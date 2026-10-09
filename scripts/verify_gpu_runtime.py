#!/usr/bin/env python
"""Run one real CUDA acceptance stage; existing result directories are refused."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import random
import sys
from time import perf_counter
import traceback


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf8")


def provenance(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def parameter_devices(model):
    return sorted({str(parameter.device) for parameter in model.parameters()})


def observe_model(torch, name, observations):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        finite = bool(torch.isfinite(output).all().item())
        observation = {
            "model": name, "input_shape": list(inputs[0].shape),
            "input_device": str(inputs[0].device), "output_shape": list(output.shape),
            "output_device": str(output.device), "output_finite": finite,
        }
        observations.append(observation)
        if not inputs[0].is_cuda or not output.is_cuda or not finite:
            raise RuntimeError("Model output failed CUDA/finite acceptance: " + name)
    return hook


def run_alignment(args, torch, result):
    from ext.t2lutils import __get_file_encoding
    from main import to_standard_lrc
    from t2l.mtl.wrapper import load_mtl_model
    from t2l.t2l import process

    method = "MTL_BDR" if args.stage == "mtl_bdr" else "MTL"
    started = perf_counter()
    models = load_mtl_model(method, cuda=True, verbose=True)
    torch.cuda.synchronize()
    result["model_load_seconds"] = perf_counter() - started
    result["model_devices"] = {}
    handles = []
    for role, model in zip(("acoustic", "boundary"), models[:2]):
        if model is not None:
            devices = parameter_devices(model)
            result["model_devices"][role] = devices
            if devices != ["cuda:0"]:
                raise RuntimeError("Model parameters are not on CUDA: " + role)
            handles.append(model.register_forward_hook(
                observe_model(torch, role, result["observed_forwards"])))
    encoding = __get_file_encoding(str(args.lyrics))
    lines = args.lyrics.read_text(encoding=encoding).splitlines()
    try:
        started = perf_counter()
        details = process(lines, str(args.audio), mtl_model=models,
                          vocalize=False, verbose=True, return_details=True)
        torch.cuda.synchronize()
        result["pipeline_seconds"] = perf_counter() - started
    finally:
        for handle in handles:
            handle.remove()
    alignment = details["alignment"]
    units = details["units"]
    if not units or not math.isfinite(alignment["score"]):
        raise RuntimeError("Alignment has no units or a nonfinite global score")
    if alignment["device"] != "cuda" or alignment["output_frames"] <= 0:
        raise RuntimeError("Alignment device or output frame count is invalid")
    for unit in units:
        start, end = unit["start_frame"], unit["end_frame"]
        if start is None or end is None or not 0 <= start <= end < alignment["output_frames"]:
            raise RuntimeError("Alignment has a missing or invalid unit interval")
        if not all(math.isfinite(unit[key]) for key in ("start_seconds", "end_seconds")):
            raise RuntimeError("Alignment has a nonfinite unit time")
    result.update(unit_count=len(units), alignment=alignment,
                  checks={"gpu_model_forwards": True, "finite_outputs": True,
                          "complete_unit_intervals": True},
                  scope="Full supplied demo, legacy layout, vocalize=False; no accuracy claim.")
    write_json(args.out_dir / "alignment.json", details)
    (args.out_dir / "enhanced.lrc").write_text(details["lrc"], encoding="utf8")
    (args.out_dir / "standard.lrc").write_text(to_standard_lrc(details["lrc"]), encoding="utf8")


def run_demucs(args, torch, result):
    import soundfile
    from t2l.t2l import load_demucs_model, preprocess_audio, separate_vocals

    audio, sample_rate = preprocess_audio(str(args.audio))
    start = round(args.clip_start * sample_rate)
    length = round(6 * sample_rate)
    clip = audio[:, start:start + length]
    if clip.shape[-1] != length or not bool(torch.isfinite(clip).all().item()):
        raise RuntimeError("The audio does not contain a finite 6-second requested clip")
    clip_path = args.out_dir / "input-clip.wav"
    soundfile.write(str(clip_path), clip.cpu().numpy().T, sample_rate, subtype="FLOAT")
    result["clip"] = {"source": str(args.audio.resolve()), "start_seconds": args.clip_start,
                      "duration_seconds": 6, "input_sample_rate": sample_rate,
                      "input_shape": list(clip.shape)}
    started = perf_counter()
    model = load_demucs_model("mdx_extra", -1, device="cuda", verbose=True)
    torch.cuda.synchronize()
    result["model_load_seconds"] = perf_counter() - started
    members = list(getattr(model, "models", []))
    if len(members) < 2:
        raise RuntimeError("mdx_extra did not load a multi-model ensemble")
    result["ensemble"] = {"name": "mdx_extra", "index": -1, "members": len(members),
                          "sources": list(model.sources), "sample_rate": model.samplerate,
                          "parameter_devices": parameter_devices(model)}
    if result["ensemble"]["parameter_devices"] != ["cuda:0"]:
        raise RuntimeError("Demucs parameters are not on CUDA")
    handles = [member.register_forward_hook(
        observe_model(torch, "demucs_member_{}".format(index), result["observed_forwards"]))
        for index, member in enumerate(members)]
    try:
        started = perf_counter()
        vocals, output_rate = separate_vocals(str(clip_path), demucs_model=model,
                                              demucs_idx=-1, verbose=True)
        torch.cuda.synchronize()
        result["separation_seconds"] = perf_counter() - started
    finally:
        for handle in handles:
            handle.remove()
    finite = bool(torch.isfinite(vocals).all().item())
    result["vocals"] = {"shape": list(vocals.shape), "sample_rate": output_rate,
                        "device": str(vocals.device), "finite": finite}
    expected_shape = [2, round(6 * output_rate)]
    if list(vocals.shape) != expected_shape or not vocals.is_cuda or not finite:
        raise RuntimeError("Separated vocals failed shape/device/finite acceptance")
    observed = {row["model"] for row in result["observed_forwards"]}
    if len(observed) != len(members):
        raise RuntimeError("Not every Demucs ensemble member ran")
    soundfile.write(str(args.out_dir / "vocals.wav"), vocals.detach().cpu().numpy().T,
                    output_rate, subtype="FLOAT")
    result.update(checks={"ensemble_all_members_ran": True, "finite_outputs": True,
                          "gpu_model_forwards": True, "six_second_vocals_shape": True},
                  scope="Only a 6-second real-audio clip and default mdx_extra ensemble; this is not the full Demucs-plus-alignment pipeline.")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("mtl", "mtl_bdr", "demucs"))
    parser.add_argument("--lyrics", type=Path, default=Path("demofile/original_txt.txt"))
    parser.add_argument("--audio", type=Path, default=Path("demofile/original_track.mp3"))
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--clip-start", type=float, default=30.0,
                        help="Start of the 6-second Demucs clip (default: 30 seconds)")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the repository root: " + str(ROOT))
    if not args.audio.is_file() or (args.stage != "demucs" and not args.lyrics.is_file()):
        parser.error("Audio and, for alignment stages, lyrics must exist")
    if not math.isfinite(args.clip_start) or args.clip_start < 0:
        parser.error("--clip-start must be finite and nonnegative")
    if args.out_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    args.out_dir.mkdir(parents=True)
    sys.path.insert(0, str(ROOT))
    result = {"stage": args.stage, "status": "running", "observed_forwards": [],
              "started_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
              "python": sys.version, "platform": platform.platform(),
              "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
              "audio": provenance(args.audio), "script": provenance(Path(__file__)),
              "seed": 7, "accuracy_metrics": {"available": False,
                  "reason": "No independently reviewed reference boundaries were supplied."}}
    write_json(args.out_dir / "result.json", result)
    torch = None
    started = perf_counter()
    try:
        result["runtime_sources"] = [provenance(ROOT / name) for name in (
            "main.py", "t2l/t2l.py", "t2l/phonetic.py", "t2l/mtl/wrapper.py",
            "t2l/mtl/model.py", "t2l/mtl/utils.py", "lid.176.ftz")]
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        import numpy as np
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing a CPU fallback")
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(7)
        result["cuda"] = {"torch": torch.__version__, "runtime": torch.version.cuda,
                          "cudnn": torch.backends.cudnn.version(),
                          "gpu": torch.cuda.get_device_name(0),
                          "capability": list(torch.cuda.get_device_capability(0)),
                          "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
        result["config"] = {"torch_threads": torch.get_num_threads(),
                            "interop_threads": torch.get_num_interop_threads(),
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if args.stage == "demucs":
            run_demucs(args, torch, result)
        else:
            names = ["checkpoint_MTL"] + (["checkpoint_BDR"] if args.stage == "mtl_bdr" else [])
            result["checkpoints"] = [provenance(ROOT / "checkpoints" / name) for name in names]
            result["lyrics"] = provenance(args.lyrics)
            run_alignment(args, torch, result)
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        (args.out_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf8")
    finally:
        result["total_seconds"] = perf_counter() - started
        if torch is not None and torch.cuda.is_initialized():
            try:
                result["peak_memory"] = {
                    "allocated_bytes": torch.cuda.max_memory_allocated(),
                    "reserved_bytes": torch.cuda.max_memory_reserved()}
            except Exception as exc:
                result["memory_measurement_error"] = str(exc)
        write_json(args.out_dir / "result.json", result)
    print(json.dumps({"stage": args.stage, "status": result["status"],
                      "result": str(args.out_dir / "result.json")}), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(cli())

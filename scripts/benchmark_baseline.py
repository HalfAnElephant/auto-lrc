#!/usr/bin/env python
"""Reproduce the CPU P0 baseline; run from the repository root."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"mtl": "MTL", "mtl_bdr": "MTL_BDR"}
LAYOUTS = ("legacy", "batch-first")
ACCURACY = {
    "available": False,
    "reason": "No independently reviewed reference boundaries were supplied.",
    "mae_seconds": None,
    "p90_seconds": None,
    "coverage": None,
}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lyrics", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, required=True,
                        help="New directory; existing results are never overwritten")
    parser.add_argument("--variants", default="mtl,mtl_bdr",
                        help="Comma-separated mtl,mtl_bdr")
    parser.add_argument("--lstm-layouts", default="legacy",
                        help="Comma-separated legacy,batch-first; correction is an ablation")
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--vocalize", type=int, choices=(0, 1), default=0)
    return parser


def reset_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def configure_lstm(models, layout):
    """Only mutate these freshly loaded instances, never model definitions/weights."""
    import torch

    if layout not in LAYOUTS:
        raise ValueError("Unknown LSTM layout: " + layout)
    result = []
    for role, model in zip(("acoustic", "boundary"), models[:2]):
        if model is None:
            continue
        before = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        layers = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.LSTM):
                original = layer.batch_first
                if layout == "batch-first":
                    layer.batch_first = True
                layers.append({"name": name, "loaded_batch_first": original,
                               "run_batch_first": layer.batch_first})
        after = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        result.append({"model": role, "layers": layers,
                       "state_dict_shapes_unchanged": before == after})
    return result


@contextmanager
def capture_timeline(models):
    """Capture real inference shapes; no extra model pass or synthetic calibration."""
    import torch
    from t2l.mtl.model import train_audio_transforms

    mel = train_audio_transforms[0]
    spec = mel.spectrogram
    pool = models[0].maxpooling
    temporal_stride = pool.stride if isinstance(pool.stride, int) else pool.stride[1]
    audit = {
        "sample_rate": mel.sample_rate,
        "stft": {key: getattr(spec, key) for key in
                 ("n_fft", "win_length", "hop_length", "pad", "center", "pad_mode")},
        "nominal_output_step_seconds": spec.hop_length * temporal_stride / mel.sample_rate,
        "timestamp_mapping": "frame_index * 256 / 22050 * 3 (existing baseline)",
        "time_origin": {
            "centered_stft_frame_zero_sample": 0 if spec.center else None,
            "pooling": "Output k pools mel frames 3k through 3k+2; midpoint is 3k+1.",
            "calibration": "Unmeasured: indexing geometry is not an acoustic onset calibration.",
            "unmeasured_offsets": ["decoder", "resampling", "vocal separation"],
        },
        "observed_shapes": {},
    }
    handles = []

    def record(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            row = {"input": list(inputs[0].shape), "output": list(output.shape)}
            if isinstance(module, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
                row.update({key: getattr(module, key) for key in
                            ("kernel_size", "stride", "padding", "dilation")})
            if isinstance(module, torch.nn.LSTM):
                row.update(batch_first=module.batch_first,
                           interpreted_time_axis=1 if module.batch_first else 0)
            audit["observed_shapes"][name] = row
        return hook

    handles.append(train_audio_transforms.register_forward_hook(record("mel")))
    for role, model in zip(("acoustic", "boundary"), models[:2]):
        if model is None:
            continue
        for name, module in model.named_modules():
            if not name or isinstance(module, (torch.nn.Conv2d, torch.nn.MaxPool2d,
                                               torch.nn.LSTM, torch.nn.Linear)):
                handles.append(module.register_forward_hook(record(role + ("." + name if name else ""))))
    try:
        yield audit
    finally:
        for handle in handles:
            handle.remove()
        shapes = audit["observed_shapes"]
        if "mel" in shapes and "acoustic" in shapes:
            mel_frames = shapes["mel"]["output"][-1]
            output_frames = shapes["acoustic"]["output"][1]
            audit["frame_geometry"] = {
                "mel_frames": mel_frames, "observed_output_frames": output_frames,
                "expected_output_frames": mel_frames // temporal_stride,
                "matches_pooling_geometry": output_frames == mel_frames // temporal_stride,
            }


def file_provenance(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def environment():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def prediction_fingerprint(details):
    # Durations vary between identical predictions and are deliberately excluded.
    value = {"units": details["units"], "lrc": details["lrc"],
             "score": details["alignment"].get("score")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    variants = args.variants.split(",")
    layouts = args.lstm_layouts.split(",")
    for values, allowed, label in ((variants, VARIANTS, "variants"), (layouts, LAYOUTS, "lstm-layouts")):
        if len(set(values)) != len(values) or any(value not in allowed for value in values):
            parser.error("--{} must contain distinct values from {}".format(label, ",".join(allowed)))
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the repository root: " + str(ROOT))
    if not args.lyrics.is_file() or not args.audio.is_file():
        parser.error("lyrics and audio must be existing files")
    if args.out_dir.exists():
        parser.error("--out-dir already exists; choose a new directory")

    sys.path.insert(0, str(ROOT))
    import torch

    # MelSpectrogram initialization performs tensor work at import time. Native
    # macOS backends must receive thread limits before that first parallel work.
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from ext.t2lutils import __get_file_encoding
    from main import to_standard_lrc
    from t2l.mtl.wrapper import load_mtl_model
    from t2l.t2l import process

    encoding = __get_file_encoding(str(args.lyrics))
    lines = args.lyrics.read_text(encoding=encoding).splitlines()
    checkpoint_paths = [ROOT / "checkpoints/checkpoint_MTL", ROOT / "lid.176.ftz"]
    if "mtl_bdr" in variants:
        checkpoint_paths.append(ROOT / "checkpoints/checkpoint_BDR")
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv if argv is None else [str(Path(__file__)), *argv],
        "environment": environment(),
        "config": {"variants": variants, "lstm_layouts": layouts, "repeats": args.repeats,
                   "seed": args.seed, "torch_threads": torch.get_num_threads(),
                   "torch_interop_threads": torch.get_num_interop_threads(),
                   "acoustic_device": "cpu", "vocalize": bool(args.vocalize),
                   "demucs_model": "mdx_extra", "demucs_idx": -1,
                   "lyrics_encoding": encoding, "deterministic_algorithms": True},
        "inputs": [file_provenance(args.lyrics), file_provenance(args.audio)],
        "models": [file_provenance(path) for path in checkpoint_paths],
        "accuracy_metrics": ACCURACY,
        "score_warning": "Raw global alignment scores are uncalibrated; they are not confidence or accuracy.",
        "ablation_warning": "Changing LSTM layout keeps checkpoint shapes compatible, but changes learned behavior. No improvement is presumed.",
    }
    args.out_dir.mkdir(parents=True)
    write_json(args.out_dir / "manifest.json", manifest)
    summary = {"runs": [], "repeatability": [], "accuracy_metrics": ACCURACY}
    for variant in variants:
        for layout in layouts:
            fingerprints = []
            for repeat in range(1, args.repeats + 1):
                stem = "{}-{}-r{}".format(variant, layout, repeat)
                reset_seed(args.seed)
                row = {"run": stem, "variant": variant, "lstm_layout": layout,
                       "repeat": repeat}
                started = perf_counter()
                try:
                    # Model construction consumes RNG before posterior smoothing in the original API.
                    models = load_mtl_model(VARIANTS[variant], cuda=False, verbose=False)
                    row["model_load_seconds"] = perf_counter() - started
                    lstm = configure_lstm(models, layout)
                    started = perf_counter()
                    with capture_timeline(models) as audit:
                        details = process(lines, str(args.audio), mtl_model=models,
                                          vocalize=bool(args.vocalize), verbose=False,
                                          return_details=True)
                    row.update(status="completed", runtime_seconds=perf_counter() - started,
                               unit_count=len(details["units"]),
                               raw_alignment_score=details["alignment"].get("score"))
                    fingerprints.append(prediction_fingerprint(details))
                    write_json(args.out_dir / (stem + ".json"),
                               {**row, "lstm": lstm, "timeline_audit": audit, "result": details})
                    (args.out_dir / (stem + ".lrc")).write_text(details["lrc"], encoding="utf-8")
                    (args.out_dir / (stem + ".standard.lrc")).write_text(
                        to_standard_lrc(details["lrc"]), encoding="utf-8")
                except Exception as exc:
                    # A failed serialization may itself have been caused by this
                    # value; do not repeat that failure while saving the error.
                    row.pop("raw_alignment_score", None)
                    row.update(status="failed", runtime_seconds=perf_counter() - started,
                               error={"type": type(exc).__name__, "message": str(exc)})
                    summary["runs"].append(row)
                    write_json(args.out_dir / (stem + ".json"), row)
                    write_json(args.out_dir / "summary.json", summary)
                    raise
                summary["runs"].append(row)
                write_json(args.out_dir / "summary.json", summary)
                print("{}: {} units, {:.3f}s".format(stem, row["unit_count"], row["runtime_seconds"]), flush=True)
            summary["repeatability"].append({
                "variant": variant, "lstm_layout": layout, "repeats": len(fingerprints),
                "identical_predictions_and_score": len(set(fingerprints)) == 1 if len(fingerprints) > 1 else None,
                "fingerprints": fingerprints,
            })
            write_json(args.out_dir / "summary.json", summary)
    print("Saved " + str(args.out_dir / "summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())

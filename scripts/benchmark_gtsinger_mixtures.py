#!/usr/bin/env python3
"""Measure CUDA predictions on clean, mixed and separated GTSinger fixtures."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter
import traceback


ROOT = Path(__file__).resolve().parents[1]
SEED = 7
INPUT_NAMES = ("source_audio", "mixture_audio", "accompaniment_audio", "lyrics")
ACCURACY = {"available": False, "reason": "Independent human boundaries remain pending."}
XCORR_WARNING = (
    "Waveform correlation is a diagnostic on a restricted lag range. It is not "
    "absolute latency or lyric-boundary calibration; no audio or prediction is shifted."
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixtures", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, required=True,
                        help="New report directory; existing results are refused")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="New directory under data/local for native-rate FLOAT vocals")
    parser.add_argument("--segment-id", action="append", default=[],
                        help="Repeat for an explicit subset; each ID must exist and be distinct")
    return parser


def provenance(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(label + " must be a positive integer")
    return value


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + " must be finite")
    return value


def validate_manifest_inputs(manifest_path):
    """Verify declared bytes, digests and sample grids before importing the runtime."""
    import soundfile

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    json.dumps(manifest, allow_nan=False)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "gtsinger-mixtures-v1":
        raise ValueError("Expected schema_version gtsinger-mixtures-v1")
    for name in ("segments_source", "generator", "config"):
        if not isinstance(manifest.get(name), dict):
            raise ValueError("Missing manifest provenance: " + name)
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected non-empty mixture rows")
    validated, identifiers = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each mixture row must be an object")
        identifier = row.get("segment_id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("Expected distinct non-empty segment_id values")
        identifiers.add(identifier)
        for name in ("song_id", "derived_from_segment_id"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError("Missing row identity: " + name)
        if row.get("split") != "unassigned":
            raise ValueError("This P1 fixture benchmark requires split='unassigned'")
        frames = _positive_integer(row.get("num_samples"), "num_samples")
        channels = _positive_integer(row.get("channels"), "channels")
        if row.get("sample_rate") != 48000:
            raise ValueError("Source mixtures must remain on the original 48000Hz grid")
        timeline = row.get("timeline")
        if (not isinstance(timeline, dict)
                or isinstance(timeline.get("source_to_mixture_offset_samples"), bool)
                or timeline.get("source_to_mixture_offset_samples") != 0):
            raise ValueError("Source-to-mixture offset must be explicitly zero")
        for name in ("snr_db", "mixture_gain", "accompaniment_gain"):
            _finite_number(row.get(name), name)
        if row["mixture_gain"] <= 0 or row["accompaniment_gain"] <= 0:
            raise ValueError("Mixture gains must be positive")
        inputs = {}
        for name in INPUT_NAMES:
            declared = row.get(name)
            if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
                raise ValueError("Missing input provenance: " + name)
            path = Path(declared["path"])
            path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
            actual = provenance(path)
            if (isinstance(declared.get("bytes"), bool)
                    or declared.get("bytes") != actual["bytes"]
                    or declared.get("sha256") != actual["sha256"]):
                raise ValueError("Input bytes/SHA256 mismatch: " + name)
            if name != "lyrics":
                info = soundfile.info(str(path))
                if (info.samplerate, info.frames, info.channels) != (48000, frames, channels):
                    raise ValueError("Input audio sample grid mismatch: " + name)
                actual["audio"] = {"sample_rate": info.samplerate, "num_samples": info.frames,
                                   "channels": info.channels, "subtype": info.subtype}
            else:
                if not path.read_text(encoding="utf-8").strip():
                    raise ValueError("Lyrics must contain UTF-8 text")
            inputs[name] = actual
        validated.append({**row, "inputs": inputs})
    return {**manifest, "rows": validated}


def select_rows(manifest, segment_ids):
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("Repeated --segment-id values are refused")
    available = {row["segment_id"] for row in manifest["rows"]}
    missing = [identifier for identifier in segment_ids if identifier not in available]
    if missing:
        raise ValueError("Requested segment_id not in manifest: " + ", ".join(missing))
    requested = set(segment_ids)
    return [row for row in manifest["rows"] if not requested or row["segment_id"] in requested]


def _first_channel(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1
            or not np.isfinite(values).all()):
        raise ValueError("Expected non-empty finite sample-major audio")
    return values[:, 0]


def _window_xcorr(reference, comparison, sample_rate, max_lag):
    import numpy as np

    count = len(reference)
    empty = {"status": "unavailable", "lag_samples": None, "lag_seconds": None,
             "normalized_peak": None, "second_peak": None, "peak_margin": None,
             "peak_at_search_edge": None,
             "confidence_kind": "diagnostic correlation statistics, not boundary confidence"}
    if count < 3:
        return {**empty, "reason": "Fewer than three common samples"}
    x = reference - reference.mean()
    y = comparison - comparison.mean()
    if np.dot(x, x) <= 0 or np.dot(y, y) <= 0:
        return {**empty, "reason": "Zero centered waveform energy"}
    limit = min(max_lag, count - 2)
    lags = np.arange(-limit, limit + 1)
    fft_size = 1 << (2 * count - 2).bit_length()
    correlation = np.fft.irfft(np.fft.rfft(y, fft_size) * np.conj(np.fft.rfft(x, fft_size)), fft_size)
    x_energy = np.r_[0.0, np.cumsum(x * x)]
    y_energy = np.r_[0.0, np.cumsum(y * y)]
    x_start, x_stop = np.maximum(-lags, 0), count - np.maximum(lags, 0)
    y_start, y_stop = np.maximum(lags, 0), count - np.maximum(-lags, 0)
    denominator = np.sqrt(np.maximum(0, x_energy[x_stop] - x_energy[x_start])
                          * np.maximum(0, y_energy[y_stop] - y_energy[y_start]))
    valid = denominator > 0
    if not valid.any():
        return {**empty, "reason": "No overlapping non-zero energy"}
    coefficients = np.full(len(lags), -np.inf)
    coefficients[valid] = np.clip(correlation[np.mod(lags[valid], fft_size)] / denominator[valid], -1, 1)
    best = int(np.argmax(coefficients))
    lag = int(lags[best])
    exclusion = max(1, round(sample_rate * 0.001))
    secondary = valid & (np.abs(lags - lag) > exclusion)
    second = float(coefficients[secondary].max()) if secondary.any() else None
    peak = float(coefficients[best])
    return {"status": "available", "lag_samples": lag, "lag_seconds": lag / sample_rate,
            "normalized_peak": peak, "second_peak": second,
            "peak_margin": peak - second if second is not None else None,
            "peak_at_search_edge": abs(lag) == limit, "searched_lag_samples": [-limit, limit],
            "overlap_samples_at_peak": count - abs(lag),
            "secondary_peak_exclusion_samples": exclusion,
            "confidence_kind": empty["confidence_kind"]}


def waveform_xcorr_diagnostic(clean, separated, sample_rate, max_lag_ms=50.0):
    """Restricted FFT cross-correlation; positive lag means comparison is later."""
    _positive_integer(sample_rate, "sample_rate")
    _finite_number(max_lag_ms, "max_lag_ms")
    if not 0 < max_lag_ms <= 50:
        raise ValueError("Correlation search must be within (0, 50] milliseconds")
    reference, comparison = _first_channel(clean), _first_channel(separated)
    count = min(len(reference), len(comparison))
    max_lag = math.floor(sample_rate * max_lag_ms / 1000)
    edges = [0, count // 3, 2 * count // 3, count]
    windows = [("overall", 0, count)] + [
        ("third-{}".format(index + 1), edges[index], edges[index + 1]) for index in range(3)]
    return {
        "diagnostic_only": True, "time_shift_applied": False, "warning": XCORR_WARNING,
        "channel_policy": "first channel, matching the existing alignment runtime",
        "positive_lag": "comparison is later than reference", "sample_rate": sample_rate,
        "max_lag_ms": max_lag_ms, "max_lag_samples": max_lag,
        "reference_samples": len(reference), "comparison_samples": len(comparison),
        "length_delta_samples": len(comparison) - len(reference), "common_samples": count,
        "windows": [{"name": name, "start_sample": start, "end_sample": end,
                     **_window_xcorr(reference[start:end], comparison[start:end], sample_rate, max_lag)}
                    for name, start, end in windows],
    }


def prediction_changes(clean, other):
    """Compare raw predicted units; these differences have no human-label reference."""
    unavailable = {"status": "unavailable", "accuracy_available": False,
                   "time_shift_applied": False, "units": []}
    identities = ("text", "line_index", "word_index", "phonemes")
    if (len(clean["units"]) != len(other["units"])
            or any(any(a.get(key) != b.get(key) for key in identities)
                   for a, b in zip(clean["units"], other["units"]))):
        return {**unavailable, "reason": "Unit identities differ; no automatic matching performed"}
    if (not clean["units"] or clean["alignment"].get("frame_resolution_seconds")
            != other["alignment"].get("frame_resolution_seconds")):
        return {**unavailable, "reason": "Missing units or differing nominal frame grids"}
    rows = []
    for index, (a, b) in enumerate(zip(clean["units"], other["units"])):
        if any(a.get(key) is None or b.get(key) is None
               for key in ("start_frame", "end_frame", "start_seconds", "end_seconds")):
            return {**unavailable, "reason": "Missing predicted intervals"}
        rows.append({"unit_index": index, "text": a["text"], "clean": dict(a), "comparison": dict(b),
                     **{key + "_delta": b[key] - a[key] for key in
                        ("start_frame", "end_frame", "start_seconds", "end_seconds")}})
    summary = {}
    for boundary in ("start", "end"):
        differences = [row[boundary + "_seconds_delta"] for row in rows]
        summary["mean_absolute_{}_change_seconds".format(boundary)] = sum(abs(v) for v in differences) / len(rows)
        summary["max_absolute_{}_change_seconds".format(boundary)] = max(abs(v) for v in differences)
    return {"status": "comparable", "accuracy_available": False, "time_shift_applied": False,
            "comparison_kind": "prediction change relative to clean, not boundary error",
            "positive_delta": "comparison prediction is later than clean prediction",
            "units": rows, "summary": summary}


def reset_seed(torch):
    import numpy as np

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _tensors(torch, value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _tensors(torch, item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _tensors(torch, item)]
    return []


@contextmanager
def cuda_forward_observations(torch, models, observations):
    handles = []
    def hook_for(role):
        def hook(module, inputs, output):
            incoming, outgoing = _tensors(torch, inputs), _tensors(torch, output)
            finite = bool(outgoing) and all(bool(torch.isfinite(value).all().item()) for value in outgoing)
            row = {"model": role, "input_shapes": [list(value.shape) for value in incoming],
                   "input_devices": [str(value.device) for value in incoming],
                   "output_shapes": [list(value.shape) for value in outgoing],
                   "output_devices": [str(value.device) for value in outgoing], "output_finite": finite}
            observations.append(row)
            if (not incoming or not outgoing or not finite
                    or any(not value.is_cuda or value.device.index != 0 for value in incoming + outgoing)):
                raise RuntimeError("Forward failed CUDA/finite acceptance: " + role)
        return hook
    try:
        for role, model in models:
            devices = sorted({str(value.device) for value in model.parameters()})
            if devices != ["cuda:0"]:
                raise RuntimeError("Model parameters are not exclusively on cuda:0: " + role)
            handles.append(model.register_forward_hook(hook_for(role)))
        yield
        observed = {row["model"] for row in observations}
        if any(role not in observed for role, _ in models):
            raise RuntimeError("A required model did not execute a real forward")
    finally:
        for handle in handles:
            handle.remove()


def measured_cuda_call(torch, record, function):
    reset_seed(torch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    record.update(status="running", seed=SEED)
    started = perf_counter()
    try:
        value = function()
        torch.cuda.synchronize()
        record["status"] = "completed"
        return value
    except Exception as error:
        record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        record["runtime_seconds"] = perf_counter() - started
        record["peak_memory"] = {"allocated_bytes": torch.cuda.max_memory_allocated(),
                                 "reserved_bytes": torch.cuda.max_memory_reserved()}


def validate_prediction(details, method):
    json.dumps(details, allow_nan=False)
    alignment = details["alignment"]
    if (alignment.get("device") not in ("cuda", "cuda:0") or alignment.get("method") != method
            or not details.get("units") or not math.isfinite(alignment["score"])):
        raise RuntimeError("Prediction failed method/CUDA/finite acceptance")
    frames = _positive_integer(alignment.get("output_frames"), "output_frames")
    resolution = _finite_number(alignment.get("frame_resolution_seconds"), "frame_resolution_seconds")
    if resolution <= 0:
        raise RuntimeError("Invalid nominal frame resolution")
    for unit in details["units"]:
        start, end = unit.get("start_frame"), unit.get("end_frame")
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
                or not isinstance(end, int) or not 0 <= start <= end < frames):
            raise RuntimeError("Missing or invalid predicted unit interval")
        for name in ("start", "end"):
            seconds = _finite_number(unit.get(name + "_seconds"), name + "_seconds")
            if not math.isclose(seconds, unit[name + "_frame"] * resolution, rel_tol=0, abs_tol=1e-9):
                raise RuntimeError("Prediction seconds differ from the raw nominal frame mapping")


def runtime_environment():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short")}


def demucs_provenance(torch):
    import demucs
    import yaml

    config_path = Path(demucs.__file__).parent / "remote/mdx_extra.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    signatures = config["models"]
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoints = []
    for signature in signatures:
        matches = sorted(cache.glob(str(signature) + "-*.th"))
        if len(matches) != 1:
            raise RuntimeError("Cannot uniquely identify loaded Demucs checkpoint: " + str(signature))
        checkpoints.append({"signature": str(signature), **provenance(matches[0])})
    return {"bag_config": provenance(config_path), "checkpoints": checkpoints,
            "association": "Model signatures from the installed mdx_extra bag configuration"}


def reserve_outputs(out_dir, audio_dir):
    if Path(out_dir).is_symlink() or Path(audio_dir).is_symlink():
        raise FileExistsError("Output paths must not be existing symlinks")
    out_dir, audio_dir = Path(out_dir).resolve(), Path(audio_dir).resolve()
    local_root = (ROOT / "data/local").resolve()
    if not audio_dir.is_relative_to(local_root) or audio_dir == local_root:
        raise ValueError("--audio-dir must be a new directory under data/local")
    if out_dir.is_relative_to(audio_dir) or audio_dir.is_relative_to(out_dir):
        raise ValueError("Report and audio directories must not overlap")
    for path in (out_dir, audio_dir):
        if path.exists() or path.is_symlink():
            raise FileExistsError("Output directory already exists: " + str(path))
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir, audio_dir


def run_row(torch, row, record, models, demucs, audio_dir, report_path, runtime):
    import numpy as np
    import soundfile

    source, source_rate = soundfile.read(row["inputs"]["source_audio"]["path"], dtype="float64", always_2d=True)
    for name in ("source_audio", "mixture_audio", "accompaniment_audio"):
        values, rate = soundfile.read(row["inputs"][name]["path"], dtype="float64", always_2d=True)
        if rate != source_rate or values.shape != source.shape or not np.isfinite(values).all():
            raise ValueError("Audio changed or has nonfinite samples: " + name)
    lines = Path(row["inputs"]["lyrics"]["path"]).read_text(encoding="utf-8").splitlines()
    record.update(status="running", inputs=row["inputs"], predictions={}, separation={})
    methods = {"MTL": (models[0], None, models[2], False, models[4]), "MTL_BDR": models}

    def infer(mode, path):
        predictions = record["predictions"].setdefault(mode, {})
        for method, loaded in methods.items():
            run = {"audio": provenance(path), "observed_forwards": [], "vocalize": False}
            predictions[method] = run
            roles = [("acoustic", loaded[0])] + ([("boundary", loaded[1])] if method == "MTL_BDR" else [])
            def invoke():
                with cuda_forward_observations(torch, roles, run["observed_forwards"]):
                    details = runtime["process"](lines, str(path), mtl_model=loaded,
                                                 vocalize=False, verbose=False, return_details=True)
                validate_prediction(details, method)
                return details
            run["result"] = measured_cuda_call(torch, run, invoke)
            if mode != "clean":
                run["change_from_clean_prediction"] = prediction_changes(
                    record["predictions"]["clean"][method]["result"], run["result"])
            write_json(report_path, record)

    infer("clean", Path(row["inputs"]["source_audio"]["path"]))
    infer("mixture", Path(row["inputs"]["mixture_audio"]["path"]))
    separation = record["separation"]
    separation["observed_forwards"] = []
    members = list(getattr(demucs, "models", []))
    if len(members) < 2:
        raise RuntimeError("Expected a multi-model mdx_extra ensemble")
    native_rate = _positive_integer(demucs.samplerate, "Demucs native sample rate")
    def separate():
        roles = [("demucs_member_{}".format(index), member) for index, member in enumerate(members)]
        with cuda_forward_observations(torch, roles, separation["observed_forwards"]):
            vocals, rate = runtime["separate_vocals"](
                row["inputs"]["mixture_audio"]["path"], sample_rate=native_rate,
                demucs_model=demucs, demucs_idx=-1, verbose=False)
        if (rate != native_rate or vocals.ndim != 2 or vocals.shape[0] != 2
                or vocals.shape[1] < 1 or not vocals.is_cuda
                or not bool(torch.isfinite(vocals).all().item())):
            raise RuntimeError("Separated output failed native-rate/shape/CUDA/finite acceptance")
        return vocals, rate
    vocals, rate = measured_cuda_call(torch, separation, separate)
    output = audio_dir / ("segment-{:04d}-vocals.wav".format(record["row_index"]))
    if output.exists():
        raise FileExistsError("Separated audio already exists: " + str(output))
    soundfile.write(str(output), vocals.detach().cpu().numpy().T, rate, subtype="FLOAT")
    expected_native = row["num_samples"] * native_rate // source_rate
    separation.update(output_audio=provenance(output), sample_rate=rate, channels=int(vocals.shape[0]),
                      num_samples=int(vocals.shape[1]), expected_floor_num_samples=expected_native,
                      length_delta_from_expected_samples=int(vocals.shape[1]) - expected_native,
                      output_subtype=soundfile.info(str(output)).subtype,
                      waveform_time_shift_applied=False, padding_applied=False)
    started = perf_counter()
    reference, reference_rate = runtime["preprocess_audio"](row["inputs"]["source_audio"]["path"], sr=native_rate)
    if reference_rate != native_rate:
        raise RuntimeError("Correlation reference resampling returned the wrong sample rate")
    diagnostic = waveform_xcorr_diagnostic(reference.detach().cpu().numpy().T,
                                           vocals.detach().cpu().numpy().T, native_rate)
    for window in diagnostic["windows"]:
        lag = window["lag_samples"]
        window["lag_equivalent_source_samples"] = lag * source_rate / native_rate if lag is not None else None
    diagnostic["equivalent_source_sample_rate"] = source_rate
    diagnostic["source_sample_mapping"] = "lag_seconds * source_rate, fractional diagnostic value; never applied"
    separation["waveform_correlation"] = diagnostic
    separation["correlation_seconds"] = perf_counter() - started
    record["resampling_routes"] = {
        "source_sample_rate": source_rate, "source_num_samples": row["num_samples"],
        "clean_and_mixture_alignment": [source_rate, 22050],
        "mixture_separation": [source_rate, native_rate],
        "separated_alignment": [native_rate, 22050],
        "diagnostic_reference": [source_rate, native_rate],
        "expected_native_samples_floor": expected_native,
        "native_tail_rounding_fraction_samples": (row["num_samples"] * native_rate % source_rate) / source_rate,
        "expected_direct_alignment_samples_floor": row["num_samples"] * 22050 // source_rate,
        "expected_separated_alignment_samples_floor": int(vocals.shape[1]) * 22050 // native_rate,
        "native_output_duration_minus_source_seconds": int(vocals.shape[1]) / native_rate - row["num_samples"] / source_rate,
        "rounding_policy": "Existing julius floor output length; no padding, 48k return trip or time shift",
        "word_boundary_calibration": "unmeasured",
    }
    infer("separated", output)
    for name in INPUT_NAMES:
        current = provenance(row["inputs"][name]["path"])
        if any(current[key] != row["inputs"][name][key] for key in ("bytes", "sha256")):
            raise RuntimeError("Input changed during the benchmark: " + name)
    record["status"] = "completed"
    write_json(report_path, record)


def cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the repository root: " + str(ROOT))
    try:
        out_dir, audio_dir = reserve_outputs(args.out_dir, args.audio_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    sys.path.insert(0, str(ROOT))
    result = {"schema_version": "gtsinger-mixture-benchmark-v1", "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv if argv is None else list(argv),
              "seed": SEED, "rows": [], "accuracy_metrics": ACCURACY,
              "interpretation": "Prediction changes relative to clean are not accuracy measurements.",
              "correlation_warning": XCORR_WARNING, "audio_dir": str(audio_dir),
              "timing_scope": "Real inference with finite/device forward hooks; hook synchronization overhead is included."}
    write_json(out_dir / "result.json", result)
    torch = None
    current = None
    started = perf_counter()
    try:
        result["environment"] = runtime_environment()
        result["mixtures_manifest"] = provenance(args.mixtures)
        manifest = validate_manifest_inputs(args.mixtures)
        selected = select_rows(manifest, args.segment_id)
        result["selection"] = {"requested_segment_ids": args.segment_id,
                               "selected_segment_ids": [row["segment_id"] for row in selected],
                               "manifest_rows": len(manifest["rows"]), "selected_rows": len(selected),
                               "policy": "Explicit subset or all rows by default, preserving manifest order",
                               "input_validation_scope": "All manifest rows; inference only on selected rows"}
        result["input_provenance"] = {key: value for key, value in manifest.items() if key != "rows"}
        result["runtime_sources"] = [provenance(ROOT / name) for name in (
            "scripts/benchmark_gtsinger_mixtures.py", "t2l/t2l.py", "t2l/phonetic.py",
            "t2l/mtl/wrapper.py", "t2l/mtl/model.py", "t2l/mtl/utils.py", "lid.176.ftz")]
        result["checkpoints"] = [provenance(ROOT / "checkpoints" / name)
                                 for name in ("checkpoint_MTL", "checkpoint_BDR")]
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing a CPU fallback")
        torch.cuda.set_device(0)
        result["cuda"] = {"torch": torch.__version__, "runtime": torch.version.cuda,
                          "cudnn": torch.backends.cudnn.version(), "device": "cuda:0",
                          "gpu": torch.cuda.get_device_name(0),
                          "capability": list(torch.cuda.get_device_capability(0)),
                          "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
        result["config"] = {"methods": ["MTL", "MTL_BDR"], "audio_modes": ["clean", "mixture", "separated"],
                            "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                            "seed_reset_before_each_inference": True, "repeatability_established": False,
                            "demucs_model": "mdx_extra", "demucs_index": -1,
                            "shared_acoustic_and_boundary_models": True, "separations_per_row": 1}
        from t2l.mtl.wrapper import load_mtl_model
        from t2l.t2l import load_demucs_model, preprocess_audio, process, separate_vocals
        result["model_load"] = {}
        models = measured_cuda_call(torch, result["model_load"].setdefault("mtl_bdr", {}),
                                    lambda: load_mtl_model("MTL_BDR", cuda=True, verbose=False))
        if models[0] is None or models[1] is None or models[4] not in ("cuda", "cuda:0"):
            raise RuntimeError("Preloaded acoustic/boundary models are not CUDA models")
        demucs = measured_cuda_call(torch, result["model_load"].setdefault("demucs", {}),
                                    lambda: load_demucs_model("mdx_extra", -1, device="cuda", verbose=False))
        result["demucs"] = {"sources": list(demucs.sources), "sample_rate": demucs.samplerate,
                            "ensemble_members": len(list(getattr(demucs, "models", []))),
                            **demucs_provenance(torch)}
        audio_dir.mkdir(parents=True, exist_ok=False)
        runtime = {"preprocess_audio": preprocess_audio, "process": process, "separate_vocals": separate_vocals}
        for index, row in enumerate(selected, 1):
            current = {"row_index": index, "segment_id": row["segment_id"], "song_id": row["song_id"],
                       "derived_from_segment_id": row["derived_from_segment_id"], "split": row["split"],
                       "snr_db": row["snr_db"], "mixture_gain": row["mixture_gain"],
                       "accompaniment_gain": row["accompaniment_gain"], "source_timeline": row["timeline"],
                       "status": "running"}
            result["rows"].append(current)
            run_row(torch, row, current, models, demucs, audio_dir,
                    out_dir / ("segment-{:04d}.json".format(index)), runtime)
            write_json(out_dir / "result.json", result)
        result["status"] = "completed"
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        result.update(status="failed", error=failure)
        if current is not None:
            current.update(status="failed", error=failure)
            write_json(out_dir / ("segment-{:04d}.json".format(current["row_index"])), current)
        (out_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        result["total_seconds"] = perf_counter() - started
        measurements = list(result.get("model_load", {}).values())
        for row in result["rows"]:
            measurements.append(row.get("separation", {}))
            measurements.extend(run for mode in row.get("predictions", {}).values() for run in mode.values())
        peaks = [record["peak_memory"] for record in measurements if "peak_memory" in record]
        if peaks:
            result["peak_memory"] = {key: max(peak[key] for peak in peaks)
                                     for key in ("allocated_bytes", "reserved_bytes")}
        write_json(out_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "result": str(out_dir / "result.json")}), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(cli())

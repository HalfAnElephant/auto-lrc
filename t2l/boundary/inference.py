"""Explicit new-checkpoint inference and non-destructive coarse adaptation."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .model import BoundaryTCN
from .neural_features import BoundaryConfig, build_candidate_windows, extract_neural_features

CHECKPOINT_SCHEMA = "boundary-checkpoint-v1"


def resolve_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        chosen = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError("Invalid boundary inference device") from error
    if chosen.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Only explicit CPU, CUDA or MPS inference is supported")
    if chosen.type == "cuda" and (not torch.cuda.is_available()
            or (chosen.index is not None and chosen.index >= torch.cuda.device_count())):
        raise ValueError("Requested CUDA device is unavailable; no CPU fallback")
    if chosen.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device is unavailable; no CPU fallback")
    if chosen.type == "cuda" and chosen.index is None:
        chosen = torch.device("cuda", torch.cuda.current_device())
    elif chosen.type == "cpu":
        chosen = torch.device("cpu")
    elif chosen.type == "mps":
        if chosen.index not in (None, 0):
            raise ValueError("Requested MPS index is unavailable; no CPU fallback")
        chosen = torch.device("mps", 0)
    return chosen


def load_boundary_checkpoint(path, *, device="auto"):
    if path is None:
        raise ValueError("A new boundary checkpoint path must be explicitly specified")
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Checkpoint must be an existing regular file")
    chosen = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
            or not isinstance(checkpoint.get("model_state_dict"), dict)):
        raise ValueError("Expected a new boundary config/model_state_dict checkpoint")
    config = BoundaryConfig.from_dict(checkpoint.get("config"))
    for value in checkpoint["model_state_dict"].values():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError("Checkpoint state must contain finite tensors")
    model = BoundaryTCN(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("Checkpoint state becomes nonfinite in model precision")
    model.to(chosen).eval()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return {"model": model, "config": config, "device": str(chosen), "checkpoint_path": str(path.absolute()),
            "checkpoint_sha256": digest.hexdigest()}


def _fallback(row, reason):
    refinement = row["refinement"]
    row["start_sample"] = refinement["coarse_start_sample"]
    row["start_seconds"] = refinement["coarse_start_seconds"]
    row["confidence"] = refinement["coarse_confidence"]
    row["confidence_status"] = refinement["coarse_confidence_status"]
    row["sample_coordinate_kind"] = refinement["coarse_sample_coordinate_kind"]
    refinement.update(status="fallback", reason=reason, delta_samples=0, delta_seconds=0.0)


def refine_boundaries(audio: np.ndarray, sample_rate: int, candidates, *, checkpoint=None,
                      loaded=None, device="auto", batch_size=32, top_k=5):
    """Return candidate dictionaries; predictions and logits remain uncalibrated."""
    if type(batch_size) is not int or batch_size <= 0 or type(top_k) is not int or top_k <= 0:
        raise ValueError("batch_size and top_k must be positive integers")
    if (checkpoint is None) == (loaded is None):
        raise ValueError("Specify exactly one explicit checkpoint or loaded boundary runtime")
    runtime = load_boundary_checkpoint(checkpoint, device=device) if loaded is None else loaded
    if not isinstance(runtime, dict) or not isinstance(runtime.get("config"), BoundaryConfig) or not isinstance(runtime.get("model"), BoundaryTCN):
        raise ValueError("loaded must be a new boundary model runtime")
    config, model = runtime["config"], runtime["model"]
    if model.config != config:
        raise ValueError("Model and runtime config differ")
    chosen = resolve_device(runtime.get("device"))
    if device != "auto" and resolve_device(device) != chosen:
        raise ValueError("Requested device differs from loaded runtime device")
    if next(model.parameters()).device != chosen:
        raise ValueError("Loaded model is not on the declared device")
    rows = [copy.deepcopy(row.to_dict() if hasattr(row, "to_dict") else row) for row in candidates]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Candidates must be dictionaries or BoundaryCandidates")
    try:
        json.dumps(rows, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Candidates must be finite JSON-serializable metadata") from error
    feature_result = extract_neural_features(audio, sample_rate, config=config)
    windows = build_candidate_windows(feature_result, rows, config=config)
    valid_rows = []
    for row in rows:
        end = row.get("end_sample")
        start = row.get("start_sample")
        if end is not None and (type(end) is not int or not 0 <= end <= feature_result["metadata"]["num_samples"]
                                or (start is not None and end < start)):
            raise ValueError("Invalid candidate end sample")
        row["refinement"] = {"method": "boundary_tcn", "status": "pending", "config": config.to_dict(),
            "coarse_start_sample": start, "coarse_start_seconds": row.get("start_seconds"),
            "coarse_confidence": row.get("confidence"), "coarse_confidence_status": row.get("confidence_status", "uncalibrated"),
            "coarse_sample_coordinate_kind": row.get("sample_coordinate_kind", "coarse"),
            "raw_frames_preserved": True, "confidence": None, "confidence_status": "uncalibrated",
            "device": str(chosen), "checkpoint_sha256": runtime.get("checkpoint_sha256"),
            "feature_metadata": feature_result["metadata"], "top_candidates": [],
            "uncertainty_semantics": "Heatmap mass, entropy and confidence head are uncalibrated model outputs"}
        if row.get("status") != "unaligned" and start is not None and row.get("start_seconds") is not None:
            valid_rows.append(row)
    if any(left["start_sample"] > right["start_sample"] for left, right in zip(valid_rows, valid_rows[1:])):
        raise ValueError("Coarse starts must be monotonic before local neural refinement")
    for skipped in windows["skipped"]:
        _fallback(rows[skipped["candidate_index"]], skipped["reason"])
    model.eval()
    with torch.inference_mode():
        for beginning in range(0, len(windows["candidate_indices"]), batch_size):
            ending = min(beginning + batch_size, len(windows["candidate_indices"]))
            features = torch.from_numpy(windows["features"][beginning:ending]).to(chosen)
            phones = torch.from_numpy(windows["phones"][beginning:ending]).to(chosen)
            mask = torch.from_numpy(windows["mask"][beginning:ending]).to(chosen)
            result = model(features, phones, mask)
            if (set(result) != {"heatmap_logits", "delta_seconds", "confidence_logit"}
                    or result["heatmap_logits"].shape != mask.shape
                    or result["delta_seconds"].shape != (ending - beginning,)
                    or result["confidence_logit"].shape != (ending - beginning,)
                    or any(not bool(torch.isfinite(value).all()) for value in result.values())):
                raise ValueError("Invalid or nonfinite boundary model outputs")
            heatmap = result["heatmap_logits"].float()
            mass = torch.softmax(heatmap.masked_fill(~mask, torch.finfo(heatmap.dtype).min), dim=1).cpu().numpy()
            logits = heatmap.cpu().numpy()
            for local, window_index in enumerate(range(beginning, ending)):
                row = rows[windows["candidate_indices"][window_index]]
                positions = windows["sample_positions"][window_index]
                valid = np.flatnonzero(windows["mask"][window_index])
                ranking = sorted(valid, key=lambda index: (-float(mass[local, index]), int(positions[index])))
                proposed = int(positions[ranking[0]])
                weights = mass[local, valid].astype(np.float64)
                weights /= weights.sum()
                mean = float(np.sum(weights * positions[valid]))
                variance = float(np.sum(weights * (positions[valid] - mean) ** 2))
                refinement = row["refinement"]
                refinement.update(status="refined", reason=None,
                    top_candidates=[{"sample": int(positions[index]), "seconds": int(positions[index]) / sample_rate,
                                     "heatmap_logit": float(logits[local, index]), "heatmap_mass": float(mass[local, index])}
                                    for index in ranking[:top_k]],
                    predicted_sample=proposed, predicted_seconds=proposed / sample_rate,
                    delta_samples=proposed - row["start_sample"], delta_seconds=(proposed - row["start_sample"]) / sample_rate,
                    regression_delta_seconds=float(result["delta_seconds"][local].cpu()),
                    confidence_logit=float(result["confidence_logit"][local].cpu()),
                    heatmap_entropy=float(-np.sum(weights * np.log(np.maximum(weights, 1e-30)))),
                    heatmap_standard_deviation_seconds=math.sqrt(variance) / sample_rate,
                    heatmap_mean_sample=mean, search_window_samples=[int(positions[valid[0]]), int(positions[valid[-1]])],
                    phone_context_ids=windows["phones"][window_index].tolist())
                row.update(start_sample=proposed, start_seconds=proposed / sample_rate, confidence=None,
                           confidence_status="uncalibrated_neural",
                           sample_coordinate_kind="TCN heatmap center on original decoded sample grid; onset remains uncalibrated")
                if row.get("end_sample") is not None and proposed > row["end_sample"]:
                    _fallback(row, "proposal_after_coarse_word_end")
    changed = True
    while changed:
        changed = False
        valid = [row for row in rows if row.get("status") != "unaligned"
                 and row.get("start_sample") is not None and row.get("start_seconds") is not None]
        for left, right in zip(valid, valid[1:]):
            if left["start_sample"] <= right["start_sample"]:
                continue
            refined = [row for row in (left, right) if row["refinement"]["status"] == "refined"]
            if not refined:
                raise ValueError("Coarse starts are not monotonic")
            for row in refined:
                _fallback(row, "crossing_proposals")
            changed = True
    json.dumps(rows, allow_nan=False)
    return rows

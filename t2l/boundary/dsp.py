"""Local acoustic boundary proposals on the original decoded sample grid."""
import copy
from dataclasses import asdict, dataclass
import math

import numpy as np

from .features import extract_features


@dataclass(frozen=True)
class DSPConfig:
    name: str = "conservative"
    prior_weight: float = 0.25
    min_evidence: float = 0.25
    min_gain: float = 0.05
    hop_samples: int = 120
    radius_seconds: float = 0.15

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be nonempty text")
        for key in ("prior_weight", "min_evidence", "min_gain", "radius_seconds"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(key + " must be finite")
            if value < 0 or (key == "radius_seconds" and value == 0):
                raise ValueError(key + " is outside its allowed range")
        if self.min_evidence > 1:
            raise ValueError("min_evidence must be <= 1")
        if isinstance(self.hop_samples, bool) or not isinstance(self.hop_samples, int) or self.hop_samples < 1:
            raise ValueError("hop_samples must be a positive integer")

    def to_dict(self):
        return asdict(self)


WEIGHTS = {
    "stop": {"energy_rise": .15, "spectral_flux": .35, "high_frequency_rise": .35,
             "periodicity_rise": .05, "spectral_change": .10},
    "fricative": {"energy_rise": .15, "spectral_flux": .20, "high_frequency_rise": .45,
                  "periodicity_rise": .05, "spectral_change": .15},
    "sonorant": {"energy_rise": .25, "spectral_flux": .20, "high_frequency_rise": .05,
                 "periodicity_rise": .30, "spectral_change": .20},
    "vowel": {"energy_rise": .25, "spectral_flux": .20, "high_frequency_rise": .05,
              "periodicity_rise": .35, "spectral_change": .15},
    "unknown": {"energy_rise": .30, "spectral_flux": .30, "high_frequency_rise": .15,
                "periodicity_rise": .10, "spectral_change": .15},
}


def phoneme_class(phonemes):
    first = str(phonemes[0]).upper().rstrip("012") if phonemes else ""
    if first in {"P", "B", "T", "D", "K", "G", "CH", "JH"}:
        return "stop"
    if first in {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}:
        return "fricative"
    if first in {"M", "N", "NG", "L", "R", "W", "Y"}:
        return "sonorant"
    if first in {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}:
        return "vowel"
    return "unknown"


def _features(audio, sample_rate, config, features):
    samples = np.asarray(audio)
    if samples.ndim == 2 and samples.shape[0] > 0:
        samples = samples[0]
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("audio must contain finite mono or channel-first samples")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    features = extract_features(audio, sample_rate, hop_samples=config.hop_samples) if features is None else features
    positions = np.asarray(features["sample_positions"])
    metadata = features["metadata"]
    if metadata["sample_rate"] != sample_rate or metadata["num_samples"] != len(samples):
        raise ValueError("features must use the same original audio sample grid")
    if metadata["hop_samples"] != config.hop_samples:
        raise ValueError("feature hop and config differ")
    if (positions.ndim != 1 or not len(positions) or positions.dtype.kind not in "iu"
            or positions[0] < 0 or positions[-1] >= len(samples) or np.any(np.diff(positions) <= 0)):
        raise ValueError("invalid feature sample positions")
    for name in set(WEIGHTS["unknown"]) | {"activity"}:
        values = np.asarray(features["features"][name])
        if values.shape != positions.shape or not np.isfinite(values).all():
            raise ValueError("invalid feature: " + name)
    return features, len(samples)


def _normalized(values):
    values = np.maximum(np.asarray(values, dtype=float), 0)
    scale = max(float(np.percentile(values, 95)), float(np.max(values)) * .05)
    return np.minimum(values / scale, 1) if scale > 1e-12 else np.zeros_like(values)


def _fallback(row, reason):
    row["refinement"].update(status="fallback", reason=reason, delta_samples=0)
    row.setdefault("fallback", []).append("dsp:" + reason)
    return row


def refine_boundaries(audio, sample_rate, candidates, config=None, features=None):
    """No annotations are accepted; confidence is deliberately uncalibrated.

    Scoring weights use the coarse model's first phone, not dataset phone labels.
    End times keep their coarse meaning; this stage proposes starts only.
    """
    config = DSPConfig() if config is None else config
    if not isinstance(config, DSPConfig):
        raise ValueError("config must be a DSPConfig")
    feature_data, num_samples = _features(audio, sample_rate, config, features)
    positions = np.asarray(feature_data["sample_positions"])
    values = feature_data["features"]
    outputs = []
    originals = {}
    for original in candidates:
        original = original.to_dict() if hasattr(original, "to_dict") else original
        if not isinstance(original, dict):
            raise ValueError("candidates must be boundary dictionaries")
        row = copy.deepcopy(original)
        originals[id(row)] = original
        if row["original_sample_rate"] != sample_rate or row["original_num_samples"] != num_samples:
            raise ValueError("candidate and audio sample grids differ")
        row["refinement"] = {"method": "dsp", "config": config.to_dict(),
                             "coarse_start_sample": row.get("start_sample"),
                             "coarse_start_seconds": row.get("start_seconds"),
                             "confidence": None, "confidence_status": "uncalibrated",
                             "feature_metadata": feature_data["metadata"]}
        outputs.append(row)
        start = row.get("start_sample")
        if row.get("status") == "unaligned" or start is None or row.get("start_seconds") is None:
            _fallback(row, "coarse_alignment_missing")
            continue
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < num_samples:
            raise ValueError("coarse start must be on the original sample grid")
        seconds = row["start_seconds"]
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
                or abs(seconds * sample_rate - start) > 1):
            raise ValueError("coarse sample and time disagree")
        radius = math.ceil(config.radius_seconds * sample_rate)
        lower, upper = max(0, start - radius), min(num_samples - 1, start + radius)
        window = row.get("search_window")
        if window is not None:
            bounds = (window.get("start_sample"), window.get("end_sample"))
            if (any(isinstance(v, bool) or not isinstance(v, int) for v in bounds)
                    or not 0 <= bounds[0] <= bounds[1] <= num_samples):
                raise ValueError("invalid candidate search window")
            lower = max(lower, window["start_sample"])
            upper = min(upper, window["end_sample"])
        if row.get("end_sample") is not None:
            end = row["end_sample"]
            if isinstance(end, bool) or not isinstance(end, int) or not start <= end <= num_samples:
                raise ValueError("invalid candidate end sample")
            upper = min(upper, row["end_sample"])
        mask = (positions >= lower) & (positions <= upper)
        indices = np.flatnonzero(mask)
        row["refinement"]["search_window_samples"] = [int(lower), int(upper)]
        if len(indices) < 3:
            _fallback(row, "insufficient_search_samples")
            continue
        kind = phoneme_class(row.get("phonemes", []))
        evidence = sum(weight * _normalized(values[name][indices]) for name, weight in WEIGHTS[kind].items())
        activity = np.asarray(values["activity"])[indices]
        evidence *= np.clip(activity, 0, 1)
        distance = np.abs(positions[indices] - start) / max(1, radius)
        score = evidence - config.prior_weight * distance
        near = int(np.argmin(distance))
        best = int(np.argmax(score))
        gain = float(score[best] - score[near])
        row["refinement"].update(phoneme_class=kind, weights=WEIGHTS[kind],
                                 score=float(score[best]), score_kind="local_acoustic_score_uncalibrated",
                                 evidence=float(evidence[best]), score_gain=gain,
                                 candidates=[])
        # Keep separated acoustic alternatives for later global selection and plots.
        selected = []
        separation = max(1, round(.005 * sample_rate))
        for i in sorted(range(len(score)), key=lambda i: (-score[i], positions[indices[i]])):
            sample = int(positions[indices[i]])
            if all(abs(sample - previous) >= separation for previous in selected):
                selected.append(sample)
                row["refinement"]["candidates"].append({"sample": sample, "seconds": sample / sample_rate,
                                                       "score": float(score[i]), "evidence": float(evidence[i])})
                if len(selected) == 8:
                    break
        if float(evidence[best]) < config.min_evidence:
            _fallback(row, "weak_acoustic_evidence")
        elif gain < config.min_gain:
            _fallback(row, "insufficient_score_gain")
        else:
            proposed = int(positions[indices[best]])
            row.update(start_sample=proposed, start_seconds=proposed / sample_rate,
                       confidence=None, confidence_status="uncalibrated_dsp",
                       sample_coordinate_kind="DSP feature center on original decoded sample grid; onset remains uncalibrated")
            row["refinement"].update(status="refined", reason=None, delta_samples=proposed - start)
    # Reject crossing proposals, preserving ties and weak adjacent boundaries.
    changed = True
    while changed:
        changed = False
        valid = [row for row in outputs if row.get("status") != "unaligned" and row.get("start_sample") is not None]
        for left, right in zip(valid, valid[1:]):
            if left["start_sample"] <= right["start_sample"]:
                continue
            refined = [row for row in (left, right) if row["refinement"]["status"] == "refined"]
            if not refined:
                raise ValueError("coarse starts are not monotonic")
            for row in refined:
                row["start_sample"] = row["refinement"]["coarse_start_sample"]
                row["start_seconds"] = row["refinement"]["coarse_start_seconds"]
                original = originals[id(row)]
                row["confidence"] = original.get("confidence")
                row["confidence_status"] = original.get("confidence_status", "uncalibrated")
                row["sample_coordinate_kind"] = original.get("sample_coordinate_kind", "coarse")
                _fallback(row, "crossing_proposals")
            changed = True
    return outputs

"""Shared, original-grid features and coarse-only windows for boundary learning."""
from dataclasses import asdict, dataclass, fields
import math
import re

import numpy as np

from .features import BLOCK_FRAMES, FEATURE_NAMES, _frames, extract_features


ARPABET = ("AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
           "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
           "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH")
PHONE_VOCABULARY = ("<UNK>",) + ARPABET
PHONE_TO_ID = {phone: index for index, phone in enumerate(PHONE_VOCABULARY)}


@dataclass(frozen=True)
class BoundaryConfig:
    schema_version: str = "boundary-tcn-v1"
    hop_seconds: float = .0025
    window_ms: float = 20.0
    mel_bins: int = 48
    mel_min_hz: float = 40.0
    mel_max_hz: float = 12000.0
    power_floor: float = 1e-12
    radius_seconds: float = .35
    window_frames: int = 281
    channels: int = 128
    dilations: tuple = (1, 2, 4, 8, 16, 32)
    kernel_size: int = 3
    embedding_dim: int = 16
    norm_groups: int = 8
    dropout: float = .1

    def __post_init__(self):
        fixed = {"schema_version": "boundary-tcn-v1", "hop_seconds": .0025, "window_ms": 20.0,
                 "mel_bins": 48, "mel_min_hz": 40.0, "mel_max_hz": 12000.0,
                 "power_floor": 1e-12, "radius_seconds": .35, "window_frames": 281,
                 "kernel_size": 3, "embedding_dim": 16}
        for key, expected in fixed.items():
            value = getattr(self, key)
            if (isinstance(value, bool) or value != expected
                    or (type(expected) in (str, int) and type(value) is not type(expected))):
                raise ValueError("Incompatible fixed boundary feature configuration: " + key)
        for key in ("channels", "norm_groups"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(key + " must be a positive integer")
        if self.channels % self.norm_groups:
            raise ValueError("channels must be divisible by norm_groups")
        if (not isinstance(self.dilations, (tuple, list)) or tuple(self.dilations) != (1, 2, 4, 8, 16, 32)
                or any(type(value) is not int for value in self.dilations)):
            raise ValueError("The six fixed TCN dilations are required")
        object.__setattr__(self, "dilations", tuple(self.dilations))
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite in [0, 1)")

    def to_dict(self):
        result = asdict(self)
        result["dilations"] = list(self.dilations)
        return result

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {item.name for item in fields(cls)}:
            raise ValueError("Boundary checkpoint config must contain exactly the versioned fields")
        return cls(**value)


def _config(config):
    if config is None:
        return BoundaryConfig()
    if not isinstance(config, BoundaryConfig):
        raise ValueError("config must be a BoundaryConfig")
    return config


def _mel_filter(sample_rate, fft_size, config):
    if sample_rate / 2 < config.mel_max_hz:
        raise ValueError("Original sample rate cannot represent the fixed 12kHz Mel upper edge")
    lower = 2595 * np.log10(1 + config.mel_min_hz / 700)
    upper = 2595 * np.log10(1 + config.mel_max_hz / 700)
    edges = 700 * (10 ** (np.linspace(lower, upper, config.mel_bins + 2) / 2595) - 1)
    frequency = np.fft.rfftfreq(fft_size, d=1 / sample_rate)
    rising = (frequency[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    falling = (edges[2:, None] - frequency[None, :]) / (edges[2:, None] - edges[1:-1, None])
    weights = np.maximum(0, np.minimum(rising, falling))
    norm = weights.sum(axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("Original FFT grid cannot represent all fixed Mel filters")
    return weights / norm, edges


def extract_neural_features(audio: np.ndarray, sample_rate: int, *, config=None) -> dict:
    """Return 57 channels on the original grid; no resampling or onset correction."""
    config = _config(config)
    if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    sample_rate = int(sample_rate)
    hop = round(sample_rate * config.hop_seconds)
    if hop < 1:
        raise ValueError("Original sample rate has no positive 2.5ms hop")
    dsp = extract_features(audio, sample_rate, hop_samples=hop, window_ms=config.window_ms)
    waveform = np.asarray(audio if audio.ndim == 1 else audio[0], dtype=np.float64)
    peak = dsp["metadata"]["normalization_peak_amplitude"]
    waveform = waveform / peak if peak > 0 else waveform.copy()
    window_samples = dsp["metadata"]["main_window_samples"]
    fft_size = 1 << (window_samples - 1).bit_length()
    weights, edges = _mel_filter(sample_rate, fft_size, config)
    hann = np.hanning(window_samples + 1)[:-1]
    spectrum_scale = hann.sum() ** 2
    positions = dsp["sample_positions"]
    mel = np.zeros((config.mel_bins, len(positions)), dtype=np.float32)
    for start in range(0, len(positions), BLOCK_FRAMES):
        stop = min(len(positions), start + BLOCK_FRAMES)
        frames = _frames(waveform, positions[start:stop], window_samples)
        spectrum = np.fft.rfft(frames * hann, n=fft_size, axis=1)
        power = np.abs(spectrum) ** 2 / spectrum_scale
        mel[:, start:stop] = np.log10(np.maximum(power @ weights.T, config.power_floor)).T
    acoustic = np.concatenate([mel, np.stack([dsp["features"][name] for name in FEATURE_NAMES]).astype(np.float32)])
    metadata = dict(dsp["metadata"], schema_version=config.schema_version, feature_channels=57,
        feature_channel_names=["log_mel_%02d" % index for index in range(config.mel_bins)] + list(FEATURE_NAMES),
        config=config.to_dict(), resampled=False, fft_size=fft_size,
        mel_scale="HTK: 2595*log10(1+Hz/700); fixed 40Hz to 12000Hz",
        mel_filter_edges_hz=edges.tolist(), mel_filter_normalization="triangular Hz filters, discrete unit-sum rows",
        mel_power_normalization="squared FFT magnitude / squared periodic Hann sum after global peak normalization",
        mel_log_transform="log10 Mel power, floor 1e-12; no learned normalization",
        actual_hop_seconds=hop / sample_rate, phone_vocabulary=list(PHONE_VOCABULARY))
    return {"sample_positions": positions, "features": acoustic, "metadata": metadata}


def phone_id(value):
    if isinstance(value, dict):
        value = value.get("label", value.get("phoneme"))
    if not isinstance(value, str):
        return 0
    return PHONE_TO_ID.get(re.sub(r"[012]$", "", value.upper().strip()), 0)


def phone_context(candidates, index):
    """Current first, previous last, next first; missing/foreign phones use UNK."""
    def phones(row):
        values = row.get("phonemes", [])
        return values if isinstance(values, list) else []
    current = phones(candidates[index])
    previous = phones(candidates[index - 1]) if index else []
    following = phones(candidates[index + 1]) if index + 1 < len(candidates) else []
    return [phone_id(current[0]) if current else 0,
            phone_id(previous[-1]) if previous else 0,
            phone_id(following[0]) if following else 0]


def build_candidate_windows(feature_result, candidates, *, config=None) -> dict:
    """Build fixed windows exclusively from coarse start_sample, never labels."""
    config = _config(config)
    metadata = feature_result.get("metadata", {}) if isinstance(feature_result, dict) else {}
    if metadata.get("config") != config.to_dict():
        raise ValueError("Feature and window config differ")
    rate, length, hop = (metadata.get(key) for key in ("sample_rate", "num_samples", "hop_samples"))
    if any(type(value) is not int or value <= 0 for value in (rate, length, hop)) or hop != round(rate * config.hop_seconds):
        raise ValueError("Invalid original feature sample grid")
    positions, acoustic = feature_result.get("sample_positions"), feature_result.get("features")
    expected = np.arange(0, length, hop, dtype=np.int64)
    if (not isinstance(positions, np.ndarray) or positions.dtype.kind not in "iu"
            or not np.array_equal(positions, expected) or not isinstance(acoustic, np.ndarray)
            or acoustic.shape != (57, len(expected)) or acoustic.dtype.kind != "f" or not np.isfinite(acoustic).all()):
        raise ValueError("Invalid feature shape, values or source coordinates")
    rows = [row.to_dict() if hasattr(row, "to_dict") else row for row in candidates]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Candidates must be dictionaries or BoundaryCandidates")
    window_features, masks, samples, contexts, indices, skipped, centers = [], [], [], [], [], [], []
    offsets = np.arange(config.window_frames) - config.window_frames // 2
    for index, row in enumerate(rows):
        if row.get("original_sample_rate") != rate or row.get("original_num_samples") != length:
            raise ValueError("Candidate and features use different original sample grids")
        center, seconds = row.get("start_sample"), row.get("start_seconds")
        if center is None or seconds is None or row.get("status") == "unaligned":
            skipped.append({"candidate_index": index, "reason": "coarse_alignment_missing"})
            continue
        if isinstance(center, bool) or not isinstance(center, (int, np.integer)) or not 0 <= center < length:
            raise ValueError("Coarse start must be on the original sample grid")
        if (isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds)
                or abs(seconds * rate - center) > 1):
            raise ValueError("Coarse sample and seconds disagree")
        center = int(center)
        frame_indices = (center + hop // 2) // hop + offsets
        original_samples = frame_indices * hop
        relative_seconds = (original_samples - center) / rate
        mask = ((frame_indices >= 0) & (frame_indices < len(expected))
                & (np.abs(relative_seconds) <= config.radius_seconds + 1e-12))
        end = row.get("end_sample")
        if end is not None:
            if type(end) is not int or not center <= end <= length:
                raise ValueError("Invalid coarse word end sample")
            end_seconds = row.get("end_seconds")
            if end_seconds is not None and (isinstance(end_seconds, bool)
                    or not isinstance(end_seconds, (int, float)) or not math.isfinite(end_seconds)
                    or abs(end_seconds * rate - end) > 1):
                raise ValueError("Coarse end sample and seconds disagree")
            mask &= original_samples <= end
        if not mask.any():
            skipped.append({"candidate_index": index, "reason": "coarse_window_has_no_feature_centers"})
            continue
        features = np.zeros((58, config.window_frames), dtype=np.float32)
        features[:57, mask] = acoustic[:, frame_indices[mask]]
        features[57] = np.clip(relative_seconds / config.radius_seconds, -1, 1)
        window_features.append(features)
        masks.append(mask)
        samples.append(original_samples)
        contexts.append(phone_context(rows, index))
        indices.append(index)
        centers.append(center)
    count = len(indices)
    return {"features": np.stack(window_features) if count else np.empty((0, 58, config.window_frames), np.float32),
            "mask": np.stack(masks) if count else np.empty((0, config.window_frames), bool),
            "sample_positions": np.stack(samples) if count else np.empty((0, config.window_frames), np.int64),
            "phones": np.asarray(contexts, dtype=np.int64).reshape(count, 3),
            "candidate_indices": indices, "coarse_start_samples": centers, "skipped": skipped,
            "metadata": {"config": config.to_dict(), "sample_rate": rate, "num_samples": length,
                "hop_samples": hop, "center_source": "coarse start_sample only; no source or human boundary labels",
                "nearest_grid_rounding": "integer half-up, preserved offset in relative_position",
                "relative_position_channel": "(original sample position - coarse start_sample)/sample_rate/radius_seconds",
                "padding": "zero acoustic channels outside original audio, physical radius or coarse word end; mask=False",
                "phone_context_order": ["current_first", "previous_last", "next_first"],
                "phone_vocabulary": list(PHONE_VOCABULARY)}}

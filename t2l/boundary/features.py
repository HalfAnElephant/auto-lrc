"""Centered acoustic screening features on the original sample coordinate grid."""
import math

import numpy as np


POWER_FLOOR = 1e-12
BLOCK_FRAMES = 256
F0_RANGE_HZ = (80.0, 1000.0)
FEATURE_NAMES = (
    "log_energy", "energy_rise", "spectral_flux", "high_frequency", "high_frequency_rise",
    "periodicity", "periodicity_rise", "activity", "spectral_change",
)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _frames(audio, positions, window_samples):
    offsets = np.arange(window_samples, dtype=np.int64) - window_samples // 2
    indices = positions[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < len(audio))
    return np.where(valid, audio[np.clip(indices, 0, len(audio) - 1)], 0.0)


def _spectrum(frames):
    window = np.hanning(frames.shape[1] + 1)[:-1]
    fft_size = 1 << (frames.shape[1] - 1).bit_length()
    return np.abs(np.fft.rfft(frames * window, n=fft_size, axis=1)), fft_size


def _periodicity(frames, minimum_lag, maximum_lag):
    """Overlap-normalized autocorrelation peaks; neither F0 nor confidence."""
    if maximum_lag < minimum_lag:
        return np.zeros(len(frames), dtype=np.float64)
    centered = frames - frames.mean(axis=1, keepdims=True)
    fft_size = 1 << (2 * frames.shape[1] - 2).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size, axis=1)
    correlation = np.fft.irfft(spectrum * spectrum.conj(), n=fft_size, axis=1)
    energy = np.concatenate([np.zeros((len(frames), 1)), np.cumsum(centered * centered, axis=1)], axis=1)
    lags = np.arange(minimum_lag, maximum_lag + 1)
    denominator = np.sqrt(np.maximum(0, energy[:, frames.shape[1] - lags])
                          * np.maximum(0, energy[:, -1, None] - energy[:, lags]))
    coefficients = np.divide(correlation[:, lags], denominator,
                             out=np.zeros_like(denominator), where=denominator > POWER_FLOOR)
    return np.clip(coefficients.max(axis=1), 0.0, 1.0)


def _positive_change(values):
    return np.maximum(np.diff(values, prepend=values[0]), 0.0)


def extract_features(audio: np.ndarray, sample_rate: int, hop_samples: int = 120,
                     window_ms: float = 20.0) -> dict:
    """Extract screening cues; positions are centered original samples, not onsets.

    Mono [N] and channel-first [C, N] inputs use the first channel. Main spectral
    and periodicity windows use ``window_ms``; onset energy/high-frequency cues
    use an auxiliary window capped at 5ms. Every window is zero padded, and no
    latency estimate or waveform shift is applied.
    """
    sample_rate = _integer(sample_rate, "sample_rate")
    hop_samples = _integer(hop_samples, "hop_samples")
    if (isinstance(window_ms, (bool, np.bool_))
            or not isinstance(window_ms, (int, float, np.integer, np.floating))
            or not math.isfinite(float(window_ms)) or window_ms <= 0):
        raise ValueError("window_ms must be finite and positive")
    window_ms = float(window_ms)
    window_samples = round(sample_rate * window_ms / 1000)
    if window_samples < 2:
        raise ValueError("window_ms must contain at least two samples")
    if not isinstance(audio, np.ndarray) or audio.dtype.kind not in "fiu":
        raise ValueError("audio must be a real numeric numpy array")
    if (audio.ndim not in (1, 2) or audio.size == 0
            or (audio.ndim == 2 and (audio.shape[0] == 0 or audio.shape[1] == 0))
            or not np.isfinite(audio).all()):
        raise ValueError("audio must be non-empty, finite mono or channel-first audio")
    waveform = np.asarray(audio if audio.ndim == 1 else audio[0], dtype=np.float64)
    peak = float(np.max(np.abs(waveform)))
    waveform = waveform / peak if peak > 0 else waveform.copy()
    positions = np.arange(0, len(waveform), hop_samples, dtype=np.int64)
    auxiliary_samples = max(2, min(window_samples, round(sample_rate * .005)))
    minimum_lag = max(1, math.ceil(sample_rate / F0_RANGE_HZ[1]))
    maximum_lag = min(window_samples - 2, math.floor(sample_rate / F0_RANGE_HZ[0]))
    high_frequency_hz = min(2000.0, sample_rate / 4)
    features = {name: np.zeros(len(positions), dtype=np.float64) for name in FEATURE_NAMES}
    short_log_energy = np.zeros(len(positions), dtype=np.float64)
    previous_spectrum = None
    for start in range(0, len(positions), BLOCK_FRAMES):
        stop = min(start + BLOCK_FRAMES, len(positions))
        main = _frames(waveform, positions[start:stop], window_samples)
        short = _frames(waveform, positions[start:stop], auxiliary_samples)
        power = np.mean(main * main, axis=1)
        short_power = np.mean(short * short, axis=1)
        features["log_energy"][start:stop] = np.log10(np.maximum(power, POWER_FLOOR))
        short_log_energy[start:stop] = np.log10(np.maximum(short_power, POWER_FLOOR))
        # Energy activity is a fixed, relative-level heuristic, not a posterior.
        features["activity"][start:stop] = np.clip((10 * short_log_energy[start:stop] + 60) / 40, 0, 1)
        magnitude, _ = _spectrum(main)
        total = magnitude.sum(axis=1, keepdims=True)
        normalized = np.divide(magnitude, total, out=np.zeros_like(magnitude), where=total > 0)
        previous = normalized[0] if previous_spectrum is None else previous_spectrum
        delta = np.diff(normalized, axis=0, prepend=previous[None, :])
        features["spectral_flux"][start:stop] = np.clip(np.maximum(delta, 0).sum(axis=1), 0, 1)
        features["spectral_change"][start:stop] = np.clip(.5 * np.abs(delta).sum(axis=1), 0, 1)
        previous_spectrum = normalized[-1].copy()
        short_magnitude, short_fft_size = _spectrum(short)
        spectral_power = short_magnitude * short_magnitude
        frequencies = np.fft.rfftfreq(short_fft_size, d=1 / sample_rate)
        spectral_total = spectral_power.sum(axis=1)
        features["high_frequency"][start:stop] = np.divide(
            spectral_power[:, frequencies >= high_frequency_hz].sum(axis=1), spectral_total,
            out=np.zeros_like(spectral_total), where=spectral_total > 0)
        features["periodicity"][start:stop] = _periodicity(main, minimum_lag, maximum_lag)
    features["energy_rise"] = np.clip(_positive_change(short_log_energy) / -math.log10(POWER_FLOOR), 0, 1)
    for name in ("high_frequency", "periodicity"):
        features[name + "_rise"] = np.clip(_positive_change(features[name]), 0, 1)
    metadata = {
        "sample_rate": sample_rate, "num_samples": len(waveform), "hop_samples": hop_samples,
        "hop_seconds": hop_samples / sample_rate, "window_ms_requested": window_ms,
        "main_window_samples": window_samples, "main_window_seconds": window_samples / sample_rate,
        "auxiliary_window_samples": auxiliary_samples, "auxiliary_window_seconds": auxiliary_samples / sample_rate,
        "window_alignment": "centered on sample_positions; zero padding outside original audio",
        "main_window_offsets_samples": [-(window_samples // 2), window_samples - window_samples // 2],
        "auxiliary_window_offsets_samples": [-(auxiliary_samples // 2),
                                              auxiliary_samples - auxiliary_samples // 2],
        "offset_interval_convention": "half-open",
        "spectral_window": "periodic Hann, next-power-of-two FFT, magnitude spectrum",
        "channel_policy": "first channel, matching the existing alignment runtime",
        "input_channels": 1 if audio.ndim == 1 else int(audio.shape[0]),
        "amplitude_normalization": "divide first channel by its global absolute peak; zero waveform stays zero",
        "normalization_peak_amplitude": peak, "power_floor": POWER_FLOOR,
        "feature_ranges": {name: [-12.0, 0.0] if name == "log_energy" else [0.0, 1.0]
                           for name in FEATURE_NAMES},
        "feature_definitions": {
            "log_energy": "log10 mean squared amplitude in the unweighted main window, floored",
            "energy_rise": "positive first difference of auxiliary log10 power / 12; first point zero",
            "spectral_flux": "positive L1 difference of main spectra normalized to unit magnitude sum",
            "spectral_change": "half the L1 difference of main spectra normalized to unit magnitude sum",
            "high_frequency": "auxiliary Hann spectral power above cutoff / total spectral power",
            "high_frequency_rise": "positive first difference of high_frequency; first point zero",
            "periodicity": "maximum nonnegative overlap-normalized autocorrelation of demeaned main windows",
            "periodicity_rise": "positive first difference of periodicity; first point zero",
            "activity": "clip((relative power dB + 60) / 40, 0, 1) using auxiliary unweighted power; not probability",
        },
        "high_frequency_cutoff_hz": high_frequency_hz,
        "activity_relative_power_db_range": [-60.0, -20.0],
        "energy_rise_log10_scale": -math.log10(POWER_FLOOR),
        "periodicity_f0_search_hz": list(F0_RANGE_HZ),
        "periodicity_lag_samples": [minimum_lag, maximum_lag] if maximum_lag >= minimum_lag else None,
        "periodicity_unavailable_value": 0.0,
        "periodicity_warning": "Auxiliary correlation cue, not pitch tracking, voicing probability or boundary confidence",
        "activity_calibrated": False, "feature_step_is_accuracy_guarantee": False,
        "latency_correction_samples": 0, "latency_calibrated": False,
        "latency_warning": "Centered windows see future samples and can respond before an onset; no guessed correction is applied",
        "block_frames": BLOCK_FRAMES,
    }
    return {"sample_positions": positions, "features": features, "metadata": metadata}

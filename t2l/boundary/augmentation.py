"""Seeded train-only waveform perturbations with the original sample timeline."""
import hashlib
import json
import math

import numpy as np


AUGMENTATION_SCHEMA = "training-waveform-augmentation-v1"
APPLICATION_PROBABILITIES = {"noise": .5, "reverb": .35, "volume": .8, "accompaniment": .5}


def _rms(waveform):
    peak = float(np.max(np.abs(waveform)))
    return peak * math.sqrt(float(np.mean((waveform / peak) ** 2))) if peak else 0.0


def _state_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Unsupported RNG state value")


def _component(waveform, reference_rms, snr_db):
    component_rms = _rms(waveform)
    if not component_rms:
        return np.zeros_like(waveform), 0.0
    gain = reference_rms * 10 ** (-snr_db / 20) / component_rms
    return waveform * gain, gain


def _synthetic_backing(length, sample_rate, rng):
    # Original bass/triad oscillators; this is not an instrumental recording.
    root_midi = int(rng.integers(40, 53))
    root_hz = 440 * 2 ** ((root_midi - 69) / 12)
    intervals = [0, 4, 7] if rng.random() < .5 else [0, 3, 7]
    frequencies = [root_hz / 2] + [root_hz * 2 ** (semitones / 12) for semitones in intervals]
    frequencies = [float(value) for value in frequencies if value < sample_rate * .45]
    phases = rng.uniform(0, 2 * np.pi, len(frequencies)).tolist()
    tempo_bpm = float(rng.uniform(60, 110))
    envelope_phase = float(rng.uniform(0, 2 * np.pi))
    time = np.arange(length, dtype=np.float64) / sample_rate
    backing = np.zeros(length, dtype=np.float64)
    for frequency, phase in zip(frequencies, phases):
        backing += np.sin(2 * np.pi * frequency * time + phase) / max(1, len(frequencies))
    backing *= .6 + .4 * np.sin(2 * np.pi * tempo_bpm / 60 * time + envelope_phase) ** 2
    return backing, {"generator": "original sine bass/triad with smooth periodic level envelope",
                     "root_midi": root_midi, "frequencies_hz": frequencies,
                     "phases_radians": phases, "tempo_bpm": tempo_bpm,
                     "envelope_phase_radians": envelope_phase,
                     "provenance": "synthetic original oscillators, not actual instrumentation or external audio"}


def augment_training_audio(audio: np.ndarray, sample_rate: int, rng: np.random.Generator, *,
                           noise=True, reverb=True, volume=True, accompaniment=True):
    """Return finite float32 mono audio and JSON metadata, without shifting labels.

    The caller owns the seed and must restrict this function to training audio.
    Enabled kinds are sampled independently at the documented probabilities.
    Causal reflection taps have positive lags and a unit direct path before the
    explicitly recorded global gains. No resampling, wrap, trim or delay
    correction occurs. Synthetic perturbations do not calibrate vocal onsets.
    """
    if (not isinstance(audio, np.ndarray) or audio.ndim != 1 or not audio.size
            or audio.dtype.kind not in "fiu" or not np.isfinite(audio).all()):
        raise ValueError("audio must be non-empty finite real mono numpy audio")
    if (isinstance(sample_rate, (bool, np.bool_)) or not isinstance(sample_rate, (int, np.integer))
            or sample_rate <= 0):
        raise ValueError("sample_rate must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a seeded numpy Generator")
    enabled = {"noise": noise, "reverb": reverb, "volume": volume, "accompaniment": accompaniment}
    if any(type(value) is not bool for value in enabled.values()):
        raise ValueError("augmentation switches must be booleans")
    waveform = audio.astype(np.float64, copy=True)
    source_peak = float(np.max(np.abs(waveform)))
    if not math.isfinite(source_peak) or source_peak > np.finfo(np.float32).max:
        raise ValueError("audio amplitude cannot be represented in float32")
    sample_rate = int(sample_rate)
    source_rms = _rms(waveform)
    initial_state = json.dumps(rng.bit_generator.state, default=_state_json, sort_keys=True, separators=(",", ":"))
    metadata = {"schema_version": AUGMENTATION_SCHEMA, "sample_rate": sample_rate,
                "num_samples": len(waveform), "input_dtype": str(audio.dtype), "output_dtype": "float32",
                "source_peak": source_peak, "source_rms": source_rms,
                "rng_bit_generator": type(rng.bit_generator).__name__,
                "rng_state_sha256_before": hashlib.sha256(initial_state.encode("ascii")).hexdigest(),
                "source_grid_offset_samples": 0, "sample_count_changed": False, "resampled": False,
                "boundary_labels": "original samples unchanged; no label-derived waveform operations",
                "onset_semantics": "synthetic perturbation, not a vocal onset calibration",
                "operation_order": ["volume", "reverb", "noise", "accompaniment", "global_peak_safety_gain"]}
    for kind in enabled:
        metadata[kind] = {"enabled": enabled[kind], "applied": False,
                          "application_probability": APPLICATION_PROBABILITIES[kind]}
    applied = {kind: enabled[kind] and rng.random() < APPLICATION_PROBABILITIES[kind] for kind in enabled}
    volume_gain = 1.0
    if applied["volume"]:
        volume_gain = float(rng.uniform(.8, 1.2))
        waveform *= volume_gain
        metadata["volume"].update(applied=True, gain=volume_gain, gain_range=[.8, 1.2])
    reference_rms = source_rms * volume_gain
    if applied["reverb"]:
        maximum_delay = min(len(waveform) - 1, max(1, round(sample_rate * .08)))
        minimum_delay = max(1, min(maximum_delay, round(sample_rate * .008)))
        if maximum_delay < 1:
            metadata["reverb"]["reason"] = "audio_too_short_for_positive_reflection_delay"
        else:
            count = int(rng.integers(2, 6))
            delays = sorted(set(int(value) for value in rng.integers(minimum_delay, maximum_delay + 1, count)))
            dry = waveform.copy()
            taps = []
            for delay in delays:
                gain = float(rng.uniform(.03, .16) * np.exp(-delay / max(1, sample_rate * .08)))
                waveform[delay:] += gain * dry[:-delay]
                taps.append({"delay_samples": delay, "delay_seconds": delay / sample_rate, "gain": gain})
            metadata["reverb"].update(applied=True, direct_path_gain=1.0, taps=taps,
                impulse_origin_sample=0, tail="causal delayed reflections truncated at original end; no circular wrap",
                direct_path_semantics="impulse[0]=1 before volume and final global peak safety gain")
    if applied["noise"]:
        if reference_rms == 0:
            metadata["noise"]["reason"] = "silent_reference_kept_silent"
        else:
            snr = float(rng.uniform(20, 35))
            component, gain = _component(rng.standard_normal(len(waveform)), reference_rms, snr)
            waveform += component
            metadata["noise"].update(applied=True, kind="original seeded Gaussian white noise",
                                      snr_db=snr, snr_range_db=[20, 35], component_gain=gain,
                                      snr_reference="original waveform RMS after sampled volume gain")
    if applied["accompaniment"]:
        if reference_rms == 0:
            metadata["accompaniment"]["reason"] = "silent_reference_kept_silent"
        else:
            component, details = _synthetic_backing(len(waveform), sample_rate, rng)
            if not details["frequencies_hz"] or not _rms(component):
                metadata["accompaniment"]["reason"] = "sample_rate_or_length_cannot_represent_synthetic_tones"
            else:
                snr = float(rng.uniform(20, 30))
                component, gain = _component(component, reference_rms, snr)
                waveform += component
                metadata["accompaniment"].update(details, applied=True, snr_db=snr,
                    snr_range_db=[20, 30], component_gain=gain,
                    snr_reference="original waveform RMS after sampled volume gain")
    peak = float(np.max(np.abs(waveform)))
    safety_gain = 1 / peak if peak > 1 else 1.0
    waveform *= safety_gain
    output = waveform.astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Augmentation produced nonfinite output")
    metadata.update(global_peak_safety_gain=safety_gain, output_peak=float(np.max(np.abs(output))),
                    output_rms=_rms(output.astype(np.float64)))
    return output, metadata

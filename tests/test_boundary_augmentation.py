import json

import numpy as np
import pytest

from t2l.boundary.augmentation import augment_training_audio


def augment(audio, seed=7, **kwargs):
    return augment_training_audio(audio, 48000, np.random.default_rng(seed), **kwargs)


def enabled_only(kind):
    return {name: name == kind for name in ("noise", "reverb", "volume", "accompaniment")}


def with_applied_kind(audio, kind):
    for seed in range(100):
        output, metadata = augment(audio, seed, **enabled_only(kind))
        if metadata[kind]["applied"]:
            return output, metadata
    raise AssertionError("No seeded application found")


def test_determinism_metadata_serializable_and_input_not_mutated():
    audio = (.1 * np.sin(np.arange(4800) * .05)).astype(np.float32)
    original = audio.copy()
    first, first_metadata = augment(audio)
    second, second_metadata = augment(audio)
    assert np.array_equal(first, second)
    assert first_metadata == second_metadata
    assert np.array_equal(audio, original)
    assert first.shape == audio.shape and first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.max(np.abs(first)) <= 1
    assert first_metadata["source_grid_offset_samples"] == 0
    assert first_metadata["sample_count_changed"] is False
    assert first_metadata["resampled"] is False
    assert "not a vocal onset calibration" in first_metadata["onset_semantics"]
    json.dumps(first_metadata, allow_nan=False)


def test_different_seed_states_are_observable_and_waveforms_can_differ():
    audio = np.ones(4800, dtype=np.float32) * .1
    one, metadata_one = augment(audio, 1)
    two, metadata_two = augment(audio, 2)
    assert metadata_one["rng_state_sha256_before"] != metadata_two["rng_state_sha256_before"]
    assert not np.array_equal(one, two)


def test_all_disabled_returns_independent_float32_copy_on_original_grid():
    audio = np.linspace(-1, 1, 101)
    output, metadata = augment(audio, noise=False, reverb=False, volume=False, accompaniment=False)
    assert np.array_equal(output, audio.astype(np.float32))
    assert not np.shares_memory(output, audio)
    assert metadata["global_peak_safety_gain"] == 1
    assert all(not metadata[kind]["applied"] for kind in enabled_only("noise"))


@pytest.mark.parametrize("length", [1, 19, 4800])
def test_silence_stays_silent_for_all_perturbations(length):
    for seed in range(10):
        output, metadata = augment(np.zeros(length), seed)
        assert output.shape == (length,)
        assert np.all(output == 0)
        assert metadata["output_rms"] == 0
        json.dumps(metadata, allow_nan=False)


def test_causal_direct_path_impulse_and_no_precursor_or_wrap():
    audio = np.zeros(12000)
    onset = 2000
    audio[onset] = .25
    output, metadata = with_applied_kind(audio, "reverb")
    assert np.all(output[:onset] == 0)
    assert output[onset] == .25
    assert metadata["reverb"]["direct_path_gain"] == 1
    assert metadata["reverb"]["impulse_origin_sample"] == 0
    assert metadata["global_peak_safety_gain"] == 1
    for tap in metadata["reverb"]["taps"]:
        assert tap["delay_samples"] > 0
        assert output[onset + tap["delay_samples"]] == pytest.approx(.25 * tap["gain"])
    final_impulse = np.zeros(12000)
    final_impulse[-1] = .25
    tail, tail_metadata = with_applied_kind(final_impulse, "reverb")
    assert np.all(tail[:-1] == 0)
    assert tail[-1] == .25
    assert "no circular wrap" in tail_metadata["reverb"]["tail"]


def test_volume_is_one_global_gain_without_timeline_movement():
    audio = np.zeros(4800)
    audio[200:600] = .1
    output, metadata = with_applied_kind(audio, "volume")
    assert np.all(output[:200] == 0) and np.all(output[600:] == 0)
    assert np.allclose(output[200:600], .1 * metadata["volume"]["gain"])
    assert .8 <= metadata["volume"]["gain"] <= 1.2


@pytest.mark.parametrize("kind,snr_bounds", [("noise", (20, 35)), ("accompaniment", (20, 30))])
def test_components_have_declared_realized_snr_without_time_shift(kind, snr_bounds):
    audio = np.ones(24000) * .1
    output, metadata = with_applied_kind(audio, kind)
    component = output.astype(np.float64) - audio
    snr = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) / np.sqrt(np.mean(component ** 2)))
    assert snr == pytest.approx(metadata[kind]["snr_db"], abs=1e-4)
    assert snr_bounds[0] <= snr <= snr_bounds[1]
    assert output.shape == audio.shape
    if kind == "accompaniment":
        assert "not actual instrumentation" in metadata[kind]["provenance"]
        assert all(frequency < 48000 * .45 for frequency in metadata[kind]["frequencies_hz"])


def test_final_gain_bounds_loud_audio_without_clipping_or_sample_shift():
    audio = np.zeros(12000)
    audio[100:1000] = 1
    for seed in range(20):
        output, metadata = augment(audio, seed)
        assert output.shape == audio.shape and np.isfinite(output).all()
        assert np.max(np.abs(output)) <= 1
        assert 0 < metadata["global_peak_safety_gain"] <= 1


def test_alternate_numpy_bit_generator_metadata_and_short_low_rate_are_supported():
    rng = np.random.Generator(np.random.MT19937(7))
    output, metadata = augment_training_audio(np.ones(3) * .1, 2, rng)
    assert output.shape == (3,) and np.isfinite(output).all()
    assert metadata["rng_bit_generator"] == "MT19937"
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("audio", [[], np.array([]), np.ones((2, 10)), np.array([np.nan]),
    np.array([np.inf]), np.array([1 + 1j]), np.array([True]), np.array([1e100])])
def test_invalid_audio_rejected(audio):
    with pytest.raises(ValueError):
        augment(audio)


@pytest.mark.parametrize("rate", [0, -1, 48000., True])
def test_invalid_sample_rate_rejected(rate):
    with pytest.raises(ValueError):
        augment_training_audio(np.ones(10), rate, np.random.default_rng(7))


def test_invalid_rng_or_nonboolean_switch_rejected():
    with pytest.raises(ValueError, match="Generator"):
        augment_training_audio(np.ones(10), 48000, np.random.RandomState(7))
    with pytest.raises(ValueError, match="booleans"):
        augment(np.ones(10), noise=1)

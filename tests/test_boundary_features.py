import json

import numpy as np
import pytest

from t2l.boundary import extract_features
from t2l.boundary.features import FEATURE_NAMES


def test_silence_has_original_sample_centers_finite_features_and_no_activity():
    result = extract_features(np.zeros(1001), 48000)
    assert np.array_equal(result["sample_positions"], np.arange(0, 1001, 120))
    assert result["sample_positions"].dtype.kind in "iu"
    assert set(result["features"]) == set(FEATURE_NAMES)
    for name, values in result["features"].items():
        assert values.shape == result["sample_positions"].shape
        assert np.isfinite(values).all()
        assert np.array_equal(values, np.full(len(values), -12 if name == "log_energy" else 0))
    assert result["metadata"]["latency_calibrated"] is False
    assert result["metadata"]["activity_calibrated"] is False
    assert result["metadata"]["feature_step_is_accuracy_guarantee"] is False
    json.dumps(result["metadata"], allow_nan=False)


def test_first_channel_policy_is_identical_to_mono_and_does_not_mutate_input():
    signal = np.random.RandomState(7).standard_normal(4000)
    audio = np.vstack([signal, np.zeros_like(signal)])
    original = audio.copy()
    mono = extract_features(signal, 48000)
    stereo = extract_features(audio, 48000)
    for name in FEATURE_NAMES:
        assert np.array_equal(mono["features"][name], stereo["features"][name])
    assert stereo["metadata"]["input_channels"] == 2
    assert np.array_equal(audio, original)


def test_features_are_stable_under_tiny_and_huge_nonzero_amplitude_scaling():
    signal = np.sin(2 * np.pi * 220 * np.arange(5000) / 48000)
    baseline = extract_features(signal, 48000)
    for factor in (1e-200, 1e200):
        scaled = extract_features(signal * factor, 48000)
        for name in FEATURE_NAMES:
            assert np.allclose(scaled["features"][name], baseline["features"][name], atol=1e-10)
        assert scaled["metadata"]["normalization_peak_amplitude"] == pytest.approx(factor)


def test_periodicity_distinguishes_tone_from_noise_without_claiming_pitch():
    rate = 48000
    tone = np.sin(2 * np.pi * 200 * np.arange(rate) / rate)
    noise = np.random.RandomState(7).standard_normal(rate)
    tone_features = extract_features(tone, rate)
    noise_features = extract_features(noise, rate)
    interior = slice(10, -10)
    assert np.median(tone_features["features"]["periodicity"][interior]) > .9
    assert np.median(noise_features["features"]["periodicity"][interior]) < .3
    assert tone_features["metadata"]["periodicity_lag_samples"] == [48, 600]
    assert "not pitch tracking" in tone_features["metadata"]["periodicity_warning"]


def test_short_high_frequency_window_separates_low_and_high_tones():
    rate = 48000
    time = np.arange(12000) / rate
    low = extract_features(np.sin(2 * np.pi * 300 * time), rate)
    high = extract_features(np.sin(2 * np.pi * 6000 * time), rate)
    assert np.median(low["features"]["high_frequency"][4:-4]) < .01
    assert np.median(high["features"]["high_frequency"][4:-4]) > .99
    assert low["metadata"]["main_window_samples"] == 960
    assert low["metadata"]["auxiliary_window_samples"] == 240
    assert low["metadata"]["auxiliary_window_offsets_samples"] == [-120, 120]
    assert low["metadata"]["activity_relative_power_db_range"] == [-60, -20]


def test_centered_impulse_response_does_not_move_coordinate_grid():
    signal = np.zeros(5000)
    signal[2400] = 1
    result = extract_features(signal, 48000)
    positions = result["sample_positions"]
    energy = result["features"]["log_energy"]
    supported = positions[energy > -12]
    assert supported.min() < 2400 < supported.max()
    assert supported.min() >= 2400 - 480 and supported.max() <= 2400 + 480
    assert result["metadata"]["main_window_offsets_samples"] == [-480, 480]
    assert result["metadata"]["latency_correction_samples"] == 0
    assert "see future samples" in result["metadata"]["latency_warning"]


def test_changes_are_nonnegative_and_first_sample_has_no_previous_frame():
    signal = np.r_[np.zeros(3000), np.sin(2 * np.pi * 6000 * np.arange(3000) / 48000)]
    result = extract_features(signal, 48000)
    for name in ("energy_rise", "high_frequency_rise", "periodicity_rise", "spectral_flux", "spectral_change"):
        assert result["features"][name][0] == 0
        assert np.min(result["features"][name]) >= 0
    assert result["features"]["energy_rise"].max() > 0
    assert result["features"]["high_frequency_rise"].max() > 0
    assert result["features"]["spectral_flux"].max() > 0


def test_fft_block_boundary_does_not_reset_flux():
    signal = np.random.RandomState(7).standard_normal(40000)
    dense = extract_features(signal, 48000, hop_samples=120)
    assert len(dense["sample_positions"]) > 256
    # Extract a local slice with the same centers far from its padded edges.
    beginning = 250 * 120
    local = extract_features(signal[beginning:beginning + 3000], 48000, hop_samples=120)
    for name in ("spectral_flux", "spectral_change"):
        assert dense["features"][name][256] == pytest.approx(local["features"][name][6])


@pytest.mark.parametrize("length,hop", [(1, 120), (120, 120), (121, 120), (999, 5000)])
def test_short_audio_and_hop_larger_than_audio_still_have_one_or_more_finite_centers(length, hop):
    result = extract_features(np.ones(length), 48000, hop_samples=hop)
    assert result["sample_positions"][0] == 0
    assert np.all(result["sample_positions"] < length)
    assert all(np.isfinite(value).all() for value in result["features"].values())


def test_odd_main_window_offsets_match_half_open_centered_definition():
    result = extract_features(np.ones(1000), 48000, window_ms=20.02)
    assert result["metadata"]["main_window_samples"] == 961
    assert result["metadata"]["main_window_offsets_samples"] == [-480, 481]


@pytest.mark.parametrize("audio", [np.array([]), np.empty((0, 3)), np.empty((2, 0)), np.ones((1, 2, 3)),
                                  np.array([np.nan]), np.array([np.inf]), np.array([1 + 2j]),
                                  np.array([True]), np.array(["1"]), [1, 2],
                                  np.array([[1., 2.], [np.nan, 0.]])])
def test_invalid_audio_is_rejected(audio):
    with pytest.raises(ValueError):
        extract_features(audio, 48000)


@pytest.mark.parametrize("kwargs", [{"sample_rate": 0}, {"sample_rate": 48000.0}, {"sample_rate": True},
                                    {"hop_samples": 0}, {"hop_samples": -1}, {"hop_samples": 1.5},
                                    {"hop_samples": True}, {"window_ms": 0}, {"window_ms": -1},
                                    {"window_ms": float("nan")}, {"window_ms": float("inf")},
                                    {"window_ms": True}, {"window_ms": .001},
                                    {"window_ms": complex(20, 1)}, {"window_ms": np.complex128(20)}])
def test_invalid_rate_hop_and_window_are_rejected(kwargs):
    options = {"sample_rate": 48000, **kwargs}
    with pytest.raises(ValueError):
        extract_features(np.ones(100), **options)

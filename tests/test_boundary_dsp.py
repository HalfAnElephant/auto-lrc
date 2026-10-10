import copy

import numpy as np
import pytest

from t2l.boundary.dsp import DSPConfig, WEIGHTS, phoneme_class, refine_boundaries


def candidate(start=4800, end=15000, **changes):
    return dict(text="test", phonemes=["T", "EH"], original_sample_rate=48000,
                original_num_samples=24000, start_sample=start, start_seconds=start / 48000,
                end_sample=end, end_seconds=end / 48000, status="candidate",
                confidence=None, confidence_status="uncalibrated", fallback=[], **changes)


def features(peak=6000):
    positions = np.arange(0, 24000, 120)
    values = {name: np.zeros(len(positions)) for name in WEIGHTS["unknown"]}
    for value in values.values():
        value[positions == peak] = 1
    values["activity"] = np.ones(len(positions))
    return {"sample_positions": positions, "features": values,
            "metadata": {"sample_rate": 48000, "num_samples": 24000, "hop_samples": 120}}


def refine(rows, feature_data=None, **config):
    return refine_boundaries(np.ones(24000), 48000, rows, DSPConfig(**config),
                             features=features() if feature_data is None else feature_data)


def test_proposal_keeps_raw_data_and_uses_original_sample_coordinate():
    original = [candidate()]
    snapshot = copy.deepcopy(original)
    result = refine(original)[0]
    assert original == snapshot
    assert result["start_sample"] == 6000
    assert result["start_seconds"] == .125
    assert result["end_sample"] == snapshot[0]["end_sample"]
    assert result["confidence"] is None
    assert result["refinement"]["delta_samples"] == 1200
    assert result["refinement"]["coarse_start_seconds"] == .1
    assert result["refinement"]["status"] == "refined"


def test_annotation_like_extra_fields_cannot_change_predictions():
    first = candidate()
    second = dict(first, boundary={"candidate_sample": 111}, human_truth_sample=222)
    third = dict(first, boundary={"candidate_sample": 9000}, human_truth_sample=9999)
    results = [refine([row])[0] for row in (first, second, third)]
    assert len({row["start_sample"] for row in results}) == 1
    assert all(row["refinement"] == results[0]["refinement"] for row in results)


def test_silence_and_weak_feature_evidence_preserve_exact_coarse_seconds():
    data = features()
    data["features"]["activity"][:] = 0
    row = candidate(start=4801)
    result = refine([row], data)[0]
    assert result["start_sample"] == 4801
    assert result["start_seconds"] == row["start_seconds"]
    assert result["refinement"]["reason"] == "weak_acoustic_evidence"
    assert result["fallback"] == ["dsp:weak_acoustic_evidence"]


def test_coarse_near_best_retains_exact_unrounded_time():
    row = candidate(start=6001)
    result = refine([row])[0]
    assert result["start_sample"] == 6001
    assert result["start_seconds"] == 6001 / 48000
    assert result["refinement"]["reason"] == "insufficient_score_gain"


def test_search_window_and_end_bound_refuse_outside_peak():
    row = candidate(end=5400, search_window={"start_sample": 4000, "end_sample": 5400})
    result = refine([row])[0]
    assert result["start_sample"] == row["start_sample"]
    assert result["refinement"]["search_window_samples"] == [4000, 5400]


def test_unaligned_candidates_remain_missing():
    row = candidate()
    row.update(start_sample=None, start_seconds=None, status="unaligned")
    result = refine([row])[0]
    assert result["start_sample"] is result["start_seconds"] is None
    assert result["status"] == "unaligned"
    assert result["refinement"]["reason"] == "coarse_alignment_missing"


def test_crossing_proposals_fall_back_and_adjacent_equal_times_are_allowed():
    rows = [candidate(start=4800), candidate(start=5700)]
    data = features()
    # First candidate reaches peak; second has no gain because the peak is excluded.
    rows[1]["search_window"] = {"start_sample": 5400, "end_sample": 5880}
    result = refine(rows, data)
    assert [r["start_sample"] for r in result] == [4800, 5700]
    assert result[0]["refinement"]["reason"] == "crossing_proposals"
    tied = refine([candidate(start=6001), candidate(start=6001)])
    assert [r["start_sample"] for r in tied] == [6001, 6001]


@pytest.mark.parametrize("key,value", [("sample_rate", 22050), ("num_samples", 22000), ("hop_samples", 240)])
def test_feature_grid_mismatch_is_explicit(key, value):
    data = features()
    data["metadata"][key] = value
    with pytest.raises(ValueError, match="grid|hop"):
        refine([candidate()], data)


def test_candidate_grid_mismatch_is_explicit():
    row = candidate()
    row["original_sample_rate"] = 22050
    with pytest.raises(ValueError, match="sample grids"):
        refine([row])


def test_corrupt_feature_values_do_not_become_scores():
    data = features()
    data["features"]["spectral_flux"][3] = np.nan
    with pytest.raises(ValueError, match="invalid feature"):
        refine([candidate()], data)


@pytest.mark.parametrize("kwargs", [{"min_gain": -1}, {"prior_weight": float("nan")},
                                     {"radius_seconds": 0}, {"hop_samples": True}, {"min_evidence": 2}])
def test_config_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        DSPConfig(**kwargs)


def test_phone_categories_strip_stress_and_do_not_consume_source_phone_labels():
    assert [phoneme_class([p]) for p in ("CH", "SH", "M", "AH0", "xyz")] == [
        "stop", "fricative", "sonorant", "vowel", "unknown"]
    assert phoneme_class([]) == "unknown"


def test_real_features_on_silence_preserve_coarse_start():
    row = candidate()
    result = refine_boundaries(np.zeros(24000), 48000, [row])[0]
    assert result["start_sample"] == row["start_sample"]
    assert result["refinement"]["reason"] == "weak_acoustic_evidence"
    assert result["refinement"]["feature_metadata"]["latency_calibrated"] is False


def test_real_features_and_scoring_share_grid_without_claiming_exact_onset():
    audio = np.zeros(24000)
    audio[6000:12000] = .5 * np.sin(2 * np.pi * 300 * np.arange(6000) / 48000)
    result = refine_boundaries(audio, 48000, [candidate()])[0]
    detail = result["refinement"]
    assert detail["feature_metadata"]["hop_samples"] == 120
    assert detail["feature_metadata"]["feature_step_is_accuracy_guarantee"] is False
    assert all(row["sample"] % 120 == 0 for row in detail["candidates"])
    assert 0 <= result["start_sample"] <= result["end_sample"]
    assert result["start_seconds"] == result["start_sample"] / 48000

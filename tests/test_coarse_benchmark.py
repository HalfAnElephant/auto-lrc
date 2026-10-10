import copy
import json
import struct
import wave

import pytest

from scripts import benchmark_coarse_gtsinger as benchmark


def reference_segment():
    return {"sample_rate": 48000, "units": [
        {"kind": "silence", "text": "SP", "boundary": None},
        {"kind": "lyric", "text": "one", "boundary": {"source": "textgrid:first_phone", "candidate_sample": 4800}},
        {"kind": "lyric", "text": "two", "boundary": {"source": "textgrid:first_phone", "candidate_sample": 9600}},
    ]}


def test_source_errors_use_sample_based_dataset_reference_and_original_unit_indices():
    segment = reference_segment()
    report = benchmark.source_errors(segment, [
        {"text": "one", "start_seconds": .11, "status": "candidate"},
        {"text": "two", "start_seconds": .18, "status": "candidate"},
    ])
    assert report["total"] == 2 and report["missing"] == report["text_mismatch"] == 0
    assert [row["unit_index"] for row in report["errors"]] == [1, 2]
    assert [row["signed_error_ms"] for row in report["errors"]] == pytest.approx([10, -20])
    assert [row["absolute_error_ms"] for row in report["errors"]] == pytest.approx([10, 20])
    assert report["errors"][0]["reference_sample"] == 4800


def test_ordered_text_mismatch_is_not_rematched_or_evaluated():
    report = benchmark.source_errors(reference_segment(), [
        {"text": "two", "start_seconds": .2}, {"text": "one", "start_seconds": .1}])
    assert report["total"] == 2 and report["text_mismatch"] == 2
    assert report["errors"] == []


def test_normalization_only_matches_ordered_case_width_and_punctuation():
    report = benchmark.source_errors(reference_segment(), [
        {"text": "ＯＮＥ!", "start_seconds": .1}, {"text": "Two?", "start_seconds": .2}])
    assert report["text_mismatch"] == 0 and len(report["errors"]) == 2
    assert all(row["absolute_error_ms"] == 0 for row in report["errors"])


@pytest.mark.parametrize("candidates", [[], [{"text": "one", "start_seconds": .1}],
                                        [{"text": "one", "start_seconds": .1}] * 3])
def test_missing_or_extra_units_cannot_be_counted_as_success(candidates):
    report = benchmark.source_errors(reference_segment(), candidates)
    assert report["total"] == 2 and report["text_mismatch"] == 2
    assert report["extra_or_missing_units"] == len(candidates) - 2
    assert report["errors"] == []


@pytest.mark.parametrize("missing", [{"start_seconds": None}, {"start_seconds": .1, "status": "unaligned"}])
def test_explicit_unaligned_or_missing_boundary_stays_in_denominator(missing):
    report = benchmark.source_errors(reference_segment(), [
        {"text": "one", **missing}, {"text": "two", "start_seconds": .2}])
    assert report["total"] == 2 and report["missing"] == 1
    assert len(report["errors"]) == 1
    summary = benchmark.summarize([report])
    assert summary["coverage_of_evaluated"]["5"] == 1
    assert summary["coverage_of_all"]["5"] == .5
    assert summary["independent_human_accuracy"] is False


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, "0.1"])
def test_invalid_numeric_prediction_is_rejected(invalid):
    with pytest.raises(ValueError, match="prediction"):
        benchmark.source_errors(reference_segment(), [
            {"text": "one", "start_seconds": invalid}, {"text": "two", "start_seconds": .2}])


def test_unexpected_reference_source_is_not_silently_evaluated():
    segment = reference_segment()
    segment["units"][1]["boundary"]["source"] = "estimated"
    with pytest.raises(ValueError, match="annotation source"):
        benchmark.source_errors(segment, [{"text": "one", "start_seconds": .1},
                                          {"text": "two", "start_seconds": .2}])


def test_summary_denominators_quantiles_and_severe_threshold():
    rows = [{"total": 7, "missing": 1, "text_mismatch": 1,
             "errors": [{"absolute_error_ms": value} for value in (0, 5, 10, 500, 501)]},
            {"total": 1, "missing": 0, "text_mismatch": 0, "errors": [{"absolute_error_ms": 50}]}]
    summary = benchmark.summarize(rows)
    assert summary["total"] == 8 and summary["evaluated"] == 6
    assert summary["missing"] == summary["text_mismatch"] == 1
    assert summary["mae_ms"] == pytest.approx(1066 / 6)
    assert summary["median_ae_ms"] == 30
    assert summary["p90_ms"] == 500.5 and summary["p95_ms"] == 500.75
    assert summary["coverage_of_evaluated"] == {"5": 2 / 6, "10": 3 / 6, "20": 3 / 6, "50": 4 / 6}
    assert summary["coverage_of_all"] == {"5": 2 / 8, "10": 3 / 8, "20": 3 / 8, "50": 4 / 8}
    assert summary["severe_over_500ms"] == 1
    assert summary["reference"] == "published_dataset_annotation"
    assert summary["independent_human_accuracy"] is False


def test_no_evaluated_units_do_not_produce_fabricated_error_metrics():
    summary = benchmark.summarize([{"total": 2, "missing": 1, "text_mismatch": 1, "errors": []}])
    assert summary["mae_ms"] is summary["median_ae_ms"] is summary["p90_ms"] is summary["p95_ms"] is None
    assert set(summary["coverage_of_evaluated"].values()) == {None}
    assert set(summary["coverage_of_all"].values()) == {0}
    empty = benchmark.summarize([])
    assert set(empty["coverage_of_all"].values()) == {None}


def test_local_windows_only_use_model_frames_and_never_source_labels():
    candidates = [{"raw_frames": {"start": 0, "end": 3}, "start_seconds": 200,
                   "boundary": {"candidate_sample": 999999}},
                  {"raw_frames": {"start": 8, "end": 9}, "start_seconds": -8}]
    original = copy.deepcopy(candidates)
    expected = {"mode": "constrained", "word_windows": [[0, 5], [6, 10]],
                "onset_windows": [[0, 3], [6, 10]], "fallback_to_legacy": True}
    assert benchmark.windows_from_legacy(candidates, 10, 2) == expected
    assert candidates == original
    candidates[0]["boundary"]["candidate_sample"] = 0
    candidates[0]["start_seconds"] = 0
    assert benchmark.windows_from_legacy(candidates, 10, 2) == expected


@pytest.mark.parametrize("start,end", [(None, 4), (1, None), (None, None)])
def test_missing_legacy_path_refuses_to_make_bands(start, end):
    with pytest.raises(ValueError, match="incomplete"):
        benchmark.windows_from_legacy([{"raw_frames": {"start": start, "end": end}}], 10, 2)


def wav_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(struct.pack("<4h", value, 1, 2, 3))


@pytest.fixture
def input_fixture(tmp_path):
    audio_root = tmp_path / "raw"
    segments, assignments, paths = [], [], []
    for index, split in enumerate(("train", "val")):
        segment_id = "English/singer/song-{}/0000".format(index)
        path = audio_root / (segment_id + ".wav")
        wav_file(path, index)
        paths.append(path)
        segments.append({"segment_id": segment_id, "song_id": "song-{}".format(index), "singer": "singer-{}".format(index),
                         "sample_rate": 48000, "num_samples": 4,
                         "sources": {"wav": {"sha256": benchmark.digest(path)}}})
        assignments.append({"segment_id": segment_id, "canonical_song_id": "song-{}".format(index), "split": split})
    manifest = {"identity_mode": "dataset", "assignments": assignments, "held_out_singers": []}
    segments_path, splits_path = tmp_path / "segments.jsonl", tmp_path / "splits.json"
    def save():
        segments_path.write_text("".join(json.dumps(segment) + "\n" for segment in segments))
        splits_path.write_text(json.dumps(manifest))
    save()
    return segments_path, splits_path, audio_root, segments, manifest, paths, save


def test_load_inputs_validates_all_rows_and_returns_list(input_fixture):
    segments_path, splits_path, audio_root, segments, _, paths, _ = input_fixture
    loaded = benchmark.load_inputs(segments_path, splits_path, audio_root)
    assert isinstance(loaded, list) and len(loaded) == 2
    assert [row["split"] for row in loaded] == ["train", "val"]
    assert [row["audio_path"] for row in loaded] == [str(path.resolve()) for path in paths]
    assert all("audio_path" not in original for original in segments)


@pytest.mark.parametrize("change", ["missing_assignment", "extra_assignment", "duplicate_assignment", "duplicate_segment"])
def test_load_inputs_rejects_nonexact_split_coverage(input_fixture, change):
    segments_path, splits_path, audio_root, segments, manifest, _, save = input_fixture
    if change == "missing_assignment":
        manifest["assignments"].pop()
    elif change == "extra_assignment":
        manifest["assignments"].append({"segment_id": "extra", "canonical_song_id": "extra", "split": "test"})
    elif change == "duplicate_assignment":
        manifest["assignments"].append(copy.deepcopy(manifest["assignments"][0]))
    else:
        segments.append(copy.deepcopy(segments[0]))
    save()
    with pytest.raises(ValueError, match="coverage|Duplicate"):
        benchmark.load_inputs(segments_path, splits_path, audio_root)


def test_song_cannot_cross_splits(input_fixture):
    segments_path, splits_path, audio_root, segments, manifest, _, save = input_fixture
    segments[1]["song_id"] = segments[0]["song_id"]
    manifest["assignments"][1]["canonical_song_id"] = segments[0]["song_id"]
    save()
    with pytest.raises(ValueError, match="crosses splits"):
        benchmark.load_inputs(segments_path, splits_path, audio_root)


def test_identical_original_audio_cannot_cross_splits_even_with_different_song_ids(input_fixture):
    segments_path, splits_path, audio_root, segments, _, paths, save = input_fixture
    paths[1].write_bytes(paths[0].read_bytes())
    segments[1]["sources"]["wav"]["sha256"] = segments[0]["sources"]["wav"]["sha256"]
    save()
    with pytest.raises(ValueError, match="crosses splits"):
        benchmark.load_inputs(segments_path, splits_path, audio_root)


def test_late_audio_checksum_failure_occurs_before_any_result_is_returned(input_fixture):
    segments_path, splits_path, audio_root, _, _, paths, _ = input_fixture
    paths[1].write_bytes(paths[1].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="frozen source"):
        benchmark.load_inputs(segments_path, splits_path, audio_root)


@pytest.mark.parametrize("key,value", [("sample_rate", 44100), ("num_samples", 5)])
def test_original_sample_grid_must_match_decoded_wav(input_fixture, key, value):
    segments_path, splits_path, audio_root, segments, _, _, save = input_fixture
    segments[1][key] = value
    save()
    with pytest.raises(ValueError, match="sample grid"):
        benchmark.load_inputs(segments_path, splits_path, audio_root)


def test_dataset_canonical_identity_must_match_source_song(input_fixture):
    segments_path, splits_path, audio_root, _, manifest, _, save = input_fixture
    manifest["assignments"][0]["canonical_song_id"] = "different"
    save()
    with pytest.raises(ValueError, match="identity changed"):
        benchmark.load_inputs(segments_path, splits_path, audio_root)


def test_held_out_singer_cannot_appear_in_training(input_fixture):
    segments_path, splits_path, audio_root, segments, manifest, _, save = input_fixture
    manifest["held_out_singers"] = [segments[0]["singer"]]
    save()
    with pytest.raises(ValueError, match="Held-out singer"):
        benchmark.load_inputs(segments_path, splits_path, audio_root)

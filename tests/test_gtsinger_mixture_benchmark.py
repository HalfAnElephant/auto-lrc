import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from scripts import benchmark_gtsinger_mixtures as benchmark


@pytest.fixture
def manifest_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    root = tmp_path / "data/local/mixtures"
    root.mkdir(parents=True)
    source = np.random.RandomState(7).standard_normal((6000, 1)) * 0.05
    paths = {}
    for name, values in (("source_audio", source), ("mixture_audio", source + 0.01),
                         ("accompaniment_audio", np.full_like(source, 0.01))):
        path = root / (name + ".wav")
        soundfile.write(str(path), values, 48000, subtype="PCM_24")
        paths[name] = path
    paths["lyrics"] = root / "lyrics.txt"
    paths["lyrics"].write_text("my love\n", encoding="utf-8")
    row = {"segment_id": "English/singer/song/0000", "song_id": "song",
           "derived_from_segment_id": "English/singer/song/0000", "split": "unassigned",
           "sample_rate": 48000, "num_samples": 6000, "channels": 1,
           "snr_db": 6.0, "mixture_gain": 1.0, "accompaniment_gain": 0.2,
           "timeline": {"source_to_mixture_offset_samples": 0},
           **{name: benchmark.provenance(path) for name, path in paths.items()}}
    value = {"schema_version": "gtsinger-mixtures-v1", "segments_source": {"path": "segments.jsonl"},
             "config": {"seed": 7}, "generator": {"kind": "fixture"}, "rows": [row]}
    path = root / "mixtures.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value, paths


def save_manifest(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_manifest_verifies_all_four_inputs_and_preserves_provenance(manifest_fixture):
    path, original, paths = manifest_fixture
    validated = benchmark.validate_manifest_inputs(path)
    row = validated["rows"][0]
    assert validated["segments_source"] == original["segments_source"]
    assert validated["generator"] == original["generator"]
    assert set(row["inputs"]) == set(benchmark.INPUT_NAMES)
    for name in benchmark.INPUT_NAMES:
        assert row["inputs"][name]["path"] == str(paths[name].resolve())
        assert row["inputs"][name]["sha256"] == original["rows"][0][name]["sha256"]
    assert row["inputs"]["mixture_audio"]["audio"] == {
        "sample_rate": 48000, "num_samples": 6000, "channels": 1, "subtype": "PCM_24"}


@pytest.mark.parametrize("name", benchmark.INPUT_NAMES)
def test_each_input_hash_is_checked(manifest_fixture, name):
    path, _, paths = manifest_fixture
    with paths[name].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="bytes/SHA256 mismatch"):
        benchmark.validate_manifest_inputs(path)


@pytest.mark.parametrize("frames,rate,channels", [(6001, 48000, 1), (6000, 44100, 1), (6000, 48000, 2)])
def test_audio_grid_is_checked_even_when_digest_matches(manifest_fixture, frames, rate, channels):
    path, value, paths = manifest_fixture
    soundfile.write(str(paths["mixture_audio"]), np.zeros((frames, channels)), rate, subtype="PCM_24")
    value["rows"][0]["mixture_audio"] = benchmark.provenance(paths["mixture_audio"])
    save_manifest(path, value)
    with pytest.raises(ValueError, match="sample grid mismatch"):
        benchmark.validate_manifest_inputs(path)


@pytest.mark.parametrize("change", [
    {"sample_rate": 44100}, {"num_samples": True}, {"channels": 0}, {"split": "train"},
    {"mixture_gain": True}, {"accompaniment_gain": 0},
    {"timeline": {"source_to_mixture_offset_samples": 1}},
    {"timeline": {"source_to_mixture_offset_samples": False}},
])
def test_manifest_rejects_unsupported_grid_or_timeline(manifest_fixture, change):
    path, value, _ = manifest_fixture
    value["rows"][0].update(change)
    save_manifest(path, value)
    with pytest.raises(ValueError):
        benchmark.validate_manifest_inputs(path)


def test_relative_input_paths_are_resolved_from_repo_root(manifest_fixture, monkeypatch):
    path, value, paths = manifest_fixture
    for name, input_path in paths.items():
        value["rows"][0][name]["path"] = str(input_path.relative_to(benchmark.ROOT))
    save_manifest(path, value)
    monkeypatch.chdir(path.parent)
    row = benchmark.validate_manifest_inputs(path)["rows"][0]
    assert row["inputs"]["lyrics"]["path"] == str(paths["lyrics"])


def test_selection_is_explicit_distinct_and_keeps_manifest_order():
    manifest = {"rows": [{"segment_id": "a"}, {"segment_id": "b"}, {"segment_id": "c"}]}
    assert benchmark.select_rows(manifest, []) == manifest["rows"]
    assert benchmark.select_rows(manifest, ["c", "a"]) == [manifest["rows"][0], manifest["rows"][2]]
    with pytest.raises(ValueError, match="Repeated"):
        benchmark.select_rows(manifest, ["a", "a"])
    with pytest.raises(ValueError, match="not in manifest"):
        benchmark.select_rows(manifest, ["missing"])


@pytest.mark.parametrize("lag,sample_rate", [(17, 48000), (-23, 48000), (7, 44100)])
def test_fft_diagnostic_has_correct_lag_sign_without_shifting_audio(lag, sample_rate):
    reference = np.random.RandomState(7).standard_normal(15000)
    if lag > 0:
        comparison = np.r_[np.zeros(lag), reference[:-lag]]
    else:
        comparison = np.r_[reference[-lag:], np.zeros(-lag)]
    original = comparison.copy()
    diagnostic = benchmark.waveform_xcorr_diagnostic(reference, comparison, sample_rate)
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["time_shift_applied"] is False
    assert diagnostic["positive_lag"] == "comparison is later than reference"
    assert diagnostic["max_lag_samples"] == sample_rate // 20
    assert [row["name"] for row in diagnostic["windows"]] == ["overall", "third-1", "third-2", "third-3"]
    for row in diagnostic["windows"]:
        assert row["status"] == "available"
        assert row["lag_samples"] == lag
        assert row["lag_seconds"] == pytest.approx(lag / sample_rate)
        assert row["normalized_peak"] > 0.9
        assert "not boundary confidence" in row["confidence_kind"]
    assert np.array_equal(comparison, original)


def test_fft_uses_first_channel_and_reports_length_difference():
    first = np.random.RandomState(9).standard_normal(15000)
    clean = np.column_stack([first, np.ones_like(first)])
    separated = np.column_stack([first, -first * 99])
    separated = np.vstack([separated, np.zeros((1, 2))])
    report = benchmark.waveform_xcorr_diagnostic(clean, separated, 48000)
    assert report["length_delta_samples"] == 1
    assert report["common_samples"] == len(first)
    assert all(row["lag_samples"] == 0 for row in report["windows"])
    assert all(row["normalized_peak"] == pytest.approx(1) for row in report["windows"])


def test_constant_signal_is_unavailable_without_inventing_confidence():
    report = benchmark.waveform_xcorr_diagnostic(np.ones(9000), np.ones(9000), 48000)
    assert all(row["status"] == "unavailable" for row in report["windows"])
    assert all(row["lag_samples"] is None and row["normalized_peak"] is None for row in report["windows"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("lag_ms", [0, 51, float("inf"), True])
def test_diagnostic_search_never_exceeds_fifty_milliseconds(lag_ms):
    with pytest.raises(ValueError):
        benchmark.waveform_xcorr_diagnostic(np.ones(9), np.ones(9), 48000, lag_ms)


def details(start=2, end=5):
    return {"units": [{"text": "word", "line_index": 0, "word_index": 0, "phonemes": [1, 2],
                       "start_frame": start, "end_frame": end,
                       "start_seconds": start * 0.1, "end_seconds": end * 0.1}],
            "alignment": {"frame_resolution_seconds": 0.1}}


def test_prediction_changes_preserve_raw_units_and_are_not_accuracy():
    clean, other = details(), details(3, 7)
    original = copy.deepcopy((clean, other))
    report = benchmark.prediction_changes(clean, other)
    assert report["status"] == "comparable"
    assert report["accuracy_available"] is False and report["time_shift_applied"] is False
    assert report["units"][0]["clean"] == clean["units"][0]
    assert report["units"][0]["comparison"] == other["units"][0]
    assert report["units"][0]["start_frame_delta"] == 1
    assert report["units"][0]["end_frame_delta"] == 2
    assert report["summary"]["mean_absolute_start_change_seconds"] == pytest.approx(0.1)
    assert report["summary"]["max_absolute_end_change_seconds"] == pytest.approx(0.2)
    assert (clean, other) == original


@pytest.mark.parametrize("mutation", ["text", "phonemes", "count", "grid", "missing"])
def test_predictions_with_unmatched_units_or_grid_are_not_force_matched(mutation):
    clean, other = details(), details()
    if mutation == "text":
        other["units"][0]["text"] = "different"
    elif mutation == "phonemes":
        other["units"][0]["phonemes"] = [7]
    elif mutation == "count":
        other["units"] = []
    elif mutation == "grid":
        other["alignment"]["frame_resolution_seconds"] = 0.2
    else:
        other["units"][0]["start_frame"] = None
    report = benchmark.prediction_changes(clean, other)
    assert report["status"] == "unavailable" and report["units"] == []
    assert report["accuracy_available"] is False


def test_existing_outputs_and_audio_outside_data_local_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    existing = tmp_path / "report"
    existing.mkdir()
    marker = existing / "preserved"
    marker.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        benchmark.reserve_outputs(existing, tmp_path / "data/local/new-audio")
    assert marker.read_bytes() == b"original"
    with pytest.raises(ValueError, match="under data/local"):
        benchmark.reserve_outputs(tmp_path / "new-report", tmp_path / "audio")
    with pytest.raises(ValueError, match="overlap"):
        benchmark.reserve_outputs(tmp_path / "data/local/new", tmp_path / "data/local/new/audio")
    assert not (tmp_path / "new-report").exists()


def test_dangling_output_symlink_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    link = tmp_path / "report"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="symlinks"):
        benchmark.reserve_outputs(link, tmp_path / "data/local/audio")
    assert link.is_symlink() and not (tmp_path / "missing").exists()


def test_cli_cpu_fallback_is_a_readable_failed_report(manifest_fixture, monkeypatch, tmp_path):
    import torch

    manifest_path, _, _ = manifest_fixture
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark, "runtime_environment", lambda: {})
    for name in ("scripts/benchmark_gtsinger_mixtures.py", "t2l/t2l.py", "t2l/phonetic.py",
                 "t2l/mtl/wrapper.py", "t2l/mtl/model.py", "t2l/mtl/utils.py", "lid.176.ftz",
                 "checkpoints/checkpoint_MTL", "checkpoints/checkpoint_BDR"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unused fixture; no model is loaded")
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output, audio = tmp_path / "reports/failure", tmp_path / "data/local/separated"
    code = benchmark.cli([str(manifest_path), "-o", str(output), "--audio-dir", str(audio),
                          "--segment-id", "English/singer/song/0000"])
    assert code == 1
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "failed"
    assert "refusing a CPU fallback" in report["error"]["message"]
    assert report["accuracy_metrics"]["available"] is False
    assert report["selection"]["selected_rows"] == 1
    assert report["rows"] == []
    assert (output / "traceback.txt").is_file()
    assert not audio.exists()

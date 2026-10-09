import hashlib
import json

import numpy as np
import pytest

from scripts import prepare_gtsinger_mixtures as mixing


def fixture_segment(tmp_path):
    audio = tmp_path / "source.wav"
    source = np.zeros(48000)
    source[[0, 1123, 47999]] = [0.25, -0.75, 0.5]
    mixing.write_pcm24_mono(audio, source)
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    record = {"segment_id": "Chinese/singer/technique/song/group/0000", "song_id": "song",
              "language": "zh", "audio_path": str(audio), "sample_rate": 48000, "num_samples": 48000,
              "sources": {"wav": {"sha256": digest}}, "units": [{"text": "a", "kind": "lyric"}],
              "split": "unassigned"}
    segments = tmp_path / "segments.jsonl"
    segments.write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / "source-manifest.json").write_text(json.dumps({"license": {"identifier": "fixture"}, "revision": "fixed"}))
    return segments, audio, source, record


def test_additive_mixture_preserves_first_last_and_internal_sample_grid(tmp_path):
    segments, audio, source, _ = fixture_segment(tmp_path)
    before = audio.read_bytes()
    result = mixing.prepare_mixtures(segments, tmp_path / "mixed")
    row = result["rows"][0]
    mixture = mixing.read_pcm24_mono(row["mixture_audio"]["path"])
    backing = mixing.read_pcm24_mono(row["accompaniment_audio"]["path"])
    recovered = (mixture - backing) / row["mixture_gain"]
    assert audio.read_bytes() == before
    assert len(mixture) == len(source)
    assert np.max(np.abs(recovered - source)) <= row["checks"]["quantization_bound"] / row["mixture_gain"]
    assert np.abs(recovered[[0, 1123, 47999]] - source[[0, 1123, 47999]]).max() < 1e-6
    assert row["timeline"]["source_to_mixture_offset_samples"] == 0
    assert row["derived_from_segment_id"] == row["segment_id"]
    assert row["split"] == "unassigned" and not result["accuracy"]["available"]
    assert row["source_audio"]["sha256"] == hashlib.sha256(before).hexdigest()


def test_requested_snr_and_peak_headroom():
    source = np.sin(np.arange(48000) * 0.07) * 0.9
    accompaniment = mixing.synthesize_accompaniment(len(source), 7)
    mixture, backing, row = mixing.mix_same_timeline(source, accompaniment, -6)
    vocal = source * row["mixture_gain"]
    measured = 20 * np.log10(np.sqrt(np.mean(vocal ** 2)) / np.sqrt(np.mean(backing ** 2)))
    assert measured == pytest.approx(-6, abs=1e-10)
    assert np.max(np.abs(mixture)) <= 0.9 + 1e-14
    assert row["mixture_gain"] < 1
    assert np.max(np.abs(backing)) <= 0.9 + 1e-14
    assert np.allclose(mixture - backing, vocal, atol=1e-15)


def test_generation_is_repeatable_and_does_not_require_labels():
    assert np.array_equal(mixing.synthesize_accompaniment(48000, 7), mixing.synthesize_accompaniment(48000, 7))
    assert not np.array_equal(mixing.synthesize_accompaniment(48000, 7), mixing.synthesize_accompaniment(48000, 8))


def test_cancelling_components_have_defined_headroom_without_dividing_by_zero():
    source = np.linspace(-0.9, 0.9, 48000)
    mixture, backing, row = mixing.mix_same_timeline(source, -source, 0)
    assert np.max(np.abs(mixture)) < 1e-14
    assert np.max(np.abs(backing)) <= 0.9 + 1e-14
    assert np.allclose(mixture - backing, source * row["mixture_gain"])


@pytest.mark.parametrize("snr", [float("nan"), float("inf"), -21, 41, True])
def test_undefined_or_extreme_snr_is_rejected(snr):
    with pytest.raises(ValueError, match="snr_db"):
        mixing.mix_same_timeline(np.ones(16), np.ones(16), snr)


@pytest.mark.parametrize("source,backing", [(np.zeros(16), np.ones(16)), (np.ones(16), np.zeros(16)),
                                         (np.ones(16), np.ones(15)), (np.array([np.nan]), np.ones(1))])
def test_invalid_signals_are_rejected(source, backing):
    with pytest.raises(ValueError):
        mixing.mix_same_timeline(source, backing)


def test_changed_source_fails_without_publishing_or_leaving_staging(tmp_path):
    segments, audio, _, _ = fixture_segment(tmp_path)
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        mixing.prepare_mixtures(segments, tmp_path / "mixed")
    assert not (tmp_path / "mixed").exists()
    assert not list(tmp_path.glob(".gtsinger-mixture-*"))


def test_existing_directory_is_not_overwritten(tmp_path):
    segments, _, _, _ = fixture_segment(tmp_path)
    out = tmp_path / "mixed"
    out.mkdir()
    marker = out / "user-file"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        mixing.prepare_mixtures(segments, out)
    assert marker.read_text() == "preserve"


def test_metadata_mismatch_is_not_repaired(tmp_path):
    segments, _, _, record = fixture_segment(tmp_path)
    record["num_samples"] -= 1
    segments.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="sample grid"):
        mixing.prepare_mixtures(segments, tmp_path / "mixed")
    assert not (tmp_path / "mixed").exists()


def test_later_segment_failure_leaves_no_partial_mixture(tmp_path):
    segments, _, _, record = fixture_segment(tmp_path)
    invalid = dict(record, segment_id="second", sources={"wav": {"sha256": "0" * 64}})
    segments.write_text(json.dumps(record) + "\n" + json.dumps(invalid) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        mixing.prepare_mixtures(segments, tmp_path / "mixed")
    assert not (tmp_path / "mixed").exists()
    assert not list(tmp_path.glob(".gtsinger-mixture-*"))

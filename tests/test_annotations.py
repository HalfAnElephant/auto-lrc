import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler
import wave

import pytest

from scripts.review_gtsinger import ReviewStore, make_handler, ThreadingHTTPServer
from t2l.datasets.annotations import LABEL_FIELDS, compare_source_annotations, validate_label


def form(tmp_path):
    audio = tmp_path / "original.wav"
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(3)
        target.setframerate(48000)
        target.writeframes(b"\0\0\0" * 48000)
    row = dict.fromkeys(LABEL_FIELDS)
    row.update(segment_id="fixture/one", song_id="song", language="English", unit_index=0,
               text="one", sample_rate=48000, audio_path=str(audio), review_status="pending")
    path = tmp_path / "manual-review.json"
    path.write_text(json.dumps([row]))
    return row, path


def label(**updates):
    result = {"human_truth_sample": 120, "human_truth_seconds": .0025, "confidence": .9,
              "review_status": "reviewed", "ambiguous": False, "reviewer": "fixture reviewer",
              "reviewed_at": "2026-10-10T00:00:00+00:00", "review_notes": None}
    result.update(updates)
    return result


@pytest.mark.parametrize("updates", [
    {"human_truth_sample": True}, {"human_truth_sample": 48000}, {"human_truth_seconds": .003},
    {"confidence": float("nan")}, {"confidence": 2}, {"reviewer": " "}, {"ambiguous": True},
    {"candidate_seconds": .1}, {"reviewed_at": "2026-10-10T00:00:00"},
])
def test_contradictory_or_unverifiable_labels_are_rejected(tmp_path, updates):
    row, _ = form(tmp_path)
    with pytest.raises(ValueError):
        validate_label(row, label(**updates), 48000)


def test_ambiguous_label_needs_reason_and_does_not_force_ground_truth(tmp_path):
    row, _ = form(tmp_path)
    ambiguous = label(review_status="ambiguous", ambiguous=True, human_truth_sample=None,
                      human_truth_seconds=None, review_notes="Cannot separate the connected consonants")
    result = validate_label(row, ambiguous, 48000)
    assert result["human_truth_sample"] is None
    with pytest.raises(ValueError, match="explanation"):
        validate_label(row, dict(ambiguous, review_notes=None), 48000)
    with pytest.raises(ValueError, match="single"):
        validate_label(row, dict(ambiguous, human_truth_sample=120), 48000)


def test_review_restores_history_rejects_stale_save_and_preserves_form_audio(tmp_path):
    row, path = form(tmp_path)
    before_form, before_audio = path.read_bytes(), Path(row["audio_path"]).read_bytes()
    session = tmp_path / "session.json"
    store = ReviewStore(path, session)
    second_process = ReviewStore(path, session)
    saved = store.update(0, label(), 0)
    assert saved["revision"] == 1 and saved["history"][0]["previous"]["review_status"] == "pending"
    assert ReviewStore(path, session).rows == saved["rows"]
    with pytest.raises(ValueError, match="another process"):
        second_process.update(0, label(), 0)
    with pytest.raises(ValueError, match="another tab"):
        store.update(0, label(human_truth_sample=240, human_truth_seconds=.005), 0)
    assert path.read_bytes() == before_form and Path(row["audio_path"]).read_bytes() == before_audio
    Path(row["audio_path"]).write_bytes(before_audio + b"changed")
    with pytest.raises(ValueError, match="audio changed"):
        store.update(0, label(), 1)
    with pytest.raises(ValueError, match="another form"):
        ReviewStore(path, session)


def test_blind_form_and_session_identity_fail_closed(tmp_path):
    row, path = form(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        ReviewStore(path, path)
    path.write_text(json.dumps([dict(row, candidate_seconds=.1)]))
    with pytest.raises(ValueError, match="candidate"):
        ReviewStore(path, tmp_path / "session.json")


def test_source_audit_excludes_pending_ambiguous_and_disagreement():
    candidates = [{"segment_id": "x", "unit_index": i, "candidate_seconds": .1} for i in range(3)]
    rows = [dict(label(), segment_id="x", unit_index=i, sample_rate=48000) for i in range(3)]
    rows[0].update(human_truth_sample=4800)
    rows[1].update(review_status="pending")
    rows[2].update(review_status="ambiguous")
    result = compare_source_annotations(candidates, rows)
    assert result["evaluated_count"] == 1 and result["mae_ms"] == 0
    assert result["pending_or_missing"] == [["x", 1]] and result["ambiguous"] == [["x", 2]]
    second = [dict(r, reviewer="second fixture reviewer") for r in rows]
    second[0]["human_truth_sample"] = 5280
    result = compare_source_annotations(candidates, rows, second_labels=second)
    assert result["status"] == "unavailable" and result["mae_ms"] is None
    assert result["disagreements"][0]["difference_ms"] == 10
    assert result["inter_reviewer_pair_count"] == 1 and result["inter_reviewer_mae_ms"] == 10
    assert result["inter_reviewer_agreed_pair_count"] == 0 and result["inter_reviewer_agreed_mae_ms"] is None
    with pytest.raises(ValueError, match="distinct"):
        compare_source_annotations(candidates, rows, second_labels=rows)


@pytest.mark.parametrize("candidate", [None, True, float("nan"), -1])
def test_reviewed_label_without_source_candidate_is_not_reported_as_pending(candidate):
    sources = [{"segment_id": "x", "unit_index": 0, "candidate_seconds": candidate}]
    rows = [dict(label(), segment_id="x", unit_index=0, sample_rate=48000)]
    result = compare_source_annotations(sources, rows)
    assert result["reviewed_count"] == 1 and result["evaluated_count"] == 0
    assert result["source_candidate_unavailable"] == [["x", 0]]
    assert result["pending_or_missing"] == [] and result["mae_ms"] is None


@pytest.mark.parametrize("status", ["pending", "ambiguous"])
def test_incomplete_second_review_does_not_hide_completed_first_review(status):
    sources = [{"segment_id": "x", "unit_index": 0, "candidate_seconds": .1}]
    rows = [dict(label(), segment_id="x", unit_index=0, sample_rate=48000)]
    second = [dict(rows[0], review_status=status, reviewer="second fixture reviewer")]
    result = compare_source_annotations(sources, rows, second_labels=second)
    assert result["reviewed_count"] == 1 and result["pending_or_missing"] == []
    assert result["second_review_" + status] == [["x", 0]]
    assert result["evaluated_count"] == 0 and result["inter_reviewer_pair_count"] == 0


def test_inter_reviewer_diagnostic_separates_agreed_and_unadjudicated_pairs():
    sources = [{"segment_id": "x", "unit_index": i, "candidate_seconds": .1} for i in range(2)]
    rows = [dict(label(), segment_id="x", unit_index=i, sample_rate=48000,
                 human_truth_sample=4800) for i in range(2)]
    second = [dict(row, reviewer="second fixture reviewer",
                   human_truth_sample=4800 + offset) for row, offset in zip(rows, (48, 4800))]
    result = compare_source_annotations(sources, rows, second_labels=second)
    assert result["inter_reviewer_mae_ms"] == 50.5
    assert result["inter_reviewer_agreed_mae_ms"] == 1
    assert result["inter_reviewer_pair_count"] == 2 and result["inter_reviewer_agreed_pair_count"] == 1
    assert result["evaluated_count"] == 1 and result["mae_ms"] == 0
    assert result["disagreements"][0]["difference_ms"] == 100


def test_http_blindness_token_range_and_original_sample_signal(tmp_path):
    _, path = form(tmp_path)
    store = ReviewStore(path, tmp_path / "session.json")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, "fixture-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:%d" % server.server_port
    urlopen = build_opener(ProxyHandler({})).open
    try:
        with urlopen(base + "/api/form") as response:
            data = json.load(response)
            assert not any("candidate" in key for row in data["rows"] for key in row)
        with urlopen(base + "/api/signal?segment=0&start=0&end=2400&width=100") as response:
            data = json.load(response)
            assert data["sample_rate"] == 48000 and data["waveform"] == [[0, 0]] * 100
            assert data["spectrogram_centers_sample"][0] == 511.5
        with urlopen(Request(base + "/audio/0", headers={"Range": "bytes=0-43"})) as response:
            assert response.status == 206 and len(response.read()) == 44
        payload = json.dumps({"row_index": 0, "label": label(), "revision": 0}).encode()
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/label", data=payload, headers={"Content-Type": "application/json"}))
        assert error.value.code == 403 and not store.session_path.exists()
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/form", headers={"Host": "unrelated.example"}))
        assert error.value.code == 403
        for origin in (None, "http://unrelated.example"):
            headers = {"Content-Type": "application/json", "X-Review-Token": "fixture-token"}
            if origin is not None:
                headers["Origin"] = origin
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + "/api/label", data=payload, headers=headers))
            assert error.value.code == 403 and not store.session_path.exists()
        with urlopen(Request(base + "/api/label", data=payload,
                             headers={"Content-Type": "application/json", "X-Review-Token": "fixture-token",
                                      "Origin": base})) as response:
            saved = json.load(response)
            assert saved["rows"][0]["human_truth_sample"] == 120
            assert ReviewStore(path, store.session_path).rows == saved["rows"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

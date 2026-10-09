import json
import os
import wave

import pytest

from scripts.adjudicate_gtsinger import load_review_documents, main, write_new_json
from scripts.audit_gtsinger_annotations import main as audit
from scripts.review_gtsinger import ReviewStore
from t2l.datasets.annotations import LABEL_FIELDS


def review_fixture(tmp_path):
    audio = tmp_path / "original.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(3)
        stream.setframerate(48000)
        stream.writeframes(b"\0\0\0" * 48000)
    rows = []
    for index, text in enumerate(("one", "two")):
        row = dict.fromkeys(LABEL_FIELDS)
        row.update(segment_id="fixture/song/0000", song_id="fixture/song", language="English",
                   unit_index=index, text=text, sample_rate=48000, audio_path=str(audio),
                   review_status="pending")
        rows.append(row)
    form = tmp_path / "manual-review.json"
    form.write_text(json.dumps(rows))
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    first, second = ReviewStore(form, first_path), ReviewStore(form, second_path)
    for index, (sample, other) in enumerate(((120, 144), (12000, 14400))):
        for store, onset, reviewer in ((first, sample, "fixture first"),
                                       (second, other, "fixture second")):
            store.update(index, {"human_truth_sample": onset, "human_truth_seconds": None,
                                 "confidence": .9, "review_status": "reviewed", "ambiguous": False,
                                 "reviewer": reviewer, "reviewed_at": None, "review_notes": None}, index)
    args = ["--form", str(form), "--review", str(first_path), "--second-review", str(second_path)]
    template = tmp_path / "adjudication.json"
    assert main(args + ["--output", str(template)]) == 0
    return args, template, first, second, form, audio


def fill_decision(template):
    document = json.loads(template.read_text())
    row = document["decisions"][0]
    row.update(human_truth_sample=12600, human_truth_seconds=None, confidence=.8,
               review_status="reviewed", ambiguous=False, reviewer="fixture adjudicator",
               reviewed_at="2026-10-10T00:00:00+00:00",
               review_notes="Synthetic test decision; explicit sample, not an averaged estimate")
    template.write_text(json.dumps(document))


def test_pending_adjudication_exports_no_invented_dispute_reference(tmp_path):
    args, template, first, second, _, _ = review_fixture(tmp_path)
    document = json.loads(template.read_text())
    assert len(document["decisions"]) == 1
    assert document["decisions"][0]["human_truth_sample"] is None
    assert document["decisions"][0]["review_status"] == "pending"
    assert "candidate" not in json.dumps(document)
    snapshots = [store.session_path.read_bytes() for store in (first, second)]
    output = tmp_path / "reference.json"
    assert main(args + ["--decisions", str(template), "--output", str(output)]) == 0
    reference = json.loads(output.read_text())
    assert reference["protocol"] == "boundary_reference"
    assert reference["rows"][0]["human_truth_sample"] == 120
    assert reference["rows"][1]["human_truth_sample"] is None
    assert reference["rows"][1]["review_status"] == "pending"
    assert reference["independent_adjudicated_test_set"] is False
    assert [store.session_path.read_bytes() for store in (first, second)] == snapshots


def test_explicit_decision_is_consumed_by_reference_and_source_audit(tmp_path):
    args, template, first, second, _, _ = review_fixture(tmp_path)
    fill_decision(template)
    output = tmp_path / "reference.json"
    assert main(args + ["--decisions", str(template), "--output", str(output)]) == 0
    reference = json.loads(output.read_text())
    assert reference["rows"][1]["human_truth_sample"] == 12600
    assert reference["rows"][1]["human_truth_seconds"] == .2625
    assert reference["rows"][1]["reviewer"] == "fixture adjudicator"
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps([
        {"segment_id": "fixture/song/0000", "unit_index": index, "candidate_seconds": seconds}
        for index, seconds in enumerate((.0025, .25))]))
    report = tmp_path / "audit.json"
    assert audit(args + ["--candidates", str(candidates), "--adjudication", str(template),
                         "--output", str(report)]) == 0
    result = json.loads(report.read_text())
    assert result["evaluated_count"] == 2 and result["mae_ms"] == pytest.approx(6.25)
    assert result["review_count"] == 2 and result["independent_adjudicated_test_set"] is False
    assert result["reference_protocol"] == "boundary_reference"
    assert result["blind_review_comparison"]["evaluated_count"] == 1
    assert len(result["blind_review_comparison"]["disagreements"]) == 1
    assert result["inter_reviewer_pair_count"] == 2
    assert result["inter_reviewer_mae_ms"] == pytest.approx(25.25)
    assert result["inter_reviewer_agreed_pair_count"] == 1
    assert json.loads(first.session_path.read_text())["rows"][1]["human_truth_sample"] == 12000
    assert json.loads(second.session_path.read_text())["rows"][1]["human_truth_sample"] == 14400


def test_changed_review_rejects_old_decisions_without_output(tmp_path):
    args, template, _, second, _, _ = review_fixture(tmp_path)
    fill_decision(template)
    update = dict(second.rows[1], human_truth_sample=14640, human_truth_seconds=None)
    second.update(1, update, second.revision)
    output = tmp_path / "reference.json"
    with pytest.raises(ValueError):
        main(args + ["--decisions", str(template), "--output", str(output)])
    assert not output.exists()


def test_changed_original_audio_rejects_session_without_output(tmp_path):
    args, template, _, _, _, audio = review_fixture(tmp_path)
    audio.write_bytes(audio.read_bytes() + b"changed")
    output = tmp_path / "reference.json"
    with pytest.raises(ValueError, match="another form"):
        main(args + ["--decisions", str(template), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("onset_definition", "vowel only"),
                                        ("audio_sources", {})])
def test_saved_metadata_must_match_original_review_protocol(tmp_path, field, value):
    _, _, first, second, form, _ = review_fixture(tmp_path)
    saved = json.loads(first.session_path.read_text())
    saved[field] = value
    first.session_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="metadata"):
        load_review_documents(form, first.session_path, second.session_path)


@pytest.mark.parametrize("existing", ["file", "symlink"])
def test_new_outputs_never_overwrite_or_follow_existing_targets(tmp_path, existing):
    args, _, _, _, _, _ = review_fixture(tmp_path)
    preserved = tmp_path / "preserved.json"
    preserved.write_text("preserve this output")
    output = preserved if existing == "file" else tmp_path / "alias.json"
    if existing == "symlink":
        output.symlink_to(preserved)
    with pytest.raises(FileExistsError):
        main(args + ["--output", str(output)])
    assert preserved.read_text() == "preserve this output"
    assert list(tmp_path.glob(".adjudication-*")) == []


def test_malformed_json_is_not_published(tmp_path):
    output = tmp_path / "report.json"
    with pytest.raises(ValueError):
        write_new_json(output, {"invalid": float("nan")})
    assert not output.exists() and list(tmp_path.glob(".adjudication-*")) == []


def test_adjudication_audit_requires_the_second_blind_review(tmp_path):
    output = tmp_path / "report.json"
    first = tmp_path / "first.json"
    first.write_text("{}")
    with pytest.raises(SystemExit) as error:
        audit(["--form", str(tmp_path / "form.json"), "--review", str(first),
               "--candidates", str(tmp_path / "candidates.json"),
               "--adjudication", str(tmp_path / "decisions.json"), "--output", str(output)])
    assert error.value.code == 2 and not output.exists()


def test_missing_decisions_file_is_a_cli_error_without_output(tmp_path):
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as error:
        main(["--form", str(tmp_path / "form.json"), "--review", str(tmp_path / "first.json"),
              "--second-review", str(tmp_path / "second.json"),
              "--decisions", str(tmp_path / "missing.json"), "--output", str(output)])
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("alias", ["same_path", "hard_link"])
def test_one_pending_session_cannot_stand_in_for_two_independent_review_files(tmp_path, alias):
    _, _, first, _, form, _ = review_fixture(tmp_path)
    pending = first.document()
    for row in pending["rows"]:
        row.update(dict.fromkeys(LABEL_FIELDS))
        row["review_status"] = "pending"
    first.session_path.write_text(json.dumps(pending))
    second_path = first.session_path
    if alias == "hard_link":
        second_path = tmp_path / "hard-link.json"
        os.link(first.session_path, second_path)
    output = tmp_path / "same-session-reference.json"
    with pytest.raises(ValueError, match="distinct session files"):
        main(["--form", str(form), "--review", str(first.session_path),
              "--second-review", str(second_path), "--output", str(output)])
    assert not output.exists()

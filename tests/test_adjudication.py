from copy import deepcopy
import hashlib
import json

import pytest

from t2l.datasets.adjudication import build_adjudication_form, resolve_adjudication
from t2l.datasets.annotations import IDENTITY_FIELDS, LABEL_FIELDS, ONSET_DEFINITION, validate_label


def label(sample=480, *, reviewer="first listener", status="reviewed", **updates):
    result = dict.fromkeys(LABEL_FIELDS)
    result["review_status"] = status
    if status != "pending":
        result.update(human_truth_sample=sample if status == "reviewed" else None,
                      confidence=.9, ambiguous=status == "ambiguous", reviewer=reviewer,
                      reviewed_at="2026-10-10T12:00:00+08:00",
                      review_notes="Indistinct consonant release" if status == "ambiguous" else None)
    result.update(updates)
    return result


def session(labels):
    rows = [dict(segment_id="fixture/song/0000", song_id="verified-song", language="English",
                 unit_index=index, text="word" + str(index), sample_rate=48000,
                 audio_path="/original/audio.wav", **item) for index, item in enumerate(labels)]
    return {"schema_version": 1, "protocol": "blind", "onset_definition": ONSET_DEFINITION,
            "form_fingerprint": "a" * 64, "revision": 2,
            "audio_sources": {"fixture/song/0000": {"sha256": "b" * 64,
                                                       "sample_rate": 48000, "num_samples": 48000}},
            "rows": rows, "history": [], "independence": "External verification is required."}


def pair(first=None, second=None):
    first = first if first is not None else [label()]
    second = second if second is not None else [label(960, reviewer="second listener")]
    return session(first), session(second)


def resolved_form(first, second, **updates):
    document = build_adjudication_form(first, second)
    document["decisions"][0].update(label(701, reviewer="adjudicator", review_notes="Audible release at sample 701"))
    document["decisions"][0].update(updates)
    return document


def test_template_is_blind_pending_and_binds_entire_original_sessions_without_mutating_them():
    first, second = pair()
    before = deepcopy((first, second))
    result = build_adjudication_form(first, second)
    assert result["protocol"] == "boundary_adjudication" and result["schema_version"] == 1
    for name, document in (("first_review_sha256", first), ("second_review_sha256", second)):
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False).encode("ascii")
        assert result[name] == hashlib.sha256(canonical).hexdigest()
    row = result["decisions"][0]
    assert set(row) == set(IDENTITY_FIELDS + LABEL_FIELDS) | {"first_review", "second_review"}
    assert row["review_status"] == "pending" and row["human_truth_sample"] is None
    assert row["first_review"] == {key: first["rows"][0][key] for key in LABEL_FIELDS}
    assert (first, second) == before
    result["audio_sources"]["fixture/song/0000"]["sha256"] = "c" * 64
    result["decisions"][0]["first_review"]["reviewer"] = "changed copy"
    assert (first, second) == before
    assert "candidate_seconds" not in json.dumps(result)
    assert result["independent_adjudicated_test_set"] is False


def test_within_tolerance_retains_first_original_sample_with_second_confirmation_without_averaging():
    first, second = pair([label(481)], [label(721, reviewer="second listener")])
    form = build_adjudication_form(first, second)
    assert form["decisions"] == []  # Exactly 240 / 48000 samples = 5ms.
    reference = resolve_adjudication(first, second, form)
    assert reference["rows"][0]["human_truth_sample"] == 481
    assert reference["rows"][0]["human_truth_seconds"] == 481 / 48000
    assert set(reference["rows"][0]) == set(IDENTITY_FIELDS + LABEL_FIELDS)
    assert reference["provenance"][0]["method"] == "retain_first_review_with_second_confirmation"
    assert reference["provenance"][0]["second_review"]["human_truth_sample"] == 721
    assert reference["counts"]["confirmed"] == 1 and reference["counts"]["reviewed"] == 1
    assert reference["independent_adjudicated_test_set"] is False
    validate_label(reference["rows"][0], reference["rows"][0], 48000)


@pytest.mark.parametrize("limit, disputed", [(5, False), (4.999999999999999, True), (0, True)])
def test_tolerance_uses_exact_integer_sample_difference(limit, disputed):
    first, second = pair([label(480)], [label(720, reviewer="second listener")])
    assert bool(build_adjudication_form(first, second, agreement_limit_ms=limit)["decisions"]) == disputed


def test_zero_tolerance_accepts_equal_samples_and_reordered_second_rows_preserve_first_order():
    first, second = pair([label(481), label(821)],
                         [label(481, reviewer="second listener"), label(821, reviewer="second listener")])
    second["rows"].reverse()
    form = build_adjudication_form(first, second, agreement_limit_ms=0)
    reference = resolve_adjudication(first, second, form, agreement_limit_ms=0)
    assert [row["unit_index"] for row in reference["rows"]] == [0, 1]
    assert form["decisions"] == []


def test_large_finite_integer_tolerance_does_not_overflow_float_conversion():
    first, second = pair()
    assert build_adjudication_form(first, second, agreement_limit_ms=10 ** 400)["decisions"] == []


@pytest.mark.parametrize("form_limit, resolution_limit", [(0, 0.0), (0.0, 0), (5, 5.0), (5.0, 5), (2.5, 2.5)])
def test_equivalent_numeric_tolerances_have_canonical_metadata_and_resolve_across_entry_points(
        form_limit, resolution_limit):
    first, second = pair()
    form = build_adjudication_form(first, second, agreement_limit_ms=form_limit)
    equivalent = build_adjudication_form(first, second, agreement_limit_ms=resolution_limit)
    assert json.dumps(form, sort_keys=True) == json.dumps(equivalent, sort_keys=True)
    expected_type = int if form_limit in (0, 5) else float
    assert type(form["agreement_limit_ms"]) is expected_type
    reference = resolve_adjudication(first, second, form, agreement_limit_ms=resolution_limit)
    assert type(reference["agreement_limit_ms"]) is expected_type
    assert reference["rows"][0]["review_status"] == "pending"


def test_normalized_numeric_tolerance_still_rejects_a_different_threshold():
    first, second = pair()
    form = build_adjudication_form(first, second, agreement_limit_ms=5.0)
    with pytest.raises(ValueError, match="agreement_limit_ms"):
        resolve_adjudication(first, second, form, agreement_limit_ms=5.000001)


def test_disagreement_and_incomplete_reviews_remain_pending_without_inventing_truth():
    first, second = pair([label(), label(status="pending"), label()],
                         [label(960, reviewer="second listener"), label(reviewer="second listener"),
                          label(status="pending")])
    form = build_adjudication_form(first, second)
    assert [row["unit_index"] for row in form["decisions"]] == [0]
    result = resolve_adjudication(first, second, form)
    assert result["status"] == "unavailable" and len(result["rows"]) == 3
    assert all(row["review_status"] == "pending" and row["human_truth_sample"] is None for row in result["rows"])
    assert result["counts"]["unresolved_adjudication"] == 1
    assert result["counts"]["incomplete_reviews"] == 2 and len(result["unresolved"]) == 3


def test_explicit_adjudication_preserves_chosen_integer_sample_and_original_evidence():
    first, second = pair()
    form = resolved_form(first, second, reviewer="first listener")  # A third identity is not mandated.
    before = deepcopy((first, second, form))
    result = resolve_adjudication(first, second, form)
    assert result["rows"][0]["human_truth_sample"] == 701
    assert result["rows"][0]["human_truth_seconds"] == 701 / 48000
    assert result["counts"]["adjudicated_reviewed"] == 1 and result["unresolved"] == []
    proof = result["provenance"][0]
    assert proof["method"] == "human_adjudication"
    assert proof["first_review"]["human_truth_sample"] == 480
    assert proof["second_review"]["human_truth_sample"] == 960
    assert proof["adjudication"]["human_truth_sample"] == 701
    assert proof["adjudication"]["review_notes"] == "Audible release at sample 701"
    assert (first, second, form) == before and result["independent_adjudicated_test_set"] is False


def test_ambiguous_review_requires_adjudication_and_ambiguous_decision_excludes_single_truth():
    first, second = pair([label(status="ambiguous")], [label(reviewer="second listener")])
    form = build_adjudication_form(first, second)
    assert len(form["decisions"]) == 1
    assert resolve_adjudication(first, second, form)["rows"][0]["review_status"] == "pending"
    form["decisions"][0].update(label(status="ambiguous", reviewer="adjudicator"))
    result = resolve_adjudication(first, second, form)
    assert result["status"] == "unavailable" and result["counts"]["adjudicated_ambiguous"] == 1
    assert result["rows"][0]["human_truth_sample"] is None
    assert result["excluded_ambiguous"] == [{"segment_id": "fixture/song/0000", "unit_index": 0}]
    assert result["unresolved"] == []


@pytest.mark.parametrize("limit", [True, False, -1, float("nan"), float("inf"), "5", None])
def test_invalid_tolerance_is_rejected(limit):
    first, second = pair()
    with pytest.raises(ValueError, match="agreement_limit_ms"):
        build_adjudication_form(first, second, agreement_limit_ms=limit)


@pytest.mark.parametrize("status", ["reviewed", "ambiguous"])
def test_complete_pair_requires_distinct_actual_reviewers_even_if_ambiguous(status):
    first, second = pair([label(status=status)], [label(960, reviewer=" first listener ")])
    with pytest.raises(ValueError, match="distinct"):
        build_adjudication_form(first, second)


@pytest.mark.parametrize("field,value", [("protocol", "boundary_reference"), ("schema_version", True),
                                         ("form_fingerprint", "not-a-hash"), ("revision", True),
                                         ("history", None), ("onset_definition", "")])
def test_invalid_blind_session_metadata_is_rejected(field, value):
    first, second = pair()
    first[field] = value
    with pytest.raises(ValueError):
        build_adjudication_form(first, second)


@pytest.mark.parametrize("field,value", [("sha256", "z" * 64), ("sha256", None),
                                         ("sample_rate", True), ("sample_rate", 0),
                                         ("num_samples", 0), ("num_samples", 48000.0)])
def test_invalid_original_audio_metadata_is_rejected(field, value):
    first, second = pair()
    first["audio_sources"]["fixture/song/0000"][field] = value
    with pytest.raises(ValueError):
        build_adjudication_form(first, second)


@pytest.mark.parametrize("field,value", [("onset_definition", "Different audible onset policy"),
                                         ("form_fingerprint", "c" * 64)])
def test_both_reviews_must_share_the_same_form_and_onset_definition(field, value):
    first, second = pair()
    second[field] = value
    with pytest.raises(ValueError, match=field):
        build_adjudication_form(first, second)


def test_both_reviews_must_share_the_same_original_audio_hash_and_length():
    first, second = pair()
    second["audio_sources"]["fixture/song/0000"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="audio_sources"):
        build_adjudication_form(first, second)
    second["audio_sources"]["fixture/song/0000"]["sha256"] = "b" * 64
    second["audio_sources"]["fixture/song/0000"]["num_samples"] += 1
    with pytest.raises(ValueError, match="audio_sources"):
        build_adjudication_form(first, second)


@pytest.mark.parametrize("field,value", [("text", "other lyric"), ("song_id", "other song"),
                                         ("audio_path", "/different.wav"), ("unit_index", 1),
                                         ("sample_rate", 44100)])
def test_review_identity_or_timeline_changes_are_rejected(field, value):
    first, second = pair()
    second["rows"][0][field] = value
    with pytest.raises(ValueError):
        build_adjudication_form(first, second)


def test_review_duplicate_missing_and_extra_source_coverage_are_rejected():
    first, second = pair()
    second["rows"].append(deepcopy(second["rows"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        build_adjudication_form(first, second)
    second = session([label(reviewer="second listener"), label(reviewer="second listener")])
    with pytest.raises(ValueError, match="coverage"):
        build_adjudication_form(first, second)
    second = session([label(reviewer="second listener")])
    second["audio_sources"]["unexpected"] = deepcopy(second["audio_sources"]["fixture/song/0000"])
    with pytest.raises(ValueError, match="coverage"):
        build_adjudication_form(first, second)


@pytest.mark.parametrize("updates", [{"human_truth_sample": 48000}, {"human_truth_sample": True},
                                     {"human_truth_seconds": .003}, {"reviewer": " "},
                                     {"reviewed_at": "2026-10-10T00:00:00"},
                                     {"candidate_seconds": .01}, {"confidence": float("nan")}])
def test_original_review_labels_are_validated(updates):
    first, second = pair()
    first["rows"][0].update(updates)
    with pytest.raises(ValueError):
        build_adjudication_form(first, second)


@pytest.mark.parametrize("updates", [{"human_truth_sample": 48000}, {"human_truth_sample": -1},
                                     {"human_truth_sample": True}, {"human_truth_sample": 701.0},
                                     {"human_truth_seconds": .02}, {"reviewer": " "},
                                     {"reviewed_at": None}, {"reviewed_at": "2026-10-10T00:00:00"},
                                     {"review_notes": None}, {"review_notes": " "},
                                     {"confidence": 2}, {"ambiguous": True}, {"review_status": "approved"}])
def test_invalid_human_adjudication_is_rejected(updates):
    first, second = pair()
    with pytest.raises(ValueError):
        resolve_adjudication(first, second, resolved_form(first, second, **updates))


def test_pending_adjudication_cannot_smuggle_a_sample_and_ambiguous_cannot_force_one():
    first, second = pair()
    form = build_adjudication_form(first, second)
    form["decisions"][0]["human_truth_sample"] = 701
    with pytest.raises(ValueError, match="Pending"):
        resolve_adjudication(first, second, form)
    form = resolved_form(first, second, review_status="ambiguous", ambiguous=True)
    with pytest.raises(ValueError, match="single"):
        resolve_adjudication(first, second, form)


@pytest.mark.parametrize("field,value", [("revision", 3), ("history", [{"new": "event"}]),
                                         ("independence", "changed declaration")])
def test_entire_snapshot_change_invalidates_old_decisions(field, value):
    first, second = pair()
    form = resolved_form(first, second)
    second[field] = value
    with pytest.raises(ValueError, match="snapshot"):
        resolve_adjudication(first, second, form)


def test_changed_review_point_invalidates_old_decisions_even_if_dispute_still_exists():
    first, second = pair()
    form = resolved_form(first, second)
    second["rows"][0]["human_truth_sample"] += 1
    with pytest.raises(ValueError, match="snapshot"):
        resolve_adjudication(first, second, form)


def test_audio_and_form_changes_in_both_reviews_still_invalidate_old_decisions():
    first, second = pair()
    form = resolved_form(first, second)
    for doc in (first, second):
        doc["audio_sources"]["fixture/song/0000"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="snapshot"):
        resolve_adjudication(first, second, form)
    first, second = pair()
    form = resolved_form(first, second)
    for doc in (first, second):
        doc["form_fingerprint"] = "d" * 64
    with pytest.raises(ValueError, match="snapshot"):
        resolve_adjudication(first, second, form)


@pytest.mark.parametrize("field,value", [("protocol", "blind"), ("schema_version", True),
                                         ("first_review_sha256", "c" * 64), ("agreement_limit_ms", 6),
                                         ("policy", {}), ("independent_adjudicated_test_set", True)])
def test_adjudication_metadata_tampering_is_rejected(field, value):
    first, second = pair()
    form = resolved_form(first, second)
    form[field] = value
    with pytest.raises(ValueError, match="metadata|snapshot"):
        resolve_adjudication(first, second, form)


@pytest.mark.parametrize("change", ["identity", "first_evidence", "second_evidence", "extra_field",
                                    "missing_label_field", "duplicate", "missing", "unexpected"])
def test_adjudication_identity_evidence_and_exact_coverage_cannot_change(change):
    first, second = pair()
    form = resolved_form(first, second)
    row = form["decisions"][0]
    if change == "identity":
        row["text"] = "other lyric"
    elif change.endswith("evidence"):
        row[change.split("_")[0] + "_review"]["human_truth_sample"] = 1
    elif change == "extra_field":
        row["candidate_seconds"] = .01
    elif change == "missing_label_field":
        del row["reviewer"]
    elif change == "duplicate":
        form["decisions"].append(deepcopy(row))
    elif change == "missing":
        form["decisions"] = []
    else:
        row["unit_index"] = 8
    with pytest.raises(ValueError):
        resolve_adjudication(first, second, form)

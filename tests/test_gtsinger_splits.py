import hashlib
import json

import pytest

from t2l.datasets.splits import assign_song_splits
from scripts.split_gtsinger import cli


def record(song, version, **extra):
    segment_id = f"{song}-{version}"
    path = f"/fixture/audio/{segment_id}.wav"
    return {"segment_id": segment_id, "song_id": song, "audio_path": path,
            "sources": {"wav": {"path": path,
                                "sha256": hashlib.sha256(("fixture audio " + segment_id).encode()).hexdigest()}},
            **extra}


def records():
    return [record(song, version)
            for song in ("first", "second", "third", "fourth", "alias")
            for version in ("control", "breathy", "another-singer")]


def review():
    return {"reviewed_by": "fixture reviewer", "song_ids": {
        "first": "A", "second": "B", "third": "C", "fourth": "D", "alias": "A"}}


def test_song_aliases_variants_and_singers_cannot_leak_and_input_order_is_irrelevant():
    result = assign_song_splits(records(), review())
    assert result == assign_song_splits(reversed(records()), review())
    groups = {}
    for row in result["assignments"]:
        groups.setdefault(row["canonical_song_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert set(row["split"] for row in result["assignments"]) == {"train", "val", "test"}
    assert sum(result["song_counts"].values()) == 4
    assert result["boundary_annotations_reviewed"] is False
    assert result["singer_independent"] is False


def test_two_sample_songs_cannot_be_presented_as_three_nonempty_splits():
    with pytest.raises(ValueError, match="at least 3"):
        assign_song_splits(records()[:6], review())


@pytest.mark.parametrize("identity_review", [{}, {"reviewed_by": ""},
    {"reviewed_by": "fixture", "song_ids": {}}])
def test_missing_identity_review_fails_closed(identity_review):
    with pytest.raises(ValueError):
        assign_song_splits(records(), identity_review)


def test_derived_audio_cannot_be_assigned_to_another_song():
    data = records()
    data[-1]["parent_segment_id"] = "second-control"
    with pytest.raises(ValueError, match="different song identities"):
        assign_song_splits(data, review())


def test_identical_audio_with_conflicting_song_identity_is_rejected():
    data = records()
    data[3]["sources"]["wav"]["sha256"] = data[0]["sources"]["wav"]["sha256"]
    assert data[0]["audio_path"] != data[3]["audio_path"]
    with pytest.raises(ValueError, match="same audio"):
        assign_song_splits(data, review())


def test_missing_audio_digests_cannot_fall_back_to_different_paths():
    data = records()
    del data[0]["sources"]
    del data[3]["sources"]
    assert data[0]["audio_path"] != data[3]["audio_path"]
    with pytest.raises(ValueError, match=r"sources\.wav\.sha256.*first-control"):
        assign_song_splits(data, review())


@pytest.mark.parametrize("source", [None, [], {}, {"wav": None}, {"wav": {}},
    {"wav": {"sha256": None}}, {"wav": {"sha256": 123}},
    {"wav": {"sha256": "a" * 63}}, {"wav": {"sha256": "a" * 65}},
    {"wav": {"sha256": "g" * 64}}, {"wav": {"sha256": " " + "a" * 64}}])
def test_missing_or_malformed_audio_digest_is_a_clear_validation_error(source):
    data = records()
    data[0]["sources"] = source
    with pytest.raises(ValueError, match=r"sources\.wav\.sha256.*first-control"):
        assign_song_splits(data, review())


def test_audio_digest_case_cannot_bypass_conflicting_identity_check():
    data = records()
    digest = data[0]["sources"]["wav"]["sha256"]
    data[0]["sources"]["wav"]["sha256"] = digest.upper()
    data[3]["sources"]["wav"]["sha256"] = digest
    with pytest.raises(ValueError, match="same audio"):
        assign_song_splits(data, review())


def test_identical_audio_under_a_reviewed_song_alias_remains_in_the_same_split():
    data = records()
    data[-1]["sources"]["wav"]["sha256"] = data[0]["sources"]["wav"]["sha256"]
    result = assign_song_splits(data, review())
    assignments = {row["segment_id"]: row["split"] for row in result["assignments"]}
    assert assignments[data[0]["segment_id"]] == assignments[data[-1]["segment_id"]]


def test_duplicates_invalid_ratios_and_unknown_parent_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        assign_song_splits(records() + records()[:1], review())
    with pytest.raises(ValueError, match="ratios"):
        assign_song_splits(records(), review(), ratios={"train": float("nan"), "val": 0.1, "test": 0.1})
    data = records()
    data[0]["parent_segment_id"] = "missing"
    with pytest.raises(ValueError, match="Unknown parent"):
        assign_song_splits(data, review())


def test_cli_refuses_existing_output_and_preserves_contents(tmp_path):
    manifest, identities, output = (tmp_path / name for name in ("segments.jsonl", "review.json", "split.json"))
    manifest.write_text("\n".join(json.dumps(row) for row in records()))
    identities.write_text(json.dumps(review()))
    args = ["--segments", str(manifest), "--song-review", str(identities), "--output", str(output)]
    assert cli(args) == 0
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        cli(args)
    assert output.read_bytes() == before


def test_singer_holdout_reserves_shared_song_and_reports_evaluation_subset():
    data = [record(song, singer, singer=singer)
            for song, singers in (("first", ("holdout", "other")), ("second", ("other",)),
                                  ("third", ("other",)), ("fourth", ("other",))) for singer in singers]
    result = assign_song_splits(data, review(), held_out_singers=["holdout"])
    assignments = {row["segment_id"]: row["split"] for row in result["assignments"]}
    assert assignments["first-holdout"] == assignments["first-other"] == "test"
    assert result["singer_independent_test_segment_ids"] == ["first-holdout"]
    assert result["singer_independent"] is True
    assert result == assign_song_splits(reversed(data), review(), held_out_singers=["holdout"])


def test_singer_holdout_fails_when_shared_songs_leave_no_training_or_singer_is_unknown():
    data = [dict(row, singer="holdout") for row in records()]
    with pytest.raises(ValueError, match="too few"):
        assign_song_splits(data, review(), held_out_singers=["holdout"])
    with pytest.raises(ValueError, match="Unknown"):
        assign_song_splits(data, review(), held_out_singers=["missing"])

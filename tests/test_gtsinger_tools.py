import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import fetch_gtsinger_samples as fetch
from scripts import build_gtsinger_representative_manifest as builder
from scripts import prepare_gtsinger as prepare


class Response(io.BytesIO):
    def __init__(self, body, url, content_length=None):
        super().__init__(body)
        self.url = url
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, body, url=None, content_length=None):
        self.body = body
        self.url = url
        self.content_length = content_length
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request.full_url, timeout))
        return Response(self.body, self.url or request.full_url, self.content_length)


def file_entry(body=b"fixture audio"):
    return {"path": fetch.STEMS[0] + ".wav", "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest()}


def fixture_inputs(tmp_path):
    manifest = json.loads(fetch.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    root = tmp_path / "source"
    for entry in manifest["files"]:
        body = ("fixture for " + entry["path"]).encode("utf-8")
        path = root / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        entry.clear()
        entry.update(path=path.relative_to(root).as_posix(), size=len(body),
                     sha256=hashlib.sha256(body).hexdigest())
    manifest_path = tmp_path / "samples.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest_path, manifest


def test_download_verifies_hash_and_reuses_without_network(tmp_path):
    body = b"fixture audio"
    entry = file_entry(body)
    opener = Opener(body)
    result = fetch.download_file(tmp_path, entry, opener)
    assert result["status"] == "downloaded"
    assert opener.calls == [(fetch.source_url(entry), 30)]
    assert (tmp_path / entry["path"]).read_bytes() == body
    assert fetch.download_file(tmp_path, entry, opener)["status"] == "reused"
    assert len(opener.calls) == 1
    assert not list(tmp_path.rglob(".download-*"))


def test_existing_different_file_is_never_overwritten(tmp_path):
    entry = file_entry()
    target = tmp_path / entry["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original user file")
    opener = Opener(b"fixture audio")
    with pytest.raises(ValueError, match="mismatch"):
        fetch.download_file(tmp_path, entry, opener)
    assert target.read_bytes() == b"original user file"
    assert not opener.calls


@pytest.mark.parametrize("body,expected_error", [(b"too long beyond declared", "exceeded"),
                                                  (b"wrong audio!!", "checksum"),
                                                  (b"short", "size/type")])
def test_invalid_download_has_no_partial_output(tmp_path, body, expected_error):
    entry = file_entry()
    with pytest.raises(ValueError, match=expected_error):
        fetch.download_file(tmp_path, entry, Opener(body))
    assert not (tmp_path / entry["path"]).exists()
    assert not list(tmp_path.rglob(".download-*"))


def test_git_blob_hash_verifies_annotation_bytes(tmp_path):
    body = b"annotation\n"
    path = tmp_path / "source"
    path.write_bytes(body)
    entry = {"path": "dataset_license.md", "size": len(body),
             "git_blob_sha1": hashlib.sha1(("blob %d\0" % len(body)).encode() + body).hexdigest()}
    assert fetch.verify_file(path, entry) == hashlib.sha256(body).hexdigest()


@pytest.mark.parametrize("path", ["../escape", "/absolute", "ok/../escape", "a\\b", "a//b"])
def test_unsafe_paths_rejected(tmp_path, path):
    with pytest.raises(ValueError, match="Unsafe"):
        fetch.safe_destination(tmp_path, path)


def test_symlink_destination_rejected(tmp_path):
    target = tmp_path / "actual"
    target.mkdir()
    (tmp_path / "link").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        fetch.safe_destination(tmp_path, "link/file")


@pytest.mark.parametrize("url", ["http://huggingface.co/file", "https://evil.example/file",
                               "https://huggingface.co.evil.example/file", "https://user@huggingface.co/file",
                               "https://huggingface.co:444/file"])
def test_redirect_origin_is_restricted(url):
    with pytest.raises(ValueError, match="origin"):
        fetch.RestrictedRedirects().redirect_request(None, None, 302, "", {}, url)


def test_manifest_fixed_revision_selection_and_total_limit(tmp_path):
    _, manifest_path, manifest = fixture_inputs(tmp_path)
    assert fetch.load_manifest(manifest_path)["revision"] == fetch.REVISION
    for mutation in (lambda m: m.update(revision="main"),
                     lambda m: m["files"][0].update(path="../escape"),
                     lambda m: m["files"][0].update(size=fetch.MAX_FILE_BYTES + 1),
                     lambda m: [entry.update(size=fetch.MAX_FILE_BYTES) for entry in m["files"]]):
        changed = copy.deepcopy(manifest)
        mutation(changed)
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            fetch.load_manifest(manifest_path)


class FakeSegment:
    def __init__(self, stem, song_id):
        self.stem, self.song_id = stem, song_id

    def to_dict(self):
        return {"segment_id": self.stem, "song_id": self.song_id, "language": self.stem.split("/")[0],
                "sample_rate": 48000, "audio_path": self.stem + ".wav",
                "validation_issues": [{"code": "json_time_difference", "path": "words[0]", "message": "fixture"}],
                "units": [
                    {"text": "hello", "kind": "lyric", "boundary": {"source": "textgrid:first_phone", "candidate_seconds": 0.125, "candidate_sample": 6000}},
                    {"text": "SP", "kind": "silence", "boundary": None},
                ]}


def test_prepare_outputs_pending_independent_reviews_without_accuracy(tmp_path):
    root, manifest_path, manifest = fixture_inputs(tmp_path)
    calls = []
    def loader(root_path, stem, *, dataset_revision, song_id):
        calls.append((root_path, stem, dataset_revision, song_id))
        return FakeSegment(stem, song_id)
    out = tmp_path / "prepared"
    summary = prepare.prepare_samples(root, manifest_path, out, segment_loader=loader)
    assert len(calls) == 2
    assert all(call[2] == fetch.REVISION for call in calls)
    reviews = json.loads((out / "manual-review.json").read_text())
    assert len(reviews) == 2
    assert not any(key.startswith("candidate_") for row in reviews for key in row)
    candidates = json.loads((out / "annotation-candidates.json").read_text())
    assert candidates[0]["candidate_sample"] == 6000
    assert all(r["human_truth_sample"] is None and r["review_status"] == "pending" for r in reviews)
    with (out / "manual-review.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["human_truth_sample"] == ""
    assert rows[0]["reviewer"] == ""
    assert not any(key.startswith("candidate_") for key in rows[0])
    assert summary["accuracy"]["status"] == "unavailable"
    assert summary["accuracy"]["mae_ms"] is None
    assert summary["independent_reviewed_boundaries"] == 0
    assert summary["validation_issue_count"] == 2
    assert summary["unit_kind_counts"] == {"lyric": 2, "silence": 2}
    assert summary["split_status"] == "unassigned"
    assert all(json.loads(line)["split"] == "unassigned" for line in (out / "segments.jsonl").read_text().splitlines())
    assert json.loads((out / "source-manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        prepare.prepare_samples(root, manifest_path, out, segment_loader=loader)


@pytest.mark.parametrize("on_invalid", ["fail", "exclude"])
def test_prepare_rejects_changed_source_before_parsing(tmp_path, on_invalid):
    root, manifest_path, manifest = fixture_inputs(tmp_path)
    (root / manifest["files"][0]["path"]).write_text("changed", encoding="utf-8")
    out = tmp_path / "prepared"
    with pytest.raises(ValueError, match="mismatch"):
        prepare.prepare_samples(root, manifest_path, out, on_invalid=on_invalid,
                                segment_loader=lambda *a, **kw: pytest.fail("must verify first"))
    assert not out.exists()


def test_prepare_parser_failure_does_not_publish_result(tmp_path):
    root, manifest_path, _ = fixture_inputs(tmp_path)
    def failed_loader(*args, **kwargs):
        raise ValueError("unreadable source annotation")
    with pytest.raises(ValueError, match="unreadable"):
        prepare.prepare_samples(root, manifest_path, tmp_path / "prepared", segment_loader=failed_loader)
    assert not (tmp_path / "prepared").exists()


def test_prepare_fail_vs_explicit_exclude_records_full_error(tmp_path):
    root, manifest_path, _ = fixture_inputs(tmp_path)
    reason = "source.json[12]: interval [6.77, 7.34] exceeds WAV duration 7.339979166666667"
    def loader(_, stem, **kwargs):
        if stem == fetch.STEMS[0]:
            raise ValueError(reason)
        return FakeSegment(stem, kwargs["song_id"])
    with pytest.raises(ValueError, match="source.json"):
        prepare.prepare_samples(root, manifest_path, tmp_path / "failed", segment_loader=loader)
    assert not (tmp_path / "failed").exists()
    out = tmp_path / "excluded"
    result = prepare.prepare_samples(root, manifest_path, out, segment_loader=loader, on_invalid="exclude")
    assert result["requested_segment_count"] == 2
    assert result["usable_segment_count"] == result["segment_count"] == 1
    assert result["excluded_segment_count"] == 1
    excluded = json.loads((out / "excluded-segments.json").read_text())
    assert excluded == result["excluded_segments"]
    assert excluded[0]["segment_id"] == fetch.STEMS[0] and excluded[0]["reason"] == reason
    assert result["validation_issue_counts"] == {"invalid_segment_excluded": 1, "json_time_difference": 1}
    assert result["validation_issue_count"] == 2
    assert result["accuracy"]["status"] == "unavailable"
    assert len(json.loads((out / "manual-review.json").read_text())) == 1


def test_prepare_exclude_all_rejected_still_fails_without_results(tmp_path):
    root, manifest_path, _ = fixture_inputs(tmp_path)
    def invalid_loader(*args, **kwargs):
        raise ValueError("invalid source interval")
    with pytest.raises(ValueError, match="No usable segments.*2 requested"):
        prepare.prepare_samples(root, manifest_path, tmp_path / "rejected", on_invalid="exclude",
                                segment_loader=invalid_loader)
    assert not (tmp_path / "rejected").exists()
    assert not list(tmp_path.glob(".gtsinger-prepare-*"))


def test_prepare_exclude_only_catches_loader_value_errors(tmp_path):
    root, manifest_path, _ = fixture_inputs(tmp_path)
    def unavailable_loader(*args, **kwargs):
        raise OSError("audio storage unavailable")
    with pytest.raises(OSError, match="storage unavailable"):
        prepare.prepare_samples(root, manifest_path, tmp_path / "unavailable", on_invalid="exclude",
                                segment_loader=unavailable_loader)
    assert not (tmp_path / "unavailable").exists()


def representative_fixture(tmp_path):
    manifest = json.loads(fetch.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    manifest["selection"] = "representative-v1"
    manifest["samples"] = [{"relative_stem": stem, "song_id": "/".join((stem.split("/")[0], stem.split("/")[3])).lower(),
                            "song_identity_review_status": "pending", "split": "unassigned"}
                           for stem in fetch.REPRESENTATIVE_STEMS]
    manifest["files"] = []
    root = tmp_path / "sources"
    paths = ["dataset_license.md"]
    for stem in fetch.REPRESENTATIVE_STEMS:
        paths.extend(stem + suffix for suffix in (".wav", ".json", ".musicxml", "_TextGrid"))
    for relative_path in paths:
        body = relative_path.encode("utf-8")
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        manifest["files"].append({"path": relative_path, "size": len(body),
                                  "sha256": hashlib.sha256(body).hexdigest()})
    manifest_path = tmp_path / "representative.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest_path, manifest


def test_representative_manifest_is_exact_revision_pinned_and_bounded(tmp_path):
    _, path, manifest = representative_fixture(tmp_path)
    assert len(fetch.load_manifest(path)["samples"]) == 20
    for mutate in (
        lambda m: m["samples"][0].update(relative_stem="Chinese/other/other/title/group/0000"),
        lambda m: m["samples"][0].update(song_identity_review_status="approved"),
        lambda m: m["samples"][0].update(split="test"),
        lambda m: m["files"].append(copy.deepcopy(m["files"][0])),
        lambda m: [entry.update(size=fetch.MAX_FILE_BYTES) for entry in m["files"]],
        lambda m: m.update(selection="full-corpus"),
    ):
        changed = copy.deepcopy(manifest)
        mutate(changed)
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            fetch.load_manifest(path)


def test_representative_selection_prepares_blind_rows_and_pending_song_identity(tmp_path):
    root, manifest_path, _ = representative_fixture(tmp_path)
    result = prepare.prepare_samples(root, manifest_path, tmp_path / "prepared",
                                     segment_loader=lambda _, stem, **kw: FakeSegment(stem, kw["song_id"]))
    assert result["segment_count"] == 20 and result["song_count"] == 6
    assert len(result["singer_counts"]) == 5
    assert result["group_counts"] == {"Control_Group": 10, "Breathy_Group": 3, "Glissando_Group": 7}
    assert result["song_identity_review_status"] == "pending"
    reviews = json.loads((tmp_path / "prepared/manual-review.json").read_text())
    assert len(reviews) == 20
    assert all(set(row) == set(prepare.REVIEW_FIELDS) for row in reviews)
    assert all(row["human_truth_sample"] is None and row["human_truth_seconds"] is None for row in reviews)
    assert not any("candidate" in key or "start" in key or "end" in key for row in reviews for key in row)


def test_publication_never_replaces_even_empty_existing_directory(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "source.txt").write_text("new result", encoding="utf-8")
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        prepare.publish_directory(staging, target)
    assert staging.is_dir() and not list(target.iterdir())
    target.rmdir()
    prepare.publish_directory(staging, target)
    assert not staging.exists()
    assert (target / "source.txt").read_text() == "new result"


def test_prepare_concurrent_destination_is_preserved_and_staging_removed(tmp_path, monkeypatch):
    root, manifest_path, _ = fixture_inputs(tmp_path)
    destination = tmp_path / "prepared"
    publish = prepare.publish_directory
    def concurrent_publish(staging, out_dir):
        out_dir.mkdir()
        (out_dir / "other.txt").write_text("other result", encoding="utf-8")
        publish(staging, out_dir)
    monkeypatch.setattr(prepare, "publish_directory", concurrent_publish)
    with pytest.raises(FileExistsError):
        prepare.prepare_samples(root, manifest_path, destination,
                                segment_loader=lambda _, stem, **kw: FakeSegment(stem, kw["song_id"]))
    assert (destination / "other.txt").read_text() == "other result"
    assert not list(tmp_path.glob(".gtsinger-prepare-*"))


def test_manifest_builder_finds_reordered_license_and_preserves_source_hashes(monkeypatch):
    smoke = json.loads(fetch.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    license_entry = copy.deepcopy(next(entry for entry in smoke["files"]
                                     if entry["path"] == smoke["license"]["path"]))
    smoke["files"].reverse()
    assert smoke["files"][0]["path"] != license_entry["path"]
    stem = fetch.REPRESENTATIVE_STEMS[0]
    metadata = [{"type": "file", "path": stem + suffix, "size": 12, "oid": "a" * 40}
                for suffix in (".wav", ".json", ".musicxml", ".TextGrid")]
    metadata[0]["lfs"] = {"oid": "b" * 64}
    def read_metadata(url, opener):
        if "/api/datasets/" in url:
            return metadata, "c" * 64
        return [{"word": "la", "end_time": 0.25, "glissando": [1]}], "d" * 64
    monkeypatch.setattr(builder, "load_manifest", lambda _: smoke)
    monkeypatch.setattr(builder, "REPRESENTATIVE_STEMS", (stem,))
    monkeypatch.setattr(builder, "read_json_url", read_metadata)
    manifest = builder.build_manifest()
    assert manifest["files"][0] == license_entry
    assert manifest["files"][1]["sha256"] == "b" * 64
    assert manifest["files"][2]["git_blob_sha1"] == "a" * 40
    assert manifest["metadata_evidence"][0]["json_content_sha256"] == "d" * 64


@pytest.mark.parametrize("license_count", [0, 2])
def test_manifest_builder_requires_unique_license_before_metadata_requests(monkeypatch, license_count):
    smoke = json.loads(fetch.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    license_entry = next(entry for entry in smoke["files"] if entry["path"] == smoke["license"]["path"])
    smoke["files"] = [entry for entry in smoke["files"] if entry["path"] != license_entry["path"]]
    smoke["files"].extend(copy.deepcopy(license_entry) for _ in range(license_count))
    monkeypatch.setattr(builder, "load_manifest", lambda _: smoke)
    monkeypatch.setattr(builder, "read_json_url", lambda *args: pytest.fail("must locate license first"))
    with pytest.raises(ValueError, match="exactly one source license file"):
        builder.build_manifest()


@pytest.mark.parametrize("failure_stage", ["build", "validation", "temporary", "publication"])
def test_manifest_builder_cli_reports_failures_and_cleans_temporary_files(
        tmp_path, monkeypatch, capsys, failure_stage):
    _, _, manifest = representative_fixture(tmp_path)
    output = tmp_path / "output.json"
    if failure_stage == "build":
        def failed_builder():
            raise ValueError("invalid source metadata")
        monkeypatch.setattr(builder, "build_manifest", failed_builder)
    else:
        monkeypatch.setattr(builder, "build_manifest", lambda: manifest)
    if failure_stage == "validation":
        manifest["revision"] = "main"
    elif failure_stage == "temporary":
        output = tmp_path / "missing-parent/output.json"
    elif failure_stage == "publication":
        native_link = builder.os.link
        def concurrent_link(source, target):
            Path(target).write_text("concurrent manifest", encoding="utf-8")
            native_link(source, target)
        monkeypatch.setattr(builder.os, "link", concurrent_link)
    with pytest.raises(SystemExit) as failure:
        builder.main(["--output", str(output)])
    assert failure.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "GTSinger manifest construction failed:" in captured.err
    assert "Traceback" not in captured.err
    assert not list(output.parent.glob(".gtsinger-manifest-*"))
    if failure_stage == "publication":
        assert output.read_text() == "concurrent manifest"
    else:
        assert not output.exists()


def test_manifest_builder_refuses_existing_output_before_source_requests(tmp_path, monkeypatch, capsys):
    output = tmp_path / "existing.json"
    output.write_text("preserved manifest", encoding="utf-8")
    monkeypatch.setattr(builder, "build_manifest", lambda: pytest.fail("must not inspect metadata"))
    with pytest.raises(SystemExit) as failure:
        builder.main(["--output", str(output)])
    assert failure.value.code != 0
    assert output.read_text() == "preserved manifest"
    assert "Manifest output already exists" in capsys.readouterr().err


@pytest.mark.parametrize("platform,symbol", [("linux", "renameat2"), ("darwin", "renamex_np")])
def test_missing_exclusive_rename_refuses_publication_without_fallback(
        tmp_path, monkeypatch, platform, symbol):
    root, manifest_path, _ = fixture_inputs(tmp_path)
    staging, existing = tmp_path / "staging", tmp_path / "existing"
    staging.mkdir()
    (staging / "source.txt").write_text("preserved staging", encoding="utf-8")
    existing.mkdir()
    monkeypatch.setattr(prepare, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(prepare.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(prepare.os, "rename", lambda *args: pytest.fail("must not fall back to rename"))
    with pytest.raises(OSError, match=symbol):
        prepare.publish_directory(staging, existing)
    assert (staging / "source.txt").read_text() == "preserved staging"
    assert not list(existing.iterdir())
    destination = tmp_path / "prepared"
    with pytest.raises(OSError, match=symbol):
        prepare.prepare_samples(root, manifest_path, destination,
                                segment_loader=lambda _, stem, **kw: FakeSegment(stem, kw["song_id"]))
    assert not destination.exists()
    assert not list(tmp_path.glob(".gtsinger-prepare-*"))

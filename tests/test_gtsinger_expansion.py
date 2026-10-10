import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from scripts import expand_gtsinger_training as expansion


class Response(io.BytesIO):
    def __init__(self, body, url, headers=None):
        super().__init__(body)
        self.url = url
        self.headers = {} if headers is None else headers

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, bodies, *, redirect=None, headers=None):
        self.bodies, self.redirect, self.headers, self.calls = bodies, redirect, headers, []

    def open(self, request, timeout):
        assert timeout == 30
        self.calls.append(request.full_url)
        return Response(self.bodies[request.full_url], self.redirect or request.full_url, self.headers)


class Client:
    def __init__(self, listings):
        self.listings = listings
        self.evidence = []

    def list_directory(self, directory):
        self.evidence.append({"directory": directory})
        return self.listings[directory]


def content_record(path, body):
    return {"path": path, "size": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            "source_url": expansion.resolve_url(path)}


@pytest.fixture
def plan_fixture(tmp_path):
    base, frozen = expansion._baseline(expansion.REPO_ROOT / "data/gtsinger-representative.json",
                                      expansion.REPO_ROOT / "reports/p1/source-mode-20261010/splits.json")
    listings = {}
    for directory in expansion.GROUP_DIRECTORIES:
        records = []
        for index in range(12):
            stem = directory + "/%04d" % index
            for extension in (".wav", ".json", ".musicxml", ".TextGrid"):
                path = stem + extension
                body = path.encode("utf-8")
                records.append({"type": "file", "path": path, "size": len(body),
                                "lfs": {"size": len(body), "oid": hashlib.sha256(body).hexdigest()}})
        listings[directory] = records
    # Match the frozen source metadata for every original quartet, including _TextGrid.
    for sample in base["samples"]:
        stem = sample["relative_stem"]
        directory = stem.rsplit("/", 1)[0]
        records = [entry for entry in listings[directory] if not entry["path"].startswith(stem + ".")]
        for entry in base["files"]:
            if entry["path"] not in {stem + suffix for suffix in (".wav", ".json", ".musicxml", ".TextGrid", "_TextGrid")}:
                continue
            record = dict(entry, type="file")
            if "sha256" in entry:
                record["lfs"] = {"oid": record.pop("sha256"), "size": entry["size"]}
            else:
                record["oid"] = record.pop("git_blob_sha1")
            records.append(record)
        listings[directory] = records
    for technique in expansion.TECHNIQUES:
        directory = "English/EN-Tenor-1/" + technique
        listings[directory] = [{"type": "directory", "path": directory + "/Rolling in the Deep"},
                               {"type": "directory", "path": directory + "/Someone Like You"}]
    client = Client(listings)
    plan = expansion.build_plan(expansion.REPO_ROOT / "data/gtsinger-representative.json",
                               expansion.REPO_ROOT / "reports/p1/source-mode-20261010/splits.json",
                               client=client)
    return plan, listings, base, frozen


def test_metadata_plan_preserves_baseline_and_bilingual_split_coverage(plan_fixture):
    plan, listings, base, frozen = plan_fixture
    assert len(plan["samples"]) == 180
    assert plan["split_overrides"]["frozen_assignments"] == frozen["assignments"]
    assert {sample["relative_stem"] for sample in base["samples"]} <= {sample["relative_stem"] for sample in plan["samples"]}
    assert plan["declared_total_bytes"] <= 500_000_000
    assert len(plan["metadata_evidence"]) == len(expansion.GROUP_DIRECTORIES) + 5 == 49
    assert {row["split"] for row in plan["split_overrides"]["assignments"] if "/EN-Tenor-1/" in row["segment_id"]} == {"test"}
    assert all(sample["song_identity_review_status"] == "pending" for sample in plan["samples"])
    assert len(plan["files"]) == 4 * 180 + 1
    json.dumps(plan, allow_nan=False)


def test_metadata_plan_deterministic_and_no_audio_fetch(plan_fixture):
    first, listings, _, _ = plan_fixture
    second = expansion.build_plan(expansion.REPO_ROOT / "data/gtsinger-representative.json",
                                  expansion.REPO_ROOT / "reports/p1/source-mode-20261010/splits.json",
                                  client=Client(listings))
    assert first == second
    assert all(entry["source_url"].startswith("https://huggingface.co/datasets/") for entry in first["files"])
    assert first["source_terms"]["license_path"] == "dataset_license.md"


def test_planner_rejects_new_song_shared_with_held_out_singer(plan_fixture):
    _, listings, _, _ = plan_fixture
    directory = "English/EN-Tenor-1/Pharyngeal"
    listings[directory].append({"type": "directory", "path": directory + "/God Is A Girl"})
    with pytest.raises(ValueError, match="held-out singer"):
        expansion.build_plan(expansion.REPO_ROOT / "data/gtsinger-representative.json",
                             expansion.REPO_ROOT / "reports/p1/source-mode-20261010/splits.json",
                             client=Client(listings))


@pytest.mark.parametrize("value", [0, 21, True, 2.5])
def test_invalid_per_song_budget_rejected_before_network(value):
    with pytest.raises(ValueError, match="segments_per_song"):
        expansion.build_plan("unused", "unused", segments_per_song=value)


def test_per_song_budget_cannot_discard_original_sources(plan_fixture):
    _, listings, _, _ = plan_fixture
    with pytest.raises(ValueError, match="preserve all existing"):
        expansion.build_plan(expansion.REPO_ROOT / "data/gtsinger-representative.json",
                             expansion.REPO_ROOT / "reports/p1/source-mode-20261010/splits.json",
                             segments_per_song=1, client=Client(listings))


def test_global_segment_budget_rejects_twenty_per_song(plan_fixture):
    _, listings, _, _ = plan_fixture
    with pytest.raises(ValueError, match="segment budget"):
        expansion.build_plan(expansion.REPO_ROOT / "data/gtsinger-representative.json",
                             expansion.REPO_ROOT / "reports/p1/source-mode-20261010/splits.json",
                             segments_per_song=20, client=Client(listings))


def test_baseline_source_change_is_rejected(plan_fixture):
    _, listings, base, _ = plan_fixture
    stem = base["samples"][0]["relative_stem"]
    entry = next(entry for entry in listings[stem.rsplit("/", 1)[0]] if entry["path"] == stem + ".wav")
    entry["lfs"]["oid"] = "f" * 64
    with pytest.raises(ValueError, match="Existing source checksum"):
        expansion.build_plan(expansion.REPO_ROOT / "data/gtsinger-representative.json",
                             expansion.REPO_ROOT / "reports/p1/source-mode-20261010/splits.json",
                             client=Client(listings))


@pytest.mark.parametrize("mutation,match", [
    (lambda plan: plan.update(revision="f" * 40), "Unsupported"),
    (lambda plan: plan.update(languages=["French"]), "Unsupported"),
    (lambda plan: plan["files"].pop(), "quartet"),
    (lambda plan: plan["files"].append(dict(plan["files"][0])), "unique"),
    (lambda plan: plan["files"][0].update(source_url="https://example.com/license"), "Source URL"),
    (lambda plan: plan["files"][0].update(size=4 * 1024 * 1024 + 1), "byte budget"),
    (lambda plan: plan.update(declared_total_bytes=0), "byte budget"),
    (lambda plan: plan["budgets"].update(max_total_bytes=1_000_000_000), "enlarged"),
    (lambda plan: plan["samples"][0].update(split="val"), "split"),
    (lambda plan: plan["samples"][0].update(song_identity_review_status="approved"), "unreviewed"),
    (lambda plan: plan["samples"][0].update(relative_stem="../escape/0000"), "catalog"),
    (lambda plan: plan["split_overrides"]["frozen_assignments"].pop(), "retained"),
    (lambda plan: plan["split_overrides"].update(held_out_singers=[]), "holdout"),
    (lambda plan: plan["held_out_catalog_song_ids"].append("english/god is a girl"), "held-out"),
    (lambda plan: plan["source_terms"].update(source_url="https://example.com/license"), "license"),
    (lambda plan: plan["source_terms"].update(identifier="CC0"), "license"),
    (lambda plan: plan.update(held_out_catalog_song_ids=[]), "catalog"),
])
def test_manifest_tampering_rejected(plan_fixture, mutation, match):
    plan = copy.deepcopy(plan_fixture[0])
    mutation(plan)
    with pytest.raises(ValueError, match=match):
        expansion.validate_plan(plan)


def test_same_waveform_cannot_cross_splits(plan_fixture):
    plan = copy.deepcopy(plan_fixture[0])
    train = next(sample["relative_stem"] for sample in plan["samples"] if sample["split"] == "train")
    test = next(sample["relative_stem"] for sample in plan["samples"] if sample["split"] == "test")
    files = {entry["path"]: entry for entry in plan["files"]}
    files[test + ".wav"]["sha256"] = files[train + ".wav"]["sha256"]
    with pytest.raises(ValueError, match="WAV crosses"):
        expansion.validate_plan(plan)


def test_identical_waveform_cannot_have_different_song_identity_even_within_split(plan_fixture):
    plan = copy.deepcopy(plan_fixture[0])
    train = [sample for sample in plan["samples"] if sample["split"] == "train"]
    first = train[0]
    second = next(sample for sample in train if sample["song_id"] != first["song_id"])
    files = {entry["path"]: entry for entry in plan["files"]}
    files[second["relative_stem"] + ".wav"]["sha256"] = files[first["relative_stem"] + ".wav"]["sha256"]
    with pytest.raises(ValueError, match="conflicting dataset song"):
        expansion.validate_plan(plan)


def test_metadata_client_only_reads_allowed_direct_listing_with_bounded_body():
    directory = sorted(expansion.GROUP_DIRECTORIES)[0]
    entry = {"type": "file", "path": directory + "/0000.json", "size": 3, "oid": "a" * 40}
    client = expansion.MetadataClient()
    url = "https://huggingface.co/api/datasets/%s/tree/%s/%s?recursive=false&expand=false" % (
        expansion.REPOSITORY, expansion.REVISION, expansion.quote(directory, safe="/"))
    client.opener = Opener({url: json.dumps([entry]).encode()})
    assert client.list_directory(directory) == [entry]
    assert len(client.evidence) == 1
    with pytest.raises(ValueError, match="outside"):
        client.list_directory("English")


@pytest.mark.parametrize("kind", ["large", "pagination", "redirect", "non_child"])
def test_metadata_client_rejects_unbounded_or_unapproved_response(kind):
    directory = sorted(expansion.GROUP_DIRECTORIES)[0]
    url = "https://huggingface.co/api/datasets/%s/tree/%s/%s?recursive=false&expand=false" % (
        expansion.REPOSITORY, expansion.REVISION, expansion.quote(directory, safe="/"))
    body = b"x" * (expansion.MAX_METADATA_BYTES + 1) if kind == "large" else b"[]"
    if kind == "non_child":
        body = json.dumps([{"path": "French/escape", "type": "file"}]).encode()
    opener = Opener({url: body}, headers={"Link": "next"} if kind == "pagination" else None,
                    redirect="https://example.com/no" if kind == "redirect" else None)
    with pytest.raises(ValueError):
        expansion.MetadataClient(opener).list_directory(directory)


def test_download_is_verified_atomic_and_reuses_existing(tmp_path):
    body = b"audio-content"
    entry = content_record("English/EN-Alto-1/Glissando/my love/Control_Group/0001.wav", body)
    opener = Opener({entry["source_url"]: body})
    assert expansion._download(tmp_path, entry, opener)["status"] == "downloaded"
    assert expansion._download(tmp_path, entry, opener)["status"] == "reused"
    assert len(opener.calls) == 1
    assert not list(tmp_path.rglob(".download-*"))
    target = tmp_path / entry["path"]
    target.write_bytes(b"not-identical")
    with pytest.raises(ValueError):
        expansion._download(tmp_path, entry, opener)
    assert target.read_bytes() == b"not-identical"


@pytest.mark.parametrize("kind", ["oversize", "checksum", "redirect", "length"])
def test_failed_download_leaves_no_partial_target(tmp_path, kind):
    body = b"expected"
    entry = content_record("dataset_license.md", body)
    delivered = body + b"x" if kind == "oversize" else (b"different" if kind == "checksum" else body)
    opener = Opener({entry["source_url"]: delivered},
                    redirect="https://other.example/license" if kind == "redirect" else None,
                    headers={"Content-Length": "999"} if kind == "length" else None)
    with pytest.raises(ValueError):
        expansion._download(tmp_path, entry, opener)
    assert not (tmp_path / entry["path"]).exists()
    assert not list(tmp_path.glob(".download-*"))


def test_source_reuse_copies_without_network_or_modifying_originals(tmp_path, monkeypatch):
    body = b"license-source"
    entry = content_record("dataset_license.md", body)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(expansion, "load_plan", lambda _: {"files": [entry], "license": {}, "declared_total_bytes": len(body)})
    original = tmp_path / "original"
    original.mkdir()
    (original / entry["path"]).write_bytes(body)
    opener = Opener({})
    output = tmp_path / "fetched"
    result = expansion.fetch_plan(manifest, output, reuse_root=original, opener=opener)
    assert result["files"][0]["status"] == "copied_existing"
    assert not opener.calls
    assert (original / entry["path"]).read_bytes() == (output / entry["path"]).read_bytes() == body


def test_plan_publication_refuses_existing_even_empty_directory(plan_fixture, tmp_path):
    plan = plan_fixture[0]
    output = tmp_path / "plan"
    summary = expansion.publish_plan(plan, output)
    assert summary["song_counts"] == {"train": 10, "val": 4, "test": 4}
    assert summary["downloaded_audio_bytes"] == 0
    assert expansion.load_plan(output / "manifest.json") == plan
    with pytest.raises(FileExistsError):
        expansion.publish_plan(plan, output)
    assert not list(tmp_path.glob(".gtsinger-plan-*"))


def test_split_overrides_use_existing_relation_validation_and_keep_assignments(plan_fixture):
    plan = plan_fixture[0]
    segments = [{"segment_id": sample["relative_stem"], "song_id": sample["song_id"],
                 "singer": sample["relative_stem"].split("/")[1],
                 "sources": {"wav": {"sha256": hashlib.sha256(sample["relative_stem"].encode()).hexdigest()}}}
                for sample in plan["samples"]]
    split = expansion._validated_splits(segments, plan)
    assert split["assignments"] == plan["split_overrides"]["assignments"]
    assert split["song_counts"] == {"train": 10, "val": 4, "test": 4}
    assert split["song_identity_reviewed"] is False
    segments[0]["parent_segment_id"] = segments[-1]["segment_id"]
    with pytest.raises(ValueError, match="different song"):
        expansion._validated_splits(segments, plan)


def test_prepare_keeps_reader_schema_and_excludes_invalid_without_approval(tmp_path, monkeypatch):
    sample = {"relative_stem": "English/EN-Tenor-1/Glissando/Someone Like You/Control_Group/0000",
              "song_id": "english/someone like you", "split": "test"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    body = b"license"
    source = tmp_path / "source"
    source.mkdir()
    (source / "dataset_license.md").write_bytes(body)
    plan = {"files": [content_record("dataset_license.md", body)], "license": {}, "samples": [sample, dict(sample, relative_stem=sample["relative_stem"][:-1] + "1")]}
    monkeypatch.setattr(expansion, "load_plan", lambda _: plan)
    monkeypatch.setattr(expansion, "_validated_splits", lambda segments, _: {
        "song_counts": {"test": 1}, "segment_counts": {"test": 1},
        "assignments": [{"segment_id": segments[0]["segment_id"], "canonical_song_id": sample["song_id"], "split": "test"}]})
    boundary = {"candidate_sample": 0, "human_truth_sample": None, "review_status": "pending"}

    def reader(root, stem, **options):
        if stem.endswith("0001"):
            raise ValueError("bad dataset annotation")
        return SimpleNamespace(to_dict=lambda: {"segment_id": stem, "song_id": options["song_id"],
            "language": "English", "singer": "EN-Tenor-1", "sample_rate": 48000, "num_samples": 1,
            "source_recording_offset_samples": None, "review_status": "pending",
            "units": [{"kind": "lyric", "text": "hello", "boundary": boundary}],
            "sources": {"wav": {"sha256": "a" * 64}}})

    output = tmp_path / "prepared"
    summary = expansion.prepare_plan(manifest, source, output, segment_loader=reader)
    segment = json.loads((output / "segments.jsonl").read_text())
    assert summary["excluded_segment_count"] == 1
    assert summary["review_status"] == segment["review_status"] == "pending"
    assert segment["source_recording_offset_samples"] is None
    assert segment["units"][0]["boundary"] == boundary
    assert summary["accuracy"]["status"] == "unavailable"
    assert segment["split"] == "test"
    with pytest.raises(FileExistsError):
        expansion.prepare_plan(manifest, source, output, segment_loader=reader)
    with pytest.raises(ValueError, match="bad dataset"):
        expansion.prepare_plan(manifest, source, tmp_path / "failed", segment_loader=reader, on_invalid="fail")
    assert not (tmp_path / "failed").exists()


def test_prepared_output_is_accepted_by_coarse_benchmark_input_contract(tmp_path, monkeypatch):
    from scripts.benchmark_coarse_gtsinger import load_inputs
    selected = [
        ("Chinese/ZH-Alto-1/Breathy/不再见/Control_Group/0000", "train"),
        ("Chinese/ZH-Alto-1/Breathy/慢慢喜欢你/Control_Group/0000", "val"),
        ("English/EN-Tenor-1/Glissando/Someone Like You/Control_Group/0000", "test"),
    ]
    source = tmp_path / "source"
    source.mkdir()
    (source / "dataset_license.md").write_bytes(b"license")
    files = [content_record("dataset_license.md", b"license")]
    samples, values = [], {}
    for index, (stem, split) in enumerate(selected):
        path = source / (stem + ".wav")
        path.parent.mkdir(parents=True)
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, 48000, 128, "NONE", "not compressed"))
            output.writeframes(bytes([index + 1, 0]) * 128)
        record = content_record(stem + ".wav", path.read_bytes())
        files.append(record)
        samples.append({"relative_stem": stem, "song_id": expansion.song_id(stem), "split": split})
        values[stem] = {"segment_id": stem, "song_id": expansion.song_id(stem),
            "language": stem.split("/")[0], "singer": stem.split("/")[1], "sample_rate": 48000,
            "num_samples": 128, "units": [], "validation_issues": [], "review_status": "pending",
            "source_recording_offset_samples": None, "sources": {"wav": {"sha256": record["sha256"]}}}
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    plan = {"files": files, "samples": samples, "license": {},
            "split_overrides": {"song_assignments": expansion.SONG_SPLITS}}
    monkeypatch.setattr(expansion, "load_plan", lambda _: plan)
    reader = lambda root, stem, **kwargs: SimpleNamespace(to_dict=lambda: copy.deepcopy(values[stem]))
    output = tmp_path / "prepared"
    expansion.prepare_plan(manifest, source, output, segment_loader=reader)
    validated = load_inputs(output / "segments.jsonl", output / "splits.json", source)
    assert len(validated) == 3
    assert {segment["split"] for segment in validated} == {"train", "val", "test"}
    assert all(segment["sample_rate"] == 48000 and segment["num_samples"] == 128 for segment in validated)
    assert all(segment["source_recording_offset_samples"] is None for segment in validated)

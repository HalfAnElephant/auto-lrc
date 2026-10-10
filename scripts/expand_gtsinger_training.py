#!/usr/bin/env python3
"""Plan, fetch, or prepare a bounded, revision-pinned bilingual training subset."""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unicodedata
from urllib.parse import quote
from urllib.request import Request, build_opener

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.fetch_gtsinger_samples import (MAX_FILE_BYTES, REPRESENTATIVE_BASES,
    REPRESENTATIVE_STEMS, REPOSITORY, REVISION, RestrictedRedirects, load_manifest as load_base_manifest,
    safe_destination, validate_url, verify_file)
from scripts.prepare_gtsinger import publish_directory

SCHEMA = "gtsinger-training-expansion-v1"
MAX_TOTAL_BYTES = 500_000_000
MAX_SEGMENTS = 250
MAX_METADATA_BYTES = 1_048_576
MAX_METADATA_REQUESTS = 64
HELD_OUT_SINGER = "EN-Tenor-1"
TECHNIQUES = ("Breathy", "Glissando", "Mixed_Voice_and_Falsetto", "Pharyngeal", "Vibrato")
BASE_SPLITS = {"chinese/不再见": "train", "chinese/慢慢喜欢你": "val",
               "chinese/给电影人的情书": "train", "english/my love": "train",
               "english/rolling in the deep": "test", "english/someone like you": "test"}
BASE_EXCLUDED = {"Chinese/ZH-Tenor-1/Breathy/慢慢喜欢你/Control_Group/0000",
                 "Chinese/ZH-Tenor-1/Glissando/给电影人的情书/Control_Group/0000"}
BASE_ASSIGNED_STEMS = set(REPRESENTATIVE_STEMS) - BASE_EXCLUDED
ADDITIONS = (
    ("Chinese/ZH-Alto-1/Breathy/匆匆那年", "Breathy_Group", "train"),
    ("Chinese/ZH-Alto-1/Breathy/奇妙能力歌", "Breathy_Group", "train"),
    ("Chinese/ZH-Alto-1/Breathy/成都", "Breathy_Group", "train"),
    ("Chinese/ZH-Alto-1/Breathy/我好想你", "Breathy_Group", "train"),
    ("English/EN-Alto-1/Glissando/god is a girl", "Glissando_Group", "train"),
    ("English/EN-Alto-1/Glissando/make it shine", "Glissando_Group", "train"),
    ("English/EN-Alto-1/Glissando/tell me that you love me", "Glissando_Group", "train"),
    ("Chinese/ZH-Alto-1/Breathy/云烟成雨", "Breathy_Group", "val"),
    ("English/EN-Alto-1/Glissando/for the first time in forever", "Glissando_Group", "val"),
    ("English/EN-Alto-1/Glissando/trouble is a friend", "Glissando_Group", "val"),
    ("Chinese/ZH-Alto-1/Breathy/可乐", "Breathy_Group", "test"),
    ("Chinese/ZH-Alto-1/Breathy/失落沙洲", "Breathy_Group", "test"),
)
CATALOG = tuple(REPRESENTATIVE_BASES) + tuple((base, group) for base, group, _ in ADDITIONS)
GROUP_DIRECTORIES = {base + "/" + group for base, variant in CATALOG
                     for group in ("Control_Group", variant)}
SONG_SPLITS = dict(BASE_SPLITS)


def song_id(base):
    parts = base.split("/")
    if len(parts) < 4 or parts[0] not in ("Chinese", "English"):
        raise ValueError("Only the explicit Chinese/English catalog is allowed")
    return "/".join(" ".join(unicodedata.normalize("NFKC", value).casefold().split())
                    for value in (parts[0], parts[3]))


SONG_SPLITS.update({song_id(base): split for base, _, split in ADDITIONS})


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            value.update(chunk)
    return value.hexdigest()


def resolve_url(path):
    safe_destination(Path("/"), path)
    return "https://huggingface.co/datasets/%s/resolve/%s/%s" % (
        REPOSITORY, REVISION, quote(path, safe="/"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


class MetadataClient:
    """Only direct listings of the fixed group paths and held-out title catalogs."""
    def __init__(self, opener=None):
        self.opener = opener or build_opener(RestrictedRedirects())
        self.evidence = []

    def list_directory(self, directory):
        heldout = {"English/" + HELD_OUT_SINGER + "/" + technique for technique in TECHNIQUES}
        if directory not in GROUP_DIRECTORIES | heldout:
            raise ValueError("Metadata path is outside the fixed expansion catalog")
        if len(self.evidence) >= MAX_METADATA_REQUESTS:
            raise ValueError("Metadata request budget exceeded")
        url = "https://huggingface.co/api/datasets/%s/tree/%s/%s?recursive=false&expand=false" % (
            REPOSITORY, REVISION, quote(directory, safe="/"))
        validate_url(url)
        with self.opener.open(Request(url, headers={"User-Agent": "auto-lrc-training-plan/1"}),
                              timeout=30) as response:
            validate_url(response.geturl())
            if response.headers.get("Link"):
                raise ValueError("Paginated metadata is outside the bounded direct-listing plan")
            body = response.read(MAX_METADATA_BYTES + 1)
        if len(body) > MAX_METADATA_BYTES:
            raise ValueError("Metadata response byte budget exceeded")
        entries = json.loads(body)
        if not isinstance(entries, list) or len(entries) > 1000:
            raise ValueError("Unexpected metadata listing shape or entry count")
        for entry in entries:
            if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                    or entry["path"].rsplit("/", 1)[0] != directory):
                raise ValueError("Metadata listing includes a non-child path")
            safe_destination(Path("/"), entry["path"])
        self.evidence.append({"directory": directory, "url": url, "bytes": len(body),
                              "response_sha256": hashlib.sha256(body).hexdigest()})
        return entries


def _file_record(entry):
    size = entry.get("size")
    if type(size) is not int or not 0 < size <= MAX_FILE_BYTES:
        raise ValueError("File exceeds per-file size budget")
    record = {"path": entry["path"], "size": size, "source_url": resolve_url(entry["path"])}
    if "lfs" in entry:
        lfs = entry["lfs"]
        if not isinstance(lfs, dict) or lfs.get("size") != size:
            raise ValueError("Inconsistent LFS size")
        record["sha256"] = lfs.get("oid")
    else:
        record["git_blob_sha1"] = entry.get("oid")
    _validate_file_record(record)
    return record


def _validate_file_record(entry):
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError("File entries must have a path")
    safe_destination(Path("/"), entry["path"])
    if type(entry.get("size")) is not int or not 0 < entry["size"] <= MAX_FILE_BYTES:
        raise ValueError("File size is outside the byte budget")
    keys = {"sha256", "git_blob_sha1"} & entry.keys()
    if len(keys) != 1:
        raise ValueError("Each file requires exactly one source checksum")
    key = keys.pop()
    if not isinstance(entry[key], str) or not re.fullmatch("[0-9a-f]{%d}" % (64 if key == "sha256" else 40), entry[key]):
        raise ValueError("Invalid source checksum")
    if entry.get("source_url") != resolve_url(entry["path"]):
        raise ValueError("Source URL does not match the pinned file")


def _quartets(directory, entries):
    records, rejected = defaultdict(list), []
    for entry in entries:
        match = re.fullmatch(re.escape(directory) + r"/(\d{4})(\.wav|\.json|\.musicxml|\.TextGrid|_TextGrid)", entry["path"])
        if entry.get("type") == "file" and match:
            records[directory + "/" + match.group(1)].append(entry)
    result = {}
    for stem, files in sorted(records.items()):
        paths = {entry["path"] for entry in files}
        required = {stem + suffix for suffix in (".wav", ".json", ".musicxml")}
        grids = paths & {stem + ".TextGrid", stem + "_TextGrid"}
        if len(files) != 4 or len(grids) != 1 or paths != required | grids:
            rejected.append({"relative_stem": stem, "reason": "incomplete or ambiguous source quartet"})
            continue
        try:
            result[stem] = sorted((_file_record(entry) for entry in files), key=lambda value: value["path"])
        except ValueError as error:
            rejected.append({"relative_stem": stem, "reason": str(error)})
    return result, rejected


def _baseline(manifest_path, splits_path):
    manifest = load_base_manifest(manifest_path)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    if (manifest.get("selection") != "representative-v1" or splits.get("identity_mode") != "dataset"
            or splits.get("song_identity_reviewed") is not False
            or splits.get("held_out_singers") != [HELD_OUT_SINGER]):
        raise ValueError("Expected the frozen representative dataset-identity singer-holdout split")
    samples = {sample["relative_stem"]: sample for sample in manifest["samples"]}
    assignments = splits.get("assignments", [])
    seen, songs, wav_splits = set(), set(), defaultdict(set)
    files = {entry["path"]: entry for entry in manifest["files"]}
    for row in assignments:
        stem = row.get("segment_id")
        if stem not in samples or stem in seen:
            raise ValueError("Baseline assignments must uniquely refer to the source selection")
        expected = song_id(stem)
        if (row.get("canonical_song_id") != expected or samples[stem]["song_id"] != expected
                or row.get("split") != BASE_SPLITS.get(expected)
                or (stem.split("/")[1] == HELD_OUT_SINGER and row["split"] != "test")):
            raise ValueError("Baseline song assignment or singer holdout changed")
        seen.add(stem)
        songs.add(expected)
        wav = files[stem + ".wav"]
        wav_splits[wav.get("sha256", wav.get("git_blob_sha1"))].add(row["split"])
    if seen != BASE_ASSIGNED_STEMS or songs != set(BASE_SPLITS) or any(len(value) != 1 for value in wav_splits.values()):
        raise ValueError("Baseline must preserve six songs without WAV leakage")
    return manifest, splits


def build_plan(base_manifest, base_splits, *, segments_per_song=10, client=None):
    if type(segments_per_song) is not int or not 1 <= segments_per_song <= 20:
        raise ValueError("segments_per_song must be an integer from 1 to 20")
    base, frozen = _baseline(base_manifest, base_splits)
    client = client or MetadataClient()
    heldout_songs = set()
    for technique in TECHNIQUES:
        for entry in client.list_directory("English/" + HELD_OUT_SINGER + "/" + technique):
            if entry.get("type") != "directory":
                raise ValueError("Held-out catalog must contain song directories only")
            heldout_songs.add(song_id(entry["path"]))
    conflicts = sorted(song for song, split in SONG_SPLITS.items() if split != "test" and song in heldout_songs)
    if conflicts:
        raise ValueError("New song conflicts with held-out singer: " + ", ".join(conflicts))
    available, rejected = {}, []
    for directory in sorted(GROUP_DIRECTORIES):
        quartets, failures = _quartets(directory, client.list_directory(directory))
        available.update(quartets)
        rejected.extend(failures)
    selected = {sample["relative_stem"] for sample in base["samples"]}
    if selected - available.keys():
        raise ValueError("Existing source quartet is absent or changed")
    prior_files = {entry["path"]: entry for entry in base["files"]}
    for stem in selected:
        for entry in available[stem]:
            prior = prior_files[entry["path"]]
            if any(prior.get(key) != entry.get(key) for key in ("size", "sha256", "git_blob_sha1")):
                raise ValueError("Existing source checksum differs from the frozen manifest")
    for song in sorted(SONG_SPLITS):
        already = {stem for stem in selected if song_id(stem) == song}
        if len(already) > segments_per_song:
            raise ValueError("Per-song budget cannot preserve all existing source segments")
        directories = sorted({stem.rsplit("/", 1)[0] for stem in available if song_id(stem) == song})
        queues = [deque(sorted(stem for stem in available if stem.rsplit("/", 1)[0] == directory and stem not in selected))
                  for directory in directories]
        while len(already) < segments_per_song and any(queues):
            for queue in queues:
                if queue and len(already) < segments_per_song:
                    stem = queue.popleft()
                    selected.add(stem)
                    already.add(stem)
        if not already:
            raise ValueError("No complete source quartets for required song: " + song)
    license_record = dict(prior_files["dataset_license.md"], source_url=resolve_url("dataset_license.md"))
    samples = [{"relative_stem": stem, "song_id": song_id(stem), "split": SONG_SPLITS[song_id(stem)],
                "song_identity_review_status": "pending", "song_identity_source": "dataset_song_id"}
               for stem in sorted(selected)]
    files = [license_record] + [entry for stem in sorted(selected) for entry in available[stem]]
    source_overrides = {"identity_mode": "dataset", "song_identity_reviewed": False,
        "aliases_confirmed": False, "held_out_singers": [HELD_OUT_SINGER],
        "song_assignments": dict(SONG_SPLITS), "frozen_assignments": frozen["assignments"],
        "assignments": [{"segment_id": sample["relative_stem"], "canonical_song_id": sample["song_id"],
                         "split": sample["split"]} for sample in samples]}
    plan = {"schema_version": SCHEMA, "repository": REPOSITORY, "revision": REVISION,
        "selection": "training-expansion-v1", "languages": ["Chinese", "English"],
        "license": dict(base["license"]), "samples": samples, "files": files,
        "declared_total_bytes": sum(entry["size"] for entry in files),
        "budgets": {"max_total_bytes": MAX_TOTAL_BYTES, "max_segments": MAX_SEGMENTS,
                    "max_file_bytes": MAX_FILE_BYTES, "segments_per_song": segments_per_song},
        "baseline": {"manifest": {"path": str(base_manifest), "sha256": digest(base_manifest)},
                     "splits": {"path": str(base_splits), "sha256": digest(base_splits)},
                     "existing_segment_ids": sorted(sample["relative_stem"] for sample in base["samples"])},
        "split_overrides": source_overrides, "metadata_evidence": client.evidence,
        "excluded_metadata_candidates": rejected,
        "held_out_catalog_song_ids": sorted(heldout_songs),
        "source_terms": {"license_path": "dataset_license.md", "source_url": resolve_url("dataset_license.md"),
                         "identifier": base["license"]["identifier"], "additional_terms": base["license"]["additional_terms"]},
        "limitations": ["Song IDs and aliases are dataset-derived and unreviewed.",
                        "Boundaries remain dataset annotations, not independent human truth.",
                        "Original whole-song sample offsets remain unknown.",
                        "Only the explicit bilingual song/group catalog is selected; no recursive dataset download."]}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if (not isinstance(plan, dict) or plan.get("schema_version") != SCHEMA
            or plan.get("repository") != REPOSITORY or plan.get("revision") != REVISION
            or plan.get("languages") != ["Chinese", "English"]):
        raise ValueError("Unsupported expansion manifest")
    samples, files = plan.get("samples"), plan.get("files")
    if not isinstance(samples, list) or not 0 < len(samples) <= MAX_SEGMENTS or not isinstance(files, list):
        raise ValueError("Invalid sample list or segment budget")
    budgets = plan.get("budgets", {})
    per_song_limit = budgets.get("segments_per_song")
    if (budgets.get("max_total_bytes") != MAX_TOTAL_BYTES or budgets.get("max_segments") != MAX_SEGMENTS
            or budgets.get("max_file_bytes") != MAX_FILE_BYTES or type(per_song_limit) is not int
            or not 1 <= per_song_limit <= 20):
        raise ValueError("Expansion budgets may not be enlarged")
    per_song, stems, expected_paths, hashes = Counter(), set(), {"dataset_license.md"}, defaultdict(set)
    wav_songs = defaultdict(set)
    paths = [entry.get("path") for entry in files if isinstance(entry, dict)]
    if len(paths) != len(files) or len(set(paths)) != len(paths):
        raise ValueError("Manifest file paths must be unique")
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("relative_stem"), str):
            raise ValueError("Malformed sample")
        stem = sample["relative_stem"]
        directory, _, index = stem.rpartition("/")
        if directory not in GROUP_DIRECTORIES or not re.fullmatch(r"\d{4}", index) or stem in stems:
            raise ValueError("Sample is outside the fixed catalog or duplicated")
        song = song_id(stem)
        if (sample.get("song_id") != song or sample.get("split") != SONG_SPLITS[song]
                or sample.get("song_identity_review_status") != "pending"):
            raise ValueError("Song split or unreviewed identity changed")
        stems.add(stem)
        per_song[song] += 1
        expected_paths.update(stem + extension for extension in (".wav", ".json", ".musicxml"))
        grids = set(paths) & {stem + ".TextGrid", stem + "_TextGrid"}
        if len(grids) != 1:
            raise ValueError("Every sample requires exactly one TextGrid")
        expected_paths.update(grids)
    if set(per_song) != set(SONG_SPLITS) or max(per_song.values()) > per_song_limit or set(paths) != expected_paths:
        raise ValueError("Song coverage, per-song budget or source quartet mismatch")
    total = 0
    for entry in files:
        _validate_file_record(entry)
        total += entry["size"]
        if entry["path"].endswith(".wav"):
            stem = entry["path"][:-4]
            if "sha256" not in entry:
                raise ValueError("WAV requires a content SHA-256 for split leakage validation")
            hashes[entry["sha256"]].add(SONG_SPLITS[song_id(stem)])
            wav_songs[entry["sha256"]].add(song_id(stem))
    if total > MAX_TOTAL_BYTES or plan.get("declared_total_bytes") != total:
        raise ValueError("Total byte budget exceeded or inconsistent")
    if any(len(splits) > 1 for splits in hashes.values()):
        raise ValueError("Same source WAV crosses splits")
    if any(len(songs) > 1 for songs in wav_songs.values()):
        raise ValueError("Same source WAV has conflicting dataset song identities")
    license_terms, source_terms = plan.get("license", {}), plan.get("source_terms", {})
    if (not isinstance(license_terms, dict) or not isinstance(source_terms, dict)
            or license_terms.get("path") != "dataset_license.md"
            or source_terms.get("license_path") != "dataset_license.md"
            or license_terms.get("identifier") != "CC-BY-NC-SA-4.0"
            or source_terms.get("identifier") != license_terms["identifier"]
            or source_terms.get("source_url") != resolve_url("dataset_license.md")
            or not isinstance(license_terms.get("additional_terms"), str)
            or not license_terms["additional_terms"].strip()
            or source_terms.get("additional_terms") != license_terms["additional_terms"]):
        raise ValueError("Source license and terms must remain bound to the manifest")
    overrides = plan.get("split_overrides", {})
    if (overrides.get("identity_mode") != "dataset" or overrides.get("song_identity_reviewed") is not False
            or overrides.get("aliases_confirmed") is not False
            or overrides.get("held_out_singers") != [HELD_OUT_SINGER]
            or overrides.get("song_assignments") != SONG_SPLITS):
        raise ValueError("Split overrides changed the dataset-identity holdout contract")
    by_stem = {sample["relative_stem"]: sample for sample in samples}
    rows = overrides.get("assignments", [])
    if (not isinstance(rows, list) or len(rows) != len(stems)
            or {row.get("segment_id") for row in rows if isinstance(row, dict)} != stems):
        raise ValueError("Split assignments must cover every planned segment exactly once")
    frozen = overrides.get("frozen_assignments")
    if (not isinstance(frozen, list) or len(frozen) != len(BASE_ASSIGNED_STEMS)
            or {row.get("segment_id") for row in frozen if isinstance(row, dict)} != BASE_ASSIGNED_STEMS
            or set(plan.get("baseline", {}).get("existing_segment_ids", [])) != set(REPRESENTATIVE_STEMS)
            or not set(REPRESENTATIVE_STEMS) <= stems):
        raise ValueError("Existing source samples and frozen assignments must all be retained")
    for row in rows + frozen:
        sample = by_stem.get(row.get("segment_id"))
        if (sample is None or row.get("canonical_song_id") != sample["song_id"]
                or row.get("split") != sample["split"]):
            raise ValueError("Frozen or incremental assignment changed")
        if sample["relative_stem"].split("/")[1] == HELD_OUT_SINGER and row["split"] != "test":
            raise ValueError("Held-out singer must remain test")
    heldout_catalog = plan.get("held_out_catalog_song_ids")
    if (not isinstance(heldout_catalog, list) or not heldout_catalog
            or any(not isinstance(value, str) or not value.startswith("english/") for value in heldout_catalog)
            or len(set(heldout_catalog)) != len(heldout_catalog)
            or not {"english/rolling in the deep", "english/someone like you"} <= set(heldout_catalog)):
        raise ValueError("Held-out singer title catalog must be retained")
    conflicts = set(heldout_catalog) & {song for song, split in SONG_SPLITS.items() if split != "test"}
    if conflicts:
        raise ValueError("Song shares held-out singer across splits")
    return plan


def load_plan(path):
    return validate_plan(json.loads(Path(path).read_text(encoding="utf-8")))


def publish_plan(plan, out_dir):
    validate_plan(plan)
    out_dir = Path(out_dir).absolute()
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError("Plan directory already exists")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gtsinger-plan-", dir=out_dir.parent))
    try:
        _write_json(staging / "manifest.json", plan)
        _write_json(staging / "split-overrides.json", plan["split_overrides"])
        summary = {"song_counts": dict(Counter(SONG_SPLITS.values())),
                   "split_language_song_counts": {split: dict(Counter(song.split("/", 1)[0] for song, assigned in SONG_SPLITS.items() if assigned == split))
                                                  for split in ("train", "val", "test")},
                   "segment_count": len(plan["samples"]), "declared_total_bytes": plan["declared_total_bytes"],
                   "metadata_request_count": len(plan["metadata_evidence"]), "downloaded_audio_bytes": 0}
        _write_json(staging / "plan-summary.json", summary)
        publish_directory(staging, out_dir)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _download(root, entry, opener):
    target = safe_destination(root, entry["path"])
    if target.exists():
        return {"path": entry["path"], "sha256": verify_file(target, entry), "status": "reused"}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", delete=False) as stream:
            temporary = Path(stream.name)
            with opener.open(Request(entry["source_url"], headers={"Accept-Encoding": "identity",
                              "User-Agent": "auto-lrc-training-fetch/1"}), timeout=30) as response:
                validate_url(response.geturl())
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != entry["size"]:
                    raise ValueError("Remote content size changed")
                count = 0
                while True:
                    chunk = response.read(min(65536, entry["size"] + 1 - count))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > entry["size"]:
                        raise ValueError("Download exceeded the declared size")
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        content_hash = verify_file(temporary, entry)
        safe_destination(root, entry["path"])
        os.link(temporary, target)
        return {"path": entry["path"], "sha256": content_hash, "status": "downloaded"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_plan(manifest_path, input_root, *, reuse_root=None, opener=None):
    plan = load_plan(manifest_path)
    root = Path(input_root).absolute()
    safe_destination(root, "dataset_license.md")
    root.mkdir(parents=True, exist_ok=True)
    opener = opener or build_opener(RestrictedRedirects())
    records = []
    for entry in plan["files"]:
        target = safe_destination(root, entry["path"])
        source = safe_destination(reuse_root, entry["path"]) if reuse_root is not None else None
        if not target.exists() and source is not None and source.exists():
            content_hash = verify_file(source, entry)
            target.parent.mkdir(parents=True, exist_ok=True)
            # An exclusive destination keeps originals intact, including concurrent publication.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".reuse-", delete=False) as stream:
                    temporary = Path(stream.name)
                    with source.open("rb") as original:
                        shutil.copyfileobj(original, stream, length=65536)
                    stream.flush()
                    os.fsync(stream.fileno())
                verify_file(temporary, entry)
                safe_destination(root, entry["path"])
                os.link(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            records.append({"path": entry["path"], "sha256": content_hash, "status": "copied_existing"})
        else:
            records.append(_download(root, entry, opener))
    return {"repository": REPOSITORY, "revision": REVISION, "manifest_sha256": digest(manifest_path),
            "license": plan["license"], "total_bytes": plan["declared_total_bytes"], "files": records}


def _validated_splits(segments, plan):
    from t2l.datasets.splits import assign_song_splits
    # Reuse existing relation/identity validation, then apply the frozen explicit song overrides.
    result = assign_song_splits(segments, identity_mode="dataset", held_out_singers=[HELD_OUT_SINGER])
    result["assignments"] = [{"segment_id": segment["segment_id"], "canonical_song_id": segment["song_id"],
                               "split": plan["split_overrides"]["song_assignments"][segment["song_id"]]}
                              for segment in sorted(segments, key=lambda value: value["segment_id"])]
    result["assignment_method"] = "frozen_existing_plus_explicit_incremental_song_overrides"
    result["requested_ratios"] = None
    by_id = {segment["segment_id"]: segment for segment in segments}
    hashes = defaultdict(set)
    for row in result["assignments"]:
        segment = by_id[row["segment_id"]]
        if segment.get("singer") == HELD_OUT_SINGER and row["split"] != "test":
            raise ValueError("Held-out singer crossed into train or validation")
        hashes[segment["sources"]["wav"]["sha256"]].add(row["split"])
    if any(len(splits) != 1 for splits in hashes.values()):
        raise ValueError("Identical WAV crosses explicit splits")
    result["song_counts"] = dict(Counter(plan["split_overrides"]["song_assignments"][song] for song in {segment["song_id"] for segment in segments}))
    result["segment_counts"] = dict(Counter(row["split"] for row in result["assignments"]))
    result["frozen_plan_manifest_schema"] = SCHEMA
    return result


def prepare_plan(manifest_path, input_root, out_dir, *, on_invalid="exclude", segment_loader=None):
    if on_invalid not in ("fail", "exclude"):
        raise ValueError("on_invalid must be fail or exclude")
    plan = load_plan(manifest_path)
    out_dir = Path(out_dir).absolute()
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError("Prepared result directory already exists")
    source_files = [dict(entry, content_sha256=verify_file(safe_destination(input_root, entry["path"]), entry))
                    for entry in plan["files"]]
    if segment_loader is None:
        from t2l.datasets.gtsinger import load_segment
        segment_loader = load_segment
    segments, excluded = [], []
    for sample in plan["samples"]:
        try:
            segment = segment_loader(Path(input_root), sample["relative_stem"], dataset_revision=REVISION,
                                     song_id=sample["song_id"]).to_dict()
            if segment["segment_id"] != sample["relative_stem"] or segment["song_id"] != sample["song_id"]:
                raise ValueError("Reader changed planned segment or song identity")
        except ValueError as error:
            if on_invalid == "fail":
                raise
            excluded.append({"segment_id": sample["relative_stem"], "song_id": sample["song_id"],
                             "exception_type": "ValueError", "reason": str(error)})
            continue
        segment.update(split=sample["split"], song_identity_review_status="pending",
                       song_identity_source="dataset_song_id")
        segments.append(segment)
    if not segments:
        raise ValueError("No usable segments remain")
    splits = _validated_splits(segments, plan)
    summary = {"schema_version": 1, "repository": REPOSITORY, "dataset_revision": REVISION,
               "license": plan["license"], "source_files": source_files, "source_manifest_sha256": digest(manifest_path),
               "requested_segment_count": len(plan["samples"]), "usable_segment_count": len(segments),
               "excluded_segment_count": len(excluded), "invalid_segment_policy": on_invalid,
               "song_counts": splits["song_counts"], "segment_counts": splits["segment_counts"],
               "language_counts": dict(Counter(segment["language"] for segment in segments)),
               "split_language_song_counts": {split: dict(Counter(song.split("/", 1)[0]
                    for song in {segment["song_id"] for segment in segments if segment["split"] == split}))
                    for split in ("train", "val", "test")},
               "missing_planned_song_ids": sorted(set(SONG_SPLITS) - {segment["song_id"] for segment in segments}),
               "unit_kind_counts": dict(Counter(unit["kind"] for segment in segments for unit in segment["units"])),
               "validation_issue_counts": dict(Counter(issue["code"] for segment in segments
                                                       for issue in segment.get("validation_issues", []))),
               "review_status": "pending", "song_identity_reviewed": False, "boundary_annotations_reviewed": False,
               "accuracy": {"status": "unavailable", "reason": "No independent human boundary labels."},
               "annotation_semantics": "Dataset annotations are supervision, not independent human truth.",
               "source_recording_offset_status": "unknown", "excluded_segments": excluded}
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gtsinger-training-prepare-", dir=out_dir.parent))
    try:
        (staging / "segments.jsonl").write_text("".join(json.dumps(segment, ensure_ascii=False, allow_nan=False) + "\n" for segment in segments), encoding="utf-8")
        for name, value in (("splits.json", splits), ("quality-summary.json", summary),
                            ("excluded-segments.json", excluded), ("source-manifest.json", plan)):
            _write_json(staging / name, value)
        publish_directory(staging, out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Fetch bounded metadata only; never download audio")
    plan.add_argument("--base-manifest", type=Path, default=REPO_ROOT / "data/gtsinger-representative.json")
    plan.add_argument("--base-splits", type=Path, default=REPO_ROOT / "reports/p1/source-mode-20261010/splits.json")
    plan.add_argument("--segments-per-song", type=int, default=10)
    plan.add_argument("--out-dir", type=Path, required=True)
    for name in ("fetch", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--input-root", type=Path, required=True)
        if name == "fetch":
            command.add_argument("--reuse-root", type=Path)
        else:
            command.add_argument("--out-dir", type=Path, required=True)
            command.add_argument("--on-invalid", choices=("fail", "exclude"), default="exclude")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            if args.out_dir.exists() or args.out_dir.is_symlink():
                raise FileExistsError("Plan directory already exists")
            result = publish_plan(build_plan(args.base_manifest, args.base_splits,
                                            segments_per_song=args.segments_per_song), args.out_dir)
        elif args.command == "fetch":
            result = fetch_plan(args.manifest, args.input_root, reuse_root=args.reuse_root)
        else:
            result = prepare_plan(args.manifest, args.input_root, args.out_dir, on_invalid=args.on_invalid)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, "GTSinger expansion failed: %s\n" % error)


if __name__ == "__main__":
    raise SystemExit(cli())

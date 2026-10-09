#!/usr/bin/env python3
"""Inspect only the fixed 20-segment selection and publish its source manifest."""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unicodedata
from urllib.parse import quote
from urllib.request import Request, build_opener

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.fetch_gtsinger_samples import (DEFAULT_MANIFEST, REPRESENTATIVE_STEMS,
    REPOSITORY, REVISION, RestrictedRedirects, load_manifest, source_url, validate_url)


def read_json_url(url, opener, max_bytes=1024 * 1024):
    validate_url(url)
    with opener.open(Request(url, headers={"User-Agent": "auto-lrc-p1-metadata/1"}), timeout=30) as response:
        validate_url(response.geturl())
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError("Fixed metadata response exceeds byte limit")
    return json.loads(body), hashlib.sha256(body).hexdigest()


def build_manifest():
    opener = build_opener(RestrictedRedirects())
    smoke = load_manifest(DEFAULT_MANIFEST)
    license_path = smoke["license"]["path"]
    license_files = [entry for entry in smoke["files"] if entry["path"] == license_path]
    if len(license_files) != 1:
        raise ValueError("Expected exactly one source license file: " + license_path)
    manifest = {key: value for key, value in smoke.items() if key not in ("samples", "files", "purpose")}
    manifest.update(selection="representative-v1", metadata_verified_on=date.today().isoformat(),
                    purpose="Fixed 20-segment bilingual technique and leakage inspection subset; song identities and human boundaries remain unreviewed.",
                    samples=[], files=[dict(license_files[0])], metadata_evidence=[])
    for stem in REPRESENTATIVE_STEMS:
        directory = stem.rsplit("/", 1)[0]
        api_url = "https://huggingface.co/api/datasets/%s/tree/%s/%s?recursive=false&expand=false" % (
            REPOSITORY, REVISION, quote(directory, safe="/"))
        entries, response_hash = read_json_url(api_url, opener)
        matches = [entry for entry in entries if entry.get("type") == "file" and entry["path"] in {
            stem + suffix for suffix in (".wav", ".json", ".musicxml", ".TextGrid", "_TextGrid")}]
        if len(matches) != 4:
            raise ValueError("Expected one source quartet: " + stem)
        files = []
        for entry in matches:
            record = {"path": entry["path"], "size": entry["size"]}
            if "lfs" in entry:
                record["sha256"] = entry["lfs"]["oid"]
            else:
                record["git_blob_sha1"] = entry["oid"]
            files.append(record)
        words_entry = next(entry for entry in files if entry["path"].endswith(".json"))
        words, words_hash = read_json_url(source_url(words_entry), opener)
        if not isinstance(words, list) or not words:
            raise ValueError("Empty or invalid published words: " + stem)
        lyrics = [word["word"] for word in words if word["word"] not in ("<SP>", "<AP>", "")]
        flags = {key: sum(str(flag) == "1" for word in words for flag in word.get(key, []))
                 for key in ("breathy", "glissando")}
        language, _, _, title, _, _ = stem.split("/")
        song_id = "/".join(" ".join(unicodedata.normalize("NFKC", value).casefold().split())
                           for value in (language, title))
        manifest["samples"].append({"relative_stem": stem, "song_id": song_id,
                                    "song_identity_review_status": "pending", "split": "unassigned"})
        manifest["files"].extend(files)
        manifest["metadata_evidence"].append({"relative_stem": stem, "api_url": api_url,
            "api_response_sha256": response_hash, "json_content_sha256": words_hash,
            "published_end_seconds": words[-1]["end_time"], "lyric_unit_count": len(lyrics),
            "lyric_preview": lyrics[:8], "published_phone_technique_flags": flags})
    manifest["declared_total_bytes"] = sum(entry["size"] for entry in manifest["files"])
    manifest["coverage"] = {
        "segment_count": len(manifest["samples"]),
        "candidate_song_count": len({sample["song_id"] for sample in manifest["samples"]}),
        "singer_counts": dict(Counter(stem.split("/")[1] for stem in REPRESENTATIVE_STEMS)),
        "group_counts": dict(Counter(stem.split("/")[4] for stem in REPRESENTATIVE_STEMS)),
        "song_identity_review_status": "pending",
    }
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            parser.error("Manifest output already exists")
        manifest = build_manifest()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                             prefix=".gtsinger-manifest-", delete=False) as target:
                temporary = Path(target.name)
                json.dump(manifest, target, ensure_ascii=False, indent=2)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            load_manifest(temporary)
            os.link(temporary, args.output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(json.dumps(manifest["coverage"], ensure_ascii=False, indent=2))
        print("Declared source bytes:", manifest["declared_total_bytes"])
    except (OSError, ValueError) as error:
        parser.exit(1, "GTSinger manifest construction failed: %s\n" % error)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Normalize a fixed GTSinger selection and create pending blind review forms."""
import argparse
from collections import Counter
import ctypes
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.fetch_gtsinger_samples import DEFAULT_MANIFEST, load_manifest, safe_destination, source_url, verify_file


REVIEW_FIELDS = ["segment_id", "song_id", "language", "unit_index", "text", "sample_rate",
                 "audio_path", "human_truth_seconds", "human_truth_sample", "confidence", "review_status",
                 "ambiguous", "reviewer", "reviewed_at", "review_notes"]


def publish_directory(staging, out_dir):
    """Atomically rename without replacing even an empty concurrent directory."""
    if sys.platform == "win32":
        # Windows rename raises FileExistsError for every existing destination.
        os.rename(staging, out_dir)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function_name = "renamex_np"
        argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(staging), os.fsencode(out_dir), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        function_name = "renameat2"
        argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = (-100, os.fsencode(staging), -100, os.fsencode(out_dir), 1)  # RENAME_NOREPLACE
    else:
        raise OSError("Atomic no-clobber directory publication requires macOS, Linux, or Windows")
    rename = getattr(libc, function_name, None)
    if rename is None:
        raise OSError("Atomic no-clobber directory publication unavailable: libc has no " + function_name)
    rename.argtypes = argtypes
    rename.restype = ctypes.c_int
    if rename(*args) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(out_dir))


def prepare_samples(input_root, manifest_path, out_dir, *, segment_loader=None, on_invalid="fail"):
    if on_invalid not in ("fail", "exclude"):
        raise ValueError("on_invalid must be fail or exclude")
    out_dir = Path(out_dir).absolute()
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError("Result directory already exists: " + str(out_dir))
    manifest = load_manifest(manifest_path)
    source_files = []
    for entry in manifest["files"]:
        source = safe_destination(input_root, entry["path"])
        source_files.append(dict(entry, content_sha256=verify_file(source, entry), source_url=source_url(entry)))
    if segment_loader is None:
        from t2l.datasets.gtsinger import load_segment
        segment_loader = load_segment
    segments, reviews, candidates, issues, excluded = [], [], [], [], []
    for sample in manifest["samples"]:
        try:
            loaded = segment_loader(Path(input_root), sample["relative_stem"],
                                    dataset_revision=manifest["revision"], song_id=sample["song_id"])
        except ValueError as error:
            if on_invalid == "fail":
                raise
            excluded.append({"segment_id": sample["relative_stem"], "song_id": sample["song_id"],
                             "reason": str(error), "exception_type": "ValueError"})
            issues.append({"code": "invalid_segment_excluded", "path": sample["relative_stem"],
                           "message": str(error), "segment_id": sample["relative_stem"]})
            continue
        segment = loaded.to_dict()
        segment["split"] = "unassigned"
        segment["song_identity_review_status"] = sample.get("song_identity_review_status", "pending")
        segments.append(segment)
        issues.extend(dict(issue, segment_id=segment["segment_id"]) for issue in segment["validation_issues"])
        for index, unit in enumerate(segment["units"]):
            if unit["kind"] != "lyric":
                continue
            boundary = unit.get("boundary") or {}
            review = dict.fromkeys(REVIEW_FIELDS)
            review.update(segment_id=segment["segment_id"], song_id=segment["song_id"],
                          language=segment["language"], unit_index=index, text=unit["text"],
                          sample_rate=segment["sample_rate"], audio_path=segment["audio_path"],
                          review_status="pending")
            reviews.append(review)
            candidates.append({"segment_id": segment["segment_id"], "unit_index": index,
                               "text": unit["text"], "candidate_source": boundary.get("source"),
                               "candidate_seconds": boundary.get("candidate_seconds"),
                               "candidate_sample": boundary.get("candidate_sample")})
    if not segments:
        raise ValueError("No usable segments: all %d requested segments were rejected" % len(manifest["samples"]))
    summary = {
        "schema_version": 1, "repository": manifest["repository"], "dataset_revision": manifest["revision"],
        "license": manifest["license"], "source_files": source_files,
        "selection": manifest.get("selection", "smoke"),
        "invalid_segment_policy": on_invalid, "requested_segment_count": len(manifest["samples"]),
        "usable_segment_count": len(segments), "excluded_segment_count": len(excluded),
        "excluded_segments": excluded,
        "segment_count": len(segments), "song_count": len({s["song_id"] for s in segments}),
        "song_identity_review_status": "pending",
        "singer_counts": dict(Counter(s["segment_id"].split("/")[1] for s in segments)),
        "technique_counts": dict(Counter(s["segment_id"].split("/")[2] for s in segments)),
        "group_counts": dict(Counter(s["segment_id"].split("/")[4] for s in segments)),
        "language_counts": dict(Counter(s["language"] for s in segments)),
        "unit_kind_counts": dict(Counter(u["kind"] for s in segments for u in s["units"])),
        "validation_issue_count": len(issues), "validation_issue_counts": dict(Counter(i["code"] for i in issues)),
        "validation_issues": issues, "review_boundary_count": len(reviews),
        "independent_reviewed_boundaries": 0, "review_status": "pending",
        "split_status": "unassigned", "split_reason": "Song identities require independent review before any train/validation/test assignment.",
        "accuracy": {"status": "unavailable", "reason": "No independent human boundary labels.",
                     "mae_ms": None, "p90_ms": None, "coverage_5ms": None, "coverage_10ms": None,
                     "coverage_20ms": None, "coverage_50ms": None},
        "boundary_semantics": "Source annotation candidates are not independent human ground truth.",
        "review_protocol": "Blind review: use audio and manual-review files first. Inspect annotation candidates and normalized segments only after independent labels are recorded.",
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gtsinger-prepare-", dir=out_dir.parent))
    try:
        (staging / "segments.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in segments), encoding="utf-8")
        for name, value in [("manual-review.json", reviews), ("annotation-candidates.json", candidates),
                            ("quality-summary.json", summary), ("source-manifest.json", manifest),
                            ("excluded-segments.json", excluded)]:
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "REVIEW.md").write_text(
            "# Independent boundary review\n\n"
            "Use the original audio and manual-review.csv or manual-review.json for the first review. "
            "The forms deliberately omit source candidate times. Record the first audible performed "
            "phoneme onset at the original sample rate, reviewer identity, and confidence; leave "
            "uncertain boundaries ambiguous with a note rather than forcing a single point.\n\n"
            "Do not inspect annotation-candidates.json, segments.jsonl, TextGrid, or JSON source "
            "timings until independent labels have been recorded. Then compare them to the source "
            "candidates. Source annotation agreement is not independent accuracy evidence.\n\n"
            "All rows start pending. These samples are unassigned inspection material; "
            "they do not establish train/validation/test splits or measured accuracy. "
            "Excluded segments and full source validation errors are recorded in "
            "excluded-segments.json and quality-summary.json; source annotations are not repaired.\n", encoding="utf-8")
        with (staging / "manual-review.csv").open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(reviews)
        if out_dir.exists() or out_dir.is_symlink():
            raise FileExistsError("Result directory appeared during preparation")
        publish_directory(staging, out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--on-invalid", choices=("fail", "exclude"), default="fail",
                        help="Fail on invalid annotations (default), or record and exclude rejected segments")
    args = parser.parse_args(argv)
    try:
        result = prepare_samples(args.input_root, args.manifest, args.out_dir, on_invalid=args.on_invalid)
        print(json.dumps({"out_dir": str(args.out_dir), "segment_count": result["segment_count"],
                          "requested_segment_count": result["requested_segment_count"],
                          "usable_segment_count": result["usable_segment_count"],
                          "excluded_segment_count": result["excluded_segment_count"],
                          "validation_issue_count": result["validation_issue_count"],
                          "review_status": result["review_status"], "accuracy": result["accuracy"]}, ensure_ascii=False, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, "GTSinger preparation failed: %s\n" % error)


if __name__ == "__main__":
    main()

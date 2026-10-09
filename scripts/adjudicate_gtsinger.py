#!/usr/bin/env python3
"""Prepare explicit human adjudications or export their original-sample references."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_gtsinger import ReviewStore
from t2l.datasets.adjudication import build_adjudication_form, resolve_adjudication


def load_review_documents(form, review, second_review):
    paths = (Path(review), Path(second_review))
    if any(not path.is_file() for path in paths):
        raise ValueError("Independent blind review sessions must already exist")
    if paths[0].samefile(paths[1]):
        raise ValueError("Two independent blind reviews require distinct session files")
    documents = []
    for path in paths:
        raw = path.read_bytes()
        saved = json.loads(raw)
        store = ReviewStore(form, path)
        if path.read_bytes() != raw:
            raise ValueError("Review session changed while loading; retry with a stable snapshot")
        validated = store.document()
        for field in ("schema_version", "protocol", "onset_definition", "form_fingerprint",
                      "revision", "audio_sources"):
            if saved.get(field) != validated[field]:
                raise ValueError("Review session metadata changed: " + field)
        documents.append(saved)
    return tuple(documents)


def write_new_json(path, document):
    payload = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".adjudication-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--form", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--second-review", required=True, type=Path)
    parser.add_argument("--decisions", type=Path,
                        help="Completed adjudication form; omit to create a pending form")
    parser.add_argument("--agreement-limit-ms", type=float, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.decisions is not None and not args.decisions.is_file():
        parser.error("Adjudication decisions file must already exist")
    first, second = load_review_documents(args.form, args.review, args.second_review)
    if args.decisions is None:
        result = build_adjudication_form(first, second, agreement_limit_ms=args.agreement_limit_ms)
    else:
        decisions = json.loads(args.decisions.read_text(encoding="utf8"))
        result = resolve_adjudication(first, second, decisions,
                                     agreement_limit_ms=args.agreement_limit_ms)
    write_new_json(args.output, result)
    print(json.dumps({"protocol": result["protocol"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

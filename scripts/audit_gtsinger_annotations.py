#!/usr/bin/env python3
"""Compare source annotation candidates to independently submitted blind reviews."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2l.datasets.annotations import compare_source_annotations
from scripts.review_gtsinger import ReviewStore
from scripts.adjudicate_gtsinger import load_review_documents, write_new_json
from t2l.datasets.adjudication import resolve_adjudication


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--form", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--second-review", type=Path)
    parser.add_argument("--adjudication", type=Path,
                        help="Explicit human decisions bound to both blind review snapshots")
    parser.add_argument("--agreement-limit-ms", type=float, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.review.is_file() or (args.second_review and not args.second_review.is_file()):
        parser.error("Independent review sessions must already exist")
    if args.adjudication and not args.second_review:
        parser.error("Adjudication requires two independent blind reviews")
    candidates = json.loads(args.candidates.read_text(encoding="utf8"))
    if args.adjudication:
        first_document, second_document = load_review_documents(args.form, args.review, args.second_review)
        reference = resolve_adjudication(first_document, second_document,
            json.loads(args.adjudication.read_text(encoding="utf8")),
            agreement_limit_ms=args.agreement_limit_ms)
        resolved_report = compare_source_annotations(candidates, reference["rows"],
                                                     agreement_limit_ms=args.agreement_limit_ms)
        resolved_report["review_count"] = 2
        resolved_report["reference_protocol"] = reference["protocol"]
        resolved_report["reference_provenance"] = reference
        blind_comparison = compare_source_annotations(candidates,
            first_document["rows"], second_labels=second_document["rows"],
            agreement_limit_ms=args.agreement_limit_ms)
        resolved_report["blind_review_comparison"] = blind_comparison
        for key, value in blind_comparison.items():
            if key.startswith("inter_reviewer_"):
                resolved_report[key] = value
        report = resolved_report
    else:
        first = ReviewStore(args.form, args.review)
        second = ReviewStore(args.form, args.second_review) if args.second_review else None
        report = compare_source_annotations(candidates, first.rows,
            second_labels=second.rows if second else None, agreement_limit_ms=args.agreement_limit_ms)
    write_new_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "evaluated_count", "total_count", "mae_ms", "p90_ms")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

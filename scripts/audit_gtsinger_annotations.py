#!/usr/bin/env python3
"""Compare source annotation candidates to independently submitted blind reviews."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2l.datasets.annotations import compare_source_annotations
from scripts.review_gtsinger import ReviewStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--form", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--second-review", type=Path)
    parser.add_argument("--agreement-limit-ms", type=float, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.review.is_file() or (args.second_review and not args.second_review.is_file()):
        parser.error("Independent review sessions must already exist")
    first = ReviewStore(args.form, args.review)
    second = ReviewStore(args.form, args.second_review) if args.second_review else None
    candidates = json.loads(args.candidates.read_text(encoding="utf8"))
    report = compare_source_annotations(candidates, first.rows,
        second_labels=second.rows if second else None, agreement_limit_ms=args.agreement_limit_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "evaluated_count", "total_count", "mae_ms", "p90_ms")}, indent=2))


if __name__ == "__main__":
    main()

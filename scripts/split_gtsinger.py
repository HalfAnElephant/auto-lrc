#!/usr/bin/env python
"""Split prepared songs using a human identity review or explicit dataset mode."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2l.datasets.splits import assign_song_splits


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True, help="Prepared segments.jsonl")
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--song-review", type=Path,
                          help="Human review JSON with reviewed_by and song_ids mapping to canonical IDs")
    identity.add_argument("--identity-mode", choices=("dataset",),
                          help="Use dataset song_id groups without human identity or alias confirmation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--held-out-singer", action="append", default=[],
                        help="Reserve this singer and every shared song for test (repeatable)")
    args = parser.parse_args(argv)
    segments = [json.loads(line) for line in args.segments.read_text(encoding="utf8").splitlines()
                if line.strip()]
    review = json.loads(args.song_review.read_text(encoding="utf8")) if args.song_review else None
    result = assign_song_splits(segments, review, seed=args.seed,
                               held_out_singers=args.held_out_singer,
                               identity_mode=args.identity_mode or "reviewed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())

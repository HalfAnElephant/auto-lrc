I've completed the read-only review of the six frozen files. Here are my findings.

## Result: no defensible regressions found

I traced the exclusion/error-metric taxonomy, persistence contract, and failure publication paths against the stated fixes and found no concrete regression that I can defend with a trigger → consequence → line citation. Details of what I checked:

**Exclusion / error metrics — `t2l/datasets/annotations.py:110-171`**
- First-review outcomes are partitioned: `reviewed` → `reviewed_count` (incremented at :115 before any comparability check), `ambiguous` → `ambiguous`, else → `pending_or_missing` (:112-113). `reviewed_count` is genuinely independent of candidate availability and of second-review status.
- Four distinct exclusion buckets are correct: `source_candidate_unavailable` (:138-141), `second_review_pending` / `second_review_ambiguous` (:122-124), and `disagreements` (:134-136). The second-review-absent branch `continue`s before candidate evaluation, so source metrics exclude non-adjudicated pairs (:120-137); agreed pairs fall through and are evaluated from the **first** reviewer's sample only (:143) — no timestamps/labels are averaged.
- `inter_reviewer_mae_ms` (:165) still averages **all** both-reviewed pairs (it is appended at :133 before the disagreement branch), with the new `inter_reviewer_mae_scope` marker (:164), plus `agreed_pair_count` / `agreed_mae` (:166-167). Verified arithmetic against the test vectors (1 ms + 100 ms → 50.5; agreed → 1).
- `validate_label` (:57-83) still enforces sample/seconds equivalence via exact Decimal half-up and forbids ambiguous rows from carrying a point.

**Persistence — `scripts/review_gtsinger.py`**
- Session save is hold-lock → re-read fingerprint + revision → temp file → fsync → `os.replace` → finally-unlink (`:120-152`). Stale saves, disappeared sessions, symlinked lock/session, and audio mutation all fail closed with the expected messages; the form and original WAV are never written. POST requires Host, exact matching Origin, and token (`:190-195`, `:259`); GET bootstrap is bound to `127.0.0.1` and returns the token only in the form document (`:220`), not via `/export`.

**Failure publication — `build_gtsinger_representative_manifest.py` / `prepare_gtsinger.py`**
- License chosen by path with an exact-one check before any metadata request (`:35-38`).
- Builder CLI catches `OSError`/`ValueError` → `parser.exit(1, …)`, cleans the temp file, and never clobbers an existing output (`:92-113`); `os.link` no-clobber publication is validated.
- `publish_directory` uses exclusive-rename primitives (`renamex_np` RENAME_EXCL / `renameat2` RENAME_NOREPLACE) with no overwrite fallback, and fails explicitly when the libc symbol is missing (`:24-48`); `FileExistsError` is preserved for an existing destination; staging is always removed on any failure (`:124-153`).

**Scope note:** I stayed within the six frozen files. I did not read `scripts/audit_gtsinger_annotations.py` (the only real caller), so I cannot certify the downstream *display* semantics of the returned report keys — but that is outside the frozen scope and the fix set, and I found nothing in the six files that would make those keys wrong. No re-runs requested.

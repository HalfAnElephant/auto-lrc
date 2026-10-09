I've completed a static (read-only) pass over the scoped subsystem, its tests, the prepare pipeline that feeds it, and the P1 spec/docs. Below are the concrete issues I can defend, ordered by severity. I did not execute anything (Read/Glob/Grep only), so these are code-reading findings.

## Findings

### 1. Medium — `compare_source_annotations` misclassifies real reviewed labels as `pending_or_missing` (lost annotations)
`t2l/datasets/annotations.py:113-137`

A row is only evaluated if a `reviewed` human label has a comparable source value. Two paths silently drop a genuine human annotation into the "no human label" bucket:

- A reviewed label whose candidate has `candidate_seconds=None`/invalid is appended to `missing` (`annotations.py:135-137`).
- When a second review is supplied, a first-review `reviewed` label whose second review is `pending`/`ambiguous` is appended to `missing` and `continue`d (`annotations.py:119-121`) — the second reviewer's actual boundary (if reviewed) is never even inspected.

Trigger:
```python
candidates = [{"segment_id": "x", "unit_index": 0, "candidate_seconds": None}]
labels = [<valid reviewed label with human_truth_sample=4800, sample_rate=48000>]
compare_source_annotations(candidates, labels)   # pending_or_missing == [["x", 0]]
```
Consequence: a real, independently reviewed human onset is excluded from all metrics *and* reported as if unreviewed; there is no distinct "reviewed but no source candidate" bucket, so `evaluation_count`/coverage understate real coverage and the audit cannot distinguish "not yet labelled" from "not comparable." This is the clearest "lost annotation / invalid evaluation" path in the subsystem. (Contrast with the explicitly required pending/ambiguous/disagreement exclusions, which are correct.)

### 2. Medium — `inter_reviewer_mae_ms` averages un-adjudicated disagreements
`t2l/datasets/annotations.py:128-131`

`agreement_rows.append(difference)` runs **before** the `difference > agreement_limit_ms` test, so pairs that exceed the disagreement threshold (and are therefore supposed to be set aside for adjudication, excluded from `errors`) still feed `inter_reviewer_mae_ms` (line 155). Two reviewers who disagree by 100 ms on one boundary and 1 ms on another report an inter-reviewer MAE of ~50.5 ms while the report simultaneously lists a >limit disagreement. This conflicts with the stated "disagreement must be adjudicated rather than averaged" contract (docs/p1-data-preparation.md:103); the summary metric blends resolved and unresolved disagreement. The main `mae_ms` correctly excludes disagreements, but the headline inter-reviewer number does not.

### 3. Medium-Low — split audio-identity fallback allows same-recording leakage
`t2l/datasets/splits.py:48-54`

When `sources.wav.sha256` is absent, identity degrades to `audio_path`:
```python
audio_key = ("sha256", digest) if digest else ("path", segment.get("audio_path"))
```
Two byte-identical recordings stored under different paths, mapped to different canonical song IDs, are not flagged as "same audio." Trigger: two segment dicts with identical audio bytes, no `sources` digest, differing `audio_path`, and `song_ids` mapping them to different canonicals → no `ValueError`, and the same recording can land in train and test. The prepare pipeline always emits `sources.wav.sha256` (t2l/datasets/gtsinger.py:69,93), so the live path is safe, but `assign_song_splits` is a public entry point and does not fail closed here. A digest-absent segment should either be rejected or keyed on content, not path.

### 4. Low — POST token is disclosed by an unauthenticated GET and Origin check accepts a missing Origin
`scripts/review_gtsinger.py:188-193,214-218,256-267`

`local_request()` treats a missing `Origin` header as matching (`self.headers.get("Origin", "http://"+host)`), and `GET /api/form` returns the per-process `token` to any caller that passes the Host/Origin check. Browsers cannot read the JSON cross-origin (no CORS header) and cannot send the custom `X-Review-Token` without a preflight, so the practical CSRF/leak surface is limited; but the intended defense is weaker than described: any same-machine non-browser client (or an extension) can fetch `/api/form`, read `token`, and POST labels. Given the requirement explicitly names "validates origin/Host plus per-process token for POST," requiring the token on `/api/form` (or rejecting requests with no `Origin` for state-changing routes) would match the stated contract. Impact is local-only.

## Areas I checked and did *not* find defects
- **Integer original-sample contract:** `validate_label` mandates an in-range integer `human_truth_sample`, validates any supplied seconds against it with `Decimal`/`ROUND_HALF_UP` on the *original* rate, then overwrites `human_truth_seconds = sample/rate` (annotations.py:58-70). No coordinate shift.
- **Display/playback DSP:** server `signal()` derives waveform/spectrogram directly from original PCM and centers STFT columns at `position+(nfft-1)/2`; the UI maps both canvases with the same `(sample-start)/(end-start)` and seeks via `currentTime = start/sample_rate` with `playbackRate` only affecting listening (review_gtsinger.py:154-185, app.js:36-113). No time shift.
- **Blindness/candidate leakage:** form fields are exactly `IDENTITY_FIELDS+LABEL_FIELDS` and required `pending`; `document()`, `/api/form`, `/api/signal`, `/export` and the static whitelist never expose `candidate_*` (review_gtsinger.py:36-37,61-63,214-252). The UI contains no candidate data.
- **Persistence:** atomic `mkstemp`+`fsync`+`os.replace`, `flock` cross-process exclusion, in-file revision + fingerprint gates on save and restore, source form/audio never written (review_gtsinger.py:100-152). The flock covers the actual write.
- **Split integrity:** canonical grouping, alias handling, parent/derived identity, duplicate-audio conflict, "≥N songs for N nonempty splits," deterministic hash ranking independent of input order, and held-out-singer songs pinned to test with the strict subset reported separately — all hold for the real `AudioSegment.to_dict()` schema (`singer`, `sources.wav.sha256`).
- **Timeline correspondence:** candidates stored with `candidate_seconds`; the audit compares `candidate_seconds` to `sample/sample_rate`, both on the original WAV timeline.

## Verification still needed
- Run the scoped suites (`tests/test_annotations.py`, `tests/test_gtsinger_splits.py`) — not run here (read-only worker).
- A targeted regression for finding 1: reviewed label with `candidate_seconds=None`, and reviewed-first + pending/ambiguous-second, asserting distinct reporting.
- A targeted regression for finding 3: two segments with identical bytes, no digest, different `audio_path`, conflicting canonicals → expect rejection (currently passes).
- Decide whether findings 2/4 warrant changes or explicit doc/limitation statements.

Files touched: **none** (read-only).

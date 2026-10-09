## P1 GTSinger data review — findings

No high-severity wrong-truth, leakage, or audio/score time-axis defect found. Source candidates remain distinct from human truth; classifications are screening-only; invalid segments are rejected, not clipped. Concrete issues:

**F1 — Mandarin "r" (and nasal finals) fall to `unknown`; mislabels real data.** `t2l/datasets/classification.py:24-31`: `_PINYIN_INITIALS` omits "r" (approximant) and `_PINYIN_FINALS` omits standalone "n"/"ng". Trigger: any syllable whose first phone is "r". Real `data/local/gtsinger-representative/Chinese/ZH-Tenor-1/Glissando/给电影人的情书/*/0000.TextGrid` contains phone `text = "r"` (2 occurrences each file). Consequence: `first_phoneme` reports `unknown` (via `_phone_category`, classification.py:41-49/113-118) instead of approximant — a screening completeness error, not a fabricated claim. Severity: Low–Medium.

**F2 — P1 long-note classification never fires on the representative corpus.** `t2l/datasets/gtsinger.py:222-226,242-266` flags every note in any measure containing a `<backup>/<forward>` with a later note as `has_note_after_time_cursor_change`; real MusicXML uses this (e.g. `English/EN-Alto-1/Glissando/my love/Control_Group/0000.musicxml:97` with notes after, plus mid-file `<divisions>` changes at lines 112/199). `classification.py:143-144` then returns `chord_grace_or_time_cursor_change`, so `audit_score` marks the whole score unmatched and `long_note` stays `unknown` (classification.py:245-254). Consequence: the new long-note signal is effectively always unknown on intended data — safe but non-functional. Also single-note `<slur type="start"/stop>` is rejected (`classification.py:167-172`). Severity: Medium (feature gap).

**F3 — Empty-interval contradiction aborts whole segments.** `gtsinger.py:179,192` open TextGrids with `includeEmptyIntervals=True`, but `_interval` (`gtsinger.py:45-49`) requires `start < end`, so any zero-length word/phone/technique interval raises and discards the entire segment rather than flagging it. Not observed in the 20 representative files (all contiguous), so latent. Severity: Low.

**F4 — Fragile license selection.** `scripts/build_gtsinger_representative_manifest.py:38` copies `smoke["files"][0]` positionally as the license; currently correct (smoke `files[0]` is `dataset_license.md`), but any reordering silently pins the wrong file. Severity: Low.

**F5 — Non-atomic/unguarded error paths.** `build_gtsinger_representative_manifest.py:90-104`: no handling for validation failure or `os.link` race — an error surfaces as traceback, not clean `parser.error`. `prepare_gtsinger.py:30-44`: `renameat2` is used unconditionally on Linux; libcs lacking the wrapper raise `AttributeError` (uncaught) instead of a clean failure. Severity: Low/portability.

### Positively verified (no defect)
- Time axis: `_sample` (gtsinger.py:52-53) uses `Decimal(str(seconds))` half-up; `Boundary.human_truth_*` null and `review_status="pending"`; no seconds derived from MusicXML (score fields are divisions/quarters only, classification.py:217,246-250).
- Alignment: inspected TextGrid decimals (`xmax = 0.166`, `0.52`) equal JSON values, so `row_aligned` holds and technique screening proceeds on real pairs.
- Exclusions are genuine time-domain overflows (TextGrid domain beyond WAV duration) rejected by `_interval`, not clipped; exclude mode retains `reason` and integrity checks (`verify_file`) run outside the try.
- Manifest internal consistency: 20 samples/81 files, declared 25,807,238 bytes; coverage 6 songs, 5 singers, groups 10/3/7; lyric counts (sum 268) minus the two excluded segments (13+24) = 231.

### Unreviewed scope
Did not read `split_gtsinger.py`, `audit_gtsinger_annotations.py`, `review_gtsinger.py`, `t2l/datasets/{annotations,splits}.py`, binary WAV bytes, or execute anything. WAV durations/exclusions taken from supplied evidence. No files were modified.

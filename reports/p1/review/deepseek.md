# P1 独立只读审查

模型：deepseek-flash[1m]；工具：Read,Glob,Grep。首轮达到14轮上限，未计为通过；检查源码/测试摘要未变化后，沿用会话 6f053e0e-8002-44f7-baee-c3cd9506f5cb 做限定收尾。续接成功，exit_code=0、无超时/权限拒绝、源码摘要仍一致。结论仅为静态审查，不替代已执行的定点测试。

## P1 read-only review — findings

**No actionable correctness defect found** in the reviewed P1 files, based on static inspection. I did not execute tests (read-only), so this is a code-reading verdict, not an independent test run.

### What I verified by inspection (no defect)

- **Time/sample conversion** (`t2l/datasets/gtsinger.py:51`): `Decimal(str(seconds)) * sample_rate`, `ROUND_HALF_UP`, using the WAV's *actual* rate. The half-value case (`0.10003125×48000 = 4801.5 → 4802`) and real candidates are arithmetically consistent (`0.52→24960`, `3.411→163728`, `8.121→389808`). Bounds are validated against `num_samples/sample_rate`.
- **SP/AP handling** (`gtsinger.py:55-60`, `318-324`): `<SP>`/blank → silence, `<AP>` → breath, boundary `None`; the first *lyric* phone supplies the candidate. In the real samples this yields the recorded 25 lyric / 5 silence / 1 breath with `<AP>` never becoming a lyric. Word labels are stored verbatim (source fidelity).
- **JSON/TG association** (`gtsinger.py:282-316`): phone→word containment is monotonic and rejects crossing phones; JSON is associated ordinally but mismatches are surfaced as `json_textgrid_mismatch` with both sources retained (never repaired); count mismatches are reported and unpaired units keep `raw_annotations=None`.
- **Source fidelity/validation** (`gtsinger.py:63-68`, `_read_json`, `_read_textgrid`, `_read_score`): parsed structure is validated (finite, ordered, in-range, array lengths) with source-path errors; raw JSON/tiers/score notes and SHA-256 per source are preserved; WAV is checked for uncompressed PCM and a real data-length match (truncation rejected).
- **Failure/overwrite** (`prepare_gtsinger.py:24-26,72-99`; `fetch_gtsinger_samples.py:112-147`): prepare refuses an existing/symlink out-dir, stages in a temp dir, re-checks before `os.rename`, and removes staging on failure; fetch verifies size+checksum before atomic `os.link` publication and never overwrites a mismatching existing file.
- **Blind review** (`prepare_gtsinger.py:18-20,42-55,76-93`): review rows carry only segment/audio/word/index plus null human fields and `review_status="pending"`; candidate times live only in `annotation-candidates.json`; no `candidate_*` keys appear in the forms.
- **Split leakage** (`t2l/datasets/splits.py`): requires a non-empty `reviewed_by` and a covering `song_ids` map (fails closed), groups whole canonical songs, enforces parent/derived identity and duplicate-audio identity conflicts, rejects fewer than N groups for the requested non-empty splits, sorts output so input order is irrelevant, and records `singer_independent=False` / `boundary_annotations_reviewed=False`.

The tests appear to target these contracts (byte-level source immutability + SHA-256, null/pending blind fields, overwrite refusal, alias/derived/duplicate-audio leakage, two-songs-can't-be-three-splits), but I did not run them.

### Material limitations / non-blocking observations (not defects)

1. **Split identity fallback** (`splits.py:50`): when `sources.wav.sha256` is absent, identity degrades to `audio_path`. Two byte-identical recordings at different paths with conflicting canonical IDs would not be flagged. The reader/prepare pipeline always emits SHA-256, so the intended pipeline is unaffected; this only narrows protection for externally constructed segment dicts. Consider requiring a digest in the splitter if other producers will feed it.
2. **Machine-specific paths in outputs** (`prepare_gtsinger.py:48`, sourced from `gtsinger.py:329`): absolute local `audio_path` is embedded in `manual-review.json/csv`, `segments.jsonl`, etc. (e.g. `reports/p1/samples/manual-review.json:9`). Harmless now, but regenerated/shared reports carry stale local paths.
3. **Token vocabulary is sample-scoped** (`gtsinger.py:25,55-60`): silence/breath detection is verified only for `<SP>`/`<AP>` in these two files; behavior for other languages, techniques, or the full corpus is unverified future scope, not a demonstrated defect.
4. **Scope is intentionally incomplete**: 2 samples / 2 songs / 2 languages, both Alto-1, no independent human labels → accuracy `unavailable`, split `unassigned`, no three-way dataset, no 5 ms evidence. This is correctly enforced and recorded, not a pass/fail misstatement.
5. **No cross-format note correspondence and no whole-recording offset** (`source_recording_offset_samples` null) — consistent with the stated contract.

### Verification still needed

- I ran no commands; the focused suites (reader 21 / tooling 21 / splits 9) are per the handoff, not independently reproduced here.
- No finding prompts a new targeted regression; the main agent need not treat anything as newly broken. If hardening item 1 is desired, the minimal targeted test would feed two segments with identical audio bytes, differing `audio_path`, and no `sha256`, mapping to different canonicals.

Files touched: **none** (read-only).

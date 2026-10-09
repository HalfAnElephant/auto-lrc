# DeepSeek 独立只读审查

模型：`deepseek-flash[1m]`；工具：`Read,Glob,Grep`。执行成功，exit_code=0、无超时、无 permission_denials。审查前后源码摘要一致，无文件修改。以下是审查原文，并非全部采纳；处置见后续技术报告。

I have enough to complete the review. Below are my findings.

---

# Independent P0 review — baseline/diagnostics change

Scope reviewed: `main.py`, `t2l/mtl/wrapper.py`, `t2l/t2l.py`, `t2l/mtl/utils.py`, `t2l/mtl/model.py`, `scripts/benchmark_baseline.py`, the four modified/new test modules, and `docs/WORK_PROGRESS.md` / `docs/p0-methodology.md` / `docs/environments.md` / `docs/progress/high-precision-spec.md`. Read-only; no files modified.

## Summary

I found **no hard correctness defect** in the core additions (frame/seconds conversion, phoneme mapping, default return contracts). The frame grid math is internally consistent with the model geometry, defaults are preserved, and the "uncalibrated / no precision claim" posture is honest. The findings below are robustness and reproducibility gaps worth resolving or explicitly documenting before P0 acceptance.

## 1. API / default compatibility — OK

- `align(..., return_details=False)` and `process(..., return_details=False)` only append parameters; existing positional/keyword callers are unaffected (`t2l/mtl/wrapper.py:29-30`, `t2l/t2l.py:23-26`).
- `t2l/t2l.py:98` `word_align, words = alignment_result[:2]` is safe for both the 2-tuple (default) and 3-tuple (`return_details=True`) returns.
- `main.py:76,78` only requests details when `--json-output` is set, so the no-diagnostic path prints/writes the legacy string byte-for-byte; the CLI test asserts this (`tests/test_cli.py:76-79`).
- `FRAME_RESOLUTION_SECONDS = HOP_LENGTH / SAMPLE_RATE * FRAME_STRIDE` reproduces the prior literal `256 / 22050 * 3` exactly (same operation order), so default LRC timestamps are unchanged (`t2l/mtl/wrapper.py:17`, `t2l/t2l.py:258`).

## 2. Raw frame / seconds and phoneme mapping — OK, one asymmetry

- `wrapper.py:105-107` measures `mel_frames` from the real transform output and `output_frames = song_pred.shape[0]`; both are batch-1 so `frames == T` and `mel_frames//3`, consistent with `MaxPool2d(kernel=(2,3))` pooling the time axis by 3 (`t2l/mtl/model.py:132,145-161`). The `observed_output_frames == mel_frames // temporal_stride` check in `scripts/benchmark_baseline.py:147-148` is the right identity.
- `units` seconds are computed as unrounded `frame * FRAME_RESOLUTION_SECONDS`, matching `gen_lrc`'s conversion, and nulls are emitted when `word_align` is short (`t2l/t2l.py:116-129`) — the test with `word_align=[[3,8]]` exercises the null branch (`tests/test_t2l.py:178-182`).
- **Minor asymmetry (low):** the `word_align` guard protects only the frames; `phone_start, phone_end = idx_word_p[word_index]` is unconditional (`t2l/t2l.py:116-118`). This is safe today because `gen_phone_gt_opt` guarantees `len(idx_word_p) == flat word count` (`t2l/mtl/utils.py:179-191`) and `align` returns one entry per `idx` (`utils.py:430-452`), but if a caller supplies heterogeneous/mocked `idx_word_p` shorter than `lines`, this raises `IndexError` instead of emitting nulls like the frame fields. Consider mirroring the guard for phoneme indices.

## 3. Failure contracts — one real JSON robustness issue

- Default/normal failure paths (`TxtValueError`, `AudioValueError`, model/CUDA exceptions) are unchanged; `format` is still validated before any audio work (`t2l/t2l.py:32-35`, test at `tests/test_t2l.py:10-20`).
- **Finding (medium):** JSON output is written with `allow_nan=False` (`main.py:94`, `scripts/benchmark_baseline.py:174-176`), but the BDR path computes `bdr_outputs = torch.log(bdr_outputs) * alpha` (`t2l/mtl/wrapper.py:84`). If a boundary sigmoid underflows to exactly `0.0` (large negative logit), `log(0) = -inf` enters the DP (`utils.py:484,517`) and `score` can become `-inf`; `float(score)` is then stored at `wrapper.py:112`. Consequences:
  - `main.py`: `json.dump(..., allow_nan=False)` raises `ValueError` **after** the standard LRC was already written and printed (`main.py:83-94`), leaving partial output.
  - benchmark: the success-path `write_json` raises inside the `try`; the `except` rewrites `row` containing `raw_alignment_score=-inf` and calls `write_json` again, which raises uncaught, masking the original error (`benchmark_baseline.py:257-270`).
  Recommend normalizing/validating non-finite `score` (or surfacing it as `null` with an explicit flag) before serialization; the non-BDR score is provably finite given `_validate_alignment_inputs` reachability (`utils.py:38-47`), so this is BDR-specific.

## 4. Output safety — precheck is sound; write ordering is not atomic

- `_check_output_paths` (`main.py:41-48`) runs before any input read, compares resolved paths, and uses `samefile` for hardlinks; the collision matrix (input/standard/extra/symlink/hardlink) is exercised at `tests/test_cli.py:85-113` and inputs are asserted untouched. Good.
- **Gap (low):** the standard LRC, enhanced LRC, and JSON are written sequentially in separate `open(...)`/`write_text` calls (`main.py:83-95`) with no temp-file/atomic rename. A failure mid-way leaves partial artifacts. Also `--json-output ""` / `--enhanced-lrc-output ""` are silently ignored (`bool("")` is falsy), so an explicit empty PATH is neither validated nor rejected. No change needed for P0 correctness, but worth noting.
- `write_csv` in `wrapper.py:236` still hardcodes `256 / 22050 * 3` instead of the new constant — harmless (function unused) but an inconsistency.

## 5. Reproducibility / measurement claims — mostly honest, one ordering caveat

- Repeatability fingerprints correctly exclude timing and include raw units/LRC/score (`benchmark_baseline.py:179-183`), and `null` is returned only when `repeats == 1` (`:276`). `accuracy_metrics` is hardwired unavailable with no MAE/P90/coverage invented (`:20-26,229,235`). The `score_kind` strings and `physical_origin_calibrated=False` markers match the "uncalibrated, no precision claim" stance (`wrapper.py:113-117`). Good.
- **Finding (medium / verify):** `p0-methodology.md:14` claims the benchmark preserves "原 `process()` 的随机数消费顺序," but the ordering differs from the original: the benchmark seeds, then constructs the model **before** calling `process` (`benchmark_baseline.py:241,247-254`), whereas original `process` runs phonetization/audio decode *then* constructs the model inside `align`. If g2p/decoding consume any torch RNG, the smoothing `P_noise` draw (`wrapper.py:77`) will not match the original reproduction, so "legacy == original" parity could differ by a frame on ties. This is likely fine (g2p uses NLTK, decoding uses torchaudio/librosa), but the claim should be verified or softened; the benchmark's own per-repeat repeatability is unaffected because its ordering is self-consistent.
- `reset_seed` calls `torch.use_deterministic_algorithms(True)` and never restores (`benchmark_baseline.py:53-60`). Fine for a one-shot CPU/no-vocalize script; note it is not exercised with CUDA/Demucs where non-deterministic-op errors could surface. `docs/p0-methodology.md:14` already scopes `batch-first` to fresh instances and disclaims precision gain, and `test_layout_ablation_is_instance_local_and_preserves_weights` verifies weights are untouched.

## 6. Test coverage notes

- Good: `test_alignment_details_preserve_legacy_path_and_rng` asserts `detailed[:2] == legacy`, RNG state unchanged, and the exact mel/output frame counts using a deterministic `RecordingTransform` (`tests/test_wrapper.py:189-227`). `test_timeline.py` grounds the STFT-center/pooling geometry against synthetic impulses.
- Gap: `tests/test_cli.py:60-64` mocks `process` wholesale, so the CLI tests don't assert JSON serialization of real `units`/`alignment` (covered only indirectly via `tests/test_t2l.py:183` and the benchmark test). No test covers a non-finite score, matching finding 3.

## Verification still needed (not performed here)

- Real MTL/MTL_BDR demo parity between untouched `2c475cc` and this working tree (frames, score, LRC) — the docs list this as unexecuted.
- Confirmation that g2p/audio decoding consume no torch RNG, to validate the `p0-methodology.md` ordering claim (finding 5).
- Whether the BDR score can actually reach `-inf` on the demo, to size finding 3.

No files touched; no commands executed.

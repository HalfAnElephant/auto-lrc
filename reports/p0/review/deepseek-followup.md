# DeepSeek 定点复审

模型：`deepseek-flash[1m]`；工具：`Read,Glob,Grep`。执行成功，exit_code=0、无超时、无 permission_denials。复审前后所有待审源码/测试摘要一致。未发现新的正确性问题。

Follow-up review complete (read-only, no files modified). Scope: `t2l/mtl/wrapper.py`, `scripts/benchmark_baseline.py`, the two tests, `docs/p0-methodology.md`.

## Nonfinite-score guard — correct

`wrapper.py:101-104` checks `math.isfinite(float(score))` inside the `return_details` branch, before the details dict is built. Default API is untouched (`:130`), verified by the new parametrized test covering `-inf/inf/nan` (`test_wrapper.py:230-242`). Because `process()` only requests details when `--json-output` is set and the raise happens inside `align` — i.e. before `gen_lrc` and all writes in `t2l.py:99` / `main.py:83-95` — the guarded path now fails without emitting any partial artifacts. Good; this is stronger than the prior state.

Residual (not blocking): with `return_details=False`, a nonfinite score still silently yields `word_align`/LRC. That is the intended legacy behavior, but nothing in the default CLI surfaces the uncalibrated/nonfinite condition. Documented only for the diagnostics branch (`p0-methodology.md:31`). Also `process()`'s docstring still names only `TxtValueError`/`AudioValueError` as failure modes (`t2l/t2l.py:29-30`) although `AlignmentValueError` is now reachable through `return_details=True`; a one-line docstring note would close the contract gap.

## Failure-report fix — works

`benchmark_baseline.py:270` pops `raw_alignment_score` before the error row is serialized, so the `except` no longer re-raises on the same bad value. Note the first failure actually occurs at `prediction_fingerprint` (`:261`, also `allow_nan=False`), before `write_json`; either way it is caught and the readable row is written, as `test_invalid_prediction_still_leaves_readable_failure_report` asserts (`:156-160`).

Residual (low): the test exercises failure before the `.lrc` writes. If `to_standard_lrc` (`:265`) failed after `:264` wrote `<stem>.lrc`, a failure row would still be emitted but a stray `.lrc` would remain — the "no lrc on failure" assertion would not hold. Only relevant if you want failures to be artifact-clean in all orderings.

## Thread initialization order — correct

There is now a single `set_num_threads` call (`:208`), before the `ext.t2lutils`/`main`/`t2l` imports (`:210-213`); `set_num_interop_threads(1)` precedes them too. `manifest["config"]` reads the effective values after import (`:225-226`), so it now records what actually took effect. The `reports/p0/original/inference.log` warning ("Cannot set number of intraop threads after parallel work has started … torch.set_num_threads(2)") is direct evidence the prior experiments were mis-threaded, so the fix is warranted and the methodology note (`:16`) correctly excludes those runs. No duplicate or late setters remain.

Residual: confirm the pending full run's `manifest.json` shows `torch_threads == 2`; if `import torch` itself were to start interop work, `set_num_interop_threads(1)` could still warn — empirical demo success suggests it does not.

## Methodology

Lines 14/16/31 now scope RNG parity to observed experiments, document the thread ordering, and disclose nonfinite handling plus non-atomic multi-file writes. Consistent with the code; no overclaim.

No new correctness defect found. Files touched: none. Verification still needed: the full MTL/MTL_BDR comparison now in progress, and confirmation that the recorded `torch_threads` equals the requested 2.

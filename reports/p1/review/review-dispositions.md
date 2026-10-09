# P1 independent review and dispositions

Date: 2026-10-10. Review model: `deepseek-flash[1m]`, dispatched through the
local DeepSeek worker with `Read,Glob,Grep` only. The worker ran no tests and
reported no file edits. Root owns implementation and acceptance. No real human
labels exist; P1 remains unaccepted and no model accuracy is reported.

## Data review

Raw findings: [deepseek-data.md](deepseek-data.md). Session
`3b13294e-7f47-46c4-b2cf-70feb3d09974` reached its turn cap, so that invocation
was a failure. Inspection found no source changes. A bounded continuation of the
same session completed successfully in one turn; it did not repeat data work.

- F1: unsupported Mandarin phone categories remain `unknown`, an acknowledged
  screening coverage limit. No unsupported category is converted into truth.
  The suggested category for `r` was not adopted without a verified rule.
- F2: the claimed corpus-wide long-note failure is not reproduced. Current
  `representative-v2/segments.jsonl` contains 18 `matched` score audits and
  long-note counts of 23 `present`, 208 `absent`, no `unknown`. The supported
  reader distinguishes measure-local cursor changes from later measures.
  Unsupported slur structures intentionally fail conservative score matching.
- F3: zero-length source intervals intentionally fail validation. None occurred
  in the inspected 20 segments; silently repairing them would change the source
  timing contract. `--on-invalid exclude` records rejected segments explicitly.
- F4/F5: fixed. License selection now requires exactly one matching source path.
  Manifest construction/validation/link-publication OS errors exit clearly and
  clean temporary files. Missing exclusive libc rename functions fail explicitly;
  no overwrite-capable fallback is used. Tools suite: 39 passed.

The raw review described exclusions as TextGrid domain overruns; the actual
recorded failures are JSON interval ends exceeding the original WAV by one
sample. The original files and exact errors remain in the exclusion reports.

## Annotation and split review

Raw findings: [deepseek-annotations.md](deepseek-annotations.md). Session
`79568f06-e5b1-4c83-894b-69220ae7b975` completed successfully. Its scoped files
were frozen during review; root and the assigned agent made subsequent fixes.

- F1: fixed. Completed first-review labels are counted regardless of source
  comparability. Missing source candidates and unfinished/ambiguous second
  reviews have separate exclusion lists; they no longer appear as absent first
  labels. Targeted regressions cover each branch.
- F2: clarified. Mean reviewer differences are a diagnostic, not averaged
  ground-truth timestamps. The preserved all-pairs metric has an explicit scope;
  agreed-pair count/MAE are reported separately. Unadjudicated disagreements never
  enter the source-error metrics. A mixed 1ms/100ms regression covers this.
- F3: fixed. Every split input requires a valid 64-character hexadecimal source
  WAV SHA-256. Different paths cannot replace missing content identity, and case
  differences cannot bypass duplicate identity checks. Split suite: 25 passed.
- F4: POST now requires an explicit matching Origin, Host and token. GET remains
  the local browser's token bootstrap. This loopback tool is not authentication
  against processes running as the same local user. Annotation/API suite:
  21 passed, including missing/foreign Origin rejection.

Reader/classification suite: 53 passed. Combined focused P1 checks: 138 passed.
P0's prior successful results were reused; no equivalent model/CI rerun.

## Fix review

[deepseek-fixes.md](deepseek-fixes.md), session
`5db6c1a5-8c1d-4959-a7b0-097025c69baa`, successfully reviewed the six frozen
annotation/API/publication files after fixes. Pre/post SHA-256 values matched;
the worker found no defensible regressions. It did not cover the split digest
guard or the unchanged audit CLI. Root separately inspected that guard and the
CLI, which passes the report through without reinterpreting its fields.

After restarting the local service, a real browser POST passed Origin/token
checks and failed at the required reviewer identity validation (HTTP 400).
No label/session was created: progress stayed 0/25, with no page errors or
horizontal overflow. The final split guard is covered by 25 focused tests.

## Integration and remaining acceptance

Mac and WSL independently prepared 18 usable / 2 rejected segments with 231
pending boundaries; 20 original WAVs are 48kHz. WSL dependency and exclusive
publication evidence is retained under `../wsl/wsl-validation-20261010/` with
the executed source hashes. Post-review code sync is separately recorded.
Browser checks and screenshots cover desktop/mobile rendering and controls;
they do not supply an actual human onset judgement.

Canonical song identity review, real independent boundary labels, disagreement
adjudication and fixed evaluation splits remain pending. No 5ms success claim
or permission to enter P2 follows from these checks.

## Controlled mixture experiment

[Generator review](deepseek-mixtures-generator.json), session
`3b76a426-bad9-4e6b-9c1d-ea5460324065`, completed successfully with
Read/Glob/Grep only. The generator and its tests were unchanged during review.

- The existing 1-LSB check applied to gain-scaled source samples. Metadata now
  explicitly names that domain and also records the inverse-gain reconstruction
  bound. This clarifies amplitude quantization, not time precision. A decoded
  silent backing now fails explicitly before calculating measured SNR.
- Generator provenance now includes both imported helper modules. Actual WSL
  output records the post-review source and dependency hashes.
- Arbitrary new generator output paths remain supported, consistent with the
  other dataset tools. Audio handling in this task uses ignored `data/local/`
  directories and explicit text-only Git staging. The CLI license notice and
  guide state that mixed vocals retain the dataset terms; no audio is committed.

[Benchmark review](deepseek-mixtures-benchmark.json), session
`513bd580-1a1f-4e3b-bb23-fe3ea3505be6`, hit the 8-turn limit and therefore
failed that invocation. Root inspected the transcript and unchanged pre/post
source/test hashes. A [bounded same-session conclusion](deepseek-mixtures-benchmark-finish.json)
then completed successfully in one turn without further tools. The reviewer
confirmed no functional defect by static inspection. It ran no tests or CUDA.

- The cached Demucs checkpoint naming assumption was checked by the real WSL
  execution: four unique checkpoints were recorded and all four models forwarded
  for each selected segment.
- This small complete-fixture benchmark intentionally rejects a missing unit
  interval instead of treating a partial alignment as success. The failed-run
  report preserves the failure and earlier completed predictions, but does not
  retain raw details of the particular prediction rejected inside validation.
  All 12 actual predictions had complete finite intervals. This strict fixture
  acceptance is not a general evaluation missing-alignment policy.

Root's two focused test modules passed **53 tests in 1.15s**, and syntax checks
passed. WSL generated 18 fixtures; the selected Chinese and English examples
completed 12 real CUDA alignments plus two four-member separations. Artifact
inspection in `../mixtures-cuda-20261010/acceptance.json` checks source/model
hashes, actual CUDA observations, sample-grid and quantization invariants.
Existing 138 P1 and 60 P0 test results were reused; no equivalent CI or baseline
inference was repeated. Realistic matched accompaniment, independent human
boundaries, canonical song identity and final P1 acceptance remain pending.

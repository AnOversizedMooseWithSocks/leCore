# Qwen3.5 v4 preflight record

This directory records two owner-authorized AWS v4 bootstrap attempts. Both
stopped at the frozen test gate **before ilxyr initialization, admission, model
download, installation, or formal execution**. They are infrastructure
preflight failures, not `execution_failure` experiment outcomes, and they
provide neither a scientific `GO` nor `NO-GO`.

| Preflight | Frozen source | Result | Blocker |
|---|---|---|---|
| 001 | `13849b0ebd2ed1f0a252534a130c2a030966c7d1` | 46 passed, 1 failed, 1 skipped | A fixture required bit-for-bit equality between two separately loaded NumPy runtimes; Linux BLAS produced sub-micro float32 rounding differences. |
| 002 | `df412cc7c938bb376014b7a68067ccc91ad9a5a6` | 47 passed, 1 failed | The same obsolete exact assertion remained in the official-Transformers fixture, which the lightweight local environment had skipped. |
| 003 | `c1ddc22916957238c00890bec13fd0609b01b521` | Runner preflight passed; ilxyr compile failed | The generated `experiment.json` contained a top-level `runner_policy` field outside the frozen ilxyr schema. |

The repaired source centralizes the portable identity gate: embedding bytes
must remain exactly equal, while independently evaluated logits must agree at
`rtol=1e-6` and `atol=1e-6`. This is still roughly three orders of magnitude
tighter than the frozen `1e-3` reference-logit acceptance threshold and does
not alter the model, treatment, 4,096-token requirement, or statistical gate.

Implementation commit `c1ddc22916957238c00890bec13fd0609b01b521` was pushed
to PR #31. The previously skipped official reload/generation test then passed
with the pinned dependency stack on macOS and in a clean Python 3.12 Linux
container. The authorized final host passed that gate and the real-model
benchmark, then stopped before admission on the generator/ilxyr schema mismatch.

## Evidence and ilxyr status

Each preflight directory contains the exact terminal status, runner status,
AWS launch record, and frozen test output. `evidence-bundles.json` binds the
complete S3 bundles by byte length and SHA-256.

The first two attempts have no `.ilxyr` ledger because they stopped before
initialization. The third produced a partial ledger with four contribution
events; independent `ilxyr verify` returned `valid: true`. Its full evidence is
recorded and permanently published separately under
`../v4-c1ddc2291695-preflight-failure/`. No experiment was compiled or admitted,
and no scientific result exists.

## Cleanup

All three instances are terminated. Their encrypted root volumes and temporary
security groups were deleted. The combined preflight compute estimate is below
approximately `$0.27`; see this record and the final attempt's cleanup receipt.

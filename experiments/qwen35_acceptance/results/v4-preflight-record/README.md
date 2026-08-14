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

The repaired source centralizes the portable identity gate: embedding bytes
must remain exactly equal, while independently evaluated logits must agree at
`rtol=1e-6` and `atol=1e-6`. This is still roughly three orders of magnitude
tighter than the frozen `1e-3` reference-logit acceptance threshold and does
not alter the model, treatment, 4,096-token requirement, or statistical gate.

Implementation commit `c1ddc22916957238c00890bec13fd0609b01b521` was pushed
to PR #31. The previously skipped official reload/generation test then passed
with the pinned dependency stack on macOS and in a clean Python 3.12 Linux
container. No third AWS host was launched.

## Evidence and ilxyr status

Each preflight directory contains the exact terminal status, runner status,
AWS launch record, and frozen test output. `evidence-bundles.json` binds the
complete S3 bundles by byte length and SHA-256.

There is intentionally no `.ilxyr` ledger or Arweave experiment publication
for these attempts: the preregistered publication procedure begins only after
an ilxyr workspace exists and `ilxyr verify` succeeds. Inventing an ilxyr
result for a bootstrap failure would misstate the scientific record. The
formal v4 one-run allowance remains unused.

## Cleanup

Both instances are terminated. Their encrypted root volumes and temporary
security groups were deleted, and the v4 completion monitor was removed. The
combined preflight compute estimate is below `$0.12`; see `aws-cleanup.json`.

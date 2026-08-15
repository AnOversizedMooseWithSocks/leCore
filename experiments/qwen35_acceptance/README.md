# Qwen3.5 acceptance experiment

The committee-facing design and review questions are in
[`PROPOSAL.md`](PROPOSAL.md). V1-v3 and v5 are preserved as execution
failures, and v4 is preserved as a preflight failure. V6 is the first formally
admitted, fully executed correctness result. V7 independently preserves that
parity while qualifying the native full-sequence C GDN recurrence on AWS. Both
resolve their frozen identities as `accepted` — **GO**.

The promoted v7 record is in
[`results/v7-cb3b1d2ac71c-accepted/`](results/v7-cb3b1d2ac71c-accepted/).
Its complete evidence bundle, excluding model weights, is permanently indexed
at [`RaDOgThG…CUSa0`](https://arweave.net/RaDOgThGonnt9eLUe_1eKjJukMY-FnbXlBYSRmCUSa0/).
All 4,096 paired positions, the statistical gate, checkpoint reloads, text
generation, and the official vision-input smoke test passed. Original and
installed perplexity were identical in this frozen run. The accepted v6 record
remains permanently indexed at
[`Ks5BCVFX…ikiTs`](https://arweave.net/Ks5BCVFX6179VUXQL8lMczLXX6hAp7j-lUpOcXikiTs/).

The layer-prepending installer remains experimental pending committee review.
V6 recorded a safe NumPy fallback and therefore validates correctness but not
compiled acceleration. V7 closes that specific gap for the full-sequence C
recurrence; it does not establish native cached-step generation. No retry is
authorized under either frozen identity.

## Compute accounting

The published compute estimates are `$4.71` (v2), `$2.66` (v3), `$0.27`
(all v4 preflights), `$2.57` (v5), `$2.70` (v6), and `$1.32` (v7): about
`$14.23` for v2-v7, plus the unestimated v1 attempt and a few cents of storage.
The `$6.86` figure in the v7 launch tracker is specifically the v4-v7 phase
subtotal. The proposal explicitly says its `$10` ceiling buys one formal
attempt; it must not be read as all-history program spend.

This directory turns the open Qwen integration questions into a frozen ilxyr
project and preserves its bounded executions, including the accepted v6 and v7
results. It generates the
hypothesis, methodology contributions, experiment contract, two explicit model
forecasts, funding record, and ordered ilxyr commands for one real checkpoint.

The contract requires:

- exact tokenizer IDs and reference-logit parity before installation;
- at least 4,096 paired evaluation positions with a moving-block bootstrap;
- peak resident/GPU memory reporting;
- reload of the emitted checkpoint from disk;
- official Transformers text generation and image-input smoke tests; and
- spectral filtering to remain disabled.

Generate a project from the leCore checkout:

```bash
python experiments/qwen35_acceptance/generate.py \
  /absolute/path/to/Qwen3.5-0.8B \
  /absolute/path/to/installation-corpus.txt \
  /absolute/path/to/evaluation-corpus.txt \
  /absolute/path/to/ilxyr-project \
  --experiment-version 7 \
  --python .venv-qwen-acceptance/bin/python
```

The two corpora must have different contents.

Create that CPU environment from the experiment-owned frozen file before
generating or running a project. It installs the matching Torch/Torchvision
pair required by Transformers' official vision path; upgrading either package
independently is not a supported experiment environment.

```bash
python3 -m venv .venv-qwen-acceptance
. .venv-qwen-acceptance/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experiments/qwen35_acceptance/requirements-cpu.txt
python -c "import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__)"
```

The generated project records the virtual environment's Python executable. The
runner source snapshot includes [`requirements-cpu.txt`](requirements-cpu.txt),
so the result also binds the frozen environment definition as evidence.

`project.json` records the exact model and both corpus digests and the ordered commands
for `~/develop/ilxyr/target/debug/ilxyr`. Review the forecasts and frozen
thresholds before contributing them to an ilxyr workspace. ilxyr executes the
runner without a shell and records its strict `metrics`/`source` envelope.
That closed envelope contains only the fields accepted by the frozen ilxyr
schema. Extended input and emitted-checkpoint provenance is stored in
`runtime-provenance.json`; the runner includes that file's SHA-256 in
`source.artifacts`, so the richer record remains content-bound evidence without
adding undeclared executor fields. Before model loading, the runner also
re-hashes the source commit, complete model/processor manifest, installation
corpus, and held-out corpus and refuses any drift from the admitted identity.

The layer-prepending installer is deliberately invoked with `--experimental`.
V7 supplies the native-qualified accepted outcome; removing that flag remains
a maintainer and committee decision.

Future-run performance controls are frozen into the generated project. The
generator requests the parity-gated C recurrence, increases chunks above 128
only when a checksummed benchmark report binds the model and evaluation corpus,
evaluates original and installed checkpoints in isolated concurrent processes
when memory permits, and uses
Bonferroni-corrected looks at 1,024-token intervals for early rejection only.
Every GO still requires all 4,096 paired positions and the unchanged final 95%
paired block-bootstrap bound.

Starting with v4, the full runner policy is canonical JSON whose SHA-256 digest
is part of the experiment ID and checker lineage. It covers chunking and worker mode,
the evaluation and reference backends, all final and sequential statistical
rules, official Torch/Transformers/Pillow versions, installation treatment,
the ban on spectral and weight-resident metadata, content-addressed sidecars,
telemetry behavior, timeout, and compute ceiling. The complete policy and its
digest are stored in `project.json`; the digest is bound into the experiment ID
and checker lineage in the schema-constrained `experiment.json`. Changing any
bound setting therefore creates a different experiment identity.

Experiment v7 and later additionally make native GDN execution a mandatory
acceptance gate when the C backend is requested. Both isolated evaluators must
report successful fresh-state and resumed-state parity checks, actual native
calls and tokens, and no refusal or fallback. A completed run that preserves
model correctness but falls back to NumPy is therefore NO-GO for the native
acceleration claim, rather than GO with a diagnostic caveat.

CI builds the exact pinned ilxyr revision, verifies a real native compilation
from an ilxyr-style environment with no inherited `PATH`, generates and admits
a miniature project, and then performs a real zero-compute `ilxyr run
--execute`. The job
requires a clean executor parse, the exact metric set, an accepted evidence
record, one `EvidenceRecorded` ledger event, and a valid workspace. This tests
the production JSON boundary without loading a model or spending cloud compute.

On AWS, pass `--progress-upload-uri s3://bucket/prefix/progress.jsonl` while
generating the project, or set `LECORE_PROGRESS_UPLOAD_URI` in the runner
environment. The durable JSONL log is then refreshed in S3 after every
evaluation chunk and stage transition; upload failures are reported but do not
change the scientific result. Use the procedure in
[`BENCHMARKING.md`](BENCHMARKING.md) with v3's observed peak before selecting a
64 or 32 GiB instance.

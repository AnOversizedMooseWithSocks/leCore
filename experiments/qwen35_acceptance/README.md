# Qwen3.5 acceptance experiment

This directory turns the open Qwen integration questions into a frozen ilxyr
project for one owner-authorized, bounded attempt. It does not claim that the
full run has passed. It generates the
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
  /absolute/path/to/ilxyr-project
```

The two corpora must have different contents.

`project.json` records the exact model and both corpus digests and the ordered commands
for `~/develop/ilxyr/target/debug/ilxyr`. Review the forecasts and frozen
thresholds before contributing them to an ilxyr workspace. ilxyr executes the
runner without a shell and records its strict `metrics`/`source` envelope.

The layer-prepending installer is deliberately invoked with `--experimental`.
An accepted ilxyr outcome is the evidence needed before removing that flag.

Future-run performance controls are frozen into the generated project. The
generator requests the parity-gated C recurrence, increases chunks above 128
only when a checksummed benchmark report binds the model and evaluation corpus,
evaluates original and installed checkpoints in isolated concurrent processes
when memory permits, and uses
Bonferroni-corrected looks at 1,024-token intervals for early rejection only.
Every GO still requires all 4,096 paired positions and the unchanged final 95%
paired block-bootstrap bound.

On AWS, pass `--progress-upload-uri s3://bucket/prefix/progress.jsonl` while
generating the project, or set `LECORE_PROGRESS_UPLOAD_URI` in the runner
environment. The durable JSONL log is then refreshed in S3 after every
evaluation chunk and stage transition; upload failures are reported but do not
change the scientific result. Use the procedure in
[`BENCHMARKING.md`](BENCHMARKING.md) with v3's observed peak before selecting a
64 or 32 GiB instance.

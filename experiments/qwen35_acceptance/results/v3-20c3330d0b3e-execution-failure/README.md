# Qwen3.5 acceptance experiment: v3 terminal record

- Experiment: `lecore.qwen35.install.20c3330d0b3e.59f551d452cf.a6c9d1135a04.f399cd6bc7f2.v3.acceptance`
- Terminal state: `execution_failure` — neither `GO` nor `NO-GO`
- Scientific result admitted by ilxyr: no
- AWS instance: `i-0f57dba006efd2f32`
- Formal execution: 2026-08-13 18:52:51.758Z–21:21:28.229Z
  (`2h 28m 36.471s`), executor exited `0`, metric ingestion failed
- AWS lifecycle: launched 18:50:56Z and self-terminated after evidence upload;
  estimated on-demand compute charge `$2.66` plus a few cents of storage
- Ledger verification: valid (`10` objects and `11` events)
- Evidence bundle SHA-256:
  `84db563708d77fb13ff93f072dec85f19e96ec7cfe33a22e56e639df78dfcc60`

## What happened

The AWS preflight passed every frozen source, model, corpus, version, timeout,
token, interpreter, tokenizer, and disabled-spectral-path check. ilxyr admitted
the project after all twelve admission gates passed. The runner then completed
the full 4,096-position paired evaluation and emitted a metric object, but four
human-readable Qwen diagnostics preceded that JSON on standard output. ilxyr's
strict parser therefore rejected the stream with:

```text
executor output is not valid metric JSON: expected value at line 1 column 1
```

No `EvidenceRecorded` event was appended, `latest_evidence` remains null, and
the diagnostic metric artifact must not be relabeled as an admitted result.

## Diagnostic observations

The unpromoted runner artifact is still operationally useful:

- all `4,096` required paired positions were evaluated;
- reference-logit relative error was `6.7877e-7` and tokenizer parity passed;
- the emitted 2,112.9 MB checkpoint reloaded in leCore and official
  Transformers;
- original perplexity was `7.9062`, installed perplexity was `10.3576`, a
  `31.0053%` regression;
- the paired moving-block interval in loss space was
  `[0.20260, 0.34247]` nats, wholly on the harmful side of the frozen gate;
- peak resident memory was `24,306.5` MB (23.74 GiB);
- text generation and vision smoke did not pass. The official processor smoke
  reported that Torchvision was absent from the environment.

These observations are not a formal `NO-GO`: the metric envelope was never
admitted. They do, however, make another paid treatment run scientifically
unattractive until the parser and vision dependency blockers are fixed and the
large quality regression is understood with a cheap local or fixture run.

## Cleanup and retry policy

No retry was launched. Instance `i-0f57dba006efd2f32` is terminated and
security group `sg-08e840b4749a47406` was deleted after detachment. The frozen
v3 identity is closed.

## Evidence map

- `project/` contains the exact ilxyr project admitted on AWS.
- `result/` contains preflight, installer, runtime, status, metric diagnostic,
  environment, checksum, and cleanup records.
- `ledger/` is the independently reverified `.ilxyr` event store, renamed for
  Git publication.
- `publication-manifest.json` is the canonical hash-and-size index.
- `publication-receipt.json` records the permanent Arweave transactions and
  verified downloads.

The redistributable corpora and the 1.7 GB public model checkpoint are omitted
from Git. Corpus hashes remain frozen in the manifests. The permanent Arweave
publication includes reconstructable chunks of both corpora and every other
evidence file, but excludes model weights.

The original AWS evidence bundle remains at
`s3://zero-training-022118847419/qwen35-acceptance/f399cd6-2fc06364-a6c9d113/v3-formal/qwen35-evidence.tgz`.


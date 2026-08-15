# Qwen3.5 acceptance experiment: v5 terminal record

- Frozen source: `d5bcfc297a3af8e4f8f0c60b3d8ad21b67e0e55a`
- Experiment: `lecore.qwen35.install.20c3330d0b3e.e4323ead9159.a6c9d1135a04.d5bcfc297a3a.be5d1f431823.v5.acceptance`
- Terminal state: `execution_failure` — neither `GO` nor `NO-GO`
- Formal execution started: yes; no retry was launched
- Scientific result admitted by ilxyr: no
- AWS instance: `i-01ed9a97837d59d00` (`r7i.4xlarge`, `us-east-1`)
- Evidence bundle SHA-256:
  `32be545966b2d1fe9254ac1d100b7c4285adddf911f2e7c428ffbe40741db500`

## What happened

The exact frozen preflight passed and ilxyr admitted the project after all
twelve gates passed. The runner completed the experimental installation, the
full 4,096-position paired evaluation, checkpoint reloads, text generation,
and the official Transformers vision-input smoke test. It emitted a valid JSON
metric object and exited successfully.

Ilxyr then rejected the otherwise valid metric envelope because the runner
included an `emitted_checkpoint` object under `source`, while the frozen
`ilxyr.run.v1` source schema permits only `repository`, `commit`, and
`artifacts`:

```text
executor output is not valid metric JSON: unknown field `emitted_checkpoint`,
expected one of `repository`, `commit`, `artifacts` at line 1 column 2838
```

The run event records executor exit code `0`, but no `EvidenceRecorded` event
was appended, `latest_evidence` remains null, and the top-level runner therefore
closed as `execution_failure`. The diagnostic metrics below must not be
relabeled as an admitted scientific result.

## Diagnostic observations

The unadmitted runner artifact is strongly GO-shaped:

- all `4,096` required paired positions were evaluated;
- original and installed perplexity were both `7.9062244629535465`;
- perplexity delta was `0%`, and the paired 95% block-bootstrap interval was
  exactly `[0, 0]` nats;
- reference-logit relative error was `6.787656774900957e-7`, with tokenizer
  parity passing;
- leCore reload, official Transformers reload, text generation, and the
  official vision-input smoke test all passed;
- spectral filtering remained disabled;
- peak resident memory was `19,611.47` MiB (about `19.15` GiB);
- the parallel 4,096-token evaluation took `515.78` seconds; and
- all four sequential looks completed without early rejection.

These observations cannot resolve the preregistered outcome because ilxyr did
not accept the evidence envelope. The native GDN acceleration metric was `0`;
this did not affect the equality result but should be investigated separately
before claiming the compiled path was exercised.

## Runtime, cost, and cleanup

The installer remained the bottleneck at `7,826.48` seconds (about 2h 10m),
while the paired evaluation took about 8m 36s. The instance existed from
08:25:56Z until terminal evidence at 10:51:32Z, about 2h 25m 36s. At the frozen
on-demand rate, estimated v5 compute is `$2.57`; including the approximately
`$0.27` from the three closed preflights, cumulative spend is approximately
`$2.84`, below the `$10` ceiling.

Instance `i-01ed9a97837d59d00` is terminated. Root volume
`vol-089fdad2568e0b3fe` and security group `sg-0d8e69dff64e4de55` are deleted.
No additional AWS instance was launched.

## Evidence map

- `project/` contains the exact ilxyr project admitted on AWS.
- `result/` contains preflight, benchmark, installer, runtime, status, metric,
  environment, dependency, source, launch, and cleanup records.
- `ledger/` is the independently reverified `.ilxyr` event store, renamed for
  Git publication.
- `evidence/evidence-bundle.json` records reconstruction details for the
  complete AWS evidence archive.
- `publication-manifest.json` and `publication-receipt.json` bind the permanent
  Arweave publication.

The redistributable corpora and public model checkpoint are omitted from Git.
The Arweave publication contains reconstructable chunks of the complete AWS
evidence archive, including the corpora, but excludes model weights.

The original AWS evidence bundle remains at
`s3://zero-training-022118847419/qwen35-acceptance/d5bcfc2-2fc06364-a6c9d113/v5-formal/qwen35-evidence.tgz`.

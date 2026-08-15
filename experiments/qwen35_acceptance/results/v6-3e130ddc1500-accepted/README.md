# Qwen3.5 acceptance experiment: v6 accepted result

- Frozen source: `3e130ddc1500af084eb7a7fe94a5da647b0ff3ef`
- Experiment: `lecore.qwen35.install.20c3330d0b3e.0215736e2060.a6c9d1135a04.3e130ddc1500.ed018efbbb3a.v6.acceptance`
- Terminal state: `accepted` — **GO**
- Formal execution started: yes; no retry was launched
- Scientific result admitted by ilxyr: yes
- AWS instance: `i-0daae54594415b2c7` (`r7i.4xlarge`, `us-east-1`)
- Evidence bundle SHA-256: `546edc88a01842c6d606590af5468699382dd4facd056761b958baacbe7fe872`

## Result

The frozen v6 experiment completed successfully. Ilxyr admitted the project,
executed the preregistered treatment once, recorded one accepted evidence
event, settled both forecasts, and independently verified all 16 ledger events
and 15 content-addressed objects.

All acceptance gates passed:

- all `4,096` required paired positions were evaluated;
- original and installed perplexity were both `7.9062244629535465`;
- perplexity delta was `0%`, and the paired 95% moving-block interval was
  exactly `[0, 0]` nats;
- reference-logit relative error was `6.787656774900957e-7`;
- tokenizer parity and source cleanliness passed;
- the emitted checkpoint reloaded in both the leCore and official
  Transformers paths;
- text generation and the official Transformers vision-input smoke test
  passed;
- spectral filtering remained disabled; and
- the experimental layer-prepending installer was used.

The emitted checkpoint was about `2,112.91` MB. Peak resident memory was
`19,560.75` MiB (about `19.10` GiB), so future runs can be right-sized well
below 128 GiB after allowing for concurrency and operating-system headroom.

## Performance qualification

The installer remained the bottleneck at `8,220.32` seconds (about 2h 17m).
The parallel paired evaluation took `556.70` seconds (about 9m 17s), using a
common 512-token chunk schedule for both checkpoints.

The requested native GDN recurrence did **not** activate. The system compiler
returned a nonzero exit status on the first native build in each process, so
the parity-gated runtime safely and explicitly fell back to NumPy. This does
not weaken the accepted correctness result—the frozen policy allowed a safe
fallback and the metrics record `native_gdn_acceleration_active = 0`—but v6
must not be cited as evidence that the C acceleration path works on AWS.

## Runtime, cost, and cleanup

The instance existed from 14:14:39Z until terminal evidence at 16:47:47Z,
about 2h 33m. At the frozen on-demand rate, estimated v6 compute is about
`$2.70`; including the approximately `$2.84` from prior attempts, cumulative
estimated compute is about `$5.54`, below the `$10` ceiling.

Instance `i-0daae54594415b2c7` is terminated. Root volume
`vol-0d82cb75c3f2aeec7` and security group `sg-027e59a8ca9edd35b` are deleted.
No additional AWS instance was launched.

## Evidence map

- `project/` contains the exact ilxyr project admitted on AWS.
- `result/` contains the preflight, benchmark, installer, runtime, status,
  metric, environment, dependency, source, launch, and cleanup records.
- `ledger/` is the independently verified `.ilxyr` event store, renamed for
  Git publication.
- `evidence/evidence-bundle.json` records reconstruction details for the
  complete AWS evidence archive.
- `publication-manifest.json` and `publication-receipt.json` bind the permanent
  Arweave publication.

The redistributable corpora and public model checkpoint are omitted from Git.
The Arweave publication contains reconstructable chunks of the complete AWS
evidence archive, including the corpora, but excludes model weights.

The original AWS evidence bundle remains at
`s3://zero-training-022118847419/qwen35-acceptance/3e130dd-2fc06364-a6c9d113/v6-formal/qwen35-evidence.tgz`.

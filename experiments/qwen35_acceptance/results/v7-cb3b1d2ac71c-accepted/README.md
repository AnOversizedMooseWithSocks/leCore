# Qwen3.5 acceptance experiment: v7 accepted native qualification

- Frozen source: `cb3b1d2ac71c183bf9307ca7145a2a619ff30c30`
- Experiment: `lecore.qwen35.install.20c3330d0b3e.e4323ead9159.a6c9d1135a04.cb3b1d2ac71c.5a73ec7002a4.v7.acceptance`
- Terminal state: `accepted` — **GO**
- Formal execution started: yes; no retry was launched
- Scientific result admitted by ilxyr: yes
- AWS instance: `i-096225051167ef900` (`r7i.2xlarge`, `us-east-1`)
- Evidence bundle SHA-256: `2871b7fa96685bd7f5d75f79368808d00649f9cf91950f4655e6099266319f8f`

## Result

The frozen v7 experiment completed successfully. Ilxyr admitted the project,
executed the preregistered treatment once, recorded one accepted evidence
event, settled both forecasts, and independently verified all 16 ledger events
and 15 content-addressed objects.

All acceptance gates passed:

- all `4,096` required paired positions were evaluated;
- original and installed perplexity were both `7.906224720549084`;
- perplexity delta was `0%`, and the paired 95% moving-block interval was
  exactly `[0, 0]` nats;
- reference-logit relative error was `6.787656774900957e-7`;
- tokenizer parity and source cleanliness passed;
- the emitted checkpoint reloaded in both leCore and official Transformers;
- text generation and the official Transformers vision-input smoke test passed;
- spectral filtering remained disabled; and
- the experimental layer-prepending installer was used.

## Native C qualification

Unlike v6, v7 required native GDN execution for acceptance. Both isolated
evaluators compiled and used the native C recurrence and completed fresh and
resumed-state parity checks. The installed evaluator recorded 200 native calls
over 82,360 tokens; the original evaluator recorded 180 calls over 74,124
tokens. Neither evaluator recorded a fallback or direct NumPy recurrence call.

The compiled library SHA-256 was
`bafe42afd0ac8aa136b3ef6c68c35dc90fbfdd5c1cbba046100da5e2817bc075`.
The compiler was `/usr/bin/gcc` on Amazon Linux, run with the frozen minimal
execution environment documented in the runtime evidence.

## Runtime, memory, cost, and cleanup

The installer remained the bottleneck at `8,056.60` seconds (about 2h 14m).
The parallel paired evaluation took `525.35` seconds (about 8m 45s), using a
common 512-token schedule. Peak resident memory was `19,503.84` MiB (about
19.05 GiB). Estimated v7 compute was about `$1.32`; cumulative estimated
compute through v7 was about `$6.86`, below the `$10` ceiling.

Instance `i-096225051167ef900` is terminated. Root volume
`vol-087fdcb57d5c79c09` and security group `sg-0a11cb364009ae8d7` are deleted.
No additional AWS instance was launched.

## Evidence map

- `project/` contains the exact ilxyr project admitted on AWS.
- `result/` contains preflight, benchmark, installer, runtime, native parity,
  metrics, environment, dependency, source, launch, and cleanup records.
- `ledger/` is the independently verified `.ilxyr` event store.
- `evidence/evidence-bundle.json` records reconstruction details for the
  complete AWS evidence archive.
- `publication-manifest.json` and `publication-receipt.json` bind the permanent
  Arweave publication.

The public model weights are excluded. The redistributable corpora are carried
inside the reconstructable evidence bundle but omitted as duplicate standalone
files. The original AWS evidence bundle remains at
`s3://zero-training-022118847419/qwen35-acceptance/cb3b1d2-2fc06364-a6c9d113/v7-formal/qwen35-evidence.tgz`.

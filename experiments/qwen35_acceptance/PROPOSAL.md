# RFC: Qwen3.5 layer-prepending acceptance experiment

- Status: draft for leCore committee comment
- Release target: `0.2.11`
- Implementation PR: #31
- Related long-term work: #30 (liblecore ABI-0; not a dependency)

## Decision requested

The committee is asked to review and either approve or revise:

1. the narrow research question and exclusions;
2. the pre-install reference-parity tolerance;
3. the paired statistical gate and minimum evaluation length;
4. the separation and choice of installation and evaluation corpora;
5. the AWS execution envelope and USD 10 spending ceiling; and
6. the rule for keeping or removing the installer's experimental flag; and
7. the permanent ilxyr/Arweave publication contract.

Approval authorizes one frozen real-model execution. It does not authorize
threshold tuning, repeated attempts under the same experiment identity, or a
claim that installed leCore capabilities improve model quality.

## Executive summary

This experiment asks whether leCore can prepend its proposed blank layers to
the public Qwen3.5-0.8B checkpoint and emit an ordinary checkpoint without
materially damaging the behavior that was present before installation.

The treatment is the experimental layer-prepending installer. The control is
the untouched checkpoint. Spectral filtering is disabled. Both checkpoints are
evaluated at the same token positions, and the acceptance decision uses a
paired moving-block bootstrap rather than independent-token or point-estimate
comparisons.

The experiment also checks that the emitted checkpoint reloads and remains
usable through the official Transformers text and image-input interfaces. The
result will be recorded by ilxyr as accepted, rejected, or execution failure
and published even when it is a no-go.

## Research question

> Can leCore prepend its experimental installation layers to
> Qwen3.5-0.8B, within a bounded memory envelope, while preserving reference
> behavior within a preregistered statistical tolerance and retaining official
> text and vision-language execution?

### Primary hypothesis

After pre-install parity is established, the installed checkpoint will:

- remain within a 1% perplexity-regression budget at the upper bound of a 95%
  paired moving-block-bootstrap interval;
- reload from its emitted safetensors files;
- generate text through official Transformers; and
- accept a synthetic image through the official Qwen processor and generate a
  response token.

The hypothesis is conjunctive: every mandatory gate must pass.

## Why this experiment is needed

The Qwen integration contains useful loader, tokenizer, configuration,
text-runtime, and diagnostic work, but the checkpoint-changing paths have not
completed a powered real-model acceptance run.

The earlier spectral experiment is not evidence for promotion. It changed only
18 of 265 eligible tensors, regressed before repair, reverted most changes, and
left an apparent improvement inside an underpowered interval. Spectral
filtering therefore remains a research-only control and is not part of this
experiment.

Miniature fixtures demonstrate structural and portability properties, but they
cannot establish real-checkpoint memory use, official-model parity, or
multimodal compatibility. This proposal closes that specific evidence gap.

## Scope

### In scope

- Qwen loader, tokenizer, and configuration compatibility;
- leCore text-runtime parity before installation;
- peak memory during the experimental installation path;
- preservation of paired token-level language-model loss;
- reload of the emitted checkpoint;
- official Transformers text generation; and
- official Transformers image-input execution.

### Out of scope

- spectral filtering or compression claims;
- improvements to intelligence, task accuracy, or benchmark scores;
- usefulness of installed memory, routing, registers, or other capabilities;
- production throughput, latency, or GPU optimization;
- architectures proposed by PR #30; and
- models other than the frozen Qwen3.5-0.8B subject.

## Frozen subject and controls

- Subject: public `Qwen/Qwen3.5-0.8B`, identified by a manifest of file hashes.
- Control: an untouched materialization of that checkpoint.
- Treatment: `assimilation/install.py --experimental` from one clean, reachable
  leCore commit.
- Spectral control: off for the complete run.
- Random seed: `0`.
- Paired unit: next-token negative log likelihood at an identical token
  position in the original and installed checkpoints.

The generated experiment identity binds the leCore commit and hashed inputs.
Any change to source, model files, corpora, thresholds, or execution contract
requires a new experiment identity.

## Corpus policy

Installation material and evaluation material must be separate files with
separate SHA-256 identities. The installer must not ground itself on the held-
out evaluation text. This avoids evaluating on material used to construct the
installed checkpoint.

Frozen inputs for the authorized run:

- installation grounding is the MIT-licensed root `REFERENCE.md` at leCore
  commit `a04ab5692be38f06120aba4b0bc5e2a284eb2c79`, 2,300,089 bytes with SHA-256
  `d6905f043e7856b93b2dd72dac5fa0dc593898c55d6c54c51f3153f4317d6b7f`;
- held-out evaluation is the complete Project Gutenberg plain-text edition of
  *The Federalist Papers*, ebook 18, retrieved 2026-08-12, 1,213,410 bytes with
  SHA-256 `a6c9d1135a04d10955fe11d210b7f642e1c2341d4f2c8369b9a832cc97839d94`;
- the run uses the first 4,097 tokenizer outputs from that held-out file to
  obtain 4,096 paired loss positions; and
- the complete corpus bytes and license notices are retained in temporary
  staging and the publication bundle because both inputs permit redistribution.

The runner now accepts the two roles as separate arguments and refuses equal
content hashes. Corpus separation is therefore satisfied rather than pending.

## Procedure

1. Freeze a clean, remotely reachable leCore commit, dependency manifest,
   Qwen file manifest, installation-document hash, evaluation-corpus hash, and
   thresholds.
2. Generate and review the ilxyr project. Register its four methodology
   contributions, two independent model forecasts, and funding commitment.
3. Admit the experiment through ilxyr before model installation begins.
4. Compare leCore token IDs with the official tokenizer.
5. Compare pre-install leCore logits with official Transformers logits.
6. Measure the untouched checkpoint on the held-out evaluation positions.
7. Invoke the installer once with its explicit experimental acknowledgement.
8. Record installer output and peak resident and accelerator memory.
9. Reload the emitted checkpoint from disk and require finite logits.
10. Measure the installed checkpoint at the same evaluation positions.
11. Compute the paired moving-block-bootstrap interval over installed-minus-
    original token loss.
12. Generate text and exercise a synthetic image input through official
    Transformers.
13. Let ilxyr resolve the declared outcome, settle forecasts, verify its ledger,
    and export the evidence.

No threshold or corpus may be changed after admission. A revised method becomes
a new experiment rather than a retry of this one.

## Acceptance contract

| Gate | Proposed requirement |
| --- | --- |
| Source | Clean checkout at the frozen, reachable commit |
| Spectral path | Disabled |
| Installer | Explicit experimental path used |
| Tokenizer parity | Exact token-ID equality on the reference prompt |
| Reference logits | Maximum relative error at or below `1e-3` before installation |
| Evaluation size | At least 4,096 paired token positions |
| Statistical method | 95% paired moving-block-bootstrap interval |
| Regression budget | Upper interval bound at or below `ln(1.01)` nats/token |
| Memory | Peak RSS reported; peak GPU allocation reported or zero on CPU |
| Artifact | Emitted safetensors size reported |
| Reload | Emitted checkpoint reloads and produces finite logits |
| Text | Official Transformers generates at least one new text token |
| Vision | Official processor accepts a synthetic image and generates a token |

`acceptance_pass` is true only when every mandatory gate passes. Point
perplexity improvement cannot override a failed confidence-bound gate.

## Declared outcomes

- `accepted`: the run completed and every mandatory gate passed.
- `rejected`: the run completed with a valid metric envelope but one or more
  mandatory gates failed.
- `execution_failure`: the runner timed out, crashed, or failed to emit its
  exact declared metric contract.

All three outcomes are publishable. An execution failure is not silently
reclassified as a model rejection, and a rejection is not discarded because it
is inconvenient.

## Proposed execution envelope

The first formal run should use a disposable AWS instance:

- region: `us-east-1`;
- instance: `r7i.4xlarge` (16 vCPU, 128 GiB RAM);
- storage: 100 GiB encrypted gp3, deleted on termination;
- operating system: Amazon Linux 2023;
- access: AWS Systems Manager with no inbound ports;
- accelerator: none; the acceptance path is CPU-bound and reports GPU use as
  zero;
- experiment timeout: six hours; and
- total spending ceiling: USD 10.

The prelaunch manifest freezes Amazon Linux image
`ami-07a5b367e8dc8bd92`, Qwen revision
`2fc06364715b967f1860aea9cf38778875588b17`, ilxyr commit `e92382f`, the
dependency versions, critical model-file hashes, and an eight-hour instance
lifetime guard. At the current On-Demand price of USD 1.0584/hour, eight compute
hours cost USD 8.4672; the 100 GiB gp3 volume and public IPv4 time leave the
bounded attempt below USD 10.

The larger memory-optimized host is insurance against repeating the known
post-emission memory failure. The ceiling buys one formal attempt, not iterative
tuning. Infrastructure setup, model download, dependency installation, and
artifact export occur outside the six-hour experiment timeout but remain inside
the spending ceiling.

## Evidence and publication

The publication should include:

- the generated ilxyr project and all contribution/forecast/funding objects;
- source, model, corpus, and dependency manifests;
- admission decision and complete run status;
- installer transcript and metric artifact;
- peak-memory and checkpoint-size measurements;
- verified ilxyr workspace summary; and
- native, RO-Crate, and in-toto evidence exports.

Weights and non-redistributable corpus contents will not be republished. Their
public handles and content hashes are sufficient for identity. The evidence PR
will preserve a rejected or failed result with the same prominence as a pass.

### Durable publication bundle

S3 is temporary execution storage, not publication. After `ilxyr verify`
succeeds, the result should be assembled as a content-addressed publication
bundle with this logical layout:

```text
qwen35-acceptance-<experiment-id>/
  publication-manifest.json
  project/
    project.json
    experiment.json
    hypothesis.json
    foundation.json
    engineering-review.json
    experiment-design.json
    forecast-empirical.json
    forecast-mechanistic.json
    funding.json
  result/
    status.json
    environment.json
    model-manifest.json
    corpus-manifest.json
    metrics.json
    install.log
  evidence/
    evidence.native.json
    evidence.ro-crate.json
    evidence.in-toto.json
  ledger/
    events.jsonl
    objects/sha256/<digest>...
```

`publication-manifest.json` is the root object. It records the experiment ID,
resolved outcome, leCore and ilxyr commits, every included relative path,
media type, byte length, and SHA-256 digest. Paths are sorted and the JSON is
canonicalized before hashing. The `ledger/` directory is the verified `.ilxyr`
workspace content, copied without credentials or unrelated experiments.

The bundle is uploaded to Arweave only after the local manifest hashes have
been checked. Publication then produces a separate
`publication-receipt.json` containing:

- the publication-manifest SHA-256;
- the Arweave bundle transaction ID and gateway URL;
- the upload timestamp, tool, and tool version;
- the uploader's declared actor identity;
- a post-upload download-and-hash verification result; and
- optional mirror locations, such as IPFS, without making them authoritative.

The receipt is committed to the ilxyr evidence PR. The Arweave transaction ID
and receipt hash are also linked from the leCore result note. If the receipt is
itself uploaded to Arweave, its second transaction ID is recorded in the PR;
the receipt does not attempt a self-referential hash.

Permanent publication excludes:

- original or emitted model weights;
- non-redistributable corpus contents;
- cloud credentials, environment secrets, package caches, and temporary paths;
- the whole mixed ilxyr workspace when it contains unrelated experiments; and
- raw S3 URLs as evidence identities.

Model/checkpoint and excluded-corpus identity remains reproducible through
public handles, licenses where available, byte sizes, and SHA-256 manifests.
S3 objects may be lifecycle-deleted after Arweave verification and PR review.

The experiment is not considered published merely because ilxyr recorded local
evidence. Publication is complete only when the GitHub evidence PR names a
verified Arweave transaction for the canonical manifest and bundle. The
current ilxyr CLI exports the native, RO-Crate, and in-toto JSON views but has
no Arweave transport; a small side-effecting publication adapter should consume
those exports without modifying the settled ledger.

## Promotion rule

The layer-prepending installer retains `--experimental` unless ilxyr records an
`accepted` outcome satisfying the complete frozen contract. Fixture success,
point-estimate improvement, partial completion, or a successful text-only smoke
test is not sufficient.

An accepted result authorizes removing the basic installation-safety gate in a
separate reviewed change. It does not establish that the installed capabilities
are useful; that requires a later capability-effect experiment.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Runtime implementation is already wrong | Require official tokenizer and logit parity before interpreting installation effects |
| Token losses are serially correlated | Use paired moving blocks and report effective sample size |
| Evaluation leakage | Separate and hash installation and held-out evaluation texts |
| Memory exhaustion | File-backed weights, float32 compute, 128 GiB runner, peak-memory evidence |
| Text preservation hides vision breakage | Require official image-input smoke after reload |
| Fork-only or dirty source cannot be reproduced | Require a clean commit reachable from a recorded remote |
| Experiment succeeds only after tuning | Freeze inputs and thresholds before admission; revisions get new IDs |
| Cloud cost runs away | On-Demand instance with a USD 10 ceiling and automatic termination |

## Questions for committee comment

1. Is `1e-3` the correct maximum relative pre-install logit error, or should
   parity use an additional absolute/percentile criterion?
2. Is a 1% upper confidence-bound regression budget acceptable for `0.2.11`?
3. Should the formal minimum remain 4,096 paired positions or be raised to
   8,192?
4. Which redistributable texts should be frozen as installation material and
   held-out evaluation material?
5. Is one CPU-only AWS attempt under USD 10 sufficient, or is a second seed or
   independent replication required before promotion?
6. Should an accepted result merely preserve `--experimental` with stronger
   documentation, or authorize a separate PR to remove it?
7. Are any additional official Transformers operations required beyond reload,
   text generation, and image-input generation?
8. Should Arweave be the authoritative permanent transport, with GitHub as the
   review/index surface and S3 only as temporary staging? Is an IPFS mirror also
   required?

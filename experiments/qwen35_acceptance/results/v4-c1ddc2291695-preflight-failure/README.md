# Qwen3.5 v4 final preflight failure

- Frozen source: `c1ddc22916957238c00890bec13fd0609b01b521`
- Experiment ID: `lecore.qwen35.install.20c3330d0b3e.e4323ead9159.a6c9d1135a04.c1ddc2291695.5773fd8e17e2.v4.acceptance`
- Terminal state: `preflight_failure`
- Formal execution started: no
- Ilxyr admission: not reached
- Scientific result: none — neither GO nor NO-GO
- Evidence bundle SHA-256: `29d4b28aa0be9896f8993ead3a93a741c831346f64356c35dc7e73bd6f8243de`

## What passed

The exact frozen Linux test gate passed. The complete public model and both
corpora matched their preregistered hashes. The real-model benchmark completed
for 128, 256, and 512-token chunks with identical loss hashes and native C
fresh/resumed-state parity. It selected 512-token chunks at 13.5247 tokens per
second and measured about 4.13 GiB peak RSS. Including the prior v3 full-run
peak, the conservative planning peak was 23.74 GiB and the benchmark found a
32 GiB instance sufficient. Its projected complete-run wall time was 50.05
minutes and projected 128 GiB compute cost was `$0.883`.

All 34 runner-side admission preflight checks passed, including source, model,
corpus, dependency, treatment, benchmark, statistical, timeout, and policy
identity bindings.

## Blocker

The runner initialized ilxyr and recorded four preregistered contribution
objects. The next command, `ilxyr compile`, rejected the generated
`experiment.json` before an experiment could be compiled or admitted:

```text
unknown field `runner_policy`, expected one of `schema`, `id`, `title`,
`hypothesis`, `rationale`, `proposer`, `family`, `shared_task_id`, `lineage`,
`baseline`, `datasets`, `models`, `metrics`, `seeds`, `outcome_contract`,
`execution`, `funding`, `security`, `evidence_authority`, `expected_outputs`
```

The v4 generator duplicated its policy metadata at the top level of
`experiment.json`, outside the frozen ilxyr schema. This is a generator/ilxyr
contract-integration failure, not a model or statistical result.

## Ledger and cleanup

Independent verification with ilxyr commit
`e92382ff2a5e8714466533a160f6609b4ef9cee8` checked 4 objects and 4 events and
returned `valid: true`. Status lookup correctly reports that the compiled
experiment does not exist. There is no `ExperimentCompiled`, `Admitted`,
`RunCompleted`, or `EvidenceRecorded` event.

Instance `i-0f022ea2acd745193` terminated after roughly eight minutes. Encrypted
root volume `vol-0503ebcb0e04f542a` and security group
`sg-075d83f594e2f78ba` were deleted. Estimated compute for this host is below
`$0.15`; all three v4 preflight hosts together are below approximately `$0.27`.
No retry was launched.

## Evidence map

- `project/` contains the exact generated v4 project.
- `result/` contains the benchmark, all preflight checks, launch identity,
  dependency record, source records, terminal status, and complete logs.
- `ledger/` contains the independently verified partial ilxyr workspace.
- `evidence/` records independent verification, the expected failed status
  lookup, and reconstruction metadata for the complete AWS bundle.
- `publication-manifest.json` and `publication-receipt.json` bind the permanent
  Arweave publication.

The 1.7 GB public model weights are not included. The original AWS bundle,
which contains no model weights, remains staged at
`s3://zero-training-022118847419/qwen35-acceptance/c1ddc22-2fc06364-a6c9d113/v4-formal/qwen35-evidence.tgz`.

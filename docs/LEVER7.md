# Lever 7 — the displacement trace

The engine's first six levers are **exact**: they fire only on identical inputs. Lever 7
is the one that amortizes across **similarity** — log what a task did, answer a *new* task
from its neighborhood, and gate the shortcut with a calibrated bound.

## The algebra

Experiences are stored **in superposition**, not in a list:

    trace  =  bundle over i of  bind(task_key_i, displacement_i)
    read   =  unbind(trace, key)         # ONE operation, O(dim log dim), flat in N

The leOS original kept a list of frames and scanned it (k-NN in Python). Storing the
bundle instead means the superposition *performs the neighborhood blend as algebra* — the
recall of a similar key returns a weighted mixture of what similar tasks did, for free.

**The update is the delta rule**, and it is adopted, not invented here:

    S_t  =  S_{t-1} (I - beta k k^T)  +  beta v k^T

`write()` first reads the trace's own prediction for the key and stores only the
**correction** `beta * (observed - predicted)`. That is Widrow-Hoff (1960), the
error-correcting write of fast-weight programmers (Schlag et al. 2021), and exactly the
state update industrialised by DeltaNet / Gated DeltaNet / RWKV-7 — and simultaneously the
displacement codec's P-frame.

**Measured** (cp41, panel bench): overwriting one of three stored pairs reaches the new
value at cos **1.000** while preserving a neighbour at **0.996**; pure Hebbian superposition
smears to **0.721**. The error-correcting term is why the trace stays usable.

## Why it is a *governed* organ

Anyone can write an outer product. What makes lever 7 trustworthy is everything around it:

| Property | What it does | Measured |
|---|---|---|
| Surprise-gated admission | a write must be surprising enough to earn its space | — |
| Calibrated reads | isotonic (UCCI) reflex-error estimate on every hit | bench CORRECT 1.00 |
| Crosstalk pricing | the read is priced by how full the trace is | predicted vs measured gap ≤0.034 (homogeneous) |
| Capacity advisory + self-tiling | at the cliff the trace **splits** (lever 6 supplies the tile size) | cliff shown, not hidden |
| Veto, not delete | `answer_feedback(ok=False)` kills the exact entry in the same breath | contradiction recovery 50/50 |
| Provenance | every answer says `taught` or `model-cached` | cp47 |
| Sessions | key-space salting isolates tenants; the algebra isolates, not a filter | no-bleed suite |
| Receipts | every call carries one; re-running matches it | MCP pins |
| Exact audit floor | the durable text record survives key-format changes (migration by replay) | 923-char roundtrip |

## Where it sits in the field (cp47 research pass)

* **Titans** (arXiv 2501.00663) memorises at test time by *gradient* steps gated by a
  surprise metric with momentum and weight-decay forgetting. Lever 7's decay **is** that
  forget gate in the linear case, but the write is a plain delta-rule outer product —
  one order cheaper and autograd-free.
* **Larimar** (ICML 2024) is the closest relative: a Kanerva distributed associative memory
  with one-shot writes, edits and selective forgetting, whose published cost is "a lossy
  associative read bounded by the memory dimension" — lever 7's capacity cliff, in print.
* **MemoryLLM** independently reports our mixed-regime law: a fixed-size pool degrades once
  history exceeds it.
* The unpublished ground: **persistent writes into a running linear-attention state**. The
  Mamba editing paper (arXiv 2404.03646) edits *slow* projections; nobody has published
  deletion from, or governed writing into, a streaming delta-rule state. Lever 7 has run
  that algebra as a governed organ for 30+ checkpoints.

## Kept negatives (the ones that shaped it)

* **Binary quantization corrupts the geometry.** Gram-matrix distortion is ~380,000x int8's.
  `auto` only picks among decision-safe levels.
* **A key is a bag of content words, and frames alias.** Stopwords aliased first; bigrams
  fixed that; then 200-way *template floods* swamped cleanup at D=2048 and every taught
  member refused. Numeric tokens are now weighted, and the **exact-repeat sidecar** (a
  normalized-text dict in front of the trace, veto-aware and session-salted) serves the
  near-exact case. That change moved the flagship bench 0.71 → **0.75** at CORRECT 1.00.
* **Rehearsing the trace's own reads is self-pollution** (0.767 → 0.730, and it damaged
  fresh memories). `consolidate()` accepts caller-owned ground truth only — which the model
  collapse literature (Shumailov, *Nature* 2024) independently says is the only safe loop.
* **Prediction is regime-dependent.** In the *mixed* regime — a stream background plus
  written facts — `capacity_report`'s prediction is an **upper bound** and said so only
  after cp46; measured recall 0.923 / 0.753 / 0.424 against predictions 0.938 / 0.886 /
  0.852. `verify_recall(pairs)` is the ground-truth path.

## Using it

    m.teach(q, a)                  # establish a fact -> provenance "taught"
    m.ask(q)                       # free rungs first; the model only wakes on a real miss
    m.answer_feedback(q, ok=False) # veto: kills the exact entry in the same breath
    m.session_open("tenant")       # key-space salting
    m.learning_save(partition)     # one container, migration by replay

Inside a model's forward pass the same algebra is `OuroborosResident` (see
`docs/UNICRON_INSTALL.md`): passive by contract, with `external_write/read/delete`,
`capacity_report` (which declares its regime) and `verify_recall`.

## The seventh lever, checkpoints 54-75 (the arc in one page)

What began as a ladder became the whole front of the house. In order of appearance:
**durable veto tombstones** (cp54: vetoes survive restart; replay honors the dead);
**the drift sentinel** (advisory conflict detection on every teach); **the void
explorer** (cp56-58: conjecture -> real-null experiment -> validated -> evidenced,
with the pheromone==delta-rule result proven twice, by literature and by
experiment, corr 1.000000); **the memory mine** (cp59: the saturation estimator --
recall margin moves BEFORE accuracy falls, corr 0.733 vs null 0.023); **the
benchmark demolition** (cp60-61: LongMemEval-protocol 0.222 -> 1.000, no model;
selection worth +14 points in reader mode); **release discipline** (cp62: the
session partition and the shipped bundle are different things -- distilled,
leakage-audited); **the chat front door** (cp63-64: substrate-first, provenance
chips, artifacts from the real raymarcher, interactive void loop with real nulls);
**Unicron audited** (cp65: AlphaEdit null-space protection measured at 1e7x
preservation; the auditor's own probe bug found and fixed; qwen preflight);
**leOS's api learning** (cp66-67: learn/call/find with discoverability cards,
restart-safe, ssrf-bounded hosted); **the above/below sweeps** (cp67-68: 0
unintended exposure gaps; the grounded answer routine pushed down to an engine
floor, closing a recorded risk); **memory portfolios** (cp69: export/import with
conflict flags, earned rungs and vetoes traveling); **source attribution** (cp70-71:
logit-lens crystallization addresses, per-fact early-exit shortcuts, the measured
speed tiers -- 97x bypass, 3.0x at exit-L7); **the expert panel** (cp72-73: seated,
deliberated, four instruments built to their conditions, one drift finding fixed
with incoherent integration); **the solana lab** (cp74: five hypotheses, two real
statistics, zero economic edges, one broken null caught by the PnL, one Lean 4
theorem exported); **discoverability** (cp75: tool_find's dead catalog arm found
and replaced with introspective cards -- 2,230 faculty docstrings became the
discovery index).

The full measured history is `CHANGELOG_lever7.md` (76 checkpoint entries, every
number reproduced by a command in `tools/`).

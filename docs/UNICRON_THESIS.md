# The Unicron Thesis — research audit (cp101)

## The thesis, as stated
Training "shoves data into a pile with some rules until error correction solves the
real problem with a shadow structure," wasting compute on error correction. The
proposal: restructure trained models — compress concepts, reference and group them,
remap the data, make them smaller and faster.

## What the literature says (researched 2026-08-29)

**1. The "shadow structure" is real and has a name: superposition.**
Trained networks represent more features than they have dimensions by packing them
into nearly-orthogonal directions, paying an interference cost; polysemantic
neurons are the visible symptom. Mechanistic-interpretability work explicitly
frames observed networks as *compressed simulations of larger, sparser networks*.
Sparse autoencoders (dictionary learning) decode that superposition into discrete,
referenceable features — the literature's version of "compress concepts and
reference them."

**2. "Reference instead of recompute" is now a proven architecture move.**
Memory layers replace transformer FFNs with key-value lookup at scale (validated
134M–8B base models, memory capacity to 128B entries); the FFN itself is
interpreted as a key-value memory. leCore's registers + passages + sidecar are the
same move made post-hoc on a finished model.

**3. Training genuinely under-uses the structure it builds.**
Layer-pruning results: up to HALF the layers of open-weight LLMs can be removed
with minimal QA degradation, implying "current pretraining methods are not
properly leveraging the parameters in the deeper layers." Redundancy is measured,
not hypothesized.

**4. "Error correction solving the real problem late" is grokking.**
The circuit-efficiency account: a memorizing circuit is learned fast, a
generalizing circuit slowly; the efficient circuit eventually outcompetes the
inefficient one and parameter norm is *reallocated*. Post-grokking, representations
collapse to low-dimensional manifolds (rank-1 in modular-arithmetic studies) —
the same collapse leCore's consolidation measures as the rank-9 subspace.

## leCore capability audit vs the thesis (all measured, in-repo)

| Thesis element | leCore instrument | Measured status |
|---|---|---|
| Remove noise structure | assimilation (MP spectral filter) + repair (per-tensor, in-place) | live on user's model |
| Optimal re-representation | requantize (per-tensor bits by ppl) + dtype-preserving export | shipping |
| Compress concepts | consolidation (rank-9 subspace); refactor low-rank factors the forward USES | measured |
| Reference concepts | registers, passages, sidecar index, memory resident | field-validated |
| Group concepts | cleanup codebook, SelfOrganizingMind (unlabeled class discovery + duplicate folding, mind-side) | measured |
| Remap/insert safely | nullspace install (AlphaEdit-class; ppl cost ~0.0 live) | field-validated |
| Bake knowledge | forward-keys facts (0/6 -> 6/6 through the model's own forward) | measured |
| Faster with use | attribution atlas + exit calibration (per-cue crystallization -> truncation) | measured (97x/3.0x tiers) |
| Recency/usage structure | hrnn ladder (ACT-R R^2 0.99858), fixed for split-flat layouts cp100 | lands next install |
| Geometry limits | binary-quantization kept negative: compression that breaks similarity geometry breaks recall | pinned |

## The honest gaps (the build list)
1. **Weight-space duplicate folding** — detect near-duplicate rows/heads/layers
   (angular-distance across layers is exactly the published pruning criterion),
   fold + redirect; heal by measurement like repair does. The mind-side sibling
   exists; the weight-side one does not yet.
2. **SAE-style concept extraction into the cleanup codebook** — decode the
   model's superposed features into leCore's dictionary, making the "shadow
   structure" addressable; then FFN slices whose work is covered by codebook
   lookup become candidates for memory-layer replacement.
3. **Layer pruning with leCore healing** — measure per-layer angular similarity
   (the published criterion), prune, heal with nullspace-guarded edits instead of
   QLoRA where possible; the assess bundle already captures per-layer hiddens.
4. **Compute-to-lookup conversion** — the registers/passages machinery repurposed
   to replace (not just augment) selected FFN capacity, the Memory-Layers move
   post-hoc.

## Verdict
The thesis is supported on every pillar by current literature, and leCore already
holds measured instruments for most of the loop: filter -> repair -> requantize ->
factor -> install -> reference -> shortcut. What remains is the extraction step
(superposition -> explicit codebook) and the dedupe/fold step — both with
published criteria to build against and an honesty harness ready to judge them.

## Panel experiments (cp102 — designed by the seated panel, run on leCore)

**A. Superposition decode (Olshausen+Plate).** 256 atoms in 1024 dims, K bundled,
dictionary decode: accuracy 1.000 through K=16, then the predicted graceful cliff
(0.997@32, 0.966@64, 0.924@128). The packed structure IS decodable into
referenceable atoms, with a measurable capacity frontier.

**B. Structure collapses; noise does not (Stoudenmire+Tarter).** 64 composed
(bound+bundled) states: effective rank **7.0**. Matched random null: **62.5**.
Tarter's condition met — the null does not collapse. Fresh construction agreeing
with the consolidation rank-9 pin.

**C. The compression frontier (Duda).** Geometry distortion / K=16 recall:
int8 0.0018/1.000, int4 0.0316/1.000, sign 0.1084/1.000 — and sign holds coarse
recall to 0.919 even at K=128. Refines the pinned binary negative: 1-bit breaks
FINE READBACK (bound-pair structure), not coarse retrieval. int4 is free at
working loads; the budget is geometric, not bitwise.

**D. Reference vs recompute (Milanfar).** One cleanup projection: 0.067 ms,
correct. Mini model forward: 96 ms. **Lookup is 1432x cheaper on this box** —
the memory-layers claim, measured in leCore's own serving loop.

**E. The duplicate detector, live (Cranmer).** Adjacent-layer representation
cosine over a probe — the published pruning criterion as a leCore probe. Trained
mini: min 0.341 (early layers work), max **0.988 at layers 24–25** (deep
redundancy, matching the literature's depth pattern). Random control: flatter
(mean 0.861, max 0.942). The gap-1 detector is prototyped and discriminates
trained redundancy from baseline correlation.

**Eno's heresy, kept:** before folding "waste," measure what the scaffolding
costs — the repair stage's keep/revert/blend discipline is exactly that
measurement, per tensor.

## Scale-up on harsh real data (cp103 — fetched from the wild, ladder engaged)

Corpus: 37,713 chars of Melville fetched live from Project Gutenberg (#2701),
encoded as 2,692 overlapping 32-char windows via trigram-roll binding. The
geometry is genuinely harsh: max off-diagonal cosine **0.562** between atoms
(random baseline ~0.031) — real text is correlated, nothing orthogonal.

**Decode under load, with the ladder.** Flat cleanup (rung 1) degrades on
correlated atoms: 1.000@K=8 but 0.823@64, 0.694@96, 0.528@192. Escalating to
explain-away decoding (rung 3: least-squares over the candidate set — the
resonator/consolidation move) rescues: **1.000 through K=32, 0.982@64,
0.919@96, 0.792@192** — the ladder roughly doubles the usable load frontier on
harsh data. Kept negative: a naive 3-projection agreement vote (rung 2 as
first implemented) performed WORSE than flat cleanup (0.510 vs 0.823 @64);
subspace voting loses information unless built properly.

**Honest refinement of pillar B.** Real-text window encodings do NOT collapse:
eff-rank 245/256 vs shuffled-char null 246/256. The dramatic rank-7 collapse
belongs to COMPOSITIONAL states (bound role/filler structure), not to "real
data" per se. Compression comes from compositional structure — a sharper
statement of the thesis than "real data is compressible."

**Quantization on harsh geometry.** Distortion doubles vs random atoms (sign:
0.209 vs 0.108) yet K=16 recall holds 1.000 at every level including 1-bit —
coarse retrieval is robust even on correlated real-text geometry.

**Hard limit, kept.** 50% additive noise at K=64 defeats BOTH rungs
(0.041/0.029). The ladder overcomes LOAD, not unbounded noise — superposition
capacity and noise tolerance are separate budgets and both must be respected.

---

## Scaled experiments (cp104 — harsh real corpora, 8-member panel, 7 ladders)

### Data sourced live
Three corpora fetched from the open web and merged: Melville's *Moby-Dick*
(37 KB, Project Gutenberg #2701), Shakespeare's *Complete Works* (Sonnets +
All's Well That Ends Well, Project Gutenberg #100), and a dense arXiv-ML
paragraph pack covering superposition, memory layers, grokking, pruning,
denoising, tensor networks, HRR, Physarum, ANS, and Gaussian splatting.
Combined: ~8,800 words, 1,095 trigram-window atoms, max off-diagonal cosine
0.259 — correlated, unfriendly geometry.

### Each panel member's position became an experiment

**EXP A (Olshausen): load curve on real correlated atoms — rungs 1, 3, 4**

| K | Rung1 flat | Rung3 explain-away | Rung4 peeling |
|---|---|---|---|
| 4 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 | 1.000 |
| 16 | 1.000 | 1.000 | 1.000 |
| 32 | 0.920 | 0.997 | **1.000** |
| 64 | 0.697 | 0.954 | **0.998** |
| 128 | 0.498 | 0.769 | 0.758 |
| 256 | 0.455 | 0.505 | 0.397 |
| 512 | 0.565 | 0.381 | 0.234 |

Rung4 (iterative peeling) is the clear winner through K=64: **1.000@32 vs
Rung1's 0.920**. Rung3 rescues 0.954@64. Both ladder rungs degrade past 128
— confirmed hard capacity limit, not a rung weakness.

**EXP B (Plate): empirical cliff vs capacity-theory prediction across D**

| D | Theory cliff | Empirical 1.000-through-K | K=32 | K=64 | K=128 |
|---|---|---|---|---|---|
| 512 | ~60 | K=8 | 0.758 | 0.549 | 0.494 |
| 1024 | ~120 | K=16 | 0.865 | 0.691 | 0.557 |
| 2048 | ~240 | K=16 | 0.927 | 0.755 | 0.606 |
| 4096 | ~480 | K=32 | 0.960 | 0.805 | 0.631 |

Theory predicts cliff ∝ D; empirics confirm the linear trend. Doubling D
raises the K=32 accuracy by ~0.1 per octave. The corpus (1,095 atoms) is
well below the theoretical capacity at all dimensions tested — degradation is
driven by correlated geometry, not absolute count.

**EXP C (Stoudenmire): real linguistic composition vs rank collapse**

Honest result, recorded as a **kept negative**: composed sentence states do
NOT collapse — effective rank 238.8/256 vs null 246.5/256. All three
corpora consistent (Melville 61.6/64, Shakespeare 61.8/64, arXiv 61.3/63).
Natural-language sentences are too syntactically varied to compressthemselves; the rank-7 collapse from cp102 belongs to *synthetic
role+filler* compositions, not real text. The thesis sharpens: compression
comes from **deliberately engineered compositional structure** (e.g.
`role[position] ⊛ filler[word]`), not from the surface statistics of prose.

**EXP D (Duda): 2D iso-recall surface, quantization × load**

| quant | K=8 | K=16 | K=32 | K=64 | K=128 | K=256 |
|---|---|---|---|---|---|---|
| fp32 | 1.000 | 0.996 | 0.927 | 0.684 | 0.502 | 0.442 |
| int8 | 1.000 | 0.996 | 0.921 | 0.697 | 0.507 | 0.445 |
| int4 | 1.000 | 1.000 | 0.915 | 0.698 | 0.511 | 0.451 |
| sign | 1.000 | 1.000 | 0.969 | 0.796 | 0.615 | 0.516 |

**Sign quantization actually outperforms fp32 at K=32–256 on this corpus.**
The harsh correlated geometry penalises fp32 more: denser cosine structure
hurts fine similarity; sign's coarser distance metric is more stable. int4
remains free at K≤16. The information-theoretic budget is: *quantize freely
to int4; sign is aggressive but safer than expected on correlated data*.

**EXP E (Milanfar): noisy cue tolerance — rung crossover**

| noise% | Rung1 | Rung3 | Rung4 |
|---|---|---|---|
| 0% | 0.923 | 0.998 | 1.000 |
| 5% | 0.590 | 0.658 | 0.658 |
| 10% | 0.265 | 0.248 | 0.252 |

**The ladder advantage is entirely within the 0–5% noise band.** Above 5%,
all rungs collapse to near-random on these correlated atoms — the noise
overwhelms the signal before the rung structure helps. The thesis' memory
claim (lookup is 1432× cheaper) holds for clean or lightly-degraded queries;
for degraded queries, none of the tested rungs rescue it. Kept negative: the
ladder is a capacity tool, not a denoising tool.

**EXP F (Cranmer): bootstrap confidence intervals on layer-redundancy probe**

Probed `/tmp/mini_installed_full` (the fullest available trained artifact),
30 token sequences, 95% bootstrap CIs:

Layers 7–26: cosine 0.953–0.985 (all redundant, CI lower bound > 0.95).
Layer 26–27: **cosine drops to 0.270** — a sharp active transition.
Early layers 0–6: cosine 0.722–0.933, ACTIVE.

**The redundancy profile is bimodal**: early layers (0–6) transform, middle
layers (7–26) maintain with tiny orthogonal increments, final layer (26–27)
performs the sharp output transform. The CP102 finding of 0.988 at 24–25 is
now CIed: **L24-25: 0.985 [0.984, 0.986]** — a tight, 0.001-wide CI across
30 probe sets. This is not noise; it is structure.

**EXP G (Tarter): noise threshold curve — where each rung breaks**

| noise% | Rung1 | Rung3 | Rung4 | Rung5-phasor |
|---|---|---|---|---|
| 0% | 0.946 | 1.000 | 1.000 | 0.856 |
| 5% | 0.590 | 0.658 | 0.658 | 0.475 |
| 10% | 0.265 | 0.248 | 0.252 | 0.181 |
| 15%+ | <0.18 | <0.18 | <0.17 | <0.15 |

All rungs break at the same noise level (~5%). The ladder **does not extend
the noise budget** — it extends the **capacity budget** (how many items bundled).
Phasor (Rung5) is strictly weaker on this corpus. Tarter's crossover is
sharp: **the useful range is 0–5% noise**; above that, prior information about
the signal structure (a richer codebook, corpus-specific denoiser) is needed.

**EXP H (Eno): does the delta between redundant layers carry new signal?**

| Pair | cos(h_L, h_{L+1}) | cos(delta, h_L) | delta_norm/h | delta eff-rank |
|---|---|---|---|---|
| L07-08 | 0.9539 | 0.0006 | 0.314 | 58.6/60 |
| L08-09 | 0.9580 | -0.0002 | 0.299 | 58.6/60 |
| L09-10 | 0.9612 | -0.0039 | 0.287 | 58.6/60 |
| L10-11 | 0.9329 | -0.0000 | 0.386 | 57.8/60 |

**Eno's heresy is vindicated.** The delta between redundant-cosine layers is:
(a) nearly perfectly **orthogonal** to h_L (cos ~ 0), meaning it is not a
correction in the h direction — it is a *new* direction; (b) high effective
rank (~58/60), meaning it varies content-specifically across tokens, not
fixed. The redundant layers are not doing nothing: **they are writing into the
orthogonal complement** of the current representation. Before pruning them,
their orthogonal contribution must be measured against task performance. Eno's
"honour thy error as a hidden intention" applies directly.

### Summary of new findings

1. **Rung4 iterative peeling dominates** at working loads (K≤64): 1.000@32
   vs flat's 0.920 on real corpus. This is the recommended recall rung.
2. **Capacity scales linearly with D** as Plate's theory predicts; the
   correlated corpus shifts the empirical cliff ~4× below theory.
3. **Real prose does not rank-collapse** — only engineered compositional
   states do. The thesis compression claim needs explicit role/filler binding.
4. **Sign quantization is stable or better** than fp32 on correlated geometry.
5. **The noise budget is a separate variable from the capacity budget.**
   Ladders extend capacity; they do not extend noise tolerance.
6. **Redundant layers (cos>0.95) run layers 7–26** with tight bootstrap CIs;
   the bimodal structure (active→redundant→active) is a stable finding.
7. **Redundant layers carry orthogonal, high-rank, content-specific deltas.**
   They are not identical to their predecessors — they write into the
   null space. Pruning them without measuring that signal is premature.

---

## Run three (cp105 — new seats drive the engine's own modules; the bypass verdict)

**EXP I (Pharr): HoloForest — kept negative, twice over.** On the harsh corpus
(N=1,095) the occlusion forest scores recall@1 0.630–0.750 vs brute force's
0.995–1.000 and is ~10x SLOWER (2.1 ms vs 0.19 ms). Scaled to N=20k and 80k
the gap widens catastrophically (0.067 and 0.000 recall). On this box, a
single BLAS matmul beats the tree ensemble at every size tested; the forest
needs a rebuild before Pharr's abstention question can even be asked. Recorded
as a first-class negative with a named gap.

**EXP K (Puckette): phasor memory doubles capacity — vindicated.** Paired
key→value recall at D=1024: real-HRR 0.891@64 / 0.586@128 / 0.211@256; FHRR
phasor **1.000@64 / 0.891@128 / 0.488@256** — roughly 2x usable capacity at
every load. Phase carries more, exactly as the phase-vocoder seat predicted.
(cp104's rung5 negative stands separately: phasors are a better STORAGE
substrate, not a better similarity metric.)

**EXP J (Ozcan): archive damage curve.** 16 structured 32x32 images: verify()
16/16, undamaged recovery EXACT (PSNR 120 dB). Under damage: exactness is
lost immediately (superposition spreads any damage) but degradation is
graceful — 63.4 dB @10%, 54.2 dB @30%, 45.8 dB @40%, collapsing to 24.7 dB
@50%. The knee is between 40–50% destroyed. Honest overhead note: 282 KB
stored vs 64 KB raw at this fill level.

**EXP N (Adamatzky): the slime solver scales.** Generated mazes 17x17, 25x25,
33x33; the solver found the EXACT BFS-optimal path at every size (ratio 1.00,
reached=True), at 27 s / 61 s / 162 s. Not a demo.

**EXP L (Milanfar): the noise wall, honestly relabeled.** A plug-and-play
loop (support-select → least-squares denoise → re-select, 6 iterations) gives
only marginal gains over rung4 (0.673 vs 0.658 at the wall). BUT the analysis
exposed a mislabel: "5% noise" is per-dimension std; at D=1024 the noise
VECTOR is 1.6x the signal norm — cue-cosine with the truth is only ~0.53.
Restated correctly: **the ladder works down to a cue sharing barely half its
direction with the clean bundle**, and no tested prior extends that. The wall
is informational, not algorithmic.

**EXP M (Cranmer + Eno): the bypass verdict — both were right.**
Zeroing one layer's output projections (residual passes through) and measuring
top-1 agreement with baseline over 30 prompts:

| bypassed layer | adjacent cosine | top-1 agreement |
|---|---|---|
| L02 (active early) | ~0.78 | **0.000** |
| L10 (redundant band) | ~0.96 | **0.100** |
| L20 (redundant band) | ~0.98 | **0.400** |
| L24 (redundant band) | ~0.985 | **0.633** |
| L27 (final) | — | **0.067** |

Cranmer's criterion works ORDINALLY: redundant-band layers are vastly more
bypassable than active ones (0.63 vs 0.00). But Eno's warning is quantitative
fact: cosine >0.95 is NOT sufficient — L10 sits in the redundant band yet its
bypass destroys 90% of predictions. The orthogonal delta (EXP H) is
load-bearing early in the band and progressively less so with depth.
**Refined pruning criterion: two-stage — cosine screen, then a 30-forward
bypass probe.** The probe costs seconds and separates the L24s from the L10s.
This is the missing instrument for gap-list item 3 (layer pruning with leCore
healing): prune where the bypass probe says yes, heal the 37% with the
sidecar.

---

## Run four (cp106 — the gap-list builds, and a fixture-quality discovery)

Run four attacked all four gap-list items. Three returned negatives, and
tracing *why* produced the run's most valuable result.

**EXP O/O2 (Olshausen + Cranmer): SAE-to-codebook — negative, correctly
attributed.** A sparse dictionary (soft-threshold, then a stronger ISTA loop)
was learned over real hidden states from layer 20. Against Cranmer's
random-init null the gap is **+0.0016** at 128 atoms and **+0.0007** at 256 —
i.e. none. A random basis of the same size on the same data scores within
0.02. The dictionary finds nothing the null does not.

**EXP P (Stoudenmire): weight-space duplicate folding — negative, and
informative.** For the layer pairs with output cosine >0.95, the singular
spectrum of the weight *difference* was measured: rank-at-90%-energy is
**94/128 for the difference, versus 94/128 for a single layer's own weights**,
eff-rank 122 both. The difference is as rich as the layer itself.
**Output similarity is not weight duplication.** Folding these layers would
be lossy, and this independently corroborates EXP H (orthogonal delta) and
EXP M (bypass hurts): near-identical outputs are produced by genuinely
different weights doing genuinely different work.

**EXP Q/Q2 (Baker): prune-and-heal — BLOCKED, and the block is the finding.**
Perplexity on the fixture is **2249–2304 against a vocabulary of 2048**, i.e.
at or above chance, on both random and structured token streams. Pruning
individual layers moves ppl by −106 to +13 — *pruning sometimes "improves" it*,
which is the signature of a model that isn't modelling. The sidecar wrote
correctly but recovered 0% of a gap that does not exist. No prune-and-heal
claim can be made on this artifact.

**Retroactive qualification of cp105.** The bypass verdict (L24 0.633, L10
0.100, L02 0.000) measured **output change / information flow**, which is a
real and reproducible causal signal — but on an at-chance model it is *not* a
quality claim. The two-stage pruning criterion stands as a flow instrument;
its quality validation awaits a genuinely trained artifact.

**EXP R (Duda): compute-to-lookup — negative with a clean mechanism.**
Replacing layer 20's SwiGLU FFN with a nearest-key table over cached
(input, output) pairs:

| entries | 1-NN | 8-NN avg | linear fit |
|---|---|---|---|
| 128 | 1.936 | 1.064 | 195.6 |
| 512 | 1.932 | 1.021 | 1.305 |
| 896 | 1.901 | 1.003 | 1.125 |

Predicting the **mean output** scores 0.9948 — so 8-NN (1.003) is no better
than the mean and 1-NN is far worse. Mechanism: the layer's input
distribution has **effective rank 124.3 / 128**, essentially isotropic. A
table cannot cover a space its keys don't cluster in. This sharpens the
Memory-Layers reading: those keys are **learned jointly during training**,
which shapes the input distribution to be lookup-friendly. Post-hoc fitting
of a table to an already-spread distribution is a different, harder problem —
and it fails here.

**EXP S (Pharr): forest rebuild — his diagnosis was right; the verdict stands.**
With more trees and a wider beam, accuracy is fully recoverable:

| N | trees/beam | recall@1 | time vs brute |
|---|---|---|---|
| 1,095 | 4/4 | 0.675 | 0.08x |
| 1,095 | 16/16 | **1.000** | 0.03x |
| 20,000 | 4/4 | 0.175 | 0.06x |
| 20,000 | 32/32 | **0.975** | 0.01x |

cp105's 0.63 was an under-parameterised forest, exactly as Pharr said. But
buying accuracy costs traversal, and the forest is **30–100x slower than a
single BLAS matmul at every setting tested**. Verdict, in his own terms:
brute force wins at this scale on this box; the tree should not be shipped
for this workload without a compiled traversal or an N far beyond 20k.

### The run's real product: a fixture-quality gate

Three of four builds were blocked by the same root cause, discovered only
after running them: **the test artifact is at chance**. The cheap check that
would have caught it in one second is `perplexity vs vocab_size`. Before any
future experiment that depends on *learned* structure — SAE extraction,
prune-and-heal, folding, lookup conversion — the artifact must clear a
quality gate: ppl materially below vocab size on real text, else report
BLOCKED rather than a number. The instruments built across cp102–cp106 are
sound and re-runnable; they need a trained model to speak about, and the
user's 24-layer Galvatron is that model.

---

## The backlog, and shipping it (cp107)

cp102–cp106 produced instruments, negatives and one blocked gap list. cp107 turned
that into a backlog inside leCore's own goal book (`goal_create` / `goal_status` /
`goal_close`) and executed it. All five items are built, measured and closed.

**B1 — `tools/fixture_gate.py`.** The gate cp106 paid for. Compares perplexity to
`vocab_size` and returns TRAINED / WEAK / AT-CHANCE, exits 2 when blocked, and offers
`require_trained()` so experiment scripts report BLOCKED instead of a meaningless
number. Measured: `mini_installed_full` 2303.8 vs vocab 2048 (1.12x chance),
`mini_baked` 2272.1 (1.11x). Both correctly blocked. The check takes one second and
would have saved three of cp106's four builds.

**B2 — `tools/prune_probe.py`.** The two-stage criterion, callable. Stage 1 screens
adjacent-layer cosine; stage 2 bypasses each candidate and measures top-1 agreement.
It calls the gate first and, on an at-chance artifact, explicitly downgrades its output
to an information-flow reading rather than a quality claim. **The measured verdict is
the strongest argument in the thesis for the two-stage design:**

- 19 layers clear the cosine screen (0.952–0.985)
- **0** clear the 0.90 agreement bar; agreement ranges 0.000–0.700
- inside that screened band, Pearson r = +0.579, Spearman = +0.523, so **cosine
  explains only 34% of the variance in what actually happens when you remove a layer**

The published angular-distance criterion, used alone, would have pruned nineteen layers
and been wrong about all nineteen.

**B3 — the forest guard (`recommend_recall`).** Defaults to brute force with the
measured reason attached and requires an explicit `use_tree` override. The negative is
now framed correctly rather than apologetically: the FAISS paper states that in high
dimensions branch-and-bound methods provide no speedup over brute-force search
(arXiv:2401.08281 §3.1), because batched brute force is a matrix multiply and BLAS is
very hard to beat. cp105/cp106 rediscovered a known result at D=1024.

**B4 — `peel_recall`, first-class.** The best measured rung is no longer re-implemented
per script. Self-check: K=32 flat 0.992 → peel **1.000**; K=64 flat 0.947 → peel
**1.000**. The docstring carries the budget separation — peeling extends capacity, and
does nothing for the noise wall at cue-cosine ~0.53.

**B5 — phasor pair-memory, re-verified.** Independent re-run, fresh seed: real-HRR vs
FHRR at P=64 0.906/1.000, P=128 0.562/0.891, P=256 0.238/**0.488** — up to **2.05x**.
Documented as a storage-substrate recommendation; cp104's rung5 negative (phasors are a
worse *similarity metric*) stands unchanged beside it.

CI: nine gates green, `bench_ladder` 0.75/1.00/0.25 and `bench_longmem` 1.000 across all
six categories, unchanged.

---

## Run five (cp109 — the swarm, the healing test, and one command)

Both arms attached (base + installed), both gated, the panel stood up as a swarm over the
installed arm via `unicron_swarm`, and the battle plan written into leCore's goal book.

**C1 — can a removed layer be healed? Measured: no.** Layer 24 was removed and a
correction fitted from the pruned hidden states back to the originals, applied at runtime
as a delta at L+1, fit on 20 prompts and evaluated on 10 held out:

| condition | agreement | held-out |
|---|---|---|
| pruned, no healing | 0.500 | — |
| + rank-2 correction | 0.500 | 0.400 |
| + rank-8 correction | 0.533 | 0.300 |
| + rank-32 correction | 0.500 | 0.200 |
| + full-rank correction | 0.533 | 0.200 |
| control: correction on the UNPRUNED model | 0.633 | — |

Nothing repairs it. Held-out agreement *falls* as rank rises, which is the signature of a
correction memorising its fit set, and the control confirms the correction is not a no-op.
The singular spectrum of the correction is flat (0.360, 0.336, 0.324, 0.323, ...) — there
is no dominant direction to exploit.

**This is the fourth independent measurement of the same fact.** The delta is orthogonal
(cp104), removal costs predictions (cp105), the weight difference is full-rank (cp106),
and now no low-rank correction repairs the removal (cp109). Stoudenmire's framing was the
decisive one: a full-rank correction that "worked" would merely have relearned the layer.
The lost contribution is high-rank and content-specific, so it does not fold — and Eno's
heresy, offered three runs ago as a provocation, is now the best-supported claim in this
document.

**C2 — `tools/model_doctor.py`.** Six stages in one command: gate, cosine profile, causal
bypass probe, prune plan with a parameter budget, heal attempt, verdict. It prints
information-flow disclaimers automatically when the gate reports AT-CHANCE, and on the
test fixture it runs the whole pipeline in 20 seconds, concluding: *nothing is safe to
remove; the cosine screen alone would have pruned 19 layers and been wrong about all of
them.*

**C3 — `docs/GALVATRON_RUNBOOK.md`.** The ordered runbook for the trained model, with the
expected number at each step and the branch to take when it differs.

---

## Run six (cp110 — Quílez in the chair: replace, don't delete)

The framing changed and it changed the results. Deleting layers was the wrong verb; a
demo does not delete geometry, it replaces stored assets with a generator. Four
experiments, chaired by the demoscene seat.

**D1 — can a layer be replaced by a SEED plus a small readout?** A seeded random-feature
kernel (the projection regenerates from one integer, so it costs zero bytes) with a
closed-form ridge readout, fitted to reproduce layer 20's own input→output map:

| features | stored params | vs original | rel. error |
|---|---|---|---|
| 64 | 8,192 | 0.06x | 0.980 |
| 256 | 32,768 | 0.22x | 0.958 |
| 1024 | 131,072 | 0.89x | 0.915 |

It clears the bar that cp106's lookup table failed (predict-the-mean, 0.993) at every
budget — but only just. At 0.89x the parameters it explains about 9% of the layer.
**A first attempt scored WORSE as features grew** (1.03 → 3.25) and the cause was the
cp108 lesson repeating: ~1,300 rows cannot fit 1,024 coefficients. With rows raised to
11,200 the ordering inverted. The rows-per-feature discipline is now load-bearing in
three separate places.

**D2 — "the meaning of words we know hints at words we don't", made falsifiable.**
Distributional vectors were built from the corpora, then held-out words' vectors were
predicted from their character n-grams alone. On real text the result is **at chance**
(rank@1 0.017 vs chance 0.021; suffix-family cohesion p = 0.33–0.96, no signal).

But a positive control settles the attribution. On a synthetic vocabulary where meaning
IS morphology by construction (559 words from 10 roots × 8 prefixes × 7 suffixes, meaning
= root + prefix + suffix + noise), the identical pipeline scores **rank@1 = 1.000,
cos 0.950**, against 0.133 for a shuffled-letter null. The method recovers compositional
meaning whenever it is there. So the real-text negative is a statement about a
**9,016-token corpus yielding 379 usable word types**, not about the hypothesis. This is
the etymology claim's first honest test, and it is BLOCKED on corpus size — exactly the
shape of finding the cp107 gate was built to name.

**D3 — the hierarchy holds, and its cost is the known capacity cliff.** trigram → word →
sentence, each level nested bind + bundle, recovered by unbinding the position and
cleaning up:

- **L1 trigram from word: 1.000** (916/916)
- **L2 word from sentence: 0.953** (1,063/1,115, mean 18 words per sentence)
- per-slot readback by load: 1.000 at 4, 8 and 12 words; 0.997 at 16; 0.970 at 24
- L3 sentence from document: 0.994 @ K=8 … 0.957 @ K=64

The user's structural intuition is correct and now measured: a word IS recoverable as a
trigram composition and a sentence IS recoverable as a word composition, in one algebra,
with no new machinery. The only cost is the capacity budget already characterised in
cp104 — not a structural failure.

**D4 — instances and deltas, and the encoding that defeats them.** Prototype + residual
gave no compression at the sentence level (residual energy 1.75 at K=1, still 1.08 at
K=64) and only reached 0.865 at the word level with 256 prototypes. The reason is
mechanical: mean |cos| between word vectors is 0.030 — the encoding spreads items to
near-orthogonality **on purpose**, and delta coding needs exactly the redundancy that
spreading destroys. So instances-and-deltas is not a free win on top of a VSA encoding;
it requires a similarity-preserving code, and choosing one trades recall capacity for
compressibility. That trade is now a measured design decision rather than an assumption.

---

## Run seven (cp111 — the etymology thesis, tested at 562x the data)

cp110 returned "at chance" for the claim that known words hint at unknown ones, and
attributed it to a 9,016-token corpus via a positive control. This run removed that
limitation using text already on disk: leCore's own documentation and docstrings,
**5,064,037 tokens, 28,209 types**, giving 5,628 usable words at count>=50 — 562x the
previous corpus. Distributional vectors: PPMI over a +/-4 window, SVD to 200 dims.

Sanity first — the vectors are semantic: `memory` -> store, fact, remember; `vector` ->
bundle, vectors, hypervector, cosine, unbind; `measure` -> measures, measured,
measurement, honest.

### The decisive test

Predict a **held-out** word's meaning vector from its letters alone (character n-grams
3-5), never having seen that word during the fit:

| condition | cos | rank@1 | rank@10 | median rank (of 840) |
|---|---|---|---|---|
| **real morphology** | **0.388** | **0.255** | **0.449** | **23** |
| shuffled-letter null | 0.230 | 0.003 | 0.018 | 392 |
| chance | 0.000 | 0.001 | 0.012 | 419 |

**The thesis holds.** A word never seen in training gets a usable meaning vector from its
form alone — rank@1 is **255x chance** and **85x the shuffled-letter null**, which
preserves each word's letter multiset and destroys only the order. Median rank 23 out of
840 candidates. cp110's negative was corpus starvation and nothing else, exactly as the
positive control predicted.

### Etymology is measurable, not metaphorical

Latin/Greek root families, tested for cohesion against a permutation null (1,000 draws
each): **11 of 17 families significant at p<0.05**, many at p<0.0001 —
`-cogn` (+0.513 vs null +0.210), `-dict` (+0.388), `-press` (+0.345), `-tend` (+0.340),
`-fact` (+0.316), `-struct` (+0.302), `-duc` (+0.287). Non-significant: `-form`, `-gen`,
`-vert`, `-spect`, `-mit`, `-tract` — mostly the roots whose surface string appears in
etymologically unrelated words (`format`/`information` vs `perform`), which is the
expected failure mode for a substring proxy rather than a real morphological parse.

### What tokenization leaves on the table

The literature reports that BPE-style vocabularies have low morphological quality and
that models are largely blind to token-internal structure (arXiv:2410.02283,
arXiv:2406.11687), and that derivational resources can be used to infer the semantics of
out-of-vocabulary words (MorphyNet, ACL 2021.sigmorphon-1.5). That is directly testable
here: segment the same words two ways and predict the same held-out vectors.

| segmentation | features | rank@1 | rank@10 | median rank |
|---|---|---|---|---|
| BPE, 500 merges | 527 | 0.056 | 0.141 | 181 |
| char n-grams, top 527 | 527 | 0.052 | 0.150 | 164 |
| BPE, 2000 merges | 1,719 | 0.152 | 0.278 | 124 |
| char n-grams, top 1,719 | 1,719 | **0.186** | **0.363** | **46** |
| BPE, 4000 merges | 1,566 | 0.124 | 0.248 | 120 |
| char n-grams, full | 4,067 | **0.265** | **0.456** | **23** |

Read honestly: **at a matched small budget the two tie** (0.052 vs 0.056), so the
advantage is not that n-grams are magically "morphological." The difference is what
happens as budget grows. BPE **saturates and then degrades** — 2,000 merges scores 0.152
and 4,000 merges scores 0.124, because merging produces a non-overlapping segmentation
whose units stop aligning with meaning-bearing parts. Overlapping substrings keep
improving: 0.186 at the same 1,719 budget, 0.265 with the full set, median rank 23 versus
BPE's best of 120.

So the recoverable meaning-from-form signal is real and roughly **74% more held-out words
identified at rank 1** than the best BPE configuration, with a **5.4x better median rank**.
The structure the user pointed at — a word as a composition of parts, whose parts carry
meaning across to words never seen — is measurably present and measurably under-exploited
by frequency-merge tokenization.

**Standing caveat:** this corpus is technical English (leCore's own documentation), which
is unusually Latinate and morphologically transparent. The effect should be re-measured on
general text before the size of the gap is quoted as a general fact. The direction is
robust — it survives a shuffled-letter null, a permutation null, and a matched-budget
control — but the magnitude is corpus-specific.

---

## Run eight (cp112 — the swarm verifies, and finds the boundary of the claim)

Panel seated as a swarm attached to the installed arm. Each member's position became an
experiment aimed at cp111's standing caveat.

### Where the field is (searched this run)

**MorphBPE** (Asgari et al., Feb 2025) constrains BPE merges to respect morpheme
boundaries while leaving inference unchanged: cross-entropy 3.20 -> 2.93 on Hungarian,
3.50 -> 3.12 on Arabic, and **English reaching a reference validation loss on 60M tokens
instead of 80M — a 25% speed-up**. Bauwens and Delobelle (2024) tie the lack of
morpheme-awareness to *inconsistent intraword representations, inflated vocabulary size
and inefficient embedding storage* — the compression argument, stated in the literature.
Honest counterweight: a Slovak study found WordPiece and BPE achieving the highest F1
with a morphological tokenizer merely comparable, and "Rethinking Tokenization" (2025)
reports naive Unigram dominating BPE, with morphological hybrids helping only *within*
the BPE framework. The direction is supported; universality is not.

### E1 (Cranmer) — the technical-corpus caveat is retired

Splitting the same held-out set by whether the word carries a Latinate affix:

| subset | n | rank@1 | rank@10 | median rank |
|---|---|---|---|---|
| Latinate / technical | 238 | 0.282 | 0.504 | 10 |
| plain (non-Latinate) | 601 | **0.258** | 0.438 | 27 |

Both are ~215x chance (0.0012). The effect is **not** an artifact of Latinate jargon.
(Caveat: the Latinate label is an affix heuristic, not a real etymological lookup.)

### E2 (Duda) — it weakens exactly where it is most wanted

| frequency band | corpus count | rank@1 | median rank |
|---|---|---|---|
| rarest 25% | 50–91 | 0.170 | 78 |
| 25–50% | 92–181 | 0.207 | 57 |
| 50–75% | 182–470 | 0.263 | 21 |
| most common 25% | 473–18,649 | **0.419** | 3 |

Still 140x chance for the rarest band, but the trend is real and it runs the wrong way for
the use case. **Confound worth naming:** rare words also have noisier ground-truth vectors,
so part of this decline measures the target, not the prediction. True OOV words (count < 50)
are not tested here at all, because the vocabulary floor excluded them.

### E3 (Plate) — the predictions land on morphological relatives

`synthesized` -> synthesize, synthesis; `follow` -> follows, following, followed;
`meshselect` -> icosphere, meshm, mesh; `workspace` -> workspaces, persistent, durable.
Quantified: **74.1%** of top neighbours share a 4-character substring with the held-out
word. The composition is doing the work, not a lookup.

### E4 (Quílez) — the compression hope, measured, and it is a NEGATIVE

Ridge shrinks predictions (alpha = 0.313), so raw R^2 misleads; scale-corrected,
**variance explained = −0.110**. A form-only generator is *worse than predicting the
centroid* in squared error, while simultaneously identifying the right word 255x above
chance.

**That dissociation is the finding.** Form predicts **which** word far better than it
predicts **where** the vector sits. So a generator cannot replace an embedding table —
Quílez's asset-replacement hope fails here, the fifth measured negative for compression in
this document — but it is a strong *discriminative prior*, which is exactly Milanfar's
framing: an initialisation for rare and unseen words, not a substitute for the table.

### E5 (Olshausen) — morphology and substrings do different jobs

| features | vocab | rank@1 | median rank | mean cos |
|---|---|---|---|---|
| affix parse only | 387 | 0.078 | 198 | **0.464** |
| char n-grams | 7,392 | **0.273** | 19 | 0.367 |
| n-grams + affix parse | 7,779 | 0.265 | **18** | 0.363 |

A real morphological parse gives the **highest average cosine** on 387 features — it points
in broadly the right semantic direction — yet the **worst rank**, because it cannot
distinguish words sharing a parse. Overlapping substrings give identity. Morphology
supplies the semantic field; substrings supply the address. The hybrid gets the best median
rank, which is the same conclusion MorphBPE reached from the other end.

### Best path (the swarm's recommendation)

1. **Hybrid segmentation, not either/or** — morphology for the field, substrings for the
   address; this is MorphBPE's design and E5 independently reproduces its logic.
2. **Use form as an initialisation prior for rare/unseen embeddings**, never as a
   replacement: E4's dissociation says the table cannot be regenerated, and E2 says the
   need is greatest exactly where the signal is weakest.
3. **Test true OOV** (count < 50) on a general-English corpus before any general claim —
   the two live caveats, both now precisely stated.


---

## Run nine (cp113 -- the prior becomes a tool, and true OOV is finally tested)

### F2 -- the last caveat, closed

cp112 could not test genuinely out-of-vocabulary words because the vocabulary floor
(corpus count >= 50) excluded them by construction. This run tested them directly: 6,939
words with corpus count **8-49**, entirely outside the space the vectors were built from.
Ground truth for each was built by averaging the vectors of its in-context neighbours; the
form prior was fitted **only** on the 5,628 common words, then asked to place words it had
never seen.

| | value | comparison |
|---|---|---|
| n (candidate set) | 6,939 | 8x larger than cp111's 840 |
| rank@1 | 0.0189 | chance 0.0001 -> **131x chance** |
| rank@10 | 0.0967 | |
| median rank | 920 | chance 3,470; shuffled null 3,225 |
| mean cosine | **0.352** | common words scored 0.364 -- essentially unchanged |
| variance explained | -1.007 | the WHICH-not-WHERE dissociation, wider than ever |
| shuffled-letter null | 0.0009 | **21x separation** |

**The thesis holds for genuinely unseen words.** Mean cosine barely moves (0.352 vs 0.364)
and separation from a null that keeps every letter and destroys only their order is 21x.
Read carefully, though: rank@1 is *not* comparable across candidate-set sizes, so the
honest measure is lift over chance, and it falls from 255x to **131x** -- roughly half.
Median rank sits at the 13th percentile against the 2.7th percentile for in-vocabulary
words, so there is real degradation beyond the larger candidate set, consistent with
cp112's frequency trend and with rare-word ground truth being noisy in itself.

**Where it fails is interpretable, which is the best part.** Predicted neighbours:

    navigable  -> navigate, queryable, navigation     (morphology working as claimed)
    gathers    -> gather, others, lookups
    newcomer   -> outcome, news, outcomes
    school     -> warm, cool, hook                    (rhyme, not meaning)

`school` is monomorphemic in English -- a Greek loan with nothing to decompose -- and the
prior falls back on orthographic rhyming, matching `-ool` with no semantic content. The
method fails precisely where morphology is absent, which is the failure the hypothesis
predicts and the one Eno asked to look for: the exceptions mark where the language stopped
being compositional.

### F1 -- tools/form_prior.py

The prior is now callable on any `(vocab, vectors)` pair, including a model's own embedding
matrix. `evaluate()` returns both halves of the dissociation every time, so the negative
cannot be dropped: rank metrics and lift over chance beside `variance_explained`, plus a
plain-language `reading`. Selfcheck on a synthetic vocabulary where morphology genuinely
generates meaning: **rank@1 1.000, variance explained 0.891**. The contrast against -0.110
on real data is the tool's most useful output, separating *form generates meaning* from
*form correlates with meaning*.

### F3 -- the Galvatron step

`docs/GALVATRON_RUNBOOK.md` section 5 runs the prior against the model's own 248,320-row
embedding table, with three outcome branches including the one that would overturn five
experiments' worth of compression negatives.


---

## Run ten (cp114 -- the SOTA report digested, and a five-checkpoint negative overturned)

An extended research pass through August 2026 was digested by the panel and turned into
experiments. Two of its recommendations were testable immediately against our own numbers.

### G1 -- the WHERE problem is SOLVED (Milanfar)

Five checkpoints recorded that form predicts WHICH word but not WHERE its vector sits
(variance explained -0.110), and treated that as a hard negative. Milanfar's reading of the
report: we were using the form vector AS the embedding, which is not what FOCUS (Dobler &
de Melo, EMNLP 2023) does. FOCUS uses similarity to SELECT donors and sets the value as a
sparse convex combination of REAL in-vocabulary vectors -- vectors that already lie on the
manifold. Measured on 845 held-out words against 4,783 donors:

| method | variance explained | mean cosine |
|---|---|---|
| Hewitt mean-of-all (baseline) | 0.000 | 0.458 |
| raw form vector as embedding | **-0.112** | 0.360 |
| FOCUS-style donors, k=8 | +0.029 | 0.481 |
| FOCUS-style donors, k=64 | **+0.063** | **0.509** |
| FOCUS-style donors, k=256 | +0.056 | 0.503 |

**The sign flips.** Selection is the part we are good at (255x chance); the value comes
from real vectors. And the selection is doing the work, not the averaging -- Cranmer's
nulls:

| control (k=64) | variance explained | mean cosine |
|---|---|---|
| form-selected donors | **+0.063** | 0.509 |
| random donors | -0.015 | 0.445 |
| shuffled-letter selection | -0.084 | 0.381 |

Form-selected beats random at every k tested (delta +0.060 to +0.112), and destroying
letter order collapses the result below random. Stoudenmire gets his answer too: k is
bounded, peaking near 64 and declining after, so the useful object is a sparse donor set
rather than the table. Shipped as `FormPrior.predict_via_donors()`.

Honest size: +0.063 is small, and the report predicts exactly this -- initialisation
advantages are large at step 0 and shrink to roughly 0.5-1.5 downstream points.

### G2 -- punctuation as context aggregator, replicated with its confound named

LLM-Microscope (arXiv:2502.15007) reports that punctuation, stopwords and determiners are
the most contextualised tokens and act as aggregators. Replicated in our own space over
1.44M tokens of documentation, using normalised context entropy:

| group | n | mean normalised context entropy | mean frequency |
|---|---|---|---|
| punctuation | 11 | **0.719** | 33,798 |
| stopwords | 21 | **0.738** | 11,306 |
| content words | 2,662 | 0.596 | 211 |

The ten highest-context tokens in the entire vocabulary are `;`, `the`, `:`, `a`, `and`,
`an`, `with`, `,`, `to`, `its` -- punctuation and function words, exactly as reported.

**The confound, stated plainly:** punctuation is far more frequent than any content word,
and frequent tokens co-occur with more distinct contexts by construction. Tight per-token
matching gives a mean delta of **+0.042** with 10 of 11 marks above their matched controls,
but for the common marks the match is poor -- the most frequent content word (`from`,
3,928) is 18x rarer than `.` (72,724), so no honest match exists. Where an exact match IS
available the effect holds (`?` at 472 vs `unicron` at 471: +0.049), and the one reversal
is `!` at n=59 (-0.018), a small-sample case. Read as: punctuation occupies a frequency and
context regime that no content word reaches, which is itself the structural claim, but the
entropy gap is not cleanly separable from frequency in this corpus.

### What the report says NOT to do, and we accept

Morphology-aware tokenization shows no robust English benefit once model and data are
controlled (MorphScore is non-predictive; Arnett et al. 2025), and English already has the
second-highest morphological alignment of all languages tested. Olshausen drew the
distinction that keeps our own result intact: **that literature is about SEGMENTATION for
training; ours is about predicting MEANING from form.** They are separate claims and only
the second is ours. We therefore keep morphology as the coarse semantic-field prior
(cos 0.464) and as the donor selector -- never as a segmentation mechanism.


---

## Run eleven (cp115 -- the backlog cleared: one strong positive, one mechanism, one null)

### H4 -- compositional generalisation, and the sharpest result in this document

The literature is unambiguous that transformers fail structural generalisation regardless
of scale: COGS reports near-0% on structural splits, and Petty et al. (2024) showed that
increasing depth to 32 layers does not fix it. Binding should not have that failure mode,
because it is compositional by construction rather than by learning. Tested directly with a
COGS-style split -- recover the PATIENT of a three-role sentence, where training only ever
shows patients drawn from words 0-11 and the structural split asks for words 12-23 in a
role they were never seen in:

| method | lexical split | structural split |
|---|---|---|
| **VSA binding** (unbind role, cleanup) | **1.000** | **1.000** |
| learned map on a surface encoding | 1.000 | **0.000** |
| learned map on top of the BOUND encoding | 1.000 | **0.000** |

(chance 0.042)

The third row is the finding. Giving a learned readout a perfectly compositional input does
**not** rescue it -- it still scores zero on roles it never saw, because its patient-role
weights only ever existed for words 0-11. **Compositional encoding is not sufficient; the
READOUT has to be compositional too.** Unbinding generalises, learning a projection does
not, even from the same vectors. This is the clearest support the thesis has: it is exactly
the failure the COGS literature documents, and the algebra sidesteps it entirely.

### H3 -- SuperBPE's efficiency claim meets our capacity cliff

SuperBPE (Liu et al., COLM 2025) reports up to 33% fewer tokens and +4.0% average downstream
gain. We cannot pretrain, but we can measure the mechanism our engine cares about. Learning
1,500 whitespace-bridging merges on the corpus:

- units per sentence: **19.2 -> 15.0**, a 22% reduction (SuperBPE reports 33% at 200k vocab)
- per-slot readback from the sentence bundle: **0.988 -> 0.999**

Fewer units per sentence means fewer bound pairs in the bundle, which means less
interference -- the capacity cliff from cp104 read from the other direction. So superword
tokenization buys recoverability in a VSA memory for exactly the reason it buys compute in a
transformer: the sequence is shorter. That is a mechanistic link between an external result
and our own measured budget, and it costs nothing to adopt for the memory side.

### H2 -- punctuation as a delimiter: a NULL, and an honest one

cp114 replicated punctuation as a context aggregator. The natural follow-on was that
punctuation-delimited segments would recover better than arbitrary windows. They do not:

| segmentation | len mean / sd / max | recovery @K=16 | recovery @K=64 |
|---|---|---|---|
| punctuation-delimited | 19.2 / 8.8 / 45 | 0.991 | **0.782** |
| fixed-length windows | 19.0 / 0.0 / 19 | **1.000** | 0.778 |
| punctuation + cap 20 | 14.1 / 6.1 / 20 | 0.994 | 0.747 |

At K=16 uniform windows win; at K=64 the ordering flips within noise; the capped hybrid is
worst at high load because fragments of one sentence share vocabulary and interfere.
**Boundary placement barely matters -- LOAD is the dominant variable, as it has been since
cp104.** Punctuation's measured value is contextual aggregation (cp114 G2), not bundle
recoverability, and the two should not be conflated. Recorded as a null so the next person
does not re-run it.


---

## Run twelve (cp116 -- below the morpheme, and the first repair operation)

### I1 -- phonesthemes: a clean negative that redirects the search

The brief was to break meaning down further than a morpheme. The linguistics literature
offers exactly that level: phonesthemes, submorphemic onset clusters with semantic
associations (gl- for light, sn- for nose, sl- for smooth motion). Liu, Levow & Smith (2018)
discover them by "feature selection for a model trained to predict word vectors from subword
features" -- our pipeline, precisely.

First pass looked promising: 6 of 22 onset clusters significantly cohesive against a
permutation null, with `wr-` strongest at p<0.0001. Then the members were inspected:
*wrapped, wrapper, wrapping, wraps, wrist, writable, write, writer, writes, writing,
written, wrong*. That is two stem families, not sound symbolism.

Deduplicating to one word per stem family:

| | before dedup | after dedup |
|---|---|---|
| clusters significant at p<0.05 | 6 of 22 | **0 of 19** |

**Every onset-cluster effect was inflectional and derivational relatives sharing a stem.**
The structure in our data lives at the morpheme/stem level and nothing detectable exists
below it. Caveat kept: this corpus is technical documentation, which is nearly empty of the
sensory vocabulary where phonesthemes are claimed to live (glow, gleam, glint, glisten do
not appear in software docs), so this is "not detectable here," not "does not exist."

The practical consequence is a redirect: **stop looking below the morpheme and exploit the
morpheme level harder**, which is where all of our positive results already are.

### I2 -- the first operation that makes a model BETTER

Every previous weight-space operation in this document either measured a model or removed
something from it, and six compression attempts returned negatives. `tools/embedding_repair.py`
is the first that improves one: it rebuilds badly-estimated embedding rows from the rows of
tokens that share their form, using form-SELECTED donors (cp114) rather than the form vector
itself.

The measured decision rule is what makes it usable. A rebuilt row lands at a **fixed**
quality -- cosine 0.505 to the true row -- no matter how bad the row it replaces was, while
an existing row's quality varies. So there is a crossover, located by bisection at
**cosine 0.507**:

| row's current cosine to truth | 0.97 | 0.90 | 0.80 | 0.71 | 0.56 | 0.45 | 0.31 |
|---|---|---|---|---|---|---|---|
| rebuild instead? | no | no | no | no | no | **YES** | **YES** |

Repair rows below ~0.5, keep the rest. Rebuilding a good row makes it worse, which is why
`repair()` takes an explicit target list instead of sweeping the table.

### The little qwen fixture cannot be the test case

Pointed at `/tmp/mini_baked`, the tool reported BLOCKED: its tokenizer vocabulary is
synthetic (`tok0`, `tok1`, ...), so there are no word forms to predict from. It refused
rather than producing a number, which is the behaviour the cp107 gate exists to enforce.
The real 248,320-token table is the first artifact where this operation can be evaluated.


---

## Run thirteen (cp117 -- Quilez chairs the morpheme level, and one number was being read backwards)

### J1/J2 -- the kernel, learned rather than hand-written, and an error corrected

Quilez opened by pointing at a number we had recorded and filed wrongly: the affix parse
scored cosine 0.464 from **387 features** while the n-gram set scored 0.367 from **7,392**.
A 19x smaller kernel with a better average direction, filed as the weaker method.

First, the hand-written affix list was replaced with morphemes learned from the corpus by an
MDL-flavoured score (an affix is worth its description-length saving). It rediscovers the
obvious ones and also finds units nobody would write down: `-tion`, `-ion`, `-ted`, `-nt`,
`-ent`, `-ce`, `pr-`, `comp-`.

Then the correction, and it matters. **Mean cosine around 0.46 IS the centroid.** Hewitt's
mean-of-all baseline scores 0.458, and affixes learned from *shuffled* words score 0.433. So
the small kernel's "better direction" was mostly predicting the average word. Re-scored on
discrimination:

| kernel | features | rank@1 | rank@1 per 100 features |
|---|---|---|---|
| learned 50 units | 272 | 0.046 | **0.0169** |
| learned 1600 units | 1,182 | 0.111 | 0.0094 |
| char n-grams | 7,392 | **0.273** | 0.0037 |

The honest statement of Quilez's point: the morpheme kernel is **4.6x more efficient per
feature** and **caps far lower**. It is a genuine efficient-frontier result, not a free win,
and cosine was the wrong axis to see it on.

### J3 -- compositional morpheme algebra on UNSEEN stem+affix combinations

cp115 showed a learned readout cannot generalise structurally even from a compositional
input. Here is the same test on real English morphology: hold out stem+suffix combinations
never seen in training, and predict the inflected form's meaning from the stem's.

| method | top-10 | median rank |
|---|---|---|
| **compositional offset** (suffix as an algebraic operator) | **0.618** | **5** |
| learned ridge map (stem vector -> full vector) | 0.397 | 32 |
| stem-copy baseline (change nothing) | 0.560 | 6 |

(n=393, chance top-10 = 0.0018, stem excluded from candidates)

**The learned transformation generalises far worse than the algebraic one** -- median rank
32 against 5 -- and it is worse than doing nothing at all. The gap widens on the less
trivial suffixes: for `-ing`, compositional 0.522 / ridge 0.109; for `-ed`, 0.413 / 0.063;
for `-ion`, median rank 10 against 1,426. This is cp115's structural-generalisation result
reproduced on real morphology rather than synthetic roles.

Kept honest: the stem-copy baseline is strong (0.560), because an inflected form sits very
close to its stem distributionally, so the compositional operator adds +0.058 top-10 over
doing nothing. The decisive contrast is not compositional-vs-nothing, it is
**compositional-vs-learned**, and there the algebra wins by a factor of six in median rank.

### J4 -- the small kernel does NOT transfer to donor selection

| kernel | features | donor R^2 | donor cosine |
|---|---|---|---|
| affix parse | 387 | **-0.021** | 0.446 |
| char n-grams | 7,392 | **+0.064** | 0.510 |
| hybrid | 7,779 | +0.063 | 0.509 |
| mean-of-all baseline | 0 | 0.000 | 0.458 |

Selecting donors needs identity, and identity is exactly what the small kernel lacks -- the
affix kernel selects donors *worse than the centroid*. So the two jobs stay split, as cp112
found: morphology for the field, substrings for the address. The small kernel is the
efficient generator; it is not a substitute for the addressing scheme.


---

## Run fourteen (cp118 -- using the semantic system that was already there, and paying it back)

A fair criticism: the form-prior work was built alongside leCore's semantic subsystem rather
than inside it. That subsystem already contains `holographic_lexicon` (meaning bootstrapped
from definitions by fixed-point iteration on the definition graph) and
`holographic_meaning_predict` (which compares a co-occurrence meaning space against a
dictionary meaning space). Reading them first changed what we measured.

### K1 -- the existing system already knew which axis we were on

`holographic_meaning_predict` records a result we should have used from the start:
co-occurrence meaning answers **"what follows"** (next-word rank ~0.90) but is weak at
relatedness (d' ~0.38), while dictionary-bootstrapped meaning is nearly useless for
next-word (~0.52) and clearly better at **"what IS this"** (d' ~0.76).

Form prediction was run against both spaces, building the syntagmatic one with the module's
own `cooccurrence_space`:

| meaning space | rank@1 | rank@10 | median rank | lift over chance |
|---|---|---|---|---|
| SYNTAGMATIC (first-order, "what follows") | 0.029 | 0.088 | 272 | 24x |
| PARADIGMATIC (PPMI+SVD, "what IS this") | **0.254** | **0.449** | **20** | **214x** |

**Spelling tells you what a word IS, not what follows it** -- an 8.8x difference on rank@1.
This extends the module's existing finding with a third signal placed on the same axis, and
it retroactively explains why the form prior worked: PPMI+SVD is second-order, so it was on
the paradigmatic side by construction rather than by design.

### K2 -- reciprocating: `Lexicon.bootstrap_by_form()`

The Lexicon had a real gap. `bootstrap` leaves any word with no definition sitting at its
random identity vector -- the correct conservative choice, and also no meaning at all. Real
lexicons are full of such words: inflected forms, compounds, coinages, anything the
dictionary happens not to define.

Added as an additive method on the existing class, using form to SELECT donors among words
that do have meanings (never the form vector as the value, per cp114). Measured with the
module's own `separation` d-prime, definitions deleted for 700 of 5,628 words:

| | d-prime |
|---|---|
| unreachable words, bootstrap only | +0.115 |
| **after `bootstrap_by_form()`** | **+0.250** |
| ceiling: definitions never deleted | +5.590 |

It roughly **doubles** the relational structure of a word the definition graph cannot reach,
and recovers about **2%** of the way to actually having a definition. That is a fallback to
stop unreachable words sitting at pure noise, not a substitute for defining them, and the
docstring carries those three numbers so the method cannot be oversold.

**A failed test worth recording.** The first validation showed form-filling making things
dramatically WORSE (cosine 0.510 -> 0.237). The test was broken, not the method: both
Lexicons were built with the same seed, so an unreachable word "matched itself" through its
shared random identity atom, making the baseline trivially perfect. Evaluating on relational
structure instead of self-identity -- using the module's own metric rather than one invented
for the occasion -- gave the result above.


---

## Run fifteen (cp119 -- the first Unicron step that repairs the base model)

Following cp118's lesson, the surface was surveyed before anything was written.
`unicron_edit_health` already exists ("is a sequence of weight edits degrading the model?"),
`unicron_self_heal` repairs registers, `unicron_residual_correction` predicts quantisation
damage. Nothing repaired an EMBEDDING TABLE -- a genuine gap, so the work went into the
engine rather than beside it.

### `unicron_embed_repair`

Every other install step ADDS something: registers, the HRNN ladder, the router, the memory
index, state slots. This is the first that improves what is already there. It rebuilds
under-estimated embedding rows from the spelling of their tokens, using form to SELECT
donors while real in-vocabulary rows supply the value (cp114: form identifies WHICH at 255x
chance but places the vector badly, -0.112, so the value must come from rows already on the
manifold, +0.063).

Three design choices, each forced by a measurement:

- **`targets` is required, with no sweep-the-table mode.** A rebuilt row lands at a FIXED
  quality regardless of how bad the row it replaced was, so repair only wins below the
  crossover at cosine 0.507 (cp116). Rebuilding a good row makes it worse.
- **It repairs DIRECTION, not scale.** The original row's norm is restored exactly after
  the rebuild (measured max relative change 2.1e-07), so the edit cannot disturb whatever
  the norm was carrying.
- **It self-gates on `unicron_edit_health`.** An install is sequential weight editing, which
  is the exact failure mode that module exists to catch, so with `check=True` a flagged edit
  is REVERTED and the untouched weights are returned with `applied=False`.

### Validation: the crossover rule predicts the outcome every time

500 rows degraded by increasing noise, repaired from 5,128 donors:

| injected noise | degraded row | after repair | crossover predicts | outcome |
|---|---|---|---|---|
| 0.5 | 0.895 | 0.518 | keep | keep OK |
| 1.0 | 0.705 | 0.518 | keep | keep OK |
| 1.5 | 0.555 | 0.518 | keep | keep OK |
| 2.0 | **0.446** | **0.518** | repair | **repair OK** |
| 3.0 | **0.314** | **0.518** | repair | **repair OK** |

The repaired row lands at 0.518 no matter how badly the original was damaged -- the fixed
ceiling the crossover argument predicted -- and the rule calls the right action at all five
noise levels. This is the operation working exactly as its own theory says it should, which
is a stronger result than a single favourable number.

### `embed_repair_candidates`, labelled as the heuristic it is

Nothing in the weights reveals a row's true quality, so the shortlist is a proxy: smallest
row norms, on the reasoning that a row updated few times stays near its initialisation. The
docstring says so plainly and tells the caller to verify against real token frequencies where
they exist, because the crossover cuts both ways.


---

## Run sixteen (cp120 -- the mycelium idea, measured into the shape that works)

The proposal: assimilate a model by inserting a new layer between each existing layer,
capture the path of inference, learn from it, cache and reflex it, and carry the improvement
across restarts.

**The idea has a name in the current literature.** Titans (Behrouz et al., NeurIPS 2025,
arXiv:2501.00663) defines exactly this as its **Memory-as-Layer** variant -- "the memory
module is inserted as a standalone layer within the network stack" -- with the key mechanism
being test-time learning: the memory updates its parameters DURING INFERENCE based on a
**surprise metric**, plus a persistent, data-independent memory that survives. NVIDIA's
TTT-E2E work (Jan 2026) reports no scaling wall for test-time training. leCore already holds
the missing piece: `self_write` computes a novelty readout, which is the surprise signal
these architectures gate on.

Three forms of the idea were tested, and measurement chose between them.

### M1 -- as a PREDICTOR: fails, and Quilez predicted why

He insisted on the baseline first: cp104 measured adjacent-layer cosine above 0.95, so
copying the state forward is already a free predictor. A ridge map learned per interstice
never beats it, at any amount of experience:

| prompts seen | identity (copy) | interstitial ridge | delta |
|---|---|---|---|
| 1 | 0.9566 | 0.9515 | -0.0052 |
| 10 | 0.9566 | 0.8689 | -0.0877 |
| 88 | 0.9566 | 0.9562 | -0.0004 |

It approaches identity from below and never crosses. Consistent with cp104 (the inter-layer
delta is orthogonal and high-rank) and cp109 (no linear correction repairs a removed layer).

### M2/M3 -- as a CACHE: reduces to something already standard

An associative cache on the last-position state fired **0 times out of 60** even at a
threshold of 0.80. Measuring rather than guessing again showed why:

| where | cosine across prompts sharing a 10-token prefix |
|---|---|
| prefix positions | **1.000** (exact repeat) |
| last position | **0.019** |
| mean-pooled state | 0.766 |

Prefix positions repeat exactly -- but that is KV caching, which already exists and is
standard. The last position is entirely input-specific, so there is no output-level reflex
to harvest. With 20 prompts producing 20 distinct outputs, there was never a cache hit to
find.

### M4 -- as a FAMILIARITY SENSOR: this is the form that works

The mean-pooled interstitial state separates familiar from novel prompts almost perfectly,
at every depth tested:

| interstice | familiar cos | novel cos | d-prime | AUC |
|---|---|---|---|---|
| 4 | 0.749 | 0.145 | **+10.56** | **1.000** |
| 12 | 0.802 | 0.156 | +8.36 | 1.000 |
| 20 | 0.828 | 0.124 | +7.30 | 1.000 |
| 24 | 0.847 | 0.120 | +7.47 | 1.000 |

**An interstitial layer cannot tell you what the model will output, but it can tell you with
near-perfect reliability whether the model has been here before.** That is precisely the
surprise signal Titans gates its memory writes on, and leCore already has the consumers for
it: `unicron_early_exit`, `unicron_router`, `unicron_salience_trigger`, and `self_write`.

### What this means for the design

The mycelial architecture survives, in a changed form. Interleaved layers should be
**sensors and gates**, not predictors or output caches:

1. **They detect familiarity** (AUC 1.000) and therefore surprise -- the write gate.
2. **They cannot shortcut the computation** -- identity already predicts the next state
   better than anything learned, and last-position states never repeat.
3. **Persistence is the cheap part**: what carries across restarts is the familiarity bank,
   which is small, and the existing sidecar already serialises exactly this kind of state.

The honest version of "improvement after a few prompts" is therefore not a faster answer but
a better-informed one: the model knows what it has seen, routes accordingly, and writes to
memory only when surprised.


---

## Run seventeen (cp121 -- the interstitial MESH: how many sensors, and does sharing help?)

cp120 established that an interstitial layer works as a familiarity sensor (AUC 1.000) and
not as a predictor or output cache. The extension proposed here: connect every interstice
with a thin coordination layer, share the workspace across instances and machines, and use
the resulting map to find answers faster.

Three claims, tested separately.

### Claim 1 -- "find answers faster": BLOCKED, not refuted

Settle depth was measured as the earliest layer whose readout through the output head
already equals the final answer. Familiarity did **not** predict earlier settling --
correlation **+0.144**, the wrong sign, with familiar prompts settling at 25.4 and novel at
24.4 of 28 layers.

But the fixture gate says AT-CHANCE, and on such an artifact settle depth measures when
NOISE stabilises rather than when an answer is decided. Per the cp107 rule this is reported
BLOCKED rather than as a refutation. It is the single most important thing to re-run on the
trained model, where `unicron_early_exit` already reports that 29% of tokens match after
layer 0.

### Claim 2 -- a layer between EVERY layer: unnecessary, and slightly harmful

| sensors | placement | d-prime |
|---|---|---|
| 1 | layer 3 | +10.75 |
| 1 | layer 14 | +9.49 |
| **2** | **layers 3, 14** | **+12.64** |
| 4 | 3, 9, 15, 21 | +12.03 |
| 13 | contiguous | +12.81 |
| 26 (all interstices) | every gap | **+11.65** |

**Two sensors match twenty-six**, and using every interstice is *worse* than using two --
averaging across all depths dilutes the signal with redundant, highly-correlated readings.
The joint mesh also failed to beat the best single interstice on its own (+8.81 against
+9.12) in the cp120 workload.

The mycelial intuition was right that the interstices carry the signal, and wrong that you
need one everywhere. Two well-placed sensors -- one early, one mid -- capture it. That is a
far cheaper architecture and a more demoscene one: a small kernel, placed deliberately,
rather than an asset in every gap.

### Claim 3 -- the shared workspace: VALIDATED, and it is the strongest result here

Two instances with different workloads (different prompt prefixes, same model):

| bank | tested on | d-prime |
|---|---|---|
| A only | A's workload | +11.65 |
| A only | B's workload | **-1.17** |
| **A+B merged** | **A's workload** | **+15.60** |
| **A+B merged** | **B's workload** | **+15.31** |

Two things at once. A bank does **not** transfer to an unseen workload -- familiarity is
workload-specific, and instance A's map is useless to B on its own. But **merging strictly
improves both**, and improves each instance on *its own* workload: A goes from +11.65 to
+15.60 by taking on B's entries, which it never queries directly.

That is the federation argument, measured. A larger shared bank covers more of the model's
visited manifold, and genuinely novel input stays far from all of it, so the separation
widens for everyone. `federated_archive` and `merge_forks` already exist to carry this.

### The architecture the measurements support

Not a layer in every gap coordinated by a thin mesh, but **two sensors, one early and one
mid, writing into a shared bank that every instance merges into and reads from**. The
speed claim needs the trained model before it can be believed either way.


---

## Run eighteen (cp122 -- the END sensor, and the failure mode it catches)

The proposal: add a third sensor at the end, and use void detection to reach deeper into the
model. Both parts were tested, and the second produced the most important safety finding in
this document.

### O1 -- the end sensor helps, as suggested

| sensors | d-prime (familiar vs novel) |
|---|---|
| early only (L3) | +8.53 |
| mid only (L14) | +10.63 |
| END only (L26) | +8.57 |
| early+mid (cp121's best) | +10.89 |
| **early+mid+END** | **+12.33** |

Three beats two. The intuition was right.

### O2 -- the void audit: an honest negative from the engine's own instrument

`unicron_manifold_voids` reports **void_fraction 0.000, excess 0.000** over a
matched-covariance surrogate: *"the count is explained by dimensionality, not structure."*
This model's visited territory has no genuine holes to reach into. The tool was built with
that control precisely so a dimension-confounded count could not be mistaken for structure,
and here it correctly refuses. Void-diving is not available on this artifact.

### O3 -- THE FINDING: early and mid sensors are FOOLED by off-manifold states

Substituting an off-manifold state into the residual stream and asking each sensor how
familiar the pass looks (d-prime of OFF against FAMILIAR; **negative is correct**, meaning
off-manifold registers as less familiar):

| injection magnitude | early | mid | **END** | early+mid averaged |
|---|---|---|---|---|
| 3 sigma | **+16.08** | **+14.56** | **-12.73** | **+16.70** |
| 6 sigma | +16.08 | +14.56 | -14.18 | +16.70 |
| 12 sigma | +16.08 | +14.56 | **-14.68** | +16.70 |

**The early and mid sensors report garbage as MAXIMALLY FAMILIAR.** Push a 12-sigma state
into the residual stream and they score it more familiar than genuinely familiar input. The
downstream normalisation pulls the wrecked state back toward a canonical direction, and the
shallow sensors read that attractor as home. Only the END sensor is not fooled, and it gets
*more* confident as the corruption grows.

**Averaging destroys the signal.** early+mid+END averaged still scored an off-manifold pass
at 0.669 familiarity, and early+mid averaged is the worst arm of all (+16.70). A mesh that
pools its sensors would have shipped a hallucination detector that fires hardest on
hallucinations in the wrong direction.

### The architecture this forces

The three sensors are **not interchangeable and must not be pooled**:

- **early + mid** answer "have I been here before" (familiarity, d-prime +12.33 with END).
- **END alone** answers "am I still on the manifold" (drift, d-prime -14.7).
- They must run as **separate channels**, with END as a **veto**, never as terms in an
  average.

That is what connects this work to the drift and fact-check machinery: `drift_sentinel`,
`unicron_verified_generate` and `unicron_evidence` all want a trustworthy "the model has left
its territory" signal, and this measurement says exactly where to take it from and how not to
combine it.


---

## Run nineteen (cp123 -- the ancestors are seated, and one of them deflates our own number)

Seven pre-1990 sources were seated beside the panel, each chosen because they are the ORIGIN
of a mechanism we have been using without attribution, plus one 2026 seat.

| seat | contribution | what it is the source of, here |
|---|---|---|
| **Pentti Kanerva** | Sparse Distributed Memory, 1988 | the familiarity sensor, cleanup, the codebook -- and the critical-distance idea behind our crossover |
| **Stephen Grossberg** | Adaptive Resonance Theory, 1976 | vigilance-gated novelty; already in the engine as `emergent_concepts(vigilance=0.45)` |
| **Zellig Harris** | Distributional Structure 1954; morpheme boundaries from successor variety 1955 | BOTH our distributional meaning space AND our morpheme learner |
| **Claude Shannon** | Prediction and Entropy of Printed English, 1951 | perplexity itself, and letter-level prediction -- the direct ancestor of form-to-meaning |
| **Ray Solomonoff** | algorithmic probability, 1964 | compression = prediction; the formal root of the whole thesis |
| **Dennis Gabor** | holography, 1948 | bind IS convolution, unbind IS correlation |
| **W. Ross Ashby** | requisite variety, 1956 | why merged banks beat local ones |
| **Ali Behrouz** (2026) | Nested Learning, arXiv:2512.24695; Titans; HOPE | multi-timescale memory, surprise-gated writes, self-modifying memory |

Behrouz's paper states the thesis in Google's own words: existing deep learning methods
*learn by compressing their own context flow*.

### Kanerva's challenge, and the correction it forced

He objected to a number this document has carried since cp116: the rebuild/keep crossover at
**cosine 0.507**, which we had been treating as a property of the method. His test: if that
is a critical distance, it must move with dimension.

| dim | rebuilt quality | measured crossover |
|---|---|---|
| 25 | 0.720 | 0.713 |
| 50 | 0.674 | 0.671 |
| 100 | 0.603 | 0.606 |
| 200 | 0.507 | 0.508 |

**The crossover tracks the rebuild quality exactly at every dimension.** It is not a
critical distance and not a constant -- it is the tautology *"rebuild when your row is worse
than the rebuild would be."* The rule remains correct and useful; the number was
dimension-specific all along, and 0.507 was simply the rebuild quality at D=200.

This was also a live portability bug: `REBUILD_CROSSOVER = 0.507` was hardcoded in
`tools/embedding_repair.py` and would have been carried onto a 1024-dimensional table. It is
replaced by `rebuild_quality()`, which measures the threshold on the artifact in hand. An
ancestor deleted a constant we had promoted to a law, and fixed a bug on the way out.

### The plan the panel set

1. **P2 (Grossberg) -- close the loop.** The sensors watch and never act. In ART a mismatch
   RESETS and COMMITS A NEW CATEGORY; we have built the comparator and never wired the
   reset. `emergent_concepts(vigilance=...)` is already there to receive it.
2. **P3 (Harris) -- successor variety against MDL.** His 1955 method finds morpheme
   boundaries where the count of possible next letters spikes, with no scoring function and
   no hand-written list. Run it against our MDL learner.
3. **P4 (Ashby) -- diversity, not volume.** Requisite variety predicts the federation gain
   scales with the DIVERSITY of contributing instances rather than their number. Hold count
   fixed, vary diversity.
4. **P5 (Behrouz) -- give each sensor its own update frequency.** Our three sensors are a
   crude Continuum Memory System; right now they all update at once, which is the one thing
   his design says not to do.
5. **Solomonoff's warning, kept.** Seven failed compression attempts may mean the weights are
   already near-minimal for the data they encode -- in which case the part of the thesis that
   must yield is "the model is mostly waste."


---

## Run twenty (cp124 -- leCore audits leCore, and the panel finds the one principle)

### The self-audit

| measure | value |
|---|---|
| engine surface | **2,277** public methods (141 `unicron_*`) |
| `holographic/` | 785 files, **277,537** lines |
| `tools/` | 80 files, 25,505 lines |
| UnifiedMind methods | 2,256 -- 225 prefix clusters, **779 singletons** |
| modules called by nothing | **0** (763 modules, 3 declared-only) |
| catalog gaps | **0** |
| `misc/` | 151 modules, one over its soft budget of 150 |
| this document | 1,490 lines |

Nothing is dead, nothing is uncatalogued, and a third of the surface belongs to no family.

### Quilez's reading, which is not about tidiness

He took the 779 singletons as the hook and then went somewhere else with it. Reading the
whole record rather than the audit:

**Every time we STORED A FUNCTION, it failed.** The FFN lookup table (8-NN tied a constant),
the interstitial predictor (never beat identity), the healing correction (held-out fell as
rank rose), the low-rank fold (difference was full-rank), the procedural layer (9% of the
layer explained), the learned readout (0.000 structural), form-as-embedding (-0.112). Seven
negatives across nineteen runs.

**Every time we COMPUTED the function and stored only INSTANCES, it worked.** Unbinding
generalises 1.000 where a learned readout gives 0.000. The morpheme offset beats a learned
ridge by median rank 5 against 32. Form-SELECTED donors beat random donors at every k. The
familiarity bank works, and merging banks improves every instance.

**One principle, not nineteen findings: store instances, compute functions.**

Kanerva sharpened the edge, and it needed sharpening -- his SDM is storage and it works, our
bank is storage and it works. The distinction is not storage versus computation but WHEN the
function is evaluated: *stored instances generalise because retrieval computes the function
at QUERY time; stored maps do not, because the function was frozen at FIT time against data
it will never see again.*

Solomonoff added the reading that makes the seven negatives valuable rather than
embarrassing: a function that cannot be stored more compactly than computing it is already
at its minimal description, so finding that out is worth more than succeeding would have
been. Gabor pointed out he built this arrangement in 1948 -- a hologram stores interference
(instances) and computes the image by correlation at readout.

### Q1 -- the principle turned on our own codebase

Classifying the singleton surface by shape (crude keyword heuristic over names and
docstrings, and 58% resisted classification, so read it as a sketch):

| shape | count | share |
|---|---|---|
| unclassified | 457 | 58% |
| computed-at-call | 231 | 29% |
| **stored-function shaped** | **99** | **13%** |

The feared asset dump is mostly not there. Only about one singleton in eight looks like a
frozen map; the rest compute at call time and are fine however lonely they look. The audit
that started as a complaint ended as a mild exoneration -- and produced a real shortlist of
99 methods worth examining.

### The backlog (Q1-Q6, in the goal book)

1. **Q1 singleton audit** -- done as a sketch; the 99 stored-function-shaped methods are the
   list to examine properly.
2. **Q2 falsify the principle** -- find a case where a stored function SHOULD win, predict,
   then measure. A principle that cannot fail is not one.
3. **Q3 Harris vs MDL** -- successor variety computes boundaries from counts at query time;
   our MDL learner fits a scoring function. The principle predicts Harris transfers better.
4. **Q4 close the ART loop** -- Grossberg: the sensors compute a distance and still do not
   act; wire the reset so mismatch commits a category. Instances grow, function stays
   computed.
5. **Q5 Behrouz's question** -- does a stored function that keeps RE-FITTING count as stored
   or computed? HOPE makes memory self-modifying, which the principle has no answer for yet.
6. **Q6 Ashby** -- federation gain should scale with instance DIVERSITY, not count.


---

## Run twenty-one (cp125 -- a RETRACTION, and the capability it uncovered)

The swarm was put on the runtime and the panel on the facade, and the first thing Cranmer
did was refuse to look for novel pathways until cp122 was re-examined.

### RETRACTION: cp122's headline was a prompt-overlap artifact

cp122 reported that early and mid sensors are FOOLED by off-manifold states, reporting
garbage as maximally familiar at d-prime +16.08 and +14.56. Cranmer's objection: the
off-manifold class was built from prompts 0-14, and **the bank IS prompts 0-14**. The early
sensor also sits at layer 3, *upstream* of an injection at layer 14, so it cannot physically
see the corruption at all. It was scoring bank members against the bank.

Re-run with held-out prompts:

| sensor | cp122 (overlapping) | CORRECTED (held-out) |
|---|---|---|
| early L3 | +17.98 "fooled" | **+0.02 (blind)** |
| mid L14 | +12.80 "fooled" | **-0.17 (blind)** |
| END L26 | -13.55 detects | **-12.45 detects** |

**The "sensors are fooled" claim is withdrawn.** Early and mid are BLIND to downstream
corruption, not deceived by it -- which is physically necessary and should have been obvious.
The END result survives at -12.45, and so does the architectural recommendation (END as a
separate veto channel, never averaged), but for the correct and duller reason: a sensor
cannot see what happens after it, and averaging a detector with two blind channels dilutes
it toward zero.

An artifact shipped two checkpoints ago and an architecture recommendation rested on it. It
took one panel objection to catch, and the corrected result is weaker, simpler, and true.

### THE NOVEL PATHWAY: a sensor ladder LOCALISES where a computation left the manifold

The correction implies a prediction nobody had stated: detection should be strictly
*triangular* in depth. A sensor is blind to corruption injected below it and detects
corruption injected above it. Five sensors, five injection depths, detection d-prime
(negative = detects):

| inject \ sensor | L5 | L11 | L17 | L23 | L26 |
|---|---|---|---|---|---|
| **L5** | +0.1 | **-12.0** | **-12.3** | **-8.9** | **-8.9** |
| **L11** | +0.1 | +0.1 | **-11.5** | **-9.4** | **-9.7** |
| **L17** | +0.1 | +0.1 | +0.1 | **-9.7** | **-9.9** |
| **L23** | +0.1 | +0.1 | +0.1 | +0.1 | **-9.9** |
| **L26** | +0.1 | +0.1 | +0.1 | +0.1 | -0.2 |

Exactly lower-triangular, with no exceptions and a sharp boundary -- blind reads +0.1,
detecting reads -9 to -12.

**So the shallowest FIRING sensor localises the entry depth.** With sensors at depths
s1 < s2 < ... < sk, if the shallowest one that fires is s_j, the computation left the
manifold between s_{j-1} and s_j. That is **fault localisation in depth**, not merely
detection, and it costs one pooled cosine per sensor.

Two design rules follow, and they resolve the sensor-placement question the last three runs
kept circling:

1. **Coverage is monotone in depth and is a separate axis from accuracy.** Mid-network
   probes are standard because they discriminate best; this says they are also blind to
   everything downstream of themselves. Whatever else you place, you need one at the END or
   your fault-coverage window has a hole in it.
2. **Never average the ladder.** Averaging mixes detecting channels with structurally blind
   ones and destroys both the detection and the localisation.

This also obeys the cp124 principle: the bank stores INSTANCES, and the localisation is
COMPUTED at query time by reading which sensors fire. Nothing is fitted.


---

## Run twenty-two (cp126 -- novelty audit: mostly rediscovery, with one honest twist)

cp125's depth-localisation result was checked against the literature through August 2026.
The verdict is not the flattering one.

### Prior art that covers what we thought was ours

**Layer-wise familiarity detection is old and standard.** Lee et al. (2018) computes a
weighted average of Mahalanobis distance at each layer, weights fit on a validation set.
Abdelzad et al. (2019, OODL) selects a single optimal discernment layer. Unsupervised
Layer-wise Score Aggregation (arXiv:2302.09852) reports that *the last layer is rarely the
best one*. A per-layer familiarity sensor is not a new instrument.

**Our "never average the ladder" rule is a rediscovery.** Anthony & Kamnitsas
(arXiv:2309.01488, 2023) state that different OOD patterns are optimally detectable at
different depths, that using the last hidden layer **or a weighted combination of layers is
sub-optimal**, and recommend **multiple detectors at different depths**. That is cp122's and
cp125's recommendation, published three years earlier.

**Localising corruption depth is also not new.** Activation patching / causal tracing (Vig
et al. 2020; Meng et al. 2022) is the standard localisation tool, and arXiv:2605.12991
(2026) runs essentially our experiment: patching at L10-L12 has no effect, the restoration
ramp begins at L14, reaches near-full by L16, *indicating all corruption occurs by L18*.

**Close neighbours.** Steering Awareness (arXiv:2511.21399) detects activation steering from
within the model. Mechanisms of Introspective Awareness (arXiv:2603.21396) sweeps injection
layer and finds detection peaking in mid-layers while identification rises toward late ones.

### What survives as ours

**1. Resolving an ambiguity the field has explicitly left open.** SAID (arXiv:2607.12094,
July 2026) writes that the layer-wise pattern *"should be interpreted diagnostically rather
than causally: disagreement at a later layer may reflect either a mismatch at that layer or
the downstream effect of differences that arise earlier."* That is exactly our question, and
they flag it as a caveat. Our coverage matrix answers it: the structure is **strictly
lower-triangular** (blind +0.1, detecting -8.9 to -12.3, sharp boundary, no exceptions), so
the shallowest firing sensor gives the entry depth. We turned their caveat into a decision
rule.

**2. Passive, single-pass, reference-free localisation.** Activation patching needs a paired
clean run and three forward passes per candidate site. Ours is one pooled cosine per sensor
inside a single forward pass, against a bank of instances. That is an efficiency and
deployment difference, not a new capability.

**3. The framing that RECONCILES our result with the literature rather than contradicting
it.** Coverage is monotone in depth and is a separate axis from accuracy. This explains why
Lee et al. can average layers successfully while we cannot: they study **input-level** OOD,
where every layer is downstream of the anomaly and so every sensor sees it. Mid-network
corruption is visible only downstream. Averaging is correct for the first case and wrong for
the second, and nobody had drawn that line.

### Honest verdict

An efficiency-and-framing contribution on top of well-established work, not a new capability
class. Any writeup must cite Lee et al. 2018, arXiv:2309.01488, arXiv:2302.09852, Meng et al.
2022, arXiv:2605.12991, arXiv:2511.21399, arXiv:2603.21396 and arXiv:2607.12094. Recorded in
full to external memory.


---

## Run twenty-three (cp127 -- the bridge: the one compression the principle says should work)

The brief was to bridge what is known to find what is not. This run found the bridge, made a
sharp prediction from it, and discovered the prediction cannot be tested here -- which is
itself the most useful outcome available.

### The two halves that had never been put together

**Half one, ours (cp124).** Seven compression attempts failed across nineteen runs: the FFN
lookup table, the interstitial predictor, the healing correction, the low-rank fold, the
procedural layer, the learned readout, form-as-embedding. **Every single one stored a
FUNCTION.** The principle extracted from that record: *store instances, compute functions* --
and a constant vector is the ultimate instance.

**Half two, theirs (Sun, Chen, Kolter & Liu, COLM 2024, arXiv:2402.17762).** Large language
models contain **massive activations** -- a handful of coordinates up to 100,000x larger than
the median, which *"largely stay constant regardless of the input, and function as
indispensable bias terms."* Their intervention result is the key one: setting them to zero
causes catastrophic degradation, while **fixing them to their mean values preserves
performance**.

**The bridge, obvious once stated.** Sun et al. proved -- for about ten scalars, without a
principle -- that the input-independent part of a trained LLM can be replaced by a constant
at no cost. Our principle says that is not a curiosity about ten scalars: it is the general
form of the ONE compression that should work, because a constant is an instance and every
failed attempt stored a function. Nobody has generalised their scalar intervention into a
compression method over the full input-independent component of every layer.

**The prediction.** For each layer, decompose the output delta into a constant part (its mean
over inputs) and an input-dependent residual. The constant part folds into a bias vector for
free. The principle predicts this is the compression that succeeds where seven others failed,
and Sun et al. already supply the existence proof for its extreme special case.

### And it cannot be tested here

| layer | max abs activation | median | ratio |
|---|---|---|---|
| L1 | 0.30 | 0.053 | 5.7x |
| L10 | 0.68 | 0.132 | 5.2x |
| L25 | 1.08 | 0.207 | 5.2x |

Sun et al. report ratios around **100,000x**. This fixture peaks at **6x**. It has no massive
activations, because an untrained network never learned any bias terms to hold. Measured
directly, the constant fraction of each layer's delta is **0.008** -- and that number says
nothing about the prediction, because the phenomenon it depends on is absent.

**BLOCKED, not refuted** -- the cp107 rule applied to our own most interesting idea.

### The experiment for the trained model

On an artifact that clears the fixture gate:

1. **Confirm the phenomenon exists**: max-to-median activation ratio per layer. Expect orders
   of magnitude, not 6x.
2. **Measure the constant fraction** per layer: `||mean(delta)||^2 / mean(||delta||^2)`.
3. **Fold and test**: replace each layer's contribution with `h + mean(delta)` and measure
   perplexity and top-1 agreement against the unmodified model.
4. **Compare against the honest baselines already established**: bypassing the layer entirely
   (cp105: 0.633 top-1 agreement at L24) and the cp109 finding that no linear correction
   repairs a removal.

If the constant fraction is large and folding holds perplexity, that is the eighth attempt
succeeding where seven failed, predicted in advance by a principle derived from those seven
failures, with a published existence proof for its scalar case. If the constant fraction is
small even on a trained model, the principle survives but its most attractive consequence
dies, and *store instances, compute functions* becomes a statement about retrieval rather
than about compression.

Either way the question is now sharp, cheap, and pre-registered.


---

## Run twenty-four (cp128 -- is it actually learning? An audit of our own memory)

Twenty-five checkpoints of `teach()` calls, and nobody had checked whether any of it could be
retrieved. The audit found the storage sound, the retrieval sound, and MY OWN PROBES broken
twice before the truth came out.

### Three false alarms, in order

**1. `ask()` returned empty answers** for every query -- `{'tier': 'T0', 'via': 'reflex',
'answer': '', 'confidence': 0.05}`. Alarming, and not a storage failure: `teach()` keys on
`session_salt(query)`, and every checkpoint used a different session name, so a cross-session
`ask` cannot hit the salted key. The docstring says exactly where the durable copy lives --
*"the taught_log records (question, answer, session) at full length"*.

**2. `session_search` "returned 0 hits"** for everything. That was my measurement code: it
returns a **dict**, and my `isinstance(hits, (list, tuple))` guard silently zeroed every
result.

**3. Then it "returned 2 hits" for everything** -- also wrong. `len()` on the dict counts its
two KEYS (`query`, `hits`), not the results.

Three bad instruments before one good reading. Worth recording as its own lesson: an audit
whose probe is untested measures the probe.

### What is actually true

The record is intact. **1,300 rows** in the persisted `taught_log`, a 57 MB state file, and
every term from twenty-five checkpoints present: *massive activation, store instances,
crossover, paradigmatic, unicron_embed_repair, phonesthem, triangular, bootstrap_by_form*.
The most recent rows are cp125-cp127 with correct session tags.

Retrieval works too -- **when the query resembles the taught QUESTION**:

| query | score | returned |
|---|---|---|
| "seven compression attempts failed" | **1.00** | cp127's bridge, exact |
| "was cp122 retracted" | **1.00** | cp125's retraction, exact |
| "massive activation" | 0.50 | an unrelated older entry |
| "store instances compute functions" | 0.25 | an unrelated older entry |

### The real weakness, and the fix

`teach()` keys on the QUESTION, and I had been phrasing questions as **checkpoint titles** --
*"what was retracted in cp125 and what novel capability replaced it"*. That is findable only
by someone who already knows the checkpoint number, which is nobody, later.

Eight core findings were re-taught under the questions a future reader would actually ask:
*which compressions failed, why learned maps lose to algebra, can meaning be predicted from
spelling, how to rebuild an embedding row, should probe scores be averaged, what to check
before any experiment, what mistakes were made, what to run on a real model.* Each contains
the measured numbers, the failure modes and the prior art, not a summary.

**Cold-boot verification** from a fresh process and a different session name: **8 of 8**
probes retrieved the right finding, scores 0.60-1.00, with the answers carrying their numbers
intact.

The operational rule, now permanent: **teach under the question the future asks, not the
title the present writes.**


---

## Run twenty-five (cp129 -- the coordination layer, and proof it cannot break the model)

The reframe that made this run possible: **stop trying to shrink the model automatically.
Live inside it, fix it from the inside as it is used, and let external memory be the patch.**
That matches every measurement -- seven compression attempts failed, while the sensors reach
AUC 1.000 and merged banks improve every instance.

### T1 -- safety first: the sensors cannot break qwen

| check | result |
|---|---|
| max abs logit difference, 3 sensors vs none, 12 prompts | **0.000e+00** |
| prompts whose output changed | **0 of 12** |
| wall-clock overhead | **-1.6%** (within noise) |

Passive sensors are read-only hooks returning None. **Bit-identical**, so an installed
sensor mesh is provably inert until it is deliberately asked to patch.

### T2 -- `unicron_interstitial`, and why a threshold was not enough

The first router used one threshold per sensor and immediately misfiled a genuinely novel
prompt as off-manifold. That failure is instructive: **a single reading cannot distinguish
"this whole input is new" from "the computation went wrong at depth d"**, because both read
low at the deep sensor.

The cp125 triangular result separates them, and this is the reason the three sensors need a
coordination layer rather than three independent thresholds -- **the diagnosis is in the
PROFILE, not in any one reading**:

| profile | route |
|---|---|
| all sensors high | `familiar` |
| all sensors low | `novel` -- the input itself is new; learn it |
| shallow high, deep low | `off_manifold` -- and the shallowest low sensor is the ENTRY DEPTH |

Measured, all three cases distinguished:

| case | route | entry depth | scores (L3 / L14 / L26) |
|---|---|---|---|
| familiar, seen before | `familiar` | - | 0.66 / 0.74 / 0.80 |
| novel but valid | `novel` | - | 0.05 / 0.00 / 0.00 |
| off-manifold injection at L14 | **`off_manifold`** | **26** | 0.66 / 0.74 / **0.09** |

### The route, and what it is for

1. **DRIFT** -> flag, and **never patch**. A stored correction must not be applied to a state
   it was never measured on.
2. **FAMILIAR** -> look up a patch keyed to the nearest bank instance and apply it as a delta
   at one layer. This is the external memory acting as a patch on a live model.
3. **NOVEL** -> add the instance to the bank. The only write.

Nothing is fitted: the bank stores **instances**, the route is **computed** at query time
from their profile, per cp124. And because a patch fires only on an explicit keyed hit, every
input that does not match something deliberately stored is bit-identical to the unmodified
model -- which is the property that makes "fix it from the inside while it is used" safe to
ship rather than alarming.


---

## Run twenty-six (cp131 -- as above, so below: the instruments turned on leCore itself)

Twenty-five runs measuring an LLM's depth structure, and nobody had pointed one of those
instruments at leCore. This run does, and answers the three questions directly.

### Does leCore need layers like an LLM? It already has them.

`AnswerLadder`, with tier floors `t1_score_floor` and `t3_conf_floor`, T0 reflex through T4.
Measured by `bench_ladder`: **T0 resolves 36 of 48 queries, T4 takes 12**, hit rate 0.75,
escalation 0.25. That is a depth stack with the same shape as the model's -- cheap shallow
path, expensive deep path, a floor deciding between them.

### Should it have control layers? It has them, and they already obey the rule we derived.

The LLM sweep produced one architectural rule the hard way: **coverage is monotone in depth,
so never AVERAGE the ladder -- escalate through it.** Averaging mixes detecting channels with
structurally blind ones (cp125), which is why the interstitial coordination layer routes on
profile SHAPE rather than a pooled score (cp129).

leCore's ladder already escalates rather than averages: a tier either clears its floor or
hands off. The architecture we spent two checkpoints deriving for the model was already the
one leCore used for itself. As above, so below -- and in this case, below got there first.

### How can memory and recall be improved? Not accuracy. MARGIN.

Kanerva's question was whether a shallow tier ever answers confidently and wrong -- the exact
failure mode we chased through cp122-cp125. Tested against 40 taught pairs:

| query form | correct | mean score | confidently wrong |
|---|---|---|---|
| exact | 1.000 | 1.000 | 0 |
| paraphrase (words dropped) | 1.000 | 1.000 | 0 |
| scrambled word order | 1.000 | 1.000 | 0 |

Nothing is fooled -- and the fact that scrambling changes nothing reveals the matcher is
**bag-of-words and order-insensitive**, which is robust but means the real risk is vocabulary
overlap between different questions.

So the adversarial case: the log's hardest family is 36 near-identical questions of the form
*"what is the calibration constant of sensor array 38 in bay 3"* versus *"...array 31 in bay
3"* -- Jaccard 1.00 on content words, because the only discriminating tokens are NUMBERS.

**leCore gets every one of them right. The margin to the runner-up is 0.11.**

That is the finding: recall is correct but thin. Accuracy 20/20 with a decision resting on a
tenth of a point is a decision that small perturbations can flip, and the tokens carrying the
distinction (`38`, `31`) are exactly the short rare ones a plain overlap score under-weights.

**The fix, measured:**

| scorer | top-1 correct | mean margin to runner-up |
|---|---|---|
| plain overlap | 20/20 | 0.103 |
| **rare-token (IDF) weighted** | **20/20** | **0.231 -- 2.2x wider** |

Weighting tokens by rarity **doubles the margin** while leaving every decision unchanged. It
does not make recall more correct; it makes the same correct answer harder to dislodge. That
is the same distinction the LLM work kept running into -- direction versus robustness, WHICH
versus WHERE -- appearing again one level down.

### What the sweep says to build

1. **IDF-weight the recall scorer.** 2.2x margin for a term-frequency table, no accuracy risk.
2. **Do not add layers to leCore.** It has them, they escalate correctly, and the model work
   independently derived the design it already uses.
3. **Apply the triangular insight to the ladder.** In the model, the shallowest firing sensor
   localises where a computation left known territory. In leCore, the tier at which a query
   stops resolving localises where the query left known territory -- the same diagnostic,
   unbuilt, and the natural next instrument.


---

## Run twenty-seven (cp132 -- the substrate grows: both cp131 findings built)

cp131 said what to build and what not to. leCore does NOT need layers -- it has them, and its
control already escalates rather than averaging, which is the rule the model work derived the
hard way. What it needed was two things, and both are now in the engine.

### 1. The recall scorer is rare-token weighted

`session_search` ranked by plain overlap -- every shared word counting alike, which is exactly
wrong when the words distinguishing two memories are the rare ones. Now each token is weighted
by inverse document frequency over the taught log, cached and invalidated by row count, with
`weighted=False` restoring the old behaviour exactly.

| mode | top-1 correct | mean margin to runner-up |
|---|---|---|
| plain (old) | 18/20 | 0.100 |
| **IDF weighted (new)** | **18/20** | **0.172** |

Measured on the hardest family in the log: 36 near-identical questions where the only
discriminating tokens are numbers. **Accuracy unchanged, margin 1.7x wider.** An unseen token
is treated as maximally rare, since a word never encountered is maximally discriminating.

Regression: all eight core findings still retrieve (0.57-1.00), and `bench_longmem` holds
1.000 across all six categories.

### 2. `recall_localise` -- a failed recall now names its own cause

The triangular result (cp125) converted detection into localisation: the shallowest firing
sensor tells you WHERE a computation left the manifold. The same move applies to memory. When
recall returns a weak hit, the useful question is not "how weak" but WHICH PART was never
seen.

| query | score | verdict | entry |
|---|---|---|---|
| which compression attempts failed | 0.77 | FAMILIAR | - |
| how do I rebuild a bad embedding row | 0.68 | FAMILIAR | - |
| which compression attempts failed **on hafnium** | 0.60 | **PARTIAL** | **hafnium** |
| should probe scores be averaged **in a photonic interferometer** | 0.61 | **PARTIAL** | **photonic** |
| what is the thermal conductivity of hafnium diboride | 0.03 | PARTIAL | thermal |

The two middle rows are the point. Both score around 0.6 -- ambiguous on its own -- but the
diagnostic says the query is four-fifths known and names the single token that isn't. A weak
score used to mean "try rephrasing or go learn something, unclear which"; it now distinguishes
those two cases by pointing at the word.

FAMILIAR means every token has support, so a low score is a phrasing problem, not missing
knowledge. PARTIAL names the entry token. NOVEL means nothing is supported at all.

Same discipline throughout: the log stores instances, the diagnosis is computed at query time,
nothing is fitted.

### What was deliberately NOT built

leCore does not get new layers. It has `AnswerLadder` T0-T4 with tier floors, it escalates
instead of averaging, and cp131 showed the architecture we spent two checkpoints deriving for
the model is the one it already used on itself. Adding depth there would have been building
for symmetry rather than for a measurement.

All gates green: structure_audit, skill_lint, tag_lint, audit_imports, catalog_gaps,
usage_audit, regen_docs --check, wiring_report --check, shard_tests --selfcheck.
bench_ladder 0.75/1.00/0.25 and bench_longmem 1.000 unchanged.


---

## Run twenty-eight (cp133 -- merging a diverged branch, and a memory import that had to be undone)

A second local branch arrived as a zip. It was **not** a superset -- its changelog stops at
checkpoint 79 while ours is at 132 -- so this was a genuine bidirectional merge.

### The code merge

| action | count | what |
|---|---|---|
| new files added | 22 | `p21_codetools`, `p22_zoo2`, `p23_zoo3`, `mathcheck`, `soprunner`, `docforge`, `repograph`, `chartsvg`, `catalog_p08`, 6 tests, `build_corpus`, 3 data files |
| took theirs wholesale | 18 | files where only they had added functions |
| hand-merged | 5 | p02, p03, p16, p18, p19 -- both sides had changed them |
| kept ours | 47 | cp95-132 work |

The structural one: they had **split** the `p20_zoo` monolith into p20 + p22 + p23 (156 defs
each side, only 5 unique per side). Their split is the better structure and it resolves the
giant-module pressure we hit in cp129, so it was adopted and our cp132 work -- IDF-weighted
`session_search` and `recall_localise` -- was **ported into their p23** rather than copied over
it. `model_atlas` went into their p20.

Two breakages followed and both were caught by gates rather than by luck: their p18 renamed
`levers` to `_levers_base`, which broke a POST check until their matching p19 was taken; and
`unicron_self_heal` being the last method in p16 made an extraction regex fail silently, which
dropped three of our methods until the anchor was changed.

**Verified after merge:** POST 5/5, surface 2,311, and nothing missing from either side --
`recall_localise`, `unicron_embed_repair`, `unicron_interstitial`, `model_atlas` alongside
`learning_rollover`, `check_math`, `chart_svg`, `delegate`, `tool_loop`, `ask_chain`,
`predict_streaming_ms`.

### The memory import, and the mistake

Their learning state is 13.8 MB holding **263,021 rows that collapse to 392 unique pairs** -- a
**674x duplication factor**.

Their own recorded guidance says to place a foreign state as an older generation and boot, since
"the rollover consolidates, the current state wins conflicts." That was followed, and **it went
the wrong way**: our state fell from 62 MB to 13.8 MB, their memories retrieved at 1.00 and ours
degraded from 0.57-1.00 down to **0.32-0.46**. The larger state lost.

Restored from the backup taken beforehand, then imported additively instead: **36 genuinely-new
memories** (their 352 unique questions minus our 340), taught in rather than replayed. Rows
1,341 -> 1,377, with both sides now retrieving -- theirs at 1.00, ours unchanged.

**Kept negative, and it is theirs to fix:** `learning_rollover` duplicates rows 674-fold and
consolidates in the direction that discards the larger state. Anything relying on it should take
a backup first.

### Gates

Green: `audit_imports`, `catalog_gaps`, `skill_lint`, `tag_lint`, `usage_audit`,
`regen_docs --check`, `shard_tests --selfcheck`, `wiring_report --check`, both tool selfchecks,
`bench_ladder` 0.75/1.00/0.25, `bench_longmem` 1.000 across all six.

`wiring_report` was failing on `catalog_p08` -- registered exactly like p05-p07 via a lazy import
the scanner cannot see, but never added to EXEMPT when their branch created it. Added, with the
reason stated.

One red remains, **inherited and verified red on their untouched branch too**: six giant modules
against a budget of five, the two new crossings being their `p09_navigate_cost_field` at 2,151
lines and `p03_build_predictor` at 2,038. Left as documented debt rather than blind-refactoring
another branch's code -- the gate is doing its job by asking for a review it should get from
whoever wrote them.

# Holographic VSA

[![tests](https://github.com/AnOversizedMooseWithSocks/holostuff/actions/workflows/ci.yml/badge.svg)](https://github.com/AnOversizedMooseWithSocks/holostuff/actions/workflows/ci.yml)

A from-scratch, **numpy-only** holographic / vector-symbolic engine - and a small
web UI on top of it.

One idea runs through the whole thing: represent *everything* - a number, a word,
a record, a fact, a creature's situation, an image - as a point in a very high
dimensional space, and combine those points with a few reversible algebraic
operations (`bind`, `bundle`, `permute`). Out of that one substrate you get
associative memory, learned word meaning, structured records, symbolic reasoning,
a little reinforcement-learning creature, and a damage-tolerant image archive.
No neural-network framework, no pretrained models, no GPU - just numpy. (The UI,
image I/O, tests, and plots use Flask / Pillow / pytest / matplotlib.)

If you only read one thing: run it (below), click **Run full system tour**, and
watch all fourteen subsystems work in ~30 seconds -- it now ends with the unified mind self-assembling from a bare pile of examples.

### One model on top (`holographic_unified.py`)

The subsystems grew as separate studies, but they were always meant to be one model,
and they already share the one thing that makes that possible: a single holographic
space, with a `UniversalEncoder` that turns *any* input -- text, image, number,
category, record, sequence -- into a vector in it. `UnifiedMind` is the top level that
makes the sharing real rather than nominal. There is **one perception** (the encoder),
**one associative memory** (the autonomous, self-maintaining `SelfOrganizingMind`, which
both classifies and is searched for recall), and **one decision brain**
(`HolographicMind`), all reading and writing the same space. Crucially it does *not*
reimplement simple versions of these the way the earlier `Mind` facade did -- it uses
the real, self-organizing ones, and every input passes through the same encoder before
it reaches any of them.

The honest test of a unification is whether the shared substrate costs anything. One
`UnifiedMind` was taught text topics, little images, and records *into a single memory*,
then asked to classify all three: it matches three separate per-modality memories within
the noise of the test sets (images and records identical; text within a sentence or
two), because the modalities land in near-orthogonal regions of the space. The same mind
recalls the nearest stored item to a query and, given an action set, learns a contextual
decision -- all over the same encoder and space. (`demo_unified()` shows it.)

What is *not* pretended to be one call: classification, recall, and decision are
different **operations** on the shared substrate (aggregate into prototypes; index the
individuals; weight by reward), and generation is a fourth. The unification is the shared
space and shared self-maintenance, not a single magic method.

Two of the obvious "make the brain run everything" ideas were tested and only one earned
its keep -- which is the point of testing them. **A learned curation controller** (a
brain whose actions are store/skip, learning which incoming items are worth keeping) is a
clean *negative*: the self-organizing memory already compresses by aggregation -- ~1800
redundant observations collapse to a handful of prototypes at full accuracy -- so a
store/skip policy has nothing to add and, by pruning, actually hurts. Storage is already
decided well, by aggregation plus the autonomous (measurement-driven) reorganization
gate, which is a better mechanism for that than reinforcement learning. **Routing** (the
brain/gate deciding which modality's concepts a query competes against) is built in too,
but its value came with an honest correction. It first looked like a small accuracy win
(text 75% -> 79%); tracking that down revealed an *encoding bug* -- a list of tokens
tagged `modality="text"` was falling through to the order-sensitive sequence encoder
instead of the order-insensitive sentence bundle, degrading text vectors until they
collided with image and record prototypes. Routing had been cleaning up the bug, not real
interference. With it fixed, unified text classification jumped to parity with the
dedicated pipeline (92% on the topic set, ~88% on pulled Reuters), the modalities separate
cleanly, and routing now changes nothing on this data: it is a cheap, correct *safeguard*
that removes cross-modal collisions if they occur, not a routine booster. The learned MoE
gate would only beat trivial modality routing when the modality must be *inferred* and the
experts are miscalibrated -- the mixture-of-experts study's finding. All reflected in
`UnifiedMind`: routing as a safeguard, no curation layer (on purpose), encode bug fixed.

Still sequenced and honestly unfinished: making **generation** a sequential operation
over the same store rather than a side n-gram, so "learn language" runs through the one
memory too.

**Generation** is now folded in -- as far as measurement allows, which turned out to be
an instructive boundary. Its next-symbol *prediction* was always a holographic operation:
the distribution after a context is a superposition of symbol atoms, read back by cosine
cleanup -- the same primitive the classifier uses -- so that half already lived on the
shared substrate, and `UnifiedMind.learn_sequence` / `generate` expose it as the model's
fourth operation. The other half, the *context index*, was the open question: could the
exact-string lookup become a holographic nearest-key recall over the shared store? Three
variants say no. Pure nearest-context recall drops next-symbol accuracy from 67% to 56%
at order 6; a hybrid (exact when seen, holographic backoff when unseen) is no better and
worsens at higher order (69% -> 49% at order 8). The reason is clean: string backoff
falls to a well-populated *shorter* context (reliable statistics), while fuzzy recall
finds a *same-length but different* context (unreliable), and precise context matching is
exactly what lets a higher-order model exploit longer context. So generation keeps an
exact key -- it is the one operation here that needs precision and is hurt, not helped, by
associative recall. That is the honest end of the list: perception, classification,
recall, decision, and a generator's prediction all share the one space and its
primitives; generation's indexing is the measured exception, kept exact on purpose.

**The inverse half of the loop is now wired in too.** Those operations -- perceive,
classify, recall, decide, generate -- are the *forward* direction: build structure and act.
The studies that grew up alongside them are the *backward* direction: take a foreign signal
APART. Three of those are now faculties of the one mind rather than disconnected modules
beside it. `decompose_signal(x, y)` detects a signal's domain topology (line / ring / Mobius /
torus), decomposes it on the matched basis (harmonics for a periodic signal, an odd-harmonic
basis for an antiperiodic Mobius one, an auto-selected additive-or-multiplicative law on a flat
line), and returns a `Formula` -- a tiny savable seed that regenerates and extrapolates the
signal, the measured-regime twin of a structure recipe. `denoise(x, ...)` exposes the engine's
manifold maps as one callable that cleans a signal by projecting it onto a low-rank subspace, a
codebook (the modern-Hopfield cleanup), or its own near-duplicates (non-local means) -- Milanfar's
thesis that *a denoiser is a map of the manifold clean signals live on*, with the modules' kept
negatives carried through (fixed-rank projection over-smooths at low noise; a projection only
helps where real structure exists; it refuses to denoise a lone vector with no prior, because
there is no free lunch). `fit_function(X, y)` fits an interpretable additive function as a
single-layer Kolmogorov-Arnold readout on the engine's encoders -- one deterministic ridge solve,
per-feature parts you can plot, and the honest boundary that an additive form cannot represent
feature interactions. Each is a thin wrapper over already-measured code, and each lands with an
end-to-end pipeline test (`test_integration.py`) that runs it THROUGH the mind -- because the
wiring's one hard lesson was that a shared *kernel* is not a shared *manifold*, so a faculty that
only imports is still a silo.

That same de-siloing pass resolved a genuine duplication on the FACTORIZATION side. The mind grew two
ways to pull a bound product apart: the original dense MAP/bipolar resonator (reached through
`factor_composite`) and the newer, higher-capacity sparse-block-code resonator that factors more
factors-by-alphabet at a fixed dimension and *verifies or abstains* rather than guessing. The SBC
factorizer had no mind-level entry point at all, so it is now a first-class faculty, `decompose_structure`,
and `factor_composite` became one entry point that delegates to it for sparse-block problems. The dense
path is kept but deprecated -- not deleted -- because it works in a different algebra (a sign-product bind,
not per-block modular addition) that the SBC resonator genuinely cannot factor; pretending one could
replace the other would have been the kind of unmeasured claim this project refuses. One factorizer for
new code, the old one honestly labelled, the boundary between them on the record.

Two smaller wirings finished the decode side. `decode_structure` is the inverse of the chain typed
structure: a linked list stored as a superposition is read back by iterated unbinding, and the whole point
-- measured -- is that each recovered pointer is noisy and that noise *compounds* into the next hop unless
you snap it back onto the node codebook first, so a raw traversal craters after a hop or two while per-peel
cleanup decodes the entire chain. And the dense associative ("modern Hopfield") cleanup from the capacity
work is now an opt-in flag on the core `cleanup` -- off by default, and at high temperature it reproduces
the existing nearest-neighbour decision bit-for-bit, so it is a strict superset whose real value is cleaning
*continuous* vectors, not changing which discrete symbol wins. With those, the mind speaks the full inverse
half of the loop: decompose a signal, factor a product, decode a chain, denoise on a manifold, fit a function
-- each a thin faculty over already-measured code, each proven through the mind, each with its negatives kept.

The mind also gained a **search** faculty and a **dynamics** faculty. `solve_maze` runs the deterministic
Tero flow-conductance model -- Physarum tubes thickening with flux until the network collapses onto the
shortest path, the same min-cost search the maze panel demonstrates but exact and ~100x faster than the
stochastic ant. `assemble` casts Rosetta-style fragment assembly as that *same* flow on a position-by-fragment
trellis, attains the global (Viterbi) optimum, and hands back the assembly as a B7 typed structure the mind can
realize -- with the honest note that the energy is a placement-mismatch stand-in, not a protein force field.
And `learn_dynamics` learns a fixed operator so that the next state is one bind of the current one (the Koopman
operator in Fourier coordinates); prediction shines on signals that genuinely have linear dynamics, only ties a
mean predictor on near-efficient-market returns (kept negative, on record), and the content-addressable
round-trip -- run a trajectory forward k steps, then back k, and land on the start -- is the durable win.

Finally the mind became **persistable** and gained two **generative** faculties. `save`/`load` route through
the kernel's versioned save, so the learned mind -- its perception, its self-organized prototype memory, its
decision brain -- round-trips with classification and decisions bit-for-bit identical, and the rate-distortion
save level (`quant='rd'`) applies to any low-rank arrays (falling back to int8 where there is none, so it is
never larger). The honest scope is on the label: what persists is the mind's learned *generalization*, not the
verbatim archive of every individual example it saw (its payloads are arbitrary inputs that do not round-trip
through a structured save -- re-learn those). `generate_vector` is the vector-level twin of the text generator,
pointed at the diffusion sampler: denoising run *backwards* from pure noise until it lands on the codebook
manifold -- generation and denoising the same operation in different regimes. And `splat_field` represents a 2-D
field as a superposition of Gaussian primitives (a splat scene IS a bundle), which both reconstructs compactly
and, because smooth Gaussians have no room for noise, doubles as a denoiser. Two later boundaries closed the
same way: an **axial** perception modality (`axial_similarity` / `decode_axial`) puts orientation-like values
on the Mobius base via the double-angle map, so the one memory stops treating theta and theta+pi as different
(measured +1.00 where the plain number modality gives +0.76 and never sees the flip); and a **splat-bundle
archive** (`splat_archive`, beside the WHT plates) stores a gallery as importance-ordered splat codes, which
buys progressive refinement and an exact region query the plates lack -- though, kept honestly on the label, the
exact WHT plates beat it on quality on DCT-friendly images, so it sits beside them, not in place of them. That
closes the integration: the
studies that grew up beside the mind are now faculties of it, each thin over already-measured code, each proven
through the mind end to end, each with its negatives kept.

**And the measurement harness became part of recognition itself.** The honesty layer -- the calibrated noise
floor, the sequential test, the false-discovery control -- had only ever been something the *tests* called to
check the engine; the mind never used it on itself. Now it does. `recognize(x)` returns not just a label but an
honest false-alarm probability: it draws random queries against the mind's OWN prototypes to learn how high pure
noise reaches, then reports the chance a match this good is an artifact (the radio-SETI / particle-physics move,
calibrated to this mind's geometry). That p-value is what lets the core operations ABSTAIN -- `classify(x,
abstain=alpha)` and `recall(x, abstain=alpha)` return the answer only when it beats noise, and None ("I don't
recognise this") when it doesn't -- on BOTH readout paths, the class prototypes and the individual store, with the
default behaviour left byte-for-byte unchanged. Two faculties build on the same own-data floor: `stream_recognize`
decides MATCH/REJECT over a stream of cues as fast as Wald's sequential test allows, and `recognize_batch` controls
the false-discovery rate across many queries so scanning a batch cannot manufacture matches by luck. An audit of
all 103 mind methods that followed found this was the one cross-cutting *property* (not *operation*) that belonged
in the core; the rest -- decompose, denoise, factor, assemble, generate -- are transformations you invoke,
correctly left as faculties (forcing them in is the same "a shared kernel is not a shared manifold" lesson denoise
already taught the hard way).

**And that audit's one flagged opportunity got the same build-and-measure treatment -- with a kept negative and
a real win.** The audit had noted the calibrated noise floor *could* drive the mind's self-maintenance:
reorganize when a novelty signal fires, rather than on a fixed clock. Built and measured (and `auto_reorganize`
is self-validating, so the honest question is the accuracy-vs-COST frontier, not accuracy alone), the flagged
idea was a NEGATIVE: a calibrated-NOVELTY trigger sat at the floor (47% on the new classes that arrive
mid-stream), because novelty detects "matches *nothing*" but online learning always leaves *something* to
match -- so it never sees the real problem, which is incoherence, not new arrivals; and calibration added
nothing over a plain cosine floor. The negative pointed straight at the right signal: COHERENCE. A
coherence-gated trigger -- reorganize only when recent inputs stop sitting on their own prototype -- matched the
best fixed schedule's accuracy at about a THIRD of the reorganize passes, by skipping the passes a coherent
store does not need. It is wired in as an opt-in `coherence_floor` (default off -- the fixed schedule is
unchanged), with one subtlety kept on the record: the gate has to read a *responsive* coherence window, because
the default smoothing is too slow to see a mid-stream shift, and the right floor is data-dependent, so it is a
parameter rather than a constant.

**The advisory panel then reviewed the live mind and asked first for FIXES, not features** — and those
landed. The calibrated recall now routes its winner through the same sublinear forest the recall path uses
(it had been doing its own exact scan, defeating the acceleration structure); its noise floor is now fit by
running that same recall path on random queries, so it is calibrated by construction rather than the
anti-conservative sample it was; a `calibration_report` draws pure noise and confirms the false-alarm rate
tracks α on both readout paths (≈0.05 at α=0.05, ≈0.10 at α=0.10 — the proof the abstention is trustworthy);
and the default `save` (`quant='auto'`) now reaches for the B5 rate-distortion code on large low-rank arrays
(a ~10× shrink over int8, cosines preserved to 0.9999), which the kernel had but the mind never asked for.
Fix and validate what shipped before building the next layer on it.

**Then the honesty layer reached ACTION.** The same calibrated-noise idea that lets recognition abstain now lets
the creature brain know when it is guessing. The brain already returned a `support` (the best cosine the current
state reaches against an action's prototypes) and gated on a HAND-SET absolute `blind_floor` — the same
uncalibrated-threshold problem the coherence floor had. `decide_confidence(state)` turns that raw support into a
false-alarm p-value via a procedure-matched brain null (run the brain's own `value()` on random states, take the
best-support distribution — calibrated by construction), returning `(action, p)`: p small means the brain has
genuinely been somewhere like here and the value estimate can be trusted, p large means it is guessing.
`decide(…, explore_if_unrecognized=α)` makes that actionable — when p>α it takes a safe random move instead of
committing to a value built on nothing. Measured: a familiar state → the learned-good action at p=0.000; a
never-seen one → p=0.300; the brain null fires on noise at ≈0.07/0.11 at α=0.05/0.10. Two Tier-0 finishers came
with it. The Wald sequential test was described as saving half a fixed window's samples, but the tour's distinct
items are well-separated from noise, so it correctly decides in ~1 sample; the savings are a property of the
OVERLAP regime, and shown there honestly — well-separated → 1.1 samples, faint overlap → 2.2, heavy overlap →
4.4, each about half the fixed-window count at matched error. And the opt-in coherence gate gained an `'auto'`
floor that drops the data-dependent absolute level: reorganize when coherence falls below ~90% of its own recent
peak — a relative retention that transfers across data scales (trading an absolute parameter for a relative one
that needs no per-dataset retuning), measured to match the hand-set floor's accuracy-vs-cost (82% at 6.7 passes
vs the schedule's 81% at 11).

**Then the honesty arc closed with `scan`.** Siemion's seat had asked for the move a real search lives by:
scan an astronomical channel count, decide each channel as fast as its own evidence allows, and still control
the trials factor across all of them. `scan(channels)` is that, and it is pure assembly of parts already
shipped — Wald's SPRT decides each channel's stream (B3), then Benjamini-Hochberg/Yekutieli FDR runs across
the channels' p-values; a channel is a confirmed detection only when the SPRT says MATCH *and* its p-value
survives FDR. The load-bearing detail was a calibration bug caught before shipping: the channel p-value needs a
noise floor, and the obvious one (the recognition null) is wrong twice over — it scores prototype rows, not the
max label score `recognize` returns, and `recognize` first runs `perceive`, which is not the identity even for a
raw vector (it lifts noise's max label score from ~0.086 to ~0.117). Either wrong floor made ~70 of 80
pure-noise channels look significant; a procedure-matched floor (random vectors through `recognize` itself, the
exact channel path) makes them uniform again. Measured on a weak target among eighty noise channels: all twelve
signal channels found, zero noise channels detected; naive per-channel thresholding would false-alarm on ~11 of
the eighty, FDR holds it to 0; and the SPRT spends 1.0 samples on a clear channel but ~1.8 on a faint one —
deciding, per channel, as fast as the evidence allows.

**And the same two disciplines reach into radio astronomy.** `detect_drifting(waterfall)` is the SETI search for a
narrowband technosignature that drifts in frequency because of the relative motion between us and its source. The
insight is that a Doppler drift is a cyclic *shift* of the spectrum over time, and a cyclic shift is the engine's
own `permute` — the same rigid-shift transform the video coder uses for motion compensation, and equally a binding
(`bind(x, δ_k) == permute(x, k)`). So "de-Doppler integration" — the matched filter that recovers a drifting signal
a stationary detector loses — is just permute-ing each frame back by the drift before summing, and the look-elsewhere
control over the (drift × channel) search grid is `bh_fdr`, exactly as `scan` controls it across channels. Measured on
synthetic spectrograms at the field's S/N≥10 regime: a stationary integration loses a drifting signal in the noise
(~2.9σ) while de-drifting at the right rate recovers it (~6.2σ) and reports the rate; naive per-cell thresholding fires
on ~100% of pure-noise scans, FDR holds it to ~0%; recall reaches ~96% at zero false-positive; and supplying an
OFF-target spectrogram lets the ON-OFF cadence reject a strong stationary RFI ~100% of the time (it persists in the
OFF pointing) while keeping the drifting ON-only signal ~94%. The kept negative is the field's own lesson: below ~10σ
integrated the dependent-FDR correction over the many cells is conservative — which is precisely why turboSETI searches
at S/N≥10. Two engine primitives — a shift-is-a-binding and a false-discovery veto — doing the radio astronomer's job.

**A store that checks itself.** `verify_store(items)` gives the engine a tamper-evident commitment — the
structural idea of a Merkle tree, built from nothing but the two kernel primitives. A Merkle tree hashes each
item, hashes pairs of hashes up to a single root, and uses that root to detect any change and the tree to
localise *which* item changed in O(log n); here the hash's structural role is played by *bundle* and the leaf
by *bind*. Each item is bound to its slot key (`leaf_i = bind(pos_i, item_i)`), every internal node is the
bundle (sum) of its children, and the root is the whole-store composite. Rebuild the root from the current
items and compare: any change to any item — or to which slot it sits in — shifts it; then descend, following
at each node the child whose composite no longer matches, and the changed item falls out in ≤ log₂(n)
comparisons regardless of how many are stored. Binding the *position* into each leaf is what makes a reordering
visible: a plain bundle is commutative, so without slot keys a swap would leave the root identical. The honest
bound is kept loud, because it is the whole distance from a cryptographic Merkle tree: the root is *linear*, so
the map from items to root is many-to-one and collisions exist — a key-aware adversary can cancel a change to
one item with a compensating change to another (a deconvolution) and leave the root bit-for-bit unchanged. So
this is evidence of *accidental* corruption and uncoordinated tampering, not tamper-proofing against an
adversary who knows the keys; and the descent localises one change per pass. (The plan had also expected that
quantising the checksums to save space would blind it to small tampers — measured, it does not, because in
1024 dimensions a tamper always pushes some component across a quantiser boundary.)

**And the engine learned to measure itself against the outside.** A `benchmarks/` harness puts two faculties
up against the off-the-shelf tool a practitioner would reach for, with the project's rule that where the
standard tool wins, it is reported, not hidden. The geometry-preserving `quant='rd'` code — consolidation/KLT,
then water-filling, then a bit-exact rANS coder — beats general-purpose `zlib`/`lzma` by ~34× on data with real
low-rank structure (it spends bits only on the directions that carry variance), and *loses* to int8+zlib on
full-rank random data, where there is nothing for the KLT to exploit and the shared basis costs more than it
saves — exactly the right read, since the engine's stored states are low-rank and random vectors are not. The
`HoloForest` matches an exact brute-force scan's recall@1 up to ~10k items while touching 3% of the comparisons
at 20k, but the exact scan is a single BLAS matrix-vector product, so the forest only wins on wall-clock past
~20k items: it buys sublinear *work*, not raw speed against a tight inner loop. The reachable baselines were
audited first — denoising, forecasting, and classification have no fair in-constraints opponent (their usual
ones are banned dependencies), so they are left out rather than measured against a strawman.

**Then continuous encoding went multi-dimensional.** Fractional Power Encoding — encode a value by raising a
base vector to a real power, so that shifting the value is a *binding* and the similarity is a kernel you
*design* — turned out to already be in the box: the 1-D `ScalarEncoder` is exactly this (`encode(x)` =
"base^x", with a Bochner kernel), and because `bind` is circular convolution it multiplies the codes' spectra
and so adds their positions, giving `bind(encode(x), encode(s)) == encode(x+s)` to numerical exactness. The
genuine new step (`vector_function_encoder`, `holographic_fpe.py`) is up from a scalar to a *vector* domain and
to whole *functions*: a point in Rⁿ is encoded by binding one per-axis FPE per coordinate (a shift along any
axis is still one binding, and the kernel is the product of the per-axis kernels — an n-D RBF), and a function
f: Rⁿ→R is a weighted superposition of encoded points that reads as a holographic kernel-density estimate when
queried and translates by a *single* binding. The standing cliff is kept on the record: a function is a bundle,
so query separation decays as the atom count grows (measured +0.39 → +0.01 from 2 to 128 atoms); and where a
scalar already suffices, the n-D machinery buys nothing — 1-D FPE *is* the ScalarEncoder.

**Then the hand-picked bases became one operator.** `decompose_signal` chose a basis by detected topology —
line→elementary, ring→harmonic — but those are not special cases to enumerate, they are two readings of one
object: the graph Laplacian's eigenbasis. The eigenbasis of a *path* graph is exactly the DCT (the elementary
basis) and the eigenbasis of a *cycle* graph is exactly the DFT (the harmonic basis), so a single operator
(`holographic_spectral.py`, faculty `spectral_basis`) subsumes both — and extends to manifolds the topology
detector cannot name. On a sphere, which the detector can only call "line", the kNN-graph Laplacian's low
eigenvectors recover a smooth field (denoise error 1.9 from a noisy 6.1) where the line/index-order basis
barely moves it (5.2): the data-driven basis is measurably right where the hand-picked fallback is not. The
same operator also reads topology — the Hodge Laplacian's harmonic dimension equals the Betti numbers (a
4-cycle → one component and one loop, a filled triangle → the loop gone), which is the spectral route the next
piece builds on. The honest scope is stated: dense `eigh` bounds it to moderate N (a huge graph would need a
sparse solver, i.e. a second dependency), and the eigenvector signs are fixed to a convention so the basis is
reproducible — the same bit-exact-tie discipline the `bind_batch` lesson taught.

**Then topology stopped being a hand-coded list.** `detect_topology` names a 1-D signal's shape by fitting
harmonics — line, ring, mobius, torus — but that is a fixed menu that only sees what it has a fit for. The
principled version reads topology straight off a point cloud, the way computational topology does: build a
Vietoris-Rips complex at a distance scale, count its holes by dimension (the Betti numbers — components,
loops, voids), and keep the signature that *persists* across the widest scale band (`holographic_topology.py`,
faculty `manifold_topology`). It reproduces what `detect_topology` gets — a contractible cloud is (1,0,0)
"line", a loop is (1,1,0) "ring" — and extends to shapes it cannot name structurally: a *torus* is (1,2,1),
two loops around a void, and a *sphere* is (1,0,1), no loops but one void (B₁ alone orders line→circle→torus
0/1/2; B₂ is what separates a sphere from a line, since both have no loops). The first cut computed Betti
numbers with a dense real `matrix_rank` and *timed out on the torus*; reducing the boundary operators over
GF(2) instead — columns as integer bitmasks, XOR elimination, the standard persistent-homology twist — is
exact and drops the torus from a timeout to half a second, and it agrees with EXP-5's Hodge-Laplacian Betti
route on every fixed complex (two operators corroborating one answer). The negatives are kept loud: persistent
homology is finicky on small, noisy, or unevenly sampled clouds (a sine's delay embedding, dense at its
turning points, fragments at the median scale — so this reads a well-sampled manifold, not an arbitrary
trajectory), it is blind to non-topological geometry by design (a circle and an ellipse are both (1,1,0), so
it wants EXP-6's geometry beside it), and the cloud is subsampled to keep the complex tractable.

**Then flows split into what they're made of.** The same boundary operators that count holes (EXP-5/7) also
take an edge flow apart. The Helmholtz-Hodge decomposition (`hodge_decomposition`, `denoise_flow`) splits any
flow on a graph into three orthogonal pieces: a *gradient* part (curl-free transport running downhill from a
vertex potential — what a source-to-sink flow is made of), a *curl* part (circulation around the filled
triangles), and a *harmonic* part (the remainder, both divergence-free and curl-free — the global circulation
that wraps the holes). The split is exact and the pieces are orthogonal to floating point, the harmonic part
is genuinely div- and curl-free, and its dimension is exactly B₁ — so the harmonic flow *is* the topology the
previous piece counts, the two operators meeting on one complex. It denoises: a noisy transport flow, with its
curl dropped, comes back closer to the truth than the noisy input (1.1 from a noisy 1.9) and well past naive
edge-smoothing (3.6), which over-smooths. And the kept negative falls straight out of the topology — on a
tree, with no cycles and no triangles, there is nothing to circulate, so curl and harmonic are exactly zero
and every flow is pure gradient.

**Then a second way to bind arrived, for the rotation-shaped corner.** Like `tensor_bind`, the geometric
product of Clifford algebra Cl(3,0) (`clifford`, `holographic_clifford.py`) is a parallel binding mode, not a
drop-in for circular convolution — its seat is geometric structure, specifically 3D rotations. A rotor
(`cos(θ/2) − sin(θ/2)·B`) rotates a vector by the sandwich `R v R̃`, and two facts make it beat convolution
here: composing rotations is *exact* (the product of two rotors is the rotor of the composed rotation —
measured error ~1e-15 against applying the rotations in sequence), and it is *non-commutative* (rotation order
matters, and the product captures it — two orders of the same pair land a measured ~0.7 apart on a probe
vector). HRR's circular convolution is commutative, so it returns one answer for both orders by construction
and must carry that whole order-gap as error; that gap, which convolution provably cannot close, is the
concrete sense in which Clifford wins on rotations. The negatives are exactly what you'd expect of a
specialised algebra and are kept on the record: Cl(n,0) needs 2ⁿ components (8 here, 1024 for ten dimensions),
so it is affordable only for low-dimensional geometric domains, not as a general high-D substrate; and it binds
*versors* — a unit rotor's inverse is its reverse, but an arbitrary multivector's is not — so it is a
geometric-transform algebra, not a general key→value memory like HRR. A parallel tool for the corner it fits,
in the same spirit as the tensor-product bind.

**Then distance learned to move mass.** A bin-wise distance — Euclidean, cosine — compares two distributions
height-by-height and is blind to *where* the mass sits: two histograms that don't overlap score maximally far
no matter how far apart they actually are, so a peak at bin 12 reads exactly as distant from a peak at bin 10
as a peak at bin 40 does. The Wasserstein (earth-mover's) distance instead measures the least work to *move*
one distribution onto the other — mass times the ground distance it travels — so it keeps growing as the
distributions move apart even with no shared support (`wasserstein`, `holographic_transport.py`). It is
computed by the Sinkhorn algorithm: add an entropy term, which turns optimal transport into a Gibbs kernel
`exp(−C/ε)` and a pair of alternating diagonal rescalings that converge to the transport plan, the distance
being the plan against the cost. It matches the 1-D closed form (the L1 distance between CDFs) to a thousandth,
and where it earns its place is exactly the case bin-wise metrics miss: a shift of 5, 10, 20 reads as a
distance of 5, 10, 20, while Euclidean saturates flat at ~0.53 for every non-overlapping shift and cosine
collapses to zero — both unable to tell a near miss from a far one. The kept negatives are the ε knob and the
cost: too large an ε blurs the plan toward independence and inflates the distance (a same-mean pair with true
distance ~3.6 reads ~4.9 at ε=50), too small underflows the kernel between separated supports into a broken
answer, and the dense kernel is O(n·m) per iteration — so the default ε scales to the cost, and a wide cost
range wants an explicit ε or a log-domain solver.

**Then the flow solver learned to see its own circulation.** An above/below sweep of the new spectral machinery
found the obvious home it hadn't reached: the Tero flow solver computes a Poiseuille flux on every edge each
step, uses it to thicken tubes, and *throws it away* once it has the path. That flux is exactly the kind of
edge flow the Helmholtz-Hodge decomposition takes apart. So `tero_flux` now exposes the converged flux and
`flow_circulation` splits it: the gradient part is the net source-to-goal transport (its divergence is the
injected current, to a part in a million), and the harmonic part is *circulation* around the graph's loops —
its dimension exactly the graph's B₁, the loop count the topology kernels already measure. A maze graph has no
filled triangles, so there is no curl. What this buys is a previously-hidden read on the flow: the harmonic
energy fraction is how much of the converged flux circulates rather than transports — exactly zero on a tree,
where the route is forced, and nonzero on a loopy grid in a way that varies with the saturation exponent μ (at
high μ competing thick tubes leave more flux going in circles). One restraint is worth recording as part of the
sweep: the flow solver builds its own conductance-weighted Laplacian, and it was *not* rerouted through the
shared `graph_laplacian` — that dynamics is tie-sensitive, and a different summation order could flip a
trajectory, the `bind_batch` lesson. The shared *helper* was extracted inside the module; the shared *kernel*
was deliberately left alone. Not everything that looks duplicated should be merged.

**Then a second sweep found the denoiser's blind spot.** Re-running the discipline over the full geometry
toolkit — spectral, topology, transport — most of it was already where it counted, which is the honest usual
result of a sweep. The one genuine gap was in the denoise faculty: it could map a *linear* subspace
(`manifold`/`adaptive`, which need an example set) but had no map for a *curved* manifold's geometry, and none
of its methods could clean a lone scalar field sitting on a point cloud. That is exactly what the EXP-5/6
graph-Laplacian eigenbasis does, so it went in as `denoise(method='spectral', points=…)` — the nonlinear-manifold
map, needing no example set and no codebook, just the cloud's own geometry. On a smooth field over a 2-sphere it
cleans error 4.1→0.9 where the geometry-blind options barely move it (trajectory/SSA 3.1, a fixed DCT low-pass
4.2), because a linear or 1-D prior cannot see a curved manifold's smoothness. The transport and topology kernels
stayed standalone, and that is a finding, not a failure: the sweep turned up no existing bin-wise distribution
comparison for Wasserstein to improve (the market compares price *windows* by cosine and forecasts by proper
score; the de-Doppler search already uses the field-standard matched-filter bank), and persistent homology is
held back from the 1-D topology detector by its delay-embedding sampling negative. Resisting the urge to force
those is the other half of the discipline.

**Then the resonator got a calibrated voice.** The SBC resonator could already say `verified` — True when the
recovered factors rebuild the product exactly — but that certificate is uselessly brittle on an approximate
input: corrupt one block of a noisy bind and `verified` goes False even though the resonator found exactly the
right factors. `resonator_confidence` (via `factor_composite(..., confidence=True)`) adds the graded answer, a
p-value on how much better the factors rebuild the input than the resonator manages on pure noise. The honest
part is the null: the resonator optimises reconstruction, so on structureless input it still manufactures ~0.27
block agreement, far above the ~1/L a random-picks null assumes — calibrate to random picks and pure noise reads
as a near-certain detection (p~0.02). The procedure-matched null (the agreement the same resonator reaches on
random SBCs) fixes it, exactly as it fixed scan's floor. Measured: corrupting up to five of sixteen blocks, the
resonator still recovers the true factors and the calibrated p holds at 0.010 (trust) the whole way, while
`verified` is True only at zero corruption; on eighty pure-noise products the median p is 0.84 (it abstains),
with a conservative false-alarm rate because block agreement is discrete.

**Then `assemble` got a real energy and a comparator.** The fragment assembler already found the global-optimal
arrangement by the same min-cost flow the maze solver runs, but on one hardcoded energy — Hamming mismatch, a
stand-in. `assemble(..., energy=callable)` makes that pluggable (defaulting to the stand-in, so nothing existing
changes), and the point is the Rosetta move: not every substitution costs the same. With a toy substitution
matrix where cross-group swaps are dear, the target "EAAE" assembles to "BABE" under Hamming but to "EEEE" under
substitution — and each is the unique global optimum under its own energy, both matching the Viterbi DP. Alongside
it, `compare_structures` superposes two assembled structures and reads their overlap both exactly (the shared
(position, fragment) motifs) and holographically (the consolidation SVD of the superposed role-bound vectors,
where shared placements collapse the rank); on clean structures the two reads agree — the holographic one
validated against the exact count, and the form you use when a fold is only available as a hypervector. A tie in
the first test instance (two assemblies at equal Hamming cost) passed alone but failed under suite ordering — a
preview of the determinism work — and the fix was to choose an instance with unique optima rather than depend on
how a tie breaks.

**Then three faculties turned out to be one engine.** Macklin's observation: the resonator's alternating
cleanup, the PnP denoise loop, and a position-based-dynamics constraint sweep are all "project onto each
constraint in turn until they jointly hold." `project_onto_constraints` is that engine, and the unification is
made real rather than nominal — `pnp_restore` now literally calls it, bit-for-bit unchanged. It shows up as
three instances: POCS (alternating projection onto two subspaces converges to their intersection in 29 sweeps),
a resonator (factor-cleanup projections plus restarts recover a bound product — a single restart falls into a
spurious fixed point, an honest reminder that the restarts are load-bearing), and PnP. Shipped alongside it is
the determinism audit the whole Macklin thread points at: every calibrated and null path added across this
program — recall_calibrated, decide_confidence, the auto coherence floor, scan, resonator confidence,
compare_structures — is run twice on a freshly rebuilt setup (null recomputed, not cached) with numpy's global
RNG scrambled in between, and all return bit-identical. The paths draw only from their own seeded RNG; the
determinism is audited, not assumed.

**Then the inverse problem became a mind operation.** The PnP/RED loop and the noise-adaptive denoiser were
already callable, but restoring a degraded measurement meant hand-building the forward and adjoint operators
every time. `restore(y, mask=..., samples=...)` closes that: pass a 0/1 mask and the operators are filled in
(a diagonal mask is its own transpose), the prior is the mind's own adaptive denoiser — and it delegates to
`denoise(method='pnp')`, so it is one mind call on the existing loop, not a reimplementation. Measured end to
end on an erased archive plate: the mind's splat archive holds a low-rank gallery, one plate has a 25-pixel
block erased plus noise, and a single denoise reaches 19.3 dB while `restore` (the loop) reaches 38.5 dB — a
19 dB win, because the one-shot projection is dragged toward zero by the erased pixels while the loop holds the
observed ones to the measurement and fills only the rest. The noise estimate it relies on is accurate at
moderate-to-high noise (and `sigma=None` matches supplying the true sigma) but over-estimates at low noise,
where a textured signal's own detail inflates the estimate — kept on the record, honest where it holds and
where it does not.

**Then the capacity cliff became a live readout.** `capacity_report` answers two questions about the same store
geometry at once. Where does the store sit relative to the noise-wins cliff (Plate's HRR capacity)? A random
query's best cosine to N stored rows — the noise floor — is the max of N cosines, ~sqrt(2 ln N / D); a genuine
match sits far above it, and the report gives the gap as a d' in noise-sigmas (a roomy D=512 store at d'=23, a
loaded D=64 store at d'=6.6 — the diagnostic sees the load), the measured floor against the theoretical bound,
and the headroom before the rising floor reaches the match (10^50x for the roomy store, 10^5x for the loaded
one — the enormous high-D capacity made concrete). And does calibrated coverage hold as the store grows
(Cranmer's open question, which Tier 0 left because it only checked a fixed store)? Building random codebooks
of increasing size and measuring the false-alarm rate, it stays at α — the procedure-matched null re-fits to
the rising floor, so the look-elsewhere discipline is load-robust, not just fixed-store-robust.

**Then a sound found the structure a market never had.** The dynamics operator (`learn_dynamics`) only tied a
trivial mean predictor on real market returns — near-efficient-market returns carry almost no linear structure
for a fixed operator to exploit, the correct result kept on the record. Audio is the honest counter-test.
`spectral_encode` is a phase vocoder in the complex domain: a frame's DFT splits into a unit-magnitude phasor
per bin (the phase — an FHRR vector that binds and recalls in the high-capacity memory like any atom) and a
magnitude per bin (the timbre), and `spectral_decode` puts them back exactly. A sustained tone framed with a
hop evolves by precisely the per-bin phase advance the propagator learns — the same advance those phasors
carry, the operator and the encoding two faces of one structure — and through the mind the propagator predicts
the next frame at error 0.001 against persistence's 1.64 and mean's 1.00, three orders of magnitude, the linear
structure the market lacked. Two negatives stay on record: a pure tone is too sparse for phase alone to
separate (its silent bins dominate, so magnitude carries identity there), and on non-integer frequencies with
noise the fixed operator is approximate (error 0.169, still ~6–10× better than either baseline).

**Then the same operator learned a fluid.** Stam's FFT fluid solver works on a periodic torus, doing the hard
step in Fourier space — the same FFT-on-a-torus the engine's bind already is. A passive scalar's linear
advection-diffusion step is exact there: each Fourier mode rotates (advection) and decays (diffusion), a
per-bin transfer the propagator fits. Through the mind, one-step prediction error on an advected field is 0.011
against persistence 0.34 and mean 1.15, the learned operator rolls out eight steps as a surrogate solver
tracking the true simulation to ~3.5%, and the operator's own forward-then-back round-trip returns the start at
cosine 1.0. The honest limit is on the record too: a nonlinear Burgers field forms shocks no fixed linear
operator can capture, and there the propagator does worse than persistence (0.054 vs 0.006). Audio and a linear
fluid are the two positives; the market and the Burgers shock are the two negatives; the faculty's docstring
carries all four so the boundary travels with the code.

**Then slime mould designed a network.** The single-source maze solver became multi-terminal: `design_network`
drives the Tero/Physarum flow between every pair of terminals at once (one multi-right-hand-side solve per
step), and the surviving tubes form the connecting network — Tero's Tokyo-rail experiment on a graph. The
feedback parameter tunes the trade-off Physarum is famous for: high μ grows a near-minimal Steiner tree (21
edges on a test grid, shorter than the 24-hop terminal-MST because the flow shares trunk segments), low μ keeps
a fault-tolerant mesh with redundant loops that survive an edge cut. And the network comes back not as a list
but as a B7 typed structure — a graph-memory M = superpose of bound node pairs — that the engine's own
unbind+cleanup queries: ask it for a node's neighbours and they come back above every non-neighbour. The
honest scope note is that the dense Laplacian solve is cubic per step, so this is for graphs of tens of nodes,
where the model's interpretability is the whole point.

**Then the image archive learned to answer a description.** The re-audit found the cross-modal machinery was
already in the DCT-plate archive — store an image with tags and it keeps a hypervector address; ask
`recall_by_tags` for a description and it returns the best match — but only the lossy splat archive had been
wired into the mind. `image_archive` wires in the exact one: recover any stored image bit-exactly (6e-15 error
at full keep), describe-then-retrieve with soft-AND over the query tags (`round`+`large` returns the ring,
`round`+`small` the circle), and — the improvement, the reverse direction `tags_of` — ask what tags an image
would get and they come back ranked. All of it survives 40% plate erasure: a text query still reconstructs the
right image at 0.002 error. The mind now has both archives and picks the right one — splats for a compact
bundle, plates for exact, addressable, damage-tolerant recall.

**Then the generator stopped repeating itself.** B10 showed that running the denoiser backwards from noise
generates a sample, but over a bare codebook it only ever returns a stored atom — a degenerate sampler.
`generate_structure` points the same annealed diffusion at a composed manifold: a valid structure is a bundle
of role-bound fillers, far too many to enumerate, so the denoiser is slot-wise — unbind each role, snap its
filler toward the vocabulary, rebind, bundle. The random start now walks onto the manifold of role-filler
structures. Ten seeds give ten distinct structures, each re-encoding to itself at cosine 1.0 (a genuine
composition, every slot decodable) and nearly orthogonal to any single atom — novel but valid — while the bare
codebook still collapses to one stored atom. Generation and denoising stay the same operation; aimed at the
composition manifold, it becomes a generator of new structure rather than a recaller of old.

**Then a single vector grew a fractal.** The demoscene move — maximal richness from a tiny deterministic kernel,
infinite detail by recursion — stated in the engine's terms. `fractal_seed` packs a fractal kernel (N copies of
the plane, each contracted and offset) into one hypervector as a bundle of role-bound grid atoms; `fractal_scene`
decodes it with pure unbind-and-cleanup, then repeats that one kernel to depth — each level a contracted copy of
the whole — and measures the box-counting dimension. A Sierpinski seed decodes to 3 copies at scale ½ and
expands to dimension 1.57 against the theoretical log3/log2 = 1.585; a five-copy scale-⅓ seed lands at 1.51
against log5/log3 = 1.465; the two seeds give distinct dimensions and expand identically every time. One vector
holds a generator, the engine reads it out, and recursion turns it into a self-similar scene of a predictable
fractal dimension.

**Then the splats learned to stretch.** The splat archive used circular Gaussians by choice; the real
3D-Gaussian-Splatting primitive is an oriented, full-covariance Gaussian fit by differentiable optimisation.
`splat_aniso` builds that core from scratch — each splat carries a Cholesky factor of its inverse covariance,
and an analytical-gradient NumPy Adam descends the reconstruction loss, warm-started from the isotropic fit, in
2-D or 3-D alike. On two elongated ridges a circular fit reaches ~18 dB and the anisotropic one ~64; on a 3-D
ellipsoid, ~24 dB against ~61 — one aligned splat where many circular ones failed. The honest Tier-4 label
stays attached: the loss is non-convex (a local optimum, more splats not monotonically better), and this is the
primitive and its fit, not the production renderer — no tile rasteriser, no view-dependent colour, no GPU.

**Then the bind itself was put on a scale.** HRR's circular convolution is a compressed projection of the
tensor product; `tensor_bind` keeps the uncompressed outer product and its low-rank tensor-train (MPS)
truncation, so all three points on the rank spectrum can be measured against the engine's bind. The tensor
product recalls far better at a fixed load — 0.95 against HRR's 0.25 at sixteen pairs — because its crosstalk is
suppressed by the key inner products, and with orthogonal keys it is exact up to M = D where circular
convolution cannot be; an MPS truncation losslessly compresses a low-rank binding matrix eightfold. But the
honest bucket is clear and stays attached: per stored number the two sit on the same capacity frontier — HRR's
compression gives up nothing there — and a generic full-rank binding cannot be MPS-compressed without losing
recall. The tensor route buys fidelity and exact structured-key capacity with storage; it is a different point
on the tradeoff, not a free improvement over the engine's bind.

**Then a parallel line came home.** A separate investigation — Path D, *computing and storing inside the
holographic space* — merged in as its own bundle (under `path_d/`, with the frontier-program and
dataset-benchmark docs), and its two reusable modules were wired into the mind as faculties. The arc is the
engine's own lesson one rung up: a single vector holds only ~0.1×D items faithfully, that budget is conserved,
and you scale not by encoding harder but by *federating* — more vectors, coordinated by a thin layer.
`storage_array` is that as a RAID-style store: a shard is a running sum, so a parity shard is the real-valued
sibling of a fountain droplet and reconstructs a lost shard exactly by subtraction (150 symbols grow to three
shards at 0.89 recall; lose one and parity restores it exactly). `superpose_compute` is the width half —
evaluate many computations inside one vector, exact at K=1 and decaying along the predicted wall — the
parallel-readout complement to the recursion that is the depth half. The headline the bundle proved, reproduced
here, is the same move applied to a neural forward pass: one weight-vector faithful to sixteen classes,
federated to eight shards holding ninety-six. The capacity walls travel with the faculties as kept negatives,
and the second lever it surfaced — RNS-phasor arithmetic — is flagged as a next step, not yet a claim.

**That next step is now built.** `exact_matmul` is the arithmetic lever wired in: a matmul read out of a lossy
superposition dies as the matrix grows (the bundled rows interfere), but a number is carried exactly as a phase
— a unit phasor is a residue, and binding phasors adds their phases — so carrying each number as residues over
coprime moduli and accumulating by phasor binding gives an *exact* modular multiply-accumulate for any number of
terms, recomposed by CRT. Integer matmul at 256×64 comes back with zero error where the lossy bundle holds 0.11
fidelity; a float matmul is exact for its quantized operands; and the exact range federates over moduli channels
(1e8 → 1e62), the arithmetic sibling of the storage array. The scope stays attached: exact for integer or
fixed-point within range, the only error a quantization rounding (not a crosstalk wall), and the FLOPs are real
— per-modulus parallelism, native on phasor hardware.

**And the index learned to skip the scan.** The long-wanted sublinear recall arrives as `pivot_index`: a tree
whose internal nodes hold explicit pivots, so routing a query is a nearest-pivot decision — cleanup against a
small codebook — applied recursively, one level per hop. Greedy top-1 routing matches an exhaustive scan (≈0.88
vs ≈0.90 on a well-separated set) while touching ~18 pivots instead of all 216, and a beam lands the true leaf
in the candidate set every time. The kept negative is the wrong-turn risk on overlapping data, with the beam as
the honest knob; the build is a NumPy k-means, no sklearn. The earlier crash — a content index that summarized
upward into a bundle and collapsed to 0.23 — is why the pivots are stored explicitly: a B-tree never bundles.

**The array got the same routing trick.** When you don't hold the directory, the array used to fall back on
broadcasting a query to every shard — O(shards), and it erodes as the array grows. `routed_recall` summarizes
each shard by a key-sketch (a bundle of its keys), matches a query against the sketches in one matmul, and
unbinds only the top-c candidate shards. At 64 shards it holds directory-level recall (0.99) where broadcast has
slipped to 0.95, while touching just eight shards instead of all sixty-four — the content-addressable, sublinear
version of the lookup, the same "summarize, route, then resolve a few" move the pivot tree makes inside its
nodes.

**And the federation move reached all the way into compute.** `distributed_forward` is the arc's headline: a
classifier's weight rows packed into one vector cap out around 0.02×D classes, but federating them across K
shards moves the wall to ~K×0.02×D — a 64-class forward pass that a single vector reads at 0.79 comes back at
1.00 across eight shards, the same federation that fixed storage now applied to the matmul (16→96 classes
faithful in the sweep). Depth — where a deep net compounds each layer's crosstalk — is cured two ways, both
wired: compute each layer exactly with the RNS arithmetic (no crosstalk to compound, exact at any depth), or
cleanup-gate between layers with a soft dense-Hopfield that snaps activations back onto the valid manifold. The
honest scope rides along: federation buys fidelity, not fewer FLOPs, and cleanup-gating's end-to-end accuracy
gain needs a trained manifold — so only the robust pieces (the exact-arithmetic depth cure, and the cleanup
primitive that demonstrably denoises onto the manifold) are claimed, not an always-win.

**The same lever turned out to fit three more locks.** The Bucket-A experiments re-opened a handful of other
single-vector walls, and each was the identical conservation law wearing a different hat — so they wired to the
faculties that already hold the move, not to new ones. `superpose_compute` gained a `shards` knob: federating
the candidates lets it pick the planted match out of 160 hypotheses (0.38 → 1.00 at K=8), and federating the
positions lets it recall a 160-symbol sequence (0.58 → 1.00), one call serving both via a `query` or a
`codebook`. `federated_archive` does it for images — K `HolographicArchive` shards behind a directory, holding
64 images at the same recovery quality a single archive gives at matched total dimension (0.965 vs 0.965):
capacity federates, fidelity is conserved. With that, every advancement Path D's experiments demonstrated is a
faculty or a measured property of one — federation across storage, width, the archive, and the forward pass;
exact arithmetic; and sublinear lookup — and the lone foil (lossy continuous matmul) and the pure conservation
measurements stay on the record as evidence, not dressed up as methods they were never meant to be.

**Translation answered inference; the learning question got its own answer.** Path D's RNS lever made the
substrate *run* trained networks exactly — matmul, deep forward pass, attention, even autoregressive
generation — but it never claimed to *train* them; that gap is closed here by gradient-free, substrate-native
learning methods the field already proved, two of which wired in as faculties on machinery the engine already
had. `reservoir` is an Echo-State Network on the engine's own recurrence: `permute` (a cyclic shift) is
norm-preserving, exactly the echo-state property, so the substrate's sequence operator *is* a near-optimal
reservoir — fixed, with only a linear readout trained by one closed-form ridge solve (no gradients,
deterministic). It reaches a literature-grade NARMA10 (NRMSE ~0.37, the reservoir features carrying it past a
linear baseline) and *learns* autoregressive text generation; the honest negative rides along — a chaotic
free-run must diverge pointwise after ~one Lyapunov time (the climate is learnable, the weather is not).
`prototype_classifier` is the HDC learner: bundle a class's encoded examples into one prototype, then a
perceptron retraining pass — pull the correct prototype toward a misclassified example, push the wrong one
away, pure add/subtract, no gradients. The encoding lifts a centroid model sharply (wine 0.67 → 0.98) and
retraining improves it (digits 0.90 → 0.95), landing — honestly, as the field reports — just below a tuned
linear model. Both are the *truly* derivative-free corner. `equilibrium_net` adds the LOCAL-GRADIENT corner:
Equilibrium Propagation trains the energy-based Hopfield cleanup itself — a free relaxation gives the
prediction, and symmetric nudged relaxations (±β·loss on the output) give a contrastive Hebbian update that
*estimates the loss gradient* (matched to finite differences at cosine ~1.0), with no backpropagation. Because
it learns the *hidden* weights, it fits a nonlinear task the linear-readout methods cannot — two interleaving
moons at ~0.92 against a linear model's ~0.85, landing honestly below exact backprop (~1.0) since the finite-β
estimate is biased. `forward_forward` completes the program with the DEPTH corner: Forward-Forward stacks
layers each trained by a local goodness objective — two forward passes (positive data; data with a *wrong*
label) instead of a backward pass, L2-normalized between layers, no backprop and no settling. Its mechanism
works (positive goodness provably separates from negative; a separable multi-class task is classified
perfectly), but the honest negative is loud: at this compact scale FF *trails* a plain linear model on every
task tried (two-moons ~0.88 tie, overlapping blobs 0.95 vs 0.99, sklearn digits 0.88 vs logistic 0.97) — its
published MNIST-scale accuracy needs the full recipe, so what ships here is the backprop-free-*depth* mechanism,
not a competitive number (Mono-Forward is the stronger refinement, not built). The standing caveat covers all
four families: this is native learning at small/moderate scale, not a route to frontier scale.

**And the learning earns back a dynamics negative.** `learn_dynamics` fits one per-frequency transfer — the
linear Koopman/DMD operator — which is exact for linearisable flow but, on a state-dependent nonlinearity,
sits at the persistence floor (the kept Burgers-shock negative). `learn_chaos` is its nonlinear companion: it
points the reservoir at the *one-step evolution map*, a fixed nonlinear expansion read out by a trained linear
map, and it learns the chaotic flow a linear transfer structurally cannot. On the canonical Lorenz '63 test
its one-step prediction is ~40x better than the *best* linear map (full DMD) and ~50x better than persistence
— a clean win, since chaos pins any linear operator at the floor. The negatives stay loud and travel with it:
closed-loop free-run holds only ~one Lyapunov time (the well-known reservoir stability wall — and the
recurrence mixing, shift vs permutation vs unitary-bind, is *not* the lever), and a high-dimensional PDE field
(a 48-D Burgers field at ~0.27, worse than persistence) is out of reach for a single global reservoir, where
the literature needs local/parallel ones and EP is weak at field regression. The win is a genuine
low-dimensional nonlinear-dynamics result, honestly bounded.

**And the learning reaches the deepest fixed object: the cleanup itself.** Every cleanup in the engine
stores its attractors — the classical one snaps to a stored atom, the modern-Hopfield energy cleanup
relaxes against a fixed codebook. `learn_cleanup` makes good on a claim that had only ever been a
docstring (that EP *is* the learning rule for the energy memory): it trains an energy whose attractors
form a *learned* manifold, so a noisy query is projected onto the manifold rather than snapped to the
nearest stored sample. The result is geometric, and that's the point. On a continuous manifold the
learned energy beats the fixed *soft* energy cleanup at every codebook size; and on a manifold of
dimension ≥ 2 it beats a matched-memory codebook of random samples (~0.43 vs ~0.49 in 2-D) — because
tiling a d-dimensional manifold with samples costs ~grid^d points, while a fixed-size learned projector
scales with the manifold's intrinsic structure, not its volume. The negatives are loud and define the
edge: for *discrete* atoms the hard 1-NN cleanup returns the exact atom (~0.02) and is unbeatable — a
learned approximate energy can't beat exact recovery, so that's the existing cleanup's job; and in 1-D
the curse of dimensionality doesn't bite, so a matched-memory codebook wins (~0.27 vs ~0.33). The
advantage over storing data is real precisely where storing data is expensive: manifolds of dimension
≥ 2. It is also the natural learned prior for the Plug-and-Play / RED denoiser the engine already runs.

**Try it.** `python unified_app.py` opens a console (http://127.0.0.1:5001) that PULLS a
real corpus on request -- Reuters categories, Brown genres, Gutenberg authors, or Europarl
languages, downloaded via NLTK from GitHub, the same place the test data came from --
trains one `UnifiedMind` on it, and lets you classify, recall the nearest stored item,
watch the memory organize itself into sub-prototypes, and generate new text in the
corpus's style, all against the one trained mind. A fifth dataset needs no network at
all and is the inception option: **"This project's own source"** -- the mind learns its
own code, classifies which subsystem a pasted snippet belongs to (held-out ~62% over
five subsystems sharing heavy vocabulary, vs 20% chance), and generates code in the
project's style. Getting that number honest transferred an old lesson to a new format:
pure-punctuation tokens are code's stopwords -- shared by every module, they dilute the
bag exactly like prose stopwords dilute topics, and dropping them lifted held-out
accuracy from 42% to 70% in the controlled comparison (generation keeps its punctuation;
code without parentheses is not code). Queries in this dataset also keep their case
(`HoloTree` is not `holotree`) and run untagged, so the mind self-discovers what it is
being shown.

**Self-discovery and self-assembly** were the next two gaps, and closing them surfaced
one re-measured trap and two clean negatives -- all kept. *Self-discovery:* the routing
safeguard used to depend on the caller's bookkeeping -- pass no modality tag and it
silently vanished. Now the encoder itself names the modality it would use
(`UniversalEncoder.infer`, the single source of truth `encode` also dispatches through,
so the tag used for routing can never disagree with the encoding used), and
`learn`/`classify` discover untagged inputs. The trap: naive type dispatch sends a LIST
of tokens to the order-sensitive sequence encoder -- the *same* encoding bug fixed once
already, sneaking back in through inference -- and scores 93.8% on the mixed-modality
demo. With the one rule that a list of strings is a bag of words, inferred routing
scores 97.5%, *exactly* matching caller-declared tags. *Self-assembly:*
`UnifiedMind.absorb(examples)` builds a working mind from a bare pile of
`(input, label)` pairs -- discovers each item's modality, pre-reads the text it sees so
word vectors carry co-occurrence meaning before anything is filed, learns everything
into the one memory, and runs a maintenance pass; it is deliberately sugar over
`read`/`learn`/`maintain_now` so there is nothing to drift out of sync. The negatives:
wiring the learned *navigator* into unified recall lost badly on the mind's own store
(48% recall@1 at ~130 comparisons, where the fixed-beam forest gets 89% within ~512 --
its arrive/keep-moving margin sense was tuned on uniform random vectors and the unified
store is clustered), so recall keeps the dumb-but-honest index and the navigator stays
a study. And the recall index's *switch-over to the forest* turned out to be 16x too
eager: measured, a single numpy matmul scan is exact AND faster than the tree's
Python-level routing until roughly 4,000 items (at the old 256-item threshold the scan
is ~7x faster and the forest already costs recall), so the threshold now sits at the
crossover -- below it the forest paid more wall-clock for less accuracy.

**Structure before meaning, on a new format.** The destination demands that the same
machinery handle text, code, images, and more by discovering each format's *structure*
first -- and the schema module always claimed its compress-by-merging move was
modality-blind ("idioms and indentation in code"), but code had never been measured. It
now is, and the corpus is -- recursively -- this project's own source: ~500k characters
of it. The discovered schema cuts held-out bits/char from 2.98 (flat character model) to
2.28 (the fractal coder), 24% fewer bits, the same shape of win it earned on Austen. The
emergent chunks, found with no labels from raw characters, ARE Python syntax:
`def __init__`, `rng = np.random.default_rng(`, `)\n        return `, whole indentation
idioms, the banner comment lines. And a compression gate can tell code from prose --
each schema claims its own held-out format -- with one honest caveat that is itself a
finding: feed the code expert this project's RAW source, which is half English
docstrings, and it becomes a *better English model* than a prose expert trained on less,
so the gate mis-routes. Representative corpora are part of the mechanism, not an
optional nicety. (Both results are locked in as tests.)

The third format closed the set: the same primitive on **images**, measured on the
project's own 712-sprite set (each pixel an opaque colour-code atom in raster order). The
honest shape of this one is that the schema is *data-hungry* here: at 60 training sprites
the rare chunks starve and the fractal coder LOSES to the flat pixel model (1.91 vs 1.96
bits/pixel); at 150 sprites it wins on every split tried (1.49 -> 1.30, and 23% fewer
bits at deeper settings -- the same magnitude as text and code). Structure exists in the
format; feeding the statistics is part of the mechanism. And the unified mind now *uses*
the format work: `learn_sequence` takes a modality (so it learns to continue code, not
just prose) and a name, so one mind holds MANY sequence schemas at once instead of a
single slot that silently overwrote. Unnamed generation routes the seed through the
compression gate -- whoever compresses the seed best understands it -- which is
content-level self-discovery, needed exactly where type inference goes blind: code and
prose are both `str`. Measured: a `def encode(self, x):` seed routes to the code schema
and continues as code; a prose seed routes to prose. One routing primitive, reused at
every level of the stack.

The same gate then moved into **classification**, and the measurement that justified it
was a surprise: not a booster, a *correctness fix*. First the encoder needed `"code"` as
a first-class text-like modality -- before that, a declared code tag silently fell to the
opaque-symbol path and two nearly identical snippets encoded as orthogonal (measured
cosine 0.04). With that fixed, one mind learned documentation and source code about the
*same* subsystems (heavy shared vocabulary -- the adversarial case) into the one memory.
The finding: routing's gain over a flat scan is ZERO on this data (bag-of-token vectors
already separate docs from code -- the safeguard story again, one level down), but the
old type-only path was actively destructive -- tags declared at learn time put code
labels in a "code" pool, an untagged classify inferred "text", and the routing safeguard
then *excluded the true labels from competition entirely*: 24% accuracy, 66% cross-pool
leakage, worse than no routing at all. The compression gate, fitted on the mind's own
learned samples (capped, refit only when the corpus grows by a third), identified the
sub-format on 100% of held-out queries and recovered declared-tag accuracy exactly --
~2s one-time fit, ~10ms per untagged string query at steady state. So `classify` now
discovers in two stages: type (`encoder.infer`) for everything, then content (the gate)
exactly where type goes blind. All pinned as tests.

**The slime mould moved into recall** -- the organize half of the navigator study,
salvaged from its negative. The learned navigator lost its place in unified recall
(recorded above), but its `ReflexCache` -- veins thicken toward what you ask most, with
a flux guard so the habit never costs more than it saves -- is separable, and it now
fronts the recall index's big regime (it moved to `holographic_tree`, since a reflex
fronts index machinery, not navigators). Measured at 16k items: on a Zipf workload the
reflex answers 70% of queries, recall@1 *rises* 96.8% -> 99.0% (a popular noisy cue
snaps to the right hot item where the forest's beam sometimes misses), and cost drops
3x; under a popularity *shift* it re-adapts within one rebuild period; on a uniform
stream the flux guard deactivates it and the cost is a wash -- measured, not promised.
Integrating it surfaced a separate embarrassment worth recording: every big-regime
recall had been re-stacking the entire store into a matrix at a measured **54 ms per
call** at 16k items -- thirty times the cost of the search it was preparing for. The
matrix is now cached, so the whole path went from ~56 ms to 0.5-1.7 ms per query.
And `absorb(pile, sequences=True)` completes self-assembly: the one call now returns a
mind that classifies, recalls, AND generates -- one named sequence schema per text-like
sub-format it discovered, fitted from the same capped samples the classify gate uses,
with unnamed generation routed by the same compression gate. The tour's closing segment
runs exactly that.

---

## Quick start

### Windows
Double-click **`run.bat`** - it installs dependencies on first run, starts the
server, and opens `http://127.0.0.1:5000`.

### Any platform
    pip install -r requirements.txt
    python app.py            # then open http://127.0.0.1:5000

If `pip install` fails with **"Access is denied" / WinError 5** (a system-wide
Python you can't write to), install into a local virtual environment instead -
which is exactly what `run.bat` now does automatically:

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python app.py

(or add `--user` to the `pip install` to install just for your account.)

The web UI has these panels: **System tour** (runs every subsystem once),
**Compression & speed** (single- and many-file size, encode/decode timing, and corruption resilience vs JPEG/PNG),
**Batch operations** (superposition capacity, 1-bit memory, cleanup throughput), **Creature** (trains the forager
with lethal poison present, then animates its random-vs-trained life step-by-step -- with a live energy bar and star
tally; an Obstacles mode adds walls to route around with the optimal route drawn
in, and a Labyrinth mode has it learn the way out of a maze -- all on a prototype memory built from the same classifier),
**Test suite** (runs the full pytest suite), **Query & recall** (the interactive image demo - degrade an
image, optionally destroy part of the plate, watch it get recalled), **Recall by description**
(cross-modal recall - describe an image in words and get the matching one back from the tag address space),
**Set packer** (delta-code a set of related images against one reference), and **Image vault** (the general store: relate by fingerprint, compress adaptively across lossless and lossy encoders with an honest table, and query by example). The Test suite panel auto-discovers and runs every test_*.py (947 at last count; up to six skip without NLTK or its downloaded corpora). The package also ships the real 712-sprite set packed to ~67 KB at `features/sprites.hsp` (which doubles as a live demo of the sprite packer), and the UI uses it in two places: the Image vault runs relate/compress/query on the whole set, and the learning creature is drawn as a real walking sprite (`amg2`) that turns to face the direction it moves and cycles its two walk frames -- with its baked-in background keyed out (flood-filled from the edges) so it shows real transparency over the grid instead of an opaque tile. The creature also runs on an energy mechanic: it starts each life with 100 energy, every step costs 1, each star it reaches gives +3, and stepping on poison empties the battery -- instant death -- so collecting stars and staying alive are the same goal. Finally, a **Vision** panel shows that the image is just numbers: RGB->HSV colour and dominant-colour extraction, Sobel edges with Hough line/circle detection and Harris corners, a geometric shape classifier, and unsupervised *emergent* classes that fall out of clustering simple feature descriptors -- then a VSA prototype classifier (bundle + cosine cleanup) labels held-out shapes, tying the vision work back to the holographic engine. The panel reports each step's accuracy honestly, including where unsupervised clustering tops out. A final **Compositional scene** panel takes the opposite stance to a holistic descriptor: it reads the DCT coefficient layout as a texture tag (finally using the DCT as a feature, not just for compression), pairs it with HSV colour and geometric shape for automatic per-object tags, then encodes each object as a product of attribute atoms and a scene as their superposition -- so a ResonatorNetwork can factor the parts back out (and that resonator now takes an optional softmax-sharpened cleanup -- the SBC readout lesson swept down -- which recovers more at high codebook load; default off, so the small-vocabulary scene case is unchanged). Multi-object scenes now decompose reliably up to ~5 objects: the old ~50%-at-three ceiling turned out to be a scale bug (normalising the scene) plus missing refinement, not a real capacity limit -- keeping the scene as an unnormalised superposition and adding coordinate-descent sweeps recovers 3-4 objects at 100%. A **Scaling** panel confronts the deepest limit head-on: one holographic trace is a bundle with finite capacity (a 2048-d memory recalls 100% of 64 pairs but ~0% of 2048), so instead of one flat store it grows a deterministic recursive tree -- each node a seeded random hyperplane splitting items at the median, each leaf a small memory kept inside capacity, queries descending with a beam that back-tracks into nearby cells. This is the random projection tree of Dasgupta & Freund and, in spirit, how slime mould beats the size limit of pure diffusion by resolving a broad mass into a hierarchical vein network. The flat memory collapses with scale while the tree holds 100%, and search reaches ~96% recall at a fraction of a full scan's comparisons; per-leaf query 'flux' shows the thick-vein / thin-vein structure. A HoloForest of several differently-seeded trees breaks the single tree's recall ceiling, reaching ~100% recall at a fraction of a full scan's comparisons. Finally, a **Content addresses** panel realises the original partitioning idea the way AWS S3 does: no folders, just a flat keyspace where each object's name encodes the hierarchy. The auto-tags (colour/shape/texture) generate a deterministic URI like `red/circle/smooth`, the key *is* the partition path, and a FacetStore supports S3-style prefix listing and CommonPrefixes roll-up. Where the RP-tree splits by meaningless random hyperplanes, this splits by meaning -- readable, queryable paths -- at the honest cost of bucket skew, with key depth as the lever. And the resonator closes the loop: it recovers an item's URI from its content vector alone, so the address is computed from the content. And the skew problem is now handled: build_indexes gives any hot bucket its own in-bucket HoloForest, so content search inside a popular prefix stays sub-linear -- the bi-level structure (semantic prefix outside, geometric forest inside) realised.

### From the command line
    python tour.py                    # guided tour of all subsystems (~20s)
    python holographic_creature.py    # any module runs its own demo
    python holographic_encoders.py    # numbers / text / records demos
    pytest -q                         # the whole test suite (947 tests)

---

## What it can do (with results from `tour.py`)

Everything below lives on the *same* vector substrate and the same memory.

**Numbers.** Encode a real value as a unit vector so nearby numbers get nearby
vectors, then read the number back out - even from a noisy vector.
`decode(encode(7.2)) = 7.19`; a noisy vector of `4.0` still decodes to `3.92`.

**Text.** Learn word vectors from raw co-occurrence (no gradients, no labels).
After a few passes over a tiny corpus: `cat ~ dog = 0.77` but `cat ~ car = 0.36`;
nearest word to `truck` is `car (0.88)`. The geometry carries meaning.

**Mixed records.** Pack a number, a category, and a free-text note into one
2048-d vector and read each field back individually: `price -> 140.7` (stored
142.5), `trend -> up`. Record-to-record similarity reflects all fields at once.

**Key -> value memory.** Store many `key -> value` pairs superposed in a *single*
vector and recall them by content, with cleanup snapping noisy results to the
nearest known symbol. A handful of `country -> capital` facts in one 1024-d
vector, all recalled correctly; it degrades gracefully when overloaded. And the
overload point moves: `recall_all` adds **successive cancellation** -- peel the
single clearest pair, subtract `bind(key, clean_value)` back out of the trace,
and the residual gets sharper for the rest; repeat (recursively) until the whole
"exposure" is developed. This roughly doubles how many pairs come back cleanly
from one trace (at 1024-d, 80 pairs go from 84% one-shot to 100%, and 160 from
42% to 70%), the way a decoder cancels interference or film develops the
strongest signal first. It is orthogonal to partitioning, so the two COMPOUND:
peeling *inside* each of 8 regions recovers 100% of 320 pairs where a single
one-shot trace gets 16% and peeling-alone has collapsed -- two compression
filters stacked, gist-then-residual. The one honest catch is error propagation:
once the clearest remaining guess is itself wrong, subtracting it injects noise,
so peeling rescues a trace inside a sane regime and partitioning is what keeps it
there. (`python holographic_ai.py` prints the full table.)

**Reasoning.** A resonator network factors a composite vector
(`subject (x) relation (x) object`, three things bound together) back into its
parts knowing only the vocabularies - recovering 6/6 facts where a single unbind
cannot. Also included: split-conformal error bars, an epistemic "how well do I
know this" map, and a semantic compass.

**A learning creature.** A grid-world forager with a purely holographic mind
(perceive -> decide -> learn by remembering experiences; no neural net). From a
random baseline of `-0.27` reward / `0.2` stars, after 120 episodes it reaches
about `+9.1` reward / `9.6` stars - it taught itself to find stars. It plays an
energy game: it starts with 100 energy, each step costs 1, each star gives +3,
and poison is lethal (energy to 0, instant death), so chasing stars and staying
alive are one and the same. The full demo in the UI also shows poison avoidance
-- it learns to collect stars AND route around the hazards, surviving most
worlds where a random walker dies. Two obstacle modes sit on the same machinery:
**Obstacles** scatters impassable walls into the forage world (kept connected,
so it is always solvable) and draws the BFS-optimal route behind the creature's
learned one -- here a working memory of recent moves collects ~1.6x more stars
than a reactive brain, because the same trick that helps when blind also helps
when the straight line is blocked. **Labyrinth** carves a fixed 7x7 perfect maze
and the creature learns the single way out over repeated tries (the classic
rat-in-a-maze), escaping in ~16 steps against an optimal of 16. The honest limit:
a 9x9 maze (a ~28-step solution) is mostly beyond this brain, because far-apart
corridors look identical through its egocentric senses. The module demo
(`python holographic_creature.py`) adds a working-memory scene with limited
vision plus the walls-and-labyrinth scene.

The creature's memory is the same holographic kit the image side uses, not a
separate store. Rather than hoarding every experience (it meets the same
egocentric situation -- "star east, wall north" -- in thousands of different
cells), the brain keeps one **prototype** per distinct situation: a new
experience is cosine-matched to that action's prototypes and either joins the
nearest class or starts a new one (the same bundle-and-cosine **classifier** the
vision panel uses), and each prototype is a **superposition** of its members
with a denoised mean return (the "superpose a gallery into plates" trick storing
experience instead of pixels). Because averaging the returns cancels the noise of
early exploration, the compressed memory is usually a *better* value estimate,
not just a smaller one: the reactive forager folds ~7,700 raw experiences into
~350 prototypes (about 22x) while foraging better, and the working-memory modes
compress 3-7x. Those prototypes are content-addressable, so `demo_introspect()`
indexes them with the same **recursive** HoloForest the image vault uses and
recalls the most similar past situation from a noisy cue in roughly a tenth of a
full scan. Honest caveat: that approximate index is for associative recall, not
the control loop -- choosing a move needs the full weighted neighbourhood, and
the approximation drops enough of it to wreck the maze policy, so deciding still
uses the exact scan, which the prototype compression already made cheap.

We tried the rest of the toolkit on the creature too, and kept only what
measurably helped -- the honest part of "use everything." Putting the recursive
forest in the *decision* loop dropped maze escapes from 93% to 0% (approximate
recall loses neighbours the value estimate needs). Deepening the working-memory
window to crack a 9x9 maze did not work (0% at both shallow and deep memory) and
at depth 10 it ballooned the prototype set to ~23k by making every state unique,
so generalization collapsed. We recorded that as "a real ceiling for an
egocentric brain, not a tuning miss" -- and the revision of that claim is part of
the record now: it was not a tuning miss, but it WAS a framing miss (see the
maze gauntlet below; the ceiling fell to 100% without changing the brain, the
senses, or the memory -- only WHERE decisions are spent). Averaging an *ensemble* of independently
trained minds was worse than picking the best single one (their policies differ
too much to average), which is why the UI trains several candidate minds and
keeps the best -- a branching search over policies that beats voting. And the
newest entry: **schema-discovered macro-actions** -- letting the compress-by-merging
schema read a *trained* creature's own trajectories (episodes joined by unique
separators so merges never cross a boundary) and handing the discovered idioms to a
fresh learner as extra actions. The discovery itself works perfectly -- the emergent
chunks are exactly the straight-line runs (`E+E+E`, `W+W+W+W`) a forager's behavior
contains -- but using them LOSES, robustly: open-loop macros drop the clean world
from 9.9 to 6.8 stars and the poison world from 6.3 to a catastrophic 2.5 (blind
commitment walks into poison); making them interruptible (stop on a star or sensed
danger) rescues the catastrophe but still loses everywhere (5.7 and 5.5), across
three seeds. The why is principled: a reactive policy that senses every step can
already produce `EEE` by deciding `E` three times -- the chunk only adds value where
deciding is expensive or perception is poor, and here it throws away exactly what
the per-step sense-decide loop provides, while doubling the action set thins the
exploration statistics. Same lesson as the curation controller: discovered
structure must beat what the substrate already does, and here it does not. The net:
the toolkit lives in the creature where it belongs (a classifier and layered
superpositions for the memory, a recursive branching partition to index it), and
stays out of the places it hurts.

**The maze gauntlet -- gamified debugging.** The mazes are now designed to mirror
challenges the system itself faced, on the principle that a puzzle the creature cracks
without cheating usually carries a lesson back to the brain -- and the first lessons ran
the other way, system to creature. The 9x9 maze ceiling is ALIASING in a costume:
far-apart corridors look identical through egocentric senses, exactly as code and prose
both look like "text". Two system cures were tried, three seeds each, same senses,
nothing global. The COMPRESSION cure (a decaying bundled trace of past actions, the
anti-23k-prototypes move) is a clean negative at every decay tried -- 0% even on the 7x7
that exact mem=4 solves at 97%, because permute-by-age ORDER is precisely the
information that breaks aliasing and bundling erases it; compression must preserve the
distinctions the task needs (the nearest-key-generation lesson again). The
DECIDE-ONLY-AT-CHOICES cure won completely: a corridor reflex (`run_episode(...,
corridor_reflex=True)`) auto-walks forced cells using nothing but the wall senses, so
the brain spends decisions and credit only at junctions. The diagnosis is quantitative:
per-step framing discounted a 26-step exit to gamma^26 ~ 0.07 at the first decision --
nearly invisible -- while junction granularity puts it near 0.4, learnable. Escapes:
9x9 0% -> 100%, 11x11 100%, 13x13 67% (and training runs ~10x faster, since an episode
is ~8 decisions instead of 90). The honest control is pinned in the tests: corridor-
following with RANDOM junction choices already escapes easy mazes (73% at 9x9 -- a
perfect maze has a small junction graph), so the gauntlet requires the brain to BEAT
that control, which it does at every size and triples at 13x13 (15% random vs 67%
trained). The transferable lesson runs both ways: where the system spends its
machinery only at real choice points, the brain must spend its decisions the same way
-- and the credit-horizon arithmetic says WHY ceilings appear when it does not.
`test_creature_gauntlet.py` holds all of it in place.

The gauntlet's second round added LOOPS and HAZARDS, and its first catch was a bug in
the first round's winner. **Braided mazes** (a fraction of dead-ends opened, so multiple
valid routes exist -- the maze costume of competing reorganization candidates) are where
corridor-following can cycle and the brain must add real routing on top: measured at
11x11 braid=0.5, the no-reflex baseline escapes 0%, reflex+brain 100%, and the
reflex-with-random control 50% -- the brain doubles the control. **The poisoned fork**
(braided maze plus hazards, each placed only if a poison-free route to the exit still
exists, so the maze stays honest: solvable, but one arm of some fork is lethal and looks
like the safe one) mirrors confusable classes with asymmetric cost -- and designing it
exposed that the corridor reflex would auto-walk the creature into poison it could
*see*: the fast path had no anomaly handoff. Measured, naive reflex with a trained
brain: 7% deaths, 93% escapes; with a one-line danger yield (the reflex returns control
to the brain when the way forward is sensed as danger): 0% deaths, 100% escapes, three
seeds -- while the random control with the same yield died 88% of the time, so the
brain's contribution stays enormous. The lesson is the flux guard's, running in both
directions: every fast path in the system -- the reflex cache, the format gate, and now
the corridor reflex -- must hand back control at anomalies, and the gauntlet is where
that class of bug gets caught in costume before it gets caught in production.

**The 16x16 room, any seed.** The escalation demanded reliability with no cherry-picked
maze and no map knowledge -- the creature only ever has its senses, learning each maze
the way a rat does, by living in it. Three walls fell, each one a system lesson. First,
ENERGY: a 16x16 maze's optimal path runs 80-108 steps against the then-default battery
of 100, so starvation beat intelligence on most seeds -- the budget must match the world
before the brain even gets a vote. That finding raised the creature's default battery to
300, and the whole any-seed sweep re-validated at the new default (same worst 95%, mean
99%) with no explicit energy override anywhere. Second, the CREDIT HORIZON arithmetic struck again one level
up: at gamma=0.9 the training is bimodal -- runs land at ~100% or collapse to 0% with
nothing between, the brain committing early to a wrong junction policy and then greedily
cycling it to death; gamma=0.97 took the failing maze/brain combinations from 1% to 98%
mean, and IMPROVED the smaller gauntlet mazes too (13x13: 67% -> 100%); epsilon floors
and longer training did nothing -- the horizon was the lever. Third, the stray collapse
that remained (15-run grid: mean 93%, worst 0%) is closed by SPECULATE-MEASURE-ADOPT
over whole policies: `learn_maze()` trains a candidate, probes its real escape rate over
a few greedy lives, and restarts with a different brain seed if it measures incompetent
-- the organizer's rule applied to training runs, and the same train-several-keep-the-
best pattern the UI already uses. Validated across 8 maze seeds: worst 95%, mean 99%,
with the restart visibly rescuing the nastiest seed. The honest frontier is recorded
with it: ZERO-SHOT transfer -- one mind trained across 30 mazes, dropped into 10 it
never saw -- measured 41% against a 36% random-junction control, i.e. no real
competence transfers; the learned junction policy is maze-specific. Earned per maze,
reliable on any seed; portable across mazes is the open problem.

**Survival foraging -- fair, and harsh on purpose.** Foraging and obstacle worlds now run
until the creature DIES (poison or starvation; `max_steps=None`), with the score being
stars collected in a life -- the energy arithmetic keeps every life finite (stars +3,
moves -1, average star ~4.7 steps away, so even a perfect forager runs a slow deficit
from its 300 battery). The fairness is in the baselines, which use the creature's exact
senses: a naive greedy chaser, and a danger-aware greedy chaser that is the bar learning
must clear. The harshness did its job immediately, three findings deep. FIRST, the
bombshell: the trained brain -- whose poison avoidance looked solid in every 50-step test
-- died on poison in 67-73% of full lives (a ~0.6%/step residual risk that short caps
simply cannot see; risks COMPOUND), collecting 13-25 stars where the two-line
danger-aware reflex collected 136 with zero deaths. Fixed by the danger reflex: lethal
moves are vetoed BELOW the brain (`decide` gained `among`, the routing lesson applied to
actions -- compete only within the survivable pool), because irreversible mistakes are
reflex business, not learned preference (the corridor reflex's danger yield and
auto_maintain's asymmetric-cost rule, a third time). Deaths: 0%. An honest negative on
the way: blinding the brain to the danger senses it no longer needs did NOT recover
efficiency (and got worse with training). SECOND, the real thief was DITHERING: the
memoryless forager spent a measured 60% of its steps stepping back where it stood two
steps before, starving at 28 stars; working memory (mem=3) cuts dithering to 10% and
lifts it to ~121 stars -- 89% of the danger-aware reflex's ceiling, the same ratio it
achieves in the clean world, so what remains is chase efficiency, not poison. THIRD, the
open problem -- recorded, then SOLVED by the system's own introspection. The brain
gained describe()/why_differ() (the relations decode turned inward: its states are
role-bound sense bundles, so a prototype reads back out in sense terms -- measured
373/373 present roles decoded, 427/427 absent roles correctly silent). Pointed at a
caught dither, the brain articulated its own bug with precision: the two 'oscillation'
states were sense-IDENTICAL, and it was choosing E at value +0.43 while its own senses
said wall_E=yes -- valuing moves into walls it could see, burning energy on no-ops. The
articulation named the fix: walls join poison in the `among` veto (the wall reflex).
Measured, three seeds: stars 5.1 -> 19.8 (the danger-aware reflex's ~20 ceiling),
dither 79% -> 43%, deaths 0%. Found by introspection, fixed by one line, measured
before believed.

**The creature, repurposed: a navigator over the data.** The grid was always a
testbed for a mind that perceives, decides, learns from what happened, and
adjusts -- and the rest of this project quietly built the *world* for it: a
recursive HoloTree that partitions data into branching regions, exactly like a
maze of corridors. `holographic_navigator.py` closes the loop ("inception"): the
*same* `HolographicMind` and `CreatureEncoder` that found the star now learn to
navigate the index to find what you asked for. The map is literal -- a cell it
stands on becomes a region (tree leaf) it is examining; "food is to the east"
becomes "the best match so far is strong / weak"; stepping toward the star
becomes examining the next-most-promising region; "I reached the star" becomes "I
have arrived at the answer, commit." The point is not to re-implement the tree's
routing but to spend its effort *adaptively*: a fixed beam reads the same number
of regions for every query, but difficulty varies enormously (here ~40% of cues
find their true neighbour in the very first region, while a third need four or
more), so a fixed beam overpays on the easy majority. The navigator senses how
confident the answer looks -- the margin between the best match and the runner-up
-- and decides arrive-or-keep-moving, learning to commit at once on easy queries
and search hard only on ambiguous ones. On a 2000-item tree it reaches **94%
recall at ~120 comparisons, matching the widest fixed beam's 96%/500 at ~4x fewer
comparisons**, from a policy of ~50 prototypes -- the same find-it / keep-looking
instinct that solved the maze, now buying retrieval efficiency. (It is trained
against an exact-scan ground truth, then deployed using only its learned senses;
this is the *access* half of "organize and access", and the natural next step is
to let the same loop decide where new items should live.)

And the organize half follows from the same instinct. Real query streams are
skewed -- a few items get most of the traffic -- so the navigator grows habits:
a small `ReflexCache` of the items it commits to most, checked *before* it
descends the tree, exactly the way slime mould thickens the veins it travels
often and lets the rest wither (and the way the engine's ReflexArc lets a
familiar input skip the expensive path). A use-reinforced hot set recognises
popular queries instantly, and a flux guard prunes the habit on an unpredictable
stream so it never costs more than it saves. On a skewed (Zipf) workload it cuts
the average query from **~125 comparisons at 96% to ~84 at 99%** -- recognising
most queries on sight -- while on a uniform stream the veins are pruned and it
falls back to the plain cost. Because each `find` can narrate its path, the
navigator has a literal train of thought: a familiar query returns
"recognised instantly -- a familiar query", an unfamiliar one walks the regions
("best match 0.18, a tie -- look further ... 0.90, clear -- arrive here"). The
little navigator in the data world, thinking out loud. (`python
holographic_navigator.py` runs both halves.)

**One shared encoder for any input.** Everything above rests on a single
fact: once you encode something into a hypervector, the same operations work on
it -- `bundle`/`bind`/`cosine` do not care where the vector came from. So the
hypervector is a universal interchange format, and the machines built on it (a
prototype classifier, the recursive index, the creature's brain) are a component
library that snaps onto *any* encoded input. `holographic_mind.py` holds the
shared pieces: a `UniversalEncoder` that turns text, numbers, categories, raw
feature vectors (an audio MFCC frame, an image embedding), images, structured
records (a dict of fields), or sequences into one unit vector in one shared
space -- and that can *name* the modality it would use (`infer`), which is what
lets the unified mind discover an input's kind instead of being told (see
below). This file used to also hold a `Mind` facade with an `assemble()` that
guessed the task from data shape; it was retired, because it re-implemented thin
versions of machinery that exists for real elsewhere -- exactly the failing
`UnifiedMind` was built to fix. Its one good idea, building a working mind
straight from a pile of examples, lives on as `UnifiedMind.absorb()`, running on
the real self-organizing memory instead of a toy one.

**A holographic mixture of experts, with a learned gate.** The general Mind routes
by a rule -- which verb you called, what type the input is. A true mixture of
experts needs the missing piece: a *learned* gate that, per input, decides which
specialist to trust, trained from outcomes. `holographic_moe.py` adds it, and the
gate is the creature's own brain -- encode the input, let the brain `decide` which
expert to consult, reward it when the expert it chose was right. Sparse top-1
routing (only the chosen expert runs), learned with no gradients, by the same
perceive/decide/remember loop that learned to forage. This closes a loop opened
earlier: blind ensembling was measured to *hurt* when experts disagree, because a
confident-but-wrong specialist drags the average down. The answer is not to mix
but to *route* -- and the gate learns to. With three cross-modal specialists
(text, image, audio) that each know only their own labels, the gate sees only the
encoded vector, never the modality, and still learns to send each input to the
right expert: **100% accuracy, matching an oracle router, against 43% for the best
single expert and 21% for random routing**. Given two experts that own different
halves of the number line (same modality), it learns to route by *value*, reaching
92% where any single expert caps at 50%. So the brain genuinely learns to route
from reward -- the capability is real.

But the honest comparison demands a gate-free baseline, and it changes the verdict:
just route to whichever expert is *surest of itself*. For a bank of holographic
specialists this is hard to beat, because an unfamiliar input naturally produces
low similarity -- "I don't know this" falls out for free -- so confidence already
tracks competence. Measured against it, confidence routing matches the learned gate
on the cross-modal task (100% vs 100%) and slightly *beats* it on the number-line
task (100% vs 92%, the gate's boundary error). So the finding is: a learned gate
works, but it is not needed for a homogeneous bank of well-calibrated holographic
experts -- confidence routing is simpler and at least as good. A learned gate earns
its keep only when confidence is unreliable: heterogeneous or miscalibrated experts
that are confidently wrong out of domain (the classic overconfident-model failure),
which is outside what this homogeneous bank can show. This is the same lesson the
forest-in-loop and the ensembling experiments taught -- keep the machinery only
where it measurably beats the simpler thing. (`python holographic_moe.py` prints
every baseline side by side.)

And there is a regime where it does beat it, which completes the story. Drop one
*heterogeneous* expert into the bank -- a linear+softmax model instead of a
holographic one -- and it behaves the way most real models do off their home turf:
it extrapolates with growing logits and is *confidently wrong*. On a task split
into two regions, where a calibrated holographic specialist owns one and the
overconfident linear specialist owns the other, the linear expert reports ~90%
confidence on the region where it is always wrong. That breaks the gate-free
heuristic -- confidence routing drops to ~72% -- while the learned gate, routing by
reliability it discovered from reward and ignoring confidence magnitude entirely,
holds at **100%, matching the oracle**, against 52% for either single expert. So
the full verdict: a learned gate is unnecessary for a homogeneous bank of
well-calibrated holographic experts, and exactly what you want the moment an expert
can be confidently wrong -- the normal case once the experts are heterogeneous.
(`demo_heterogeneous` in the same file shows it.)

**A self-organizing memory that reorganizes a shadow copy and swaps it in.** The
hard part of a system that keeps learning is keeping its data ORGANIZED as it
arrives. A holographic class is stored as a bundle, which is fine until a class is
several things at once -- "vehicle" is cars and trucks and motorbikes, in different
directions of the space. Bundle those into one prototype and you get their average,
a point that is none of them; on genuinely multi-modal classes a
one-prototype-per-label store collapses (measured: 49% where the structure allows
100%). `holographic_organizer.py` fixes this the way databases and CPUs do
in-place updates safely -- read-copy-update / double buffering. A small team of
organizer experts builds a SHADOW copy of the store from an experience buffer: a
`SplitExpert` discovers how many modes each label really has (raising k only while
it buys coherence, so a one-mode class stays one prototype and a three-mode class
becomes three -- chosen from the data, the self-classifying step), and a
`MergeExpert` folds away near-duplicates and flags cross-label collisions. The live
model is never touched during the build, so a query mid-reorganization sees a
complete, consistent store; then a single atomic SWAP makes the reorganized copy
live. Streaming multi-modal data, a naive one-prototype store sits at ~50% while
the self-organizing one climbs to **100% after each reorganization**, having found
the two modes per class on its own; and the swap is verified non-destructive (the
live model's answers are identical while the shadow is built, changing only at the
swap). This is the scaffolding for the self-* goal: self-learning (it absorbs a
stream with no training phase), self-organizing (it restructures its own memory on
the shadow and swaps), self-classifying (it discovers the sub-categories inside each
label). (`python holographic_organizer.py` runs it.)

And it can pull the trigger itself, which is what makes the cold-start problem
tractable. A system that starts blind has nothing to classify against; it files its
first data into immature prototypes and only later has enough to see the real
structure -- so early data ends up in the wrong place. A `TriggerExpert` watches two
signals it reads off the model itself -- *incoherence* (recent examples sitting far
from their own prototype: a class has gone multi-modal but is still one blurry blob)
and *novelty* (recent inputs matching no prototype: a new kind of thing has begun
arriving) -- and fires a reorganization with no schedule, only when a signal crosses
and enough new data has accumulated to be sure. Because the reorganization rebuilds
from the experience buffer, the early, badly-filed data gets re-placed in the swap.
Streaming two multi-modal classes and then introducing a third halfway through, the
self-triggering store goes from 52% to **100%, firing exactly twice on its own**:
once on incoherence (splitting the early blurry classes -- the cold-start data
re-placed) and once on novelty (organizing in the class that arrived mid-stream),
while a store that never reorganizes stays at 52%. And it does not thrash: on data
that really is one mode per class, coherence stays high and it never fires. The
reorganization can also recurse -- reorganize, re-check, and reorganize again while
it still buys coherence -- for structure that is modal at several scales. This is
the loop the self-* goal needs: notice your own organization has gone stale, fix it
on a shadow, swap it in, without being told when. (`demo_self_triggering` shows it.)

And, like the brain, it now does this with **no thresholds at all**. `auto_reorganize`
holds out a slice of recent experience, speculates a few organizations of itself at
different resolutions (one prototype per label, up to a few sub-modes each), and keeps
whichever *classifies the held-out slice best*, breaking near-ties (within one
standard error, read off the data) toward the fewest prototypes. Held-out accuracy is
the only judge, and it replaces both hand-set signals at once: a blurry cold-start
blob is beaten on accuracy by a split, so it splits; a new class that arrives
mid-stream is absorbed when a finer organization starts predicting it; and a class
that really is one mode ties at every resolution, so the single prototype wins on
leanness and nothing over-splits. On the same streaming cold-start-plus-new-class
test it reaches 100% (vs 56% for never reorganizing), choosing `k=2` to split the
early blur and `k=2` again to take in the new class -- matching, and here slightly
beating, the hand-tuned trigger, with nothing tuned. (`demo_autonomous_organizing`
shows it narrating each choice.)


**The same tools, turned on the brain itself (inception).** The brain that runs the
system -- the creature's `HolographicMind`, the same class the navigator and the MoE
gate are built from -- can go stale just like any other memory. It never forgets:
its bundles only grow, and near-duplicate prototypes (cosine below the merge
threshold) pile up. That redundancy turns out to be the exact thing that makes it
stale. When the world *shifts* -- the right action for a situation changes -- each of
those duplicates still holds the old value up, and an online update only ever touches
the single nearest one, so the value barely moves and the orchestrator cannot
unlearn. Measured plainly: after a regime shift where every situation's best action
changes, a plain brain is stuck near chance across thousands of steps (its old action
still reads 0.87 while the new correct one reads 0.19), carrying ~440 prototypes.
The fix is the data-organizer's own merge tool, run on the brain's value memory:
fold the duplicate prototypes into one, combining their returns by count. A single
prototype that every update touches *can* be unlearned where a cloud of duplicates
cannot -- so folding both compresses the memory (~440 -> ~70 prototypes, 6x) and
restores adaptation (recovery to 100%). And the brain triggers it itself, off two
signals it reads about its own state: *redundancy* (it has gone bloated) and
*surprise* (an EMA of how far its value predictions miss the returns it actually
gets -- which spikes the moment the world moves). With `maintain=True` it reorganized
itself six times over the run, unprompted, and ended both fresh and lean; with it off
(the default) the brain is byte-for-byte its old self.

And the last hand-hold is now gone too. `maintain='auto'` runs the whole thing with
**no behavioural thresholds at all** -- no surprise floor, no redundancy floor, no
fixed fold grain. Instead the brain keeps a window of recent experience and, every so
often, *speculates*: it builds a handful of reorganised versions of itself -- fold its
duplicates at a few grains (compress, forget nothing); rebuild from recent experience
(forget the stale regime) -- and measures each one the way that actually matters, by
the reward its greedy decisions would have earned on a held-out slice of that window.
A rebuild wins as soon as its decisions are better than the best fold -- it does not
have to win by a margin, because the costs are asymmetric: a needless rebuild in a
stable world just re-derives the same policy from still-valid recent experience and
costs nothing, while a missed rebuild after a shift strands the brain on a stale
policy. When a rebuild is chosen it is refit on the *full* recent window, not just the
slice it was selected on. Otherwise the brain compresses without forgetting, taking the
leanest candidate that is statistically as good as the best. One rule, and it does the
right thing in both regimes: while the world holds still it picks a fold (matching the
6x compression to 72 prototypes), and the moment the world shifts it picks a rebuild and
recovers to 100% -- with nothing tuned for either case. The only knobs left are resource
budgets (how large a window, how often to look), not behavioural thresholds.

That eager-commit rule is itself a correction. The first version made a rebuild win only
if it beat the best fold by a full standard error -- the same instinct that, in the data
organizer, made splitting candidates (trained on a fit slice) lose to a "keep" model
trained on all the data. On easy shifts it was invisible, but a hard, noisy, narrow-gap
shift exposed it: right after the shift the recent window is still half old experience,
which flatters the stale memory enough that the one-SE margin keeps the gate sitting on
"keep" for thousands of steps while the world has plainly moved -- the autonomous brain
crawling back on online relearning alone instead of committing. Dropping the margin (the
costs are asymmetric, so "better at all" is the right bar) and deploying the rebuild on
the full recent window roughly halves the recovery time on those hard shifts and fires a
rebuild where the old gate fired none -- while the stationary case shows no churn and no
loss of accuracy or leanness. The same conservatism, found in two different self-* gates,
fixed the same way: judge on held-out data, then commit and deploy on all the (currently
valid) data. This is the inception the self-* goal asks for, fully closed: the
orchestrator does not rot while the things it orchestrates stay fresh, and it does not
need a human to tell it when or how hard to clean itself. (`demo_self_maintaining` shows
the autonomous brain narrating each choice; the data organizer's trigger runs the very
same speculate-measure-adopt rule, judged by held-out classification accuracy.)


**Text, from a system that knows no language.** There is no dictionary here and no
grammar -- every word and every letter begins as a meaningless random vector. What
the engine can learn is the *statistical structure* of text, and `holographic_text.py`
shows that structure alone carries surprisingly far. It ships small original datasets
(topical sentences across cooking / space / sports / money, and a short four-language
set) and answers four questions, each measured honestly:

- *Can it learn?* Random indexing gives each word a vector that is the sum of the
  words it appears near. After reading the corpus, words used alike sit closer than
  words that are not (same-topic cosine ~5x the cross-topic one; `oven` lands nearest
  `cake`/`bread`, `ball` nearest `team`/`striker`) -- distributional meaning, with no
  labels and no training loop. On a corpus this small the signal is real but modest,
  and the writeup says so.
- *Can it analyze?* A character-trigram classifier identifies English / Spanish /
  French / German from raw letters at ~83% on held-out sentences, knowing nothing
  about any of them -- each language is just a direction built from its common
  letter-triples.
- *Can it organize?* Representing each sentence by the bundle of its learned word
  vectors, a prototype-per-topic classifier labels held-out sentences at ~92%, and
  clustering the sentences with no labels recovers the topics at ~82% purity. The
  unsupervised result leans directly on step 1: it only works this well *because* the
  learned meanings pull related sentences together (raw random atoms cluster at ~50%).
- *Can it produce?* A holographic character n-gram -- storing each context's next
  character as a superposition, read back by cleanup, backing off to shorter contexts
  -- predicts the next letter ~51% of the time (vs ~19% for always guessing a space)
  and the words it emits are ~100% real, despite working one character at a time with
  no notion of a word. The honest verdict: it reads like the training text up close
  and drifts into nonsense from afar -- it learned letter and word statistics, not
  meaning or grammar. (`python holographic_text.py` runs all four.)

The built-in datasets are deliberately tiny -- enough to read and test offline. To
see how far the *same* code goes on real text, `holographic_text.py` can also pull
public-domain corpora from NLTK (which hosts them on GitHub): Project Gutenberg
books, the Universal Declaration of Human Rights in many languages, and the
genre-labeled Brown corpus. Nothing about the algorithms changes -- only the amount
of text they learn from -- and everything gets sharper: language ID rises to **97%
across 11 languages**, word neighbours become genuine (`woman` -> `man, lady, ladies,
person`; `happy` -> `sorry, glad`), genre classification reaches **73% across five
Brown genres on 145 real documents**, and a 6-gram trained on *Alice in Wonderland*
predicts the next letter **62%** of the time with **96% real words**, generating
recognisable Carroll ("the cook and the mouse shook its head ... she was not a
serpent"). The scaling is opt-in (`pip install nltk`, one download) and the module
still runs fully offline on the built-in data without it. The limits stay the honest
ones -- distributional, not grammatical; an n-gram, not an understanding -- they just
arrive much later with real data. (`demo_text_scaled()` runs these.)

Generation in particular turns out not to be English-specific. The same character
n-gram, with nothing language-aware in it, was trained on five European languages
(European Parliament proceedings) and generates text that reads unmistakably as each:
**next-letter prediction holds at 63-68% across English, French, German, Spanish and
Italian, with 85-97% real words**, and the samples keep what makes each language look
like itself -- French accents, German compounds, Spanish endings. This is a capability
test, not a new mechanism (the generator is unchanged; only the text differs), and the
limit is the same honest one -- it spells and chains words plausibly without knowing
what any of them mean. (`demo_text_multilingual()` runs it.)

These text demos at first used a plain local k-means and single prototypes rather
than the self-organizing memory built earlier -- so the obvious question is whether
that heavier machinery helps here. Wiring text into it (a new `observe_vector` front
door lets the memory ingest the sentence vectors it did not encode itself) and
measuring gives a clean, honest negative that is itself the point. On the topic
classifier the autonomous memory scores **92%, identical to one-prototype-per-topic,
and keeps exactly one prototype each (zero reorganizations)** -- while naively forcing
three sub-prototypes per topic also scores 92% but spends 3x the memory. Text topics
are linearly separable, so a single prototype per class is already optimal, and the
autonomous memory *measures* that and refuses to split. The same holds on the
heterogeneous Brown genres (it keeps one prototype per genre, matching the baseline).
So the self-organizing machinery's contribution on text is discipline, not accuracy:
the same self-* logic that splits genuinely multi-modal data here verifies that text
does not need splitting and declines to over-engineer. (For blind clustering the plain
k-means stays ahead; the coherence-driven split is fragile on these embeddings -- a
limit worth stating plainly. `demo_text_self_organizing()` shows the comparison.)

That clean negative held only because the topics were easy. Pushed onto genuinely HARD
text -- the Reuters financial categories, whose vocabularies overlap heavily
(crude / trade / money-fx / interest all read alike) and whose classes are internally
multi-modal -- the picture flips. A single averaged prototype mis-files the off-centre
members (~77% across seeds), splitting each class into ~3 sub-prototypes genuinely
helps (~82%), and the autonomous memory now *fires*: it measures the gain on held-out
data and reorganizes, reaching ~81% (76.8% -> 80.7% averaged over seeds). The hard
problem also exposed a real bug: the gate had been comparing split candidates (trained
on a fit slice) against a "keep" model trained on ALL the data, so the splits were
handicapped and the gate under-fired. Fixing it to judge every resolution on equal
footing and then refit the winner on all the data made the autonomous version fire
reliably -- a fairness fix that only a hard, confusable dataset would surface. And the
discipline still cuts both ways: on sentiment (movie reviews), where the failure is the
representation carrying no good/bad signal rather than multi-modality, splitting cannot
help and the memory correctly declines. So the honest full picture is: the machinery is
inert on easy, linearly separable topics (and says so), earns its keep on hard
confusable ones (and fires), and refuses to chase problems splitting cannot fix.
(`demo_text_hard()` runs the Reuters and sentiment stress tests.)

One more question the hard data invites: now that the gate fires, is the rule that
*picks how far to split* itself any good? It chooses one resolution for all labels and,
among resolutions that tie on held-out accuracy within one standard error, keeps the
leanest. Three alternatives were measured against it on Reuters (hard) and the clean
topics (easy): a per-label resolution chosen by greedy coordinate-ascent, a "climb while
each step still earns more accuracy than its own noise" rule, and sweeping the duplicate-
merge grain alongside k. All three were worse or a wash -- per-label and climb both
under-fired (78-79% vs 80%), and sweeping the merge grain changed nothing. The current
rule already sits close to the pure best-k accuracy (80% vs 81%) at roughly half the
prototypes AND still picks one prototype per class on the easy topics, which pure best-k
would over-split on noise. So unlike the two gates above, this one is not a hidden
conservatism bug -- it is a deliberate accuracy-for-leanness trade that measurement says
is already well placed, and it was left alone. The rule has now outlasted a fourth challenger: FRACTAL recursive bisection (the same split applied self-similarly at every node, each accept measured) was a wash on hard uneven data and wins leanness only where accuracy saturates -- full numbers in the design notes; recursive self-similarity stays where it measurably pays, the HoloForest index. The discipline is in checking, and in not
"fixing" what the data says is right.


**A damage-tolerant image archive.** Store a gallery of images superposed into a
few plates; recall the clean original from a noisy / blurred / occluded query,
even after destroying a large fraction of the plate. The bundled demo recalls
6/6 images from each corruption, and still 6/6 (reconstructing at ~52 dB) with
40% of every plate destroyed. Two further capabilities sit on the same store:
**cross-modal recall** (`recall_by_tags`) addresses an image by word/number tags
bundled into a hypervector, so "radial pink" alone returns the right image (6/6
on the demo gallery, no picture needed); and **quantized plates** (`quantize(4)`)
shrink the store ~8x (844 KB -> 107 KB) with content recall still 6/6 and
recovery degrading gracefully (79 -> 30 dB). Both work together - the tag
addresses live outside the plates, so cross-modal recall is unaffected by
quantization.

**A delta set-packer for related images** (`holographic_pack.py`). A follow-on, `pack_sprites.py`, handles palette GIF sprite sets a different way, and the honest lesson there is that delta coding was the WRONG tool: on a real 712-sprite set, unifying everything to one shared 88-colour palette and compressing the index planes with LZMA packs to 58 KB (bit-exact, the v2 format) -- 13.6x under the loose GIFs -- while every delta variant did worse. The win was the representation, not the diff: the v2 gain came the same way, by measuring that these CHARACTER sprites are more self-similar down a column than across a row, so the packer stores the index planes column-major when that wins (per-pack, recorded in one flag, never regressing below row-major) for a further 15% (68,632 -> 58,041 bytes) -- lossless, with v1 blobs still decoding. Reordering planes and predictive/delta filters were re-measured with the newer tools and rejected again (LZMA's window already finds the cross-sprite matches; filters break its runs). Generalising past sprites, `image_vault.py` is a format-, size- and codec-agnostic store that NORMALISES any input to RGBA, RELATES images by a size-invariant fingerprint (similarity, clustering, query-by-example), and COMPRESSES adaptively -- it measures shared-palette+LZMA, LZMA over related-ordered pixels, and per-image PNG, plus optional lossy JPEG/WebP for photographs (measured with PSNR), then keeps whichever is smallest for that set and reports the comparison honestly. The orientation win from the sprite packer generalizes here too: the palette and LZMA methods each try both row- and column-major layouts and keep the smaller (a per-set flag), so the vault inherits the sprite transpose gain automatically while correctly choosing row-major on horizontally-structured data -- adaptive by measurement, never regressing. (The same orientation trial was measured on the truecolour delta-packer and DECLINED -- on related images with localized edits the residual is already structure-free, so it would add complexity for no gain; the principle is applied only where the data has directional redundancy.)

**A dictionary-first curriculum for word meaning** (`holographic_lexicon.py`). Asked what happens if the brain learns a dictionary before any other reading, the engine builds each word's meaning as a bundle of its definition words' meaning vectors and iterates that on the definition graph -- a fixed point, the resonator dynamic applied to a lexicon. Measured on WordNet (a real machine-readable dictionary) against a synonyms-vs-random separation test: random vectors separate nothing (d'=0), one definition pass jumps to d'=1.5, and iterating peaks at ~3 passes (d'=1.9) before over-diffusing -- the same fixed-point-then-collapse sweet spot seen elsewhere. Co-occurrence reading alone scores only d'=0.5, so definitions are far denser meaning than prose. The curriculum verdict is two-sided and honest: dictionary-then-reading beats reading-alone (+0.8 d', the intuition confirmed), but full-rate reading washes out the clean definitional structure (1.9->1.3) -- the dictionary is so much cleaner than prose that reading must refine, not overwrite, and gentle reading preserves the seed. The grammar and encyclopedia layers map onto sibling subsystems: grammar is sequence structure (the sequentiality test), an encyclopedia is relational fact (the KnowledgeStore + ask/raytrace machinery).

**Real photographs, not just sprites** (`holographic_photos.py`). The image stack was only ever measured on GIF sprites (palette, <=88 colours); a wallpaper category of real photos (thousands of colours, continuous tone) is the opposite regime and the honest test of the lossy path. The findings, stated plainly: JPEG beats our DCT coder on efficiency (~31 dB at ~2 KB vs the holographic plate's ~5 KB) -- the plate keeps top-K global DCT coefficients in a fixed-size store, not entropy-coded 8x8 blocks, and is not a competitive photo codec. But it is robust where JPEG is brittle: erasing a random 50% of the holographic coefficients barely moves PSNR (28.7 dB at both 0% and 50%), because every coefficient carries a little of the whole image, while JPEG loses ~15 dB after a 10% byte loss and often fails to decode. And the vault's adaptive chooser generalizes -- on these 12k-colour photos the palette path is correctly unavailable and it picks lossy WebP, the opposite of its sprite choice, driven entirely by the data. The orientation trick correctly stays idle (photos lack the directional self-similarity of character sprites). Robustness, not efficiency, is the honest contribution.

**Coarse-to-fine resolution** (`holographic_resolution.py`). leOS queries a trained Matryoshka embedding at 32/128/full dimensions, escalating on a cheap confidence totem. Adapted honestly to holostuff's random hypervectors -- where information is spread evenly, so a truncated prefix is a random subsample of the cosine, not an energy-concentrated one -- the totem becomes statistical: the top-1/top-2 gap measured against the spread of the field at that resolution. Measured, it returns the same nearest neighbour as a full-dimension scan 100% of the time while using ~5% of the dimension-work over a 500-item store (easy matches settle at ~128 of 4096 dims); it degrades to a full scan on near-ties (no error, no saving) and abstains on a no-match. Wired into the central `_cleanup` path (so the KnowledgeStore and, through the brain, find/read_role/climb all benefit -- verified identical to the exhaustive scan) and into the brain's record `find`, but deliberately not the already-hierarchical RP-tree or the already-low-dim image fingerprints. The companion measurement -- `resolution_profile`/`stabilisation_dim`, the persistent-homology idea in practical form -- tracks at which truncation a query's winner stabilises (low = robust to truncation, full width = a close call), surfaced as a 'How much resolution?' button in the unified-mind app.

**Fractal structure** (`holographic_fractal.py`). leOS's self-similarity detector, ported to the data holostuff actually has. The instrument is verified against known fractals first (box-counting recovers Sierpinski 1.59, a filled square 1.95, a line 0.98), then applied honestly: natural-photo edge maps have fractal dimension ~1.55 (rough, scale-invariant -- the statistics of natural scenes) while a smooth synthetic circle's edge is ~1.0, so dimension is a real natural-vs-synthetic signal and a texture descriptor the shape work lacked; DAI/WETH returns read Hurst ~0.30 (mean-reverting), independently agreeing with the -0.175 autocorrelation the market rounds found; and a Barnsley fern compresses to 4 affine maps (28 numbers, >500x) while random points have no compact IFS (the kept negative -- IFS compression pays off only on self-similar data). Wired into the brain (`fractal_dimension`, `self_affinity` as perceptual readouts) and the vision panel (a 'fractal dimension' demo contrasting natural photos with synthetic shapes).

**Context-conditioned generation, and an honest negative** (`holographic_generation.py`). The recurring question of why this brain is not an LLM, answered by measurement rather than assertion. The generator is a shallow n-gram `P(next | last few tokens)`; an LLM's power is a high-capacity *learned* `P(next | whole context)`. To test whether merely deepening the conditioning helps on this substrate, the word-level `ContextGenerator` re-ranks a word n-gram's candidates by how well each candidate's learned meaning vector aligns with a running topic vector, with a tunable topic_weight (0 = bare n-gram). The measured result, kept as the finding: topic-pull re-ranking does **not** buy genuine coherence -- it is flat where the n-gram has only one candidate (~85% of order-2 contexts), and when pushed hard it raises the coherence *number* only as lexical diversity collapses (0.78 to 0.09) into degenerate repetition that even keeps transition-validity high. You can only re-rank structure the proposer already offers; a shallow proposer offers none, so the ceiling is the proposer, not the loop. Surfaced as `learn_word_generator`/`generate_words`/`topic_pull_tradeoff` on the brain and a 'topic pull' panel in the app that shows the coherence-rises-as-diversity-collapses table live.

**Fountain codes: a second robustness axis** (`holographic_fountain.py`). leOS's last clean idea, built from scratch as Luby's LT code. k source blocks become an unlimited stream of droplets, each the XOR of a random subset (the binary sibling of a bundle); collect any k(1+eps) of them, whichever survived, in any order, and decode recovers every block exactly by peeling -- a degree-1 droplet reveals its block, which is XORed out of the rest, unlocking the next, the same loop-until-resolved pattern as the resonator and coarse-to-fine. This complements the holographic plate rather than competing: the plate degrades gracefully when one analog representation is partly corrupted (lossy, never exact), while the fountain survives whole-packet loss and recovers exactly -- lose 40% of a provisioned stream and the blob returns bit-for-bit, lose so much that fewer than k droplets survive and nothing returns (the information floor, an honest hard cliff). The price is a ~20% droplet overhead at large k; at small k the overhead is higher and more variable (kept as a measured caveat). Surfaced as an 'Erasure robustness: the other axis' app panel with a channel-loss slider, beside the plate's graceful-decay demo.

**A predictive loop: anticipate and correct** (`holographic_predictive.py`). The active layer on top of storage -- the engine moves from retrieving what it holds to *acting* on it. Adapted from the predictive-coding architecture in a friend's Closure-SDK (its upper stack is Rao & Ballard predictive coding; here it is rebuilt on bind/bundle/permute instead of the S3 quaternion substrate). The living cycle: predict the next symbol from recent context, measure surprise (1 - cosine of predicted vs actual), correct error-gated (reinforce when right, nudge or create an entry when wrong, scaled by surprise), and report free energy (smoothed prediction error) and valence. What is new: prediction by *resonance* rather than exact match -- each entry is an order-aware context vector paired with a next symbol, so a context never seen exactly still predicts sensibly when a similar one was seen (an unseen 'my cat' predicts 'sat' from the cat-contexts; exact n-gram backoff is blind here). Measured on Brown news, held-out accuracy rises with exposure and it scores ~7% on never-seen contexts where exact lookup scores 0; on a periodic stream surprise falls to ~0 within one period and free energy converges to 0 (on non-repeating prose it honestly does not, because each context is mostly novel). Generation runs the same predictor forward by anticipation. Wired as `build_predictor`/`observe_sequence`/`anticipate`/`generate_predictive`/`prediction_report` on the brain and a 'predictive loop' app panel showing the learning curve, free-energy trace, generalisation score, and a generated sample. Also ported: `zread`, a coupling-weighted order-aware soft read -- now also SUPPORT-weighted, so a soft next-symbol read of a context with several successors at different rates returns the MAP (most-frequent) one. Without that weighting a 70/30 split blended 50/50 and decoded to the 30% symbol; weighting the blend by each entry's reinforcement count (the engine's own frequency record) fixes it (measured MAP-correct across 60/40..90/10), and the same revisit of the soft path surfaced that the hard read is itself fragile near a tie -- so support-weighted soft is the reliable read on stochastic streams.

**Meaning-level prediction: generation with structure** (`holographic_meaning_predict.py`). The next rung: instead of returning one stored next symbol, the predictor *composes* a next-meaning vector as the coupling-weighted blend of the next-meanings of every resonating context, *settles* it by iterated cleanup in a meaning space (the resonator pattern), and reads off the nearest word. The prediction is built from many entries, so it can land where no single entry sits, and even when the exact word is missed it is missed toward semantically near words -- reported as a semantic rank (percentile of the actual next word under the prediction; 0.5 is chance) alongside exact accuracy. Tested the way good data demands -- dictionary/encyclopedia curriculum as the prior, then real corpora -- which overturned the obvious guess: measured on Brown and Reuters alike, co-occurrence (syntagmatic) meaning predicts the next word well (semantic rank ~0.85) while the dictionary-curriculum (paradigmatic) space is near chance at it (~0.50); the reverse holds for relatedness, where the dictionary space separates related words at d-prime ~0.8-0.93 vs ~0.45. The principle kept: match the space to the query -- co-occurrence for what follows, the dictionary/encyclopedia prior for what a thing is. Wired as `build_meaning_predictor`/`anticipate_meaning`/`meaning_prediction_report` on the brain, with `cooccurrence_space`, `relatedness_dprime`, and a `set_space` hook to plug in any prior.

**Proof of structure: verifying meaning, not trusting it** (`holographic_structure.py`). A predictor that lands each step in the right neighbourhood can still emit locally-plausible, globally-meaningless text, so a prediction needs a verifier -- and the proof has to come from projecting each word onto its context, not from the word's meaning alone. Single-step coherence (cosine of what the context predicts to the actual word) separates real text from shuffled and random, but is gameable: text generated greedily by the predictor scores *higher* than real text, because each step maximised exactly that. The proof that works is the lag-coherence profile -- the similarity between each word and the word k positions back, for k=1..6. Real text has a moderate, even profile; salad deviates (too low for random/shuffled, too high and periodic for degenerate loops). The structure score (z-distance of a sequence's profile from a real-text band) rates real text ~ -1.2, shuffled ~ -2.2, random ~ -4.1, and self-generated loops ~ -15 -- catching the locally-coherent salad single-step coherence missed. Used as a process, `steered_generate` picks the next word that best preserves the running structure: greedy decoding collapses into a loop (~ -15) while steered generation stays in the real-text band (~ -0.8) -- the opposite of the earlier topic-pull collapse, because it steers by trajectory structure rather than a static topic bag. An honest limit kept: it rejects random and degenerate loops but not reliably shuffled real words (exact-order corruption would need a running composition). Wired as `verify_structure`/`generate_structured` on the brain and a proof block in the predictive app panel.

**Query-and-generate** (`holographic_respond.py`). The synthesis: ask the system something and get a structured, on-topic continuation back, built entirely from the substrate's own predict/compose/verify machinery. A query implies a target region in meaning space (the bundle of its content words' meanings); generation runs forward with the meaning predictor, each step chosen under two forces -- the structure guard (keep the running window coherent, escape loops) and the query-pull (prefer candidates pointing toward the query target). Measured on Brown news, relevance rises monotonically with the query weight (0.47 -> 0.66) while structure holds in the real-text band through a real operating window, degrading only under a hard pull. The load-bearing finding, and why this is not a repeat of the earlier topic-pull collapse: the structure guard is what makes that window exist -- at a hard pull, with the guard structure stays ~ -2.0 while without it (the topic-pull regime) it collapses to ~ -6.7 for the same relevance. `respond_report` returns the response with both its relevance and its structure, so an answer is never trusted blindly. Wired as `respond`/`respond_report` on the brain and a 'query & generate' app panel that shows the steered response beside the unsteered baseline.

**Deliberation: think before answering** (`holographic_deliberate.py`). Rather than emit the first draft, the system forms the gist (the query's meaning target), realizes it into a draft, judges it (relevance plus structure), and iterates -- keeping the best and stopping early once a draft is good enough. The iteration count is the thinking time and it adapts: easy queries settle in one or two passes, hard ones use the full budget (measured 1-8 across Brown queries), which is the 'sometimes fast, sometimes slow' of human deliberation. The loop helps (deliberated quality ~0.40-0.43 vs a single greedy pass ~0.34-0.36). Two plan-elaboration ideas were tried and kept as negatives: rolling the meaning predictor forward into a trajectory plan drifts into function words and hurts relevance, and enriching the gist with meaning-neighbours is neutral -- so the gist stays simple and the effort goes into the loop. The full trace of drafts and their scores is returned, so the inner deliberation is visible. Wired as `deliberate` on the brain and a 'deliberate' app panel that shows the thinking time, the kept response, and the draft trace.

**Multi-judge negotiation** (in `holographic_deliberate.py`). Deliberation under several competing judges instead of one quality score: coherence (structure), relevance (on-query), and novelty (anti-repetition). The negotiated score is the minimum across the judges -- the binding pressure -- so the kept draft is the most balanced rather than one that wins a single axis while failing another, and the per-judge trace makes the tension visible. With the structure guard already suppressing most loops, the novelty judge is mostly a safety net (it rescues the occasional repetitive draft, type-token ratio 0.85 -> 0.96, and matches otherwise). Wired as `negotiate` on the brain; backward compatible with the single-quality `deliberate`.

**Cross-domain structure** (`holographic_signal_structure.py`). The structure verifier's idea -- match a sample's autocorrelation signature to a band of real data -- generalised beyond text. For a continuous series it is the lag-autocorrelation profile; for an image, the spatial autocorrelation across pixel offsets. It transfers cleanly to images (a natural patch scores ~ -0.6 against a real-patch band while noise and pixel-shuffled versions crash to ~ -14, the same separation text gives) and to returns only with the right signature: raw returns are nearly uncorrelated, so the distinguishing structure is volatility clustering (lag-1 autocorrelation of absolute returns vs a shuffled control), unmistakable on a long synthetic GARCH series (z ~ 9) but only ~1 sigma on the short real DAI/WETH sample -- too little data to call. The lesson kept: the machinery is cross-domain, but the choice of what to autocorrelate is the domain knowledge. Wired into the brain as `verify_image_structure` and `volatility_structure`.

**Compression: better structure, fewer bits** (`holographic_compress.py`). A predictor is a compressor -- rank-code each symbol by the predictor's ranking and well-predicted (structured) symbols cost few bits. Measured on Brown news against a uniform baseline of log2(vocabulary) (~11.8 bits/symbol): real text costs ~7.0 (ratio 0.59), shuffled real words ~8.9 (0.75), random ~10.4 (0.88) -- more structure, fewer bits. The structure score predicts the compression ratio (correlation ~ -0.6 across windows from real to shuffled), and the predictor beats a frequency-only unigram model (~9.5 bits/symbol), so it is exploiting order and context, not just word counts. This sits beside the fractal IFS compressor already in the stack (a self-similar fern compresses ~500x, random data does not): two kinds of structure, two kinds of compression, one principle. Wired as `compress_cost`/`structure_compresses` on the brain, with real-vs-shuffled compression ratios shown in the app's structure-proof block.

**Self-discovery of structure** (`holographic_segment.py`). Find the units in a stream with no labels, by listening to where prediction breaks down. Inside a unit the next symbol is constrained; at a unit's end many can follow, so uncertainty peaks at boundaries. Strip the spaces from text and the word boundaries are recoverable from this signal alone: for each exact context, accumulate the bundle of symbols that followed, read its entropy, and cut at the peaks. Measured on Brown (spaces removed), the recovered boundaries hit F1 ~0.6 against the true word boundaries vs ~0.2 for a random cut, and the discovered chunks compress to ~2.1 bits/char vs ~4.2 over single characters -- finding the right decomposition roughly halves the description length, the same structure-to-compression principle reached by discovering the units. A kept negative chose the method: a resonance-blended readout smears the signal (F1 ~0.26), because segmentation needs exact-context successor diversity, the opposite of what generalisation wants. Wired as `discover_units` on the brain with a 'self-discovery' app panel.

**Factorization by searching in superposition** (`holographic_resonator.py`). The inverse of binding, and the most powerful single primitive here -- a Resonator Network (Frady, Kent, Olshausen & Sommer, 2020). Binding combines several vectors into one; given only the composite and the codebooks of possible parts, the resonator recovers which part came from each codebook by entertaining a weighted superposition of all of a codebook's vectors at once, unbinding the others, cleaning up toward the codebook, and iterating until the true factors resonate out and the rest cancel. It never enumerates the combinatorial space (the product of codebook sizes -- a million for three codebooks of 100). Measured on this substrate: three codebooks of 50 (125,000 combinations) solved 20/20 with ~2 random restarts; three of 100 (a million combinations) ~11/20 at dimension 3000 -- the classic dimension-vs-capacity tradeoff. A kept negative shaped the build: the engine's native circular-convolution binding amplifies noise too much to factor (0-1/20), so the module uses self-inverse MAP/bipolar binding internally -- the operation you can invert in superposition depends on the algebra you bind with. Wired as `factor_composite` on the brain with a 'factorize' app panel that binds three random vectors and recovers them.

**Lossless codec and source attribution** (`holographic_codec.py`). Going both directions exactly. The predictor ranks the vocabulary at each step; each token is encoded by its rank, and because the decoder runs the identical predictor it decompresses to the exact original -- a lossless round-trip (the compressed form is the seed plus the rank stream, with the model shared like a codebook). The honest answer to 'compress to a seed and decompress back': real and exactly lossless, but bounded by structure -- measured on Brown, real text rank-codes to ~0.63 of the uniform baseline, random data barely shrinks (~0.74, no free lunch), and a perfectly periodic stream collapses to ~0 bits/token (the seed alone). That is the same statement as the IFS fern compressing ~500x while random data does not: compression is the search for the shortest generator. Source attribution, hard before and now tractable because resonance couplings are exposed, tags each stored context with its source and traces a passage's provenance from which sources' contexts resonated with and predicted it (a held-out news passage attributes ~0.74 to news vs romance). Wired as `compress_lossless`/`decompress_lossless`/`attribute_sources` on the brain with a 'lossless codec' app panel. Together with the resonator this completes a reversibility sweep: the engine can both compose and factor, and both predict forward and compress-and-restore.

**Many minds, one substrate** (`holographic_partition.py`). For running a population of NPCs or agents without a full brain each. Train one base mind on the common knowledge and freeze it; every instance shares that base by reference (including its encoder, so all instances perceive into the same vector space) and holds only its own small delta of what it personally learned. An instance reads by scoring over base + delta, so it inherits all shared knowledge and adds its private knowledge on top; it writes only into its delta, so the base stays shareable. Because instances share the same atoms, learned vectors are comparable and additive, which makes two things free: merging an instance's learning back into the base (so everyone inherits it) is just superposition/bundling -- a federated average in VSA terms -- and isolation costs nothing, since an instance never sees another's private delta until it is propagated. Measured: with a base of B prototypes and N instances each holding ~d private, the population costs B + N*d versus N*B for separate minds (50 NPCs over a 1,000-prototype base with 20 private each is ~2,000 vs ~50,000, ~25x, growing with base size). Verified inheritance, isolation, propagation, and that merge-by-superposition preserves recall. Wired as `share()` on the brain returning a `SharedMind` you `branch()` into lightweight instances, with a 'many NPCs, one mind' app panel. (Honest limit: superposition merge has a capacity ceiling -- bundling very many distinct deltas into one base label eventually degrades recall, the classic VSA capacity cliff.)

**An encyclopedia for understanding beyond words** (`holographic_encyclopedia.py`). The third curriculum rung: structured knowledge about complex topics, not just what words mean. Concepts load from WordNet's is_a hierarchy and part-of relations (keyed by synset so senses don't collide) into a KnowledgeStore, and a taxonomy chain (dog -> canine -> carnivore -> mammal -> animal) is walked as a relation ray -- each hop a bounce, the cleanup confidence its reflectance. Measured: one-hop is_a retrieval is exact (100%), and multi-hop climbing is exact at 2-4 hops once scored honestly in a closed world with consistent senses (a first naive measurement read 43% and looked like a failure -- it was a test artifact from sense ambiguity, the model was right; kept as a lesson). Chain throughput decays with depth (0.50->0.25 over 2-4 hops), a calibrated 'how far has this deduction traveled' that lets a chain abstain rather than emit noise. And the point of the exercise: taxonomic siblings are related knowledge even when their definitions barely overlap -- ~58% of sibling pairs share at most one definition word, so the dictionary is nearly blind to their kinship while the encyclopedia links them through the shared parent. Relatedness lives in the structure, not the words. A Curriculum class stacks the three layers and reports the capability each adds that the previous lacks. And the curriculum is wired into the brain itself, not left in tests: `UnifiedMind` gained `learn_dictionary` (bootstraps word meaning into its own text encoder), `define` (nearest words by learned meaning), `learn_encyclopedia` (absorbs is_a facts into its own memory as records), and `climb`/`is_a` (walks taxonomy chains over its own memory as a path-traced ray). The unified-mind app exposes this as a hand-built **Dictionary + encyclopedia** dataset (no network) and a curriculum panel: type a word and see its meaning-neighbours (dictionary layer) beside its is_a chain with throughput (encyclopedia layer), side by side over one trained mind. And because a typed sentence otherwise just gets *completed* (the generation path is not a chatbot), the mind gained a `answer(question)` router and an **ask** panel: it recognizes a question's shape -- 'what is a dog?' -> meaning + is_a chain, 'is a dog an animal?' -> taxonomic check, 'what is the capital of france?' -> a record's role, 'classify: <text>' -> nearest category -- and dispatches to the real operation, falling back to clearly-labelled text completion (never pretending a continuation is an answer) when it can't map the question. Lossless modes are bit-exact; pull any image back by index/name or hand it an example to find the nearest stored ones. Single-file
codecs compress each image alone, so a family of images that shares structure
pays for that shared content in every file. The packer stores a set as one
reference plus per-image deltas (residual mod 256, zlib'd), bit-exact and in 8-bit
integers throughout. On a six-logo suite that shares a background and ring it packs
to ~39% of per-file PNG and beats gzip-ing the whole set; honestly, on images that
are already compressible on their own (gradients, photos) per-file PNG/JPEG win,
and the built-in benchmark shows both so the choice is clear. A lossy
Walsh-Hadamard tier was prototyped and dropped because it never beat JPEG.

**Then the engine moved two more of its own loops into the substrate.** A self-hosting audit asked which
hand-written Python procedures could be re-expressed as VSA programs running on the engine's own stored-program
machine — where a program is a hypervector, control flow is `IFMATCH`/`ITERATE`/`CALL`, and `APPLY <faculty>`
invokes a real faculty as a step. The rule is exact: a procedure is movable iff it is *orchestration* whose
every step is a hypervector→hypervector map and whose state is one accumulator — which the data-analysis
pipeline (PIPE-1) and the recurrent linear map (`ITERATE [APPLY matmul]`) already showed. Two of the engine's
remaining canonical iterate-to-fixed-point loops fit and were knocked out. `restore_procedure` runs Plug-and-Play/
RED restoration as the program `ITERATE [APPLY datafit; APPLY denoise]`: on a half-masked low-rank signal it
recovers to the *same* error-to-truth as the Python `denoise(method='pnp')` (raw rel-error 0.86 → 0.17),
converging in six iterations where the Python loop runs a fixed forty. `generate_procedure` runs the B10
generative diffusion as `ITERATE [APPLY diffuse]` from a noise seed — a self-cooling step that anneals β up and
injected noise down, with the loop halting when the sharpened cleanup lands on the manifold (cosine 1.000, the
same sample the Python `generate` produces). The payoff is not speed — both carry the honest procedure tax of a
noisy unbind-and-clean per instruction read, and the numerics underneath never leave NumPy — it is that the
restoration *loop* and the generative *process* are now stored, composable, recipe-savable objects rather than
control flow: process, not object. The boundary stayed where the audit drew it; the numerical primitives (the
FFT bind, the SVD, Sinkhorn) *are* the substrate and cannot become programs that run on themselves.

**Then a faculty that had become fast was promoted to a gate, and the basis it guards learned to scale.** A
performance audit found persistent homology (made 83× faster a session earlier) was the lone heavy outlier;
fixing it left a dividend. Now that naming a cloud's topology is sub-second even on a structureless blob, it
becomes a first-class *gate*: `is_manifold` reads the Betti signature and answers whether a cloud is a single
connected manifold (sphere → `True`, B0=1) or a dense blob with no clean topology (random 4-D cloud → `False`,
fragmented and dense). The gate earns its keep next door, on the spectral denoiser, whose premise is a *smooth
field on a curved manifold* — exactly the thing `is_manifold` checks. With `check_manifold=True`,
`denoise(method='spectral')` runs the gate first and refuses loudly on a blob (where the "denoise" is only graph
low-pass, not manifold denoising) with an escape hatch to override; measured, the spectral map cleans a 2-sphere
field 3.74 → 1.08 but barely moves a blob (4.37 → 4.20), so the gate names which case you are in for free. The
second half was harder. The spectral basis built its modes with `np.linalg.eigh`, which computes *all* n
eigenvectors at O(n³) to keep the lowest twelve — fine to ~1500 points, painful past 3000 (4.4 s) and worse at
5000 (22 s). A partial eigensolver should compute only the smooth modes, but four textbook shortcuts were
prototyped and *measured to fail*: shifted Lanczos and block subspace iteration both compress the wanted modes
against the shift and converge to the wrong subspace; unshifted Lanczos works small then degrades; a
graph-Tikhonov low-pass (no eigendecomposition at all) is robust and fast but a *soft* filter attenuates the
signal modes along with the noise (denoise 3.2 vs eigh 1.2); Nyström landmark-extension drifts worse as n grows.
The honest root cause is degeneracy — a 2-sphere's Laplacian carries 2l+1 modes at each eigenvalue, so a *count*
cutoff like n_basis=12 lands inside a degenerate block and no count-based iterative method can cleanly separate
it. The method that does is the one real sparse eigensolvers use: Chebyshev-filtered subspace iteration, where a
polynomial filter amplifies the wanted low-eigenvalue subspace by orders of magnitude so a block iteration
converges *through* the degeneracy. Built on a *sparse* kNN-Laplacian matvec (O(n·k), byte-identical to the
dense operator to 2e-14, never forming the n×n matrix) it matches the exact eigh's denoise to a projector
difference of ~0 in the verified range with a speedup that grows as eigh's O(n³) bites — ~4× at 3000 points,
~7× at 4500 — and `SpectralBasis` switches to it automatically above a point threshold while keeping the exact
dense eigh below (faster there, and bit-identical, so every existing result is unchanged). The kept negatives
travel with it: it is an approximation whose projector error grows slowly with n, and it lifts the *eigh* cost,
not the O(n²) distance build, which still caps practical use at a few thousand points — a spatial index would be
the next rung.

**Then the honesty discipline became something you can lint.** A forwarded backlog proposed turning a session's
worth of analysis methods into VSA programs, and grounding it against the live code was the usual humbling
exercise: most of it was already built (the stored-program VM, the self-hosting procedure layer, and every part
of the "honesty harness keystone" — `walk_forward_recall` *is* the six-check harness, `scan` *is* the parallel
SPRT+FDR detector), one flagship item was the capacity cliff in disguise (running N independent computations in
one superposed bundle and reading N exact scalars back is exactly the thing the engine has measured can't be
done), and the market half bet against the efficient-market negative already on the books. The one genuinely-new,
on-mission idea survived the audit: turn the honesty discipline from a habit you maintain into a *structural
property you can check*. Because a protocol — an analysis procedure — is program-as-data, its step structure can
be read back from its program vector (the VM's own unbind+cleanup), and the canonical artifact-factory becomes a
structural query: a SEARCH/recall step that proposes candidates from the data with no procedure-matched NULL to
confirm they beat noise. `audit_procedure` builds a protocol from a list of faculty-steps, recovers the structure
holographically, and flags three anti-patterns — a search with no null, a searched-and-scored family with no FDR
control, and selecting-then-scoring with no out-of-sample split between them — while staying targeted enough not
to flag an honest no-search procedure like a restoration loop. The kept negatives are explicit: it is a
structural lint on *declared* steps, not a data-flow analysis (so "scores the same rows it selected on" is
approximated by the order check — no split between select and decide — since the single-accumulator VM doesn't
track data lineage), and the per-step decode is bounded by the program vector's capacity, so a protocol must be
short enough to read reliably (the procedure tax, verified to round-trip exactly at protocol length). The
discipline the whole project runs on — the procedure-matched null, the out-of-sample split, the exact arbiter —
is now a thing a program can be checked *for*, not only a thing a person remembers to do.

**Then the research log learned to catch its own contradictions.** The other genuinely-new idea from the same
backlog — a registry of findings you query by similarity and that flags its own tensions — turned out to have its
substrate already present (the relations layer's role-bound records and analogy-as-unbind), missing only the one
operation a research log actually needs: contradiction detection. A finding is encoded the way every record in
the engine is — `bind(SUBJECT, x) + bind(OBJECT, y) + bind(CONDITION, c)` with a stored ±1 polarity — so the
existing explain/analogy operations compose with it for free. What the registry adds is two things. Recall is
*role-sensitive*: querying for findings where momentum is the OBJECT does not surface ones where it is the
subject, the dividend of structured encoding over a bag of words. And tension detection distinguishes a *flat
contradiction* — two opposite-polarity findings making the same claim under the same or absent condition, where
one must be wrong — from a *conditioned tension* — the same opposition under *different* conditions, which is
reconcilable because the outcome is conditioned on the differing dimension. "Efficiency-ratio strengthens
momentum at the 10-day horizon" versus "efficiency-ratio backfires intraday" is the latter, and the registry
says so rather than calling it a contradiction. The discipline holds: retrieval is holographic (a cosine over the
bound claim finds the candidate conflicts) but the verdict is exact (the polarity sign and the condition
equality decide), the same one-exact-door rule the whole engine runs on. The honest boundary stays where the
audit drew it — findings are *structured* claims, not free prose; turning a 2300-line narrative log into
structured claims is an NLP step this engine, with no embeddings and no parser, does not do.

**More engine pieces** (each with a runnable demo): residue (exact integer)
arithmetic on vectors, signed-distance regions of the sphere, a predictive filter
that stays quiet on the expected, a unified scalar "field" abstraction,
two-timescale diffusion, Kuramoto-style emergent grouping, a tool orchestrator
with circuit-breakers and reusable skeletons, online unsupervised concept
formation, and a hypervector reaction-diffusion cellular automaton.

---

## How it works

**The core.** Atoms are random high-dimensional vectors, nearly orthogonal by the
blessing of dimensionality. `bind` (a reversible element-wise combine) ties two
together into something dissimilar to both; `unbind` recovers a partner.
`bundle` (normalised sum) overlays things into a set you can still query. A
`Vocabulary` mints clean atoms and `cleanup` snaps a noisy vector back to the
nearest known one. That is the whole toolkit; every subsystem is a different way
of arranging those operations.

**The image archive** adds three image-specific steps:

1. *DCT, keep the big coefficients.* Each colour channel goes through a pure-numpy
   orthonormal 2-D DCT; only the largest `K` coefficients are kept (their
   positions stored as a small bitmask, counted honestly in the size).
2. *Spread with structured keys.* The kept coefficients are scattered and run
   through a **Walsh-Hadamard transform** with a fixed random sign pattern, so
   each one is smeared across *all* `D` plate values - no plate value is special.
   The key operator is matrix-free and an exact isometry, so an undamaged plate
   decodes exactly with one adjoint pass.
3. *Recover.* Undamaged - one multiply. Damaged - a mask marks survivors and a
   small conjugate-gradient solve recovers from what is left, graceful until the
   survivors drop below the stored coefficient count.

Multiple images share one plate via *disjoint* key-slot pools (keeping the
combined keys orthonormal), and content-addressable recall keeps a tiny thumbnail
fingerprint *outside* the plate so recognition survives even when the plate is
wrecked.

---

## Benchmarks vs. existing tech (image archive)

Measured by `bench_vs_jpeg.py` on a 240x240 colour image. Reproduce: `python bench_vs_jpeg.py`.

**Plain compression - JPEG/PNG win, and that's fine.** The hologram is not a
compressor: PNG 1.5 KB, JPEG q85 5.3 KB (29.8 dB), hologram 4-bit 42.2 KB
(27.9 dB). Use a real codec if you want small files.

**Corruption resilience - the hologram wins enormously.** Corrupt the same
*fraction* of a JPEG file vs. plate cells (mean PSNR over 8 trials; 0 dB = no
longer decodes):

| corrupted | JPEG q85 | Hologram |
|-----------|----------|----------|
| 0.1%      | 9.2 dB   | 27.9 dB  |
| 1%        | 4.0 dB   | 27.9 dB  |
| 10%       | 0.0 dB   | 27.8 dB  |
| 40%       | 0.0 dB   | 27.0 dB  |

A JPEG dies at a tenth of a percent (its headers, DC terms, and entropy-coder
state are single points of failure); the hologram is essentially untouched at 40%,
because corruption is just uniform noise with no privileged bytes to destroy. See
`figures/bench_corruption.png`.

**Also measured:** the Walsh-Hadamard keys use ~3,200x less memory than a dense
random-projection matrix and run ~57x faster, with identical fidelity; conjugate-
gradient decoding gives ~8x the usable capacity of a matched filter; 10 images
multiplex into one plate with no crosstalk.

**Batch retrieval (1-bit vs float).** Finding the right stored item from a noisy
query, over a 10,000-item database (`bench_batch.py`): 1-bit sign hypervectors
with Hamming similarity match float32 cosine on accuracy (**100% vs 100% recall@1**
on a 20%-corrupted query) while using **32x less memory** (10 MB vs 328 MB). In
pure numpy the float matmul is faster (BLAS is hard to beat without a dedicated
popcount kernel); the 1-bit win is the footprint, which fits in cache at scale.

---

## Project layout (flat on purpose - everything imports cleanly)

The engine (pure numpy):

    holographic_ai.py         bind/bundle/cleanup, key->value memory, learner, reflex, drift
    holographic_unified.py    TOP LEVEL: one encoder + one memory + one brain + named sequence schemas
    unified_app.py            web console to test the unified mind on pulled corpora
    holographic_encoders.py   numbers (scalar/fractional-power), text, mixed records
    holographic_reasoning.py  resonator, conformal intervals, epistemic map, compass
    holographic_creature.py   grid-world + a holographic RL mind (the forager)
    holographic_navigator.py  the same mind, repurposed to navigate the data tree
    holographic_mind.py       the shared UniversalEncoder (with modality self-discovery)
    holographic_moe.py        mixture of experts with a learned holographic gate
    holographic_organizer.py  self-organizing memory: reorganize a shadow, then swap
    holographic_schema.py     structure by compression: chunk schemas, fractal coder, the gate
    holographic_text.py       text from scratch: learn / analyze / organize / produce
    holographic_tree.py       recursive RP-tree + HoloForest + slime-mould ReflexCache
    holographic_graph_memory.py  routed-descent memory -- a recorded negative for classification
    holographic_slime.py      slime-mould maze solver (discover, then thin to shortest)
    holographic_vision.py     pixels -> features: HSV, edges, Hough, shapes, emergent classes
    holographic_scene.py      compositional scenes: per-object tags, resonator factoring
    holographic_uri.py        content addresses: S3-style keyspace + bi-level buckets
    holographic_extras.py     residue arithmetic, SDF regions, predictive filter
    holographic_field.py      scalar field abstraction (one field, many roles)
    holographic_diffusion.py  two-timescale double diffusion
    holographic_sync.py       Kuramoto-style emergent grouping
    holographic_orchestrator.py  tool planner with circuit-breakers + skeletons
    holographic_emergence.py  online unsupervised concept formation
    holographic_automaton.py  hypervector reaction-diffusion CA (demoscene)
    holographic_image.py      WHT keys, DCT codec, quantised plate, damage decode
    holographic_archive.py    content-addressable multi-image memory
    holographic_pack.py       lossless delta set-packer for related images
    pack_sprites.py           palette-indexed packer for GIF sprite sets (+ bench_sprites.py)
    image_vault.py            format-agnostic store: relate, compress, retrieve any images

The app and tour:

    app.py        Flask UI (system tour + test runner + image recall)
    tour.py       command-line tour of every subsystem
    run.bat       Windows launcher

Tests (947 total):

    test_holographic.py           core engine (bind/bundle/memory/reflex/drift)
    test_holographic_image.py     image store / WHT / quantisation
    test_holographic_archive.py   archive recall + damage
    test_holographic_pack.py      delta set-packer round-trip + size
    test_pack_sprites.py          sprite packer round-trip + size
    test_image_vault.py           vault round-trip, query, clustering, lossy tier
    test_holographic_vision.py    HSV / edges / Hough / shape ID / clustering
    test_holographic_scene.py     compositional tags + multi-object resonator
    test_holographic_tree.py      capacity curve, RP-tree + forest recall
    test_holographic_uri.py       S3-style keyspace, prefixes, bi-level buckets
    test_holographic_navigator.py learned data-tree navigator vs fixed beam
    test_holographic_mind.py      universal encoder + modality self-discovery + index regime
    test_holographic_moe.py       learned gate routes to specialists, beats single
    test_holographic_organizer.py self-organizing + autonomous reorg (no thresholds)
    test_holographic_text.py      word learning, language ID, topic sort, generate, scale, hard, multilingual
    test_holographic_schema.py    schema discovery across text / CODE / IMAGES + the gates
    test_holographic_slime.py     slime maze solving + tube thinning
    test_holographic_orchestrator.py  typed tool chains, circuit-breakers
    test_holographic_graph_memory.py  routed descent -- pins the recorded negative
    test_holographic_brain.py     self-maintaining, autonomous, hard-shift recovery
    test_holographic_reservoir.py gradient-free reservoir: fixed permute recurrence + ridge readout
    test_holographic_classifier.py gradient-free HDC prototypes + perceptron retraining
    test_learning_faculties.py    reservoir + classifier + EP + FF + chaos + learned-energy through UnifiedMind
    test_holographic_equilibrium.py  Equilibrium Propagation: gradient matches finite-diff, learns two moons
    test_holographic_forward.py   Forward-Forward: local goodness objectives separate pos/neg, classify blobs
    test_holographic_chaos.py     nonlinear dynamics: reservoir beats best-linear >10x on chaotic one-step
    test_holographic_energy.py    learned energy memory: EP-trained cleanup beats fixed cleanup on a 2-D manifold
    test_holographic_unified.py   top level: one memory across modalities, self-discovery,
                                  absorb, named schemas routed by the compression gate
    test_integration.py           the recent modules wired into UnifiedMind as faculties, proven end to
                                  end (the whole integration plan): decompose_signal -> save -> realize ->
                                  denoise; SBC factorizer + decode_structure (peel); factor_composite
                                  de-siloed; opt-in energy cleanup; solve_maze / assemble / learn_dynamics;
                                  the mind saves/loads (quant='rd'); generate_vector + splat_field; the
                                  axial modality (theta==theta+pi, Mobius); the splat-bundle archive; the
                                  is_manifold gate + check_manifold guard on spectral denoise; spectral
                                  denoise scaling to a large cloud via the Chebyshev partial eigensolver
    test_holographic_dedoppler.py de-Doppler drift detection: a permute-based matched-filter bank +
                                  bh_fdr look-elsewhere + ON-OFF cadence (the detect_drifting faculty)
    test_holographic_verify.py    self-verifying storage: the holographic Merkle tree (bind+bundle) detects
                                  + localises a single tamper in <= log2(n) checks (the verify_store faculty)
    test_benchmarks.py            external-baseline harness (BLD-2): the rd code vs zlib/lzma and the forest
                                  vs exact brute-force NN, asserting the honest direction (incl. where stdlib wins)
    test_theory_references.py     BLD-3: every test THEORY.md cites must exist, so the guarantees doc can't rot
    test_holographic_fpe.py       BLD-7: N-D fractional power encoding -- n-D shift-as-bind, product kernel,
                                  compute-on-functions (a function as a bundle, queried + shifted by bind);
                                  pins that 1-D FPE was already the ScalarEncoder, plus the capacity cliff
    test_holographic_spectral.py  EXP-5/6/8: the graph/Hodge Laplacian operator -- cycle eigenbasis IS the DFT,
                                  Hodge harmonic dim == Betti numbers, the eigenbasis as a basis-selector
                                  (matches elementary/DCT on a line, beats it on a sphere), the Chebyshev-filtered
                                  partial eigensolver matching exact eigh at scale (sparse matvec, no O(n^3)),
                                  and the Hodge decomposition of an edge flow (gradient + curl + harmonic)
    test_holographic_topology.py  EXP-7: principled topology by persistent homology -- a point cloud's Betti
                                  signature names line/ring (matching detect_topology) and torus (1,2,1) /
                                  sphere (1,0,1) it cannot; GF(2) Betti agrees with the Hodge route
    test_holographic_clifford.py  EXP-9: Cl(3,0) geometric algebra as a parallel binding mode -- rotors compose
                                  3D rotations EXACTLY (~1e-15) and non-commutatively, the rotation-shaped win
                                  the commutative convolution bind can't reach; 2^d growth is the kept negative
    test_holographic_transport.py BLD-8: Wasserstein distance by Sinkhorn -- tracks how far distributions sit
                                  apart even with no overlap (where Euclidean/cosine saturate); matches the 1-D
                                  closed form; the eps knob (blur high / underflow) is the kept negative
    test_holographic_flow.py      B6 + sweep: the Tero flow solver, and `tero_flux` exposing the converged edge
                                  flux as a Hodge-decomposable flow -- gradient divergence == injected current,
                                  harmonic dimension == B1 (loops), zero circulation on a tree (flow_circulation)
                                  (recover/refine/region) + splat_bundle/recall_region; and the honesty layer
                                  woven into recognition (calibrated recognize + classify/recall abstention,
                                  SPRT stream, FDR-controlled batch); coherence-gated maintenance
                                  (reorganize only when incoherent -- fewer passes at comparable accuracy);
                                  and the panel's Tier-0 fixes (sublinear+calibrated recall, p-value coverage
                                  on noise, the default save using the rate-distortion code where it helps); the
                                  honesty layer reaching ACTION (calibrated decide_confidence + the
                                  explore-when-unrecognized policy, over a procedure-matched brain null); the
                                  SPRT's sample-savings shown on OVERLAPPING densities (decisive on separated
                                  ones); the auto coherence floor (reorganize on a relative drop below the
                                  recent peak -- no absolute threshold); and the `scan` faculty (SPRT per
                                  channel + FDR across channels -- streaming detection and look-elsewhere
                                  control in one pass, the noise floor procedure-matched through perceive); and a calibrated soft
                                  confidence for the SBC resonator on approximate inputs (a p-value vs the
                                  agreement the resonator manufactures on noise -- it rescues confidence when
                                  the exact-reconstruction certificate is uselessly False, and abstains on real
                                  noise); and a pluggable placement energy for `assemble` (the Rosetta move -- a
                                  supplied substitution energy changes the global-optimal assembly, the flow
                                  search still matching the Viterbi DP) with a structure-compare that reads the
                                  overlap of two assembled folds, the consolidation SVD read matching the exact
                                  placement count; one iterate-a-projection faculty (`project_onto_constraints`)
                                  under which the resonator, the PnP denoise loop, and a PBD constraint sweep are
                                  the same engine -- shown as POCS / resonator / PnP, with `pnp_restore` now
                                  literally calling it; and a determinism audit proving the new calibrated/null
                                  paths are bit-identical across a rebuild with the global RNG scrambled; and a `restore` faculty (Plug-and-Play/RED inverse
                                  problem -- pass a mask and it inpaints an erased measurement using the mind's
                                  adaptive denoiser as the prior, the loop beating a single denoise by ~19 dB on
                                  an erased archive plate); and a capacity/SNR diagnostic (`capacity_report`) reporting where a
                                  store sits versus the HRR noise-wins cliff (d' above the floor, the measured
                                  floor tracking sqrt(2 ln N / D), the headroom before noise wins) and that the
                                  calibrated false-alarm rate holds at alpha as the store grows; and a spectral/audio FHRR modality (`spectral_encode` / `spectral_decode`,
                                  Puckette's phase vocoder -- an exactly-invertible split into a unit-phasor
                                  FHRR vector and a magnitude) plus a validation that `learn_dynamics` predicts
                                  audio frames three orders of magnitude better than persistence or mean, the
                                  linear structure market returns lacked; and the same `learn_dynamics` validated on a fluid
                                  field (Stam's FFT-on-a-torus advection-diffusion -- prediction error 0.011
                                  vs persistence/mean, a surrogate rollout tracking the true sim to ~3.5%, and
                                  the honest limit where shock-forming nonlinear Burgers flow beats it); and multi-terminal network design (`design_network`,
                                  the Tero/Physarum Tokyo-rail flow model -- mu tunes a near-minimal Steiner
                                  tree against a fault-tolerant mesh, returned as a queryable B7 typed
                                  graph-memory whose unbind+cleanup recalls a node's neighbours); and cross-modal recall (`image_archive`, the
                                  exact DCT-plate store now reachable from the mind -- describe an image with
                                  tags and `recall_by_tags` retrieves it, `tags_of` runs the reverse, robust
                                  to 40% plate erasure); and generation over a COMPOSED subspace (`generate_structure`,
                                  the B10 denoise-from-noise sampler run over the manifold of role-filler
                                  structures via a slot-wise projection -- novel-but-valid compositions where
                                  the bare codebook only returns a stored atom); and a fractal scene from a seed vector (`fractal_seed` /
                                  `fractal_scene`, one kernel encoded holographically in a single vector,
                                  decoded and repeated to depth into a self-similar scene whose box-dimension
                                  matches log(N)/log(1/scale)); and anisotropic splats with a 3-D extension
                                  (`splat_aniso`, the real 3DGS primitive -- oriented full-covariance
                                  Gaussians fit by an analytical-gradient NumPy Adam, beating isotropic by
                                  ~45 dB on oriented 2-D and 3-D structure, a local optimum honestly labelled); and a tensor-product / tensor-train (MPS) bind mode
                                  (`tensor_bind`, the uncompressed cousin of HRR's circular convolution --
                                  far higher fidelity at fixed load and exact recall for orthogonal keys, with
                                  a lossless MPS compression of low-rank bindings, all bought with storage, not
                                  free over HRR); and the Path D federation + width faculties
                                  (`storage_array`, a RAID-style federated store whose parity
                                  reconstructs a lost shard exactly, and `superpose_compute`, the
                                  WIDTH faculty that evaluates K computations in one vector -- the
                                  conservation-law arc, with the capacity walls kept as negatives); plus exact RNS-phasor arithmetic
                                  (`exact_matmul`, integer / fixed-point matmul carried as residues
                                  over coprime moduli with phasor-binding modular accumulation --
                                  exact where a lossy bundle degrades, range federating over moduli); and a recursive pivot-tree index
                                  (`pivot_index`, sublinear nearest-item recall as nearest-pivot
                                  cleanup applied recursively -- greedy top-1 matching an exhaustive
                                  scan at ~O(log N) comparisons, a beam recovering recall); and sketch-routed array recall
                                  (`storage_array(...).routed_recall`, content-addressable shard
                                  routing by per-shard key-sketches -- as accurate as the directory
                                  while unbinding only the top-c shards, where broadcast erodes O(K)); and a distributed forward pass
                                  (`distributed_forward`, the storage federation applied to the
                                  matmul -- weight rows across K shards move the class wall ~Kx,
                                  with depth cured by exact per-layer arithmetic or cleanup-gating); the same federation extended to candidate
                                  SELECTION and SEQUENCE recall (`superpose_compute(..., shards=K)`)
                                  and to the image archive (`federated_archive`, K HolographicArchive
                                  shards with a directory -- capacity federates, recovery conserved)
    test_holographic_relations.py meaning as the recovered relationship: explain/name/map/chain
    test_holographic_protocol.py  D1: protocol-as-data anti-pattern audit -- a protocol's step structure read
                                  back from its program vector, flagging a SEARCH with no procedure-matched
                                  NULL and a select-then-score with no out-of-sample SPLIT (the audit_procedure faculty)
    test_holographic_knowledge.py D3: the findings registry -- a research log as a holographic knowledge
                                  structure: structured claims recalled by (role-sensitive) similarity, and the
                                  log's own contradictions detected, flat vs conditioned (the finding_registry faculty)
    test_creature_gauntlet.py     the maze gauntlet: gamified debugging, system lessons in mazes
    test_app_creature.py          the app's creature endpoint round-trip

Research / provenance -- one-off scripts whose results are recorded above and in
`figures/`; none is imported by the library, the app, or the tests. They were
moved out of the root into `archive/` to keep the working set readable, and each
adds the repo root to its own import path so it still runs from anywhere:

    archive/exp_*.py, archive/bench_vs_jpeg.py (add --fig for the corruption figure),
    archive/bench_batch.py, archive/bench_sprites.py, archive/bench_fig.py,
    archive/make_test_image.py        (e.g. `python archive/exp_wht.py`)
    benchmark_holographic.py, stress_holographic.py    (still at the root: these are
                                       the live measurement suites, not one-offs)
    figures/   rendered results

---

## Variance and credibility (`holographic_measure.py`)

This whole engine runs on random vectors -- random atoms, random projection trees,
shuffled splits -- so any single-seed score is one sample from a distribution. A
variance harness (`measure` / `assert_robust`) runs each load-bearing claim across
seeds and reports a mean, a standard deviation, and a 95% bootstrap confidence
interval; the load-bearing tests then assert the **lower CI bound**, not the mean, so a
lucky seed can't pass a test the typical seed would fail. Measured across seeds on the
real corpora (run `python holographic_measure.py` for the live table):

| Claim (real corpus) | mean ± std | 95% CI | verdict |
|---|---|---|---|
| next-char accuracy (Gutenberg *Alice*, 6-gram) | 0.61 ± 0.00 | [0.61, 0.62] | solid |
| language ID (UDHR, 6 languages) | 0.99 ± 0.01 | [0.98, 1.00] | solid |
| word-boundary F1 (Brown, spaceless) | 0.60 ± 0.01 | [0.60, 0.61] | solid |
| topic classification (Reuters, 5 categories) | 0.83 ± 0.05 | [0.79, 0.86] | solid |

The first three are tight enough that the headline number is essentially seed-proof;
the Reuters figure carries a real ±0.05 spread (a single seed could read anywhere from
~0.79 to ~0.86), which is exactly why it's reported with its interval rather than as a
point estimate, and why its test asserts robustness to a conservative 0.72 floor.

## Where VSA is load-bearing (`holographic_ablate.py`, `ABLATIONS.md`)

The sections above repeatedly find a simple baseline tying the holographic one. Taken
together that raises a fair question: which subsystems genuinely *need* VSA, and which are
a showcase where VSA isn't the reason they work? The ablation table answers it the honest
way — for each subsystem, the dumbest non-holographic baseline runs on the same real data
and metric, both are measured across seeds, and the confidence intervals decide the
verdict:

| Subsystem | Holographic | Dumbest honest baseline | Verdict |
|---|---|---|---|
| topic classify (Reuters, 5-cat) | **0.83 ± 0.05** | bag-of-words centroid: 0.61 ± 0.06 | **VSA load-bearing** |
| key→value, **noisy** keys | **0.89 ± 0.07** | exact dict: 0.00 ± 0.00 | **VSA load-bearing** |
| language ID (UDHR, 6 lang) | 0.99 ± 0.01 | bag-of-trigrams centroid: 0.99 ± 0.00 | uniformity |
| segmentation (Brown) | 0.60 ± 0.01 | exact count-based entropy: 0.61 ± 0.00 | not load-bearing |
| recall index (2000 items) | 0.82 ± 0.03 | exact brute-force scan: 1.00 ± 0.00 | scale, not accuracy |

The finding is sharper than "it does text and memory": **VSA is load-bearing exactly where
the problem is approximate or compositional** — recovering a value from a *corrupted* cue
(an exact dict scores a flat zero the moment a key is perturbed; cosine cleanup recovers it
at ~0.89), or folding co-occurrence structure into a representation (topic classification
beats raw word counts by ~0.22). It is **decorative where an exact, countable statistic
already settles the task**: character-trigram counts tell languages apart and branching
entropy finds word boundaries whether or not you express them in VSA, and the exact
estimators tie or marginally beat the holographic ones. The recall forest is honestly
behind exact scan on recall but reaches it at ~41% of the comparisons — a sublinear *scale*
win, recorded as such rather than dressed up. Full write-up in `ABLATIONS.md`; reproduce
with `python holographic_ablate.py`.

## Procedural generation, four modalities (`holographic_generate.py`)

The engine can also *produce* output, not just analyse and remember it -- procedurally,
by driving decoders it already has, with no learned distribution and no gradients. Each
generator clears the same bar as everything else: it beats the dumbest honest baseline on
a measurable metric (`python holographic_generate.py` prints the live numbers).

- **Image & video morph.** Spherical interpolation between two stored images *in the DCT
  coefficient domain*, inverse-transformed per frame. The win over a pixel crossfade is
  measurable and specific: a crossfade midpoint *is* the double-exposure of the two
  pictures (ghosting, distance 0 from `0.5a + 0.5b`); the coefficient-domain morph blends
  structure, so its midpoint sits measurably away from that double image (~0.06). Threaded
  through a list of keyframes, that's procedural video.
- **Text.** Nucleus (top-p) decoding with an optional repetition penalty, over the
  existing holographic n-gram distribution. Trimming the unlikely tail raises the
  real-word fraction from ~0.79 (plain temperature sampling) to ~1.00 -- markedly more
  coherent -- at a modest diversity cost (distinct-4-gram ~0.87 → ~0.77). That's the real
  top-p tradeoff, reported rather than hidden; the repetition penalty defaults off because
  this n-gram doesn't loop.
- **Audio.** Sonify an existing symbolic sequence: map each symbol to a fixed pitch and
  render short sine tones to a real WAV. This is faithful *rendering*, not a learned
  synthesiser -- the honest claim (asserted in the tests) is that distinct symbols produce
  distinguishable, repeatable pitches, deterministically.

This is the on-ramp to native generation: the next step up, below, runs the resonator
*forward* to compose new attribute scenes rather than interpolate stored ones.

## Forward compositional generation (`holographic_compose.py`)

The step up from interpolating what's stored to **composing what was never stored**. The
scene decomposition path runs backward (image → auto-tag → resonator factors the scene
vector into its colour/shape/texture atoms); this runs the same machinery *forward* — pick
attribute tags, `encode_scene` binds and superposes them into a scene vector, and
`make_scene` renders it to an actual image. No new model, no gradients: existing structure
driven forward.

The honest part is the bar. A generated scene is only meaningful if it can be **analysed
straight back to the spec it was built from**, so every generator is measured by
*round-trip fidelity*, and the combinations are drawn to be **novel** (excluded from any
"seen" set) so a correct round-trip proves composition, not recall:

- **Novel single-object compositions** factor back to their exact tags ~100% (40/40), and
  the whole composable vocabulary (7 colours × 4 shapes × 4 textures) round-trips ≥97%.
- **Novel multi-object scenes** recover the full set of objects through 4 per scene
  (~29–30/30), via the resonator's explain-away peel.
- **Rendered pixels auto-tag back** to the composed shape and colour (≈40/40 each) — the
  generated image is a real, analysable picture, not noise.
- **Animation** is composing a frame per step while one attribute sweeps a sequence; the
  trajectory is "real" because every frame's vector factors back to its intended value
  (100% on-target for a colour or shape sweep), each frame carrying its rendered image —
  procedural video by composition.
- **Nested composition (fractal).** The same bind-and-superpose that builds a scene from
  objects builds a *scene-of-scenes* one level up: each sub-scene is composed, bound to a
  seed-derived group atom, and superposed; `decompose_nested` peels each group back out and
  factors it — the identical unbind-then-factor at two levels. A sub-scene is to the
  super-scene exactly what an object is to a scene ("same above, same below"). Measured
  recovery: exact at 2 groups (1.00), ~0.97 at 3, then graceful decay (0.89 at 4, 0.82 at
  5) as group-binding cross-talk accumulates — the flat scene's capacity limit, one level
  up. The group atoms are derived, so the whole nesting regenerates from one seed.

All of this is **wired into the live console**, not just the library: the UnifiedMind app
(`unified_app.py`) carries panels for *compose a scene* (forward composition + round-trip +
animation), *morph* (coefficient-domain slerp vs crossfade), *nucleus text* (top-p decoding
vs temperature), and *save & reload* (persist the learned meaning space through the
versioned core). Each is backed by a real endpoint and covered by tests that hit the live
routes.

## Frozen core + persistence (`holographic_core.py`)

The kernel that everything builds *on* now has a stable, documented import surface, and a
trained mind can be saved and reloaded instead of recomputed. This is the gate that turns
"interesting modules" into "a thing you can build on."

- **The kernel facade.** `holographic_core` re-exports the primitives —
  `random_vector`, `unitary_vector`, `bind`, `unbind`, `bundle`, `permute`, `cosine`,
  `slerp`, `cleanup`, `Vocabulary` — with stable signatures. It's an *extraction, not a
  rewrite*: the functions still live in `holographic_ai.py`, so nothing existing changes;
  build-on-top code (the reservoir, the generation bundle, the forward renderer to come)
  imports from the frozen surface rather than reaching into a subsystem's internals.
- **Versioned save/load.** `save(obj, path)` / `load(path)` round-trip any object that
  exposes `to_state()` / `from_state()` — `Vocabulary`, `HolographicMind`, `HoloForest`,
  and the whole `SelfOrganizingMind` (the heart of `UnifiedMind`: its learned encoder plus
  the classified prototype bank) — through a single npz, stamped with a `STATE_VERSION`. A
  trained, *consolidated* brain (one that has projected into a low-rank basis) reloads with
  its banks and basis intact and decides identically; the recall forest's seed-derived
  trees rebuild byte-for-byte from the stored items; and a saved memory classifies
  identically after reload **even on never-seen words**, because the vocabulary's rng state
  is persisted so atoms minted after a reload match a run that was never saved. An
  incompatible version **fails loudly** on load rather than returning a silently-wrong
  object. The recent-experience buffer and transient EMAs are deliberately not persisted —
  they're self-healing scratch state.

## Performance and storage

A couple of measured wins that touch the whole stack, both behaviour-preserving:

- **Vectorised cleanup.** `Vocabulary.cleanup` over the full vocabulary (the hot path used
  by text classification, scene factoring, and recall) now runs as one cached
  matrix-vector product instead of a Python loop of per-name cosines — about **9× faster**
  on a 500-atom vocabulary (~1900µs → ~200µs), with bit-for-bit the same answer (stored
  atoms are unit length, so the dot is the cosine up to the query's norm, which doesn't
  change the argmax). The stacked matrix is cached and rebuilt only when the atom set
  changes. The explicit-candidate-subset path keeps the simple loop, since those sets are
  small and one-off.
- **Vectorised prototype classification.** `SubPrototypeMemory.classify` /`label_scores` —
  the single most-called scan in the brain, since every `classify`, `recall`, `decide`, and
  `classify_robust` routes through it — uses the same cached unit-matrix product instead of
  a per-prototype loop. The winner is identical to the old loop (scores match to machine
  epsilon, since a matrix product and a per-element loop sum in a different order). The
  cache is keyed on a mutation counter so an in-place online update can never leave it
  stale.
- **Vectorised FHRR cleanup.** `PhasorVocabulary.cleanup` over the full atom set is the same
  loop-to-matvec win in the complex domain, found by the INV-5 profiling pass (INV-1's
  cosine-grep had missed it, because the FHRR similarity is a *helper* call, not a literal
  cosine). The FHRR similarity is the real part of the normalised complex inner product, and
  `real(vdot(b, q)) == Re(b)·Re(q) + Im(b)·Im(q)`, so the whole atom set scores as **two real
  matrix-vector products** — about **11× faster** at 2000 atoms (~19.6ms → ~1.8ms),
  argmax-identical with similarities matching to ~1e-16. The honest subtlety kept on the
  record: the *obvious* `np.real(B.conj() @ q)` is no faster than the loop, because `B.conj()`
  allocates a full complex copy whose cost matches the matvec — the no-copy real-matvec form is
  the actual win. Cached like the others, invalidated when a new atom is minted; the
  candidate-subset path scores the same way without caching. (This is a value-cleanup path, not
  the creature's tie-sensitive decision path, which the `bind_batch` lesson keeps hands off.)
- **Half-size saves.** `holographic_core.save` stores float arrays as **float32** by
  default, roughly halving every saved mind (e.g. a small `SelfOrganizingMind` 126 KB →
  65 KB). These vectors are only ever compared by cosine, where float32 is far more
  precision than the decisions need; on realistic probes behaviour is unchanged (a
  classification only flips on an exact tie, where either answer is equally valid). Pass
  `compress=False` for a bit-exact round-trip when that matters.
- **FHRR for high-capacity binding** (`holographic_fhrr.py`, opt-in). The literature's
  most-cited cross-VSA comparison recommends FHRR (complex unit-phasor atoms, binding by
  element-wise complex multiplication) for capacity. Measured on this substrate, an FHRR
  key→value trace holds far more pairs than the real-valued HRR core before readback breaks
  (at 40 pairs / 256-d: ~0.90 vs ~0.61; at 60: ~0.74 vs ~0.40). It's offered as an owned
  faculty (`UnifiedMind.high_capacity_memory()`) for that specific high-load regime — the
  readable real-valued HRR stays the default, since at the few-factor loads the engine
  normally runs both are already perfect, and the project's `unitary` atoms (real domain) do
  *not* capture FHRR's advantage on their own.
- **int8 quantized saves** (`save(..., quant="int8")`, opt-in). The scalar-quantisation
  trick vector databases use to shrink stored embeddings: each float array is stored as
  signed 8-bit integers with a per-array scale, dequantised on load. ~3× smaller than
  float32 (~5–6× vs float64). Measured **lossless for classification** at the working
  dimension — the prototypes are near-orthogonal there, so 8-bit precision leaves the
  nearest-neighbour argmax unchanged (verified on crowded 100-class spaces). Like float32 it
  can flip an exact tie, so float32 stays the default; int8 is for when stored size matters.
- **Dynamic quantization** (`save(..., quant="auto")`, opt-in). Precision per array follows
  the data's own complexity and size: an array whose separation proves 8-bit lossless is
  stored as int8 (~4×), tiny or marginal arrays as float32. The level is chosen by a measured
  margin gate (int8 is taken only when it preserves every row's self-recognition and ≥70 % of
  its top1–top2 margin), so an auto save matches a float32 save's *decisions* — verified
  decision-safe across the whole stack: the classification memory (0 flips), the recall forest
  (0 flips), and the creature value-brain (tie-level). Only magnitude-preserving levels are
  auto-selected: a 1-bit binary level was tested and dropped, because it distorts the
  pairwise-similarity geometry by ~0.1–0.2 on every array and only survives where the decision
  is a wide-margin classification argmax (it flipped 62/200 of the value-brain's actions) — its
  safety is decision-specific and can't be verified at the generic persistence layer.

## Honest limitations

- **Not a competitive image compressor** - use JPEG/WebP/AVIF for small files.
- **Hard capacity / damage cliff** - image recovery is graceful only until the
  surviving plate cells drop below the stored coefficient count (the demo archive
  sits at load 0.37, so its cliff is ~63%; the UI caps damage at 70%).
- **Small-scale by design** - this is a clear, readable, from-scratch engine for
  learning and experimenting, not a tuned production system. The text corpus is
  tiny, the creature world is small, and the vectors are modest. It is built to be
  understood and extended.

## Design notes

- **Relations -- meaning as the recovered relationship** (`holographic_relations.py`,
  `UnifiedMind.explain()`): similarity says THAT two things are alike; these
  operations say WHY and HOW. Over role-bound records (what the encoder already
  builds from dicts): EXPLAIN decodes the per-role verdict ("france is like
  belgium BECAUSE currency/language/continent match; UNLIKE because the capitals
  differ" -- 4/4), NAME recovers how a filler relates ("paris relates to france
  AS capital" -- 100%), MAP answers "what is the dollar of mexico?" (100%), and
  CHAIN composes hops ("the language of the country with the currency of the
  country whose capital is X" -- exact through three hops). The measured law
  that shaped the API: meaning survives composition only when it touches
  SYMBOLS between steps -- the direct algebraic relation vector scores ~94%,
  its failures are pure HRR noise, and dimension does not save it (96/94/90% at
  1024/2048/4096), while routing each hop through a cleanup is exact: the
  discrete vocabulary is the error correction that makes chained meaning
  reliable. And the operations are CROSS-MODAL with zero new machinery
  (`holographic_scene.explain_objects`): two raw images go through the existing
  auto-tagger (colour/shape/texture), the tags become role-bound records, and
  the system answers why one image is like another -- "shape SHARED (circle),
  colour differs (red vs green)" -- measured end-to-end on generated shapes at
  72/72 = 100% verdicts (the tagger itself is 36/36 on ground truth), with
  chains working over image stores too ("the colour of the rectangle-shaped
  object"). And the operations are UNIFIED, not a side library: the mind runs
  them on its OWN memory -- find(role, filler) scans the records absorb()
  already stored, read_role() decodes a role from a learned class prototype
  against the filler vocabulary learn() registered from experience, ask()
  chains multi-hop questions over the absorbed store, and explain() takes
  either fresh dicts or two LEARNED labels. The measured payoff of unification:
  classes built from six noisy, incomplete observations each (one random role
  dropped per copy) still decode perfectly -- read 40/40, explain verdicts
  180/180 over all pairs, 3-hop chains 100% -- because superposition linearity
  reinforces the shared role-filler terms while the dropouts average out. The
  mind explains concepts it LEARNED, not just records it is handed. And the
  INCEPTION step: explain_splits()/explain_organization() turn the relations
  decode on the mind's own memory ORGANIZATION -- when the organizer splits a
  class, the sub-prototypes' roles are decoded and contrasted, naming what the
  split separated ('A divided because one mode is colour=red/shape=circle, the
  other blue/square'). Separation is judged by CONTRAST (each mode's winner
  genuinely absent from the other: ~0.5 for real structure vs <=0.1 for
  incidental skew), and the statistic's first outing caught the organizer
  red-handed: one XOR label's split turned out to separate the NOISE role --
  accuracy-sufficient, structurally arbitrary -- and the explanation reported
  it honestly. The same introspection on the creature's brain
  (describe()/why_differ()) articulated and SOLVED the cluttered-forager open
  problem (see the survival section). And the capabilities live in the MAIN
  MODELS, propagated to every consumer: the safety reflexes moved INSIDE the
  brain (`HolographicMind.decide(senses=...)` vetoes moves into seen poison or
  walls; run_episode's flags, the demos, and the showcase app's own creature
  loop all route through the one mechanism -- the on-camera creature no longer
  suicides or wall-bumps), and the unified mind keeps a JOURNAL: every road to
  auto_reorganize narrates itself ("reorganized: 'A' went from 1 to 2
  sub-prototypes, the modes differ in colour, shape"), maintain_now adds the
  decision brain's measured keep/fold/refresh verdict to the same entry (the
  whole self-maintenance story in one place -- and honestly: a brain that
  self-maintained mid-stream reports "kept", because by maintenance time
  nothing WAS stale), with the console's organize panel showing the mind's own
  account verbatim. The operations now run on REAL data twice over: the
  712-sprite library absorbs as image + auto-tag/name record per label (roles:
  colour, texture, family, facing, frame), and the measured new result is that
  role decode survives a MIXED prototype -- an image vector superposed with the
  record -- at 100% (750/750), with the cross-modal loop closing at 96%: SEE a
  sprite, classify it, SAY its colour in symbols. And the console gained a
  'Countries (records)' dataset (ten countries from eight noisy observations
  each, 97% held-out) plus a RELATIONS panel: explain two learned labels
  per-role, find by attribute, chain a two-hop ask -- the mind answering WHY
  over its own memory, in the browser -- the ask row takes arbitrary chains
  ('capital>currency, currency>language'), each hop cleaned up to a symbol. PROVENANCE -- can a generated or pasted passage be traced to its sources?
  The stores already hold the answer: the recall index keeps every absorbed item
  with its payload (find() returns exact provenance), and the sequence model now
  records, for every context->token transition, WHICH source documents taught it
  (a doc-counter beside each count, zero cost when unused). attribute(text) ranks
  the fitted sources by the transitions a passage actually uses -- and the
  measurements drew a sharp, honest boundary. Attributing GIVEN text is the
  well-posed question: 70% top-1 on a clean four-book Gutenberg split, 92% on
  five, 8/9 windows localized in spliced text -- and the level matters
  (coarsest-chunk-first measured 70% vs 42% atom-only and 48% all-levels,
  because an author's characteristic multi-character chunks are the signal while
  'th'->'e' is shared by everyone). Attributing freely-GENERATED low-order text
  is NOT well-posed: after the seed it drifts into transitions every source
  shares, so the ranking goes near-uniform -- and the UI says so rather than
  faking confidence. An inverse-document-frequency refinement was tried and
  measured a wash (the multi-order context already carries the distinctiveness),
  so it was dropped. COHERENT RESOLUTION (the default) realizes a sharper
  principle: a passage usually comes from ONE source, so a transition only one
  source taught (the word 'fillet' in one book) is near-certain provenance while
  a shared one ('butterfly' in three) is weak -- so each transition's vote is
  weighted by its SPECIFICITY (inverse number of sources that taught it), and the
  unique tokens PIN the source while the shared tokens confirm rather than smear.
  Measured: lifts confidence on ambiguous short passages (+2-4 points top-1 at 60-100
  chars, a wash where evidence already saturates) and sharpens the margin
  (a Melville probe went 0.37 -> 0.68). A sequential RUNNING-PRIOR on top of this
  (let the leader-so-far bias later tokens) was tried and measured a
  wash-to-negative -- specificity already captures the insight, and a feedback
  loop risks runaway commitment to an early wrong guess -- so it was kept out.
  The console gained a provenance panel: generate then 'Trace sources', or paste
  any passage and see the source ranking as weighted bars.

  SEQUENCE ALIGNMENT then closed the gap the bag could not: it answers "whose
  STYLE" but not "whose actual MATERIAL", and a sentence sharing every word with
  sources of OPPOSITE message (a bullish vs bearish thesis differing only in
  'up'/'down') is attributed by the bag to the wrong source. Meaning is in the
  ORDERING, and nature solves this the way genome alignment does -- identify a
  fragment by its longest contiguous verbatim match, not its token composition.
  align() scores maximal verbatim spans by length x specificity; measured 100%
  top-1 on verbatim-clause probes (bag 97%) at a ~3.5x margin, and it gets the
  bull/bear theses BOTH right where the bag confidently picks wrong. trace()
  reports STYLE and MATERIAL and leads with whichever is decisive (a long
  verbatim span => quoted/assembled; none => paraphrase/original-in-style); the
  console shows the verdict, its basis, the deciding span, and both rankings.

  SEQUENCE / ORDER / TIME as a first-class property (holographic_sequence.py).
  A sweep prompted by a sharp observation -- the same steps of a peanut-butter
  sandwich in the wrong order are not a worse recipe, they are not a recipe --
  found the stack treats most things as order-FREE, rightly (topic = bag of
  words, class = bundle of examples, record = set of bindings; "what is this
  about" does not depend on order). But some meaning lives ONLY in the sequence
  (plans, recipes, proofs, protocols, timelines), and nothing could QUERY order.
  SequenceMemory fixes that with the same primitives (bind/bundle/permute): each
  step rotated by its position, order recoverable. Measured 100%: step(i) reads
  the i-th step, position_of(x) finds where x occurs, precedes(a, b) answers
  whether a precedes b, and validate(constraints) runs the PB&J test -- does
  every 'a before b' rule hold? -- naming exactly which step is out of order.
  (A what-comes-next encoding measured ~64%, the bundle-capacity ceiling the
  scaling work charted, so next-step is left to the exact list; this memory owns
  the ORDER RELATIONS no bag store can answer.) Wired into the unified mind
  (learn_plan/step_at/precedes/validate_plan) over the SHARED symbol space. The
  encoder already made the right call elsewhere: a word-list infers as an
  order-free bag for classification (97.5% vs 93.8% via the sequence path) --
  order is restored where it carries meaning, not blanket-applied.

  SELF-DISCOVERED SEQUENTIALITY -- the organizer learning, without being told
  and without a magic number, that a class is ORDERED. The honest test is a
  permutation test against the data's OWN shuffle: does the real order of a
  class's members predict the next element better than the same members with
  order destroyed? A transition model is built leave-one-out and scored by how
  much higher the true next element ranks than the others (a graded margin, not
  argmax -- argmax saturates on small step vocabularies); the baseline is the
  mean over shuffled copies, so the class is its own null hypothesis. The result
  is a z-score (signal in units of the null's own spread), and z>2 -- the
  standard 'two sigma above noise' bar, a statement not a tuned constant --
  calls a class sequential. Measured: ~+16 for genuinely ordered classes, ~0 for
  an order-free bag of the same elements (real order indistinguishable from
  shuffled), degrading gracefully through partial noise (still strong at 30%
  scrambled, at the boundary near 50%, silent once order is gone). A class that
  passes gets its canonical order SELF-ASSEMBLED from the members by a pairwise-
  precedence vote -- the mind reconstructs a sequence it was never shown whole,
  exactly, even from drop-one partial observations -- and gains order queries
  (precedes/validate). The mind absorbs sequential and bag classes mixed,
  discovers which is which, and acts only on the real structure: order as a
  DISCOVERED organizational property, measured into existence, not declared.

  RECURSIVE / FRACTAL discovery -- the same order-test applied at every layer.
  Once a class proves sequential, each of its steps is tested for its OWN
  internal order (where the data provides sub-observations), and the structure
  unfolds into a tree the mind was never given the shape of: a nested recipe's
  top order is recovered, its expandable steps (make_sauce, prep) recurse into
  their sub-recipes, and the recursion STOPS honestly -- at atomic steps with no
  sub-observations, and (the real test) at steps that HAVE sub-observations but
  in unordered form (a garnish whose ingredients carry no order is correctly
  NOT expanded, told from an ordered sub-recipe by the permutation test alone).
  No depth is declared, no shape assumed; each layer is measured into existence
  by the same z>2 bar, and self-assembly recovers each layer's canonical order.
  Sequence discovery made fractal: structure all the way down, until the data
  says stop.

  SELF-PROOF + CONTEXT-BINDING -- structure must prove itself before its meaning
  is trusted, and steps are generic until context fills them. Two additions. (1)
  A discovered order can score z>2 yet be INCONSISTENT: if members' pairwise
  precedences form a cycle (A before B, B before C, C before A) no ordering
  satisfies them and the plan cannot be executed. prove_executable does a
  topological feasibility check -- structure earns trust by passing, not by z
  alone -- and gating registration on it immediately caught a real bug: a
  score-heuristic canonical sort had misplaced a rare step against a 4-0
  majority; the proof surfaced it, and a proper topological sort fixed it
  (structure validating structure found an error before it shipped). (2)
  extract_template discovers the generic SCHEMA and its context-bound SLOTS in a
  repeated step: 'the material has density X' is fixed words plus a slot that
  varies across observations (5g, 3g, ...), separated by per-position entropy and
  split at the natural largest GAP (the data's own scale, no constant). This is a
  physical law -- 'F = m*a' is generic until a scenario BINDS the values; the
  schema is the law, the slots are where context enters, exactly as 'open the
  book' leaves 'book' to be filled from prior context.

  EXECUTION -- the closed loop, from discovering structure to ACTING on it.
  execute_plan runs a discovered, PROVEN plan under an honest contract: a step
  fires only when every step that must precede it has already fired AND its
  context slots can be bound from the scenario; otherwise it BLOCKS, reported
  with its reason (an unmet precondition naming the steps still needed, or an
  unbound slot naming the missing context) rather than silently assumed away. A
  templated step fires as its bound form ('cut into 2 pieces' when context
  supplies pieces=2); without the binding it blocks, and steps behind it
  cascade-block truthfully. An unproven plan cannot be run -- you cannot execute
  what discovery and proof never registered. The full arc stands: discover order
  (permutation test) -> prove it executable (topological feasibility) -> bind
  context into its slots -> RUN it, every stage measured or proven, every failure
  informative.

  WIRED THROUGH THE STACK (not living in tests). absorb() now AUTO-DISCOVERS
  order: hand it ordered list-examples and it runs the permutation test on each
  class, proves the winners executable, and registers them -- order is a property
  of self-assembly, not a manual call. Verified at scale: 240 mixed
  procedure/bag examples absorbed in one call, all four procedures identified
  with EXACT canonical-order recovery and both bag classes left alone. The
  CREATURE uses it: a trained maze brain's successful escape routes are captured
  (capture_route) and the sequence machinery discovers their route is genuinely
  ordered (z up to ~68 vs its own shuffle) and proves it executable -- the
  creature acts, then understands the structure of its own action, surfaced live
  in the showcase maze panel. And the UI exposes the whole pipeline: a
  plan-discovery panel (/api/plan) absorbs noisy procedure observations mixed
  with bag distractors and shows, with no labels, the mind discovering which are
  ordered, proving them, recovering the order, and executing under the honest
  contract (in-order fires, out-of-order blocks). And the discovered plan composes with action:
  replay_plan drives navigation from a proven route instead of re-deciding each
  step, validating every move -- in its own maze the plan escapes 10/10, in a
  CHANGED maze it detects exactly where it breaks (the blocked cell) rather than
  falsely succeeding, so the creature knows the boundary of its learned structure
  and the seam where it would need to re-learn.

  THROUGHPUT (the raytracing parallel). A relation chain is a ray bouncing
  through the holographic space -- each hop a bounce, the cleanup-to-a-symbol the
  surface intersection, the cleanup confidence that bounce's reflectance. Path
  tracing accumulates THROUGHPUT (the product of reflectances) and terminates
  paths that lose too much energy; both transfer. ask_traced accumulates the
  per-hop confidences, and the product is a calibrated trust in the chained
  answer: on a dense interfering store it separates correct chains (~0.23) from
  wrong (~0.10), and abstaining on the low-throughput half lifts answered
  accuracy from 71% to 85%. A chain whose throughput decays below a floor
  ABSTAINS rather than emitting noise -- the ray that ran out of energy
  contributes nothing. The console relations panel shows the answer, its
  throughput, and the per-hop confidences. (A revisit of the kept negatives
  against the new machinery re-confirmed them: competence-weighted flocking still
  loses to best-pick -- in the regime a committee should help, no candidate is
  good enough to yield a signal to align toward, and once they are, best-pick
  already wins -- and the fractal/curation negatives still stand, the new tools
  not displacing a gate that already harvests its signal. A negative is overturned
  by a measured win, not a fresh analogy.)

  MULTI-RAY (many rays per pixel). One query is a noisy point sample; path tracing
  fires many rays and averages, and the same recovers errors a single encoding
  makes. classify_robust fires several word-resampled views of a text query and
  combines them -- the crucial step is Z-SCORING each ray's per-label evidence
  before summing, so a confident-but-wrong view cannot dominate (the naive vote's
  failure, and flocking's). Measured: with feature lenses ranging 100/100/50/17%
  the z-scored ensemble reaches the best single lens BLIND, and on a noisy text
  task it lifted classification 89% -> 100% with no regression on clean queries.
  Each view is a SHADOW of the input from a different angle; the ensemble is the
  form no single shadow shows. The console classify panel reports the multi-ray
  label and the fraction of rays that agreed.

  MULTI-RAY CHAINS, by contrast, are a clean NEGATIVE for accuracy -- and the
  contrast is the lesson. Firing several throughput-traced relation routes to one
  answer and combining them does not help: a route through a unique key is already
  exact (nothing to add), and routes through shared values fail for the SAME
  reason (correlated errors), so combining averages noise -- naive voting even
  made a perfect route worse (100% -> 75%), reliability-weighting only matched the
  best route, and where all routes were ambiguous the combo (27%) lost to the best
  single (52%). Multi-ray helps only when the rays' errors are INDEPENDENT, which
  the feature-lens classification views are and the chain routes are not. The kept
  artifact is route_reliability: a self-measured 1 / mean-fan-out (unique role =
  exact key = 1.0, shared role = ambiguous = low) that ranks which find()
  operations to trust, no magic number -- a good negative leaves something behind.

  PROJECTION TO CREATE NEW THINGS. Casting one record's attributes onto another's
  frame synthesizes a NOVEL entity -- 'france with japanese language and the yen'
  -- that exists in no training data and decodes back to exactly the intended
  blend (100% over random blends). blend() does this directly; project_transform()
  does analogy AS GENERATION (the a->b per-role delta projected onto c creates a
  coherent hybrid: japan's geography, germany's distinctive capital and language).
  The honest split the investigation found: retrieval analogy (FIND the existing
  d) hits a uniqueness wall -- the cleanup law makes every entity an exact key, so
  there is no graded nearness for a transform to climb -- but GENERATION (MAKE the
  specified new thing) is well-posed and exact. Creation sidesteps the wall
  retrieval hits; the line a good negative names is the one between finding and
  making. Wired into the knowledge store and the unified mind (synthesizing over
  its own learned classes), and shown in the tour. And projection lifts to multi-object
  SCENES, where the parts must first be discovered: blend_scenes takes two scene
  vectors (objects unknown), factors each into its objects via the resonator,
  projects one factor across (scene A's forms wearing scene B's palette), and
  recomposes a NOVEL scene that factors back to exactly the intended hybrid
  (100% across all three factors and 2-4 separable objects). The full decompose
  -> project -> recompose loop, all through the resonator -- the part that was a
  recovery tool now drives generation. Honest boundary: recovery rides the
  resonator's capacity (separable objects exact; colliding objects degrade as
  multi-object factoring always does). Shown as a third 'projection blend' demo in
  the scene panel and in the tour. Projection then unfolds over TIME: a smooth
  attribute morph is impossible (interpolating one colour atom red->blue is a
  crossfade-with-snap -- the resonator reports red until t~0.55 then flips hard,
  the cleanup law holding discrete coherent states), so the honest morph is a
  SEQUENCE of discrete coherent frames: a control parameter sweeps 0->1 and the
  objects adopt B's attribute one at a time, every frame factoring exactly, A
  first and B's full pattern last. morph_scenes builds it, and the loop closes
  with the sequence machinery -- the morph as a flip-count token sequence passes
  the sequentiality permutation test (z~10 vs its own shuffle), so projection
  generates the frames and sequence-discovery confirms the order. Shown as a morph
  strip in the scene panel and in the tour. And cardinality itself morphs: the
  object COUNT is self-measured (the scene is an unnormalised superposition of
  near-orthogonal unit products, so round(||v||^2) IS the count -- 96% exact over
  n=1..7, nobody tells the system n), and the scene vector is ALGEBRAICALLY
  EDITABLE -- removing an object is subtracting its factored product (explain-away
  repurposed as an editor), adding is adding one. morph_cardinality chains such
  edits from a 3-object scene down to one and up into a different 2-object scene,
  the count discovered at every frame, each frame factoring exactly, the final
  edited vector holding exactly the target -- never re-encoded. The composite is
  countable and editable, not just decodable. The same algebra becomes the creature's
  PERCEPTION: WorldView encodes the world's contents (exit, poison, walls) as a
  superposition of type(x)position products, so the count is the norm and the
  DIFF of two snapshots is itself a composite of the changes -- appeared objects
  positive, vanished negative, unchanged content cancelling exactly. The diff's
  norm counts the changes and count-driven peeling names them (100%/100% over
  mutated 16x16 mazes). The integration: a wall dropped on the creature's learned
  route makes replay_plan break at exactly that cell and WorldView independently
  names that wall -- perception explains the plan failure (6/6 at 9x9, 5/5 at
  16x16). Stress sweep: 16x16 escapes 100% across six seeds, braided+poison forks
  100%; 20x20 is the measured wall -- partly budget-shaped (one seed 0%->83% with
  more episodes/steps) and partly the BOOTSTRAP problem (another seed stays 0%
  under any budget or horizon: epsilon-greedy exploration never finds the first
  success in a deep-enough maze, so no reward signal ever arrives). The honest
  next step there is curiosity-driven exploration or a curriculum, recorded as a
  future thread. That thread is now pulled: the BOOTSTRAP RESCUE. (And a
  refinement found while pinning it: the wall's mechanism is not 'luck is
  hopeless' -- sustained high epsilon occasionally escapes -- but 'the
  loop-attractor policy locks in as epsilon decays, before luck consolidates';
  plain probes 0% at every budget tried.) Curiosity (first-visit bonus =
  exit_reward / n_free_cells, the world's own arithmetic; off at first escape,
  because visited-ness is not in the creature's state and the crumbs are
  unlearnable after their job is done) finds the first success (episode 4 vs
  never); rehearsal (one stored successful trajectory re-remembered per episode)
  consolidates it; capacity (512/30 where 256/15 loops in a 14-cell attractor)
  holds it. And measurement cut both ways: the same protocol HURT the seed where
  luck already sufficed (83% plain -> 0% with it; rehearsal alone 33%), so the
  integration is a rescue summoned by self-measurement -- candidates run plain,
  and only an observed starvation (zero training escapes, the data's own signal,
  no threshold) enables the bootstrap for subsequent candidates
  (bootstrap="auto", learn_maze's default). Verified: the luck-sufficient seed
  routes plain to 83%, the formerly impossible seed starves, rescues, and probes
  100%, and the 16x16 six-seed regression stays 100%.

  MARKET DATA (real numeric time series on the same substrate, with the honest
  split between memory and prophecy). User-supplied DEX candles (DAI/WETH, 100
  one-minute bars, checked in under data/) through holographic_market.py: a
  candle is ONE record (five roles bound to graded scalar codes; round-trip
  decode 1.6-2.9 bp vs the data's own 8.2 bp return sd -- resolution finer than
  the signal); the permutation test REDISCOVERS market structure (price levels
  provably ordered, z=+6.8 vs their own shuffle; return signs indistinguishable
  from shuffle, z=-0.6 -- the efficient-market property, found by the engine's
  own instrument); candle-level novelty at the data's own scale catches the real
  anomalies (the 2685-volume candle at z=4.2, the +21bp swing at z=5.7). And the
  kept negative, pinned in the suite as the honest claim: walk-forward next-sign
  prediction from the nearest motif is a COIN FLIP (49%), as is every baseline
  tried (46-55%), all inside the binomial 39-61% chance band at this sample --
  recall is memory, not prophecy, and the test asserts the band, not a win. AT
  SCALE (a second user dataset -- 15,793 SOL ticks over 2.2 days, ~1-second
  jupiter bursts plus 5-minute coingecko points, checked in as
  data/sol_5min.npz; analysis within-burst only, never across a hole): the
  SAME sequentiality instrument flips its verdict where real structure exists
  -- tick return signs are strongly ordered (z=+44, momentum, +0.20 sign
  autocorr) where DAI minutes were shuffle-like (z=-0.6). And the chance band,
  now +/-2.6% at n=1454, PROVES the momentum edge while assigning it honestly:
  persistence (last nonzero sign) 60.2%, outside the band; the holographic
  motif 54.1%, also outside chance but decisively below the trivial rule;
  always-predict-flat wins raw next-tick (ticks are 88% flat). Measurement
  beats sophistication -- the motif's value is memory and novelty, not
  direction-calling -- and the suite pins the ordering (persistence > motif >
  chance), not a win. STRUCTURE-FIRST, the follow-up round: the momentum is
  located (intra-burst only -- 60.5% within, chance across the holes, no
  burst-to-burst carry-over), bursts are mildly drifty, and move-shapes do NOT
  recur beyond their shuffled marginal (z=-0.6: momentum and drift, not chart
  patterns). And the validated win, RAY-PROJECTED PRICE TARGETS: a matched
  K-move pattern's R most similar past windows each carry the outcome that
  followed; the bundle's quantiles are the target distribution for the next H
  moves. Proper-scored and selection-quarantined (R chosen on the first half
  only): on the held-out half the rays beat the unconditional distribution at
  pinball loss (paired z=+3.3) with ~13% tighter calibrated intervals. The
  pattern locates the current context's outcome SCALE rather than calling
  direction -- and the confidence gauge is honestly labelled (it gates
  difficulty, not skill: the confident quartile helps the baseline exactly as
  much). THE HORIZON MAP answers 'how accurately, per timeframe' with the
  decay measured end to end: direction one move ahead only (persistence 58.3%,
  chance by H=2); point error never beats predict-zero and grows like sqrt(H)
  (2.3 -> 7.2 bp, H=1 -> 8) -- diffusion plus a one-move memory; the rays'
  calibrated-interval advantage is significant at H=1-3 and gone by H=8 (pinned
  held-out: win at H=1, absent at H=8); per-second direction dilutes to ~51%
  within bursts; and across the ~5-minute holes everything is chance (54.8 +/-
  4.9 at 5-10 min). The only validated predictive product is the calibrated
  target interval a few moves out -- and the suite enforces that the claim
  stays horizon-qualified. AT SCALE AND ACROSS INSTRUMENTS: a
  1000-candle DAI/WETH pull (10x the first slice, a 3x-tighter chance band)
  confirms every structural finding -- round-trip finer than the signal, levels
  ordered (z=+100), return signs STILL shuffle-like (the efficient-market
  verdict survives, and the tighter band makes the negative stronger), the
  53652-volume outlier flagged at z=8.1 -- while prediction stays chance (all
  baselines inside +/-3.2%). And the calibrated-interval win REPRODUCES on this
  second instrument (held-out H=3, paired z=+2.98, ~9% tighter intervals), with
  the edge living at H=3-5 on mean-reverting DAI candles vs H=1-3 on
  momentum-driven SOL ticks -- same product, opposite microstructure. The
  engine does not predict prices; it locates and calibrates uncertainty, now
  shown across two markets.

  TEMPORAL & SPECTRAL COMPRESSION (holographic_video.py). The physics compression
  win -- translation in value-space IS binding -- turns out to be the same thing a
  VIDEO CODEC exploits: a rigid pixel SHIFT is one binding, so an object moving
  across frames is one operator applied repeatedly. Keyframe + motion-compensated-
  residual (GOP) coding makes that free: a one-number motion search recovers a
  whole-pixel shift exactly, the residual goes to L2 0.000, and against per-frame
  INTRA storage GOP is ~10% smaller AND +0.4 dB on rigid motion -- a strict
  rate-distortion win. The honest boundary, pinned: deformation (a growing/morphing
  object) is not a rigid shift, so the motion model is wrong, residuals stay large,
  and GOP loses -3.7 dB -- motion compensation pays exactly when motion is the
  change. And the audio lesson unifies it: a 1-D signal's DCT is the spectral
  (MP3) basis, so the same HolographicImage machinery compresses a tone+transient
  8x at 34 dB and survives 30% erasure with no extra loss. Spatial, temporal, and
  spectral compression are one operation on the substrate. SUB-PIXEL MOTION extends this: a fractional
  drift is recovered exactly by a Fourier-shift motion search (residual to
  numerical zero, vs the residual integer search leaves) -- a pixel shift is a
  phase ramp in frequency, the scalar code's fractional-power principle in 2-D.

  VERSIONED HISTORY (holographic_history.py) -- a version history IS a video. The
  substrate is otherwise always 'now': a reorganization swaps the store in place,
  with no undo and no record of how it changed. Storing each version as a frame
  fixes both, using the GOP structure (keyframe + deltas) -- but LOSSLESSLY, since
  rollback needs the exact prior state (completing the compression picture: lossy
  spectral for perceptual data, lossless sparse-delta for state/history). Rows are
  keyed by stable id, not position (the git lesson, learned by measurement: a
  naive diff miscounts a deletion as an 86%-matrix change), so reorganization is
  genuinely sparse and the history compresses ~29x losslessly. checkout(v)
  reconstructs any version exactly; proof-gated commits reject a reorganization
  that violates an invariant (the store stays valid, the attempt stays in the
  audit log); rollback(v) restores a past version exactly and is itself recorded
  -- the timeline is append-only, nothing erased. The honest boundary: a dense
  update (every entry nudged) does not compress -- versioning is for structural
  history, not dense trajectories. The video codec became version control for the
  mind's own state.

  PHYSICS ON THE SUBSTRATE (holographic_physics.py). Additive kinematics turns
  out to be NATIVE to the scalar code: encode(a+b) == bind(encode(a),
  encode(b)) exactly, so motion is repeated binding -- a 15-step constant-
  acceleration trajectory integrates by pure vector algebra (max decode error
  0.06) and velocity is read off two observations by unbinding. Boundaries:
  trajectories must stay in the encoder's range, and multiplicative dynamics
  (damping, oscillation) are not native -- binding adds, never scales. On the
  market, the price-as-particle framing was tested against the validated
  tools and split honestly: at H=1 the two-number state (v, a) is EQUIVALENT
  to the 5-move shape rays (physics as compression -- the one-step structure
  IS kinematic) while beating the unconditional distribution; at H=3 the
  shape still wins; and prices have NO INERTIA -- kinematic extrapolation
  loses to predict-zero at every horizon, pinned so the metaphor stays
  honest.
  A WIRING SWEEP then made sure nothing
  stayed hidden in tests: the app's labyrinth pane carried a stale "no
  reactive brain can hold a 16x16 maze" early-return from before the gauntlet
  broke that ceiling, so the panel now shows TWO SOLVERS, ONE SUBSTRATE -- the
  brain that LEARNED the maze on the left (learn_maze protocol; escapes the
  braided 16x16 in 42 steps on camera) and the slime-mold colony computing the
  optimal 38-step tube on the right; the forage modes now TRAIN with the
  brain's safety reflexes (the veto shapes the experience learned from, not
  just the final moves -- measured in the app's own walls world: 48 -> 83
  stars across six lives, both deathless); the unified mind's decide() passes
  senses through to the same model-level vetoes; and every decision frame in
  the creature animation now carries the brain's own account, decoded live
  ("senses food_x=west, wall_S=yes -> W (value +0.79)") -- introspection on
  camera. GENERATION FIDELITY (user-caught): generated text had no capitals or
  punctuation -- the engines were innocent (the fractal coder takes raw
  characters), but every console loader fed them a scrubbed token diet
  (lowercased, isalpha-filtered), and BOTH generate endpoints lowercased the
  seed. The loaders now feed TRUE corpus text, measured on Austen: ~12% more
  bits/char (1.949 -> 2.175) and 8 points of word coherence for output that
  reads as PROSE -- capitals, commas, apostrophes, sentences. The flat n-gram
  gained a fold_case switch (default preserves every pinned number). And one
  engine lesson the debugging earned: tiling a small corpus (block x N) makes
  the whole block the OPTIMAL compression unit, so the chunk schema learns a
  corpus-sized mega-chunk and generation replays it wholesale -- the schema
  was right, the diet was degenerate; varied, independently-shuffled passes
  fix it. A seed that encodes to nothing at coarse chunk levels now descends
  to finer conditioning instead of trusting the unconditional prior
  (bits/char unchanged at 2.175). The showcase app also joined in with a
  'Compare two sprites' panel: pick (or randomize) two real sprites, see the
  per-role verdict decoded holographically next to the actual images, and
  below it the cross-modal loop live -- the mind is shown each IMAGE with no
  name, classifies it against the whole library, and states the colour in
  symbols (SEE -> SAY; the first run builds the relations memory over all 712
  sprites, ~1 min, then instant). Absorbing the library with FAMILY as the
  label closed the inception loop on real data too: every family split, and
  the journal named the splits by the genuine within-family modes -- facing
  and frame for the walk-cycle families, colour for the npc grab-bag. Building
  the panel also caught a propagation miss the suite could not see: the
  showcase's embedded unified panel had its own organize endpoint that never
  learned to show the journal story -- fixed, both consoles now narrate. Pinned in
  test_holographic_relations.py, test_holographic_unified.py, and
  test_holographic_brain.py.
- **Projection consolidation** (`HolographicMind.consolidate()`): the brain's
  thousands of 512-D prototypes are shadows of one low-rank object (the span of
  its sense-atom vocabulary -- measured: 99.9% of their energy in 22-24
  dimensions), so the memory is re-stored as coefficients in the SVD-discovered
  subspace: **21x smaller, ~5x faster decisions, at behavioural parity** (forage
  122 -> 120 stars, 16x16 maze 90% -> 95%). The measured hazard ships with its
  cure: a shadow hides new structure (a poison-free consolidation left the
  danger sense at 4% in-basis energy -- nearly invisible), so a residual guard
  tracks out-of-basis energy and EXPANDS the basis when the world grows
  structure the shadow cannot show (measured under a shift: rank 9 -> 13,
  danger 4% -> 100% visible). Compress when stable, grow at anomaly -- the
  flux-guard pattern's fourth appearance. Pinned in test_holographic_brain.py.
- **`THEORY.md`** gathers the load-bearing claims in one place, each tagged *cited theorem* vs
  *measured here* with a pointer to the test that backs it (a `test_theory_references.py` check keeps those
  pointers live). It is the project's notion of rigor — cited where the math is known, measured where it is
  not, and the kept negatives that bound every claim.
- **`NOTES_concepts.md`** records natural-process analogies (double diffusion /
  salt fingering, surface tension, gravity lensing, flocking, prism/spectral
  decomposition, demoscene) considered as possible improvements, and what honest
  measurement said about each. Most were parked or tested to clean negatives, with
  the recorded reasoning the real value. One was later **re-opened**: the
  salt-finger variance pre-screen, originally judged unavailable at 512-d, was
  re-measured on the *real encoded substrate* (not synthetic blobs) after the
  consolidation work gave us a low-rank lens — the signal is strong there (~7 sigma)
  and predicts split benefit (r ≈ 0.94), so it now ships as a conservative,
  default-off fast path on `auto_reorganize` that can only skip work, never change
  the measured choice. The flocking and prism negatives stand (local policy
  consensus loses to measured best-pick; wall-pocket dithering is not caused by
  state fusion). The lesson: an honest negative is real for the test it ran, but a
  capability unlocked later can make the same idea worth re-measuring on real data.

## License

Released under the MIT License — see the [LICENSE](LICENSE) file. In short: do what you like with it, keep the copyright notice, no warranty.

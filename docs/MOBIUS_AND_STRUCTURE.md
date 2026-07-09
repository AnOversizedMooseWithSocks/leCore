# Möbius topology and structure-first computation in holostuff

*A research note prompted by two questions: (1) would a Möbius strip define some things better than a
circle — things with multiple states or sign-flipping noise? and (2) the fruit-fly connectome that
drove a virtual fly with no training — what does it say about how we structure, and reorganize, what
the engine knows?*

Both questions turn out to be one question: **does the topology and structure of a representation
carry computation, beyond the contents it stores?** Everything below is measured on the real engine.
Negatives are kept.

---

## Part 1 — When a circle is the wrong shape: Möbius / non-orientable representations

### The observation behind the question
Circles, sign flips, and noise recur all through holostuff: binding *is* circular convolution; the
phasor work lives on the unit circle; involution flips a vector and applied twice returns it; binary
quantization (a kept negative) hinges on sign. The intuition — that a Möbius strip might fit some of
this better than a circle — is correct, and it has a precise statement.

### What the literature says
Neural population activity traces out a low-dimensional manifold whose **topology matches the variable
being represented**: a ring (circle) for head direction, a torus for grid cells, and — importantly —
a **Klein bottle / Möbius structure for orientation** in visual cortex (orientation + spatial phase;
Swindale 1996, Tanaka 1995; topological-data-analysis confirmations since). Continuous-attractor
network theory now builds Möbius-band and Klein-bottle attractors explicitly, using a *custom
non-orientable metric* because a circle's metric is simply wrong there (eLife MADE framework, 2025).

The dividing line is **orientability**. A *directed* angle (a heading, a phase from 0 to 2π) lives on
a circle. An *axial* quantity — where θ and θ+π are the **same state** — does not. An unoriented
line's direction, a nematic/liquid-crystal director, a crystal axis, a phase defined only mod π: for
all of these, θ and θ+π are identical, and the correct base space is the **projective line RP¹**, the
base of the Möbius double-cover. On a circle, θ and θ+π sit at **opposite** points.

### Measured on the substrate
holostuff binds by circular convolution, so the circle is its native shape. I tested whether that
shape actually hurts axial data, and whether the standard fix helps.

**Axial data (θ ≡ θ+π).** Encoding the angle directly (a plain circle) versus the **double-angle map
θ → 2θ** (which makes θ and θ+π coincide exactly — this map *is* the 2-to-1 cover of the circle onto
the Möbius base):

| | similarity(θ, θ+π) | axial recovery error |
|---|---|---|
| naive circle | **−0.22** (says they're far apart — wrong) | **0.470 rad** (≈ 27°) |
| Möbius / double-angle | **+1.00** (recognizes them as identical) | **0.002 rad** |

When each measurement is reported as θ *or* θ+π at random (the real situation with unoriented data),
the circle is effectively guessing; the Möbius encoding is essentially exact.

**Sign-flipping data, f(t+T) = −f(t).** A pattern that inverts every period and only returns after
two is *antiperiodic* — a Möbius double-cover in time. Measured: **100% of its energy lives in the odd
harmonics** (the antiperiodic / Möbius subspace); the periodic (circular) component is ~1e-14. The
ordinary circular basis literally cannot see a sign-flipping pattern. This is the concrete form of
"noise that flips sign" — it has its own subspace, and it is not the circle's.

### What shipped
`holographic_mobius.py`:
- `AxialEncoder` — double-angle phasor encoder; θ and θ+π map to the *same* hypervector (RP¹, not S¹).
- `antiperiodic_fraction`, `antiperiodic_split` — diagnose and extract the sign-flipping component a
  circular representation can't hold.

Six tests pin it (593 total now). The honest scope is a **kept negative**: use these *only* for
genuinely axial or sign-flipping data. On directed data the circle is correct, and the double-angle
encoder deliberately throws away the half-turn distinction — it would wrongly merge a heading with its
reverse. Topology must match the data; this is a tool for when it doesn't.

### A bonus: it names an old kept negative
Binary quantization (removed from `auto` because it distorted pairwise-similarity geometry) maps
values to ±1 — which is itself a **Z₂ / antipodal (Möbius-like) identification**. That is *exactly*
why it corrupted circular geometry (mean pairwise distance collapses 1.27 → 1.00 on circular points),
and — the flip side — exactly why it would be the *right* move for axial / sign-flip data. The old
negative was a topology mismatch. Now it has a name.

---

## Part 2 — Structure carries computation: the fruit-fly connectome parallel

### The experiment
The FlyWire consortium published the full adult *Drosophila* connectome (Dorkenwald et al., *Nature*
2024; ~140,000 neurons, ~50M synapses). Shiu et al. (*Nature* 2024) then wired a leaky-integrate-and-
fire model **straight from that connectome — no training, no reward, no reinforcement learning** — and
it reproduced sensorimotor behavior (sugar → feeding, touch → grooming) at ~95% accuracy. In 2026 an
embodied version drove a physics-simulated fly body (walking, grooming) from the wiring alone.

The honest, load-bearing claim across all the coverage (and the careful critiques): **structure
carries computation.** Biological wiring beat random graphs and standard neural-net controls. The
architecture itself holds the work — not a trained set of weights.

### Why this is holostuff's thesis
holostuff is deterministic and structure-first: no backprop, no gradient training. The bind/bundle
**organization** is the computation. The connectome result is an existence proof, at fruit-fly scale,
of the principle the engine is built on. I backed the parallel with proofs on real data (Brown corpus).

**Proof 1 — structure is the computation, with no training.** Build a classifier by *structuring*
real documents: encode each, bundle them per category into a prototype, classify held-out documents by
nearest prototype. No gradients, no training loop. Held-out accuracy:

> **0.76** correct vs **0.17** chance (6 classes).

The bundled prototypes *are* the classifier. That is the engine's analog of wiring driving behavior —
the structure does the work the moment it exists.

**Proof 2 — learning is structural reorganization (the honest version).** It is tempting to claim the
representation "collapses to low rank as it learns." It does not, naively: the **raw document cloud's
effective rank grows** with more samples (8.9 → 20 → 32 → 45 as docs/class go 2 → 20). Accumulation is
*not* learning. What *is* learning is the reorganization that follows: the **task** structure is
low-rank, and consolidation (SVD — the engine's existing consolidation faculty) finds it.

| consolidate prototypes to rank | held-out accuracy |
|---|---|
| 6 (full) | **0.76** (lossless) |
| 4 | 0.70 |
| 2 | 0.43 (broken) |

So the six categories live in a ~4–6 dimensional subspace even though the raw vectors span 45+. Learning
is the move that **separates the low-rank task structure (kept) from high-rank sample noise (discarded)**
— reorganizing the representation onto the subspace that carries the decision. That is precisely the
holostuff analog of a connectome being a specific, low-complexity wiring that holds the behavior: the
structure that matters is far smaller than the raw activity, and the work is in finding it, not in
piling up examples.

### The honest boundary (kept)
The fly result is not "a brain was uploaded." The connectome models have no plasticity, coarse sensing,
and some embodied variants trained controllers *on top of* the wiring. The clean, defensible claim — the
one holostuff shares — is narrower and solid: **a fixed structure, with no gradient training, can carry
real computation, and a good structure beats a random one.** holostuff measures both halves of that.

---

## The single thread

A representation is not just *what* it stores. Its **shape** decides what it can say (a circle cannot
represent an orientation or a sign flip; a Möbius strip can), and its **structure** — not a trained
weight matrix — can carry the computation outright (bundled prototypes classify; a connectome walks).
Learning, in this view, is choosing the right topology and reorganizing onto the right low-rank
structure — both of which holostuff now does, and measures.

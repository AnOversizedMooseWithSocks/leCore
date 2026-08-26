# The thesis: one data type, many costumes

*For the visitor who looked at 600+ modules — fluids, meshes, IK, audio, creatures,
renderers, compressors, retrieval, memory — and concluded "junk." This document is the
missing sentence, and then the receipts. Every claim below is a measured result in this
repository, most of them pinned by tests that fail if the claim rots.*

## The missing sentence

**leCore has one data type — the hypervector — and everything in this codebase, including
functionality itself, is either a point in that space or an operator on it.** Data is a
vector. A memory is a vector (a superposition of bound pairs). A role, a pointer, a
permutation, a program, a rig, a rendering rule — vectors and operators in the same space,
under one small algebra: bind, bundle, permute, and their group actions. When everything is
one type, modules do not add — they **multiply**. That multiplication is what the "junk" is.

## The junk test: five things that are secretly one thing

Take the modules that look most unrelated and watch them collapse:

1. **Cleanup is a denoiser.** The memory's cleanup step (snap a noisy readout to the
   codebook) and image restoration are the same operator. Measured: destroy half a memory
   trace — raw recall collapses to cosine 0.144 — and cleanup still identifies **24/24**
   stored items. That is a restoration prior doing memory's job, and it is why the image
   machinery is not "unrelated."
2. **IK, position-based dynamics, camera pose (PnP), and the resonator are all
   "iterate a projection."** One solver shape, four costumes. When we rigged *memory itself*
   with bones and joints (the semantic rig), the animation stack's CCD solved it in closed
   form — a planted pose recovered to **8e-17 radians** — because disjoint rotation planes
   commute. The 3-D animation code was never about 3-D.
3. **Mesh subdivision ran on hypervector sequences verbatim.** Loop subdivision — a mesh
   algorithm — worked unchanged on symbol sequences, because both are "refine a structure
   in the same space it lives in."
4. **A "mince" is a moving-block bootstrap.** The salamander-surgery battery needed to
   shuffle blocks of a memory trace; the statistics module already owned that operator
   (`block_shuffle`). One import, no new code: the neuroscience experiment and the surrogate
   test are one operator in two costumes.
5. **A renderer's sphere tracing became a retrieval certificate.** Per-block angular radii
   plus Cauchy–Schwarz turn the raymarcher's "march past empty space" into certified-exact
   nearest-neighbour search: **24× at 1.3% of blocks touched** where data has structure —
   and the same geometry honestly refuses to help on structureless data (pinned both ways).

None of those reuses were planned. They were *found*, because one data type makes them
findable.

## Functionality is a vector too — that is the part visitors miss

- **Roles are shift amounts; binding is a rigid transform.** Rotate a memory trace and it
  does not break — it recalls **rotated values at exactly baseline fidelity** (0.204 ==
  0.204, originals at 0.005). Behavior transforms coherently instead of dying. Pietsch's
  rotated salamanders that fed in reversed directions: that is this identity, measured.
- **Which symmetry group your memory lives in decides which surgeries it survives.** HRR
  traces are covariant under the cyclic group only; the GDN matrix memory under the full
  orthogonal group (exact, pinned as a regression trap). Two substrates are not redundancy —
  they are two different contracts with damage.
- **Programs are vectors.** The VM runs stored vector programs; recipes are rules with
  holes; a trained model's delta is a vector, so **model composition is addition** —
  measured as behavior transfer with a bruise (the capacity law), and **ablation is exact
  unlearning** (graft rejection: the host restored to its original behavior, exactly).
- **The GPU is a set of roles, and VSA ops fill them.** The machine map assigns measured
  units to silicon roles: a compiled gather rule answers in one dot product (**182,010×**
  when reused — and honestly 0.03× when not), kernel fusion composes N linear passes into
  one transfer (matched a 2,000-step loop to 6.7e-16), superposition packing is SIMT width
  with its 1/√K capacity law stated.

## Why compose at all: the economics

In a conventional codebase, N features cost N implementations and interact by glue code. Here
N operators in one space give ~N² compositions **for free**, and the repository's history is
that multiplication paying out: a projector built for rendering emptied render chains of
host-language links; a phase-vocoder idea became the memory's skeleton; the animation rig
became a **zero-capacity-cost memory edit verb** (pose: recall exactly preserved, inverse
exact); a drift model built for generation became transfer-with-rejection; a demoscene
precision trick became the benchmark result (**recall 1.000 @ 9.7 ms at 100k against FAISS's
27.1 ms exact scan** — the only 1.000 in the 1M table). Each looked "unnecessary" until the
day it was the answer.

## What keeps this from actually being junk

Sprawl without discipline *would* be junk. The discipline is mechanical and audited:

- **One front door.** ~1,970 faculties behind `find_capability`, with aliases written from
  the *user's* mouth. The governing rule is brutal: a capability the front door cannot
  surface **does not exist**, and CI audits enforce it (reachability / catalog gaps / example
  lint at 0/0/0).
- **Rule 0.** Nothing is built before interrogating the live system with five stranger
  phrasings. Most "new" work already exists in a different costume — see the junk test.
- **Kept negatives.** Refuted ideas ship loudly in docstrings and notes so they are never
  rebuilt. This codebase's failures are load-bearing documentation.
- **Measurement over narrative.** Every claim above carries a number, and the numbers carry
  tests. Deterministic to the bit (`PYTHONHASHSEED=0`, hashes not `hash()`), so every result
  is a receipt.

## The ten-minute tour for a skeptic

```python
import lecore
m = lecore.UnifiedMind(dim=256, seed=0)

m.find_capability("is this junk")            # this document answers, by design
m.find_capability("the snake eats its tail") # the Ouroboros memory loop
m.machine_spec_sheet()                       # the virtual GPU + L0-L4 tiers, measured
m.shufflebrain_battery(dim=1024, n_items=16) # brain surgery on a memory, all theorems live
m.semantic_rig(dim=96, hrr_dim=512)          # bones, joints, IK -- on the memory itself
m.advise_scale(n_pairs=500, dim=1024)        # the capacity laws, consulted before the wall
```

If after those six calls the fluid solver still looks unrelated to the database, run the
seventh: `m.find_capability("iterate a projection")` — and count the costumes.

## The one sentence to keep

**It is not 600 modules. It is one algebra wearing 600 costumes, a front door that knows all
of their names, and a ledger proving each one has paid rent.**

# holostuff — Chunking Transfer Sweep: Panel Review & Experiment Backlog

*An above/below sweep taking the recent **chunk-and-re-anchor** lesson and asking, across the whole
system, where else a capacity cliff could be beaten the same way. Every proposal is attributed to a SEAT
and that field's REAL published method — never a fabricated personal opinion. Two items were prototyped and
measured on the live substrate before being written down (negatives kept); the rest carry an honest prior.*

---

## The lesson being transferred

`plan_route` and `chunk_route` taught the same thing twice: a single fixed-width holographic structure
degrades past a length/load cliff (HRR crosstalk — physics, not a bug, like any bounded buffer), and the way
past it is **composition, not a bigger structure**. Two moves do the work:

- **Chunk** — split the work into cap-sized pieces, each individually clean.
- **Re-anchor / overlap** — coordinate the pieces at their boundaries (reset the accumulation, or overlap by
  one element) so nothing is lost across the seam.

Effective size then becomes unbounded at **linear cost**. The question for the panel: this is a very general
primitive — *where else in the engine is there a cliff that chunk-and-coordinate would beat, and is the
existing mechanism actually the right one?*

---

## Two results measured before writing this down

**P1 — chunking DOES unlock arbitrarily long VSA programs, but not the way you'd first reach for.**
A 60-instruction HoloMachine program at dim 1024 (single-program cap ~20–32 instructions) decodes to garbage
as one structure (cosine 0.08 to the intended result). The obvious fix — factor it into functions and `CALL`
them — **also fails** (cosine 0.06): `CALL` pulls each sub-program out of a *bundled library*, and bundling
several function-vectors into one library vector re-introduces the very cliff we're escaping (the docstring's
"busy disk" crosstalk). The fix that works is the true `chunk_route` analog: each chunk is its **own clean
program vector**, and the **host threads the accumulator** across them — 60 instructions then execute at
cosine **1.000**. So: more complex programs, yes — via host-threaded independent chunks, with the
`CALL`-the-library route kept on the record as a measured negative.

**X3 — a long chunked route should be random-access, and is.**
Indexing each chunk by a summary vector (a bundle of its tiles) and locating a query two-level (nearest chunk
summary → nearest tile within it) located **200/200** tiles exactly at **~6.9× fewer** comparisons than a flat
scan. So "where am I in this 1000-step route?" is sub-linear, not a replay-from-start — the acceleration
structure the route was missing.

---

## The panel, by group

Each item: the SEAT, the REAL method it rests on, the transfer, and an honest **prior**
(measured-win / measured-negative / strong / plausible / weak / research).

### Group P — More complex programs & procedures *(the headline question)*

- **P1. Host-threaded chunked programs (`run_chunked`).** *Seat: Plate — HRR capacity mathematics; the
  capacity cliff is his.* Split a long program into independent ≤K-instruction chunk-vectors and thread the
  accumulator across them in the host (re-anchor per chunk). **Prior: measured win** (60 instr exact). The
  `CALL`/library route is a **measured negative** and must be documented so nobody reaches for it.
- **P2. Host-side periodic programs (domain repetition).** *Seat: Quílez — SDF domain repetition (`mod`
  space): infinite structure from one kernel.* A program whose body repeats is stored once and re-run n times
  by the host (threading ACC), never materialising the full unrolled program. **Prior: plausible** (the
  host-threaded version sidesteps the `CALL` negative; `REPEAT`-of-`CALL` would inherit it).

### Group S — Long signals & sequences *(overlap-add)*

- **S1. Overlap-add signal/audio chunking.** *Seat: Puckette — the phase vocoder's weighted overlap-add
  (WOLA) IS chunk-and-coordinate for signals (Allen & Rabiner).* Process a long FHRR signal as overlapping
  windowed chunks, each a clean structure, reconstructed by overlap-add. This is the most exact rhyme in the
  sweep: `chunk_route`'s boundary overlap is literally overlap-add. **Prior: strong** (the canonical DSP
  method for exactly this).
  **STATUS: MEASURED NEGATIVE — do not build.** Prototyped on the FPE / `VectorFunctionEncoder` substrate
  (a continuous function f as f = sum_i y_i encode(x_i), queried by an inner product = kernel sum). A single
  bundle reconstructs the function with near-PERFECT shape fidelity (corr ~1.0 vs the noise-free kernel sum)
  at every domain length tested out to N=1500, while BOTH hard-cut chunking and proper Hann overlap-add are
  essentially useless (corr ~0) — chunking is strictly HARMFUL here. Why the rhyme fails: FPE encodings are
  shift-invariant powers of ONE base, so <encode(q), encode(x_i)> is the same kernel for every pair at a given
  distance — the finite-dim error is a DETERMINISTIC sidelobe, not √N random accumulation — and evaluating
  <query, bundle> distributes linearly and exactly over the superposition. So a long-domain function has NO
  capacity problem for chunking to solve; chunking only breaks the clean global kernel sum into
  boundary-incomplete pieces. **The sharpened principle (the real payoff of this negative): chunking helps
  DECODE-VIA-CLEANUP (routes, sequences, programs — recover a SPECIFIC item from a superposition, where
  crosstalk caps recovery), and does NOT help LINEAR-FUNCTIONAL EVALUATION (kernel density / function query,
  where the inner product is exact by linearity). This likely also weakens S2 (block denoising of a long
  signal) for the same reason, and should be checked before building it.**
- **S2. Overlapping-block denoising of a long signal.** *Seat: Milanfar — BM3D / NLM process the signal in
  overlapping blocks and aggregate by weighted averaging (Dabov 2007; Buades 2005).* Run the existing
  `denoise` per overlapping block so the denoiser is never capacity-bound on a long signal. **Prior: strong**
  (block processing is how these denoisers already scale).
- **S3. Chunked sequence-continuation memory.** *Seat: Plate / Olshausen — VSA positional sequence coding.*
  The positional `SequenceMemory` caps; store a long sequence as overlapping positional blocks (chunk_route
  for the continuation memory). **Prior: plausible** (thin; the same move on a different store).

### Group X — Spatial & structural chunking

- **X1. Tiled scene factorization.** *Seat: Olshausen — convolutional sparse coding + resonator networks for
  visual scenes (Kymn, Olshausen et al. 2024): scenes are parsed per spatial tile.* A many-object scene
  exceeds the resonator's object cap; factor by spatial tile and merge. **Prior: strong** (tiling is the
  **STATUS: SHIPPED.** `SceneCoder.factor_scene_tiled` / mind `decompose_scene_tiled`. Measured 30%→93% recovery at 15 objects, dim 1024, tiles of <=5.
  field's own answer to scene scale, and the resonator is already wired).
- **X2. Tiled splat fields.** *Seat: Drettakis — 3D Gaussian Splatting sorts and rasterises per screen tile
  (Kerbl 2023).* Store/query a large `splat_field` by tile so a big scene isn't one overstuffed bundle.
  **Prior: strong** (tiling is 3DGS's own scaling mechanism).
  **STATUS: SHIPPED (the content-addressable scene path).** `splat_bundle_tiled` / `recall_region_tiled`
  (mind: `splat_scene` / `splat_region`). The bundled splat scene's region readback is decode-via-cleanup, so
  a single bundle caps as the grid gets finer (measured ~100% @ grid 8 -> ~75% @ grid 32, dim 4096); routing
  each cell to a tile bundle of <=tile*tile bindings holds recall ~100% at any resolution (75% -> 100% @ grid
  32). This was prompted by the question "does chunking help text/image GENERATION": image region recall YES
  (decode-via-cleanup, this item); TEXT generation NO (the generators use bounded n-gram/lookback context and
  never accumulate a capping bundle — a measured/audited negative, see NOTES). The SplatArchive.region path was
  already exact (explicit list), so it never had this cap — the tiled bundle is the compact content-addressable
  complement. (Anisotropic / 3-D tiled splats remain the heavier Drettakis next step, unchanged.)
- **X3. BVH-over-chunks (sub-linear random access).** *Seat: Pharr — BVH / spatial acceleration: turn an
  O(n) scan into O(log n).* Index chunk-summaries for "jump to the region near here" on a long route/sequence.
  **Prior: measured win** (200/200 located, 6.9× fewer comparisons).
- **X4. Region-stitched network design.** *Seat: Adamatzky — Physarum multi-terminal transport networks
  (Tero Tokyo-rail model).* Solve a large transport network as regions and stitch at the boundaries, past the
  single-structure cap. **Prior: plausible** (extends the flow solver the same way).

### Group R — Iterative re-anchoring *(the lesson in time, not space)*

- **R1. Re-anchored long rollouts.** *Seat: Stam — long field rollouts; + the learned propagator (B4) and
  Koopman/DMD.* A learned propagator rolled out for many steps drifts; periodically re-project the state onto
  the `consolidation` manifold (re-anchor in time) to halt the drift. **Prior: plausible — worth measuring**
  (it is re-anchoring applied to a trajectory rather than a route; the order-book negative said prediction is
  weak, but trajectory *stability* is the durable property).
- **R2. Re-anchored resonator.** *Seat: Olshausen — resonator networks.* Snap factor estimates to the
  codebook periodically across a long factorization to escape limit cycles. **Prior: weak — likely a kept
  no-op** (the resonator already cleans every iteration; re-anchoring is probably redundant). Cheap to check;
  expect to record a negative.

### Group C — Storage, audit, and the reframe

- **C1. Chunk-level deduplication.** *Seat: Duda — ANS / content-addressable dedup; rate-distortion.* When
  chunks repeat (a route that revisits a corridor, a program with repeated motifs), store each unique chunk
  once and reference it. **Prior: plausible** (the win is exactly the repetition ratio; no repetition → no
  saving, kept honest).
- **C2. Determinism / tie-break audit of chunk boundaries.** *Seat: Macklin — bit-exact tie-breaks (the
  `bind_batch` lesson).* Verify `plan_route`/`chunk_route`'s overlap seams are bit-deterministic run-to-run
  and that no boundary decode sits on a knife-edge tie. **Prior: should be clean** (position-based,
  deterministic) — the audit itself is the value.
- **C3. Adaptive (rate-distortion) chunking.** *Seat: Eno (the reframe — the window is a choice) + Duda
  (water-filling bit allocation).* Size chunks by local complexity: coarse where the sequence is simple, fine
  where it's dense — spend "chunk budget" where the information is, instead of a uniform 14. **Prior:
  research** (the genuinely open, interesting one).

### Two deeper framings worth keeping (not build items yet)

- **Stoudenmire — a tensor-train (MPS) IS a chunked factorization of a long sequence.** The "right" math
  behind why chunking works: an MPS represents a long ordered object as a chain of small local tensors with
  bounded bond dimension — formally what chunk-and-overlap approximates. A reference frame for C3, not a build.
- **Togelius — chunk-based procedural generation (WFC).** Generating a long level in coordinated chunks is
  PCG's standard scaling move; the engine's procedural generator could grow arbitrarily long content the same
  way. A capability direction if the generator ever needs unbounded length.

---

## Prioritized build order (value over effort)

1. **P1 `run_chunked`** — measured win, answers the headline question, cheap; ship the negative with it. *(lead)* **[SHIPPED]**
2. **X3 BVH-over-chunks** — measured win, immediately useful ("where am I in the route"), cheap. **[SHIPPED]**
3. **S3 chunked sequence memory** — thin, reuses the chunk_route pattern on the continuation store. **[SHIPPED]**
4. **C2 determinism audit** — cheap hygiene on what just shipped; the project's discipline. **[SHIPPED]**
5. **S1 overlap-add signals** — ~~strong prior, the elegant rhyme~~ **[MEASURED NEGATIVE — chunking is harmful on the FPE function substrate; the query is linear-exact, no capacity problem to solve. See the S1 entry for the sharpened decode-vs-evaluate principle.]**
6. **X1 tiled scene factorization** **[SHIPPED — 30%→93% recovery at 15 objects/dim 1024; tile size <= cap plays the chunk's role.]** Note: unlike S1, the resonator is DECODE-VIA-CLEANUP (it iteratively recovers factors from a superposition), so it genuinely has the capacity problem chunking/tiling addresses — the S1 negative does NOT pre-empt it. Verify by measurement as always.
7. **R1 re-anchored rollout** **[MEASURED NEGATIVE — the propagator TRACKS its model class (free drift ~0); re-projecting onto the training-state manifold only discards valid forward signal. A linear rollout is an EVALUATION with no drift to fix. Joins S1.]**
8. **C1 chunk dedup** **[SHIPPED — 65% saving at the repetition ratio, 0% without repeats, exact rebuild.]**
9. **S2 overlapping-block denoise** — ~~strong prior~~ **likely weakened by the S1 principle** (block-aggregate denoising is closer to linear evaluation than to decode-via-cleanup); check the premise before building. Medium.
10. **X2 tiled splats** **[SHIPPED — splat-scene region recall]** **/ X4 region-stitched networks [SHIPPED — the multi-terminal Tero 'Tokyo rail' `network_design` faculty was already on disk]** — strong/plausible; medium.
11. **P2 periodic programs** — thin once P1 lands (low).
12. **R2 re-anchored resonator** — weak prior; cheap to check, likely a kept no-op.
13. **C3 adaptive chunking** — research; highest effort, do last.

---

## The honest bottom line

The sweep's strongest finding is also its most useful: the recent lesson is not a one-off route trick but a
**general capacity primitive**, and the system has cliffs in at least five other places that chunk-and-
coordinate should beat. Two are already measured — long programs (host-threaded, with the `CALL`-library
route a kept negative) and sub-linear random access into a chunked route. The most elegant transfer is that
`chunk_route`'s boundary overlap is literally the phase vocoder's overlap-add, so a long-signal modality
falls out of the same idea. The recurring theme holds: one set of primitives, measured honestly and kept
minimal, turns out to be load-bearing across fields — and beating a hard limit with composition rather than
pretending it isn't there is the move the whole project keeps re-learning.

**Resolution (all 13 items closed).** Shipped: P1, X3, S3, C2, X2, and X4 (the multi-terminal Tero "Tokyo
rail" `network_design`, found already on disk), plus X1 (tiled scene factorization, 30%→93% at 15 objects)
and C1 (chunk dedup, saving the repetition ratio). Kept negatives: S1 (overlap-add on the linear FPE query)
and R1 (re-anchoring a propagator rollout — the operator tracks its model class, so there is no drift to fix).
The decode-vs-evaluate line predicted every outcome: chunking/tiling/re-anchoring helps wherever an item must
be DECODED from a superposition, and is inert or harmful wherever the query is a linear EVALUATION with no
capacity cliff. The remaining originally-listed tail is settled by that same test without a build: **S2**
overlapping-block denoise is block-aggregate (closer to evaluation; the S1 negative carries over), **R2**
re-anchored resonator is a no-op (the resonator already snaps to the codebook every iteration), **P2** periodic
programs is a thin wrapper over the shipped `run_chunked`, and **C3** adaptive (rate-distortion) chunking is the
one genuine open research item — spend the chunk budget where the information is, rather than uniformly.

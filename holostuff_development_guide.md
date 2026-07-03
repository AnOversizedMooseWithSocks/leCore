# holostuff — Development Guide

*Layout, capabilities, and the design & development practices of the project.
Grounded in an audit of the live codebase (this snapshot: 239 engine modules,
260 test files, 2,170 collected tests, `UnifiedMind` at 510 public faculties,
`tour.py` at ~5,900 lines). Written for anyone — human or AI session — picking
up the project cold.*

---

## 1. What this is

holostuff is a from-scratch, **NumPy-only** engine for Vector Symbolic
Architectures (VSA) and Holographic Reduced Representations (HRR), grown into
the authoritative back-end core for a Blender-class 3D application (three.js /
WebGPU front end planned; the boundary is binary glTF via
`holographic_gltf.py`).

The thesis, stated once and proven module by module: **VSA is geometry — as
above, so below.** Everything — a number, a word, a record, an image, a mesh,
a fluid field, a creature's situation, a program — is a point (or a field of
points) in one high-dimensional space, and a *small* set of reversible
algebraic operations composes and decomposes them:

| Operation | Math | What it means |
|---|---|---|
| `bind(a, b)` | circular convolution via the real FFT | attach / associate / rigidly transform |
| `unbind(c, a)` | correlation (bind with the involution) | detach / query / inverse transform |
| `bundle(a, b, …)` | normalized superposition (sum) | a set / a memory / a scene |
| `permute(a, k)` | cyclic shift | order / direction / protection |
| `cleanup(x, vocab)` | nearest codebook atom by cosine | recognize / denoise / decide |

Out of those five you get associative memory, structured records, symbolic
reasoning, a reinforcement-learning creature, a damage-tolerant image archive,
a physically-based renderer, a fluid solver, a mesh modeling kernel, Gaussian
splatting, calibrated statistical detection, and a stored-program machine —
all on the same substrate, all deterministic, all measured.

### The non-negotiable constraints

These are constitutional. Violating them is a rejected change, no matter how
good the numbers look (the C-backend PR was declined for exactly this:
measured regression at operating dims, non-bit-exact, banned dependencies).

1. **NumPy / Flask / stdlib / hashlib only** in the core. No PyTorch, scipy,
   sklearn, PIL-in-core, autodiff, or learned weights. `numba` and `sympy` are
   **opt-in accelerators** only (`holographic_jit.py`, `holographic_codegen.py`)
   — the engine must run and pass every test without them.
2. **Deterministic outputs.** `PYTHONHASHSEED=0`; `hashlib` for content hashes,
   never Python's `hash()`; seeded `default_rng` everywhere randomness is
   needed. The tie-break/sign rule is stated *once* and made executable in
   `holographic_determinism.py` (ISA-1).
3. **Backward-compatible, additive changes only.** New capability lands as a
   new module or a default-off flag. Existing decisions must not flip — the
   `bind_batch` lesson: a change bit-identical to 1e-12 still flipped a
   creature's maze trajectory, so it was kept out of the tie-sensitive path.
4. **Readable, WHY-commented code.** Comments explain *why* a line exists —
   the trade-off, the negative it avoids, the paper it implements — not what
   the syntax does.
5. **Rigorous honest measurement.** Every claim has a baseline, a variance
   estimate, and its negatives kept loud and on record (see §6).

---

## 2. The substrate, bottom to top

The stack reads like an operating system, and that's deliberate — the project
calls it "the assembly tower":

```
 Layer 6  UnifiedMind (holographic_unified.py)      — 510 faculties, one mind
 Layer 5  structure language (holographic_lang.py)  — declarative descriptions
 Layer 4  recipes & templates (recipe/template/ops) — generative IR with holes
 Layer 3  the VSA ISA (ISA.md, machine, reference)  — stored-program machine,
          calling convention, registers, macro hygiene, reversible model
 Layer 2  binding modes (fhrr, clifford, fpe, sbc,  — governed extensions,
          tensor, mobius)                             each earns a regime win
 Layer 1  the frozen kernel (holographic_ai.py,     — bind/bundle/permute/
          holographic_core.py, determinism, fft)      cleanup + persistence
```

Key documents, in reading order for a new contributor:

* `README.md` — the whole story with tour results inline.
* `THEORY.md` — the mathematical grounding and citations.
* `ISA.md` (+ `ISA_EXTENSIONS.md`, `ISA_REVERSIBLE.md`) — the instruction-set
  **contract**: the one determinism rule, the base instructions, the
  architecture/microarchitecture boundary, the ABI/calling convention, and
  what the contract deliberately does *not* freeze. Enforced by
  `holographic_reference.py` (definitional reference implementations +
  conformance harness) and `test_isa_conformance.py`.
* `writing_vsa_programs.md` — how to write programs *in* the substrate:
  LOAD/BIND/BUNDLE/PERMUTE, CALL and named sub-programs, IFMATCH triggers,
  REPEAT loops, STORE/RECALL registers, PUSH/POP stack, and APPLY handlers
  that call out to the mind's faculties.
* `NOTES_concepts.md` — the append-only research log: every faculty's test
  delta and kept negatives, newest at the bottom. **Read the tail before
  starting work** — it's the most recent ground truth.
* `ABLATIONS.md` — where VSA is actually load-bearing, with FDR-controlled
  ablation results.
* The backlog family (`WIRING_BACKLOG.md`, `holostuff_*_backlog.md`,
  `PANEL_*.md`) — panel-ranked future work.

### The kernel (`holographic_ai.py`, `holographic_core.py`)

`holographic_ai.py` holds bind/bundle/cleanup, key→value memory, the learner,
reflexes, drift. `holographic_core.py` is the **frozen core**: versioned
persistence (`to_state`/`from_state`, `save`/`load`), scalar quantization
(`int8`/`float32`/`auto`), and the rate-distortion save path (`quant='rd'`:
KLT via consolidation → water-filling bit allocation → bit-exact rANS). At
360 lines it is small on purpose — everything builds *on* it, nothing reaches
*into* it.

`holographic_fft.py` and `holographic_backend.py` (CuPy) and
`holographic_jit.py` (numba) are the optional fast paths. Remember the hard
lesson: **`@njit` with `cache=True` crashes under dynamically-loaded modules**
("No module named dynamic") — omit `cache=True`.

---

## 3. Project layout — flat on purpose

Everything lives in one directory and imports cleanly; there is no package
nesting to fight. The 239 engine modules fall into families. This taxonomy is
the map — the per-module one-liner is its own docstring, which is always the
authoritative description.

### 3.1 Kernel, ISA & program layer
`holographic_ai`, `holographic_core`, `holographic_determinism`,
`holographic_reference` (conformance harness), `holographic_fft`,
`holographic_jit`, `holographic_backend` (GPU opt-in),
`holographic_machine` (the stored-program VM), `holographic_lang`
(structure-description language), `holographic_recipe` /
`holographic_recipeops` (generative recipe store + validator/edit operators),
`holographic_template` (recipes with named holes), `holographic_compile`
(content-addressed compile cache), `holographic_codegen` (opt-in SymPy
design-time gradients), `holographic_dispatch` (per-element operator
dispatch), `holographic_route` (representation routing policy),
`holographic_reversible` (error-correction model + auto-cleanup scheduler).

### 3.2 Encoders & binding modes
`holographic_encoders` (scalar/FPE/text/records), `holographic_mind` (the one
`UniversalEncoder` with modality self-discovery), `holographic_fpe` /
`holographic_fpefield` (fractional power encoding, N-D), `holographic_fhrr`
(complex phasors), `holographic_sbc` (sparse block codes + scaled resonator),
`holographic_clifford` (geometric algebra bind mode), `holographic_tensor`
(tensor-product bind + tensor-train truncation), `holographic_mobius`
(non-orientable topologies), `holographic_steering` (anisotropic kernels),
`holographic_harmonic`, `holographic_octnormal` (manifold-correct normal
quantization), `holographic_sphere` (Riemannian mean/transport on the
hypersphere).

### 3.3 Memory, recall & storage
`holographic_archive` / `holographic_image` (WHT-plate content-addressable
image memory, damage-tolerant), `holographic_splat_archive`,
`holographic_pack` / `pack_sprites` / `image_vault` (delta packers, vault),
`holographic_tree` (RP-tree + `HoloForest` sublinear recall),
`holographic_pivot` (recursive pivot-tree index), `holographic_organizer`
(self-organizing memory, shadow-and-swap reorg, coherence-gated maintenance),
`holographic_hopfield` (dense associative cleanup + generation-by-denoising),
`holographic_energy` (learned attractors), `holographic_resolution`
(coarse-to-fine cleanup), `holographic_multires` (mipmap pyramids),
`holographic_graph_memory` (a recorded negative), `holographic_occlusion`
(alpha-composited bundle readout), sparse-recovery trio `holographic_cosamp`
/ `holographic_iht` / `holographic_mis`, `holographic_superposed` (compute in
superposition — the WIDTH faculty), `holographic_history` (versioned rollback
store), `holographic_verify` (O(log n) tamper evidence),
`holographic_fountain` (rateless erasure codes), `holographic_rns` (exact
integer arithmetic over phasors), `holographic_ratedistortion` (B5),
`holographic_deltachain` (chunked delta chain with integrity proof),
`holographic_uri` (content addressing).

### 3.4 Honesty & measurement (the instrument panel)
`holographic_honesty` (`RecallNull` calibrated false-alarm probability, SPRT
streaming detection, `bh_fdr`), `holographic_measure` (the variance harness —
every headline number gets mean, spread, CI), `holographic_ablate` (the
FDR-controlled ablation table), `holographic_structure` /
`holographic_signal_structure` (proof-of-structure verifiers),
`holographic_protocol` (honesty as a structural property of a program),
`holographic_knowledge` (the queryable findings registry),
`holographic_flatness` (spectral flatness predicts binding distortion),
`benchmark_holographic` / `stress_holographic` (quantitative + adversarial
benchmarks).

### 3.5 Structure, symbols & language
`holographic_typed` (B7 keystone: ONE typed structure), `holographic_peel`
(denoised per-peel decode), `holographic_resonator` / `holographic_compose`
(factorize ↔ compose), `holographic_planshape` (schema-guided typed plans),
`holographic_schema` (compressive hierarchy discovery), `holographic_sequence`
/ `holographic_directed` (order as a first-class property),
`holographic_relations`, `holographic_symbolic` (MDL symbolic regression,
multiplicative/log mode), `holographic_manifold` (topology-aware decompose),
`holographic_grammar` (L-systems as recipes), `holographic_text` /
`holographic_lexicon` / `holographic_encyclopedia` / `holographic_intent` /
`holographic_answer` / `holographic_respond` / `holographic_deliberate` /
`holographic_generation` / `holographic_meaning_predict` (the language stack),
`holographic_segment` (unsupervised unit discovery), `holographic_codec` /
`holographic_compress` (lossless sequence codecs).

### 3.6 Dynamics, physics & simulation
`holographic_dynamics` (Propagator: prediction as one bind — **note:
`rollout()` returns the whole `(k, dim)` trajectory; compare against
`rollout(...)[-1]`**), `holographic_chaos` (nonlinear flows the linear
propagator can't), `holographic_fluid` (Stam Stable Fluids) +
`holographic_fields` (grids/particles), `holographic_softbody` (PBD),
`holographic_collide` (SDF environment collision as projection),
`holographic_emitter` (surface emission), `holographic_physics`,
`holographic_equilibrium` (equilibrium propagation), `holographic_automaton`
(reaction-diffusion CA), `holographic_diffusion` / `holographic_diffuse`
(denoise-as-diffusion), `holographic_energy`, `holographic_eulerops`.

### 3.7 Geometry, modeling & rendering (the DCC stack)
The **explicit mesh kernel** `holographic_mesh` (half-edge, Euler/genus,
OBJ/glTF) plus its operator suite: `meshverbs`/`meshverbs2` (extrude, inset,
dissolve, bevel, bridge, loop-cut), `meshsubdiv`, `meshsmooth`, `meshqem`
(decimation), `meshcurvature`, `meshgeodesic`, `meshuv` + `meshseam`
(unwrap/atlas), `meshik` + `meshskin` + `blendpose` + `deform` (rig/skin/
pose), `meshtools` (mirror/weld), `meshpoly`, `meshbridge` (mesh ↔ SDF ↔
splat, three views of one surface), `eulerops`, `gltf`, `svg`.

The **implicit/field side**: `sdf` (analytic SDF expression trees + CSG),
`sdf_render` / `sdfbake` / `raymarch` / `sculpt` / `sparsefield` /
`fpefield` (field-first surface-as-one-hypervector), `noise` (band-limited
fields, fBm), `terrain`, `procgen` / `generate` / `grammar` (procedural
generation), `tiling` (domain repetition as bind+bundle), `displace`,
`regionfield`, `octree` (capacity-adaptive 3D tiling).

The **splat side** (7 modules): `splat` (scene = superposition of Gaussians),
`splatdensify`, `splatprune`, `splatexport` (.ply/JSON), `splat_archive`,
`jittersplat` (a kept negative), `splatsharpen` (a kept negative).

The **renderer**: `render` (camera/lights/rasterizer/volume ray-marcher),
`brdf` (Cook-Torrance/GGX), `pathtrace` (Monte-Carlo GI), `globalillum`
(GI + caustics), `prt` (precomputed radiance transfer), `radiance`
(render = query on a radiance field), `postfx` (composable post chain),
`lens`, `volint` (closed-form volumetric integrals), `mis` (Veach balance
heuristic), `traverse` (Russian roulette), `raydiff` / `raycoherence` /
`rayindex` (ray beams, coherence, bidirectional ray↔object index),
`adaptive` (a render call that picks its own methods), `lod`, `anim`
(timeline + tiered frame cache), `scene` / `scenegraph` / `scenedelta` /
`semantic` (compositional scenes, algebra, deltas, describe/generate),
`material` / `materialio`, `backwardwarp`, `video` (temporal compression),
`phasemorph`, `cache` (irradiance-gradient decode), `accumulate` (firefly
clamping), `sampling` / `lowdiscrepancy` (blue-noise / quasi-random).

### 3.8 Navigation, search & planning
`holographic_slime` (Physarum maze solver, elitist-ant),
`holographic_flow` (deterministic Tero flow — ~100× faster than the ant),
`holographic_assembly` (fragment assembly as flow search),
`holographic_navigator` (the creature on the data tree), `holographic_plan`
(corridor planning: bake, run cheap, re-anchor), `holographic_dirtyfield`
(cost fields that update only where changed), `holographic_lens`
(gradient-field navigation with caustics), `holographic_reanchor`
(load-bearing for deep traversal — audited), `holographic_lookahead` and
`holographic_ldexplore` (both KEPT NEGATIVES, on record).

### 3.9 Learning & agents
`holographic_creature` / `holographic_creature_mind` (the RL forager and the
reference demo of building a specialized mind ON UnifiedMind),
`holographic_agent` (affect, action library, pain-driven behavior),
`holographic_valuehead` (policy = hypervectors, learn = bundling, decide = a
dot), `holographic_drives` (homeostatic scheduling of faculties),
`holographic_classifier` (gradient-free classification), `holographic_kan`
(deterministic Kolmogorov-Arnold readout), `holographic_moe` (learned
holographic gate), `holographic_forward` (Forward-Forward: depth from local
objectives), `holographic_recurrent` / `holographic_reservoir` (gradient-free
sequence learning), `holographic_predictive`, `holographic_dream`
(consolidation + Nyström), `holographic_partition` (many minds, one shared
frozen base), `holographic_emergence`, `holographic_sync` (Kuramoto
grouping), `holographic_voidsynth` (program synthesis when no tool chain
reaches a goal), `holographic_orchestrator` (tool planner with
circuit-breakers).

### 3.10 Scale, distribution & analysis
`holographic_distribute` (commutative-monoid distribution: partition, LPT
load balance, sum/min/max/bundle reducers, read-only shared cache — see the
NOTES tail for the measured bit-exactness results and the three loud
negatives), `holographic_nystrom` (landmark spectral embedding past the
O(N³) wall), `holographic_spectral` (Laplacian/Hodge kernel),
`holographic_graphsignal` (Taubin filtering over k-NN graphs),
`holographic_simgraph` (cotangent Laplacian turned inward),
`holographic_chart` (nonlinear manifold charts), `holographic_topology`
(persistent homology), `holographic_transport` (Sinkhorn Wasserstein),
`holographic_kde`, `holographic_cosmic` (point-cloud structure
classification), `holographic_market` (the honest time-series study),
`holographic_dedoppler` (drifting narrowband detection),
`holographic_denoise` / `holographic_downscale` / `holographic_sharpen`
(the denoising family), `holographic_bandwidth`, `holographic_fractal`.

### 3.11 App layer
`app.py` (Flask UI: tour, test runner, image recall), `unified_app.py`
(console for the one model), `tour.py` (the command-line tour of *every*
subsystem — the living integration demo), `run.bat`.

`path_d/` holds Path-D experiments, docs, plots and figures; `data/`,
`figures/`, `archive/`, `features/`, `benchmarks/` hold assets and results.

---

## 4. UnifiedMind — one mind, not a drawer of experiments

`holographic_unified.py` (~7,800 lines) is the top level. The audit of this
snapshot counts **510 public faculties** on `UnifiedMind`. The architecture is
one encoder (`UniversalEncoder`), one associative memory (the self-organizing,
self-maintaining store), one decision brain, one kernel — and every study that
grew up beside the mind is wired in as a *faculty* of it, default-off and
backward-compatible.

The organizing frame is two halves of one loop, inverse operations on the
same substrate:

* **Forward:** COMPOSE · RECALL · PREDICT · GENERATE — `compose_scene`,
  `typed_structure`, `realize`, `recall`, `learn_sequence`, `generate`,
  `generate_vector`, `learn_dynamics`, procedural generation…
* **Inverse:** DECOMPOSE · DENOISE · SEARCH · RECOGNIZE-HONESTLY —
  `decompose_signal`, `decompose_structure`, `denoise`, `solve_maze`,
  `assemble`, `recognize` (calibrated p), `classify(abstain=α)`,
  `stream_recognize` (SPRT), `recognize_batch` (FDR), `scan`,
  `decide_confidence`…

Honesty is wired into recognition *itself*: every readout path can return a
calibrated false-alarm probability, abstain when the match is noise-level,
decide streams sample-optimally (Wald), and control the look-elsewhere effect
across batches (Benjamini-Hochberg). `calibration_report` verifies empirically
that p≤α actually holds the false-alarm rate at α.

Rules for adding a faculty: it must **earn its method** (no forced wiring); it
delegates to a module, never reimplements; it defaults off or to the old
behavior; and it lands with a **cross-faculty pipeline integration test**, not
just an import check. The hardest integration lesson is on record: naive
cross-module chaining *regressed* (a denoiser fed a recall output dropped
cosine 0.13 → −0.06) because **a shared kernel is not a shared manifold**.
Pipelines must be proven end to end in `test_integration.py`.

---

## 5. The four design principles (the project's working philosophy)

These are the practices that shape every session. They are not slogans — each
one has produced shipped modules, and each has a failure mode the discipline
in §6 guards against.

### 5.1 Everything is a wave/vector — so generalize on contact

In holographic space, a number, an image, a mesh, a force field, and a
program are all the same kind of object: a hypervector (or a field of them),
manipulated by the same five operations. The practical consequence: **when an
improvement lands in one area, assume it applies elsewhere until measured
otherwise.** When something works in more than one place, extract it into a
generalized module and make the call sites thin.

The codebase is full of proof this pays:

* A rigid shift is a single binding → motion-compensated video compression
  (`holographic_video`) *and* the propagator's O(1) advancement
  (`holographic_dynamics`) *and* scene transforms (`scenegraph`).
* Cleanup and consolidation turned out to already *be* denoisers — the
  Milanfar reframe. Naming them correctly unified restoration (PnP/RED),
  splatting-as-denoising, self-similar NLM, and generation-from-noise into
  one operation seen five ways (`holographic_denoise`, `_splat`,
  `_hopfield`, `_diffuse`).
* "Iterate a projection" is one engine wearing four costumes: the resonator's
  alternating projection, the PnP restoration loop, PBD constraint solving,
  and FABRIK IK — `holographic_meshik` literally runs through the shipped
  iterate-a-projection machinery, and `project_onto_constraints` is the one
  entry point.
* The N-D field pattern (`holographic_ndfield`) exists precisely because the
  same field abstraction kept being rebuilt per-domain.
* Loop subdivision built for meshes (`meshsubdiv`) was turned *inward* onto
  hypervector sequences (`subdivcurve`); the cotangent Laplacian likewise
  (`simgraph`). Outward tools become inward tools and vice versa.

**Practice:** before writing new machinery, ask "which existing module is
this, in a different costume?" After shipping, ask "which other three modules
should now delegate to this?"

### 5.2 As above, so below — think fractally, like a demoscene developer

The same structure recurs at every scale, and the discipline is to *check for
it deliberately* rather than notice it by luck. A scene is a bundle of
objects; an object is a bundle of parts; a part is a bundle of role-bound
attributes — and the resonator that factors one level factors them all
(`compose_nested`, `nested_scene_structure`, `decompose_nested`,
`tile_field_recursive`, `menger_fractal`, `fractal_scene`). Structure
recursion costs no new machinery because bind+bundle already compose.

This is the demoscene aesthetic made constitutional: maximal richness from a
minimal deterministic kernel — Quílez's seat on the panel, and the reason the
tour ends with a mind self-assembling from a bare pile of examples.

**Practice — the up/down check, run at every close-out:**

1. *Down:* does this new operation still work when applied to the components
   of its own input? (If a denoiser helps a scene, does it help a peel?)
2. *Up:* does it work when the input is itself a component of something
   larger? (If it helps a splat, does it help a bundle of splat scenes?)
3. *Sideways:* which of the §5.1 costumes does it wear — field, structure,
   sequence, program?

Missing one of these is a missed faculty. Several shipped modules exist only
because the check was run (e.g. the mesh Taubin smoother becoming the
hypervector codebook denoiser).

### 5.3 There are no limitations, just bad approaches

When something looks impossible or too slow, the assumption is that the
*approach* is wrong, not that the wall is real. The project's standing toolkit
for turning a wall into an approach, with the shipped evidence for each:

* **The cache hierarchy (L1/2/3/4 + RAM) for speed.** Design data layouts and
  access patterns so the hot loop stays cache-resident: contiguous NumPy
  arrays over object graphs, batch/vectorized kernels, bake-once-sample-O(1)
  (`sdfbake`), content-addressed compile caches (`compile`), gradient caches
  (`cache`), adaptive anchor placement (`adaptive_cache`), tiered frame
  caches (`anim`). When Python-loop-bound is the honest verdict (the mesh
  kernel's kept negative), it is *written down*, and the numba/CuPy opt-in
  paths exist for exactly those loops.
* **RAID-style width for scale.** Most holostuff computations are
  **commutative monoids** — force/radiance/density fields ADD, SDF unions
  MIN, coverage MAXes, memories BUNDLE — so they partition into buckets that
  run independently and reassemble by the monoid's own operator, with no
  stitch pass and no seams. `holographic_distribute` ships this: `partition`,
  `adaptive_partition` (LPT — isolate the heavy item so the slowest bucket
  doesn't bound wall time), the four reducers, and the read-only shared-cache
  pattern (bake once on the main node, hand the immutable object to every
  bucket). Measured: SDF-union and tiled-render reassembly are BIT-EXACT and
  bucket-order-invariant; SUM is order-independent only to ~1e-12 (float
  addition isn't associative — the render-farm caveat); nonlinear steps with
  feedback do NOT superpose and must scatter disjointly. Width also lives
  *inside* one vector: `holographic_superposed` computes in superposition.
* **Deterministic patterns/structures for depth.** Because the engine is
  deterministic and seed-reproducible, depth can be *computed instead of
  stored*: recipes and templates regenerate structures from seeds; the
  spectral iterate (`iterate`) diagonalizes a bind operator once and
  evaluates any level or the limit in closed form; procedural
  generation replaces asset storage; delta chains and O(change) patches
  (`deltachain`, scene deltas) replace full-state copies; the reversible ISA
  model recovers rather than re-runs.
* **Just add more dimensions.** When a representation can't hold a property,
  can't separate two things, or can't be solved by the machinery at hand, the
  move is often to *widen the space* rather than fight in the current one.
  Dimensions are cheap in this engine — capacity, orthogonality, and
  linear-separability all improve with D — and extra dimensions can carry
  extra **properties and accumulators that track simultaneously**, each
  role-protected so they don't interfere. The shipped evidence spans every
  flavor of the move:
    * *Extra properties, in parallel:* a record is properties bound to roles
      and bundled — one vector tracks all of them at once, and adding a
      property is adding a role, never a schema migration (`typed`,
      `encoders`, `material`). Per-vertex data rides as extra field channels
      beside geometry (`attributes`); occlusion readout adds an alpha
      accumulator alongside color so opacity and radiance composite in one
      pass (`occlusion`); phasors carry magnitude *and* phase as two
      simultaneous tracks per frequency (`fhrr`).
    * *Extra accumulators:* the width faculty runs many computations in
      superposition inside one vector (`superposed`); distributed memory
      buckets are independent accumulators reassembled by bundle
      (`distribute`); the agent tracks reward *and* pain as separate affect
      channels rather than one collapsed scalar (`agent`).
    * *Lifting to make the impossible linear:* the propagator's kept negative
      — no fixed linear map can turn a chaotic flow — was answered not by a
      cleverer linear map but by a **nonlinear feature expansion into a
      higher-dimensional space** where a plain linear readout suffices: the
      reservoir's echo-state lift beats the best linear baseline (full DMD)
      ~40× on Lorenz one-step prediction (`reservoir`, `chaos`). Same move
      elsewhere: FPE encodes continuous N-D coordinates natively (`fpe`,
      `ndfield`), SBC partitions D into blocks to raise factorization
      capacity past the dense cliff (`sbc`), the log transform lifts
      multiplicative structure into a space where it's additive (`symbolic`),
      and Clifford multivectors add graded dimensions to make rotations exact
      (`clifford`).
  The honest boundary, kept loud: more dimensions cost memory and bandwidth
  linearly, capacity gains are sublinear past the operating point (Plate's
  capacity math; the cliff is shown, not hidden), and a lift that helps must
  still beat the strongest baseline in the *original* space — the reservoir
  win was measured against full DMD, not persistence.
* **Bricks/tiles + an orchestration layer for chunking.** When one wave can't
  hold it, tile the domain so each tile's wave stays within capacity, and
  orchestrate: the capacity-adaptive octree (`octree`), tiled scene
  factorization past the resonator cap (`decompose_scene_tiled`), the
  narrow-band sparse field (`sparsefield`), multires pyramids, `chunk_route`,
  the brick ray index, and the coherence-gated maintenance that reorganizes
  only when the store actually goes incoherent.

**Practice:** when blocked, walk the five levers in order — *can the hot data
fit closer to the ALU? can the work split into a monoid? can determinism
replace storage or recomputation? can more dimensions carry the property, run
the accumulators in parallel, or lift the problem to where it's linear? can
the domain tile under an orchestrator?* — before concluding anything is
impossible.

### 5.4 We do things others have not done — do not assume; build, test, verify

"Impossible" usually means "not yet figured out." The project's history is a
list of things the standard answer says need a learned model or a heavyweight
framework, done here deterministically on NumPy: calibrated statistical
detection, a self-explaining RL agent, damage-tolerant holographic image
storage, a physically-based path tracer, Gaussian splatting, a stored-program
machine — all from five primitives.

But the same principle cuts the other way, and this is the part that keeps it
honest: **do not assume it works, either.** Confidence in the destination,
zero confidence in any unverified step. Every idea — especially the exciting
cross-cutting ones — gets built small, measured against a *proper* baseline,
and kept whether it wins or loses. The probe sweep (`probesweep`) exists to
measure six transfers the panel pre-judged as likely no-ops — and kept all
six negatives. `misgen`, `ldexplore`, `lookahead`, `jittersplat`,
`splatsharpen` are all KEPT NEGATIVES, named as such in their own docstrings,
so no future session re-invents them.

Baseline discipline matters as much as measuring at all: the zeros-Dijkstra
baseline shared a tie-break with the router and hid the win; persistence is a
strawman baseline for market returns (the mean predictor is the honest one);
routing's "win" turned out to be masking an encoding bug. A win without a
proper baseline is not a result.

---

## 6. Measurement & honesty discipline (how claims are made)

* **The variance harness** (`holographic_measure`): every headline number is a
  mean, a spread, and a bootstrap CI across seeds — never a single lucky run.
* **The honesty harness** (`holographic_honesty`): score a candidate, then
  re-run the identical machinery on a shuffled null and demand it collapse.
  `RecallNull` gives calibrated false-alarm probabilities; SPRT decides
  streams sample-optimally; `bh_fdr` controls false discovery across scans —
  including across the engine's *own ablation table*.
* **Ablations** (`holographic_ablate`, `ABLATIONS.md`): "is this component
  actually load-bearing?" answered with paired permutation p-values under FDR.
* **Kept negatives are first-class.** They live in module docstrings, in
  `NOTES_concepts.md`, in the README, and in tests that pin them. A vision
  document that hides its negatives betrays the project. Current loud ones
  include: navigation routes are grid/L1-constrained so the meaningful metric
  is field cost crossed, not cell count; the SOL manifold is near-vertical so
  routing gains are modest (~8%); POCS on non-convex sets can stall from
  degenerate starts; manifold-projection denoising hurts at low noise;
  bare-codebook generation is a degenerate sampler; the mesh kernel is
  Python-loop bound.
* **Probe first, always.** Before claiming a feature is missing or new, audit
  the live code. This has repeatedly redirected work — `project_onto_constraints`
  already shipped when it was about to be "proposed"; energy cleanup was
  already inside the SBC resonator; PnP/RED was already wired; the panel's
  Tier-1 list got *shorter* after the re-audit, which is the method working.
* **Believe the measurement over the narrative.** When a result looks great,
  hunt for the bug or the strawman before celebrating (the routing/encoding
  story). When it looks bad, write it down and keep it loud.

---

## 7. The advisory panel framework

The project maintains a ~19-seat named panel of real researchers (Plate,
Olshausen, Milanfar, Drettakis, Pharr, Macklin, Stam, Quílez, Togelius,
Tarter, Siemion, Baker, Cranmer, Adamatzky, Ozcan, Puckette, Duda,
Stoudenmire, Eno — roster and hooks in the project docs). It is an
**attribution and grounding framework**: every proposal is attributed to a
SEAT and that field's REAL published methods, cited by paper — never a
fabricated personal opinion. The panel's job in practice:

* Ground every idea in real literature before building it.
* Rank backlogs by value-over-effort with honest effort labels ("pure
  assembly" vs "research-heavy, speculative").
* Re-audit the live code before each review cycle, and *shorten* the list
  when the audit finds work already done.

Panel documents in the repo: `PANEL_occlusion_speed_backlog.md`,
`holostuff_panel_*.md`, plus the research program and reviews kept alongside
the project.

---

## 8. The close-out ritual (every faculty, no exceptions)

Every build follows the same eight steps. Skipping one is how silos and
phantom features happen.

1. **Module + `_selftest()`** — the module runs standalone and proves itself.
2. **A pytest file** (`test_holographic_<name>.py`) covering the win, the
   edge cases, and pinning the kept negatives.
3. **Wire the faculty into `holographic_unified.py`** — default-off,
   backward-compatible, delegating (never duplicating).
4. **An integration test** with *locally-scoped* imports of
   cosine/bind/Propagator (never top-level) proving a cross-faculty pipeline
   end to end.
5. **README count markers updated via `sed`**, using the pytest
   collect-only count as ground truth (currently 2,170).
6. **`NOTES_concepts.md` append** — what was built, the test-count delta, and
   the kept negatives, loud.
7. **`tour.py` demo block** inserted before the S3 title line —
   underscore-prefixed locals, NumPy scalars cast to `float` in f-strings,
   closing line preserved.
8. **Clean zip rebuild** with a clean-extract verification. The canonical
   deliverable is `/mnt/user-data/outputs/holographic_vsa_complete.zip`; each
   session picks up from the canonical zip (filesystem snapshot rollbacks are
   a known hazard — recovery is restore-from-zip).

---

## 9. Environment & tooling notes (sandbox specifics)

* Working directory `/home/claude/work`; deliverables to
  `/mnt/user-data/outputs`.
* **pytest:** always `-p no:cacheprovider`.
* **Shell is sh/dash-flavored:** no here-strings, no process substitution, no
  brace expansion. Use temp files where process substitution would be used.
* **`tour.py` exceeds the bash wall-clock limit** — run it via a `setsid`
  wrapper script with a poll loop; kill stray `tour.py` processes before
  heavy work.
* **numba:** never `cache=True` under dynamically-loaded modules.
* **Blueprint-mounted demos:** `importlib.util.spec_from_file_location` with
  `sys.path` for clean module names.
* **Server-rendered PNGs are the verifiable deliverable** — WebGL/three.js
  cannot be verified headless here.
* Determinism env: `PYTHONHASHSEED=0`; hashes via `hashlib`.

---

## 10. Where the project is heading

The strategic fork is still open: **mesh-first** (deepen the half-edge kernel
toward Blender polygon parity) vs **native-first** (SDF/splat-first, which
plays to the NumPy-only constraint and the measured strengths). The mesh ↔
SDF ↔ splat bridge (`meshbridge`) deliberately keeps both doors open — three
views of one surface, convertible.

Panel-ranked near-term themes (see the backlog files for the live lists):
data-oriented/ECS composability (scene, simulation, and dataset as one VSA
program over a shared field), one shared spatial structure for cull and
navigation, dirty-flag physics/nav deltas (`dirtyfield` is the opening move),
the photo-to-3D pipeline (depth → unproject → per-pixel Gaussians, with the
honest boundary that learned depth models and learned back-geometry
hallucination are the genuinely non-borrowable parts), and the DCC/rendering
backlogs.

The recurring lesson, which is also the guide's summary: **one small set of
primitives, measured honestly and kept minimal, keeps turning out to be
load-bearing across fields that don't talk to each other.** A binding is a
rigid shift is a convolution is a phase rotation is a tensor contraction.
Generalize on contact, check above and below, treat every wall as a bad
approach, and never claim a win you haven't measured against a proper
baseline — with the negatives kept loud when the data says no.

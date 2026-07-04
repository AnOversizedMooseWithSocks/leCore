# holostuff Wiring Backlog — the integration audit (above/below sweep)

*Grounded in an audit of the LIVE code (165 modules, `UnifiedMind` at 296 faculties), not memory. The question: is
there functionality sitting siloed in tests/demos that the one mind cannot reach? The method: enumerate every
`holographic_*.py`, find which are imported by `UnifiedMind` (121) and which are not (44), then triage the 44 by
asking — for each — is it (a) a helper already reachable through a wired module, (b) a genuine capability the mind
should expose but does not, (c) a superseded duplicate of something already wired, or (d) deliberately standalone
(an app, a VM, a measurement harness, or an on-record kept-negative). Only (b) and (c) are real work.*

---

## Headline

**The recent work is fully wired, and the engine is mostly integrated.** Every module shipped in the last several
sessions — `bandwidth`, `kde`, `lod`, `flatness`, `splatprune`, `scenedelta`, `occlusion`, `harmonic`,
`splatdensify`, `relocate` — is imported by `UnifiedMind` and exposed as a faculty, with an integration test. No
recent functionality is siloed.

Of the 44 modules the mind does not import directly: most are **helpers reachable through a wired module**
(`encoders`, `image`, `field`, `relations`, `slime`, `mobius`, `ratedistortion`, `reasoning`, …) or **deliberately
standalone** (apps, VMs, measurement harnesses, and the on-record kept-negatives). The sweep found **six genuinely
siloed capabilities** worth wiring (one of them low-value), **one superseded duplicate** to retire, and a clear set
of modules that are correctly left out — listed below with the reason, so the "what NOT to wire" decisions are on the
record the way the integration plan asks.

---

## Tier 1 — genuine siloed capabilities to wire (faculty-shaped, no mind coverage, tested)

Each below is a real capability with a clean API and a passing test wrapper, that `UnifiedMind` does **not** expose and
does **not** cover under another name (verified by keyword + import search — the keyword hits that exist are docstring
mentions, not faculties).

### W1. Mixture of experts with a learned gate — `holographic_moe` (`GatedMixture`)  *(highest value)*
- **What it is.** A mixture of experts whose **gate is learned** — per input, a trained gate picks which specialist to
  trust. API: `add_expert` / `add_linear_expert`, `fit`, `predict`.
- **Why it is genuinely missing.** The mind dispatches by a **rule** (which verb you called, what type the input is) —
  that is routing, but it is *not* a learned gate, which is MoE's defining piece. The module's own docstring draws
  exactly this distinction. No faculty covers it (`moe`/`mixture of expert` → 0 faculty hits).
- **Wiring.** A `mixture_of_experts(...)` (or `gated_mixture`) faculty returning a `GatedMixture`; integration test that
  a learned gate routes a held-out input to the right specialist where the rule-dispatch cannot. Serves the
  Olshausen/Togelius seats (learned, interpretable routing).

### W2. Kinematics on the substrate — `holographic_physics` (`Kinematics`)  *(high value; ties the VSA-is-geometry thesis)*
- **What it is.** Physics as an algebra of binds: encode position/velocity as hypervectors, advance a trajectory by
  binding, read velocity back by **unbind**. API: `state`, `step`, `trajectory`, `read_velocity`.
- **Why it is genuinely missing.** The `physics` keyword appears only in docstrings as analogy; there is no kinematics
  faculty. This is the direct embodiment of "binding is a rigid transform" pointed at motion — the Stam/Macklin seats'
  territory, and a clean demonstration of the engine's core thesis.
- **Wiring.** A `kinematics(...)` faculty (or `learn_dynamics`-adjacent) exposing `trajectory` / `read_velocity`;
  integration test that a bound trajectory round-trips and the unbind reads the right velocity. Note honestly that the
  related `learn_dynamics` (Propagator) is already wired — this is the *closed-form kinematic* twin, not the learned one.

### W3. Motion-compensated temporal/video compression — `holographic_video` (`HolographicVideo`)  *(medium value)*
- **What it is.** The video-codec insight made concrete: estimate inter-frame **shift**, motion-compensate (a
  translation is a single bind, which **zeroes the residual**), and store the small residual. API: `encode`, `decode`,
  `mean_psnr`, with `estimate_shift` / `fourier_shift`.
- **Why it is genuinely missing.** The mind has token/sequence compression (`compress_lossless`, `learn_sequence`) and
  the rate-distortion code, but **no motion-compensated image/frame codec** (`temporal compress` → 0 faculty hits;
  `video` → docstring only). This is the image-domain application of the rigid-shift-is-a-bind property.
- **Wiring.** A `compress_video(frames)` / `video_codec(...)` faculty returning a `HolographicVideo`; integration test
  on a synthetic panning sequence (motion-compensated PSNR beats per-frame independent storage). Serves Stam/Puckette
  (temporal) and Duda (compression).

### W4. Versioned compressed store with rollback — `holographic_history` (`VersionedStore`)  *(medium value; practical for the 3D-app vision)*
- **What it is.** A store timeline: `commit` rows (with an optional proof/note), `checkout` / `rollback` to any
  version, `head`, `history`. Versioning with compression on the holographic substrate.
- **Why it is genuinely missing.** `versioned` appears once in a docstring; `rollback` → 0 hits. No faculty exposes
  commit/checkout/rollback. This is directly useful for the editable-mesh authoring vision (undo/redo, scene
  versioning) the project is heading toward.
- **Wiring.** A `versioned_store(...)` faculty returning a `VersionedStore`; integration test that a commit→edit→rollback
  round-trips exactly. A natural companion to the scene-delta faculty already shipped.

### W5. Hierarchical (sublinear) self-organizing memory — `holographic_graph_memory` (`GraphMemory`)  *(low–medium; partial redundancy, stated)*
- **What it is.** The maze lesson applied to store/retrieve: replace the flat prototype memory's O(prototypes) scan
  with a **hierarchical** cosine-kmeans tree. API: `observe_vector`, `organize`, `classify_vector`, `counts_by_label`.
- **Why it is partly redundant (the honest caveat).** The mind already has **sublinear recall** (`HoloForest`,
  random-projection trees) and **unsupervised class discovery** (`SelfOrganizingMind`, the flat version this upgrades).
  So the *purpose* (sublinear classification) is covered; `GraphMemory` is a *different mechanism* (a cosine-kmeans
  hierarchy) and the explicit hierarchical upgrade of the flat self-organizing memory. Wire it only if the hierarchical
  variant measurably beats the flat one at scale — otherwise leave it as a documented alternative. **Probe before wiring.**

---

## Tier 2 — cleanup (a superseded duplicate, not a wiring need)

### C1. `holographic_recurrent` is superseded by `holographic_reservoir`
- Both implement **reservoir computing / an echo-state network** (gradient-free recurrence, `run`/`fit`/`predict`).
  The mind's `reservoir` faculty is backed by `holographic_reservoir.HolographicESN`. `holographic_recurrent` is a
  parallel, tested, but **unreferenced** implementation of the same capability.
- **Action:** do NOT wire (the capability is already exposed). Either retire `holographic_recurrent` (and its test) or
  add a one-line deprecation note pointing at `holographic_reservoir`. This is the only true duplicate the sweep found.

---

## Correctly standalone — examined and deliberately NOT wired (on the record)

Per the integration plan's "a faculty must earn its method," these are reachable-as-source but should not become mind
methods. Listed so the decision is explicit, not an oversight:

- **Apps / runtimes above the mind** (like the `machine` VM, which the plan already leaves standalone):
  `holographic_orchestrator` (a task-planning / tool-registry / circuit-breaker agent layer), `holographic_navigator`
  (the *already-wired* creature repurposed as a data-navigation app), `holographic_creature_mind` (explicitly the
  *reference demo* of building a specialized mind ON `UnifiedMind`). Each has demo/`__main__` blocks — they are
  applications of wired primitives, not new primitives.
- **Measurement / audit harnesses** (the plan keeps these as module functions used by tests): `holographic_ablate`
  (the ablation table — "where is VSA load-bearing?"), `holographic_reanchor` (the re-anchoring audit), and the
  `holographic_probesweep` cross-cutting probe set.
- **Demos of capabilities already wired**: `holographic_backwardwarp` (demonstrates that backward warp *is* the
  engine's `unbind` — already exposed), `holographic_photos` (tests the wired image/archive stack on real photographs).
- **Knowledge / curriculum content layers** (borderline; judged content-over-capability): `holographic_encyclopedia`
  (a structured-knowledge layer over the wired structure faculties) and `holographic_lexicon` (a dictionary-first
  word-meaning curriculum). Wire only if a concrete need arises; today they are content built on wired primitives.
- **On-record kept-negatives** — must NOT be wired (wiring a documented negative would misrepresent it):
  `holographic_jittersplat` (sub-pixel jitter does not sharpen past the refit), `holographic_ldexplore`
  (low-discrepancy → creature exploration, no-op), `holographic_lookahead` (re-anchored creature lookahead, negative),
  `holographic_misgen` (multiple-importance → steered generation, no-op), `holographic_splatsharpen` (negative-lobe
  sharpening, negative). These are the engine's "negatives kept loud" — correctly siloed by design.

---

## Recommended sequence

Value over effort, each shipped under the standard close-out ritual (module exists and is tested for all but the
note below, so the work is: add the faculty, append an integration test, update README/NOTES/tour, rebuild + verify):

1. **W1 mixture_of_experts** — highest value (a genuinely distinct routing capability), clean API, tested. Lead.
2. **W2 kinematics** — high value, ties the core VSA-is-geometry thesis, clean + tested.
3. **W4 versioned_store** — practical for the editable-mesh authoring vision; pairs with the shipped scene-delta.
4. **W3 video codec** — the rigid-shift-is-a-bind property made into a codec; measure motion-comp vs per-frame.
5. **W5 hierarchical memory** — **probe first**; wire only if the hierarchy beats the flat self-organizing memory at
   scale (the purpose is already covered by `HoloForest` / `SelfOrganizingMind`).
6. **C1 retire `holographic_recurrent`** — cleanup; the reservoir capability is already wired via `holographic_reservoir`.

*Not in scope:* one `holographic_automaton` (HyperCA — Turing patterns in hypervector space) exists as a generative
demo but has **no test wrapper**, so it is untested in the suite; if its generative output is wanted as a faculty it
needs a test first, then wiring — lowest priority, and the only siloed module without a test.

---

## The honest bottom line

The sweep was reassuring in the engine's usual way: the *recent* work is all wired, and most of the 44 unimported
modules are either helpers reachable through a wired faculty or deliberately standalone (apps, VMs, audits, and the
kept-negatives that are siloed *on purpose*). The genuine debt is small and specific — **five capabilities the mind
should expose but doesn't** (a learned MoE gate, substrate kinematics, a motion-compensated video codec, a versioned
store, and — pending a probe — a hierarchical memory), plus **one duplicate to retire** (`recurrent`, superseded by
`reservoir`). Wiring W1–W4 closes the real gap; W5 and the `automaton` note are conditional on a measurement. That is
the integration plan's discipline applied to the whole tree: one mind those primitives serve, not a drawer of
disconnected experiments beside it — and where something is correctly left out, the reason is written down.

---

## RESOLUTION (all items closed)

- **W1 mixture_of_experts** — WIRED. Learned gate routes by content (>=0.85 on number-line split, beats either single expert). +1 integration test.
- **W2 kinematics** — WIRED. Binding-is-motion: trajectory by bind tracks truth, velocity by unbind, out-of-range raises. +1 integration test.
- **W4 versioned_store** — WIRED. Commit/edit/rollback exact round-trip, proof-gated, history never erased. +1 integration test.
- **W3 video_codec** — WIRED. Rigid pan: motion-compensated GOP beats per-frame intra (fewer bytes AND higher PSNR); non-rigid change kept as the honest loss. +1 integration test.
- **W5 graph_memory** — PROBED → NOT WIRED. Measured vs the flat store at 12→400 labels: comparisons sub-linear (10→33 vs 12→400) but accuracy collapses (1.00→0.55) while flat stays perfect. The flat scan is optimal for classification; the hierarchy's home is sparse navigable structure, not high-dim NN. Consistent with the module's own tests. Left as a documented alternative.
- **C1 recurrent** — DOCUMENTED → NOT RETIRED. Not a pure duplicate: two reservoir flavours + a ReservoirSequenceClassifier used by tour.py + real-corpora kept-negatives. Added a cross-reference note; the mind's clean reservoir is holographic_reservoir.HolographicESN. Nothing deleted (retiring would break the tour).

Faculties added across the wiring backlog: **4** (mixture_of_experts, kinematics, versioned_store, video_codec). Tests added: **4** integration tests. The engine is now either wired or documented as deliberately-standalone, end to end.

## From the demo-gallery handoff + modeling-gaps docs (July 2026 above/below audit)

Landed this pass: `holographic_pattern` (demo-gallery -> core, verbatim), `SurfaceMaterial` + `render_surface`
(CORE_NOTES 2.1, the biggest gap -- material channels as Param sockets resolved per hit, from_name consuming the ONE
canonical MATERIAL_RENDER table). Remaining threads, ranked by the docs' own build order:

1. Progressive/yielding `path_trace` (small; unblocks the refine stream).
2. SDF -> surface-splat bridge (one call over poisson_disk_sample + aniso_fit + splatexport; every scene splat-viewable).
3. `RenderSession` + one stream transport (wraps IncrementalRenderer + progressive path_trace; preview/final never diverge).
4. Expose `pick()` + object-id/depth AOV streaming (build_ray_index already has primary[pixel]=id -- exposure, not build).
5. Per-object TRS state + generalise incremental move() to rotate/scale/add/delete.
6. Vectorise the PBD constraint solve (measured 83 ms/step at 432 pts -- THE physics bottleneck; batched Jacobi).
7. `SimulationSession` (step/pause/scrub/poke; splat-proxy previews for everything-moves frames).
8. Field-predicate RegionField (regions by height/slope/noise predicates, not just nested shells).
9. Dual contouring / surface nets (feature-preserving field -> mesh).
10. Per-vertex normals in rasterize_mesh; material preview stage helper.

## Physical-definitions fork integrated (July 2026)
Landed: matlib + quantities + definitions modules (verbatim, 55 shipped tests green) + `data/definitions/`.
Integrated: SurfaceMaterial.from_matlib + faculties render_material/material_catalog/fractal_planet (render),
physical_material/resolve_scenario (sim), quantity/estimate_bill (grammar). Remaining fork phases (from
holostuff_scientific_databases_backlog.md): (2) upgrade every Definition with unit/uncertainty/source/external_ids +
extend the resolver grammar to parse quantities ("a 2 kg steel ball") and cost triggers; (3) ingest adapters per source
(OPTIMADE/Materials Project, PDG, PubChem, USGS, ICE, NASA/JPL) -- NETWORK-GATED, tested against vendored fixtures;
(4) new categories (particles, astronomical bodies, organisms, minerals) as new kinds in the one registry.
Next engine build regardless: SimulationSession (now has resolve_scenario/physical_material to feed it).

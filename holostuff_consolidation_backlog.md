# holostuff — consolidation backlog (the do-it list)

*Distilled from the two audits (`holostuff_consolidation_audit.md` = the datatype capabilities;
`holostuff_render_sim_consolidation_audit.md` = the render/sim domain) into one ordered checklist you can
work top-down. The audits hold the "why"; this holds the "what to do." Ordered so each item builds on the
last — but after Phase 0, the promotes are largely independent, so grab whichever fits your day.*

---

## How to use this — the same six steps for every item (so each task below doesn't repeat them)

1. **Pick the canonical impl** — the best one that already exists (named per item below).
2. **Wrap it in a small, readable home** — a plain class/module with a clear API and comments. No cleverness.
3. **Point the duplicates at it** — turn the scattered copies into thin shims that call the home. Keep the old
   function names working so nothing downstream breaks (additive only — the engine's rule).
4. **Pin it bit-exact** where it matters — on the tie-sensitive paths a result must be *measured identical*
   (the "1e-12 flipped a maze trajectory" lesson). If it can't match exactly, keep the home out of that path.
5. **Register it in the catalog** (Phase 0) — one line: what it does, an example call, and whether it's native.
6. **Prove it with one integration test** — a cross-module (or through-the-pipeline) test that shows a real
   caller now uses the home end to end. That test is what makes "wired up" true instead of claimed.

**Golden rule, kept loud:** *route, don't rewrite.* A different algorithm (path-trace vs PRT; fluid vs cloth)
is **not** a duplicate — unify the scaffolding around the strategies, never the strategies themselves.

---

## Phase 0 — the catalog (do this first; it's cheap and it stops the bleeding)

**C1. [DONE -- batch C1] The capability catalog — "search before you build."**
- Source → home: promote `holographic_knowledge` (query-by-similarity) + `capability_registry` into one
  `catalog` with `register_capability(...)` and `find_capability(problem)`.
- Do: give each entry a plain-English `does`, a copy-paste `example`, and a `native` flag (True = batched/fusable/
  stays in the vector domain; False = hops to Python). `find_capability` runs a fuzzy search over `does`.
- Effort: low.
- Done when: `find_capability("search a big pile of vectors")` returns the search home; seed it with the **7
  indices, 9 caches, and 8 field types** the audits already named so they're findable today.
- **DONE:** holographic_catalog.py (Catalog + register_capability/find_capability; default_catalog seeds the homes; seed_from_mind adds live faculties). Wired UnifiedMind.find_capability/register_capability. 'search a big pile of vectors' -> Index home; 702 capabilities findable through the mind. +9 tests + 1 integration.

---

## Phase 1 — the datatype and its capability homes (the foundation everything sits on)

**D1. [DONE -- batch D1] The first-class hypervector datatype.**
- Source → home: a small `Hypervector` wrapper — the numpy array + its `dim` + which encoder made it + a
  "what am I" tag; the five verbs (`bind`/`unbind`/`bundle`/`cleanup`/`permute`) as methods.
- Do: the **make** side is the encoders (`UniversalEncoder` + the typed scalar/text/record/FPE encoders) — label
  them as "the constructors." The **consume** side is the verbs + the homes below + the readouts
  (`decode`/`sample`). Keep it a thin wrapper — don't hide the raw array (some hot paths need it).
- Effort: medium.
- Done when: you can build one from any data, call the five verbs as methods, and get the raw array back cheaply. **DONE:** holographic_hypervector.Hypervector -- thin wrapper (array+dim+encoder+tag); wrap/encode constructors; bind/unbind/bundle/permute/cleanup as methods (bit-identical to the bare ops); .array/.raw()/np.asarray(hv) return the raw array with no copy; mind.hypervector() faculty. +8 tests.

**H1. [DONE -- batch H1] `Index` — search / recall.** Canonical: `HoloForest` (`tree`) + `pivot` + `rayindex`.
Delegates: `archive`, `organizer`, `analog`, `navigator`. Do: one `Index.nearest(query, k)` (exact scan for
tiny sets, RP-forest for large, spatial variant for rays) with a calibrated confidence + abstain (the recall-null
work). Done when: two of the delegates call `Index` and the recall benchmark is unchanged. **DONE:** holographic_index.Index -- Index.nearest(q,k,abstain) routing exact-scan/RP-forest by size + calibrated abstain (RecallNull); lexicon.nearest & TextEncoder.nearest delegate with byte-identical rankings; bench_recall unchanged. Scout found extra indices (spatial.knn, uri.nearest, octree.query) -- kept distinct (different metrics), registered in catalog. +7 tests. *Med.*

**H2. [DONE -- batch H2] `Cache` — caching / baking (= `bake_and_query`).** Canonical: `cache`/`compile` + the bakes
`matbake`/`sdfbake`/`viewlut`/`residency`/`anim`. Do: one `Cache.bake(evaluator, vary=...)` where `vary` is
`constant`/`position`/`view`/`time`; bake the slow factor, look it up. Done when: `matbake` and one other bake
call `Cache` and produce bit-identical output. *Med.*

**H3. [DONE -- batch H3] `Scale` — scaling / distribution.** Canonical: `distribute` (`partition` + monoid `reduce`).
Strategies: `octree`, `multires`, `superposed`, `tiling`, `sparsefield`. Do: `Scale.map_reduce(...)` with those
as pluggable strategies. Done when: one domain scaler delegates and the result matches. **DONE:** holographic_scalehome.Scale -- map_reduce/partition/tiles/bricks over holographic_distribute; strategies tiling/octree/multires/superposed/sparsefield named. The mind distribute_compute/partition_domain/partition_grid/distribute_bricks all delegate, bit-identical; map_reduce == np.sum verified. +9 tests. *Med.*

**H4. [DONE -- batch H4] `Blend` — blend / merge / interpolate.** Canonical: `bundle` (superposition) + `sphere`
(slerp / Riemannian mean) + a merge-with-conflict-policy (the workspace/scene-combine discipline). Delegates:
`mixture`, `occlusion`, `phasemorph`, `blendpose`. Done when: two delegates call `Blend`. **DONE:** holographic_blendhome.Blend -- bundle(weighted)/lerp/slerp/mean/alpha_composite/merge over ai.bundle+ai.slerp+sphere.frechet_mean. blendpose.blend_pose + generate.morph_images delegate, bit-identical; phasemorph/mixture/occlusion kept distinct (specialised). +8 tests. *Low–med.*

**H5. [DONE -- batch H5] `Transform` — transform / warp.** Canonical: `bind` (rigid) + `permute` (order) + `clifford`
(rotate) + `steering` (anisotropic). Delegates: `backwardwarp`, scenegraph transforms. Done when: two delegates
call `Transform`. **DONE:** holographic_transformhome.Transform -- facade over bind/permute (VSA) + 4x4 matrices (holographic_transform) + clifford rotor + steering. Found+fixed a real DUP: scenegraph copied transform's translation/scaling/compose; scenegraph + procgen now delegate, bit-identical (scenegraph Rodrigues rotation kept distinct, ~1e-12). +8 tests. *Low–med.*

**H6. [DONE -- batch H6] `Memory` — the cache hierarchy (L1–L4 + RAM).** Canonical: `residency` (keep reused FFT spectra
resident) + the contiguous-array/`bind_batch` layout + tiling-to-fit + the opt-in `backend` (GPU)/`jit` paths.
Do: a home that says "keep the hot working set where the CPU can reach it fast" and exposes those levers. Done
when: the `residency` path is reachable through `Memory` and a batched kernel is measurably cache-resident. **DONE:** holographic_memoryhome.Memory -- residency (spectrum_cache/bind_cached, bit-identical)/bind_batch/tiles/backend. mind.spectrum_cache routes here; batched record encode measured ~2-3x faster than the per-pair loop (cache-resident). +7 tests. *Med.*

**H7. [DONE -- batch H7] `Compute` — stay VSA-native (no Python hops).** Canonical: `fuse` (bind chain → ~2 FFTs, no
Python between ops) + `superschedule` (width) + `schedule` (the cost model) + `machine` (run logic as a VSA
**program**). Do: `Compute.fuse(chain)` and `Compute.as_program(...)`; the rule is **push decisions/cleanups to
the boundaries**, keep the hot middle in the vector domain. Done when: a multi-op chain runs fused (measure the
FFT-count drop) and a small routine runs as a `machine` program. *Med.*

**H7 DONE:** holographic_computehome.Compute -- fuse/fuse_record (measured 2*len+2 FFTs vs ~3*len; a 32-pair record 66 vs 96, ~31% fewer, agrees to ~1e-15) + run_recipe/run_scheduled + machine. mind.fuse_record/fuse_expression route here, bit-identical. +6 tests.

> As each H-item lands, **register it in the catalog** with `native=True`. That's what makes "always take full
> advantage in all scenarios" real rather than aspirational.

---

## Phase 2 — the render / sim homes (these compose *through the pipeline*)

**R1. [DONE -- batch R1] `Pipeline` — the one entry point.** Canonical: `holographic_pipeline` (already built for this —
Stage `needs`/`produces`). Do: make it *the* way to compose a run; register the render methods
(`pathtrace`/`prt`/`radiance`/`raymarch`/`render`) as **strategies** it dispatches (with `adaptive` as the
picker) — a routing, not a merge. Done when: a render goes through `Pipeline` and the `needs`/`produces` check
catches a deliberately-missing input. *Med. Highest leverage — do it first in this phase.*
- **DONE:** RENDER_STRATEGIES registry (pathtrace/raymarch/prt/radiance), each declaring `needs` and delegating to its module; `RenderSpec.method` + `dispatch_render` (auto-picker + needs-check raising a clear PipelineError). pathtrace byte-identical (pinned); all 27 prior pipeline tests green. Registered 'Pipeline (render/sim)' in the catalog. +5 tests.

**R2. [DONE -- batch R2] `Field` — one field, several backends.** Canonical: `ndfield` (already "the reusable pattern").
Strategies under it: dense (`fields`), narrow-band (`sparsefield`), spectral (`spectralfield`), FPE (`fpefield`),
region (`regionfield`), dirty (`dirtyfield`). Do: one `Field` interface; the reps are backends chosen by cost.
Done when: two of the field modules are reachable as `Field` backends with identical values. **DONE:** holographic_fieldhome.Field -- field.sample(points) routing to callable/dense/sparse backends; dense & callable agree to 0.0 at grid nodes; sparse routes to SparseField.sample. +6 tests. *Med. The biggest
genuine de-duplication; everything sim/render sits on it.*

**R3. [DONE -- batch R3] `Material` + `Shading`.** Canonical: `material` (the channel record) + `brdf` (the shade model).
Do: `material` stays the home; its **bakes delegate to `Cache` (H2)**; its **procedural sources go to `Texture`
(R6)**; render methods call `Shading` (brdf) instead of re-deriving. Done when: a shaded surface pulls channels
from `Material`, shades via `Shading`, and bakes via `Cache`. **DONE:** Shading (brdf) already centralised (no re-derived Fresnel/GGX found); added the missing brdf.lambert diffuse term + routed globalillum bit-identical; matbake.bake_material bakes channels via Cache (H2); three-way integration test green. Registered Material + Shading in catalog. Compound shades kept their own expression. +2 tests. *Med.*

**R4. [DONE -- batch R4] `Sampling`.** Over shipped parts: `sampling` (Poisson/blue-noise) + `lowdiscrepancy` +
`adaptive_sample` + `mis` + `traverse` (Russian-roulette) + `accumulate` (firefly). Do: one home — patterns,
adaptive, MIS, termination. Done when: `pathtrace` gets its samples from `Sampling`. **DONE:** holographic_samplinghome.Sampling -- routes low_discrepancy/poisson/mis/accumulate, OWNS the cosine-hemisphere that was copied in 3 modules. pathtrace AA offsets + globalillum + lightcache + mind.low_discrepancy_sample + mind.blue_noise_sample all route through it, bit-identical. +8 tests. *Low–med.*

**R5. [DONE -- batch R5] `Denoise`.** Over shipped parts: `svgf` + `denoise` (manifold/NLM) + `sharpen` + the VSA
`cleanup`/`consolidation`. Do: one home with the demodulation refinement from the modulation backlog (divide out
albedo → denoise the smooth part → multiply back). Done when: the pipeline's denoise stage calls `Denoise`. **DONE:** holographic_denoisehome.Denoise -- .image(svgf/demodulated) / .sharpen / .signal(manifold family), all routing. Pipeline denoise stage + mind.svgf_denoise + mind.sharpen_loop route through it, bit-identical; signal family stays on mind.denoise. +9 tests. *Low–med.*

**R6. [DONE -- batch R6] `Texture`.** Over shipped parts: `noise`/`curlnoise` (procedural) + `texturesynth`
(example-based) + the weathering set (`burn`/`oxidation`/`cellular`/`inclusions`). Do: one home that feeds
`Material` channels. Done when: a `Material` channel is sourced through `Texture`. **DONE:** holographic_texturehome.Texture -- fbm/voronoi/curl/synth returning channel-ready fields. A Voronoi crack field drives a SurfaceMaterial roughness channel; integration test walks Texture->Material->Cache->Shading. +7 tests. *Low–med.*

**R7. [DONE -- batch R7] `Lighting`.** No home today — shading is recomputed in `brdf`/`prt`/`globalillum`/`radiance`/
`spharm`/`raymarch`. Do: one home for light types (directional/point/area/env, SH) + the shade integral; render
methods call it. Done when: two render methods get lighting from `Lighting`. **DONE:** holographic_lightinghome.Lighting -- re-exports the 10 light types + the shade integral in modes direct/prt/environment_sh. Cached soft-light + indirect passes route to Lighting.direct, pipeline PRT strategy to Lighting.prt, all bit-identical. +6 tests. *Med.*

**R8. [DONE -- batch R8] `Shadow`.** No home today — visibility is redone in `prt`/`pathtrace`/`globalillum`/`raymarch`/
`occlusion`. Do: one home with shadow-rays / PRT-visibility / SDF-soft-shadow as strategies. Done when: two
render paths get visibility from `Shadow`. **DONE:** holographic_shadowhome.Shadow -- soft/ambient_occlusion/hard shadow-ray strategies (+ PRT baked note). raycoherence & semantic route their visibility through it, bit-identical (25 tests pass). +6 tests. *Med.*

**R9. [DONE -- batch R9] `Simulation` scaffold.** The solvers exist and are legitimately distinct (`fluid`, `combustion`
(fire), `smokepresets`, `softbody`, `cosserat`/`groom`, `mpm`, `collide`); what's missing is a **shared step
loop** — each has a different step signature today. Do: one `Simulation.step()` over the `Field` (R2), with each
solver as a strategy that plugs into the `Pipeline` (R1). Keep the solvers separate. Done when: two solvers step
through the same loop and the pipeline renders their field. **DONE:** holographic_simulationhome.Simulation -- a shared step loop over integrate.SolverAdapter; Simulation.for_fluid + for_automaton step two DISTINCT solvers (Stable Fluids + reaction-diffusion) through one loop; each field exposed as a Field (R2) and rendered via volume_render (R1). Solvers kept separate. +7 tests. *Med. Do after R1 + R2.*

---

## Dependencies (so you can sequence)

- **C1 first** — unblocks discovery for everything.
- **D1** underpins the H-homes (they're what the datatype *does*); but the H-promotes can start before D1 and be
  hung off it later.
- **R3 Material → needs H2 Cache** (for its bakes). **R9 Simulation → needs R2 Field.** **R2 Field → connects to
  D1** (a field is a hypervector-valued thing). **R1 Pipeline** unblocks proving every R-item end to end.
- Everything else is independent — pick by appetite.

Suggested order to knock out: **C1 → R1 → R2 → H1 → H2 → R3 → R4/R5/R6 → H3/H4/H5 → R7/R8 → H6/H7 → R9**, each
landing with its catalog entry and one integration test.

---

## The honest boundaries (don't skip these)

- **Route, don't rewrite.** Different algorithms (render methods, sim solvers) are strategies — unify the
  scaffolding, keep the strategies distinct. Merging them is a leaky abstraction that helps no one.
- **Pin the tie-sensitive paths bit-exact**, or keep the shared home out of them.
- **Don't over-consolidate.** If a domain impl has genuine special knowledge (`rayindex` knows rays; `matbake`
  knows material channels), unify the shared core and let the special part stay a thin layer on top.
- **Some Python hops are unavoidable** — a real decision must cross; push it to the boundaries, and mark it
  honestly in the catalog (`native=False`) rather than faking it.
- **Coupled multi-physics doesn't fully generalise** — the shared step loop gives you the frame; the coupling
  terms (fire heating a fluid, cloth on rigid) are per-pair work, kept honest.

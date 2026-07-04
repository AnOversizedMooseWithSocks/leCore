# Render Pipeline & Materials — Backlog

*Findings from the demo-scene / gallery journey (spheres, glass, fractal, one-surface-many-materials, fur,
ocean). This is a working list of what to FIX, WIRE, PROMOTE, or BUILD in the render + materials + lighting +
hair path. It complements the existing `holostuff_rendering_backlog.md` and
`holostuff_panel_review_rendering_backlog.md` — cross-check before starting so we don't duplicate.*

## Progress (this session)

**DONE:** J1 (render bench — the speed guard), A1 (pipeline render/gbuffer/svgf stages now run the REAL scene via
a `RenderSpec` + `converge_samples`, matching `render_auto` bit-for-bit; demo path unchanged), A4 (variance-guided
SVGF passed through the pipeline), E1 (deleted the duplicate tonemap — `aces_tonemap` now delegates to
`holographic_postfx`, which gained `auto_exposure`), E2 (pipeline `postfx="aces"` present stage). Paired showcase
I1 (smoke & fire from one fluid sim through `volume_render`) shipped as the visual proof. **DONE (batch 2):** B1 (spheres + glass + identities now pull physical materials from `matlib`; only ocean
hand-codes, and it uses a bespoke water shader anyway), D1 (sun-sky HDRI via `sky_dome` is the gallery
default), H1 (low-discrepancy sub-pixel anti-aliasing wired into `path_trace` as opt-in `antialias=`, on by
default in `render_auto`; measurably lowers edge jaggedness; the bench comparison made AA-consistent).
**DONE (batch 3):** H7 -- the renderer now consumes the canonical SCENE DOCUMENT. New holographic_scene_render.py flattens a holographic_scene_doc.Scene into (sdf, material_fn): every object placed by its transform, union-ed into one scene SDF, each hit shaded by its owning object's LIBRARY material; render_scene_document renders it in one call. Wired as UnifiedMind faculties (render_scene_document / scene_to_render) + integration test + tour block. **DONE (batch 14):** BUILD M1 (modulate/demodulate primitive = bind/unbind, holographic_modulate.py) + M4 (demodulated denoise: divide albedo out, denoise smooth irradiance, multiply back). Probed first -- the 5 bakes (matcompile/matbake/viewlut/prt/radiance) already exist, svgf was guide-only, no demodulate primitive. Measured 33% less denoise error on textured diffuse; KEPT NEGATIVE: neutral on uniform albedo (so NOT the matte-speckle fix), diffuse only, background masked. Wired render_auto(demodulate=False)->render_scene_document->mind, default-off/byte-identical. NEXT for the matte speckle specifically: extend the dome-style soft-light cache to the area lights.

**DONE (batch 13):** BUILD the CACHED DOME (RENDER-DC1) -- holographic_domecache.py serves the soft dome/sky-ambient light as a three-tier cache (warm: bake PRT transfer at a coarse anchor grid; hot: smooth normal-aware interpolation; cold: recompute exactly at the edges). 15-16x faster than baking every pixel, ~40x vs brute path-traced AO, noise-free, ~96% hit rate; the 323s dome is now sub-second. Wired into render_scene_document(dome_cache=False) + the mind, default-off/byte-identical. Fixed the blocky-shadow facets (Gaussian gather, not bilinear). This is the first shipped piece of the two-mode cached-lighting design. NEXT: extend the cache to placed area lights + route hard contact shadows to per-pixel NEE (the cold tier calling the tracer).

**DONE (batch 12):** BUILD the full light rig -- expanded holographic_lights from 3 to 8 types (point, directional, ambient, spot+gobo, rect/area, sphere, mesh, IES) with colour/intensity FIELDS and a load_ies() parser (showcase render_light_types). NEXT build: holographic_water driven by spectral_ocean.

**DONE (batch 11):** BUILD placed lights + next-event estimation -- closed the tracer's own flagged kept-negative: new holographic_lights.py (Point/Directional/Sphere lights + NEE with shadow rays), wired through path_trace/render_auto/scene_doc/pipeline as lights=None (byte-compatible). Real lamps with correct hard/soft shadows (showcase render_lit_scene). NEXT build: holographic_water driven by spectral_ocean.

**DONE (batch 10):** BUILD thin-film iridescence -- the one MISSING material: new holographic_thinfilm.py (two-beam interference -> CIE colour), a 7th material channel (film thickness nm) tinting the reflection by view angle in the path tracer, presets soap_bubble/oil_slick/beetle_shell + matlib.iridesce() (showcase render_iridescence). The materials story is now complete. NEXT builds: light objects + next-event estimation (the tracer only has environment lighting), holographic_water driven by spectral_ocean.

**DONE (batch 9):** H4 -- HAIR/FUR is a first-class pipeline stage: render_hair gained an opt-in coverage alpha, and the pipeline hair stage over-composites fur onto a path-traced body (showcase render_fur_over_scene). The render-STAGE wiring trio (volume H5, particles H6, hair H4) is COMPLETE -- all three share one shape: build -> render to (image,alpha) -> over-composite. NEXT: the genuine BUILDs -- thin-film iridescence (the one missing material), light objects + next-event estimation, holographic_water.

**DONE (batch 8):** H6 -- PARTICLES are a first-class pipeline stage: new holographic_pointsplat.py projects+splats a point cloud (the missing renderer for the particle sim), and the pipeline particle stage over-composites it (showcase render_sparks_over_scene). NEXT: hair as a pipeline stage (H4), then the genuine builds (thin-film iridescence, NEE/lights, holographic_water).

**DONE (batch 7):** H5 -- VOLUME is a first-class pipeline stage: a RenderSpec can carry a smoke/fire/fog volume, and the pipeline volume stage over-composites it onto the surface render (showcase render_smoke_over_scene). NEXT structural items: particle render stage (H6), hair as a pipeline stage (H4), and the genuine gaps (thin-film iridescence, NEE/lights, holographic_water).

**DONE (batch 6):** H2 cont. -- physical-structure materials: crystal grains (Voronoi) and ore inclusions drive albedo via per-point sockets carried on scene objects (showcase render_crystal). The physical-property-materials thread (SSS, thermal, crystal/inclusions) is now wired; thin-film iridescence remains the one MISSING material.

**DONE (batch 5):** H2 cont. -- thermal emission: a material glows by its TEMPERATURE (matlib.heat -> blackbody -> emission; showcase render_hot_metal). Also rebuilt the SSS demo with varied-thickness shapes (torus/displaced/hollow) so the effect actually reads.

**DONE (batch 4):** H2 (partial) -- subsurface scattering wired into the path tracer, DRIVEN BY THE MATERIAL (wax/jade/skin/marble carry an sss strength; the tracer adds the thin-glow term; showcase render_subsurface + tour). **NEXT:** the remaining builds -- scene-graph as render input (H7), volume/particle/hair render STAGES in the
pipeline (H5/H6/H4), subsurface + physical-property materials into the shader (H2), and the genuine gaps
(thin-film iridescence, NEE/lights, a `holographic_water` module driven by `spectral_ocean`).

## The one-line theme

**Most of this is a wiring problem, not a missing-capability problem.** The engine already had the good pieces —
the calibrated convergence stop rule, variance-guided denoise, an HDR sun-and-sky, an ACES tonemapper, guide-hair
interpolation, a 130-material library — but the renderer and the gallery didn't use them. So scenes kept
*reinventing* worse versions: grainy raw path tracing, flat two-tone skies, Reinhard tonemapping, hand-typed
material tuples, brute-forced strand counts. A smaller number of items are genuine gaps (open-surface dielectrics,
distance-based absorption, a fiber material model).

## Tags

- **WIRE** — the capability exists but nothing connects it to the renderer.
- **FIX** — something is incorrect or broken.
- **PROMOTE** — real rendering logic currently lives in the gallery or a grab-bag module; it should be library code.
- **BUILD** — genuinely missing.
- **ORG** — organization / naming / dedup.
- **PERF** — too slow.

---

## A. Render pipeline & sampling

| ID | Tag | Item | Where | Fix |
|----|-----|------|-------|-----|
| A1 | WIRE/FIX | The pipeline's stages are **demo-backed**. `_gbuffer_run` / `_render_run` / `_svgf_run` / `_adaptive_run` run a synthetic left/right-split `_demo_scene`, not a real render — so `PipelineConfig(denoise="svgf")` never denoised an actual image. | `holographic_pipeline.py` | Point the stages at the real path (`path_trace` + `primary_gbuffer` + variance-guided `svgf`). `render_auto` already composes exactly this — fold it in as the pipeline's render body. |
| A2 | PROMOTE | `render_auto`, `primary_gbuffer`, `render_denoised`, `declfirefly` landed in `holographic_gbuffer.py`, which is a grab-bag sitting next to the gallery. | `holographic_gbuffer.py` | Move into the render subsystem and make it the canonical render entry the pipeline calls. |
| A3 | WIRE | The calibrated convergence stop rule was **siloed** — `render_auto` now drives it, but the pipeline's own `adaptive_samples` stage still doesn't. | `holographic_adaptive_sample.py`, `holographic_pipeline.py` | Unify on one adaptive path. |
| A4 | FIX (done — verify) | SVGF wasn't **variance-guided** (fixed colour sigma → it over-smoothed already-converged pixels; this is what forced the per-scene `svgf_levels` hand-tuning). | `holographic_svgf.py` | Fixed via `variance=` in `atrous_bilateral`. Confirm every render path passes the variance map, not just `render_auto`. |

## B. Materials

| ID | Tag | Item | Where | Fix |
|----|-----|------|-------|-----|
| B1 | WIRE | The renderer **never consumed a material object** — every scene hand-codes a `material(P)` tuple. | `make_gallery.py`, `holographic_pathtrace.py` | Route scenes through `matlib.shade(mat)`; make "material → shader inputs" the one bridge (started: `matlib.shade` / `fiber_params`). |
| B2 | FIX (done — audit) | `matlib.material()` didn't populate physical dielectric properties — glass/gem/liquid presets came back **opaque** (water had `ior=1.5, transmission=0`). | `holographic_matlib.py` | Fixed by class/name. Audit every glass/gem/liquid/ice preset for correct IOR + attenuation colour. |
| B3 | BUILD (done — verify) | `PBRMaterial` lacked the physical channels the renderer needs: **IOR, transmission, absorption, fiber**. | `holographic_materialio.py` | Added as the real glTF extensions (`KHR_materials_ior`/`_transmission`/`_volume`). Verify glTF round-trip and that the render actually reads them. |
| B4 | ORG | There are **two material systems** with an unclear relationship: `holographic_material.Material` (VSA texture-field record) and `materialio.PBRMaterial` (glTF factor set). | `holographic_material.py`, `holographic_materialio.py` | Document which is the render source of truth and how the VSA record relates to it. |
| B5 | BUILD | Scenes hand-write the **region logic** ("this half of the floor is checker A, that half is B; this x-range is copper"). | `make_gallery.py` | A scene should assign a *material to an SDF region* and let a shader resolve it. Lighter scenes, one code path. |

## C. Dielectrics — glass, water, caustics, dispersion

| ID | Tag | Item | Where | Fix |
|----|-----|------|-------|-----|
| C1 | FIX/BUILD | The path tracer's glass is a **closed-object** model (enter one face, march to the far face, exit). An open surface — water over a floor — doesn't fit it, so the ocean water went black. | `holographic_pathtrace.py` (`_march_through`) | Either a general refractive-interface model, or an explicit water/volume path. (For now the ocean uses a separate water shader — see C5.) |
| C2 | BUILD | **No distance-based absorption.** `attenuation_distance` is carried on the material and exported to glTF, but the tracer only tints transmitted light per-interface by albedo — so thick vs. thin coloured glass look the same. | `holographic_pathtrace.py` | Beer-Lambert `exp(-sigma * path_length)` over the marched distance through the dielectric. |
| C3 | WIRE/BUILD | **Caustics are a bolt-on.** They're a separate forward-light-trace composited onto the floor in the gallery, not part of the render. | `holographic_globalillum.caustics`, `make_gallery.py` | Integrate a caustic pass into the render path; keep the reusable `add_caustics` (currently in `gbuffer`). |
| C4 | BUILD | **Dispersion is a 3× RGB composite** at the gallery level (trace R/G/B with different IOR). Fine for a hero shot, not a core capability. | `holographic_gbuffer.render_dispersion` | If wanted properly, a hero-wavelength spectral path inside `path_trace`. |
| C5 | PROMOTE | The **water shader lives in the gallery** (`render_ocean`): Fresnel + Snell refraction + Beer-Lambert depth + caustics. It's reusable and shouldn't be buried there. | `make_gallery.py` | Promote to a `holographic_water` module. |

## D. Lighting & environment

| ID | Tag | Item | Where | Fix |
|----|-----|------|-------|-----|
| D1 | WIRE | Scenes used a **flat two-tone sky**; `sky_dome`'s bright HDR sun disk + HDRI-image support went unused (this was the "washed out" cause). | `holographic_raymarch.sky_dome`, `make_gallery.py` | Make a proper sun-and-sky / HDRI environment the default lighting. |
| D2 | BUILD | The path tracer has **no explicit lights / no next-event estimation** — it only gathers light when a bounce randomly hits the emissive environment. Great for a big sky, very noisy for small/bright emitters. | `holographic_pathtrace.py` | NEE / light sampling with MIS is the honest next step (the module's own kept-negative). |
| D3 | ORG | The **rasterizer has light objects** (`render.light`) but the **path tracer doesn't** — two separate lighting worlds. | `holographic_render.py`, `holographic_pathtrace.py` | Unify one light representation across both. |

## E. Tonemapping / post / colour

| ID | Tag | Item | Where | Fix |
|----|-----|------|-------|-----|
| E1 | ORG/DEDUP | **`holographic_postfx` already has `aces()`, `reinhard()`, `exposure()`, `gamma()`, bloom, vignette, and a graded-frame preset.** My `aces_tonemap` (in `gbuffer`/gallery) is a **duplicate**. | `holographic_postfx.py`, `holographic_gbuffer.py` | Delete `aces_tonemap`; tonemap via `postfx.aces` + `postfx.exposure`. Add the **auto-exposure** (log-average → mid-grey) I wrote as a `postfx` helper so it lives with the rest. |
| E2 | WIRE | The pipeline has a `present` stage but **no postfx/tonemap stage** — a rendered HDR buffer never becomes a graded display frame in-pipeline. | `holographic_pipeline.py` | Add a `postfx` stage (exposure → tonemap → grade) before `present`. |

## F. Hair / fur

| ID | Tag | Item | Where | Fix |
|----|-----|------|-------|-----|
| F1 | PROMOTE/PERF | **Density should use guide interpolation, not brute force.** `groom.interpolate_strands(guides, render_roots, clump)` already exists, but I grew 24,000 strands from scratch (a ~4-minute render). Clumping would also fill gaps naturally. | `holographic_groom.interpolate_strands`, `make_gallery.render_fur` | Groom a few hundred guides, interpolate thousands of render strands with clumping. |
| F2 | PROMOTE/BUILD | **Combing/styling isn't a groom feature.** `groom` grows strands along the surface normal with only an arbitrary per-strand `lean` — no coherent comb/flow/gravity. I did `_comb` + gravity droop at the gallery level. | `holographic_groom.py`, `make_gallery.py` | Promote comb-along-a-direction, gravity-relax, and clump into `groom`. |
| F3 | PERF | **The strand rasterizer is a Python loop of 1-px DDA lines** — slow (the 4-min fur) and aliased (needed 2× supersampling to hide it). | `holographic_hairshade._draw_segment` / `render_hair` | Vectorize: batch all segments and splat with `np.add.at`; add line width + anti-aliasing. Single biggest hair win. |
| F4 | WIRE | **Fur is lit by its own key/rim rig**, separate from the scene's sun/environment and the path tracer. | `holographic_hairshade.render_hair` | Light strands by the same environment as the rest of the scene. |
| F5 | FIX (optional) | The fiber "material" is only **partially physical** — absorption from colour, but no true eumelanin/pheomelanin model and no azimuthal roughness. | `holographic_hairshade.marschner`, `holographic_matlib` | Deeper physical fiber model if we ever want hero hair. |

## G. Organization (cross-cutting)

- **G1 [ORG]** — `make_gallery.py` carries real rendering logic (the water shader, the comb, material-region
  assignment, the benchmark harness). The gallery should *compose* library calls, not *contain* the renderer.
- **G2 [ORG]** — render utilities are scattered across `holographic_gbuffer` (grab-bag), the gallery,
  `holographic_pipeline`, `holographic_postfx`, `holographic_svgf`, `holographic_globalillum`. There's no clean
  render-module boundary. The pipeline is the intended composition layer — but it's demo-backed (A1), so it isn't
  actually the front door yet. Decide the boundary and make the pipeline real.
- **G3 [ORG]** — reconcile this list with `holostuff_rendering_backlog.md` and the panel review, fold in, drop dupes.

---

## Already done this journey (partial fixes — most still need promotion/hardening per the items above)

- `primary_gbuffer`, `render_denoised`, `render_auto`, `declfirefly` — `holographic_gbuffer.py` *(promote: A2)*
- variance-guided `atrous_bilateral` — `holographic_svgf.py` *(A4)*
- `render_auto` wired as a `UnifiedMind` faculty
- `PBRMaterial` physical extensions; `matlib.material()` physical props + `fiber` class + `shade`/`fiber_params` *(B1–B3)*
- `render_hair` reads physical fiber params *(F4/F5 partial)*
- `aces_tonemap` — **should be replaced by `postfx.aces`** *(E1)*
- `render_dispersion`, `add_caustics` — gallery/`gbuffer` *(promote: C3/C4)*
- the water shader in `render_ocean` — **promote to a module** *(C5)*
- fur comb + gravity droop + supersample + key/rim lighting — gallery *(promote: F1–F3)*

## Suggested order

1. **Cheap wins that stop the reinvention:** E1 (dedup tonemap → `postfx`), A1/A2 (make the pipeline call the real
   render path), B1 (route scenes through materials).
2. **Real gaps worth building:** C5 (water module) + C2 (distance absorption), F3 (vectorize the hair rasterizer —
   the biggest perf item), F1 (guide-interpolated density).
3. **Deeper / optional:** D2 (NEE + light objects), C4 (spectral dispersion), F2 (comb as a groom feature),
   B4/B5 + G1/G2 (the organization cleanup once the wiring above is settled).

---

# H. Wiring Sweep — every render capability vs. what actually reaches a frame

*Method: the tour is the running index of what the engine can do (everything we build gets a tour block), and
`UnifiedMind` is the faculty surface. I swept both against three questions per capability: does it exist (tour +
faculty), is it wired into the **pipeline** (`holographic_pipeline.py`, the frame-compositing layer), and does it
reach the **actual render path** (`render_auto` / `make_gallery`)?*

**The headline.** There are ~100 render-related `UnifiedMind` faculties — materials, physical-property materials,
lights, GI, caustics, fog, fluid/smoke/fire, hair, particles, ocean/waves, textures, splats, scene-graph, postfx.
**Nearly all exist and are wired to the mind. Nearly none compose into a rendered frame.** The pipeline's render
stages are explicit demo stand-ins (it shades a 24×24 `_demo_scene`; the source says so at lines ~30/120/133/241),
and the real render path is SDF path tracing with hand-coded materials and a flat sky. So the gap across the board
is not "does it exist" — it's **"does it compose into one render."** Legend: ✓ wired · ~ partial · ✗ not wired ·
demo = pipeline stage is a stand-in.

## H1. Render core — sampling / denoise / compositing

| Capability | Faculty / module (exists) | Pipeline | Render path | Gap |
|---|---|---|---|---|
| Path tracer | `path_trace`, `render_auto` | demo `_render_run` | ✓ (`render_auto`) | Make the pipeline render stage call `render_auto`, not the demo. |
| Primary G-buffer | `primary_gbuffer` | demo `_gbuffer_run` | ✓ | Real G-buffer instead of `_demo_scene`. |
| Adaptive sampling | `adaptive_sample_budget`, `adaptive_sample` | demo `_adaptive_run` | ✓ (via `render_auto`) | Unify the pipeline stage onto the real one. |
| Variance-guided SVGF | `svgf_denoise` | demo `_svgf_run` (legacy stand-in) | ✓ | Pass the real variance; drop the stand-in. |
| **Blue-noise / low-discrepancy sampling** | `blue_noise_sample`, `low_discrepancy_sample` | ✗ | **✗ — `path_trace` uses plain `default_rng`** | Wire the low-discrepancy samplers into the tracer (big quality/noise win, already built). |
| Firefly accumulate | `accumulate` (`robust_accumulate`) | ✗ | ~ (`declfirefly` is a re-do) | Use `robust_accumulate` in the tracer's accumulation. |
| Temporal reproject | `render_frame_delta`, reproject | demo | ✗ | For interactive/anim; not in stills path. |
| Gaussian splats | `splat_*` (fit/render/lod/densify/prune…) | demo `_splat_run` | ✗ | A whole splat render path exists, unused by the frame. |

## H2. Materials & physical properties

| Capability | Faculty / module | Render path | Gap |
|---|---|---|---|
| PBR material + adapter | `pbr_material`, `physical_material`, `surface_material`, `material` (lib), `matlib.shade` | ~ (identities/glass wired; rest hand-coded) | Route ALL scenes through `matlib.shade` (backlog B1). |
| **Physical-property → appearance** | `material_thermal` (density/specific-heat/conductivity), `material_elemental` (composition), `material_inclusions` (impurities), `crystal_material` (polycrystalline grains), `grain_material` | **✗** | These compute real physical/structural properties but **don't feed the shader**. Wire them as material inputs (e.g. thermal → emission/blackbody, inclusions/crystal → albedo sockets). Directly answers "physical properties affecting materials." |
| **Subsurface scattering** | `subsurface` (field-native translucency) | **✗ — not in the path tracer** | Wire SSS into the surface shading path (skin, wax, jade, leaves). |
| Textures / patterns / noise | `texture_map`, `synthesize_texture`, `pattern_field`, `procedural_noise`, `material_catalog` | **✗ (gallery uses patterns only in a 2-D chart)** | Let materials carry texture/pattern sockets the tracer samples. |
| Displacement / bump | `displace`, `auto_displace` | ✗ | Wire displacement into the SDF/shading path. |
| Thin-film iridescence | **no faculty found** | ✗ | **MISSING (BUILD)** — soap-bubble/oil-slick/beetle look. |

## H3. Lights & light transport

| Capability | Faculty / module | Render path | Gap |
|---|---|---|---|
| Environment / sun-sky / HDRI | `sky_dome`, `estimate_light_direction`, `sample_directional` | ~ (now used, but flat before) | Make sun-sky/HDRI the default env (D1). |
| **Light objects + NEE** | `light` (rasterizer only) | **✗ — path tracer has no lights/NEE** | Add light sampling / MIS to the tracer (D2/D3). |
| GI / irradiance cache | `irradiance_cache`, `read_irradiance`, `radiance_transfer`, `plan_render` (bake), `render_baked` | ✗ | `render_auto` uses none of the GI cache / PRT machinery. |
| PRT | `radiance_transfer`, `prt` | ✗ | Relighting path unused. |
| Radiance field | `holographic_radiance_field` | ✗ | Precomputed radiance, unused by the frame. |
| Caustics | `caustics`, `caustic_focus`, `detect_caustic` | ~ (bolt-on composite) | Integrate as a render pass (C3). |
| Lightning / emissive FX | `grow_lightning` | ✗ | Emissive FX not in the render path. |

## H4. Fur / hair

| Capability | Faculty / module | Render path | Gap |
|---|---|---|---|
| Groom + strand shading | `groom_hair`, `render_hair` | ✓ (gallery) | Working, but its own renderer (F3/F4). |
| Guide interpolation (density) | `interpolate_hair` (`interpolate_strands`, clumping) | **✗ — I brute-forced 24k strands** | Use guide+clump interpolation (F1). |
| Strand DYNAMICS | `simulate_hair` (PBD), `cosserat_strand` (twist), `hair_wind` (curl-noise) | ✗ | Rest groom only; dynamics/wind/twist unused. |

## H5. Volumes — smoke / fire / fog

| Capability | Faculty / module | Pipeline | Render path | Gap |
|---|---|---|---|---|
| Smoke / fire sim | `fluid_solver`, `fluid_step`, `smoke_step`, `smoke_preset`, `fire` | ✓ `sim_fluid` stage | ✗ | Sim runs in-pipeline; **the volume RENDER doesn't**. |
| Volume render | `render_volume`, `fractal_volume` | ✗ | **✗ (only tour/tests)** | Wire volumetric ray-march into the render path so smoke/fire actually appear in a frame. |
| Atmospheric fog | `holographic_fog_volume`, `volint` (closed-form integrals), `measure_volume` | ✗ | **✗** | No fog in the main render (my water reinvented Beer-Lambert instead of using `volint`). |

## H6. Particles / liquid / ocean

| Capability | Faculty / module | Pipeline | Render path | Gap |
|---|---|---|---|---|
| Particle system | `particle_system`, `particle_sim`, `advance_particles`, `emit_from_surface`, `collide_sdf` | ~ (`sim_collide`, `sim_granular`) | **✗ — no particle rendering** | Particles simulate but never render. Add a particle render (points/splats). |
| Ocean / waves | `spectral_ocean`, `spectral_wave`, `wave_field`, `wave_packets`, `solve_waves`, `plan_waves` | ~ (`sim_waves`) | **✗ — my gallery water shader reinvents sum-of-sines** | The water shader should drive its surface from `spectral_ocean`, and spectral ocean should be renderable. |
| Free surface / breaking | `break_wave` (`freesurface`), `nonnewtonian_fluid` | ~ | ✗ | Liquid sims not rendered. |
| Snow (MPM) | `snow_mpm`, `simulate_snow` | ✗ | ✗ | Exists, never rendered. |

## H7. Scene organization

| Capability | Faculty / module | Render path | Gap |
|---|---|---|---|
| **Scene graph / assembly** | `scene_graph`, `compose_scene`, `make_scene`, `new_scene`, `nested_scene_structure`, `scene_flatten`, `scene_delta`, `scene_to_recipe` | **✗ — gallery hand-builds ad-hoc SDF classes** | The renderer should consume a **scene graph**, not bespoke Python per scene. This is the biggest organization win: one scene model in, one frame out. |
| Scene DSL / description | `parse_scene_description`, `render_scene_description`, `scene_control_spec` | ✗ | A scene description exists but the real render doesn't use it. |
| `render_scene` faculty | `render_scene` | limited | Only paints (shape, colour) tags via `make_scene` — not materials/lights/volumes. |
| Sampler (read-probe) | `place_sampler`, `sampler` | ✗ | The placeable read-probe isn't in the render path. |
| Camera / session | `camera`, `camera_controller`, `render_session`, `incremental_renderer` | ✗ (gallery uses ad-hoc `_Cam`) | Use the real camera + progressive `render_session`. |

## H8. Post-processing / display

| Capability | Faculty / module | Pipeline | Render path | Gap |
|---|---|---|---|---|
| Tonemap / exposure / grade / bloom | `postfx_chain` (`aces`/`reinhard`/`exposure`/`gamma`/bloom/vignette/preset) | ✗ (only `present`) | **✗ — I built a duplicate `aces_tonemap`** | Delete the dup; add a `postfx` stage; tonemap via `postfx` (E1/E2). |

---

## What the sweep changes about the plan

The first half of this doc (A–G) was "fix the things I touched." This sweep (H) shows the deeper shape: **the
pipeline is an empty shell with stand-in stages, and every real capability is a faculty that never composes into
it.** So the highest-leverage single item is **making the pipeline real** — give it actual stages (scene → material
→ light → render(`render_auto`) → volume → hair → particles → postfx → present) that DELEGATE to the faculties
that already exist, with the scene-graph as the input. Almost everything else on this list ("wire X into the
render") becomes "add/enable X's stage" once that spine exists.

Suggested first cut, lowest-risk first: (1) pipeline render stage calls `render_auto`; (2) postfx stage via
`postfx` (kills the dup); (3) materials via `matlib.shade`; (4) sun-sky env default; (5) wire the low-discrepancy
sampler into `path_trace`. Those five are all "use what exists," no new capability, and they'd make the pipeline
produce a real frame. Then the bigger builds: scene-graph as render input, volume/particle/hair render stages,
subsurface + physical-property materials into the shader, and the genuine gaps (thin-film iridescence, NEE/lights,
a `holographic_water` module driven by `spectral_ocean`).

---

# I. Missing showcase demos — features with no gallery presence

*The gallery currently shows 6 renders (spheres, glass, fractal, identities, fur, ocean) + patterns/RD + data
charts. Diffed against the H-sweep, these engine features have NO visual showcase at all. Tagged: **ready** = can
be demoed today with existing machinery; **after-wire** = blocked on an H-item first.*

| # | Demo | Shows off | Machinery | Status |
|---|------|-----------|-----------|--------|
| I1 | **Smoke & fire** | Stable-fluids sim + volumetric render (the Houdini-class story) | `fluid_solver`/`smoke_preset` + `render_volume` (mode smoke/fire) | **ready** — both exist, just never composed into a gallery frame |
| I2 | **Fractal planet cross-section** | matlib's own crown jewel: biomes + interior layers + ore deposits in one slice | `fractal_planet().cross_section()` + `write_png` (all built, selftested) | **ready** |
| I3 | **Gaussian-splat scene** | The 3DGS story: fit → render → LOD chain (prune) side by side | `splat_fit`/`splat_render`/`splat_lod_chain` | **ready** |
| I4 | **Vegetated terrain** | fBm heightfield + grammar vegetation (the demoscene layer) | `terrain`, `vegetated_terrain`, `terrain_to_mesh` | **ready** |
| I5 | **Mesh rasterizer w/ lights** | The OTHER renderer (camera/lights/raster) — gallery only ever shows the path tracer | `render_mesh` + `light` objects | **ready** |
| I6 | **Physical-property materials** | crystal grains + impurity inclusions as colour sockets — "physical structure IS the texture" | `crystal_material`, `material_inclusions` (colour sockets f(points)→rgb) | **ready** — sockets can drive the tracer's albedo today |
| I7 | **Caustic map standalone** | The forward-traced caustic itself (striking as a pure image; currently only composited faintly) | `globalillum.caustics` | **ready** |
| I8 | **Cloth / softbody drape** | PBD cloth falling over a body, with SDF collision | `SoftBody` + `collide_sdf` + `render_mesh` | **ready** |
| I9 | **Snow (MPM)** | The Frozen-tech story — pile/footprint | `snow_mpm` + a render of the particle/height result | ~ready (needs a simple particle/height render) |
| I10 | **Animation filmstrip** | Anything moving: a deform loop, smoke over N frames, a breaking wave — as a strip of frames | `deform`/`fluid_step`/`break_wave` + existing renders | **ready** (still images in a row; no video needed) |
| I11 | **Scene-DSL → frame** | "One text description in, one rendered frame out" — the scene-organization story | `parse_scene_description` → scene → render | **after-wire** (H7: renderer must consume the scene graph) |
| I12 | **Subsurface scattering** | Jade/wax/skin translucency | `subsurface` | **after-wire** (H2: wire SSS into the shader) |
| I13 | **Fog / atmosphere** | Depth-cued scene in fog (volint closed-form) | `holographic_fog_volume`, `volint` | **after-wire** (H5: fog into the render path) |
| I14 | **Sculpt before/after** | FS-1 brushes: field edit → re-mesh | sculpt + `render_mesh` | ready |
| I15 | **Lightning** | `grow_lightning` emissive bolt over a night scene | after-wire (emissive FX in tracer) or composite | ~ready as composite |

**Read on this list:** I1–I8 are pure composition — the machinery exists and is tested, the gallery just never
called it. They'd roughly double the visual showcase without building anything new. I2 (planet) and I1 (smoke/fire)
are probably the two biggest "wow per line of code." The after-wire ones are exactly the demos that VALIDATE the
H-items — e.g. the scene-DSL demo is the acceptance test for making the pipeline consume a scene graph, and the
fog demo is the acceptance test for wiring volint in. Suggest pairing them: each H wiring item ships WITH its
showcase demo, so the wiring is proven by a picture, not just a unit test.

---

# J. Benchmarks, experiments, and other hiding spots

*Swept: `benchmark_holographic.py`, `benchmarks/`, `test_benchmarks.py`, `figures/`, `path_d/` (frontier program),
`tools/`, and the `app.py` browser UI.*

| # | Find | Detail | Action |
|---|------|--------|--------|
| J1 | **No render performance benchmarks exist** | The bench suite covers compression/recall only; the ONLY render measurement is the ad-hoc `[BENCH render_spheres]` print inside `make_gallery`. Nothing tracks path_trace samples/sec, `render_auto` convergence time, the hair rasterizer, fluid steps, or `render_volume` — so the H rewiring work has **no speed-regression guard**. | BUILD a `benchmarks/bench_render.py` (spp/sec, time-to-quality, strands/sec, sim steps/sec), wired into the same harness as the existing benches. Do this FIRST — it's the safety net for everything in H. |
| J2 | **17 orphaned figures in `figures/`** | Measured results with no home — `bench_clustering/conformal/resonator/streaming`, `holo_contrast/degradation/multiplex`, `stress_disappearance/predictive/separability`, `improve_decode/robust`, `ca_evolution`, `wht_capable`, `image_bench`, `exp_critique`, `test_image`. GALLERY.md references 12 of 29; the rest are invisible. | Triage: add the good ones to GALLERY.md (the stress_* family and holo_degradation are strong "graceful degradation" visuals), delete the stale, and confirm each generating script still runs. |
| J3 | **The whole path_d frontier program is showcased nowhere** | `path_d/figures/` holds ~10 real measured results — array scaling, capacity cliff, factor wall, distributed/superposed forward pass, RNS resolution — a complete "VSA compute at scale" story, visible only if you browse the subfolder. | Decide: a "Scaling & frontier experiments" GALLERY.md section (with the honest caveats those experiments carry), or an explicit pointer from the README/GALLERY to path_d. Not render, but it's the biggest pile of hidden measured work in the repo. |
| J4 | **Bench harness coverage unknown** | `benchmarks/` (compression, recall) + `benchmark_holographic.py` + `test_benchmarks.py` exist — but verify they run in CI and that their figure outputs are the ones GALLERY.md embeds (chart drift = silent lying docs). | Audit: CI runs them; gallery charts regenerate from them; add J1's render bench to the same harness. |
| J5 | **Demo scaffolding exists but the gallery ignores it** | `tools/demo_kit.py` (PNG response + smoke-check helpers "every gallery demo re-writes by hand" — its own words) and `tools/new_demo.py` (demo skeleton stamper). `make_gallery.py` hand-rolls exactly what demo_kit provides. | Fold the gallery onto demo_kit when promoting gallery logic (G1) — one demo convention, not two. |
| J6 | **Browser-UI demos may have the same disease the gallery had** | `app.py` exposes `scene_a`/`scene_b`, `boil_water`/`pour_water`. If they call raw `path_trace`/raw sims directly, they bypass `render_auto`/denoise the same way the old grainy gallery did. | Audit the UI demo endpoints; route their renders through the same pipeline front door once H lands. |

**Read:** J1 is the one genuinely new must-do — the H rewiring needs a speed guard before we start moving render
code around. J2/J3 are cheap honesty wins (measured work exists; nobody can see it). J5/J6 fold into the G1/H
organization work rather than standing alone.

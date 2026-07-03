# holostuff core audit — from the seat of someone who just built the garage on the old version

**Method.** Probe-first. I diffed the new zip (330 modules) against the version the demo gallery is built on
(237 modules) — 93 added, none removed, so it's a clean additive update. Then I read the actual docstrings and
public APIs (via AST) of the modules that touch the areas you named. Two honest limits up front: (1) this is a
static read plus one small chain smoke-test, not an end-to-end run of the new pipeline — the "it connects" claims
below come from the API contracts, which I'd want to confirm with a smoke test; (2) my image-view tool has been
blank this whole session, so I can't visually judge the *quality* of the new denoiser/upscaler output, only that
the pieces exist and are shaped right.

---

## Headline

Building the garage, I hand-rolled a camera controller, a reprojection "move", an edge anti-alias, an FFT
denoise, a two-level BVH, a floor-bounce/irradiance term, and an oriented-surfel splat builder — because none of
them existed as callable core pieces. **Almost every one of those now has a dedicated, self-tested module in this
update.** The update doesn't just add features; it retroactively turns my garage hacks into "should have called
the core." So my answer to "what should we add" is mostly **not "build" — it's "connect and demonstrate."** The
primitives are here; they aren't yet wired into the one place a demo-builder reaches for (`RenderSession`), and
they aren't shown off in a demo.

---

## Part 1 — the garage pain, mapped to what now exists

Every row is something I wrote by hand in `demos/08_garage/backend.py` because the old core lacked it:

| What I hand-rolled in the garage | Now in core | API |
| --- | --- | --- |
| Mouse orbit / zoom / pan camera math (frontend + `_camera`) | `holographic_camera.py` | `CameraController.orbit / pan / dolly / zoom / frame / to_camera` |
| The `_move` reprojection (warp cached frame, re-trace only disocclusions) | `holographic_temporal.py` | `TemporalReuse.solve(solve_fn, dirty, reproject, accumulate)` |
| `_edge_antialias` + FFT `denoise` for the preview | `holographic_svgf.py` | `atrous_bilateral(...)` — edge-aware, cosine edge-stop, coarse-to-fine |
| (never had it) render small + upscale | `holographic_fsr.py`, `holographic_superres.py` | `fsr_upscale` (EASU+RCAS), `guided_upsample` |
| `_floor_bounce` / the "irradiance" fill; bake-once diffuse | `holographic_matbake.py` | `bake_field`, `bake_material` (view-independent channels) |
| The specular that "drifts" during reprojection | `holographic_viewlut.py` | `ViewLUT`, `bake_view_lut` (view-dependent specular, baked) |
| The `_Scene` two-level bounding-sphere BVH | `holographic_spatial.py` | `SpatialGrid` — one shared grid for radius/knn/closest |
| My variance-harness "when is it converged" guesswork | `holographic_adaptive_sample.py` | `converged_mask`, `samples_to_target`, `sample_budget` |
| `_build_proxy_splats` oriented lit surfels | `holographic_splatexport` + `holographic_meshbridge` | anisotropic records; `marching_tetrahedra` for SDF↔mesh↔splat |
| Per-pixel G-buffer I extracted by hand for debugging | `holographic_renderchannels.py` | `render_channels`, `composites_to_beauty` (AOVs) |

If this update had landed before the garage, that demo would have been perhaps a third of the code, and I
wouldn't have spent this session chasing artifacts that were really "I re-implemented X worse than the core would."

---

## Part 2 — what to add (the gaps), by your categories

The recurring gap: these primitives are wired into `UnifiedMind` (the brain) and `holographic_pipeline.py`
assembles several of them, **but `RenderSession` — what a demo/app calls to render a scene — was not upgraded.**
Its public surface is still `preview()`, `render_final()`, `to_splats()`, `set_material()`, `edit_channel()`. No
`move()`, no denoise/upscale/temporal path. So a demo-builder still has to hand-wire the stack.

Good news that narrows the work: `Pipeline.run(scene, seed, prev_frame, renderer)` is explicitly designed to take
a `RenderSession` as its `renderer` and a `prev_frame` for temporal reuse (its docstring says so). And there are
ready presets: `PipelineConfig.preview() / .final() / .interactive() / .ocean()`. So the assembly exists — it just
isn't connected to the SDF render path or surfaced.

**Viewport interactivity (highest leverage).**
Add `RenderSession.move(camera_controller)` that runs `TemporalReuse` over the session's own render — reproject
the last frame, re-solve only the dirty region. Equivalently: wire `Pipeline.interactive()` with
`renderer=session`. Either way, every demo (and the modeling app) gets the smooth-orbit-with-reprojection I
hand-rolled, for free, and correctly (my version dropped the view-dependent specular; `ViewLUT` fixes that).
Pair it with `CameraController` on the front end so orbit/pan/dolly/zoom stop being bespoke per demo.

**Preview render speed.**
Three levers, all now present, none wired into `preview()`:
- `matbake.bake_material` — bake the view-independent lighting (diffuse + ambient + the floor bounce) **once**, reuse
  it across every camera move. This is the "irradiance map" intuition made real; it's exactly what would have made
  the garage settle instant instead of ~4 s.
- `fsr_upscale` — render at half resolution, upscale sharp. Roughly a 4× cut on the trace for the moving preview.
- `SpatialGrid` — replace the per-demo hand-rolled BVH so many-part SDF scenes stop being O(parts) in `eval`.
- `adaptive_sample.converged_mask` — for the progressive photo, stop sampling pixels that have converged instead of
  running every pixel to a fixed spp. (Note: `converged_mask` appears **unused anywhere** right now — it's the one
  siloed piece in this set.)

**Preview render quality.**
Route `preview()` through `svgf.atrous_bilateral` instead of a plain blur. It's edge-aware (won't smear across the
wall/ceiling seam the way my FFT denoise did) and is the standard way to make a 1-spp image look clean. This single
swap addresses the whole class of "the preview looks noisy / the denoise blurs detail" complaints.

**Mesh & LOD export.**
This chain now exists end to end and shares the `Mesh` type (`vertices, faces, normals, uvs, colours`):
`SDF → mesh` (`meshbridge.marching_tetrahedra`) → `build_lod_chain(mesh, [fractions])` (QEM decimation to
screen-space-error LODs) → `mesh_to_glb(mesh)` / `write_glb` (binary glTF, "the boundary between the NumPy back end
and three.js"). What's missing is a **one-call convenience and a demo**: e.g. `RenderSession.export_glb(lods=[...])`
that runs SDF→mesh→LOD chain→glb. The parts are all there; nobody has strung them into a "download this scene as a
.glb with three LODs" button.

---

## Part 3 — things worth demoing that are missing a piece (or just not built)

You asked specifically what we could show off that doesn't yet have all the pieces. Ranked by value:

**1. A real interactive viewport demo (CameraController + TemporalReuse + SVGF).** *Pieces exist; not assembled
into a demo.* This is the flagship "holostuff is interactive" showcase, and it's precisely what the garage strains
to fake by hand. Missing piece: the RenderSession/Pipeline connection from Part 2. Build that once and the demo is
mostly free. High value — it answers "can you actually move around in this thing" with yes.

**2. A mesh → LOD → glTF export demo (the "holostuff → Blender / three.js" bridge).** *All pieces exist
(`meshbridge`, `lod`, `gltf`); no demo emits a .glb.* Sculpt or field-edit a shape, hit export, download a real
`.glb` with a LOD chain, drop it into any glTF viewer. This is the single most convincing demonstration of the
"authoritative backend brain, browser is the muscle" thesis — the backend *computes* the geometry, the browser
just displays the exported mesh. Missing piece: only the wrapper + the download button. Very high value, low effort.

**3. photo-to-3D (`photo3d.photo_to_gaussians`).** *Works, but needs a depth map it can't produce.* There is
deliberately **no monocular depth estimator** (that needs learned weights, which the constitution bans — a correct
kept-negative). So a *single-photo-from-the-wild* demo can't be honest yet. But a **self-contained** demo is
available today: render a scene's depth + colour with the engine, feed both to `photo_to_gaussians`, and show the
abstaining reconstruction (it only reconstructs the observed front surface — which is itself a nice honesty story).
Medium value; be upfront that arbitrary-photo input is the missing (and intentionally-absent) piece.

**4. A physics set-piece.** The update adds a large simulation surface I have not seen demoed — `mpm` (material
point method), `gas`, `combustion`, `freesurface`, `nonnewtonian`, `tear`, and the hair/groom/cosserat stack. That's
a lot of "wow" sitting undemoed. One flashy sim (MPM sand collapsing, free-surface water, or a combustion plume)
would show breadth. No missing piece that I found — just not built into the gallery. Worth one demo.

---

## Part 4 — honest caveats and kept-negatives

- **I did not run the new pipeline end to end.** The connections in Part 2 are read from API contracts and
  docstrings, not executed. Before committing to the `RenderSession.move()` design, a 20-line smoke test
  (`Pipeline.interactive().run(scene, prev_frame=last, renderer=session)`) would confirm the shapes line up. My one
  attempted chain test failed only because I guessed the `Mesh` constructor args wrong, not because the chain is
  broken.
- **I can't see render output this session** (image tool returns blanks), so I can't vouch for how the new
  `svgf`/`fsr` results *look* — only that the modules exist, self-test, and have the right signatures.
- **The audit is scoped to your named areas.** I focused the deep read on rendering/preview/viewport/mesh-LOD and
  skimmed the other ~70 new modules (physics, query system, materials). There may be more show-off material in
  there than Part 3 captures.
- **`converged_mask` is the one genuinely siloed render primitive** — present, self-tested, imported by nobody.
  Cheap to wire into the progressive photo path.
- **Monocular depth stays a kept-negative** by design (no learned weights). photo3d is honest about only
  reconstructing what a single view observes; a demo should keep that framing rather than pretend to full 3D.

---

## The one-line version

The pain I hit building the garage is now almost entirely solved at the primitive level — camera, reprojection,
denoise, upscale, bake, spatial index, LOD, glTF all exist and are tested. The highest-leverage next move is to
**connect that stack to `RenderSession`** (a `move()` and a denoised/upscaled `preview()`), and then the two
"bridge" demos — an interactive viewport and a glTF/LOD export — write themselves and show off exactly the pieces
this update added.

---
name: lecore-render
description: "Render, preview and export 3D with the leCore engine: SDF/mesh scenes, materials, lights, the adaptive pipeline (converge-by-confidence, denoise, upscale, post), holographic bake-relight, sim output to frames, realtime previews and turntables, import/export, equations and datasets, GPU and caches, and the memory loop. Use when leCore should render, path trace, bake, relight, animate, turntable or export a 3D scene. Boot with lecore-boot; solvers live in lecore-simulate."
---

# Rendering with leCore

Everything here is the engine's own guidance (`RENDERING_GUIDE.md`, `docs/RENDER_PATH_AUDIT.md`,
`docs/SDF_COOKBOOK.md`, `CAPABILITIES.md`, `docs/NOTES_concepts.md`) plus two things measured
in sessions: five crystal renders and a render lab (`render_lab.py`: one cube on a plane, one sun,
every render path timed against a 256-spp reference). Numbers below say which they are. The
engine's rule applies to you: **`find_capability` before you hand-roll a render step** — the
holographic render lineage went eleven sweeps unused because its cards were missing, and the
renders that stopped using the substrate are the ones that took 30–100 minutes.

## 1. The mental model

**Surface or volume is the first decision.** A surface is an SDF and goes to a surface renderer;
a cloud, smoke, fog or fire is a density field and goes to `render_volume` (a cloud has no
surface). The semantic scene system tags each object and routes it.

**Bake the light-independent half once, relight by dot product.** The geometry of a pixel —
entry point, normal, refracted path, exit direction, Fresnel split — is deterministic; only the
light changes. `bake_scene → render_baked`, `radiance_transfer`, `bake_glass → relight_glass`.

**The adaptive pipeline is the default, not an option.** Converge each pixel to a confidence
target, denoise with the measured variance, spend extra samples only where the denoiser cannot
help, and stop. Fixed spp is for references and A/Bs. Section 4 is the how.

**Grade last, once, on linear HDR.** The post chain owns the tonemap.

**Two ladders, not one knob.** Iterate on the fast ladder until the picture is right; then run
the quality ladder once. Section 3 has both, measured.

## 2. Building a scene

```python
import lecore
from holographic.mesh_and_geometry.holographic_sdf import box, plane, sphere     # primitives are functions
from holographic.rendering.holographic_render import Camera, save_png, save_gif
from holographic.rendering.holographic_lights import make_light
from holographic.materials_and_texture import holographic_matlib as matlib
mind = lecore.UnifiedMind(dim=512, seed=0)          # or lecore.autoboot(partition=...) for memory + model end

cube  = box(0.35, 0.35, 0.35).translate((0, 0.35, 0))   # HALF-extents, three scalars; sits on y=0
floor = plane(0.0)
scene = cube.union(floor)                               # combinators are METHODS: union, smooth_union(k), subtract, rotate(axis, angle), scale, rounded
cam   = Camera(eye=(1.7, 1.2, 2.3), target=(0, 0.35, 0), fov_deg=40, aspect=4/3)   # eye/target IS the look-at
sun   = [make_light("sun", direction=(0.45, 0.8, 0.35), intensity=3.0)]
```

**There is no `sdf_box`, `op_union`, `look_at` or `marching_cubes`** in the repo. Isosurfacing is
marching tetrahedra (`sdf_to_mesh(sdf, resolution=48)`), primitives are module functions,
combinators are methods, `Camera(eye, target)` is the look-at.

**The light-direction trap (measured in the lab).** The path tracer's `DirectionalLight` /
`make_light("sun", direction=…)` wants the direction *to* the light (up, toward the sun). The
simple `Light` class used by `render_volume` and the rasteriser wants the direction light
*travels*. Point the sun down and it sits under the floor: the whole ladder renders sky-lit
and shadowless with no error. First render: check that the shadow exists.

**Materials for the path tracer** come from `matlib.trace_channels(name)` → the eight-channel
callback (albedo, metallic, roughness, emission, ior, sss, iridescence, absorption). Hand-assembling
that tuple is how a whole crystal arc ran with absorption silently unset. Per-object materials
need a switch on "which part is nearest" (`|part.eval(P)|` argmin) — a plain union does not
remember which child won. Several doors (`bake_scene`, `RenderSession`, material-ID renderers)
want a scene object with **`eval(P)` and `ids(P)`**: wrap the parts (`eval = min`, `ids = argmin`).
Presets (~85, hand-authored artist defaults, not measured spectra): `matte_white`, `matte_gray`,
`plastic_red`, `steel`, `gold`, `copper`, `chrome`, `marble`, `concrete`, `wood_oak`; glass and
gems (alpha < 1 transmits): `glass_clear`, `glass_frosted`, `ice`, `diamond`, `ruby`, `amethyst`,
`quartz`. `glass_optics(name)` → `n_d`, `abbe`, per-RGB `absorb`. Imported PBR texture sets:
`load_texture_set(folder)` (base/roughness/metallic/normal/height/ao by filename keyword).

**Lights**: `make_light(kind, target=, **kw)` for `sun | point | spot | area/softbox | dome`
(`target=` orients the panel/disk/spot for you); `sky=sky_model(hour=…)` on a sun syncs its
direction, colour and intensity to the sky. Environments as fields: `hdri_env(img)` (reads the
map for irradiance, dominant direction, sun share) then `env_add_light(env, direction, radiance=,
sigma=)` lobes; `env_field()` only has `add_sky(fn)` / `add_softbox(...)` — `env_add_light`
needs an `HdriEnv` (a small lat-long of `sky_dome` works as the base).

**Text → scene**: `build_scene(text)` → `scene.render(quality=…)`, `scene.adjust`,
`scene.feedback` (why a word did nothing); `render_scene_description(text, camera, w, h,
quality="fast"|"hyperreal")`. Trap: `mind.render_scene(tag_list)` draws attribute tags, not 3D.

## 3. The two ladders (lab numbers: 160×120, 2 cores, PSNR vs a 256-spp jittered reference)

**Fast ladder — iterate until the picture is right**

| Path | Time | PSNR | Use for |
|---|---|---|---|
| `render_sdf(sdf, cam, w, h)` | 0.20 s | — (own palette) | geometry, camera, silhouette; display range out |
| `RenderSession(scene, mats, cam).preview()` | 0.10 s (0.02 s at half) | — | material and lighting glance; edit a material, re-preview |
| `realtime_session(sess).frame(known_shift=)` | 0.036 s | — | a moving viewport; 20 % of pixels re-shaded, bit-identical on those |
| `path_trace` spp 4 / 8 / 16 | 0.9 / 1.8 / 3.6 s | 31.3 / 34.4 / 37.5 dB | a raw look at GI and noise |
| `path_trace` spp 8 + firefly + SVGF(variance) | 1.9 s | **40.8 dB** | the iteration frame: +6.4 dB for 0.14 s |
| `render_auto(quality="draft")` | 4.9 s | **44.9 dB** | the hands-off draft — beats raw spp 64 (13.4 s, 44.6 dB) |
| `path_trace_adaptive(tol=0.03, min_spp=16)` | 5.4 s | 35.5 dB | uneven scenes (notes: 48 % faster on a geode); no win on a flat cube |

**Quality ladder — run once, at the delivery size**

| Path | Time | PSNR | Notes |
|---|---|---|---|
| `render_auto("medium")` | 9.1 s | 45.9 dB | mean 36 spp, 74 % converged |
| `render_auto("high")` | 12.7 s | 46.2 dB | mean 48 spp; 320×240 took 45.9 s (4× pixels, 3.6× time) |
| `path_trace` spp 64 + SVGF | 15.1 s | 46.3 dB | the denoiser adds only +1.8 dB near convergence |
| post chain | 3–14 ms | — | free; do it once on the linear frame |

Read the ladders as: **`render_auto draft` is the iteration workhorse; `high` is the beauty
frame; raw spp is for references.** Halving the confidence target costs ~2× time for < 1 dB
here. Bounces: `max_bounce=1` is direct light only (21 dB, black shadows), `2` recovers most GI
(31.7 dB at spp 8). The notes' large-scene numbers agree: the tracer is dispatch-bound at
preview sizes (16× pixels for 2.8× time), so **passes and bounces are the lever, not pixels** —
`render_preview` (draft, 1 bounce) is 12× the full path at equal size.

## 4. The adaptive pipeline — use it to cut time and lift quality

Four adaptive mechanisms exist; they compose.

- **Converge-by-confidence**: `render_auto(scene, cam, w, h, material, quality=, lights=,
  return_stats=True)` — `quality` is a CI half-width (draft 0.08 / medium 0.04 / high 0.022 /
  ultra 0.012), passes of `pass_spp` until each pixel's 95 % CI is inside it, then
  `declfirefly(k)` → SVGF steered by the measured per-pixel variance. Stats: `passes,
  mean_samples, converged_frac, median_ci, seconds`. The stop rule lives in
  `holographic_adaptive_sample` (verified bit-identical on 100 k variances).
- **Per-block adaptive spp**: `path_trace_adaptive(sdf, cam, w, h, tol, block=8, min_spp=16,
  max_spp, tol_map=)` → `(image, {spp, saving, why})`. **`min_spp` is not a quality floor, it is
  what makes the CI trustworthy**: at 8 the CI declared convergence everywhere. Feed
  `tol_map = denoiser_relaxation(depth, normal, relax=6)` so the sampler relaxes where the
  denoiser will work (grain 0.080 → 0.032; 83 % of a flat 48-spp render's samples avoided).
- **Adaptive AA**: `render_scene(adaptive=True)` supersamples edges only (< 0.6× the rays of
  brute SSAA at |diff| < 0.002); `path_trace(antialias=True)` jitters low-discrepancy sub-pixel
  positions — **spp never removes jaggies, that is a sampling-position problem**. For bakes,
  re-bake only silhouette ±1 px and facet edges through a pixel-list camera (crystal v5: edge
  specks 3.0 % → 0.5 %, ~4× the main bake per ray).
- **Method dispatch**: `plan_render(objects, frames, relight)` / `render_adaptive(objects, …)`
  chooses grid-bake vs analytic (bake wins 6.4× at 64 primitives, loses 0.3–0.7× at 5 objects
  and one frame) and collapse (PRT) vs trace per surface. Name collision: `render_adaptive`
  also names the per-pixel one in `holographic_pathtrace`.
- **Budgets**: `render_plan(budget_s)` times a probe tile and shrinks (w, h, max_spp) to fit;
  `path_trace(should_stop=, on_progress=, progress_every=)` for a refine stream;
  `frame_budget_ms(target_fps)` for a viewport ladder with hysteresis.

## 5. Denoise, upscale, post — when, and in which order

**Denoise low spp, with the variance map, before anything else.** Lab: spp 8 raw 34.4 dB →
firefly then SVGF 40.8 dB for 0.14 s; **without `variance=` at `levels=5` it scored 29.2 dB,
below raw** — always `path_trace(return_variance=True)` and pass it. Firefly-before-SVGF and
SVGF-before-firefly measured identical on a matte scene; the two shipped pipelines disagree
(`render_auto` clamps first, `render_specimen` denoises first) — clamp first, so a firefly is
never smeared into a blotch. Near convergence the denoiser only softens (+1.8 dB at spp 64; the
notes: −1 dB at `high` vs an equal-budget raw trace) — **denoise the draft, consider skipping it
on the beauty**. Demodulate albedo (`denoise_demodulated`) only on textured diffuse (33 % less
error on texture, neutral on flat albedo, wrong on metal — "the irradiance IS the reflection").
A G-buffer costs 0.09 s (`primary_gbuffer(scene, cam, w, h, material)` → normal, albedo, depth;
`mind.render_gbuffer` returns depth, normal, albedo — different order).

**Upscale is not a speed lever for quality work.** Lab: native spp 8 + SVGF 1.95 s at 40.8 dB;
half-res + SVGF + `guided_upsample` 0.6–0.7 s at 27.3 dB (`levels=4`, the default) or 33.3 dB
(`levels=1`); plain `upscale` (FSR) 33.3 dB. On a box-downsampled copy of the reference itself,
`guided_upsample` scored 27.4 / 33.1 / 34.6 dB at levels 4 / 2 / 1 against FSR 35.9 and
nearest-neighbour 32.6 — on a flat-albedo scene the guide says "same surface" everywhere and
the bilateral flattens the shading. The notes' three-way on a copper still life agrees: native
64.8 s cleanest; masked native 33.1 s; upscale-from-half 15.8 s visibly softer. Rule: **render
native at delivery size; `out_res`/half-res is the quick-grid tool.** When you do upsample:
denoise at low res first (`levels=2`), demodulate by the *area-downsampled* high albedo (point
sampling lost −19 %), guided-upsample with depth+albedo guides, remodulate, box-down in linear,
then display, then FXAA — and `upscale(image, scale, sharpness)` (FSR EASU+RCAS) as the very
last step when a bigger delivery is wanted. `guided_upsample` clips to [0,1]: feed it display range.

**Post, once, on linear HDR.** `PostChain().then("exposure", ev=).then("bloom", …).then("aces")
.then("vignette").then("gamma")`; presets `display_chain()` (auto-exposure → ACES → gamma) and
`default_chain()` (+ bloom, chromatic aberration, vignette, film grain). Feeding a graded frame
applies ACES twice and washes it out. After a denoise **drop `film_grain`** (it puts the grain
back). ACES is auto-exposed and scale-invariant — its control is `key`, not exposure; lower the
key for a dark plate. Hold `exposure(ev=)` fixed only for a lighting A/B (the meter hides
lighting changes). `fuse=True` is a no-op on the shipped chains. `save_png(path, rgb01)`;
`save_gif(path, frames, fps)` is 8-bit with a **fixed 6×7×6 palette and no dither** — measured:
grey (32,36,40) → (51,42,51) mauve, a grey floor → blue, copper → yellow. It is for "is it
moving", never for colour: write clips with an adaptive palette (Pillow median cut +
Floyd–Steinberg, one palette per clip) or an MP4 via `imageio_ffmpeg` with frames padded to
16-px macroblocks (`render_lab.write_gif_adaptive` / `write_mp4`); GIF delays are 1/100 s, so
"60 fps" is written as 50. Tone-map before writing or an HDR buffer just clips.
`render_auto`, `path_trace`, `relight_glass` return **linear HDR**; `render_sdf`, `render_scene`,
`preview` return **display range** — never mix them in one frame list without one view transform.

## 6. Physics and simulation → frames

The solvers, materials, fields, noise, growth and procedural geometry have their own skill,
**lecore-simulate** — read it for anything that moves, burns, grows or is generated. What a
renderer needs to know: a sim hands you a density grid (wrap it as a callable for
`render_volume(field, cam, bounds, mode="smoke"|"fire"|"density")`, 0.03 s/frame at 160×120 for
a 32³ grid), a point cloud (`splat_points(P, cam, w, h, colors, radius_px)` → `(image, alpha)`,
5 ms/frame for 600 points), or a body pose (rebuild the SDF per frame and render it with
`render_sdf` / `RenderSession.preview()`, 100–300 ms/frame). Step many times per rendered frame —
the sim is 100–1000× cheaper than the picture. Two contact bugs found while building the lab
(a rigid cube falling through another, a cloth sliding off a cube) and their fixes are in
lecore-simulate §3; the working recipes are `rigid_contact.py` and the patched
`SoftBody.step`. Composite volumes over the stage by alpha, keep one view transform per clip.

## 7. The holographic VSA path

`docs/RENDER_PATH_AUDIT.md`: render by *projecting from a superposition* — the scene is
hypervectors, an image is a readout, and staying in the Fourier domain is the win. Measured:

- `bake_scene(scene, cam, w, h, methods={id: "collapse"|"trace"}, colors)` → `render_baked
  (baked, env_fn)` — PRT; the light is a **function of directions `w → (n,3)`**, not a vector.
  Lab: bake 12.4 s at 160×120, each relight 2 ms. Notes: 15× vs all-trace, break-even ~160 relights.
- `radiance_transfer`: 3.8 s precompute, 0.4 ms per relight (57×).
- `bake_glass(sdf, cam, w, h, n_d, abbe, samples=, lam_jitter=, floor_y=)` → `relight_glass
  (bake, env, floor_albedo=, sdf=, sun_dir=, material=, flaw_volume=, absorb_volume=,
  flaw_ambient_occlusion=, reflection_footprint=)` — lights bundled as one hypervector over
  directions (`EnvField` / `HdriEnv`). Lab: bake 0.19 s, relight 0.05 s per light for a clear
  cube; notes: 1.6 s + 7 s per light for a Mandelbox vs ~5 min at 192 spp still grainy. TIR loss
  past `max_internal=4` is 3 % (65.5 % before the interior-march fix).
- `holographic_fog_volume`: density as one FPE vector, closed-form ray integral — exact vs a
  160-step march (corr 1.0000), ~90× at image scale; absorption only, emissive media still march.
- Caustics: `holographic_holocaustic.spectral_landings` → `caustic_pass(rgb, cam, w, h, plane_y, window, center,
  occluder_sdf=)` → `composite_caustic(beauty, px, strength=, receiver_mask=)` — RGB is three
  unbinds. Capacity-bounded (~√dim cells per axis); the histogram (`spectral_caustics`, 2.3 s)
  won on speed. Brute-force path tracing is the wrong algorithm for a caustic (no NEE: a
  brighter key made grain worse, 0.76 → 1.19). Lab: a path-traced clear cube casts a black
  shadow at spp 32 — glass goes through the bake.
- `HolographicRadianceField` ("render = query"), `fpefield` (edit = bind; delta edits
  model-size-independent), semantic scene as one bundled vector (12/12 attributes decode).

Choose it for relighting over fixed geometry, absorption-only media, free-viewpoint reads after
a bake, composability (bundle = add, bind = move). Not for a single still (histogram/MC wins),
sharp spectral colour at low dim (λ crosstalk floor 0.75 at 2048 against a ~3 % physical
effect), or emissive self-shadowing media. MCP doors: `scene_create`, `scene_adjust`,
`scene_export` (STL only), `image_tool`, `math_eval`, `chart_make` — scenes live in process
memory, not on disk.

## 8. Realtime preview, animation, video

- **Viewport**: `RenderSession(scene, {id: SurfaceMaterial.from_name(...)}, cam, w, h)` —
  `.preview()` (lab 0.10 s), `.set_material()`, `.render_final(spp, on_progress=)` (5.7 s at
  spp 32), `.to_splats()`; `mind.realtime_session(sess, budget=0.2)` — `.frame(known_shift=)`
  re-shades only the news (20 % mask, 3.2× faster, bit-identical on touched pixels; pass the
  shift, recovering it costs 2,280 traces and −4.5 dB), `.refine()`, `.payload(kinds)`.
  HTTP: `POST /frame`, SSE `GET /frame/stream`, `frame_server().next_frame(session, target_fps)`.
  Feedback zoom: 9.8 ms/frame at ⅛ band vs 101 ms full recompute (60 fps).
- **Offline animation**: `render_animation` (keyframes, GIF), `Timeline`/`Transport`/
  `FrameCache` (O(1) scrub-back), `render_progressive(door, args, n_buckets, spp, workers)` —
  byte-identical across pauses and processes, 1.76× on 2 workers, **each bucket needs its own
  seed** (`sample_buckets`), a material *callback* cannot be checkpointed (describe the scene as a
  document). Long frame runs go through the job system, not one HTTP request.
- **Turntable via three.js (lab)**: `sdf_to_mesh(sdf, resolution=48)` (0.14 s, 6 k verts) →
  set `mesh.normals = sdf_normal(sdf, V)` (**tet-mesh averaged normals speckle** — the first
  turntable looked like static) → `mesh_to_glb(mesh, material=matlib.material(name))` (4 ms,
  215 KB, validates) → an HTML page embedding the arrays with the repo's inlined three.js
  (`pages/vendor/three.inline.js`, r185, no GLTFLoader/OrbitControls/RoomEnvironment — build a
  6-canvas gradient `CubeTexture` for `scene.environment` or metals render black) → Playwright
  Chromium (`--use-gl=angle --use-angle=swiftshader`) `renderFrame(i)` + screenshot: 68 ms/frame
  at 320×240 including the screenshot round trip (launch 1.2 s), i.e. a 60-frame turntable in
  ~5 s against ~300 ms per `render_sdf` frame and 13 s per converged frame. Write with an
  adaptive-palette GIF or MP4 (section 5) or stills — the first turntable went out through
  `save_gif` and came back mauve, blue and yellow. Nothing in the repo does this for you: the engine refuses to own
  video decoding/encoding (`FrameSource` asserts no cv2/ffmpeg/imageio import; `save_gif` says an
  MP4 is a licensing question). **MP4/WebM**: leStudio's `POST /api/render/animation`
  (`format="mp4"`, imageio-ffmpeg, ≤1280×720, sizes snapped to 16 px) or your own
  `imageio_ffmpeg` in the sandbox. OBS browser-source capture profiles exist for the live path.

## 9. Import and export

| Direction | What | Door | Notes |
|---|---|---|---|
| in | OBJ (+MTL), GLB/glTF (nodes, skins, morphs, animations, images) | `load_obj`, `load_glb`, `import_asset`, `preview_asset`, `pose_asset(lm, time, clip)` | no STL/PLY/USD/FBX/OFF readers; PLY only for splats |
| in | texture sets | `load_texture_set(folder)` | `.png/.jpg/.tga/.bmp/.tif/.exr`; Substance `.spp/.sbsar` not parsed |
| in | HDRI | `load_hdr`, `load_exr` (read-only) | cached maps ship as `.npy` |
| in | raw volume | `load_volume(path, dims, dtype, bounds)` → `GridField` | |
| mesh ↔ field | `sdf_to_mesh(sdf, bounds, resolution, level)` (tetrahedra); `mesh_to_sdf_grid(mesh, bounds, res, sign="auto")` | `winding_flood` is not a drop-in for `winding`; bake the FULL mesh, never decimate to fit | winding 113× via cluster dipoles; band 8–18 % of voxels |
| out | GLB with one material (+ `baseColorTexture` if UVs) | `mesh_to_glb` / `mind.mesh_to_gltf(mesh, material=)`; deterministic bytes | emits `KHR_materials_ior/transmission/volume` for glass; **drops abbe, per-RGB absorb, sss, iridescence**; `from_gltf_dict` drops ior/transmission back in (asymmetric) |
| out | ASCII STL, DXF (2D) | `mesh_to_stl`, `polylines_to_dxf`; MCP `scene_export` = STL only | |
| out | splats | `export_splats(fmt="ply"\|"json")` | |
| out | images | `save_png` (8-bit), `save_gif`; no PNG-16/TIFF/EXR writers | keep the linear `.npy` yourself |
| materials | `PBRMaterial.to_gltf_dict / from_gltf_dict / to_mtl / materials_from_mtl / to_vsa_record` | | |
| scenes | `.lews` (`holographic_lews`, spec 1.0): zip of typed sections `lecore.image/mesh/material/sdf/camera/scene/asset/journal`; `Workspace(root, app)` multi-app, locked, journaled, `expected_rev` conflicts | the only on-disk scene format; `Scene`/`SceneObject` documents have no save/load; MCP scene handles die with the process | 200 MB workspace rewrites 200 MB per put |

Texture baking: `bake_texture(graph, res)`, `bake_material`, `bake_normal_map(low, low_uv, high,
size, ao=)`, `transfer_uv`, `make_uv_shell`, `auto_retopo`; `mesh_lscm` needs a seam cut first.

## 10. Where to get models, textures, HDRIs

All CC0 unless noted; send a unique `User-Agent`. **The Anthropic sandbox's egress blocked all
of these (HTTP 000) when probed; on the user's machine they work.** Cache downloads beside the
project and key them by asset id.

- **Poly Haven** — `https://api.polyhaven.com/assets?t=hdris|textures|models`, `/files/{id}`
  (per resolution 1k–16k: HDRI `hdr`/`exr`; textures png/jpg per map; models gltf/blend/fbx, with
  size and md5), `/info/{id}`, `/categories/{type}`, `/types`. Credit "Powered by Poly Haven" if
  you ship an integration. Swagger at `api.polyhaven.com/api-docs/swagger.json`.
- **ambientCG** — v3 `https://ambientcg.com/api/v3/assets|categories|collections|rss`; v2
  `api/v2/full_json?q=&type=&limit=` still serves `downloadLink`/`rawLink` per `1K-JPG`, `2K-PNG`,
  HDRI, model formats. Docs: `docs.ambientcg.com`.
- **Sketchfab** — `https://api.sketchfab.com/v3/models?downloadable=true&q=` lists; download
  needs an OAuth token; licenses vary per model (many CC-BY, not CC0).
- **Smithsonian Open Access** (3D scans, CC0) — `https://api.si.edu/openaccess/api/v1.0/search`
  with a free api.data.gov key.
- **Kenney** (CC0 game asset packs, zip downloads, no API) and the `awesome-cc0` list
  (github.com/madjin/awesome-cc0) for more.

Import with `load_glb`/`load_obj` + `load_texture_set`; a Sketchfab scan with 71 % boundary
edges shredded under `sign="flood"` — use `auto`/`winding`.

## 11. Equations and datasets; Lean 4

- **Equation → surface**: SymPy expression → kernel → raymarch. `compile_field(expr, ("x","y","z"))`
  → value and gradient closures; plugins `[symbolic]` (`compiled_sdf_normal`, `exact_sdf_normal`)
  and `[jit]` (`compiled_sdf_numba`, `render_sdf_fast(expr, cam)`, 9–15× the NumPy renderer;
  compile 140–390 ms once). Any callable `f(P) → distance` renders through `render_sdf`; an SDF
  node has `.to_jit_expr()` (refuses fractals/ellipsoid/twist/bend). Height fields:
  `terrain(...)`, `terrain_to_mesh/sdf`, `displace_from_height`. Isosurfaces of sampled
  functions: `sample_field(func, bounds, res)` → `surface_nets` / `marching_tetrahedra`.
- **Datasets**: 2-D series → `chart_svg(kind=line|bar|scatter, …)` / MCP `chart_make`
  (Okabe-Ito palette, non-finite refused; no heatmap kind; matplotlib is import-blocked in the
  unified mind); points → `scatter_to_field_3d` / `splat_render` / `field_to_splats`; arrays →
  `render_volume(field, cam, bounds, mode=)` is "volume from array". Analysis (not rendering), MCP tools:
  `series_analyze(tasks=demux|regimes|forecast|formula|drift)`, `dataset_decompose`.
  **No LaTeX/MathJax/KaTeX anywhere** — typeset equations in leStudio or a browser page.
- **Lean 4** (optional): `python3 tools/install_lean.py [--status|--remove]` — pinned 4.15.0,
  sha256-checked, ~265 MB → ~1.3 GB under `$LECORE_LEAN_PREFIX` (default `~/.lecore/lean4`), prints
  the PATH line. Doors `lean_status`, `lean_verify(source)`, `lean_export(goal, rules, check=)`,
  `lean_fuzz`; `lean_check` demotes `ok` on `declaration uses 'sorry'` (lean exits 0 on sorry).
  It buys the `lean_verified` provenance tier for `proof_store` / `proof_recall(min_provenance=)`
  (300 theories, 793 exports, 0 failures). Render-adjacent: `tet_certificate_lean(mesh, …)` emits
  a Lean proof of tet-mesh connectivity; `lean/LeCoreHeadSpec.lean` mirrors `holographic_headspec.check_invariants`.
  **No door turns a Lean verdict into a picture; mathlib is unreachable (no lakefile).**

## 12. Acceleration and caches — say only what is measured

- **Per-call overhead first.** A scene field with a Python loop over N primitives costs ~N numpy
  calls per evaluation regardless of point count, and AO/shadow traces call it 64 steps × rays ×
  sub-bakes. Vectorise small batches (points × primitives distance matrix, bit-identical, 402 s
  → 115 s on the crystal edge pass) before reaching for a backend.
- **Grid bake**: `bake_sdf(sdf, lo, hi, 256)` — 4× on a crystal bake at 1.2 % mean difference,
  9.6× per frame on a geode; not bit-accurate (quantised normals, ~0.2 radiance error that does
  not converge with resolution); 256³ is the ceiling on a 5–8 GB box; chunk the field at ~400 k
  points; key the cache by a hash of field values on a probe grid, not by parameters (a stale
  grid once rendered old geometry silently).
- **Caches to reuse across iterations**: HDRI prefilters (`HdriEnv._pre_cache`, buckets
  0.4/0.15/0.05, in-memory), the AO cache shared across `MultiBake` subs, the dome cache (`render_dome_term` / `cached_dome_shade`,
  15–16× vs per-pixel, ~96 % hit), the G-buffer (compute once before the trace: `tol_map`, SVGF guide,
  upsample guide), `SceneRenderer` dirty-region frames (`set_attr` re-renders one object's
  pixels), `CompileCache` (sha256 of source, LRU 128), Zig/C kernels on disk
  (`LECORE_ZIG_CACHE`, `LECORE_CC_CACHE`), the realtime session's reprojection. `LightCache`
  paints false shadows on curved mirrors — off for those.
- **Backends** (extras in `setup.py`; `requirements-accel.txt` says plain `@njit` only, never
  `parallel=`/`fastmath=`): `[jit]` numba — eikonal SDF ~270× (2-D), `render_sdf_fast` ~15× for
  field-native shading only (PBR/reflect/refract stay NumPy); `[zig]` — batch kernels 2–5×,
  raymarch 3.8× (f64 safe mode byte-identical, f32 is not; 1–2 s first compile); `[wgsl]` wgpu —
  elementwise f32 maps only, correctness bit-exact on a software adapter, **`gpu_crossover` on a
  software adapter is flagged MEANINGLESS and publishes `crossover: never`; no real-GPU number in
  the repo**; `[gpu]` CuPy (`HOLOSTUFF_GPU=1`, `mind.use_gpu(True)`, `cupy-cuda12x`) — fluid
  solver, FFT post-FX chain, LLM forward; **no GPU raymarch, path trace, denoise or upscale;
  matches NumPy to a tolerance, not bit-identical; speed unmeasured in-tree**; GLSL/WebGL2 —
  verified kernel sources (`glsl_kernel(name)`), fields win 51–307× on an RTX A4500, retrieval
  loses, no timing for the render kernels. Determinism is a CPU property: keep references,
  A/Bs and anything tie-sensitive on CPU; put fluid steps, FFT post and the LLM on the GPU when
  one exists, and **measure the crossover on that machine before claiming it**. In this sandbox
  numba is not installable (pip finds no version for 3.11 through the proxy) and there is no GPU.

## 13. Learn as you render — the memory loop

The external memory is where render knowledge compounds. `render_lab_teach.py` is the worked
example; the loop:

- **Before**: `mind.ask(query)` → `{tier, confidence, why, provenance}` (T0 memory … T4;
  `refused` is a result); `find_capability(problem, k=3)` and `suggest_pipeline(start_kind,
  goal_kind)`; `wisdom(query=, k=8)` for kept lessons and negatives; `image_recall(query)` for
  prior renders; `recall_localise(query)` says which token is NOVEL.
- **During**: `mind.teach(question, answer)` — two positional strings, nothing else; provenance
  is set to `taught`. **Key the question the way a later session would type it** ("when to
  denoise a path traced render and in what order") and **put the numbers and the baseline in the
  answer** ("spp 8 raw 34.4 dB → firefly+SVGF 40.8 dB in 0.14 s; without variance 29.2 dB").
  `teach_about(question, answer, paths)` fingerprints files so `stale_facts()` can flag them;
  `image_remember(image, label, source="render")` stores a content-addressed render;
  `tool_reflex_teach(pattern, service, endpoint, params)` turns a recipe into a T-tier reflex.
  Negatives are lessons: `bequeath(lesson, author=, topic=)` → provenance `wisdom:<author>`.
- **After**: `answer_feedback(query, ok=False)` tombstones a wrong row across restarts;
  `hypothesis_propose → hypothesis_test(alpha) → conjecture_promote(level="validated"|
  "evidenced")` upgrades a conjecture that passed a measurement; `escalations()` / `resolve
  (question, answer, by="human")` closes what the swarm could not; then
  **`learning_save(root, path=<root>/learning/state-<UTC stamp>Z.lecore)`** — `learning_save(root)`
  alone writes the legacy `state.lecore`, which the rollover ranks *oldest* (lab: ten taught
  rows vanished that way once). Boot next session with `boot(partition, doctrine=True)`; it
  rolls the generation. MCP: `zoo_ask`, `zoo_teach`, `zoo_feedback`, `memory_write(text, tags)`,
  `memory_search`. Lab: 476 → 488 rows, re-ask returned T0 with the numbers after a reload.
  Recall quality is measured for sensor questions (2.2× runner-up margin) — **not for render
  recipes; that benchmark does not exist yet.**

## 14. Budgets and the sandbox

- The memory cgroup kills at ~5.1 GB anon-rss on a box that reports 8 GB, with no traceback
  (`dmesg | grep Killed`). Big bakes: render in row buckets (a StripCamera whose `ray_dirs`
  returns rows of the full frame, `np.vstack` the strips — bit-identical) and chunk `bake_sdf`.
- One Bash call dies at 10 min: launch renders detached (`nohup setsid … < /dev/null &`), one
  process per stage with its own log, poll the log; A/B batches in a shell script.
  `pkill -f name` from inside a tool shell matches itself — kill by scanning `/proc` or use a
  `[b]racket` pattern.
- Iteration sizes from the crystal arc: 320×200 at 1 bake sample ≈ 100 s, 480×300 ≈ 4 min,
  1280×800 at 3 samples ≈ 60–70 min plus edge and denoise passes. Lab cube: everything under 15 s
  at 160×120, 46 s for a converged 320×240. Preview small; look at the PNG.

## 15. Decompose before you tune

When a frame looks wrong, read the per-pixel terms for the offending pixels before touching a
material: Fresnel R, path length, exit target (sky / floor / object), env radiance at the exit
and reflection directions, lost wavelengths (`bake.exit_ok`), the G-buffer, the variance map.
Three wrong guesses in one crystal session — absorption, in-scatter, geometry — each cost a
preview; one probe found a studio key lobe 27× the sky. Give every probe a fixed region and a
metric so runs compare (`holocaustic.caustic_concentration`, `spectral_detail`, `highlight_fraction`, a
silhouette-speck count, |lum − 3×3 median| > 0.2 over glass pixels, PSNR against a fixed
reference). Add env knobs that default to the old behaviour instead of editing constants.
Probe any procedural geometry before rendering it: march random directions, cluster the normals,
compare facet counts and angles to the target (the hexagonal form expansion shipped four
renders of rhombic blades before that probe caught it).

## 16. Worked notes: glass and crystals (from five renders)

Bake with anti-aliasing (`samples=3..8, lam_jitter=True`) — one ray per pixel is a stair-stepped
silhouette that no post undoes; then re-bake silhouette and facet-edge pixels alone at 8–12
samples. Size lights by irradiance relative to `env.floor_irradiance × sun_share`; under an HDRI
sun, daylight-size the lobes (~0.12× the map, sigma ~0.35) — a studio key at sigma 0.22 turned
every facet white. Light glass with contrast: a bright uniform sky kills dispersion and caustics.
`flaw_ambient_occlusion=8` lights milky in-scatter by the occluded hemisphere instead of open sky
(the source of chalky bases); `reflection_footprint=1.0` reads the environment over the pixel's
fan of reflection directions with a prefiltered map. `sky_dome` reads a map nearest-neighbour —
re-read the backdrop bilinearly from the 2k map (u = atan2(x,−z)/2π + 0.5, v = 0.5 − asin(y)/π).
Denoise composites: only on glass and floor pixels, gently on bright refractive glass (levels=4
erased interior facets; half of a levels=1 pass kept them). Sit objects in the ground (bury a
rock ~25 %, add an apron). Crystals: `crystal_grow_on(substrate, bounds, habit=, size=, seeds=,
real_cell=True, clip_to_substrate=True)`; `quartz_rz` is class-32 quartz (6 prism faces at 60°,
m^r 141°47′).

## 17. Kept negatives (do not re-litigate)

`anchors=` does not pay in the path-traced view path. `spectrum_to_rgb` is for emission.
`caustic_pass` is plane-only. `PostChain(fuse=True)` is a no-op on the shipped chains. Field
smoothing does not help a corrected mesh bake. `winding_flood` is not a drop-in for `winding`.
`svgf_denoise` cannot add detail; `upscale` cannot invent it; `guided_upsample` at its default
`levels=4` scored below nearest-neighbour on a flat scene. SVGF without a variance map is worse
than no denoise. `max_bounce=1` is direct light only. A sun pointed "down" is a sun under the
floor. Grain after a denoise, a bright uniform sky, per-pixel spectrum fits (twenty minutes per
1080p frame), studio-strength lobes under an HDRI sun, decimating a mesh to fit a bake,
`learning_save` without a stamped path, `save_gif` for anything judged by colour, and a GPU
speedup quoted from a software adapter are the recurring mistakes.

## 18. Deliver like the engine reports

Save the linear frame (`.npy`), the masks and the G-buffer beside the PNG — every later stage
needs them. Report bake, relight, converge, denoise, edge and post seconds, `mean_samples` and
`converged_frac`, the auto-EV or `key`, clip and crush fractions, and each quality metric next
to the previous version's and the reference's. Say what did not work, with its number. Teach the
recipe and the negatives before the session ends (section 13). A finish in leStudio does only
what the chain cannot, at 1×, with the engine's upscale last.

## Long frames in a sandbox that reaps jobs; the shader that matches the render (Sep 2026, measured)

- **Render in kill-safe row strips.** A 30-minute whole-frame `render_auto` died at a turn boundary with
  nothing written. `StripCamera(Camera)` overrides `ray_dirs(w, h, jitter)` to return only rows [r0, r1)
  of the FULL frame's grid (same eye, fov, aspect — the trace is the whole frame's trace); render each
  strip with 4 rows of overlap so the per-strip denoiser has no seam, save the LINEAR strip as `.npy` the
  moment it finishes, let a rerun skip strips on disk, `np.vstack`, then run `display_chain().apply`
  ONCE on the stacked frame (per-strip auto-exposure would differ). Measured: 800×500 draft in 10 strips,
  66–113 s each, mean 15–18 spp, every strip ≥99.8% converged, 15 min, no visible seams; medium was
  247 s per strip on one core. `speaker_render_strips.py` is the worked example.
- A render script without a `__main__` guard renders on import (three minutes gone when another script
  imported its scene). Guard it.
- **The viewport can be the render's geometry.** `mind.sdf_scene_shader(parts, camera)` compiles a whole
  multi-part analytic scene to one WebGL2 fragment shader — each part's `map()` from the tree's own
  `to_glsl()`, combined by min into `mapAll(p, out id)`, rays from the SAME basis `Camera.ray_dirs` uses —
  and returns the uniforms to bind. Measured on the 15-part speaker: silhouette IoU 0.9852 against the
  engine's own sphere-trace, part-id agreement 0.9924, every disagreeing pixel on an edge, 2 s per frame
  in Playwright Chromium on SwiftShader vs minutes for the path trace. `tools/scene_shader_iou.py`
  reproduces it. `sdf_validate_glsl` now compiles the rotated cylinder / capsule primitives too (vec2 and
  swizzles were added to the g++ shim); worst diff 2.9e-7, the float32 tolerance the docstring states.
- The app's `/api/photo` path bakes meshes onto a grid (244 s for a noisy 320×200 at spp 8, thin parts
  destroyed); for an all-analytic scene rebuild the same constants as exact SDFs and render through
  `render_auto` (draft 400×250: 197 s, 99.9% converged). Contracts probed: `cylinder(h, r)` is along Y
  with half-height h; `rotate` takes a vector axis; `plane(y0)` is inside below y0;
  `make_light("softbox", target=, width=, height=, position=)`.

- **Decide, don't guess (sweep 176).** Material for a part, a light rig, a quality preset: `typed(description,
  [options], examples=...)` decides with a margin and an `id`, and `systemone_lint` flags a state that carries
  two observations or a contrastive clause before you ask. Measured on the speaker's parts: materials from a
  one-line description 8 of 8 from the 141-name library. Report `decision_outcome(id, what was used)` and the
  repeat brief is answered from experience; `route(task)` is tiered (act / choose / abstain) and learns the
  same way; `plan_change("render X with Y")` answers reuse / extend / build with the catalog tier and the
  source as evidence before you write a renderer that already exists.


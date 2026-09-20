---
name: lestudio3d
description: "Model, sculpt, texture, render and share 3-D scenes with leStudio3d, the C4D/Blender-style polygon-and-field modeller built on the leCore engine (the 3-D sibling of the 2-D leStudio): install, launch, drive it over its JSON API and the engine's own /engine service, describe scenes in words, run its mesh and CAD ops, path-trace photos, work in a shared .lews workspace with leStudio and other agents, export GLB/OBJ/STL/shaders. Use when the user says leStudio3d, lestudio3d, the 3-D studio, or asks to model, sculpt, generate, render, retopologise, measure or export 3-D geometry with that tool."
---

# leStudio3d, driven as an agent

leStudio3d is a polygon **and** signed-distance-field modeller on leCore: primitives born as exact
SDF trees (so they export as shaders and trace with no bake), meshes (imported GLB/OBJ, sculpted,
retopologised, decimated), 141 physical materials, a procedural node graph, CAD measurement
(volume, inertia, draft, sections, printability), generators (landscapes, planets, oceans, creatures,
supershapes, crystals, trees), a raymarched preview and a path-traced GI photo, and one shared
`.lews` workspace it edits together with leStudio (2-D) and any other leCore app. The browser UI
and an agent use the SAME JSON HTTP API; the whole engine is mounted beside it. Every route is
listed with its summary at `GET /api/agent/tools`; read that first, then this.

Repo: https://github.com/AnOversizedMooseWithSocks/leOS-Studio, folder `3d/lestudio3d` (until the
rename lands it is `3d/polystudio` at 1.4.0 — that checkout predates everything below; prefer the
current standalone zip `lestudio3d_standalone.zip` if the folder still says polystudio). Engine:
https://github.com/AnOversizedMooseWithSocks/leCore, PyPI `leos-core`, import name `lecore`.
`RELEASE_NOTES.md`, `LESTUDIO3D_BACKLOG.md` (open app items), `LECORE_CORE_BACKLOG.md` (engine
gaps with reproducers) and `LESTUDIO_COMPAT_REPORT.md` (the 2-D bridge) are the lab notebook —
grep them before filing anything. Numbers below are measured baselines (2026-09-14, app 1.8.1,
leCore 0.2.22, 4-core sandbox); report yours beside them.

## 1. Install and launch

    [ -d leOS-Studio ] || git clone https://github.com/AnOversizedMooseWithSocks/leOS-Studio.git
    cd leOS-Studio && git pull --ff-only; git log -1 --format='%h %ad' --date=short
    cd 3d/lestudio3d 2>/dev/null || cd 3d/polystudio        # pre-rename checkout: unzip the release over it instead
    cat VERSION                                               # 1.8.1 or later is this manual's build
    pip install --break-system-packages -r requirements.txt     # leos-core[ui] >= 0.2.22 (Flask, Pillow, numpy)
    # a local leCore checkout ahead of PyPI: pip install --break-system-packages --no-build-isolation -e /abs/leCore
    export PYTHONHASHSEED=0 LESTUDIO3D_WORKSPACE=/abs/shared.lews.d     # the live workspace directory (optional but do it)
    nohup setsid python3 app.py --no-browser > lestudio3d.log 2>&1 < /dev/null &
    for i in $(seq 1 30); do sleep 1; curl -sf http://127.0.0.1:5000/api/engine_status > /dev/null && break; done
    curl -s http://127.0.0.1:5000/api/scene | head -c 200      # {"objects":[{"id":"O1","name":"Cube",...

Port 5000, one process, Flask's development server, no authentication — localhost / trusted LAN
only, never expose it. Boot prints the mount: `[engine] 0.2.22  missing: nothing this app calls`,
`[lews] live workspace ...`, `[agent] /api/agent/{tools,invoke} + /api/{mind,engine,events,presence}`,
`[engine] whole engine mounted at /engine/ (N routes)`. `GET /api/engine_status` is the engine's own
`engine_status()` plus the app's `features()` gate (`missing` names any faculty this build lacks).
Nothing is vendored; there is no bundled engine and no version-pin fallback. In a cloud sandbox
background processes are reaped between turns — probe `/api/engine_status` each turn and relaunch;
keep every scene in a script that can rebuild it, and `GET /api/scene/save` when a scene matters.
Tests: `sh tests/run_all.sh` (real engine: route sweep 114 calls / 0 failures, render api 44,
lews bridge 33, scene document 20); `python3 quality_gate.py` before touching a render path.

Always send two headers on POSTs: `X-Client: <this run>` (echo suppression on the change feed) and
`X-User: <stable identity>` (presence, host role, memory partition, journal author). Reuse the same
`X-User` across runs. Helper:

    import json, time, urllib.request
    BASE = "http://127.0.0.1:5000"
    H = {"Content-Type": "application/json", "X-Client": "run-%d" % time.time(), "X-User": "agent-1"}
    def post(path, body):
        req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=H, method="POST")
        try: return json.loads(urllib.request.urlopen(req, timeout=600).read())
        except urllib.error.HTTPError as e: raise SystemExit("HTTP %d on %s: %s" % (e.code, path, e.read()[:400]))
    def get(path): return urllib.request.urlopen(BASE + path, timeout=600).read()

## 2. Finding tools (before hand-rolling anything)

1. `GET /api/agent/tools` — every app route (106) with `methods`, `summary` (first sentence of the
   docstring) and `returns: json|image|stream`. `POST /api/agent/invoke {"tool": "render", "args":
   {"w": 320}, "json": {...}}` calls any of them by name; image routes come back as
   `{"result": {"image": "data:image/png;base64,..."}}` — the only way an agent sees its own work.
   Streams (`photo`, `photo/*`) are refused there; call them directly.
2. `GET /api/scene` is the truth: every object with `id`, `name`, `counts {v, f}`, `bbox`, `faceMats`
   (one material name per face), `faces`, `positions`, `vertColor`, `sculpt`; plus `rev`. It is
   ~1 MB for a 3000-face scene — `?view=summary` for ids/bbox/counts only. `GET /api/scene/doc` is
   the ENGINE's reading (`scene_info`: handles, geometry kind, pre-flight `problems`) and carries
   `ref`, the handle of this scene's engine Scene document.
3. `POST /api/mind {"name", "args"}` — the read-only allowlist of engine faculties (`find_capability`,
   `suggest`, `describe_skill`, `complete_method`, `code_search`, `features`, `version`, image
   analysis: `compare_images`, `seam_continuity`, `est_dx`, `vanishing_point`, `image_colours`,
   `image_signature`). A rejected name returns the allowlist WITH python signatures — post a junk
   name first. Arg names are the python ones (`problem`, not `query`).
4. **The whole engine**: `GET /engine/tools` (2,408 faculties, name / description / params) and
   `POST /engine/invoke {"name": "mass_properties", "args": {...}, "budget": 20000}`. Meshes go in as
   `{"vertices": [...], "faces": [...]}` where the faculty takes a Mesh; objects JSON cannot carry
   come back as `{"type": "Scene", "ref": "ref:Scene:1"}` and any `ref:` string passed as an arg
   resolves to the live object — post `/api/scene/doc`'s `ref` as `scene` to run `scene_info`,
   `place`, `scene_set_texture`, `render_preview` on YOUR scene. `budget` bounds big results to a
   preview with a `ref`. Also `/engine/capabilities/search {"query","k"}`, `/engine/jobs/*`,
   `/engine/documents`, `/engine/skills/*`, `/engine/bus/*`, `/engine/sql`. Faculty errors return
   `{"ok": false, "error"}` with HTTP 200 (the engine's contract) or 400 — check `ok`.
5. `GET /api/docs?q=...` is the catalog ranked by meaning (`find_capability`); no q browses it.
6. Verify, do not eyeball: `GET /api/measure?object=` (bbox, watertight volume), `/api/validate`
   (thin walls, non-manifold, holes), `/api/mass_properties`, `/api/section/measure`,
   `/api/draft_report`, `/api/bounding`, `/api/field_report` (bake method and exactness of an
   object's render field), `/api/render_stats`. Read every PNG before calling it done.

## 3. Objects, ops, materials

- Primitives: `POST /api/new {"kind": cube|icosphere|plane|tetra|humanoid, ...}` — an unknown kind
  is REFUSED with the vocabulary (it used to silently make a cube). Everything else analytic comes
  from `POST /api/compose {"text": "sphere radius .6 subtract rounded box size .5"}` — the SDF DSL,
  kept as the object's exact tree — or `POST /api/extrude_profile`, `/api/sweep`, `/api/sketch/solve`,
  `/api/floorplan`. Objects born analytic render exactly (`analytic-exact`, no bake) until a
  topology edit or sculpt turns them into a grid field.
- `POST /api/op {"op": ..., "object": id, ...}` — 58 ops: `delete_object`, `duplicate`, `place|
  translate|rotate|scale|drop|rename`, `transform {matrix}`, `merge_objects`, `mirror`, `array`,
  `boolean {other, mode}`, `lathe`, `extrude`, `inset`, `loopcut`, `bevel`, `bevel_selection`,
  `dissolve`, `poke`, `subdivide`, `smooth`, `halve`, `delete_faces`, `fill_holes`, `bridge`,
  `triangulate`, `shell`, `band`, `opening`, `wall`, `lattice`, `cloth`, `flute`, `weather`,
  `variants`, `surface_graph`, `sdf_modifier`, `shader_displace`, `shader_material`, `set_layers`,
  `draft_check`, `decimate {target}`, and the remeshers `retopo {density, fast}` (field-aligned
  quads; manifold cleanup + orient are applied first because the engine requires consistent
  winding), `cvt_remesh {sites, iterations}`, `manifold_cleanup`, `orient`. Each returns the scene
  (`only` that object), remeshers add `faces {before, after}` and the engine's `report`
  (`quad_fraction`, `sites`, ...). A refused op is a 400 naming why. Every op is undoable
  (`POST /api/undo`, `/api/redo`, engine `EditHistory`), and every single-object op is RECORDED in
  the object's history (`GET /api/history?object=`, `POST /api/history/op {action: checkout|branch|
  ...}`) so a scene can be replayed and re-parameterised (`/api/anim/keyable`, `/api/anim/bake`).
- Sub-object: `POST /api/select {"object", "op": edge_loop|face_ring|boundary|box|symmetric, ...}`
  (engine meshselect; `op`, not `mode`) returns index lists in the op's element mode plus touched
  vertices; `/api/select/geodesic` grows along the surface; `/api/softsel`; `/api/verts` (the
  ~20 Hz drag stream); `/api/snap`. Sculpt: `sculpt/enter` (mesh → distance grid), `sculpt/begin_stroke`, `sculpt/stroke
  {points, brush, r, s}`, `sculpt/exit` (re-extracts the mesh). Hierarchy: `POST /api/parent`.
- Materials: `GET /api/materials` → 141 presets in 17 classes (metal 15, biome 14, deposit 13,
  diffuse 11, ground 10, emissive 10, stone 8, gem 7, liquid 7, fiber 7, plastic 7, layer 7,
  fabric 6, wood 6, organic 6, glass 4, ceramic 3); default `clay`, floor `concrete`.
  `POST /api/assign {"material", "object", all|faces:[..]|objects:[..]}` — per face, undoable,
  surviving topology edits by nearest-centroid transfer. `POST /api/material/custom` authors one
  from physical parameters; `GET /api/material_ball?name=&res=` previews it. Colour words map
  through one vocabulary everywhere (red→ruby, green→emerald, blue→sapphire, gold→gold,
  purple→amethyst, white→porcelain, black→obsidian, orange→copper, grey→clay).
- Textures/UVs: `POST /api/uv {object}` (engine isomap unwrap, planar fallback), `GET
  /api/uv/maps.zip`, `GET /api/object_texture?object=` (PNG), `POST /api/import_glb` (raw bytes;
  `?mode=auto|asis|decimate|retopo|voxel|rebake&target=`; multi-material scenes split per material,
  textures kept through decimation by `transfer_uv` with the residual reported), `/api/import_obj`,
  exports `GET /api/export_glb|export_obj|export_stl?object=`, `GET /api/export_shader?object=&lang=glsl|wgsl`
  (an analytic object as a distance shader, with the engine's ALU verdict), `GET /api/scene_wgsl`.
- Generators: `POST /api/generate {"kind", "seed", ...}` — `landscape` (biomes, sea level),
  `planet`, `moon`, `star`, `solar_system`, `asteroids`, `galaxy_field`, `ocean` (Gerstner, animate
  with `/api/ocean/animate`), `clouds`, `creature` (+ `/api/creature/walk`), `tree`, `supershape`,
  `scatter {source, target, count}` (also `POST /api/scatter`), `obj`; and the field modifiers
  `bend|twist|displace|elongate|onion|rounded|chamfer|smooth`. `POST /api/erode` runs the engine's
  droplet erosion on a heightfield. `POST /api/data/to_geometry` turns tables into heightfields,
  bars, curves. Units: `GET/POST /api/units`.
- Words: `POST /api/scene/describe {"text": "a red cube next to a green sphere"}` builds objects
  through the engine's `describe_to_scene` (Scene document); returns `made` (object id, handle,
  material), `unknown` and `suggestions` — 400 with those when nothing was understood (~2.4 s for
  two objects). `POST /api/semantic {"command": "make the sphere bigger" | "paint the cube red" |
  "give the cube a metal material"}` edits the existing scene through the controlled grammar (no
  LLM); `GET /api/describe_scene` reads it back in English. Described objects carry the document's
  own SDF classes, which lack the DSL contract, so they are mesh-only afterwards (no shader export);
  build with `/api/compose` when you need the exact tree.

## 4. Rendering

- `GET /api/render?w=&h=&eye=x,y,z&target=x,y,z&fov=&quality=` — one raymarched preview PNG
  (0.35 s at 320×240 for one analytic object). Pass `session=<id>&target_fps=` and the engine's
  frame-budget controller picks the rung (`X-LeCore-Render` says `AUTO[medium] ... frame=42ms
  budget`); `render_progressive` refines a parked camera round by round (`X-Converged: 1` ends it),
  `render_cancel?session=`. The header reports the bake method per object (`analytic-exact`,
  `grid(exact-dist)`, `grid(refine+shell+flood)`) and bake/trace seconds — read it: a baked object
  in an otherwise analytic scene costs seconds (a humanoid: bake 2.9 s), and `bake=0.00s (cached)`
  means the field cache held.
- `GET /api/scene_preview?w=&h=&quality=draft|good&eye=&target=` — the engine's `render_preview` of
  the Scene document, the fast look for the see→fix loop (0.5 s at 320×240). `GET /api/camera/fit?w=&h=`
  frames every vertex exactly (`fit_camera`); with no camera given, previews use the last render
  camera, else a fit.
- `GET /api/photo?w=&h=&spp=&grid=&aov=normal|depth|albedo` — path-traced GI, an NDJSON stream: a
  `meta` line (`w` may be widened to the aspect the tracer needs, `batches`, `grid`, `warnings`),
  then `frame` lines with `png` base64 and `done`, then `{"type": "done", "seconds"}`. The client
  disconnecting cancels the trace. Measured: one analytic cube 160×120 spp 8 streams its first frame
  in 0.1 s; a 7-object scene with a baked humanoid at 320×240 spp 32 took 192 s on 4 cores — budget
  by `spp × pixels × baked objects`, preview with `scene_preview` first. `GET /api/photo_post?
  exposure=&sharpen=&session=` re-grades the last finished photo WITHOUT re-tracing (0.01 s);
  `POST /api/upscale` is the engine's edge-adaptive 2× on a finished frame (data URL in, no
  shrink-then-enlarge; blob URLs refused by name). `POST /api/scene/animate {"keys": {<object id>:
  {"position": [[t, [x,y,z]], ...], "rotation": ..., "scale": ...}}, "n_frames", "fps", "w", "h"}` →
  an animated GIF (data URL) through the engine's `render_animation`.
- Environment: `POST /api/env` with `{"preset": day|sunset|night|starfield|nebula|galaxy}`
  (procedural), `{"studio": classic|soft|dramatic, "gain"}` (the engine's three-point rig as a dome),
  `{"sky": {"hour": 19, "clouds": [["cirrus", 0.5]], "moon": true, "seed"}}` (parametric sky),
  `{"hdr": "<path or data URL>"}` (Radiance .hdr), `{"exr": "<path>"}` (OpenEXR, `pip install
  OpenEXR`). It becomes the preview sky AND the photo's dome light; the reply carries a b64 `preview`
  and `max_radiance`. `DELETE /api/env` clears. `POST /api/photo/light|depth|shapes|texture|scene`
  go the other way: a photo in, a light estimate / relief mesh / fitted primitives / matched fBm
  texture / starter scene out.
- `GET /api/render_engine` is the mesh rasteriser (textured), `GET /api/field_meta` +
  `/api/field_tex.bin` the baked volume for a client GPU tracer.

## 5. The shared workspace (`.lews`) — leStudio, agents, other apps

Set `LESTUDIO3D_WORKSPACE=<dir>` (the same directory leStudio opens as its `_WS_ROOT`). Then every
`GET /api/scene/save` puts the private `lestudio3d.scene` (the full JSON: objects with stable ids,
custom materials, node graph, render assets, units) AND the canonical kinds: `lecore.mesh`
(`lestudio3d:mesh:<id>`, with uv when present), `lecore.sdf` (`lestudio3d:sdf:<id>`, the exact DSL of
every analytic object), `lecore.material` (`lestudio3d:mat:<name>`, matlib name in
`overrides.matlib`), `lecore.image` textures (`lestudio3d:tex:<id>`), one `lecore.scene`
(`lestudio3d:scene`, bindings by section id), `lecore.camera` (`lestudio3d:camera`, the last render
camera), a `lecore.journal` per edited object (`lestudio3d:journal:<id>`, the recorded ops) and the
environment as a content-addressed `lecore.asset` referenced from `lestudio3d:env` so GC keeps it.
Unchanged sections are skipped by hash; deleted objects' sections are deleted. Object ids come from
the workspace's own counter (`Workspace.mint("O")`) and survive save → load.

- `GET /api/workspace?since=<rev>` → `open`, `root`, `describe`, `roster`, `presence` (who, app,
  host), `kinds`, `assets`, `changes` since a rev. `GET /api/events` (SSE `{rev, changes,
  participants}`, a comment ping every 2 s) and `GET /api/presence` come from the engine surface.
  `GET /api/workspace/wait?rev=&t=` is the long-poll for clients without SSE. `POST
  /api/workspace/note {"kind", "data"}` records a selection or viewport as one journal line.
- `POST /api/workspace/import` with JSON `{"from": "workspace"}` pulls from the live directory:
  `lestudio.document` sections (leStudio's paintings, layers composited by the engine's
  `composite_layers`; R67 journal-first layers use their replay base and are named in
  `journal_only`) and `lecore.image` become textures (`?object=<id>` applies the first one, UVs
  unwrapped if missing); other apps' `lecore.mesh` and `lecore.sdf` become objects (the SDF's tree
  is kept); `lecore.scene` bindings supply names and materials; a `lecore.camera` is adopted.
  Posting raw `.lews` bytes instead imports a FILE through `Workspace.from_file` (journalled as
  "opened a file", every section carried). `POST /api/workspace/inspect` (raw bytes) reports what a
  file holds without importing. `GET /api/workspace/export` downloads the live directory as one
  `.lews` (`Workspace.export_bytes`) — every app's sections, assets and presets.
- `POST /api/scene/load {"from": "workspace"}` reloads our own scene from the directory; `{"objects":
  [...]}` (a saved JSON) restores a file.
- `POST /api/workspace/gc {"dry_run": true}` lists unreferenced assets (`gc_assets`). NOTE: leStudio
  hoists shared arrays into `lecore.asset` referenced by `array_refs`, which the engine's GC does not
  count as a reference — do not GC a directory a painter is using.
- `GET/POST/DELETE /api/presets` — `lecore.preset` sections (`{"name", "target":
  "lestudio3d.render", "params": {...}, "tags"}`), listable by every app. `POST /api/memory
  {"action": remember|recall|observe|suggest|habits|forget|stats, ...}` is per-user memory through
  the engine's `app_substrate` (partition keyed by `X-User`). `POST /api/invite {}` mints a
  shareable link + code (`create_invite_link`), `POST /api/join {"link_or_code"}` admits a guest —
  bookkeeping, not a gate.
- Bridge facts: leStudio (2-D) only file-opens `.lews` whose meta says `app == "lestudio"` and has
  no `lecore.preset` reader yet; its journal-first layers cannot be replayed here (ask for baked
  pixels for an exact texture). The engine's `scene_section` drops `name` and `sdf` bindings
  (LC-8), so bind by mesh; the app publishes the SDF twin under a parallel id.

## 6. MCP — what exists and what does not

leStudio3d exposes no MCP server; its agent contract is the HTTP API plus `/engine/`. The engine's
own MCP (`holographic_mcp.py`, stdio; `scene_create`, `scene_adjust`, `scene_export`, `image_tool`,
`lecore_find|describe|invoke`, `memory_write|search`) reasons about scenes in ITS document, not
this app's; the bridge between them is the `.lews` directory (its sections import here) or the
agent. Prefer `/engine/invoke` with the `ref` from `/api/scene/doc` when you already have the
studio up: same mind, same scene.

## 7. Several agents on one scene

Nothing here is a swarm feature — no roles, ownership or merge strategy beyond the workspace lock,
the document lock and the recorded history. Build on documented primitives:

- **One object per worker, by convention**: each worker addresses only its own object ids (name
  them for the worker), coordinates through `X-User` + `/api/events`, checkpoints with
  `/api/scene/save`; audit overlap with `/api/measure` and `/api/bounding` (OBB overlap), not by eye.
- **One workspace, one app instance per worker**: each worker runs its own leStudio3d on a different
  port with the SAME `LESTUDIO3D_WORKSPACE`, publishes with `/api/scene/save` and pulls with
  `import {"from": "workspace"}`; canonical sections carry the geometry; ids never collide
  (workspace-minted). Presence, host and the change feed are the engine's.
- **Journal-first**: every single-object op is recorded; a worker can hand over ONLY its journal
  (`lecore.journal` section) and a coordinator replays it (`/api/history/op`).

Discipline: distinct stable `X-User`/`X-Client` per worker; long work (photos, retopo, GLB rebake)
serialised per worker — one process, one document lock; verify with numbers.

## 8. Sharp edges (measured)

- `/api/scene` is big; `?view=summary`. Image results through `/api/agent/invoke` are data URLs;
  `/api/photo` will not go through invoke. The engine classifies image routes by the FIRST path
  segment (LC-4), which is why the preview is `/api/scene_preview`, not `/api/scene/preview`.
- `retopo` on a marching-tetrahedra + decimate mesh fails `is_oriented` unless cleaned first — the
  route does `manifold_cleanup` → `orient` for you; do the same before `/engine/invoke
  surface_retopo`.
- `refine_scene` (the engine's self-improving loop) cannot take the Scene document (LC-3); use
  `/api/semantic` for edits or run the loop on `build_scene` via `/engine/invoke`.
- `scene_add` refuses a Mesh; our objects enter the document as their FIELDS (exact tree or baked
  grid) — anything with `.eval` is accepted. `Scene.add(tags=)` wants a dict (LC-6).
- `lews_material_section(library=)` only knows the optics library (LC-5); matlib names go in
  `overrides.matlib`.
- `quality_gate.py` fails `edge_tones` at 0.5985 vs 0.90 on the exact-field render — pre-existing
  (PS-1), not a regression; look at the frames (`--write`) before moving a threshold.
- `/api/new` vocabulary is cube/icosphere/plane/tetra/humanoid; spheres, tori, cylinders come from
  `/api/compose`. `/api/generate` kinds are listed in §3; an unknown kind is refused.
- Deliverables follow the user's standing rule: one zip of changed/new files, paths relative to the
  app root, the `.lews` alongside any PNG that matters, no commits or pushes.

## 9. Typed decisions, the fast loop, and the exact frame — measured in the speaker session (sweep 176)

Everything below was run on a 14-object model built from the blueprint; the numbers are in
`docs/research/RESEARCH_04_tool_use_experiments.md`.

- **Materials and tools as typed decisions.** For each part, `typed(description, [materials])` chose from
  the 141-name library 8 of 8 against hand choices; the *tool* decision (extrude_profile / sweep / compose /
  new primitive) was 1–3 of 8 from the same functional descriptions and 8 of 8 once the parts were described
  as GEOMETRY ("a tube: a circle profile extruded to a length", "a shallow dome: a sphere clipped") with
  balanced example budgets. `systemone_lint` flags the imbalance (the shortest option owns the smoothing
  floor) and the wrong-vocabulary states before you decide; heed its warns. Report every pick with
  `decision_outcome(id, material_or_tool_used)` — the repeat brief is then answered from experience.
- **The iteration loop is the rasteriser.** `/api/render_engine` at 0.38–0.7 s per 640×400 frame;
  `/api/scene_preview` took 48 s (it bakes); the field preview warns thin parts are thinner than a grid cell.
  A three.js turntable through Playwright on SwiftShader from `/api/scene` arrays (positions are flat,
  faces are index lists, `faceMats` per face) ran 350 ms/frame for 35k triangles and is what caught four
  buttons reading as one bar.
- **The photo route bakes meshes to a grid** — 244 s for a noisy 320×200 at spp 8, caps/rings/feet lost.
  For an all-analytic scene rebuild the same constants as exact SDFs and render through `render_auto`
  (draft 400×250: 197 s, 99.9% converged), in kill-safe row strips for the delivery frame (`lecore-render`
  skill). `sdf_scene_shader(parts, camera)` compiles that exact scene to one WebGL2 shader whose silhouette
  matches the engine's trace at IoU 0.985 — the viewport can show the render's geometry.
- **Two traps that cost hours.** `pip install -r requirements.txt` here pulls a PyPI `leos-core` that
  SHADOWS the checkout for any script not run from the repo root (`PYTHONPATH=.`); the app's engine is that
  PyPI build, so new checkout verbs are not on it — decisions go to your own service, geometry to the app.
  And a `Layer` node screened in the graph washes the sheet white (transparent reads as white): bake labels
  with `add`.
- **Several agents on one scene (§7) with the swarm contract.** Each worker's step is a `swarm_step` with
  `done_when` and evidence from `/api/analyze` (region brightness, dominant hue) — refused without them —
  and `swarm_evaluate()` is the exit. The region map, palette and layer ids live in the shared partition
  (`teach` / `ask`), so a worker reads what the layout worker decided instead of re-deciding it.
- **Verify the object the way you verify a decision:** `/api/analyze` regions for what landed,
  `sdf_validate_glsl` for every analytic part (worst diff 2.9e-7 after vec2/swizzles joined the shim),
  `verify_decision(state, answer)` for a served pick that might contradict what was learned.


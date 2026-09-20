---
name: lestudio
description: "Create, paint, composite or finish images with leStudio (leOS-Studio/2d/lestudio), the layer-and-node image editor built on the leCore engine: install, launch, drive it over its JSON API, discover nodes and engine faculties, paint with real media, run its simulations, work in a shared multi-user workspace, coordinate several agents on one picture, export. Use when the user says leStudio, lestudio, leOS-Studio, or asks to make, paint, filter, composite, finish, animate or export an image with that tool."
---

# leStudio, driven as an agent

leStudio is a Photoshop/GIMP-style layered editor with a non-destructive node graph (105 node
ops, most a direct door into a leCore faculty), a physical paint engine (oil, acrylic,
watercolour, PBR materials, impasto), simulations (fluid, smoke, ink, fire, reaction–diffusion,
erosion, cloth-like stroke rigs) and one shared workspace. The browser UI and an agent use the
SAME JSON HTTP API — the browser is just one client; an agent is another. A document is a
recipe (stroke paths, brush settings, node parameters, seeds), so the same `.lews` renders
byte-identically across saves and processes.

Repo: https://github.com/AnOversizedMooseWithSocks/leOS-Studio, folder `2d/lestudio`. Engine:
leCore (`import lecore`), the `leos-core` PyPI package or a local checkout. `AGENT.md` in that
folder is the agent manual AND the lab notebook (1,300+ lines): read its first 50 lines every
session (Orientation, Conventions, worked example) and grep the rest by topic when a feature
misbehaves — the answer is usually already recorded. Numbers below are measured baselines
(2026-09-11/13, repo 3e28ebe, leCore 0.2.21); report yours next to them.

## 1. Install and launch

    [ -d leOS-Studio ] || git clone https://github.com/AnOversizedMooseWithSocks/leOS-Studio.git
    cd leOS-Studio && git pull --ff-only; git log -1 --format='%h %ad' --date=short
    cd 2d/lestudio
    # engine first (a local checkout is newer than the PyPI floor 0.2.9), then the studio
    [ -d ../../../leCore ] && pip install --break-system-packages --no-build-isolation -e ../../../leCore \
        || pip install --break-system-packages 'leos-core>=0.2.9'
    pip install --break-system-packages --no-build-isolation --no-deps -e .
    pip install --break-system-packages flask pillow numpy opencv-python-headless
    python3 -c "import lecore, lestudio; print('lecore', lecore.__version__, lecore.__file__)"

    export PYTHONHASHSEED=0 LESTUDIO_PORT=5050 LESTUDIO_THREADS=4
    nohup setsid python3 -m lestudio > lestudio.log 2>&1 < /dev/null &
    for i in $(seq 1 20); do sleep 1; curl -sf http://127.0.0.1:5050/api/health && break; done
    curl -s http://127.0.0.1:5050/api/ready        # {"canvas":[768,512],"ok":true}  (ready in ~1 s)

`/api/health` is liveness and takes no lock (0.3 ms even while saturated with watercolour
strokes); `/api/ready` takes the document lock and fails while starting or wedged — probe with
health, gate traffic with ready. `LESTUDIO_THREADS` caps BLAS threads and must run before numpy
imports, hence from the entry point. `--no-build-isolation` matters behind a proxy. Optional
accelerators (`run.sh accel`: ziglang, numba, pyfftw, cupy) are never installed silently. In a
cloud sandbox background processes are reaped between turns — start every turn with `/api/health`
and relaunch when silent; the workspace is in memory, so keep every graph in a script that can
rebuild it and save `.lews` when a document matters.

Always send two headers on POSTs: `X-Client: <this run>` (echo suppression) and `X-User:
<stable identity>` (presence, host role, kick). Reuse the same `X-User` across runs or you are a
crowd of ghosts. A stdlib helper:

    import json, time, urllib.request
    BASE = "http://127.0.0.1:5050"
    H = {"Content-Type": "application/json", "X-Client": "run-%d" % time.time(), "X-User": "agent-1"}
    def post(path, body):
        req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=H, method="POST")
        try: return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as e: raise SystemExit("HTTP %d on %s: %s" % (e.code, path, e.read()[:400]))
    def get(path): return urllib.request.urlopen(BASE + path, timeout=120).read()

## 2. Finding tools and capabilities (before hand-rolling anything)

1. `GET /api/schema` once: every `/api` route with its docstring, the full node-op catalog
   (`ops`: category, inputs, params with `kind`/`lo`/`hi`/`choices`/`hint`/`when`, and
   `available` — false when this build's leCore lacks a `requires` faculty; filter on it or the
   graph 500s), and `hints`. It hides PATCH/DELETE routes and any route whose docstring sits on
   a wrapper — not exhaustive for verbs. Unknown node params are silently ignored (every Blur in
   two sessions ran at the default because the param is `sigma`, not `radius`): check `ops`.
2. `GET /api/state` is the complete truth: docs, active_doc, layers (full meta + `has_strokes`),
   groups, masks, selections, splines, the last 64 strokes (summaries), media_status, brushes,
   graph, ops, blend_modes, paper, papers, auto_stratum, palette_ready, brush_state, materials,
   peers, capabilities, can_undo/can_redo, dpi.
3. No node fits → ask the engine through the studio's own door, `POST /api/mind {"name",
   "args"}`: a read-only allowlist of 14 leCore faculties — `find_capability`, `find_scored`,
   `suggest`, `describe_skill`, `complete_method`, `capabilities`, `features`, `version`, and the
   analysis helpers `compare_images`, `seam_continuity`, `est_dx`, `vanishing_point`,
   `image_colours`, `image_signature`. A rejected name returns the whole allowlist WITH python
   signatures — post a junk name first to harvest them. Arg names are the python signatures
   (`problem`, `task`, `prefix`), not `query`. Nested lists become float32 arrays; arrays over
   4096 elements come back truncated; mutating faculties are not exposed. It bumps the sync
   revision even though it is read-only.
4. Purpose-built routes exist for shader work (`POST /api/sdf/shader`, `/api/postfx/shader`,
   `/api/shader/match` ~10 s, `GET /api/shader/presets`, `/api/shader/palette`), shape
   recognition (`POST /api/shape/recognise`), re-rendering strokes at a new size
   (`/api/layer/revector`), near-duplicate layers (`GET /api/layers/duplicates`), the Image menu
   (`POST /api/crop`, `/api/reorient` — rotate/flip are lossless array reorderings).
5. Verify rather than eyeball: `POST /api/analyze {node?, sock?, regions:[{name, box:[x0,y0,x1,y1]
   in 0..1, metrics:[mean_rgb, dominant_hue, brightness, fraction_matching]}]}` — "is the sky
   actually blue?" — so a script can check a `.lews` without pulling PNGs apart. Also
   `GET /api/histogram`, `GET /api/graph/timings` (per-node seconds of the last real compute),
   `POST /api/layer/compare`. And Read every PNG before calling it done.
6. Probe signatures, not names: `have("fluid_step")` was true long before it could hold a
   simulation at the canvas edge; the studio inspects the signature for `boundary`.

## 3. MCP — what exists and what does not

leStudio exposes no MCP server and consumes none (a grep of the whole tree finds nothing);
its agent contract is the HTTP API above. The leCore engine has its own MCP
(`holographic_mcp.py`, stdio, `lecore-mcp` after pip install; client block
`{"mcpServers": {"lecore": {"command": "python3", "args": ["/abs/leCore/holographic_mcp.py"],
"cwd": "/abs/leCore", "env": {"PYTHONHASHSEED": "0", "LECORE_MEMORY_ROOT": "/abs/leCore/lecore_memory"}}}}`)
with `lecore_map`, `lecore_find`, `lecore_describe`, `lecore_invoke`, `image_tool`,
`scene_create`, `zoo_ask`/`zoo_teach` (call `zoo_boot` first), `memory_write`/`memory_search`.
The two compose with the agent as the bridge: use the engine MCP for reasoning, discovery and
cross-session notes about a painting; use `/api/mind` when you are already driving a studio
(it answers from the SAME mind the studio uses, so `capabilities`/`features`/`version` match
what the studio will do); use the studio's HTTP API for everything that touches the picture.
Do not route image bytes casually between them — the studio's image doors are
`GET /api/composite.png`, `/api/layer/<lid>.png`, `/api/export.png`, `/api/graph/render.png?w=&h=`.

## 4. The node editor

`POST /api/graph {"nodes": [{id, type, params, inputs, x, y}]}` replaces the whole graph, then
commits `Layer out` nodes and returns `{ok, committed, conflicts}`. `PATCH /api/graph/node/<nid>
{params?, inputs?, pos?}` is the cheap per-node iteration (`null` removes a wire). Inputs are
`"NID"`, `"NID.socket"` or `[id, socket]`; an input key `param:<name>` drives any numeric
parameter from a node (a `Value`, `Color value`, `Light direction.x/.y/.angle`,
`Shadertoy.value`, `Perceptual diff.score` directly; any image by its mean luminance).
`mute: true` bypasses a node. `Layer` / `Layer group` / `Mask` nodes take `params.doc` to read
another workspace document without copying pixels. `POST /api/graph/group {ids}` /
`/ungroup` encapsulate a subgraph. Cycles are refused.

Render: `GET /api/graph/output.png` (the `Output` node; nothing wired = composite of all
layers; on any exception it silently falls back to the composite — an agent gets a picture,
not an error), `GET /api/graph/preview/<nid>.png?sock=`, `GET /api/graph/render.png?w=&h=&node=`
(arbitrary resolution, ≤ ~67 MP; procedural nodes gain real detail), `GET /api/graph/sigs`
(poll this tiny JSON and refetch only what moved). Everything is memoised under a hash of
type + params + upstream signatures — O(change). Long nodes (Clouds ~6 s, Segment, Shadertoy
match ~10 s) run as jobs: `POST /api/graph/run {"id"}` → `{job}`, poll `GET /api/job/<id>`,
`POST /api/job/<id>/cancel`, `GET /api/job/<id>/result`. `POST /api/render/animation {node,
param, from, to, frames≤120, fps≤30, format: gif|mp4}` sweeps one parameter (sizes snap to
multiples of 16; mp4 needs imageio-ffmpeg). Bake with `POST /api/graph/apply {"id": nid,
"layer"?: lid, "name"?}` — the docstring says `node`, the code reads `id`; without `layer` it
makes a new `"<Type> bake"` layer; onto an existing layer it REPLACES the pixels. A `Layer out`
node is a continuous bake with snapshot semantics; one that targets a layer the graph reads is
refused and reported in `conflicts`.

Node families (105): Generate (`Solid`, `Gradient`, `Radial gradient`, `Pattern`, `Fractal`,
`Procedural texture` — marble/wood/brick/voronoi/musgrave/…, `Sky`, `Clouds`, `Water`,
`Scatter`, `Warped noise`, `Seamless noise`, `SDF render` — DSL/mandelbulb/mandelbox, compiles
to GLSL, `Branching growth`, `Reaction diffusion`, `Smoke`, `3D model`, `Shadertoy` — renders
in the browser's GPU, outputs `out` + `value`, `time`); Color / Adjust (`Palette map`, `Color
ramp`, `Levels`, `Curves`, `Hue / Saturation`, `Color balance`, `Gradient map`, `Threshold`,
…); Filter (`Blur` — `sigma` 0–30, `Sharpen`, `Denoise`, `Edges`, `Segment`, `Inpaint`,
`Upscale 2x`, `Displace`, `Texture synth`, `Flow warp`, `Smart smooth`, `Deconvolve`,
`Erosion`, `Motion blur`, `Band`, `Spectral chain`, …); Comp (`Merge`, `Chroma key` — `despill`
0 keeps the subject's own greens, `Luma key`, `Grade`, `Transform`, `Layer style`, `Glow`,
`Switch`, `Perceptual diff`, `Distance field`, `Premult`/`Unpremult`, channel ops); Combine
(`Blend` with the 10 blend modes, `Morph`, `Mask mix`, `Align`, `Seamless clone`, `Group`);
FX (`Fluid`, `Smoke`, `Paint relief 3D` — a layer's paint body as a lit mesh, `Light shafts`,
`Grain`, `Chromatic aberration` — any `chroma` > 0 fringes hard, `Vignette`, `Refract`,
`Stroke FX`, `Depth fog`, `Post FX` — bloom/grain/vignette/chroma/tonemap, grain 0.06 is heavy,
0.012 reads as film, `Splatify`, `ASCII art`, `Perspective grid`); Values (`Value`, `Light
direction`, `Sample image`, `Color value`); Input (`Media in` — a file path, a direct video/
stream URL, or `test:clock`; `Layer`, `Layer group`, `Mask`); Output (`Output`, `Layer out`,
`Mask out`, `Paint out`, `Brush out`, `Fill out`); Utility (`Reroute`, free). `SDF render` has
no alpha: render on a pure green Solid and Chroma key it. A studio graph at 2560×1600 was
OOM-killed on a small box — finish at 1× and upscale afterwards.

## 5. Layers and what only layers can do

`POST /api/layer {"action": …}`: `add {name?, below?}`, `duplicate`, `merge_down`, `merge
{ids}`, `merge_visible`, `clear`, `remove`/`delete` (refuses the last canvas layer), `fill
{content:{kind:"solid",color}}`, `flip {axis}`, `move {index}`, `edit` — the whitelist is
`name, visible, opacity, blend, mask, mask_invert, alpha_lock, clip, thickness, vol_kind,
vol_ior, vol_density, absorbency, emissive, emissive_color, reflect, dispersion, media_rate,
z_off, tilt_x, tilt_y, curve, dome, field, field_mode, field_strength, curve_axis,
curve_profile, dome_profile, locked, relief, gravity, gravity_angle, optical, media_res,
media_time, bg`. Ten blend modes: normal, multiply, screen, overlay, add, subtract,
difference, darken, lighten, softlight. Masks: `POST /api/mask {add|duplicate|remove|edit|
move|merge}`, attach with `edit {mask: mid}`; a selection becomes a mask via
`POST /api/selection {"action":"to_mask"}`. Selections: `POST /api/select {"tool": rect|ellipse|
wand|lum|obj, "params": {x0,y0,x1,y1…}, "mode": new|add|sub}` (NOT `action`/`x,y,w,h`); one
unsaved working slot, `POST /api/selection/keep` promotes it; only kept selections reach the
`.lews`. Groups: `POST /api/group {add|remove|edit}`. Splines: `POST /api/spline {add|remove|
edit|stroke}` (`stroke` rails a brush along it).

What is unique here: a layer is made of STUFF — `vol_kind` none|water|glass|fog|absorb|puff
(with `vol_ior`, `vol_density`, `thickness`) and the dynamic media `inkwater|smoke|fire|air`
whose fluid state survives between strokes; `emissive`, `reflect`, `dispersion`, `optical`;
`gravity`/`gravity_angle` as a SURFACE property (an easel runs down, a wall runs down that
wall, a flat canvas levels outward — a different code path); `z_off`, `tilt_x/y`, `curve`,
`dome` and `POST /api/view3d {"mode": flat|ortho|persp, "vantage"}` for multiplane — a
directional light with `shadows=True` throws each pane's shadow onto the one behind it, which
is most of the win (perspective itself measured near-inert: ~2.4% of pixels whether panes are
26 or 88 units apart; key ~0.6, fill ~0.2, per-layer `relief` ~0.35 or impasto lights like
craters); `relief`; walls (`POST /api/wall`), fields (`POST /api/field`, point|direct|vortex),
lights (`POST /api/light`, kinds view|directional|point|spot|dome, presets sun|studio|
three_point|dome — every number is finite-checked because one NaN intensity corrupted the whole
picture invisibly), `POST /api/contact_print` (a tilted slab printing through the one below).

Paint strata (`POST /api/stratum {"on": true}`): paint height caps at 4.0 per layer and flattens
to a plateau; with strata on, the excess cascades into build-up layers (`"sky · build-up 2"`)
that inherit clip/blend/opacity/alpha_lock and shade on the total height — 9 loaded passes give
22 units of relief across 6 strata. Cost ~815 ms vs ~240 ms per stroke, and it is document-wide
(a watercolour ground spilled into stripes when left on): base → glazes → impasto on its own
layer above, and turn it on only for that layer's passes. Glazing that makes forms turn: local
colour on a base, a shadow glaze on its own MULTIPLY layer (~0.55 opacity — at 0.24 it barely
darkens), a light scumble on SCREEN, both `clip=True`, specular on a small ADD layer; lay the
silhouette solid then `alpha_lock=True` before modelling — one layer per object. The palette is
its own 560×150 surface, not a layer: `POST /api/palette {colors, layer?, x?, y?, size?, media?}`
squeezes mounds, `GET /api/palette.png` (region in the `X-Palette-Box` header),
`POST /api/palette/paint {points, color, radius, load?, mix?, mode: blend|knife}` both dips and
mixes, `POST /api/palette/clear` scrapes undoably; keep it at the bottom of the stack in the
volumetric view (a known grey-embossing artifact).

## 6. Brushes and mediums

`POST /api/paint {layer, points: [[x, y, pressure?]…], color, radius (8), opacity (1), hardness
(0.7), media?: oil|acrylic|water, material?, load? (0.6), mix? (0), real_brush? (false),
stroke_taper?, erase?, target_mask?, selection?, brush?, live?, record? (true), patch_ok?,
mode?}` → `{ok, sid, warning?}`. Modes: brush, knife, blend, smudge, clone, heal, erase_strokes,
erase_top, erase_undo, erase_depth, node (a `Paint out` node's image as the brush source). Wrong
shape or non-finite numbers → 400 naming the field; out-of-range real numbers are CLAMPED. A
stroke that can have no visible effect returns a `warning` (hidden layer, opacity ≤ 0.02,
alpha-lock over empty, clip onto an empty base) rather than a refusal — read it. `patch_ok`
returns just the dirty window. `POST /api/fill {layer, x, y, tolerance (0.12), contiguous,
source: color|gradient|pattern|node}` — seed clearly inside a region (a seed on an outline pixel
repainted a 25,000-px outline network) and guard with an expected-area check.

The deposit model turns coverage into paint by four mechanisms in stroke-local coordinates:
load depletion (long strokes run dry), a bristle comb (~4 px lane pitch at any size), canvas
tooth (a fixed-seed fbm+weave substrate, so thin paint catches only on the peaks) and a berm
(bristles bank paint into two rims, volume conserved); the weave is filled by paint
(`exp(−h/0.40)`), not printed on top. Marks are the union of per-hair bristle tracks (~240 ms
for a 1080p 200-point stroke). Pressure is the third point component; speed is inferred from
point spacing, and a two-point line from the API reads as a maximum-speed flick — send ≥ 6
points or the stroke lands three times too thin. Mice always paint at full pressure.

Media (`_MEDIA`): oil (hold 0.62, flow 0.22, 10 flow iterations, gloss 0.34), acrylic (0.80,
0.16, 6), water (0.10, 0.45, 26 iterations, absorb 0.9, edge-darkening, granulation) — water is
the slow case (~150 ms flush on a big brush) and always will be. Materials: 12 PBR presets
(gold, silver, copper, chrome, brushed_steel, lacquer, plastic, rubber, wax, clay, chalk…) as
`"material": "gold"` or a dict `{preset, rough, metal, grain, hold, flow}`; the brush colour is
the albedo; a material wins over a medium if both are sent. Papers: `POST /api/paper {"paper":
canvas|rough|cold_press|hot_press|smooth|linen}` — stamped onto each layer as it is painted;
measure a stock by how pigment tracks its dips (`corr(tooth, alpha)`: smooth +0.10, rough −0.21,
linen −0.23), not by variance. Watercolour is a fluid IN paper (Curtis 1997): wicking, edge
darkening that MOVES pigment to the rim, granulation that pools in the dips (the opposite sign
from stiff paint); its knobs are fixed per medium (UX gap).

Real brush mode (`real_brush: true`): the brush is a physical object holding finite paint
(`brush_state` in `/api/state`); it recharges only from a genuine pile on the canvas and lays
nothing at 0 charge (fades over the last ~18%). The single biggest cause of flat paintings was
painting as if the brush were infinite — reload (`POST /api/brush_load {color?, amount?}` or dip
the palette) whenever charge drops below ~0.55. Wet-on-wet `mix`: the rate is per pixel travelled;
mix at the palette, model at mix ≈ 0.2 (at 1.0 the form converges to one muddy average); dipping
MIXES, it does not replace. The blender (`mode: "blend"`) is a clean brush that softens the paint
already there — recorded and replayable, unlike `smudge` — but it averages and it carries paint:
many SHORT strokes across the form, never long radial passes (a starburst) or concentric ones
(a vinyl-record spiral). The palette knife (`mode: "knife"`, blades smooth|push|scrape|spread,
radius 26) re-levels the whole paint column, which is also the cure for stepped strata. Custom
tips: `POST /api/brush {add|remove|edit}` with `spacing, follow, j_angle, j_size, j_scatter`, tip
128×128 grey from a `Brush out` node. Every stroke is a live record: `/api/strokes/select|width|
rig|simulate|key|points|move|pull|smooth|split|join|duplicate|delete|tolayer|clipboard`,
`/api/nudge`, `/api/transform {kind:"strokes"}`; `POST /api/stroke_group {create|dissolve}`
bundles strokes so a group id works anywhere a stroke id does. Paint Effects: an inkless
recorded stroke (`opacity: 0, record: true` → `sid`) feeding a `Stroke FX` node (`spline: sid`,
comma-join several; `mode: particles|tubes`, gravity, wind, `field` attract|repel|flow|contain)
into a `Layer out` — strokes stay procedural and retunable. Studio setups (oil / watercolour /
ink) drive only ordinary endpoints (`/api/paper`, `/api/palette`, `/api/stratum`), so an agent
reproduces them with the same POSTs; there is no setup endpoint.

## 7. Simulations

There is no `/api/sim`. Simulation is driven five ways: node `steps` params re-evaluated through
the graph cache (`Fluid` 1–200 steps, buoyancy, swirl, viscosity, obstacle matte on `solid`,
`density` output, solved on a ≤192-wide grid — 20 steps at 96×128 in 0.05 s, deterministic per
seed; `Smoke` FFT solver with presets rising|plume|swirl|opposing|shear|buoyant; `Reaction
diffusion`; `Branching growth` lightning/frost; `Erosion` droplets; `Clouds` raymarched ~6 s;
`Water`); per-layer media slabs — paint into an ink/smoke/fire layer and `POST /api/media/step
{layer, steps}` or `POST /api/media/cook {steps ≤ 2400, layer?, until: "settled"}` (negative steps
un-cook; `settled` uses the engine's regime detection so a cap-stop is distinguishable from a
settle; `media_res` coarse 128 cells ~11 ms / normal 192 ~18 ms / fine 320 ~45 ms per frame);
the timeline (`POST /api/timeline {action: "frame"}`, `media_rate`, `media_time`); rigged stroke
physics (`POST /api/strokes/rig` then `/api/strokes/simulate {id, steps, gravity, wind, damping,
stiffness, seed}` → joints); and jobs / `POST /api/render/animation` when it is slow. Paint flow
under gravity and watercolour run inline per stroke (`POST /api/paint_run {layer, steps, gx, gy,
gz}` presses paint downhill along the layer's own surface).

GPU: painting runs on the CPU (brush, impasto, media, blurs are pure NumPy — porting the deposit
to CuPy is a project, not a sweep item, because every stroke would pay a host↔device copy);
the GPU, when present, is used for simulation and node work, and shader previews run in the
browser's GPU. Read `GET /api/status` → `subsystems` (painting: cpu; simulation: gpu|cpu;
shaders: gpu), `gpu` (a boolean, deliberately separate from the `gpu_report` dict), `accel`,
`accel_missing` (install line and what each unlocks: ziglang 2–5× kernels and 3.8× raymarch
for Clouds/SDF, numba JIT for SDF render, pyfftw for spectral nodes, cupy for the engine
backend), `advice`, and `determinism` — enabling a GPU path makes renders bit-approximate and
the byte-identical `.lews` guarantee lapses, with a warning.

## 8. One workspace, many people and apps

The workspace is ONE SHARED STUDIO, not a canvas per visitor: `WS`/`DOC` are module globals, it
must run as one worker, and `app.run()` is Flask's development server. There is no
authentication of any kind — the host role and the kick list rest on the self-asserted `X-User`
header — so treat the API as a localhost / trusted-LAN surface and never instruct anyone to
expose it. Presence is holding `GET /api/events?client=&user=&name=` open (SSE: `{rev, src,
editors, names}` when the revision moves; a 2 s ping reaps dead tabs — the fix for immortal
ghost editors). Every mutating `/api/*` POST bumps `rev` (except graph/run, live, autosave, jobs);
a client refreshes from `/api/state` when `rev` moves and `src` is not its own `X-Client`. The
host is the earliest-seen user still present; `GET /api/editors`, `POST /api/editors/kick {id}`
and `/allow` (host only, 403 otherwise; a kicked user's POSTs get 403 from every tab). Invites:
`POST /api/invite` mints a single-use code (leCore `create_invite_link`), `POST /api/join` —
bookkeeping only, not a gate. Names: `POST /api/presence/name`.

Documents: `POST /api/new {name, width, height, background|null, dpi}` (8×8 … 16384 px/side,
≤ 80 MP, dpi 1–2400) activates the new document; `POST /api/doc {"action": activate|rename|
close|settings}` — `close` on a document with edits returns 409 `{needs_confirm, edits}` unless
`force: true`; `settings {width, height, mode: resample|canvas}` is two different operations.
`activate` records who is viewing what (`state.peers[].doc`) — but `WS.active` is a single
global and paint routes act on the active document, so participants must never activate
concurrently. Cross-document composition without moving pixels: `Layer`/`Layer group`/`Mask`
nodes with `params.doc`. `GET /api/workspace.lews` / `POST /api/workspace/open` carry EVERY
open document (the clean handoff unit; it grows with each open document — close stale ones,
20 MB with two finish docs), `POST/GET /api/autosave` never 500s (200 with `ok:false` says save
manually), `POST /api/undo`/`/redo` follow whichever surface was edited last. Compositing,
resize, close and new serialise on one re-entrant lock (an old race composited old-shaped
layers into a new frame: 5/40 → 0/120). Other apps: the engine's `App("name", user=…)`
substrate gives a separate memory partition per (app, user) — physical isolation, a directory,
not a salt — but this build of leStudio does not use it; per-user learning is unimplemented.
Streaming: `POST /api/live {"action": start|stop, "fps"}` (0.5–30), `GET /api/stream.mjpg`,
`GET /obs` (chromeless capture page, turns Live on), `GET /api/stream/health` (measured
sustainable fps: ~120 ms per 720p frame, ~990 ms at 4K).

## 9. Several agents on one picture

Nothing in the studio is a swarm feature — no roles, queue, ownership, locks beyond the
document lock, or merge strategy; `/api/job/*` is single-process async evaluation. Say so, and
build only on documented primitives:

- **One document per worker, merged by graph** (safest): each worker `POST /api/new`, works on
  its own document, reports the id; a coordinator composites them in a merge document with
  `Layer` nodes carrying `params.doc` and the ten blend modes; one `.lews` carries all of it.
  Serialise `activate` through the coordinator — activate, do the whole batch, hand on.
- **One document, one layer per worker by convention** (most useful, least protected): a layer
  named for each worker; workers only ever address their own layer id; `alpha_lock`/`clip` per
  layer; audit overlap with `POST /api/layer/compare` and `GET /api/layers/duplicates`. The
  server enforces no ownership.
- **Inkless recorded strokes + one `Stroke FX` node** (no pixel conflicts at all): every worker
  paints `{opacity: 0, record: true}` and returns only `sid`s, optionally bundled with
  `POST /api/stroke_group`; the coordinator comma-joins them into one `Stroke FX` → `Layer out`.
  Contributions stay procedural, retunable and individually removable; the cost is FX rendering
  instead of native deposit.

Discipline for any of them: a distinct stable `X-User` and `X-Client` per worker; everyone
holds `/api/events` open and refreshes on foreign `rev`; long evaluations through `/api/graph/run`;
checkpoint as `.lews` between phases; verify with `POST /api/analyze` regions, not by looking.
Coordination transport lives outside the studio: the leCore service bus (`POST /bus/publish|
poll|history`) and roles (`mind.role(name, topic, handler, emit=)`), the shared memory
(`teach`/`ask`, `memory_write`/`memory_search`) for notes about the painting — the agent is the
bridge in both directions. Divide by subject (a reference scout, a background painter, a
finisher), give each a fixed region and a metric, and let the leCore boot ritual's contract
(ask memory first, find_capability before hand-rolling, teach as you go) govern the workers.

## 10. Finishing a leCore render

The engine's post chain already did exposure, bloom, ACES, CA, vignette, grain. The studio pass
does only what the chain cannot: `Media in` the render and its glass/floor/background masks;
DoF on the backdrop (`Blur` sigma 6 through the background mask); a haze ramp on the far floor
(`Gradient` → `Color ramp` × floor mask → `Mask mix` into a warm `Solid`); a restrained inner
light through the glass mask (`Grade` + `Glow` screened, a violet halo from a sigma-18 blur of
the mask) sized to how dark the crystals ARRIVE — the dose that suited deep-absorbing spires
washed brighter ones to pink-white at the same setting; `Light shafts` weight 0.10; `Post FX`
bloom 0.04, grain 0, vignette 0, chroma 0. Finish at 1× and let the engine upscale last.

## 11. Sharp edges (measured)

- `POST /api/graph/apply` reads `id`, not `node`. `GET /api/export.png` ignores `?w=&h=` and
  `?layer=` — use `/api/graph/render.png` and `/api/layer/<lid>.png`. `/api/mask` has no
  `from_selection`; `/api/brush` and `/api/spline` use `remove`, not `delete`; `/api/live` takes
  `action`, not `on`. The README says 35 operators; there are 105.
- `/api/new` returns the document only; layers come from `/api/state`; a fresh server already
  holds "Untitled 1". `/api/open` puts an uploaded image in a layer named after the file at
  index 1. Baking onto an existing layer replaces its pixels. `state.strokes` is the last 64,
  summaries only.
- NaN is the recurring defect: it never raises, it propagates, and it returned 200 into pixels
  and height maps. Any new number-taking endpoint needs `_finite()`.
- Harness errors outnumbered real bugs in the studio's own scenario rounds (5 vs 7): check the
  request shape (`/api/select` takes `tool`+`params`), the `warning` field, and whether an undo
  removed the layer, before filing a bug.
- Editing `index.html`: never put an emoji in a patch string (a surrogate pair truncated the
  file to zero bytes, twice); always `io.open(..., encoding="utf-8")` both ways; balance-check
  tags; the entry point stays at the end of `server.py`.
- Deliverables follow the user's standing rule: one zip of only the changed/new files, paths
  relative to the repo root, the `.lews` alongside any PNG that matters, no commits or pushes.

## Painting from numbers with a swarm: the benchmark-poster recipe (Sep 2026, measured)

`bench_poster_swarm.py` (in the leCore zip under `docs/research/evidence/`) is the worked example: four
workers with their own `X-User` / `X-Client`, the region map, palette and layer ids TAUGHT into the
leCore partition and read back with `ask` by every worker, the design choices (ground, accent) as typed
decisions with `reflex=True` so a repeat brief is answered from experience, every worker step published
with `swarm_step` carrying `/api/analyze` evidence, `swarm_evaluate` as the exit. 1,705 strokes, 28 s.

- Bars are combs of horizontal strokes (`fill_rect`: one stroke per 3 px, radius 0.75 × step); the studio
  deposits paint, it does not fill polygons. A curve is a polyline of ≥8-point segments; a marker is a
  25-point circle stroke.
- Labels are a PIL PNG of light text on BLACK, loaded with `Media in` and baked with `/api/graph/apply`,
  then the layer set to blend `add`. Never `screen` a `Layer` node in the graph — transparent pixels read
  as white and the sheet washes out.
- **Verify text by GEOMETRY, not by looking.** `/api/analyze` brightness cannot see a subtitle colliding
  with the title or a panel title running into its neighbour — the first run had both. Measure every text
  box with `ImageDraw.textbbox` in the SAME font and anchor the draw uses, wrap titles to the panel's
  pixel width (`textlength`), and report `text_overflow` as step evidence; the second run reported 0.
- Each worker's done_when is a region brighter than the ground (or an accent hue present) by
  `/api/analyze` with `"metrics":["brightness","dominant_hue"]`, and `text_overflow == 0` for the labels
  worker. A step without evidence is refused by `swarm_step`.
- Save `/api/composite.png` and `/api/workspace.lews` together; the `.lews` carries the layers.


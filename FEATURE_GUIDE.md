# leCore Feature Guide

A hands-on guide to the features added most recently — composable materials and textures, the describe-a-scene
authoring flow (naming, texturing, external files), external-asset relocation and the queryable file map, the
message-bus + optional-agent harness, the opt-in language layer (dictionary + semantic word search), and cold storage
for compressing inactive data.

Everything here is reached through one object, the `UnifiedMind`:

```python
from holographic.misc.holographic_unified import UnifiedMind
mind = UnifiedMind(dim=1024, seed=0)   # dim = hypervector width; seed keeps everything deterministic
```

Three ground rules worth knowing up front:

- **Deterministic.** Same inputs + same seed → same output, every run (ids come from `hashlib`, never Python's `hash()`).
- **Opt-in / lazy.** The heavy parts (the dictionary, a semantic index, an image decoder) load only when you actually
  use them, so importing the library to build on top of it costs you nothing.
- **Honest.** Where a feature is approximate or has a known limit, this guide says so plainly under *"Kept limits"*.

If you ever forget which call does what, ask the engine in plain English:

```python
for home in mind.find_capability("paint a texture onto an object and render it"):
    print(home.name)
```

---

## 1. Composable materials and textures

A texture is built as a small **graph of operations** over typed inputs — leaves (procedural sources or constant
colours) feed operators (`mix`, `multiply`, `scale`, `over`, `saturate`, …) that feed other operators. The whole graph
is type-checked when you compose it, not when you render.

```python
# leaves: a procedural source, or a constant colour (colour NAMES work too)
noise  = mind.texture_leaf("fbm", n_dims=2, seed=0)     # a fractal-noise field
orange = mind.texture_leaf(value="orange")
purple = mind.texture_leaf(value="purple")

# an OP node blends the two colours by the noise field. NOTE the name: texture_OP, not texture_map --
# texture_map is the older image-based texture; texture_op builds a procedural graph node.
tex = mind.texture_op("mix", a=orange, b=purple, t=noise)

rgb = mind.sample_texture(tex, [0.3, 0.7])              # sample the graph at a UV -> an rgb value
```

Four "costumes" wrap the same composition machinery for common jobs:

```python
# CMP3 -- blend whole MATERIALS by per-point masks (weights become a partition of unity)
blend = mind.multi_material([mat_a, mat_b], weights=[w_mask, 1.0], mode="mask")

# CMP2 -- LAYER materials in a fixed order (base < diffuse < specular/reflection < coat); the order is
# schema-checked at compose time, so you can't stack a base coat on top of a clear coat by mistake
layered = mind.layered_material([base_layer, coat_layer])

# CMP4 -- INSTANCE shared geometry: define once, place many; edit the definition and every instance updates
scene = mind.instanced_scene(definitions, instances)

# CMP5 -- a RENDER GRAPH that decides what to bake once vs sample live, then prepares a scene to render
rg = mind.render_graph()
```

**Preview** a texture or material without a full render:

```python
swatch  = mind.preview_texture(tex)          # a flat RGB thumbnail of the texture
matball = mind.preview_material(material)     # a Cook-Torrance-shaded sphere of the material
```

**Kept limits.** A texture's `sample` is a cosine read-out of a vector field — it's direction/scale-normalised, so tune
by ratios, not absolute values. Layer *ordering* is a correctness rule, not energy-conserving radiometry.

---

## 2. Describe a scene, name it, texture it

Describe a scene in plain words; adjust it by talking to it.

```python
scene = mind.build_scene("a big red metal sphere and a small blue box on a sunny day")
scene.adjust("make the sphere bigger")
scene.adjust("change the box to metal")
img = scene.render(width=320, height=240)    # an (H, W, 3) image in [0,1]
```

**Name objects** so you can refer to them easily. A nickname always wins over description-matching, so once you name
something you can always reach it:

```python
scene.name("the red sphere", "hero")         # give it a nickname
scene.adjust("make hero glass")              # reference it by that nickname
scene.adjust("rename hero to champion")      # rename in plain English
scene.adjust("call the box crate")           # name a second object
print(scene.labels())                        # {'champion': 'red big glass sphere', 'crate': 'blue small box'}
```

**Texture objects** with built-in procedural textures — by talking to the scene, or via the API:

```python
scene.adjust("give champion a rusty texture")   # rusty / marbled / mossy / cloudy / lava / striped / noisy
scene.adjust("make the box mossy")
scene.paint("crate", "marbled")                 # the same thing through the API
img = scene.render()                            # render() automatically paints attached textures on
```

**Attach an EXTERNAL image file** as a texture (see §3 for what happens when those files move):

```python
scene.attach_texture_file("the sphere", "project/textures/wave.png")
img = scene.render()                            # loads the file's pixels and paints them on
```

**Kept limits.** UV mapping is the textbook kind (a seam + pole pinch on a sphere, face seams on a box); the fast render
uses a single hard light — reach for the path tracer (`render(quality="hyperreal")`) for soft shadows / GI. External
image decoding uses PIL, imported *lazily* only when you actually draw an image file, so the core stays NumPy-only.

---

## 3. External assets and the queryable file map

### 3a. Track external files and repair paths when they move

Real pipelines move folders around and break every reference at once. An `AssetLibrary` fixes them the way you'd reason
about it: re-point **one** file, and it works out the parent that moved and re-finds the rest.

```python
lib = mind.asset_library()
lib.add("/proj/textures/water/wave.png")
lib.add("/proj/textures/stone/wall.png")
lib.add("/proj/models/boat.obj")

# ... the whole /proj folder was moved to /work/proj ...
print(len(lib.missing()))                      # 3 -- all broken

# re-point ONE; the other two are found automatically (shared moved-parent + a structural search)
report = lib.relink(lib.assets[0], "/work/proj/textures/water/wave.png")
print(len(lib.missing()))                      # 0
```

Know when an external file was **edited on disk**:

```python
for ref in lib.changed():                      # size/mtime (cheap) or content hash (definitive)
    print("re-import:", ref.path)
    ref.refresh()                              # acknowledge it
```

**Distributed / cross-machine.** Absolute paths differ per machine, so identify files by **content hash** and resolve
them wherever they landed:

```python
lib.add_hashes()                               # record a content hash per file (do this once)
path = lib.resolve(some_ref, roots=["/mnt/shared/assets"])   # finds it by content, not by path
lib.save("assets.json")                        # a portable JSON manifest
```

A `SemanticScene` carries its own library, so external textures self-heal at render time:

```python
scene.set_asset_roots(["/work/proj"])          # where to search if files moved
scene.resolve_assets()                         # re-find any missing files
img = scene.render()                           # missing files fall back to the object's colour -- never crashes
```

**Kept limits.** The path logic is POSIX-tested; Windows drive letters are handled simply, not exhaustively.

### 3b. Digest a folder or zip into a queryable file map

Point at a folder, a `.zip`, or a single file and get back a `FileMap` you can query five ways:

```python
fm = mind.ingest_files("my_project.zip")       # a folder, a .zip, or one file

fm.find("*.png")                               # by NAME / glob
fm.by_kind("model")                            # by KIND: image / text / model / data / code / archive / other
fm.larger_than(1_000_000)                      # by METADATA (also newer_than, by_ext)
fm.search_text("normal caustic")               # by text CONTENT (an inverted index over the text/code files)
fm.tree()                                       # the folder hierarchy as nested dicts -- the "file map"

# by MEANING (opt-in: builds a small vector index over the text, then searches by description)
fm.build_meaning_index()
fm.find_by_meaning("lighting setup")
```

Every ingested file is tracked in a built-in `AssetLibrary`, so the same map self-heals: `fm.missing()`, `fm.changed()`,
`fm.relink(one, new)`, `fm.resolve_assets(roots)`.

**Kept limits.** Text indexing reads only text/code kinds under a size cap (so a pile of big binaries stays cheap).
Meaning search is approximate random-indexing — reliable for the top hits, noisy in the tail.

---

## 4. The agent harness: a message bus + an optional LLM

leCore is the core of an AI-substrate harness: a person and an agent can both be attached to the running tool, and the
app **pushes** to the agent instead of the agent polling. It's built on a small message bus.

```python
bus = mind.bus()                               # a topic-based message bus (publish / subscribe / mailboxes / history)
bus.subscribe("render.*", lambda m: print("event:", m.topic))
bus.publish("render.start", {"w": 320})
```

Connect an **optional** agent — any callable `text -> reply` (your wrapper around any model; no LLM library is imported,
so this is entirely optional and the app runs fine with no agent attached):

```python
bridge = mind.agent_bridge(llm=my_llm_function)     # llm=None also works -- events just get logged
bridge.notify_on("render.done", "Does this look right?")   # PUSH to the agent when a render finishes
bridge.on_reply(lambda m: print("agent said:", m.payload["reply"]))
answer = bridge.ask("what can you do?")             # ask it directly
```

Run any job as a **task** that announces itself — this is the "check after the render is done" pattern, with no polling:

```python
# runs the render in the background and publishes 'render.done' (with a small summary) when it finishes;
# the bridge above then calls the LLM automatically
mind.run_task("render",
              lambda: scene.render(width=640, height=480),
              background=True,
              summarize=lambda img: {"shape": list(img.shape)})   # an LLM can't read a NumPy image, so hand it a summary
```

Over HTTP, a remote agent uses `POST /bus/publish` and `POST /bus/poll` (its inbox) — see `SERVICE.md`.

**Kept limits.** The push side is in-process (a callback) or a pulled HTTP inbox; there is no live server-push (SSE/
websocket) yet.

---

## 5. The language layer: dictionary + semantic word search

A ~144k-word English dictionary (Princeton WordNet) gives the engine real-world grounding. It is **opt-in and lazy** —
it never loads from importing leCore or building a mind, only from the first language call, then it lives in RAM as a
plain dict (fast lookups).

```python
mind.lookup("gravity")                         # {'definition': ..., 'pos': ..., 'synonyms': [...], ...}
mind.word_taxonomy("dog")                       # 'a dog is a kind of ...'

import holographic.misc.holographic_dictionary as hd
hd.stats()                                      # {'loaded': False, 'source': 'dictionary.json.xz', ...} -- reading this does NOT load it
hd.preload()                                    # force the one-time load at startup (optional)
hd.unload()                                     # drop the ~22 MB back; the next lookup transparently reloads
```

Search the dictionary by **meaning** (the fuzzy reverse of a lookup) — opt-in, since it builds a vector index:

```python
idx = mind.build_semantic_index(words=my_vocab)     # or words=None for the whole dictionary (~150 MB at dim=256)
idx.find("unexpected good luck")               # -> 'serendipity'
idx.similar("puppy")                           # -> 'dog', 'kitten'
```

**Kept limits.** The dictionary itself is exact. The semantic index is approximate random-indexing over one gloss per
word — great for the top hit, noisy in the tail, and word-sense sensitive (it only sees the single stored sense).

---

## 6. Cold storage: compress inactive data, inflate on demand

A long-running app holds a lot of *idle* data — tables nobody has queried lately, a database belonging to another
session, a cache you built once. Cold storage folds those up (serialize + compress, freeing the live object) and
unfolds them transparently the next time something touches them. Nothing is lost; it's the same object, just compressed
while it wasn't needed.

Wrap **one** value:

```python
c = mind.cool(big_table)          # or codec="lzma" for a smaller blob, spill_dir="/tmp/cold" to write it to disk
c.cool()                          # serialize + compress + free the live object's RAM
big_table = c.get()               # bit-identical, inflated on access
print(c.ratio())                  # cold / warm size -- smaller is better
```

Or bound memory across **many** values with an auto-cooling store — it keeps only the K most-recently-used live and
compresses the rest, warming any of them the instant you `get()` it:

```python
store = mind.cold_store(keep_warm=8)     # at most 8 stay warm
for name, table in my_tables.items():
    store.put(name, table)
t = store.get("orders")                  # if it was cold, it's transparently warmed here
print(store.stats())                     # {'warm': 8, 'cold': N, 'cold_bytes': ..., 'approx_saved_bytes': ...}
```

It works on anything picklable — a `Table`, a whole `Database`, a big NumPy array, an ordinary dict. `spill_dir=...`
writes cold blobs to a file so even the compressed bytes leave RAM.

**The query Database can auto-cool its own idle tables.** Turn it on (it's off by default), and tables you haven't
queried lately compress; the next query warms them back transparently:

```python
db.enable_cold_storage(keep_warm=8)   # keep the 8 most-recently-used tables warm
db.cool_idle()                        # compress the rest (call this when the DB is idle -- no query in flight)
db.resolve("app.orders")              # a query warms a cold table automatically
db.cold_stats()                       # {'warm': ..., 'cold': ..., 'cold_bytes': ..., 'enabled': True}
```

This is **safe in distributed compute**: if a cold-enabled database is shipped to a worker (used as a shared read-only
cache), it arrives *warm with cooling disabled* — a plain, immutable copy — so a worker's reads never mutate the shared
cache, and the lock and any spill-file paths never cross the process boundary. Cool on the long-lived main node to save
memory; workers get safe warm copies. (Cool only when idle: cooling swaps a table for its compressed form, and warming
later builds a fresh object, so doing it mid-transaction could strand a live reference.)

**Kept limits.** How much you save depends entirely on the data. Redundant / text / structured data compresses a lot
(a repetitive array can drop to ~0.1% of its size). But leCore's **VSA record vectors are near-random (high-entropy),
so they barely compress** — there the real win is freeing the live Python object and (optionally) spilling the blob to
disk, not the compression ratio. And because it uses `pickle`, only cool data your own app produced — never thaw a blob
from an untrusted source.

---

## 7. Importing artist file formats

Bring in the files artists actually hand you. One dispatcher, `mind.import_asset(path)`, picks by extension; or call the
specific loader.

```python
# Wavefront OBJ (+ its .mtl): positions, per-corner UVs/normals, the material each face uses, and the materials
# themselves (Kd/Pr/Pm factors + map_* textures loaded)
lm = mind.load_obj("chair.obj")
lm.positions        # (Nv, 3)
lm.faces            # (Nf, 3) triangles (polygons are fan-triangulated)
lm.materials        # {name: PBRMaterial}
mesh = lm.mesh()    # a plain engine Mesh for the geometry pipeline

# glTF / GLB: geometry AND its PBR materials (base colour / metallic-roughness / normal / occlusion / emissive),
# per-vertex UVs and normals, embedded textures, and -- for rigged models -- animation and skinning
glb = mind.load_glb("robot.glb")
mat = list(glb.materials.values())[0]     # .base_color, .metallic, .roughness, .base_color_map, .normal_map, .ao_map...
glb.uv                                    # per-vertex UVs (TEXCOORD_0), or None
glb.normals                               # per-vertex normals, or None

# rigged/animated glTF: keyframed node transforms + the skeleton
for clip in glb.animations:               # each is an AnimationClip
    print(clip.name, clip.duration)       # e.g. "Walk 1.20"
    pose = clip.sample(0.5)               # {node_index: 4x4 local matrix} at t=0.5s (rotations SLERPed)
glb.skins                                 # [{'joints': [...], 'inverse_bind': (J,4,4)}] -- the skeleton
glb.joints, glb.weights                   # per-vertex skin binding (JOINTS_0 / WEIGHTS_0), or None

# DEFORM the rig -- make it actually move. Morph-blends the base shape (if it has blend shapes) then applies
# linear-blend skinning by the posed skeleton, returning the deformed mesh at time t.
posed = mind.deform_mesh(glb, clip=glb.animations[0], t=0.5)   # a Mesh with vertices moved to the pose at t=0.5s
rest = mind.deform_mesh(glb, clip=None)                        # the rest pose (no animation)

# A folder of maps exported from Adobe Substance 3D Painter (or any tool) -> one PBRMaterial. Maps are matched by
# file name: basecolor / roughness / metallic / normal / height / ao / emissive.
brick = mind.load_texture_set("exports/brick")
brick.channels_found    # e.g. ['ao', 'base_color', 'height', 'metallic', 'normal', 'roughness']

# A volumetric density grid -> a field the volume renderer marches
field, bounds = mind.load_volume("smoke.npy")          # or raw floats: load_volume("d.raw", dims=(nx,ny,nz))
img, alpha = mind.render_volume(field, camera, bounds, mode="smoke")
```

**Kept limits (stated plainly).** We import the *open, exported* forms. The proprietary project files need their
vendor's engine and are **not** parsed: Substance's `.sbsar` / `.spp` (export the texture maps from Painter instead),
and OpenVDB's sparse `.vdb` (export a dense `.npy`/`.raw` grid, or convert with the OpenVDB tools — `load_volume`
refuses a `.vdb` rather than guessing). Image decoding uses PIL, imported lazily only when a texture is actually
loaded, so the core stays NumPy-only; a texture that can't be found becomes `None` and the factor-level material still
works. OBJ handling covers the common case (v/vt/vn/f/usemtl/mtllib, fan-triangulated polygons); exotic OBJ features
are ignored, not errored. The deformer applies **linear-blend** skinning (the standard method; it has the classic
candy-wrapper collapse at extreme twists that dual-quaternion skinning avoids — not implemented) and blends morph
targets; it uses the first skin and moves positions (normals aren't re-skinned). OBJ carries no animation.

---

## 8. Unlabeled data exploration: demux, scaffold, decompose, reunite

Hand the engine a raw stream -- no labels, no schema -- and get back the sources,
the primary axis, the generating laws, and the leftovers. Every stage returns its
evidence (score tables, correlation matrices, merge tolerances), and every verdict
is decided by measurement: noise is never dressed as law.

```python
import numpy as np
import lecore

mind = lecore.UnifiedMind(dim=256, seed=0)

# --- One interleaved stream, two sources (the "Contact" protocol, zero hints).
u = np.linspace(0, 1, 200)
stream = np.empty(400)
stream[0::2] = np.sin(2 * np.pi * 2 * u)      # a lawful harmonic
stream[1::2] = 0.8 * u + 0.1                  # a lawful trend
report = mind.explore_series(stream, auto_demux=True)
print(report["demux"]["stride"], report["verdict"])   # 2 structured
```

The stride is FOUND, not assumed: at the true interleave every strided sub-stream
is smooth; deinterleaving is a permutation, so recovery is bit-exact.

```python
# --- A multi-channel series: which channels move together (which are one object)?
motion = np.sin(2 * np.pi * 1.5 * u)
series = np.stack([motion, 0.7 * motion, -0.4 * motion,   # one "mesh" (mirror incl.)
                   np.cumsum(np.random.default_rng(0).standard_normal(200)) * 0.1],
                  axis=1)                                   # an unrelated walker
d = mind.demux_series(series)
print(d["groups"])                            # [[0, 1, 2], [3]]
```

```python
# --- Packetized bursts (no cyclic stride): boundaries by statistics shifts,
#     sources by noise-calibrated assignment, drift reunited by continuation.
rng = np.random.default_rng(2)
x = np.concatenate([0.02 * np.arange(60),               # a ramp...
                    8.0 + rng.standard_normal(120),      # ...a loud burst...
                    0.02 * np.arange(180, 240)])         # ...the ramp resumes on trend
pk = mind.packet_demux(x, min_seg=24, continuation=True)
print(pk["n_sources"], pk["continuation_merges"][0]["predicted"])  # 2 3.6
```

```python
# --- The full loop on a bare cube: scaffold -> rectify -> decompose -> residuals.
t_irr = np.cumsum(np.random.default_rng(0).exponential(1.0, size=200))
uu = (t_irr - t_irr[0]) / (t_irr[-1] - t_irr[0])
cube = np.stack([np.sin(2 * np.pi * 2 * uu), 0.8 * uu + 0.1], axis=1)
res = mind.explore_series(cube, coords={0: t_irr})
print(res["scaffold"], res["verdict"],
      [round(c["explained_fraction"], 2) for c in res["channels"]])  # 0 structured [1.0, 1.0]
```

Beneath the orchestrator, each stage is its own faculty: `analyze_axes` (which
axis is the carrier -- boring AND organising), `rectify_carrier` (repair a
wobbling axis by the arc-length lift), `winding_map` (a largely-reversing carrier:
function / hysteresis / path, with merging refused exactly where it would
fabricate), `analytic_signal` (rotation kinematics), `identify_dynamics` (masses
and force laws behind a gauge-breaking channel), `cross_channel_links` (delayed
copies the per-channel view cannot see), and `diagnose_scaling` / `auto_scale`
(which knob to double when a stage hits a limit). Ask
`mind.find_capability("explore unlabeled data")` for the live menu.

## 9. Physics, astronomy, polarization & code (the merged arc)

A later merge added five families. Each is field-native, deterministic, and wired into `mind`; ask
`mind.find_capability("...")` for the live menu. Every snippet below runs as written.

```python
import numpy as np, lecore
mind = lecore.UnifiedMind(dim=256, seed=0)

# --- Quantum: a wavefunction evolved UNITARILY by the split-operator Schrodinger solver.
qf = mind.quantum_field((64, 64)); qf.gaussian_packet((20, 32), (4, 4), (2.0, 0.0))
sol = mind.quantum_solver(qf); sol.run(5, 0.1)          # norm is conserved to machine precision
# quantum_dot_well / quantum_solenoid_A build a scatterer or an Aharonov-Bohm ring;
# probability_current gives the flow (and quantum_velocity feeds advect_field -- the sideways reuse).

# --- Gravity: an N-body sim with a symplectic integrator (energy stays bounded) + closed-form Kepler orbits.
vc = mind.circular_orbit_velocity(1000.0, 1, 1.0)
r = mind.nbody_simulate(np.array([[0., 0.], [1., 0.]]), np.array([[0., 0.], [0., vc]]),
                        np.array([1000., 1.]), 0.001, 500, G=1.0, softening=1e-4, record_every=10)
print(r["energy_drift"])                                 # ~0; r["trajectory"] scrubs through mind.transport(...)

# --- Astronomy: assemble a star system, a cluster (Salpeter IMF), or a volumetric nebula.
sy = mind.star_system({"star": {"temp_K": 5772}, "planets": [{"a": 1.0, "e": 0.02, "radius": 0.09, "temp_K": 288}]})
print(sy["planets"][0]["biome"])                         # 'temperate'
vol = mind.nebula_volume(res=16, seed=0)                 # a 3-D density field; mind.nebula_field_fn feeds render_volume

# --- Period finding on gappy data (Lomb-Scargle) -- what a plain FFT cannot do.
rng = np.random.default_rng(0); t = np.sort(rng.uniform(0, 20, 120)); y = np.sin(2 * np.pi * t / 2.5)
print(mind.best_period(t, y, min_period=0.5, max_period=8)["period"])   # 2.5

# --- Polarization: the SAME Stokes core reads a mantis eye AND a radio telescope.
lam2 = np.linspace(0.03, 0.24, 160); P = 2.0 * np.exp(2j * (0.3 + 42.0 * lam2)); phi = mind.rm_phi_grid(lam2)
print(mind.rm_peak(mind.rm_synthesis(lam2, phi, P=P), phi)["rm"])       # 42.0 (Faraday depth recovered)
# stokes_linear/mueller_matrix/apply_mueller do the optics; observe_spectrum turns a spectrum into sensor
# readings (a human eye reproduces blackbody_rgb exactly); mantis_view + mantis_falsecolor show 12 bands + handedness.

# --- Code: describe a kernel in English -> one Python IR -> emit Zig / WGSL / C from the SAME IR.
k = mind.kernel_from_description("a sphere of radius 1", dialect="python")
print("fn " in str(mind.translate_kernel(k, "python", "zig_f64")))     # True
# triage_code makes honest structural observations of unknown code; explain_code is a deterministic description.
# Zig native kernels (zig_batch_eval) are OPT-IN like numba -- they raise a clear error without the ziglang wheel.
```

## Where to look next

- **`mind.find_capability("...")`** — ask the engine, in plain English, which call does what.
- **`CAPABILITIES.md`** — the full menu of capability "homes" (auto-generated from the catalog).
- **`API_QUICKREF.md`** — one scannable line per public function (auto-generated).
- **`SERVICE.md`** — the HTTP endpoints, including the message bus.
- **`tour.py`** — a runnable tour that exercises these features end to end and prints what each one did.

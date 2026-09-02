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

## Building a game: the authoritative world shard

Every ingredient of a game already lives in the engine (rigid bodies, CCD, spatial hashing, the
distributed farm and bus, fork/merge worlds, durability). The `game_shard` faculty is the
composition: an authoritative, **deterministic** fixed-tick world fed by an ordered player-command
queue. Same command log, same sha256 digest -- deterministic lockstep verification for free.
Clients pay only for what's near them (`snapshot(center, radius)`), sync with stateless
`delta_since(baseline)` diffs, and a `region` box reports departing entities so a massive world
can be sharded across the farm. `run_game_shard` is the same thing as one JSON-in/JSON-out call.

```python
import lecore
m = lecore.UnifiedMind(dim=256, seed=0)
shard = m.game_shard(seed=0, gravity=(0, -9.8, 0), region=((-50, -50, -50), (50, 50, 50)))
shard.submit({"tick": 0, "player": "alice", "seq": 0, "op": "spawn", "id": 1, "pos": (0, 10, 0)})
shard.submit({"tick": 0, "player": "bob",   "seq": 0, "op": "spawn", "id": 2, "pos": (0.4, 10.6, 0)})
shard.submit({"tick": 5, "player": "alice", "seq": 1, "op": "impulse", "id": 1, "j": (3, 0, 0)})
shard.step()
base = shard.snapshot()
for _ in range(29):
    out = shard.step()
print("tick", out["tick"], "digest", out["digest"][:12])
print("alice sees", shard.snapshot(center=(0, 5, 0), radius=20)["ids"])
print("moved since baseline:", [e["id"] for e in shard.delta_since(base)["moved"]])
r = m.run_game_shard([{"tick": 0, "player": "a", "seq": 0, "op": "spawn", "id": 9, "pos": (0, 0, 0)}], ticks=3)
r2 = m.run_game_shard([], ticks=3, state=r["state"])
print("resumed to tick", r2["state"]["tick"])
```

### Scaling it: the sharded world

`game_world` turns the single shard into a lazy grid of them -- cost tracks *occupied* cells, not
world size. Entities that cross a cell boundary migrate deterministically (exact velocity and mass
carried over), snapshots span the seams, and `tick(collect_only=True)` + `receive()` are the
bus-transport pair for spreading shards across the distributed farm -- identical payloads either
way.

```python
world = m.game_world(cell=4.0, dt=0.1, seed=0)
world.spawn(1, (3.5, 1, 1), vel=(2, 0, 0))
world.spawn(2, (1.0, 1, 1))
migrated = []
for _ in range(5):
    out = world.tick()
    migrated += out["migrated"]
print("shards:", len(world.shards), "migrated:", migrated, "digest:", out["digest"][:12])
print("seam-free AOI:", sorted(world.snapshot(center=(4, 1, 1), radius=6)["ids"]))
```

### Plugging into the distributed system (the layering)

The distributed stack is the **data layer** -- the coordinator's own rule is that non-monoid
feedback work (and a game tick is exactly that) runs *whole on one worker*; the bus moves
messages; presence says who's alive. The game world is the **interaction layer**. `game_bus_host`
is the entire handshake between them: each node owns a set of world cells and exchanges entity
handoffs over per-cell bus topics. Local `MessageBus` and cross-machine `DistributedBus` are the
same call -- swap the bus, keep the game.

```python
from holographic.scene_and_pipeline.holographic_distbus import MessageBus
bus = MessageBus()
wa, wb = m.game_world(cell=4.0, dt=0.1), m.game_world(cell=4.0, dt=0.1)
node_a = m.game_bus_host(bus, wa, [(0, 0, 0)], world_id="demo")
node_b = m.game_bus_host(bus, wb, [(1, 0, 0)], world_id="demo")
wa.spawn(1, (3.5, 1, 1), vel=(2, 0, 0))
for _ in range(6):
    node_a.tick(); node_b.tick()
print("entity 1 now lives on node B:", node_b.world.owner.get(1))
```

### Watching it from a browser (the three.js seam)

`POST /game` creates a room and routes player commands; `GET /game/stream` is an SSE push of
per-client **deltas** -- first event is the full area-of-interest as `added`, later events only
what changed. That's the wire format a three.js client feeds straight into its scene graph:

```js
const es = new EventSource("/game/stream?world=demo&session=me&target_fps=30&cx=0&cy=0&cz=0&r=50");
es.onmessage = (m) => {
  const d = JSON.parse(m.data);
  d.added.forEach(e => scene.add(makeSphere(e)));
  d.moved.forEach(e => meshes[e.id].position.set(...e.pos));
  d.removed.forEach(id => scene.remove(meshes[id]));
};
fetch("/game", {method: "POST", body: JSON.stringify({world: "demo",
  cmds: [{player: "me", seq: 1, op: "impulse", id: 1, j: [5, 0, 0]}]})});
```

`advance=1` (default) makes the stream the world's designated clock; run extra viewers with
`advance=0`. Start the service with `serve(threads=True)` -- an open stream must never block input.

## 10. The substrate for agents: ask first, remember forever, share wisely

Everything in this section is what an agent harness (openzoo, an MCP client, a script)
should reach for *before* asking a model. Every block here opens with `# guide-check`
and runs verbatim in CI (`tests/test_guide_examples.py`); blocks elsewhere in this guide
are fragments that build on names introduced by their surrounding prose.

**The front door.** `orient()` is generated live from the catalog, so it cannot rot like
a static skill file: the five-move workflow and the top three existing doors for a topic.

```python
# guide-check
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
o = mind.orient(topic="merge two directory trees")
assert "merge" in o["directed_to"][0]["name"].lower()
print(o["workflow"][0])          # serve(query) -- ask before anything; escalation is honest
```

**Ask before anything.** `serve()` answers from memory, then from a taught *tool reflex*
(a learned API called with deterministically extracted arguments), then escalates with
the reason. The model is called only when the substrate cannot serve.

```python
# guide-check
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
mind.teach("what is the boiling point of water at sea level", "100 C, 212 F")
assert mind.serve("what is the boiling point of water at sea level")["via"] == "memory"
r = mind.serve("what colour is a quark")
assert not r["served"] and "escalate" in r["via"] and r["reason"]
```

**Virtually limitless context, with citations.** `study(root)` digests a directory and
answers with the source file (and symbol, for code) or refuses off-corpus honestly.

```python
# guide-check
import lecore, os, tempfile
d = tempfile.mkdtemp()
open(os.path.join(d, "law.md"), "w").write("# Tension\n\n" + "The rope tension law states pull scales with the winch drum radius. " * 4)
st = lecore.UnifiedMind(dim=256, seed=0).study(d)
a = st["ask"]("what does the rope tension law state")
assert a["answerable"] and a["citations"][0].endswith("law.md")
assert not st["ask"]("recipe for banana bread")["answerable"]
```

**Wisdom that outlives the model, and a commons that respects privacy.** `bequeath` records
a lesson with the author's name in its provenance; `wisdom` inherits it in any later mind.
`contribute` screens a user's shared rows (session-salted rows never leave; path, email,
phone and key shapes are rejected with reasons on a review sheet); `commons_pool` merges
bundles with conflicts flagged.

```python
# guide-check
import lecore, os, tempfile
t = tempfile.mkdtemp()
a = lecore.UnifiedMind(dim=256, seed=0)
a.teach("what is the derivative of x squared", "2x")
a.teach("my email", "moose@example.com")
a.bequeath("write the failure down louder than the win", author="model-a", topic="discipline")
sheet = a.contribute(os.path.join(t, "a"), author="user-a")
assert sheet["kept"] == 2 and any("email" in why for _q, why in sheet["rejected"])
pool = a.commons_pool([os.path.join(t, "a")], os.path.join(t, "commons"))
b = lecore.UnifiedMind(dim=256, seed=0)
b.memory_import(os.path.join(t, "commons"))
assert b.ask("what is the derivative of x squared")["tier"] == "T0"
assert b.wisdom()["authors"] == ["model-a"] and b.ask("my email")["tier"] != "T0"
```

**Non-LLM backends.** `api_learn(spec)` turns an OpenAPI spec -- a forecaster, a robot's
status endpoint -- into callable, discoverable tools that survive save/load; `api_use`
calls them. A tool reflex then lets `serve()` answer from the tool with no model call.

```python
# guide-check
import lecore, json, threading, http.server, socketserver, time, tempfile
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
        body = json.dumps({"fahrenheit": float(req.get("celsius", 0)) * 9 / 5 + 32}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
srv = socketserver.TCPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start(); time.sleep(0.2)
spec = {"openapi": "3.0.0", "info": {"title": "convertd", "version": "1"}, "servers": [{"url": "http://127.0.0.1:%d" % srv.server_address[1]}],
        "paths": {"/c2f": {"post": {"operationId": "c_to_f", "summary": "celsius to fahrenheit", "responses": {"200": {"description": "f"}}}}}}
mind = lecore.UnifiedMind(dim=256, seed=0)
mind.api_learn(spec, name="convertd")
mind.tool_reflex_teach("convert 25 celsius to fahrenheit", "convertd", "c_to_f", extract_numbers=["celsius"])
r = mind.serve("convert 100 celsius to fahrenheit")
assert r["via"] == "tool-reflex" and r["result"]["fahrenheit"] == 212.0
root = tempfile.mkdtemp(); mind.learning_save(root)
again = lecore.UnifiedMind(dim=256, seed=0); again.learning_load(root)
assert again.serve("convert -40 celsius to fahrenheit")["result"]["fahrenheit"] == -40.0   # the reflex survived a restart
srv.shutdown()
```

**Lean partitions, honest fallbacks.** `learning_save(root, audit="regen")` drops the audit
arrays and replays taught text on load (~195x smaller for pure-taught minds); when a mind
holds non-taught rows the guard falls back and `audit_regen_reason` says why.
`partition_report(root)` tells you where the bytes went.

```python
# guide-check
import lecore, tempfile, os, glob
mind = lecore.UnifiedMind(dim=256, seed=0)
for i in range(30):
    mind.teach("lean fact %d" % i, "answer %d" % i)
root = tempfile.mkdtemp()
r = mind.learning_save(root, audit="regen")
assert r["audit_regen"] is True
assert os.path.getsize(glob.glob(os.path.join(root, "**", "state.lecore"), recursive=True)[0]) < 5000
back = lecore.UnifiedMind(dim=256, seed=0); back.learning_load(root)
assert back.ask("lean fact 7")["answer"] == "answer 7"
```

**Merging trees without losing anything.** `merge_trees` gives a sha census with verdicts
(`ours_is_base`, `theirs_is_base`, `both_changed`) and applies only the unambiguous ones.
The lesson on record from a real merge: after *any* three-way merge, census definitions
and signatures against the pre-merge tree -- a "clean" diff3 will silently honour the
other side's deletions.

```python
# guide-check
import lecore, os, tempfile, shutil
a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
open(os.path.join(a, "same.py"), "w").write("X = 1\n"); shutil.copy(os.path.join(a, "same.py"), b)
open(os.path.join(b, "new.py"), "w").write("Y = 2\n")
r = lecore.UnifiedMind(dim=256, seed=0).merge_trees(a, b, apply=True)
assert r["identical"] == 1 and os.path.exists(os.path.join(a, "new.py"))
```

**What a big result costs the prompt.** A faculty that returns a million floats returns a million
JSON numbers, and an agent's context is the scarce resource. `bounded_preview(value)` gives the
type, the *true* length or shape, a head/tail sample and the byte cost of both renderings;
`value_cost(value)` gives the cost alone — measured exactly for a small value, sampled and
flagged `exact: False` for a large one. Over the wire, `POST /invoke` takes an optional `budget`:
a result over it comes back as this preview *with* a `ref` handle to the live value, and a result
under it is returned whole, byte for byte as before.

```python
# guide-check
import lecore, numpy as np
mind = lecore.UnifiedMind(dim=256, seed=0)
big = np.arange(1000000, dtype=float)
p = mind.bounded_preview(big, max_bytes=512)
assert p["size"] == 1000000 and p["shape"] == [1000000]      # the TRUE size, never the truncated one
assert p["truncated"] and p["bytes_preview"] <= 512
assert mind.value_cost(big)["bytes"] > 1000000               # what the unbounded reply would have cost
assert mind.value_cost([1, 2, 3]) == {"bytes": 9, "exact": True, "leaves": 3}
nested = [[float(j) for j in range(1000)] for _ in range(1000)]
assert mind.bounded_preview(nested)["head"][0]["length"] == 1000   # bounded at EVERY level, not just the outer
```

*Kept limits.* A preview is lossy: everything between head and tail is reachable only through the
`ref` handle the service mints beside it, and that handle is process-local and evictable. Below
roughly sixteen floats — ten dict keys, two hundred characters — the preview envelope costs *more*
than the value it describes, which is why `/invoke` bounds only what is already over budget instead
of bounding by reflex.

**Holding a return to a contract.** `result_usable` asks "is this usable at all"; `result_contract`
asks the sharper question — "is this the typed thing I said this step must return" — and lists
*every* violation rather than only the first, so one round-trip buys the whole fix. The contract is
plain JSON, so it crosses `/invoke` intact. `validated_call(fn, contract, retries)` hands the
executor its own typed violation back as `feedback=` and asks again, a bounded number of times.

```python
# guide-check
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
contract = {"expect": "nonempty", "require": ["evidence", "verdict"], "types": {"evidence": "list"}}
bad = mind.result_contract({"verdict": "ok"}, contract)
assert not bad["ok"] and "evidence" in bad["violations"][0]
seen = []
def step(feedback=None):
    seen.append(feedback)
    return {"verdict": "ok", "evidence": ["cited"]} if feedback else {"verdict": "ok"}
r = mind.validated_call(step, contract=contract, retries=1)
assert r["ok"] and r["attempts"] == 2 and r["informed_retries"] == 1
assert seen[0] is None      # attempt 1 is a plain call -- byte-identical to calling it yourself
```

*Kept limit.* Neither call can detect a *wrong* answer, only an absent or malformed one — the limit
`result_usable` has, inherited rather than fixed.

**Did the merge lose anything?** `merge_trees` runs *before* a merge and decides what to apply.
`merge_census(base, new)` is its after-partner and answers what a clean diff3 cannot: which
definitions disappeared, which signatures changed, which files shrank — and whether the content
merely *moved* somewhere else in the tree.

```python
# guide-check
import lecore, os, tempfile
base, new = tempfile.mkdtemp(), tempfile.mkdtemp()
open(os.path.join(base, "mod.py"), "w").write("def keep():\n    return 1\ndef gone():\n    return 2\n")
open(os.path.join(new, "mod.py"), "w").write("def keep(extra=None):\n    return 1\n")
c = lecore.UnifiedMind(dim=256, seed=0).merge_census(base, new)
assert c["counts"]["lost"] == 1 and c["lost"][0]["name"] == "gone"
assert c["counts"]["signature_changed"] == 1 and c["signature_changed"][0]["name"] == "keep"
```

**Auditing your own seams.** A faculty is meant to *delegate*. When a parameter is added to a module
function and never plumbed through its wrapper, the capability goes on listing itself in `/tools` and
answering `/invoke` while part of it becomes unreachable -- and every other audit passes, because the
module has a docstring, the catalog example still runs, and nothing is unwired. `delegation_drift()`
is the only instrument that looks at that seam. It also reports what a wrapper *binds itself*
(`mind=self`, `seed=self.seed`) with the binding shown, because a parameter the faculty decides is not
a parameter the caller lost.

```python
# guide-check
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
d = mind.delegation_drift()
assert d["checked"] > 1000                                  # every faculty that declares a delegate
assert d["total_missing"] == 0                              # the backlog is cleared; the budget floor is 0
assert any(r["bound_to"].startswith("self") for r in d["supplied"])  # decided, not lost
# Shape is checked on whichever section HAS rows. The first version of this block indexed
# d["missing"][0] and went red the day the backlog reached zero -- an example that asserts a
# backlog exists rots the moment somebody clears it.
rows = d["missing"] or d["supplied"]
assert rows and isinstance(rows[0], dict)                   # named rows, not tuples
```

*Kept limits.* It checks **names, not semantics**: a faculty forwarding `seed` to a delegate's
`rng_seed` still reads as drift, and one forwarding a value to the *wrong* delegate parameter still
reads as clean. It is a seam-shaped net, not a proof of correctness. The logic lives in `tools/`, so it
needs a source checkout -- and it raises rather than reporting a zero it never computed.

**Programs you can run, not snippets you can read.** `apps()` lists the applications library -- each
entry says what it *proves*, because "it ran" is not a demonstration -- and `app_run(name)` runs one end
to end, returning the numbers it asserts beside its measured runtime. The whole library is 0.29 s, so it
is an example anyone can afford to run. Every application reaches the engine only through faculties;
`tests/test_applications.py` parses each file and fails on a `holographic.*` import, so an application
cannot quietly decay into a script that bypasses the surface every other audit protects.

```python
# guide-check
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
names = [a["name"] for a in mind.apps()]
# NAME the applications rather than counting them: an exact count is an assertion with an expiry
# date, and this one expired the day a fifth application landed.
assert {"spectral_heat", "interleaved_sources", "infinite_zoom"} <= set(names)
assert all(len(a["proves"]) > 40 for a in mind.apps())       # every entry says what it demonstrates

heat = mind.app_run("spectral_heat")
assert heat["proved"]["max_error"] < 1e-13                   # a PDE at t=20, exact, in ONE step
assert heat["proved"]["fd_steps_at_longest"] > 200           # what the marching baseline had to do

demux = mind.app_run("interleaved_sources")
assert demux["proved"]["strides_recovered"] == 3             # K=2,3,4 recovered from the mixture alone
assert mind.app_run("request_to_record")["proved"]["fabricated"] == 0   # refuses instead of inventing
```

*Kept limits.* This is a **first tranche** -- four of the six domains the backlog names; 3-D and
advanced-algorithms are not here yet. And the comparison worth being careful about: Torchhd, the
reference open-source VSA/HDC library, ships an `examples/` directory and leCore shipped none, which is
why this exists -- but their examples are ML tasks scored on public datasets and these are end-to-end
programs over leCore's own machinery. Different claims, and anyone reading both repos should find that
sentence true. `applications/` sits beside the engine, so a wheel install raises rather than reporting
an empty library.

**As above, so below: one operator, two scales.** A feedback buffer is frame N composited with a
transform of frame N−1. A deep zoom is a coordinate window composited with a transform of itself. They
are the same operator — and that is not a metaphor, it is an optimisation: zooming in means the new view
is a magnified *subset* of the old one, so the previous frame already holds every pixel, just blurrier.
Magnify it (one resample) and recompute one narrow band exactly. `feedback_step()` is the operator,
`deep_zoom()` is the zoom that rides it, `feedback_fixed_point()` says whether a buffer converges or
blows up, and `zoom_floor()` says where float64 gives out.

```python
# guide-check
import lecore, numpy as np
mind = lecore.UnifiedMind(dim=256, seed=0)

# decay is THE control parameter, and the critical value is exactly 1.0
buf = np.random.default_rng(0).random((32, 48))
assert mind.feedback_fixed_point(buf, steps=200, zoom=1.0, decay=0.9)["verdict"] == "converged"
assert mind.feedback_fixed_point(buf, steps=400, zoom=1.0, decay=1.6)["verdict"] == "diverged"

# the float64 wall, bracketed from both sides -- and it depends on WHERE you look
wall = mind.zoom_floor((-0.743643887037151, 0.13182590420533), 320)
assert 13.0 < wall["decades"] < 14.5 and wall["verified"]
assert mind.zoom_floor((0.0, 0.0), 320)["decades"] > 100      # no eps wall at the origin

# the zoom refuses to render arithmetic noise
past = mind.deep_zoom(span0=1e-12, rate=0.1, frames=8, width=64, height=36, max_iter=16, band=4)
assert past["stopped"].startswith("precision floor")
```

**One operator, two costumes — and the claim is a number.** Hand `feedback_step()` a 1-D hypervector and
`rotate` becomes a cyclic **permute**: the VSA sequence operator, and the fixed recurrence
`mind.reservoir` already uses. With `decay < 1` that *is* a leaky echo-state update. So the demo scene's
oldest effect and this engine's sequence recurrence are the same operator — and the way to know that
rather than merely say it is to find something that is **the same number** in both. It is the critical
decay, and it is exactly **1.0**.

```python
# guide-check
import lecore, numpy as np
mind = lecore.UnifiedMind(dim=256, seed=0)
vec = np.random.default_rng(3).random(256)
frame = np.random.default_rng(4).random((48, 64))

# the SAME constant, to 1e-12, in both costumes
seq = mind.feedback_fixed_point(vec, steps=16, zoom=1.0, rotate=3, decay=0.95, tol=0.0)
field = mind.feedback_fixed_point(frame, steps=16, zoom=1.0, rotate=0.0, decay=0.95, tol=0.0)
assert abs(seq["ratio"] - 0.95) < 1e-12 and abs(field["ratio"] - 0.95) < 1e-12

# and it is PERMUTATION-ness that buys it, not rank
assert mind.is_permutation(vec, zoom=1.0, rotate=3)["permutation"] is True
rot = mind.is_permutation(frame, zoom=1.0, rotate=0.15)
assert not rot["permutation"] and rot["sampled_once"] < rot["cells"]   # rounding is many-to-one
```

*The finding is sharper than "field vs sequence".* The constant holds whenever the transform is a
**permutation**, and rank has nothing to do with it — a 2-D integer roll lands on 1.0 exactly too. Two
tidier hypotheses died to get there: that rank was the cause (no), and that the clamped edges were
(wrapped 1.0001997 vs clamped 1.0001981 — indistinguishable). It is nearest-neighbour **rounding**,
which is many-to-one: at 0.15 rad a 48×64 rotation samples only 2,658 of 3,072 cells exactly once, and
its critical decay sits 2.0e-04 above 1. `is_permutation()` reports that, so a ratio that looks wrong
comes with the reason.

*Kept limits, all measured.* The acceleration is real but not free: reuse costs ~1.1 % mean error
against a full recompute, bounded by the refresh band. It is only real-time at `band=8` — `band=4` is
18.7 ms/frame, already over a 16.7 ms budget before you add trails. And `zoom_floor` **detects** the
float64 wall; it cannot take you past it. Going deeper needs arbitrary precision or a perturbation
reference orbit, which is a different and much larger build. Run the whole effect with
`mind.app_run('infinite_zoom')`.

**Over the wire.** The same doors ride the MCP server: `lecore-mcp` is on PATH after
`pip install leos-core`; `study` / `study_ask` / `wisdom_record` / `wisdom_ask` are curated
tools, and `lecore_invoke` reaches every faculty. The `initialize` banner carries the
workflow contract to every connected model.

## 11. Meshes, end to end

The largest verb family in the engine, and the one that had the thinnest human coverage: ninety-odd
`mesh_*` doors spanning build, measure, repair, edit, subdivide, decimate, unwrap, the field bridge
and export. Every block below opens with `# guide-check` and runs verbatim in CI.

One habit before you start, because it costs a crash otherwise: **probe the return, do not read it
off the prose.** Several of these hand back a tuple whose arity is easy to guess wrong — a
`mesh_lod_chain()` rung is `(mesh, faces, error, ratio)`, not the `(mesh, log)` pair that
`mesh_decimate_to()` and `mesh_repair()` return. `mind.shape_of(mind.mesh_lod_chain, mesh)` settles it
in one call, and `mind.signature_of(fn)` answers the arity without executing anything.

### 11a. Build one, then measure it

`mesh_box()`, `mesh_grid()` and `mesh_tetrahedron()` are the primitives; `mesh_from_gltf()` parses binary
glTF back into a mesh, `mesh_from_sdf()` marches an implicit field into a surface, and
`points_to_mesh()` takes oriented points the whole way to a watertight quad mesh. Measurement is a
family of its own: `mesh_report()` is the one-call summary (counts, face-type fractions, boundary and
non-manifold edges, bbox, centroid), `mesh_euler()` gives V−E+F with genus and the closed/manifold
flags, `mesh_face_counts()` gives the tri/quad/n-gon split, and `mesh_volume()` and
`mesh_connected_components()` answer the two questions a broken import usually fails.

```python
# guide-check
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
box = mind.mesh_box(1.0, 1.0, 1.0)
r = mind.mesh_report(box)
assert r["verts"] == 8 and r["faces"] == 6 and r["quad_fraction"] == 1.0
assert r["is_closed"] and r["is_manifold"] and r["euler_characteristic"] == 2
assert mind.mesh_volume(box) == 1.0 and mind.mesh_connected_components(box) == 1
assert mind.mesh_euler(box)["genus"] == 0
tri = mind.mesh_triangulate(box)
assert mind.mesh_face_counts(tri) == {3: 12, 4: 0, 5: 0}
assert len(mind.mesh_creases(tri)) == 12          # a cube has exactly twelve sharp edges
```

Deeper measurements, all read-only: `mesh_curvature()` (mean or Gaussian) with
`mesh_curvature_confidence()` beside it so a noisy estimate says so; `mesh_creases()` for the sharp
edges; `mesh_geodesic()` for single-source distance *across the surface*; `mesh_section()` for an exact
planar cross-section (polylines, area, perimeter); `mesh_winding_number()` for inside/outside at any
query point; `mesh_closest_point()` and `mesh_point_distance()` for the correspondence and distance
queries every fitting loop needs; `mesh_orientation_report()` and `mesh_is_oriented()` for the property
a half-edge structure assumes — is every directed edge traversed exactly once.

### 11b. Repair what an importer handed you

`mesh_repair()` composes the standard cleanup and returns a log of what it actually did, which is the
part you want in a pipeline. Its pieces are callable on their own: `mesh_weld()` merges vertices
closer than a tolerance, `mesh_fill_holes()` closes boundary loops, `mesh_make_manifold()` splits
non-manifold vertices into their connected fans, `mesh_orient()` makes winding consistent by
flood-fill, `mesh_drop_small_components()` removes disconnected specks, and `mesh_triangulate()`
ear-clips every face. `mesh_split_vertices()` and `mesh_rip_vertex()` go the other way — they *un*-weld
— and `mesh_topology_delta()` reports whether an op changed topology it had no business changing.

```python
# guide-check
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
tri = mind.mesh_triangulate(mind.mesh_box(1.0, 1.0, 1.0))
torn = mind.mesh_split_vertices(tri)                    # every face gets its own corners: 36 loose verts
assert mind.mesh_report(torn)["verts"] == 36 and not mind.mesh_report(torn)["is_closed"]
fixed, log = mind.mesh_repair(torn)
assert mind.mesh_report(fixed)["verts"] == 8 and mind.mesh_report(fixed)["is_closed"]
assert log["vertices_delta"] == -28 and log["became_manifold"] is not None
holed = mind.mesh_grid(nx=3, ny=3)
assert mind.mesh_report(holed)["boundary_edges"] == 12
closed = mind.mesh_fill_holes(holed)
closed = closed[0] if isinstance(closed, tuple) else closed
assert mind.mesh_report(closed)["boundary_edges"] == 0
```

`mesh_face_type()` converts a triangle mesh's face standard to quads or n-gons *without* moving a
vertex — the operation an artist expects when a tool says "quadrangulate".

### 11c. Edit it: the Euler operators

These are the modelling verbs, and they are Euler operators rather than mesh rewrites, so the
combinatorial invariants stay checkable after every step (`mesh_euler()` is the check).
`mesh_extrude()` lifts a face along its normal, `mesh_inset()` shrinks one toward its centre,
`mesh_bevel_vertex()` rounds a corner, `mesh_poke()` fans a face from a new centre vertex, and
`mesh_loop_cut()` inserts an edge loop. Lower down: `mesh_split_edge()`, `mesh_split_face()`,
`mesh_flip_edge()`, `mesh_collapse_edge()`, `mesh_dissolve_vertex()` and `mesh_bridge()` (join two edge
loops). `mesh_symmetrize()` mirrors across an axis plane. Selections are first-class and persistent:
`mesh_selection()` holds a vertex/edge/face set, and `mesh_soft_selection()` gives a geodesic falloff
in [0,1] so a deformation fades out instead of stopping at a hard boundary.

### 11d. Subdivide and smooth

```python
# guide-check
import lecore, numpy as np
mind = lecore.UnifiedMind(dim=256, seed=0)
box = mind.mesh_box(1.0, 1.0, 1.0)
quads = mind.mesh_catmull_clark(box, levels=2)
assert mind.mesh_face_counts(quads) == {3: 0, 4: 96, 5: 0}      # Catmull-Clark stays all-quad
loop = mind.mesh_subdivide(mind.mesh_triangulate(box), levels=2)
assert len(loop.faces) == 12 * 4 ** 2
smoothed = mind.mesh_smooth(loop, iters=8)
span = lambda mesh: float(np.ptp(np.asarray(mesh.vertices), axis=0).max())
assert span(smoothed) > 0.9 * span(loop)     # Taubin lambda|mu denoises WITHOUT the shrink a plain Laplacian causes
limit_pts, limit_normals = mind.mesh_limit_surface(loop)
assert limit_pts.shape == np.asarray(loop.vertices).shape
```

`mesh_catmull_clark()` is the quad subdivision and takes a crease map; build one with
`mesh_crease_edges()` from an explicit edge list, or let `mesh_auto_crease()` tag the sharp edges for
you by dihedral angle. `mesh_subdivide()` is Loop subdivision for triangles, and `mesh_limit_surface()`
jumps straight to where infinite subdivision would put every vertex — closed form, no iteration.

### 11e. Decimate, and prove the loss

A decimator that does not report its error is a decimator you cannot trust. `mesh_decimate_to()`
takes an explicit budget (`target_faces` or `target_fraction`) and returns a log beside the mesh;
`mesh_surface_deviation()` gives the mean and max point-to-surface error against the original, and
`mesh_egi_compare()` checks that the *orientation field* survived (Horn's Extended Gaussian Image),
which is the failure a face-count target alone will not catch.

```python
# guide-check
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
dense = mind.mesh_subdivide(mind.mesh_triangulate(mind.mesh_box(1.0, 1.0, 1.0)), levels=3)
assert len(dense.faces) == 768
lean, log = mind.mesh_decimate_to(dense, target_fraction=0.25)
assert log["source_faces"] == 768 and len(lean.faces) <= 200
mean_dev, max_dev = mind.mesh_surface_deviation(dense, lean)
assert max_dev < 0.05                                   # the quality metric, on a unit box
chain = mind.mesh_lod_chain(lean, targets=(0.5,))
assert [len(level[0].faces) for level in chain][0] == len(lean.faces)   # rung 0 is the source itself
assert len(chain) == 2 and len(chain[1][0].faces) < len(lean.faces)     # each rung is (mesh, faces, error, ratio)
assert mind.mesh_select_lod(chain, 2.0, 1.0) == 0 and mind.mesh_select_lod(chain, 500.0, 1.0) == 1
```

`mesh_qem_decimate()` is the Garland–Heckbert quadric decimator underneath; `mesh_cluster_decimate()`
is the parallel vertex-clustering alternative when you care more about throughput than about the
silhouette, with `mesh_cluster_lod_chain()` as its chain form. `mesh_field_lod()` builds the chain in
the field domain instead, `mesh_select_lod()` picks a rung by screen-space error, and
`mesh_textured_lod()` is the one call for a decimated mesh that still wears its texture.

### 11f. Unwrap it

```python
# guide-check
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
flat = mind.mesh_triangulate(mind.mesh_grid(nx=4, ny=4))
uv = mind.mesh_uv_unwrap(flat, method="lscm")
assert uv.shape == (25, 2)
angle = mind.mesh_uv_angle_distortion(flat, uv)
assert angle["flipped"] == 0 and abs(angle["max"] - 1.0) < 1e-6   # a developable surface unwraps EXACTLY
assert mind.mesh_uv_distortion(flat, uv) < 1e-6
assert set(mind.mesh_uv_report(flat, methods=("lscm", "planar"))) == {"lscm", "planar"}
assert mind.mesh_pack_uv(flat).shape == (25, 2)
```

`mesh_lscm()` is the least-squares conformal map (Lévy et al., 2002) that `mesh_uv_unwrap()` reaches
for by default; `mesh_pack_uv()` unwraps each connected component and packs the islands. Seams first,
if the surface is not a disk: `mesh_auto_seam()` marks them by dihedral angle, `mesh_shortest_seam()`
finds a path between two vertices, and `mesh_cut_seam()` opens the mesh along one. Never ship a chart
without measuring it — `mesh_uv_angle_distortion()` is the quantity LSCM actually minimises,
`mesh_uv_area_distortion()` catches the opposite failure, `mesh_uv_distortion()` is the per-edge
stretch, and `mesh_uv_report()` runs every chart against every metric so nobody picks by vibe.
`mesh_stable_uv()` gives UVs that are a deterministic function of world position (they do not swim
when the topology changes); after a decimation, `mesh_reproject_uv()` carries the old chart onto the
new topology and `mesh_rebake_texture()` bakes the pixels across.

### 11g. Meshes are fields wearing a different costume

The bridge runs both ways, and it is the reason CSG, collision and skeletons all work on an
imported mesh with no extra machinery. `mesh_to_sdf()` gives signed distance at query points,
`mesh_to_sdf_grid()` produces a full re-marchable field, and `mesh_from_sdf()` marches one back.

```python
# guide-check
import lecore, numpy as np
mind = lecore.UnifiedMind(dim=256, seed=0)
tri = mind.mesh_triangulate(mind.mesh_box(1.0, 1.0, 1.0))
pts = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [2.0, 0.0, 0.0]])
assert np.allclose(mind.mesh_to_sdf(tri, pts), [-0.5, 0.4, 1.5])            # signed: inside is negative
assert np.allclose(mind.mesh_winding_number(pts, tri.vertices, tri.faces), [1, 0, 0])
cut = mind.mesh_section(tri, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
assert cut["area"] == 1.0 and cut["perimeter"] == 4.0 and cut["contours"] == 1
glb = mind.mesh_to_gltf(tri)
assert glb[:4] == b"glTF" and len(mind.mesh_from_gltf(glb).faces) == 12     # a real round trip
assert mind.mesh_to_stl(tri.vertices, tri.faces).startswith("solid ")
```

`mesh_csg()` routes a boolean of two solids through the field rather than through fragile polygon
clipping; `mesh_to_field()` and `mesh_sample_field()` are the banded-grid form; `mesh_to_field_vector()`
carries a whole surface as a *single* hypervector, so an edit becomes a bind. Structure comes out
of the same bridge: `mesh_skeleton()` for the curve skeleton / medial axis, `mesh_parts()` for limbs
and body via the Reeb graph of geodesic distance, `mesh_laplacian_eigenmaps()` for the low spectrum
of the cotan Laplacian, and `mesh_fiedler_order()` for a stable linear vertex order. `mesh_to_softbody()`
turns any mesh into a simulatable body, and `mesh_program_obj()` runs a compiled mesh-transform
program on the installed machine.

### 11h. Get it out again

`mesh_to_stl()` is the CAD export, `mesh_to_gltf()` writes single-file binary glTF (and
`mesh_from_gltf()` reads it back). For storage rather than interchange, `mesh_encode()` compresses at a
stated error budget and `mesh_decode()` inverts it; `mesh_to_tokens()` serialises to a stable token
stream in Morton order, which is what a sequence model wants.

*Kept limits.* `mesh_csg()`, `mesh_skeleton()` and `mesh_to_sdf_grid()` all go through a voxel grid, so
their fidelity is the resolution you pass — a thin feature below the cell size will not survive,
and raising `res` costs cubically. `mesh_uv_unwrap()` expects disk topology: cut seams first for
anything else, or use `mesh_pack_uv()`, which does the per-component unwrap for you.

## Where to look next

- `docs/WHY_A_HOLOGRAPHIC_VM.md` -- why run one; swarm memory topologies; group learning;
  many models; importing and synthesizing skills. Runnable, like §10.
- `docs/USE_CASES.md` -- three swarms that run in CI: a customer-service swarm that learns
  from its humans, a development swarm with one understanding of the codebase, a lab of
  focused roles on one bus and one memory.

- **`mind.find_capability("...")`** — ask the engine, in plain English, which call does what.
- **`CAPABILITIES.md`** — the full menu of capability "homes" (auto-generated from the catalog).
- **`API_QUICKREF.md`** — one scannable line per public function (auto-generated).
- **`SERVICE.md`** — the HTTP endpoints, including the message bus.
- **`tour.py`** — a runnable tour that exercises these features end to end and prints what each one did.

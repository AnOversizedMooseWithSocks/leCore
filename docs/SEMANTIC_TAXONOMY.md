# SEMANTIC_TAXONOMY.md — the capability action hierarchy

The **semantic path** is the `File → Export → PNG` grouping for capabilities: a verb-category tree that groups a
capability by *what a user does with it*, orthogonal to the physical location URI (which module the code lives in).
A capability's `semantic=` tag (optional, on `register_capability`) is a `/`-joined path whose first segment is one
of the roots below.

**Rule (grounded, not aspirational):** a root or sub-branch exists here only because real capabilities populate it.
An empty menu branch is the discovery equivalent of a dark module. The drift gate (S4.1) fails any `semantic` tag
whose root is not in this file.

## The verb roots

Each root is a top-level *action*, the way a DCC context menu's top bar reads. Ordered by how a modeling session
flows: create something, select part of it, transform it, modify its topology, measure it, convert it, then
render/simulate/analyze/io.

- **`create/`** — bring new geometry or data into being. `create/primitive` (sdf/mesh boxes, spheres),
  `create/procedural` (fractal, generate, curve), `create/scene` (compose an sdf_scene, scene graph).
- **`select/`** — choose a subset to act on. `select/element` (mesh_selection), `select/loop` (edge loop, face
  ring, boundary), `select/region` (box/frustum), `select/symmetry` (mirror a selection), `select/soft`
  (proportional-editing weights).
- **`transform/`** — move points/objects without changing topology. `transform/translate`, `transform/rotate`
  (axis-angle, quaternion, scenegraph rotation, skin rotation), `transform/scale`, `transform/gizmo`
  (transform_selection with pivot/space/constraint), `transform/snap` (grid/vertex/edge snapping),
  `transform/pivot`.
- **`modify/`** — change topology or shape. `modify/extrude`, `modify/bevel`, `modify/inset`, `modify/boolean`
  (csg), `modify/subdivide`, `modify/weld`, `modify/deform`, `modify/sculpt`.
- **`measure/`** — read a quantity off geometry. `measure/distance`, `measure/bounds` (bbox), `measure/area`,
  `measure/curvature`, `measure/geodesic`.
- **`convert/`** — change representation. `convert/voxelize`, `convert/isosurface` (points↔mesh, sdf↔mesh),
  `convert/uv` (unwrap), `convert/emit` (glsl/wgsl/ascii dialects), `convert/split` (decompose one mesh into
  parts, e.g. per-material submeshes -- a representation change: merged scene -> keyed dict of meshes).
- **`render/`** — produce an image/frame. `render/raymarch` (sphere_trace, ray_sdf), `render/raster`,
  `render/splat`, `render/shade`, `render/frame` (the real-time frame serving).
- **`simulate/`** — evolve a physical field over time. `simulate/fluid`, `simulate/pbd`, `simulate/cloth`,
  `simulate/particles`.
- **`animate/`** — drive values over time. `animate/keyframe` (timeline + easing), `animate/pose` (blend_pose),
  `animate/skin` (skin_bind_weights, skin_mesh).
- **`analyze/`** — factor, recall, or reason over structure. `analyze/factor` (resonator), `analyze/recall`
  (archive), `analyze/spectral`, `analyze/pipeline` (find/resolve/browse capabilities).
- **`io/`** — cross the boundary to/from disk or another format. `io/import` (obj/gltf), `io/export`,
  `io/workspace` (save/load, checkpoints), `io/file` (the agentic file tools).

## Branches the tags actually use (S5 — the doc caught up with the code)

The list above was written before the tag drive. When `semantic=` coverage went from 108 to ~650 (derived at
registration, see `holographic_semantictag`), **15 branches were in real use that this file never listed** — 102
capabilities filed under menu paths the taxonomy did not admit existed. The S4.1 gate only checked the ROOT, so
the drift was invisible to CI. Per this file's own rule — *a branch exists here because real capabilities populate
it* — the branches earned their place and the DOC was what had drifted. They are documented here, and the gate now
checks the FULL PATH so neither side can drift again.

- **`create/emit`** (21) — generators that emit a made thing: creature, humanoid, quadruped_spec, body_params,
  ifs_generate, mandelbulb, escape_time. *Kept observation: this overlaps `create/procedural` almost exactly, and
  two names for one action is a discoverability tax. Merging is a candidate — it is deliberately NOT done here,
  because re-filing 21 live tags is a change that should be measured (does the menu get easier to browse?), not a
  tidy-up smuggled into a documentation pass.*
- **`simulate/step`** (37) — the generic time-stepper for solvers that are not fluid/pbd/cloth/particles:
  Schrodinger TDSE, N-body gravity, quantum transmission, a fixed-tick game shard.
- **`analyze/measure`** (15) — FITTERS: fit_pose, fit_primitives, fit_shape, ifs_fit, fold_fit, solve_ik_limited.
  They *analyse* structure by fitting a model to it. Distinct from the `measure/` root, which reads a quantity
  straight off geometry (a distance, a bbox) with no model in between.
- **`analyze/route`** (6) — pick a path/policy: route_repair, route_representation, route_structured.
- **`analyze/match`** (4) — match a record or prototype against a store.
- **`analyze/describe`** (3) — produce a description of a thing (doc/report generation).
- **`analyze/decide`** (1) — a decision node with an abstain.
- **`select/pick`** (4) — choose one from a set (distinct from `select/element`, which selects sub-geometry).
- **`transform/warp`** (3) — non-rigid spatial warps.
- **`modify/transform`** (1), **`modify/perturb`** (1) — damage/noise applied to data, **`modify/filter`** (1).
- **`convert/parse`** (1) — text/format in, structure out.
- **`measure/eval`** (1) — evaluate an expression/field at coordinates.
- **`simulate/run`** (2) — run a whole simulation to completion (vs `simulate/step`, one tick).

- **`modify/graph`** (3) — edit a NODE GRAPH's topology: `collapse` (group a selection into one reusable subgraph
  node), `expand` (the inverse), `remove` (delete a node and prune its edges). *Why `modify/` and not `transform/`:
  these were first filed under an invented "transform/edit" (quoted, not back-ticked: this file's own gate reads
  any back-ticked root/sub as DOCUMENTED, so naming a retired tag in backticks would resurrect it as an empty
  branch -- which the gate then fails, correctly), which is wrong at the ROOT — this file defines
  `transform/` as "move points/objects **without** changing topology", and deleting or collapsing a node changes
  the graph's topology outright. `modify/` is "change topology or shape", and a graph has topology even though it
  has no geometry. Filed here per this file's own rule: three capabilities share the action, so the branch has
  members and has earned its place, rather than being hung on `modify/deform` (geometric) or minted as a singleton.*

Two branches listed above were **documented but EMPTY** — `convert/isosurface` and `simulate/pbd` — which by this
file's rule is itself the bug. They were empty because their members were UNTAGGED, not because they were
aspirational: `occupancy_to_mesh` / `mesh_from_sdf` / `points_to_mesh` are exactly "points↔mesh, sdf↔mesh", and
`resolve_swept_collision` is a PBD solver's collision step that the stem "resolve" had mis-filed under
`analyze/pipeline`. Both are now populated via `_SEMANTIC_OVERRIDES` (see `holographic_catalog`).

## How the path disambiguates a collision

`rotation` collides between `mesh_and_geometry/meshskin` and `scene_and_pipeline/scenegraph`. Their semantic paths
disambiguate by *intent*: the scenegraph one is `transform/rotate` (a general node rotation), the meshskin one is
`transform/rotate/skin` (a bone rotation used by skinning). A user browsing `transform/rotate/` sees both, each
labeled by what it's for — exactly the File → Export → PNG resolution. The location URI still ground-truths *where*
each lives; the semantic path says *what each is for*.

## Adding a branch

When a new capability needs a home not listed here: if several existing capabilities share that action, add the
branch (it has members). If it's a one-off, hang it on the nearest existing branch rather than minting a singleton.
Never add a root — the ten above are the complete top-level verb set; a genuinely new root is a design conversation,
not a registration.

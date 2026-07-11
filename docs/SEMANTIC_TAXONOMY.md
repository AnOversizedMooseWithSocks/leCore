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
  `convert/uv` (unwrap), `convert/emit` (glsl/wgsl/ascii dialects).
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

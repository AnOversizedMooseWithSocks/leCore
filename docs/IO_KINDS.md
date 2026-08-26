# IO_KINDS.md — the capability datatype vocabulary

A capability may declare what it **consumes** and **produces** in coarse *kinds*. This is the compatibility layer
that turns capability discovery into pipeline-building: "what can I run on this mesh?" is a filter on `consumes`, and
"what gets me from points to a mesh?" is a `produces`→`consumes` chain.

The kinds live in `holographic/caching_and_storage/holographic_iokinds.py` as a **closed, coarse** vocabulary. Closed
so the tags form a shared language (if one capability *produces* `points` and the next *consumes* `points`, they
link); coarse so the routing stays simple — the fine distinctions live in each capability's own docstring, not in
the kind.

## The kinds

| kind | what it is | example capabilities |
|------|-----------|----------------------|
| `mesh` | vertices + faces (the polygon half) | `mesh_selection`, `select_edge_loop`, `points_to_mesh` |
| `points` | an unordered point set — clouds, particles, splat centers (coarse: all "points") | `ray_mesh_intersect`, voxelize inputs |
| `sdf` | a signed-distance function/field, callable or node (the implicit half) | `ray_sdf_intersect`, sphere-trace |
| `sdf_scene` | a *composed* sdf scene: parts + materials, the whole implicit scene as one object | `sdf_scene`, scene emit |
| `field` | a sampled grid field over space — density, velocity, heat | `sample_field`, fluid solvers |
| `image` | a 2-D raster — RGB(A) or scalar image / texture | `render_scene`, ascii/orbit-trap render |
| `hypervector` | a VSA/HRR vector or bundle — the engine's native representation | encoders, resonator, archive |
| `transform` | an affine transform — matrix, (R,t), or a delta you *apply* | `transform_selection`, `snap_transform_delta` |
| `selection` | a sub-object selection — vertex/edge/face indices + mode | `mesh_selection`, `select_in_box` |
| `scalar` | a single number or small tuple — a measurement, a cost | `measure_bbox`, `scene.cost()` |
| `curve` | a 1-D curve/spline — control points + parameterization | spline/knot builders |
| `skeleton` | a bone hierarchy / armature — joints + parenting (drives skinning) | `skin_bind_weights`, `skin_mesh` |

## Rules

- **Coarse on purpose.** `points` covers point clouds, particle sets, and splat centers alike. Don't mint a new kind
  for a sub-case; hang the distinction in the docstring. A new kind is justified only when *many* capabilities take
  or return something none of the existing kinds describes.
- **Grounded in members.** Every kind above has real capabilities that take or return it (measured across the
  catalog: `field` 199, `mesh` 151, `sdf` 95, `image` 78, `points` 51, `transform` 50, `scalar` 35, `selection` 18,
  `curve` 22, `skeleton`/`bone` a handful, plus `sdf_scene` as the composed container). A kind with no members would
  be a dead menu entry — the io equivalent of a dark module.
- **Validated at registration.** `register_capability(..., consumes=(...), produces=(...))` runs
  `holographic_iokinds.validate_kinds`, so a typo (`point` vs `points`) fails loudly at registration, not silently at
  pipeline time when the mismatched kinds never link.
- **Default empty = unspecified.** A capability with no `consumes`/`produces` is treated as "always shown" by the
  filter — tagging is additive and opt-in, never a gate that hides untagged capabilities.

## How this composes (S3.4 preview)

Given a start kind and a goal kind, the catalog can chain capabilities whose `produces` feeds the next's `consumes`
— e.g. `image → (depth) → points → mesh`. That is the render-graph idea (nodes typed by what they consume/produce)
applied to the whole catalog: the system proposes a *pipeline*, not just a single capability.

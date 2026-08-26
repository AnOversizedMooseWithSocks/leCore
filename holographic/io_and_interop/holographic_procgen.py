"""Procedural generation (S2): 3D objects from a seed, greebled & fractal models, vegetated terrain.

WHY THIS MODULE EXISTS
----------------------
The pieces were all in the box -- the SDF algebra (S1), L-system plants and greeble panels (G5),
fractal terrain (G4), noise (G1), materials (G2). This module is the COMPOSITION layer that turns them
into the three things asked for: procedurally generated 3D objects, fractal/greeble models, and
procedurally vegetated terrain. The through-line is the demoscene seed: a tiny integer unfolds into a
whole, deterministic world (Quilez's seat).

WHAT IT GIVES
-------------
  * `procedural_object(seed)` -- a random SDF tree (primitives under CSG/smooth-union/transform/warp),
    so one seed is one composable, renderable, emittable object. object_to_mesh marches it.
  * `greeble_mesh(base, seed)` -- cover a base mesh's faces with extruded greeble boxes (the G5 panel
    idea lifted from a flat rectangle onto any surface) for that mechanical sci-fi-hull detail.
  * `menger` re-exported from S1 -- the canonical recursive fractal model, as an SDF that also emits GLSL.
  * `scatter_on_terrain` / `vegetated_terrain` -- instance L-system plants (G5) across a fBm terrain
    (G4) at the surface height, with per-instance jitter -> one scenegraph.

HONEST SCOPE (kept negatives)
-----------------------------
  * `procedural_object` makes VARIED objects, not curated ones: a random tree can occasionally subtract
    most of itself away or leave a disconnected surface. It is a generator with a seed, not an art
    director; marching at a fixed resolution rounds sub-cell features.
  * `greeble_mesh` instances boxes on faces; it does NOT boolean them into the hull (no CSG weld), so
    greebles can intersect the base -- which is exactly how real greebling looks, but it is instancing,
    not a solid model.
  * Scatter places instances at the terrain height with no collision/clustering ecology -- it is a
    deterministic scatter, not a growth simulation.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_sdf import sphere, box as sdf_box, torus, cylinder, menger, to_callable
from holographic.mesh_and_geometry.holographic_mesh import Mesh, box as mesh_box
from holographic.scene_and_pipeline.holographic_scenegraph import SceneNode, flatten_scene
from holographic.misc.holographic_transformhome import Transform   # translate/scale via the Transform home  consolidation H5
from holographic.agents_and_reasoning.holographic_grammar import _align_z_to


# ---------------------------------------------------------------------------
# Procedural 3D objects: a seed -> an SDF tree.
# ---------------------------------------------------------------------------

def procedural_object(seed=0, complexity=3):
    """Build a deterministic SDF object tree from a seed: a few transformed primitives combined by random
    CSG / smooth-union, with an occasional rounding or twist. Returns an SDF (render with object_to_mesh,
    emit with .to_glsl(), represent with .to_tree())."""
    rng = np.random.default_rng(seed)

    def rand_prim():
        which = rng.integers(4)
        if which == 0:
            node = sphere(float(rng.uniform(0.3, 0.7)))
        elif which == 1:
            node = sdf_box(*rng.uniform(0.2, 0.6, 3).tolist())
        elif which == 2:
            node = torus(float(rng.uniform(0.4, 0.7)), float(rng.uniform(0.1, 0.25)))
        else:
            node = cylinder(float(rng.uniform(0.3, 0.6)), float(rng.uniform(0.2, 0.5)))
        if rng.random() < 0.7:
            node = node.translate(rng.uniform(-0.5, 0.5, 3).tolist())
        if rng.random() < 0.4:
            node = node.rotate(rng.uniform(-1, 1, 3).tolist(), float(rng.uniform(0, np.pi)))
        return node

    obj = rand_prim()
    for _ in range(max(1, complexity) - 1):
        nxt = rand_prim()
        op = rng.integers(3)
        if op == 0:
            obj = obj.smooth_union(nxt, float(rng.uniform(0.1, 0.3)))
        elif op == 1:
            obj = obj.union(nxt)
        else:
            obj = obj.subtract(nxt)
    if rng.random() < 0.4:
        obj = obj.rounded(float(rng.uniform(0.02, 0.1)))
    if rng.random() < 0.3:
        obj = obj.twist(float(rng.uniform(0.5, 1.5)))
    return obj


def object_to_mesh(sdf_node, bounds=((-2, -2, -2), (2, 2, 2)), res=40):
    """March an SDF object tree to a triangle mesh (uses the engine's existing marching bridge)."""
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
    vals, axes = sample_field(to_callable(sdf_node), bounds, res)
    return marching_tetrahedra_vec(vals, axes, 0.0)


# ---------------------------------------------------------------------------
# Greeble a mesh: extruded boxes on the faces (the G5 panel idea on any surface).
# ---------------------------------------------------------------------------

def _face_centroid_normal(verts, face):
    pts = verts[list(face)]
    c = pts.mean(axis=0)
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    nn = np.linalg.norm(n)
    return c, (n / nn if nn > 0 else np.array([0.0, 0.0, 1.0])), pts


def greeble_mesh(base_mesh, seed=0, density=0.7, max_height=0.15, footprint=0.5):
    """Cover a base mesh's faces with extruded greeble boxes -> a new merged Mesh (base + greebles).

    For each face, with probability `density`, place a small box centred on the face, oriented to the
    face normal, extruded outward by a random height. `footprint` scales the box to a fraction of the
    face size. Deterministic given `seed`. Instancing, not CSG (greebles may intersect -- that is the look).
    """
    rng = np.random.default_rng(seed)
    V = base_mesh.vertices
    children = [SceneNode(mesh=base_mesh)]
    for face in base_mesh.faces:
        if rng.random() > density:
            continue
        c, n, pts = _face_centroid_normal(V, face)
        extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) * footprint * 0.5
        extent = max(extent, 1e-3)
        h = float(rng.uniform(0.2, 1.0)) * max_height
        w = extent * float(rng.uniform(0.4, 0.9)); d = extent * float(rng.uniform(0.4, 0.9))
        strut = mesh_box(width=w, height=d, depth=h)           # along +z, extruded by h
        T = np.eye(4); T[:3, 3] = c + n * (h * 0.5)            # sit on the face, extruded outward
        children.append(SceneNode(transform=T @ _align_z_to(n), mesh=strut))
    return flatten_scene(SceneNode(children=children, name="greebled"))


# ---------------------------------------------------------------------------
# Vegetation scatter on terrain.
# ---------------------------------------------------------------------------

def scatter_on_terrain(terrain, instance_fn, count=12, seed=0, scale_range=(0.5, 1.0),
                       jitter_yaw=True):
    """Place `count` instances on the terrain surface -> one scenegraph (instances + the terrain mesh).

    `instance_fn(rng)` returns a Mesh to place (e.g. a grown plant). Each is dropped at a random (x, y) in
    the terrain bounds, lifted to z = terrain.height([x, y]), randomly scaled and (optionally) yaw-rotated.
    Deterministic given `seed`. Returns (scene_node, placements) where placements is the list of (x,y,z).
    """
    from holographic.mesh_and_geometry.holographic_terrain import terrain_to_mesh
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1) = terrain.bounds
    children = [SceneNode(mesh=terrain_to_mesh(terrain, 24), name="terrain")]
    placements = []
    for _ in range(count):
        x = float(rng.uniform(x0, x1)); y = float(rng.uniform(y0, y1))
        z = terrain.height([x, y])
        s = float(rng.uniform(*scale_range))
        T = Transform.translation([x, y, z]) @ Transform.scaling(s)
        if jitter_yaw:                                         # spin about the up (+z) axis for variety
            from holographic.scene_and_pipeline.holographic_scenegraph import rotation
            T = T @ rotation([0, 0, 1], float(rng.uniform(0, 2 * np.pi)))
        children.append(SceneNode(transform=T, mesh=instance_fn(rng)))
        placements.append((x, y, z))
    return SceneNode(children=children, name="vegetated_terrain"), placements


def vegetated_terrain(seed=0, n_plants=10, terrain_kwargs=None, plant_iterations=3):
    """Convenience: a fBm terrain with `n_plants` L-system plants scattered on it. Returns (scene, terrain)."""
    from holographic.mesh_and_geometry.holographic_terrain import Terrain
    from holographic.agents_and_reasoning.holographic_grammar import LSystem, grow_plant
    tk = dict(bounds=[(0, 6), (0, 6)], octaves=4, gain=0.5, base_bandwidth=2.0, dim=512)
    if terrain_kwargs:
        tk.update(terrain_kwargs)
    terr = Terrain(seed=seed, **tk)

    def make_plant(rng):
        plant = LSystem("X", {"X": "F[+X][-X]FX", "F": "FF"}, rng_seed=int(rng.integers(1 << 30)))
        mesh, _, _ = grow_plant(plant, plant_iterations, angle_deg=25, step=0.2, radius=0.02)
        return mesh

    scene, placements = scatter_on_terrain(terr, make_plant, count=n_plants, seed=seed + 1)
    return scene, terr


# ---------------------------------------------------------------------------

def _selftest():
    # (1) PROCEDURAL OBJECT is deterministic (same seed -> same tree) and varies with seed.
    a = procedural_object(7, complexity=3)
    a2 = procedural_object(7, complexity=3)
    assert a.to_dsl() == a2.to_dsl(), "same seed must give the same object"
    b = procedural_object(8, complexity=3)
    assert a.to_dsl() != b.to_dsl(), "different seeds should give different objects"
    # it renders to a non-empty mesh and emits a shader
    mesh = object_to_mesh(a, res=32)
    assert mesh.n_faces > 0, "procedural object should march to a surface"
    assert "mainImage" in a.to_glsl(), "procedural object should emit a shader"

    # (2) MENGER fractal model renders to a mesh with many faces (the holes add surface).
    spng_mesh = object_to_mesh(menger(2, 1.0), bounds=((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)), res=48)
    assert spng_mesh.n_faces > 0

    # (3) GREEBLE_MESH adds geometry on top of the base and is deterministic.
    base = mesh_box(1, 1, 1)
    g1 = greeble_mesh(base, seed=3, density=1.0)
    g1b = greeble_mesh(base, seed=3, density=1.0)
    assert g1.n_vertices > base.n_vertices, "greebles should add geometry"
    assert np.allclose(g1.vertices, g1b.vertices), "greeble must be deterministic"

    # (4) SCATTER places instances AT the terrain height (each instance's translation z == height there).
    from holographic.mesh_and_geometry.holographic_terrain import Terrain
    terr = Terrain(bounds=[(0, 4), (0, 4)], octaves=3, dim=512, seed=2)
    scene, placements = scatter_on_terrain(terr, lambda rng: mesh_box(0.1, 0.1, 0.3), count=8, seed=1)
    assert len(placements) == 8
    for (x, y, z) in placements:
        assert abs(z - terr.height([x, y])) < 1e-9, "instance not placed at the terrain height"
    flat = flatten_scene(scene)
    assert flat.n_vertices > 0, "vegetated scene should flatten to a mesh"

    # (5) end-to-end vegetated_terrain produces a scene with the terrain plus plants.
    vscene, vterr = vegetated_terrain(seed=5, n_plants=4, plant_iterations=2)
    vmesh = flatten_scene(vscene)
    assert vmesh.n_faces > 0

    print("holographic_procgen selftest passed:",
          f"object_faces={mesh.n_faces} object_dsl_len={len(a.to_dsl())} menger_faces={spng_mesh.n_faces} "
          f"greeble {base.n_vertices}->{g1.n_vertices} verts, scatter={len(placements)} veg_faces={vmesh.n_faces}")


if __name__ == "__main__":
    _selftest()

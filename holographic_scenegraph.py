"""Holographic scene-graph algebra: a scene that is simultaneously GEOMETRY and STRUCTURE.

WHY THIS MODULE EXISTS
----------------------
The FWD work gave the engine a mesh kernel and a set of modeler verbs; the §ARCH work gave the StructureRecipe its
own Euler operators (ARCH-1) and showed program/tree/scene all reduce to one recipe (B7). This module is the
capstone that joins the two: a SCENE GRAPH whose leaves are real meshes and whose edges are transforms, which can
be read TWO ways at once --

  * as GEOMETRY: flatten_scene instances every leaf through its accumulated transform and MERGES them into one mesh
    (the renderer's view -- a concrete pile of triangles);
  * as STRUCTURE: scene_to_recipe encodes the same graph as a StructureRecipe -- transforms bound to content,
    siblings bundled -- realising to one hypervector (the engine's view -- a composed structure).

THE ALGEBRA (the point): these two views are CONSISTENT, because the engine's bundle and a mesh merge are BOTH
commutative. Swapping two siblings changes neither the flattened geometry (same set of triangles) nor the realised
vector (bundle is order-free) -- so a structural edit from ARCH-1 (reorder_members on the sibling bundle) is a
no-op on the geometry too. VSA is geometry: the scene graph is one object wearing both costumes, and they agree.

WHAT IT PROVIDES
  * SceneNode(transform, mesh, children) -- a node: a 4x4 transform, an optional leaf mesh, optional child nodes.
  * identity / translation / scaling / rotation / compose_transforms -- 4x4 transform builders.
  * flatten_scene(node) -- instance every leaf through its accumulated transform, merge into one Mesh.
  * scene_to_recipe(node, dim, seed) -- encode the graph as a StructureRecipe (a valid recipe ARCH-1 operates on).

THE MEASUREMENT BAR (checked exactly in the self-test)
  * INSTANCING: a scene of N leaf meshes flattens to a merged mesh with the summed vertex/face counts; a leaf under
    a translation lands with its centroid at that translation; NESTED transforms compose (parent then child).
  * CONSISTENCY THEOREM: swapping two siblings leaves the flattened geometry identical (same sorted vertices, same
    face count) AND the realised hypervector identical (bundle commutativity) -- the two views agree.
  * the scene recipe is a WELL-FORMED recipe (passes ARCH-1's validate), so the recipe operators apply to scenes.
  * deterministic: same scene -> identical mesh and identical vector.

DETERMINISM (per ISA.md)
  Transforms are fixed matrices; the merge and the encoding visit children in fixed order; leaf/transform atom
  names are content hashes (hashlib). Same scene -> byte-identical mesh and vector (asserted).

KEPT NEGATIVES (loud)
  * flatten_scene INSTANCES and concatenates -- it does not weld coincident vertices or boolean-merge overlapping
    geometry (that is mesh_csg / ARCH-7's job); a flattened scene of two touching cubes is two components, not one
    solid. Honest: this is scene assembly, not constructive solid geometry.
  * scene_to_recipe encodes the scene's STRUCTURE (which transform holds which content), not its geometry -- two
    different meshes with the same vertex hash would collide (by construction the hash distinguishes them); the
    recipe is a structural index, and recovering geometry is flatten_scene's job, not the vector's.
  * The encoding bundles siblings, so (by the decode ceiling) a node with very many children loses per-child
    recoverability in the bundle -- the same capacity cliff every bundle carries; deep/wide scenes index
    structurally but are not meant to be decoded child-by-child from the root vector.
"""

import hashlib

import numpy as np

from holographic_mesh import Mesh
from holographic_recipe import StructureRecipe


# ---- 4x4 transform builders ----------------------------------------------------------------------
def identity():
    return np.eye(4)


def translation(t):
    # consolidation H5: delegate to the Transform home (dedup -- this matrix was duplicated with holographic_transform)
    from holographic_transformhome import Transform
    return Transform.translation(t)


def scaling(s):
    """Uniform scale (scalar s) or per-axis scale (length-3 s). Delegates to the Transform home (consolidation H5)."""
    from holographic_transformhome import Transform
    return Transform.scaling(s)


def rotation(axis, angle):
    """A 4x4 rotation about `axis` by `angle` radians (Rodrigues)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([
        [c + x * x * (1 - c),     x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c),     y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    return M


def compose_transforms(*matrices):
    """The product M0 @ M1 @ ... (apply right-to-left, parent then child). Delegates to the Transform home (H5)."""
    from holographic_transformhome import Transform
    return Transform.compose(*matrices)


class SceneNode:
    """A scene-graph node: a 4x4 `transform`, an optional leaf `mesh`, and optional child nodes. A node may carry
    both geometry and children (a group with its own mesh)."""

    def __init__(self, transform=None, mesh=None, children=None, name=None):
        self.transform = np.eye(4) if transform is None else np.asarray(transform, float)
        self.mesh = mesh
        self.children = list(children) if children else []
        self.name = name


def _merge(meshes):
    """Concatenate meshes into one (offset face indices). Instancing/assembly -- does NOT weld coincident vertices."""
    meshes = [m for m in meshes if m.n_vertices > 0]
    if not meshes:
        return Mesh(np.zeros((0, 3)), [])
    verts, faces, off = [], [], 0
    for m in meshes:
        verts.append(m.vertices)
        faces.extend(tuple(i + off for i in f) for f in m.faces)
        off += m.n_vertices
    return Mesh(np.vstack(verts), faces)


def flatten_scene(node, _world=None):
    """Instance every leaf mesh through its ACCUMULATED transform (parent transforms composed down the graph) and
    merge them into one Mesh -- the geometry view of the scene. Returns a Mesh."""
    world = node.transform if _world is None else _world @ node.transform
    parts = []
    if node.mesh is not None:
        V = node.mesh.vertices
        Vh = np.hstack([V, np.ones((len(V), 1))])
        parts.append(Mesh((Vh @ world.T)[:, :3], [tuple(f) for f in node.mesh.faces]))
    for c in node.children:
        parts.append(flatten_scene(c, world))
    return _merge(parts)


def _sig(arr):
    """A short deterministic content hash of an array (hashlib) -- the atom name for a transform or a leaf mesh."""
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()[:12]


def scene_to_recipe(node, dim=512, seed=0):
    """Encode the scene graph as a StructureRecipe -- transforms BOUND to content, siblings BUNDLED -- realising to
    one hypervector (the structure view of the scene). A well-formed recipe that ARCH-1's operators apply to.
    Returns a StructureRecipe with the root marked as output."""
    r = StructureRecipe(dim, seed)

    def build(n):
        parts = []
        if n.mesh is not None:
            parts.append(r.atom("mesh_" + _sig(n.mesh.vertices)))
        for c in n.children:
            parts.append(build(c))
        content = parts[0] if len(parts) == 1 else r.bundle(parts)
        return r.bind(r.atom("xform_" + _sig(n.transform)), content)

    r.mark_output(build(node))
    return r


# =====================================================================================================
# Self-test -- instancing, nested transforms, and the geometry<->structure consistency theorem.
# =====================================================================================================
def _selftest():
    from holographic_mesh import box
    from holographic_recipeops import validate

    cube = box()

    # --- INSTANCING: a scene of two cubes flattens to one merged mesh with summed counts ---
    scene = SceneNode(children=[SceneNode(translation([2, 0, 0]), mesh=cube),
                                SceneNode(translation([0, 2, 0]), mesh=cube)])
    flat = flatten_scene(scene)
    assert flat.n_vertices == 2 * cube.n_vertices and flat.n_faces == 2 * cube.n_faces, "two cubes merged"
    plus_x = flat.vertices[flat.vertices[:, 0] > 1].mean(axis=0)
    assert np.allclose(plus_x, [2, 0, 0], atol=1e-9), "the +x instance lands at its translation"

    # --- NESTED transforms compose (parent then child) ---
    nested = SceneNode(translation([1, 0, 0]), children=[SceneNode(translation([1, 0, 0]), mesh=cube)])
    assert abs(flatten_scene(nested).vertices[:, 0].mean() - 2.0) < 1e-9, "two +1 translations compose to +2"

    # --- CONSISTENCY THEOREM: swapping siblings leaves geometry AND vector identical ---
    swapped = SceneNode(children=[scene.children[1], scene.children[0]])
    flat_s = flatten_scene(swapped)
    geo_same = (np.allclose(np.sort(flat.vertices, axis=0), np.sort(flat_s.vertices, axis=0))
                and flat.n_faces == flat_s.n_faces)
    vec_same = np.allclose(scene_to_recipe(scene).outputs()[0], scene_to_recipe(swapped).outputs()[0], atol=1e-12)
    assert geo_same, "swapping siblings leaves the flattened geometry identical (merge is commutative)"
    assert vec_same, "swapping siblings leaves the realised vector identical (bundle is commutative)"

    # --- the scene recipe is a well-formed recipe ARCH-1 operates on ---
    assert validate(scene_to_recipe(scene))[0], "the scene encodes to a valid StructureRecipe"

    # --- determinism: same scene -> identical mesh and identical vector ---
    assert np.array_equal(flatten_scene(scene).vertices, flatten_scene(scene).vertices)
    assert np.array_equal(scene_to_recipe(scene).outputs()[0], scene_to_recipe(scene).outputs()[0])

    print(f"holographic_scenegraph selftest: ok (INSTANCING -- 2 cubes flatten to one mesh V={flat.n_vertices} "
          f"F={flat.n_faces}, +x instance at its translation; NESTED transforms compose to +2; CONSISTENCY THEOREM "
          f"-- swapping siblings leaves geometry identical={geo_same} AND the holographic vector identical={vec_same} "
          f"(both commutative); the scene is a valid StructureRecipe; deterministic)")


if __name__ == "__main__":
    _selftest()

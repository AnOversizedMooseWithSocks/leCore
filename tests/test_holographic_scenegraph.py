"""Tests for the holographic scene-graph algebra: a scene read as GEOMETRY (flatten_scene instances + merges) and as
STRUCTURE (scene_to_recipe encodes to a recipe), with the two views consistent -- swapping siblings changes neither
the flattened geometry nor the realised vector (both commutative)."""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import box
from holographic.misc.holographic_recipeops import validate
from holographic.scene_and_pipeline.holographic_scenegraph import SceneNode, identity, translation, scaling, rotation, compose_transforms, flatten_scene, scene_to_recipe


def _two_cube_scene():
    cube = box()
    return SceneNode(children=[SceneNode(translation([2, 0, 0]), mesh=cube),
                               SceneNode(translation([0, 2, 0]), mesh=cube)]), cube


# ---- transform builders ---------------------------------------------------------------------------
def test_translation_moves_a_point():
    p = translation([3, -1, 2]) @ np.array([0, 0, 0, 1.0])
    assert np.allclose(p[:3], [3, -1, 2])


def test_rotation_rotates_a_vector():
    p = rotation([0, 0, 1], np.pi / 2) @ np.array([1, 0, 0, 1.0])
    assert np.allclose(p[:3], [0, 1, 0], atol=1e-9)        # +x -> +y under 90 deg about z


def test_scaling_scales_a_point():
    p = scaling(2.0) @ np.array([1, 1, 1, 1.0])
    assert np.allclose(p[:3], [2, 2, 2])


def test_compose_transforms_is_matrix_product():
    A = translation([1, 0, 0]); B = scaling(2.0)
    assert np.allclose(compose_transforms(A, B), A @ B)


# ---- instancing -----------------------------------------------------------------------------------
def test_instancing_merges_meshes():
    scene, cube = _two_cube_scene()
    flat = flatten_scene(scene)
    assert flat.n_vertices == 2 * cube.n_vertices and flat.n_faces == 2 * cube.n_faces


def test_instance_lands_at_its_translation():
    scene, _ = _two_cube_scene()
    flat = flatten_scene(scene)
    assert np.allclose(flat.vertices[flat.vertices[:, 0] > 1].mean(axis=0), [2, 0, 0], atol=1e-9)


def test_nested_transforms_compose():
    cube = box()
    nested = SceneNode(translation([1, 0, 0]), children=[SceneNode(translation([1, 0, 0]), mesh=cube)])
    assert abs(flatten_scene(nested).vertices[:, 0].mean() - 2.0) < 1e-9


def test_identity_node_leaves_mesh_in_place():
    cube = box()
    flat = flatten_scene(SceneNode(identity(), mesh=cube))
    assert np.allclose(np.sort(flat.vertices, axis=0), np.sort(cube.vertices, axis=0))


# ---- the consistency theorem ----------------------------------------------------------------------
def test_sibling_swap_leaves_geometry_identical():
    scene, _ = _two_cube_scene()
    swapped = SceneNode(children=[scene.children[1], scene.children[0]])
    a, b = flatten_scene(scene), flatten_scene(swapped)
    assert np.allclose(np.sort(a.vertices, axis=0), np.sort(b.vertices, axis=0)) and a.n_faces == b.n_faces


def test_sibling_swap_leaves_vector_identical():
    scene, _ = _two_cube_scene()
    swapped = SceneNode(children=[scene.children[1], scene.children[0]])
    assert np.allclose(scene_to_recipe(scene).outputs()[0], scene_to_recipe(swapped).outputs()[0], atol=1e-12)


# ---- structure view -------------------------------------------------------------------------------
def test_scene_encodes_to_a_valid_recipe():
    scene, _ = _two_cube_scene()
    assert validate(scene_to_recipe(scene))[0]


def test_distinct_scenes_give_distinct_vectors():
    cube = box()
    a = scene_to_recipe(SceneNode(translation([1, 0, 0]), mesh=cube)).outputs()[0]
    b = scene_to_recipe(SceneNode(translation([5, 0, 0]), mesh=cube)).outputs()[0]
    assert not np.allclose(a, b, atol=1e-6)               # different transforms -> different structure vector


# ---- determinism ----------------------------------------------------------------------------------
def test_flatten_and_encode_are_deterministic():
    scene, _ = _two_cube_scene()
    assert np.array_equal(flatten_scene(scene).vertices, flatten_scene(scene).vertices)
    assert np.array_equal(scene_to_recipe(scene).outputs()[0], scene_to_recipe(scene).outputs()[0])

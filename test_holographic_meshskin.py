"""Tests for skinning (FWD-9): linear blend skinning as a soft mixture of expert bone-transforms. Covers rigid
reproduction (partition of unity), single-bone exactness, identity, translation interpolation, faces-preserved,
and the candy-wrapper collapse measured to its exact closed form (the kept negative), plus determinism."""

import numpy as np

from holographic_mesh import box
from holographic_meshskin import linear_blend_skin, skin_mesh, make_transform, rotation


def test_shared_rigid_transform_is_reproduced_exactly():
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((15, 3))
    M = make_transform(rot=rotation([0.2, 0.9, 0.4], 0.6), translation=[1.0, -0.5, 0.3])
    transforms = np.stack([M, M, M])
    weights = rng.uniform(0.1, 1.0, (15, 3))               # ARBITRARY weights
    expected = (np.hstack([pts, np.ones((15, 1))]) @ M.T)[:, :3]
    assert np.allclose(linear_blend_skin(pts, transforms, weights), expected, atol=1e-12)


def test_identity_transforms_leave_points_fixed():
    rng = np.random.default_rng(2)
    pts = rng.standard_normal((15, 3))
    out = linear_blend_skin(pts, np.stack([np.eye(4), np.eye(4)]), np.ones((15, 2)))
    assert np.allclose(out, pts, atol=1e-12)


def test_single_bone_weight_is_exact():
    rng = np.random.default_rng(3)
    pts = rng.standard_normal((15, 3))
    M = make_transform(axis=[0, 0, 1], angle=0.9, translation=[2, 1, 0])
    two = np.stack([M, make_transform(translation=[10, 0, 0])])
    w = np.zeros((15, 2)); w[:, 0] = 1.0
    expected = (np.hstack([pts, np.ones((15, 1))]) @ M.T)[:, :3]
    assert np.allclose(linear_blend_skin(pts, two, w), expected, atol=1e-12)


def test_unnormalized_weights_are_treated_as_partition_of_unity():
    pts = np.array([[1.0, 0.0, 0.0]])
    a = make_transform(translation=[0, 0, 0])
    b = make_transform(translation=[2, 0, 0])
    # weights (3, 1) -> normalised to (0.25, 0.75): result = 0.25*[1,0,0] + 0.75*[3,0,0] = [2.5,0,0]
    out = linear_blend_skin(pts, np.stack([a, b]), np.array([[1.0, 3.0]]))
    assert np.allclose(out, [[2.5, 0.0, 0.0]], atol=1e-12)


def test_translation_interpolation():
    pts = np.array([[0.0, 0.0, 0.0]])
    a = make_transform(translation=[0, 0, 0])
    b = make_transform(translation=[4, 0, 0])
    out = linear_blend_skin(pts, np.stack([a, b]), np.array([[0.5, 0.5]]))
    assert np.allclose(out, [[2.0, 0.0, 0.0]], atol=1e-12)   # midpoint of the two bone translations


def test_candy_wrapper_collapse_matches_cos_half_theta():
    # the kept negative, measured to closed form: a unit ring blended 50/50 across a theta twist -> radius cos(theta/2)
    phi = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
    half = np.full((32, 2), 0.5)
    for theta in (np.pi / 3, np.pi / 2, 2 * np.pi / 3):
        bones = np.stack([np.eye(4), make_transform(axis=[0, 0, 1], angle=theta)])
        radius = float(np.mean(np.linalg.norm(linear_blend_skin(ring, bones, half)[:, :2], axis=1)))
        assert abs(radius - abs(np.cos(theta / 2))) < 1e-9


def test_full_collapse_at_180_degrees():
    phi = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
    bones = np.stack([np.eye(4), make_transform(axis=[0, 0, 1], angle=np.pi)])
    radius = float(np.mean(np.linalg.norm(linear_blend_skin(ring, bones, np.full((32, 2), 0.5))[:, :2], axis=1)))
    assert radius < 1e-9                                    # a 180-degree 50/50 twist collapses to the axis


def test_skin_mesh_preserves_faces():
    b = box()
    transforms = np.stack([make_transform(translation=[1, 0, 0]), make_transform(translation=[0, 1, 0])])
    skinned = skin_mesh(b, transforms, np.ones((b.n_vertices, 2)))
    assert skinned.faces == b.faces                        # connectivity untouched
    assert not np.allclose(skinned.vertices, b.vertices)   # but vertices moved


def test_skinning_is_deterministic():
    rng = np.random.default_rng(4)
    pts = rng.standard_normal((15, 3))
    M = make_transform(axis=[1, 0, 0], angle=0.5, translation=[1, 2, 3])
    transforms = np.stack([M, np.eye(4)])
    weights = rng.uniform(0.1, 1.0, (15, 2))
    assert np.array_equal(linear_blend_skin(pts, transforms, weights), linear_blend_skin(pts, transforms, weights))

"""Tests for holographic_transformhome -- the Transform home (H5: VSA/matrix/rotor/steer transforms, one facade)."""
import numpy as np
from holographic.misc.holographic_transformhome import Transform, transform_kinds


def test_matrices_route_bit_identical():
    import holographic.misc.holographic_transform as TF
    assert np.array_equal(Transform.translation([1, 2, 3]), TF.translation([1, 2, 3]))
    assert np.array_equal(Transform.scaling([2, 0.5, 1]), TF.scaling([2, 0.5, 1]))
    A = TF.translation([1, 0, 0]); B = TF.scaling(2)
    assert np.array_equal(Transform.compose(A, B), TF.compose(A, B))


def test_translation_moves_a_point():
    M = Transform.translation([1.0, 2.0, 3.0])
    p = M @ np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(p[:3], [1.0, 2.0, 3.0])


def test_trs_round_trip():
    from holographic.misc.holographic_transform import quat_from_axis_angle
    t = np.array([1.0, -2.0, 0.5]); s = np.array([2.0, 2.0, 2.0])
    q = quat_from_axis_angle([0, 1, 0], 0.6)
    t2, q2, s2 = Transform.decompose(Transform.compose_trs(t, q, s))
    assert np.allclose(t2, t) and np.allclose(s2, s)


def test_vsa_bind_permute_route():
    from holographic.agents_and_reasoning.holographic_ai import bind, permute
    rng = np.random.default_rng(0)
    a = rng.standard_normal(256); b = rng.standard_normal(256)
    assert np.array_equal(Transform.bind(a, b), bind(a, b))
    assert np.array_equal(Transform.permute(a, 3), permute(a, 3))


def test_clifford_rotor_rotates():
    R = Transform.rotor([0, 0, 1], np.pi / 2)
    v = Transform.rotate_vec(R, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(v, [0.0, 1.0, 0.0], atol=1e-9)


def test_scenegraph_rotation_kept_distinct():
    # scenegraph keeps its own Rodrigues rotation -- NOT bit-identical to the quaternion one (kept difference)
    import holographic.scene_and_pipeline.holographic_scenegraph as SG
    a = [0.3, 0.8, 0.5]; ang = 0.7
    assert not np.array_equal(SG.rotation(a, ang), Transform.rotation(a, ang))
    assert np.allclose(SG.rotation(a, ang), Transform.rotation(a, ang), atol=1e-12)


def test_kinds_listed():
    assert "matrix(4x4)" in transform_kinds() and "rotor(clifford)" in transform_kinds()

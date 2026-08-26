"""Modeling-app backlog item G: transform utilities -- decompose/compose TRS, quaternions+slerp, euler, look_at."""
import numpy as np
from holographic.misc.holographic_transform import decompose, compose_trs, quat_from_euler, quat_to_euler, quat_to_matrix, quat_from_matrix, quat_from_axis_angle, quat_to_axis_angle, quat_mul, quat_slerp, quat_rotate, look_at


def test_decompose_compose_roundtrip():
    t = np.array([2.0, -3.0, 5.0]); q = quat_from_euler(0.3, -0.7, 1.1); s = np.array([2.0, 0.5, 1.5])
    t2, q2, s2 = decompose(compose_trs(t, q, s))
    assert np.allclose(t2, t) and np.allclose(s2, s)
    assert np.allclose(quat_to_matrix(q2), quat_to_matrix(q), atol=1e-9)


def test_quat_matrix_roundtrip_is_proper_rotation():
    q = quat_from_euler(0.4, 1.2, -0.5); R = quat_to_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1.0) < 1e-9
    assert np.allclose(quat_to_matrix(quat_from_matrix(R)), R, atol=1e-9)


def test_euler_roundtrip():
    e = np.array([0.4, -0.6, 0.9])
    assert np.allclose(quat_to_euler(quat_from_euler(*e)), e, atol=1e-6)


def test_axis_angle_roundtrip():
    axis = np.array([1.0, 2.0, -1.0]); axis /= np.linalg.norm(axis)
    ax2, an2 = quat_to_axis_angle(quat_from_axis_angle(axis, 1.2))
    assert np.allclose(ax2, axis, atol=1e-9) and abs(an2 - 1.2) < 1e-9


def test_quat_mul_composes():
    qx = quat_from_axis_angle([1, 0, 0], 0.5); qy = quat_from_axis_angle([0, 1, 0], 0.9)
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(quat_rotate(quat_mul(qx, qy), v), quat_rotate(qx, quat_rotate(qy, v)), atol=1e-9)


def test_slerp_endpoints_and_midpoint():
    a = quat_from_axis_angle([0, 0, 1], 0.0); b = quat_from_axis_angle([0, 0, 1], 1.0)
    assert np.allclose(quat_slerp(a, b, 0.0), a) and np.allclose(quat_slerp(a, b, 1.0), b)
    mid = quat_slerp(a, b, 0.5)
    assert abs(np.linalg.norm(mid) - 1.0) < 1e-9
    _, ang = quat_to_axis_angle(mid); assert abs(ang - 0.5) < 1e-6


def test_slerp_shortest_path():
    # a and -b are the same rotation as a and b; slerp must take the short arc regardless of sign
    a = quat_from_axis_angle([0, 1, 0], 0.1); b = quat_from_axis_angle([0, 1, 0], 0.2)
    assert np.allclose(quat_to_matrix(quat_slerp(a, -b, 0.5)), quat_to_matrix(quat_slerp(a, b, 0.5)), atol=1e-9)


def test_look_at():
    eye = np.array([3.0, 4.0, 5.0]); target = np.zeros(3)
    V = look_at(eye, target)
    oe = V @ np.array([*eye, 1.0]); ot = V @ np.array([*target, 1.0])
    assert np.allclose(oe[:3], 0.0, atol=1e-9)                 # eye -> origin
    assert abs(ot[0]) < 1e-9 and abs(ot[1]) < 1e-9 and ot[2] < 0   # target down -z

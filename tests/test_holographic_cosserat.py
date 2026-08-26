"""Cosserat rod (H2b): orientation frames give twist and curl-holding for hair."""
import numpy as np
from holographic.simulation_and_physics.holographic_cosserat import CosseratStrand, from_strand, qmul, qconj, quat_rotate, quat_from_axis_angle, quat_between


def _helix(n=16, cr=0.12):
    s = np.linspace(0, 1, n)
    return np.stack([cr * (np.cos(2 * np.pi * 2 * s) - 1.0) * s, s * 0.8, cr * np.sin(2 * np.pi * 2 * s) * s], axis=1)


def test_quaternion_helpers():
    q = quat_from_axis_angle([1, 0, 0], np.pi / 2)
    assert np.allclose(quat_rotate(q, [0, 0, 1]), [0, -1, 0], atol=1e-6)
    assert np.allclose(quat_rotate(qconj(q), quat_rotate(q, [0.3, 0.4, 0.5])), [0.3, 0.4, 0.5], atol=1e-6)
    r = quat_between([0, 0, 1], [1, 0, 0])
    assert np.allclose(quat_rotate(r, [0, 0, 1]), [1, 0, 0], atol=1e-6)


def test_curl_holds_under_gravity():
    pts = _helix()
    rest = 1.0 - np.linalg.norm(pts[-1] - pts[0]) / np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    rod = CosseratStrand(pts, bend_stiffness=0.6, shape_stiffness=0.7).settle(steps=120, gravity=(0, -9.8, 0))
    plain = CosseratStrand(pts, bend_stiffness=0.0, shape_stiffness=0.0).settle(steps=120, gravity=(0, -9.8, 0))
    assert abs(rod.curl_amount() - rest) < abs(plain.curl_amount() - rest)
    assert rod.curl_amount() > 0.5 * rest


def test_stretch_uncurls():
    pts = _helix()
    rest = 1.0 - np.linalg.norm(pts[-1] - pts[0]) / np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    rod = CosseratStrand(pts, bend_stiffness=0.6, shape_stiffness=0.7)
    axis = (pts[-1] - pts[0]); axis = axis / np.linalg.norm(axis)
    target = pts[0] + axis * 0.96 * rod._arc
    for _ in range(150):
        rod.step(gravity=(0, 0, 0)); rod.x[-1] = target; rod.v[-1] = 0.0; rod.x[0] = pts[0]; rod.v[0] = 0.0
    assert rod.curl_amount() < 0.5 * rest


def test_twist_propagates():
    pts = _helix()
    rod = CosseratStrand(pts, bend_stiffness=0.8, shape_stiffness=0.5)
    before = abs(rod.twist_of(len(pts) // 2))
    rod.set_root_twist(1.2)
    for _ in range(40):
        rod.step(gravity=(0, 0, 0))
    assert abs(rod.twist_of(len(pts) // 2)) > before


def test_deterministic_and_inextensible():
    pts = _helix()
    a = CosseratStrand(pts).settle(steps=30, gravity=(0, -9.8, 0))
    b = CosseratStrand(pts).settle(steps=30, gravity=(0, -9.8, 0))
    assert np.array_equal(a.x, b.x)
    seg = np.linalg.norm(np.diff(a.x, axis=0), axis=1)
    assert np.abs(seg - a.L).max() < 1e-6                          # inextensible (segment lengths preserved)

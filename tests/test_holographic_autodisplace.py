"""Inverse-rendering IR5: displacement from a confident height -- promote an IR1 height to real geometry, gated."""
import numpy as np
from holographic.mesh_and_geometry.holographic_mesh import grid
from holographic.mesh_and_geometry.holographic_autodisplace import displace_from_height, auto_displace, _bilinear


def _bump_map(N):
    yy, xx = np.mgrid[0:N + 1, 0:N + 1]
    return np.exp(-(((xx - N / 2) / (N / 5)) ** 2 + ((yy - N / 2) / (N / 5)) ** 2))


def test_confident_height_makes_relief():
    N = 24; plane = grid(nx=N, ny=N)
    disp, abst = displace_from_height(plane, _bump_map(N), amount=0.3, confidence=0.2, min_confidence=0.02)
    assert not abst
    z = disp.vertices[:, 2]
    assert z.max() > 0.2
    ci = int(np.argmin(np.linalg.norm(plane.vertices[:, :2], axis=1)))
    co = int(np.argmax(np.linalg.norm(plane.vertices[:, :2], axis=1)))
    assert z[ci] > z[co] + 0.1                              # relief follows the height


def test_low_confidence_abstains_mesh_unchanged():
    N = 24; plane = grid(nx=N, ny=N)
    same, abst = displace_from_height(plane, _bump_map(N), amount=0.3, confidence=0.005, min_confidence=0.02)
    assert abst and np.allclose(same.vertices[:, 2], 0.0)   # not deformed


def test_no_confidence_given_displaces():
    N = 16; plane = grid(nx=N, ny=N)
    disp, abst = displace_from_height(plane, _bump_map(N), amount=0.2, confidence=None)
    assert not abst and disp.vertices[:, 2].max() > 0.05    # no gate -> displaces


def test_bilinear_samples_corners():
    h = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert abs(_bilinear(h, 0.0, 0.0) - 0.0) < 1e-9
    assert abs(_bilinear(h, 1.0, 1.0) - 3.0) < 1e-9
    assert abs(_bilinear(h, 0.5, 0.5) - 1.5) < 1e-9         # centre = mean of the four


def test_auto_displace_bumpy_vs_flat():
    Ni = 48
    u = np.linspace(0, 6 * np.pi, Ni)
    bump = 0.5 + 0.4 * np.outer(np.sin(u), np.cos(u))
    _, info = auto_displace(grid(nx=20, ny=20), np.stack([bump, bump, bump], axis=-1), amount=0.2)
    assert info["displaced"]
    _, info_flat = auto_displace(grid(nx=20, ny=20), np.full((Ni, Ni, 3), 0.5), amount=0.2)
    assert not info_flat["displaced"]


def test_deterministic():
    N = 16; plane = grid(nx=N, ny=N)
    d1, _ = displace_from_height(plane, _bump_map(N), amount=0.3, confidence=0.2)
    d2, _ = displace_from_height(plane, _bump_map(N), amount=0.3, confidence=0.2)
    assert np.array_equal(d1.vertices, d2.vertices)

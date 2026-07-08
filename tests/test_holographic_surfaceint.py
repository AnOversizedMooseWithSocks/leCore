"""Inverse-rendering IR7: surface-from-gradient by FFT (Frankot-Chellappa) -- consistent, tileable height."""
import numpy as np
from holographic.mesh_and_geometry.holographic_surfaceint import gradient_from_normals, height_from_gradient, height_from_normals, consistent_normals
from holographic.mesh_and_geometry.holographic_autobump import normal_from_height


def _periodic_height(H=48, W=64):
    y = np.arange(H)[:, None]; x = np.arange(W)[None, :]
    h = np.sin(2 * np.pi * 2 * x / W) * np.cos(2 * np.pi * 3 * y / H)
    return h - h.mean(), x, y


def test_recovers_height_from_analytic_gradient():
    h, x, y = _periodic_height()
    H, W = h.shape
    p = (2 * np.pi * 2 / W) * np.cos(2 * np.pi * 2 * x / W) * np.cos(2 * np.pi * 3 * y / H)
    q = -(2 * np.pi * 3 / H) * np.sin(2 * np.pi * 2 * x / W) * np.sin(2 * np.pi * 3 * y / H)
    z = height_from_gradient(p * np.ones_like(h), q * np.ones_like(h)); z -= z.mean()
    assert np.corrcoef(z.ravel(), h.ravel())[0, 1] > 0.999
    assert np.sqrt(np.mean((z - h) ** 2)) / np.std(h) < 0.02


def test_roundtrip_from_normal_map():
    h, _, _ = _periodic_height()
    z = height_from_normals(normal_from_height(h, strength=1.0)); z -= z.mean()
    assert np.corrcoef(z.ravel(), h.ravel())[0, 1] > 0.99


def test_gradient_from_normals_matches():
    h, _, _ = _periodic_height()
    n = normal_from_height(h, strength=1.0)
    p, q = gradient_from_normals(n)
    gy, gx = np.gradient(h)
    assert np.corrcoef(p.ravel(), gx.ravel())[0, 1] > 0.99   # p ~ dh/dx


def test_seamlessly_tileable():
    h, _, _ = _periodic_height()
    z = height_from_normals(normal_from_height(h, strength=1.0))
    seam = np.abs(z[:, 0] - z[:, -1]).mean()
    interior = np.abs(np.diff(z, axis=1)).mean()
    assert seam < 3.0 * interior                             # periodic -> no big seam discontinuity


def test_consistent_normals_are_integrable():
    h, _, _ = _periodic_height()
    n = normal_from_height(h, strength=1.0)
    cn = consistent_normals(n, strength=1.0)
    z = height_from_normals(cn); z -= z.mean()
    assert np.corrcoef(z.ravel(), h.ravel())[0, 1] > 0.99


def test_deterministic():
    h, _, _ = _periodic_height()
    n = normal_from_height(h, strength=1.0)
    assert np.array_equal(height_from_normals(n), height_from_normals(n))

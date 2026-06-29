"""Tests for G3 displacement/bump (holographic_displace): SDF displacement is a field delta (apply_delta of
-amount*scalar) with EXACT remove_delta undo; mesh displacement moves vertices along normals by exactly
amount*scalar; bump perturbs shading normals only where the scalar field varies."""

import numpy as np

from holographic_fpe import VectorFunctionEncoder
from holographic_fpefield import HolographicField
from holographic_mesh import Mesh
from holographic_displace import displace_sdf, displace_mesh, bump_normals, _selftest


def _slab():
    enc = VectorFunctionEncoder(3, dim=2048, bounds=[(-1, 1)] * 3, kernel="rbf", bandwidth=6.0, seed=1)
    axes = np.linspace(-0.8, 0.8, 6)
    gx, gy, gz = np.meshgrid(axes, axes, axes, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return HolographicField(enc, P, P[:, 2])      # sdf = z


def test_sdf_displacement_raises_surface():
    field = _slab()
    disp, _ = displace_sdf(field, lambda x: 1.0, 0.2)
    pt = [[0.0, 0.0, 0.0]]
    assert disp.value(pt)[0] < field.value(pt)[0] - 0.05


def test_sdf_undo_is_exact():
    field = _slab()
    disp, delta = displace_sdf(field, lambda x: 1.0, 0.2)
    restored = disp.remove_delta(delta)
    assert np.max(np.abs(restored.f - field.f)) < 1e-9


def _quad():
    return Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [(0, 1, 2), (0, 2, 3)])


def test_mesh_displacement_along_normals():
    quad = _quad()
    raised = displace_mesh(quad, lambda x: 1.0, 0.3)
    assert np.allclose(raised.vertices[:, 2] - quad.vertices[:, 2], 0.3, atol=1e-6)


def test_mesh_displacement_varies_with_scalar():
    quad = _quad()
    ramp = displace_mesh(quad, lambda x: x[0], 0.5)
    assert np.allclose(ramp.vertices[:, 2], 0.5 * quad.vertices[:, 0], atol=1e-6)


def test_bump_only_where_field_varies():
    quad = _quad()
    base = quad.vertex_normals(store=False)
    assert np.allclose(bump_normals(quad, lambda x: 1.0, 1.0), base, atol=1e-6)
    assert np.max(np.abs(bump_normals(quad, lambda x: x[0], 1.0) - base)) > 0.05


def test_selftest_runs():
    _selftest()

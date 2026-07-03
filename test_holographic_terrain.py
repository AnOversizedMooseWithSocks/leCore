"""Tests for G4 terrain (holographic_terrain): a 2-D fBm heightfield (composition of G1), liftable to a
displaced-grid mesh (z = height) or a heightfield SDF (z - height: sign-correct). Roughness tracks
persistence; no erosion (kept negative)."""

import numpy as np

from holographic_terrain import Terrain, terrain_to_mesh, terrain_to_sdf, _selftest


def test_deterministic():
    a = Terrain(bounds=[(0, 4), (0, 4)], octaves=4, dim=512, seed=7).heightmap(20)
    b = Terrain(bounds=[(0, 4), (0, 4)], octaves=4, dim=512, seed=7).heightmap(20)
    assert np.allclose(a, b)


def test_roughness_tracks_persistence():
    def rough(gain):
        hm = Terrain(bounds=[(0, 4), (0, 4)], octaves=4, gain=gain, base_bandwidth=2.5, dim=512, seed=3).heightmap(40)
        return (np.abs(np.diff(hm, axis=0)).mean() + np.abs(np.diff(hm, axis=1)).mean()) / (hm.std() + 1e-9)
    assert rough(0.85) > rough(0.30)


def test_terrain_to_mesh_shape_and_height():
    t = Terrain(bounds=[(0, 4), (0, 4)], octaves=4, dim=512, seed=7)
    mesh = terrain_to_mesh(t, 12)
    assert mesh.n_vertices == 144 and len(mesh.faces) == 2 * 11 * 11
    v = mesh.vertices[5 * 12 + 5]
    assert abs(v[2] - t.height([v[0], v[1]])) < 1e-9


def test_heightmap_matches_direct_point_stack_readout():
    t = Terrain(bounds=[(0, 4), (0, 4)], octaves=3, dim=512, seed=4)
    res = 14
    xs = np.linspace(0, 4, res)
    ys = np.linspace(0, 4, res)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    direct = t.heights(pts).reshape(res, res)
    assert np.allclose(t.heightmap(res), direct, atol=1e-12)


def test_heightfield_sdf_sign():
    t = Terrain(bounds=[(0, 4), (0, 4)], octaves=4, dim=512, seed=7)
    fld = terrain_to_sdf(t, z_bounds=(-2, 2), res=8, dim=1024, bandwidth=8.0, seed=1)
    h = t.height([2.0, 2.0])
    below = fld.value([[2.0, 2.0, h - 1.0]])[0]
    above = fld.value([[2.0, 2.0, h + 1.0]])[0]
    assert below < 0 < above


def test_selftest_runs():
    _selftest()

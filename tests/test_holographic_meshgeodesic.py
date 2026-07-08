"""Tests for surface geodesics (FWD-5): accuracy against the analytic great-circle distance (correlation, net
overestimate with bounded undercut), the antipode-is-farthest property, the geodesic-vs-Euclidean soft-selection
contrast (no bleed across the surface), matrix structure, reachability, a flat-grid sanity check, determinism."""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh, box, grid
from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere
from holographic.mesh_and_geometry.holographic_meshgeodesic import geodesic_distances, geodesic_matrix, geodesic_soft_selection, _edge_graph


def _sphere_geo():
    s = _icosphere(3)
    north = int(np.argmax(s.vertices[:, 2]))
    return s, north, geodesic_distances(s, north)


# ---- accuracy vs the analytic great circle ----------------------------------------------------------
def test_geodesic_correlates_with_great_circle():
    s, north, g = _sphere_geo()
    true_geo = np.arccos(np.clip(s.vertices[:, 2], -1.0, 1.0))   # great-circle distance from the pole
    assert float(np.corrcoef(g, true_geo)[0, 1]) > 0.99


def test_geodesic_net_overestimate_with_bounded_undercut():
    s, north, g = _sphere_geo()
    true_geo = np.arccos(np.clip(s.vertices[:, 2], -1.0, 1.0))
    far = true_geo > 0.2
    signed = (g[far] - true_geo[far]) / true_geo[far]
    assert 0.0 < float(signed.mean()) < 0.12          # net overestimate (edge restriction)
    assert float(signed.min()) > -0.05                # bounded chord-effect undercut near source


def test_antipode_is_farthest_and_exceeds_euclidean():
    s, north, g = _sphere_geo()
    south = int(np.argmin(s.vertices[:, 2]))
    assert abs(g[south] - np.pi) < 0.2                # great-circle north->south = pi on the unit sphere
    assert g[south] > float(np.linalg.norm(s.vertices[south] - s.vertices[north]))   # > straight-line


# ---- the geodesic-vs-Euclidean contrast (no bleed) -------------------------------------------------
def test_soft_selection_excludes_antipode_a_euclidean_ball_would_include():
    s, north, _ = _sphere_geo()
    south = int(np.argmin(s.vertices[:, 2]))
    sel = geodesic_soft_selection(s, north, radius=2.5)
    assert sel[north] == 1.0
    assert sel[south] == 0.0                          # geodesic (~pi > 2.5) excludes the antipode
    assert float(np.linalg.norm(s.vertices[south] - s.vertices[north])) < 2.5   # Euclidean ball would NOT


def test_soft_selection_in_range():
    s, north, _ = _sphere_geo()
    for falloff in ("smooth", "linear"):
        sel = geodesic_soft_selection(s, north, radius=1.5, falloff=falloff)
        assert np.all((sel >= 0.0) & (sel <= 1.0))


# ---- matrix structure & reachability ---------------------------------------------------------------
def test_geodesic_matrix_is_symmetric_with_zero_diagonal():
    m = _icosphere(2)
    D = geodesic_matrix(m)
    assert D.shape == (m.n_vertices, m.n_vertices)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T, atol=1e-9)             # geodesic distance is symmetric


def test_all_vertices_reachable_on_a_closed_mesh():
    s, north, g = _sphere_geo()
    assert np.all(np.isfinite(g))                     # a closed sphere is connected


# ---- flat-grid sanity: edge-graph geodesic is the along-grid (>= straight-line) distance -----------
def test_flat_grid_geodesic_exceeds_straight_line():
    g_mesh = grid(5, 5, width=1.0, height=1.0)        # quad grid: edges along the axes only
    d = geodesic_distances(g_mesh, 0)                 # from a corner
    far_corner = int(np.argmax(np.linalg.norm(g_mesh.vertices - g_mesh.vertices[0], axis=1)))
    straight = float(np.linalg.norm(g_mesh.vertices[far_corner] - g_mesh.vertices[0]))
    assert d[far_corner] >= straight - 1e-9           # along-grid path is at least the straight-line distance


# ---- determinism -----------------------------------------------------------------------------------
def test_geodesic_is_deterministic():
    s, north, _ = _sphere_geo()
    assert np.array_equal(geodesic_distances(s, north), geodesic_distances(s, north))

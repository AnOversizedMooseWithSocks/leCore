"""F3 -- the points -> SDF -> mesh path, and three negatives about which error is which.

The engine could turn an SDF into points and never the reverse. `convert splats to a mesh`, `surface reconstruction
from points`, `marching cubes` and `dual contouring` all returned fallbacks before this.

THE BASELINE IS THE CELL SIZE. A dual isosurface extractor places one vertex per cell; it cannot do better than the
cell it lives in. Scoring against zero would be scoring against a resolution nobody asked for.
"""

import numpy as np
import pytest

from holographic.mesh_and_geometry.holographic_isosurface import (
    is_oriented, is_watertight, mesh_report, points_to_mesh, sdf_from_points, surface_nets)


LO, HI = np.full(3, -1.6), np.full(3, 1.6)


def _sphere_cloud(n=600, seed=0):
    """A unit sphere's normal IS its position -- the cleanest oriented cloud there is, and analytic ground truth."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p, p.copy()


def _sphere_sdf(X):
    return np.linalg.norm(np.asarray(X, float), axis=-1) - 1.0


def test_selftest_runs():
    from holographic.mesh_and_geometry import holographic_isosurface as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# the mesh
# ---------------------------------------------------------------------------------------------------------

def test_a_sphere_reconstructs_watertight_with_the_right_topology():
    p, n = _sphere_cloud()
    V, Q, _F, _g = points_to_mesh(p, n, LO, HI, 24)
    rep = mesh_report(V, Q, sdf=_sphere_sdf)
    assert rep["watertight"] is True
    assert rep["euler"] == 2                       # V - E + F for a sphere; E = 2Q on a closed quad mesh
    assert rep["n_vertices"] > 500 and rep["n_quads"] > 500


def test_the_surface_error_is_sub_cell_which_is_the_only_honest_bar():
    p, n = _sphere_cloud()
    V, Q, _F, grids = points_to_mesh(p, n, LO, HI, 24)
    cell = float(grids[0][1] - grids[0][0])
    err = mesh_report(V, Q, sdf=_sphere_sdf)["surface_error"]
    assert err < cell                              # measured 0.0496 against a cell of 0.1391
    assert err > 0.0                               # ... and not zero: a dual extractor is not exact


def test_the_mesh_is_deterministic():
    p, n = _sphere_cloud()
    a = points_to_mesh(p, n, LO, HI, 20)
    b = points_to_mesh(p, n, LO, HI, 20)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_surface_nets_on_an_analytic_field_beats_the_point_cloud_path():
    # Feed the extractor a PERFECT sdf: the only error left is the extractor's own.
    res = 24
    grids = [np.linspace(LO[d], HI[d], res) for d in range(3)]
    G = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    V, Q = surface_nets(_sphere_sdf(G), grids)
    cell = float(grids[0][1] - grids[0][0])
    assert is_watertight(Q)
    assert np.abs(_sphere_sdf(V)).max() < 0.25 * cell        # the extractor alone is much better than sub-cell


def test_a_shifted_iso_extracts_a_different_level_set():
    res = 20
    grids = [np.linspace(LO[d], HI[d], res) for d in range(3)]
    G = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    field = _sphere_sdf(G)
    V, _Q = surface_nets(field, grids, iso=0.3)
    assert abs(float(np.linalg.norm(V, axis=1).mean()) - 1.3) < 0.1


# ---------------------------------------------------------------------------------------------------------
# THE THREE KEPT NEGATIVES
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_the_point_cloud_sdf_is_worst_near_the_surface():
    # THE COUNTERINTUITIVE ONE, and it matters because that is exactly where the extractor reads the field.
    # Distance-to-nearest-SAMPLE overestimates distance-to-SURFACE by up to the sample spacing.
    p, n = _sphere_cloud()
    res = 24
    F, grids = sdf_from_points(p, n, LO, HI, res)
    G = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    truth = _sphere_sdf(G)
    err = np.abs(F - truth)

    near = np.abs(truth) < 0.1
    far = np.abs(truth) > 0.6
    assert err[near].max() > 2.0 * err[far].max()   # measured 0.2352 vs 0.0705


def test_kept_negative_accuracy_is_set_by_the_cloud_not_the_grid():
    # At a FIXED grid, more points means less error. Refining the grid under a sparse cloud buys nothing.
    res = 20
    G = np.stack(np.meshgrid(*[np.linspace(LO[d], HI[d], res) for d in range(3)], indexing="ij"), axis=-1)
    truth = _sphere_sdf(G)
    near = np.abs(truth) < 0.1

    errs = []
    for n_pts in (150, 600, 2400):
        p, nr = _sphere_cloud(n_pts)
        F, _g = sdf_from_points(p, nr, LO, HI, res)
        errs.append(float(np.abs(F - truth)[near].max()))
    assert errs == sorted(errs, reverse=True)       # strictly improving with the cloud

    # ... and the error tracks the SAMPLE SPACING, at roughly 1.3-1.7x it
    for n_pts, e in zip((150, 600, 2400), errs):
        spacing = np.sqrt(4 * np.pi / n_pts)
        assert 1.0 < e / spacing < 2.5, (n_pts, e / spacing)


def test_kept_negative_the_mesh_is_more_accurate_than_its_own_field():
    # Not a paradox: the zero crossing is interpolated, and averaging twelve edge crossings cancels per-sample
    # noise. Do not read the field's error as the mesh's, in either direction.
    p, n = _sphere_cloud()
    V, _Q, F, grids = points_to_mesh(p, n, LO, HI, 24)
    G = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    truth = _sphere_sdf(G)
    near = np.abs(truth) < 0.1

    field_err = float(np.abs(F - truth)[near].max())
    mesh_err = float(np.abs(_sphere_sdf(V)).max())
    assert mesh_err < 0.5 * field_err              # measured 4.7x better


def test_honest_scope_sharp_features_are_rounded_off():
    # Naive surface nets averages the crossings, which is exactly what Dual Contouring's QEF solve avoids. A cube's
    # corner should land ON the corner; here it is pulled inward. Stated, not hidden.
    res = 24
    grids = [np.linspace(-1.6, 1.6, res) for _ in range(3)]
    G = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    cube = np.abs(G).max(axis=-1) - 1.0            # an axis-aligned cube SDF: sharp edges and corners
    V, Q = surface_nets(cube, grids)
    assert is_watertight(Q)

    # the extreme corner of the reconstruction falls short of (1,1,1)
    corner = V[np.argmax(V.sum(axis=1))]
    assert np.linalg.norm(corner - np.ones(3)) > 0.05


# ---------------------------------------------------------------------------------------------------------
# the round trip, through the engine's own forward direction
# ---------------------------------------------------------------------------------------------------------

def test_round_trip_against_the_engines_own_sdf_surface_points():
    # sdf -> points -> sdf -> mesh. The forward direction already shipped; this closes the loop.
    from holographic.scene_and_pipeline.holographic_session import sdf_surface_points

    class Sphere:
        """`sdf_surface_points` takes an SDF OBJECT with `.eval`, not a bare callable -- a contract worth citing
        rather than assuming (the first version of this test passed a function and got an AttributeError)."""

        def eval(self, P):
            return _sphere_sdf(P)

    pts = np.asarray(sdf_surface_points(Sphere(), (LO, HI), n=800, seed=0), float)
    assert len(pts) > 300
    assert np.abs(_sphere_sdf(pts)).max() < 0.05            # they really are on the surface

    nrm = pts / np.linalg.norm(pts, axis=1, keepdims=True)  # a sphere's normal is its position
    V, Q, _F, grids = points_to_mesh(pts, nrm, LO, HI, 24)
    cell = float(grids[0][1] - grids[0][0])
    assert is_watertight(Q)
    assert np.abs(_sphere_sdf(V)).max() < cell


# ---------------------------------------------------------------------------------------------------------
# guards + wiring
# ---------------------------------------------------------------------------------------------------------

def test_degenerate_inputs_raise():
    p, n = _sphere_cloud(50)
    with pytest.raises(ValueError):
        sdf_from_points(np.zeros((0, 3)), np.zeros((0, 3)), LO, HI, 8)
    with pytest.raises(ValueError):
        sdf_from_points(p, n[:-1], LO, HI, 8)
    with pytest.raises(ValueError):
        sdf_from_points(p[:, :2], n[:, :2], LO, HI, 8)
    with pytest.raises(ValueError):
        surface_nets(np.zeros((4, 4)), [np.linspace(0, 1, 4)] * 3)


def test_an_empty_isosurface_is_an_empty_mesh_not_a_crash():
    grids = [np.linspace(0, 1, 6)] * 3
    V, Q = surface_nets(np.ones((6, 6, 6)), grids)
    assert V.shape == (0, 3) and Q.shape == (0, 4)
    assert is_watertight(Q) is False                        # an empty mesh is not watertight; it is empty


def test_chunking_does_not_change_the_answer():
    p, n = _sphere_cloud(200)
    a, _g = sdf_from_points(p, n, LO, HI, 12, chunk=10 ** 6)
    b, _g = sdf_from_points(p, n, LO, HI, 12, chunk=37)      # a deliberately awkward chunk
    assert np.array_equal(a, b)


def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    p, n = _sphere_cloud(400)
    V, Q, _F, _g = m.points_to_mesh(p, n, LO, HI, 20)
    rep = m.mesh_report(V, Q, sdf=_sphere_sdf)
    assert rep["watertight"] and rep["euler"] == 2 and rep["surface_error"] < 0.2

    for phrase in ("convert splats to a mesh", "surface reconstruction from points", "marching cubes",
                   "dual contouring"):
        assert "Points to mesh" in str(m.find_capability(phrase)[:3]), phrase


# ---------------------------------------------------------------------------------------------------------
# ORIENTATION -- a defect found when a downstream consumer (a half-edge structure) refused the mesh
# ---------------------------------------------------------------------------------------------------------

def _sdf_grid(fn, res=18, ext=1.7):
    grids = [np.linspace(-ext, ext, res) for _ in range(3)]
    G = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    return fn(G), grids


def test_the_extracted_mesh_is_oriented_not_merely_watertight():
    # WATERTIGHT IS NOT ORIENTED. Before the fix: is_watertight True, is_oriented False, 228 directed edges
    # duplicated, 98 of 200 normals inward. `is_watertight` counts UNDIRECTED edges and cannot see it.
    field, grids = _sdf_grid(lambda G: np.linalg.norm(G, axis=-1) - 1.0)
    V, Q = surface_nets(field, grids)
    assert is_watertight(Q)
    assert is_oriented(Q)


def test_every_normal_points_along_the_field_gradient():
    field, grids = _sdf_grid(lambda G: np.linalg.norm(G, axis=-1) - 1.0)
    V, Q = surface_nets(field, grids)
    outward = sum(1 for q in Q if np.cross(V[q][1] - V[q][0], V[q][2] - V[q][0]) @ V[q].mean(axis=0) > 0)
    assert outward == len(Q)                                  # outward for an SDF; 576/576 on this sphere


def test_orientation_holds_on_a_genus_one_surface_too():
    # A torus exercises the frame-parity flip on every axis. Only the crossing sign was fixed at first, and that
    # left 136 of 408 normals outward -- the permutation (1, 0, 2) is odd, so axis 1 needs a second flip.
    def torus(G):
        return np.sqrt((np.sqrt(G[..., 0] ** 2 + G[..., 1] ** 2) - 1.0) ** 2 + G[..., 2] ** 2) - 0.35

    field, grids = _sdf_grid(torus, res=22)
    V, Q = surface_nets(field, grids)
    assert is_watertight(Q) and is_oriented(Q)


def test_an_unoriented_mesh_is_watertight_which_is_the_whole_problem():
    # Reverse ONE quad: still watertight (undirected counts unchanged), no longer oriented.
    field, grids = _sdf_grid(lambda G: np.linalg.norm(G, axis=-1) - 1.0)
    _V, Q = surface_nets(field, grids)
    Q2 = Q.copy()
    Q2[0] = Q2[0][::-1]
    assert is_watertight(Q2)                                  # ... says nothing is wrong
    assert not is_oriented(Q2)                                # ... and something is


def test_the_oriented_mesh_builds_a_half_edge_structure():
    # THE CONSUMER THAT CAUGHT IT. `Mesh.half_edges()` raises on a directed edge seen twice.
    from holographic.mesh_and_geometry.holographic_mesh import Mesh

    field, grids = _sdf_grid(lambda G: np.linalg.norm(G, axis=-1) - 1.0)
    V, Q = surface_nets(field, grids)
    tris = np.array([t for a, b, c, d in Q for t in ([a, b, c], [a, c, d])], int)
    mesh = Mesh(V, tris)
    assert mesh.is_closed()
    assert mesh.euler_characteristic() == 2                   # a sphere, and the topology proves the winding


def test_the_torus_has_euler_characteristic_zero():
    from holographic.mesh_and_geometry.holographic_mesh import Mesh

    def torus(G):
        return np.sqrt((np.sqrt(G[..., 0] ** 2 + G[..., 1] ** 2) - 1.0) ** 2 + G[..., 2] ** 2) - 0.35

    field, grids = _sdf_grid(torus, res=22)
    V, Q = surface_nets(field, grids)
    tris = np.array([t for a, b, c, d in Q for t in ([a, b, c], [a, c, d])], int)
    mesh = Mesh(V, tris)
    assert mesh.is_closed() and mesh.euler_characteristic() == 0 and mesh.genus() == 1


def test_is_oriented_is_wired_to_the_mind():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    field, grids = _sdf_grid(lambda G: np.linalg.norm(G, axis=-1) - 1.0)
    _V, Q = surface_nets(field, grids)
    assert m.mesh_is_oriented(Q) is True

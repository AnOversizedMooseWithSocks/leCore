"""F2 -- the smoothest 4-RoSy cross field, and the retraction of its own bar.

The previous session probed F2, failed, and recorded the failure honestly: *"a cross field that fails its own
topological invariant is not a cross field. F2 stays open, with its bar written down."* The bar was Poincare-Hopf:
the singularity indices must sum to the Euler characteristic.

**The bar was wrong.** It is true, it is exact, and it holds for a random field, an all-zero field and an
adversarial one. The matching integers are antisymmetric, so their contribution cancels around every dual edge and
what remains is a function of the MESH alone.

    field                      sum(index)   singularities   energy
    smoothest (eigenvector)       +2.0            49          54.7
    uniformly random              +2.0           127        1542.2
    all-zero                      +2.0           203

**A bar that passes for every input is not a bar.** Judge a field by its singularity COUNT and its Dirichlet
ENERGY. Poincare-Hopf validates the transport and the dual rings -- which is worth having, and is not what it was
advertised as.
"""

import numpy as np
import pytest

from holographic.mesh_and_geometry.holographic_crossfield import (
    angle_defect, connection, connection_laplacian, cross_field, field_energy, field_report, singularity_index,
    vertex_rings)
from holographic.mesh_and_geometry.holographic_isosurface import is_oriented, surface_nets
from holographic.mesh_and_geometry.holographic_mesh import Mesh, tetrahedron


def _from_sdf(fn, res=14, ext=1.7):
    grids = [np.linspace(-ext, ext, res)] * 3
    G = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    V, Q = surface_nets(fn(G), grids)
    assert is_oriented(Q)                                      # the module needs it; surface_nets learned it here
    return Mesh(V, np.array([t for a, b, c, d in Q for t in ([a, b, c], [a, c, d])], int))


def _sphere(res=14):
    return _from_sdf(lambda G: np.linalg.norm(G, axis=-1) - 1.0, res)


def _torus(res=18):
    return _from_sdf(lambda G: np.sqrt((np.sqrt(G[..., 0] ** 2 + G[..., 1] ** 2) - 1.0) ** 2 + G[..., 2] ** 2) - 0.35,
                     res)


def test_selftest_runs():
    from holographic.mesh_and_geometry import holographic_crossfield as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# the mesh side: Gauss-Bonnet, and the rings that Poincare-Hopf DOES validate
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mesh_fn,chi", [(tetrahedron, 2), (_sphere, 2), (_torus, 0)])
def test_gauss_bonnet_is_exact(mesh_fn, chi):
    mesh = mesh_fn()
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    assert mesh.euler_characteristic() == chi
    assert abs(angle_defect(V, F).sum() / (2 * np.pi) - chi) < 1e-9


def test_every_vertex_ring_closes_on_a_closed_mesh():
    for mesh in (tetrahedron(), _sphere(), _torus()):
        V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
        _rho, opp, nxt, _de = connection(V, F)
        rings = vertex_rings(F, opp, nxt, len(V))
        assert len(rings) == len(V)


def test_kept_negative_the_transport_is_antisymmetric_by_construction():
    # Computing rho from both directed edges lets atan2's branch cut differ by 2pi, which shifts the matching by 4
    # and the index by 1 PER EDGE. The sphere's indices summed to -43 instead of +2.
    mesh = _sphere()
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    rho, _opp, _nxt, dual = connection(V, F)
    for (f, g) in dual:
        assert rho[(f, g)] == -rho[(g, f)]                     # exactly, not approximately


def test_the_connection_laplacian_is_hermitian_and_psd():
    mesh = _sphere()
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    rho, _o, _n, dual = connection(V, F)
    L = connection_laplacian(F, rho, dual)
    assert np.abs(L - L.conj().T).max() < 1e-12
    assert np.linalg.eigvalsh(L).min() > -1e-9


def test_an_unoriented_mesh_is_refused():
    mesh = _sphere()
    F = np.asarray(mesh.faces, int).copy()
    F[0] = F[0][::-1]
    with pytest.raises(ValueError, match="oriented|twice"):
        cross_field(Mesh(np.asarray(mesh.vertices, float), F))


# ---------------------------------------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------------------------------------

def test_a_tetrahedron_admits_a_perfectly_smooth_field():
    phi, ctx = cross_field(tetrahedron())
    rep = field_report(tetrahedron(), phi, ctx)
    assert rep["lambda_min"] < 1e-9                            # the connection is trivial here
    assert rep["energy"] < 1e-20
    assert rep["sum_index"] == 2.0 and rep["n_singularities"] == 4   # four +1/2 cone points


def test_the_eigen_solve_beats_random_restarts():
    mesh = _sphere()
    phi, ctx = cross_field(mesh)
    e = field_energy(phi, ctx)
    rng = np.random.default_rng(0)
    for _ in range(8):
        junk = rng.uniform(-np.pi / 4, np.pi / 4, len(mesh.faces))
        assert e < field_energy(junk, ctx)


def test_the_field_is_deterministic():
    mesh = _sphere()
    a, _ = cross_field(mesh)
    b, _ = cross_field(mesh)
    assert np.array_equal(a, b)


def test_the_field_angles_live_in_a_quarter_turn():
    phi, _ctx = cross_field(_sphere())
    assert phi.min() > -np.pi / 4 - 1e-9 and phi.max() <= np.pi / 4 + 1e-9


# ---------------------------------------------------------------------------------------------------------
# THE INDEX, and THE RETRACTION
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mesh_fn,chi", [(tetrahedron, 2), (_sphere, 2), (_torus, 0)])
def test_the_index_is_exactly_a_multiple_of_a_quarter_and_sums_to_chi(mesh_fn, chi):
    mesh = mesh_fn()
    phi, ctx = cross_field(mesh)
    idx = singularity_index(phi, ctx)
    q = idx * 4.0
    assert np.abs(q - np.round(q)).max() < 1e-9                # an exact quarter, not "close to" one
    assert abs(idx.sum() - chi) < 1e-9


@pytest.mark.parametrize("mesh_fn,chi", [(_sphere, 2), (_torus, 0)])
def test_kept_negative_a_random_field_satisfies_poincare_hopf_just_as_exactly(mesh_fn, chi):
    # THE RETRACTION. The bar recorded for this item is vacuous: it constrains the MESH, not the field.
    mesh = mesh_fn()
    _phi, ctx = cross_field(mesh)
    rng = np.random.default_rng(0)
    for field in (rng.uniform(-np.pi / 4, np.pi / 4, len(mesh.faces)),
                  np.zeros(len(mesh.faces)),
                  np.where(np.arange(len(mesh.faces)) % 2 == 0, 0.0, np.pi / 4)):
        idx = singularity_index(field, ctx)
        assert abs(idx.sum() - chi) < 1e-9                     # holds for garbage
        q = idx * 4.0
        assert np.abs(q - np.round(q)).max() < 1e-9            # and the quarters hold too


def test_what_actually_separates_a_good_field_from_a_bad_one():
    mesh = _sphere()
    phi, ctx = cross_field(mesh)
    rep = field_report(mesh, phi, ctx)

    junk = np.random.default_rng(0).uniform(-np.pi / 4, np.pi / 4, len(mesh.faces))
    junk_idx = singularity_index(junk, ctx)
    n_junk = int((np.abs(junk_idx) > 1e-9).sum())

    assert rep["n_singularities"] < n_junk                     # the COUNT separates
    assert rep["energy"] < field_energy(junk, ctx) / 10.0      # ... and so does the ENERGY, by an order of magnitude
    assert rep["poincare_hopf"] is True                        # ... while the invariant does not


def test_the_report_carries_every_number_and_warns_about_the_one_that_lies():
    rep = field_report(_sphere())
    for k in ("lambda_min", "energy", "n_singularities", "sum_index", "euler",
              "quarter_residual", "poincare_hopf"):
        assert k in rep
    assert rep["quarter_residual"] < 1e-9
    assert rep["sum_index"] == float(rep["euler"])


def test_a_torus_can_be_smoother_than_a_sphere_because_chi_is_zero():
    # A torus admits a singularity-free field in principle (chi = 0); a sphere cannot (chi = 2). On these irregular
    # meshes neither reaches its ideal, which is stated rather than hidden.
    sphere, torus = field_report(_sphere()), field_report(_torus())
    assert sphere["euler"] == 2 and torus["euler"] == 0
    assert sphere["sum_index"] == 2.0 and torus["sum_index"] == 0.0
    assert sphere["n_singularities"] > 8                       # far from the ideal 8 cone points of index 1/4


# ---------------------------------------------------------------------------------------------------------
# guards + wiring
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_surface_nets_is_not_manifold_at_every_resolution_and_the_guard_catches_it():
    # A torus at grid resolution 14, 18 and 20 extracts watertight AND oriented. At 16 it is NEITHER: the ambiguous
    # cell, where one dual vertex per cell assumes the surface crosses the cell once. Non-monotonic in resolution
    # (14 works, 16 does not), so it is a CONFIGURATION failure and no "use a finer grid" rule fixes it. The remedy
    # is to CHECK, and `cross_field` refuses such a mesh rather than transporting a frame across a bad face.
    from holographic.mesh_and_geometry.holographic_isosurface import is_watertight

    def _extract(res):
        grids = [np.linspace(-1.7, 1.7, res)] * 3
        G = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
        field = np.sqrt((np.sqrt(G[..., 0] ** 2 + G[..., 1] ** 2) - 1.0) ** 2 + G[..., 2] ** 2) - 0.35
        return surface_nets(field, grids)

    for good in (14, 18, 20):
        _V, Q = _extract(good)
        assert is_watertight(Q) and is_oriented(Q), good

    V, Q = _extract(16)
    assert is_watertight(Q) is False                            # not merely mis-wound: NOT MANIFOLD
    assert is_oriented(Q) is False

    tris = np.array([t for a, b, c, d in Q for t in ([a, b, c], [a, c, d])], int)
    with pytest.raises(ValueError):
        cross_field(Mesh(V, tris))


def test_a_tiny_or_open_mesh_is_refused():
    with pytest.raises(ValueError, match="at least 4 faces"):
        cross_field(Mesh(np.eye(3), np.array([[0, 1, 2]])))

    from holographic.mesh_and_geometry.holographic_meshuv import flat_grid_mesh
    with pytest.raises(ValueError, match="CLOSED"):
        cross_field(flat_grid_mesh(4))                         # a disk has boundary edges


def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    phi, ctx = m.cross_field(tetrahedron())
    idx = m.singularity_index(phi, ctx)
    assert abs(idx.sum() - 2.0) < 1e-9

    rep = m.field_report(tetrahedron(), phi, ctx)
    assert rep["poincare_hopf"] is True and rep["energy"] < 1e-20

    assert "Cross field" in str(m.find_capability("smoothest direction field")[:3])

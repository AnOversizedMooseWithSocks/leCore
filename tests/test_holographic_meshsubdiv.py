"""Tests for Loop subdivision (FWD-8): exact topological refinement (faces x4, V'=V+E, chi preserved, closed
manifold), affine reproduction (a flat mesh stays flat to machine precision -- the rigor reference), the
smoothing low-pass signature (dihedral spread drops on an angular mesh), multi-level, triangle output,
determinism."""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import box
from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere
from holographic.mesh_and_geometry.holographic_meshuv import flat_grid_mesh
from holographic.mesh_and_geometry.holographic_meshcurvature import dihedral_angles
from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide, _triangles
from holographic.mesh_and_geometry.holographic_mesh import Mesh


def test_subdivide_quadruples_faces():
    s = _icosphere(1)
    assert loop_subdivide(s, 1).n_faces == 4 * s.n_faces


def test_subdivide_adds_one_vertex_per_edge():
    s = _icosphere(1)
    assert loop_subdivide(s, 1).n_vertices == s.n_vertices + len(s.edges())


def test_subdivide_preserves_chi_closed_manifold():
    s = _icosphere(1)
    sub = loop_subdivide(s, 1)
    assert sub.euler_characteristic() == s.euler_characteristic()
    assert sub.is_closed() and sub.is_manifold()


def test_subdivide_flat_mesh_stays_flat():
    # the affine-reproduction rigor reference: the Loop masks are barycentric, so a planar input is planar out
    flat = flat_grid_mesh(5)
    assert float(np.max(np.abs(loop_subdivide(flat, 2).vertices[:, 2]))) < 1e-12


def test_subdivide_smooths_an_angular_mesh():
    cube = box()
    before = float(np.std(list(dihedral_angles(Mesh(cube.vertices.copy(), _triangles(cube))).values())))
    after = float(np.std(list(dihedral_angles(loop_subdivide(cube, 2)).values())))
    assert after < before * 0.5                            # the low-pass smooth roughly halves the spread (or more)


def test_two_levels_quadruple_faces_twice():
    s = _icosphere(1)
    assert loop_subdivide(s, 2).n_faces == 16 * s.n_faces  # x4 per level


def test_subdivide_output_is_all_triangles():
    # Loop is a triangle scheme; a quad input is triangulated, and the output is pure-triangle
    sub = loop_subdivide(box(), 1)
    assert all(len(f) == 3 for f in sub.faces)


def test_subdivide_is_deterministic():
    s = _icosphere(1)
    assert np.array_equal(loop_subdivide(s, 1).vertices, loop_subdivide(s, 1).vertices)


# --- Change 3: the vectorized subdivision-matrix path must be bit-identical to the reference loop
#     (positions within TOL, topology EXACT) ---
import numpy as _np_sd
from holographic.mesh_and_geometry.holographic_meshsubdiv import _one_level as _ref_one_level, _one_level_matrix as _fast_one_level
from holographic.mesh_and_geometry.holographic_mesh import box as _sd_box, grid as _sd_grid, tetrahedron as _sd_tet


def test_subdivision_matrix_bit_identical_to_loop():
    for m in (_sd_box(2, 2, 2), _sd_grid(5, 4), _sd_tet()):
        ref = _ref_one_level(m); fast = _fast_one_level(m)
        # topology EXACT
        assert [tuple(f) for f in ref.faces] == [tuple(f) for f in fast.faces]
        assert ref.vertices.shape == fast.vertices.shape
        # positions within TOL (only float summation order differs)
        assert _np_sd.abs(ref.vertices - fast.vertices).max() < 1e-9


def test_subdivision_matrix_multilevel_and_euler():
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
    m = _sd_box(1, 1, 1)
    sub = loop_subdivide(m, levels=2)
    # each level: faces x4; a closed cube (chi=2) stays chi=2
    assert len(sub.faces) == len(_sd_box(1, 1, 1).faces) * 2 * 16  # box has quads -> triangulated x2, then x4 x4
    assert sub.euler_characteristic() == 2


# ======================================================================================================
# The LIMIT surface: iterate's k -> infinity, on the local Loop operator. Closes the last unifier PENDING.
# ======================================================================================================
def _sphere():
    from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere
    return _icosphere(1)


def test_the_loop_ring_block_is_a_bind_operator_and_iterate_owns_its_spectrum():
    """The part of the local Loop operator that is NOT shift-invariant is only the centre vertex. The ring-to-ring
    block is exactly a circulant, so iterate.transfer (an rfft) IS its eigendecomposition -- for free."""
    from holographic.mesh_and_geometry.holographic_meshsubdiv import _ring_kernel
    from holographic.misc.holographic_iterate import transfer
    for n in (3, 5, 6, 7):
        c = _ring_kernel(n)
        circ = np.stack([np.roll(c, i) for i in range(n)])          # the ring block as a dense matrix
        got = np.sort_complex(np.linalg.eigvals(circ))
        want = np.sort_complex(np.fft.fft(c))
        assert np.max(np.abs(got - want)) < 1e-12, n                # transfer really is the eigendecomposition
        lam = np.real(transfer(c))
        assert abs(lam[0] - 0.625) < 1e-12, (n, lam[0])             # mode 0: 5/8 at EVERY valence


def test_warrens_beta_is_read_off_the_spectrum_not_hard_coded():
    """lambda_1 = 3/8 + cos(2 pi / n)/4 is, to the last bit, the term inside Warren's beta. The subdivision mask's
    own parameter is a property of the ring's eigenvalues."""
    from holographic.mesh_and_geometry.holographic_meshsubdiv import _ring_kernel
    from holographic.misc.holographic_iterate import transfer
    for n in (3, 4, 5, 6, 7, 9):
        lam = np.real(transfer(_ring_kernel(n)))
        assert abs(lam[1] - (0.375 + 0.25 * np.cos(2 * np.pi / n))) < 1e-12, n
        beta_spec = (1.0 / n) * (lam[0] - lam[1] ** 2)
        beta_warren = (1.0 / n) * (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * np.cos(2.0 * np.pi / n)) ** 2)
        assert abs(beta_spec - beta_warren) < 1e-15, (n, beta_spec, beta_warren)
    assert abs((1.0 / 3.0) * (0.625 - 0.25 ** 2) - 3.0 / 16.0) < 1e-15   # the classical valence-3 beta


def test_the_closed_form_limit_is_what_infinite_subdivision_converges_to():
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_limit, loop_subdivide
    s = _sphere()
    P, _ = loop_limit(s)
    errs = [float(np.max(np.abs(loop_subdivide(s, k).vertices[:s.n_vertices] - P))) for k in (4, 6, 8)]
    assert errs[0] > errs[1] > errs[2], errs                        # monotone
    assert errs[2] < 1e-5, errs                                     # measured 2.3e-6
    assert errs[0] / errs[1] > 8.0 and errs[1] / errs[2] > 8.0, errs  # at the subdominant eigenvalue's rate


def test_the_limit_normal_is_exact_not_area_weighted():
    """Modes +-1 of the ring span the tangent plane, so the normal is EXACT. On a sphere it must be radial to
    machine precision -- an area-weighted face normal is not."""
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_limit
    P, N = loop_limit(_sphere())
    radial = P / np.linalg.norm(P, axis=1, keepdims=True)
    assert float(np.abs(np.abs((N * radial).sum(1)) - 1.0).max()) < 1e-9


def test_affine_reproduction_survives_the_limit_and_boundaries_are_handled():
    from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_limit
    from holographic.mesh_and_geometry.holographic_meshuv import flat_grid_mesh
    flat = flat_grid_mesh(5)                                        # planar, and it has a BOUNDARY
    P, N = loop_limit(flat)
    assert float(np.max(np.abs(P[:, 2]))) < 1e-12                   # a planar mesh has a planar limit surface
    assert float(np.abs(np.abs(N[:, 2]) - 1.0).max()) < 1e-9        # ...and every limit normal is +-z
    assert np.isfinite(P).all() and np.isfinite(N).all()


def test_limit_surface_through_the_mind_and_is_deterministic():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    s = _sphere()
    P1, N1 = m.mesh_limit_surface(s)
    P2, N2 = m.mesh_limit_surface(s)
    assert np.array_equal(P1, P2) and np.array_equal(N1, N2)        # bit-identical, no RNG anywhere
    assert P1.shape == (s.n_vertices, 3) and N1.shape == (s.n_vertices, 3)
    assert any("limit surface" in c.name.lower() for c in m.find_capability("subdivision limit"))

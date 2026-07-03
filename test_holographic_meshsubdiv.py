"""Tests for Loop subdivision (FWD-8): exact topological refinement (faces x4, V'=V+E, chi preserved, closed
manifold), affine reproduction (a flat mesh stays flat to machine precision -- the rigor reference), the
smoothing low-pass signature (dihedral spread drops on an angular mesh), multi-level, triangle output,
determinism."""

import numpy as np

from holographic_mesh import box
from holographic_meshsmooth import _icosphere
from holographic_meshuv import flat_grid_mesh
from holographic_meshcurvature import dihedral_angles
from holographic_meshsubdiv import loop_subdivide, _triangles
from holographic_mesh import Mesh


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
from holographic_meshsubdiv import _one_level as _ref_one_level, _one_level_matrix as _fast_one_level
from holographic_mesh import box as _sd_box, grid as _sd_grid, tetrahedron as _sd_tet


def test_subdivision_matrix_bit_identical_to_loop():
    for m in (_sd_box(2, 2, 2), _sd_grid(5, 4), _sd_tet()):
        ref = _ref_one_level(m); fast = _fast_one_level(m)
        # topology EXACT
        assert [tuple(f) for f in ref.faces] == [tuple(f) for f in fast.faces]
        assert ref.vertices.shape == fast.vertices.shape
        # positions within TOL (only float summation order differs)
        assert _np_sd.abs(ref.vertices - fast.vertices).max() < 1e-9


def test_subdivision_matrix_multilevel_and_euler():
    from holographic_meshsubdiv import loop_subdivide
    m = _sd_box(1, 1, 1)
    sub = loop_subdivide(m, levels=2)
    # each level: faces x4; a closed cube (chi=2) stays chi=2
    assert len(sub.faces) == len(_sd_box(1, 1, 1).faces) * 2 * 16  # box has quads -> triangulated x2, then x4 x4
    assert sub.euler_characteristic() == 2

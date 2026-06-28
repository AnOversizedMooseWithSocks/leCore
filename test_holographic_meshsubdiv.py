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

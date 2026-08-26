"""Tests for the FWD-7 modeler-verb remainder (holographic_meshverbs2): bevel, bridge, loop-cut -- the three verbs
that need vertex duplication or edge-loop tracing. Bevel chamfers a corner (chi preserved), bridge joins two loops
into a tube, loop-cut inserts an edge loop through a quad strip (chi preserved)."""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh, box, grid
from holographic.mesh_and_geometry.holographic_meshverbs2 import bevel_vertex, bridge_loops, loop_cut
from holographic.mesh_and_geometry.holographic_meshseam import _boundary_loop_count


# ---- bevel ----------------------------------------------------------------------------------------
def test_bevel_is_closed_manifold_with_chi_preserved():
    bev = bevel_vertex(box(), 0, ratio=0.3)
    assert bev.is_manifold() and bev.is_closed() and bev.euler_characteristic() == 2


def test_bevel_makes_three_pentagons_and_a_triangle_cap():
    sizes = sorted(len(f) for f in bevel_vertex(box(), 0, ratio=0.3).faces)
    assert sizes.count(5) == 3 and sizes.count(3) == 1


def test_bevel_removes_the_corner_and_adds_three_vertices():
    cube = box()
    assert bevel_vertex(cube, 0, 0.3).n_vertices == cube.n_vertices - 1 + 3


def test_bevel_new_vertices_sit_near_the_corner_for_small_ratio():
    cube = box()
    corner = cube.vertices[0]
    bev = bevel_vertex(cube, 0, ratio=0.2)
    # the 3 new vertices (not among the original cube vertices) cluster near the original corner
    originals = {tuple(np.round(v, 6)) for v in cube.vertices}
    news = [v for v in bev.vertices if tuple(np.round(v, 6)) not in originals]
    assert len(news) == 3
    assert all(np.linalg.norm(v - corner) < 0.5 for v in news)   # ratio 0.2 of a unit-edge cube


# ---- bridge ---------------------------------------------------------------------------------------
def _two_squares():
    sq0 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    sq1 = sq0.copy(); sq1[:, 2] = 1.0
    return np.vstack([sq0, sq1])


def test_bridge_two_squares_is_an_open_tube():
    tube = bridge_loops(_two_squares(), [0, 1, 2, 3], [4, 5, 6, 7], closed=True)
    assert tube.is_manifold() and not tube.is_closed()
    assert tube.n_faces == 4 and tube.euler_characteristic() == 0


def test_bridge_tube_has_two_boundary_loops():
    tube = bridge_loops(_two_squares(), [0, 1, 2, 3], [4, 5, 6, 7], closed=True)
    assert _boundary_loop_count(tube) == 2


def test_bridge_unequal_loops_raises():
    try:
        bridge_loops(_two_squares(), [0, 1, 2], [4, 5, 6, 7])
        assert False
    except ValueError:
        pass


# ---- loop-cut -------------------------------------------------------------------------------------
def test_loop_cut_box_preserves_chi_and_adds_four_faces():
    cube = box()
    f0 = tuple(cube.faces[0])
    lc = loop_cut(cube, 0, (f0[0], f0[1]))
    assert lc.is_manifold() and lc.is_closed() and lc.euler_characteristic() == 2
    assert lc.n_faces == cube.n_faces + 4


def test_loop_cut_grid_preserves_chi_and_adds_three_faces():
    g = grid(3, 3)
    fg = tuple(g.faces[0])
    lcg = loop_cut(g, 0, (fg[0], fg[1]))
    assert lcg.is_manifold() and lcg.euler_characteristic() == 1
    assert lcg.n_faces == g.n_faces + 3


def test_loop_cut_on_a_triangle_mesh_raises():
    tri = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float), [(0, 1, 2)])
    try:
        loop_cut(tri, 0, (0, 1))
        assert False
    except ValueError:
        pass


# ---- determinism ----------------------------------------------------------------------------------
def test_verbs_are_deterministic():
    cube = box()
    f0 = tuple(cube.faces[0])
    assert np.array_equal(bevel_vertex(cube, 0, 0.3).vertices, bevel_vertex(cube, 0, 0.3).vertices)
    assert np.array_equal(loop_cut(cube, 0, (f0[0], f0[1])).vertices, loop_cut(cube, 0, (f0[0], f0[1])).vertices)

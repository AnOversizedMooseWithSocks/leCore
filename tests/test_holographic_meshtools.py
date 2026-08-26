"""Tests for the classic mesh tools mirror and merge_by_distance / weld (ANIM-3)."""

import numpy as np
from holographic.mesh_and_geometry.holographic_mesh import Mesh, box, grid
from holographic.mesh_and_geometry.holographic_meshtools import merge_by_distance, mirror


def test_weld_collapses_duplicate_vertices():
    b = box()
    # stack a duplicate copy of every vertex; faces still index the first copy
    dup = Mesh(np.vstack([b.vertices, b.vertices]), [tuple(f) for f in b.faces])
    w = merge_by_distance(dup, tol=1e-5)
    assert w.n_vertices == b.n_vertices                       # the duplicates fused back to the originals


def test_weld_drops_degenerate_faces():
    # a triangle whose two vertices coincide collapses and must be removed
    V = np.array([[0., 0, 0], [1, 0, 0], [1.0 + 1e-9, 0, 0]])  # verts 1 and 2 are within tol
    m = Mesh(V, [(0, 1, 2)])
    w = merge_by_distance(m, tol=1e-5)
    assert w.n_faces == 0                                     # the degenerate triangle was dropped


def test_mirror_is_symmetric_about_the_plane():
    g = grid(4, 4)
    g.vertices[:, 0] = np.abs(g.vertices[:, 0])               # fold to the +x half
    m = mirror(g, axis=0, plane=0.0, weld=True)
    assert np.allclose(m.vertices[:, 0].min(), -m.vertices[:, 0].max(), atol=1e-6)
    assert m.n_vertices < g.n_vertices * 2                    # the seam on x=0 welded


def test_mirror_reverses_winding_on_the_reflected_half():
    # a single triangle mirrored across x=0: the reflected copy must have reversed winding so its normal
    # points consistently (a reflection flips orientation)
    V = np.array([[1.0, 0, 0], [2, 0, 0], [1, 1, 0]])
    m = mirror(Mesh(V, [(0, 1, 2)]), axis=0, plane=0.0, weld=False)
    assert m.n_faces == 2
    # original winding (0,1,2); reflected face indices are reversed
    assert tuple(m.faces[1]) == (5, 4, 3)


def test_solidify_closes_an_open_sheet_into_a_watertight_solid():
    from holographic.mesh_and_geometry.holographic_mesh import grid
    from holographic.mesh_and_geometry.holographic_meshtools import solidify
    g = grid(5, 5)
    solid = solidify(g, 0.1)
    top = solid.validate_topology()
    assert top["watertight"] and top["manifold_edges"]       # open sheet -> closed manifold slab
    assert solid.n_faces > g.n_faces * 2                     # two shells + the bridge ring
    assert abs((solid.vertices[:, 2].max() - solid.vertices[:, 2].min()) - 0.1) < 1e-6

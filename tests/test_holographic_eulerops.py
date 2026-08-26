"""Tests for the local Euler edit operators (FWD-7): the chi/manifold invariants each operator must preserve,
the make/kill round-trips (flip-back, split-then-collapse) that are the exact-inverse correctness witnesses,
the guarded preconditions (the collapse link condition, the flip duplicate-edge refusal), and determinism.
The per-element-loop speed bound is a docstring negative, not a test -- only correctness is asserted here."""

import numpy as np
from collections import Counter

from holographic.mesh_and_geometry.holographic_mesh import Mesh, box
from holographic.misc.holographic_eulerops import flip_edge, split_edge, collapse_edge, split_face, _face_with_directed_edge, _third


# ---- shared fixtures / helpers ----------------------------------------------------------------------
def _tri_cube():
    """A triangulated unit-ish cube: a closed triangle manifold, chi = 2."""
    c = box(2.0, 2.0, 2.0)
    return Mesh(c.vertices.copy(), [tuple(t) for t in c.triangulate()])


def _canon(mesh):
    """Canonical connectivity (face rotated to lowest vertex, faces sorted) -- 'same mesh' up to order."""
    out = []
    for f in mesh.faces:
        k = f.index(min(f))
        out.append(tuple(f[k:] + f[:k]))
    return tuple(sorted(out))


def _a_flippable_edge(tm):
    """Return (a, b, c, d): an interior edge {a,b} of two triangles whose apexes c,d are not already an edge."""
    dirset = Counter()
    for f in tm.faces:
        for k in range(3):
            dirset[(f[k], f[(k + 1) % 3])] += 1
    edgeset = set(tm.edges())
    for (x, y) in dirset:
        if (y, x) not in dirset:
            continue
        fa, fb = _face_with_directed_edge(tm.faces, x, y), _face_with_directed_edge(tm.faces, y, x)
        if len(tm.faces[fa]) != 3 or len(tm.faces[fb]) != 3:
            continue
        cc, dd = _third(tm.faces[fa], x, y), _third(tm.faces[fb], x, y)
        if (min(cc, dd), max(cc, dd)) not in edgeset:
            return x, y, cc, dd
    raise AssertionError("no flippable interior edge found")


def _bipyramid():
    """A triangular bipyramid: closed, chi = 2, with non-collapsible equatorial edges (link condition)."""
    v = np.array([[0, 0, 1], [1, 0, 0], [-0.5, 0.87, 0], [-0.5, -0.87, 0], [0, 0, -1]], dtype=float)
    f = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (4, 2, 1), (4, 3, 2), (4, 1, 3)]
    return Mesh(v, f)


# ---- flip_edge --------------------------------------------------------------------------------------
def test_flip_edge_is_chi_and_count_invariant():
    tm = _tri_cube()
    a, b, _, _ = _a_flippable_edge(tm)
    out = flip_edge(tm, a, b)
    assert out.is_manifold() and out.is_closed()
    assert out.euler_characteristic() == tm.euler_characteristic()      # chi invariant
    assert (out.n_vertices, out.n_faces) == (tm.n_vertices, tm.n_faces)  # V and F unchanged


def test_flip_edge_round_trips():
    tm = _tri_cube()
    a, b, c, d = _a_flippable_edge(tm)
    once = flip_edge(tm, a, b)
    assert _canon(flip_edge(once, c, d)) == _canon(tm)                  # flip the new edge -> original


def test_flip_edge_refuses_creating_a_duplicate_edge():
    # a cube EDGE (not a quad diagonal) flips into an already-existing edge -> must be refused
    tm = _tri_cube()
    edgeset = set(tm.edges())
    dirset = Counter()
    for f in tm.faces:
        for k in range(3):
            dirset[(f[k], f[(k + 1) % 3])] += 1
    bad = None
    for (x, y) in dirset:
        if (y, x) not in dirset:
            continue
        fa, fb = _face_with_directed_edge(tm.faces, x, y), _face_with_directed_edge(tm.faces, y, x)
        cc, dd = _third(tm.faces[fa], x, y), _third(tm.faces[fb], x, y)
        if (min(cc, dd), max(cc, dd)) in edgeset:
            bad = (x, y)
            break
    assert bad is not None, "the triangulated cube should have an edge whose flip duplicates another"
    try:
        flip_edge(tm, *bad)
        assert False, "flip into an existing edge must raise"
    except ValueError:
        pass


# ---- split_edge / collapse_edge (the make/kill pair) ------------------------------------------------
def test_split_edge_adds_one_vertex_and_preserves_chi():
    tm = _tri_cube()
    a, b, _, _ = _a_flippable_edge(tm)
    out, m = split_edge(tm, a, b)
    assert m == tm.n_vertices                                           # new vertex appended (deterministic)
    assert out.n_vertices == tm.n_vertices + 1
    assert out.euler_characteristic() == tm.euler_characteristic()      # chi unchanged across refinement
    assert out.is_manifold() and out.is_closed()


def test_split_then_collapse_is_an_exact_inverse():
    tm = _tri_cube()
    a, b, _, _ = _a_flippable_edge(tm)
    split, m = split_edge(tm, a, b)
    back = collapse_edge(split, keep=a, remove=m)
    assert back is not None
    assert back.n_vertices == tm.n_vertices
    assert _canon(back) == _canon(tm)                                  # do-then-undo restores the mesh exactly


def test_collapse_edge_refuses_link_condition_violation():
    bp = _bipyramid()
    # edge {1,2}: endpoints share neighbour 3 which is not an apex of {1,2} -> unsafe -> None
    assert collapse_edge(bp, keep=1, remove=2) is None


def test_collapse_edge_legal_case_yields_tetrahedron():
    bp = _bipyramid()
    out = collapse_edge(bp, keep=0, remove=1)                          # apex onto equator: legal
    assert out is not None
    assert out.n_vertices == 4 and out.euler_characteristic() == 2
    assert out.is_manifold() and out.is_closed()


def test_split_edge_rejects_a_non_triangle_face():
    quad = box(2.0, 2.0, 2.0)                                          # quad faces
    # vertices 0 and 1 share a quad edge; that face is a quad -> split_edge must refuse
    try:
        split_edge(quad, 0, 1)
        assert False, "split_edge on a non-triangle incident face must raise"
    except ValueError:
        pass


# ---- split_face (the n-gon operator) ----------------------------------------------------------------
def test_split_face_on_a_quad_preserves_chi():
    quad = box(2.0, 2.0, 2.0)
    out = split_face(quad, 0, 0, 2)                                     # diagonal of the first quad
    assert out.n_faces == quad.n_faces + 1
    assert out.euler_characteristic() == 2 and out.is_manifold() and out.is_closed()


def test_split_face_rejects_adjacent_corners():
    quad = box(2.0, 2.0, 2.0)
    try:
        split_face(quad, 0, 0, 1)                                       # adjacent corners = an existing edge
        assert False, "split_face on adjacent corners must raise"
    except ValueError:
        pass


# ---- determinism ------------------------------------------------------------------------------------
def test_operators_are_deterministic():
    tm = _tri_cube()
    a, b, c, d = _a_flippable_edge(tm)
    s1, _ = split_edge(tm, a, b)
    s2, _ = split_edge(tm, a, b)
    assert np.array_equal(s1.vertices, s2.vertices) and s1.faces == s2.faces
    assert flip_edge(tm, a, b).faces == flip_edge(tm, a, b).faces

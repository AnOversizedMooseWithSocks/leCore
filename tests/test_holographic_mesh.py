"""Tests for the explicit polygon mesh kernel (FWD-1): the half-edge adjacency, the Euler well-formedness
invariants, normals, the OBJ/buffer round-trips, and the guards (non-manifold rejection, deterministic
integer buffers). The kept negative (Python-loop-bound build) is a docstring statement, not a test -- speed
is not asserted, only correctness and determinism."""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh, box, tetrahedron, grid


# ---- counts & Euler invariants ----------------------------------------------------------------------
def test_box_counts_and_euler():
    m = box(2.0, 2.0, 2.0)
    assert m.n_vertices == 8
    assert m.n_faces == 6
    assert m.n_edges == 12
    assert m.euler_characteristic() == 2          # genus-0 closed surface: V - E + F = 2
    assert m.is_closed()
    assert m.is_manifold()
    assert m.genus() == 0


def test_tetrahedron_invariant():
    t = tetrahedron()
    assert (t.n_vertices, t.n_edges, t.n_faces) == (4, 6, 4)
    assert t.euler_characteristic() == 2
    assert t.is_closed()
    assert t.genus() == 0


def test_grid_is_open_with_boundary():
    # an open (boundary-having) surface: chi = 1, not closed, genus undefined
    g = grid(4, 4)
    assert not g.is_closed()
    assert g.euler_characteristic() == 1
    assert g.genus() is None


# ---- half-edge adjacency ----------------------------------------------------------------------------
def test_half_edge_twins_are_reciprocal():
    he = box(1, 1, 1).half_edges()
    H = len(he["origin"])
    assert H == 24                                # 6 quads * 4 corners
    for h in range(H):
        t = int(he["twin"][h])
        assert t >= 0                             # the box is closed: every half-edge has a twin
        assert int(he["twin"][t]) == h            # and the relationship is symmetric


def test_half_edge_next_cycles_close_on_the_face():
    m = box(1, 1, 1)
    he = m.half_edges()
    for h in range(len(he["origin"])):
        cur, steps = h, 0
        while True:
            cur = int(he["nxt"][cur]); steps += 1
            if cur == h:
                break
            assert steps < 8
        assert steps == 4                         # a quad face: 4 corners, cycle length 4


def test_vertex_neighbours_and_faces():
    m = box(1, 1, 1)
    # every corner of a box touches exactly 3 faces and 3 edge-neighbours
    for v in range(m.n_vertices):
        assert len(m.vertex_faces(v)) == 3
        assert len(m.vertex_neighbours(v)) == 3


# ---- normals ----------------------------------------------------------------------------------------
def test_vertex_normals_point_outward_on_a_centred_box():
    m = box(2, 2, 2)
    nrm = m.vertex_normals()
    pos_dir = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    dots = np.sum(nrm * pos_dir, axis=1)
    assert np.all(dots > 0.5)                     # each vertex normal roughly along its outward position


def test_normals_are_unit_length():
    m = box(1, 1, 1)
    nrm = m.vertex_normals()
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-6)


# ---- round-trips ------------------------------------------------------------------------------------
def test_buffer_round_trip_positions_exact():
    m = box(2, 2, 2)
    buf = m.to_buffers()
    assert buf["indices"].size == 36             # 6 quads -> 12 tris -> 36 indices
    m2 = Mesh.from_buffers(buf["position"], buf["indices"], normal=buf["normal"])
    assert np.allclose(m2.vertices, m.vertices)
    assert m2.triangulate().shape == (12, 3)


def test_obj_round_trip_preserves_quad_topology():
    m = box(2, 2, 2)
    m2 = Mesh.from_obj(m.to_obj())
    assert m2.n_vertices == 8 and m2.n_faces == 6
    assert all(len(f) == 4 for f in m2.faces)    # OBJ keeps quads as quads (buffers would not)
    # vertex sets match (order is preserved by OBJ, but compare robustly)
    assert np.allclose(np.sort(m2.vertices, axis=0), np.sort(m.vertices, axis=0))


def test_obj_parses_slash_face_form():
    # f a/vt/vn form: take the vertex index before the first slash
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/1 3/3/1\n"
    m = Mesh.from_obj(text)
    assert m.n_vertices == 3 and m.faces == [(0, 1, 2)]


# ---- guards ----------------------------------------------------------------------------------------
def test_non_manifold_orientation_is_rejected():
    # two faces both traversing the directed edge (0 -> 1): inconsistent orientation / non-manifold
    verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    bad = Mesh(verts, [(0, 1, 2), (0, 1, 3)])
    assert not bad.is_manifold()
    try:
        bad.half_edges()
        assert False, "should raise on a directed edge appearing twice"
    except ValueError:
        pass


def test_degenerate_face_rejected():
    try:
        Mesh([[0, 0, 0], [1, 0, 0]], [(0, 1)])   # a 2-vertex "face"
        assert False, "a face with < 3 vertices should be rejected"
    except ValueError:
        pass


def test_index_buffer_is_deterministic():
    # the integer index buffer is the EXACT class: bit-reproducible run to run (ISA contract)
    a = box(2, 2, 2).to_buffers()["indices"]
    b = box(2, 2, 2).to_buffers()["indices"]
    assert np.array_equal(a, b)


def test_edges_listing_is_sorted_and_deterministic():
    m = box(1, 1, 1)
    e1 = m.edges()
    e2 = box(1, 1, 1).edges()
    assert e1 == e2                              # same mesh -> same edge list, same order
    assert e1 == sorted(e1)                      # returned in sorted order


def test_vertex_normals_vectorized_matches_loop_and_points_outward():
    """vertex_normals has a vectorized triangle fast-path (Newell's method over all faces at once, face-order
    scatter-add). It must be BIT-IDENTICAL to the per-face loop, and on a sphere the normals point radially outward."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    v, a = sample_field(sphere, ((-1, -1, -1), (1, 1, 1)), 14)
    m = marching_tetrahedra_vec(v, a, 0.0)

    def ref(mesh):
        V = mesh.vertices
        acc = np.zeros((len(V), 3))
        for f in mesh.faces:
            n = len(f)
            nx = ny = nz = 0.0
            for k in range(n):
                cu = V[f[k]]; nx_ = V[f[(k + 1) % n]]
                nx += (cu[1] - nx_[1]) * (cu[2] + nx_[2])
                ny += (cu[2] - nx_[2]) * (cu[0] + nx_[0])
                nz += (cu[0] - nx_[0]) * (cu[1] + nx_[1])
            fn = np.array([nx, ny, nz])
            for vi in f:
                acc[vi] += fn
        nrm = np.linalg.norm(acc, axis=1, keepdims=True)
        return np.where(nrm > 1e-12, acc / np.where(nrm > 1e-12, nrm, 1.0), np.array([0.0, 0.0, 1.0]))

    vn = m.vertex_normals(store=False)
    assert np.array_equal(vn, ref(m))                          # vectorized == loop, bit for bit
    # on a sphere, the outward normal at a vertex aligns with its radial direction
    radial = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    assert np.mean(np.sum(vn * radial, axis=1)) > 0.95         # strongly outward on average


def test_validate_topology_clean_bowtie_degenerate():
    """The full topology report: a clean sphere passes; a bowtie (non-manifold VERTEX that the edge test misses)
    and a degenerate face are caught."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
    def sphere(P): P = np.asarray(P, float); return np.linalg.norm(P, axis=1) - 0.6
    v, ax = sample_field(sphere, (np.array([-1., -1, -1]), np.array([1., 1, 1])), 32)
    M = marching_tetrahedra_vec(v, ax)
    r = M.validate_topology()
    assert r["ok"] and r["manifold_edges"] and r["manifold_vertices"] and r["watertight"]
    assert r["euler"] == 2 and r["genus"] == 0

    bow = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0.]]), [(0, 1, 2), (0, 3, 4)])
    rb = bow.validate_topology()
    assert rb["manifold_edges"] and not rb["manifold_vertices"] and rb["non_manifold_verts"] == [0] and not rb["ok"]

    deg = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]]), [(0, 1, 1)])
    assert deg.validate_topology()["degenerate_faces"] == 1


def test_marching_keys_are_stable_identity_across_local_edit():
    """A vertex's edge-key is a STABLE identity: across a local field edit, the array INDEX of an unchanged
    vertex can change, but its KEY does not -- and its position is unchanged where the field is."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
    def base(P): P = np.asarray(P, float); return np.linalg.norm(P, axis=1) - 0.6
    def edited(P):
        P = np.asarray(P, float)
        return base(P) - 0.15 * np.exp(-(((P - np.array([0., 0, 0.6])) ** 2).sum(1)) / (2 * 0.08 ** 2))
    b = (np.array([-1., -1, -1]), np.array([1., 1, 1]))
    v, ax = sample_field(base, b, 36); M1, k1 = marching_tetrahedra_vec(v, ax, return_keys=True)
    v2, _ = sample_field(edited, b, 36); M2, k2 = marching_tetrahedra_vec(v2, ax, return_keys=True)
    assert len(k1) == M1.n_vertices and len(set(k1.tolist())) == len(k1)   # one unique key per vertex
    p1 = {int(k): M1.vertices[i] for i, k in enumerate(k1.tolist())}
    p2 = {int(k): M2.vertices[i] for i, k in enumerate(k2.tolist())}
    far = [k for k in np.intersect1d(k1, k2).tolist() if p1[k][2] < 0.2]
    assert far and all(np.allclose(p1[k], p2[k], atol=1e-9) for k in far)   # key tracks the vertex, position stable


def test_stable_uv_is_edit_invariant():
    """Position-deterministic UVs don't move under a local edit (unlike the global unwrap, which re-solves)."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
    from holographic.mesh_and_geometry.holographic_meshuv import stable_uv
    def base(P): P = np.asarray(P, float); return np.linalg.norm(P, axis=1) - 0.6
    def edited(P):
        P = np.asarray(P, float)
        return base(P) - 0.15 * np.exp(-(((P - np.array([0., 0, 0.6])) ** 2).sum(1)) / (2 * 0.08 ** 2))
    b = (np.array([-1., -1, -1]), np.array([1., 1, 1]))
    v, ax = sample_field(base, b, 36); M1, k1 = marching_tetrahedra_vec(v, ax, return_keys=True)
    v2, _ = sample_field(edited, b, 36); M2, k2 = marching_tetrahedra_vec(v2, ax, return_keys=True)
    uv1 = stable_uv(M1, bounds=b); uv2 = stable_uv(M2, bounds=b)
    u1 = {int(k): uv1[i] for i, k in enumerate(k1.tolist())}
    u2 = {int(k): uv2[i] for i, k in enumerate(k2.tolist())}
    p1 = {int(k): M1.vertices[i] for i, k in enumerate(k1.tolist())}
    far = [k for k in np.intersect1d(k1, k2).tolist() if p1[k][2] < 0.2]
    assert far and all(np.allclose(u1[k], u2[k], atol=1e-9) for k in far)


# --- mesh performance fix: the vectorized triangle build must be BYTE-IDENTICAL to the loop, and the
#     CSR-backed queries must match the old full-scan outputs exactly ---
import numpy as _np_perf
from holographic.mesh_and_geometry.holographic_mesh import _half_edges_tri, _half_edges_loop, box as _box, grid as _grid, tetrahedron as _tet


def _triangulate(m):
    tris = []
    for f in m.faces:
        for k in range(1, len(f) - 1):
            tris.append([f[0], f[k], f[k + 1]])
    m.faces = tris; m._he = None; m._adj = None
    return m


def _ref_vertex_faces(m, v):
    return sorted({fi for fi, f in enumerate(m.faces) if v in f})


def _ref_vertex_neighbours(m, v):
    nb = set()
    for f in m.faces:
        if v in f:
            n = len(f); i = f.index(v); nb.add(f[(i + 1) % n]); nb.add(f[(i - 1) % n])
    nb.discard(v)
    return sorted(nb)


def test_tri_build_bit_identical_to_loop():
    for m in (_triangulate(_box(2, 2, 2)), _triangulate(_grid(6, 5)), _triangulate(_tet())):
        F = _np_perf.asarray(m.faces, dtype=_np_perf.int64)
        fast = _half_edges_tri(F); ref = _half_edges_loop(m.faces)
        for key in ("origin", "face", "nxt", "twin"):
            assert _np_perf.array_equal(fast[key], ref[key]), (key, m.n_vertices)


def test_boundary_twins_minus_one():
    m = _triangulate(_grid(5, 5))                                    # an open sheet -> has boundary edges
    he = m.half_edges()
    assert (he["twin"] == -1).any()                                 # boundary half-edges exist
    # every non-boundary twin is reciprocal
    for h, t in enumerate(he["twin"]):
        if t != -1:
            assert he["twin"][t] == h


def test_non_manifold_raises_in_both_paths():
    import pytest
    bad = [[0, 1, 2], [0, 1, 3]]                                    # directed edge (0,1) appears twice
    with pytest.raises(ValueError):
        _half_edges_tri(_np_perf.asarray(bad, dtype=_np_perf.int64))
    with pytest.raises(ValueError):
        _half_edges_loop(bad)


def test_csr_queries_match_full_scan():
    for m in (_triangulate(_box(2, 2, 2)), _triangulate(_grid(6, 5))):
        for v in range(m.n_vertices):
            assert m.vertex_faces(v) == _ref_vertex_faces(m, v)
            assert m.vertex_neighbours(v) == _ref_vertex_neighbours(m, v)


def test_cache_invalidation_adj_with_he():
    m = _triangulate(_box(2, 2, 2))
    _ = m.vertex_faces(0); assert m._adj is not None
    m._he = None; m._adj = None                                     # simulate an edit invalidation
    assert m.vertex_faces(0) == _ref_vertex_faces(m, 0)            # rebuilds correctly

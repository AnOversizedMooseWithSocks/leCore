"""Tests for the core modeler verbs (FWD-7): extrude / inset / dissolve-vertex. Each must produce a VALID mesh
(chi preserved, still closed + manifold) and hit its exact geometric signature, on both a triangle mesh
(icosphere) and a quad mesh (box)."""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import box
from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere
from holographic.mesh_and_geometry.holographic_meshverbs import extrude_face, inset_face, dissolve_vertex, _face_normal


def _signed_volume(m):
    """Signed volume via the divergence theorem (sum of tetrahedra from the origin). For a closed outward-oriented
    mesh this is the enclosed volume."""
    V = m.vertices
    vol = 0.0
    for f in m.faces:
        for k in range(1, len(f) - 1):
            a, b, c = V[f[0]], V[f[k]], V[f[k + 1]]
            vol += float(np.dot(a, np.cross(b, c))) / 6.0
    return vol


def _tri_area(verts, f):
    a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


# ---- EXTRUDE ---------------------------------------------------------------------------------------
def test_extrude_preserves_chi_closed_manifold():
    s = _icosphere(2)
    ex = extrude_face(s, 0, distance=0.3)
    assert ex.euler_characteristic() == s.euler_characteristic()
    assert ex.is_closed() and ex.is_manifold()


def test_extrude_cap_moves_exactly_distance_along_normal():
    s = _icosphere(2)
    face = s.faces[0]
    nrm = _face_normal(s.vertices, face)
    cap_before = np.mean([s.vertices[v] for v in face], axis=0)
    ex = extrude_face(s, 0, distance=0.3)
    cap_after = ex.vertices[ex.n_vertices - 3:].mean(axis=0)
    moved = cap_after - cap_before
    assert abs(float(np.dot(moved, nrm)) - 0.3) < 1e-9            # exactly `distance` along the normal
    assert float(np.linalg.norm(moved - np.dot(moved, nrm) * nrm)) < 1e-9   # and only along the normal


def test_extrude_outward_increases_volume():
    s = _icosphere(2)
    assert _signed_volume(extrude_face(s, 0, distance=0.3)) > _signed_volume(s)


# ---- INSET -----------------------------------------------------------------------------------------
def test_inset_preserves_chi_closed_manifold():
    s = _icosphere(2)
    ins = inset_face(s, 0, ratio=0.4)
    assert ins.euler_characteristic() == s.euler_characteristic()
    assert ins.is_closed() and ins.is_manifold()


def test_inset_central_area_is_one_minus_ratio_squared():
    s = _icosphere(2)
    area0 = _tri_area(s.vertices, s.faces[0])
    ins = inset_face(s, 0, ratio=0.4)
    central = ins.faces[s.n_faces - 1]                            # appended right after dropping face 0
    assert abs(_tri_area(ins.vertices, central) - (1 - 0.4) ** 2 * area0) < 1e-9


def test_inset_central_face_is_coplanar_with_original():
    s = _icosphere(2)
    n0 = _face_normal(s.vertices, s.faces[0])
    ins = inset_face(s, 0, ratio=0.4)
    n1 = _face_normal(ins.vertices, ins.faces[s.n_faces - 1])
    assert float(np.dot(n0, n1)) > 1.0 - 1e-9                     # same orientation -> coplanar (in-plane inset)


# ---- DISSOLVE --------------------------------------------------------------------------------------
def test_dissolve_preserves_chi_closed_manifold():
    s = _icosphere(2)
    diss = dissolve_vertex(s, vertex=5)
    assert diss.euler_characteristic() == s.euler_characteristic()
    assert diss.is_closed() and diss.is_manifold()


def test_dissolve_removes_exactly_one_vertex():
    s = _icosphere(2)
    assert dissolve_vertex(s, vertex=5).n_vertices == s.n_vertices - 1


# ---- robustness on a QUAD mesh (box: degree-3 vertices, quad faces) --------------------------------
def test_verbs_work_on_a_quad_box():
    b = box()
    for m in (extrude_face(b, 0, 0.5), inset_face(b, 0, 0.3), dissolve_vertex(b, 0)):
        assert m.euler_characteristic() == b.euler_characteristic()
        assert m.is_closed() and m.is_manifold()
    assert dissolve_vertex(b, 0).n_vertices == b.n_vertices - 1


# ---- determinism -----------------------------------------------------------------------------------
def test_extrude_is_deterministic():
    s = _icosphere(2)
    assert np.array_equal(extrude_face(s, 0, 0.3).vertices, extrude_face(s, 0, 0.3).vertices)

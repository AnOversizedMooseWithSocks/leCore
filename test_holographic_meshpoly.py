import numpy as np
from holographic_mesh import Mesh, box
from holographic_meshpoly import triangles_to_quads, merge_coplanar, face_type_counts


def test_box_triangles_repair_to_quads():
    """A box's flat faces, triangulated, re-pair exactly back into 6 quads."""
    tri = Mesh(box().vertices, [tuple(t) for t in box().triangulate()])
    assert face_type_counts(tri)[3] == 12
    q = triangles_to_quads(tri)
    c = face_type_counts(q)
    assert c[4] == 6 and c[3] == 0


def test_quad_conversion_stays_watertight_manifold():
    """Quad-dominant conversion of a marched sphere preserves watertight, manifold-edge topology."""
    from holographic_meshbridge import sample_field, marching_tetrahedra_vec
    def sphere(P): P = np.asarray(P, float); return np.linalg.norm(P, axis=1) - 0.6
    v, ax = sample_field(sphere, (np.array([-1., -1, -1]), np.array([1., 1, 1])), 28)
    M = marching_tetrahedra_vec(v, ax)
    Q = triangles_to_quads(M, planarity=0.85)
    r = Q.validate_topology()
    assert r["watertight"] and r["manifold_edges"]
    assert face_type_counts(Q)[4] > face_type_counts(Q)[3]    # quad-dominant


def test_coplanar_merge_collapses_flat_regions_to_ngons():
    """Flat box faces collapse to single polygons; the result stays watertight."""
    from holographic_meshbridge import sample_field, marching_tetrahedra_vec
    def boxsdf(P):
        P = np.asarray(P, float); q = np.abs(P) - 0.5
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2])), 0)
    v, ax = sample_field(boxsdf, (np.array([-1., -1, -1]), np.array([1., 1, 1])), 20)
    M = marching_tetrahedra_vec(v, ax)
    ng = merge_coplanar(M, normal_tol=0.999)
    assert sum(face_type_counts(ng).values()) < M.n_faces     # far fewer faces after collapse
    assert ng.validate_topology()["watertight"]

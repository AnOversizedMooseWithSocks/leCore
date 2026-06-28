"""Tests for the mesh<->SDF<->splat bridge (FWD-11): marching-tetrahedra isosurface extraction (SDF->mesh),
mesh->SDF sampling, and the splat->mesh path. Measured against analytic references: a closed-manifold sphere with
vertices on the sphere, outward orientation, signed-distance correctness, the splat blob, resolution scaling,
determinism."""

import numpy as np

from holographic_meshbridge import (sample_field, marching_tetrahedra, mesh_to_sdf,
                                     sphere_sdf, metaball_field)


def _extract_sphere(radius=1.0, res=24, half=1.5):
    vals, axes = sample_field(sphere_sdf(radius=radius), ((-half,) * 3, (half,) * 3), res=res)
    return marching_tetrahedra(vals, axes, level=0.0)


# ---- SDF -> mesh ----------------------------------------------------------------------------------
def test_sdf_to_mesh_is_a_closed_manifold_sphere():
    m = _extract_sphere()
    assert m.n_faces > 0 and m.is_manifold()
    assert m.is_closed() and m.euler_characteristic() == 2      # genus-0 closed surface (watertight)


def test_extracted_vertices_lie_on_the_sphere():
    radii = np.linalg.norm(_extract_sphere(radius=1.0).vertices, axis=1)
    assert abs(float(radii.mean()) - 1.0) < 0.02 and float(radii.std()) < 0.03


def test_extracted_sphere_radius_scales():
    radii = np.linalg.norm(_extract_sphere(radius=0.7, half=1.2).vertices, axis=1)
    assert abs(float(radii.mean()) - 0.7) < 0.02               # a different radius extracts correctly


def test_marching_tets_orientation_is_outward():
    m = _extract_sphere(res=20)
    V = m.vertices
    outward = sum(1 for (a, b, c) in m.faces
                  if np.dot(np.cross(V[b] - V[a], V[c] - V[a]), (V[a] + V[b] + V[c]) / 3.0) > 0)
    assert outward == m.n_faces                                # every face normal points outward


def test_resolution_scaling_adds_faces():
    f12 = _extract_sphere(res=12).n_faces
    f20 = _extract_sphere(res=20).n_faces
    f28 = _extract_sphere(res=28).n_faces
    assert f12 < f20 < f28                                     # finer grid -> more, finer triangles


# ---- mesh -> SDF ----------------------------------------------------------------------------------
def test_mesh_to_sdf_matches_analytic():
    m = _extract_sphere()
    probes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -1.5]])
    got = mesh_to_sdf(m, probes)
    analytic = np.linalg.norm(probes, axis=1) - 1.0
    assert np.allclose(got, analytic, atol=0.05)


def test_mesh_to_sdf_sign_is_inside_negative_outside_positive():
    m = _extract_sphere()
    got = mesh_to_sdf(m, np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    assert got[0] < 0 and got[1] > 0


# ---- splat -> mesh --------------------------------------------------------------------------------
def test_splat_field_meshes_to_a_closed_blob():
    vals, axes = sample_field(metaball_field(np.array([[-0.4, 0, 0], [0.4, 0, 0]]), radius=0.4),
                              ((-1.5,) * 3, (1.5,) * 3), res=24)
    blob = marching_tetrahedra(vals, axes, level=0.5)
    assert blob.n_faces > 0 and blob.is_manifold() and blob.is_closed()


# ---- determinism ----------------------------------------------------------------------------------
def test_marching_tets_is_deterministic():
    vals, axes = sample_field(sphere_sdf(), ((-1.5,) * 3, (1.5,) * 3), res=20)
    assert np.array_equal(marching_tetrahedra(vals, axes, 0.0).vertices,
                          marching_tetrahedra(vals, axes, 0.0).vertices)

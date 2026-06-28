"""Tests for mesh curvature & feature detection (FWD-6): the EXACT discrete Gauss-Bonnet check (total angle
defect = 2*pi*chi), unit-sphere mean/Gaussian curvature near 1, cube crease detection (12 sharp 90-degree
edges, none on a smooth sphere), the per-vertex noise negative + confidence, and determinism."""

import numpy as np

from holographic_mesh import Mesh, box
from holographic_meshsmooth import _icosphere
from holographic_meshcurvature import (vertex_areas, angle_defects, gaussian_curvature, gauss_bonnet_defect,
                                       mean_curvature, dihedral_angles, detect_creases, curvature_confidence)


# ---- the EXACT references -------------------------------------------------------------------------
def test_gauss_bonnet_is_exact_on_a_closed_mesh():
    # discrete Gauss-Bonnet: total angle defect = 2*pi*chi to floating point -- the curvature estimate is
    # validated against the Euler characteristic the mesh kernel computes
    sphere = _icosphere(3)
    assert abs(gauss_bonnet_defect(sphere)) < 1e-6
    assert abs(float(angle_defects(sphere).sum()) - 4.0 * np.pi) < 1e-6   # chi=2 -> 4*pi


def test_gauss_bonnet_holds_for_a_cube_too():
    # a (triangulated) cube is also chi=2 -> total defect 4*pi, independent of the geometry
    cube = box(2.0, 2.0, 2.0)
    tcube = Mesh(cube.vertices.copy(), [tuple(t) for t in cube.triangulate()])
    assert abs(gauss_bonnet_defect(tcube)) < 1e-6


def test_unit_sphere_gaussian_curvature_near_one():
    K = gaussian_curvature(_icosphere(3))
    assert 0.8 < float(K.mean()) < 1.25                 # K = 1/R^2 = 1 on the unit sphere (discrete band)


def test_unit_sphere_mean_curvature_near_one():
    H = mean_curvature(_icosphere(3))
    assert 0.8 < float(H.mean()) < 1.25                 # H = 1/R = 1 on the unit sphere


def test_vertex_areas_sum_to_total_surface_area():
    # barycentric areas split each triangle 1/3 per vertex, so their sum is the total mesh area
    m = _icosphere(2)
    V = m.vertices
    total_tri = sum(0.5 * float(np.linalg.norm(np.cross(V[j] - V[i], V[k] - V[i])))
                    for (i, j, k) in m.triangulate())
    assert abs(float(vertex_areas(m).sum()) - total_tri) < 1e-9


# ---- creases ---------------------------------------------------------------------------------------
def test_cube_has_twelve_creases():
    assert len(detect_creases(box(2.0, 2.0, 2.0), threshold_deg=30.0)) == 12


def test_cube_dihedral_angles_are_ninety_degrees():
    ang = dihedral_angles(box(2.0, 2.0, 2.0))
    assert len(ang) == 12
    assert all(abs(a - np.pi / 2) < 1e-6 for a in ang.values())


def test_triangulated_cube_still_twelve_creases():
    # triangulation adds 6 FLAT diagonals (0-degree dihedral) which are not creases
    cube = box(2.0, 2.0, 2.0)
    tcube = Mesh(cube.vertices.copy(), [tuple(t) for t in cube.triangulate()])
    assert len(detect_creases(tcube, threshold_deg=30.0)) == 12


def test_smooth_sphere_has_no_creases():
    assert len(detect_creases(_icosphere(3), threshold_deg=30.0)) == 0


# ---- the kept negative + confidence ---------------------------------------------------------------
def test_per_vertex_curvature_is_noisy():
    # the MEAN is accurate but per-vertex values vary on a coarse mesh -- the documented negative
    H = mean_curvature(_icosphere(3))
    assert (H.std() / H.mean()) > 0.02


def test_curvature_confidence_in_range():
    conf = curvature_confidence(_icosphere(3))
    sphere = _icosphere(3)
    assert conf.shape == (sphere.n_vertices,)
    assert np.all((conf >= 0.0) & (conf <= 1.0))


# ---- determinism -----------------------------------------------------------------------------------
def test_curvature_and_creases_are_deterministic():
    sphere = _icosphere(2)
    assert np.array_equal(mean_curvature(sphere), mean_curvature(sphere))
    assert np.array_equal(gaussian_curvature(sphere), gaussian_curvature(sphere))
    assert detect_creases(box(2, 2, 2)) == detect_creases(box(2, 2, 2))

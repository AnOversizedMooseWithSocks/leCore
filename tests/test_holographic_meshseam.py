"""Tests for seam cutting (ARCH-4): opening a closed surface into a disk by vertex duplication -- a real FWD-3
seam. Covers the disk topology (chi=1, one boundary, manifold), interior-vertex duplication, the robust
geometry-preservation payback (cut keeps faces, puncture deletes them), the distortion payback with a good seam,
the kept negative (a full meridian is worse), seam-path validity, and determinism."""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import box
from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere
from holographic.mesh_and_geometry.holographic_meshuv import uv_unwrap, uv_distortion, puncture
from holographic.mesh_and_geometry.holographic_meshseam import cut_seam, shortest_seam, _boundary_loop_count


def _sphere_and_meridian():
    s = _icosphere(3)
    north = int(np.argmax(s.vertices[:, 2]))
    south = int(np.argmin(s.vertices[:, 2]))
    return s, shortest_seam(s, north, south), north, south


# ---- the disk topology ----------------------------------------------------------------------------
def test_cut_seam_opens_a_closed_sphere_into_a_disk():
    s, meridian, _, _ = _sphere_and_meridian()
    disk = cut_seam(s, meridian)
    assert disk.is_manifold() and not disk.is_closed()
    assert disk.euler_characteristic() == 1                # a disk


def test_cut_disk_has_exactly_one_boundary_loop():
    s, meridian, _, _ = _sphere_and_meridian()
    assert _boundary_loop_count(cut_seam(s, meridian)) == 1


def test_cut_seam_duplicates_interior_seam_vertices():
    s, meridian, _, _ = _sphere_and_meridian()
    disk = cut_seam(s, meridian)
    assert disk.n_vertices == s.n_vertices + (len(meridian) - 2)   # endpoints stay single


# ---- the robust geometry-preservation payback -----------------------------------------------------
def test_cut_seam_preserves_every_face():
    s, meridian, _, _ = _sphere_and_meridian()
    assert cut_seam(s, meridian).n_faces == s.n_faces      # non-destructive


def test_puncture_deletes_faces_but_cut_does_not():
    s, meridian, _, _ = _sphere_and_meridian()
    assert puncture(s, 0).n_faces < s.n_faces              # the contrast: puncture loses geometry
    assert cut_seam(s, meridian).n_faces == s.n_faces


# ---- the distortion payback (with a good seam) and the kept negative ------------------------------
def test_well_chosen_seam_beats_puncture_on_distortion():
    s, _, north, _ = _sphere_and_meridian()
    equator = int(np.argmin(np.abs(s.vertices[:, 2])))
    good = cut_seam(s, shortest_seam(s, north, equator))
    good_dist = uv_distortion(good, uv_unwrap(good))
    punct_dist = uv_distortion(puncture(s, 0), uv_unwrap(puncture(s, 0)))
    assert good_dist < punct_dist


def test_full_meridian_is_a_worse_cut_than_the_puncture():
    # the kept negative: seam choice matters -- a full pole-to-pole meridian makes a long thin lune
    s, meridian, _, _ = _sphere_and_meridian()
    full = cut_seam(s, meridian)
    assert uv_distortion(full, uv_unwrap(full)) > uv_distortion(puncture(s, 0), uv_unwrap(puncture(s, 0)))


# ---- seam path validity ---------------------------------------------------------------------------
def test_shortest_seam_is_a_valid_edge_path():
    s, _, north, south = _sphere_and_meridian()
    seam = shortest_seam(s, north, south)
    assert seam[0] == north and seam[-1] == south
    edges = {frozenset(e) for e in s.edges()}              # normalise (edges() yields sorted tuples)
    for u, v in zip(seam, seam[1:]):
        assert frozenset((u, v)) in edges                  # consecutive seam vertices share an edge


# ---- determinism ----------------------------------------------------------------------------------
def test_cut_seam_is_deterministic():
    s, meridian, _, _ = _sphere_and_meridian()
    assert np.array_equal(cut_seam(s, meridian).vertices, cut_seam(s, meridian).vertices)

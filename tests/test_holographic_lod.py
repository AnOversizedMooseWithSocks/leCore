"""Tests for the screen-space-error LOD policy (holographic_lod): the geometric instance of the engine's
error-budget resolution selection, on top of QEM decimation + surface_deviation. Pick the coarsest mesh whose error,
projected to the screen, stays under a pixel budget."""

import math

from holographic.misc.holographic_lod import build_lod_chain, screen_space_error, select_lod
from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere

# build the chain once (decimation is the expensive part) and reuse across tests
_FULL = _icosphere(2)                                      # V66 F128 unit sphere
_CHAIN = build_lod_chain(_FULL, targets=(0.5, 0.25, 0.125))


def test_chain_has_several_levels():
    assert len(_CHAIN) >= 3


def test_first_level_is_the_original_with_zero_error():
    assert _CHAIN[0].n_faces == _FULL.n_faces
    assert _CHAIN[0].max_error == 0.0 and _CHAIN[0].mean_error == 0.0


def test_face_count_strictly_decreases():
    faces = [l.n_faces for l in _CHAIN]
    assert all(faces[i] > faces[i + 1] for i in range(len(faces) - 1))


def test_deviation_only_grows():
    err = [l.max_error for l in _CHAIN]
    assert all(err[i] <= err[i + 1] for i in range(len(err) - 1))


def test_screen_space_error_falls_with_distance():
    e = _CHAIN[-1].max_error
    assert screen_space_error(e, 5.0) > screen_space_error(e, 50.0) > screen_space_error(e, 500.0)


def test_screen_space_error_scales_with_resolution():
    e = _CHAIN[-1].max_error
    assert screen_space_error(e, 10.0, screen_height_px=4320) > screen_space_error(e, 10.0, screen_height_px=540)


def test_lod_coarsens_with_distance():
    picks = [select_lod(_CHAIN, d, 2.0) for d in (2.0, 5.0, 15.0, 50.0, 200.0)]
    assert all(picks[i] <= picks[i + 1] for i in range(len(picks) - 1))


def test_full_detail_up_close_and_coarser_far():
    assert select_lod(_CHAIN, 2.0, 2.0) == 0
    assert select_lod(_CHAIN, 200.0, 2.0) > 0


def test_selection_is_tight():
    d, thr = 50.0, 2.0
    p = select_lod(_CHAIN, d, thr)
    if p + 1 < len(_CHAIN):
        assert screen_space_error(_CHAIN[p + 1].max_error, d) >= thr


def test_tighter_threshold_is_never_coarser():
    assert select_lod(_CHAIN, 50.0, 0.5) <= select_lod(_CHAIN, 50.0, 10.0)


def test_higher_resolution_is_never_coarser():
    assert select_lod(_CHAIN, 50.0, 2.0, screen_height_px=4320) <= select_lod(_CHAIN, 50.0, 2.0, screen_height_px=540)


def test_deterministic():
    chain2 = build_lod_chain(_FULL, targets=(0.5, 0.25, 0.125))
    assert [l.n_faces for l in chain2] == [l.n_faces for l in _CHAIN]
    assert [l.max_error for l in chain2] == [l.max_error for l in _CHAIN]
    assert select_lod(_CHAIN, 37.0, 1.5) == select_lod(chain2, 37.0, 1.5)


def test_build_cluster_lod_chain_parallel_imported_mesh():
    """build_cluster_lod_chain is the PARALLEL imported-mesh LOD: cluster-decimate at decreasing grids, deviation from
    the original. Levels must coarsen monotonically in FACE COUNT, and selection must pick coarser far away. Small
    mesh so the brute-deviation error metric stays quick (the decimation itself is O(n)). KEPT NEGATIVE asserted: the
    per-level ERROR is NOT monotonic in grid resolution -- cell alignment with the surface matters, so a coarser grid
    can land representatives closer; we record that, and do not require monotone error."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
    from holographic.misc.holographic_lod import build_cluster_lod_chain, select_lod

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    v, a = sample_field(sphere, ((-1, -1, -1), (1, 1, 1)), 18)
    m = marching_tetrahedra_vec(v, a, 0.0)
    chain = build_cluster_lod_chain(m, grids=(16, 10, 6))

    assert len(chain) >= 3
    faces = [lvl.n_faces for lvl in chain]
    assert all(faces[i] > faces[i + 1] for i in range(len(faces) - 1))         # strictly coarser in face count
    assert chain[0].max_error == 0.0                                           # level 0 is the original (zero error)
    assert all(lvl.max_error >= 0.0 for lvl in chain)                          # errors are distances (non-negative)

    near = select_lod(chain, 0.3, 2.0)
    far = select_lod(chain, 500.0, 2.0)
    assert chain[far].n_faces <= chain[near].n_faces                           # coarser (or equal) far away

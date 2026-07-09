"""Tests for UV unwrapping (FWD-3): the shipped classical-MDS chart fed the mesh's geodesic matrix (Isomap on
explicit edges). Covers near-isometric unwrap of a developable patch, the curved-surface distortion (Gauss),
Isomap-beats-linear on curved AND the honest reverse on flat, flip-free packing (no overlap), the closed-needs-a
-seam negative, unit-square packing, and determinism."""

import numpy as np

from holographic.mesh_and_geometry.holographic_meshuv import uv_unwrap, uv_distortion, flat_grid_mesh, hemisphere_cap, puncture
from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere


def _orientation_consistency(mesh, uv):
    """Fraction of faces sharing the majority winding sign in UV. 1.0 == no flipped triangles == no local
    overlap (a locally injective chart)."""
    s = []
    for (a, b, c) in mesh.faces:
        e1 = uv[b] - uv[a]
        e2 = uv[c] - uv[a]
        s.append(np.sign(e1[0] * e2[1] - e1[1] * e2[0]))
    s = np.asarray(s)
    return max(int((s > 0).sum()), int((s < 0).sum())) / len(s)


# ---- developable patch: near-isometric, non-degenerate, flip-free ---------------------------------
def test_flat_patch_unwraps_near_isometric():
    flat = flat_grid_mesh(9)
    assert uv_distortion(flat, uv_unwrap(flat)) < 0.07     # isotropic developable -> nearly isometric


def test_uv_is_non_degenerate():
    flat = flat_grid_mesh(9)
    uv = uv_unwrap(flat)
    distinct = {(round(float(x), 6), round(float(y), 6)) for x, y in uv}
    assert len(distinct) == flat.n_vertices                # vertices map to distinct UV points


def test_uv_has_no_flipped_triangles():
    # the bar's "charts don't overlap after packing": a flip-free unwrap is locally injective
    for m in (flat_grid_mesh(9), hemisphere_cap(3)):
        assert _orientation_consistency(m, uv_unwrap(m)) == 1.0


# ---- curvature: Gauss distortion, and Isomap vs the linear baseline -------------------------------
def test_curved_cap_distorts_more_than_flat():
    flat_d = uv_distortion(flat_grid_mesh(9), uv_unwrap(flat_grid_mesh(9)))
    cap_d = uv_distortion(hemisphere_cap(3), uv_unwrap(hemisphere_cap(3)))
    assert cap_d > flat_d + 0.05                            # a curved surface cannot flatten without stretch


def test_isomap_beats_planar_projection_on_a_curved_surface():
    cap = hemisphere_cap(3)
    iso = uv_distortion(cap, uv_unwrap(cap, method="isomap"))
    planar = uv_distortion(cap, uv_unwrap(cap, method="planar"))
    assert iso < planar                                    # geodesic chart wins where the surface bends


def test_planar_projection_beats_isomap_on_a_flat_surface():
    # the honest reverse: on a developable patch a linear projection is EXACTLY isometric; Isomap carries the
    # edge-graph geodesic's small approximation error -- so linear is the right tool there (chart.py's philosophy)
    flat = flat_grid_mesh(9)
    iso = uv_distortion(flat, uv_unwrap(flat, method="isomap"))
    planar = uv_distortion(flat, uv_unwrap(flat, method="planar"))
    assert planar < iso
    assert planar < 1e-6                                   # linear projection of a planar mesh is isometric


# ---- closed surfaces need a seam ------------------------------------------------------------------
def test_punctured_sphere_is_a_disk_and_distorts_most():
    sphere = _icosphere(3)
    assert sphere.is_closed()
    disk = puncture(sphere, vertex=0)
    assert disk.euler_characteristic() == 1 and not disk.is_closed()   # chi 2 -> 1: a disk
    dist_punct = uv_distortion(disk, uv_unwrap(disk))
    dist_cap = uv_distortion(hemisphere_cap(3), uv_unwrap(hemisphere_cap(3)))
    assert dist_punct > dist_cap                           # flattening most of a closed sphere distorts more


# ---- packing & determinism ------------------------------------------------------------------------
def test_uv_is_packed_to_unit_square():
    uv = uv_unwrap(hemisphere_cap(3))
    assert uv.min() >= -1e-9 and uv.max() <= 1.0 + 1e-9    # ~[0,1]^2 with a single uniform scale


def test_uv_unwrap_is_deterministic():
    flat = flat_grid_mesh(9)
    assert np.array_equal(uv_unwrap(flat), uv_unwrap(flat))

"""point_set_to_mesh_grid is STREAMED: peak memory is set by a byte budget, not by the query count.

WHY THIS FILE EXISTS. Every mesh_distance_grid bake above ~112^3 on a 7 GB box was OOM-killed, and the
cause was not the output (a 176^3 float64 grid is 43 MB) but the kernel underneath: it materialised a
(N, (2r+1)^3, 3) neighbour block plus a candidate edge list, MEASURED at 36.8 KB of peak RSS per query
and growing linearly, so a ~1M-voxel surface shell demanded ~37 GB to produce a 43 MB answer.

The fix processes queries in blocks. That is only defensible because every stage of the kernel is
per-query independent, so the result must be BIT-IDENTICAL at any chunk size -- these tests assert
exactly that, because the failure mode of a chunked reduction is a silent, chunk-size-dependent answer
rather than an exception. Determinism is a hard constraint of this engine; a bake that changes with a
memory budget would flip decisions everywhere downstream.
"""
import numpy as np
import pytest

from holographic.mesh_and_geometry.holographic_meshbridge import (
    point_set_to_mesh_grid, _grid_query_chunk, marching_tetrahedra_vec, mesh_distance_grid)


def _sphere(n=30):
    """A marched unit sphere -- finely tessellated, the shape the bake kernel actually sees."""
    ax = np.linspace(-1.5, 1.5, n)
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)
    return marching_tetrahedra_vec(np.linalg.norm(g, axis=-1) - 1.0, (ax, ax, ax))


def _probes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n, 3))
    return P / np.linalg.norm(P, axis=1)[:, None] * rng.uniform(0.8, 1.2, n)[:, None]


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("chunk", [1, 7, 103, 399, 10 ** 9])
def test_chunking_is_bit_identical(signed, chunk):
    """THE CONTRACT. chunk is a memory knob and must not be able to change the answer -- not to 1e-12,
    not at all. The chunk sizes are deliberately NOT divisors of the query count, so a ragged final
    block is exercised; chunk=1 and chunk=1e9 bracket the extremes (one query per block vs the whole
    set in one block, i.e. the original unchunked path)."""
    msh = _sphere()
    P = _probes()
    ref = point_set_to_mesh_grid(P, msh.vertices, msh.faces, radius=2, signed=signed, chunk=10 ** 9)
    got = point_set_to_mesh_grid(P, msh.vertices, msh.faces, radius=2, signed=signed, chunk=chunk)
    assert np.array_equal(np.isinf(ref), np.isinf(got)), \
        "chunking changed WHICH queries found no triangle -- the +inf pattern is part of the contract"
    fin = np.isfinite(ref)
    assert fin.any(), "the fixture must actually reach the surface, or this test asserts nothing"
    assert np.array_equal(ref[fin], got[fin]), \
        "chunk=%r is not bit-identical (max delta %.3g)" % (chunk, float(np.abs(ref[fin] - got[fin]).max()))


def test_signs_survive_chunking():
    """The signed path has the one piece of per-query state that a careless chunking would break: a
    lexsort picking the nearest triangle per query, whose winning index is LOCAL to the block. If the
    block offset were dropped, signs would land on the wrong queries -- which is invisible in the
    unsigned distance and catastrophic in a bake (inside/outside inverted in stripes)."""
    msh = _sphere()
    P = _probes(300, seed=3)
    ref = point_set_to_mesh_grid(P, msh.vertices, msh.faces, radius=2, signed=True, chunk=10 ** 9)
    got = point_set_to_mesh_grid(P, msh.vertices, msh.faces, radius=2, signed=True, chunk=13)
    fin = np.isfinite(ref)
    assert np.array_equal(np.sign(ref[fin]), np.sign(got[fin]))
    # and the signs must be RIGHT, not merely stable: inside the unit sphere is negative
    inside = np.linalg.norm(P, axis=1) < 0.95
    assert (got[inside & fin] < 0).mean() > 0.98


def test_chunk_model_scales_with_radius_and_budget():
    """The block size is a MODEL, not a constant, because the per-query cost is set by the radius:
    (2r+1)^3 neighbour rows. A fixed chunk tuned at radius 2 would be 3.4x over budget at radius 3."""
    assert _grid_query_chunk(7 ** 3, 0.13) < _grid_query_chunk(5 ** 3, 0.13)
    assert _grid_query_chunk(125, 0.13, max_bytes=8 * 1024 ** 3) > _grid_query_chunk(125, 0.13, max_bytes=64 * 1024 ** 2)
    assert _grid_query_chunk(125, 0.13, max_bytes=1) >= 256, "floored -- a tiny budget cannot be met by chunking anyway"
    assert _grid_query_chunk(125, 0.0) > _grid_query_chunk(125, 5.0), "denser cells mean more candidates, smaller blocks"


def test_bake_is_unchanged_by_the_budget():
    """END TO END, and the one that would actually have caught a mistake: the same bake under a
    generous and a punishing memory budget must produce the SAME GRID. This is the property that makes
    the fix safe to leave on by default."""
    msh = _sphere(26)
    lo, hi = np.array([-1.4] * 3), np.array([1.4] * 3)
    a, _ = mesh_distance_grid(msh, (lo, hi), res=32, method="shell")
    import holographic.mesh_and_geometry.holographic_meshbridge as mb
    real = mb.point_set_to_mesh_grid
    try:                                     # force a pathologically small block through the same call
        mb.point_set_to_mesh_grid = lambda *a_, **k: real(*a_, **dict(k, chunk=17))
        b, _ = mesh_distance_grid(msh, (lo, hi), res=32, method="shell")
    finally:
        mb.point_set_to_mesh_grid = real
    assert np.array_equal(np.asarray(a), np.asarray(b)), "a memory budget must never move a baked SDF"


def test_voxel_remesh_accepts_quads():
    """REGRESSION, and it was live: voxel_remesh's job is messy-input cleanup, but it forwarded faces
    straight to a triangle-only sampler, so voxel_remesh(box()) -- a QUAD primitive from this very
    package -- raised "needs TRIANGLES, got 4-gon faces". A cleanup verb refusing input for needing
    cleanup. It now ear-clips first. This also broke the module's own selftest, which is how it hid."""
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.mesh_and_geometry.holographic_meshbridge import voxel_remesh
    # resolution 36 matches the module selftest: below ~32 the box's own SDF band does not clear the
    # grid edge and the marched surface legitimately opens -- a documented property of `pad`, not of quads.
    out = voxel_remesh(box(), resolution=36)
    assert out.n_faces > 0
    edges = {}
    for f in out.faces:
        for i in range(len(f)):
            e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
            edges[e] = edges.get(e, 0) + 1
    assert sum(1 for c in edges.values() if c == 1) == 0, "the remesh must be watertight"


# ---------------------------------------------------------------------------------------------------
# sign="winding_flood": the cheap sign path, and the scope limit that makes it shippable.
#
# The full-volume winding sign prices the generalised winding number at every voxel -- MEASURED 38.8
# minutes for a 192^3 bake of a 151,582-face .glb. Most of that cost is in the wrong place: the winding
# number far from a surface was never in doubt. winding_flood prices it only on the BAND (a watertight
# blocking shell is all flood_fill_sign needs) and floods the rest.
#
# But the two are NOT the same question. "winding" asks *is this point enclosed by the solid angle*;
# winding_flood asks *can this point escape to the grid boundary*. They coincide for the class winding
# was built for -- soups whose boundary edges are seams between coincident shells, no opening a voxel
# wide -- and diverge as soon as there is a real hole, because the flood needs only one passage. A hole
# 0.8 band widths across already flips 5.05% of voxels, and it does NOT get better with a smaller hole.
# So the path self-checks against a random winding subsample and REFUSES rather than shipping a hollow
# field. These tests pin both halves; without the refusal test this would be a footgun.
# ---------------------------------------------------------------------------------------------------

def _punctured(cap):
    """A marched unit sphere with the polar cap above z = 1-cap removed: an opening of known size."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    ax = np.linspace(-1.5, 1.5, 26)
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)
    sph = marching_tetrahedra_vec(np.linalg.norm(g, axis=-1) - 1.0, (ax, ax, ax))
    V = np.asarray(sph.vertices, float)
    F = [tuple(f) for f in sph.faces]
    if cap <= 0:
        return sph
    zc = np.array([V[list(f)].mean(0)[2] for f in F])
    return Mesh(V, [f for f, z in zip(F, zc) if z < 1.0 - cap])


def test_winding_flood_matches_winding_when_there_is_no_passage():
    """The claim the fast path rests on. Not 'close', not 'near the surface' -- the same sign field."""
    from holographic.mesh_and_geometry.holographic_meshbridge import mesh_to_sdf_grid
    bs = ((-1.6,) * 3, (1.6,) * 3)
    sph = _punctured(0.0)
    a, _ = mesh_to_sdf_grid(sph, bs, res=30, sign="winding")
    b, _ = mesh_to_sdf_grid(sph, bs, res=30, sign="winding_flood")
    same = (np.asarray(a) < 0) == (np.asarray(b) < 0)
    assert same.all(), "%d of %d voxels differ" % (int((~same).sum()), same.size)
    assert (np.asarray(b) < 0).mean() > 0.05, "an all-positive field would pass the line above vacuously"


def test_winding_flood_refuses_when_the_flood_can_walk_in():
    """THE LOAD-BEARING TEST. The failure mode is silent: a hollow object that renders as glass with
    nothing inside it. It must raise, and the message must name the escape route."""
    from holographic.mesh_and_geometry.holographic_meshbridge import mesh_to_sdf_grid
    with pytest.raises(ValueError, match="winding_flood"):
        mesh_to_sdf_grid(_punctured(0.05), ((-1.6,) * 3, (1.6,) * 3), res=30, sign="winding_flood")


def test_winding_flood_underfills_which_is_why_it_refuses():
    """Pin the DIRECTION of the error, so a future change that alters it re-opens the scope question."""
    from holographic.mesh_and_geometry.holographic_meshbridge import mesh_to_sdf_grid
    bs = ((-1.6,) * 3, (1.6,) * 3)
    holed = _punctured(0.05)
    wf, _ = mesh_to_sdf_grid(holed, bs, res=30, sign="winding_flood", verify=0)
    w, _ = mesh_to_sdf_grid(holed, bs, res=30, sign="winding")
    assert (np.asarray(wf) < 0).mean() < (np.asarray(w) < 0).mean()


def test_unknown_sign_names_the_valid_options():
    """An error that does not list the alternatives costs a round trip to the source."""
    from holographic.mesh_and_geometry.holographic_meshbridge import mesh_to_sdf_grid
    with pytest.raises(ValueError, match="winding_flood"):
        mesh_to_sdf_grid(_punctured(0.0), ((-1.6,) * 3, (1.6,) * 3), res=16, sign="nonsense")

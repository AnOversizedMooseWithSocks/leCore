"""Tests for FS-2 the narrow-band sparse field (holographic_sparsefield): store/edit/re-extract only the thin shell of
voxels around the surface, so a brush touches O(brush) voxels not O(res^3). A sphere SDF is the ground truth."""

import numpy as np

from holographic.misc.holographic_sparsefield import SparseField, _smooth_falloff


R = 0.6
BOUNDS = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
VOXEL = 2.0 / 42                      # ~42^3 grid, modest for a pure-Python test
BAND = 4 * VOXEL


def _sphere(P):
    return np.linalg.norm(P, axis=1) - R     # |grad| = 1 everywhere


# one read-only field shared by the non-mutating tests
_SHARED = SparseField.from_field(_sphere, BOUNDS[0], BOUNDS[1], VOXEL, BAND, tile=6)


def test_band_is_sparse():
    full = int(np.prod(_SHARED.ncorner))
    assert 0 < len(_SHARED.values) < 0.5 * full          # the surface shell is a fraction of the volume


def test_sample_matches_sdf_in_band():
    probe = np.array([[R * 0.5, 0.0, 0.0], [0.0, R, 0.0], [0.0, 0.0, -R * 0.9]])
    got = _SHARED.sample(probe)
    true = _sphere(probe)
    near = np.abs(true) < BAND
    assert np.allclose(got[near], true[near], atol=VOXEL)


def test_extract_is_watertight_and_on_the_surface():
    mesh = _SHARED.extract_local()
    assert mesh.n_faces > 0
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert np.max(np.abs(radii - R)) < VOXEL            # vertices lie on the true sphere within a voxel
    assert mesh.is_manifold()                            # welded watertight across brick seams


def test_stroke_touches_O_brush_not_whole_grid():
    sf = SparseField.from_field(_sphere, BOUNDS[0], BOUNDS[1], VOXEL, BAND, tile=6)
    full = int(np.prod(sf.ncorner))
    p = np.array([R, 0.0, 0.0])
    brush_r = 0.25

    def inflate(points):
        d = np.linalg.norm(points - p, axis=1)
        return -0.5 * BAND * _smooth_falloff(d, brush_r)

    dirty, touched = sf.apply_local(inflate, p, brush_r)
    assert 0 < touched < 0.1 * full                      # O(brush) << res^3
    assert len(dirty) >= 1
    # the re-meshed patch (dirty bricks only) is non-empty
    assert sf.extract_local(dirty_bricks=dirty).n_faces > 0


def test_reinitialize_moves_gradient_toward_one():
    sf = SparseField.from_field(_sphere, BOUNDS[0], BOUNDS[1], VOXEL, BAND, tile=6)
    mask = np.abs(sf.field) < BAND                       # squish the band: |grad| -> ~0.6 (within band, no clamp)
    sf.field[mask] *= 0.6
    before = SparseField.grad_norm_stats(sf.values, sf.h)
    after = sf.reinitialize(iters=12)
    assert before < 0.8
    assert abs(after - 1.0) < abs(before - 1.0)


def test_cache_skips_unchanged_bricks():
    """The brick-mesh working-set cache (the ReflexCache idea): a cold extract marches every active brick; after a
    local brush, a warm re-extract re-marches only the dirty bricks -- the loop's O(dirty) re-extract."""
    sf = SparseField.from_field(_sphere, BOUNDS[0], BOUNDS[1], VOXEL, BAND, tile=6)
    cold = sf.extract_cached()
    cold_marched = sf._last_marched
    assert cold_marched == len(sf.active)               # nothing cached yet

    p = np.array([0.6, 0.0, 0.0])

    def inflate(points):
        d = np.linalg.norm(points - p, axis=1)
        return -0.5 * BAND * _smooth_falloff(d, 0.25)

    dirty, _touched = sf.apply_local(inflate, p, 0.25)
    warm = sf.extract_cached()
    assert sf._last_marched < cold_marched              # most bricks reused from cache
    assert sf._last_marched <= len(dirty) + 2

    sf.cache_clear()                                     # a from-scratch rebuild of the edited field matches
    fresh = sf.extract_cached()
    assert warm.n_faces == fresh.n_faces


def test_deterministic_build():
    a = SparseField.from_field(_sphere, BOUNDS[0], BOUNDS[1], VOXEL, BAND, tile=6)
    b = SparseField.from_field(_sphere, BOUNDS[0], BOUNDS[1], VOXEL, BAND, tile=6)
    assert a.active == b.active
    av, bv = a.values, b.values                          # grab the (rebuilt) dicts once -- the property is not cheap
    assert av.keys() == bv.keys()
    for k in av:
        assert av[k] == bv[k]


def test_extract_dirty_returns_only_changed_bricks_for_realtime_sculpting():
    """The sculpting fast path: extract_dirty re-meshes only the bricks a stroke touched and returns them as a
    per-brick delta, never reassembling the whole mesh -- so the per-frame projection is O(brush), not O(model)."""
    import numpy as np, time
    from holographic.misc.holographic_sparsefield import SparseField, _smooth_falloff
    bounds = ((-1., -1, -1), (1., 1, 1)); voxel = 2.0 / 64; band = 4 * voxel

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    sf = SparseField.from_field(sphere, bounds[0], bounds[1], voxel, band, tile=8)
    cold = sf.extract_dirty()
    assert set(cold["updated"]) == set(sf.active) and not cold["removed"]   # cold = all active bricks

    p = np.array([0.6, 0., 0.]); r = 0.12
    def inflate(P):
        return -0.5 * band * _smooth_falloff(np.linalg.norm(P - p, axis=1), r)
    sf.apply_local(inflate, p, r)
    warm = sf.extract_dirty()
    assert 0 < len(warm["updated"]) < len(sf.active)                        # warm = only the touched bricks

    # a delta brick equals a fresh march of that brick (correct geometry)
    from holographic.mesh_and_geometry.holographic_meshbridge import marching_tetrahedra_vec
    for b, bm in warm["updated"].items():
        blk, ax = sf._materialize(b[0] * sf.tile, b[1] * sf.tile, b[2] * sf.tile, sf.tile + 1, sf.tile + 1, sf.tile + 1)
        assert bm.n_faces == marching_tetrahedra_vec(blk, ax, level=0.0).n_faces

    # MODEL-SIZE INDEPENDENCE: the per-frame delta is BRUSH-bounded on any model -- a small fraction of the total
    # bricks, whatever the model's size. (The full measurement of constant per-frame cost is in NOTES.)
    assert len(warm["updated"]) < 0.25 * len(sf.active)                    # the stroke's delta is a small slice of the model
    def four(P):
        cs = [np.array([0.5, 0, 0.]), np.array([-0.5, 0, 0.]), np.array([0, 0.5, 0.]), np.array([0, -0.5, 0.])]
        return np.min([np.linalg.norm(P - c, axis=1) - 0.4 for c in cs], axis=0)
    big = SparseField.from_field(four, bounds[0], bounds[1], voxel, band, tile=8); big.extract_dirty()
    big.apply_local(inflate, np.array([0.5 + 0.4, 0., 0.]), r)
    bw = big.extract_dirty()
    assert 0 < len(bw["updated"]) < 0.25 * len(big.active)                 # same brush, different model -> still brush-bounded

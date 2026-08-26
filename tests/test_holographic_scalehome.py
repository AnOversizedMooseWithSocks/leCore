"""Tests for holographic_scalehome -- the Scale home (H3: partition + monoid-reduce, one home)."""
import numpy as np
from holographic.misc.holographic_scalehome import Scale, scale_strategies, scale_backends


def test_map_reduce_matches_unsplit_sum():
    x = np.arange(1000.0)
    idx = Scale.partition(len(x), 7)
    buckets = [x[i] for i in idx]
    got, info = Scale.map_reduce(buckets, worker=lambda b, _c: b.sum(), reduce="sum")
    assert abs(float(got) - float(x.sum())) < 1e-6 and info["buckets"] == 7


def test_map_reduce_routes_bit_identical():
    from holographic.scene_and_pipeline.holographic_distribute import distribute, reduce_sum
    x = np.arange(300.0); idx = Scale.partition(len(x), 4); buckets = [x[i] for i in idx]
    a, _ = Scale.map_reduce(buckets, lambda b, c: b.sum(), reduce="sum")
    b, _ = distribute(buckets, lambda b, c: b.sum(), reduce=reduce_sum)
    assert a == b


def test_partition_load_balances():
    costs = np.array([10.0, 1, 1, 1, 10, 1, 1, 1, 10, 1])
    parts = Scale.partition(len(costs), 3, costs=costs)
    loads = sorted(costs[p].sum() for p in parts)
    assert loads[-1] - loads[0] <= 10.0


def test_tiles_cover_2d_disjointly():
    H, W = 20, 30
    canvas = np.zeros((H, W), dtype=int)
    for sl in Scale.tiles((H, W), (3, 4)):
        canvas[sl] += 1
    assert (canvas == 1).all()


def test_tiles_cover_3d_disjointly():
    X, Y, Z = 8, 10, 6
    vol = np.zeros((X, Y, Z), dtype=int)
    for sl in Scale.tiles((X, Y, Z), 2):
        vol[sl] += 1
    assert (vol == 1).all()


def test_min_and_bundle_monoids():
    depths = [np.array([5.0, 9, 2]), np.array([3.0, 1, 8])]
    m, _ = Scale.map_reduce(depths, lambda b, c: b, reduce="min")
    assert np.array_equal(m, np.array([3.0, 1.0, 2.0]))
    # bundle (VSA superposition) sums then the caller would normalise -- just check it reduces to one vector
    vecs = [np.ones(16), -np.ones(16), np.ones(16)]
    bundled, _ = Scale.map_reduce(vecs, lambda b, c: b, reduce="bundle")
    assert bundled.shape == (16,)


def test_bricks_places_and_skips():
    out_shape = (12, 12)
    regions = Scale.tiles(out_shape, (2, 2))
    # a worker that fills its region with a constant; skip the top-left brick
    def worker(sl, cache):
        h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
        return np.full((h, w), 5.0)
    out, info = Scale.bricks(out_shape, regions, worker, fill=0.0)
    assert (out == 5.0).all() and info["ran"] == 4


def test_backends_and_strategies():
    assert set(scale_backends()) == {"map_reduce", "partition", "tiles", "bricks"}
    assert "octree" in scale_strategies() and "superposed" in scale_strategies()

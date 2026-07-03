"""Distributed computation: reassembly is the computation's own commutative monoid, so buckets are order-independent
(=> distributable) and the shared cache is computed once. Correctness is proven against the monolithic result."""
import numpy as np
from holographic_distribute import (partition, adaptive_partition, distribute, distribute_scatter,
                                     reduce_sum, reduce_min, reduce_max, reduce_bundle)


def test_force_superposition_exact_and_order_independent():
    from holographic_fields import attractor_force
    rng = np.random.default_rng(1)
    P = rng.standard_normal((120, 2)); centers = rng.standard_normal((20, 2)) * 3
    mono = sum(attractor_force(P, c) for c in centers)
    buckets = partition(len(centers), 6)
    worker = lambda b, c: sum(attractor_force(P, centers[i]) for i in b)
    dist, info = distribute(buckets, worker, reduce=reduce_sum)
    assert np.allclose(mono, dist, atol=1e-12)                   # superposition is exact vs monolithic
    d2, _ = distribute(buckets[::-1], worker, reduce=reduce_sum)
    assert np.allclose(dist, d2, atol=1e-12)                     # order-independent (to float rounding)
    assert info["buckets"] == len(buckets)


def test_sdf_union_min_is_bit_exact_order_independent():
    rng = np.random.default_rng(2)
    Q = rng.standard_normal((200, 3)); c = rng.standard_normal((12, 3)) * 2
    ev = lambda idxs: np.minimum.reduce([np.linalg.norm(Q - c[i], axis=1) - 0.7 for i in idxs])
    mono = ev(range(len(c)))
    pb = partition(len(c), 4)
    d, _ = distribute(pb, lambda b, _c: ev(b), reduce=reduce_min)
    assert np.array_equal(mono, d)                               # min monoid: BIT-exact
    d2, _ = distribute(pb[::-1], lambda b, _c: ev(b), reduce=reduce_min)
    assert np.array_equal(d, d2)                                 # and bit-exactly order-independent


def test_shared_cache_built_once_read_by_all_buckets():
    calls = {"build": 0, "read": 0}
    def build():
        calls["build"] += 1; return np.arange(10)
    cache = build()
    def w(b, c):
        calls["read"] += 1; return c.sum() * 0 + len(b)          # touches the shared cache
    pb = partition(30, 5)
    distribute(pb, w, reduce=reduce_sum, cache=cache)
    assert calls["build"] == 1 and calls["read"] == len(pb)


def test_scatter_tiles_equal_monolithic_no_seams():
    rng = np.random.default_rng(3)
    vals = rng.standard_normal((256, 3))
    mono = vals.copy()
    buckets = partition(256, 8)
    out, info = distribute_scatter((256, 3), buckets, lambda b, c: (b, vals[b]))
    assert np.array_equal(mono, out)                             # disjoint scatter == monolithic, bit-exact
    out2, _ = distribute_scatter((256, 3), buckets[::-1], lambda b, c: (b, vals[b]))
    assert np.array_equal(out, out2)                             # order-independent


def test_adaptive_partition_isolates_heavy_items():
    costs = np.array([20., 1, 1, 1, 1, 1, 1, 1])
    buckets = adaptive_partition(costs, 4)
    heaviest = max(sum(costs[i] for i in b) for b in buckets)
    assert heaviest <= 20 + 1e-9                                 # the heavy item is not stacked with others
    even_heaviest = max(sum(costs[i] for i in b) for b in partition(len(costs), 4))
    assert heaviest <= even_heaviest                             # never worse than an even split


def test_distributed_vsa_memory_bundle_reassembly():
    """A VSA key->value memory built in buckets and reassembled by BUNDLE equals the monolithic memory: unbind still
    recovers each value. This is the 'shortcut reassembly' for the VSA scene/memory case -- superposition."""
    from holographic_core import random_vector, bind, unbind, cosine
    rng = np.random.default_rng(4)
    keys = [random_vector(1024, rng) for _ in range(12)]
    vals = [random_vector(1024, rng) for _ in range(12)]
    records = [bind(keys[i], vals[i]) for i in range(12)]
    mono = reduce_sum(records)                                   # one-node memory (raw superposition)
    buckets = partition(12, 4)
    part_mems = [reduce_sum([records[i] for i in b]) for b in buckets]   # each bucket bundles its records
    reassembled = reduce_bundle(part_mems)                       # reassemble buckets by bundle (superposition)
    for i in range(12):
        rec = unbind(reassembled, keys[i])                       # recover the value bound under this key
        best = int(np.argmax([cosine(rec, v) for v in vals]))
        assert best == i                                         # every value still recoverable after distributed build
    assert cosine(mono, reassembled) > 0.99                      # same memory direction as the monolithic build


def test_reduce_sum_exact_is_bit_exact_order_independent():
    """The float-sum negative is removable: fixed-point (integer) accumulation carries the value in a wider
    representation where addition is exact + commutative, so the reassembly is bit-identical across bucket order."""
    from holographic_distribute import reduce_sum_exact, reduce_sum
    rng = np.random.default_rng(7)
    parts = [rng.standard_normal(300) * (10.0 ** rng.integers(-2, 2)) for _ in range(11)]
    ea = reduce_sum_exact(parts); eb = reduce_sum_exact(parts[::-1])
    assert np.array_equal(ea, eb)                                # BIT-exact regardless of order
    assert not np.array_equal(reduce_sum(parts), reduce_sum(parts[::-1]))   # plain float sum is not
    assert np.abs(ea - reduce_sum(parts)).max() < 1e-6           # and it still agrees with the float sum


def test_partition_2d_tiles_cover_disjointly():
    from holographic_distribute import partition_2d, distribute_bricks
    H, W = 40, 60
    tiles = partition_2d((H, W), (4, 5))
    field = np.random.default_rng(0).standard_normal((H, W))
    out, info = distribute_bricks((H, W), tiles, lambda r, c: field[r])
    assert np.array_equal(field, out) and info["regions"] == 20   # disjoint cover == monolithic
    out2, _ = distribute_bricks((H, W), tiles[::-1], lambda r, c: field[r])
    assert np.array_equal(out, out2)                              # order-independent


def test_partition_3d_bricks_bake_exact_and_order_independent():
    from holographic_distribute import partition_3d, distribute_bricks
    res = 32; g = np.linspace(-3, 3, res)
    GX, GY, GZ = np.meshgrid(g, g, g, indexing="ij"); Pts = np.stack([GX, GY, GZ], -1)
    balls = np.array([[0., 0, 0], [1.3, 0.2, 0.1]])
    sdf = lambda P: np.minimum.reduce([np.linalg.norm(P - c, axis=-1) - 0.7 for c in balls])
    mono = sdf(Pts)
    bricks = partition_3d((res, res, res), (4, 4, 4))
    brk, info = distribute_bricks((res, res, res), bricks, lambda r, c: sdf(Pts[r]))
    assert np.array_equal(mono, brk) and info["regions"] == 64    # brick bake == dense bake, bit-exact
    import random; bs = bricks[:]; random.Random(1).shuffle(bs)
    brk2, _ = distribute_bricks((res, res, res), bs, lambda r, c: sdf(Pts[r]))
    assert np.array_equal(brk, brk2)                              # bricks distributable (order-independent)


def test_sparse_brick_skip_keeps_surface():
    from holographic_distribute import partition_3d, distribute_bricks
    res = 48; g = np.linspace(-6, 6, res)                         # large volume
    GX, GY, GZ = np.meshgrid(g, g, g, indexing="ij"); Pts = np.stack([GX, GY, GZ], -1)
    sdf = lambda P: np.linalg.norm(P, axis=-1) - 0.8              # one small ball -> mostly empty
    mono = sdf(Pts)
    nb = 6; bricks = partition_3d((res, res, res), (nb, nb, nb)); diag = 12.0 / nb * np.sqrt(3)
    skip = lambda r: abs(float(sdf(Pts[r].reshape(-1, 3).mean(0)[None])[0])) > diag
    sparse, info = distribute_bricks((res, res, res), bricks, lambda r, c: sdf(Pts[r]), fill=99.0, skip=skip)
    assert info["skipped"] > info["ran"]                          # low occupancy -> most bricks skipped
    surf = np.abs(mono) < 0.2
    assert np.array_equal(sparse[surf], mono[surf])              # surface untouched by the skip

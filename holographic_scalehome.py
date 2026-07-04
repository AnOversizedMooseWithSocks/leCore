"""holographic_scalehome.py -- the SCALE home (consolidation backlog H3): one place for "make this bigger than one
box / one pass can hold", i.e. partition a job, run the pieces independently, and reassemble.

WHY THIS EXISTS
---------------
The scale-out machinery already ships in holographic_distribute -- partition / adaptive_partition (load balanced),
partition_2d / partition_3d (image tiles / volume bricks), distribute_bricks (place a worker's output per region),
and the commutative-monoid reducers (reduce_sum / min / max / bundle / sum_exact). The UnifiedMind wires it as
faculties (distribute_compute, partition_domain, partition_grid, distribute_bricks). What was missing is the plain
library HOME the other homes have -- a single import you reach for when you want to scale something out -- and a
canonical `map_reduce`.

`Scale` is that home. It PROMOTES holographic_distribute (route, don't rewrite):

    Scale.map_reduce(buckets, worker, reduce='sum'|'min'|'max'|'bundle'|'exact'|callable, cache=None)
        -> the core: hand every bucket the SAME shared read-only `cache`, run worker(bucket, cache), reassemble with a
           COMMUTATIVE monoid so the result is independent of bucket order (=> buckets can run on separate machines).
    Scale.partition(n, k, costs=None)     -> split n items into k buckets (load-balanced when costs are given)
    Scale.tiles(shape, blocks)            -> 2D image tiles / 3D volume bricks (disjoint slice tuples)
    Scale.bricks(out_shape, regions, worker, ..., skip=...)  -> run a worker per region and PLACE it; skip empty bricks

THE STRATEGIES (how you scale depends on the shape of the problem; each is a real technique the engine ships):
    tiling      -- Scale.tiles / Scale.bricks: cut the domain into disjoint blocks (this home).
    octree      -- holographic_octree: adaptive subdivision, prune empty space (scale by NOT storing nothing).
    multires    -- holographic_multires: a coarse-to-fine pyramid (scale by resolution, spend detail where it shows).
    superposed  -- reduce_bundle / VSA bundling: pack many items into ONE vector (scale by superposition, not storage).
    sparsefield -- holographic_sparsefield: store only the active cells (scale by occupancy).

"Limitations are usually just bad approaches": most apparent size limits fall to one of these -- partition it,
prune the empty part, drop resolution where it doesn't show, superpose, or store only what's there.
"""
import numpy as np


class Scale:
    """A namespace of staticmethods over the engine's scale-out machinery. Partition, map-reduce, tile, brick."""

    # the reassembly monoids, by name -- a commutative reducer makes bucket order irrelevant (so, distributable)
    @staticmethod
    def _reducer(reduce):
        from holographic_distribute import (reduce_sum, reduce_min, reduce_max, reduce_bundle, reduce_sum_exact)
        table = {"sum": reduce_sum, "min": reduce_min, "max": reduce_max, "bundle": reduce_bundle,
                 "exact": reduce_sum_exact}
        return table.get(reduce, reduce) if isinstance(reduce, str) else reduce

    @staticmethod
    def map_reduce(buckets, worker, reduce="sum", cache=None):
        """The core scale-out: decompose a job into `buckets`, hand each the SAME shared read-only `cache`, run
        worker(bucket, cache) on each, and reassemble with a COMMUTATIVE monoid `reduce` so the result does not
        depend on bucket order -- which is exactly the property that lets the buckets run on separate machines with
        no stitch pass. `reduce` is 'sum' (linear accumulation), 'min'/'max' (depth / bounds), 'bundle' (VSA
        superposition), 'exact' (fixed-point deterministic sum), or your own callable. Routes to
        holographic_distribute.distribute."""
        from holographic_distribute import distribute
        return distribute(buckets, worker, reduce=Scale._reducer(reduce), cache=cache)

    @staticmethod
    def partition(n, k, costs=None):
        """Split a domain of `n` items into `k` disjoint buckets (index arrays). With per-item `costs` it LOAD
        BALANCES -- heaviest-first onto the lightest bucket -- so the slowest bucket (which bounds wall-time) is
        minimised. Routes to holographic_distribute.partition / adaptive_partition."""
        from holographic_distribute import partition, adaptive_partition
        return adaptive_partition(np.asarray(costs, float), k) if costs is not None else partition(n, k)

    @staticmethod
    def tiles(shape, blocks):
        """Cut a 2D image (shape=(H,W)) into TILES or a 3D volume (shape=(X,Y,Z)) into BRICKS. `blocks` is an int or
        a per-axis tuple. Returns disjoint slice-tuples covering the domain -- each an independent bucket, and also
        the cache-blocking layout (a tile sized to a working budget streams through a fast cache level). Routes to
        holographic_distribute.partition_2d / partition_3d."""
        from holographic_distribute import partition_2d, partition_3d
        return partition_3d(shape, blocks) if len(shape) == 3 else partition_2d(shape, blocks)

    @staticmethod
    def bricks(out_shape, regions, worker, cache=None, fill=0.0, skip=None):
        """Run worker(region, cache) on each tile/brick and PLACE its result at that region -- disjoint, so
        order-independent and seamless (a shared read-only cache makes borders agree). `skip(region)->bool` drops
        EMPTY bricks (sparse volumes: the real 3D win beyond parallelism). Returns (out, info). Routes to
        holographic_distribute.distribute_bricks."""
        from holographic_distribute import distribute_bricks
        return distribute_bricks(out_shape, regions, worker, cache=cache, fill=fill, skip=skip)


def scale_strategies():
    """The scaling strategies the home names (for the catalog / discovery)."""
    return ("tiling", "octree", "multires", "superposed", "sparsefield")


def scale_backends():
    """The scale facilities the home exposes."""
    return ("map_reduce", "partition", "tiles", "bricks")


def _selftest():
    # map_reduce == the direct computation: sum a big vector by partitioning it into buckets and reduce-summing
    x = np.arange(1000.0)
    idx = Scale.partition(len(x), 7)                                  # 7 buckets of indices
    buckets = [x[i] for i in idx]
    got, info = Scale.map_reduce(buckets, worker=lambda b, _c: b.sum(), reduce="sum")
    assert abs(float(got) - float(x.sum())) < 1e-6                    # partition+reduce matches the un-split sum
    assert info["buckets"] == 7

    # load-balanced partition: heavy items spread out so the buckets' costs are close
    costs = np.array([10.0, 1, 1, 1, 10, 1, 1, 1, 10, 1])
    parts = Scale.partition(len(costs), 3, costs=costs)
    loads = sorted(costs[p].sum() for p in parts)
    assert loads[-1] - loads[0] <= 10.0                              # the max bucket isn't wildly heavier than the min

    # tiles cover a 2D domain disjointly and completely
    H, W = 20, 30
    canvas = np.zeros((H, W), dtype=int)
    for sl in Scale.tiles((H, W), (3, 4)):
        canvas[sl] += 1
    assert (canvas == 1).all()                                       # every pixel covered exactly once

    # 'min' monoid: a depth-style reduction over buckets
    depths = [np.array([5.0, 9, 2]), np.array([3.0, 1, 8])]
    m, _ = Scale.map_reduce(depths, worker=lambda b, _c: b, reduce="min")
    assert np.array_equal(m, np.array([3.0, 1.0, 2.0]))

    print("OK: holographic_scalehome self-test passed (map_reduce matches un-split sum; partition load-balances; "
          "tiles cover disjointly; min monoid works; strategies %s)" % ", ".join(scale_strategies()))


if __name__ == "__main__":
    _selftest()

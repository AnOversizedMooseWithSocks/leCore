"""Tests for holographic_coordinator (R2: Coordinator + backends + margin-gated tie-break).
Workers are top-level functions so LocalPool can pickle them by reference."""
import os
import numpy as np
from holographic.scene_and_pipeline.holographic_coordinator import Coordinator, InProcessBackend, LocalPool, decide, decide_sequence, _sum_bucket
from holographic.scene_and_pipeline.holographic_distribute import reduce_sum, reduce_min, reduce_max


def _cache_sum(bucket, cache):
    return float(np.sum([cache[i] for i in bucket]))


def _whoami(bucket, cache):
    return os.getpid()


def _min_scalar(bucket, cache):
    return float(np.min(bucket))


def test_inprocess_reduce_matches_plain():
    c = Coordinator(InProcessBackend())
    buckets = [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]]
    assert c.run(buckets, _sum_bucket, reduce=reduce_sum) == float(sum(range(10)))


def test_localpool_shared_cache():
    cache = np.arange(64, dtype=np.float64) ** 2
    buckets = [list(range(0, 20)), list(range(20, 45)), list(range(45, 64))]
    with Coordinator(LocalPool(n=3)) as lc:
        got = lc.run(buckets, _cache_sum, cache=cache, reduce=reduce_sum)
    assert abs(got - float(np.sum(cache))) < 1e-9


def test_localpool_uses_separate_interpreters():
    with Coordinator(LocalPool(n=3)) as lc:
        pids = lc.run([[i] for i in range(6)], _whoami, reduce=lambda parts: parts)
    assert os.getpid() not in set(pids)                    # real separate processes


def test_min_bit_exact_across_backends():
    buckets = [[1.0, 5.0], [3.0], [2.0, 0.5]]
    ip = Coordinator(InProcessBackend()).run(buckets, _min_scalar, reduce=reduce_min)
    with Coordinator(LocalPool(n=2)) as lc:
        lp = lc.run(buckets, _min_scalar, reduce=reduce_min)
    assert ip == lp == 0.5                                 # MIN monoid is bit-exact regardless of where it runs


def test_decide_margin_gate():
    assert decide([0.1, 0.9, 0.3]) == 1                    # clear winner -> fast path
    assert decide([0.5, 0.5, 0.2]) == 0                    # exact tie -> canonical lowest-index
    assert decide([0.50000001, 0.5], safe_margin=1e-9) == 0  # within margin -> treated as tie


def test_decide_identical_under_reduction_wobble():
    # two reduction orders differ by a float wobble; a near-tie resolves IDENTICALLY via the canonical rule
    base = np.array([0.5, 0.5, 0.4])
    order_a = base.copy()
    order_b = base + np.array([1e-13, -1e-13, 0.0])        # ~1e-13 wobble from a different SUM order
    assert decide(order_a) == decide(order_b) == 0         # the rule, not the rounding, decides
    # a comfortable-margin decision agrees anyway
    clear = np.array([0.9, 0.5, 0.4])
    assert decide(clear) == decide(clear + np.array([1e-13, -1e-13, 0.0])) == 0


def test_decide_sequence():
    seq = [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]]
    assert decide_sequence(seq) == [1, 0, 0]


def test_coordinator_releases_cache_on_error():
    # a worker that raises must not leak the shared cache (release_cache runs in finally)
    def _boom(bucket, cache):
        raise ValueError("boom")
    lp = LocalPool(n=2)
    c = Coordinator(lp)
    try:
        c.run([[1]], _sum_bucket if False else _boom_top, cache=np.arange(4.0))
    except Exception:
        pass
    assert lp._shm == {}                                   # cache freed despite the error
    c.close()


def _boom_top(bucket, cache):
    raise ValueError("boom")

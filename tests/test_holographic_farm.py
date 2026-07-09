"""Tests for holographic_farm (R3: worker daemon + NetworkFarm coordinator backend, over real http/json)."""
import numpy as np
import pytest
from holographic.scene_and_pipeline.holographic_coordinator import Coordinator
from holographic.misc.holographic_farm import WorkerDaemon, NetworkFarm, _sum_indices, _min_indices, _content_hash
from holographic.scene_and_pipeline.holographic_distribute import reduce_sum, reduce_min


@pytest.fixture
def daemon():
    d = WorkerDaemon(port=0)
    d.register_worker("sum_indices", _sum_indices)
    d.register_worker("min_indices", _min_indices)
    addr = d.start()
    yield d, addr
    d.stop()


def test_node_registers_and_reports_speed(daemon):
    d, addr = daemon
    farm = NetworkFarm([addr])
    assert farm.nodes and farm.nodes[0]["speed"] > 0
    farm.close()


def test_network_sum_matches_direct(daemon):
    d, addr = daemon
    cache = np.arange(30, dtype=np.float64) ** 2
    buckets = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]
    with Coordinator(NetworkFarm([addr])) as coord:
        got = coord.run(buckets, "sum_indices", cache=cache, reduce=reduce_sum)
    assert abs(got - float(np.sum(cache))) < 1e-6


def test_network_min_bit_exact(daemon):
    d, addr = daemon
    cache = np.arange(30, dtype=np.float64) ** 2
    buckets = [list(range(0, 15)), list(range(15, 30))]
    with Coordinator(NetworkFarm([addr])) as coord:
        got = coord.run(buckets, "min_indices", cache=cache, reduce=reduce_min)
    assert got == float(np.min(cache))                         # MIN monoid: bit-exact over the network too


def test_cache_shipped_once_by_hash(daemon):
    d, addr = daemon
    cache = np.arange(30, dtype=np.float64) ** 2
    with Coordinator(NetworkFarm([addr])) as coord:
        coord.run([list(range(30))], "sum_indices", cache=cache, reduce=reduce_sum)
    assert _content_hash(cache) in d.caches                    # the node kept it for reuse


def test_unregistered_worker_refused(daemon):
    d, addr = daemon
    resp = d._handle("/task", {"worker": "rm_rf", "bucket": [1], "cache_hash": None})
    assert not resp["ok"] and "not registered" in resp["error"]


def test_worker_error_reported_not_crashed(daemon):
    d, addr = daemon
    def _bad(bucket, cache):
        raise ValueError("boom")
    d.register_worker("bad", _bad)
    resp = d._handle("/task", {"worker": "bad", "bucket": [1], "cache_hash": None})
    assert not resp["ok"] and "boom" in resp["error"]


def test_network_worker_needs_a_name():
    farm = NetworkFarm([])
    try:
        farm.submit(lambda b, c: 0, [1], ("net", None)); assert False   # a lambda has no registered name
    except (ValueError, RuntimeError):
        pass
    farm.close()

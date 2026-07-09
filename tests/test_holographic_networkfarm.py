"""Tests for the NetworkFarm coordinator backend + serve_worker (cross-machine compute over stdlib sockets/JSON)."""
import threading
import time
import numpy as np
import pytest
from http.server import HTTPServer
from holographic.scene_and_pipeline.holographic_coordinator import Coordinator, InProcessBackend, NetworkFarm, WorkerNode, _make_worker_handler, _sum_bucket, _encode, _decode
from holographic.scene_and_pipeline.holographic_distribute import reduce_sum, reduce_max


def _vsum(bucket, cache):
    return np.sum(np.array(bucket), axis=0)


@pytest.fixture
def farm_node():
    """A worker daemon on an OS-assigned free port (port 0 -> parallel-safe), with a couple of workers registered."""
    node = WorkerNode(token="farm", workers={"sum": _sum_bucket, "vsum": _vsum})
    httpd = HTTPServer(("127.0.0.1", 0), _make_worker_handler(node))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.15)
    yield "127.0.0.1:%d" % port
    httpd.shutdown(); httpd.server_close()


def test_farm_matches_inprocess(farm_node):
    buckets = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    local = Coordinator(InProcessBackend()).run(buckets, _sum_bucket, cache=None, reduce=reduce_sum)
    farm = Coordinator(NetworkFarm([farm_node], token="farm"))
    remote = farm.run(buckets, "sum", cache=None, reduce=reduce_sum)
    farm.close()
    assert remote == local == 45.0


def test_array_buckets_roundtrip(farm_node):
    farm = Coordinator(NetworkFarm([farm_node], token="farm"))
    vb = [[np.ones(4), np.full(4, 2.0)], [np.full(4, 3.0)]]
    r = farm.run(vb, "vsum", cache=None, reduce=reduce_sum)
    farm.close()
    assert np.allclose(r, [6, 6, 6, 6])


def test_unregistered_worker_refused(farm_node):
    farm = Coordinator(NetworkFarm([farm_node], token="farm"))
    with pytest.raises(Exception):
        farm.run([[1]], "not_a_real_worker", cache=None, reduce=reduce_sum)
    farm.close()


def test_auth_required(farm_node):
    farm = Coordinator(NetworkFarm([farm_node], token="WRONG"))
    with pytest.raises(Exception):
        farm.run([[1, 2]], "sum", cache=None, reduce=reduce_sum)
    farm.close()


def test_deterministic_reduce_order(farm_node):
    # results come back in BUCKET order regardless of completion, so repeated runs are identical
    farm = Coordinator(NetworkFarm([farm_node], token="farm"))
    buckets = [[float(i)] for i in range(20)]
    r1 = farm.run(buckets, "sum", cache=None, reduce=reduce_sum)
    r2 = farm.run(buckets, "sum", cache=None, reduce=reduce_sum)
    farm.close()
    assert r1 == r2 == float(sum(range(20)))


def test_codec_preserves_dtype_and_shape():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = _decode(_encode(a))
    assert b.shape == (2, 3) and str(b.dtype) == "float32" and np.allclose(a, b)


def test_health_lists_workers():
    node = WorkerNode(workers={"sum": _sum_bucket})
    assert node.handle_health()["workers"] == ["sum"]


def test_multi_node_round_robin(farm_node):
    # two references to the same node: buckets still distribute + reduce correctly
    farm = Coordinator(NetworkFarm([farm_node, farm_node], token="farm"))
    r = farm.run([[1, 2], [3, 4], [5, 6]], "sum", cache=None, reduce=reduce_sum)
    farm.close()
    assert r == 21.0

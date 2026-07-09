"""holographic_farm.py -- R3: the network backend (render farm / SETI@home). Run the coordinator's workers on OTHER
machines. Stdlib only: http.server + json + base64. No framework, no broker.

THE SHAPE
---------
  WorkerDaemon  -- runs ON each node. Holds a registry of NAMED workers (YOUR code, present on the node) and a cache
                   store keyed by content hash. Endpoints: /ping (health + measured speed for LPT), /cache (receive a
                   read-only cache ONCE, keep it by hash), /task (run a named worker on a bucket + cached cache).
  NetworkFarm   -- a Coordinator BACKEND. Registers nodes, publishes the cache to each ONCE (content-hashed; nodes
                   reuse it), and dispatches buckets to nodes concurrently. Plugs into the SAME Coordinator.run() as
                   the local pool -- only WHERE the worker runs changes.

So `Coordinator(NetworkFarm([...])).run(buckets, "worker_name", cache, reduce)` fans a monoid job across machines and
reduces the parts, exactly like the local pool -- one coordinator, one monoid reduce, now across a farm.

THE LOAD-BEARING SECURITY PRINCIPLE (loud)
------------------------------------------
Buckets are DATA; workers are YOUR REGISTERED CODE. The coordinator sends a worker NAME + a bucket of data -- never
code. A daemon runs only the workers its operator registered, and refuses an unknown name. You never ship code you
wouldn't run on a stranger's machine, and you never run a stranger's bucket AS code.

Other honest caveats (kept):
  * DO NOT expose a daemon openly. Bind to localhost for a trusted LAN, or put auth + TLS in front of it if it crosses
    the internet. This module ships the wire protocol, not the auth -- that is a deployment responsibility, stated so.
  * UNTRUSTED nodes need redundant compute + voting (R5) -- a node can faithfully return a plausible WRONG answer,
    which repair (cleanup/fountain/verify) does not catch. This backend is for a TRUSTED farm until R5 lands.
  * Offload COARSE: a bucket must be compute-heavy vs its transfer, or the network dominates (same rule as the pool).
"""
import base64
import hashlib
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


# ---- array <-> json helpers (a cache/result may be a numpy array) ------------------------------------------
def _pack_array(arr):
    """A numpy array as a json-safe dict (base64 bytes + shape + dtype)."""
    arr = np.ascontiguousarray(arr)
    return {"__ndarray__": base64.b64encode(arr.tobytes()).decode("ascii"),
            "shape": list(arr.shape), "dtype": str(arr.dtype)}


def _unpack_array(d):
    return np.frombuffer(base64.b64decode(d["__ndarray__"]), dtype=np.dtype(d["dtype"])).reshape(d["shape"])


def _content_hash(arr):
    """A stable content hash of a cache array -- so a node stores it once and reuses it (bake-once, share)."""
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode()).hexdigest()[:16]


# ============================================================================================================
# The worker daemon -- runs ON a node.
# ============================================================================================================
class WorkerDaemon:
    """An http worker node. Register your trusted workers by name, start it, and a NetworkFarm can dispatch buckets to
    it. Buckets are data; the workers are the code YOU registered here."""

    def __init__(self, host="127.0.0.1", port=0):
        self.host = host
        self.port = port
        self.workers = {}                                  # name -> fn(bucket, cache) -- YOUR registered code
        self.caches = {}                                   # content_hash -> array (shared read-only, kept for reuse)
        self._server = None
        self._thread = None

    def register_worker(self, name, fn):
        """Register a trusted worker (name -> fn(bucket, cache)). Only registered names can ever run."""
        self.workers[name] = fn
        return name

    # -- the request handling (called by the http handler below) --------------------------------------------
    def _handle(self, path, payload):
        if path == "/ping":
            return {"ok": True, "speed": self._measure_speed(), "workers": sorted(self.workers)}
        if path == "/cache":
            arr = _unpack_array(payload["array"])
            self.caches[payload["hash"]] = arr             # store the read-only cache by its content hash
            return {"ok": True, "hash": payload["hash"]}
        if path == "/task":
            name = payload["worker"]
            if name not in self.workers:                   # SECURITY: only registered code runs
                return {"ok": False, "error": "worker %r is not registered on this node" % name}
            cache = self.caches.get(payload.get("cache_hash"))
            bucket = payload["bucket"]
            try:
                result = self.workers[name](bucket, cache)
            except Exception as e:                         # a worker error is reported, not crashed
                return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
            return {"ok": True, "result": _pack_result(result)}
        return {"ok": False, "error": "unknown path %r" % path}

    def _measure_speed(self):
        """A crude throughput probe so the farm can LPT-balance faster nodes -- iterations/second on a fixed loop."""
        t0 = time.perf_counter()
        acc = 0
        for i in range(200000):
            acc = (acc + i) % 1000003
        return 200000.0 / (time.perf_counter() - t0 + 1e-9)

    # -- lifecycle ------------------------------------------------------------------------------------------
    def start(self):
        """Start serving in a background thread. Returns the bound address 'host:port' (port may have been 0=auto)."""
        daemon = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._reply(daemon._handle(self.path, payload))

            def do_GET(self):
                self._reply(daemon._handle(self.path, {}))

            def _reply(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):                     # silence the default stderr access log
                pass

        self._server = HTTPServer((self.host, self.port), _Handler)
        self.port = self._server.server_address[1]         # resolve an auto-assigned port
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return "%s:%d" % (self.host, self.port)

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def _pack_result(result):
    """A worker result as json-safe: an array is packed, scalars/lists pass through."""
    if isinstance(result, np.ndarray):
        return _pack_array(result)
    if isinstance(result, (np.floating, np.integer)):
        return float(result)
    return result


def _unpack_result(r):
    if isinstance(r, dict) and "__ndarray__" in r:
        return _unpack_array(r)
    return r


# ============================================================================================================
# The network farm -- a Coordinator BACKEND that dispatches to daemons.
# ============================================================================================================
class NetworkFarm:
    """Dispatch a monoid job's buckets across worker daemons. Plugs into Coordinator.run() like any backend: publish
    the cache once per node, submit worker(bucket, cache) per bucket, collect. Concurrent dispatch via a thread pool
    (http requests release the GIL during I/O, so nodes compute in parallel)."""

    by_name = True                                     # submit() takes a worker NAME (the daemon holds the code)

    def __init__(self, nodes=None, timeout=30, max_inflight=16):
        from concurrent.futures import ThreadPoolExecutor
        self.timeout = timeout
        self.nodes = []                                    # list of {"addr", "speed"}
        self._pool = ThreadPoolExecutor(max_workers=max_inflight)
        self._rr = 0                                       # round-robin cursor (fallback when speeds are unknown)
        for addr in (nodes or []):
            self.register_node(addr)

    def register_node(self, addr):
        """A node announces itself; we ping it for a speed estimate (feeds the least-loaded pick / LPT)."""
        info = self._get(addr, "/ping")
        self.nodes.append({"addr": addr, "speed": float(info.get("speed", 1.0))})
        return addr

    # -- the Coordinator backend interface ------------------------------------------------------------------
    def publish_cache(self, cache):
        """Send the read-only cache to each node ONCE, keyed by content hash; nodes keep it for reuse. Returns a
        handle the workers use to look the cache up locally -- no re-send per bucket."""
        if cache is None:
            return ("net", None)
        arr = np.asarray(cache)
        h = _content_hash(arr)
        packed = _pack_array(arr)
        for node in self.nodes:
            self._post(node["addr"], "/cache", {"hash": h, "array": packed})
        return ("net", h)

    def submit(self, worker, bucket, handle):
        """Dispatch one bucket to a node. `worker` is a NAME (or a function whose __name__ is registered on the
        nodes) -- code is never shipped. Returns a future."""
        name = worker if isinstance(worker, str) else getattr(worker, "__name__", None)
        if name is None:
            raise ValueError("network workers are referenced by NAME (registered on each node) -- pass a string")
        node = self._least_loaded()
        cache_hash = handle[1] if handle and handle[0] == "net" else None
        return self._pool.submit(self._dispatch, node["addr"], name, bucket, cache_hash)

    def release_cache(self, handle):
        pass                                               # nodes keep caches by hash for reuse (evict is optional)

    def close(self):
        self._pool.shutdown(wait=True)

    # -- internals ------------------------------------------------------------------------------------------
    def _least_loaded(self):
        """Pick the fastest node (LPT spirit: heaviest work to the quickest). Falls back to round-robin if speeds
        are equal/unknown, so no single node is hammered."""
        if not self.nodes:
            raise RuntimeError("no nodes registered with the farm")
        fastest = max(self.nodes, key=lambda n: n["speed"])
        if all(abs(n["speed"] - self.nodes[0]["speed"]) < 1e-6 for n in self.nodes):
            node = self.nodes[self._rr % len(self.nodes)]
            self._rr += 1
            return node
        return fastest

    def _dispatch(self, addr, name, bucket, cache_hash):
        resp = self._post(addr, "/task", {"worker": name, "bucket": _jsonable(bucket), "cache_hash": cache_hash})
        if not resp.get("ok"):
            raise RuntimeError("node %s failed task %r: %s" % (addr, name, resp.get("error")))
        return _unpack_result(resp["result"])

    def _post(self, addr, path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request("http://%s%s" % (addr, path), data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _get(self, addr, path):
        with urllib.request.urlopen("http://%s%s" % (addr, path), timeout=self.timeout) as r:
            return json.loads(r.read())


def _jsonable(bucket):
    """Buckets travel as json. numpy arrays/lists -> plain lists; anything already json-safe passes through."""
    if isinstance(bucket, np.ndarray):
        return bucket.tolist()
    return bucket


# ---- module-level workers for the self-test (registered on the daemon by name) -----------------------------
def _sum_indices(bucket, cache):
    """Sum a bucket of indices' values from the shared read-only cache (or the indices themselves if no cache)."""
    if cache is None:
        return float(np.sum(bucket))
    return float(np.sum([cache[i] for i in bucket]))


def _min_indices(bucket, cache):
    return float(np.min([cache[i] for i in bucket]))


def _selftest():
    from holographic.scene_and_pipeline.holographic_coordinator import Coordinator
    from holographic.scene_and_pipeline.holographic_distribute import reduce_sum, reduce_min

    # start a daemon on loopback with two registered workers (this stands in for a remote machine)
    node = WorkerDaemon(port=0)
    node.register_worker("sum_indices", _sum_indices)
    node.register_worker("min_indices", _min_indices)
    addr = node.start()
    try:
        farm = NetworkFarm([addr])
        assert farm.nodes and farm.nodes[0]["speed"] > 0        # the node registered + reported a speed

        cache = np.arange(30, dtype=np.float64) ** 2
        buckets = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]

        with Coordinator(farm) as coord:
            got = coord.run(buckets, "sum_indices", cache=cache, reduce=reduce_sum)
        assert abs(got - float(np.sum(cache))) < 1e-6           # every index summed once, over the network

        # MIN over the network reassembles bit-exact vs a direct compute
        with Coordinator(NetworkFarm([addr])) as coord:
            gmin = coord.run(buckets, "min_indices", cache=cache, reduce=reduce_min)
        assert gmin == float(np.min(cache))

        # SECURITY: an unregistered worker name is refused by the node
        resp = node._handle("/task", {"worker": "rm_rf", "bucket": [1], "cache_hash": None})
        assert not resp["ok"] and "not registered" in resp["error"]

        # cache is shipped ONCE and reused: the node holds it by content hash
        assert _content_hash(cache) in node.caches
    finally:
        node.stop()

    print("OK: holographic_farm self-test passed (worker daemon over http/json on loopback, cache shipped once by "
          "content hash, network SUM matches + MIN bit-exact via the same Coordinator.run, unregistered worker "
          "refused -- R3; buckets are data, workers are registered code)")


if __name__ == "__main__":
    # Run a worker daemon on THIS machine so a coordinator elsewhere can dispatch to it.
    #   python holographic_farm.py --host 0.0.0.0 --port 8763
    # SECURITY: only bind a public host behind auth/TLS on a trusted network -- see the module docstring.
    import argparse
    p = argparse.ArgumentParser(description="leCore worker daemon (a node in the render farm).")
    p.add_argument("--host", default="127.0.0.1", help="bind address (127.0.0.1 = local only; 0.0.0.0 = all NICs)")
    p.add_argument("--port", type=int, default=8763)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        _selftest()
    else:
        d = WorkerDaemon(host=args.host, port=args.port)
        d.register_worker("sum_indices", _sum_indices)          # register YOUR trusted workers here
        d.register_worker("min_indices", _min_indices)
        addr = d.start()
        print("leCore worker daemon serving on %s (workers: %s). Ctrl-C to stop." % (addr, sorted(d.workers)))
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            d.stop()

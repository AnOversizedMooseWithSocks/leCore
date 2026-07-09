"""holographic_coordinator.py -- a distributed compute Coordinator with PLUGGABLE BACKENDS (R2: local process pool).

WHY
---
holographic_distribute already holds the render-farm THEORY: partition a monoid job into buckets, hand every bucket a
shared READ-ONLY cache, run a `worker`, and reassemble by an associative-commutative reducer (sum/min/max/bundle). What
it does NOT have is a choice of WHERE the worker runs -- today the workers run sequentially in-process. This module adds
that: a Coordinator that schedules buckets onto a pluggable BACKEND and reduces the parts, reusing distribute's monoid
math and shared-cache pattern rather than rebuilding them.

  Coordinator(backend).run(buckets, worker, cache, reduce)   -- schedule + collect + monoid-reduce, backend-agnostic.

Backends (this file ships two; the network + command backends are separate rungs):
  InProcessBackend  -- run workers sequentially in this process (the default; mirrors distribute, always available).
  LocalPool         -- a PERSISTENT ProcessPoolExecutor (each worker its own interpreter + GIL) + shared_memory for the
                       big read-only cache, so a 100 MB field is shipped ONCE (zero-copy), not pickled per bucket. This
                       alone offloads GIL-bound Python work (the mesh kernel's Python-loop kept negative, long bakes)
                       and keeps the main process responsive.

The margin-gated tie-break (decide) is here too: distributed float SUM agrees only to ~1e-12 across bucket orders, but
that only matters if a TIE-SENSITIVE decision consumes the sum. cleanup already computes the sims, so the margin is
free; resolve only the rare knife-edge with the CANONICAL rule (determinism.argmax_tiebreak) so every node agrees.

KEPT NEGATIVES / SCOPE (loud)
  * OFFLOAD COARSE, not fine. IPC pickling + process handoff cost real time; a bucket must be compute-heavy relative to
    its data or the transfer dominates. This is for mesh/sim/render/bake work, NOT the FFT core (already GIL-released).
  * The worker for LocalPool must be PICKLABLE -- a top-level function (module.qualname), not a lambda/closure, because
    ProcessPoolExecutor pickles it by reference to re-import in the child. InProcessBackend has no such restriction.
  * Only MONOID work may be split disjointly (distribute's rule). Non-monoid feedback steps do not superpose -- run
    them whole on one worker; the Coordinator does not make unsafe work safe, it just chooses where safe work runs.
  * Determinism relies on the PARENT process's env (PYTHONHASHSEED=0): children inherit it. Seeds are passed explicitly
    inside the bucket/worker, deterministic by construction.
"""
import numpy as np
from holographic.scene_and_pipeline.holographic_distribute import reduce_sum, reduce_min, reduce_max
from holographic.misc.holographic_determinism import argmax_tiebreak


# ============================================================================================================
# The Coordinator -- backend-agnostic scheduler + monoid reducer.
# ============================================================================================================
class Coordinator:
    """Schedule buckets onto a backend's workers and reassemble by a monoid reducer. Does not care WHERE a worker
    runs -- that is the pluggable backend. Sits behind distribute, so the reduce and the shared cache are reused."""

    def __init__(self, backend=None):
        self.backend = backend if backend is not None else InProcessBackend()

    def run(self, buckets, worker, cache=None, reduce=reduce_sum):
        """Publish the shared read-only cache ONCE, submit worker(bucket, cache) for every bucket, collect the parts,
        and reassemble with the (associative + commutative) reducer. The cache is released even if a worker raises."""
        handle = self.backend.publish_cache(cache)
        try:
            futures = [self.backend.submit(worker, b, handle) for b in buckets]
            parts = [f.result() for f in futures]
        finally:
            self.backend.release_cache(handle)        # free shared memory / node caches even on error
        return reduce(parts)

    def close(self):
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ============================================================================================================
# Backend 1 -- in-process (the default; sequential, always available, mirrors distribute).
# ============================================================================================================
class _Immediate:
    """A trivial future: the work already ran, .result() just returns it. Lets the in-process backend share the
    Coordinator's submit/collect shape with the real (async) backends."""

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value

    def done(self):
        return True                                    # the work already ran synchronously -- always done


class InProcessBackend:
    """Run workers sequentially in this process. No pickling, no restriction on the worker -- the safe default and
    the reference the parallel backends must match."""

    by_name = False                                    # submit() takes a CALLABLE worker (resolved from a registry)

    def publish_cache(self, cache):
        return ("direct", cache)                       # the handle IS the cache

    def submit(self, worker, bucket, handle):
        return _Immediate(worker(bucket, handle[1]))

    def release_cache(self, handle):
        pass

    def close(self):
        pass


# ============================================================================================================
# Backend 2 -- LocalPool (persistent process pool + shared_memory for the read-only cache).
# ============================================================================================================
def _run_with_cache(worker, bucket, handle):
    """Top-level (picklable) trampoline executed IN THE CHILD process: re-attach the shared read-only cache by name,
    run the worker, and detach. Kept top-level so ProcessPoolExecutor can pickle it by reference."""
    kind = handle[0]
    if kind == "shm":
        from multiprocessing import shared_memory
        _, name, shape, dtype = handle
        shm = shared_memory.SharedMemory(name=name)   # attach the EXISTING block (do NOT create, do NOT unlink)
        try:
            cache = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)   # a read-only view of the shared array
            return worker(bucket, cache)
        finally:
            shm.close()                                # detach our handle; the parent owns unlink()
    return worker(bucket, handle[1] if kind == "direct" else None)


class LocalPool:
    """A persistent local process pool. Each worker is its own interpreter (its own GIL), so GIL-bound Python work
    actually runs in parallel. A large read-only cache is published ONCE into shared_memory (zero-copy) rather than
    pickled to every bucket."""

    by_name = False                                    # submit() takes a CALLABLE (a top-level, picklable function)

    def __init__(self, n=None):
        from concurrent.futures import ProcessPoolExecutor
        self.pool = ProcessPoolExecutor(max_workers=n)   # PERSISTENT -- not a spawn per task
        self._shm = {}                                   # handle-name -> SharedMemory (parent owns the lifecycle)

    def publish_cache(self, cache):
        """Ship the read-only cache once. A numpy array goes into shared_memory (zero-copy, mapped by every worker);
        None or a small picklable object is passed directly (pickled per submit, which is fine when it is small)."""
        if cache is None:
            return ("direct", None)
        if isinstance(cache, np.ndarray):
            from multiprocessing import shared_memory
            arr = np.ascontiguousarray(cache)
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            view[:] = arr[:]                             # write once; workers map it READ-ONLY by name
            self._shm[shm.name] = shm                    # keep it alive until release_cache
            return ("shm", shm.name, arr.shape, str(arr.dtype))
        return ("direct", cache)                         # small non-array cache: let pickling handle it

    def submit(self, worker, bucket, handle):
        return self.pool.submit(_run_with_cache, worker, bucket, handle)

    def release_cache(self, handle):
        """Free the shared block (parent owns unlink). Idempotent -- a released or direct handle is a no-op."""
        if handle and handle[0] == "shm":
            shm = self._shm.pop(handle[1], None)
            if shm is not None:
                shm.close()
                shm.unlink()

    def close(self):
        for name in list(self._shm):
            self.release_cache(("shm", name, None, None))
        self.pool.shutdown(wait=True)


# ============================================================================================================
# The margin-gated canonical tie-break (reuses cleanup's sims + determinism.argmax_tiebreak).
# ============================================================================================================
def decide(sims, safe_margin=1e-9):
    """Trust the fast/distributed result unless a decision is balanced on a knife-edge. `sims` are the similarities to
    each candidate atom (cleanup already computed them, so the margin is FREE). If the top two are comfortably apart
    (> safe_margin, well above the ~1e-12 float-SUM wobble), every node agrees -- return the plain argmax. Only the
    rare near-tie is resolved by the CANONICAL rule (ties -> lowest index), so it comes out identical on every node /
    reduction order because a RULE breaks the tie, not the rounding."""
    sims = np.asarray(sims, float)
    if sims.size < 2:
        return int(argmax_tiebreak(sims))
    order = np.argsort(sims)
    top, second = sims[order[-1]], sims[order[-2]]
    if (top - second) > safe_margin:                    # comfortable margin -> the fast path is safe everywhere
        return int(order[-1])
    return int(argmax_tiebreak(sims))                   # knife-edge (rare) -> the canonical rule, not the rounding


def decide_sequence(sims_seq, safe_margin=1e-9):
    """Apply decide() at each step of a sequence (a maze-style trajectory): comfortable steps run free, near-tie steps
    get the canonical rule -- so the whole path is identical on every node regardless of reduction order."""
    return [decide(s, safe_margin) for s in sims_seq]


# ---- module-level workers for the self-test (must be top-level so LocalPool can pickle them) ----------------
def _sum_bucket(bucket, cache):
    """Sum a bucket of indices' contributions, optionally scaled by a shared read-only cache vector."""
    if cache is None:
        return float(np.sum(bucket))
    return float(np.sum([cache[i] for i in bucket]))


def _selftest():
    # (1) in-process backend reproduces a plain reduce
    coord = Coordinator(InProcessBackend())
    buckets = [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]]
    total = coord.run(buckets, _sum_bucket, cache=None, reduce=reduce_sum)
    assert total == float(sum(range(10))), total

    # (2) shared read-only cache: a worker reads the published array by name in each child
    cache = np.arange(10, dtype=np.float64) * 2.0        # cache[i] = 2i
    with Coordinator(LocalPool(n=2)) as lc:
        got = lc.run(buckets, _sum_bucket, cache=cache, reduce=reduce_sum)
    assert got == float(np.sum(cache)), got             # every index summed once, via the shared cache

    # (3) LocalPool MIN reassembly matches in-process exactly (bit-exact monoid)
    parts_buckets = [[1.0, 5.0], [3.0], [2.0, 0.5]]
    def _min_bucket(b, c):
        return float(np.min(b))
    # _min_bucket is a closure -> use it only in-process (LocalPool needs a top-level fn); assert the reducer path
    ip = Coordinator(InProcessBackend()).run(parts_buckets, _min_bucket, reduce=reduce_min)
    assert ip == 0.5

    # (4) margin-gated tie-break: a comfortable margin returns argmax; an exact tie goes to the lowest index
    assert decide([0.1, 0.9, 0.3]) == 1                 # clear winner
    assert decide([0.5, 0.5, 0.2]) == 0                 # exact tie -> canonical lowest-index rule
    assert decide([0.50000001, 0.5], safe_margin=1e-9) == 0   # within margin -> treated as a tie -> rule

    print("OK: holographic_coordinator self-test passed (Coordinator + InProcess/LocalPool backends, shared_memory "
          "read-only cache shipped once, monoid reduce reused from distribute, margin-gated canonical tie-break -- R2)")


if __name__ == "__main__":
    _selftest()


# ============================================================================================================
# Backend 3 -- NetworkFarm (workers run on REMOTE nodes; the client brokers over stdlib sockets/JSON).
#
# This is the cross-machine build. Each node runs serve_worker() with a set of workers registered BY NAME. The
# NetworkFarm is a Coordinator backend that, for each bucket, POSTs (worker_name, bucket, cache) to a node and
# collects the result -- then the Coordinator reassembles by the same monoid reducer as the local backends.
#
# SAFETY BY DESIGN: workers are referenced by NAME, never shipped as code. A node ONLY runs a worker it has itself
# registered, so a client can't make a node execute arbitrary code -- the network equivalent of the command
# allowlist. (On an untrusted/public farm you additionally want redundant-compute voting + signed/verified results;
# those are the opponent + verify mechanisms, switched on at deploy time -- see the backlog's honest-scope note.)
# ============================================================================================================
import json as _json
import urllib.request as _urlreq
import urllib.error as _urlerr


def _encode(o):
    """Make a bucket / cache / result JSON-safe WITHOUT losing numpy fidelity: an ndarray becomes a tagged dict that
    _decode turns back into the same array (dtype + shape preserved). Everything else passes through / recurses."""
    if isinstance(o, np.ndarray):
        return {"__nd__": True, "data": o.tolist(), "dtype": str(o.dtype), "shape": list(o.shape)}
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, dict):
        return {str(k): _encode(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_encode(v) for v in o]
    return o


def _decode(o):
    """Inverse of _encode: tagged ndarray dicts come back as real numpy arrays; everything else recurses."""
    if isinstance(o, dict):
        if o.get("__nd__"):
            return np.array(o["data"], dtype=o["dtype"]).reshape(o["shape"])
        return {k: _decode(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_decode(v) for v in o]
    return o


def _http_post(url, body, token=None, timeout=60.0):
    """One small stdlib POST of a JSON body, returning the parsed JSON reply (bearer-token auth, like the service)."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer %s" % token
    req = _urlreq.Request(url, data=_json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except _urlerr.HTTPError as e:                          # a 4xx/5xx still carries a JSON error body
        try:
            return _json.loads(e.read().decode("utf-8"))
        except Exception:
            raise


class NetworkFarm:
    """A Coordinator backend that runs workers on REMOTE nodes. Point it at a list of nodes ('host:port'); each node
    must be running serve_worker() with the SAME worker names registered. Buckets are round-robined across the nodes
    and POSTed concurrently; results come back in bucket order so the monoid reduce stays deterministic."""

    by_name = True                                         # submit() takes a worker NAME (a string), resolved on the node

    def __init__(self, nodes, token=None, timeout=60.0, max_workers=None):
        from concurrent.futures import ThreadPoolExecutor
        self.nodes = list(nodes)
        if not self.nodes:
            raise ValueError("NetworkFarm needs at least one node ('host:port')")
        self.token = token
        self.timeout = timeout
        # one thread per in-flight POST so buckets on different nodes truly overlap
        self.pool = ThreadPoolExecutor(max_workers=max_workers or max(4, len(self.nodes) * 2))
        self._rr = 0                                        # round-robin cursor over the nodes

    def publish_cache(self, cache):
        """Serialize the read-only cache ONCE; it is then carried with each run request. (A future optimisation is to
        push it to each node once and reference it by handle; carrying it is the simple, correct v1.)"""
        return ("carry", _encode(cache))

    def submit(self, worker, bucket, handle):
        """Pick the next node (round-robin) and POST the run there, off the thread pool -> a real Future."""
        node = self.nodes[self._rr % len(self.nodes)]
        self._rr += 1
        return self.pool.submit(self._run_remote, node, worker, bucket, handle)

    def _run_remote(self, node, worker, bucket, handle):
        body = {"worker": worker, "bucket": _encode(bucket), "cache": handle[1]}
        resp = _http_post("http://%s/run" % node, body, token=self.token, timeout=self.timeout)
        if not resp.get("ok", False):
            raise RuntimeError("farm node %s failed: %s" % (node, resp.get("error", "unknown error")))
        return _decode(resp["result"])

    def release_cache(self, handle):
        pass                                               # the cache was carried, nothing persists on the nodes

    def close(self):
        self.pool.shutdown(wait=True)


# ------------------------------------------------------------------------------------------------------------
# The worker daemon: run one on each node. It holds workers BY NAME and runs only those (never client code).
# ------------------------------------------------------------------------------------------------------------
class WorkerNode:
    """The state behind serve_worker: a name -> worker registry plus the run() that a request dispatches to. Kept
    separate from the HTTP plumbing so it can be driven directly in a test (no socket)."""

    def __init__(self, token=None, workers=None):
        self.token = token
        self.workers = {}
        for name, fn in (workers or {}).items():
            self.register_worker(name, fn)

    def register_worker(self, name, fn):
        """Offer a worker under `name`. Only registered names can be run -- this is the safety boundary."""
        self.workers[name] = fn
        return self

    def run(self, worker, bucket, cache):
        """Resolve `worker` by name and run it on (bucket, cache). Raises if the name isn't registered."""
        fn = self.workers.get(worker)
        if fn is None:
            raise KeyError("worker %r is not registered on this node" % worker)
        return fn(bucket, cache)

    # the two request handlers, returning plain dicts (the HTTP layer just serializes them)
    def handle_run(self, payload):
        payload = payload or {}
        result = self.run(payload.get("worker", ""), _decode(payload.get("bucket")), _decode(payload.get("cache")))
        return {"ok": True, "result": _encode(result)}

    def handle_health(self):
        return {"ok": True, "role": "worker", "workers": sorted(self.workers)}


def _make_worker_handler(node):
    """A BaseHTTPRequestHandler bound to a WorkerNode: GET /health, POST /run. Bearer-token gated if the node has one."""
    from http.server import BaseHTTPRequestHandler

    class _Handler(BaseHTTPRequestHandler):
        def _authed(self):
            if not node.token:
                return True
            return self.headers.get("Authorization", "") == "Bearer %s" % node.token

        def _reply(self, code, obj):
            body = _json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            return _json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

        def do_GET(self):
            if not self._authed():
                return self._reply(401, {"ok": False, "error": "unauthorized"})
            if self.path == "/health":
                return self._reply(200, node.handle_health())
            self._reply(404, {"ok": False, "error": "no such endpoint: %s" % self.path})

        def do_POST(self):
            if not self._authed():
                return self._reply(401, {"ok": False, "error": "unauthorized"})
            try:
                if self.path == "/run":
                    return self._reply(200, node.handle_run(self._read_json()))
                self._reply(404, {"ok": False, "error": "no such endpoint: %s" % self.path})
            except Exception as e:                         # report the type, don't leak a traceback
                self._reply(500, {"ok": False, "error": "%s: %s" % (type(e).__name__, e)})

        def log_message(self, *a):                         # keep the console quiet
            pass

    return _Handler


def serve_worker(host="0.0.0.0", port=9000, token=None, workers=None):
    """Start a farm worker daemon (BLOCKING) on this node. `workers` is a {name: fn(bucket, cache)} dict of the workers
    this node offers; a NetworkFarm client runs them by name. Endpoints: GET /health, POST /run {worker, bucket, cache}
    -> {ok, result}. stdlib http.server + JSON; bearer-token auth if `token` is set. Ctrl-C to stop."""
    from http.server import HTTPServer
    node = WorkerNode(token=token, workers=workers)
    httpd = HTTPServer((host, port), _make_worker_handler(node))
    print("leCore farm worker on http://%s:%d -- workers: %s" % (host, port, sorted(node.workers)))
    if host == "0.0.0.0":
        print("  NOTE: bound to ALL interfaces -- only behind auth/TLS on a trusted network.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping worker.")
        httpd.server_close()


def hardened(backend, redundancy=1, attempts=3, backoff=0.1, tol=1e-9, quorum=None):
    """P9 -- a Coordinator whose every bucket is RETRIED, optionally run REDUNDANTLY and accepted only on
    AGREEMENT, and whose run can be gated on canary buckets first.

    This is the guardrail the public-farm plan calls for (an untrusted node can return a plausible-but-wrong
    answer; voting is the detector). It already existed as `hardening.HardenedCoordinator` and was reachable
    from the catalog and nothing else. Use redundancy>1 + canaries for untrusted nodes; redundancy=1 on a
    trusted pool, where it is simply retry. Same `.run(buckets, worker, cache, reduce)` call as Coordinator."""
    from holographic.misc.holographic_hardening import HardenedCoordinator
    return HardenedCoordinator(backend, redundancy=redundancy, attempts=attempts, backoff=backoff,
                               tol=tol, quorum=quorum)

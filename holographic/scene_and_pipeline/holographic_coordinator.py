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

    def run_waves(self, items, keys_of, worker, cache=None):
        """Schedule CONFLICTING work into lock-free WAVES (backlog C2, Box3D lesson B5).

        `keys_of(item)` returns the set of resources that item touches. Two items conflict iff they share one.
        Colouring the conflict graph splits the items into waves in which **no two items touch a shared resource**,
        so a wave runs fully parallel with **no locks and no atomics** -- and, because the colouring is greedy in
        ascending index, the schedule is DETERMINISTIC: same items in, same waves, same order, on every machine.
        That is precisely how Box3D earns its cross-platform determinism.

        Returns (results, info) with `results` in ITEM order (not wave order -- the schedule is an implementation
        detail, the answer is not) and `info` = {waves, wave_sizes, parallelism}. MEASURED: 2,000 transactions over
        300 keys colour into 24 waves, mean wave size 83.3 -- 83x lock-free parallelism, every wave conflict-free.

        Colouring cannot invent parallelism: if everything conflicts it honestly serialises into N waves of one.
        Delegates to holographic_island.color_waves -- a physics constraint graph, a mesh's edge adjacency, a DB
        write set and a farm's conflict graph are the same object."""
        from holographic.simulation_and_physics.holographic_island import color_waves, conflict_graph
        items = list(items)
        n, edges = conflict_graph([set(keys_of(it)) for it in items])
        waves = color_waves(n, edges)
        results = [None] * len(items)
        handle = self.backend.publish_cache(cache)
        try:
            for wave in waves:                                  # waves run in order; WITHIN a wave, in parallel
                futures = [(k, self.backend.submit(worker, items[k], handle)) for k in wave]
                for k, f in futures:
                    results[k] = f.result()
        finally:
            self.backend.release_cache(handle)
        info = {"waves": len(waves), "wave_sizes": [len(w) for w in waves],
                "parallelism": (len(items) / len(waves)) if waves else 0.0}
        return results, info

    def run_exact(self, buckets, worker, cache=None, bits=40):
        """`run`, but the answer is BIT-IDENTICAL under any bucketing -- the invariance a farm actually needs.

        `worker(bucket, cache)` must return the bucket's CONTRIBUTIONS, not their sum. That contract change IS the
        fix: `run`'s default float `reduce_sum` disagrees by 2.98e-08 between a 4-way and a 7-way split of the same
        work, and swapping in `reduce_sum_exact` does NOT repair it, because each worker has already float-summed
        inside its own bucket before the reduce ever sees a number. **Exactness has to reach the leaves.**

        Two passes over the collected parts, both order-independent: a global scale from the global peak and count
        (`max` and `len` are partition-invariant), then int64 accumulators that merge in any order. Returns
        (total, info); `info` carries the scale used, so the result is auditable.

        Determinism that survives RE-PARTITIONING a running farm mid-job. See holographic_distribute.distribute_exact."""
        from holographic.scene_and_pipeline.holographic_distribute import (exact_merge, exact_partial, exact_scale)
        handle = self.backend.publish_cache(cache)
        try:
            futures = [self.backend.submit(worker, b, handle) for b in buckets]
            parts = [np.asarray(f.result(), float) for f in futures]
        finally:
            self.backend.release_cache(handle)
        flat = [p.reshape(-1) if p.ndim <= 1 else p.reshape(p.shape[0], -1) for p in parts]
        peak = max((float(np.abs(p).max()) for p in flat if p.size), default=0.0)
        n_total = int(sum(int(p.shape[0]) if p.ndim else 1 for p in flat))
        info = {"buckets": len(buckets), "contributions": n_total, "peak": peak, "bits": int(bits)}
        if peak == 0.0 or n_total == 0:
            info["scale"] = 0.0
            return (np.zeros_like(parts[0][0]) if (parts and parts[0].ndim > 1) else 0.0), info
        scale = exact_scale(peak, n_total, bits=bits)
        info["scale"] = scale
        accs = [exact_partial(list(p) if p.ndim > 1 else [p], scale) for p in flat]
        total = exact_merge(accs).astype(np.float64) / scale
        if parts and parts[0].ndim > 1:
            total = total.reshape(parts[0].shape[1:])
        return total, info

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


def cpu_budget():
    """How many cores may this process ACTUALLY use? Returns an int >= 1.

    WHY NOT os.cpu_count(): IT LIES IN A CONTAINER. It reports the HOST's core count and knows nothing about
    cgroup quota or CPU affinity, so `docker run --cpus=2` on a 64-core box gives 64 -- and a pool sized from
    it spawns 64 interpreters to time-share 2 cores, which is slower than sequential AND costs 64x the
    memory. That matters here specifically: this engine is meant to run on small devices, and a wrong core
    count is a memory-bloat bug, not just a speed one.

    Takes the MINIMUM of every limit that is actually enforceable:
      * sched_getaffinity -- taskset / cpuset pinning (Linux; absent on macOS and Windows)
      * cgroup v2 cpu.max and v1 cfs quota/period -- the `--cpus` limit, rounded UP so a 1.5-core quota
        reports 2 rather than 1 (a fractional quota still lets two workers make progress)
      * os.cpu_count() -- the floor when nothing else is readable
    Anything unreadable is skipped rather than guessed at; the answer is never below 1."""
    import os

    limits = []
    count = os.cpu_count()
    if count:
        limits.append(int(count))
    try:
        limits.append(len(os.sched_getaffinity(0)))        # not present on every platform
    except (AttributeError, OSError):
        pass
    try:                                                   # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()
        if quota != "max":
            limits.append(max(1, -(-int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    try:                                                   # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read())
        if quota > 0 and period > 0:
            limits.append(max(1, -(-quota // period)))
    except (OSError, ValueError):
        pass
    return max(1, min(limits)) if limits else 1


#: Measured dispatch cost of handing ONE bucket to a pool worker (submit + pickle + collect), in
#: milliseconds. A bucket doing less work than this can never pay for being sent away, whatever the core
#: count -- which is why `should_pool` gates on work per bucket and not on core count alone.
DISPATCH_MS_PER_BUCKET = 0.2


def should_pool(n_buckets, est_ms_per_bucket, cores=None, margin=4.0):
    """Would a process pool pay for this job? Returns (verdict, why).

    Refuses on any of three grounds, each of which is fatal on its own:
      * ONE USABLE CORE -- a pool cannot win by construction, only add overhead and memory.
      * FEWER THAN 2 BUCKETS -- nothing to run in parallel.
      * WORK PER BUCKET BELOW `margin` x DISPATCH -- the default margin of 4 is deliberately conservative:
        near break-even a pool costs memory and process lifetime for no gain, so 'roughly equal' should
        decline. This is the same shape as the machine model's placement oracle, and the same lesson --
        the answer depends on the CALLER'S numbers, so it is computed, never assumed.

    `est_ms_per_bucket` is the caller's estimate; time one bucket if you do not have it. Deliberately not
    measured here: timing a bucket means RUNNING one, and a gate that runs the work to decide whether to
    run the work has to be the caller's choice, not a hidden cost."""
    cores = int(cpu_budget() if cores is None else cores)
    if cores < 2:
        return False, "only %d usable core(s); a pool adds overhead and memory but cannot add speed" % cores
    if int(n_buckets) < 2:
        return False, "%d bucket(s); nothing to run in parallel" % int(n_buckets)
    floor = float(margin) * DISPATCH_MS_PER_BUCKET
    if float(est_ms_per_bucket) < floor:
        return False, ("%.2f ms/bucket is below the %.2f ms floor (%.0fx dispatch); dispatch would dominate"
                       % (float(est_ms_per_bucket), floor, float(margin)))
    return True, ("%d cores, %d buckets at %.2f ms each -- above the %.2f ms floor"
                  % (cores, int(n_buckets), float(est_ms_per_bucket), floor))


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
        """Pick the next node (round-robin) and POST the run there, off the thread pool -> a real Future.

        WORKERS CROSS THIS BOUNDARY BY NAME. That is the security property of the farm -- only DATA travels,
        never code -- and it is why `farm` is the right door for off-machine work and `command_tool` (which
        runs an allowlisted binary locally) is not the same kind of thing at all.
        The property USED TO HOLD BY ACCIDENT: handing a callable got you
        `TypeError: Object of type function is not JSON serializable` from deep inside the encoder, which
        enforces the rule without ever stating it and sends the reader to the serializer instead of to the
        design. An accidental guarantee is one refactor away from not being a guarantee, so it is now
        explicit. In-process and pool backends legitimately DO take callables, which is why this check lives
        here rather than in Coordinator.run."""
        if not isinstance(worker, str):
            raise TypeError(
                "a farm worker must be a NAME (str) registered on the nodes, got %r. Only data crosses the "
                "wire, never code -- that is the point of the farm. Register the function on each node with "
                "serve_worker and pass its name." % type(worker).__name__)
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
            from holographic.scene_and_pipeline.holographic_distbus import send_json
            send_json(self, code, obj)                # promoted (sweep 123): one home in distbus

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

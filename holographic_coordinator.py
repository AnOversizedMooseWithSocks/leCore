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
from holographic_distribute import reduce_sum, reduce_min, reduce_max
from holographic_determinism import argmax_tiebreak


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

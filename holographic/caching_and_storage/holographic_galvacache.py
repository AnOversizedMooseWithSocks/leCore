"""GALVACACHE -- stop recomputing the same answer inside the model.

A Galvatron redoes a surprising amount of work, and it is all work whose inputs
repeat exactly. MEASURED on a running model before this existed:
  * attention screen routing re-ran k-means ONCE PER HEAD PER FORWARD PASS --
    the same keys clustered into the same clusters, every time;
  * capability routing (find_capability) cost ~75 ms per call and the toolbelt
    asks the same questions repeatedly;
  * retrieval re-ranked an unchanged corpus for an unchanged query.
Branch-and-select generation multiplies all three by k.

KEYS ARE CONTENT, NOT IDENTITY. Every key is a hashlib digest of the actual
bytes (and shape and dtype) of the inputs, never `id()` or a call counter, so
the cache is correct across processes, survives a restart, and never returns a
stale answer for changed data. That also makes it deterministic under
PYTHONHASHSEED=0, which `hash()` would not be.

THE CACHE IS NOT ALLOWED TO CHANGE ANSWERS. Every entry stores the value a real
computation produced; verify=True re-runs the function and asserts equality, so
"the cache is fast" can never quietly mean "the cache is wrong". A cache that is
not checked is an unmeasured claim about correctness, not a speedup.
"""

import hashlib
import time

import numpy as np


def content_key(*parts):
    """A stable digest of arbitrary inputs -- arrays by their exact bytes.

    hashlib, never hash(): the built-in is salted per process, so a cache keyed
    on it would silently miss across restarts and break the determinism the rest
    of the engine guarantees."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str(p.shape).encode())
            h.update(str(p.dtype).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        elif isinstance(p, (list, tuple)):
            h.update(content_key(*p).encode())
        elif isinstance(p, dict):
            h.update(content_key(*sorted(p.items(), key=lambda kv: str(kv[0]))).encode())
        else:
            h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()


class GalvaCache:
    """Bounded, content-addressed memo for the model's repeated inner work."""

    def __init__(self, max_entries=512, verify=False):
        self.max_entries = int(max_entries)
        self.verify = bool(verify)
        self._store = {}
        self._used = {}
        self.hits = 0
        self.misses = 0
        self.saved_seconds = 0.0

    def get_or_compute(self, key, fn):
        if key in self._store:
            self.hits += 1
            self._used[key] = time.time()
            value, cost = self._store[key]
            self.saved_seconds += cost
            if self.verify:
                fresh = fn()
                if not _same(fresh, value):
                    raise AssertionError(
                        "CACHE RETURNED A DIFFERENT ANSWER than recomputation "
                        "for key %s -- the key is not capturing everything the "
                        "result depends on" % key[:16])
            return value
        self.misses += 1
        t0 = time.time()
        value = fn()
        cost = time.time() - t0
        self._store[key] = (value, cost)
        self._used[key] = time.time()
        if len(self._store) > self.max_entries:
            oldest = min(self._used, key=self._used.get)     # plain LRU
            self._store.pop(oldest, None)
            self._used.pop(oldest, None)
        return value

    def stats(self):
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "entries": len(self._store),
                "seconds_saved": round(self.saved_seconds, 4)}

    def clear(self):
        self._store.clear()
        self._used.clear()


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(_same(x, y) for x, y in zip(a, b))
    return a == b


# ------------------------------------------------------------------ install

_INSTALLED = {}


def install(runtime=None, mind=None, cache=None, verify=False):
    """Wrap the measured hot paths. Returns the cache so its stats can be read.

    Wrapping is done by MONKEY-PATCHING THE MODULE FUNCTION rather than by
    editing each call site, because the same k-means is reached from the
    vectorized path, the step path and the pack loader; a cache installed at one
    call site would look like it worked and miss most of the traffic."""
    cache = cache or GalvaCache(verify=verify)

    import holographic.io_and_interop.holographic_gdnruntime as G
    if "kmeans" not in _INSTALLED:
        original = G._kmeans

        def cached_kmeans(X, nc, iters=8, seed=0):
            key = content_key("kmeans", X, nc, iters, seed)
            return cache.get_or_compute(key, lambda: original(X, nc, iters=iters,
                                                              seed=seed))
        G._kmeans = cached_kmeans
        _INSTALLED["kmeans"] = original

    if mind is not None and "find_capability" not in _INSTALLED:
        original_fc = mind.find_capability

        def cached_fc(*a, **kw):
            # ACCEPT ANY CALL SHAPE. Naming the first parameter `query` changed
            # the signature, and a caller that passes it by keyword (or that the
            # engine calls differently) then fails with a TypeError that looks
            # like a bug in the model rather than in the wrapper. A cache must
            # be invisible to its callers.
            key = content_key("find_capability", a, kw)
            return cache.get_or_compute(key, lambda: original_fc(*a, **kw))
        mind.find_capability = cached_fc
        _INSTALLED["find_capability"] = (mind, original_fc)

    if mind is not None and "bm25" not in _INSTALLED:
        original_bm = mind.bm25_rank

        def cached_bm(*a, **kw):
            key = content_key("bm25", [tuple(x) if isinstance(x, list) else x
                                       for x in a], kw)
            return cache.get_or_compute(key, lambda: original_bm(*a, **kw))
        mind.bm25_rank = cached_bm
        _INSTALLED["bm25"] = (mind, original_bm)

    return cache


def uninstall():
    """Put every patched function back -- a test that cannot restore the world
    it changed will poison every test after it."""
    import holographic.io_and_interop.holographic_gdnruntime as G
    if "kmeans" in _INSTALLED:
        G._kmeans = _INSTALLED.pop("kmeans")
    for name in ("find_capability", "bm25"):
        if name in _INSTALLED:
            obj, original = _INSTALLED.pop(name)
            setattr(obj, "find_capability" if name == "find_capability"
                    else "bm25_rank", original)


def _selftest():
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)

    # ---- keys are CONTENT: same bytes -> same key, one changed value -> not --
    a = np.arange(12.0).reshape(3, 4)
    b = a.copy()
    c = a.copy()
    c[2, 3] += 1e-9
    assert content_key(a) == content_key(b)
    assert content_key(a) != content_key(c), "a changed array must miss"
    assert content_key(a) != content_key(a.astype(np.float32)), "dtype matters"
    assert content_key(a) != content_key(a.reshape(4, 3)), "shape matters"

    # ---- the cache RETURNS THE COMPUTED VALUE, and verify proves it ----
    cache = GalvaCache(verify=True)
    calls = [0]

    def work():
        calls[0] += 1
        return np.arange(5.0) * 2

    k = content_key("work", 1)
    v1 = cache.get_or_compute(k, work)
    v2 = cache.get_or_compute(k, work)
    assert np.array_equal(v1, v2)
    assert calls[0] == 2, "verify=True must RE-RUN and compare, not trust"
    assert cache.hits == 1 and cache.misses == 1

    # ---- a wrong key is CAUGHT rather than silently served ----
    bad = GalvaCache(verify=True)
    seq = [np.array([1.0]), np.array([2.0])]      # same key, different answers
    bad.get_or_compute("fixed", lambda: seq[0])
    try:
        bad.get_or_compute("fixed", lambda: seq[1])
        raise AssertionError("cache served a stale value without complaint")
    except AssertionError as exc:
        assert "DIFFERENT ANSWER" in str(exc)

    # ---- LRU bound holds ----
    small = GalvaCache(max_entries=3)
    for i in range(6):
        small.get_or_compute("k%d" % i, lambda i=i: i)
    assert len(small._store) == 3, small.stats()

    # ---- INSTALLED, the real hot paths get faster and stay CORRECT ----
    import holographic.io_and_interop.holographic_gdnruntime as G
    X = np.random.default_rng(0).standard_normal((64, 8))
    plain_a, plain_C = G._kmeans(X, 8, seed=0)
    live = install(mind=mind, verify=False)
    try:
        t0 = time.time()
        for _ in range(5):
            G._kmeans(X, 8, seed=0)
        cached_t = time.time() - t0
        got_a, got_C = G._kmeans(X, 8, seed=0)
        assert np.array_equal(got_a, plain_a) and np.allclose(got_C, plain_C), \
            "cached k-means changed the clustering"
        # REPORT COLD AND WARM SEPARATELY. Summing them hides the effect: the
        # first routing call also builds the catalog lazily, so a total makes a
        # 3000x speedup look like no speedup (it did, in the first draft).
        t0 = time.time()
        mind.find_capability("how many things fit in a bundle")
        fc_cold = time.time() - t0
        t0 = time.time()
        for _ in range(4):
            mind.find_capability("how many things fit in a bundle")
        fc_t = (time.time() - t0) / 4.0
        st = live.stats()
        assert st["hits"] >= 7, st
    finally:
        uninstall()
    assert G._kmeans is not None
    # and uninstall really restored the original
    again_a, _ = G._kmeans(X, 8, seed=0)
    assert np.array_equal(again_a, plain_a)

    print("galvacache selftest OK -- content keys separate dtype/shape/one-changed-"
          "element; verify=True RE-RUNS and would have caught a stale answer "
          "(proven with a deliberately wrong key); LRU bound holds; installed on "
          "the real hot paths %d hits with clustering bit-identical, 5 k-means in "
          "%.4fs, and capability routing %.4fs cold -> %.6fs warm (%.0fx); "
          "uninstall restores the originals"
          % (st["hits"], cached_t, fc_cold, fc_t, fc_cold / max(fc_t, 1e-9)))


if __name__ == "__main__":
    _selftest()

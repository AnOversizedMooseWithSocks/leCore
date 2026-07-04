"""holographic_memoryhome.py -- the MEMORY home (consolidation backlog H6): keep the hot working set where the CPU
can reach it fast. One place for the cache-hierarchy levers the engine actually has.

THE HIERARCHY (the organizing frame)
------------------------------------
registers -> L1 -> L2 -> L3 -> RAM -> disk, each ~10x bigger and ~10x slower than the one before. You do not place
data in the CPU caches by hand, but you decide what LANDS there, and that is most of the speed. The levers:

  * RESIDENCY  -- keep a reused FFT spectrum around instead of recomputing it. holographic_residency.SpectrumCache
                  is an LRU of atom -> rfft(atom); a bind/unbind against a KNOWN operand then skips the forward
                  transform. Pure speed-up, bit-identical to the plain op.
  * BATCHED / CONTIGUOUS LAYOUT -- do a whole record in ONE FFT over a stacked, contiguous array (holographic_ai's
                  bind_batch / bundle_bind), not a Python loop of per-pair binds. Contiguous memory streams through a
                  cache level; a loop of small ops thrashes it.
  * TILING TO FIT -- cut a big working set into tiles sized to a cache level so each tile stays resident while it is
                  worked (delegates to the Scale home's partition/tiles).
  * OPT-IN BACKENDS -- move the hot array math to the GPU (CuPy, holographic_backend) or a numba-jitted kernel
                  (holographic_jit) when present. Accelerators only: the engine runs and passes every test on plain
                  NumPy without them.

`Memory` exposes those levers behind one door (route, don't rewrite):

    Memory.spectrum_cache(max_items)               # an LRU residency cache of FFT spectra
    Memory.bind_cached(a, b, cache) / unbind_cached(c, a, cache)   # bind/unbind reusing resident spectra
    Memory.bind_batch(keys, values)                # a whole record in one batched, contiguous FFT
    Memory.tiles(shape, blocks)                    # cut a working set into cache-sized tiles (-> Scale)
    Memory.backend(kind) / gpu_available()         # select the opt-in GPU / numba fast path (falls back to NumPy)
"""
import numpy as np


class Memory:
    """A namespace of staticmethods over the cache-hierarchy levers. Residency / batched layout / tiling / backend."""

    @staticmethod
    def spectrum_cache(max_items=4096):
        """An LRU RESIDENCY cache mapping an atom to its real-FFT spectrum, so a repeated bind/unbind against a known
        operand skips the forward transform. A pure speed-up that can never change a result. Routes to
        holographic_residency.SpectrumCache."""
        from holographic_residency import SpectrumCache
        return SpectrumCache(max_items=max_items)

    @staticmethod
    def bind_cached(a, b, cache):
        """bind(a, b) reusing whichever operand spectra `cache` already holds -- BIT-IDENTICAL to bind(), just
        skipping the forward FFT on a cache hit. Routes to holographic_residency.bind_cached."""
        from holographic_residency import bind_cached
        return bind_cached(a, b, cache)

    @staticmethod
    def unbind_cached(composite, a, cache):
        """unbind(composite, a) reusing the cached spectrum of the (usually known) key `a`. Routes to
        holographic_residency.unbind_cached."""
        from holographic_residency import unbind_cached
        return unbind_cached(composite, a, cache)

    @staticmethod
    def bind_batch(keys, values):
        """Encode a whole record -- bundle(bind(k_i, v_i)) -- in ONE batched FFT over CONTIGUOUS stacked arrays,
        instead of a Python loop of per-pair binds. The cache-resident layout for records/scenes/recipes. Routes to
        holographic_ai.bundle_bind. (Kept negative: the batched FFT differs from the scalar-loop bind by ~1e-12, so
        it is kept OUT of tie-sensitive decision paths.)"""
        from holographic_ai import bundle_bind
        return bundle_bind(keys, values)

    @staticmethod
    def tiles(shape, blocks):
        """Cut a working set (2D image / 3D volume) into tiles sized to fit a cache level, so each tile stays
        resident while worked. Delegates to the Scale home (consolidation H3)."""
        from holographic_scalehome import Scale
        return Scale.tiles(shape, blocks)

    @staticmethod
    def gpu_available():
        """True if a CuPy GPU backend is importable. Routes to holographic_backend.gpu_available."""
        from holographic_backend import gpu_available
        return bool(gpu_available())

    @staticmethod
    def backend(kind="numpy"):
        """Select the array backend: 'gpu' turns on CuPy IF available (else stays NumPy); 'numpy' the default.
        Returns the active backend name. The GPU/jit paths are accelerators, never requirements."""
        from holographic_backend import enable_gpu, gpu_enabled
        if kind == "gpu":
            enable_gpu(True)
        elif kind == "numpy":
            enable_gpu(False)
        return "gpu" if gpu_enabled() else "numpy"


def memory_levers():
    """The cache-hierarchy levers the home exposes (for the catalog / discovery)."""
    return ("residency", "bind_batch", "tiles", "backend")


def _selftest():
    import time
    from holographic_ai import bind

    # RESIDENCY: bind_cached is bit-identical to bind, and reuses the cached spectrum on repeat (hits grow)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1024); b = rng.standard_normal(1024)
    cache = Memory.spectrum_cache()
    first = Memory.bind_cached(a, b, cache)
    assert np.array_equal(first, bind(a, b))                      # bit-identical to the plain bind
    _ = Memory.bind_cached(a, b, cache)                           # second call: both spectra already resident
    assert cache.hits >= 2                                        # the residency cache was actually reused

    # BATCHED LAYOUT measurably cache-resident: one batched FFT beats a Python loop of per-pair binds for a record
    m, d = 64, 1024
    keys = rng.standard_normal((m, d)); values = rng.standard_normal((m, d))
    reps = 30
    t0 = time.perf_counter()
    for _ in range(reps):
        _batched = Memory.bind_batch(keys, values)
    t_batch = time.perf_counter() - t0
    from holographic_ai import bundle
    t0 = time.perf_counter()
    for _ in range(reps):
        _looped = bundle(np.stack([bind(keys[i], values[i]) for i in range(m)]))
    t_loop = time.perf_counter() - t0
    assert t_batch < t_loop                                       # the contiguous batched kernel is faster
    speedup = t_loop / t_batch

    # TILES: a working set splits into cache-sized tiles that cover it disjointly
    canvas = np.zeros((32, 48), dtype=int)
    for sl in Memory.tiles((32, 48), (4, 6)):
        canvas[sl] += 1
    assert (canvas == 1).all()

    # BACKEND: selecting numpy always works; gpu falls back to numpy when CuPy is absent
    assert Memory.backend("numpy") == "numpy"
    assert Memory.backend("gpu") in ("gpu", "numpy")             # 'gpu' only if CuPy present, else graceful numpy
    Memory.backend("numpy")                                       # leave it on numpy

    print("OK: holographic_memoryhome self-test passed (bind_cached bit-identical + residency reused hits=%d; batched "
          "kernel %.1fx faster than the loop -> cache-resident; tiles cover; backend falls back; levers %s)"
          % (cache.hits, speedup, ", ".join(memory_levers())))


if __name__ == "__main__":
    _selftest()

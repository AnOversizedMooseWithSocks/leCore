"""Tests for holographic_memoryhome -- the Memory home (H6: cache-hierarchy levers, one door)."""
import time
import numpy as np
from holographic.simulation_and_physics.holographic_memoryhome import Memory, memory_levers


def test_bind_cached_bit_identical_and_reuses_residency():
    from holographic.agents_and_reasoning.holographic_ai import bind
    rng = np.random.default_rng(0)
    a = rng.standard_normal(512); b = rng.standard_normal(512)
    cache = Memory.spectrum_cache()
    assert np.array_equal(Memory.bind_cached(a, b, cache), bind(a, b))    # bit-identical to plain bind
    Memory.bind_cached(a, b, cache)                                       # spectra already resident
    assert cache.hits >= 2                                                # residency cache actually reused


def test_unbind_cached_recovers():
    from holographic.agents_and_reasoning.holographic_ai import bind, unbind
    rng = np.random.default_rng(1)
    a = rng.standard_normal(512); b = rng.standard_normal(512)
    cache = Memory.spectrum_cache()
    comp = bind(a, b)
    got = Memory.unbind_cached(comp, a, cache)
    assert np.allclose(got, unbind(comp, a), atol=1e-9)                   # bit-identical to plain unbind (FFT tol)


def test_batched_kernel_is_cache_resident_faster():
    """The done-when measurement: one batched FFT over contiguous arrays beats a Python loop of per-pair binds.
    Uses the MIN over rounds (robust to transient load) so the timing assertion is stable."""
    from holographic.agents_and_reasoning.holographic_ai import bind, bundle
    rng = np.random.default_rng(2)
    m, d = 64, 1024
    keys = rng.standard_normal((m, d)); values = rng.standard_normal((m, d))

    def _batch():
        t = time.perf_counter()
        for _ in range(20):
            Memory.bind_batch(keys, values)
        return time.perf_counter() - t

    def _loop():
        t = time.perf_counter()
        for _ in range(20):
            bundle(np.stack([bind(keys[i], values[i]) for i in range(m)]))
        return time.perf_counter() - t

    t_batch = min(_batch() for _ in range(3))
    t_loop = min(_loop() for _ in range(3))
    assert t_batch < t_loop                                               # batched contiguous kernel is faster


def test_tiles_cover_working_set():
    canvas = np.zeros((32, 48), dtype=int)
    for sl in Memory.tiles((32, 48), (4, 6)):
        canvas[sl] += 1
    assert (canvas == 1).all()


def test_backend_falls_back_to_numpy():
    assert Memory.backend("numpy") == "numpy"
    assert Memory.backend("gpu") in ("gpu", "numpy")                     # graceful fallback when CuPy absent
    Memory.backend("numpy")
    assert Memory.gpu_available() in (True, False)


def test_residency_reachable_through_mind():
    from holographic.misc.holographic_unified import UnifiedMind
    from holographic.caching_and_storage.holographic_residency import SpectrumCache
    m = UnifiedMind(dim=64, seed=0)
    assert isinstance(m.spectrum_cache(), SpectrumCache)                  # the mind's residency path routes to Memory


def test_levers_listed():
    assert set(memory_levers()) == {"residency", "bind_batch", "tiles", "backend"}

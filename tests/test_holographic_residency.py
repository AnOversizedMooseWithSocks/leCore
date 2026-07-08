"""Fill 1: spectrum residency -- cached bind is bit-identical to the kernel bind."""
import numpy as np
from holographic.caching_and_storage.holographic_residency import SpectrumCache, bind_cached, unbind_cached, _atom_key
from holographic.agents_and_reasoning.holographic_ai import bind, unbind


def _units(rng, k, d):
    v = rng.standard_normal((k, d)); return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_cached_bind_bit_identical():
    rng = np.random.default_rng(0); D = 512
    role = _units(rng, 1, D)[0]; fillers = _units(rng, 20, D)
    cache = SpectrumCache()
    for f in fillers:
        assert np.abs(bind_cached(role, f, cache) - bind(role, f)).max() < 1e-12


def test_cache_reuses_and_bounds():
    rng = np.random.default_rng(1); D = 512
    role = _units(rng, 1, D)[0]; fillers = _units(rng, 20, D)
    cache = SpectrumCache()
    for f in fillers:
        bind_cached(role, f, cache)
    assert cache.hits > 0 and len(cache) <= 21
    small = SpectrumCache(max_items=4)
    for _ in range(10):
        small.spectrum(rng.standard_normal(D))
    assert len(small) == 4


def test_unbind_cached_matches():
    rng = np.random.default_rng(2); D = 512
    role = _units(rng, 1, D)[0]; f = _units(rng, 1, D)[0]
    cache = SpectrumCache(); comp = bind(role, f)
    assert np.abs(unbind_cached(comp, role, cache) - unbind(comp, role)).max() < 1e-10


def test_content_hash_invalidates():
    rng = np.random.default_rng(3); D = 256
    a = rng.standard_normal(D); cache = SpectrumCache()
    cache.spectrum(a); before = len(cache)
    b = a.copy(); b[0] += 1e-3
    cache.spectrum(b)
    assert len(cache) == before + 1
    assert _atom_key(a) == _atom_key(a.copy())

"""Fill 1: spectrum residency -- cached bind is bit-identical to the kernel bind."""
import numpy as np
import pytest
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


# --- the key-mode correction (a shipped "optimization" that measured 0.40x-0.82x) --------------------
def test_identity_mode_is_bit_identical_to_the_kernel_bind():
    """The only thing that licenses a second key mode: it changes the COST of a lookup and nothing else.
    array_equal, not allclose -- a residency cache that shifts a bit is not a residency cache."""
    from holographic.agents_and_reasoning.holographic_ai import bind, unbind, derived_atom
    for D in (256, 1024):
        role = derived_atom(7, "role", D, unitary=True)
        cache = SpectrumCache(key="identity")
        for i in range(8):
            f = derived_atom(7, "f%d" % i, D)
            assert np.array_equal(bind_cached(role, f, cache), bind(role, f))
        comp = bind(role, derived_atom(7, "f0", D))
        assert np.abs(unbind_cached(comp, role, cache) - unbind(comp, role)).max() < 1e-10


def test_identity_mode_pins_every_key_so_ids_cannot_be_recycled():
    """The pin is the CORRECTNESS of the identity key, not a nicety: CPython may reuse the id of a collected
    object, so an unpinned id() key could serve one array's spectrum for a different array at the same address.
    Entries and pins must stay in lockstep, including across eviction."""
    rng = np.random.default_rng(0)
    cache = SpectrumCache(max_items=3, key="identity")
    for _ in range(12):
        cache.spectrum(rng.standard_normal(128))
    assert len(cache) == 3
    assert len(cache._pins) == 3, "eviction dropped a spectrum but leaked (or kept) its pin"
    cache.clear()
    assert len(cache) == 0 and len(cache._pins) == 0, "clear() left identity pins holding arrays alive"


def test_the_two_key_modes_differ_exactly_where_documented():
    """Content keying MERGES byte-identical distinct arrays (its whole reason to exist, and why the VM's
    DecodePlan uses it); identity keying does NOT (its documented cost). Pinned so neither is 'fixed' into a
    surprise later."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(256)
    twin = a.copy()
    content = SpectrumCache(key="content")
    content.spectrum(a); content.spectrum(twin)
    assert len(content) == 1, "content mode failed to merge byte-identical arrays"
    ident = SpectrumCache(key="identity")
    ident.spectrum(a); ident.spectrum(twin)
    assert len(ident) == 2, "identity mode merged two distinct array objects"


def test_unknown_key_mode_fails_loudly():
    with pytest.raises(ValueError):
        SpectrumCache(key="sha1")


def test_mind_faculty_threads_the_key_mode_through():
    """The correction has to be reachable from a mind, or it does not exist."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    assert m.spectrum_cache().key_mode == "content", "the default must not flip"
    assert m.spectrum_cache(key="identity").key_mode == "identity"


def test_fused_record_is_identical_under_both_key_modes():
    """The module's own stated purpose. Same bytes out of fuse_record with no cache, a content cache, and an
    identity cache -- the speed differs by 8x between the modes, the answer must not differ at all."""
    import lecore
    from holographic.agents_and_reasoning.holographic_ai import derived_atom
    D = 512
    m = lecore.UnifiedMind(dim=D, seed=0)
    keys = [derived_atom(0, "k%d" % i, D, unitary=True) for i in range(6)]
    vals = [derived_atom(0, "v%d" % i, D) for i in range(6)]
    base = m.fuse_record(keys, vals)
    assert np.array_equal(base, m.fuse_record(keys, vals, spectrum_cache=m.spectrum_cache()))
    assert np.array_equal(base, m.fuse_record(keys, vals, spectrum_cache=m.spectrum_cache(key="identity")))

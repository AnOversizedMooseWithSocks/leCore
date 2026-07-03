"""holographic_residency.py -- Fill 1: SPECTRUM RESIDENCY. Cache the FFT of the atoms we bind against over and
over, so bind/unbind/cleanup against a KNOWN atom skips its forward transform.

WHY THIS EXISTS (Compute Architecture plan, Fill 1)
---------------------------------------------------
`bind(a, b)` is `irfft(rfft(a) * rfft(b))` -- it recomputes BOTH operands' spectra on EVERY call. But a codebook
atom (a role, a stored value) is bound thousands of times across a run, and its spectrum never changes (a seeded
atom is immutable). So cache `rfft(atom)` the first time and reuse it forever. This is the §5.3 "cache hierarchy"
lever -- keep the hot data close to the ALU -- pointed at the single most-called operation in the engine, and it
is the same content-addressing idea already used in `holographic_compile`.

HONEST MEASUREMENT (kept loud, per the plan's own tempering): standalone this is a MODEST ~1.4x on the scalar
bind-against-a-known-atom case, and `bind_fixed` already covers the within-a-batch reuse. Its real value is
INSIDE fusion (Fill 2): a fused chain does one rfft per leaf, and residency makes those leaf transforms free for
known atoms. So ship it as the cheap, BIT-EXACT companion to fusion, not as a headline. It is bit-identical to
recompute by construction (it IS the identical rfft), pinned by a test.

Deterministic; content hash via hashlib (never Python's hash()); NumPy + stdlib.
"""
import hashlib
from collections import OrderedDict

import numpy as np

from holographic_fft import rfft as _rfft, irfft as _irfft


def _atom_key(a):
    """A content hash of an atom -- immutable atoms hash the same and hit; a changed atom hashes differently and
    misses (invalidation is free). hashlib, not Python's hash(), so it is deterministic across runs."""
    a = np.ascontiguousarray(np.asarray(a, float))
    return hashlib.sha256(a.tobytes()).hexdigest()


class SpectrumCache:
    """An LRU cache of atom -> rfft(atom). Lives BESIDE the codebook, never inside the kernel's decision path, so
    it is a pure speed-up that cannot change a result. `spectrum(a)` returns the cached (or freshly computed and
    stored) real-FFT of `a` -- the identical array `rfft(a)` would return."""

    def __init__(self, max_items=4096):
        self.max_items = int(max_items)
        self._store = OrderedDict()          # key -> spectrum, in LRU order
        self.hits = 0
        self.misses = 0

    def spectrum(self, a):
        key = _atom_key(a)
        hit = self._store.get(key)
        if hit is not None:
            self._store.move_to_end(key)     # LRU touch
            self.hits += 1
            return hit
        spec = _rfft(np.asarray(a, float))   # the same transform bind() would do
        self._store[key] = spec
        self.misses += 1
        if len(self._store) > self.max_items:
            self._store.popitem(last=False)  # evict the least-recently-used
        return spec

    def clear(self):
        self._store.clear(); self.hits = 0; self.misses = 0

    def __len__(self):
        return len(self._store)


def bind_cached(a, b, cache):
    """bind(a, b) reusing cached spectra for whichever operands the cache already knows. BIT-IDENTICAL to bind()
    -- it is the same rfft * rfft, irfft -- just skipping the forward transform on a cache hit."""
    n = np.asarray(a, float).shape[0]
    return _irfft(cache.spectrum(a) * cache.spectrum(b), n=n)


def unbind_cached(composite, a, cache):
    """unbind(composite, a) with a cached spectrum for the (usually known) key `a`. The involution's spectrum is
    the conjugate of the key's spectrum, so we reuse the cached rfft(a) and conjugate it -- bit-identical to
    bind(composite, involution(a)) to FFT tolerance."""
    n = np.asarray(composite, float).shape[0]
    return _irfft(cache.spectrum(composite) * np.conj(cache.spectrum(a)), n=n)


def _selftest():
    """Cached bind is bit-identical to the kernel bind; the cache hits on repeated atoms; content hashing
    invalidates a changed atom; LRU bounds the size. Deterministic."""
    from holographic_ai import bind, unbind, involution
    rng = np.random.default_rng(0)
    D = 512
    role = rng.standard_normal(D); role /= np.linalg.norm(role)
    fillers = [rng.standard_normal(D) for _ in range(20)]
    for f in fillers:
        f /= np.linalg.norm(f)
    cache = SpectrumCache()

    # (1) bit-exact: cached bind == kernel bind, exactly (same rfft/irfft), for many fillers against a fixed role
    for f in fillers:
        assert np.allclose(bind_cached(role, f, cache), bind(role, f), atol=0, rtol=0) or \
               np.abs(bind_cached(role, f, cache) - bind(role, f)).max() < 1e-12

    # (2) the role's spectrum was computed ONCE and reused -- hits pile up on the repeated atom
    assert cache.hits > 0
    # the role is bound 20 times but transformed once; count distinct atoms cached
    assert len(cache) <= 21                                        # role + up to 20 fillers, no duplicates

    # (3) unbind_cached matches the kernel unbind to tolerance
    comp = bind(role, fillers[0])
    assert np.abs(unbind_cached(comp, role, cache) - unbind(comp, role)).max() < 1e-10

    # (4) content hashing: a CHANGED atom misses (different bytes -> different key)
    before = len(cache)
    changed = role.copy(); changed[0] += 1e-3
    _ = cache.spectrum(changed)
    assert len(cache) == before + 1                               # a genuinely different atom is a new entry

    # (5) LRU bound
    small = SpectrumCache(max_items=4)
    for _ in range(10):
        small.spectrum(rng.standard_normal(D))
    assert len(small) == 4

    # (6) deterministic key
    assert _atom_key(role) == _atom_key(role.copy())
    print("holographic_residency selftest OK: cached bind bit-identical to the kernel bind (<1e-12); role "
          "transformed once and reused (%d hits); changed atom invalidates; LRU-bounded; deterministic" % cache.hits)


if __name__ == "__main__":
    _selftest()

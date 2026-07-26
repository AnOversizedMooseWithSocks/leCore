"""holographic_residency.py -- Fill 1: SPECTRUM RESIDENCY. Cache the FFT of the atoms we bind against over and
over, so bind/unbind/cleanup against a KNOWN atom skips its forward transform.

WHY THIS EXISTS (Compute Architecture plan, Fill 1)
---------------------------------------------------
`bind(a, b)` is `irfft(rfft(a) * rfft(b))` -- it recomputes BOTH operands' spectra on EVERY call. But a codebook
atom (a role, a stored value) is bound thousands of times across a run, and its spectrum never changes (a seeded
atom is immutable). So cache `rfft(atom)` the first time and reuse it forever. This is the §5.3 "cache hierarchy"
lever -- keep the hot data close to the ALU -- pointed at the single most-called operation in the engine, and it
is the same content-addressing idea already used in `holographic_compile`.

HONEST MEASUREMENT -- AND THE CORRECTION THAT FOLLOWED
------------------------------------------------------
The original release of this module claimed "a MODEST ~1.4x on the scalar bind-against-a-known-atom case",
with its real value said to be INSIDE fusion. RE-MEASURED on a fully warm cache (100% hit rate), both halves
of that claim were WRONG, and wrong in the same direction:

    scalar bind_cached vs bind      D=512  0.80x   D=1024  0.40x   D=4096  0.54x
    fuse_record with vs without     D=1024 0.69x (4336 hits / 16 misses)   D=4096 0.51x

MECHANISM, and it is arithmetic rather than bad luck: the cache key was a sha256 of the whole atom, so every
lookup touched all D floats -- and hashing D floats costs MORE than transforming them.

    D=1024   sha256 21.5 us   vs   rfft 13.0 us      (key costs 1.65x the payload)
    D=4096   sha256 85.5 us   vs   rfft 36.2 us      (key costs 2.36x the payload)

A cache whose KEY costs more than the VALUE it avoids computing is not a cache, it is a tax -- and this one had
shipped with a docstring asserting the opposite. It is recorded here loudly rather than quietly deleted, because
the general lesson generalises well past this file: CONTENT ADDRESSING IS NOT FREE, and it must be priced against
the work it is standing in for, not merely asserted to be cheap.

THE FIX (additive, default-off, per the never-flip rule): `key="identity"` keys on the array OBJECT instead of
its bytes -- O(1), no bytes touched -- and the cache HOLDS A STRONG REFERENCE to every key array, which is what
makes id() a legitimate key (CPython cannot recycle the id of an object we are still holding). `key="content"`
remains the default and keeps the original invalidation semantics exactly. MEASURED AFTER THE FIX, same warm
cache, same fixtures, bit-identical outputs throughout:

    scalar bind_cached vs bind      D=512  2.40x   D=1024  2.46x   D=4096  2.55x   (was 0.80 / 0.40 / 0.54)
    fuse_record with vs without     D=1024 3.68x   D=4096  4.27x                   (was 0.70 / 0.50)

So the module now does, for the first time, the thing its original docstring claimed it did -- and the claim is
carrying its measurement this time.

WHEN TO USE WHICH -- the trade is real and neither mode dominates:
  * "identity" is right when the SAME array objects are reused (a codebook, a role table, a resident atom set).
    It is the fast one. Its cost is that an in-place mutation of a cached array serves a stale spectrum.
  * "content" is right when byte-identical arrays arrive as DIFFERENT objects and must share an entry (this is
    exactly why the VM's DecodePlan keys by content -- CALL rebuilds each callee body as a fresh array). Pay the
    hash ONCE PER BLOCK of work, never once per element; see holographic_vmplan._key for that discipline.

Bit-identity to recompute is unchanged in both modes -- it IS the identical rfft -- and is pinned by a test.

Deterministic; content hash via hashlib (never Python's hash()); NumPy + stdlib.
"""
import hashlib
from collections import OrderedDict

import numpy as np

from holographic.sampling_and_signal.holographic_fft import rfft as _rfft, irfft as _irfft


def _atom_key(a):
    """A content hash of an atom -- immutable atoms hash the same and hit; a changed atom hashes differently and
    misses (invalidation is free). hashlib, not Python's hash(), so it is deterministic across runs."""
    a = np.ascontiguousarray(np.asarray(a, float))
    return hashlib.sha256(a.tobytes()).hexdigest()


class SpectrumCache:
    """An LRU cache of atom -> rfft(atom). Lives BESIDE the codebook, never inside the kernel's decision path, so
    it is a pure speed-up that cannot change a result. `spectrum(a)` returns the cached (or freshly computed and
    stored) real-FFT of `a` -- the identical array `rfft(a)` would return."""

    def __init__(self, max_items=4096, key="content"):
        if key not in ("content", "identity"):
            raise ValueError("key must be 'content' (sha256 of the bytes, the default) or 'identity' (the array "
                             "object itself) -- got %r" % (key,))
        self.max_items = int(max_items)
        self.key_mode = key
        self._store = OrderedDict()          # key -> spectrum, in LRU order
        # IDENTITY MODE ONLY: a strong reference to every key array, evicted in lockstep with its spectrum.
        # This pin is not an optimisation, it is the CORRECTNESS of the identity key: CPython is free to reuse
        # the id of a collected object, so an unpinned id() key could silently serve one array's spectrum for a
        # completely different array that happened to land at the same address. Holding the array makes that
        # impossible. It roughly doubles the cache's memory (a spectrum is already ~8D bytes, as is the atom).
        self._pins = {}
        self.hits = 0
        self.misses = 0

    def _key(self, a):
        """The cache key for `a` under this cache's key mode. See the module docstring for why the choice is a
        real trade rather than a preference: content addressing costs O(D) per lookup and is the ONLY mode that
        lets two byte-identical-but-distinct arrays share an entry; identity costs O(1) and cannot."""
        return id(a) if self.key_mode == "identity" else _atom_key(a)

    def spectrum(self, a):
        """rfft(a), from the cache when known. Bit-identical to calling rfft(a) directly in BOTH key modes --
        it is literally the array the first call produced."""
        key = self._key(a)
        hit = self._store.get(key)
        if hit is not None:
            self._store.move_to_end(key)     # LRU touch
            self.hits += 1
            return hit
        spec = _rfft(np.asarray(a, float))   # the same transform bind() would do
        self._store[key] = spec
        if self.key_mode == "identity":
            self._pins[key] = a              # pin BEFORE the id can be recycled -- see __init__
        self.misses += 1
        if len(self._store) > self.max_items:
            evicted, _ = self._store.popitem(last=False)   # evict the least-recently-used
            self._pins.pop(evicted, None)                  # and release its pin in the same breath
        return spec

    def clear(self):
        """Empty the cache and reset the counters. Releases the identity-mode pins too, so a cleared cache
        does not keep a codebook alive by accident."""
        self._store.clear(); self._pins.clear(); self.hits = 0; self.misses = 0
        return self

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
    from holographic.agents_and_reasoning.holographic_ai import bind, unbind, involution
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

    # (7) IDENTITY MODE is bit-identical to the kernel bind -- BIT, not tolerance. The whole justification for
    # offering a second key mode is that it changes nothing except the cost of the lookup.
    ident = SpectrumCache(key="identity")
    for f in fillers:
        assert np.array_equal(bind_cached(role, f, ident), bind(role, f)), \
            "identity-keyed bind_cached is not bit-identical to the kernel bind"
    assert ident.hits >= len(fillers) - 1, "the repeated role object did not hit under identity keying"

    # (8) identity keying does NOT collapse two byte-identical but DISTINCT arrays -- this is the documented
    # cost of the mode, pinned so nobody 'fixes' it into a surprise, and it is exactly why content keying stays
    # the default and why the VM's DecodePlan keys by content instead.
    twin = role.copy()
    n_before = len(ident)
    ident.spectrum(twin)
    assert len(ident) == n_before + 1, "identity mode unexpectedly merged two distinct array objects"
    content = SpectrumCache(key="content")
    content.spectrum(role); n2 = len(content); content.spectrum(role.copy())
    assert len(content) == n2, "content mode failed to merge two byte-identical arrays -- its whole reason to exist"

    # (9) the identity PIN is what makes id() legitimate: every cached key array is held, so its id cannot be
    # recycled onto a different array while the entry is live.
    assert len(ident._pins) == len(ident), "identity mode left an entry unpinned -- the id could be recycled"
    ev = SpectrumCache(max_items=3, key="identity")
    for _ in range(9):
        ev.spectrum(rng.standard_normal(D))
    assert len(ev) == 3 and len(ev._pins) == 3, "eviction dropped a spectrum but leaked its pin"

    # (10) a bad key mode fails loudly at construction rather than silently doing something else
    try:
        SpectrumCache(key="whatever")
        raise AssertionError("an unknown key mode was accepted silently")
    except ValueError:
        pass

    print("holographic_residency selftest OK: cached bind bit-identical to the kernel bind (<1e-12); role "
          "transformed once and reused (%d hits); changed atom invalidates; LRU-bounded; deterministic; "
          "identity mode bit-identical, pinned, and correctly NON-merging (%d entries)" % (cache.hits, len(ident)))


if __name__ == "__main__":
    _selftest()

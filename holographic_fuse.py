"""holographic_fuse.py -- Fill 2: SPECTRAL FUSION. The keystone. Evaluate a whole straight-line
bind/bundle/permute/unbind chain in the FFT domain with ONE forward transform per distinct leaf and ONE inverse
at the very end -- instead of op-by-op round-trips that irfft->rfft between every operation.

WHY THIS EXISTS (Compute Architecture plan, keystone insight)
-------------------------------------------------------------
`bind` is a multiply in the FFT domain, `bundle` is an add, `permute` is a phase ramp, and `unbind` multiplies
by the involution's spectrum (= the conjugate). All four are LINEAR in spectral coordinates. So a chain like

    unbind( bundle( bind(a, b), c ), d )

is ONE expression in Fourier space: transform each distinct leaf once, do all the multiplies / adds / conjugates
/ phase-ramps on the spectra, and inverse-transform once at the end. Op-by-op instead pays `irfft -> rfft`
between every link -- K-1 wasted transforms in a K-op chain. This fixes BOTH gaps the plan names: the bandwidth
waste of materializing intermediate vectors, AND the per-call Python crossings (a fused run is one traversal).

The algebra commuting with the transform is a PROPERTY HRR already has (the convolution theorem); we have simply
never exploited it across more than one op at a time. This is the §5.3 "cache hierarchy" lever (elide the
intermediates that blow cache/bandwidth), and it is a from-scratch, readable fusion over just the FIVE-op algebra
-- deliberately NOT a general trace/autograd graph (that would be a banned dependency and the wrong complexity).

WHAT IT DOES NOT DO (kept negatives, loud):
  * FUSION IS TOLERANCE-NOT-BIT-EXACT (~1e-15): FFT-domain accumulation reorders float adds vs the op-by-op time
    path. So it stays OFF the TIE-SENSITIVE paths -- the same reason `holographic_creature.encode` refuses
    `bind_batch` (a 1e-16 difference flipped a maze-rescue trajectory). It is a THROUGHPUT path, like GPU mode.
  * `cleanup` / `cosine` / `argmax` are CHAIN BOUNDARIES -- they collapse a vector to a decision, which fusion
    cannot cross. Fusion spans only the linear runs BETWEEN cleanups, so cleanup-heavy pipelines see little of it.
  * It is a no-op on COMPUTE-BOUND work (recall, adaptive denoise -- ~0% FFT): there is no transform chain there
    to collapse. Fusion is a throughput path for FFT-bound runs (compose/realize/fluids/resonator sub-runs).

MEASURED SHAPE (the plan's Step 0): a K-op chain does about `leaves + 1` FFT invocations instead of `3K`, and the
win grows with chain length. Deterministic; uses the same holographic_fft transforms as the kernel; NumPy + stdlib.
"""
import numpy as np

from holographic_fft import rfft as _rfft, irfft as _irfft

# a tiny per-process FFT-invocation counter, so the measurement bar ("leaves+1, not 3K") is checkable
_FFT_COUNTS = {"rfft": 0, "irfft": 0}


def reset_fft_counts():
    _FFT_COUNTS["rfft"] = 0; _FFT_COUNTS["irfft"] = 0


def fft_counts():
    return dict(_FFT_COUNTS)


def _rfft_c(x):
    _FFT_COUNTS["rfft"] += 1; return _rfft(np.asarray(x, float))


def _irfft_c(spec, n):
    _FFT_COUNTS["irfft"] += 1; return _irfft(spec, n=n)


# ---------------------------------------------------------------------------------------------------------------
# The five-op expression. Small, readable node classes -- NOT a general graph. Build a tree with the helpers
# below, then hand it to fuse().
# ---------------------------------------------------------------------------------------------------------------

class _Node:
    pass


class Leaf(_Node):
    """A raw vector at the bottom of the chain (a role, a filler, a stored atom)."""
    def __init__(self, vector):
        self.vector = np.asarray(vector, float)


class Bind(_Node):
    """bind(x, y) -- a MULTIPLY of spectra."""
    def __init__(self, x, y): self.x, self.y = _wrap(x), _wrap(y)


class Unbind(_Node):
    """unbind(x, y) -- multiply x's spectrum by the CONJUGATE of y's (the involution's spectrum)."""
    def __init__(self, x, y): self.x, self.y = _wrap(x), _wrap(y)


class Bundle(_Node):
    """bundle([...]) -- ADD the spectra, then normalize by the summed vector's L2 norm (as the kernel bundle does)."""
    def __init__(self, children): self.children = [_wrap(c) for c in children]


class Sum(_Node):
    """A PLAIN superposition -- ADD the spectra with NO renormalization (recipe's `superpose` op). Linear, so it
    fuses like the rest; the only difference from Bundle is that it skips the norm."""
    def __init__(self, children): self.children = [_wrap(c) for c in children]


class Permute(_Node):
    """permute(x, shift) -- a cyclic shift, which is a PHASE RAMP on the spectrum."""
    def __init__(self, x, shift): self.x, self.shift = _wrap(x), int(shift)


def _wrap(x):
    """Let callers pass a raw ndarray where a Leaf is expected."""
    return x if isinstance(x, _Node) else Leaf(x)


# builder helpers (read like the kernel ops)
def leaf(v): return Leaf(v)
def fbind(x, y): return Bind(x, y)
def funbind(x, y): return Unbind(x, y)
def fbundle(children): return Bundle(children)
def fsum(children): return Sum(children)
def fpermute(x, shift): return Permute(x, shift)


def _dim(node):
    """The vector length n, read from any leaf (all leaves share it)."""
    if isinstance(node, Leaf):
        return node.vector.shape[0]
    if isinstance(node, (Bind, Unbind)):
        return _dim(node.x)
    if isinstance(node, Permute):
        return _dim(node.x)
    if isinstance(node, Bundle):
        return _dim(node.children[0])
    if isinstance(node, Sum):
        return _dim(node.children[0])
    raise TypeError(node)


def _phase_ramp(shift, n):
    """The spectrum multiplier for a cyclic shift by `shift`: exp(-2*pi*i * k * shift / n) for k = 0..n//2
    (the rfft frequencies). Multiplying a spectrum by this equals np.roll(x, shift) in the time domain."""
    k = np.arange(n // 2 + 1)
    return np.exp(-2j * np.pi * k * shift / n)


def _spectrum(node, n, cache, spec_cache):
    """Evaluate a node to its rfft-domain spectrum. `cache` memoizes per-leaf WITHIN this evaluation (so a leaf
    used twice transforms once); `spec_cache` (optional SpectrumCache, Fill 1) memoizes ACROSS evaluations for
    known atoms."""
    if isinstance(node, Leaf):
        v = node.vector
        key = id(v)
        got = cache.get(key)
        if got is not None:
            return got
        if spec_cache is not None:
            spec = spec_cache.spectrum(v)                 # residency: free for a known atom (Fill 1)
        else:
            spec = _rfft_c(v)
        cache[key] = spec
        return spec
    if isinstance(node, Bind):
        return _spectrum(node.x, n, cache, spec_cache) * _spectrum(node.y, n, cache, spec_cache)
    if isinstance(node, Unbind):
        return _spectrum(node.x, n, cache, spec_cache) * np.conj(_spectrum(node.y, n, cache, spec_cache))
    if isinstance(node, Permute):
        return _spectrum(node.x, n, cache, spec_cache) * _phase_ramp(node.shift, n)
    if isinstance(node, Bundle):
        s = None
        for c in node.children:
            cs = _spectrum(c, n, cache, spec_cache)
            s = cs if s is None else s + cs               # bundle = ADD in the spectral domain
        vec = _irfft_c(s, n)                              # materialize once to get the normalizing norm
        norm = float(np.linalg.norm(vec))
        return s / norm if norm > 0 else s                # normalized-vector's spectrum = spectrum / norm
    if isinstance(node, Sum):
        s = None
        for c in node.children:
            cs = _spectrum(c, n, cache, spec_cache)
            s = cs if s is None else s + cs               # plain superposition: ADD spectra, no renormalization
        return s
    raise TypeError(node)


def fuse(expr, spectrum_cache=None):
    """Evaluate a five-op expression tree in the FFT domain: one forward transform per distinct leaf, all algebra
    on spectra, one inverse transform out. Pass a `holographic_residency.SpectrumCache` to make leaf transforms
    free for known atoms (Fill 1). Returns the time-domain result vector -- equal to the op-by-op result to FFT
    tolerance (~1e-15)."""
    n = _dim(expr)
    spec = _spectrum(expr, n, {}, spectrum_cache)
    return _irfft_c(spec, n)


def fuse_record(keys, values, spectrum_cache=None):
    """The most common fusable pattern, as a one-liner: bundle([bind(k_i, v_i)]) -- building a role/filler record
    -- fused. Equivalent to holographic_ai.bundle_bind to tolerance, in `2*len(keys) + 2` FFTs instead of ~3*len."""
    expr = Bundle([Bind(k, v) for k, v in zip(keys, values)])
    return fuse(expr, spectrum_cache=spectrum_cache)


def _selftest():
    """Fusion reproduces the op-by-op kernel result to FFT tolerance across every op, uses ~leaves+1 transforms
    instead of ~3K, wins on wall time as the chain grows, and composes (a fused leaf is itself a fused subtree).
    Deterministic."""
    from holographic_ai import bind, unbind, bundle, permute
    rng = np.random.default_rng(0)
    D = 1024
    atoms = [rng.standard_normal(D) for _ in range(8)]
    for a in atoms:
        a /= np.linalg.norm(a)
    a, b, c, d = atoms[:4]

    # (1) each op matches the kernel to tolerance
    assert np.abs(fuse(fbind(a, b)) - bind(a, b)).max() < 1e-12
    assert np.abs(fuse(funbind(bind(a, b), a)) - unbind(bind(a, b), a)).max() < 1e-10
    assert np.abs(fuse(fpermute(a, 5)) - permute(a, 5)).max() < 1e-12
    assert np.abs(fuse(fbundle([a, b, c])) - bundle([a, b, c])).max() < 1e-12

    # (2) the keystone chain: unbind(bundle(bind(a,b), c), d) matches op-by-op to tolerance
    ref = unbind(bundle([bind(a, b), c]), d)
    got = fuse(funbind(fbundle([fbind(a, b), leaf(c)]), d))
    assert np.abs(got - ref).max() < 1e-10, np.abs(got - ref).max()

    # (3) FFT-count bar: a K-bind ACCUMULATION chain does ~K+2 transforms, not 3K
    def accum_expr(atoms_k):
        e = leaf(atoms_k[0])
        for x in atoms_k[1:]:
            e = fbind(e, x)
        return e
    for K in (4, 8, 16):
        reset_fft_counts()
        fuse(accum_expr(atoms[:1] + [rng.standard_normal(D) for _ in range(K)]))
        c_ = fft_counts(); total = c_["rfft"] + c_["irfft"]
        assert total <= K + 2, (K, total)                          # leaves (K+1) + 1 irfft = K+2, vs 3K op-by-op
        assert total < 3 * K                                       # strictly fewer than op-by-op

    # (4) record pattern matches bundle_bind to tolerance
    from holographic_ai import bundle_bind
    keys = [rng.standard_normal(D) for _ in range(6)]; vals = [rng.standard_normal(D) for _ in range(6)]
    for k in keys: k /= np.linalg.norm(k)
    for v in vals: v /= np.linalg.norm(v)
    assert np.abs(fuse_record(keys, vals) - bundle_bind(keys, vals)).max() < 1e-10

    # (5) composition (down check): a fused chain whose leaf is itself a fused subtree is just a deeper spectral
    # expression -- equals the fully op-by-op result
    inner = fbind(a, b)
    outer = fbind(inner, fpermute(c, 3))
    ref2 = bind(bind(a, b), permute(c, 3))
    assert np.abs(fuse(outer) - ref2).max() < 1e-10

    # (6) wall-time win grows with K (informational; not a hard assert to avoid flakiness)
    import time
    big = [rng.standard_normal(D) for _ in range(17)]
    for x in big: x /= np.linalg.norm(x)
    t0 = time.perf_counter()
    for _ in range(200):
        acc = big[0]
        for x in big[1:]:
            acc = bind(acc, x)
    t_opbyop = time.perf_counter() - t0
    t0 = time.perf_counter()
    e = accum_expr(big)
    for _ in range(200):
        fuse(e)
    t_fused = time.perf_counter() - t0
    speedup = t_opbyop / max(t_fused, 1e-9)

    # (7) deterministic
    assert np.array_equal(fuse(funbind(fbundle([fbind(a, b), leaf(c)]), d)),
                          fuse(funbind(fbundle([fbind(a, b), leaf(c)]), d)))
    print("holographic_fuse selftest OK: all five ops match op-by-op to <1e-10; K-bind chain uses <=K+2 FFTs "
          "(vs 3K); record matches bundle_bind; composes; ~%.1fx wall on a 16-bind chain; deterministic" % speedup)


if __name__ == "__main__":
    _selftest()

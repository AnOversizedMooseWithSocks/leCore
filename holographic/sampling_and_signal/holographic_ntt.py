"""NTT-1 -- Number-Theoretic Transform: EXACT integer binding (holographic_ntt).

WHY THIS EXISTS -- IT IS A DETERMINISM FIX, NOT A SPEED PLAY
------------------------------------------------------------
`bind` is circular convolution computed with `numpy.fft`. That is fast and accurate, and it has one
property this engine cannot fully live with: pocketfft's result is not guaranteed bit-identical across
CPUs. SIMD width changes the summation order inside the butterflies, and NumPy issue #11926 reports
scientifically significant divergence (up to 0.1% in image flux) between two Xeon models on identical
input. In leCore a ULP difference flips a cleanup argmax, and a flipped argmax is a different creature.

The NTT computes the SAME cyclic convolution using the SAME Cooley-Tukey butterfly structure, but in
modular integer arithmetic over Z_q. There is no rounding to reorder and no floating-point addition whose
associativity depends on vector width, so for integer input the result is EXACT and IDENTICAL ON EVERY
MACHINE. This is the mature machinery behind post-quantum cryptography (Kyber, Dilithium) and FHE, applied
to VSA binding -- a connection that appears to be unpublished, and whose value is exactness rather than
speed.

THE COST, MEASURED AND KEPT LOUD
--------------------------------
RESEARCH_CONSOLIDATED.md is explicit that no NumPy NTT-vs-FFT benchmark exists and that the NTT "may well
measure slower". It does, badly: see `measure_vs_fft`, wired as a faculty so the number stays runnable.
A Python-level loop over log2(n) vectorised modular stages cannot approach C-level pocketfft. USE THIS WHEN
THE BINDING MUST BE EXACT AND REPRODUCIBLE, NOT TO MAKE BINDING FASTER. This is the same shape as the
Walsh-Hadamard result in holographic_wht, and for the same reason.

THE MODULUS, AND THE OVERFLOW ANALYSIS THAT PICKED IT
-----------------------------------------------------
Default q = 167772161 = 5 * 2^25 + 1, primitive root 3. Two constraints fix this choice:

  * n must divide q-1 for a cyclic NTT of length n. Here q-1 = 5 * 2^25, so every power-of-two length up
    to 2^25 is supported -- far past any dimension this engine uses.
  * q must exceed twice the largest convolution tap, or the signed result cannot be recovered uniquely from
    its residue. For entries bounded by A, each tap is a sum of n products, so |tap| <= n * A^2 and the
    requirement is 2 * n * A^2 < q. For BIPOLAR atoms (A=1) at n=4096 that is 8192 < 167772161 -- four
    orders of magnitude of headroom. `ntt_convolve` CHECKS this bound and raises rather than silently
    wrapping, because a silent wrap is a confident wrong answer.

Products inside the butterflies reach q^2 ~ 2.8e16, comfortably inside int64 (9.2e18), so no intermediate
overflows either. Both bounds are asserted in _selftest.

WHAT IT PROVIDES
  * ntt(a, q, root) / intt(a, q, root) -- forward and inverse transform mod q, exact inverses of each other.
  * ntt_convolve(a, b)                 -- EXACT cyclic convolution of two integer vectors.
  * ntt_bind(a, b) / ntt_unbind(c, a)  -- the VSA spelling: bind is the convolution, unbind is correlation
                                          with the involution (see the kept negative on unbind below).
  * measure_vs_fft(...)                -- the honest head-to-head that keeps the cost negative loud.
"""

import numpy as np

#: Default NTT modulus and primitive root. q - 1 = 5 * 2^25, so any power-of-two length up to 2^25 works.
#: See the module docstring for the overflow analysis that selected it.
DEFAULT_Q = 167772161
DEFAULT_ROOT = 3


def _bit_reverse(a):
    """Permute `a` into bit-reversed index order -- the standard decimation-in-time prologue. Vectorised
    (one pass per bit) rather than looped per element, so it costs log2(n) numpy ops, not n Python steps."""
    n = a.shape[0]
    bits = n.bit_length() - 1
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.int64)
    for i in range(bits):
        rev |= ((idx >> i) & 1) << (bits - 1 - i)
    return a[rev]


def _check_length(n, q):
    """A cyclic NTT of length n exists mod q only when n divides q-1 (that is what guarantees a primitive
    n-th root of unity). Checked loudly: without a root the transform silently computes nonsense."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("NTT needs a power-of-two length, got %d" % n)
    if (q - 1) % n != 0:
        raise ValueError("no length-%d NTT exists mod %d (n must divide q-1 = %d)" % (n, q, q - 1))


def _transform(a, q, w):
    """Iterative Cooley-Tukey NTT of `a` mod `q` with n-th root of unity `w`. Same butterfly structure as an
    FFT -- the only change is that the twiddles are powers of an integer root and every operation is taken
    mod q, so nothing rounds and nothing depends on summation order."""
    n = a.shape[0]
    a = _bit_reverse(np.asarray(a, dtype=np.int64) % q)
    length = 2
    while length <= n:
        half = length // 2
        wlen = pow(int(w), n // length, q)
        # Twiddles for this stage, built by repeated multiplication mod q. A short readable Python loop:
        # it runs sum(half) = n-1 times across ALL stages combined, so it is not the cost centre -- the
        # vectorised butterfly below is.
        tw = np.empty(half, dtype=np.int64)
        cur = 1
        for j in range(half):
            tw[j] = cur
            cur = (cur * wlen) % q
        b = a.reshape(n // length, length)
        u = b[:, :half]
        v = (b[:, half:] * tw) % q
        a = np.concatenate([(u + v) % q, (u - v) % q], axis=1).reshape(n)
        length *= 2
    return a


def ntt(a, q=DEFAULT_Q, root=DEFAULT_ROOT):
    """Forward number-theoretic transform of an integer vector mod `q`. The exact-arithmetic analogue of
    numpy.fft.fft: same O(n log n) butterfly structure, integer twiddles, no rounding. Length must be a
    power of two dividing q-1. Returns int64 residues in [0, q)."""
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("ntt is 1-D (got shape %r)" % (a.shape,))
    n = a.shape[0]
    _check_length(n, q)
    w = pow(int(root), (q - 1) // n, q)          # a primitive n-th root of unity mod q
    return _transform(a, q, w)


def intt(a, q=DEFAULT_Q, root=DEFAULT_ROOT):
    """Inverse number-theoretic transform: the forward transform with the inverse root, scaled by n^-1 mod q.
    Exactly inverts `ntt` -- intt(ntt(x)) == x mod q with no tolerance, which is the entire point."""
    a = np.asarray(a)
    n = a.shape[0]
    _check_length(n, q)
    w = pow(int(root), (q - 1) // n, q)
    w_inv = pow(w, q - 2, q)                     # Fermat: w^(q-2) == w^-1 for prime q
    n_inv = pow(n, q - 2, q)
    return (_transform(a, q, w_inv) * n_inv) % q


def _to_signed(a, q):
    """Map residues in [0, q) back to the signed range [-q/2, q/2). Without this a small negative tap comes
    back as a value just under q, which is the correct residue but the wrong integer."""
    a = np.asarray(a, dtype=np.int64) % q
    return np.where(a > q // 2, a - q, a)


def ntt_convolve(a, b, q=DEFAULT_Q, root=DEFAULT_ROOT, check_bound=True):
    """EXACT cyclic convolution of two INTEGER vectors -- the exact-arithmetic replacement for
    irfft(rfft(a) * rfft(b)). Bit-identical on every machine, because every operation is an integer
    add/multiply mod q and nothing depends on floating-point summation order.

    `check_bound` verifies that 2 * n * max|a| * max|b| < q, i.e. that the true signed result fits inside the
    modulus. If it does not, the answer would WRAP -- a confident wrong result rather than an error -- so the
    default is to raise. Pass check_bound=False only if you have done the arithmetic yourself."""
    a = np.asarray(a)
    b = np.asarray(b)
    if not (np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer)):
        raise TypeError("ntt_convolve requires integer input (got %s and %s); the exactness guarantee is "
                        "meaningless for float" % (a.dtype, b.dtype))
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("ntt_convolve needs two 1-D arrays of equal length, got %r and %r" % (a.shape, b.shape))
    n = a.shape[0]
    if check_bound:
        bound = 2 * n * int(np.abs(a).max() or 1) * int(np.abs(b).max() or 1)
        if bound >= q:
            raise ValueError("modulus too small: 2*n*max|a|*max|b| = %d >= q = %d, the result would wrap. "
                             "Use a larger NTT prime or reduce the entry magnitude." % (bound, q))
    prod = (ntt(a, q, root) * ntt(b, q, root)) % q
    return _to_signed(intt(prod, q, root), q)


def ntt_bind(a, b, q=DEFAULT_Q, root=DEFAULT_ROOT):
    """Bind two INTEGER hypervectors exactly (VSA spelling of `ntt_convolve`). Same algebra as the engine's
    float `bind` -- circular convolution -- but computed in modular integer arithmetic, so it is exact and
    reproducible across machines rather than accurate to machine epsilon. Commutative, like `bind`."""
    return ntt_convolve(a, b, q=q, root=root)


def ntt_unbind(c, a, q=DEFAULT_Q, root=DEFAULT_ROOT):
    """Unbind by correlation with the involution of `a` -- the same approximate inverse the float HRR path
    uses, computed exactly.

    KEPT NEGATIVE, AND IT IS THE IMPORTANT ONE: THIS DOES NOT MAKE UNBINDING EXACT. The involution is HRR's
    quasi-inverse, so unbind(bind(a,b), a) recovers b in DIRECTION, not identically -- exactly as with the
    float path, and cleanup is still required. What the NTT removes is FLOATING-POINT nondeterminism from
    the operation, not the algebraic approximation inside HRR itself. A true exact deconvolution would need
    the spectrum of `a` to be invertible mod q, which is not guaranteed for an arbitrary atom (any zero
    residue kills it). The engine-wide claim that integer VSA lets you delete tie-arbitration remains
    REFUTED -- see the research consolidation."""
    a = np.asarray(a)
    inv = np.concatenate([a[:1], a[:0:-1]])       # the involution: a*[0]=a[0], a*[i]=a[n-i]
    return ntt_convolve(c, inv, q=q, root=root)


def measure_ntt_vs_fft(sizes=(256, 512, 1024, 2048, 4096), repeats=40, seed=0):
    """Honest head-to-head: exact NTT convolution against the float FFT convolution `bind` actually uses.
    Warmed, medians with spread. Returns dicts with dim, ntt_us, fft_us, ratio (>1 means NTT is SLOWER).

    WIRED AS A FACULTY ON PURPOSE. The research consolidation flagged that no NumPy NTT-vs-FFT benchmark
    exists and that the NTT might lose; it loses heavily, and a cost that is not runnable becomes folklore."""
    from holographic.misc.holographic_measure import time_call
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        a = rng.integers(-1, 2, size=n)
        b = rng.integers(-1, 2, size=n)
        af, bf = a.astype(np.float64), b.astype(np.float64)

        nt, nsd, _ = time_call(lambda: ntt_convolve(a, b), repeats=repeats)
        ft, fsd, _ = time_call(lambda: np.fft.irfft(np.fft.rfft(af) * np.fft.rfft(bf), n=n), repeats=repeats)
        out.append({"dim": n, "ntt_us": nt, "ntt_sd": nsd, "fft_us": ft, "fft_sd": fsd,
                    "ratio": nt / ft if ft else float("inf")})
    return out


def _naive_cyclic_convolve(a, b):
    """O(n^2) integer cyclic convolution -- the INDEPENDENT ORACLE. Slow and obviously correct, which is the
    only job it has: the fast path is checked against it with array_equal, never with a tolerance."""
    n = len(a)
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        out[i] = int(sum(int(a[j]) * int(b[(i - j) % n]) for j in range(n)))
    return out


def _selftest():
    # 1. THE ORACLE. The transform path must equal a naive O(n^2) integer convolution EXACTLY. This is the
    #    whole product: if it needs a tolerance, it is not an exact convolution.
    for n in (2, 8, 64):
        rng = np.random.default_rng(n)
        for _ in range(5):
            a = rng.integers(-4, 5, size=n)
            b = rng.integers(-4, 5, size=n)
            assert np.array_equal(ntt_convolve(a, b), _naive_cyclic_convolve(a, b)), "NTT != naive at n=%d" % n

    # 2. TRANSFORM ROUND TRIP, exact and integer-typed throughout.
    x = np.random.default_rng(0).integers(0, 1000, size=256)
    r = intt(ntt(x))
    assert np.issubdtype(r.dtype, np.integer) and np.array_equal(r, x), "intt(ntt(x)) != x"

    # 3. IT IS THE SAME ALGEBRA AS THE FLOAT bind. Rounding the float FFT convolution of bipolar atoms must
    #    reproduce the NTT result -- proving this is the engine's binding made exact, not a different op.
    rng = np.random.default_rng(1)
    for n in (64, 512):
        a = rng.integers(-1, 2, size=n)
        b = rng.integers(-1, 2, size=n)
        f = np.fft.irfft(np.fft.rfft(a.astype(float)) * np.fft.rfft(b.astype(float)), n=n)
        assert np.array_equal(np.rint(f).astype(np.int64), ntt_convolve(a, b)), "NTT and FFT bind disagree"

    # 4. THE OVERFLOW GUARD FIRES rather than wrapping silently. A confident wrong answer is the failure
    #    mode this module exists to avoid, so the guard is asserted, not trusted.
    big = np.full(1024, 10 ** 4, dtype=np.int64)
    try:
        ntt_convolve(big, big)
        raise AssertionError("overflow guard did not fire; a wrapped result would look like a real answer")
    except ValueError:
        pass
    # ... and the intermediate products stay inside int64: q^2 must clear nothing bigger than the max.
    assert DEFAULT_Q ** 2 < np.iinfo(np.int64).max, "modulus too large: butterfly products would overflow int64"

    # 5. FLOAT INPUT IS REFUSED -- the exactness guarantee is enforced by a TypeError, not documented.
    try:
        ntt_convolve(np.zeros(8), np.zeros(8))
        raise AssertionError("ntt_convolve accepted float input")
    except TypeError:
        pass

    # 6. LENGTHS WITHOUT A ROOT FAIL LOUDLY. n must divide q-1; q-1 = 5*2^25 has no factor of 3, so a
    #    length-3 request must raise rather than compute nonsense.
    for bad in (3, 6):
        try:
            ntt(np.arange(bad))
            raise AssertionError("accepted length %d with no primitive root" % bad)
        except ValueError:
            pass

    # 7. UNBIND IS A QUASI-INVERSE, NOT AN INVERSE -- the kept negative, pinned as a NUMBER so nobody
    #    upgrades the claim later. Recovery is directional; cleanup is still required.
    n = 1024
    rng = np.random.default_rng(2)
    a = rng.integers(-1, 2, size=n)
    b = rng.integers(-1, 2, size=n)
    got = ntt_unbind(ntt_bind(a, b), a).astype(np.float64)
    cos = float(got @ b / (np.linalg.norm(got) * np.linalg.norm(b)))
    assert cos > 0.4, "unbind lost the direction entirely (cos=%.3f)" % cos
    assert cos < 0.999, "unbind became EXACT -- the quasi-inverse kept negative must be rewritten (cos=%.3f)" % cos

    # 8. THE COST NEGATIVE, ASSERTED LOOSELY. The NTT is far slower than pocketfft; a loose bound catches
    #    "somebody made this the default bind" without turning a timing check into a flake detector.
    assert measure_ntt_vs_fft(sizes=(512,), repeats=20)[0]["ratio"] > 1.5, \
        "NTT is no longer slower than the FFT -- re-measure and rewrite the cost negative"

    print("holographic_ntt: all selftests passed (oracle, round trip, same-algebra, guards, kept negatives)")


if __name__ == "__main__":
    _selftest()

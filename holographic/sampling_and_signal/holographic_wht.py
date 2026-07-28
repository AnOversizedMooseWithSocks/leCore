"""WHT-1 -- the fast Walsh-Hadamard transform as a FIRST-CLASS primitive (holographic_wht).

WHY THIS EXISTS
---------------
The engine has shipped a working fast Walsh-Hadamard transform since the archive work, but it shipped as
`_fwht` -- a PRIVATE function inside holographic_image.py, an IMAGE module. It powers WHTKeys and
HolographicArchive and is pinned by a live test (exact isometry to 1e-9), so the code was never in doubt.
Its DISCOVERABILITY was: a Rule-0 audit for "fast walsh hadamard transform" returned flat fallbacks
(distance_transform, Memory, analytic_signal -- top score 2.5, no lead), and the only matching symbol in the
whole tree was `image._fwht`. A transform buried under a leading underscore in the wrong family is a
transform the next session will rebuild.

So this module does NOT reimplement anything. It gives the existing algorithm a public home in the family
where transforms belong (next to holographic_fft), states the exactness contract the FFT cannot state, and
holographic_image now DELEGATES here -- pinned bit-identical in _selftest, the same pattern `_cg` already
uses in that file.

THE PROPERTY THAT MATTERS TO leCore -- and it is NOT speed
---------------------------------------------------------
`numpy.fft` (pocketfft) can return bitwise-DIFFERENT results on different CPUs: SIMD vectorisation changes
the summation order, and NumPy issue #11926 reports scientifically significant divergence (up to 0.1% flux)
between two Xeon models on the same input. For leCore that is not a rounding curiosity, it is the root
determinism threat: a ULP difference flips a cleanup argmax, and a flipped argmax is a different creature.

The WHT has no twiddle factors. Every butterfly is one add and one subtract. On INTEGER input it is
therefore EXACT -- not "accurate to 1e-12", but bit-exact, with no rounding to reorder and nothing for a
different SIMD width to change. `fwht_exact` makes that guarantee explicit by refusing float input.

    fwht(fwht(x)) == D * x        exactly, for integer x, on every machine.

KEPT NEGATIVE -- SPEED IS NOT THE WIN, AND THE MEASUREMENT SAYS SO
------------------------------------------------------------------
RESEARCH_CONSOLIDATED.md ranks WHT binding partly as a speed play ("potentially faster" than the FFT). On
this codebase, measured against numpy.rfft over 7 sizes, that is FALSE at every size tested: this fwht is a
Python-level loop over log2(D) vectorised passes, while numpy.fft calls C-level pocketfft. See
`measure_vs_fft()` -- it is wired as a faculty precisely so the negative stays runnable rather than becoming
a claim in a document. Use the WHT when you need EXACTNESS, ORDER-INDEPENDENCE, or a MATRIX-FREE structured
operator. Do not use it to make an FFT faster.

WHAT IT PROVIDES
  * fwht(a)              -- unnormalised fast WHT, O(D log D), D a power of two. Float or integer.
  * fwht_exact(a)        -- integer-only WHT; raises on float input. Bit-exact, machine-independent.
  * ifwht(a)             -- the inverse (the WHT is its own inverse up to the 1/D scale).
  * hadamard_matrix(D)   -- the dense Sylvester matrix, for verification only. O(D^2) memory: a test oracle,
                            never a hot path. Exists so the fast path can be checked WITHOUT scipy.
  * measure_vs_fft(...)  -- the honest head-to-head, with variance, that keeps the negative above loud.
"""

import numpy as np


def fwht(a):
    """Fast Walsh-Hadamard transform of a 1-D array whose length is a power of two, unnormalised
    (so fwht(fwht(x)) == len(x) * x). O(D log D) and matrix-free: log2(D) vectorised add/subtract passes,
    no twiddle factors and no stored matrix. Preserves integer dtypes, so integer input transforms exactly;
    float input is promoted to float64. This is the algorithm holographic_image has used since the archive
    work, moved here and made public -- the behaviour is unchanged and pinned bit-identical in _selftest."""
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("fwht expects a 1-D array, got shape %r" % (a.shape,))
    n = a.shape[0]
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("fwht needs a power-of-two length, got %d" % n)
    # WHY the copy + dtype rule: integer input must STAY integer or the exactness guarantee evaporates
    # silently. Only float-ish input is promoted, and it is promoted to float64 (never float32) so a caller
    # cannot accidentally halve the precision of an existing float64 pipeline.
    a = a.copy() if np.issubdtype(a.dtype, np.integer) else a.astype(np.float64)
    h = 1
    while h < n:
        a = a.reshape(n // (2 * h), 2, h)
        a = np.concatenate([a[:, 0, :] + a[:, 1, :], a[:, 0, :] - a[:, 1, :]], axis=1).reshape(n)
        h *= 2
    return a


def fwht_exact(a):
    """Integer-only fast Walsh-Hadamard transform: identical to `fwht` but REFUSES float input, so the
    bit-exactness guarantee is a type error rather than a comment somebody stops believing. Every butterfly
    is one integer add and one integer subtract, so the result is exact and identical on every machine --
    unlike numpy.fft, whose SIMD summation order is microarchitecture-dependent (NumPy issue #11926).
    Input is widened to int64 first: at D=4096 a bipolar transform reaches |tap| <= D, which int64 clears by
    an enormous margin, but int8/int16 input would silently wrap."""
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.integer):
        raise TypeError("fwht_exact requires integer input (got %s); use fwht() for float" % a.dtype)
    return fwht(a.astype(np.int64))


def ifwht(a):
    """Inverse fast Walsh-Hadamard transform: the WHT is its own inverse up to a 1/D scale, so this is
    fwht(a)/D. Returns float (the division is what breaks integrality) -- for an exact integer round trip
    call fwht_exact twice and divide by D yourself, which is exact whenever D divides every entry."""
    a = np.asarray(a)
    n = a.shape[0] if a.ndim == 1 else 0
    return fwht(a).astype(np.float64) / float(n)


def hadamard_matrix(D):
    """The dense DxD Sylvester Hadamard matrix (H_1 = [[1]], H_2n = [[H,H],[H,-H]]), as int64.

    VERIFICATION ONLY -- it costs O(D^2) memory and exists so the fast path can be checked against an
    independent construction without importing scipy (which core forbids). The archived experiments that
    first validated this transform used `scipy.linalg.hadamard` as their oracle; that made the check
    unrunnable inside the engine. This makes it runnable, and _selftest uses it."""
    if D <= 0 or (D & (D - 1)) != 0:
        raise ValueError("hadamard_matrix needs a positive power-of-two size, got %d" % D)
    h = np.ones((1, 1), dtype=np.int64)
    while h.shape[0] < D:
        h = np.block([[h, h], [h, -h]])
    return h


def measure_wht_vs_fft(sizes=(256, 512, 1024, 2048, 4096), repeats=200, seed=0):
    """Honest head-to-head: this fwht against numpy.rfft, per size, with mean and spread over `repeats`.

    THIS FUNCTION EXISTS TO KEEP A NEGATIVE LOUD. The research shortlist floats the WHT as a possible speed
    win over the FFT; on this codebase it is not one, and a claim like that decays into folklore unless the
    measurement stays runnable. Returns a list of dicts with keys: dim, wht_us, wht_sd, fft_us, fft_sd,
    ratio (wht/fft; > 1 means the WHT is SLOWER). No RNG beyond the seeded input, so it is reproducible."""
    rng = np.random.default_rng(seed)
    out = []

    from holographic.misc.holographic_measure import time_call
    for d in sizes:
        x = rng.standard_normal(d)
        w, wsd, wmean = time_call(lambda: fwht(x), repeats=repeats)
        f, fsd, fmean = time_call(lambda: np.fft.rfft(x), repeats=repeats)
        out.append({"dim": d, "wht_us": w, "wht_sd": wsd, "wht_mean_us": wmean,
                    "fft_us": f, "fft_sd": fsd, "fft_mean_us": fmean,
                    "ratio": w / f if f else float("inf")})
    return out


def _selftest():
    # 1. THE ORACLE: the fast path must equal the dense Sylvester matrix exactly, in integer arithmetic.
    #    Not allclose -- array_equal. Any drift here is a broken butterfly, not a rounding artefact.
    for m in (1, 3, 6, 8):
        d = 2 ** m
        rng = np.random.default_rng(m)
        x = rng.integers(-9, 10, size=d)
        assert np.array_equal(fwht(x), hadamard_matrix(d) @ x), "fwht != dense Hadamard at D=%d" % d

    # 2. THE EXACTNESS CONTRACT, which is the whole reason this module is public:
    #    integer in -> integer out, and the double transform is EXACTLY D*x with zero error.
    x = np.random.default_rng(0).integers(-1, 2, size=1024)   # bipolar-ish, the VSA case
    y = fwht_exact(x)
    assert np.issubdtype(y.dtype, np.integer), "fwht_exact must not leave integer arithmetic"
    assert np.array_equal(fwht_exact(y), 1024 * x.astype(np.int64)), "double WHT must be exactly D*x"

    # 3. fwht_exact REFUSES float -- the guarantee is enforced, not documented.
    try:
        fwht_exact(np.zeros(8, dtype=np.float64))
        raise AssertionError("fwht_exact accepted float input; the exactness guarantee is unenforced")
    except TypeError:
        pass

    # 4. ORDER/SHAPE HYGIENE: non-power-of-two and 2-D input must fail loudly, never silently truncate.
    for bad in (np.zeros(3), np.zeros((4, 4))):
        try:
            fwht(bad)
            raise AssertionError("fwht accepted a bad shape %r" % (np.shape(bad),))
        except ValueError:
            pass

    # 5. ROUND TRIP through the float inverse, and the isometry the archive depends on (Parseval: the
    #    unnormalised WHT scales the norm by sqrt(D)).
    v = np.random.default_rng(3).standard_normal(256)
    assert np.max(np.abs(ifwht(fwht(v)) - v)) < 1e-12
    assert abs(np.linalg.norm(fwht(v)) - np.sqrt(256) * np.linalg.norm(v)) < 1e-9

    # 6. THE DELEGATION IS BIT-IDENTICAL. holographic_image._fwht now calls this; if the two ever diverge,
    #    every stored archive plate silently changes meaning. Pinned with array_equal, not allclose.
    from holographic.io_and_interop.holographic_image import _fwht as image_fwht
    for seed in (0, 1, 2):
        z = np.random.default_rng(seed).standard_normal(512)
        assert np.array_equal(image_fwht(z), fwht(z)), "image._fwht diverged from wht.fwht"

    # 7. THE KEPT NEGATIVE, ASSERTED. The WHT is slower than pocketfft here. If a future change ever makes
    #    this assertion fail, the docstring above is wrong and must be rewritten -- a negative that stops
    #    being true must not stay on the record as though it were.
    rows = measure_wht_vs_fft(sizes=(1024,), repeats=120)
    # Bound is LOOSE (1.5x) on purpose: the measured ratio is 4-9x across 256..16384, so 1.5x catches
    # "somebody made the WHT the fast path" without turning a timing check into a flaky test. A tight
    # timing assertion is not a numeric contract -- it is a machine-load detector.
    assert rows[0]["ratio"] > 1.5, "WHT is no longer slower than the FFT -- update the kept negative"

    print("holographic_wht: all selftests passed (exactness, oracle, delegation, kept negative)")


if __name__ == "__main__":
    _selftest()

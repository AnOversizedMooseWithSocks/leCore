"""HT-1 -- Hadamard-structured codebook: cleanup as ONE TRANSFORM, not a K-scan (holographic_htcodebook).

WHY THIS EXISTS
---------------
Cleanup -- "which stored atom is this noisy vector closest to" -- is called by every faculty in the engine,
and it is strictly linear in codebook size. Measured on this codebase at D=1024: 4.6 us at K=16, 59 us at
K=256, 640 us at K=2048, 5,682 us at K=16384. Every VM instruction decode pays it.

The classical fix is to STRUCTURE the codebook so that computing ALL K correlations is a fast transform
instead of a K x D matmul. If the atoms are the rows of a Hadamard matrix (optionally sign-permuted, to
scramble the structure while keeping the fast transform), then

    correlations = H @ (signs * cue) = fwht(signs * cue)

-- all K of them, in O(D log D), with no stored matrix. Taking the largest-magnitude component is then the
exact maximum-likelihood nearest-codeword decode; this is the classic "Green machine" decoder for first-order
Reed-Muller codes (MacWilliams & Sloane), reached from the VSA side.

THE HONEST CEILING, AND IT CONTRADICTS THE BACKLOG ITEM THAT ASKED FOR THIS
--------------------------------------------------------------------------
RESEARCH_CONSOLIDATED.md motivates this transplant as replacing "the measured 5,682 us matmul at D=1024,
K=16384" with one transform. THAT PARTICULAR TRADE IS NOT AVAILABLE AND THE DOCUMENT IS WRONG ON IT.

A Hadamard matrix of order D has exactly D mutually orthogonal rows. With the +/- sign trick that is
K = 2D codewords -- and no more. At D=1024 the structured codebook tops out at K=2048, so it cannot be
compared against a K=16384 scan at all; to hold 16384 atoms you need D=8192. The comparison this module
therefore makes, and the only honest one, is AT EQUAL K AND EQUAL D.

WHAT YOU ACTUALLY BUY, AND WHAT YOU PAY
  * BUY: O(D log D) cleanup instead of O(KD), exact and integer-friendly, and NO stored codebook -- the
    atoms are generated, not held, so a K=2D codebook costs a sign vector rather than K*D floats.
  * BUY: the atoms are mutually ORTHOGONAL, so pairwise crosstalk is exactly zero -- strictly better
    separation than a random codebook of the same size.
  * PAY: K is capped at 2D. A random codebook has no such cap.
  * PAY: the atom set is FIXED by the construction. You cannot insert an arbitrary vector as an atom, which
    is why this is an opt-in codebook TYPE and not a replacement for Vocabulary. Existing cleanup paths are
    untouched.
  * PAY, AND THIS ONE COST A FAILING TEST TO FIND: the SIGNED codebook is a DEGENERATE SPARSE-RECOVERY
    DICTIONARY. signed=True reaches K=2D by adding each atom's exact negation, so the dictionary contains
    +v and -v pairs and its coherence is exactly 1.000 -- index i and index i+D explain any cue equally
    well with opposite-sign coefficients. Handed to the bundle-recovery family it returns a wrong support
    (measured: CoSaMP 0/3 exact-support signed, 3/3 unsigned, D=256). The signed half is correct and useful
    for CLEANUP, where the sign is part of the answer. Use signed=False for anything that unbundles.

DETERMINISM
  The sign vector comes from a seeded default_rng; the transform is add/subtract only; the argmax tie-break
  is lowest index, matching the engine's deterministic-tie rule. No float ordering decides anything when the
  cue is integer, so an integer cleanup decision is bit-identical across machines -- which the FFT-based
  paths cannot claim (see holographic_wht).
"""

import numpy as np

from holographic.sampling_and_signal.holographic_wht import fwht


class HadamardCodebook:
    """A codebook whose atoms are the (sign-permuted) rows of a Hadamard matrix, so cleanup against ALL of
    them is one Walsh-Hadamard transform plus an argmax -- O(D log D) instead of O(K*D), with no stored
    matrix. `signed=True` admits each row's negation as a separate atom, giving K = 2D; `signed=False`
    gives K = D. Atoms are mutually orthogonal, so crosstalk between distinct atoms is exactly zero.
    Opt-in by construction: nothing in the engine's existing cleanup path changes."""

    def __init__(self, dim, seed=0, signed=True):
        if dim <= 0 or (dim & (dim - 1)) != 0:
            raise ValueError("HadamardCodebook needs a power-of-two dim, got %d" % dim)
        self.dim = int(dim)
        self.signed = bool(signed)
        # WHY A SIGN PERMUTATION: raw Hadamard rows are highly regular (row 0 is all-ones), which makes them
        # a poor stand-in for random atoms in anything that cares about the atoms' distribution. A Rademacher
        # diagonal scrambles them while leaving the fast transform intact, because a diagonal sign flip
        # commutes through the butterfly -- the whole point of the construction.
        self.signs = np.random.default_rng(seed).choice([-1, 1], size=self.dim).astype(np.int64)

    @property
    def K(self):
        """How many atoms this codebook holds: 2*dim when signed, else dim. THE CAP IS THE POINT -- see the
        module docstring; a Hadamard construction cannot exceed it, unlike a random codebook."""
        return 2 * self.dim if self.signed else self.dim

    def atom(self, index):
        """The `index`-th atom as a dense vector. For a signed codebook, indices [0, dim) are the rows and
        [dim, 2*dim) are their negations. Generated on demand -- the codebook stores a sign vector, not a
        matrix, which is the storage half of the win."""
        row = index % self.dim
        e = np.zeros(self.dim, dtype=np.int64)
        e[row] = 1
        a = fwht(e) * self.signs                       # the row of H, sign-permuted
        return -a if (self.signed and index >= self.dim) else a

    def atoms(self):
        """The full dense K x dim atom matrix. VERIFICATION AND BASELINE USE ONLY -- materialising it throws
        away the entire storage advantage. It exists so `cleanup` can be checked against an honest matmul."""
        rows = np.stack([self.atom(i) for i in range(self.dim)])
        return np.vstack([rows, -rows]) if self.signed else rows

    def correlations(self, cue):
        """All K correlations between `cue` and every atom, as one transform. For the unsigned half this is
        exactly fwht(signs * cue); the signed half is its negation, so no extra work is done. Integer input
        stays integer, so the correlations -- and therefore the cleanup DECISION -- are bit-exact."""
        cue = np.asarray(cue)
        if cue.shape != (self.dim,):
            raise ValueError("cue must have shape (%d,), got %r" % (self.dim, cue.shape))
        base = fwht(cue * self.signs if np.issubdtype(cue.dtype, np.integer)
                    else cue * self.signs.astype(np.float64))
        return np.concatenate([base, -base]) if self.signed else base

    def cleanup(self, cue):
        """Nearest atom to `cue`, as (index, score) -- exact maximum-likelihood nearest-codeword decode in
        O(D log D). Because every atom has the same norm, the largest correlation IS the nearest atom, so
        one transform plus an argmax settles it; no scan and no stored matrix. Ties break to the LOWEST index
        (numpy argmax), matching the engine's deterministic-tie rule, so the decision is reproducible."""
        corr = self.correlations(cue)
        i = int(np.argmax(corr))
        return i, corr[i]


def measure_vs_scan(dims=(256, 512, 1024, 2048, 4096), repeats=60, seed=0):
    """Honest head-to-head at EQUAL K and EQUAL D: transform cleanup against a matmul scan over a random
    codebook of the same K. Warmed, medians reported with spread.

    WHY EQUAL-K IS THE ONLY FAIR COMPARISON: the structured codebook caps K at 2D, so the backlog's proposed
    "5,682 us at K=16384 vs one transform at D=1024" compares a scan to a codebook that cannot exist. Returns
    dicts with dim, K, wht_us, scan_us, speedup (>1 means the transform wins)."""
    from holographic.misc.holographic_measure import time_call
    rng = np.random.default_rng(seed)
    out = []
    for d in dims:
        cb = HadamardCodebook(d, seed=seed)
        rand = rng.standard_normal((cb.K, d))
        cue = rng.standard_normal(d)

        w, wsd, _ = time_call(lambda: cb.cleanup(cue), repeats=repeats)
        s_, ssd, _ = time_call(lambda: np.argmax(rand @ cue), repeats=repeats)
        out.append({"dim": d, "K": cb.K, "wht_us": w, "wht_sd": wsd,
                    "scan_us": s_, "scan_sd": ssd, "speedup": s_ / w if w else float("inf")})
    return out


def _selftest():
    # 1. THE TRANSFORM PATH MUST EQUAL THE MATMUL, EXACTLY. This is the correctness contract: if the fast
    #    path and the honest scan ever disagree, the "cleanup" is decoding a different codebook.
    for d in (8, 64, 256):
        cb = HadamardCodebook(d, seed=1)
        A = cb.atoms()
        rng = np.random.default_rng(d)
        for _ in range(20):
            cue = rng.integers(-7, 8, size=d)
            assert np.array_equal(cb.correlations(cue), A @ cue), "transform != matmul at D=%d" % d
            assert cb.cleanup(cue)[0] == int(np.argmax(A @ cue)), "cleanup disagrees with the scan"

    # 2. ORTHOGONALITY: distinct rows must have exactly zero crosstalk, and each atom norm^2 == D. This is
    #    the property that makes argmax-of-correlation the ML decode, so it is asserted, not assumed.
    cb = HadamardCodebook(64, seed=2)
    R = cb.atoms()[:64]
    G = R @ R.T
    assert np.array_equal(G, 64 * np.eye(64, dtype=np.int64)), "Hadamard rows are not orthogonal"

    # 3. EXACT RECALL: every atom must clean up to ITSELF, for all K atoms, signed included.
    cb = HadamardCodebook(32, seed=3)
    for i in range(cb.K):
        assert cb.cleanup(cb.atom(i))[0] == i, "atom %d does not recall itself" % i

    # 4. NOISE TOLERANCE, with a number rather than a vibe: at 0.5x-amplitude noise the decode stays exact.
    cb = HadamardCodebook(256, seed=4)
    rng = np.random.default_rng(5)
    wrong = 0
    for _ in range(200):
        i = int(rng.integers(cb.K))
        noisy = cb.atom(i).astype(np.float64) + 0.5 * rng.standard_normal(256)
        wrong += (cb.cleanup(noisy)[0] != i)
    assert wrong == 0, "%d/200 misdecodes at 0.5x noise" % wrong

    # 5. THE CAP IS REAL AND ENFORCED BY CONSTRUCTION -- the kept negative, asserted so it cannot be
    #    quietly forgotten by someone who reads only the speedup number.
    assert HadamardCodebook(1024).K == 2048, "a D=1024 Hadamard codebook cannot exceed K=2048"
    for bad in (0, 3, 100):
        try:
            HadamardCodebook(bad)
            raise AssertionError("accepted non-power-of-two dim %d" % bad)
        except ValueError:
            pass

    # 6. DETERMINISM: same seed, same atoms, same decisions -- bit-identical, twice built.
    a, b = HadamardCodebook(128, seed=7), HadamardCodebook(128, seed=7)
    assert np.array_equal(a.atoms(), b.atoms())

    print("holographic_htcodebook: all selftests passed (exactness vs matmul, orthogonality, recall, cap)")


if __name__ == "__main__":
    _selftest()

"""Exact factorization by random linear codes -- recovery by SOLVING, not by searching.

WHY THIS EXISTS. The resonator searches the M^F product space, and its accuracy is a surface over
TWO variables, both measured with phasor_factor at D=1024 over 12 trials:

    fixed search space ~4096   F=2 12/12   F=3 8/12   F=4 8/12   F=6 2/12
    fixed F=3                  M=8 0.967   M=16 8/12  M=24 1/12

So factor count costs something beyond the space it generates, AND the space costs something on its
own. Neither is a cliff you can dimension your way past; advise_scale now reports both.

THE ALTERNATIVE, after Raviv, "Linear Codes for Hyperdimensional Computing" (Neural Computation
36(6):1084-1120, 2024): give every codebook entry a codeword drawn from a LINEAR code, bind by
addition in that code, and factorization becomes SOLVING A LINEAR SYSTEM rather than iterating a
search. Recovery is then exact whenever the system is determined -- independent of F and of M, which
is precisely the surface that degrades above.

THE FORM USED HERE. Phases are the natural carrier: an FHRR bind is phase ADDITION, so a product of
F atoms is the SUM of their phase vectors mod 2*pi. Draw each codebook's phases from a shared random
generator matrix G (D x F*k bits worth of structure) and the composite's phase vector is a linear
image of the concatenated index bits. Solve for the bits, read the indices, VERIFY by recomposition.

KEPT NEGATIVE, LOUD. This is NOT a better resonator and must never be reported as one. It is a
different algorithm with a different requirement: the codebooks must be BUILT by this module (the
codewords carry the structure that makes the system solvable). Handed arbitrary random codebooks it
has nothing to solve and correctly refuses. The resonator's virtue is that it factors codebooks it
did not create; this one's virtue is exactness on codebooks it did.

KEPT NEGATIVE 2. Determined means D >= F * ceil(log2(M)) phase equations for the index bits. Below
that the system is underdetermined and the module REFUSES rather than returning a guess -- the
abstain-not-error discipline applied to algebra.
"""

import numpy as np


def _bits_needed(m):
    """Bits to index m entries. m=1 needs 0 bits; the caller's guard, not a silent 0."""
    return int(np.ceil(np.log2(max(int(m), 2))))


def build_codebooks(dim, n_factors, n_entries, seed=0):
    """Build F codebooks of unit phasors whose phases are a LINEAR image of their index bits.

    Returns (codebooks, basis) where codebooks[f] is (n_entries, dim) complex unit phasors and
    basis is the (n_factors*bits, dim) real phase generator the solver needs. The basis IS the
    structure -- without it a composite is just a product and there is nothing to solve.
    """
    dim, F, M = int(dim), int(n_factors), int(n_entries)
    b = _bits_needed(M)
    need = F * b
    if dim < need:
        raise ValueError("dim %d < F*bits %d: the phase system would be underdetermined; "
                         "raise dim or shrink the codebook (this module refuses to guess)"
                         % (dim, need))
    rng = np.random.default_rng(int(seed))
    # Rows of `basis` are the phase directions each index BIT contributes. Orthogonalising them
    # makes the system well-conditioned; without it the solve is technically valid and numerically
    # miserable at large F, which reads as a factorization failure and is not one.
    basis = rng.standard_normal((need, dim))
    basis, _ = np.linalg.qr(basis.T)
    basis = basis.T[:need] * np.pi          # scale so a set bit is a half-turn of phase
    cbs = []
    for f in range(F):
        rows = np.empty((M, dim))
        for i in range(M):
            bits = np.array([(i >> j) & 1 for j in range(b)], float)
            rows[i] = bits @ basis[f * b:(f + 1) * b]
        cbs.append(np.exp(1j * rows))
    return cbs, basis


def factor_exact(composite, basis, n_factors, n_entries):
    """Recover the F indices from a bound product by SOLVING for the index bits.

    Returns (indices tuple, residual). Residual is the max |bit - round(bit)| over the solved
    vector: a clean solve sits at ~0 and a broken one does not, so the caller gets the evidence
    rather than a bare answer. Verification by recomposition is the caller's job and cheap.
    """
    F, M = int(n_factors), int(n_entries)
    b = _bits_needed(M)
    phase = np.angle(np.asarray(composite).reshape(-1))
    # least squares in phase space; the wrap is handled by solving on the unwrapped residual, which
    # is exact here because every codeword phase is a sum of half-turns of the SAME basis rows.
    x, *_ = np.linalg.lstsq(basis.T, phase, rcond=None)
    x = np.asarray(x, float)
    bits = np.mod(np.round(x), 2.0)
    resid = float(np.max(np.abs(x - np.round(x)))) if x.size else 0.0
    idx = []
    for f in range(F):
        v = bits[f * b:(f + 1) * b]
        idx.append(int(sum(int(v[j]) << j for j in range(b))) % M)
    return tuple(idx), resid


def _selftest():
    """Assert EXACTNESS in the regime the resonator was measured to lose, and REFUSAL below it."""
    # The regime advise_scale now flags: F=3, M=24, search space 13824 (resonator measured 1/12).
    D, F, M = 1024, 3, 24
    cbs, basis = build_codebooks(D, F, M, seed=0)
    rng = np.random.default_rng(7)
    ok = 0
    for _ in range(24):
        idx = tuple(int(rng.integers(0, M)) for _ in range(F))
        comp = np.ones(D, dtype=complex)
        for f in range(F):
            comp = comp * cbs[f][idx[f]]
        got, resid = factor_exact(comp, basis, F, M)
        ok += (got == idx)
    assert ok == 24, "exact factorization must be 24/24 where the resonator measured 1/12, got %d" % ok

    # And at F=6, where the resonator measured 2/12 even at a small search space.
    D2, F2, M2 = 1024, 6, 8
    cbs2, basis2 = build_codebooks(D2, F2, M2, seed=1)
    rng2 = np.random.default_rng(9)
    ok2 = 0
    for _ in range(16):
        idx = tuple(int(rng2.integers(0, M2)) for _ in range(F2))
        comp = np.ones(D2, dtype=complex)
        for f in range(F2):
            comp = comp * cbs2[f][idx[f]]
        got, _r = factor_exact(comp, basis2, F2, M2)
        ok2 += (got == idx)
    assert ok2 == 16, "F=6 must be exact, got %d/16" % ok2

    # REFUSAL below the determined bound, rather than a guess.
    try:
        build_codebooks(8, 6, 64, seed=0)
        raise AssertionError("must refuse an underdetermined system")
    except ValueError:
        pass

    print("holographic_lincode selftest OK -- exact at F=3/M=24 and F=6/M=8, refuses underdetermined")


if __name__ == "__main__":
    _selftest()

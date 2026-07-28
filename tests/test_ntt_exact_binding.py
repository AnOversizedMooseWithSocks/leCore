"""Regression traps for holographic_ntt -- exact integer binding.

The product here is EXACTNESS, so every contract below is asserted with array_equal, never a tolerance.
The two negatives (cost, and unbind still being a quasi-inverse) are pinned as hard as the positives,
because the claim this whole line came from -- "integer VSA lets us delete tie-arbitration" -- is
REFUTED, and a refutation that is not tested quietly comes back.
"""
import numpy as np
import pytest

import lecore
from holographic.sampling_and_signal.holographic_ntt import (DEFAULT_Q, _naive_cyclic_convolve, intt,
                                                             measure_ntt_vs_fft, ntt, ntt_bind,
                                                             ntt_convolve, ntt_unbind)


def test_ntt_convolution_equals_the_naive_oracle_exactly():
    # THE PRODUCT. An independent O(n^2) integer convolution is the oracle; array_equal, no tolerance.
    for n in (2, 8, 64):
        rng = np.random.default_rng(n)
        for _ in range(5):
            a = rng.integers(-4, 5, size=n)
            b = rng.integers(-4, 5, size=n)
            assert np.array_equal(ntt_convolve(a, b), _naive_cyclic_convolve(a, b))


def test_transform_round_trip_is_exact_and_stays_integer():
    x = np.random.default_rng(0).integers(0, 1000, size=256)
    r = intt(ntt(x))
    assert np.issubdtype(r.dtype, np.integer)
    assert np.array_equal(r, x)


def test_ntt_bind_is_the_same_algebra_as_the_float_bind():
    # Proves this is the ENGINE'S binding made exact, not a different operation wearing the name.
    # Rounding the float FFT convolution of bipolar atoms must reproduce the NTT result exactly.
    rng = np.random.default_rng(1)
    for n in (64, 512):
        a = rng.integers(-1, 2, size=n)
        b = rng.integers(-1, 2, size=n)
        f = np.fft.irfft(np.fft.rfft(a.astype(float)) * np.fft.rfft(b.astype(float)), n=n)
        assert np.array_equal(np.rint(f).astype(np.int64), ntt_convolve(a, b))


def test_overflow_guard_fires_instead_of_wrapping():
    # A wrapped modular result LOOKS like a real answer. This module exists to avoid confident wrong
    # answers, so the bound check is a hard contract.
    big = np.full(1024, 10 ** 4, dtype=np.int64)
    with pytest.raises(ValueError):
        ntt_convolve(big, big)


def test_intermediate_products_cannot_overflow_int64():
    # The butterflies multiply two residues < q, so q^2 must clear int64 or the arithmetic is silently
    # wrong for large inputs. Pinned so nobody swaps in a bigger prime without redoing this analysis.
    assert DEFAULT_Q ** 2 < np.iinfo(np.int64).max


def test_float_input_is_refused():
    with pytest.raises(TypeError):
        ntt_convolve(np.zeros(8), np.zeros(8))


def test_lengths_without_a_primitive_root_fail_loudly():
    # n must divide q-1. q-1 = 5*2^25 has no factor of 3, so length 3 has no root and must raise
    # rather than compute nonsense.
    for bad in (3, 6):
        with pytest.raises(ValueError):
            ntt(np.arange(bad))


def test_unbind_is_a_quasi_inverse_not_an_inverse_kept_negative():
    # THE REFUTATION, PINNED AS A NUMBER. The NTT removes floating-point nondeterminism from the
    # OPERATION; it does not remove the algebraic approximation inside HRR. Recovery is directional
    # (~0.7, matching the float path's measured band), so cleanup is NOT deleted. The upper bound is
    # the load-bearing half: if this ever becomes exact, the claim must be rewritten, not the test.
    n = 1024
    rng = np.random.default_rng(2)
    a = rng.integers(-1, 2, size=n)
    b = rng.integers(-1, 2, size=n)
    got = ntt_unbind(ntt_bind(a, b), a).astype(np.float64)
    cos = float(got @ b / (np.linalg.norm(got) * np.linalg.norm(b)))
    assert 0.4 < cos < 0.999


def test_ntt_is_slower_than_the_fft_kept_negative():
    # Measured 19-50x slower across D=256..4096. Loose bound: a timing assertion is a machine-load
    # detector, not a numeric contract -- but if it ever fails, re-measure and rewrite the claim.
    assert measure_ntt_vs_fft(sizes=(512,), repeats=20)[0]["ratio"] > 1.5


# --------------------------------------------------------------------------------------
# CROSS-FACULTY
# --------------------------------------------------------------------------------------

def test_ntt_faculties_round_trip_through_the_mind():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    a, b = rng.integers(-1, 2, size=256), rng.integers(-1, 2, size=256)
    c = mind.ntt_bind(a, b)
    assert np.array_equal(c, mind.ntt_convolve(a, b))
    got = mind.ntt_unbind(c, a).astype(np.float64)
    assert float(got @ b / (np.linalg.norm(got) * np.linalg.norm(b))) > 0.4


def test_exact_binding_is_reproducible_across_repeated_construction():
    # The determinism claim, exercised the only way a single machine can: the operation must be a pure
    # integer function of its inputs, so repeated evaluation and a fresh mind give bit-identical bytes.
    rng = np.random.default_rng(3)
    a, b = rng.integers(-1, 2, size=512), rng.integers(-1, 2, size=512)
    first = lecore.UnifiedMind(dim=64, seed=0).ntt_bind(a, b)
    for _ in range(3):
        assert np.array_equal(lecore.UnifiedMind(dim=64, seed=0).ntt_bind(a, b), first)
    assert first.dtype == np.int64


def test_exact_bind_cleans_up_through_a_hadamard_codebook():
    # CROSS-FACULTY: exact integer binding feeding the transform-based cleanup built alongside it. Both
    # are integer-exact, so the whole bind -> cleanup path is machine-independent end to end -- which is
    # the actual point of the pairing, and something neither module's selftest can check alone.
    mind = lecore.UnifiedMind(dim=256, seed=0)
    cb = mind.hadamard_codebook(256, seed=0, signed=False)
    key = np.random.default_rng(4).integers(-1, 2, size=256)
    for i in (0, 7, 130):
        bound = mind.ntt_bind(cb.atom(i), key)
        recovered = mind.ntt_unbind(bound, key)
        assert cb.cleanup(recovered)[0] == i, "atom %d did not survive exact bind/unbind + cleanup" % i


def test_ntt_capability_is_discoverable_by_stranger_phrasing():
    mind = lecore.UnifiedMind(dim=128, seed=0)
    for query in ("exact circular convolution with integers", "number theoretic transform",
                  "convolution that is identical on every machine", "bind without floating point"):
        assert "NTT exact" in str(mind.find_capability(query)[:3]), "%r no longer surfaces the NTT" % query

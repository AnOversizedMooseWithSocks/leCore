"""A prebuilt map of hypervector transforms -- a GROUP REPRESENTATION, not a lookup table.

Moose's idea, measured. The idea is right; the payoff is not where it looks; and one of the four things named
cannot go in the map at all.

    caching one transform's spectrum       1.42x   (the operand's rfft is 28% of a bind)
    composing a chain of 8 into one bind  13.5x   exact to 5.7e-17   <- THE POINT
    batching one transform over M=512      2.3x   (the transforms dominate, not the loop)

**SCALE IS NOT IN THE BANK.** A dilation is not shift-invariant, so it is not diagonal in the Fourier basis and no
spectrum represents it: fit one on a vector, apply it to a second, relative error 1.579. The wrong object, not a
lossy fit. DL11 said so already, and gave the remedy: on a log axis a dilation becomes a shift.
"""

import numpy as np
import pytest

from holographic.agents_and_reasoning.holographic_ai import bind
from holographic.caching_and_storage.holographic_transformbank import TransformBank, scale_is_not_a_bind


D = 512


def _bank(seed=0, n=6):
    b = TransformBank(D, seed=seed)
    for i in range(n):
        b.add_random_unitary("t%d" % i)
    b.add_rotation("rot7", 7)
    return b


def test_selftest_runs():
    from holographic.caching_and_storage import holographic_transformbank as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# what the bank holds
# ---------------------------------------------------------------------------------------------------------

def test_a_cyclic_rotation_really_is_a_bind():
    b = _bank()
    v = np.random.default_rng(1).normal(size=D)
    assert np.abs(b.apply("rot7", v) - np.roll(v, 7)).max() < 1e-10


def test_one_transform_matches_bind():
    b = _bank()
    v = np.random.default_rng(1).normal(size=D)
    assert np.abs(b.apply("t0", v) - bind(v, b._atoms["t0"])).max() < 1e-10


def test_a_batch_matches_the_loop():
    b = _bank()
    V = np.random.default_rng(2).normal(size=(16, D))
    loop = np.stack([b.apply("t1", v) for v in V])
    assert np.abs(loop - b.apply_batch("t1", V)).max() < 1e-12


# ---------------------------------------------------------------------------------------------------------
# THE POINT: composition
# ---------------------------------------------------------------------------------------------------------

def test_a_chain_of_transforms_collapses_into_one_bind():
    b = _bank()
    names = ["t%d" % i for i in range(6)]
    v = np.random.default_rng(3).normal(size=D)

    seq = v
    for n in names:
        seq = bind(seq, b._atoms[n])
    assert np.abs(seq - b.apply_chain(names, v)).max() < 1e-12


def test_kept_negative_composition_is_exact_but_not_bit_identical():
    # One inverse transform instead of k. Same product, different rounding -- the same reassociation the emitted C
    # twin and `encode_many` both show. Reported as a distance, never as a boolean.
    b = _bank()
    names = ["t%d" % i for i in range(6)]
    v = np.random.default_rng(3).normal(size=D)
    seq = v
    for n in names:
        seq = bind(seq, b._atoms[n])
    got = b.apply_chain(names, v)
    assert not np.array_equal(seq, got)
    assert np.abs(seq - got).max() < 1e-12


def test_the_chain_is_actually_faster():
    import time

    b = _bank()
    names = ["t%d" % i for i in range(6)]
    v = np.random.default_rng(3).normal(size=D)

    def _t(fn, n=20):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    def _seq():
        x = v
        for n in names:
            x = bind(x, b._atoms[n])
        return x

    assert _t(_seq) > 2.0 * _t(lambda: b.apply_chain(names, v))     # measured 13.5x at D=4096


def test_composition_is_commutative_because_convolution_is():
    b = _bank()
    names = ["t0", "t3", "rot7"]
    assert np.abs(b.compose(names) - b.compose(list(reversed(names)))).max() < 1e-10


def test_the_empty_chain_is_the_identity():
    b = _bank()
    v = np.random.default_rng(4).normal(size=D)
    assert np.abs(b.apply_chain([], v) - v).max() < 1e-10


def test_a_power_matches_repeated_application():
    b = _bank()
    v = np.random.default_rng(5).normal(size=D)
    rep = v
    for _ in range(5):
        rep = bind(rep, b._atoms["t0"])
    pw = np.fft.irfft(np.fft.rfft(v) * b.power("t0", 5), n=D)
    assert np.abs(rep - pw).max() < 1e-10


def test_a_rotation_power_is_a_bigger_rotation():
    b = _bank()
    v = np.random.default_rng(6).normal(size=D)
    got = np.fft.irfft(np.fft.rfft(v) * b.power("rot7", 3), n=D)
    assert np.abs(got - np.roll(v, 21)).max() < 1e-10


# ---------------------------------------------------------------------------------------------------------
# unitarity: the inverse
# ---------------------------------------------------------------------------------------------------------

def test_a_unitarys_inverse_is_its_conjugate_and_a_gaussians_is_refused():
    b = _bank()
    v = np.random.default_rng(7).normal(size=D)
    assert b.is_unitary("t0") and b.is_unitary("rot7")
    back = np.fft.irfft(np.fft.rfft(b.apply("t0", v)) * b.inverse_spectrum("t0"), n=D)
    assert np.abs(back - v).max() < 1e-10

    b.add("gauss", np.random.default_rng(8).normal(size=D))
    assert not b.is_unitary("gauss")
    with pytest.raises(ValueError, match="not unitary"):
        b.inverse_spectrum("gauss")                            # N11: cosine 0.744, not 1.0


# ---------------------------------------------------------------------------------------------------------
# THE REFUSAL: scale is not a bind
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_scale_is_not_diagonal_in_the_fourier_basis():
    # Fit the "spectrum" of a dilation on one vector; apply it to another. It is the wrong object, not a lossy fit.
    rel = scale_is_not_a_bind(dim=256, s=1.5, seed=0)
    assert rel > 0.5                                           # measured 1.579

    for s in (1.2, 2.0):
        assert scale_is_not_a_bind(dim=256, s=s, seed=1) > 0.2


def test_the_bank_offers_no_way_to_add_a_scale():
    # Refusing what the algebra does not diagonalise is the feature. A bank that "supported" scale would return a
    # confidently wrong vector.
    b = _bank()
    assert not hasattr(b, "add_scale")
    assert "scale" not in b.names()


# ---------------------------------------------------------------------------------------------------------
# accounting + guards
# ---------------------------------------------------------------------------------------------------------

def test_the_bank_costs_the_same_as_its_atoms_not_twice():
    # I guessed 2x. An rfft of a real vector is Hermitian, so numpy stores D/2+1 complex128 -- the same bytes as D
    # float64. The bank is free.
    b = _bank()
    st = b.stats()
    assert st["n_transforms"] == 7
    assert 0.99 < st["vs_atoms"] < 1.02


def test_unknown_and_malformed_inputs_raise():
    b = _bank()
    with pytest.raises(KeyError, match="no transform"):
        b.spectrum("nope")
    with pytest.raises(ValueError, match="atom must be"):
        b.add("bad", np.zeros(D + 1))


def test_the_bank_is_deterministic():
    a, b = _bank(seed=0), _bank(seed=0)
    assert np.array_equal(a.spectrum("t0"), b.spectrum("t0"))
    assert not np.array_equal(_bank(seed=1).spectrum("t0"), a.spectrum("t0"))


def test_wired_to_the_mind_and_discoverable():
    import lecore

    m = lecore.UnifiedMind(dim=256, seed=0)
    b = m.transform_bank(256)
    b.add_rotation("r3", 3)
    v = np.random.default_rng(0).normal(size=256)
    assert np.abs(b.apply("r3", v) - np.roll(v, 3)).max() < 1e-10
    assert m.scale_is_not_a_bind() > 0.5
    assert "Transform bank" in str(m.find_capability("prebuilt map of transforms")[:3])

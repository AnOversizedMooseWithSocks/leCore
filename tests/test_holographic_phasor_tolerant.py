"""tests/test_holographic_phasor_tolerant.py -- holographic_phasor.factor(tolerant=True) (CLM backlog, wave 2).

The default phasor resonator (one run from the codebook means, a hard snap, a fixpoint exit) solves 0 of 20 exact
products at 20 x 50 x 50, D=2048. tolerant=True adds the superposition (FHRR) update with restarts, the agreement
exits and a procedure-matched null p-value -- the pattern of holographic_resonator.factor(tolerant=True). Pinned
here on small shapes (fast): the default path is untouched, exact products certify, noisy ones are right AND
significant, structureless composites are refused, and everything is deterministic.
"""
import os

import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "0")

import holographic.agents_and_reasoning.holographic_phasor as PH  # noqa: E402

D = 512
SIZES = (6, 8, 8)
CBS = [np.stack([PH.atom("t%d_%d" % (g, i), D) for i in range(n)]) for g, n in enumerate(SIZES)]


def _product(idx):
    c = np.ones(D, dtype=complex)
    for f, i in enumerate(idx):
        c = c * CBS[f][i]
    return c


def test_the_default_path_is_unchanged_and_still_a_tuple():
    truth = (2, 5, 1)
    got = PH.factor(_product(truth), CBS)
    assert isinstance(got, tuple) and all(isinstance(i, int) for i in got)
    # the new keyword arguments are inert unless tolerant=True
    assert PH.factor(_product(truth), CBS, restarts=3, m_null=5, seed=9) == got


def test_an_exact_product_certifies():
    r = PH.factor(_product((1, 7, 3)), CBS, tolerant=True, m_null=19, accept_p=0.05)
    assert r["factors"] == (1, 7, 3) and r["exit"] == "exact" and r["solved"] is True
    assert r["agreement"] > 0.999 and r["accepted"] is True and r["search_space"] == 6 * 8 * 8


def test_a_noisy_product_is_right_and_significant():
    rng = np.random.default_rng(4)
    right = 0
    for trial in range(5):
        truth = tuple(int(rng.integers(n)) for n in SIZES)
        c = _product(truth) * np.exp(1j * 0.6 * rng.standard_normal(D))     # heavy phase jitter
        r = PH.factor(c, CBS, tolerant=True, m_null=19, accept_p=0.05)
        assert r["solved"] is False                                         # 'solved' keeps its exact meaning
        right += (r["factors"] == truth and r["accepted"])
    assert right == 5


def test_a_structureless_composite_is_not_accepted_and_the_null_is_cached():
    rng = np.random.default_rng(5)
    for _ in range(3):
        junk = np.exp(2j * np.pi * rng.random(D))
        r = PH.factor(junk, CBS, tolerant=True, m_null=19, accept_p=0.05)
        assert r["accepted"] is False and r["p_value"] > 0.05
    n0 = len(PH._TOLERANT_NULL_CACHE)
    PH.factor(junk, CBS, tolerant=True, m_null=19, accept_p=0.05)
    assert len(PH._TOLERANT_NULL_CACHE) == n0                               # one null per codebook shape


def test_the_tolerant_path_is_deterministic():
    c = _product((0, 2, 6)) * np.exp(1j * 0.5 * np.random.default_rng(6).standard_normal(D))
    a = PH.factor(c, CBS, tolerant=True, m_null=19)
    b = PH.factor(c, CBS, tolerant=True, m_null=19)
    assert a == b

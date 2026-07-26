"""The Group-A null layer, exercised ACROSS faculties and through the mind.

The layer spans two families on purpose -- the surrogate generators live in sampling_and_signal, the
evaluators in agents_and_reasoning -- so the thing most likely to rot is the seam between them, not either
half. These tests drive that seam: mind faculty -> honesty evaluator -> surrogate generator, plus the whole
screening chain (pipeline -> pipeline_null -> split_half -> bh_fdr) that the campaign this layer came from
ran on every candidate.

The headline contract, asserted here and in holographic_honesty._selftest_null_layer, is the one that pays
for the module: a chain that MANUFACTURES structure on pure noise must be caught by its own pipeline null
and must NOT be caught by comparison against a textbook baseline.
"""
import math

import numpy as np

import lecore
from holographic.agents_and_reasoning.holographic_honesty import bh_fdr


def _smoothed_persistence(d, a=0.8):
    """An exponential smoother followed by a direction-persistence count -- innocent-looking, and it invents
    momentum out of white noise. Deliberately the pipeline under test."""
    y = np.empty(len(d))
    y[0] = d[0]
    for i in range(1, len(d)):
        y[i] = a * y[i - 1] + (1 - a) * d[i]
    s = np.sign(y)
    s = s[s != 0]
    return float(np.mean(s[1:] == s[:-1]))


def _markov_sign_series(n, hold, seed):
    """A series with GENUINE sign persistence: a Markov sign chain times independent magnitudes."""
    rng = np.random.default_rng(seed)
    sgn = np.ones(n)
    for i in range(1, n):
        sgn[i] = sgn[i - 1] if rng.random() < hold else -sgn[i - 1]
    return sgn * np.abs(rng.normal(size=n))


def test_pipeline_null_catches_manufactured_structure_that_a_textbook_baseline_misses():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    noise = np.random.default_rng(0).normal(size=3000)
    r = mind.pipeline_null(_smoothed_persistence, noise, surrogate="iid_shuffle", n=100, seed=0, side="greater")
    # against the textbook 0.5 this is a spectacular "momentum effect"...
    assert r["observed"] - 0.5 > 0.25
    # ...and against the null the SAME chain produces, it is nothing at all.
    assert abs(r["z"]) < 2.0
    assert not r["collapsed"]
    # the surrogates' own persistence proves the manufacturing: ~0.79, nowhere near 0.5.
    assert r["null_mean"] > 0.7


def test_pipeline_null_still_has_power_through_the_same_manufacturing_chain():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    real = _markov_sign_series(3000, hold=0.75, seed=1)
    r = mind.pipeline_null(_smoothed_persistence, real, surrogate="iid_shuffle", n=100, seed=0, side="greater")
    assert r["z"] > 4.0
    assert r["collapsed"]


def test_the_full_screening_chain_composes_pipeline_null_split_half_and_fdr():
    """The campaign's actual per-candidate gauntlet, run over a small family of candidates: one real, three
    null. Every honest candidate must clear ALL THREE gates and every null candidate must fail at least one.
    This is the cross-faculty test proper -- the mind's surrogate faculties, the honesty evaluators, and the
    pre-existing bh_fdr all in one pass."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(7)
    candidates = {
        "real": _markov_sign_series(3000, hold=0.75, seed=2),
        "null_a": rng.normal(size=3000),
        "null_b": rng.normal(size=3000),
        "null_c": rng.normal(size=3000),
    }
    pvals, passed_split = [], {}
    for name, series in candidates.items():
        r = mind.pipeline_null(_smoothed_persistence, series, surrogate="iid_shuffle", n=100, seed=0,
                               side="greater")
        pvals.append(r["p"])
        # gate 2: does the per-event effect replicate across halves? Score each point by whether its smoothed
        # direction agreed with the previous one -- the same quantity the pipeline aggregates.
        y = np.convolve(series, np.ones(9) / 9.0, mode="valid")
        agree = (np.sign(y[1:]) == np.sign(y[:-1])).astype(float) - 0.5
        passed_split[name] = mind.split_half(agree)["passed"]
    names = list(candidates)
    rejected, n_rejected = bh_fdr(pvals, alpha=0.1)          # (mask, count) -- the family-wide gate
    verdict = {n: bool(rej) and passed_split[n] for n, rej in zip(names, rejected)}
    assert n_rejected == 1, (n_rejected, pvals)
    assert verdict["real"], verdict
    assert not any(verdict[n] for n in names if n != "real"), verdict


def test_surrogate_choice_is_load_bearing_and_the_degenerate_case_is_visible():
    """The layer's sharpest kept negative, pinned across the seam: a null that PRESERVES the statistic under
    test is not a null. sign_flip keeps magnitudes exactly, so a magnitude-only statistic has a zero-spread
    null; iid_shuffle keeps the sample mean exactly, so a mean test against it degenerates to a 0/1 step."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = np.cumsum(np.random.default_rng(3).normal(size=800))
    # sign_flip preserves every magnitude EXACTLY -> a magnitude-only statistic cannot vary across the null.
    mag_null = np.array([float(np.mean(np.abs(s)))
                         for s in mind.surrogate_ensemble(x, "sign_flip", n=16, seed=0)])
    assert mag_null.std() < 1e-12

    def ttest_p(v):
        se = v.std(ddof=1) / math.sqrt(len(v))
        return math.erfc(abs(v.mean() / se) / math.sqrt(2.0))

    base = np.random.default_rng(11).normal(size=400)
    graded = mind.min_detectable_effect(ttest_p, base, [0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
                                        surrogate="sign_flip", n_trials=60, seed=0)
    degenerate = mind.min_detectable_effect(ttest_p, base, [0.0, 0.02, 0.05],
                                            surrogate="iid_shuffle", n_trials=40, seed=0)
    assert graded["floor"] == 0.15                                   # 3 sigma at n=400, sd=1
    assert not all(pw in (0.0, 1.0) for pw in graded["power_curve"])
    assert all(pw in (0.0, 1.0) for pw in degenerate["power_curve"])


def test_split_half_modes_separate_a_regime_bound_effect_from_a_replicating_one():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(21)
    replicating = np.concatenate([rng.normal(0.5, 1, 200), rng.normal(0.5, 1, 200)])
    regime_bound = np.concatenate([rng.normal(0.6, 1, 200), rng.normal(0.0, 1, 200)])
    assert mind.split_half(replicating)["passed"]
    assert mind.split_half(replicating, mode="interleave")["passed"]
    # the regime-bound effect passes interleaved (both halves share the regime) and fails contiguous. That
    # DIFFERENCE is the finding: the effect is real inside its regime and absent outside it.
    assert mind.split_half(regime_bound, mode="interleave")["passed"]
    assert not mind.split_half(regime_bound, mode="contiguous")["passed"]


def test_arrow_of_time_flags_a_nonlinear_process_and_spares_a_linear_one():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    n = 1024
    saw = (np.arange(n) % 50) / 50.0                                 # slow rise, instant fall: irreversible
    rng = np.random.default_rng(5)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + rng.normal()                       # linear Gaussian: reversible
    a_saw = mind.time_arrow_test(saw, n_surrogates=60, seed=1)
    a_ar = mind.time_arrow_test(ar, n_surrogates=60, seed=1)
    assert a_saw["z"] < -10.0
    assert abs(a_ar["z"]) < 3.0
    assert a_saw["p"] < a_ar["p"]


def test_the_null_layer_is_deterministic_end_to_end():
    """Two independently constructed minds must agree bit-for-bit -- the layer is seeded throughout and any
    accidental use of global randomness would show up here first."""
    a = lecore.UnifiedMind(dim=256, seed=0)
    b = lecore.UnifiedMind(dim=256, seed=0)
    x = np.cumsum(np.random.default_rng(4).normal(size=600))
    assert np.array_equal(a.sign_flip(x, seed=2), b.sign_flip(x, seed=2))
    assert np.array_equal(a.block_shuffle(x, 32, seed=2), b.block_shuffle(x, 32, seed=2))
    assert a.trev(x) == b.trev(x)
    assert a.time_arrow_test(x, n_surrogates=20, seed=1) == b.time_arrow_test(x, n_surrogates=20, seed=1)
    assert a.pipeline_null(_smoothed_persistence, x, n=20, seed=1) == \
           b.pipeline_null(_smoothed_persistence, x, n=20, seed=1)

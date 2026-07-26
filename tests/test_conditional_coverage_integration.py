"""D1 through the mind: conditional coverage catching the guarantee that holds on average and fails in the
state that matters -- and its composition with the causal gate (build the condition causally, diagnose the
coverage under it)."""
import numpy as np

import lecore


def test_marginal_coverage_hides_a_degraded_state_and_the_split_reveals_it():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    calib = rng.normal(0, 1, 600)
    storm = np.arange(1000) % 4 == 0
    test = np.where(storm, rng.normal(0, 3.0, 1000), rng.normal(0, 1.0, 1000))
    row = mind.conditional_coverage(calib, test, storm, alphas=(0.1,))[0]
    assert row["degraded"] and row["reliable"]
    assert row["empirical_inside"] < row["nominal"] - 0.1          # storms uncovered
    assert row["empirical_all"] > row["empirical_inside"] + 0.1    # the average hid it


def test_uniform_forecaster_mostly_passes_and_thin_sides_refuse_to_pretend():
    """The 2-SE degraded alarm has a DESIGNED ~5%% false-alarm rate on a genuinely uniform forecaster (two
    sides at ~2.5%% each) -- so the honest test is the rate, not any single draw. (The first draft asserted one
    seed and promptly drew the 5%%; kept as the reason this test looks the way it does.)"""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    trips = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        calib = rng.normal(0, 1, 600)
        storm = np.arange(400) % 4 == 0
        if mind.conditional_coverage(calib, rng.normal(0, 1, 400), storm, alphas=(0.1,))[0]["degraded"]:
            trips += 1
    assert trips <= 3, trips                       # ~0.5 expected; 4+ of 10 would mean a broken alarm
    rng = np.random.default_rng(1)
    tiny = mind.conditional_coverage(rng.normal(0, 1, 600), rng.normal(0, 1, 40),
                                     (np.arange(40) % 4 == 0), alphas=(0.1,))[0]
    assert not tiny["reliable"]


def test_composition_with_a_causal_gate_condition():
    """The condition built the way it would be USED: a causal trailing-volatility gate over the test-period
    context, its mask feeding the coverage split -- C1 and D1 in one chain."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(2)
    n = 800
    vol = np.where(np.arange(n) % 200 < 50, 3.0, 1.0)              # periodic high-vol regimes
    context = rng.normal(0, 1, n) * vol
    gate = mind.causal_gate(stat="std", window=30, threshold=1.8, compare="ge", context=context)
    mask = np.asarray(gate["mask"], bool)
    calib = rng.normal(0, 1, 600)
    test = rng.normal(0, 1, n) * vol                                # residuals inherit the state
    rows = mind.conditional_coverage(calib, test, mask, alphas=(0.1,))
    assert rows[0]["degraded"]                                      # the causal gate finds the uncovered state
    assert gate["audit"]["passed"]                                  # and its causality proof travelled with it

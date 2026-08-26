"""Tests for holographic_refine.py -- the produce/critique/adjust/retry loop."""
import numpy as np
from holographic.misc.holographic_refine import refine, opponent_critic


def test_converges_and_reports_tries():
    target = 100.0
    log = refine(produce=lambda: 0.0,
                 critique=lambda v: 1.0 - abs(target - v) / target,
                 adjust=lambda v, s: v + (target - v) * 0.5,
                 accept=0.95, budget=10)
    assert log["accepted"] and log["score"] >= 0.95
    assert 0 < log["tries"] <= 10


def test_first_attempt_already_good_uses_zero_tries():
    log = refine(produce=lambda: 1.0, critique=lambda v: 1.0, adjust=lambda v, s: v, accept=0.9, budget=5)
    assert log["accepted"] and log["tries"] == 0


def test_budget_exhausted_is_honest():
    log = refine(produce=lambda: 0.0, critique=lambda v: 0.2, adjust=lambda v, s: v, accept=0.9, budget=3)
    assert not log["accepted"] and log["tries"] == 3 and log["score"] == 0.2


def test_opponent_critic_scores_by_agreement():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(128); ref /= np.linalg.norm(ref)
    crit = opponent_critic(ref)
    assert crit(ref) > 0.99                                   # identical -> agreement ~1
    assert crit(-ref) < -0.99                                 # opposed -> negative


def test_refine_with_opponent_critic_drives_to_reference():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(256); ref /= np.linalg.norm(ref)
    out = refine(produce=lambda: ref + 1.5 * rng.standard_normal(256),
                 critique=opponent_critic(ref),
                 adjust=lambda v, s: 0.5 * (v / np.linalg.norm(v)) + 0.5 * ref,
                 accept=0.9, budget=12)
    assert out["accepted"]

"""Tests for holographic_tokensample and its wiring (PredictiveMemory.sample, mind.sample_instruction).

The contracts pinned here are the MEASURED ones, not smoke tests:
  * temperature limits (argmax at T->0 including the underflow guard; flattening at high T)
  * nucleus trims the tail and NEVER samples it; top_p=1.0 is bit-identical to plain temperature
  * determinism under a seeded rng
  * PredictiveMemory.sample is distribution-faithful (a 70/30 successor split reads ~70/30 -- the same
    MAP-correctness contract the soft predict path documents)
  * generate_sampled does NOT limit-cycle where greedy generate does (the reason the module exists)
  * mind-level: sample_instruction round-trips through learn_recipe_grammar (cross-faculty integration)
"""
import numpy as np
import pytest

from holographic.agents_and_reasoning.holographic_tokensample import sample_from_distribution
from holographic.agents_and_reasoning.holographic_predictive import PredictiveMemory


DIST = {"a": 0.7, "b": 0.2, "c": 0.1}


def test_low_temperature_is_argmax_with_underflow_guard():
    # T=1e-4 underflows weight**(1/T) to zero mass; the guard must fall back to argmax, not NaN.
    rng = np.random.default_rng(0)
    picks = {sample_from_distribution(DIST, temperature=1e-4, rng=rng) for _ in range(50)}
    assert picks == {"a"}


def test_high_temperature_flattens():
    rng = np.random.default_rng(1)
    hot = [sample_from_distribution(DIST, temperature=8.0, rng=rng) for _ in range(3000)]
    frac_a = hot.count("a") / len(hot)
    assert 0.34 < frac_a < 0.50          # far below the raw 0.70, toward uniform 1/3


def test_nucleus_never_samples_the_trimmed_tail():
    rng = np.random.default_rng(2)
    picks = {sample_from_distribution(DIST, temperature=1.0, top_p=0.7, rng=rng) for _ in range(200)}
    assert picks == {"a"}                # 'a' alone reaches the 0.7 mass; b, c are trimmed


def test_top_p_one_is_plain_temperature_exactly():
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    a = [sample_from_distribution(DIST, 0.8, top_p=1.0, rng=r1) for _ in range(100)]
    b = [sample_from_distribution(DIST, 0.8, rng=r2) for _ in range(100)]
    assert a == b


def test_deterministic_under_seeded_rng():
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    assert [sample_from_distribution(DIST, 0.6, 0.9, r1) for _ in range(30)] == \
           [sample_from_distribution(DIST, 0.6, 0.9, r2) for _ in range(30)]


def test_empty_and_dead_distributions_return_none():
    assert sample_from_distribution({}, rng=np.random.default_rng(0)) is None
    assert sample_from_distribution({"x": 0.0}, rng=np.random.default_rng(0)) is None


def test_predictive_memory_sample_is_distribution_faithful():
    # a 70/30 successor split must SAMPLE ~70/30 (support weighting), not 50/50 and not argmax-only.
    pm = PredictiveMemory(dim=512, order=1, seed=0)
    seq = []
    for i in range(100):
        seq += ["x", "y" if i % 10 < 7 else "z"]      # after 'x': 70% 'y', 30% 'z'
    pm.learn_sequence(seq)
    rng = np.random.default_rng(3)
    draws = [pm.sample(["x"], temperature=1.0, rng=rng)[0] for _ in range(800)]
    frac_y = draws.count("y") / len(draws)
    assert 0.55 < frac_y < 0.85, frac_y               # faithful, not collapsed to either mode
    assert draws.count("z") > 0                        # the minority successor is REACHABLE


def test_generate_sampled_breaks_the_limit_cycle():
    # a cycle with one stochastic branch: greedy locks the loop; sampling must escape it sometimes.
    pm = PredictiveMemory(dim=512, order=2, seed=0)
    rng0 = np.random.default_rng(0)
    seq = []
    for _ in range(60):
        seq += ["a", "b"]
        seq += ["c"] if rng0.random() < 0.3 else ["d"]
    pm.learn_sequence(seq)
    greedy = pm.generate(["a", "b"], length=60)
    sampled = pm.generate_sampled(["a", "b"], length=60, temperature=1.0, seed_rng=5)
    # greedy emits ONE branch symbol forever; sampled must visit BOTH branches
    assert len({t for t in greedy if t in ("c", "d")}) <= 1
    assert {"c", "d"} <= set(sampled)


def test_mind_sample_instruction_round_trip():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    recipes = [[("op%d" % (i % 3), str(i % 4)) for i in range(12)] for _ in range(6)]
    m.learn_recipe_grammar(recipes, order=2)
    rng = np.random.default_rng(1)
    out = m.sample_instruction(recipes[0][:2], temperature=1.0, rng=rng)
    assert out and out[0] is not None                 # (opcode, operand) came back
    rec = m.sample_recipe(recipes[0][:2], length=6, temperature=1.0, seed_rng=2)
    assert 1 <= len(rec) <= 6 and all(isinstance(x, tuple) and len(x) == 2 for x in rec)


def test_repeated_recipe_context_is_stored_not_absorbed():
    """Regression (routing bug found by dogfooding the sampler): a (context -> next) transition
    identical to a stored one must be reachable by that context, not silently absorbed into an
    UNRELATED same-successor entry (the old reinforce path gated on surprise alone, so a lucky
    argmax reinforced the zero-context row and the real context read back an EMPTY distribution)."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    m.learn_recipe_grammar([[("a", "1"), ("b", "2"), ("a", "1"), ("c", "3")] for _ in range(4)], order=2)
    jm = m._recipe_grammar_joint
    dist = jm.next_distribution([m._instr_token(("a", "1")), m._instr_token(("b", "2"))])
    assert dist, "the ['a|1','b|2'] context must have a stored successor distribution"
    assert "a|1" in dist
    out = m.sample_instruction([("a", "1"), ("b", "2")], rng=np.random.default_rng(0))
    assert out[0] is not None, "sample_instruction must not return (None, None) for a trained context"

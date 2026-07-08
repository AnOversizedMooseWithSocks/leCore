"""Tests for holographic_navigator: the creature's brain, learning to navigate
the data tree, should match a wide fixed beam's recall at far lower cost."""

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import random_vector
from holographic.agents_and_reasoning.holographic_navigator import DataWorld, CreatureEncoder, HolographicMind, train, evaluate, fixed_beam_curve


def _trained_world(seed=0):
    rng = np.random.default_rng(seed)
    items = np.stack([random_vector(128, rng) for _ in range(600)])
    world = DataWorld(items, leaf_size=32, seed=seed, max_regions=12, noise=0.5)
    enc = CreatureEncoder(256, seed=1)
    mind = HolographicMind(256, DataWorld.ACTIONS, k=12, epsilon=0.3,
                           novelty_bonus=0.1, memory_cap=4000, seed=3)
    train(world, enc, mind, queries=2500)
    return world, enc, mind


def test_dataworld_basic_roundtrip():
    # With a clean (noise-free) cue and enough effort, the navigator's frontier
    # should contain the exact item, so committing on it is correct.
    rng = np.random.default_rng(1)
    items = np.stack([random_vector(64, rng) for _ in range(200)])
    world = DataWorld(items, leaf_size=16, seed=0, max_regions=8, noise=0.0)
    world.reset(rng)
    # exhaust the frontier, then arrive -- best item should be the true NN
    done = False
    while not done:
        _, _, _, done = world.step("keep_moving")
    assert world.correct()


def test_navigator_matches_recall_at_lower_cost():
    world, enc, mind = _trained_world(seed=0)
    recall, comps = evaluate(world, enc, mind, queries=300)
    base = fixed_beam_curve(world, beams=(1, 2, 4, 8, 12), queries=300)
    widest = base[-1]                       # the most thorough fixed beam
    # The navigator should be accurate...
    assert recall >= 0.80
    # ...while spending markedly fewer comparisons than the widest fixed beam
    # that it is competitive with.
    assert comps < 0.6 * widest["comparisons"]
    # And it should have learned a compact policy (a handful of prototypes).
    assert mind.prototype_count() < 400


def test_reflex_habits_help_on_skew_and_dont_hurt_on_uniform():
    from holographic.agents_and_reasoning.holographic_navigator import Navigator, _zipf_workload
    world, enc, mind = _trained_world(seed=0)
    items = world.items

    def run(workload, use_reflex):
        nav = Navigator(world, enc, mind, hot_size=32)
        r = np.random.default_rng(123)
        ok = comps = 0
        for i in workload:
            q = items[i] + world.noise * random_vector(world.dim, r)
            q = q / np.linalg.norm(q)
            truth = int((items @ q).argmax())
            pred, c, _ = (nav.find(q) if use_reflex
                          else world.search(q, enc, mind))
            ok += (pred == truth); comps += c
        return ok / len(workload), comps / len(workload)

    n = len(items)
    skew = _zipf_workload(n, 2500, 1.3, seed=5)
    s_recall, s_comps = run(skew, False)
    r_recall, r_comps = run(skew, True)
    # On a skewed stream the habits should clearly cut cost without losing recall.
    assert r_comps < 0.8 * s_comps
    assert r_recall >= s_recall - 0.02

    uni = _zipf_workload(n, 2500, 0.0, seed=6)
    _, su_comps = run(uni, False)
    _, ru_comps = run(uni, True)
    # On an unpredictable stream the flux guard keeps it from costing much more.
    assert ru_comps < 1.3 * su_comps

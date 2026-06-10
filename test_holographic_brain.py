"""The orchestrator brain must not go stale. These tests pin the inception fix:
the data-organizer's tools (fold redundant prototypes, self-trigger on a signal the
model reads off itself) applied to the HolographicMind that runs the system.

Background measured in the session: a plain brain cannot recover from a regime
shift, because near-duplicate prototypes each hold the old value up and online
updates only touch one at a time -- so it cannot unlearn. Folding the duplicates
both compresses the memory and restores adaptability."""

import numpy as np
from holographic_creature import HolographicMind


def _world(dim=256, seed=0, C=24, A=3):
    rng = np.random.default_rng(seed)
    base = [b / np.linalg.norm(b) for b in (rng.standard_normal(dim) for _ in range(C))]
    best = [int(rng.integers(A)) for _ in range(C)]

    def state(i):
        v = base[i] + 0.02 * rng.standard_normal(dim)
        return v / np.linalg.norm(v)

    return rng, state, best, C, A


def _train(brain, rng, state, best, C, steps):
    for _ in range(steps):
        i = int(rng.integers(C)); s = state(i)
        a = brain.decide(s, explore=True, epsilon=0.25)
        brain.remember([s], [a], [1.0 if a == best[i] else 0.0])


def _acc(brain, rng, state, best, C, n=600):
    ok = sum(brain.decide(state(i := int(rng.integers(C))), explore=False, epsilon=0.0) == best[i]
             for _ in range(n))
    return ok / n


def test_self_maintaining_brain_recovers_from_a_regime_shift():
    dim = 256
    rng, state, best, C, A = _world(dim=dim, seed=0)
    plain = HolographicMind(dim, [f"a{j}" for j in range(A)], k=8, epsilon=0.2,
                            novelty_bonus=0.2, seed=0)
    fresh = HolographicMind(dim, [f"a{j}" for j in range(A)], k=8, epsilon=0.2,
                            novelty_bonus=0.2, seed=0, maintain=True,
                            surprise_floor=0.35, redundancy_floor=0.3, maintain_gap=400)
    # learn the task
    _train(plain, np.random.default_rng(1), state, best, C, 2500)
    _train(fresh, np.random.default_rng(1), state, best, C, 2500)
    # the world shifts: every situation's best action changes
    best[:] = [(x + 1) % A for x in best]
    _train(plain, np.random.default_rng(2), state, best, C, 3000)
    _train(fresh, np.random.default_rng(2), state, best, C, 3000)

    plain_acc = _acc(plain, np.random.default_rng(3), state, best, C)
    fresh_acc = _acc(fresh, np.random.default_rng(3), state, best, C)
    assert plain_acc < 0.5                     # plain brain stuck on the old policy
    assert fresh_acc >= 0.9                     # self-maintaining brain recovered
    assert fresh.reorganizations >= 1           # it decided to do so on its own
    assert fresh.prototype_count() < plain.prototype_count()   # and stayed lean


def test_maintain_off_is_unchanged_and_never_reorganizes():
    dim = 256
    rng, state, best, C, A = _world(dim=dim, seed=5)
    brain = HolographicMind(dim, [f"a{j}" for j in range(A)], k=8, epsilon=0.2,
                            novelty_bonus=0.2, seed=5)   # maintain defaults to False
    _train(brain, np.random.default_rng(6), state, best, C, 1500)
    assert brain.reorganizations == 0           # default path does not self-maintain
    assert brain.surprise == 0.0                # and tracks no surprise
    assert _acc(brain, np.random.default_rng(7), state, best, C) >= 0.9   # still learns


def test_reorganize_folds_duplicates_without_losing_the_policy():
    dim = 256
    rng, state, best, C, A = _world(dim=dim, seed=8)
    brain = HolographicMind(dim, [f"a{j}" for j in range(A)], k=8, epsilon=0.2,
                            novelty_bonus=0.2, seed=8)
    _train(brain, np.random.default_rng(9), state, best, C, 2500)
    before_acc = _acc(brain, np.random.default_rng(10), state, best, C)
    before_n = brain.prototype_count()
    b, a = brain.reorganize(duplicate=0.85)
    assert (b, a) == (before_n, brain.prototype_count())
    assert brain.prototype_count() < before_n          # duplicates were folded
    after_acc = _acc(brain, np.random.default_rng(10), state, best, C)
    assert after_acc >= before_acc - 0.05              # policy preserved


def test_autonomous_brain_recovers_from_shift_with_no_thresholds():
    # maintain='auto' sets NO behavioural thresholds (no surprise/redundancy floor,
    # no fixed fold grain). It must still recover from a regime shift and stay lean,
    # deciding purely by measuring its own decisions on held-out recent experience.
    dim = 256
    rng, state, best, C, A = _world(dim=dim, seed=0)
    plain = HolographicMind(dim, [f"a{j}" for j in range(A)], k=8, epsilon=0.2,
                            novelty_bonus=0.2, seed=0)
    auto = HolographicMind(dim, [f"a{j}" for j in range(A)], k=8, epsilon=0.2,
                           novelty_bonus=0.2, seed=0, maintain='auto')
    _train(plain, np.random.default_rng(1), state, best, C, 2000)
    _train(auto, np.random.default_rng(1), state, best, C, 2000)
    best[:] = [(x + 1) % A for x in best]
    _train(plain, np.random.default_rng(2), state, best, C, 2800)
    _train(auto, np.random.default_rng(2), state, best, C, 2800)

    plain_acc = _acc(plain, np.random.default_rng(3), state, best, C)
    auto_acc = _acc(auto, np.random.default_rng(3), state, best, C)
    assert plain_acc < 0.5                                  # no upkeep -> stuck
    assert auto_acc >= 0.9                                  # autonomous -> recovered
    assert auto.reorganizations >= 1                        # it acted on its own
    assert auto.prototype_count() < plain.prototype_count() # and stayed lean


def test_autonomous_brain_compresses_without_churn_when_stationary():
    dim = 256
    rng, state, best, C, A = _world(dim=dim, seed=4)
    plain = HolographicMind(dim, [f"a{j}" for j in range(A)], k=8, epsilon=0.2,
                            novelty_bonus=0.2, seed=4)
    auto = HolographicMind(dim, [f"a{j}" for j in range(A)], k=8, epsilon=0.2,
                           novelty_bonus=0.2, seed=4, maintain='auto')
    _train(plain, np.random.default_rng(5), state, best, C, 4000)
    _train(auto, np.random.default_rng(5), state, best, C, 4000)
    # stays accurate, and folds itself far leaner than the unmaintained brain
    assert _acc(auto, np.random.default_rng(6), state, best, C) >= 0.9
    assert auto.prototype_count() < plain.prototype_count() / 2


def test_autonomous_brain_refreshes_and_recovers_on_a_hard_noisy_shift():
    # A harder, noisy non-stationary problem: more situations, a graded/noisy reward,
    # and a small recent window -- the regime where the gate used to UNDER-fire, sitting
    # on stale prototypes because a half-old buffer right after the shift flatters them.
    # The fixed gate commits to a refresh as soon as recent decisions are better and
    # rebuilds it from the full recent window, so the brain recovers instead of crawling
    # back on online relearning alone.
    dim, C, A = 256, 60, 4
    rng = np.random.default_rng(0)
    base = [b / np.linalg.norm(b) for b in (rng.standard_normal(dim) for _ in range(C))]
    best = [int(rng.integers(A)) for _ in range(C)]

    def state(i):
        v = base[i] + 0.02 * rng.standard_normal(dim)
        return v / np.linalg.norm(v)

    def reward(i, a):
        return 1.0 if rng.random() < (0.8 if a == best[i] else 0.2) else 0.0

    def acc(b):
        return sum(b.decide(state(i), explore=False, epsilon=0.0) == best[i] for i in range(C)) / C

    def mk(**kw):
        return HolographicMind(dim, [f"a{j}" for j in range(A)], k=8, epsilon=0.2,
                               novelty_bonus=0.2, seed=0, buffer_cap=800, check_every=400, **kw)
    plain, auto = mk(), mk(maintain='auto')
    refreshed_after_shift, shift = 0, 4500
    for t in range(1, 9001):
        if t == shift:
            best = [(x + 2) % A for x in best]
        i = int(rng.integers(C)); s = state(i)
        for b in (plain, auto):
            a = b.decide(s, explore=True, epsilon=0.25)
            prev = b.reorganizations
            b.remember([s], [a], [reward(i, a)])
            if b is auto and b.reorganizations > prev and "refresh" in (b.last_choice or "") and t > shift:
                refreshed_after_shift += 1

    pa, aa = acc(plain), acc(auto)
    assert pa < 0.5                              # no upkeep -> stuck on the stale policy
    assert aa >= 0.6                             # autonomous brain recovered
    assert aa >= pa + 0.25                       # by a wide margin over the plain brain
    assert refreshed_after_shift >= 1            # it committed to a refresh, not just folds
    assert auto.prototype_count() < plain.prototype_count()   # and stayed lean

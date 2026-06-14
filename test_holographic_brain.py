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


def test_consolidate_projects_memory_at_behavioral_parity():
    # PROJECTION: thousands of 512-D prototypes are shadows of one low-rank
    # object (the span of the sense-atom vocabulary -- the overlap between
    # concepts is the registration mark). consolidate() discovers the subspace
    # by SVD and re-stores the memory in it. Measured at dim 512: rank ~24
    # (21x smaller), decide() ~5x faster, forage 122 -> 120 stars and 16x16
    # maze 90% -> 95% (parity). Fast pinned version at dim 256.
    import numpy as np
    from holographic_creature import (HolographicMind, CreatureEncoder,
                                      GridWorld, run_episode)
    dim = 256
    enc = CreatureEncoder(dim, seed=1)
    mind = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.45,
                           novelty_bonus=0.2, memory_cap=12000, seed=2)
    world = GridWorld(7, 7, n_poison=0, seed=3)
    for ep in range(120):
        mind.epsilon = max(0.05, 0.45 * (1 - ep / 120))
        run_episode(world, enc, mind, learn=True, explore=True, max_steps=80)

    def stars(n=8):
        out = []
        for _ in range(n):
            w = GridWorld(7, 7, n_poison=0, seed=3)
            run_episode(w, enc, mind, learn=False, explore=False,
                        eval_epsilon=0.05, max_steps=200)
            out.append(w.stars)
        return float(np.mean(out))

    before = stars()
    r = mind.consolidate()
    assert r is not None and r <= dim // 4          # genuinely low-rank
    after = stars()                                  # raw states in; perceive projects
    assert after >= 0.8 * before                     # behavioural parity floor


def test_consolidate_guard_expands_when_the_world_grows_structure():
    # THE SHADOW HAZARD, pinned: a brain consolidated in a poison-free world
    # leaves the danger sense nearly invisible (measured 4% in-basis energy) --
    # so the residual guard MUST trip when poison appears, grow the basis, and
    # make danger visible again (measured 4% -> 100%). The flux-guard pattern's
    # fourth appearance: compress when stable, expand at anomaly.
    import numpy as np
    from holographic_creature import (HolographicMind, CreatureEncoder,
                                      GridWorld, run_episode)
    dim = 256
    enc = CreatureEncoder(dim, seed=1)
    mind = HolographicMind(dim, GridWorld.ACTIONS, k=15, epsilon=0.45,
                           novelty_bonus=0.2, memory_cap=12000, seed=2)
    world = GridWorld(7, 7, n_poison=0, seed=3)
    for ep in range(100):
        mind.epsilon = max(0.05, 0.45 * (1 - ep / 100))
        run_episode(world, enc, mind, learn=True, explore=True, max_steps=80)
    r0 = mind.consolidate()
    danger = enc.encode({"danger_N": "yes"})
    inb0 = float(np.linalg.norm(danger @ mind._basis) ** 2
                 / np.linalg.norm(danger) ** 2)
    assert inb0 < 0.5                                # the shadow really hides it

    poison_world = GridWorld(7, 7, n_poison=2, seed=3)
    for ep in range(40):                             # live where the new structure is
        mind.epsilon = 0.2
        run_episode(poison_world, enc, mind, learn=True, explore=True,
                    max_steps=80, danger_reflex=True)
    inb1 = float(np.linalg.norm(danger @ mind._basis) ** 2
                 / np.linalg.norm(danger) ** 2)
    assert mind._basis.shape[1] > r0                 # the basis grew
    assert inb1 > 0.9                                # danger is visible again


def test_describe_decodes_states_and_why_differ_explains():
    # INTROSPECTION: the brain's states are role-bound sense bundles, so
    # describe() decodes them back (relations decode turned inward). Measured:
    # present roles 373/373 correct, absent roles 427/427 silent, with the 0.18
    # floor sitting in a real gap (present min 0.28 vs absent max 0.13).
    import numpy as np
    from holographic_creature import HolographicMind, CreatureEncoder, GridWorld
    enc = CreatureEncoder(512, seed=1)
    mind = HolographicMind(512, GridWorld.ACTIONS, seed=2)
    rng = np.random.default_rng(0)
    ok_pres = ok_abs = n_pres = n_abs = 0
    for _ in range(40):
        s = {"food_x": rng.choice(["east", "west", "none"]),
             "food_y": rng.choice(["north", "south", "none"])}
        for w in ("wall_N", "wall_S", "danger_E"):
            if rng.random() < 0.4:
                s[w] = "yes"
        d = mind.describe(enc.encode(s), enc)
        for role in ("food_x", "food_y", "wall_N", "wall_S", "danger_E"):
            if role in s:
                n_pres += 1
                ok_pres += (d.get(role, (None,))[0] == s[role])
            else:
                n_abs += 1
                ok_abs += (role not in d)
    assert ok_pres / n_pres >= 0.95
    assert ok_abs / n_abs >= 0.95
    # why_differ: the per-role verdict between two crafted states
    s1 = enc.encode({"food_x": "east", "wall_N": "yes"})
    s2 = enc.encode({"food_x": "west", "wall_N": "yes"})
    v = {r: (a, b, sh) for r, a, b, sh in mind.why_differ(s1, s2, enc)}
    assert v["food_x"] == ("east", "west", False)
    assert v["wall_N"] == ("yes", "yes", True)


# ---------------------------------------------------------------------------
# Capacity-aware layering and tiered/blind/online decision refinements
# ---------------------------------------------------------------------------

def test_capacity_caps_prototype_load_and_preserves_fidelity():
    # A prototype is a bundle with finite capacity: fold too many distinct members
    # into one and the unit stops resembling any of them. capacity= caps members per
    # prototype, splitting instead of blurring. Off (=0) blurs without bound.
    import numpy as np
    from holographic_creature import HolographicMind
    rng = np.random.default_rng(0)
    base = rng.standard_normal(256)
    members = [base + 0.01 * rng.standard_normal(256) for _ in range(40)]  # all merge-close

    capped = HolographicMind(dim=256, actions=["N", "S", "E", "W"], merge=0.5, capacity=8, seed=0)
    blurred = HolographicMind(dim=256, actions=["N", "S", "E", "W"], merge=0.5, capacity=0, seed=0)
    for s in members:
        capped.remember([s], [0], [1.0])
        blurred.remember([s], [0], [1.0])
    rc, rb = capped.capacity_report(), blurred.capacity_report()
    assert rc["max_count"] <= 8                       # capacity respected
    assert rc["overloaded"] == 0                      # nothing past the soft cap
    assert rb["max_count"] == 40 and rb["overloaded"] >= 1   # one over-loaded bundle


def test_capacity_off_is_unchanged():
    # default capacity=0 reproduces the old single-prototype folding exactly
    import numpy as np
    from holographic_creature import HolographicMind
    rng = np.random.default_rng(1)
    b = HolographicMind(dim=128, actions=["N", "S"], merge=0.5, seed=0)
    base = rng.standard_normal(128)
    for _ in range(20):
        b.remember([base + 0.01 * rng.standard_normal(128)], [0], [1.0])
    assert b.capacity_report()["max_count"] == 20     # all folded into one, as before


def test_soft_veto_waits_on_temporary_block():
    # tiered veto: boxed in by permanent walls + a temporary block on the goal side ->
    # wait on the soft (temporary) block rather than guessing among walls.
    import numpy as np
    from holographic_creature import HolographicMind
    b = HolographicMind(dim=256, actions=["N", "S", "E", "W"], seed=0)
    senses = {"goal_N": "yes", "traffic_N": "yes",
              "wall_E": "yes", "wall_S": "yes", "wall_W": "yes"}
    choice = b.decide(np.zeros(256), senses=senses, avoid=("wall",),
                      soft=("traffic", "red"), explore=False)
    assert b.actions[choice] == "N"


def test_soft_default_empty_is_unchanged():
    # soft=() (default) reproduces today's veto-and-lift behaviour: fully blocked ->
    # all actions back in play (no crash, a real choice returned).
    import numpy as np
    from holographic_creature import HolographicMind
    b = HolographicMind(dim=256, actions=["N", "S", "E", "W"], seed=0)
    senses = {"wall_N": "yes", "wall_E": "yes", "wall_S": "yes", "wall_W": "yes"}
    choice = b.decide(np.zeros(256), senses=senses, avoid=("wall",), explore=False)
    assert choice in range(4)


def test_blind_floor_follows_compass_when_lost():
    # blind-state compass: no memory anywhere + a goal token present -> follow the compass,
    # don't guess. Off by default (blind_floor=0.0).
    import numpy as np
    from holographic_creature import HolographicMind
    b = HolographicMind(dim=256, actions=["N", "S", "E", "W"], seed=0)
    assert b.blind_floor == 0.0                       # off by default
    b.blind_floor = 0.15
    choice = b.decide(np.zeros(256), senses={"goal_E": "yes"}, explore=False)
    assert b.actions[choice] == "E"


def test_penalize_recent_lowers_a_repeated_move():
    # online stuck-signal: a detected loop teaches itself -- the penalised
    # (state, action) gets its learned value lowered.
    import numpy as np
    from holographic_creature import HolographicMind
    rng = np.random.default_rng(2)
    b = HolographicMind(dim=256, actions=["N", "S", "E", "W"], maintain="auto", merge=0.5, seed=0)
    s = rng.standard_normal(256)
    b.remember([s], [0], [1.0])
    before, _ = b.value(b.perceive_vec(s), 0)
    hit = b.penalize_recent(amount=1.0, n=4)
    after, _ = b.value(b.perceive_vec(s), 0)
    assert hit >= 1 and after < before


def test_penalize_recent_noop_without_buffer():
    # without a recent-experience buffer (maintain off) it's a safe no-op
    from holographic_creature import HolographicMind
    b = HolographicMind(dim=64, actions=["N", "S"], seed=0)
    assert b.penalize_recent() == 0

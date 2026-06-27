"""The creature brain's batched-scoring API (value_batch / _value_projected) and the maintenance
BUDGET knob (auto_maintain grains/refresh). The load-bearing property is BIT-IDENTITY: the
creature is tie-sensitive (a 1e-16 difference at the top-k boundary flips a maze trajectory --
the same hazard that kept bind_batch out of the encoder), so value_batch must reproduce the
value() loop exactly, not approximately. It does -- it just moves the basis projection out of the
per-action loop (one projection, not N) and drops the width-check branch. The budget knob defaults
to the full 8-way search (unchanged); only an explicit lean setting trims candidates."""
import numpy as np
from holographic_creature import GridWorld, CreatureEncoder, HolographicMind, run_episode


def _trained_brain(dim=128, episodes=40, seed=2):
    world = GridWorld(width=7, height=7, n_poison=2, seed=0)
    enc = CreatureEncoder(dim, seed=1)
    mind = HolographicMind(dim, actions=["N", "S", "E", "W"], seed=seed, maintain='auto')
    for _ in range(episodes):
        run_episode(world, enc, mind, learn=True, explore=True, max_steps=40, mem=0)
    # gather a fixed set of states to score, plus a recent-experience buffer big enough that
    # auto_maintain runs (it returns early when the held-out slice is < 20)
    states = []
    buf = []
    senses = world.reset(); s = enc.build_state(senses, [], 0)
    for _ in range(400):
        if np.linalg.norm(s) > 1e-9:
            states.append(s.copy())
        a = mind.decide(s, explore=True, epsilon=0.3)
        a = mind.actions.index(a) if isinstance(a, str) else a
        senses, r, ate, alive = world.step(mind.actions[a])
        if np.linalg.norm(s) > 1e-9:
            buf.append((mind.perceive_vec(s), a, float(r)))
        s = enc.build_state(senses, [], 0)
        if not alive:
            senses = world.reset(); s = enc.build_state(senses, [], 0)
    return mind, states, buf[-150:]


def test_value_batch_is_bit_identical_to_value_loop():
    """value_batch must reproduce a per-action value() loop EXACTLY (tie-sensitivity demands it)."""
    mind, states, buf = _trained_brain()
    for st in states:
        sv = mind.perceive_vec(st)
        vb_v, vb_s = mind.value_batch(sv)
        for a in range(4):
            v, sp = mind.value(sv, a)
            assert vb_v[a] == v and vb_s[a] == sp        # bit-for-bit, not approximately


def test_value_batch_bit_identical_on_consolidated_raw_states():
    """The one real saving: a consolidated brain handed a RAW state projects ONCE in value_batch
    instead of once per action -- and still matches the per-action value() loop bit-for-bit."""
    mind, states, buf = _trained_brain()
    mind.consolidate()
    for st in states[:120]:
        vb_v, vb_s = mind.value_batch(st)                # raw full-dim -> projected once inside
        for a in range(4):
            v, sp = mind.value(st, a)                     # each value() projects independently
            assert vb_v[a] == v and vb_s[a] == sp


def test_value_projected_matches_value_on_projected_state():
    """_value_projected is value() minus the basis-width branch; identical on an already-projected
    (or raw un-consolidated) state."""
    mind, states, buf = _trained_brain()
    for st in states:
        sv = mind.perceive_vec(st)
        for a in range(4):
            assert mind._value_projected(sv, a) == mind.value(sv, a)


def test_value_batch_action_subset():
    """Scoring a subset of actions returns aligned arrays for exactly those actions."""
    mind, states, buf = _trained_brain()
    sv = mind.perceive_vec(states[0])
    vals, sups = mind.value_batch(sv, action_idxs=[1, 3])
    assert vals.shape == (2,) and sups.shape == (2,)
    assert vals[0] == mind.value(sv, 1)[0] and vals[1] == mind.value(sv, 3)[0]


def test_maintain_budget_default_is_full_search():
    """The budget knob defaults to the full 8-way search: a default auto_maintain() and an explicit
    full-grain/refresh call must reach the SAME decision from the same buffer."""
    import copy
    mind, _, buf = _trained_brain()
    a = copy.deepcopy(mind); a._buf = list(buf); a.auto_maintain()
    b = copy.deepcopy(mind); b._buf = list(buf)
    b.auto_maintain(grains=(0.9, 0.82, 0.75), refresh=True)
    assert a.last_choice == b.last_choice
    assert a.prototype_count() == b.prototype_count()


def test_maintain_budget_lean_runs_and_trims_candidates():
    """A lean budget (fewer grains, no refresh) still produces a valid memory and never adopts a
    refresh (the family is absent), so a stable-deployment tick is cheap and forget-free."""
    import copy
    mind, _, buf = _trained_brain()
    m = copy.deepcopy(mind); m._buf = list(buf)
    m.auto_maintain(grains=(0.82,), refresh=False)
    assert m.last_choice in ("keep", "fold@0.82")        # only preserving options were offered
    assert m.prototype_count() >= 1
    # keep-only budget: a single candidate, must be a no-op 'keep'
    m2 = copy.deepcopy(mind); m2._buf = list(buf)
    m2.auto_maintain(grains=(), refresh=False)
    assert m2.last_choice == "keep"


def test_creature_plan_bakes_a_corridor():
    """Phase-0 migration: a creature can bake a corridor plan via the shared planning module (the same
    capability UnifiedMind.plan has), without touching its value/decide path."""
    import numpy as np
    from holographic_creature import HolographicMind
    rng = np.random.default_rng(0)
    mind = HolographicMind(1024, actions=["N", "S", "E", "W"], seed=0)
    tiles = rng.standard_normal((11, 1024)); tiles /= np.linalg.norm(tiles, axis=1, keepdims=True)
    def field_step(cur):
        i = int(np.argmax(tiles @ (cur / (np.linalg.norm(cur) + 1e-12))))
        return tiles[i + 1] if i + 1 < len(tiles) else None
    p = mind.plan(tiles[0], field_step, max_steps=10, floor=0.12, action_of=lambda a, b: "go")
    assert p.route == list(range(1, 11)) and len(p.actions) == 10
    assert mind.replan_needed(p, 0, floor=0.12) is False
    assert mind.replan_needed(p, len(p.route), floor=0.12) is True

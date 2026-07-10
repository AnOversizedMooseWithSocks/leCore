"""X9 (fat-margin caching for a drifting query) + X11 (the escalation ladder).

X9 is Catto's enlarged AABB read as a cache policy: a query that DRIFTS should be served from a baked region, not
re-keyed exactly. X11 is the dispatcher that makes his "4 substeps" dial and our closed form two ends of one axis.

KEPT NEGATIVE pinned here: X9 is NOT the sleep tracker's two-threshold hysteresis. A margin cache has exactly ONE
radius -- a cache entry has no state to hover at a bar and flicker between, so an inner threshold is never read.
I predicted otherwise before measuring; the prediction was wrong and the test says so.
"""

import numpy as np
import pytest

from holographic.caching_and_storage.holographic_cachehome import (
    MarginCache, drift_scale, replay_margin, suggest_margin)
from holographic.simulation_and_physics.holographic_modal import escalation_plan
from holographic.simulation_and_physics.holographic_island import SleepTracker


def _walk(n=400, d=2, seed=0):
    """A drifting query: a unit-step random walk -- a camera, a cursor, an agent, a recall neighbourhood."""
    return np.cumsum(np.random.default_rng(seed).normal(size=(n, d)), axis=0)


# ============================================================================================
# X9 -- fat-margin caching
# ============================================================================================

def test_selftest_runs():
    from holographic.caching_and_storage import holographic_cachehome as mod
    mod._selftest()


def test_exact_key_caching_never_hits_on_a_drifting_query():
    # The premise. margin=0 means "key on the exact query", and a drifting query is never twice the same.
    st = replay_margin(_walk(), 0.0)
    assert st["rebuilds"] == 400 and st["hits"] == 0 and st["hit_rate"] == 0.0


def test_the_margin_trade_is_monotone_and_reproduces_the_measured_table():
    q = _walk()
    table = {m: replay_margin(q, m) for m in (0.0, 1.0, 3.0, 6.0)}
    hits = [table[m]["hits"] for m in (0.0, 1.0, 3.0, 6.0)]
    assert hits == sorted(hits)                                  # more margin, never fewer hits
    rebuilds = [table[m]["rebuilds"] for m in (0.0, 1.0, 3.0, 6.0)]
    assert rebuilds == sorted(rebuilds, reverse=True)            # ... and never more rebuilds

    # the measured numbers on THIS stream (they move with the drift scale -- reported, not universal)
    assert table[0.0]["rebuilds"] == 400
    assert table[6.0]["rebuilds"] == 20
    assert table[6.0]["hit_rate"] == pytest.approx(0.95, abs=0.005)


def test_drift_scale_is_the_variation_probe_on_the_query_stream():
    q = _walk()
    assert drift_scale(q) == pytest.approx(float(np.mean(np.linalg.norm(np.diff(q, axis=0), axis=1))), abs=1e-12)
    assert drift_scale([np.zeros(2)]) == 0.0                     # fewer than two queries: no drift
    assert drift_scale([np.zeros(2)] * 5) == 0.0                 # a stream that never moves


def test_suggest_margin_meets_the_target_and_is_not_wastefully_large():
    q = _walk()
    m = suggest_margin(q, target_hit_rate=0.9)
    assert replay_margin(q, m)["hit_rate"] >= 0.9                # meets it ...
    assert replay_margin(q, m * 0.5)["hit_rate"] < 0.9           # ... and half of it does not
    assert suggest_margin(q, 0.9) == suggest_margin(q, 0.9)      # deterministic (fixed bisection count)
    assert suggest_margin([np.zeros(2)] * 5) == 0.0              # a still query needs no margin


def test_suggest_margin_reports_its_ceiling_rather_than_silently_clamping():
    # A target no margin in the search bound can reach must come back as the bound, not as a lie.
    q = _walk(n=50)
    huge = suggest_margin(q, target_hit_rate=1.0, max_multiple=0.01)
    assert huge == pytest.approx(0.01 * drift_scale(q))
    assert replay_margin(q, huge)["hit_rate"] < 1.0              # honest: it did NOT meet the target


def test_the_cache_bakes_once_per_rebuild_and_serves_the_rest():
    q = _walk()
    calls = {"n": 0}

    def build(p):
        calls["n"] += 1
        return ("baked", tuple(np.round(np.asarray(p, float), 6)))

    mc = MarginCache(build, margin=6.0)
    values = [mc.get(x) for x in q]
    assert calls["n"] == mc.rebuilds                             # one bake per rebuild, never a spare
    assert mc.rebuilds == 20 and mc.hits == 380
    assert mc.stats()["hit_rate"] == pytest.approx(0.95, abs=0.005)
    assert all(v[0][0] == "baked" for v in values)
    assert values[1][1] is True                                  # the second query hit the first bake


def test_the_cache_accepts_any_metric_so_it_serves_hypervectors_too():
    # SIDEWAYS: the same policy on a cosine-distance recall neighbourhood, not just a Euclidean camera.
    rng = np.random.default_rng(1)
    base = rng.normal(size=64)
    stream = [base + 0.02 * rng.normal(size=64) for _ in range(50)]
    cosine = lambda a, b: 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    tight = replay_margin(stream, 1e-6, metric=cosine)
    loose = replay_margin(stream, 0.05, metric=cosine)
    assert tight["rebuilds"] == 50                               # exact key: always a miss
    assert loose["hits"] > 40                                    # a fat cosine margin serves the neighbourhood


def test_kept_negative_a_margin_cache_has_one_radius_not_a_hysteresis_band():
    # I predicted X9 would reuse SleepTracker's two-threshold band. It does not, and here is the difference.
    q = _walk()

    # SleepTracker NEEDS two bars: with one, an energy sitting at the bar flickers between two STATES.
    bar = 1e-8
    noisy = [bar * (1.0 + 0.5 * ((-1) ** k)) for k in range(12)]
    one = SleepTracker(sleep_energy=bar, sleep_frames=1, wake_energy=bar)
    states = [one.update(0, e) for e in noisy]
    assert sum(1 for a, b in zip(states, states[1:]) if a != b) >= 8      # it thrashes

    # A margin cache has NO such state: a query is inside the region or it is not, evaluated fresh each time.
    # So the replay is a pure function of (stream, radius) -- there is nothing an inner threshold could damp.
    assert replay_margin(q, 3.0) == replay_margin(q, 3.0)
    assert replay_margin(q, 3.0)["rebuilds"] > replay_margin(q, 6.0)["rebuilds"]


def test_margin_cache_validates_its_dial():
    with pytest.raises(ValueError):
        MarginCache(lambda p: p, margin=-1.0)


# ============================================================================================
# X11 -- the escalation ladder
# ============================================================================================

def test_the_ladder_is_ordered_sleep_then_defect_then_breakeven():
    # An asleep island is never asked whether it is diagonalizable; a defective one is never asked whether the
    # jump would pay. The ORDER of the tests is the ladder, and the order is the contract.
    assert escalation_plan(24, 3840, energy=0.0, sleep_energy=1e-8)["rung"] == "sleep"
    assert escalation_plan(24, 3840, energy=0.0, sleep_energy=1e-8, diagonalizable=False)["rung"] == "sleep"
    assert escalation_plan(24, 16, energy=0.0, sleep_energy=1e-8)["rung"] == "sleep"

    assert escalation_plan(24, 3840, diagonalizable=False)["rung"] == "substep"   # defect beats break-even
    assert escalation_plan(24, 3840)["rung"] == "jump"
    assert escalation_plan(24, 16)["rung"] == "substep"


def test_the_ladder_reports_a_reason_and_catto_s_substep_count():
    sleep = escalation_plan(24, 3840, energy=0.0, sleep_energy=1e-8)
    assert sleep["substeps"] == 0 and "fixed point" in sleep["why"]

    jump = escalation_plan(24, 3840)
    assert jump["substeps"] == 0 and "eigendecomposition" in jump["why"]

    sub = escalation_plan(24, 4)
    assert sub["substeps"] == 4 and "amortize" in sub["why"]      # the substep rung carries the count

    defect = escalation_plan(24, 3840, diagonalizable=False)
    assert defect["substeps"] == 3840 and "defective" in defect["why"]


def test_the_ladder_agrees_with_the_gates_it_dispatches_over():
    from holographic.simulation_and_physics.holographic_modal import should_jump
    for dim in (4, 24, 100):
        for k in (1, 16, 480, 3840):
            plan = escalation_plan(dim, k, energy=1.0, sleep_energy=0.0)
            assert (plan["rung"] == "jump") == should_jump(dim, k)   # one gate, not two


def test_a_loud_island_is_never_put_to_sleep():
    assert escalation_plan(24, 3840, energy=1.0, sleep_energy=1e-8)["rung"] != "sleep"
    assert escalation_plan(24, 3840, energy=None)["rung"] != "sleep"   # no energy given: never assume rest


def test_x9_and_x11_are_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    q = _walk()

    assert m.drift_scale(q) == pytest.approx(drift_scale(q))
    assert m.replay_margin(q, 6.0)["rebuilds"] == 20
    marg = m.suggest_margin(q, 0.9)
    assert m.replay_margin(q, marg)["hit_rate"] >= 0.9

    mc = m.margin_cache(lambda p: "v", margin=6.0)
    for x in q:
        mc.get(x)
    assert mc.stats()["hits"] == 380

    assert m.escalation_plan(24, 3840, energy=1.0, sleep_energy=1e-8)["rung"] == "jump"
    assert m.escalation_plan(24, 16)["rung"] == "substep"

    assert "Fat-margin" in str(m.find_capability("cache a result for a query that keeps moving slightly")[:3])
    assert "Modal jump" in str(m.find_capability("choose how many substeps to use")[:3])


def test_cross_faculty_the_ladder_reads_the_islands_own_sleep_probe():
    # X11 meets X3: the ladder's `energy` argument is exactly `island_energy`, and its sleep bar is the tracker's.
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    at_rest = np.zeros((6, 3))
    moving = np.ones((6, 3))
    assert m.escalation_plan(36, 3840, energy=m.island_energy(at_rest), sleep_energy=1e-8)["rung"] == "sleep"
    assert m.escalation_plan(36, 3840, energy=m.island_energy(moving), sleep_energy=1e-8)["rung"] == "jump"

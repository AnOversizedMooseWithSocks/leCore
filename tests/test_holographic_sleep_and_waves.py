"""C1 + C2 -- island sleep in the softbody solver, and the coordinator's deterministic wave schedule.

C1  A sleeping island is AT ITS FIXED POINT. Skipping it is bit-identical to stepping it, and the saving is exactly
    the awake fraction -- `body.last_step_stats` COUNTS it rather than asserting it.

C2  Two tasks in one colour touch disjoint resources, so a wave runs with no locks and no atomics, in a
    reproducible order. MEASURED: 2,000 transactions over 300 keys -> 24 waves, 83.3x lock-free parallelism.

Both are default-off / opt-in, and both come with the negative that bounds them:
  * a sleeping island CANNOT wake itself (its velocity is never integrated) -- an external event must;
  * colouring CANNOT invent parallelism -- if everything conflicts it honestly serialises.
"""

import numpy as np
import pytest

from holographic.scene_and_pipeline.holographic_coordinator import Coordinator
from holographic.simulation_and_physics.holographic_island import SleepTracker, color_waves, conflict_graph
from holographic.simulation_and_physics.holographic_softbody import SoftBody


def _four_chains(moving_speed=3.0):
    """Four disconnected 3-node chains -- four islands. Only the first is given a velocity."""
    x = np.zeros((12, 3))
    for k in range(4):
        for n in range(3):
            x[k * 3 + n] = [n * 0.5, k * 2.0, 0.0]
    sb = SoftBody(x)
    for k in range(4):
        sb.add_distance(k * 3, k * 3 + 1, 0.5)
        sb.add_distance(k * 3 + 1, k * 3 + 2, 0.5)
    sb.v[0:3] = [moving_speed, 0.0, 0.0]
    return sb


# ===========================================================================================
# C1 -- island sleep
# ===========================================================================================

def test_the_body_knows_its_islands():
    sb = SoftBody(np.zeros((6, 3)))
    sb.add_distance(0, 1)
    sb.add_distance(1, 2)
    sb.add_distance(4, 5)
    assert sb.islands() == [[0, 1, 2], [3], [4, 5]]             # node 3 is its own singleton island

    # cached, and invalidated when the constraint graph changes
    first = sb.islands()
    assert sb.islands() is first
    sb.add_distance(2, 3)
    assert sb.islands() == [[0, 1, 2, 3], [4, 5]]


def test_only_the_moving_island_is_awake_and_the_saving_is_counted():
    sb = _four_chains()
    tracker = SleepTracker(sleep_energy=1e-8, sleep_frames=1)
    sb.step(dt=1 / 60.0, gravity=np.zeros(3), sleep=tracker)

    st = sb.last_step_stats
    assert st["islands"] == 4 and st["awake"] == 1 and st["asleep"] == 3
    # the saving is EXACTLY the awake fraction: 2 of 8 constraints touched
    assert st["constraints_total"] == 8 and st["constraints_solved"] == 2
    assert st["constraints_solved"] / st["constraints_total"] == pytest.approx(st["awake"] / st["islands"])


def test_a_sleeping_island_is_carried_through_bit_identically():
    # THE CORRECTNESS CONTRACT. Not "to 1e-12" -- bit for bit, because skipping must not perturb.
    sb = _four_chains()
    start = sb.x.copy()
    tracker = SleepTracker(sleep_energy=1e-8, sleep_frames=1)
    for _ in range(5):
        sb.step(dt=1 / 60.0, gravity=np.zeros(3), sleep=tracker)
    assert np.array_equal(sb.x[3:], start[3:])                  # the three sleeping chains never moved
    assert np.array_equal(sb.v[3:], np.zeros((9, 3)))


def test_the_awake_island_matches_the_no_sleep_reference():
    # Sleeping the others must not change the answer for the one that is awake.
    a, b = _four_chains(), _four_chains()
    tracker = SleepTracker(sleep_energy=1e-8, sleep_frames=1)
    for _ in range(5):
        a.step(dt=1 / 60.0, gravity=np.zeros(3), sleep=tracker)
        b.step(dt=1 / 60.0, gravity=np.zeros(3))
    assert np.allclose(a.x[:3], b.x[:3], atol=1e-12)


def test_sleep_is_default_off_and_bit_identical_when_absent():
    a, b = _four_chains(), _four_chains()
    for _ in range(3):
        a.step(dt=1 / 60.0, gravity=np.array([0.0, -9.81, 0.0]))
        b.step(dt=1 / 60.0, gravity=np.array([0.0, -9.81, 0.0]), sleep=None)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.v, b.v)


def test_a_body_that_is_all_moving_sleeps_nothing():
    sb = _four_chains()
    sb.v[:] = [1.0, 0.0, 0.0]                                   # every island has kinetic energy
    tracker = SleepTracker(sleep_energy=1e-8, sleep_frames=1)
    sb.step(dt=1 / 60.0, gravity=np.zeros(3), sleep=tracker)
    st = sb.last_step_stats
    assert st["awake"] == 4 and st["asleep"] == 0
    assert st["constraints_solved"] == st["constraints_total"]  # no saving, and it says so


def test_kept_negative_a_sleeping_island_cannot_wake_itself():
    # Its velocity is never integrated, so nothing inside it can change. That is not an oversight -- it is Catto's
    # design (contact-begin wakes a body). An external event must say so, and `wake_all` is that door.
    sb = _four_chains()
    tracker = SleepTracker(sleep_energy=1e-8, sleep_frames=1)
    for _ in range(3):
        sb.step(dt=1 / 60.0, gravity=np.array([0.0, -9.81, 0.0]), sleep=tracker)
    assert sb.last_step_stats["asleep"] == 3                     # gravity did NOT wake them

    frozen = sb.x[3:].copy()
    for _ in range(3):
        sb.step(dt=1 / 60.0, gravity=np.array([0.0, -9.81, 0.0]), sleep=tracker)
    assert np.array_equal(sb.x[3:], frozen)                      # ... and they stay put, forever

    sb.wake_all()
    sb.step(dt=1 / 60.0, gravity=np.array([0.0, -9.81, 0.0]), sleep=tracker)
    assert sb.last_step_stats["awake"] == 4                      # ... until something wakes them
    assert not np.array_equal(sb.x[3:], frozen)                  # and then gravity bites


def test_hysteresis_is_mandatory_and_the_tracker_supplies_it():
    # A single sleep threshold thrashes on float noise at the bar. Pinned in the island module; re-asserted here
    # because a softbody using a single-threshold tracker would flicker its islands every frame.
    bar = 1e-8
    noisy = [bar * (1.0 + 0.5 * ((-1) ** k)) for k in range(12)]

    def flips(tr):
        states = [tr.update(0, e) for e in noisy]
        return sum(1 for a, b in zip(states, states[1:]) if a != b)

    assert flips(SleepTracker(sleep_energy=bar, sleep_frames=1, wake_energy=bar)) >= 8
    assert flips(SleepTracker(sleep_energy=bar, sleep_frames=1)) <= 1


# ===========================================================================================
# C2 -- the coordinator's wave schedule
# ===========================================================================================

def test_run_waves_reproduces_the_measured_workload():
    rng = np.random.default_rng(0)
    tx = [tuple(int(x) for x in rng.choice(300, 2, replace=False)) for _ in range(2000)]
    c = Coordinator()
    try:
        results, info = c.run_waves(tx, keys_of=set, worker=lambda item, cache: sum(item))
    finally:
        c.close()
    assert info["waves"] == 24
    assert info["parallelism"] == pytest.approx(83.3, abs=0.1)
    assert sum(info["wave_sizes"]) == 2000


def test_results_come_back_in_item_order_not_wave_order():
    # The schedule is an implementation detail; the answer is not.
    items = [("a",), ("a", "b"), ("c",), ("b",)]
    c = Coordinator()
    try:
        results, _ = c.run_waves(items, keys_of=set, worker=lambda item, cache: len(item))
    finally:
        c.close()
    assert results == [1, 2, 1, 1]


def test_every_wave_is_conflict_free():
    rng = np.random.default_rng(1)
    tasks = [set(int(k) for k in rng.choice(60, 2, replace=False)) for _ in range(200)]
    n, edges = conflict_graph(tasks)
    for wave in color_waves(n, edges):
        seen = set()
        for i in wave:
            assert not (seen & tasks[i]), "two items in one wave share a resource"
            seen |= tasks[i]


def test_the_schedule_is_deterministic():
    items = [("a", "b"), ("b", "c"), ("d",), ("a",), ("e", "f")]
    c = Coordinator()
    try:
        r1, i1 = c.run_waves(items, keys_of=set, worker=lambda it, ca: it)
        r2, i2 = c.run_waves(items, keys_of=set, worker=lambda it, ca: it)
    finally:
        c.close()
    assert r1 == r2 and i1 == i2
    assert i1["wave_sizes"] == [3, 2]                            # greedy ascending: the exact schedule


def test_kept_negative_colouring_cannot_invent_parallelism():
    same = [("k",)] * 5
    c = Coordinator()
    try:
        _, info = c.run_waves(same, keys_of=set, worker=lambda it, ca: 1)
    finally:
        c.close()
    assert info["waves"] == 5 and info["wave_sizes"] == [1] * 5   # honest serialisation
    assert info["parallelism"] == pytest.approx(1.0)


def test_disjoint_work_collapses_to_one_wave():
    items = [("a",), ("b",), ("c",)]
    c = Coordinator()
    try:
        _, info = c.run_waves(items, keys_of=set, worker=lambda it, ca: 1)
    finally:
        c.close()
    assert info["waves"] == 1 and info["parallelism"] == 3.0


def test_run_waves_publishes_the_shared_cache_once():
    seen = []
    c = Coordinator()
    try:
        c.run_waves([("a",), ("b",)], keys_of=set, worker=lambda it, cache: seen.append(cache), cache={"shared": 1})
    finally:
        c.close()
    assert len(seen) == 2 and all(s == seen[0] for s in seen)


# ===========================================================================================
# wiring
# ===========================================================================================

def test_both_are_wired_to_the_mind():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    tracker = m.softbody_sleep_tracker(sleep_energy=1e-8, sleep_frames=1)
    sb = _four_chains()
    sb.step(dt=1 / 60.0, gravity=np.zeros(3), sleep=tracker)
    assert sb.last_step_stats["awake"] == 1

    results, info = m.run_waves([("a", "b"), ("b",), ("c",)], keys_of=set, worker=lambda it, ca: len(it))
    assert results == [2, 1, 1] and info["waves"] == 2


def test_the_lint_records_both_clients_as_wired():
    import os
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "tools"))
    from unifiers import PENDING, cites

    assert ("island.color_waves (deterministic lock-free scheduling)", "holographic_coordinator") not in PENDING
    assert ("island.SleepTracker (solve only what moves)", "holographic_softbody") not in PENDING
    assert cites("holographic_coordinator", "island.color_waves (deterministic lock-free scheduling)", repo)
    assert cites("holographic_softbody", "island.SleepTracker (solve only what moves)", repo)

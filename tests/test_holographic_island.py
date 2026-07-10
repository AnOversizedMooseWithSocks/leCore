"""X3 -- island decomposition + the sleep probe (Box3D lesson B3).

The claims under test, each with its baseline:
  1. components are deterministic and independent of edge order/direction  (a farm must not disagree with itself)
  2. skipping a sleeping island is BIT-IDENTICAL to not having touched it  (the correctness contract)
  3. sleep IS the closed form: settle_island == iterate.limit, exactly, and the fixed point is NOT rest
  4. hysteresis is load-bearing: one threshold flickers on float noise, two do not  (kept negative)
  5. the cost saving is exactly the awake fraction  (the honest win, counted not asserted)
"""

import numpy as np
import pytest

from holographic.simulation_and_physics.holographic_island import (
    connected_components, island_energy, SleepTracker, step_islands, settle_island)
from holographic.misc.holographic_iterate import step_k


def test_selftest_runs():
    from holographic.simulation_and_physics import holographic_island as mod
    mod._selftest()


def test_components_are_deterministic_and_order_independent():
    edges = [(0, 1), (1, 2), (4, 5)]
    want = [[0, 1, 2], [3], [4, 5]]                 # ordered by smallest member; singleton 3 kept
    assert connected_components(6, edges) == want
    assert connected_components(6, list(reversed(edges))) == want
    assert connected_components(6, [(1, 0), (2, 1), (5, 4)]) == want      # direction-independent
    assert connected_components(6, edges + [(0, 0)]) == want              # self-loops connect nothing
    assert connected_components(3, []) == [[0], [1], [2]]                 # no edges: all singletons


def test_mesh_components_delegate_to_the_generic_flood_fill():
    # GENERALIZE ON CONTACT: the mesh-only counter now delegates, so a mesh shell and a constraint island
    # are literally the same computation. Two disjoint triangles => 2 components.
    from holographic.scene_and_pipeline.holographic_route import connected_components as mesh_cc

    class _M:
        n_vertices = 6
        faces = [(0, 1, 2), (3, 4, 5)]
    assert mesh_cc(_M()) == 2
    _M.faces = [(0, 1, 2), (2, 3, 4)]               # sharing vertex 2 => one shell, vertex 5 isolated
    assert mesh_cc(_M()) == 2


def test_skipping_a_sleeping_island_is_bit_identical():
    # THE CORRECTNESS CONTRACT. A false 'asleep' freezes a moving body and nothing downstream can detect it,
    # so a sleeping island's rows must be carried through untouched -- not 'to 1e-12', bit for bit.
    rng = np.random.default_rng(0)
    comps = connected_components(6, [(0, 1), (1, 2), (4, 5)])
    state = np.zeros((6, 3))
    state[0:3] = rng.normal(size=(3, 3))            # island 0 is loud; islands 1 and 2 are exactly at rest
    tr = SleepTracker(sleep_energy=1e-8, sleep_frames=1)

    step_islands(state, comps, lambda s: s + 1.0, tracker=tr)          # frame 1: accumulate quiet frames
    new, awake, asleep = step_islands(state, comps, lambda s: s + 1.0, tracker=tr)
    assert awake == [0] and asleep == [1, 2]
    assert np.array_equal(new[3:], state[3:])                          # untouched, bit for bit
    assert np.array_equal(new[0:3], state[0:3] + 1.0)                  # the awake island really stepped


def test_tracker_none_is_the_old_behaviour_exactly():
    # BACKWARD COMPATIBILITY: no tracker => everything awake => identical to stepping every island.
    comps = connected_components(4, [(0, 1)])
    state = np.zeros((4, 2))
    new, awake, asleep = step_islands(state, comps, lambda s: s + 2.0, tracker=None)
    assert asleep == [] and awake == [0, 1, 2]
    assert np.array_equal(new, state + 2.0)


def test_sleep_is_the_closed_form_and_the_fixed_point_is_not_rest():
    # SLEEP IS iterate.limit(). And the measured surprise, kept loud: the fixed point of a diffusive island
    # is its MEAN, not zero -- modes with |eigenvalue| ~ 1 persist. Only a strictly contractive island rests.
    rng = np.random.default_rng(0)
    n = 64
    U = np.zeros(n)
    U[0], U[1], U[-1] = 0.90, 0.05, 0.05            # taps sum to 1 => DC mode has |eigenvalue| == 1
    x0 = rng.normal(size=n)

    lim = settle_island(x0, U)
    assert np.allclose(lim, x0.mean())                          # settles to the MEAN, not to rest
    assert np.abs(lim).max() > 1e-3                             # ... so it is emphatically NOT zero

    assert np.linalg.norm(step_k(x0, U, 100) - lim) > 1.0       # 100 steps: nowhere near
    assert np.linalg.norm(step_k(x0, U, 100_000) - lim) < 1e-9  # stepping only APPROACHES the closed form

    Uc = np.zeros(n)
    Uc[0] = 0.5                                                 # strictly contractive
    assert np.abs(settle_island(x0, Uc)).max() < 1e-12          # ... this one really does settle to rest

    with pytest.raises(ValueError):                             # a diverging island has no limit; say so
        Ud = np.zeros(n)
        Ud[0] = 1.5
        settle_island(x0, Ud)


def test_kept_negative_one_threshold_flickers_two_do_not():
    bar = 1e-8
    noisy = [bar * (1.0 + 0.5 * ((-1) ** k)) for k in range(12)]   # straddles the bar every frame

    def flips(tracker):
        states = [tracker.update(0, e) for e in noisy]              # update() has side effects: once per frame
        return sum(1 for a, b in zip(states, states[1:]) if a != b)

    one = flips(SleepTracker(sleep_energy=bar, sleep_frames=1, wake_energy=bar))
    two = flips(SleepTracker(sleep_energy=bar, sleep_frames=1))     # default hysteresis band = 4x
    assert one >= 8                                                 # the single threshold thrashes
    assert two <= 1                                                 # at most the one legitimate transition


def test_an_impulse_wakes_an_island_the_same_frame():
    tr = SleepTracker(sleep_energy=1e-8, sleep_frames=1)
    assert tr.update(7, 0.0) is True and tr.asleep(7)
    assert tr.update(7, 1.0) is False and not tr.asleep(7)          # loud energy beats the counter
    assert tr.update(7, 0.0) is True
    tr.wake(7)                                                      # an explicit contact/grab event
    assert not tr.asleep(7)


def test_sleep_energy_and_wake_energy_are_validated():
    with pytest.raises(ValueError):
        SleepTracker(sleep_energy=-1.0)
    with pytest.raises(ValueError):
        SleepTracker(sleep_frames=0)
    with pytest.raises(ValueError):
        SleepTracker(sleep_energy=1e-6, wake_energy=1e-9)           # wake bar must be the OUTER band


def test_the_win_is_the_awake_fraction_counted_not_asserted():
    # THE HONEST MEASUREMENT: with most islands asleep, `step` is CALLED only for the awake ones.
    # Count the calls; the saving is exactly the awake fraction, no more.
    n_islands, per = 20, 3
    edges = [(i * per + k, i * per + k + 1) for i in range(n_islands) for k in range(per - 1)]
    comps = connected_components(n_islands * per, edges)
    assert len(comps) == n_islands

    rng = np.random.default_rng(1)
    state = np.zeros((n_islands * per, 2))
    loud = [0, 1, 2]                                                # only 3 of 20 islands are moving
    for i in loud:
        state[comps[i]] = rng.normal(size=(per, 2))

    calls = {"n": 0}

    def step(s):
        calls["n"] += 1
        return s

    tr = SleepTracker(sleep_energy=1e-8, sleep_frames=1)
    step_islands(state, comps, step, tracker=tr)                    # frame 1: quiet islands bank a frame
    calls["n"] = 0
    _, awake, asleep = step_islands(state, comps, step, tracker=tr)

    assert sorted(awake) == loud and len(asleep) == n_islands - len(loud)
    assert calls["n"] == len(loud)                                  # 3 calls, not 20
    assert calls["n"] / n_islands == pytest.approx(0.15)            # the saving IS the awake fraction


def test_islands_are_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    assert m.islands(6, [(0, 1), (1, 2), (4, 5)]) == [[0, 1, 2], [3], [4, 5]]
    assert m.island_energy(np.zeros((3, 3))) == 0.0

    U = np.zeros(8)
    U[0] = 0.5
    assert np.abs(m.settle_island(np.arange(8.0), U)).max() < 1e-12

    tr = m.island_sleep_tracker(sleep_energy=1e-8, sleep_frames=1)
    st = np.zeros((6, 2))
    m.step_islands(st, m.islands(6, [(0, 1)]), lambda s: s, tracker=tr)
    _, awake, asleep = m.step_islands(st, m.islands(6, [(0, 1)]), lambda s: s, tracker=tr)
    assert awake == [] and len(asleep) == 5

    assert "Islands + sleep" in str(m.find_capability("put resting bodies to sleep")[:3])


def test_cross_faculty_islands_partition_a_mesh_and_a_constraint_graph_alike():
    # SIDEWAYS: the same flood fill serves geometry and dynamics. Two disjoint triangles as a mesh, and the
    # same adjacency as a physics constraint graph, must yield the same partition.
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    faces = [(0, 1, 2), (3, 4, 5)]
    edges = [(f[k], f[(k + 1) % 3]) for f in faces for k in range(3)]
    assert m.islands(6, edges) == [[0, 1, 2], [3, 4, 5]]

    from holographic.scene_and_pipeline.holographic_route import connected_components as mesh_cc

    class _M:
        n_vertices = 6
        faces = [(0, 1, 2), (3, 4, 5)]
    assert mesh_cc(_M()) == len(m.islands(6, edges))

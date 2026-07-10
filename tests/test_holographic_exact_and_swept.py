"""A1 + A2 -- the two CORRECTNESS bugs from the superpowers audit, fixed.

A1  `distribute()`'s default reduce is a float sum, so a 4-way and a 7-way farm split of the same work disagree by
    2.98e-08. Swapping in `reduce_sum_exact` does NOT repair it: by the time the reduce sees the parts, each worker
    has already float-summed inside its own bucket. **Exactness has to reach the leaves.** So the contract changes --
    `distribute_exact` / `Coordinator.run_exact` take a worker that returns CONTRIBUTIONS, not a sum.

A2  `softbody` called the discrete `resolve_sdf_collision`, which we proved tunnels: a node stepping 0.5 m per frame
    across a 0.1 m wall lands past it having never sampled the interior. `continuous=True` sweeps the segment.

Both fixes are DEFAULT-OFF and strictly additive: where they do not apply, the result is bit-identical to today.
"""

import numpy as np
import pytest

from holographic.scene_and_pipeline.holographic_coordinator import Coordinator
from holographic.scene_and_pipeline.holographic_distribute import (
    distribute, distribute_exact, reduce_sum, reduce_sum_exact)
from holographic.simulation_and_physics.holographic_collide import resolve_swept_collision, resolve_sdf_collision
from holographic.simulation_and_physics.holographic_softbody import SoftBody


WALL = lambda P: np.abs(np.asarray(P, float)[:, 0]) - 0.05


def _wide_data(n=700, d=6, seed=0):
    """Contributions spanning 16 orders of magnitude -- the realistic farm case. Float non-associativity bites on
    DYNAMIC RANGE, not on term count; without this spread the float baseline agrees and the test proves nothing."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) * (10.0 ** rng.integers(-8, 8, size=(n, 1)))


def _split(a, k):
    return [a[i] for i in np.array_split(np.arange(len(a)), k)]


# ===========================================================================================
# A1 -- partition-invariant reduction
# ===========================================================================================

def test_the_premise_the_default_reduce_is_not_partition_invariant():
    # Without this the exact path is solving nothing.
    d = _wide_data()
    bucket_sum = lambda b, c: np.asarray(b, float).sum(axis=0)
    r4, _ = distribute(_split(d, 4), bucket_sum)
    r7, _ = distribute(_split(d, 7), bucket_sum)
    assert not np.array_equal(r4, r7)
    assert np.abs(r4 - r7).max() > 1e-9


def test_kept_negative_swapping_in_reduce_sum_exact_does_not_repair_it():
    # THE TRAP. The obvious fix looks right and is wrong: the worker has already float-summed inside its bucket.
    d = _wide_data()
    bucket_sum = lambda b, c: np.asarray(b, float).sum(axis=0)
    e4, _ = distribute(_split(d, 4), bucket_sum, reduce=reduce_sum_exact)
    e7, _ = distribute(_split(d, 7), bucket_sum, reduce=reduce_sum_exact)
    assert not np.array_equal(np.asarray(e4), np.asarray(e7))


def test_distribute_exact_is_bit_identical_under_any_bucketing():
    d = _wide_data()
    contribs = lambda b, c: np.asarray(b, float)
    base, _ = distribute_exact(_split(d, 4), contribs)
    for k in (1, 2, 4, 7, 13, len(d)):
        got, _ = distribute_exact(_split(d, k), contribs)
        assert np.array_equal(got, base), k

    # ... and under a row shuffle: the farm re-partitioned mid-job and handed the rows out differently
    rng = np.random.default_rng(9)
    shuffled, _ = distribute_exact(_split(d[rng.permutation(len(d))], 7), contribs)
    assert np.array_equal(shuffled, base)


def test_the_result_is_auditable():
    d = _wide_data(n=64, d=3)
    total, info = distribute_exact(_split(d, 4), lambda b, c: np.asarray(b, float))
    assert info["buckets"] == 4 and info["contributions"] == 64
    assert info["scale"] > 0.0 and info["bits"] == 40
    assert info["peak"] == pytest.approx(float(np.abs(d).max()))
    assert total.shape == (3,)
    # the exact total agrees with the float total to the quantization, not to nothing
    assert np.abs(total - d.sum(axis=0)).max() < 1.0


def test_distribute_exact_edge_cases():
    zero, info = distribute_exact([np.zeros((4, 3)), np.zeros((2, 3))], lambda b, c: np.asarray(b, float))
    assert np.array_equal(zero, np.zeros(3)) and info["scale"] == 0.0
    one, _ = distribute_exact([np.array([[1.0, 2.0]])], lambda b, c: np.asarray(b, float))
    assert np.allclose(one, [1.0, 2.0])


def test_coordinator_run_exact_is_partition_invariant_where_run_is_not():
    d = _wide_data()
    c = Coordinator()
    try:
        contribs = lambda b, cache: np.asarray(b, float)
        base, _ = c.run_exact(_split(d, 4), contribs)
        for k in (1, 7, 13):
            got, _ = c.run_exact(_split(d, k), contribs)
            assert np.array_equal(got, base), k

        # the float path it replaces still disagrees -- that is the bug, still there, still opt-out
        bucket_sum = lambda b, cache: np.asarray(b, float).sum(axis=0)
        f4 = c.run(_split(d, 4), bucket_sum)
        f7 = c.run(_split(d, 7), bucket_sum)
        assert not np.array_equal(f4, f7)
    finally:
        c.close()


def test_coordinator_run_exact_reports_its_scale():
    c = Coordinator()
    try:
        total, info = c.run_exact(_split(_wide_data(n=32, d=2), 4), lambda b, cache: np.asarray(b, float))
        assert info["contributions"] == 32 and info["scale"] > 0.0
        assert total.shape == (2,)
    finally:
        c.close()


# ===========================================================================================
# A2 -- swept collision
# ===========================================================================================

def test_the_premise_the_discrete_resolve_tunnels():
    # 30 m/s at dt=1/60 => 0.5 m per step across a 0.1 m wall. Neither endpoint is inside.
    landed = np.array([[0.20, 0.0, 0.0]])
    assert float(WALL(np.array([[-0.30, 0.0, 0.0]]))[0]) > 0
    assert float(WALL(landed)[0]) > 0
    assert np.allclose(resolve_sdf_collision(landed, WALL), landed)      # nothing to resolve: it tunnelled


def test_resolve_swept_collision_catches_the_crossing():
    prev = np.array([[-0.30, 0.0, 0.0]])
    now = np.array([[0.20, 0.0, 0.0]])
    out = resolve_swept_collision(prev, now, WALL)
    assert abs(float(out[0, 0]) + 0.05) < 1e-3                            # stopped on the NEAR face
    assert float(WALL(out)[0]) >= -1e-3


def test_the_sweep_leaves_untouched_nodes_bit_identical():
    # STRICT ADDITION: nodes whose segment does not hit are returned unchanged, bit for bit.
    prev = np.array([[-3.0, 0.0, 0.0], [-0.30, 0.0, 0.0]])
    now = np.array([[-2.9, 0.0, 0.0], [0.20, 0.0, 0.0]])
    out = resolve_swept_collision(prev, now, WALL)
    assert np.array_equal(out[0], now[0])                                 # the far node: untouched
    assert not np.array_equal(out[1], now[1])                             # the crossing node: caught

    # a node that moved nowhere near the collider at all
    far = np.array([[-5.0, 0.0, 0.0]])
    assert np.array_equal(resolve_swept_collision(far, far + 0.01, WALL), far + 0.01)


def test_a_stationary_node_is_never_swept():
    x = np.array([[0.5, 0.0, 0.0]])
    assert np.array_equal(resolve_swept_collision(x, x, WALL), x)


def test_softbody_tunnels_without_the_flag_and_is_stopped_with_it():
    make = lambda: SoftBody(np.array([[-0.30, 0.0, 0.0]]), velocities=np.array([[30.0, 0.0, 0.0]]))

    loose = make()
    loose.step(dt=1 / 60.0, gravity=np.zeros(3), collider=WALL)
    assert float(loose.x[0, 0]) > 0.1, "the premise: the discrete resolve is supposed to tunnel here"

    tight = make()
    tight.step(dt=1 / 60.0, gravity=np.zeros(3), collider=WALL, continuous=True)
    assert float(tight.x[0, 0]) < 0.0                                     # stopped on the near side
    assert float(WALL(tight.x)[0]) >= -1e-3


def test_softbody_continuous_is_default_off_and_bit_identical_where_it_does_not_apply():
    # BACKWARD COMPATIBILITY, pinned. No collider at all: the flag changes nothing.
    a = SoftBody(np.array([[0.0, 1.0, 0.0]]), velocities=np.array([[0.1, 0.0, 0.0]]))
    b = SoftBody(np.array([[0.0, 1.0, 0.0]]), velocities=np.array([[0.1, 0.0, 0.0]]))
    a.step(dt=1 / 60.0, gravity=np.array([0.0, -9.81, 0.0]))
    b.step(dt=1 / 60.0, gravity=np.array([0.0, -9.81, 0.0]), continuous=True)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.v, b.v)

    # a SLOW node against the same wall: the discrete resolve already handles it, so the sweep must not perturb it
    c = SoftBody(np.array([[-0.30, 0.0, 0.0]]), velocities=np.array([[1.0, 0.0, 0.0]]))
    d = SoftBody(np.array([[-0.30, 0.0, 0.0]]), velocities=np.array([[1.0, 0.0, 0.0]]))
    c.step(dt=1 / 60.0, gravity=np.zeros(3), collider=WALL)
    d.step(dt=1 / 60.0, gravity=np.zeros(3), collider=WALL, continuous=True)
    assert np.array_equal(c.x, d.x)


def test_softbody_continuous_survives_a_sweep_of_speeds():
    for speed in (5.0, 30.0, 200.0, 5000.0):
        sb = SoftBody(np.array([[-0.30, 0.0, 0.0]]), velocities=np.array([[speed, 0.0, 0.0]]))
        sb.step(dt=1 / 60.0, gravity=np.zeros(3), collider=WALL, continuous=True)
        if speed / 60.0 > 0.25:                                           # fast enough to reach the wall
            assert float(sb.x[0, 0]) <= 0.0, (speed, sb.x)


# ===========================================================================================
# wiring
# ===========================================================================================

def test_both_fixes_are_wired_to_the_mind():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    d = _wide_data(n=200, d=3)
    contribs = lambda b, c: np.asarray(b, float)
    t4, _ = m.distribute_exact(_split(d, 4), contribs)
    t7, _ = m.distribute_exact(_split(d, 7), contribs)
    assert np.array_equal(t4, t7)

    out = m.resolve_swept_collision([[-0.30, 0, 0]], [[0.20, 0, 0]], WALL)
    assert abs(float(np.asarray(out)[0, 0]) + 0.05) < 1e-3


def test_the_lint_now_records_both_clients_as_wired():
    # The mechanism: wiring a client FORCES its PENDING line to be deleted, so progress is recorded, not remembered.
    import os
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "tools"))
    from unifiers import PENDING, cites

    assert ("distribute.reduce_sum_exact_partitioned", "holographic_coordinator") not in PENDING
    assert ("collide.advance_ccd / time_of_impact", "holographic_softbody") not in PENDING
    assert cites("holographic_coordinator", "distribute.reduce_sum_exact_partitioned", repo)
    assert cites("holographic_softbody", "collide.advance_ccd / time_of_impact", repo)

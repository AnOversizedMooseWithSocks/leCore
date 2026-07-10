"""X5 + X6 + X10 -- deterministic parallelism by construction, and partition-invariant accumulation.

X5  graph-colour waves: two tasks in one wave never touch a shared resource, so a wave runs lock-free and
    atomics-free; the colouring is greedy in ascending index, so the SCHEDULE is deterministic.
X10 the same colouring applied to database write batches (holographic_querylock.plan_write_waves).
X6  partition invariance: a physics trajectory that is BIT-IDENTICAL under 4-way vs 7-way farm splits.

KEPT NEGATIVE, and it is the whole reason X6 needed new code: `reduce_sum_exact` is ORDER-independent but not
PARTITION-independent. Its scale comes from the peak and count of the parts it is handed, so a farm that
float-sums inside each bucket has already diverged before the exact merge sees anything. Exactness must reach
the leaves.
"""

import numpy as np
import pytest

from holographic.simulation_and_physics.holographic_island import (
    conflict_graph, graph_coloring, color_waves)
from holographic.scene_and_pipeline.holographic_distribute import (
    reduce_sum_exact, reduce_sum_exact_partitioned, exact_scale, exact_partial, exact_merge)
from holographic.agents_and_reasoning.holographic_querylock import plan_write_waves


# ============================================================================================
# X5 -- graph colouring
# ============================================================================================

def test_coloring_basics():
    assert graph_coloring(3, []) == [0, 0, 0]                       # no edges: everything in one wave
    assert graph_coloring(3, [(0, 1), (1, 2), (0, 2)]) == [0, 1, 2]  # a triangle needs three colours
    assert graph_coloring(2, [(0, 0)]) == [0, 0]                     # a self-loop conflicts with nothing
    assert graph_coloring(0, []) == []


def test_conflict_graph_is_deterministic_and_order_independent():
    tasks = [{"a", "b"}, {"b", "c"}, {"d"}, {"a"}]
    n, edges = conflict_graph(tasks)
    assert n == 4
    assert edges == sorted(edges)                                   # sorted, u < v
    assert all(u < v for u, v in edges)
    assert edges == [(0, 1), (0, 3)]                                # share 'b', share 'a'
    assert conflict_graph(tasks) == (n, edges)                      # same in, same out


def test_every_wave_is_conflict_free_and_every_task_scheduled_once():
    rng = np.random.default_rng(0)
    tasks = [set(int(k) for k in rng.choice(300, 2, replace=False)) for _ in range(2000)]
    n, edges = conflict_graph(tasks)
    waves = color_waves(n, edges)

    scheduled = [i for w in waves for i in w]
    assert sorted(scheduled) == list(range(2000))                   # exactly once each

    for w in waves:
        seen = set()
        for i in w:
            assert not (seen & tasks[i]), "two tasks in one wave share a resource"
            seen |= tasks[i]

    # THE MEASURED NUMBER, reported not asserted loosely: 24 waves over 2,000 tasks = 83x lock-free parallelism
    assert len(waves) == 24
    assert 2000 / len(waves) == pytest.approx(83.3, abs=0.1)


def test_the_schedule_is_deterministic_across_runs():
    rng = np.random.default_rng(1)
    tasks = [set(int(k) for k in rng.choice(50, 3, replace=False)) for _ in range(200)]
    n, edges = conflict_graph(tasks)
    first = color_waves(n, edges)
    for _ in range(4):
        assert color_waves(n, edges) == first                       # same colours, same order, every run
    # ... and shuffling the EDGE list must not change the colouring (edges are a set, not a sequence)
    perm = list(rng.permutation(len(edges)))
    shuffled = [edges[i] for i in perm]
    assert color_waves(n, shuffled) == first


def test_greedy_is_bounded_by_max_degree_plus_one():
    rng = np.random.default_rng(2)
    tasks = [set(int(k) for k in rng.choice(40, 3, replace=False)) for _ in range(120)]
    n, edges = conflict_graph(tasks)
    deg = np.zeros(n, int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    assert max(graph_coloring(n, edges)) + 1 <= deg.max() + 1        # the greedy guarantee


# ============================================================================================
# X10 -- the same colouring, applied to database write batches
# ============================================================================================

def test_write_batches_colour_into_key_disjoint_waves():
    batches = [{"a", "b"}, {"b", "c"}, {"d"}, {"a"}, {"e", "f"}]
    waves = plan_write_waves(batches)
    assert sum(len(w) for w in waves) == len(batches)
    assert waves[0] == [0, 2, 4]                                     # greedy ascending: the exact schedule
    for w in waves:
        seen = set()
        for i in w:
            assert not (seen & batches[i])
            seen |= batches[i]
    assert plan_write_waves(batches) == waves                        # deterministic


def test_write_waves_collapse_to_one_when_nothing_conflicts():
    batches = [{"a"}, {"b"}, {"c"}]
    assert plan_write_waves(batches) == [[0, 1, 2]]                  # no lock needed between any of them


def test_write_waves_serialise_when_everything_conflicts():
    batches = [{"a"}, {"a"}, {"a"}]
    assert plan_write_waves(batches) == [[0], [1], [2]]              # honest: colouring cannot invent parallelism


# ============================================================================================
# X6 -- partition invariance
# ============================================================================================

def _contribs(n=700, d=6, seed=0):
    """Contributions spanning 16 orders of magnitude -- the realistic farm case where float sums disagree."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) * (10.0 ** rng.integers(-8, 8, size=(n, 1)))


def _split(a, k):
    return [a[i] for i in np.array_split(np.arange(len(a)), k)]


def test_plain_float_reduction_is_not_partition_invariant():
    # The premise. Without this the exact path is solving nothing.
    c = _contribs()
    f = lambda parts: sum((p.sum(axis=0) for p in parts), np.zeros(c.shape[1]))
    f4, f7 = f(_split(c, 4)), f(_split(c, 7))
    assert not np.array_equal(f4, f7)
    assert np.abs(f4 - f7).max() > 1e-9


def test_kept_negative_reduce_sum_exact_is_order_invariant_but_not_partition_invariant():
    c = _contribs()
    # ORDER invariance: shuffle the SAME parts list -> bit-identical. This is what it promises, and it holds.
    parts = [row for row in c]
    rng = np.random.default_rng(3)
    perm = list(rng.permutation(len(parts)))
    assert np.array_equal(reduce_sum_exact(parts), reduce_sum_exact([parts[i] for i in perm]))

    # PARTITION invariance: float-sum inside each bucket first, then merge exactly -> NOT bit-identical, because
    # the rounding diverged before the exact merge ever saw the numbers. Exactness must reach the leaves.
    e4 = np.asarray(reduce_sum_exact([p.sum(axis=0) for p in _split(c, 4)]))
    e7 = np.asarray(reduce_sum_exact([p.sum(axis=0) for p in _split(c, 7)]))
    assert not np.array_equal(e4, e7)


def test_reduce_sum_exact_partitioned_is_bit_identical_under_any_bucketing():
    c = _contribs()
    base = reduce_sum_exact_partitioned(_split(c, 4))
    for k in (1, 2, 4, 7, 13, len(c)):
        assert np.array_equal(reduce_sum_exact_partitioned(_split(c, k)), base), k

    # ... and under a row shuffle (the farm re-partitioned mid-run and handed the rows out differently)
    rng = np.random.default_rng(4)
    assert np.array_equal(reduce_sum_exact_partitioned(_split(c[rng.permutation(len(c))], 7)), base)


def test_the_accumulator_is_a_monoid_merging_in_any_order_or_grouping():
    c = _contribs(n=50, d=3)
    scale = exact_scale(float(np.abs(c).max()), len(c))
    accs = [exact_partial([row], scale) for row in c]
    a = exact_merge(accs)
    b = exact_merge(list(reversed(accs)))
    grouped = exact_merge([exact_merge(accs[:17]), exact_merge(accs[17:40]), exact_merge(accs[40:])])
    assert np.array_equal(a, b) and np.array_equal(a, grouped)       # any order, any grouping, same int64
    assert a.dtype == np.int64


def test_partitioned_reduction_edge_cases():
    assert reduce_sum_exact_partitioned([]) == 0.0
    z = reduce_sum_exact_partitioned([[np.zeros(3)], [np.zeros(3)]])
    assert np.array_equal(z, np.zeros(3))
    one = reduce_sum_exact_partitioned([[np.array([1.0, 2.0])]])
    assert np.allclose(one, [1.0, 2.0])


def test_a_physics_trajectory_is_bit_identical_under_4_way_and_7_way_farm_splits():
    # X6, THE DEMO. One island of particles; each step accumulates pairwise forces spanning a wide magnitude
    # range (near pairs dominate, far pairs are tiny -- exactly the case float summation reorders badly).
    # Split the force accumulation across a 4-node farm and a 7-node farm; the trajectories must agree BIT for
    # BIT, not to 1e-12. That is determinism that survives re-partitioning the farm mid-run.
    rng = np.random.default_rng(5)
    n_p = 24
    X0 = rng.normal(size=(n_p, 3))
    V0 = np.zeros((n_p, 3))
    pairs = [(i, j) for i in range(n_p) for j in range(i + 1, n_p)]

    # THE CONDITION, measured and kept: float non-associativity bites on DYNAMIC RANGE, not on term count. A plain
    # 1/r^2 scene spans only ~2.6 orders of magnitude across its pair forces, and summing 23 similar-magnitude
    # terms is reorder-stable -- the float baseline agreed bit-for-bit under 4-way vs 7-way, so the test proved
    # nothing. Charges spanning 1e-6..1e6 push the forces across 22 orders, which is the realistic farm case
    # (heavy bodies beside light ones) and the case reduce_sum_exact was written for. Do not remove the charges:
    # without them this test passes for the wrong reason.
    q = 10.0 ** rng.integers(-6, 6, size=n_p).astype(float)

    def pair_force(X, i, j):
        d = X[j] - X[i]
        r2 = float(d @ d) + 1e-6
        return d / (r2 * np.sqrt(r2)) * q[i] * q[j]                  # 1/r^2 x charges: 22 orders of magnitude

    def step(X, V, n_buckets, dt=1e-3):
        # every particle's force is the exact sum of its pair contributions, bucketed n_buckets ways
        buckets = [[] for _ in range(n_buckets)]
        for b, (i, j) in enumerate(pairs):
            f = pair_force(X, i, j)
            contrib = np.zeros((n_p, 3))
            contrib[i] += f
            contrib[j] -= f
            buckets[b % n_buckets].append(contrib)                    # round-robin: a DIFFERENT split per n_buckets
        F = reduce_sum_exact_partitioned(buckets)
        Vn = V + dt * F
        return X + dt * Vn, Vn

    X4, V4 = X0.copy(), V0.copy()
    X7, V7 = X0.copy(), V0.copy()
    for _ in range(6):
        X4, V4 = step(X4, V4, 4)
        X7, V7 = step(X7, V7, 7)

    assert np.array_equal(X4, X7), "trajectories must be bit-identical under different farm splits"
    assert np.array_equal(V4, V7)
    assert np.abs(X4 - X0).max() > 1e-9                              # it actually moved: not a trivial match

    # and the float baseline does NOT survive the same re-partitioning
    def float_step(X, V, n_buckets, dt=1e-3):
        buckets = [np.zeros((n_p, 3)) for _ in range(n_buckets)]
        for b, (i, j) in enumerate(pairs):
            f = pair_force(X, i, j)
            buckets[b % n_buckets][i] += f
            buckets[b % n_buckets][j] -= f
        F = np.zeros((n_p, 3))
        for bk in buckets:
            F = F + bk
        Vn = V + dt * F
        return X + dt * Vn, Vn

    Xf4, Vf4 = X0.copy(), V0.copy()
    Xf7, Vf7 = X0.copy(), V0.copy()
    for _ in range(6):
        Xf4, Vf4 = float_step(Xf4, Vf4, 4)
        Xf7, Vf7 = float_step(Xf7, Vf7, 7)
    assert not np.array_equal(Xf4, Xf7), "the float baseline is supposed to drift -- that is the premise"
    assert np.abs(Xf4 - Xf7).max() > 0.0


def test_colour_waves_and_exact_sums_compose_on_the_mind():
    # CROSS-FACULTY: the wave schedule (X5) decides WHAT runs together; the exact accumulator (X6) makes the
    # result independent of how it was bucketed. Two faculties, one determinism story.
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    n, edges = m.conflict_graph([{"a", "b"}, {"b", "c"}, {"d"}])
    assert m.color_waves(n, edges) == [[0, 2], [1]]
    assert m.graph_coloring(n, edges) == [0, 1, 0]
    assert m.plan_write_waves([{"a"}, {"a", "b"}, {"c"}]) == [[0, 2], [1]]

    c = _contribs(n=60, d=4, seed=6)
    a = m.reduce_sum_exact_partitioned(_split(c, 3))
    b = m.reduce_sum_exact_partitioned(_split(c, 7))
    assert np.array_equal(a, b)

    assert "Graph-colour waves" in str(m.find_capability("run tasks in parallel without locks")[:3])
    assert "Partition-invariant" in str(m.find_capability("same answer no matter how many machines i use")[:3])

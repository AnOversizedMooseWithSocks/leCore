"""Regression traps for local process-pool scale-out (`local_pool` + the `backend` seam).

THE DEFECT THIS CLOSES was not a missing mechanism -- `LocalPool` was written, correct, and tested. It was
UNREACHABLE: no faculty exposed it, `distribute_compute` had no backend parameter at all, and
`find_capability("spin up another instance")` returned "Describe a scene" and "Texture graph". By this
repo's governing rule -- a capability find_capability cannot surface and /invoke cannot call does not exist
-- it did not exist.

The property that matters more than the speed is BIT-IDENTITY: the reduce is a commutative monoid, so a
pooled run must give byte-for-byte the same answer as a sequential one. A faster wrong answer is not a
result.
"""
import numpy as np
import pytest

import lecore
import poolwork


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


@pytest.fixture(scope="module")
def buckets():
    return [list(range(i * 200, (i + 1) * 200)) for i in range(6)]


def test_the_default_is_unchanged(mind, buckets):
    # The seam is ADDITIVE. Every existing caller passes no backend and must get exactly today's behaviour.
    total, info = mind.distribute_compute(buckets, poolwork.light, reduce="sum")
    assert info["buckets"] == 6
    assert total == sum(sum(b) for b in buckets)


def test_a_pooled_run_is_bit_identical_to_sequential(mind, buckets):
    """THE PROPERTY THAT DECIDES WHETHER PARALLELISM IS SAFE HERE. Not 'close' -- identical. The reduce is a
    commutative monoid and the workers touch disjoint buckets, so partitioning across processes must not
    move a single bit."""
    sequential, _ = mind.distribute_compute(buckets, poolwork.heavy, reduce="sum")
    pool = mind.local_pool(n=2)
    try:
        pooled, _ = mind.distribute_compute(buckets, poolwork.heavy, reduce="sum", backend=pool)
    finally:
        pool.close()
    assert sequential == pooled, "pooled %r != sequential %r" % (pooled, sequential)


def test_bucket_order_does_not_change_the_answer(mind):
    # The reason a pool is safe at all: the reducer is commutative, so reordering buckets is a no-op.
    a = [list(range(0, 300)), list(range(300, 600))]
    b = [list(range(300, 600)), list(range(0, 300))]
    assert mind.distribute_compute(a, poolwork.light, reduce="sum")[0] == \
        mind.distribute_compute(b, poolwork.light, reduce="sum")[0]


def test_the_pool_is_persistent_and_reusable(mind, buckets):
    # LocalPool is a PERSISTENT pool, not spawn-per-task. Two runs on one pool must both work -- if the
    # second failed, callers would be paying process startup on every call and the design intent is lost.
    pool = mind.local_pool(n=2)
    try:
        first, _ = mind.distribute_compute(buckets, poolwork.light, reduce="sum", backend=pool)
        second, _ = mind.distribute_compute(buckets, poolwork.light, reduce="sum", backend=pool)
        assert first == second
    finally:
        pool.close()


def test_a_shared_cache_survives_the_process_boundary(mind, buckets):
    # The zero-copy shared_memory path: a numpy cache is published ONCE and mapped read-only by every
    # worker. If this regressed to per-bucket pickling it would still be CORRECT, so only a test that
    # actually uses the cache would notice.
    cache = np.arange(16, dtype=float)
    pool = mind.local_pool(n=2)
    try:
        pooled, _ = mind.distribute_compute(buckets, poolwork.with_cache, reduce="sum",
                                            cache=cache, backend=pool)
    finally:
        pool.close()
    sequential, _ = mind.distribute_compute(buckets, poolwork.with_cache, reduce="sum", cache=cache)
    assert pooled == sequential


def test_local_pool_is_discoverable(mind):
    """The actual gap. Before this, every one of these returned unrelated fallbacks."""
    for query in ("spin up another instance", "start a second worker", "use more cores",
                  "launch a local worker pool", "balance load across instances"):
        assert "local worker processes" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the local pool" % query


def test_the_measurement_confound_is_recorded(mind):
    """The break-even numbers taken for this change were measured on a ONE-CORE machine, where a process
    pool cannot win by construction. They are recorded as a confound rather than a result, and the default
    was left alone because of it -- not because pools were shown not to pay."""
    doc = mind.local_pool.__doc__ or ""
    assert "ONE CORE" in doc
    assert "CONFOUND" in doc


# --------------------------------------------------------------------------------------
# Core detection and the automatic pooling decision.
# --------------------------------------------------------------------------------------

def test_cpu_budget_respects_limits_that_cpu_count_ignores(mind):
    """os.cpu_count() reports the HOST's cores inside a container and ignores cgroup quota and affinity, so
    `docker run --cpus=2` on a 64-core box answers 64 -- and a pool sized from it spawns 64 interpreters to
    time-share 2 cores: slower than sequential AND 64x the memory. On an engine meant for small devices that
    is a memory-bloat bug, not just a speed one."""
    import os

    budget = mind.cpu_budget()
    assert isinstance(budget, int) and budget >= 1
    assert budget <= (os.cpu_count() or 1), "cpu_budget exceeded the machine's own core count"
    try:
        assert budget <= len(os.sched_getaffinity(0)), "cpu_budget ignored CPU affinity"
    except (AttributeError, OSError):
        pass                                    # not every platform has affinity; skipping is correct


def test_should_pool_refuses_on_one_core(mind):
    refused, why = mind.should_pool(8, 50.0, cores=1)
    assert refused is False and "core" in why


def test_should_pool_refuses_a_single_bucket(mind):
    refused, why = mind.should_pool(1, 50.0, cores=8)
    assert refused is False and "bucket" in why


def test_should_pool_refuses_work_smaller_than_dispatch(mind):
    # ~0.2 ms dispatch per bucket, x4 margin. Below that, sending the work away costs more than doing it.
    refused, why = mind.should_pool(8, 0.1, cores=8)
    assert refused is False and "dispatch" in why


def test_should_pool_accepts_a_job_that_clears_every_gate(mind):
    ok, why = mind.should_pool(8, 50.0, cores=8)
    assert ok is True and "8 cores" in why


def test_auto_is_bit_identical_to_sequential(mind, buckets):
    sequential, _ = mind.distribute_compute(buckets, poolwork.heavy, reduce="sum")
    automatic, _ = mind.distribute_compute(buckets, poolwork.heavy, reduce="sum",
                                           backend="auto", est_ms_per_bucket=50.0)
    assert sequential == automatic


def test_auto_leaves_no_processes_behind(mind, buckets):
    """AUTOMATIC DECISION, NOT AUTOMATIC SPAWN. A library call must not silently leave interpreters running
    after it returns -- that is the caller's decision, and on a small device it is the difference between
    idle and out of memory."""
    import multiprocessing

    before = len(multiprocessing.active_children())
    mind.distribute_compute(buckets, poolwork.light, reduce="sum",
                            backend="auto", est_ms_per_bucket=50.0)
    assert len(multiprocessing.active_children()) == before


def test_auto_without_an_estimate_declines_rather_than_guessing(mind, buckets):
    # No estimate means no evidence. Falling back to sequential is the conservative reading, and it keeps
    # the default behaviour of every existing caller who adopts auto without measuring first.
    total, _ = mind.distribute_compute(buckets, poolwork.light, reduce="sum", backend="auto")
    assert total == sum(sum(b) for b in buckets)


# --------------------------------------------------------------------------------------
# auto routes through the ONE oracle, and the decision is auditable (backlog W7).
# --------------------------------------------------------------------------------------

def test_auto_consults_place_work_not_should_pool_directly(mind, buckets):
    """ONE ORACLE, NOT TWO. `auto` used to call should_pool directly, so it could only answer 'pool or not'
    — it could not see the device and it ignored the resource policy. place_work already composes unit, pool
    and device with the policy veto first; a second copy of the routing logic inside `auto` would re-create
    the three-unrelated-switches problem inside the thing built to fix it."""
    mind.distribute_compute(buckets, poolwork.light, reduce="sum",
                            backend="auto", est_ms_per_bucket=50.0)
    decision = mind.last_placement()
    assert decision is not None
    assert set(decision["considered"]) == {"unit", "pool", "device"}


def test_the_auto_decision_is_inspectable(mind, buckets):
    # An automatic decision that cannot be inspected is indistinguishable from a bug. The gate can decline
    # for four different reasons and a caller deserves to know which.
    mind.distribute_compute(buckets, poolwork.light, reduce="sum",
                            backend="auto", est_ms_per_bucket=50.0)
    why = mind.last_placement()["considered"]["pool"]["why"]
    assert why and ("core" in why or "dispatch" in why or "bucket" in why or "policy" in why)


def test_a_policy_veto_reaches_the_auto_path(mind, buckets):
    mind.resource_policy(pool="deny")
    total, _info = mind.distribute_compute(buckets, poolwork.light, reduce="sum",
                                           backend="auto", est_ms_per_bucket=5000.0)
    assert mind.last_placement()["considered"]["pool"]["verdict"] is False
    assert "policy" in mind.last_placement()["considered"]["pool"]["why"]
    assert total == sum(sum(b) for b in buckets), "the result changed when the pool was denied"


def test_a_device_verdict_does_not_silently_do_nothing(mind, buckets):
    """This seam partitions BUCKETS across workers; a device kernel is a different shape entirely (one
    dispatch over an array, not N independent Python callables). So a 'device' verdict runs on CPU here —
    but it is REPORTED, because silently discarding the recommendation would leave a caller believing the
    device had been used."""
    mind.resource_policy(pool="deny")
    mind.distribute_compute(buckets, poolwork.light, reduce="sum", backend="auto",
                            est_ms_per_bucket=50.0, n_bytes=10 ** 8, flops_per_byte=50.0)
    decision = mind.last_placement()
    assert decision["considered"]["device"]["verdict"] in (True, False)
    assert decision["considered"]["device"]["why"]

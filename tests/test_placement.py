"""Regression traps for the unified placement oracle (backlog W4).

The claim under test is an ORDERING, not an arithmetic: policy veto first, then cheapest-correct. A version
that consulted the three oracles in any other order would pass most of these and fail the two that encode
the argument — the veto test and the tie test.
"""
import pytest

import lecore
from holographic.scene_and_pipeline.holographic_placement import place_work
from holographic.scene_and_pipeline.holographic_policy import ResourcePolicy


@pytest.fixture
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


def test_nothing_supplied_falls_back_to_cpu(mind):
    result = mind.place_work()
    assert result["placement"] == "cpu"
    assert result["provisional"] is False


def test_an_uncosted_candidate_is_not_evaluated_rather_than_lost(mind):
    """A placement nobody costed and a placement that lost are DIFFERENT FACTS. Reporting a missing estimate
    as a rejection would quietly hide the fact that the caller never supplied the numbers."""
    result = mind.place_work(n_buckets=8, est_ms_per_bucket=50.0)
    assert result["considered"]["device"]["verdict"] is None
    assert "not evaluated" in result["considered"]["device"]["why"]
    assert result["considered"]["pool"]["verdict"] in (True, False)


def test_the_policy_veto_beats_arithmetic_that_would_say_yes():
    """THE FIRST HALF OF THE ARGUMENT. No amount of arithmetic makes a forbidden device faster, and an
    oracle that recommends what the operator has banned is worse than no oracle — it produces a plan that
    cannot be executed."""
    result = place_work(n_bytes=10 ** 9, flops_per_byte=100.0, policy=ResourcePolicy(gpu="off"))
    assert result["placement"] != "device"
    assert result["considered"]["device"]["verdict"] is False
    assert "forbids" in result["considered"]["device"]["why"]


def test_the_pool_veto_is_honoured_too():
    result = place_work(n_buckets=64, est_ms_per_bucket=5000.0, policy=ResourcePolicy(pool="deny"))
    assert result["considered"]["pool"]["verdict"] is False
    assert result["placement"] != "pool"


def test_cheapest_correct_wins_a_tie(mind):
    """THE SECOND HALF. When a unit and a device both pay, the unit wins — weaker requirements first. The
    device is last specifically because it is the only placement that changes the NUMBERS rather than only
    the speed: GPU matches NumPy to a tolerance, while the pooled path is verified bit-identical."""
    result = mind.place_work(unit="t2_baked_grid", baseline_ns=1e7, n_calls=100,
                             n_bytes=10 ** 8, flops_per_byte=50.0)
    assert result["considered"]["unit"]["verdict"] is True
    # GPU-GATED, like every other device-dependent test in the tree (test_gpu_crossover.py carries a
    # module-level skipif). A device verdict is False wherever no compute adapter exists -- every CI runner
    # included -- so asserting True unconditionally tests the RUNNER, not the tie-break. What this test
    # actually pins is the ORDER (a unit that pays beats a device that also pays), asserted below and
    # reached only when the device arm can genuinely pay.
    if result["considered"]["device"]["verdict"] is not True:
        import pytest as _pytest
        _pytest.skip("no compute adapter available: the device arm cannot pay here")
    assert result["placement"] == "unit", "the device was preferred over a unit that also pays"


def test_a_device_recommendation_is_marked_provisional(mind):
    # should_offload's thresholds are arithmetic from PCIe bandwidth; no host<->device crossover has ever
    # been measured in this project, and a recommendation must not be mistaken for a measured result.
    result = mind.place_work(n_bytes=10 ** 8, flops_per_byte=50.0)
    if result["placement"] == "device":
        assert result["provisional"] is True
    assert mind.place_work()["provisional"] is False


def test_the_evidence_travels_with_the_answer(mind):
    result = mind.place_work(n_buckets=8, est_ms_per_bucket=50.0, n_bytes=10 ** 8, flops_per_byte=50.0)
    assert set(result["considered"]) == {"unit", "pool", "device"}
    for row in result["considered"].values():
        assert "verdict" in row and row["why"]


def test_it_composes_rather_than_reimplements():
    """Every verdict must come from the existing oracle for that question — this module contributes the
    order, the veto and the report, not a fourth cost model that could drift from the other three."""
    import inspect

    from holographic.scene_and_pipeline import holographic_placement as mod
    src = inspect.getsource(mod.place_work)
    assert "should_pool" in src and "should_offload" in src and "machine_place_unit" in src


def test_placement_is_discoverable(mind):
    for query in ("where should this work run", "should this go on the gpu or cpu", "cpu pool or gpu"):
        assert "one placement oracle" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the placement oracle" % query

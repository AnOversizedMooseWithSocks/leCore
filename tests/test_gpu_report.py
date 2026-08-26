"""Regression traps for the GPU report and offload pre-gate (GPU backlog A2 + A3).

These are the GPU mirror of `cpu_budget` / `should_pool` and are tested the same way: the report must
distinguish states a bare bool conflates, and every gate must be shown to fire on its own ground rather
than being masked by the "no device" check that comes first.
"""
import pytest

import lecore
from holographic.io_and_interop.holographic_gpureport import (MIN_BYTES_PROVISIONAL, gpu_report,
                                                              should_offload)
from holographic.scene_and_pipeline.holographic_policy import ResourcePolicy


@pytest.fixture
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


# --------------------------------------------------------------------------------------
# A2 -- the report.
# --------------------------------------------------------------------------------------

def test_the_report_never_raises_on_a_machine_with_no_gpu():
    # The common case. A report that fails where there is no device is useless precisely when it is needed.
    report = gpu_report()
    for key in ("cupy", "wgsl", "any_available", "policy_allows", "wired_modules", "note"):
        assert key in report


def test_an_unavailable_path_says_why():
    """A bare bool conflates four states — no CuPy, CuPy but no device, a device the policy forbids, and
    enabled — and three of the four are fixable by the user. The `why` is the whole point of the report."""
    report = gpu_report()
    assert report["cupy"]["available"] is False
    assert report["cupy"]["why"], "an unavailable path gave no reason"


def test_both_paths_are_reported_separately():
    # A CuPy-only report would tell an Apple or AMD user they have no GPU, which is false.
    report = gpu_report()
    assert "NVIDIA" in report["cupy"]["vendor"]
    assert "Vulkan" in report["wgsl"]["vendor"] or "Metal" in report["wgsl"]["vendor"]


def test_wired_modules_are_discovered_by_IMPORT_not_by_mention():
    """THE DEFECT THIS TEST EXISTS FOR. The first version substring-matched 'holographic_backend' and
    counted every module that merely NAMES the backend in a docstring — including two written the same
    afternoon that discuss it in prose — reporting 7 consumers where there are 4. `deptrace` is the standing
    example: it documents `holographic_backend.accelerator_report` and never imports it.

    A capability audit that counts documentation as wiring is exactly the kind of number this project
    refuses to publish."""
    modules = gpu_report()["wired_modules"]
    assert modules, "the discovery walk found nothing"
    assert all("/" in name for name in modules)
    assert not any("deptrace" in name for name in modules), \
        "deptrace only MENTIONS the backend in a docstring; it must not count as wired"
    assert not any("gpureport" in name or "wgpurun" in name for name in modules), \
        "a module that discusses the backend in prose is being counted as a consumer again"


def test_the_report_honours_the_resource_policy(mind):
    mind.resource_policy(gpu="off")
    assert mind.gpu_report()["policy_allows"] is False
    mind.resource_policy(gpu="auto")
    assert mind.gpu_report()["policy_allows"] is True


# --------------------------------------------------------------------------------------
# A3 -- the pre-gate. Each ground checked with availability forced, so the device check cannot mask it.
# --------------------------------------------------------------------------------------

def test_no_device_refuses_regardless_of_arithmetic():
    refused, why = should_offload(10 ** 9, 100.0, available=False)
    assert refused is False and "no GPU" in why


def test_too_little_data_refuses():
    refused, why = should_offload(1000, 100.0, available=True)
    assert refused is False and "transfer floor" in why


def test_a_transfer_bound_job_refuses():
    # An elementwise pass reads and writes everything and computes almost nothing, so the bus is the cost.
    refused, why = should_offload(10 ** 9, 0.5, available=True)
    assert refused is False and "transfer-bound" in why


def test_repeated_round_trips_point_at_fusion_rather_than_offload():
    """The interesting refusal. When data would cross the bus N times the answer is not 'is it big enough'
    but 'collapse the passes first' — which is what shader_pipeline does before any transfer happens."""
    refused, why = should_offload(10 ** 9, 100.0, round_trips=4, available=True)
    assert refused is False
    assert "shader_pipeline" in why


def test_a_job_clearing_every_floor_is_accepted():
    ok, why = should_offload(10 ** 8, 50.0, available=True)
    assert ok is True
    assert "provisional" in why.lower(), "the verdict must flag its thresholds as unmeasured"


def test_the_thresholds_are_marked_provisional():
    # No host<->device crossover has ever been measured in this project. The constant is arithmetic from
    # PCIe bandwidth, not a result, and saying so is what keeps it from becoming a folklore number.
    import inspect

    from holographic.io_and_interop import holographic_gpureport as mod
    src = inspect.getsource(mod)
    assert "PROVISIONAL" in src
    assert MIN_BYTES_PROVISIONAL > 0


def test_a_policy_veto_beats_any_amount_of_arithmetic():
    refused, _why = should_offload(10 ** 9, 100.0, policy=ResourcePolicy(gpu="off"))
    assert refused is False


def test_wired_and_discoverable(mind):
    assert mind.should_offload(10 ** 8, 50.0)[0] in (True, False)
    for query in ("what gpu do i have", "is the gpu worth using here", "why is my gpu not being used"):
        assert "What GPU do I have" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the GPU report" % query


# --------------------------------------------------------------------------------------
# D1's gate: is there anything to fuse-then-dispatch?
# --------------------------------------------------------------------------------------

def test_the_shipped_postfx_chain_has_no_fusable_runs(mind):
    """D1'S GATE, PINNED. 'Fuse-then-dispatch' assumes there are adjacent linear stages to collapse before
    handing work to a device. Measured on the shipped default chain: ZERO. All seven stages
    (exposure, bloom, aces, chromatic_aberration, vignette, film_grain, gamma) fall in a single NON-fused
    run, because every linear stage is separated by a nonlinear one.

    That is the same shape of finding as the refuted fusion rung: the mechanism is real and there is nothing
    in the shipped code for it to act on. If a chain ever gains adjacent linear stages this fails, and the
    fuse-then-dispatch item becomes worth costing again."""
    steps = mind.postfx_chain().steps
    runs = mind.postfx_fusable_runs(steps)
    fusable = [s for fused, s in runs if fused and len(s) > 1]
    assert fusable == [], "the default chain now has fusable runs: %r -- re-open D1" % fusable


def test_finding_fusable_runs_already_ships(mind):
    # Rule 0 on my own backlog item: D1's stated first task ("a pass that finds maximal LSI runs") already
    # existed as postfx_fusable_runs. Pinned so it is not rebuilt under a new name.
    assert hasattr(mind, "postfx_fusable_runs")
    runs = mind.postfx_fusable_runs([("exposure", {}), ("gamma", {})])
    assert isinstance(runs, list) and runs

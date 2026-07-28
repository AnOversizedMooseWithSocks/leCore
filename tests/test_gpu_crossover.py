"""Regression traps for the host<->device crossover benchmark (backlog M1).

The benchmark's value is entirely in its HONESTY, not its numbers — on this machine the numbers are
meaningless by construction. So these tests check the guards: that a software adapter is flagged rather than
flattered, that "the device never won" is reported as a result rather than swallowed, and that the timing
path forces completion instead of measuring kernel launch.
"""
import numpy as np
import pytest

import lecore
from holographic.io_and_interop.holographic_gpubench import crossover, crossover_report as report
from holographic.io_and_interop.holographic_wgpurun import available

pytestmark = pytest.mark.skipif(not available(), reason="wgpu / no compute adapter available")


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


@pytest.fixture(scope="module")
def small():
    return crossover(kind="cleanup", dims=(64,), counts=(32,), batches=(1, 4), repeats=1)


def test_the_report_has_the_fields_downstream_needs(small):
    assert set(small) == {"adapter", "trustworthy", "rows", "crossover", "note"}
    assert small["rows"], "an available adapter produced no rows"


def test_a_software_adapter_is_flagged_not_flattered(small):
    """THE POINT OF THE WHOLE MODULE. llvmpipe and WARP are CPU adapters, so a timing there is NumPy against
    a CPU driver emulating a GPU. Emitting a plausible-looking table without saying so would be worse than
    emitting nothing — someone would quote it."""
    if str(small["adapter"].get("type", "")).upper() == "CPU":
        assert small["trustworthy"] is False
        assert "MEANINGLESS" in small["note"]
    else:
        assert small["trustworthy"] is True


def test_the_banner_prints_before_the_table(small):
    # Ordering matters: a reader who skims sees the caveat first, not a tidy table of numbers.
    text = report(small)
    assert text.index(small["note"][:20]) < text.index("dim")


def test_never_winning_is_reported_as_a_result(small):
    """`crossover: never` must be published, not swallowed. A device that loses at every tested size is a
    finding — and on a CPU adapter it is the EXPECTED finding."""
    text = report(small)
    if small["crossover"] is None:
        assert "never" in text and "RESULT" in text
    else:
        assert "bytes" in text


def test_every_timed_row_forces_completion(small):
    """GPU calls are ASYNCHRONOUS: timing a dispatch without forcing completion measures KERNEL LAUNCH, not
    execution, and produces numbers that look spectacular and are wrong. Every device path here reads its
    result back, so a device time can never be implausibly near zero."""
    for row in small["rows"]:
        if "error" in row:
            continue
        assert row["gpu_ms"] > 0.0
        assert row["cpu_ms"] > 0.0


def test_an_unsupported_shape_is_a_skipped_row_not_a_crash():
    # A sweep must survive a configuration a kernel refuses (e.g. a non-power-of-two dim for bind), or one
    # bad cell loses the whole table.
    out = crossover(kind="bind", dims=(96,), counts=(8,), batches=(2,), repeats=1)
    assert out["rows"], "the sweep produced nothing at all"
    assert any("error" in row for row in out["rows"]) or all("gpu_ms" in row for row in out["rows"])


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        crossover(kind="nonsense", dims=(64,), counts=(32,), batches=(1,), repeats=1)


def test_the_benchmark_is_wired_and_discoverable(mind):
    result = mind.gpu_crossover(dims=(64,), counts=(32,), batches=(1,), repeats=1)
    assert "crossover" in result
    assert isinstance(mind.gpu_crossover(dims=(64,), counts=(32,), batches=(1,), repeats=1, text=True), str)
    for query in ("measure the gpu crossover", "benchmark cpu vs gpu", "is my gpu actually faster"):
        assert "crossover" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the benchmark" % query


def test_it_names_what_to_do_with_the_answer(mind):
    # A measurement nobody knows how to apply is a number, not a result. The docstring must say which
    # constants it replaces, or M1 stays open even after someone runs it.
    doc = mind.gpu_crossover.__doc__ or ""
    assert "MIN_BYTES_PROVISIONAL" in doc

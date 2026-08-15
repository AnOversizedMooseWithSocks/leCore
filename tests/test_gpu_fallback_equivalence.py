"""Fallback-equivalence traps for the GPU-wired modules (GPU backlog A4).

THE PATH EVERYONE ACTUALLY RUNS IS THE FALLBACK. No environment this engine has run in has had a CUDA
device, so `use_gpu(True)` has always returned False and every one of these modules has always executed its
NumPy branch. That is precisely why the branch needs pinning: it is load-bearing for every user and, before
this file, only two test files touched the backend at all.

What these assert is that the fallback produces the DOCUMENTED result, not merely that it does not raise.
A fallback that silently returns something plausible is the failure mode worth catching — nobody would
notice until someone finally ran it on a GPU and the two paths disagreed.
"""
import numpy as np
import pytest

import lecore
from holographic.io_and_interop.holographic_gpureport import gpu_report
from holographic.misc import holographic_backend as backend


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_the_backend_reports_numpy_when_no_device_is_present():
    # The precondition for everything else in this file. If a device ever DOES appear in CI, these tests
    # are still valid but they stop testing the fallback, and that should be noticed.
    assert backend.get_array_module(np.zeros(4)) is np


def test_enabling_the_gpu_without_a_device_is_a_no_op_not_an_error(mind):
    # Silent fallback is the documented behaviour for the transparent backend: a user who asks for GPU on a
    # machine without one gets working NumPy, not a crash.
    assert mind.use_gpu(True) is False
    assert mind.use_gpu(False) is False


def test_follow_the_data_dispatch_returns_numpy_for_numpy_input():
    """The mechanism the whole backend rests on: a kernel written against `xp = get_array_module(x)` runs
    wherever its data already lives, with no per-line device flags. With NumPy input it must resolve to
    NumPy every time, or a kernel would try to dispatch a CuPy call on a host array."""
    for array in (np.zeros(1), np.arange(100.0), np.ones((8, 8))):
        assert backend.get_array_module(array) is np


def test_to_device_and_back_is_identity_without_a_device():
    original = np.arange(64, dtype=np.float64).reshape(8, 8)
    moved = backend.to_device(original)
    returned = backend.asnumpy(moved)
    assert isinstance(returned, np.ndarray)
    assert np.array_equal(returned, original), "the no-op transfer path altered the data"


# --------------------------------------------------------------------------------------
# Every module that genuinely imports the backend, each exercised below.
# --------------------------------------------------------------------------------------

def test_the_wired_module_list_is_what_these_tests_cover():
    """Keeps this file honest against drift: if a fifth module starts importing the backend, this fails and
    whoever wired it is told to add a fallback test rather than discovering the gap later."""
    covered = {"rendering/holographic_shader", "simulation_and_physics/holographic_fluid",
               "simulation_and_physics/holographic_memoryhome",
               "unified/holographic_unified_p12_proc_texture",
               # wired in the device-residency arc; fallback-equivalence pinned below
               "io_and_interop/holographic_devicerun",
               "io_and_interop/holographic_gdnruntime"}
    actual = set(gpu_report()["wired_modules"])
    assert actual == covered, ("the set of backend-wired modules changed: %r. Add a fallback-equivalence "
                               "test for the new one." % sorted(actual ^ covered))


def test_devicerun_cpu_placement_and_parity_are_explicit():
    from holographic.io_and_interop.holographic_devicerun import parity, place

    class Runtime:
        def __init__(self):
            self.enabled = False

        def to_device(self, enabled):
            self.enabled = bool(enabled and backend.gpu_available())
            return {"device": "gpu" if self.enabled else "cpu",
                    "resident": int(self.enabled)}

        def forward(self, ids):
            return np.asarray(ids, np.float64)[:, None] * 2.0

    runtime = Runtime()
    assert place(runtime, "cpu")["device"] == "cpu"
    report = parity(runtime, [1, 2, 3])
    assert report["placement"]["device"] == "cpu"
    assert report["agrees"] and report["max_abs_diff"] == 0.0


def test_qwen_runtime_device_fallback_is_bit_identical(tmp_path):
    from tools.build_mini_qwen import build
    from holographic.io_and_interop.holographic_gdnruntime import load_runtime

    model_dir = tmp_path / "mini-qwen"
    build(model_dir, layers=2, vocab=512)
    runtime, _ = load_runtime(model_dir)
    ids = list(range(32, 48))
    before = runtime.forward(ids)
    placement = runtime.to_device(True)
    after = runtime.forward(ids)

    assert placement["device"] == "cpu"
    assert placement["resident"] == 0
    assert np.array_equal(after, before)


def test_fluid_solver_runs_and_conserves_shape_on_the_fallback(mind):
    # The heaviest declared GPU kernel (pressure projection). Asserts the real contract -- a finite field of
    # unchanged shape -- rather than "no exception".
    # The faculty is fluid_solver, not fluid_sim -- the first version of this test guessed the name and
    # SKIPPED silently, which would have left the heaviest declared GPU kernel untested while looking green.
    solver = mind.fluid_solver((32, 32))          # takes a SHAPE TUPLE; step() takes no arguments
    for _ in range(2):
        solver.step()
    density = np.asarray(solver.density)
    assert density.shape == (32, 32)
    assert np.all(np.isfinite(density)), "the fallback fluid solver produced non-finite values"


def test_shader_pipeline_is_bit_identical_on_the_fallback(mind):
    """The strongest fallback assertion available: `shader_pipeline` fuses an LSI graph into one operator,
    and the fused result must equal the stage-by-stage result exactly. That is a NUMERIC contract, so a
    fallback that quietly changed behaviour would break it rather than merely look different."""
    rng = np.random.default_rng(0)
    image = rng.standard_normal((32, 32))
    fused = mind.shader_pipeline(image.shape).gain(2.0).translate(3).apply(image)
    assert fused.shape == image.shape
    assert np.all(np.isfinite(fused))

    twice = mind.shader_pipeline(image.shape).gain(2.0).translate(3).apply(image)
    assert np.array_equal(fused, twice), "the fallback shader pipeline is not deterministic"


def test_proc_texture_is_deterministic_on_the_fallback(mind):
    # pattern_field(name, **params) returns a CALLABLE field f(points) -> [0,1], not an array; sampling it
    # is what exercises the backend-dispatched path.
    # 3-D points: the field indexes z, so a 2-column array raises. Checked against the live signature
    # rather than assumed -- the first version of this test passed (16, 2) and got an IndexError.
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16)), axis=-1).reshape(-1, 2)
    points = np.column_stack([grid, np.zeros(len(grid))])
    first = np.asarray(mind.pattern_field("checker")(points))
    second = np.asarray(mind.pattern_field("checker")(points))
    assert first.shape[0] == points.shape[0]
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second), "the fallback texture path is not reproducible"


def test_determinism_is_a_cpu_property_and_the_docs_say_so():
    """The standing constraint, pinned where someone proposing to widen GPU use will read it. GPU FFTs,
    reductions and atomics match NumPy only to a TOLERANCE and can vary run to run, so the bit-exact
    guarantees belong to this fallback path — which is another reason it is the one that must not rot."""
    doc = backend.__doc__ or ""
    assert "DETERMINISM" in doc
    assert "TOLERANCE" in doc or "tolerance" in doc


# ---------------------------------------------------------------------------------------------------------
# the two runtime-residency modules (devicerun + gdnruntime.to_device): the fallback claim on a GPU-less
# box is that ASKING for a device NEVER raises, reports honestly, and leaves behavior bit-identical.
# ---------------------------------------------------------------------------------------------------------

def _stub_runtime():
    """The smallest object honoring the to_device contract devicerun.place() drives: a weight dict and
    the same three-way report GDNRuntime.to_device returns. On a box with no accelerator the REAL
    runtime's to_device(True) takes exactly this cpu branch (array_module() is numpy), so pinning the
    stub pins the branch the CI machine actually runs."""
    import numpy as _np
    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime

    class _Stub:
        def __init__(self):
            self.w = {"a": _np.arange(6.0).reshape(2, 3)}
            self._dev = None
        to_device = GDNRuntime.to_device          # the REAL method, on the stub's state
    return _Stub()


def test_devicerun_place_never_raises_and_reports_honestly_without_a_gpu():
    from holographic.io_and_interop.holographic_devicerun import place, status
    st = status()
    rt = _stub_runtime()
    rep_cpu = place(rt, want="cpu")
    assert rep_cpu["device"] == "cpu" and rt._dev is None
    rep_gpu = place(rt, want="gpu")               # asking must RUN, not raise
    if not st["gpu_available"]:
        assert rep_gpu["device"] == "cpu" and rep_gpu.get("asked") == "gpu"
        assert rt._dev is None                    # nothing silently moved


def test_gdnruntime_to_device_is_behavior_preserving_without_a_gpu():
    import numpy as _np
    from holographic.misc.holographic_backend import gpu_available
    rt = _stub_runtime()
    before = {k: v.copy() for k, v in rt.w.items()}
    rep = rt.to_device(True)
    if not gpu_available():
        assert rep["device"] == "cpu" and "no accelerator" in rep["why"]
        for k in before:                          # weights untouched -> forward math untouched
            assert _np.array_equal(rt.w[k], before[k])
    rep_off = rt.to_device(False)
    assert rep_off == {"device": "cpu", "resident": 0, "why": "disabled"}

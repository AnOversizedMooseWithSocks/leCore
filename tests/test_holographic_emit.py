"""K8 -- dialect emitters. The bar is EXECUTED: the emitted C is compiled with `cc` and run.

leCore's kernels are written once, in Python, and the browser needs them in WGSL. A dialect emitter makes the
hand-written compute shader a projection of the authoritative Python kernel -- one source of truth, two runtimes,
no drift.

WGSL cannot be run here (no GPU, no browser). The C dialect can, and it is: `c_f64` comes out BIT-IDENTICAL to
Python, and `c_f32` -- the executable stand-in for WGSL's single precision -- differs by 8e-08 to 3.4e-07.

    THREE KEPT NEGATIVES:
      1. A WGSL kernel CANNOT be bit-identical to its Python original. WGSL is f32, NumPy is f64. The bar is
         "float tolerance", and the tolerance is f32 epsilon -- not a number anybody chooses.
      2. The emitted WGSL is not executed by any test in this repo. Its arithmetic is validated through `c_f32`,
         which shares the IR; its own precision rules and compilability are NOT. A real gap, stated.
      3. `bind` is not emittable, and that is not a missing feature. It is a circular convolution by FFT -- a
         whole-array cooperative algorithm whose WGSL is a workgroup FFT, a different artifact.
"""

import math

import numpy as np
import pytest

from holographic.io_and_interop.holographic_emit import (
    DIALECTS, EmitError, emit, emit_source, run_c, validate_c)


SDF = """
def sdf_sphere(px: float, py: float, pz: float, r: float) -> float:
    d = sqrt(px * px + py * py + pz * pz)
    return d - r
"""

SMOOTHSTEP = """
def smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)
"""

COSINE = """
def cosine_of(dot: float, na: float, nb: float) -> float:
    return dot / (sqrt(na) * sqrt(nb))
"""


def _live(src):
    """The kernel as a live Python function. `as_python` compiles the SAME TEXT the emitter emits, so the two sides
    of the comparison are one program in two dialects -- not two implementations, which would test neither."""
    from holographic.io_and_interop.holographic_emit import as_python
    return as_python(src)


def test_selftest_runs():
    from holographic.io_and_interop import holographic_emit as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# THE BAR, EXECUTED
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("src,args", [
    (SDF, np.random.default_rng(0).normal(size=(40, 4))),
    (SMOOTHSTEP, np.stack([np.full(40, -1.0), np.full(40, 1.0),
                           np.random.default_rng(1).uniform(-1.5, 1.5, 40)], axis=1)),
    (COSINE, np.random.default_rng(2).uniform(0.2, 3.0, (40, 3))),
])
def test_the_emitted_c_f64_is_bit_identical_to_the_python_original(src, args):
    calls = [tuple(float(v) for v in row) for row in args]
    rep = validate_c(src, calls, "c_f64")                      # TEXT in, both sides from the same text
    assert rep["bit_identical"] is True
    assert rep["max_abs_diff"] == 0.0


@pytest.mark.parametrize("src,args", [
    (SDF, np.random.default_rng(0).normal(size=(40, 4))),
    (COSINE, np.random.default_rng(2).uniform(0.2, 3.0, (40, 3))),
])
def test_kept_negative_c_f32_cannot_be_bit_identical_and_its_error_is_the_wgsl_tolerance(src, args):
    calls = [tuple(float(v) for v in row) for row in args]
    rep = validate_c(src, calls, "c_f32")
    assert rep["bit_identical"] is False
    assert 0.0 < rep["max_abs_diff"] < 1e-5                   # measured 8e-08 .. 3.4e-07
    assert rep["max_rel_diff"] < 1e-5                          # ... which is f32 epsilon, not a chosen number


def test_the_compiled_kernel_actually_runs():
    got = run_c(SDF, [(0.3, -0.7, 1.1, 0.85)], "c_f64")
    assert abs(got[0] - _live(SDF)(0.3, -0.7, 1.1, 0.85)) == 0.0


# ---------------------------------------------------------------------------------------------------------
# the dialects
# ---------------------------------------------------------------------------------------------------------

def test_wgsl_is_structurally_well_formed_and_free_of_python_isms():
    w = emit_source(SDF, "wgsl")
    assert w.startswith("fn sdf_sphere(px: f32, py: f32, pz: f32, r: f32) -> f32")
    assert "let d =" in w and w.rstrip().endswith("}")
    for banned in ("def ", "double", "**", "None", "np."):
        assert banned not in w


def test_every_dialect_emits_the_same_kernel_shape():
    for dialect in DIALECTS:
        out = emit_source(SDF, dialect)
        assert "sdf_sphere" in out and "return" in out and out.count("{") == out.count("}")


def test_the_dialect_table_renames_the_intrinsic():
    assert "sqrtf(" in emit_source(SDF, "c_f32")
    assert "sqrt(" in emit_source(SDF, "c_f64") and "sqrtf(" not in emit_source(SDF, "c_f64")
    assert "sqrt(" in emit_source(SDF, "wgsl")
    assert "Math.sqrt(" in emit_source(SDF, "js")


def test_float_literals_carry_the_dialects_suffix():
    assert "0.0f" in emit_source(SMOOTHSTEP, "wgsl")
    assert "0.0f" in emit_source(SMOOTHSTEP, "c_f32")
    assert "0.0f" not in emit_source(SMOOTHSTEP, "c_f64")


def test_emit_source_takes_text_because_a_kernel_is_text():
    text = "def lerp(a: float, b: float, t: float) -> float:\n    return a + (b - a) * t\n"
    assert emit_source(text, "wgsl").startswith("fn lerp(a: f32, b: f32, t: f32) -> f32")
    assert emit_source(text, "js").startswith("function lerp(a, b, t)")


# ---------------------------------------------------------------------------------------------------------
# K10's RULE: refuse rather than guess
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("src,needle", [
    ("def f(x, y: float) -> float:\n    return x + y\n", "annotation"),
    ("def f(x: int, y: float) -> float:\n    return y\n", "annotation"),
    ("def f(x: float) -> float:\n    return numpy_thing(x)\n", "unknown call"),
    ("def f(x: float) -> float:\n    for i in range(3):\n        x = x + 1.0\n    return x\n", "unsupported statement"),
    ("def f(x: float) -> float:\n    if x > 0.0:\n        return x\n    return 0.0\n", "unsupported statement"),
    ("def f(x: float) -> float:\n    y = x + 1.0\n", "never returns"),
    ("def f(x: float):\n    return x\n", "-> float"),
    ("def f(x: float) -> float:\n    return x % 2.0\n", "unsupported operator"),
    ("def f(x: float) -> float:\n    return True\n", "float constants"),
    ("def f(x: float, *rest) -> float:\n    return x\n", "positional float"),
])
def test_the_emitter_refuses_and_names_the_construct(src, needle):
    with pytest.raises(EmitError, match=needle):
        emit_source(src, "wgsl")


def test_an_unknown_dialect_raises():
    with pytest.raises(EmitError, match="unknown dialect"):
        emit_source(SDF, "glsl")


def test_statements_after_return_are_refused():
    src = "def f(x: float) -> float:\n    return x\n    y = 1.0\n"
    with pytest.raises(EmitError, match="unreachable"):
        emit_source(src, "wgsl")


def test_run_c_refuses_a_dialect_it_cannot_execute():
    with pytest.raises(EmitError, match="cannot be executed here"):
        run_c(SDF, [(1.0, 0.0, 0.0, 0.5)], "wgsl")


def test_a_function_with_no_source_points_at_emit_source():
    fn = eval("lambda x: x")                                   # no retrievable source
    with pytest.raises(EmitError, match="emit_source"):
        emit(fn, "wgsl")


# ---------------------------------------------------------------------------------------------------------
# KEPT NEGATIVE 3: bind is not a scalar kernel
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_bind_is_an_fft_and_is_refused():
    # The backlog names "cosine, a bind, an SDF eval". Two of the three are scalar and project cleanly. `bind` is a
    # circular convolution by FFT -- a whole-array cooperative algorithm. Its WGSL is a workgroup FFT, a different
    # artifact, and a scalar emitter that "supported" it would emit an O(D^2) loop nest and call it a bind.
    src = "def bind(a, b):\n    return _irfft(_rfft(a) * _rfft(b), n=a.shape[0])\n"
    with pytest.raises(EmitError):
        emit_source(src, "wgsl")


def test_the_two_scalar_kernels_the_backlog_names_do_emit():
    for src in (SDF, COSINE):
        assert emit_source(src, "wgsl").startswith("fn ")


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    w = m.emit_kernel(SDF, "wgsl")
    assert "fn sdf_sphere" in w and "-> f32" in w

    rep = m.validate_kernel(SDF, [(0.3, -0.7, 1.1, 0.85), (1.0, 0.0, 0.0, 0.5)], "c_f64")
    assert rep["bit_identical"] is True

    rep32 = m.validate_kernel(SDF, [(0.3, -0.7, 1.1, 0.85)], "c_f32")
    assert rep32["bit_identical"] is False

    assert "Dialect emitters" in str(m.find_capability("emit wgsl")[:3])

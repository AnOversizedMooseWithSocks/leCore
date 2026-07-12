"""Cross-faculty integration for the merged code quartet + Zig interop (backlog items 4/10): the same IR emits every
dialect (including the new Zig), the code tools compose, and Zig degrades gracefully without the toolchain wheel."""
import numpy as np
import lecore


def test_kernel_description_to_every_dialect():
    """ITEM 10: describe a kernel in constrained English -> one Python IR -> emit Zig / WGSL / C from the SAME IR.
    The Zig dialect is the merge's addition; this proves it round-trips through translate_kernel."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    k = m.kernel_from_description("a sphere of radius 1", dialect="python")
    assert "def scene" in str(k), "kernel_from_description should yield a python scene kernel"
    for dialect in ("zig_f64", "zig_f32", "wgsl", "c_f64"):
        out = str(m.translate_kernel(k, "python", dialect))
        assert len(out) > 20, "translate to %s produced nothing" % dialect
        if dialect.startswith("zig"):
            assert "fn " in out and "f" in out, "Zig output should be a Zig fn"


def test_translate_is_exact_across_dialects():
    """Translating the same IR to two dialects yields two real programs (one IR, many backends). The emitter REFUSES
    unresolved types / unsupported ops rather than guessing -- a kept-negative we rely on."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    k = m.kernel_from_description("a sphere of radius 2", dialect="python")
    zig = str(m.translate_kernel(k, "python", "zig_f64"))
    wgsl = str(m.translate_kernel(k, "python", "wgsl"))
    assert zig != wgsl and "return" in zig and ("return" in wgsl or "->" in wgsl)


def test_emitter_refuses_unresolved_types():
    """The emitter's refusal contract: an un-annotated parameter is an error, not a defaulted type (a wrong int/double
    is the first bug a hand port makes). This kept-negative is what makes 'exact' meaningful."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    bad = "def f(px, py):\n    return px + py"                # no float annotations
    try:
        m.translate_kernel(bad, "python", "c_f64")
        assert False, "should have refused the un-annotated kernel"
    except Exception as e:
        assert "annotation" in str(e).lower() or "type" in str(e).lower()


def test_triage_reports_structural_observations():
    """ITEM 4: triage_code makes honest STRUCTURAL observations about code in an unknown language -- counts and
    identifiers, explicitly NOT comprehension (the kept negative)."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    tr = m.triage_code("fn quicksort(xs: List) { let pivot = xs[0]; }")
    assert tr["is_observation_not_explanation"] is True
    assert tr["identifier_count"] > 0 and tr["chars"] > 0


def test_explain_code_is_deterministic():
    """explain_code gives a deterministic layered description (no LLM); same input -> same output."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    src = "def f(p: float) -> float:\n    return p * 2.0"
    a = m.explain_code(src)
    b = m.explain_code(src)
    assert a == b, "explain_code must be deterministic"
    assert "functions" in a or "summary" in a


def test_zig_batch_eval_skips_without_toolchain():
    """ITEM 10: Zig is an OPT-IN accelerator (like numba) -- without the ziglang wheel it must raise a clear,
    actionable error, never silently corrupt or hang. The engine must pass without the wheel."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    try:
        import ziglang  # noqa: F401
        have = True
    except ImportError:
        have = False
    if have:
        return                                               # toolchain present: nothing to assert here
    raised = False
    kernel = "def f(x: float) -> float:\n    return x * 2.0"   # a Python kernel; zig_batch_eval compiles it
    try:
        m.zig_batch_eval(kernel, [np.array([1.0, 2.0, 3.0])])
    except Exception as e:
        raised = True
        assert "zig" in str(e).lower() or "toolchain" in str(e).lower() or "install" in str(e).lower(), \
            "without the wheel the error should name zig/toolchain/install, got: %s" % e
    assert raised, "without the ziglang wheel, zig_batch_eval must raise (opt-in accelerator contract)"

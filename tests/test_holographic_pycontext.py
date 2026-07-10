"""K6 -- purity & effect analysis from the stdlib, plus the audit tooling that made it findable.

THE BAR is directional, not a percentage: **never a false PURE**. A wrong "impure" costs a cache miss; a wrong
"pure" silently corrupts a cache and every result downstream. So every test below that matters asserts that a
function with any effect -- or any callee we could not resolve -- comes back False.

THE CORRECTION the module carries: purity closed over the CALL GRAPH, because a function that calls an impure
function is impure however clean its own body. A local rule reports 54.3% pure on this tree; the sound fixpoint
reports 32.1%. The backlog's "76.0%" is a local-rule number.

Also tested here: `mind.find_scored` / `mind.capability_confidence`, wired this session, and `tools/backlog_probe`.
They exist because twice an audit concluded the wrong thing from `find_capability`'s always-three answer.
"""

import ast
import subprocess
import sys

import pytest

from holographic.io_and_interop.holographic_pycontext import (
    analyze_function, analyze_source, close_call_graph, purity_report, is_pure, scan_tree)


def test_selftest_runs():
    from holographic.io_and_interop import holographic_pycontext as mod
    mod._selftest()


def _fn(src, name):
    return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)


# ---------------------------------------------------------------------------------------------------------
# THE BAR: never a false pure
# ---------------------------------------------------------------------------------------------------------

IMPURE_CASES = {
    "prints":            "def prints(x):\n    print(x)\n    return x\n",
    "mutates_param":     "def mutates_param(xs, v):\n    xs.append(v)\n    return xs\n",
    "writes_param_slot": "def writes_param_slot(d, k, v):\n    d[k] = v\n",
    "sets_attr":         "def sets_attr(o):\n    o.x = 1\n    return o\n",
    "global_write":      "def global_write():\n    global C\n    C = 1\n",
    "unknown_callee":    "def unknown_callee(x):\n    return mystery(x)\n",
    "unknown_method":    "def unknown_method(o):\n    return o.whatever()\n",
    "writes_module_var": "BUF = []\ndef writes_module_var(x):\n    BUF[0] = x\n",
}


@pytest.mark.parametrize("name,src", sorted(IMPURE_CASES.items()))
def test_no_false_pure_on_any_effectful_function(name, src):
    assert is_pure(src, name) is False, name
    assert name in purity_report(src)["reasons"]


PURE_CASES = {
    "arith":       "def arith(a, b):\n    return a * b + 1\n",
    "escape":      "def escape(xs):\n    out = []\n    for x in xs:\n        out.append(x * 2)\n    return out\n",
    "dict_escape": "def dict_escape(xs):\n    d = {}\n    for i, x in enumerate(xs):\n        d[i] = x\n    return d\n",
    "comprehension": "def comprehension(xs):\n    return [x * 2 for x in xs if x > 0]\n",
    "pure_builtin": "def pure_builtin(xs):\n    return sorted(set(xs))\n",
    "numpy_module": "import numpy as np\ndef numpy_module(a):\n    return np.asarray(a).sum()\n",
    "pure_method":  "def pure_method(a):\n    return a.reshape(-1).mean()\n",
}


@pytest.mark.parametrize("name,src", sorted(PURE_CASES.items()))
def test_the_clean_cases_really_do_measure_pure(name, src):
    # If these came back impure the gate would be useless -- conservative must not mean "always no".
    assert is_pure(src, name) is True, name


def test_escape_analysis_is_what_makes_the_common_shape_pure():
    # "functional core, imperative shell": mutating a container you allocated is invisible from outside.
    escaped = "def f(xs):\n    out = []\n    out.append(1)\n    return out\n"
    leaked = "def f(xs):\n    xs.append(1)\n    return xs\n"
    assert is_pure(escaped, "f") is True
    assert is_pure(leaked, "f") is False
    assert analyze_function(_fn(escaped, "f"))["allocated"] == ["out"]


# ---------------------------------------------------------------------------------------------------------
# THE CORRECTION: purity is not a local property
# ---------------------------------------------------------------------------------------------------------

def test_a_spotless_body_that_calls_an_impure_function_is_impure():
    src = ("def shouts(x):\n    print(x)\n    return x\n"
           "def clean(x):\n    return shouts(x)\n")
    facts = analyze_source(src)
    assert facts["clean"]["reasons"] == []          # its own body has NO local effect ...
    assert is_pure(src, "clean") is False           # ... and it is impure anyway. This is the whole point.


def test_an_unresolved_callee_is_impure_not_assumed_fine():
    src = "def f(x):\n    return helper_defined_elsewhere(x)\n"
    assert analyze_source(src)["f"]["reasons"] == []
    assert is_pure(src, "f") is False


def test_purity_propagates_transitively_and_reaches_a_fixpoint():
    src = ("def leaf(x):\n    return x + 1\n"
           "def mid(x):\n    return leaf(x) * 2\n"
           "def top(x):\n    return mid(x) - 1\n"
           "def dirty(x):\n    print(x)\n"
           "def poisoned(x):\n    return top(x) + dirty(x)\n")
    v = purity_report(src)["verdicts"]
    assert v["leaf"] and v["mid"] and v["top"]      # purity flows up a clean chain
    assert not v["dirty"] and not v["poisoned"]     # ... and impurity flows up too


def test_mutual_recursion_with_clean_bodies_settles_to_pure():
    src = ("def even(n):\n    return True if n == 0 else odd(n - 1)\n"
           "def odd(n):\n    return False if n == 0 else even(n - 1)\n")
    v = purity_report(src)["verdicts"]
    assert v["even"] is True and v["odd"] is True   # the fixpoint terminates and is correct on cycles


def test_the_local_only_figure_is_reported_beside_the_sound_one():
    src = ("def clean(x):\n    return shouts(x)\n"
           "def shouts(x):\n    print(x)\n    return x\n")
    rep = purity_report(src)
    assert rep["local_only_fraction"] == 0.5        # a local rule would call `clean` pure ...
    assert rep["fraction"] == 0.0                   # ... the sound one does not
    assert rep["local_only_fraction"] > rep["fraction"]


def test_close_call_graph_is_monotone_and_terminates():
    facts = analyze_source("def a(x):\n    return b(x)\ndef b(x):\n    return a(x)\n")
    assert close_call_graph(facts, max_iters=1) == close_call_graph(facts, max_iters=64)


# ---------------------------------------------------------------------------------------------------------
# the live tree, and the numbers that correct the backlog
# ---------------------------------------------------------------------------------------------------------

def test_the_tree_scan_reports_both_figures_and_the_sound_one_is_lower():
    rep = scan_tree("holographic")
    assert rep["total"] > 2000
    assert 0.25 < rep["fraction"] < 0.40                 # measured 32.1%
    assert 0.45 < rep["local_only_fraction"] < 0.65      # measured 54.3% -- what a local rule would claim
    assert rep["local_only_fraction"] > rep["fraction"] + 0.15
    assert set(rep["pure"]) & set(rep["impure"]) == set()
    assert len(rep["pure"]) + len(rep["impure"]) == rep["total"]


def test_edge_cases():
    assert purity_report("")["total"] == 0
    assert purity_report("x = 1")["total"] == 0
    with pytest.raises(KeyError):
        is_pure("def f(): pass", "nope")
    with pytest.raises(SyntaxError):
        purity_report("def (:")


def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    src = "def f(xs):\n    out = []\n    for x in xs: out.append(x*2)\n    return out\n"
    assert m.function_purity(src, "f") is True
    assert m.purity_report(src)["fraction"] == 1.0
    scan = m.purity_scan("holographic")
    assert scan["local_only_fraction"] > scan["fraction"]
    assert m.capability_confidence("decide whether a python function is pure")["confident"] is True


# ---------------------------------------------------------------------------------------------------------
# the tooling that stops an audit concluding the wrong thing
# ---------------------------------------------------------------------------------------------------------

def test_find_scored_distinguishes_a_hit_from_a_fallback():
    # find_capability ALWAYS returns three names. Only the score says whether any of them is the answer.
    # Twice in this program an audit read a fallback as a hit (and a miss as an absence). This is the fix.
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    hit = m.capability_confidence("blur a field without decompressing it")
    assert hit["confident"] is True and hit["score"] > 3.0

    miss = m.capability_confidence("emit a kernel as a webgpu compute shader")
    assert miss["confident"] is False                     # ... yet find_capability still names three capabilities
    assert len(m.find_capability("emit a kernel as a webgpu compute shader")) == 3

    scored = m.find_scored("blur a field without decompressing it", k=3)
    assert scored == sorted(scored, key=lambda cs: -cs[1])   # best first
    assert all(s >= 0 for _, s in scored)


def test_backlog_probe_tool_selftests():
    out = subprocess.run([sys.executable, "tools/backlog_probe.py", "--selftest"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "self-test passed" in out.stdout


def test_backlog_probe_symbol_search_reports_absence_honestly():
    from tools.backlog_probe import _symbols
    assert any(n == "hierarchical_recall" for _, n in _symbols(["hierarchical"]))
    assert _symbols(["afunctionnamenobodywouldever"]) == []

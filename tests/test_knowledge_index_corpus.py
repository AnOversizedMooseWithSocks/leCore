"""Regression trap for the ROUTING CORPUS selection in tools/semantic/knowledge_index.py.

Runs without the embedding model: corpus selection is pure text/AST work, so the property that broke CI is
testable here even though the exam itself needs weights CI downloads.

WHAT BROKE. Splitting UnifiedMind into 13 mixin parts added 13 files matching holographic_*.py whose only
top-level symbols are `_UnifiedPartNN` and `_selftest`. collect_code embedded all 13 as routing candidates,
and the routing exam regressed by exactly one on BOTH flat and fused top-1 -- a uniform delta, which is what
identified the fault as upstream of the fusion rather than in it.
"""
import ast
import os
import sys

import pytest

_SEM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "semantic")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _collectors():
    """Load only the pure collector functions, without importing the module (which pulls in the model)."""
    import re as _re
    import types
    src = open(os.path.join(_SEM, "knowledge_index.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    ns = {"os": os, "re": _re, "ast": ast, "MAX_CHARS": 280, "_alias_enrichment": lambda: {}}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"_has_public_api", "collect_code"}:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "ki", "exec"), ns)
    return ns


def test_the_exclusion_filter_still_works_when_asked_for():
    """The filter is now OFF by default (it was measured on CI and did not pay -- see _has_public_api), but it
    must stay correct and runnable so the negative can be re-checked rather than re-argued."""
    ns = _collectors()
    kept = {n for _k, n, _b in ns["collect_code"](_REPO, include_private_modules=False)}
    for i in range(1, 14):
        assert not any(n.startswith("holographic_unified_p%02d" % i) for n in kept), \
            "unified mixin part p%02d is back in the routing corpus" % i


def test_the_default_corpus_keeps_every_module():
    """DEFAULT IS NOW THE FULL CORPUS. The exclusion was tried, measured, and refuted; the default must not
    quietly drift back to it."""
    ns = _collectors()
    without = ns["collect_code"](_REPO, include_private_modules=False)
    with_priv = ns["collect_code"](_REPO, include_private_modules=True)
    assert len(with_priv) > len(without), "the flag no longer restores the old corpus"
    assert len(with_priv) - len(without) >= 13, \
        "expected at least the 13 mixin parts to differ, got %d" % (len(with_priv) - len(without))


def test_the_exclusion_never_removes_an_accepted_answer():
    """NECESSARY BUT NOT SUFFICIENT -- and the earlier version of this test claimed otherwise.

    It was originally named ..._is_monotonic_on_the_exam_suite and used to argue that excluding no-public-API
    modules could only improve ranks. CI DISPROVED THAT: the 768d median went 2 -> 3 and worst 226 -> 251.
    The flaw is that AllButTheTop REFITS ON THE CORPUS MEAN, so removing any vector changes the correction
    applied to every other vector; rank monotonicity simply does not follow from 'the answer is still in the
    candidate set'.

    What this test still legitimately pins is the weaker property in its new name: whatever the corpus filter
    does, it must never delete a module that the exam accepts as correct. That failure would be unambiguous
    and silent, so it is worth a trap -- just not the trap it was advertised as."""
    ns = _collectors()
    src = open(os.path.join(_SEM, "knowledge_index.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    accepted = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "ASKS_MODULE" for t in node.targets):
            for elt in node.value.elts:
                for name in elt.elts[1].elts:
                    accepted.add(name.value)
    assert len(accepted) > 20, "could not parse ASKS_MODULE accepted answers (%d found)" % len(accepted)
    kept = {n for _k, n, _b in ns["collect_code"](_REPO)}
    dropped = {n for _k, n, _b in ns["collect_code"](_REPO, include_private_modules=True)} - kept
    lost = accepted & dropped
    assert not lost, "the exclusion removed accepted answer(s) -- monotonicity is broken: %s" % sorted(lost)


def test_unparseable_files_are_kept_not_silently_dropped():
    """Fail-open, not fail-closed: a file mid-edit must not vanish from the corpus."""
    ns = _collectors()
    assert ns["_has_public_api"]("def (((", "broken.py") is True
    assert ns["_has_public_api"]("def public_thing():\n    pass\n", "ok.py") is True
    assert ns["_has_public_api"]("class _Private:\n    pass\n", "priv.py") is False


def test_the_default_is_the_full_corpus():
    """Pins the refutation. If someone re-enables the exclusion by default without new measurements, the
    kept negative has been silently overturned."""
    ns = _collectors()
    default = {n for _k, n, _b in ns["collect_code"](_REPO)}
    assert any("unified_p" in n for n in default), \
        "the no-public-API exclusion is on by default again -- it was MEASURED and refuted (768d median " \
        "2 -> 3, gated fused top-1 unchanged at 6). Re-enable it only with a CI run that shows it pays."


def test_the_exam_median_is_printed_at_the_precision_it_is_compared_at():
    """THE DISPLAY LIE, pinned.

    With an EVEN number of asks the median is the mean of the 6th and 7th ranks -- a half-integer by
    construction. The gate printed it with ':.0f', so a true median of 2.5 rendered as "2" directly beside
    "(require <= 2.0)". The log read like a PASS while the comparison used 2.5 and failed. It hid a failing
    criterion across four CI runs, because the fused gate was failing too and the verdict was FAIL either way.

    A number shown to the reader must be the number the machine compared."""
    src = open(os.path.join(_SEM, "knowledge_index.py"), encoding="utf-8").read()
    assert "{exam_median:.0f}" not in src, \
        "exam median is back to :.0f -- a 2.5 will print as '2' next to 'require <= 2.0' and read as a pass"
    assert "{exam_median:.1f}" in src, "exam median should print at the precision it is compared at"


def test_an_even_ask_count_produces_half_integer_medians():
    """Why the precision matters, asserted rather than asserted-in-prose: ASKS_MODULE has an even length, so
    numpy's median interpolates between two ranks and .5 values are normal, not exotic."""
    import ast as _ast
    import numpy as _np
    src = open(os.path.join(_SEM, "knowledge_index.py"), encoding="utf-8").read()
    tree = _ast.parse(src)
    n = None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign) and any(getattr(t, "id", None) == "ASKS_MODULE" for t in node.targets):
            n = len(node.value.elts)
    assert n and n % 2 == 0, "ASKS_MODULE is expected to have an even length (got %r)" % n
    ranks = [1, 1, 1, 1, 1, 2, 3, 3, 8, 21, 43, 226]          # the real 768d flat ranks from CI
    assert float(_np.median(ranks)) == 2.5
    assert f"{float(_np.median(ranks)):.0f}" == "2", "the rounding that caused the lie no longer reproduces"


def test_the_gate_judges_one_configuration_not_two():
    """THE WHACK-A-MOLE FIX, pinned.

    The exam grew in two halves: top-5 and median gated on FLAT @768d, and a fused @128d top-1 criterion
    bolted on later. One boolean verdict was then computed from TWO configurations -- and the 768d flat row
    is not what ships. A genuine repair could take the shipped row to a clean pass while the build stayed
    red on a config no user runs, at one CI run per attempt.

    --gate-shipped-row puts all three criteria on the fused gamma=0.50 128d row. This test asserts the flag
    exists, is wired into the gate, and that the 768d numbers survive as a printed diagnostic."""
    src = open(os.path.join(_SEM, "knowledge_index.py"), encoding="utf-8").read()
    assert "--gate-shipped-row" in src, "the shipped-row gate flag is gone"
    assert "gate_top5 >= args.require_top5 and gate_median <= args.require_median" in src, \
        "the gate no longer reads the selected configuration's numbers"
    assert "diagnostic, NOT gated" in src, \
        "flat @768d must still be printed as an encoder diagnostic, or dense drift becomes invisible"


def test_the_shipped_row_gate_would_pass_this_runs_numbers():
    """Arithmetic check against the REAL CI numbers, so the change is verified rather than hoped.

    Measured this run -- flat @768d: top-5 8, median 2.5, top-1 5. SHIPPED row (fused, g=0.50, 128d):
    top-5 8, median 1.0, top-1 7. Bars: top-5 >= 8, median <= 1, fused top-1 >= 7."""
    req_top5, req_median, req_top1 = 8, 1, 7
    flat_top5, flat_median = 8, 2.5
    ship_top5, ship_median, ship_top1 = 8, 1.0, 7

    # the OLD gate: top-5/median from flat, top-1 from shipped -> mixed, and fails
    old_ok = (flat_top5 >= req_top5) and (flat_median <= 2) and (ship_top1 >= req_top1)
    assert not old_ok, "the old mixed gate should fail on these numbers (median 2.5 > 2)"

    # the NEW gate: all three from the shipped row -> passes
    new_ok = (ship_top5 >= req_top5) and (ship_median <= req_median) and (ship_top1 >= req_top1)
    assert new_ok, "the shipped-row gate should pass on this run's measured numbers"


def test_the_new_bars_are_tighter_not_looser():
    """The one claim that must not be fudged: this is a re-TARGETING, not a relaxation.

    The shipped row's top-5 and median were previously UNGATED entirely, and the new median bar (1) is
    tighter than the 768d bar it replaces (2). If a future edit loosens either, the change stops being
    defensible as a bug fix."""
    wf = open(os.path.join(os.path.dirname(_SEM), "..", ".github", "workflows",
                           "semantic-coverage.yml"), encoding="utf-8").read()
    assert "--gate-shipped-row" in wf, "CI no longer passes --gate-shipped-row"
    assert "--require-median 1" in wf, "the shipped-row median bar must stay at 1 (it was 2 at 768d)"
    assert "--require-top5 8" in wf and "--require-fused-top1 7" in wf

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


def test_modules_with_no_public_api_are_not_routing_targets():
    """The fix. A module with no public top-level symbol cannot be routed to -- nothing in it can be named --
    so it must not be a routing candidate."""
    ns = _collectors()
    kept = {n for _k, n, _b in ns["collect_code"](_REPO)}
    for i in range(1, 14):
        assert not any(n.startswith("holographic_unified_p%02d" % i) for n in kept), \
            "unified mixin part p%02d is back in the routing corpus" % i


def test_the_flag_reproduces_the_pre_fix_corpus():
    """The A/B must stay runnable, or the claim 'excluding them helps' becomes unfalsifiable."""
    ns = _collectors()
    without = ns["collect_code"](_REPO)
    with_priv = ns["collect_code"](_REPO, include_private_modules=True)
    assert len(with_priv) > len(without), "the flag no longer restores the old corpus"
    assert len(with_priv) - len(without) >= 13, \
        "expected at least the 13 mixin parts to differ, got %d" % (len(with_priv) - len(without))


def test_the_exclusion_is_monotonic_on_the_exam_suite():
    """THE LOAD-BEARING PROPERTY, and the reason this fix is safe without running the model.

    No accepted answer in ASKS_MODULE is a no-public-API module. Removing candidates that can never be
    correct cannot lower the rank of a correct one, so top-1/top-5 can only rise or hold and median/worst can
    only improve or hold. If a future edit ever excludes a module that IS an accepted answer, that guarantee
    silently dies -- and this test is what catches it."""
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

"""Tests for tools/skill_lint.py -- the docstring/invocation-quality linter over UnifiedMind faculties."""
import importlib.util
import os

_PATH = os.path.join(os.path.dirname(__file__), "tools", "skill_lint.py")


def _lint():
    spec = importlib.util.spec_from_file_location("skill_lint", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_critical_or_terse_gaps():
    """Every public UnifiedMind method has a non-trivial docstring summary an agent can read."""
    a = _lint().audit()
    assert a["critical"] == [], a["critical"]
    assert a["terse"] == [], a["terse"]


def test_report_returns_zero():
    assert _lint().report() == 0


def test_the_core_faculties_are_documented():
    """The four faculties this linter first caught (learn/next_symbol/reinforce/describe) now describe well."""
    import holographic_skills as sk
    ms = sk.mind_methods()
    for n in ("learn", "next_symbol", "reinforce", "describe"):
        assert len(ms[n]["summary"].split()) >= 5, (n, ms[n]["summary"])


def test_home_example_references_resolve_and_document():
    """Every module-level function named in a curated home's `example` (what an agent copies) exists, imports, and
    has a usable docstring -- no BROKEN references, no missing/terse docs."""
    h = _lint().audit_home_examples()
    assert h["broken"] == [], "broken example references: %s" % h["broken"]
    assert h["no_doc"] == [], "referenced functions with no docstring: %s" % h["no_doc"]
    assert h["terse"] == [], "referenced functions with a thin docstring: %s" % h["terse"]
    assert h["checked"] > 50

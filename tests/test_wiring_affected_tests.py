"""mind.affected_tests -- the DOOR added onto tools/select_tests.py + tools/test_changed.py. Both already existed,
both already had their own tests (test_select_tests.py) and their own CI wiring (ci.yml runs select_tests on every
push/PR) -- but NEITHER was reachable through UnifiedMind. `find_capability` returned only unrelated fallbacks for
"which tests should I run for my change" / "reduce how many tests run every commit" / etc. (audited with 7
phrasings before building; see NOTES_concepts.md). This file tests the DOOR itself: that it delegates faithfully
(never reimplements the selection logic) and is now discoverable. The selector's own correctness is
tests/test_select_tests.py's job, not this file's.
"""
import os

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _a_known_module():
    """A real project module + its test file, discovered from the import graph (not hard-coded) so this survives
    any future reorg -- mirrors test_select_tests.py's own _rel_of helper."""
    from tools.select_tests import build_graph
    mods, _ = build_graph(ROOT)
    for dotted, rel in mods.items():
        if dotted.split(".")[-1] == "holographic_render":
            return rel
    raise AssertionError("holographic_render.py not found -- can't run this test meaningfully")


def test_delegates_faithfully_not_a_reimplementation():
    """The mind's answer for an explicit changed-file list must be BIT-IDENTICAL to calling tools.select_tests
    directly with the same arguments -- if these ever diverge, the wiring re-derived the logic instead of
    delegating to it (the thing Faculty rule #2 forbids)."""
    import lecore
    from tools.select_tests import affected_tests as direct

    m = lecore.UnifiedMind(dim=64, seed=0)
    m.set_file_root(ROOT)
    rel = _a_known_module()

    via_mind = m.affected_tests(changed_paths=[rel])
    via_tool = direct([rel], root=ROOT)
    assert via_mind == via_tool
    assert via_mind != "ALL"
    assert any(os.path.basename(p) == "test_holographic_render.py" for p in via_mind)


def test_docs_only_change_selects_nothing():
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    m.set_file_root(ROOT)
    assert m.affected_tests(changed_paths=["README.md"]) == []


def test_unknown_file_fails_safe_to_all():
    """An unscopable change (unknown extension / not in the module map) must widen to "ALL", never silently
    narrow -- the safety property the whole faculty exists to preserve."""
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    m.set_file_root(ROOT)
    assert m.affected_tests(changed_paths=["features/sprites.hsp"]) == "ALL"


def test_auto_detect_threads_changed_files_through(monkeypatch):
    """With changed_paths=None, the mind must call tools.test_changed.changed_files() and feed its output into the
    SAME selection tools.select_tests.affected_tests() would give directly -- tested via monkeypatch so the result
    is deterministic and doesn't depend on the ambient git state of whatever checkout the suite happens to run in
    (asserting on live git diff output would be flaky-by-construction, not a real regression trap)."""
    import lecore
    import tools.test_changed as test_changed_mod
    from tools.select_tests import affected_tests as direct

    rel = _a_known_module()
    monkeypatch.setattr(test_changed_mod, "changed_files", lambda since=None: [rel])

    m = lecore.UnifiedMind(dim=64, seed=0)
    m.set_file_root(ROOT)
    result = m.affected_tests()          # no args -> must go through the (monkeypatched) auto-detect path
    assert result == direct([rel], root=ROOT)


def test_no_changes_short_circuits_without_calling_the_selector(monkeypatch):
    """An empty change list (clean tree) must return [] directly -- not call into affected_tests at all, since
    there's nothing to scope. Verified by making the underlying selector raise if it's ever reached."""
    import lecore
    import tools.test_changed as test_changed_mod
    import tools.select_tests as select_tests_mod

    monkeypatch.setattr(test_changed_mod, "changed_files", lambda since=None: [])

    def _boom(*a, **kw):
        raise AssertionError("affected_tests should not be called when there are no changed files")
    monkeypatch.setattr(select_tests_mod, "affected_tests", _boom)

    m = lecore.UnifiedMind(dim=64, seed=0)
    m.set_file_root(ROOT)
    assert m.affected_tests() == []


def test_discoverable_via_find_capability():
    """The whole point: a stranger's phrasing of the actual problem must now surface this capability, where before
    the audit it returned only unrelated fallbacks (Code / file editing, Partition-invariant sums, etc.)."""
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    for q in ("which tests should I run for my change", "reduce how many tests run every commit",
              "avoid running the full test suite locally"):
        hits = [c.name for c in m.find_capability(q)[:3]]
        assert any("affected" in h.lower() or "test" in h.lower() and "select" in h.lower() or
                   "affected-test" in h.lower() for h in hits), (q, hits)

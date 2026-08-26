"""Regression traps for function-granularity reachability (holographic_orphanaudit).

The tool invites action on its output -- "this function is reachable by nothing" -- so its FALSE POSITIVE
behaviour is the contract that matters, not its recall. Every test here pins conservatism.
"""
import os
import tempfile

import pytest

from holographic.io_and_interop.holographic_orphanaudit import (
    audit, orphan_report, public_definitions, referenced_names, ORPHAN_BUDGET)


def _write(d, name, text):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def test_a_name_mentioned_only_in_a_string_counts_as_reached():
    """THE FALSE-ORPHAN TRAP. This repo routes by name through the catalog's `example=` snippets, so a
    function whose only mention is inside a string literal is genuinely reachable. Dropping string mentions
    would manufacture orphans -- the one error a tool like this must never make."""
    with tempfile.TemporaryDirectory() as d:
        a = _write(d, "a.py", "def by_string():\n    pass\n")
        b = _write(d, "b.py", "EXAMPLE = 'mind.by_string()'\n")
        assert "by_string" in referenced_names([a, b])


def test_same_file_callers_count():
    """A helper called only by its own module's public API is REACHED. An earlier version discarded all
    same-file uses and over-reported by 5x against an independent oracle."""
    with tempfile.TemporaryDirectory() as d:
        a = _write(d, "a.py", "def helper():\n    pass\n\n\ndef public():\n    return helper()\n")
        assert "helper" in referenced_names([a])


def test_decorated_functions_are_never_audited():
    """A decorated function is REGISTERED, not called -- Flask routes, properties, staticmethods. Absence of
    a caller proves nothing about them, and auditing them produced 23 false orphans (every HTTP handler)."""
    with tempfile.TemporaryDirectory() as d:
        a = _write(d, "a.py", "def plain():\n    pass\n\n\n@app.route('/x')\ndef routed():\n    pass\n")
        defs = public_definitions([a])
        assert "plain" in defs and "routed" not in defs


def test_private_names_are_not_audited():
    with tempfile.TemporaryDirectory() as d:
        a = _write(d, "a.py", "def _helper():\n    pass\n")
        assert public_definitions([a]) == {}


def test_buckets_partition_exactly_and_stay_within_budget():
    """Every public engine function lands in exactly one bucket, and the orphan gate holds."""
    b = audit()
    names = [n for v in b.values() for n, _p, _l in v]
    assert len(names) == len(set(names)), "a function landed in two buckets"
    assert len(names) > 2000, "the scan missed the tree (%d functions)" % len(names)
    assert len(b["orphan"]) <= ORPHAN_BUDGET, (
        "orphans rose to %d (budget %d) -- wire, catalogue or declare, but do not delete on a static signal"
        % (len(b["orphan"]), ORPHAN_BUDGET))


def test_the_dynamic_oracles_actually_loaded():
    """If the mind or the catalog fails to import, the buckets silently over-report -- which is exactly how
    this tool first reported 0 faculties and 537 orphans. A near-empty faculty bucket means a broken oracle,
    not a broken engine."""
    b = audit()
    assert len(b["faculty"]) > 1000, (
        "faculty oracle looks broken (%d) -- a mind failed to boot and the counts are meaningless"
        % len(b["faculty"]))
    assert len(b["catalog"]) > 300, "catalog oracle looks broken (%d)" % len(b["catalog"])


def test_mind_faculty_round_trip():
    """leCore auditing leCore, through the front door."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    r = m.audit_orphans(limit=5)
    assert set(r["counts"]) == {"faculty", "catalog", "engine", "test_only", "tool_only", "orphan"}
    assert r["ok"] is True and r["budget"] == ORPHAN_BUDGET
    assert len(r["orphan"]) <= 5 and len(r["test_only"]) <= 5
    assert all({"name", "path", "line"} <= set(d) for d in r["orphan"])
    assert "your_capability" not in str(m.find_capability("find dead code")[:1])
    assert "Function-granularity" in str(m.find_capability("find dead code")[0])


# ---------------------------------------------------------------------------
# AGENT REACHABILITY -- the second question: does the reference GO anywhere?
# ---------------------------------------------------------------------------

def test_consolidation_home_is_a_cul_de_sac():
    """The single load-bearing rule. A `*home.py` facade is import-only BY DESIGN, so a reference from one
    is not a route to an agent. If this ever inverts, `shadowed` silently goes to zero and the audit reads
    CLEAN while being blind -- which is the exact failure the pass exists to catch."""
    from holographic.io_and_interop.holographic_orphanaudit import _is_terminal
    assert not _is_terminal("/x/holographic_lightinghome.py")
    assert not _is_terminal("/x/holographic_lookahead.py"), "declared negatives are cul-de-sacs too"
    assert _is_terminal("/x/holographic_lights.py")
    # ...and the service is the OPPOSITE of a cul-de-sac: it is the door agents come through. The first
    # draft classified it as plumbing and manufactured five false positives in a single run.
    assert _is_terminal("/x/holographic_service.py")


def test_classes_are_part_of_the_surface():
    """audit() collects FunctionDef only, so a class an agent cannot construct was invisible to it by
    construction. public_classes is the half of the surface nobody was auditing."""
    from holographic.io_and_interop.holographic_orphanaudit import public_classes, public_definitions
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.py")
        with open(p, "w") as f:
            f.write("class Public:\n    def meth(self):\n        pass\n\n\nclass _Private:\n    pass\n")
        assert set(public_classes([p])) == {"Public"}, "public classes must be collected"
        assert "Public" not in public_definitions([p]), "the function scan must stay unchanged (additive)"


def test_from_import_counts_as_a_reference():
    """`from lights import DomeLight` binds a name that is neither a Name nor an Attribute node. This repo
    has been bitten by that before -- a sweep found 43 facade imports the wiring audit could not see -- so
    missing it here would manufacture findings in the one direction that costs live code."""
    from holographic.io_and_interop.holographic_orphanaudit import _referenced_in
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.py")
        with open(a, "w") as f:
            f.write("from thing import Named as Aliased\n")
        where = _referenced_in([a])
        assert a in where.get("Named", set()), "the real name must be seen"
        assert a in where.get("Aliased", set()), "the alias must be seen"


def test_agent_reach_through_the_mind():
    """Cross-faculty: the audit runs through the front door and its finding is DISCOVERABLE. A capability
    find_capability cannot surface does not exist, and an audit nobody can find is worth nothing."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    r = m.audit_agent_reach(limit=5)
    assert set(r["counts"]) == {"shadowed", "dark_classes"}
    assert len(r["shadowed"]) <= 5 and len(r["dark"]) <= 5
    assert all({"name", "path", "line"} <= set(e) for e in r["dark"])
    assert r["counts"]["dark_classes"] > 100, "the class scan is not seeing the tree"
    assert "Agent reachability" in str(m.find_capability("which classes can I not construct")[0])
    # the orphan audit must be UNCHANGED by all of this -- additive means additive
    assert set(m.audit_orphans(limit=1)["counts"]) == {
        "faculty", "catalog", "engine", "test_only", "tool_only", "orphan"}


def test_lighting_classes_expose_the_factory_blind_spot():
    """The trap I wrote one item ago said "when DomeLight is finally wired this test SHOULD fail". It was
    wired -- mind.scene_light('dome') builds one -- and this test did NOT fail. That is the finding.

    `dark` asks whether the class NAME is a faculty or appears in catalog text. A class reachable only
    through a factory door is genuinely constructible by an agent and still reads as dark. Measured: nine
    light classes went reachable behind one factory and dark_classes moved 312 -> 311, the single move
    caused by catalog prose rather than by the fix.

    Both facts are pinned here on purpose, so the blind spot cannot go quiet: the class IS reachable, and
    the audit does NOT see it. Crediting "constructed by a faculty-reachable function in the same module"
    would clear all nine in one line and over-credit everything else -- a metric edited to score its
    author's work. The honest reading of `dark` is 'not constructible BY NAME'."""
    import lecore
    from holographic.io_and_interop.holographic_orphanaudit import agent_reach_report
    m = lecore.UnifiedMind(dim=128, seed=0)
    # fact 1: an agent CAN build one, through the front door, with no import past lecore
    assert type(m.scene_light("dome")).__name__ == "DomeLight"
    # fact 2: and the audit still reports it dark. When someone builds the constructor-edge version, THIS
    # is the assertion that should fail -- and that failure is the good news, so update it, don't relax it.
    dark = {e["name"] for e in agent_reach_report(limit=10000)["dark"]}
    assert "DomeLight" in dark, "constructor edges are now followed -- rewrite the negative in the module head"
    # RectLight is missed for the OTHER reason: the catalog oracle is lexical and prose names it.
    assert "RectLight" not in dark, "if this fires the catalog oracle changed -- rewrite the negative above"

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

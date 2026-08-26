"""Traps for the shared L3 source index -- and above all for its CALL PATTERN.

The index itself is easy. The thing that must never regress is HOW OFTEN its key is computed: the first
version resolved the index inside the per-file helper, recomputing a 39.5 ms tree digest on every one of
3,543 lookups, and the orphan audit went from 7.3 s to 87 s -- TWELVE TIMES SLOWER WITH THE CACHE THAN
WITHOUT. A cache's amortisation is a property of the call pattern, not of the cache.
"""
import time

import pytest

from holographic.io_and_interop import holographic_srcindex as si
from holographic.scene_and_pipeline.holographic_compile import CompileCache


def test_index_is_built_once_per_distinct_tree():
    c = CompileCache(maxsize=4)
    a = si.parsed_trees(cache=c)
    b = si.parsed_trees(cache=c)
    assert a is b, "the second call rebuilt instead of hitting"
    assert c.stats["compiles"] == 1
    assert len(a) > 400, "only %d files indexed" % len(a)


def test_edit_and_rename_both_invalidate(tmp_path):
    """The digest IS the invalidation. If either of these stops changing the key, the audits silently read a
    stale tree -- and an audit reporting on code that no longer exists is worse than no audit."""
    p = tmp_path / "a.py"
    p.write_text("def x():\n    pass\n")
    d1 = si.tree_digest([str(p)])
    p.write_text("def x():\n    return 1\n")
    d2 = si.tree_digest([str(p)])
    assert d1 != d2, "an edit did not change the digest"
    q = tmp_path / "b.py"
    p.rename(q)
    assert si.tree_digest([str(q)]) != d2, "a rename kept the digest; reported locations would be wrong"


def test_the_audit_resolves_the_index_ONCE_not_per_file():
    """THE NAMED REGRESSION, and the whole reason this file exists.

    A full orphan audit must cost exactly ONE compile and a handful of index resolutions -- not one per
    source file. If `compiles` climbs with the file count, someone moved parsed_trees() back inside a loop
    and the cache has become a 12x pessimisation wearing a cache's clothes."""
    from holographic.io_and_interop.holographic_orphanaudit import audit
    si.index_clear()
    audit()
    first = si.index_stats()
    assert first["compiles"] == 1, "one audit caused %d compiles" % first["compiles"]
    assert first["hits"] + first["misses"] <= 4, (
        "the index was resolved %d times for a single audit -- it must be resolved once and threaded down"
        % (first["hits"] + first["misses"]))
    audit()
    second = si.index_stats()
    assert second["compiles"] == 1, "a repeat audit re-parsed the tree"


def test_the_index_actually_pays_on_the_audit():
    """A cache that does not measurably pay is a liability, and this one has already been a 12x loss once.
    Measured 1.81x end-to-end (the parse is ~45% of the audit, so Amdahl caps it there). The gate is loose --
    this asserts the sign of the effect on shared CI hardware, not the exact figure."""
    from holographic.io_and_interop.holographic_orphanaudit import audit
    si.index_clear(); audit(); audit()                       # warm every other cache first
    t0 = time.perf_counter(); audit(); t_idx = time.perf_counter() - t0
    real = si.parsed_trees
    si.parsed_trees = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index disabled"))
    try:
        t0 = time.perf_counter(); audit(); t_raw = time.perf_counter() - t0
    finally:
        si.parsed_trees = real
    assert t_idx < t_raw, "the index made the audit SLOWER (%.0f ms vs %.0f ms)" % (t_idx * 1e3, t_raw * 1e3)


def test_audit_survives_a_broken_index():
    """The index is an optimisation and must never be load-bearing: if it raises, the audit still runs."""
    from holographic.io_and_interop.holographic_orphanaudit import audit
    real = si.parsed_trees
    si.parsed_trees = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        b = audit()
    finally:
        si.parsed_trees = real
    assert len(b["faculty"]) > 1000, "the audit degraded to nonsense when the index failed"

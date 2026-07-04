"""Guards the buried-tech audit finding: the once-buried unique modules stay wired (discoverable via a curated home),
and the superseded duplicates stay clearly flagged so nobody re-wires a duplicate by accident."""
import os
import glob
import re


def test_workspace_and_durability_are_discoverable():
    from holographic_catalog import default_catalog
    names = {c.name for c in default_catalog().all()}
    assert "Workspaces (durable DB + transient sessions)" in names
    assert "Durability & crash recovery" in names


def test_find_capability_reaches_them():
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=128, seed=0)
    assert any("Workspace" in str(h) for h in m.find_capability("isolate a per-session scratch database"))
    assert any("Durability" in str(h) for h in m.find_capability("crash recovery journal for the database"))


def test_superseded_duplicates_are_flagged():
    """Each duplicate names its wired twin in a SUPERSEDED banner -- so it reads as intentionally-unwired, not lost."""
    twins = {
        "holographic_query_concurrency.py": "holographic_querylock",
        "holographic_query_graph.py": "holographic_querygraph",
        "holographic_query_history.py": "holographic_querytime",
        "holographic_query_programs.py": "holographic_queryprog",
    }
    for fname, twin in twins.items():
        head = open(fname, encoding="utf-8").read()[:600].lower()
        assert "superseded by" in head and twin in head, fname


def test_no_module_is_buried():
    """No engine module is left unwired AND unhomed AND not-declared -- the whole point of the audit."""
    mods = [os.path.basename(p)[:-3] for p in glob.glob("holographic_*.py")
            if not os.path.basename(p).startswith("test_")]
    unified = open("holographic_unified.py", encoding="utf-8").read()
    from holographic_catalog import default_catalog
    blob = " ".join((c.name + " " + c.does + " " + c.example + " " + " ".join(c.aliases)).lower()
                    for c in default_catalog().all())
    allpy = glob.glob("*.py") + glob.glob("tools/*.py")

    def referenced(mod):
        for p in allpy:
            b = os.path.basename(p)[:-3]
            if b == mod or b.startswith("test_"):
                continue
            if re.search(r"\b" + re.escape(mod) + r"\b", open(p, encoding="utf-8", errors="replace").read()):
                return True
        return False

    buried = []
    for m in mods:
        if m == "holographic_unified" or m in unified:
            continue
        doc = open(m + ".py", encoding="utf-8", errors="replace").read().lower()
        if "superseded by" in doc:
            continue
        short = m.replace("holographic_", "")
        if referenced(m) or short in blob or m in blob:
            continue
        buried.append(m)
    assert buried == [], "buried (unwired, unhomed, not declared): %s" % buried

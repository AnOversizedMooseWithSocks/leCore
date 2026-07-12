"""Guards the buried-tech audit finding: the once-buried unique modules stay wired (discoverable via a curated home),
and the superseded duplicates stay clearly flagged so nobody re-wires a duplicate by accident."""
import os
import glob
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_module_path(bare_name):
    """Resolve a bare 'holographic_x' name to its real file, wherever it landed under holographic/."""
    matches = glob.glob(os.path.join(_REPO_ROOT, "holographic", "**", bare_name + ".py"), recursive=True)
    return matches[0] if matches else None


def test_workspace_and_durability_are_discoverable():
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    names = {c.name for c in default_catalog().all()}
    assert "Workspaces (durable DB + transient sessions)" in names
    assert "Durability & crash recovery" in names


def test_find_capability_reaches_them():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=128, seed=0)
    assert any("Workspace" in str(h) for h in m.find_capability("isolate a per-session scratch database"))
    assert any("Durability" in str(h) for h in m.find_capability("crash recovery journal for the database"))


def test_superseded_duplicates_are_flagged():
    """Each duplicate names its wired twin in a SUPERSEDED banner -- so it reads as intentionally-unwired, not lost."""
    twins = {
        "holographic_query_concurrency": "holographic_querylock",
        "holographic_query_graph": "holographic_querygraph",
        "holographic_query_history": "holographic_querytime",
        "holographic_query_programs": "holographic_queryprog",
    }
    for stem, twin in twins.items():
        path = _find_module_path(stem)
        assert path, "could not find %s.py anywhere under holographic/" % stem
        head = open(path, encoding="utf-8").read()[:600].lower()
        assert "superseded by" in head and twin in head, stem


def test_no_module_is_buried():
    """No engine module is left unwired AND unhomed AND not-declared -- the whole point of the audit."""
    holo_paths = glob.glob(os.path.join(_REPO_ROOT, "holographic", "**", "holographic_*.py"), recursive=True)
    mods = [os.path.basename(p)[:-3] for p in holo_paths if not os.path.basename(p).startswith("test_")]
    unified_path = _find_module_path("holographic_unified")
    unified = open(unified_path, encoding="utf-8").read()
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    blob = " ".join((c.name + " " + c.does + " " + c.example + " " + " ".join(c.aliases)).lower()
                    for c in default_catalog().all())
    # every real source file that could plausibly reference a module by name: the repo-root scripts
    # (app.py, lecore.py, etc.), the whole holographic/ tree, and the flat tools/ scripts (tests are
    # excluded -- referencing a module in its own test doesn't count)
    allpy = glob.glob(os.path.join(_REPO_ROOT, "*.py")) + \
            glob.glob(os.path.join(_REPO_ROOT, "holographic", "**", "*.py"), recursive=True) + \
            glob.glob(os.path.join(_REPO_ROOT, "tools", "*.py"))

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
        doc_path = _find_module_path(m)
        doc = open(doc_path, encoding="utf-8", errors="replace").read().lower()
        if "superseded by" in doc:
            continue
        short = m.replace("holographic_", "")
        if referenced(m) or short in blob or m in blob:
            continue
        buried.append(m)
    assert buried == [], "buried (unwired, unhomed, not declared): %s" % buried


def test_no_dark_method_capabilities():
    """D1 regression trap: a public mind method auto-registered as a capability must be DISCOVERABLE -- surface in the
    top-15 for its own name, OR carry aliases (a bare generic name like 'scene'/'material'/'query' is otherwise
    shadowed by a descriptively-titled sibling and becomes unreachable via find_capability). This pins the fix and
    fails CI if a new bare-name faculty is registered without aliases."""
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=128, seed=0)
    c = m._capability_catalog()
    dark = []
    for cap in c.all():
        top15 = [x.name for x in m.find_capability(cap.name)[:15]]
        if cap.name not in top15 and not (getattr(cap, "aliases", ()) or ()):
            dark.append(cap.name)
    assert not dark, "dark (undiscoverable, alias-less) capabilities regressed: %r" % dark[:20]


def test_previously_dark_bare_names_are_now_discoverable():
    """Spot-check several of the 39 once-dark bare method-names: a user phrasing now surfaces each in the top-3."""
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=256, seed=0)
    checks = {
        "reinforcement learning agent": "agent",
        "physically based material": "material",
        "run a sql query": "query",
        "add a light to a scene": "light",
        "variance harness": "measure",
        "re-extract a mesh from a field": "surface_mesh",
    }
    for phrasing, name in checks.items():
        top3 = [x.name for x in m.find_capability(phrasing)[:3]]
        assert name in top3, (phrasing, "->", top3)

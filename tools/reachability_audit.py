"""reachability_audit.py -- is every built module REACHABLE and DISCOVERABLE? "There's no point in having
unreachable or buried functionality -- we built it, let's make sure it can be used."

For each holographic_*.py engine module it checks three things (stdlib/AST only, never imports the module):
  * DOCSTRING present?  -- a module with no docstring can't be surfaced by the catalog's find_capability. A real gap.
  * PUBLIC API present?  -- top-level defs/classes not starting with '_'. No public API + not referenced -> dead.
  * REFERENCED by UnifiedMind?  -- its name appears in holographic_unified.py (reachable as / through a faculty).
  * KEPT NEGATIVE?  -- the docstring/source flags it as a deliberately-unwired negative (fine, by design).

It prints a summary and the two lists that actually need attention: NO-DOCSTRING (undiscoverable) and
IMPORT-ONLY-NO-NEGATIVE (built, not a faculty, not a declared negative -- decide: catalog note, faculty, or leave).

Usage:  python tools/reachability_audit.py
"""
import ast
import os
import glob
import sys


def _engine_modules(root):
    out = []
    # engine modules live under the holographic/ package (holographic/<family>/holographic_*.py); recurse it.
    # Fall back to a flat root glob too, so this still works on an un-reorganized checkout.
    patterns = [os.path.join(root, "holographic", "**", "holographic_*.py"),
                os.path.join(root, "holographic_*.py")]
    seen = set()
    for pat in patterns:
        for path in sorted(glob.glob(pat, recursive=True)):
            base = os.path.basename(path)
            if base.startswith("test_"):
                continue
            if base in seen:                                    # don't double-count if both patterns hit
                continue
            seen.add(base)
            out.append(path)
    return sorted(out)


def _public_api(tree):
    """Top-level function/class names not starting with '_' (the module's public surface)."""
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            names.append(node.name)
    return names


# the modules that are DELIBERATELY not wired -- recorded negatives named in the dev guide / their own docstrings.
_KNOWN_NEGATIVES = {
    "holographic_misgen", "holographic_ldexplore", "holographic_lookahead", "holographic_jittersplat",
    "holographic_splatsharpen", "holographic_graph_memory", "holographic_probesweep",
}


def _is_facade(name, src):
    """A CONSOLIDATION HOME (`holographic_*home.py`) is a library facade -- "one door, route don't rewrite" -- and
    is import-only BY DESIGN. It is not a failed idea and it is not a gap.

    Before this distinction existed, all 13 homes sat in the IMPORT-ONLY "review" bucket forever, indistinguishable
    from real gaps. **A number that never moves is a blind spot, not a baseline.** A facade must still SAY it is one:
    the naming convention alone is not the declaration."""
    if not name.endswith("home"):
        return False
    head = src[:1200].lower()
    return ("home (consolidation" in head or "one facade" in head or "the one door" in head
            or "single door" in head or "route, don't rewrite" in head or "one place for" in head
            or "scaffold" in head)


def audit(root):
    # holographic_unified.py moved into the package (holographic/misc/); find it wherever it lives.
    _unified = glob.glob(os.path.join(root, "holographic", "**", "holographic_unified.py"), recursive=True) \
               or glob.glob(os.path.join(root, "holographic_unified.py"))
    mind_src = open(_unified[0], encoding="utf-8", errors="replace").read() if _unified else ""

    modules = _engine_modules(root)
    no_doc, no_public, import_only, kept_neg, documents_neg, superseded = [], [], [], [], [], []
    facades = []
    referenced = 0
    for path in modules:
        name = os.path.basename(path)[:-3]
        if name == "holographic_unified":
            continue
        src = open(path, encoding="utf-8", errors="replace").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        doc = (ast.get_docstring(tree) or "").strip()
        api = _public_api(tree)
        dl = doc.lower()
        is_neg = name in _KNOWN_NEGATIVES                        # deliberately-unwired (explicit list)
        is_facade = _is_facade(name, src)                        # a consolidation home: import-only BY DESIGN
        is_superseded = "superseded by" in dl                    # an older twin, declared and pointed at the wired one
        if ("kept negative" in dl) or ("recorded negative" in dl):
            documents_neg.append(name)                          # merely DOCUMENTS a negative -- honest, good
        in_mind = name in mind_src

        if not doc:
            no_doc.append(name)
        if not api:
            no_public.append(name)
        if is_neg:
            kept_neg.append(name)
        if is_superseded:
            superseded.append(name)
        if is_facade:
            facades.append(name)
        if in_mind:
            referenced += 1
        elif not is_neg and not is_superseded and not is_facade:  # facades are import-only BY DESIGN
            import_only.append(name)

    n = len([p for p in modules if os.path.basename(p)[:-3] != "holographic_unified"])
    print("REACHABILITY AUDIT over %d engine modules\n" % n)
    print("  referenced by UnifiedMind (reachable as/through a faculty): %d" % referenced)
    print("  deliberately NOT wired (recorded negatives):               %d  %s" % (len(kept_neg), sorted(kept_neg)))
    print("  modules that DOCUMENT a kept negative (honest measurement): %d" % len(documents_neg))
    print("  SUPERSEDED by a wired twin (declared, use the twin):        %d  %s" % (len(superseded), sorted(superseded)))
    print("  CONSOLIDATION HOMES (one door, import-only BY DESIGN):      %d  %s" % (len(facades), sorted(facades)))
    print()
    print("  NO DOCSTRING -> UNDISCOVERABLE by find_capability (FIX these): %d" % len(no_doc))
    for m in sorted(no_doc):
        print("      %s" % m)
    print()
    print("  NO PUBLIC API (dead or all-underscore): %d  %s" % (len(no_public), sorted(no_public)))
    print()
    print("  IMPORT-ONLY, not a declared negative (findable via the catalog, but NOT a mind faculty -- review): %d"
          % len(import_only))
    for m in sorted(import_only):
        print("      %s" % m)
    return {"no_doc": no_doc, "import_only": import_only, "kept_neg": kept_neg, "no_public": no_public}


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    audit(root)

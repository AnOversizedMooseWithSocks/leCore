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

# INFRASTRUCTURE / PLUMBING -- import-only BY DESIGN, reached THROUGH a higher faculty or the transport layer, not a
# direct agent faculty. Same spirit as the consolidation-home facades: a module that exists to be used by other code,
# not called by an agent, must SAY so or the audit can't tell it from a real gap. Verified: each is imported by other
# engine modules (delegated) or is the transport/query spine itself.
#   service/toolclient/uri  -- the HTTP tool server + remote-tool client + URI scheme (the transport the agent uses,
#                              not a capability it invokes).
#   sync/farm/provenance    -- cross-node sync, the compute farm, and provenance tracking (cross-cutting plumbing).
#   determinism             -- the determinism harness (imported by ~15 modules to pin seeds); infrastructure, not a op.
#   query_durable/queryfolder/querygraph/queryprog/querytime -- EXTENSIONS of the wired query-database faculty; the
#                              agent reaches them through mind's query/database doors, not as standalone methods.
#   objectref               -- the /invoke object-handle registry. It belongs to the SERVICE, not the mind: it is
#                              minted and resolved by the HTTP boundary on the agent's behalf, and the agent uses
#                              it by passing a 'ref:Type:N' string back as an argument, never by calling it. Wiring
#                              a mind faculty for it would be forced -- a mind running in-process has no registry
#                              and no need of one. Its capability IS catalogued (agents must discover the ref
#                              convention); what is import-only is the implementation, which is the definition of
#                              plumbing.
#   catalog_p01..p06        -- the capability REGISTRY, split off holographic_catalog when it reached 81% of the
#                              1 MB agent-read cap. They are the catalog itself, not separate capabilities:
#                              default_catalog() calls register(c) on each IN ORDER (the sequence is part of the
#                              contract -- find_capability ranks by score and ties break by registration order).
#                              A mind faculty per part would be nonsense; every capability they register is
#                              already discoverable through the one door they exist to fill.
_KNOWN_INFRASTRUCTURE = {
    "holographic_service", "holographic_toolclient", "holographic_uri", "holographic_sync", "holographic_farm",
    "holographic_provenance", "holographic_determinism", "holographic_query_durable", "holographic_queryfolder",
    "holographic_querygraph", "holographic_queryprog", "holographic_querytime",
    # A SHARED PARSE CACHE, not a faculty: codehealth / codemap / orphanaudit all reach it, and each of THOSE
    # is wired (audit_complexity, code_search/code_similar, audit_orphans). Giving it its own mind method
    # would publish a cache-warming detail as a user-facing verb; declaring it keeps the import-only list
    # meaningful instead of carrying a permanent known-good entry nobody reviews.
    "holographic_srcindex",
    # THE CATALOG'S OWN PARTS + the object-handle registry: same reasoning one level up. The parts are
    # reached through holographic_catalog (their only door, exactly as the unified parts are reached
    # through holographic_unified), and objectref is plumbing the service holds on the caller's behalf.
    "holographic_objectref",
    "holographic_catalog_p01", "holographic_catalog_p02", "holographic_catalog_p03",
    "holographic_catalog_p04", "holographic_catalog_p05", "holographic_catalog_p06",
}


def _is_evidence(name, src):
    """An EVIDENCE / HARNESS module runs to DEMONSTRATE a property (an ablation table, a robustness curve) -- it is
    import-only BY DESIGN and says so in its own docstring. Like a facade, the module must DECLARE it: recognising the
    self-declaration keeps these out of the "real gap" bucket without a central list going stale. Markers are the exact
    phrases these modules already use ("not a callable faculty", "stays a harness", "ablation table")."""
    head = src[:1400].lower()
    return ("not a callable faculty" in head or "not a callable library capability" in head
            or "stays a harness" in head or "ablation table" in head
            or "run to demonstrate a property, not" in head)


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
    # THE FACULTY SURFACE IS NOW SEVERAL FILES, so read all of them. UnifiedMind used to be one 17.4k-line
    # file; it is now a shim plus holographic/unified/holographic_unified_p*.py mixin parts. Reading only the
    # shim would report almost every module as unreferenced -- the references did not disappear, they moved.
    _unified = glob.glob(os.path.join(root, "holographic", "**", "holographic_unified.py"), recursive=True) \
               or glob.glob(os.path.join(root, "holographic_unified.py"))
    _unified += sorted(glob.glob(os.path.join(root, "holographic", "unified", "holographic_unified_p*.py")))
    mind_src = "\n".join(open(p, encoding="utf-8", errors="replace").read() for p in _unified)

    modules = _engine_modules(root)
    no_doc, no_public, import_only, kept_neg, documents_neg, superseded = [], [], [], [], [], []
    facades = []
    infrastructure, evidence = [], []
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
        is_infra = name in _KNOWN_INFRASTRUCTURE                 # plumbing / query-extension: reached via a faculty
        is_evidence = _is_evidence(name, src)                    # a harness/ablation: runs to demonstrate, not called
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
        if is_infra:
            infrastructure.append(name)
        if is_evidence:
            evidence.append(name)
        if in_mind:
            referenced += 1
        elif not is_neg and not is_superseded and not is_facade and not is_infra and not is_evidence:
            import_only.append(name)                             # every exclusion above is import-only BY DESIGN

    n = len([p for p in modules if os.path.basename(p)[:-3] != "holographic_unified"])
    print("REACHABILITY AUDIT over %d engine modules\n" % n)
    print("  referenced by UnifiedMind (reachable as/through a faculty): %d" % referenced)
    print("  deliberately NOT wired (recorded negatives):               %d  %s" % (len(kept_neg), sorted(kept_neg)))
    print("  modules that DOCUMENT a kept negative (honest measurement): %d" % len(documents_neg))
    print("  SUPERSEDED by a wired twin (declared, use the twin):        %d  %s" % (len(superseded), sorted(superseded)))
    print("  CONSOLIDATION HOMES (one door, import-only BY DESIGN):      %d  %s" % (len(facades), sorted(facades)))
    print("  INFRASTRUCTURE / PLUMBING (reached via a faculty, by design): %d  %s" % (len(infrastructure), sorted(infrastructure)))
    print("  EVIDENCE / HARNESS (runs to demonstrate, not a faculty):    %d  %s" % (len(evidence), sorted(evidence)))
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
    # -- C7 canary: file-size census against the agent-facing read cap ------------------------------------
    # codeedit's capped read (1 MB) exists to protect context windows; internal callers read uncapped. This
    # census is a NON-GATING warning at 80% of the cap so the next crossing is seen approaching instead of
    # discovered when a tool refuses -- which is exactly how holographic_unified.py's crossing was discovered.
    cap = 1_000_000
    watch = []
    for dirpath, _dirs, files in os.walk(os.path.join(root, "holographic")):
        for fn in files:
            if fn.endswith(".py"):
                sz = os.path.getsize(os.path.join(dirpath, fn))
                if sz >= int(cap * 0.8):
                    watch.append((sz, fn))
    print()
    print("  SIZE CANARY (>= 80%% of the 1 MB agent-read cap; non-gating heads-up): %d" % len(watch))
    for sz, fn in sorted(watch, reverse=True):
        # Over the cap, ONLY the whole-file reads refuse -- that is the cap doing its one job (protecting a
        # context window). Every bounded-output tool works: view/read_lines (a slice), count_occurrences (an
        # int), and every WRITE (replace/insert/replace_lines/delete_lines) read uncapped, so a big module
        # stays fully agent-navigable and agent-EDITABLE. That was not true until the cap-scope fix: view and
        # all four writes were capped too, so unified.py could not be edited through the mind at all -- pinned
        # now by tests/test_codeedit_cap_scope.py. Size alone is therefore a heads-up, not a capability loss.
        over = " -- OVER the cap: whole-file read() refuses BY DESIGN; view/read_lines/count/edits all work" \
               if sz > cap else ""
        print("      %s  %d bytes (%.0f%%)%s" % (fn, sz, 100.0 * sz / cap, over))
    # DUPLICATE FACULTY DEFINITIONS (F24 -- the blind spot that let THREE silently-shadowed
    # faculties through at 0/0/0): same-name `def` twice in one class body means Python keeps
    # the second silently and the first becomes dead code with a live-looking docstring. AST
    # walk per facade part; ten lines, permanent. HARD ERROR: a shadowed faculty is a silent
    # behavior flip waiting to be re-edited.
    dup = []
    for fn in glob.glob(os.path.join(root, "holographic", "unified", "holographic_unified_p*.py")):
        tree_ = ast.parse(open(fn, encoding="utf-8", errors="replace").read())
        for node in ast.walk(tree_):
            if isinstance(node, ast.ClassDef):
                seen = {}
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name in seen:
                            dup.append((os.path.basename(fn), item.name, seen[item.name], item.lineno))
                        seen[item.name] = item.lineno
    # ...and the SAME disease in the catalog organ: a duplicate string KEY in an alias dict is
    # kept-last silently by Python, darkening the first block's aliases (found mirroring all
    # three shadowed faculties). Regex-level scan of the catalog files' top-level alias keys.
    import re as _re
    for fn in glob.glob(os.path.join(root, "holographic", "caching_and_storage", "holographic_catalog*.py")):
        src_ = open(fn, encoding="utf-8", errors="replace").read()
        keys = _re.findall(r'^    "([a-z0-9_]+)": \(', src_, _re.M)
        seen_ = {}
        for i, k in enumerate(keys):
            if k in seen_:
                dup.append((os.path.basename(fn), "ALIAS-KEY " + k, seen_[k], i))
            seen_[k] = i
    print()
    print("  DUPLICATE (silently shadowed) faculty definitions -- HARD ERROR: %d" % len(dup))
    for fn, name, first, second in dup:
        print("      %s: %s at lines %d and %d (first is DEAD CODE)" % (fn, name, first, second))
    return {"no_doc": no_doc, "import_only": import_only, "kept_neg": kept_neg, "no_public": no_public,
            "size_watch": watch, "duplicates": dup}


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    audit(root)

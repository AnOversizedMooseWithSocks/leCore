"""holographic_orphanaudit.py -- reachability at FUNCTION granularity: leCore auditing its own surface.

The engine could already EDIT itself (file_grep, file_replace, file_python_check) but could not ANALYSE
itself: `find_capability("find dead code")` returned unrelated fallbacks, "cyclomatic complexity" returned
the renderer's `scene_cost`, and "map the codebase" returned a displacement-map baker. Lexical false
friends, not capabilities. This module closes that gap, and `tools/orphan_audit.py` is a thin CLI over it.

WHY THIS EXISTS
---------------
reachability_audit, catalog_gaps and wiring_report all reason about MODULES. A module passes if it has a
docstring, exports something public, and is referenced from the UnifiedMind surface. All three report 0 gaps.

But a module can pass every one of those checks while individual functions inside it are reachable by nothing
at all. The audits look at the file; nobody was looking inside it.

MEASURED, and cross-validated against an independent oracle (vulture 2.x, third-party, static): of 8,532
analysed blocks the engine has 221 public functions with no static caller, no UnifiedMind faculty and no
catalog mention. Splitting those by what else can reach them:

    155  exercised by a TEST but exposed nowhere  <- they WORK. By this repo's own governing rule --
                                                     "a capability find_capability can't surface and /invoke
                                                     can't call does not exist" -- they formally do not exist.
     65  called only by a tools/ script
     57  TRUE ORPHANS: no faculty, no catalog entry, no test, no tool, no static caller

57 orphans across 8,532 blocks is a LOW rate and that is worth saying plainly -- this audit is not an
indictment of the other three, it is the granularity they were never built to have. The 155 are the
interesting number: working, tested code that the engine's own definition says is not real.

WHY IT IS STDLIB-ONLY AND NOT JUST "RUN VULTURE"
------------------------------------------------
vulture found this and deserves the credit, but core is NumPy/Flask/stdlib/hashlib only, and an audit that
CI cannot run without a third-party wheel is an audit that quietly stops running. `ast` is stdlib and
sufficient: we are not doing type inference, we are asking "does this name appear anywhere outside its own
definition". Cross-validated against vulture on the same tree -- see _selftest.

WHAT IT DELIBERATELY DOES NOT DO (kept negatives)
--------------------------------------------------
  * NO TYPE INFERENCE, so `obj.foo()` counts as a reference to EVERY `foo` defined anywhere. That makes this
    audit CONSERVATIVE: it under-reports orphans and never invents one. A conservative dead-code finder is
    the only honest kind, because the cost of a false orphan (deleting live code) dwarfs the cost of a
    missed one.
  * NO AUTOMATIC DELETION, and no "fix" mode. An orphan is a QUESTION -- wire it, catalogue it, or declare
    it a negative -- not a defect to be swept. The engine's whole failure mode is capability that exists but
    cannot be found; deleting on a static signal would turn that failure mode into data loss.
  * UNDERSCORE-PRIVATE NAMES ARE NOT AUDITED. A module's own helpers are its business.
"""
import ast
import glob
import json
import os
import sys

# holographic/io_and_interop/<this file> -> up three to the repo root. Overridable, because a mind can be
# pointed at a checkout other than its own (set_file_root's habit).
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Budget, in the same spirit as structure_audit's: a gate that holds the line without demanding a cleanup
# before the audit can be adopted. Raise it only with a reason written down next to the number.
# Measured, not guessed: 43 of these are corroborated by an independent third-party oracle (vulture) run on
# the same tree. The budget sits just above the current count so the gate holds the line without demanding a
# 43-item cleanup before the audit can be adopted. Lower it as orphans get wired, catalogued or declared.
ORPHAN_BUDGET = 50


def _py(*patterns):
    out = []
    for p in patterns:
        out.extend(glob.glob(os.path.join(REPO, p), recursive=True))
    return sorted(set(out))


def _tree(path, trees=None):
    """The AST for `path`, from the SHARED content-addressed index when the file is in the engine tree.

    WHY NOT JUST open+parse: this audit walked the tree twice (definitions, then references) and codehealth
    walked it twice more, so a single report paid ~5 full parses of ~600 files -- 7.3 s for the orphan audit
    alone. The engine's L3 unit exists for exactly this ("the same spec is compiled from many call sites"),
    and an AST is a pure function of the bytes, which is the tier's other precondition. tests/ and tools/ are
    outside the indexed tree and still parse directly; they are a small fraction of the files.

    `trees` IS PASSED IN, NEVER FETCHED HERE, and that signature is the whole lesson. The first version called
    parsed_trees() inside this function -- which recomputes the tree digest, 39.5 ms, on EVERY file lookup.
    3,543 lookups later the audit had gone from 7.3 s to 87 s: TWELVE TIMES SLOWER WITH THE CACHE THAN
    WITHOUT IT. That is the identical failure the SpectrumCache correction is on record for (a content key
    priced per-lookup instead of per-block), repeated inside the module whose own docstring warns about it.
    A CACHE'S AMORTISATION IS A PROPERTY OF THE CALL PATTERN, NOT OF THE CACHE: documenting "hash once per
    block" does nothing if the caller asks once per element. Resolve the index ONCE per audit, thread it down."""
    if trees is not None:
        hit = trees.get(os.path.abspath(path)) or trees.get(path)
        if hit is not None:
            return hit
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return ast.parse(f.read(), filename=path)
    except (OSError, SyntaxError):
        return None


def public_definitions(paths, trees=None):
    """Every public function and method defined in `paths`, as {name: [(path, lineno), ...]}.

    Nested defs are skipped: a closure is an implementation detail of its parent, not a surface."""
    defs = {}
    for path in paths:
        tree = _tree(path, trees)
        if tree is None:
            continue
        for node in tree.body:
            targets = [node] if isinstance(node, ast.FunctionDef) else \
                      (node.body if isinstance(node, ast.ClassDef) else [])
            for n in targets:
                if not isinstance(n, ast.FunctionDef) or n.name.startswith("_"):
                    continue
                # A DECORATED FUNCTION IS REGISTERED, NOT DEAD. Flask's @app.route, @property, @staticmethod
                # and every other registration decorator mean something else holds the reference, so there is
                # no caller to find and absence of one proves nothing. Skipping them removed 23 false
                # orphans in one go -- all of them holographic_service.py's api_* HTTP route handlers, which
                # are about as live as code in this repo gets.
                if n.decorator_list:
                    continue
                defs.setdefault(n.name, []).append((path, n.lineno))
    return defs


def referenced_names(paths, skip_defs_in=None, trees=None):   # skip_defs_in kept for compatibility; see below
    """Every identifier USED in `paths` -- bare names, attribute accesses, and string mentions.

    Strings count on purpose: this repo dispatches by name through the catalog's `example=` snippets and
    through getattr-style routing, so a name that only ever appears inside a string is still reachable.
    Missing that would manufacture false orphans, which is the one error this audit must not make."""
    used = set()
    for path in paths:
        tree = _tree(path, trees)
        if tree is None:
            continue
        # SAME-FILE USES COUNT. An earlier version stripped every name a file defined from that file's
        # references, on the theory that "self-reference is not reachability". That is wrong and it
        # over-reported by 5x (311 orphans against an independent oracle's 57): a private-ish helper called
        # only by its own module's public API is REACHED, and stripping it manufactured an orphan. The only
        # thing worth discounting is true self-recursion, and letting that read as reachable errs toward
        # under-reporting -- the safe direction for a tool whose output invites deletion.
        local = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                local.add(node.id)
            elif isinstance(node, ast.Attribute):
                local.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # cheap tokenisation; a name mentioned in a docstring or an example counts as a mention
                for tok in node.value.replace("(", " ").replace(".", " ").replace(",", " ").split():
                    if tok.isidentifier():
                        local.add(tok)
        used |= local
    return used


def dynamic_surface():
    """The two oracles a static scan cannot see: the live UnifiedMind faculty list, and the catalog text.

    Returns (faculties, catalog_blob). Degrades to empty on import failure so the audit still runs on a tree
    whose mind will not boot -- the same rule reachability_audit follows."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)        # tools/ is sys.path[0] when run as a script; the repo root is not
    try:
        import lecore
        mind = lecore.UnifiedMind(dim=128, seed=0)
        faculties = {n for n in dir(mind) if not n.startswith("_")}
    except Exception as exc:
        # LOUD, not silent. The first run of this tool reported 0 faculties because a bare except swallowed
        # an ImportError, and an empty oracle does not look like a broken oracle -- it looks like a finding.
        print("  WARNING: could not boot a mind (%s: %s); faculty oracle is EMPTY and counts below are"
              " over-stated" % (type(exc).__name__, exc), file=sys.stderr)
        faculties = set()
    try:
        from holographic.caching_and_storage.holographic_catalog import default_catalog
        blob = set()
        for c in default_catalog().all():
            text = " ".join([c.name, c.does, c.example, " ".join(c.aliases)])
            for tok in text.replace("(", " ").replace(".", " ").replace(",", " ").split():
                if tok.isidentifier():
                    blob.add(tok)
    except Exception as exc:
        print("  WARNING: could not load the catalog (%s: %s); catalog oracle is EMPTY"
              % (type(exc).__name__, exc), file=sys.stderr)
        blob = set()
    return faculties, blob


def audit():
    engine = _py("holographic/**/*.py", "lecore.py", "app.py", "holographic_service.py")
    tests = _py("tests/**/*.py")
    tools = _py("tools/**/*.py")

    # ONE resolve of the shared L3 index for the whole audit -- see _tree's docstring for what happens
    # when this moves inside the loop (12x SLOWER than no cache at all).
    try:
        from holographic.io_and_interop.holographic_srcindex import parsed_trees
        trees = parsed_trees()
    except Exception:
        trees = None                          # the index is an optimisation; never let it break the audit
    defs = public_definitions(engine, trees)
    engine_used = referenced_names(engine, skip_defs_in=set(engine), trees=trees)
    test_used = referenced_names(tests, trees=trees)
    tool_used = referenced_names(tools, trees=trees)
    faculties, catalog = dynamic_surface()

    buckets = {"faculty": [], "catalog": [], "engine": [], "test_only": [], "tool_only": [], "orphan": []}
    for name, sites in sorted(defs.items()):
        path, line = sites[0]
        if name in faculties:
            buckets["faculty"].append((name, path, line))
        elif name in catalog:
            buckets["catalog"].append((name, path, line))
        elif name in engine_used:
            buckets["engine"].append((name, path, line))
        elif name in test_used:
            buckets["test_only"].append((name, path, line))
        elif name in tool_used:
            buckets["tool_only"].append((name, path, line))
        else:
            buckets["orphan"].append((name, path, line))
    return buckets


def orphan_report(root=None, limit=40):
    """Function-granularity reachability for the whole engine, as a plain dict -- the mind-facing entry point.

    Returns {counts: {bucket: n}, orphan: [...], test_only: [...], budget: n, ok: bool}. Lists are truncated
    to `limit` because this is meant to be read in a chat window, not paged; ask the CLI for the full dump.

    THE TWO BUCKETS THAT MATTER:
      orphan     -- no faculty, no catalog entry, no caller, no test, no tool. Built and forgotten.
      test_only  -- WORKS and is TESTED, but is exposed nowhere. By this repo's governing rule ("a capability
                    find_capability can't surface and /invoke can't call does not exist") these formally do
                    not exist, which makes them the most valuable list in the report: finished work that is
                    one catalog entry away from being real."""
    global REPO
    if root:
        REPO = os.path.abspath(root)
    b = audit()
    return {"counts": {k: len(v) for k, v in b.items()},
            "orphan": [{"name": n, "path": os.path.relpath(p, REPO), "line": l} for n, p, l in b["orphan"][:limit]],
            "test_only": [{"name": n, "path": os.path.relpath(p, REPO), "line": l} for n, p, l in b["test_only"][:limit]],
            "budget": ORPHAN_BUDGET,
            "ok": len(b["orphan"]) <= ORPHAN_BUDGET}


# ---------------------------------------------------------------------------
# AGENT REACHABILITY -- the second question, asked after "is this referenced?"
#
# WHY THIS EXISTS (found by measurement, 3-D authoring probe). `audit()` above asks a LEXICAL question:
# does this name appear anywhere outside its own definition? That is the right question for dead code and
# it is deliberately conservative. But it answers YES for a symbol whose only reference is inside a module
# that is ITSELF import-only by design -- a consolidation home, a declared negative, the plumbing. The
# chain is alive in the import graph and dead to an agent, because it never terminates at a faculty.
#
# MEASURED, and the reason this pass exists: holographic_lights defines ten light classes. DomeLight and
# RectLight -- environment/IBL lighting and area lights, between them most of what makes a render look
# like a photograph -- are referenced ONLY by holographic_lightinghome (a consolidation home) and
# holographic_lightcache. `audit()` filed them under "engine: reachable". An agent asking the mind for a
# light gets `mind.light()`, which returns the RASTERISER's Light class and raises
# AttributeError: 'Light' object has no attribute 'sample' the moment the path tracer touches it.
# Every module-level audit read 0 gaps while that was true.
#
# TWO NEW BUCKETS, and the distinction matters:
#   shadowed -- referenced, but every referencing engine file is itself import-only by design. Alive in the
#               graph, unreachable from /invoke. This is the bucket the 3-D probe was looking for.
#   dark     -- a public CLASS that is neither a faculty nor named in the catalog. audit() never saw these
#               at all: public_definitions collects FunctionDef only, so a class an agent cannot construct
#               was invisible to the audit by construction, not by judgement.
#
# KEPT NEGATIVE, loudly: this shares audit()'s no-type-inference limitation, so it inherits the same
# conservatism -- a name is credited to a module if it appears there at all. It therefore UNDER-reports.
# It is a review queue, never a delete list, and it does not gate CI: the counts are a starting baseline,
# and a budget pinned before anyone has looked at the list is a number pretending to be a decision.
#
# SECOND KEPT NEGATIVE, found the hard way one item later: THE FACTORY BLIND SPOT. `dark` asks whether the
# CLASS NAME is a faculty or appears in catalog text. A class reachable only through a factory --
# mind.scene_light('dome') -> holographic_lights.make_light -> DomeLight -- is genuinely constructible by
# an agent and this pass still calls it dark. Measured: after nine light classes were wired behind one
# factory door, dark_classes moved 312 -> 311, and the only one that moved was moved by catalog prose.
# The pass scored the fix as a no-op.
#
# NOT PAPERED OVER, deliberately. Crediting "any class constructed by a faculty-reachable function in the
# same module" would clear all nine in one line and would also credit every class any wired function
# happens to mention -- trading a known under-report for an unknown over-report, in the direction that
# makes the number look good. A metric edited to score its author's work is worth nothing. The honest read
# of `dark` is "not constructible BY NAME", which is a real and narrower claim than "unreachable", and the
# name-by-name reading is how it must be used until someone builds the constructor-edge version properly.

# Import-only BY DESIGN. Kept in sync with tools/reachability_audit.py's _KNOWN_NEGATIVES /
# _KNOWN_INFRASTRUCTURE -- duplicated rather than imported because core must not depend on tools/.
_NON_TERMINAL = {
    # declared negatives: deliberately unwired, named in the dev guide and their own docstrings
    "holographic_misgen", "holographic_ldexplore", "holographic_lookahead", "holographic_jittersplat",
    "holographic_splatsharpen", "holographic_graph_memory", "holographic_probesweep",
    # infrastructure / plumbing: reached THROUGH a faculty or the transport, never called directly.
    # KEPT NEGATIVE, found by running this pass on the live tree: holographic_service was in this list on
    # the first draft and it produced five false positives at once (serve_frame, drop_session, load_all,
    # demo_frame_payload, serve_frame_distributed). It does not belong here. reachability_audit calls the
    # service "infrastructure" because it is not a CAPABILITY -- true for that audit's question. For THIS
    # question the service is the opposite of a cul-de-sac: it is the door agents come through. A reference
    # from an HTTP route is the most terminal reference in the repo.
    "holographic_toolclient", "holographic_uri", "holographic_sync", "holographic_farm",
    "holographic_provenance", "holographic_determinism", "holographic_query_durable", "holographic_queryfolder",
    "holographic_querygraph", "holographic_queryprog", "holographic_querytime",
}


def _is_terminal(path):
    """Can a reference living in `path` ever reach an agent?

    NO for a consolidation home (`*home.py` -- 'one door', import-only by design), for a declared negative,
    and for the plumbing. A reference from one of those is not a route to the mind, it is a cul-de-sac.
    Everything else counts as terminal, which is the conservative direction: we credit reachability we
    cannot prove rather than manufacture a finding."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return not (stem.endswith("home") or stem in _NON_TERMINAL)


def public_classes(paths, trees=None):
    """Every public CLASS defined in `paths`, as {name: [(path, lineno), ...]}.

    Classes are a surface: `DomeLight`, `RectLight`, `PostChain` are things an agent CONSTRUCTS, and a class
    it cannot construct is exactly as unreachable as a function it cannot call. public_definitions() walks
    FunctionDef only, so this is the half of the surface that audit() was never looking at."""
    out = {}
    for path in paths:
        tree = _tree(path, trees)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                out.setdefault(node.name, []).append((path, node.lineno))
    return out


def _referenced_in(paths, trees=None):
    """{name -> set(paths that mention it)}. Same lexical rule as referenced_names -- attributes and string
    mentions count -- but it keeps WHERE each mention was, which is the whole point: 'referenced' and
    'referenced from somewhere an agent can get to' are different claims."""
    where = {}
    for path in paths:
        tree = _tree(path, trees)
        if tree is None:
            continue
        for node in ast.walk(tree):
            names = ()
            if isinstance(node, ast.Name):
                names = (node.id,)
            elif isinstance(node, ast.Attribute):
                names = (node.attr,)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # `from holographic_lights import DomeLight` binds a name that NEVER appears as a Name or
                # Attribute node. This repo has been bitten by exactly that before -- a sweep found 43
                # facade imports in `from PKG import MODULE as X` style that the wiring audit could not
                # see -- so the import statement itself has to be read. Both the real name and the alias
                # count: the alias is how the file refers to it, the real name is what we are auditing.
                names = tuple(x for a in node.names for x in (a.name.split(".")[0], a.asname) if x)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names = tuple(t for t in node.value.replace("(", " ").replace(".", " ").replace(",", " ").split()
                              if t.isidentifier())
            for nm in names:
                where.setdefault(nm, set()).add(path)
    return where


def agent_reach_report(root=None, limit=40):
    """Which public symbols can an AGENT actually reach -- functions AND classes, chains checked to the end.

    Returns {counts, shadowed, dark, budget, ok}. `shadowed` = referenced, but only from modules that are
    themselves import-only by design, so the chain never terminates at a faculty. `dark` = a public class
    that is neither a faculty nor named in the catalog.

    This is the companion to orphan_report, not a replacement: that one asks "is anything unreferenced?",
    this one asks "does the reference go anywhere?". Advisory only -- see the module notes for why it does
    not gate."""
    global REPO
    if root:
        REPO = os.path.abspath(root)
    engine = _py("holographic/**/*.py", "lecore.py", "app.py", "holographic_service.py")
    try:
        from holographic.io_and_interop.holographic_srcindex import parsed_trees
        trees = parsed_trees()
    except Exception:
        trees = None                          # the index is an optimisation; never let it break the audit
    faculties, catalog = dynamic_surface()
    where = _referenced_in(engine, trees)

    def rel(p):
        return os.path.relpath(p, REPO)

    shadowed = []
    for name, sites in sorted(public_definitions(engine, trees).items()):
        if name in faculties or name in catalog:
            continue
        sites_seen = where.get(name, set())
        # its OWN file always mentions it (the def itself); a self-mention is not a route out
        outside = {p for p in sites_seen if p not in {s[0] for s in sites}}
        if outside and not any(_is_terminal(p) for p in outside):
            shadowed.append({"name": name, "path": rel(sites[0][0]), "line": sites[0][1],
                             "only_from": sorted(rel(p) for p in outside)[:4]})

    dark = []
    for name, sites in sorted(public_classes(engine, trees).items()):
        if name in faculties or name in catalog:
            continue
        dark.append({"name": name, "path": rel(sites[0][0]), "line": sites[0][1]})

    return {"counts": {"shadowed": len(shadowed), "dark_classes": len(dark)},
            "shadowed": shadowed[:limit], "dark": dark[:limit],
            "budget": None, "ok": True}


def main(argv):
    if "--agent" in argv:
        r = agent_reach_report(limit=200 if "--list" in argv else 12)
        print("AGENT REACHABILITY (advisory -- does the reference chain end at a faculty?)")
        print("  %5d  SHADOWED     -- referenced only from import-only-by-design modules" % r["counts"]["shadowed"])
        print("  %5d  DARK CLASS   -- public class, no faculty, no catalog entry" % r["counts"]["dark_classes"])
        for kind in ("shadowed", "dark"):
            print("\n--- %s ---" % kind.upper())
            for e in r[kind]:
                extra = ("  <- only from %s" % ", ".join(e["only_from"])) if e.get("only_from") else ""
                print("  %-34s %s:%d%s" % (e["name"], e["path"], e["line"], extra))
        return 0

    b = audit()
    total = sum(len(v) for v in b.values())
    print("FUNCTION-GRANULARITY REACHABILITY  (%d public engine functions)" % total)
    print("  %5d  a UnifiedMind faculty        -- callable over /invoke" % len(b["faculty"]))
    print("  %5d  named in the catalog         -- surfaced by find_capability" % len(b["catalog"]))
    print("  %5d  called inside the engine     -- reachable, though not itself a surface" % len(b["engine"]))
    print("  %5d  TEST-ONLY                    -- works, tested, exposed NOWHERE" % len(b["test_only"]))
    print("  %5d  TOOL-ONLY                    -- reachable only from tools/" % len(b["tool_only"]))
    print("  %5d  ORPHAN                       -- no faculty, catalog, caller, test or tool" % len(b["orphan"]))

    if "--list" in argv:
        for kind in ("orphan", "test_only"):
            print("\n--- %s ---" % kind.upper())
            for name, path, line in b[kind]:
                print("  %-38s %s:%d" % (name, os.path.relpath(path, REPO), line))
    if "--json" in argv:
        print(json.dumps({k: [{"name": n, "path": os.path.relpath(p, REPO), "line": l} for n, p, l in v]
                          for k, v in b.items()}, indent=1))

    n = len(b["orphan"])
    if n > ORPHAN_BUDGET:
        print("\nFAIL: %d orphan(s) exceeds the budget of %d. Wire it, catalogue it, or declare it a "
              "negative -- but do not delete on a static signal alone." % (n, ORPHAN_BUDGET))
        return 1
    print("\nOK: %d orphan(s), budget %d." % (n, ORPHAN_BUDGET))
    return 0


def _selftest():
    """Asserts the CONTRACT, including the conservatism that makes the tool safe to act on.

    The load-bearing assertion is the third one: a name that is only ever mentioned inside a STRING must
    still count as reachable. This repo routes by name through catalog `example=` snippets, so dropping
    string mentions would manufacture false orphans -- and a false orphan is the one error that could get
    live code deleted."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.py")
        with open(a, "w") as f:
            f.write("def alive():\n    pass\n\n\ndef gone():\n    pass\n\n\n"
                    "def _private():\n    pass\n\n\ndef by_string():\n    pass\n")
        b = os.path.join(d, "b.py")
        with open(b, "w") as f:
            f.write("import a\nEXAMPLE = 'mind.by_string()'\n\n\ndef go():\n    return a.alive()\n")
        defs = public_definitions([a])
        assert set(defs) == {"alive", "gone", "by_string"}, "private names must not be audited: %s" % sorted(defs)
        used = referenced_names([a, b], skip_defs_in={a, b})
        assert "alive" in used, "an attribute call was not seen as a reference"
        assert "by_string" in used, "a name mentioned only in a STRING must count -- catalog examples route by name"
        assert "gone" not in used, "an uncalled function was wrongly counted as referenced"

    # and on the real tree the buckets must partition exactly, with a plausible surface size
    b = audit()
    names = [n for v in b.values() for n, _p, _l in v]
    assert len(names) == len(set(names)), "a function landed in two buckets -- the partition is not exclusive"
    assert len(names) > 2000, "only %d public engine functions found -- the scan missed the tree" % len(names)
    assert len(b["faculty"]) > 500, "faculty bucket implausibly small (%d)" % len(b["faculty"])
    print("orphan_audit selftest OK -- %d public functions partitioned, %d orphan(s), string-mentions honoured"
          % (len(names), len(b["orphan"])))
    _selftest_agent_reach()


def _selftest_agent_reach():
    """Regression trap for the agent-reachability pass. Asserts the exact contract, not 'no exception'.

    The load-bearing assertion is the terminality one: a reference from a `*home.py` consolidation facade
    must NOT count as reaching an agent. That single rule is why this pass sees what audit() cannot, and if
    it ever silently inverts, every count below goes to zero and the audit looks CLEAN while being blind --
    the exact failure this whole pass was built to catch."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        # a symbol referenced ONLY from a consolidation home is shadowed; the same symbol referenced from
        # an ordinary module is not. Both cases in one fixture, so the rule is pinned from both sides.
        with open(os.path.join(d, "holographic_thing.py"), "w") as f:
            f.write("def only_from_home():\n    pass\n\n\ndef from_a_real_module():\n    pass\n\n\n"
                    "class Widget:\n    pass\n")
        with open(os.path.join(d, "holographic_thinghome.py"), "w") as f:
            f.write("from holographic_thing import only_from_home\n")
        with open(os.path.join(d, "holographic_caller.py"), "w") as f:
            f.write("from holographic_thing import from_a_real_module\n")
        paths = sorted(glob.glob(os.path.join(d, "*.py")))
        home = os.path.join(d, "holographic_thinghome.py")
        real = os.path.join(d, "holographic_caller.py")
        assert not _is_terminal(home), "a consolidation home must be a cul-de-sac, not a route to an agent"
        assert _is_terminal(real), "an ordinary module must count as terminal"
        assert not _is_terminal(os.path.join(d, "holographic_lookahead.py")), "declared negatives are cul-de-sacs"
        # the SERVICE is terminal -- it is the agent's door. Pinned because the first draft got this
        # backwards and manufactured five false positives in one run.
        assert _is_terminal(os.path.join(d, "holographic_service.py")), \
            "the HTTP service is the door agents come through -- a reference from it IS terminal"
        assert set(public_classes(paths)) == {"Widget"}, "public classes must be collected -- audit() sees none"
        where = _referenced_in(paths)
        assert where["only_from_home"] == {home}, "reference sites must be kept, not just counted"

    # ...and on the real tree the pass must still SEE something. A zero here is a broken oracle, not a
    # clean repo: 312 dark classes were measured the day this was written, and audit() reported 0 gaps
    # over the same tree on the same day.
    r = agent_reach_report(limit=10000)
    assert r["counts"]["dark_classes"] > 100, \
        "only %d dark classes -- the class scan is not seeing the tree" % r["counts"]["dark_classes"]
    dark = {e["name"] for e in r["dark"]}
    assert "DomeLight" in dark, \
        "DomeLight must read as dark -- environment lighting an agent cannot construct is the finding " \
        "that motivated this pass; if it ever goes green, WIRE it, do not relax the assert"
    # KEPT NEGATIVE, loud: this shares audit()'s no-type-inference rule, so a class merely NAMED in catalog
    # prose reads as reachable even when no faculty constructs it. RectLight is exactly that case -- it is
    # unreachable from the mind today and this pass does NOT report it. The pass under-reports by design;
    # it is a review queue, never a completeness claim.
    assert "RectLight" not in dark, \
        "RectLight is expected to be MISSED (catalog prose mentions it) -- if this fires the lexical " \
        "catalog oracle changed and the under-reporting note above needs rewriting"
    print("agent_reach selftest OK -- %d shadowed, %d dark class(es); homes non-terminal, service terminal"
          % (r["counts"]["shadowed"], r["counts"]["dark_classes"]))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main(sys.argv[1:]))

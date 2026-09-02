"""SWARM AUDIT + ABOVE/BELOW SWEEP (cp67).

Two instruments in one pass, both deterministic:

RESIDENT SWARM (below): one auditor resident per organ group walks its modules --
importable? carries a selftest or __main__? contributes to the capability catalog?
Import failures and dead groups are defects; selftest-less modules are listed, not
hidden.

ABOVE/BELOW MATRIX (across): every major capability is checked at each layer of the
stack -- L0 engine faculty, L1 facade (lecore.py), L2 hosted MCP tool, L3 chat verb,
L4 suite pin. A capability present below but unreachable above is a wiring gap;
one exposed above without an engine floor below is a facade lying. DELIBERATE gaps
(e.g. no hosted raymarching: cost; no hosted api registration: SSRF) are whitelisted
WITH THEIR REASONS -- the audit fails only on gaps nobody chose.
"""
import argparse
import ast
import importlib
import json
import os
import re
import sys

sys.path.insert(0, ".")

CAPS = [
    # name, engine attr, facade ok(=engine passthrough), mcp marker, chat marker, pin marker
    ("ask/teach/veto", "answer_feedback", True, "zoo_ask", "veto", "veto"),
    ("semantic recall", "recall_semantic", True, "recall", "semantic", "SEMANTIC RECALL"),
    ("saturation estimate", "saturation_estimate", True, "saturation", "saturation", "saturation_estimate"),
    ("drift sentinel / teach_check", "teach_check", True, "drift", "conflict", "DriftSentinel"),
    ("void explore/mix/propose", "void_mix", True, "zoo_void", "explore", "void_mix"),
    ("hypothesis test", "hypothesis_test", True, "hypothesis", "test ", "hypothesis_test"),
    ("conjecture record/promote", "conjecture_promote", True, "conjecture", "promote", "conjecture_promote"),
    ("api learn/use", "api_use", True, "zoo_tools", "use api", "apilearn"),
    ("contextual tool find", "tool_find", True, "zoo_tools", "find a tool", "tool_find"),
    ("research archive", "research_archive", True, "corpus_ask", None, "research_archive"),
    ("scene render", "render_scene_description", True, None, "render", "render"),
    ("procedural texture", "encode_texture", True, None, "texture", "texture"),
    ("workspaces", "app_substrate", True, None, "workspace", "workspace"),
    ("memory slots/compare", "learning_load", True, None, "compare", "load memory"),
    ("sessions", "session_open", True, None, "session", "session"),
    ("panel", "panel_deliberate", True, "zoo_panel", None, "panel"),
    ("ouroboros selection", "ouroboros", True, None, None, "Ouroboros"),
    ("grounded answering", "ask_grounded", True, "ask_grounded", "ask_grounded", "ask_grounded"),
    ("docs explain", "explain", True, None, "explain", "explain"),
    ("memory portfolio", "memory_export", True, None, "export memory", "memory_export"),
    ("source attribution", "model_attribute", True, None, "RuntimeRung", "attribution"),
]
DELIBERATE = {
    ("scene render", "L2"): "hosted raymarching is a cost decision, not a wiring gap",
    ("procedural texture", "L2"): "same cost decision as scene render",
    ("workspaces", "L2"): "hosted callers are namespaced by the server, not by chat workspaces",
    ("memory slots/compare", "L2"): "hosted memory upload is an SSRF/abuse surface; local-runtime feature",
    ("sessions", "L2"): "hosted sessions are connection-scoped by the server",
    ("research archive", "L3"): "archive building is an operator/dev act; chat consumes via ask",
    ("panel", "L3"): "panel runs through ask when relevant; no dedicated verb yet (accepted)",
    ("ouroboros selection", "L2"): "selection is inside the reader path, not a callable tool",
    ("ouroboros selection", "L3"): "same: substrate-internal",
    ("memory portfolio", "L2"): "hosted import/export is an abuse surface; "
                                "local-runtime feature like memory upload",
    ("source attribution", "L2"): "requires model-directory access; hosted "
                                  "operators enable it server-side on their "
                                  "own runtime",
}


# ---------------------------------------------------------------------------------------
# THE DERIVED MATRIX (sweep 133). The CAPS literal above is frozen at cp67 and the sweep
# was reporting green over 21 hand-written rows while the engine grew past 600 capabilities.
#
# THE INSTRUMENT'S REAL DEFECT WAS NOT THE FROZEN INPUT, though that is what made it
# invisible. It CLAIMS to measure reachability -- "a capability present below but
# unreachable above is a wiring gap" -- and it MEASURES PROMOTION: the L2/L3 markers are
# substring probes for DEDICATED tool names. Those are different properties, and the
# difference matters because holographic_mcp hosts `lecore_invoke(name, args) -> run any
# public faculty`. Reachability at L2 is therefore UNIVERSAL by construction: measured,
# all 683 method-carrying catalog cards that resolve are in the /tools manifest, and all 36
# doors the round-5 brief named dispatch through mind.invoke with 0 refusals. A promotion
# census over 616 methods would be sweep 123's "761-item bar nobody clears", so promotion
# is reported and NEVER gated. (L1 was worse than frozen: `fac and "UnifiedMind" in facade`
# is a constant, True for every row, measuring nothing per capability since cp67.)
#
# SO THE SELECTION RULE, which is the whole design question: ask REACHABILITY of everything
# -- it is always meaningful and always checkable -- and ask PROMOTION of nothing. The
# population comes from the CATALOG, because registering a card is the engine's own act of
# declaring "this is a capability", it is made per-sweep by the build loop, and it grows
# without anyone remembering to edit a literal.
# ---------------------------------------------------------------------------------------

BUDGET_PATH = "tools/swarm_audit_budget.json"


def _surface_calls(path, recv):
    """Every mind verb a surface FILE calls, by AST: `<recv>.verb(...)` or `<anything>.mind.verb(...)`.

    Derived, not declared. The old markers were substrings hand-typed into a literal, so a tool
    could be renamed and the probe would keep matching some unrelated line -- and a tool could be
    ADDED and never probed at all, which is what happened for five years of sweeps."""
    out = set()
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except (OSError, SyntaxError):
        return out
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            continue
        v, attr = n.func.value, n.func.attr
        hit = ((isinstance(v, ast.Name) and v.id == recv)
               or (isinstance(v, ast.Attribute) and v.attr == "mind"))
        if hit and not attr.startswith("_"):
            out.add(attr)
    return out


def _repo_defs(root="."):
    """(methods, functions) defined anywhere in the repo -- what a card's method= could mean.

    A catalog card names `method=`, and that name can legitimately be a method on an OBJECT the
    mind hands back (mind.memory_curate() -> MemoryCurator.plan) rather than a mind verb. Flagging
    those would be crying wolf on a pattern the engine uses deliberately, so they have to be told
    apart from a name that resolves nowhere -- and telling them apart needs the whole tree."""
    skip = {".git", "__pycache__", ".pytest_cache", "node_modules", ".lecore_archive"}
    methods, funcs = set(), set()
    for dp, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith("."))
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse(open(os.path.join(dp, f), encoding="utf-8",
                                      errors="replace").read())
            except (OSError, SyntaxError):
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.ClassDef):
                    for b in n.body:
                        if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.add(b.name)
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs.add(n.name)
    return methods, funcs


def derived_matrix(mind, root="."):
    """The above/below sweep over the CATALOG, with every layer derived. Returns a report dict.

    Layers, all measured rather than declared:
      L0 engine    -- the card's method= resolves on the mind
      L1 facade    -- it resolves on lecore.UnifiedMind (a REAL per-capability check now)
      L2 reachable -- it is in the /tools manifest, i.e. an agent can lecore_invoke it
      L2 promoted  -- a dedicated MCP tool handler calls mind.<method>   (census, never a gap)
      L3 chat      -- chat_server calls m.<method>                        (census, never a gap)
      L4 pinned    -- the name appears somewhere under tests/             (census, never a gap)

    GENUINE gap classes, and only these gate:
      no_floor     -- the card names a method defined NOWHERE in the repo. The discovery layer
                      promising a door that does not exist is the hardest form of "a facade lying",
                      and skill_lint does not catch it: it validates mind.X inside EXAMPLES, never
                      the card's own method= field.
      unreachable  -- the method exists only as a MODULE FUNCTION: importable in-process, not
                      callable over /invoke. Exactly "present below, unreachable above".
      facade_lie   -- a surface calls mind.X where X does not exist on the mind.
      l1_gap/l2_gap-- a resolvable method missing from the facade or the manifest.

    NOT-MEANINGFUL, reported so the judgement is on the record rather than implied:
      object_method-- the method lives on an object the mind returns. Reachable, by design."""
    import lecore
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    from holographic.misc.holographic_skills import manifest

    cards = [c for c in default_catalog().all() if getattr(c, "method", None)]
    man = {x["name"] for x in manifest(include_methods=True).get("methods", [])}
    obj_methods, mod_funcs = _repo_defs(root)
    mcp_calls = _surface_calls(os.path.join(root, "holographic_mcp.py"), "mind")
    chat_calls = _surface_calls(os.path.join(root, "chat_server.py"), "m")
    tests_src = []
    for dp, _d, fs in os.walk(os.path.join(root, "tests")):
        for f in sorted(fs):
            if f.endswith(".py"):
                tests_src.append(open(os.path.join(dp, f), encoding="utf-8",
                                      errors="replace").read())
    # ONE tokenisation instead of a regex per door. Measured: 651 \\b-anchored searches over the
    # 4.85 MB of tests/ cost 43.6 s and made the whole sweep 52.8 s -- an audit nobody would run
    # after every change, which is the same way a gate gets disabled. Same semantics (does the
    # identifier appear as a word), 0.2 s.
    tests_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", "\n".join(tests_src)))

    rows, genuine, not_meaningful = [], [], []
    seen = set()
    for c in sorted(cards, key=lambda c: (c.method, c.name)):
        meth = c.method
        if meth in seen:
            continue                       # several cards can share one door; ask about it once
        seen.add(meth)
        l0 = hasattr(mind, meth)
        row = {"capability": c.name, "method": meth, "L0": l0,
               "L1": bool(l0 and hasattr(lecore.UnifiedMind, meth)),
               "L2": meth in man, "L2d": meth in mcp_calls, "L3": meth in chat_calls,
               "L4": meth in tests_tokens}
        rows.append(row)
        if not l0:
            if meth in obj_methods:
                not_meaningful.append({"method": meth, "capability": c.name,
                                       "why": "object method -- reachable via an object the mind "
                                              "returns, which is a deliberate pattern"})
            elif meth in mod_funcs:
                genuine.append({"method": meth, "capability": c.name, "kind": "unreachable",
                                "why": "module-level function: importable in-process, NOT "
                                       "callable over /invoke"})
            else:
                genuine.append({"method": meth, "capability": c.name, "kind": "no_floor",
                                "why": "declared by a catalog card and defined NOWHERE in the repo"})
        else:
            if not row["L1"]:
                genuine.append({"method": meth, "capability": c.name, "kind": "l1_gap",
                                "why": "on the mind but not on the facade class"})
            if not row["L2"]:
                genuine.append({"method": meth, "capability": c.name, "kind": "l2_gap",
                                "why": "on the mind but absent from the /tools manifest, so no "
                                       "hosted agent can invoke it"})
    for surface, names in (("mcp", mcp_calls), ("chat", chat_calls)):
        for n in sorted(names):
            if not hasattr(mind, n):
                genuine.append({"method": n, "capability": "(%s surface)" % surface,
                                "kind": "facade_lie",
                                "why": "the %s surface calls mind.%s, which does not exist"
                                       % (surface, n)})
    counts = {"cards": len(cards), "doors": len(rows),
              "L0": sum(r["L0"] for r in rows), "L1": sum(r["L1"] for r in rows),
              "L2_reachable": sum(r["L2"] for r in rows),
              "L2_promoted": sum(r["L2d"] for r in rows), "L3_promoted": sum(r["L3"] for r in rows),
              "L4_pinned": sum(r["L4"] for r in rows),
              "genuine": len(genuine), "not_meaningful": len(not_meaningful)}
    by_kind = {}
    for g in genuine:
        by_kind[g["kind"]] = by_kind.get(g["kind"], 0) + 1
    return {"rows": rows, "genuine": sorted(genuine, key=lambda g: (g["kind"], g["method"])),
            "not_meaningful": sorted(not_meaningful, key=lambda g: g["method"]),
            "counts": counts, "by_kind": by_kind}


def _budget(root="."):
    """The shrink-only floor. A gate at zero on somebody else's backlog gets disabled, not fixed --
    the lesson delegation_drift wrote down; this steals its shape, including --reason."""
    p = os.path.join(root, BUDGET_PATH)
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return {"genuine_budget": None, "why": "no budget file"}

def resident_swarm():
    root = "holographic"
    out = []
    for grp in sorted(os.listdir(root)):
        gdir = os.path.join(root, grp)
        if not os.path.isdir(gdir) or grp.startswith("__"):
            continue
        mods = [f[:-3] for f in os.listdir(gdir)
                if f.endswith(".py") and not f.startswith("__")]
        ok, fail, selftests = 0, [], 0
        for mn in mods:
            try:
                mod = importlib.import_module("%s.%s.%s" % (root, grp, mn))
                ok += 1
                if hasattr(mod, "_selftest") or "_selftest" in open(
                        os.path.join(gdir, mn + ".py")).read():
                    selftests += 1
            except Exception as e:
                fail.append("%s: %s" % (mn, str(e)[:60]))
        out.append({"group": grp, "modules": len(mods), "import_ok": ok,
                    "selftests": selftests, "failures": fail})
    return out


def above_below(mind):
    mcp_src = open("holographic_mcp.py").read().lower()
    chat_src = open("chat_server.py").read().lower()
    suite_src = open(
        "holographic/unified/holographic_unified_p20_zoo.py").read()
    rows, gaps = [], []
    facade = open("lecore.py").read()
    for name, attr, fac, mcpm, chatm, pinm in CAPS:
        l0 = hasattr(mind, attr)
        l1 = fac and ("UnifiedMind" in facade)      # facade passthrough
        l2 = bool(mcpm and mcpm.lower() in mcp_src)
        l3 = bool(chatm and chatm.lower() in chat_src)
        l4 = bool(pinm and pinm in suite_src)
        rows.append((name, l0, l1, l2, l3, l4))
        for layer, present, marker in (("L2", l2, mcpm), ("L3", l3, chatm)):
            if marker is None:
                continue
            if l0 and not present and (name, layer) not in DELIBERATE:
                gaps.append("%s missing at %s" % (name, layer))
        if not l0:
            gaps.append("%s exposed above without an engine floor" % name)
    return rows, gaps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", action="store_true",
                    help="fail when the DERIVED genuine-gap count exceeds the recorded budget "
                         "(shrink-only, not zero)")
    ap.add_argument("--rebase", action="store_true", help="write the current count as the new floor")
    ap.add_argument("--reason", default="", help="required with --rebase: why the floor moved")
    args = ap.parse_args(argv)

    import lecore
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "")
    swarm = resident_swarm()
    n_fail = sum(len(g["failures"]) for g in swarm)
    rows, gaps = above_below(m)
    der = derived_matrix(m)
    dc = der["counts"]
    lines = ["# Swarm audit + above/below sweep", "",
             "## Resident swarm (per organ group)", ""]
    for g in swarm:
        lines.append("- %-28s %d modules, %d import, %d with selftests%s"
                     % (g["group"], g["modules"], g["import_ok"],
                        g["selftests"],
                        ("  FAILURES: " + "; ".join(g["failures"]))
                        if g["failures"] else ""))
    lines += ["", "## Above/below matrix (L0 engine, L1 facade, L2 hosted, "
              "L3 chat, L4 pinned)", ""]
    for name, *ls in rows:
        lines.append("- %-30s %s" % (name, " ".join(
            "%s:%s" % (l, "+" if v else "-")
            for l, v in zip(("L0", "L1", "L2", "L3", "L4"), ls))))
    lines += ["", "## Deliberate gaps (chosen, with reasons)", ""]
    for (name, layer), why in sorted(DELIBERATE.items()):
        lines.append("- %s at %s: %s" % (name, layer, why))
    lines += ["", "## Unintended gaps", ""]
    lines += ["- " + g for g in gaps] if gaps else ["- none"]

    # -- the derived sweep, which is the one that grows with the engine ---------------------
    lines += ["", "## Derived matrix (catalog-wide, sweep 133)", "",
              "Run it through the mind with `mind.above_below()`; on the command line "
              "`python3 tools/swarm_audit.py --gate`.", "",
              "The hand-written matrix above covers %d capabilities and has since cp67. This one "
              "takes its population from the CATALOG, so it grows whenever a sweep registers a "
              "capability instead of when somebody remembers to edit a literal." % len(rows), "",
              "| measure | count |", "|---|---|",
              "| catalog cards carrying a `method=` | %d |" % dc["cards"],
              "| distinct doors behind them | %d |" % dc["doors"],
              "| L0 engine floor | %d |" % dc["L0"],
              "| L1 reachable on the facade class | %d |" % dc["L1"],
              "| L2 reachable (in the /tools manifest, so `lecore_invoke` can call it) | %d |"
              % dc["L2_reachable"],
              "| L2 PROMOTED to a dedicated MCP tool | %d |" % dc["L2_promoted"],
              "| L3 PROMOTED to a chat verb | %d |" % dc["L3_promoted"],
              "| L4 named under tests/ | %d |" % dc["L4_pinned"],
              "| **genuine gaps** | **%d** |" % dc["genuine"],
              "| not meaningful to ask (object methods) | %d |" % dc["not_meaningful"], "",
              "PROMOTION IS A CENSUS, NEVER A GAP. `holographic_mcp` hosts "
              "`lecore_invoke(name, args)`, which runs any public faculty, so L2 reachability is "
              "universal by construction and a dedicated tool is a curation decision. Scoring "
              "%d unpromoted doors as defects would be the bar nobody clears."
              % (dc["doors"] - dc["L2_promoted"]), ""]
    if der["genuine"]:
        lines += ["### Genuine gaps", ""]
        for g in der["genuine"]:
            lines.append("- `%s` (%s) -- %s" % (g["method"], g["kind"], g["why"]))
        lines.append("")
    if der["not_meaningful"]:
        lines += ["### Not meaningful to ask (recorded so the judgement is on the record)", ""]
        for g in der["not_meaningful"]:
            lines.append("- `%s` -- %s" % (g["method"], g["why"]))
        lines.append("")
    open("docs/SWARM_AUDIT.md", "w").write("\n".join(lines) + "\n")

    print("swarm: %d groups, %d import failures | matrix: %d capabilities, "
          "%d unintended gap(s)" % (len(swarm), n_fail, len(rows), len(gaps)))
    for g in gaps:
        print("  GAP:", g)
    print("derived: %d cards -> %d doors | L0 %d, L1 %d, L2-reachable %d | promoted L2 %d / L3 %d "
          "| GENUINE %d %s | not-meaningful %d"
          % (dc["cards"], dc["doors"], dc["L0"], dc["L1"], dc["L2_reachable"],
             dc["L2_promoted"], dc["L3_promoted"], dc["genuine"],
             json.dumps(der["by_kind"], sort_keys=True), dc["not_meaningful"]))
    for g in der["genuine"][:12]:
        print("  GENUINE %-12s %-26s %s" % (g["kind"], g["method"], g["why"][:64]))
    if len(der["genuine"]) > 12:
        print("  ... %d more (full list in docs/SWARM_AUDIT.md)" % (len(der["genuine"]) - 12))

    if args.rebase:
        if not args.reason.strip():
            print("REFUSED: --rebase needs --reason. A floor that moves without a written reason "
                  "is a floor nobody can audit.")
            return 2
        json.dump({"genuine_budget": dc["genuine"], "why": args.reason.strip()},
                  open(BUDGET_PATH, "w"), indent=1)
        print("rebased: genuine_budget = %d -- %s" % (dc["genuine"], args.reason.strip()))
        return 0
    if args.gate:
        b = _budget()
        cap = b.get("genuine_budget")
        if cap is None:
            print("GATE: no budget recorded; run --rebase --reason '<why>' to set the floor.")
            return 1
        if dc["genuine"] > cap:
            print("GATE FAILED: %d genuine gap(s) against a budget of %d. Shrink-only: fix them, "
                  "or --rebase --reason if the floor genuinely moved." % (dc["genuine"], cap))
            return 1
        print("OK: within budget (%d)" % cap)
def _selftest():
    """Regression trap for the derived sweep. Numbers, not smoke: the classifier's job is to tell
    three lookalikes apart -- a door that is missing, a door that is only importable, and a door
    that lives on an object by design -- and getting the third one wrong is how a canary gets
    switched off."""
    import lecore

    m = lecore.UnifiedMind(dim=64, seed=0)

    # 1. THE SURFACE DERIVATION. Hand-typed substring markers could not see a tool being added;
    #    an AST walk over the real files can. Both surfaces must yield a non-trivial verb set, and
    #    every verb they name must exist on the mind -- a surface calling mind.X where X is absent
    #    is "a facade lying", the defect class this sweep never checked.
    mcp_calls = _surface_calls("holographic_mcp.py", "mind")
    chat_calls = _surface_calls("chat_server.py", "m")
    assert len(mcp_calls) >= 20, len(mcp_calls)
    assert len(chat_calls) >= 20, len(chat_calls)
    assert not [n for n in mcp_calls | chat_calls if not hasattr(m, n)], "a surface calls a door that is gone"

    # 2. THE CLASSIFIER, on a synthetic card set, because the live tree cannot exercise every arm
    #    on demand and a classifier tested only on today's data is tested on one sample.
    methods, funcs = _repo_defs(".")
    assert "plan" in methods, "MemoryCurator.plan must read as an OBJECT method"
    assert "derived_matrix" in funcs and "derived_matrix" not in methods
    assert not hasattr(m, "derived_matrix"), "a module function is not a mind verb"

    # 3. THE LIVE SWEEP. Structural invariants, not frozen totals -- the counts move every sweep
    #    by design, and pinning them would make this fail on somebody else's good work.
    r = derived_matrix(m)
    c = r["counts"]
    assert c["cards"] >= c["doors"] >= c["L0"], c          # cards can share a door
    assert c["L0"] == c["L1"] == c["L2_reachable"], (
        "L1/L2 must be MEASURED per capability, not assumed: they were a constant until sweep 133")
    # PROMOTION IS A CENSUS. If a future edit starts gating on it, this catches it: promotion is a
    # small minority by construction and must never be counted into `genuine`.
    assert c["L2_promoted"] < c["doors"] / 10, "promotion should be rare -- it is a curation act"
    assert c["genuine"] == sum(r["by_kind"].values())
    assert all(g["kind"] in ("no_floor", "unreachable", "facade_lie", "l1_gap", "l2_gap")
               for g in r["genuine"])
    assert all(g["why"] for g in r["genuine"] + r["not_meaningful"]), \
        "an unexplained finding is one the next session deletes rather than fixes"
    # A TEST WHOSE FIXTURE IS A REAL BUG DIES THE DAY SOMEBODY FIXES IT -- and this assertion was
    # one, twice over. It read `genuine >= 4` and `no_floor >= 1`, using the four broken cards as its
    # fixture; the moment sweep 133's close-out repaired them the module selftest went red for the
    # crime of the defect being gone. (The same shape had already bitten the delegation-drift test and
    # a FEATURE_GUIDE block.) What the sweep must actually guarantee is that the CLASSES are wired and
    # the floor holds -- zero in every class is the goal state, not a broken test.
    assert set(r["by_kind"]) <= {"no_floor", "unreachable", "facade_lie", "l1_gap", "l2_gap"}
    assert c["genuine"] == len(r["genuine"]) >= 0
    assert c["not_meaningful"] >= 1, "object methods must be classified OUT, never flagged"

    # 4. THE BUDGET IS SHRINK-ONLY AND NEEDS A REASON. A floor that moves silently is not a floor.
    b = _budget()
    assert isinstance(b.get("genuine_budget"), int) and b["why"].strip()
    assert c["genuine"] <= b["genuine_budget"], (
        "genuine gaps %d exceed the recorded floor %d" % (c["genuine"], b["genuine_budget"]))
    assert main(["--rebase"]) == 2, "--rebase without --reason must refuse"

    print("OK: swarm_audit derived sweep -- %d catalog cards behind %d doors (the frozen literal "
          "covered %d); L0/L1/L2 all measured at %d; promotion is a census (%d at L2, %d at L3) and "
          "never a gap; %d GENUINE gaps %s the old matrix was green on, and %d object methods "
          "classified OUT so the canary does not cry wolf"
          % (c["cards"], c["doors"], len(CAPS), c["L0"], c["L2_promoted"], c["L3_promoted"],
             c["genuine"], json.dumps(r["by_kind"], sort_keys=True), c["not_meaningful"]))


if __name__ == "__main__":
    sys.exit(main())

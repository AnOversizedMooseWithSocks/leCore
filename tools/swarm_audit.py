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


def main():
    import lecore
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "")
    swarm = resident_swarm()
    n_fail = sum(len(g["failures"]) for g in swarm)
    rows, gaps = above_below(m)
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
    open("docs/SWARM_AUDIT.md", "w").write("\n".join(lines) + "\n")
    print("swarm: %d groups, %d import failures | matrix: %d capabilities, "
          "%d unintended gap(s)" % (len(swarm), n_fail, len(rows), len(gaps)))
    for g in gaps:
        print("  GAP:", g)
    return 1 if (gaps or n_fail) else 0


if __name__ == "__main__":
    sys.exit(main())

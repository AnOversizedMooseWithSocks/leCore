"""doc_coverage.py -- which UnifiedMind verbs does NO human-written document mention?

WHY THIS EXISTS (sweep 123). Generated docs (CAPABILITIES.md, REFERENCE.md, API_QUICKREF.md)
cover every faculty by construction -- they are regenerated from the live catalog and the
docstrings, and regen_docs --check refuses drift. What they cannot measure is the HUMAN
layer: README, FEATURE_GUIDE, WHATS_NEW, docs/*.md, the integration guides. Measured at
the start of the sweep: the doors built across thirty sweeps (orient, serve, study,
bequeath, contribute, api_learn, merge_trees ...) were mentioned by ONE human document,
written the day before. Nobody had an instrument that said so.

WHAT IT MEASURES. The public methods of UnifiedMind (the faculty surface a caller reaches
by name), against the union of human docs. A verb counts as mentioned when `name(` or
`.name` appears in any of them. The number is loud, not gated: the surface is ~2,000
verbs and a human guide should not mention them all -- it should mention the doors.
So `--check` enforces ONE thing: the count of UNMENTIONED verbs may not grow past the
recorded budget (tools/doc_coverage_budget.json -- recorded when this tool shipped,
lowered whenever docs improve). The catalog-door number is REPORTED, not gated:
`native=True` is on ~894 cards, so "every native door documented" would be a 761-item
gate on day one -- a bar nobody clears is a bar nobody runs. Measured on the day this
shipped: 2,318 verbs, 240 (10%) mentioned, 761 of 894 native-card verbs unmentioned.
Those two numbers are the honest state of the human layer; drive them down per sweep.

USAGE
    python3 tools/doc_coverage.py             # the report
    python3 tools/doc_coverage.py --check     # CI mode: budget + every native door documented
    python3 tools/doc_coverage.py --rebase    # record today's count as the budget (with a reason)
"""
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = os.path.join(REPO, "tools", "doc_coverage_budget.json")
HUMAN_DOCS = (["README.md", "FEATURE_GUIDE.md", "AGENTS.md", "DEVELOPMENT_STRATEGY.md",
               "RENDERING_GUIDE.md", "GALLERY.md", "writing_vsa_programs.md"]
              + glob.glob(os.path.join(REPO, "docs", "*.md"))
              + glob.glob(os.path.join(REPO, "integrations", "**", "*.md"), recursive=True))
GENERATED = ("NOTES_concepts.md", "CAPABILITIES.md", "REFERENCE.md", "API_QUICKREF.md",
             "FACULTY_MAP.md", "DOC_MAP.md", "PIPELINE_MAP.md", "ZOO.md", "SERVICE.md")


def human_text():
    out = []
    for p in HUMAN_DOCS:
        p = p if os.path.isabs(p) else os.path.join(REPO, p)
        if os.path.exists(p) and not any(g in p for g in GENERATED):
            out.append(open(p, encoding="utf-8", errors="ignore").read())
    return "\n".join(out)


def mind_verbs():
    sys.path.insert(0, REPO)
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    return sorted(n for n in dir(m) if not n.startswith("_") and callable(getattr(m, n, None)))


def native_doors():
    """Catalog cards registered native=True whose example calls a mind verb: the declared doors."""
    sys.path.insert(0, REPO)
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    doors = set()
    for cap in default_catalog().all():
        if getattr(cap, "native", False):
            for v in re.findall(r"\.([a-zA-Z_][a-zA-Z0-9_]*)\(", str(getattr(cap, "example", "") or "")):
                doors.add(v)
    return doors


def report():
    text = human_text()
    verbs = mind_verbs()
    mentioned = {v for v in verbs if ("%s(" % v) in text or (".%s" % v) in text}
    unmentioned = [v for v in verbs if v not in mentioned]
    doors = native_doors() & set(verbs)
    dark_doors = sorted(d for d in doors if d not in mentioned)
    return {"verbs": len(verbs), "mentioned": len(mentioned), "unmentioned": len(unmentioned),
            "native_doors": len(doors), "dark_doors": dark_doors, "unmentioned_list": unmentioned}


def main(argv):
    r = report()
    print("mind verbs %d | mentioned in human docs %d (%.0f%%) | unmentioned %d | native doors %d, undocumented %d"
          % (r["verbs"], r["mentioned"], 100.0 * r["mentioned"] / max(r["verbs"], 1), r["unmentioned"],
             r["native_doors"], len(r["dark_doors"])))
    if "--rebase" in argv:
        with open(BUDGET, "w") as f:
            json.dump({"unmentioned_budget": r["unmentioned"],
                       "why": "recorded by --rebase; may only shrink -- lower it when docs improve"}, f, indent=1)
        print("budget recorded:", r["unmentioned"])
        return 0
    if "--check" in argv:
        budget = json.load(open(BUDGET))["unmentioned_budget"] if os.path.exists(BUDGET) else None
        bad = 0
        if budget is not None and r["unmentioned"] > budget:
            print("FAIL: unmentioned verbs grew %d -> %d; document the new ones or --rebase WITH A REASON"
                  % (budget, r["unmentioned"]))
            bad = 1
        print("OK: within budget (%s)" % budget if not bad else "")
        return bad
    print("undocumented native doors:", r["dark_doors"][:20])
    print("unmentioned verbs (first 30):", r["unmentioned_list"][:30])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

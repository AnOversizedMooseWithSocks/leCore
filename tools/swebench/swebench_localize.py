"""swebench_localize.py -- SWE-bench Verified FILE LOCALISATION with the substrate only (no model).

Task: given the issue text, rank the repository's files (the tree at the instance's exact base_commit) so the
files the gold patch touched come first. Scored as any-gold-in-top-k and all-gold-in-top-k, k in 1/3/5/10.
The retriever is the catalog router's family: token overlap between the issue and each path, IDF-weighted over
the tree, with exact bonuses for a dotted module path or a basename the issue mentions, and test/doc paths
down-weighted (patches touch source). Deterministic; ~4k files per instance; 500 instances.

    PYTHONHASHSEED=0 python3 swebench_localize.py [--per-repo]
"""
import json
import math
import re
import sys
import collections

rows = json.load(open("/home/claude/swebench/verified.json"))
trees = json.load(open("/home/claude/swebench/trees_at_base.json"))
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOP = set("the a an to of in on for and or is it this that with as by be are was were not from at if when which"
           " should would could can does do did has have had i we you they he she but so than then there here"
           " also into up out about over after before its their our your my me my us them him her what how why"
           " where who whom whose all any some no yes more most much many very just like get got set use used"
           " using see error errors issue issues bug bugs fix fixed fixes problem expected actual result results"
           " code python file files line lines return returns value values object objects test tests testing"
           " model models function functions method methods class classes call calls called example examples".split())


def toks(s):
    out = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s):
        parts = [p.lower() for p in CAMEL.findall(w.replace("_", " ")) if p]
        out += [p for p in parts if len(p) > 2 and p not in STOP]
    return out


def path_tokens(p):
    stem = p.rsplit(".", 1)[0]
    return toks(stem.replace("/", " "))


def localize(inst, files, k=10):
    text = inst["problem_statement"] + " " + inst.get("hints_text", "")
    q = collections.Counter(toks(text))
    # IDF over the tree: rare path tokens carry the evidence
    df = collections.Counter()
    ptoks = {}
    for f in files:
        t = set(path_tokens(f))
        ptoks[f] = t
        df.update(t)
    N = len(files)
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}
    # exact mentions: dotted module paths and basenames in the issue
    dotted = set(m.replace(".", "/") for m in re.findall(r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){1,6}\b", text.lower()))
    bases = set(re.findall(r"\b([a-z_][a-z0-9_]*\.py)\b", text.lower()))
    scored = []
    for f in files:
        if not f.endswith(".py"):
            continue
        s = sum(idf[t] * min(q[t], 3) for t in ptoks[f] if t in q)
        fl = f.lower()
        stem = fl[:-3]
        if any(stem.endswith(d) or d.endswith(stem) or ("/" + d + "/") in ("/" + stem + "/") for d in dotted if len(d) > 8):
            s += 12.0
        if fl.rsplit("/", 1)[-1] in bases:
            s += 8.0
        if "/tests/" in "/" + fl or fl.startswith("tests/") or "/test_" in "/" + fl or "/docs/" in "/" + fl or fl.startswith("doc"):
            s *= 0.3
        if s > 0:
            scored.append((s, f))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [f for _, f in scored[:k]], (scored[0][0] - scored[1][0] if len(scored) > 1 else 0.0)


res = []
for inst in rows:
    files = trees.get(inst["instance_id"], [])
    top, margin = localize(inst, files)
    gold = set(inst["gold_files"])
    res.append({"id": inst["instance_id"], "repo": inst["repo"], "top": top, "gold": sorted(gold), "margin": margin,
                "any": {k: bool(gold & set(top[:k])) for k in (1, 3, 5, 10)},
                "all": {k: gold <= set(top[:k]) for k in (1, 3, 5, 10)}})
json.dump(res, open("/home/claude/swebench/localize_results.json", "w"))
n = len(res)
print("SWE-bench Verified FILE LOCALISATION, substrate only (lexical routing over path tokens), %d instances:" % n)
print("  any gold file in top-k : " + "  ".join("k=%-2d %.3f" % (k, sum(r["any"][k] for r in res) / n) for k in (1, 3, 5, 10)))
print("  ALL gold files in top-k: " + "  ".join("k=%-2d %.3f" % (k, sum(r["all"][k] for r in res) / n) for k in (1, 3, 5, 10)))
if "--per-repo" in sys.argv:
    by = collections.defaultdict(list)
    for r in res:
        by[r["repo"]].append(r)
    for repo, rs in sorted(by.items(), key=lambda t: -len(t[1])):
        print("    %-26s n=%3d  top-1 %.3f  top-5 %.3f  top-10 %.3f" % (repo, len(rs), sum(r["any"][1] for r in rs) / len(rs), sum(r["any"][5] for r in rs) / len(rs), sum(r["any"][10] for r in rs) / len(rs)))
# the abstention view: answer top-1 only when the margin clears a floor
import numpy as np
mg = np.array([r["margin"] for r in res]); ok1 = np.array([r["any"][1] for r in res])
for q in (0.0, 5.0, 10.0, 15.0):
    m = mg >= q
    print("  margin >= %4.1f: answered %.3f at top-1 accuracy %.3f" % (q, m.mean(), ok1[m].mean() if m.any() else float("nan")))

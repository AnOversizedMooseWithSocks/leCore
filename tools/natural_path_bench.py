#!/usr/bin/env python3
"""tools/natural_path_bench.py -- how the decision surface does THROUGH THE DOORS AGENTS USE (sweep 176).

One table, re-runnable, numbers with their baselines:
  1. route() on ablated paraphrases -- act / choose / abstain at first contact; after the outcome is reported, a
     repeat and a NEW paraphrase answered from experience (the reflex behind the seen gate).
  2. typed() on Banking77 20/intent -- forced accuracy, what the substrate keeps at p >= 0.9 / 0.8 and at what
     accuracy, conformal coverage under the MARGINAL guarantee and the TRAINING-CONDITIONAL one, cost per decision.
  3. the acting loop (agent_benchmark) -- resolution and false actions, old gate vs the tiered operating point.
  4. the MCP front door -- off-catalog probes answered by lecore_find (must be 0), decide -> outcome round trip.

    PYTHONHASHSEED=0 python3 tools/natural_path_bench.py [--seeds 3] [--rows 300]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lecore  # noqa: E402
from holographic.caching_and_storage import holographic_catalog as C  # noqa: E402

SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 3
ROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 300


def bench_route():
    rows = []
    for seed in range(SEEDS):
        m = lecore.UnifiedMind(dim=256, seed=0)
        cat = m._capability_catalog()
        caps = [c for c in cat.all() if len(c.aliases) >= 4]
        rng = np.random.default_rng(seed)
        sample = [caps[i] for i in rng.permutation(len(caps))[:100]]
        held = {}
        for c in sample:                                   # ablate two aliases: the paraphrases the router never saw
            held[c.name] = (c.aliases[1], c.aliases[2])
            c.aliases = tuple(a for i, a in enumerate(c.aliases) if i not in (1, 2))
            c._hay = c._nw | set(C._tokens(c.does)) | C._alias_tokens(c.aliases)
            c._al = tuple(a.lower() for a in c.aliases)
        m._reflex_encs = None
        t = {"act": 0, "choose": 0, "abstain": 0}
        act_ok = menu_ok = 0
        for c in sample:
            r = m.route(held[c.name][0])
            t[r["decision"]] += 1
            if r["decision"] == "act":
                act_ok += r["skill"]["name"] == c.name
            if r["decision"] == "choose":
                menu_ok += any(o["name"] == c.name for o in r["options"])
            m.decision_outcome(r["id"], c.name)
        rep = rep_ok = par = par_ok = 0
        for c in sample:
            r = m.route(held[c.name][0]); rep += r.get("via") == "reflex"; rep_ok += r.get("via") == "reflex" and r["decision"] == "act" and r["skill"]["name"] == c.name
            r2 = m.route(held[c.name][1]); par += r2.get("via") == "reflex"; par_ok += r2.get("via") == "reflex" and r2["decision"] == "act" and r2["skill"]["name"] == c.name
        n = len(sample)
        rows.append((t["act"] / n, act_ok / max(t["act"], 1), t["choose"] / n, menu_ok / max(t["choose"], 1), t["abstain"] / n,
                     rep / n, rep_ok / max(rep, 1), par / n, par_ok / max(par, 1)))
    return np.array(rows).mean(0), np.array(rows).std(0)


def bench_typed():
    B = json.load(open(os.path.join(os.path.dirname(__file__), "_banking77_cache.json")))
    Cn = 77
    L = [str(i) for i in range(Cn)]
    by = {y: [t for t, yy in B["train"] if yy == y] for y in range(Cn)}
    rng = np.random.default_rng(0)
    perm = {y: rng.permutation(len(by[y])) for y in range(Cn)}
    ex = {L[y]: [by[y][i] for i in perm[y][:20]] for y in range(Cn)}
    lab = [[by[y][i], {"answer": L[y]}] for y in range(Cn) for i in perm[y][20:22]]
    stream = [B["eval"][i] for i in np.random.default_rng(7).permutation(len(B["eval"]))[:ROWS]]
    out = {}
    for conf in (None, 0.9):
        m = lecore.UnifiedMind(dim=256, seed=0)
        t0 = time.time()
        ok = 0; kept = {0.9: [0, 0], 0.8: [0, 0]}; cov = []; sz = []; state_findings = 0
        for t, y in stream:
            a = m.typed(t, L, examples=ex, labeled=lab, scorer="nb", conformal_alpha=0.05, conformal_confidence=conf)
            v = a["value"] if a["value"] is not None else a["ranked"][0][0]
            ok += v == L[y]; state_findings += bool(a["lint"])
            for f in kept:
                if a.get("p") is not None and a["p"] >= f:
                    kept[f][0] += 1; kept[f][1] += v == L[y]
            if a.get("set"):
                cov.append(L[y] in a["set"]); sz.append(len(a["set"]))
            m.decision_outcome(a["id"], L[y])
        n = len(stream)
        out[conf] = {"forced": ok / n, "ms": 1000 * (time.time() - t0) / n, "state_findings": state_findings / n,
                     "kept90": (kept[0.9][0] / n, kept[0.9][1] / max(kept[0.9][0], 1)), "kept80": (kept[0.8][0] / n, kept[0.8][1] / max(kept[0.8][0], 1)),
                     "coverage": float(np.mean(cov)), "set_size": float(np.mean(sz)), "size_le2": float(np.mean([s <= 2 for s in sz])), "n_cal": len(lab)}
    return out


def bench_loop():
    m = lecore.UnifiedMind(dim=256, seed=0)
    old = m.agent_benchmark(n_has=60, n_no=20, seed=0)
    new = m.agent_benchmark(n_has=60, n_no=20, seed=0, z_min=0.1)
    return old, new


def bench_mcp():
    import holographic_mcp as H
    s = H.MCPServer()
    s.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
    call = lambda n, a: json.loads(s.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": n, "arguments": a}})["result"]["content"][0]["text"])
    probes = ["purple monkey dishwasher", "asdf qwer zxcv", "blue elephant tuesday", "book me a table for two", "how do I boil an egg",
              "sing happy birthday", "who won the match last night", "translate this into french", "order more coffee pods", "the weather in lisbon"]
    t0 = time.time()
    answered = sum(call("lecore_find", {"query": q}).get("tier") == "answer" for q in probes)
    ms = 1000 * (time.time() - t0) / len(probes)
    d = call("lecore_decide", {"state": "courier lost the package", "options": ["billing", "shipping"]})
    o = call("lecore_outcome", {"id": d["id"], "outcome": "shipping"})
    return answered, len(probes), ms, d["value"], o.get("was_correct"), (o.get("reflex") or {}).get("learned")


def main():
    print("NATURAL-PATH BENCH -- %d seeds, %d Banking77 rows" % (SEEDS, ROWS))
    mu, sd = bench_route()
    print("1. route() on ablated paraphrases, first contact: act %.3f (right %.3f) | choose %.3f (menu holds it %.3f) | abstain %.3f" % tuple(mu[:5]))
    print("   after the outcome is reported: repeat via reflex %.3f (right %.3f) | NEW paraphrase via reflex %.3f (right %.3f)   [spread over seeds: act ±%.3f, repeat ±%.3f]" % (mu[5], mu[6], mu[7], mu[8], sd[0], sd[5]))
    ty = bench_typed()
    a, b = ty[None], ty[0.9]
    print("2. typed() Banking77 20/intent, p from %d labeled rows, outcomes reported: forced %.3f | %.0f ms/decision | state findings on %.2f of rows" % (a["n_cal"], a["forced"], a["ms"], a["state_findings"]))
    print("   substrate keeps p>=0.9: %.3f at %.3f | p>=0.8: %.3f at %.3f" % (a["kept90"] + a["kept80"]))
    print("   conformal 0.95 MARGINAL:              coverage %.3f | mean set %.1f | <=2 options %.3f" % (a["coverage"], a["set_size"], a["size_le2"]))
    print("   conformal 0.95 CONDITIONAL (conf 0.9): coverage %.3f | mean set %.1f | <=2 options %.3f" % (b["coverage"], b["set_size"], b["size_le2"]))
    old, new = bench_loop()
    print("3. acting loop: old gate (z 0.8) resolution %.2f, false actions %d/20 | tiered (z 0.1) resolution %.2f, false actions %d/20, menus %d" % (old["resolution_rate"], old["false_actions"], new["resolution_rate"], new["false_actions"], new.get("menus", 0)))
    ans, n, ms, v, wc, learned = bench_mcp()
    print("4. MCP front door: off-catalog probes answered %d/%d | lecore_find %.0f ms | decide -> outcome: %s, was_correct %s, reflex learned %s" % (ans, n, ms, v, wc, learned))


if __name__ == "__main__":
    main()

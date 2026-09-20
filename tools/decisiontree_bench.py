"""decisiontree_bench.py -- reproduce every sweep-173 number from the LIVE catalog, negatives included.

The ground truth needs no dataset: every catalog capability carries aliases (mean 9 per card), i.e.
different phrasings that SHOULD reach the same place. The honest split holds one alias out and ABLATES
it from the router's index on both sides, so nothing is a lookup. The earlier un-ablated "baseline" of
0.987 was exactly that lookup (aliases are tokenised into the router's haystack, catalog line ~116).

Experiments (3 seeds, spread reported):
  1. ROUTER GENERALISATION  top-1 / top-3 / top-8 on held-out aliases -- the ceilings everything sits under.
  2. OUTCOME MEMORY AS A ROUTER, closed world  -- and the one-line FILTER that matches it (the strawman).
  3. OPEN WORLD  -- half the queries target capabilities the store never saw. The router wins outright.
  4. EQUIVALENCE (AUROC) -- the one job the fingerprint does better than every baseline.

Run:  PYTHONHASHSEED=0 python3 tools/decisiontree_bench.py [--seeds 3] [--n 150]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIM = 1024


def auroc(pos, neg):
    """Probability a random SAME-result pair scores above a random DIFFERENT-result pair."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def split(seed, n_seen, n_unseen, ablate):
    """A fresh catalog with the held-out alias(es) removed from each sampled card's index."""
    from holographic.caching_and_storage import holographic_catalog as C
    cat = C.default_catalog()
    caps = [c for c in cat.all() if len(getattr(c, "aliases", None) or []) >= 3]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(caps))
    seen = [caps[i] for i in idx[:n_seen]]
    unseen = [caps[i] for i in idx[n_seen:n_seen + n_unseen]]
    held = {}
    for c in seen + unseen:
        held[c.name] = tuple(c.aliases[i] for i in ablate)
        c.aliases = tuple(a for i, a in enumerate(c.aliases) if i not in ablate)
        c._hay = c._nw | set(C._tokens(c.does)) | C._alias_tokens(c.aliases)
        c._al = tuple(a.lower() for a in c.aliases)
    return cat, seen, unseen, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    from holographic.agents_and_reasoning.holographic_decisiontree import (
        OutcomeMemory, routing_fingerprint)
    from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode
    from holographic.mesh_and_geometry.holographic_planshape import cosine
    from holographic.caching_and_storage import holographic_catalog as C
    seeds = range(args.seeds)

    def fmt(v):
        v = np.asarray(v)
        return "%.3f (spread %.3f)" % (v.mean(), v.max() - v.min())

    # ---- 1 + 2: closed world -------------------------------------------------------------
    r1 = {"top1": [], "top3": [], "top8": [], "filter": [], "memory": [], "bag": []}
    for s in seeds:
        cat, seen, _, held = split(s, args.n, 0, ablate=(1,))
        known = {c.name for c in seen}
        fp = routing_fingerprint(cat, dim=DIM, seed=0, k=8)
        om = OutcomeMemory(dim=DIM, seed=0, encoder=fp)
        ob = OutcomeMemory(dim=DIM, seed=0, encoder=hashed_ngram_encode(dim=DIM))
        for c in seen:
            om.record(c.aliases[0], c.name, None)
            ob.record(c.aliases[0], c.name, None)
        t1 = t3 = t8 = fi = me = ba = 0
        for c in seen:
            q = held[c.name][0]
            names = [cp.name for cp, _ in cat.find_scored(q, k=874)]
            t1 += bool(names[:1] == [c.name]); t3 += c.name in names[:3]; t8 += c.name in names[:8]
            f = [nm for nm in names if nm in known]
            fi += bool(f and f[0] == c.name)                       # the one-line filter
            rk = om.recall(q, None, margin=0.02)["ranked"]; me += bool(rk and rk[0][0] == c.name)
            rb = ob.recall(q, None, margin=0.02)["ranked"]; ba += bool(rb and rb[0][0] == c.name)
        n = len(seen)
        for k, v in zip(r1, (t1, t3, t8, fi, me, ba)):
            r1[k].append(v / n)
    print("1) ROUTER GENERALISATION on held-out aliases (ablated), %d seeds x %d:" % (args.seeds, args.n))
    print("   top-1 (deployed BASELINE) %s | top-3 recall %s | top-8 recall %s"
          % (fmt(r1["top1"]), fmt(r1["top3"]), fmt(r1["top8"])))
    print("2) OUTCOME MEMORY AS A ROUTER, closed world (answer always in the store):")
    print("   memory (fingerprint) %s | bag encoder %s" % (fmt(r1["memory"]), fmt(r1["bag"])))
    print("   STRAWMAN: router filtered to seen capabilities %s  <- no hypervectors, same result"
          % fmt(r1["filter"]))

    # ---- 3: open world -------------------------------------------------------------------
    r3 = {"router": [], "filter": [], "memory": [], "abstained_unseen": []}
    for s in seeds:
        cat, seen, unseen, held = split(s, args.n, args.n, ablate=(1,))
        known = {c.name for c in seen}
        om = OutcomeMemory(dim=DIM, seed=0, encoder=routing_fingerprint(cat, dim=DIM, seed=0, k=8))
        for c in seen:
            om.record(c.aliases[0], c.name, None)
        ro = fi = me = ab = 0
        for c in seen + unseen:
            q = held[c.name][0]
            names = [cp.name for cp, _ in cat.find_scored(q, k=874)]
            ro += bool(names[:1] == [c.name])
            f = [nm for nm in names if nm in known]; fi += bool(f and f[0] == c.name)
            r = om.recall(q, None, margin=0.02, min_score=0.35)
            me += (r["result"] == c.name)
            if c in unseen and r["result"] is None:
                ab += 1
        N = len(seen) + len(unseen)
        r3["router"].append(ro / N); r3["filter"].append(fi / N); r3["memory"].append(me / N)
        r3["abstained_unseen"].append(ab / len(unseen))
    print("3) OPEN WORLD, 50% of queries target never-observed capabilities:")
    print("   router %s | filter %s (wrong BY CONSTRUCTION on unseen) | memory %s"
          % (fmt(r3["router"]), fmt(r3["filter"]), fmt(r3["memory"])))
    print("   memory abstained on unseen queries %s  <- its one virtue" % fmt(r3["abstained_unseen"]))

    # ---- 4: equivalence ------------------------------------------------------------------
    r4 = {"fingerprint": [], "bag": [], "jaccard": []}
    for s in seeds:
        cat, seen, _, held = split(s, args.n, 0, ablate=(0, 1))
        fp = routing_fingerprint(cat, dim=DIM, seed=0, k=8)
        bag = hashed_ngram_encode(dim=DIM)
        A = [held[c.name][0] for c in seen]; B = [held[c.name][1] for c in seen]; n = len(seen)
        for name, enc in (("fingerprint", fp), ("bag", bag)):
            ea = [enc(t) for t in A]; eb = [enc(t) for t in B]
            pos = [float(cosine(ea[i], eb[i])) for i in range(n)]
            neg = [float(cosine(ea[i], eb[(i + j) % n])) for i in range(n) for j in (1, 7, 31)]
            r4[name].append(auroc(pos, neg))
        tok = lambda t: set(C._tokens(t))
        jac = lambda a, b: len(tok(a) & tok(b)) / len(tok(a) | tok(b)) if (tok(a) | tok(b)) else 0.0
        pos = [jac(A[i], B[i]) for i in range(n)]
        neg = [jac(A[i], B[(i + j) % n]) for i in range(n) for j in (1, 7, 31)]
        r4["jaccard"].append(auroc(pos, neg))
    print("4) EQUIVALENCE -- does input similarity tell same-result pairs from different-result pairs? (AUROC)")
    print("   routing fingerprint %s | bag encoder %s | raw word overlap %s"
          % (fmt(r4["fingerprint"]), fmt(r4["bag"]), fmt(r4["jaccard"])))


def gate_table(seeds=3, n=150):
    """5) THE GATE TABLE (sweep 176, backlog A2): what route_or_abstain's z_min accepts, and what the tiered
    router does with the same rows -- re-run every sweep so drift in the setpoint shows. Ground truth is the
    ablated alias split; gibberish is the 7 logged off-catalog probes."""
    import numpy as np
    from collections import Counter
    from holographic.caching_and_storage import holographic_catalog as C
    gib = ["purple monkey dishwasher", "counter traders", "make the thing do the stuff with the wobbly bits",
           "asdf qwer zxcv", "blue elephant tuesday", "hello there how are you today", "what is the meaning of life"]
    zs, ok, tiers, good = [], [], Counter(), 0
    for s in range(seeds):
        cat, seen, _, held = split(s, n, 0, ablate=(1,))
        for c in seen:
            q = held[c.name][0]
            r = cat.route_or_abstain(q, z_min=-1e9)
            top = r["hits"][0] if r["hits"] else None                      # hits are (capability, score) tuples
            top = top[0] if isinstance(top, tuple) else top
            zs.append(r["z"]); ok.append(bool(top is not None and top.name == c.name))
            t = cat.route_tiered(q, k=5); tiers[t["tier"]] += 1
            good += (t["answer"] == c.name) if t["tier"] == "answer" else (t["tier"] != "refuse" and c.name in [o["name"] for o in t["options"]])
        gz = [cat.route_or_abstain(g, z_min=-1e9)["z"] for g in gib]
    zs, ok = np.array(zs), np.array(ok)
    print("5) GATE TABLE -- route_or_abstain z_min vs real paraphrases (%d) and gibberish (7):" % len(zs))
    for zmin in (0.8, 0.5, 0.1, 0.0, -0.2, -0.5):
        a = zs >= zmin
        print("   z_min %5.2f: accepts real %5.1f%% | gibberish %d/7 | top-1 correct among accepted %.3f"
              % (zmin, 100 * a.mean(), sum(1 for z in gz if z >= zmin), ok[a].mean() if a.any() else float("nan")))
    N = len(zs)
    print("   tiered (k=5): answer %.3f menu %.3f clarify %.3f refuse %.3f | right card answered-or-in-options %.3f"
          % (tiers["answer"] / N, tiers["menu"] / N, tiers["clarify"] / N, tiers["refuse"] / N, good / N))


if __name__ == "__main__":
    main()
    gate_table()

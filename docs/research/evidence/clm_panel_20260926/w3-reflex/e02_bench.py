"""E0.2 -- catalog routing baseline from free pairs (alias -> its card), scratch only, read-only.

Protocol (swarm brief, w3-reflex):
  * aliases come from the LIVE skills catalog (holographic_skills._catalog()), each card's .aliases;
  * held-out half = the guard's insertion-stable hash rule: int(sha256(alias)[:2], 16) % 2 == 1;
  * queries go through an IN-PROCESS UnifiedMind(dim=256, seed=0) -- its own catalog (the one the service's
    find_capability/route_tiered use, 3,952 cards) -- never the shared partition;
  * two arms:
      in-index  -- aliases left in the haystack (the LOOKUP: exact-alias +5 bonus; an upper bound, not a result)
      ablated   -- every held-out alias string removed from EVERY card's alias list first (the honest baseline
                   E4.1 must beat: the train half stays, the test half is unseen);
  * per query: rank of the best true card in the full _score_all ordering (the one scorer find_capability,
    find_scored and route_tiered all share), route_tiered's tier and whether its answer / menu holds the truth;
  * latency: _score_all alone, mind.find_capability (memo miss, as deployed), mind.route_tiered, per query.
The find_capability memo is redirected to this worker's scratch tmp so the service's /tmp memo is untouched.
"""
import hashlib
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
tempfile.tempdir = os.path.join(HERE, "tmp")          # isolate the fc memo from the service's /tmp file
os.makedirs(tempfile.tempdir, exist_ok=True)
for f in os.listdir(tempfile.tempdir):                # a COLD memo every run: latency means first-call cost
    if f.startswith("lecore_fc_"):
        os.remove(os.path.join(tempfile.tempdir, f))
sys.path.insert(0, "/home/claude/leCore")

import numpy as np


def held_out(alias):
    return int(hashlib.sha256(alias.encode()).hexdigest()[:2], 16) % 2 == 1


def rebake(cat):
    """Forget the catalog's baked haystacks, memo and null cache so the next call re-bakes from the aliases."""
    for attr in ("_fc_baked", "_fc_vocab", "_fc_hash", "_fc_memo", "_fc_memo_path", "_null_floor_cache"):
        if hasattr(cat, attr):
            delattr(cat, attr)


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, float), q)) if xs else None


def run_arm(mind, cat, queries, truth, fams, label, do_route=True, time_it=True):
    from holographic.caching_and_storage.holographic_catalog import _tokens
    cat._ensure_fc_baked()
    ranks, t_score, t_find, t_route = [], [], [], []
    tiers = {"answer": 0, "menu": 0, "clarify": 0, "refuse": 0}
    ans_right = menu_hit = 0
    top1_wrong_same_family = top1_wrong_n = 0
    mismatch_find = 0
    for i, q in enumerate(queries):
        tq = set(_tokens(q))
        t0 = time.perf_counter()
        scored = cat._score_all(q, tq) if tq else []
        t_score.append(time.perf_counter() - t0)
        names = [n for _, n, _ in scored]
        tr = truth[q]
        r = next((j + 1 for j, n in enumerate(names) if n in tr), None)
        ranks.append(r)
        if names and names[0] not in tr:
            top1_wrong_n += 1
            tf = {fams.get(n, (None, None))[0] for n in tr} - {None}
            if fams.get(names[0], (None, None))[0] in tf:
                top1_wrong_same_family += 1
        if time_it:
            t0 = time.perf_counter()
            fc = mind.find_capability(q, k=5)
            t_find.append(time.perf_counter() - t0)
            if [c.name for c in fc] != names[:5]:
                mismatch_find += 1
        if do_route:
            t0 = time.perf_counter()
            rt = mind.route_tiered(q, k=5)
            t_route.append(time.perf_counter() - t0)
            tiers[rt["tier"]] = tiers.get(rt["tier"], 0) + 1
            if rt["tier"] == "answer" and rt["answer"] in tr:
                ans_right += 1
            if rt["tier"] in ("menu", "clarify") and any(o["name"] in tr for o in rt["options"]):
                menu_hit += 1
    n = len(queries)
    rr = [r if r is not None else 10 ** 9 for r in ranks]
    out = {
        "arm": label, "n_queries": n, "catalog_cards": len(cat),
        "top1": sum(1 for r in ranks if r == 1) / n,
        "top5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "top8": sum(1 for r in ranks if r is not None and r <= 8) / n,
        "median_rank": float(np.median(rr)),
        "unranked_zero_overlap": sum(1 for r in ranks if r is None) / n,
        "top1_wrong_same_family_share": (top1_wrong_same_family / top1_wrong_n) if top1_wrong_n else None,
        "find_capability_top5_differs_from_score_all": mismatch_find,
        "latency_ms": {
            "score_all": {"mean": 1e3 * float(np.mean(t_score)), "p50": 1e3 * pct(t_score, 50), "p95": 1e3 * pct(t_score, 95)},
        },
    }
    if t_find:
        out["latency_ms"]["find_capability"] = {"mean": 1e3 * float(np.mean(t_find)), "p50": 1e3 * pct(t_find, 50),
                                                "p95": 1e3 * pct(t_find, 95), "first": 1e3 * t_find[0]}
    if do_route:
        out["route_tiered"] = {
            "tiers": {k: v / n for k, v in tiers.items()},
            "answer_accuracy": (ans_right / tiers["answer"]) if tiers["answer"] else None,
            "answered_right_of_all": ans_right / n,
            "menu_holds_truth": (menu_hit / (tiers["menu"] + tiers["clarify"])) if (tiers["menu"] + tiers["clarify"]) else None,
        }
        out["latency_ms"]["route_tiered"] = {"mean": 1e3 * float(np.mean(t_route)), "p50": 1e3 * pct(t_route, 50),
                                             "p95": 1e3 * pct(t_route, 95), "max": 1e3 * max(t_route)}
    return out


def main():
    T0 = time.time()
    from holographic.misc.holographic_skills import _catalog
    import lecore
    skills = _catalog()
    # snapshot BEFORE any ablation (objects may be shared between catalogs)
    card_aliases = {c.name: tuple(c.aliases or ()) for c in skills.all()}
    uniq = sorted({a for al in card_aliases.values() for a in al})
    held = [a for a in uniq if held_out(a)]
    mind = lecore.UnifiedMind(dim=256, seed=0)
    cat = mind._capability_catalog()
    mind_aliases = {c.name: tuple(c.aliases or ()) for c in cat.all()}
    # truth = every card of the SCORED catalog that carries the alias (78 skills aliases sit on >1 card)
    truth = {}
    for name, al in mind_aliases.items():
        for a in al:
            truth.setdefault(a, set()).add(name)
    queries = [a for a in held if a in truth]
    fams = cat.families()
    info = {"skills_cards": len(skills), "mind_cards": len(cat), "unique_skills_aliases": len(uniq),
            "held_out": len(held), "held_out_multi_card": sum(1 for a in queries if len(truth[a]) > 1),
            "family_resolved_cards": sum(1 for v in fams.values() if v[0])}
    print(json.dumps(info), flush=True)
    res = {"info": info}

    # arm 1: in-index (the lookup) -- ranks only (route on a 1-in-8 sample to keep CPU down)
    res["in_index"] = run_arm(mind, cat, queries, truth, fams, "in-index", do_route=False, time_it=False)
    print(json.dumps(res["in_index"]), flush=True)

    # arm 2: ablated -- the held-out alias strings leave EVERY card; the train half stays
    hs = set(held)
    for c in cat.all():
        c.aliases = tuple(a for a in (c.aliases or ()) if a not in hs)
    rebake(cat)
    t0 = time.perf_counter()
    cat._ensure_fc_baked()
    bake_ms = 1e3 * (time.perf_counter() - t0)
    res["ablated"] = run_arm(mind, cat, queries, truth, fams, "ablated", do_route=True, time_it=True)
    res["ablated"]["latency_ms"]["bake_once"] = bake_ms
    print(json.dumps(res["ablated"]), flush=True)
    res["cpu_s"] = time.time() - T0
    with open(os.path.join(HERE, "e02_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("total s", round(res["cpu_s"], 1))


if __name__ == "__main__":
    main()

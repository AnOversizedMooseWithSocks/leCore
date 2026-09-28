"""E0.2 attribution (scratch): (a) where route_tiered's 16 ms goes; (b) is the answer-tier drop (0.833 -> 0.604) from
CATALOG SIZE or from ABLATION DENSITY -- rerun the same hash-held-out protocol on default_catalog() (893 cards, the
catalog route_tiered's thresholds were set on); (c) the same-family share of top-1 errors when BOTH families resolve."""
import hashlib
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
tempfile.tempdir = os.path.join(HERE, "tmp")
sys.path.insert(0, "/home/claude/leCore")
import numpy as np


def held_out(a):
    return int(hashlib.sha256(a.encode()).hexdigest()[:2], 16) % 2 == 1


def rebake(cat):
    for attr in ("_fc_baked", "_fc_vocab", "_fc_hash", "_fc_memo", "_fc_memo_path", "_null_floor_cache"):
        if hasattr(cat, attr):
            delattr(cat, attr)


def arm(cat, queries, truth, label):
    from holographic.caching_and_storage.holographic_catalog import _tokens
    fams = cat.families()
    n = len(queries)
    t1 = t5 = 0
    tiers = {}
    ans_ok = menu_ok = 0
    both = same = 0
    t_fam, t_null, t_find = [], [], []
    for q in queries:
        tr = truth[q]
        names = [nm for _, nm, _ in cat._score_all(q, set(_tokens(q)))]
        t1 += bool(names[:1] and names[0] in tr)
        t5 += any(x in tr for x in names[:5])
        if names and names[0] not in tr:
            wf = fams.get(names[0], (None, None))[0]
            tf = {fams.get(x, (None, None))[0] for x in tr} - {None}
            if wf is not None and tf:
                both += 1
                same += wf in tf
        r = cat.route_tiered(q, k=5)
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
        ans_ok += r["tier"] == "answer" and r["answer"] in tr
        menu_ok += r["tier"] == "menu" and any(o["name"] in tr for o in r["options"])
    # latency breakdown on the first 300 queries (warm nulls)
    for q in queries[:300]:
        t = time.perf_counter(); cat.families(); t_fam.append(time.perf_counter() - t)
        t = time.perf_counter(); cat.route_or_abstain(q, k=8, z_min=-1e9); t_null.append(time.perf_counter() - t)
        t = time.perf_counter(); cat.find_scored(q, k=8); t_find.append(time.perf_counter() - t)
    return {"arm": label, "cards": len(cat), "n": n, "top1": t1 / n, "top5": t5 / n,
            "tiers": {k: v / n for k, v in tiers.items()},
            "answer_accuracy": ans_ok / max(tiers.get("answer", 0), 1),
            "menu_holds_truth": menu_ok / max(tiers.get("menu", 0), 1),
            "top1_errors_both_families_resolved": both, "same_family_share_of_those": same / max(both, 1),
            "ms_per_call": {"families()": 1e3 * float(np.mean(t_fam)), "route_or_abstain(warm)": 1e3 * float(np.mean(t_null)),
                            "find_scored": 1e3 * float(np.mean(t_find))}}


def main():
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    from holographic.misc.holographic_skills import _catalog
    import lecore
    uniq = sorted({a for c in _catalog().all() for a in (c.aliases or ())})
    held = {a for a in uniq if held_out(a)}
    out = {}
    for label, cat in (("default_catalog (893)", default_catalog()),
                       ("mind catalog (3952)", lecore.UnifiedMind(dim=256, seed=0)._capability_catalog())):
        truth = {}
        for c in cat.all():
            for a in (c.aliases or ()):
                truth.setdefault(a, set()).add(c.name)
        queries = sorted(a for a in held if a in truth)
        for c in cat.all():
            c.aliases = tuple(a for a in (c.aliases or ()) if a not in held)
        rebake(cat)
        out[label] = arm(cat, queries, truth, label)
        print(json.dumps(out[label]), flush=True)
    json.dump(out, open(os.path.join(HERE, "e02_attrib.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

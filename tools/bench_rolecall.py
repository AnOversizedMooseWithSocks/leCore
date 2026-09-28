"""tools/bench_rolecall.py -- measure the phase-E holographic candidates (CLM backlog E3.1, E3.2, E5.1) with baselines.

    PYTHONHASHSEED=0 python3 tools/bench_rolecall.py                      # everything -> docs/research/evidence/bench_rolecall.json
    PYTHONHASHSEED=0 python3 tools/bench_rolecall.py --only direction,compose --out /tmp/x.json

Sections (each reports its baseline next to its number, and its kept negatives):
  direction   E3.1 -- the context-role direction learner vs the POSITIONAL rule (first currency named = FROM) on the
              hand-labelled tests/data/exchange_direction.json: the fixed hash split (train -> test), 5-fold CV over
              all items, per-source counts, the discordant pairs, and the ablations (mask, window, rule).
  decode      E3.2 -- the panel's synthetic sum-form replication (20 verbs, 50 values, FROM/TO; D=1024/2048): clean,
              noise = one term's norm, EQUAL 2-call blends (the chimera case), DOMINANT blends (0.7 / 0.3); per-role
              argmax (the baseline) vs compose() with the chimera check.
  compose     E3.2 acceptance -- REAL wording: 5-fold CV over the exchange set; for each held-out question the reader
              is trained on the other folds, the question's soft state is built and composed; right / verdict /
              p_model / cross-fitted calibrated p (isotonic, DoorCalibrator) and ECE on the calls NEVER SEEN WHOLE in
              the training folds; plus the refuse-a-direction probe on the SYMMETRIC sentences ('between X and Y').
  resonator   the product form with permutation roles (20 x 50 x 50, D=2048): ResonatorNetwork exact exit (before)
              vs the noise-tolerant exit (after) at 0 / 5 / 10 % flipped components; refusal latency; p-values.
  trajectory  E5.1 -- sum_t rho^t(step_t): cosine distributions of SAME-plan re-encodings (the noise floor: every
              step's incidental text changed) vs adjacent swaps, random swaps, one inserted step (anywhere, and at
              the END), one deleted step, one changed argument; detection at the floor's 1% false-alarm point;
              decode-and-diff accuracy.

numpy + stdlib; deterministic (seeded generators, hashlib-derived atoms). No external data needed.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic.agents_and_reasoning.holographic_rolecall import (  # noqa: E402
    ContextRoles, ProductCodec, RoleCodec, amounts_in, call_string, encode_step, expected_calibration_error,
    find_span, naive_decode, positional_read, question_state, question_tokens, trajectory_compare,
    trajectory_decode, trajectory_diff, trajectory_encode)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data", "exchange_direction.json")
SLOTS = ["from", "to"]
SKIPPED = []            # the data file's skipped rows (set by main; the symmetric ones feed the no-direction arm)


# ================================================================================================ helpers
def load_items(path=DATA):
    d = json.load(open(path))
    return d["items"], d["skipped"]


def mentions_of(it):
    """[(words, token_index)] of every currency mention in an item, in question order (FROM, TO, extra TOs)."""
    toks = question_tokens(it["text"])
    fa, ta = it["from_at"], it["to_at"]
    ms = [(it["from_words"], fa), (it["to_words"], ta)]
    taken = [(fa, fa + len(question_tokens(it["from_words"]))), (ta, ta + len(question_tokens(it["to_words"])))]
    for w in it.get("to_also_words", []):
        sp = find_span(toks, w, taken)
        taken.append(sp)
        ms.append((w, sp[0]))
    return sorted(ms, key=lambda m: m[1])


def value_of(it):
    """{mention words: ISO code} for an item's mentions."""
    v = {it["from_words"]: it["from"], it["to_words"]: it["to"]}
    for w, c in zip(it.get("to_also_words", []), it.get("to_also", [])):
        v[w] = c
    return v


def direction_right(it, asg):
    """A reading is right when FROM is the labelled FROM words and TO is one of the labelled TO words."""
    return asg["from"] == it["from_words"] and asg["to"] in [it["to_words"]] + it.get("to_also_words", [])


def teach(reader, items):
    for it in items:
        reader.learn(it["text"], {"from": (it["from_words"], it["from_at"]), "to": [(it["to_words"], it["to_at"])]
                                  + [(w, a) for w, a in zip(it.get("to_also_words", []),
                                                            [m[1] for m in mentions_of(it)
                                                             if m[0] in it.get("to_also_words", [])])]})
    return reader


def teach_either(reader, skipped, split="train"):
    """Teach the NO-DIRECTION verdicts ({'either': [w1, w2]}) of the symmetric sentences in one split."""
    for s in skipped:
        if s.get("words") and s.get("split") == split:
            reader.learn(s["text"], {"either": list(s["words"])})
    return reader


def folds_of(items, k=5):
    """Deterministic k-fold: by the item's position in the split-sorted id order (the hash split already
    shuffles by sha256), fold = rank % k."""
    order = sorted(range(len(items)), key=lambda i: (items[i]["split"], items[i]["id"]))
    f = [0] * len(items)
    for rank, i in enumerate(order):
        f[i] = rank % k
    return f


def summary(xs):
    xs = np.asarray(xs, float)
    if not len(xs):
        return None
    return {"n": int(len(xs)), "mean": round(float(xs.mean()), 4), "sd": round(float(xs.std()), 4),
            "p01": round(float(np.quantile(xs, 0.01)), 4), "p50": round(float(np.quantile(xs, 0.5)), 4),
            "p99": round(float(np.quantile(xs, 0.99)), 4), "min": round(float(xs.min()), 4),
            "max": round(float(xs.max()), 4)}


def auroc(neg, pos):
    """P(a random 'pos' score > a random 'neg' score), ties 0.5 -- by ranks (no sklearn)."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    allv = np.concatenate([neg, pos])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                     # average ties
        sel = allv == v
        ranks[sel] = ranks[sel].mean()
    rp = ranks[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ================================================================================================ E3.1 direction
def bench_direction(items):
    train = [it for it in items if it["split"] == "train"]
    test = [it for it in items if it["split"] == "test"]
    out = {"n_items": len(items), "n_train": len(train), "n_test": len(test)}

    def run(cfg, tr, te):
        cr = teach(ContextRoles(**cfg), tr)
        rows = []
        for it in te:
            ms = mentions_of(it)
            r = cr.read(it["text"], ms, SLOTS)
            rows.append({"id": it["id"], "text": it["text"], "source": it["source"],
                         "ctx": direction_right(it, r["assignment"]),
                         "pos": direction_right(it, positional_read(ms, SLOTS)),
                         "margin": r["margin"], "p": r["p"], "truth": "%s->%s" % (it["from"], it["to"]),
                         "read": "%s->%s" % (value_of(it)[r["assignment"]["from"]], value_of(it)[r["assignment"]["to"]])})
        return rows

    rows = run({}, train, test)
    ctx, pos = sum(r["ctx"] for r in rows), sum(r["pos"] for r in rows)
    out["split"] = {
        "context_role": "%d/%d" % (ctx, len(rows)), "positional": "%d/%d" % (pos, len(rows)),
        "discordant": {"context_right_positional_wrong": sum(r["ctx"] and not r["pos"] for r in rows),
                       "positional_right_context_wrong": sum(r["pos"] and not r["ctx"] for r in rows)},
        "by_source": {src: {"context_role": "%d/%d" % (sum(r["ctx"] for r in rows if r["source"] == src),
                                                       sum(1 for r in rows if r["source"] == src)),
                            "positional": "%d/%d" % (sum(r["pos"] for r in rows if r["source"] == src),
                                                     sum(1 for r in rows if r["source"] == src))}
                      for src in ("clinc150", "handwritten")},
        "positional_wrong_items_in_test": sum(1 for r in rows if not r["pos"]),
        "context_errors": [{k: r[k] for k in ("id", "text", "truth", "read", "margin")} for r in rows if not r["ctx"]],
        "train_fit": "%d/%d" % (sum(r["ctx"] for r in run({}, train, train)), len(train))}
    # the same held-out split, the panel's configuration against its variations (ablations, not tuning: the
    # default was fixed to the panel's before this ran)
    abl = {}
    for name, cfg in [("panel_default L1+L2+R1 masked centroid", {}),
                      ("no mention mask", {"mask_mentions": False}),
                      ("L1+R1", {"window": ("L1", "R1")}),
                      ("L1+L2+R1+R2", {"window": ("L1", "L2", "R1", "R2")}),
                      ("rule=infonce (ProtoStore)", {"rule": "infonce"})]:
        rr = run(cfg, train, test)
        abl[name] = "%d/%d" % (sum(r["ctx"] for r in rr), len(rr))
    out["ablations_on_split"] = abl
    # with the no-direction prototype taught from the symmetric TRAIN sentences: does direction accuracy hold, and
    # are held-out symmetric sentences recognised as having no direction?
    sym_te = [x for x in SKIPPED if x.get("words") and x.get("split") == "test"]
    for arm, use in (("without_either (kept negative)", False), ("with_either", True)):
        cr = teach(ContextRoles(), train)
        if use:
            teach_either(cr, SKIPPED)
        d_ok = d_nodir = 0
        for it in test:
            r = cr.read(it["text"], mentions_of(it), SLOTS)
            d_ok += r["direction"] and direction_right(it, r["assignment"])
            d_nodir += not r["direction"]
        s_nodir = sum(not cr.read(x["text"], x["words"], SLOTS)["direction"] for x in sym_te)
        out.setdefault("no_direction", {})[arm] = {
            "directional_test_right": "%d/%d" % (d_ok, len(test)),
            "directional_test_called_no_direction": d_nodir,
            "symmetric_test_called_no_direction": "%d/%d" % (s_nodir, len(sym_te)),
            "symmetric_taught": sum(1 for x in SKIPPED if x.get("words") and x.get("split") == "train") if use else 0}
    # 5-fold CV over ALL items: every item held out once
    f = folds_of(items)
    cv_ctx = cv_pos = 0
    for k in range(5):
        rr = run({}, [it for i, it in enumerate(items) if f[i] != k], [it for i, it in enumerate(items) if f[i] == k])
        cv_ctx += sum(r["ctx"] for r in rr)
        cv_pos += sum(r["pos"] for r in rr)
    out["cv5_all_items"] = {"context_role": "%d/%d" % (cv_ctx, len(items)), "positional": "%d/%d" % (cv_pos, len(items))}
    # learning curve on the split: how many taught verdicts before the learner beats positional
    curve = []
    for n in (4, 8, 16, 32, 64, len(train)):
        rr = run({}, train[:n], test)
        curve.append({"taught": n, "context_role": sum(r["ctx"] for r in rr), "positional": pos, "of": len(rr)})
    out["learning_curve"] = curve
    return out


# ================================================================================================ E3.2 synthetic
def bench_decode(trials=500):
    out = []
    for D in (2048, 1024):
        codec = RoleCodec(dim=D, seed=0)
        verbs = ["verb%02d" % i for i in range(20)]
        vals = ["val%02d" % i for i in range(50)]
        sv = {"from": vals, "to": vals}
        rng = np.random.default_rng(2000)

        def call():
            v, x, y = int(rng.integers(20)), int(rng.integers(50)), int(rng.integers(50))
            while y == x:
                y = int(rng.integers(50))
            return verbs[v], {"from": vals[x], "to": vals[y]}

        res = {"D": D, "trials": trials}
        for cond in ("clean", "noise_1x_term", "equal_blend", "dominant_0.7_0.3"):
            naive_ok = naive_chim = ck_ok = ck_flag = ck_chim_slip = 0
            ms = []
            for t in range(trials):
                v1, a1 = call()
                s = codec.encode_call(v1, a1)
                c1 = call_string(v1, a1)
                c2 = None
                if cond == "noise_1x_term":
                    # the panel's sigma=1 arm: per-component sd sqrt(K/D) for K=3 unit terms, i.e. a noise vector
                    # as long as the whole state (exp_a_resonator.py: noise_sigma * sqrt(3) * N(0,1) on +/-1 terms)
                    s = s + math.sqrt(3.0 / D) * rng.standard_normal(D)
                if cond in ("equal_blend", "dominant_0.7_0.3"):
                    while True:
                        v2, a2 = call()
                        if v2 != v1 and len({a1["from"], a1["to"], a2["from"], a2["to"]}) == 4:
                            break
                    c2 = call_string(v2, a2)
                    w1, w2 = (1.0, 1.0) if cond == "equal_blend" else (0.7, 0.3)
                    s = w1 * s + codec.encode_call(v2, a2, weight=w2)
                nd = naive_decode(codec, s, verbs, sv)
                ncall = call_string(nd["verb"], nd["args"])
                t0 = time.perf_counter()
                got = codec.compose(s, verbs, sv)
                ms.append(1000 * (time.perf_counter() - t0))
                ok_set = {c1} if cond != "equal_blend" else {c1, c2}
                parts = [(v1, a1), (v2, a2)] if c2 else [(v1, a1)]

                def chimera(call_s, dec):
                    if c2 is None or call_s in (c1, c2):
                        return False
                    return (dec["verb"] in (parts[0][0], parts[1][0]) and
                            all(dec["args"].get(r) in (parts[0][1][r], parts[1][1][r]) for r in ("from", "to")))

                naive_ok += ncall in ok_set
                naive_chim += chimera(ncall, nd)
                flagged = got["verdict"] in ("ambiguous", "empty")
                ck_flag += flagged
                ck_ok += (not flagged) and got["call"] in ok_set
                ck_chim_slip += (not flagged) and chimera(got["call"], got)
            res[cond] = {"naive_right": naive_ok / trials, "naive_chimera": naive_chim / trials,
                         "checked_right_and_served": ck_ok / trials, "checked_flagged": ck_flag / trials,
                         "checked_chimera_served": ck_chim_slip / trials,
                         "compose_ms_mean": round(float(np.mean(ms)), 3)}
        out.append(res)
    return {"by_dim": out, "setup": "20 verbs, 50 values shared by FROM/TO, HRR sum with unitary roles; "
                                    "baseline = per-role argmax (naive_decode); checked = compose() verdict"}


# ================================================================================================ E3.2 real wording
def bench_compose(items, skipped, z_margin=3.0, with_either=True):
    """5-fold CV: the reader never saw the held-out question; the call's (verb, from, to, amount) whole may or may
    not have occurred in the training folds -- the acceptance counts the ones that did NOT."""
    from holographic.agents_and_reasoning.holographic_protostore import DoorCalibrator
    codec = RoleCodec(dim=2048, seed=0)
    f = folds_of(items)

    def truth_call(it):
        a = {"from": it["from"], "to": it["to"]}
        if it["amount"] is not None:
            a["amount"] = it["amount"]
        return call_string("fx", a)

    rows = []
    for k in range(5):
        tr = [it for i, it in enumerate(items) if f[i] != k]
        te = [it for i, it in enumerate(items) if f[i] == k]
        reader = teach_either(teach(ContextRoles(), tr), skipped) if with_either else teach(ContextRoles(), tr)
        seen_whole = {truth_call(it) for it in tr}
        seen_pair = {(it["from"], it["to"]) for it in tr}
        seen_vals = {c for it in tr for c in [it["from"], it["to"]] + it.get("to_also", [])}
        for it in te:
            vals = value_of(it)
            # the amount: the numbers the question gives; 'a' / 'one' before a currency counts as 1 (the labels do)
            nums = amounts_in(it["text"])
            if not nums and it["amount"] == "1":
                nums = ["1"]
            t0 = time.perf_counter()
            st, rd = question_state(codec, reader, it["text"], "fx", vals, SLOTS, numbers=nums)
            codes = sorted(set(vals.values()))
            got = codec.compose(st, {"fx": ["from", "to", "amount"]},
                                {"from": codes, "to": codes, "amount": nums or []}, z_margin=z_margin)
            ms = 1000 * (time.perf_counter() - t0)
            right = got["call"] == truth_call(it) or (
                it.get("to_also") and got["args"].get("from") == it["from"]
                and got["args"].get("to") in [it["to"]] + it["to_also"]
                and got["args"].get("amount") == it["amount"])
            rows.append({"id": it["id"], "fold": k, "right": bool(right), "verdict": got["verdict"],
                         "score": got["score"], "p_model": got["p_model"], "ms": ms,
                         "direction_right": got["args"].get("from") == it["from"]
                         and got["args"].get("to") in [it["to"]] + it.get("to_also", []),
                         "amount_right": got["args"].get("amount") == it["amount"],
                         "never_seen_whole": truth_call(it) not in seen_whole,
                         "new_pair": (it["from"], it["to"]) not in seen_pair,
                         "known_values": it["from"] in seen_vals and it["to"] in seen_vals,
                         "call": got["call"], "truth": truth_call(it)})
    # cross-fitted calibration: fold k's p comes from a DoorCalibrator fed ONLY the other folds' (score, right)
    for k in range(5):
        cal = DoorCalibrator("compose", min_count=8)
        for r in rows:
            if r["fold"] != k:
                cal.observe(r["score"], r["right"], refit=False)
        cal.refit()
        for r in rows:
            if r["fold"] == k:
                r["p_cal"] = cal.p_correct(r["score"])

    def block(sel):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return {"n": 0}
        served = [r for r in rs if r["verdict"] not in ("ambiguous", "empty")]
        e_cal, tab = expected_calibration_error([r["p_cal"] for r in rs], [r["right"] for r in rs], bins=10)
        e_mod, _ = expected_calibration_error([r["p_model"] for r in rs], [r["right"] for r in rs], bins=10)
        return {"n": len(rs), "right": sum(r["right"] for r in rs), "right_rate": round(np.mean([r["right"] for r in rs]), 4),
                "direction_right": sum(r["direction_right"] for r in rs),
                "amount_right": sum(r["amount_right"] for r in rs),
                "verdicts": {v: sum(r["verdict"] == v for r in rs) for v in ("clean", "dominant", "ambiguous", "empty")},
                "served_not_ambiguous": len(served), "right_when_served": sum(r["right"] for r in served),
                "ece_p_calibrated_crossfit": round(e_cal, 4), "ece_p_model_coldstart": round(e_mod, 4),
                "mean_p_calibrated": round(float(np.mean([r["p_cal"] for r in rs])), 4),
                "reliability_table_p_calibrated": [[round(a, 2), round(b, 2), n, round(mp, 3), round(acc, 3)]
                                                   for a, b, n, mp, acc in tab]}

    out = {"all_items_cv5": block(lambda r: True),
           "never_seen_whole": block(lambda r: r["never_seen_whole"] and r["known_values"]),
           "new_from_to_pair": block(lambda r: r["new_pair"] and r["known_values"]),
           "compose_ms_mean": round(float(np.mean([r["ms"] for r in rows])), 3),
           "wrong_calls": [{k: r[k] for k in ("id", "call", "truth", "verdict", "p_cal")} for r in rows if not r["right"]]}
    # the SYMMETRIC probe: held-out sentences with NO direction ('between X and Y'). The reader is trained on EVERY
    # directional item (none of these is among them) plus -- in the with_either arm -- the no-direction verdicts of
    # the symmetric TRAIN half; the right behaviour on the TEST half is to refuse a direction, not pick one.
    reader = teach(ContextRoles(), items)
    if with_either:
        teach_either(reader, skipped)
    cal = DoorCalibrator("compose", min_count=8)
    for r in rows:
        cal.observe(r["score"], r["right"], refit=False)
    cal.refit()
    sym = []
    for s in skipped:
        if not s.get("words") or s.get("split") != "test":
            continue
        w1, w2 = s["words"]
        vals = {w1: "A:" + w1, w2: "B:" + w2}
        st, rd = question_state(codec, reader, s["text"], "fx", vals, SLOTS, numbers=[])
        got = codec.compose(st, {"fx": ["from", "to"]}, {"from": sorted(vals.values()), "to": sorted(vals.values())},
                            z_margin=z_margin)
        pc = cal.p_correct(got["score"])
        # what the mind faculty SERVES: an ambiguous verdict's calibrated p is capped by the model p, because the
        # isotonic map clamps below its lowest seen score and has never seen a wrong answer down there
        served_p = min(pc if pc is not None else 1.0, got["p_model"]) if got["verdict"] in ("ambiguous", "empty") else pc
        sym.append({"verdict": got["verdict"], "p_cal": pc, "reader_p": rd["p"], "p_model": got["p_model"],
                    "p_served": served_p})
    dir_flag = sum(r["verdict"] == "ambiguous" for r in rows)
    out["with_either_prototype"] = bool(with_either)
    out["symmetric_probe"] = {
        "n": len(sym), "flagged_ambiguous": sum(r["verdict"] == "ambiguous" for r in sym),
        "p_calibrated_below_0.5": sum((r["p_cal"] or 0) < 0.5 for r in sym),
        "mean_p_calibrated": round(float(np.mean([r["p_cal"] or 0 for r in sym])), 4),
        "mean_reader_p": round(float(np.mean([r["reader_p"] for r in sym])), 4),
        "mean_p_model": round(float(np.mean([r["p_model"] for r in sym])), 4),
        "mean_p_served_ambiguous_capped": round(float(np.mean([r["p_served"] for r in sym])), 4),
        "p_served_below_0.5": sum(r["p_served"] < 0.5 for r in sym),
        "directional_items_flagged_ambiguous": "%d/%d" % (dir_flag, len(rows)),
        "mean_p_calibrated_directional": round(float(np.mean([r["p_cal"] for r in rows])), 4)}
    return out


# ================================================================================================ resonator
def bench_resonator(trials=20):
    from holographic.misc.holographic_resonator import ResonatorNetwork, map_bind, map_codebook
    D, NV, NC = 2048, 20, 50
    verbs = map_codebook(NV, D, 1)
    vals = map_codebook(NC, D, 2)
    Bf, Bt = np.roll(vals, 1, axis=1), np.roll(vals, 2, axis=1)   # permutation roles (rho^1, rho^2)
    net = ResonatorNetwork([verbs, Bf, Bt])
    t0 = time.perf_counter()
    null = net.noise_null(restarts=20, iters=200, patience=8, m=100)
    out = {"D": D, "codebooks": [NV, NC, NC], "roles": "permutation (rho^1, rho^2)", "restarts": 20, "iters": 200,
           "patience": 8, "m_null": 100, "accept_p": 0.01, "null_fit_s_once_per_shape": round(time.perf_counter() - t0, 2),
           "null_agreement": {"median": float(np.median(null)), "max": float(null.max())}, "rows": []}
    for noise in (0.0, 0.05, 0.10):
        rng = np.random.default_rng(1000 + int(noise * 100))
        before = {"right": 0, "solved": 0, "ms": []}
        after = {"right": 0, "accepted": 0, "accepted_wrong": 0, "ms": [], "p_right": [], "p_wrong": []}
        for t in range(trials):
            v, x, y = int(rng.integers(NV)), int(rng.integers(NC)), int(rng.integers(NC))
            while y == x:
                y = int(rng.integers(NC))
            c = map_bind(verbs[v], Bf[x], Bt[y])
            if noise:
                c = np.where(rng.random(D) < noise, -c, c)
            t1 = time.perf_counter()
            r0 = net.factor(c, restarts=20, iters=200)
            before["ms"].append(1000 * (time.perf_counter() - t1))
            before["right"] += tuple(int(i) for i in r0["factors"]) == (v, x, y)
            before["solved"] += bool(r0["solved"])
            t1 = time.perf_counter()
            r1 = net.factor(c, restarts=20, iters=200, tolerant=True, patience=8, accept_p=0.01, m_null=100)
            after["ms"].append(1000 * (time.perf_counter() - t1))
            ok = tuple(int(i) for i in r1["factors"]) == (v, x, y)
            after["right"] += ok
            after["accepted"] += r1["accepted"]
            after["accepted_wrong"] += r1["accepted"] and not ok
            (after["p_right"] if ok else after["p_wrong"]).append(r1["p_value"])
        out["rows"].append({
            "noise_flip": noise, "trials": trials,
            "before_exact_exit": {"right": before["right"], "solved_flag": before["solved"],
                                  "ms_mean": round(float(np.mean(before["ms"])), 1),
                                  "ms_max": round(float(np.max(before["ms"])), 1)},
            "after_tolerant_exit": {"right": after["right"], "accepted": after["accepted"],
                                    "accepted_but_wrong": after["accepted_wrong"],
                                    "ms_mean": round(float(np.mean(after["ms"])), 1),
                                    "ms_max": round(float(np.max(after["ms"])), 1),
                                    "p_value_when_right_median": float(np.median(after["p_right"])) if after["p_right"] else None,
                                    "p_value_when_wrong_median": float(np.median(after["p_wrong"])) if after["p_wrong"] else None}})
    junk = map_codebook(1, D, 999)[0]
    t1 = time.perf_counter()
    r0 = net.factor(junk, restarts=20, iters=200)
    b_ms = 1000 * (time.perf_counter() - t1)
    t1 = time.perf_counter()
    r1 = net.factor(junk, restarts=20, iters=200, tolerant=True)
    a_ms = 1000 * (time.perf_counter() - t1)
    out["refuse_random_composite"] = {"before_ms": round(b_ms, 1), "before_solved": bool(r0["solved"]),
                                      "after_ms": round(a_ms, 1), "after_accepted": bool(r1["accepted"]),
                                      "after_p_value": r1["p_value"]}
    # the ProductCodec face of the same thing (named verbs / values, the faculty's path), 10 noisy calls
    pc = ProductCodec(["v%02d" % i for i in range(NV)], ["x%02d" % i for i in range(NC)], ["from", "to"], dim=D)
    rng = np.random.default_rng(77)
    ok = acc = 0
    for t in range(10):
        a = {"from": "x%02d" % int(rng.integers(NC)), "to": "x%02d" % int(rng.integers(NC))}
        vb = "v%02d" % int(rng.integers(NV))
        c = pc.encode(vb, a)
        c = np.where(rng.random(D) < 0.05, -c, c)
        g = pc.factor(c)
        ok += g["call"] == call_string(vb, a)
        acc += g["accepted"]
    out["product_codec_5pct_noise"] = {"right": ok, "accepted": acc, "of": 10}
    return out


# ================================================================================================ E5.1 trajectories
def bench_trajectory(trials=200):
    codec = RoleCodec(dim=2048, seed=0)
    cur = ["USD", "EUR", "JPY", "GBP", "CAD", "MXN", "CHF", "INR"]
    cities = ["paris", "tokyo", "denver", "lima", "oslo", "cairo"]
    words = ["alpha", "bravo", "delta", "echo", "golf", "hotel", "india", "kilo", "lima", "mike", "oscar", "papa"]

    def rand_call(rng):
        k = int(rng.integers(3))
        if k == 0:
            a, b = rng.choice(len(cur), 2, replace=False)
            return ("fx", {"from": cur[int(a)], "to": cur[int(b)]})
        if k == 1:
            return ("weather", {"location": cities[int(rng.integers(len(cities)))]})
        return ("search", {"query": words[int(rng.integers(len(words)))]})

    def rand_text(rng):
        # incidental text from a BOUNDED vocabulary: hashed_ngram_encode caches one 2048-float vector per distinct
        # n-gram, and random digit strings grew that cache past 1.6 GB (the box OOM-kills at ~2 GB) -- measured
        return " ".join(words[int(i)] for i in rng.integers(len(words), size=6))

    def enc(calls, texts, tw):
        return trajectory_encode([encode_step(codec, v, a, text=t, text_weight=tw) for (v, a), t in zip(calls, texts)])

    out = {"D": 2048, "trials": trials, "step": "VERB + slots + text_weight * TEXT(incidental text)", "rows": []}
    for tw in (1.0, 0.5):
        for T in (5, 10, 20):
            rng = np.random.default_rng(5000 + T + int(tw * 10))
            cos = {k: [] for k in ("same_plan_noise_floor", "adjacent_swap", "random_swap", "insert_anywhere",
                                   "insert_at_end", "delete_one", "change_one_arg", "unrelated_plan")}
            diff_ok = {"adjacent_swap": 0, "insert_anywhere": 0, "delete_one": 0}
            for t in range(trials):
                calls = [rand_call(rng) for _ in range(T)]
                A = enc(calls, [rand_text(rng) for _ in range(T)], tw)

                def other(cs):
                    return enc(cs, [rand_text(rng) for _ in range(len(cs))], tw)   # every re-run has new text

                cos["same_plan_noise_floor"].append(float(np.dot(A, other(calls))))
                i = int(rng.integers(T - 1))
                sw = calls[:i] + [calls[i + 1], calls[i]] + calls[i + 2:]
                cos["adjacent_swap"].append(float(np.dot(A, other(sw))))
                i, j = sorted(int(x) for x in rng.choice(T, 2, replace=False))
                rs = list(calls)
                rs[i], rs[j] = rs[j], rs[i]
                cos["random_swap"].append(float(np.dot(A, other(rs))))
                new = rand_call(rng)
                k = int(rng.integers(T + 1))
                ins = calls[:k] + [new] + calls[k:]
                cos["insert_anywhere"].append(float(np.dot(A, other(ins))))
                cos["insert_at_end"].append(float(np.dot(A, other(calls + [new]))))
                k2 = int(rng.integers(T))
                dl = calls[:k2] + calls[k2 + 1:]
                cos["delete_one"].append(float(np.dot(A, other(dl))))
                ch = list(calls)
                k3 = int(rng.integers(T))
                v, a = ch[k3]
                a2 = dict(a)
                slot = sorted(a2)[0]
                pool = cur if slot in ("from", "to") else (cities if slot == "location" else words)
                a2[slot] = [x for x in pool if x not in a.values()][int(rng.integers(len(pool) - len(a)))]
                ch[k3] = (v, a2)
                cos["change_one_arg"].append(float(np.dot(A, other(ch))))
                cos["unrelated_plan"].append(float(np.dot(A, other([rand_call(rng) for _ in range(T)]))))
                # decode-and-diff: read both back against the book of clean calls, name the edit
                if t < 40:
                    book = {}
                    for vv, aa in calls + [new]:
                        book[call_string(vv, aa)] = codec.encode_call(vv, aa)
                    base = [n for n, _ in trajectory_decode(A, book, max_len=T + 2)]
                    for name, cs in (("adjacent_swap", sw), ("insert_anywhere", ins), ("delete_one", dl)):
                        got = [n for n, _ in trajectory_decode(other(cs), book, max_len=T + 2)]
                        want = [call_string(vv, aa) for vv, aa in cs]
                        d = trajectory_diff(base, got)
                        okd = got == want and base == [call_string(vv, aa) for vv, aa in calls] and len(d) >= 1 and (
                            (name == "adjacent_swap" and d[0]["op"] == "swap") or
                            (name == "insert_anywhere" and d[0]["op"] == "insert") or
                            (name == "delete_one" and d[0]["op"] == "delete"))
                        diff_ok[name] += bool(okd)
            floor = np.asarray(cos["same_plan_noise_floor"])
            thr = float(np.quantile(floor, 0.01))
            row = {"T": T, "text_weight": tw, "noise_floor": summary(floor), "edits": {}}
            for k, v in cos.items():
                if k == "same_plan_noise_floor":
                    continue
                v = np.asarray(v)
                row["edits"][k] = {"cosine": summary(v), "detected_at_1pct_false_alarm": round(float((v < thr).mean()), 4),
                                   "auroc_vs_floor": round(auroc(v, floor), 4)}   # P(floor cos > edit cos)
            row["decode_diff_names_the_edit"] = {k: "%d/40" % n for k, n in diff_ok.items()}
            out["rows"].append(row)
    return out


# ================================================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "research", "evidence", "bench_rolecall.json"))
    ap.add_argument("--only", default="direction,decode,compose,resonator,trajectory")
    ap.add_argument("--trials", type=int, default=500, help="decode trials per condition")
    a = ap.parse_args(argv)
    only = set(a.only.split(","))
    items, skipped = load_items()
    global SKIPPED
    SKIPPED = skipped
    res = {"cmd": "PYTHONHASHSEED=0 python3 tools/bench_rolecall.py" + ("" if len(only) == 5 else " --only " + a.only),
           "date": time.strftime("%Y-%m-%d"), "loadavg_at_start": [round(x, 2) for x in os.getloadavg()],
           "data": "tests/data/exchange_direction.json (%d directional items, %d skipped with reasons)"
                   % (len(items), len(skipped))}
    T0 = time.perf_counter()
    for name, fn in (("direction", lambda: bench_direction(items)), ("decode", lambda: bench_decode(a.trials)),
                     ("compose", lambda: {"with_either": bench_compose(items, skipped),
                                          "without_either_kept_negative": bench_compose(items, skipped,
                                                                                        with_either=False)}),
                     ("resonator", bench_resonator),
                     ("trajectory", bench_trajectory)):
        if name in only:
            t0 = time.perf_counter()
            res[name] = fn()
            res[name]["cpu_s"] = round(time.perf_counter() - t0, 1)
            print("[%s] %.1fs" % (name, time.perf_counter() - t0), flush=True)
    res["total_s"] = round(time.perf_counter() - T0, 1)
    res["loadavg_at_end"] = [round(x, 2) for x in os.getloadavg()]
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
        fh.write("\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("trajectory",)}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()

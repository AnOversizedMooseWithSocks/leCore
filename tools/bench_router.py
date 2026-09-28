#!/usr/bin/env python3
"""bench_router.py -- the capability router: is it fast (E3.4), does it LEARN (E4.1), and does serve ask it (E4.5)?

WHY THIS EXISTS (docs/BACKLOG_contrastive.md, the CLM panel's router items)
-------------------------------------------------------------------------
The panel (docs/research/evidence/clm_panel_20260926/w3-reflex/e02_*.py) measured the catalog router on held-out
aliases: find_capability top-1 0.402 over 3,954 held-out aliases and 3,952 cards, route_tiered answering 37.1% at
0.604 accuracy, 16.7 ms per route of which families() was 15.2 ms, and serve() escalating 20 of 20 alias probes
because it never asked the router. This harness re-measures each of those on the CURRENT catalog, then measures what
the backlog changed, with the same protocol:

    * aliases come from the live skills catalog (holographic_skills._catalog()); an alias is HELD OUT by the panel's
      insertion-stable rule int(sha256(alias)[:2], 16) % 2 == 1 and every held-out string is removed from EVERY card
      of an in-process UnifiedMind(dim=256, seed=0)'s catalog before scoring (an unseen wording; its card is the
      truth -- free, noiseless labels);
    * E4.1 needs a VALIDATION split to choose settings on: carved from the TRAINING aliases by
      int(sha256('router-val|' + alias)[:8], 16) % 5 == 0, and those strings leave the index too (a val alias left
      in the index would win by its own +5 exact-alias bonus). So this bench's lexical baseline is scored on a
      slightly POORER index than the panel's -- both numbers are reported;
    * out-of-scope questions for the E0.4 gate protocol are CLINC150's oos_test (real human wording that matches no
      intent; off-catalog BY ASSUMPTION -- a few may touch a real capability, which only makes the gate look worse),
      split by bench_contrastive.oos_half_a: half A calibrates, half B is reported. Precision is under the stated
      deployment prior 4,500 in-scope : 1,000 out-of-scope (bench_contrastive.PRIOR), never a split's own mix.

SECTIONS (all by default; each writes into docs/research/evidence/bench_router.json)
    latency   E3.4  route_tiered per call, the pre-cache implementation (kept verbatim in tests/test_router_learning.py
                    as the oracle) vs the shipped one, same queries, and how many records differ (must be 0)
    panel     E0.2  the panel's protocol reproduced on today's catalog (test ablated only): top-1 / top-5, tiers,
                    answer accuracy
    learn     E4.1  the learned router (holographic_routerlearn.RouteLearner) pretrained on TRAIN aliases with
                    leave-one-out rivals; the lam x mu grid chosen on VAL; test top-1 with a paired bootstrap CI vs the
                    lexical ranking on the same index; arms kept for the record (prior only, prototype only, untrained)
    gate      E4.1  answer / menu / refuse re-derived with the E0.4 protocol (bench_contrastive.gate_protocol):
                    signals compared by val AURC, thresholds calibrated on val + oos half A, reported on test + half B,
                    against the shipped z tiers (z_answer -0.1, z_refuse -0.5, ties == 1)
    serve     E4.5  the 20 alias probes (e02_service.py's list) through mind.serve(): in a fresh mind (aliases in the
                    index, as in the live service) and with the probes ablated; each served capability checked
                    against the alias's true card

Run:  PYTHONHASHSEED=0 python tools/bench_router.py --data <dir with clinc150_full.json> [--sections ...]
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
os.environ.setdefault("PYTHONHASHSEED", "0")

OUT = os.path.join(REPO, "docs", "research", "evidence", "bench_router.json")
PANEL = {"top1": 0.402, "top5": 0.638, "answer_share": 0.371, "answer_accuracy": 0.604, "route_ms": 16.7,
         "families_ms": 15.2, "serve_served": 0}
# The 20 service probes (e02_service.py: every len/20-th held-out alias in sorted order, 2026-09-26) -- listed
# verbatim so a catalog edit cannot silently change the probe set.
PROBES = ["1d curve from medial voxels", "audio drives parameters", "cache invalidation", "combine shader variants",
          "decode a png", "draw the mandelbrot set", "faraday rotation", "generate many surrogates cheaply",
          "identify a generator", "kernel", "mandelbrot field", "my glb bake is hollow", "pcg hash",
          "put the sphere on top of the box", "render points to an image", "scattering off a well",
          "silhouette iou sweep", "store a frame stack", "too many factors", "wave state encoder carrier and envelope"]
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0)
MUS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.7, 1.0)


def held_out(a):
    """The panel's insertion-stable test rule (the guard's rule): odd first sha256 byte."""
    return int(hashlib.sha256(a.encode()).hexdigest()[:2], 16) % 2 == 1


def is_val(a):
    """This bench's val carve of the TRAINING aliases: 1 in 5 by a salted sha256 (independent of the test rule)."""
    return int(hashlib.sha256(("router-val|" + a).encode()).hexdigest()[:8], 16) % 5 == 0


def rebake(cat):
    """Forget the catalog's baked haystacks / memo / null cache so the next call re-bakes from the aliases."""
    for attr in ("_fc_baked", "_fc_vocab", "_fc_hash", "_fc_memo", "_fc_memo_path", "_null_floor_cache"):
        if hasattr(cat, attr):
            delattr(cat, attr)


def ms(xs):
    """{mean, p50, p95} in milliseconds of a list of seconds."""
    a = 1e3 * np.asarray(xs, float)
    return {"mean": round(float(a.mean()), 3), "p50": round(float(np.median(a)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "n": int(len(a))}


class World:
    """One mind + its catalog, the alias truth, and the splits. ablate() removes strings from every card."""

    def __init__(self):
        from holographic.misc.holographic_skills import _catalog
        import lecore
        skills = _catalog()
        uniq = sorted({a for c in skills.all() for a in (c.aliases or ())})
        self.skills_aliases = len(uniq)
        self.held = {a for a in uniq if held_out(a)}
        self.mind = lecore.UnifiedMind(dim=256, seed=0)
        self.cat = self.mind._capability_catalog()
        self.truth = {}
        for c in self.cat.all():
            for a in (c.aliases or ()):
                self.truth.setdefault(a, []).append(c.name)
        self.test = sorted(a for a in self.held if a in self.truth)
        pool = sorted(a for a in self.truth if a not in self.held)
        self.val = [a for a in pool if is_val(a)]
        self.train = [a for a in pool if not is_val(a)]
        self._orig = {c.name: tuple(c.aliases or ()) for c in self.cat.all()}

    def ablate(self, drop):
        """Every string in `drop` leaves every card (the rest restored to the original lists)."""
        drop = set(drop)
        for c in self.cat.all():
            c.aliases = tuple(a for a in self._orig.get(c.name, ()) if a not in drop)
        rebake(self.cat)
        self.cat._ensure_fc_baked()

    def lexical(self, q, k):
        """[(capability, score)] -- the one scorer, top k."""
        return self.cat.find_scored(q, k=k)


# ============================================================================================ E3.4 latency
def section_latency(w, n=300):
    """route_tiered per call: the pre-E3.4 code (oracle, from the test file) vs the shipped method, same queries."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("trl", os.path.join(REPO, "tests", "test_router_learning.py"))
    trl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trl)
    w.ablate(w.test)
    qs = w.test[:n]
    for q in qs[:40]:
        w.cat.route_tiered(q)                                # warm the per-token-count nulls for both arms
    t_old, t_new, diff = [], [], 0
    for q in qs:
        t = time.perf_counter()
        a = trl._route_tiered_pre_e34(w.cat, q)
        t_old.append(time.perf_counter() - t)
        t = time.perf_counter()
        b = w.cat.route_tiered(q)
        t_new.append(time.perf_counter() - t)
        diff += trl._canon(a) != trl._canon(b)
    t_fam = []
    for _ in range(50):
        t = time.perf_counter()
        w.cat.families()
        t_fam.append(time.perf_counter() - t)
    t_mind = []
    for q in qs[:100]:
        t = time.perf_counter()
        w.mind.route_tiered(q)
        t_mind.append(time.perf_counter() - t)
    return {"cards": len(w.cat), "queries": len(qs), "records_differing": diff,
            "route_tiered_ms_before": ms(t_old), "route_tiered_ms_after": ms(t_new),
            "families_ms_memoised": ms(t_fam), "mind_route_tiered_ms": ms(t_mind),
            "panel": {"route_ms": PANEL["route_ms"], "families_ms": PANEL["families_ms"]},
            "machine_load_1min": os.getloadavg()[0]}


# ============================================================================================ E0.2 reproduction
def tiers_of(w, qs, truth, **kw):
    """The shipped z tiers over queries: shares, answer accuracy, menu recall."""
    t = {"answer": 0, "menu": 0, "clarify": 0, "refuse": 0}
    ans_ok = menu_ok = 0
    for q in qs:
        r = w.cat.route_tiered(q, k=5, **kw)
        t[r["tier"]] += 1
        tr = truth.get(q, ())
        ans_ok += r["tier"] == "answer" and r["answer"] in tr
        menu_ok += r["tier"] in ("menu", "clarify") and any(o["name"] in tr for o in r["options"])
    n = len(qs)
    return {"n": n, "shares": {k: round(v / n, 4) for k, v in t.items()},
            "answer_accuracy": round(ans_ok / t["answer"], 4) if t["answer"] else None,
            "menu_holds_truth": round(menu_ok / (t["menu"] + t["clarify"]), 4) if (t["menu"] + t["clarify"]) else None}


def section_panel(w):
    """The panel's protocol verbatim on today's catalog: only the test half leaves the index."""
    w.ablate(w.test)
    ranks = []
    for q in w.test:
        names = [c.name for c, _ in w.lexical(q, 8)]
        ranks.append(next((j + 1 for j, n in enumerate(names) if n in w.truth[q]), None))
    n = len(w.test)
    out = {"cards": len(w.cat), "skills_aliases": w.skills_aliases, "held_out": len(w.test),
           "top1": round(sum(r == 1 for r in ranks) / n, 4),
           "top5": round(sum(1 for r in ranks if r is not None and r <= 5) / n, 4),
           "route_tiered": tiers_of(w, w.test, w.truth),
           "panel": {k: PANEL[k] for k in ("top1", "top5", "answer_share", "answer_accuracy")}}
    return out


# ============================================================================================ E4.1 the learner
def pretrain(w, learner, pairs, k):
    """One pass of alias -> card verdicts, each with its LEAVE-ONE-OUT lexical rivals (the hard negatives)."""
    from holographic.agents_and_reasoning.holographic_routerlearn import loo_candidates
    t = time.perf_counter()
    items = [(a, w.truth[a][0], [c.name for c, _ in loo_candidates(w.cat, a, k=k)]) for a in pairs]
    learner.learn_many(items)
    return time.perf_counter() - t


def fused_rank(learner, q, ranked, lam, mu):
    """The learner's rerank at (lam, mu) without mutating it."""
    l0, m0 = learner.lam, learner.mu
    learner.lam, learner.mu = lam, mu
    try:
        return learner.rerank(q, ranked)
    finally:
        learner.lam, learner.mu = l0, m0


def section_learn(w, dims=(1024, 2048)):
    """E4.1: pretrain on TRAIN, choose (lam, mu) on VAL, report on TEST with a paired bootstrap CI."""
    import bench_contrastive as BC
    from holographic.agents_and_reasoning import holographic_routerlearn as RL
    w.ablate(list(w.test) + list(w.val))
    K = RL.ROUTER_K
    lex = {q: w.lexical(q, K) for q in w.test + w.val}
    right = lambda q, names: bool(names) and names[0] in w.truth[q]
    out = {"splits": {"train": len(w.train), "val": len(w.val), "test": len(w.test)}, "K": K,
           "lexical": {"val_top1": round(float(np.mean([right(q, [c.name for c, _ in lex[q]]) for q in w.val])), 4),
                       "test_top1": round(float(np.mean([right(q, [c.name for c, _ in lex[q]]) for q in w.test])), 4)},
           "grid": {}, "configs_tried": 0}
    learners = {}
    for dim in dims:
        feats = RL.RouteFeatures(w.cat, dim=dim)
        # untrained: an empty store (every card scored against its text seed, no verdicts)
        untrained = RL.RouteLearner(w.cat, dim=dim, features=feats)
        lr = RL.RouteLearner(w.cat, dim=dim, features=feats)
        secs = pretrain(w, lr, w.train, K)
        learners[dim] = (untrained, lr)
        out["pretrain_s_%d" % dim] = round(secs, 1)
        out["store_rows_%d" % dim] = len(lr.store)
        # the grid is run in full at the first dim; the others only at the chosen point (fewer configurations)
        grid = [(la, mu) for la in LAMS for mu in MUS] if dim == dims[0] else None
        if grid is None:
            continue
        for la, mu in grid:
            acc = float(np.mean([right(q, [c.name for c, *_ in fused_rank(lr, q, lex[q], la, mu)]) for q in w.val]))
            out["grid"]["lam%.2f_mu%.2f" % (la, mu)] = round(acc, 4)
            out["configs_tried"] += 1
    best = max(out["grid"].items(), key=lambda kv: (kv[1], -float(kv[0].split("_")[0][3:]), -float(kv[0].split("mu")[1])))
    la, mu = float(best[0].split("_")[0][3:]), float(best[0].split("mu")[1])
    out["chosen_on_val"] = {"lam": la, "mu": mu, "val_top1": best[1],
                            "module_constants": {"lam": RL.ROUTER_LAM, "mu": RL.ROUTER_MU, "dim": RL.ROUTER_DIM},
                            "constants_match": (la == RL.ROUTER_LAM and mu == RL.ROUTER_MU)}
    # report on TEST at the VAL CHOICE (the module constants must equal it -- constants_match says whether they do)
    SL, SM, SD = la, mu, RL.ROUTER_DIM
    arms = {"lexical": (None, 0, 0), "val_choice": (SD, SL, SM), "prior_only": (SD, 0.0, SM),
            "prototype_only": (SD, SL, 0.0), "untrained": ("u", SL, SM)}
    for d in dims:
        if d != SD:
            arms["val_choice_dim%d" % d] = (d, SL, SM)
    base = np.array([right(q, [c.name for c, _ in lex[q]]) for q in w.test], float)
    base5 = np.array([any(c.name in w.truth[q] for c, _ in lex[q][:5]) for q in w.test], float)
    out["test"] = {}
    for arm, (d, la_, mu_) in arms.items():
        if d is None:
            ok, ok5 = base, base5
        else:
            lrn = learners[SD][0] if d == "u" else learners[d][1]
            rr = {q: [c.name for c, *_ in fused_rank(lrn, q, lex[q], la_, mu_)] for q in w.test}
            ok = np.array([right(q, rr[q]) for q in w.test], float)
            ok5 = np.array([any(n in w.truth[q] for n in rr[q][:5]) for q in w.test], float)
        row = {"top1": round(float(ok.mean()), 4), "top5": round(float(ok5.mean()), 4)}
        if d is not None:
            ci = BC.paired_bootstrap(base, ok, n=2000, seed=0)
            row["vs_lexical"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ci.items()}
        out["test"][arm] = row
    out["acceptance"] = {"panel_top1": PANEL["top1"], "val_choice_test_top1": out["test"]["val_choice"]["top1"],
                         "beats_panel": out["test"]["val_choice"]["top1"] > PANEL["top1"],
                         "ci_vs_lexical_excludes_zero": out["test"]["val_choice"]["vs_lexical"]["lo"] > 0}
    w._learner = learners[SD][1]
    w._learner.lam, w._learner.mu = SL, SM                # the gate section scores the val choice
    w._lex = lex
    return out


# ============================================================================================ E4.1 the gate
def section_gate(w, data_dir, targets=(0.70, 0.75, 0.80, 0.90)):
    """answer / menu / refuse re-derived with the E0.4 protocol. Needs section_learn's learner (w._learner)."""
    import bench_contrastive as BC
    from holographic.agents_and_reasoning import holographic_routerlearn as RL
    lr = w._learner
    K = RL.ROUTER_K
    clinc = json.load(open(os.path.join(data_dir, "clinc150_full.json")))
    oos = [q for q, _ in clinc["oos_test"]]
    half_a = BC.oos_half_a(oos)
    lex = dict(w._lex)
    for q in oos:
        lex[q] = w.lexical(q, K)
    rows = {}
    for q in w.val + w.test + oos:
        z = float(w.cat.route_or_abstain(q, k=8, z_min=-1e9)["z"])
        L = lex[q]
        lx = [s for _, s in L]
        fu = lr.rerank(q, L)
        fs = [f for _, f, *_ in fu]
        rows[q] = {"z": z,
                   "lex_top": L[0][0].name if L else None, "lex_margin": (lx[0] - lx[1]) if len(lx) > 1 else (lx[0] if lx else 0.0),
                   "lex_ties": sum(1 for s in lx[:5] if lx and s == lx[0]),
                   "fused_top": fu[0][0].name if fu else None, "fused_margin": (fs[0] - fs[1]) if len(fs) > 1 else (fs[0] if fs else 0.0),
                   "lex_menu": [c.name for c, _ in L[:5]], "fused_menu": [c.name for c, *_ in fu[:5]]}
    qs = w.val + w.test + oos
    is_oos = np.array([False] * (len(w.val) + len(w.test)) + [True] * len(oos))
    is_cal = np.concatenate([np.ones(len(w.val), bool), np.zeros(len(w.test), bool), half_a])
    truth = lambda q: w.truth.get(q, ())
    ok_lex = np.array([rows[q]["lex_top"] in truth(q) for q in qs])
    ok_fu = np.array([rows[q]["fused_top"] in truth(q) for q in qs])
    NEG = -1e9                                            # no candidate at all: never served
    sig = {
        "lexical|z": (np.array([rows[q]["z"] if rows[q]["lex_top"] else NEG for q in qs]), ok_lex),
        "lexical|margin": (np.array([rows[q]["lex_margin"] if rows[q]["lex_top"] else NEG for q in qs]), ok_lex),
        "lexical|z+margin": (np.array([rows[q]["z"] + rows[q]["lex_margin"] if rows[q]["lex_top"] else NEG
                                       for q in qs]), ok_lex),
        "fused|z": (np.array([rows[q]["z"] if rows[q]["fused_top"] else NEG for q in qs]), ok_fu),
        "fused|margin": (np.array([rows[q]["fused_margin"] if rows[q]["fused_top"] else NEG for q in qs]), ok_fu),
        "fused|z+margin": (np.array([rows[q]["z"] + rows[q]["fused_margin"] if rows[q]["fused_top"] else NEG
                                     for q in qs]), ok_fu),
    }
    out = {"n": {"val": len(w.val), "test": len(w.test), "oos": len(oos), "oos_cal": int(half_a.sum())},
           "prior": list(BC.PRIOR), "signals": {}, "configs_tried": len(sig)}
    for name, (s, ok) in sig.items():
        g = BC.gate_protocol(s, ok, is_oos, is_cal, targets=targets, oracle=True)
        cal_aurc = BC.aurc(s[is_cal], ok[is_cal], ~is_oos[is_cal], BC.PRIOR)
        out["signals"][name] = {"val_aurc": round(cal_aurc, 4), "report": g}
    chosen = min((k for k in sig if k.startswith("fused")), key=lambda k: out["signals"][k]["val_aurc"])
    out["chosen_signal"] = chosen
    # ---- tiers. BASELINE: the shipped z tiers (z_answer -0.1, z_refuse -0.5, ties == 1) on the same items
    rep = ~is_cal
    rep_in, rep_oos = rep & ~is_oos, rep & is_oos

    def tier_stats(ans, ref, ok, menus):
        """shares on the report side: in-scope answer share / accuracy / refuse share / menu recall; oos answered /
        refused."""
        menu = ~ans & ~ref
        mi = menu & rep_in
        return {"in_scope": {"answer": round(float(ans[rep_in].mean()), 4),
                             "answer_accuracy": round(float(ok[ans & rep_in].mean()), 4) if (ans & rep_in).any() else None,
                             "menu": round(float(menu[rep_in].mean()), 4), "refuse": round(float(ref[rep_in].mean()), 4),
                             "menu_holds_truth": round(float(np.mean([any(n in truth(q) for n in menus[i])
                                                                      for i, q in enumerate(qs) if mi[i]])), 4) if mi.any() else None},
                "oos": {"answer": round(float(ans[rep_oos].mean()), 4), "menu": round(float(menu[rep_oos].mean()), 4),
                        "refuse": round(float(ref[rep_oos].mean()), 4)}}
    z = np.array([rows[q]["z"] for q in qs])
    has = np.array([rows[q]["lex_top"] is not None for q in qs])
    ties = np.array([rows[q]["lex_ties"] for q in qs])
    base_ans = has & (z >= -0.1) & (ties == 1)
    base_ref = ~has | (z < -0.5)
    out["tiers_baseline_z"] = tier_stats(base_ans, base_ref, ok_lex, [rows[q]["lex_menu"] for q in qs])
    # RE-DERIVED: answer = chosen signal >= its calibrated threshold (P under the prior, on val + oos half A); refuse =
    # z below the threshold that refuses AT LEAST the baseline's share of calibration oos (refusal rate not lower)
    # while refusing the fewest calibration in-scope questions.
    s, ok = sig[chosen]
    base_oos_ref_cal = float(base_ref[is_cal & is_oos].mean())
    cand = np.unique(z[is_cal])
    t_ref = next((float(t) for t in cand if float(((z < t) | ~has)[is_cal & is_oos].mean()) >= base_oos_ref_cal), -0.5)
    out["refuse"] = {"baseline_z_refuse": -0.5, "baseline_oos_refused_cal": round(base_oos_ref_cal, 4),
                     "rederived_z_refuse": round(t_ref, 4)}
    out["tiers"] = {}
    for P in targets:
        thr = out["signals"][chosen]["report"]["calibrated"]["P%.2f" % P]["thr"]
        ans = has & (s >= thr) & (z >= t_ref)
        ref = ~has | (z < t_ref)
        out["tiers"]["P%.2f" % P] = {"answer_threshold": thr, **tier_stats(ans, ref, ok, [rows[q]["fused_menu"] for q in qs])}
    # the lexical ranking under the same re-derived protocol (is the gain the learner's or the gate's?) -- with the
    # best LEXICAL signal by val AURC (what a mind without a learned router gates serve() on)
    lex_sig = min((k for k in sig if k.startswith("lexical")), key=lambda k: out["signals"][k]["val_aurc"])
    out["chosen_lexical_signal"] = lex_sig
    s_l, ok_l = sig[lex_sig]
    out["tiers_lexical_rederived"] = {}
    for P in targets:
        thr = out["signals"][lex_sig]["report"]["calibrated"]["P%.2f" % P]["thr"]
        ans = has & (s_l >= thr) & (z >= t_ref)
        out["tiers_lexical_rederived"]["P%.2f" % P] = {"answer_threshold": thr, **tier_stats(
            ans, ~has | (z < t_ref), ok_l, [rows[q]["lex_menu"] for q in qs])}
    return out


# ============================================================================================ E4.5 serve
def section_serve(ablate=False):
    """The 20 probes through mind.serve() in a FRESH mind (as the live service holds them: in the index), or with the
    probes ablated. Every served capability is checked against the probe's true card(s)."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    cat = m._capability_catalog()
    truth = {}
    for c in cat.all():
        for a in (c.aliases or ()):
            truth.setdefault(a, set()).add(c.name)
    if ablate:
        for c in cat.all():
            c.aliases = tuple(a for a in (c.aliases or ()) if a not in set(PROBES))
        rebake(cat)
    rows, t = [], []
    for q in PROBES:
        t0 = time.perf_counter()
        r = m.serve(q)
        t.append(time.perf_counter() - t0)
        cap = r.get("capability")
        rows.append({"q": q, "via": r.get("via"), "capability": cap,
                     "right": (cap in truth.get(q, ())) if cap else None,
                     "p_correct": dict.get(r, "p_correct"), "gate": r.get("gate")})
    served = [x for x in rows if x["via"] == "route"]
    return {"ablated": ablate, "probes": len(PROBES), "served": len(served),
            "served_right": sum(1 for x in served if x["right"]), "served_wrong": sum(1 for x in served if not x["right"]),
            "escalated": sum(1 for x in rows if x["via"] == "escalate"), "serve_ms": ms(t), "rows": rows,
            "panel_served": PANEL["serve_served"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=None, help="directory holding clinc150_full.json (for the gate section)")
    ap.add_argument("--sections", default="latency,panel,learn,gate,serve")
    ap.add_argument("--json", default=OUT)
    a = ap.parse_args()
    tempfile.tempdir = tempfile.mkdtemp(prefix="bench_router_")   # the catalog memo lives here, not in the shared /tmp
    secs = a.sections.split(",")
    try:
        res = json.load(open(a.json))
    except Exception:
        res = {}
    res["command"] = "PYTHONHASHSEED=0 python tools/bench_router.py --data <dir> --sections " + a.sections
    T0 = time.time()
    w = World() if any(s in secs for s in ("latency", "panel", "learn", "gate")) else None
    if w is not None:
        print("world: %d cards, test %d, val %d, train %d (%.1f s)" % (len(w.cat), len(w.test), len(w.val),
                                                                     len(w.train), time.time() - T0), flush=True)
    if "latency" in secs:
        res["latency"] = section_latency(w)
        print("latency", json.dumps(res["latency"]), flush=True)
    if "panel" in secs:
        res["panel"] = section_panel(w)
        print("panel", json.dumps(res["panel"]), flush=True)
    if "learn" in secs or "gate" in secs:
        res["learn"] = section_learn(w)
        print("learn", json.dumps(res["learn"]), flush=True)
    if "gate" in secs:
        if not a.data:
            raise SystemExit("the gate section needs --data (CLINC150's oos_test is the out-of-scope set)")
        from bench_contrastive import sha256_file
        res["gate"] = section_gate(w, a.data)
        res["gate"]["data_sha256"] = sha256_file(os.path.join(a.data, "clinc150_full.json"))
        print("gate", json.dumps(res["gate"])[:3000], flush=True)
    if "serve" in secs:
        res["serve"] = {"in_index": section_serve(False), "ablated": section_serve(True)}
        print("serve", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in res["serve"].items()}),
              flush=True)
    res["seconds"] = round(time.time() - T0, 1)
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("wrote", a.json, "in %.1f s" % res["seconds"])


if __name__ == "__main__":
    main()

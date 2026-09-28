"""tools/bench_rank.py -- E3.3: mind.rank(state, candidates), measured on real wording and on the clock.

WHAT IS MEASURED (CLM backlog E3.3, wave 2)
-------------------------------------------
mind.rank is CLM's "Score / Choice over arbitrary candidates" done our way: state and candidates in ONE space (hashed
character 3..5-grams at 2048), the ORDER by absolute cosine, an absolute floor that refuses, p_correct from the rank
door's own calibrator, p_null against in-vocabulary word salad scored by the same procedure, a conformal-style set, and
a ledger id whose outcome trains the door. This tool reproduces every number the door's constants cite.

  quality   CLINC150 (CC BY 3.0): the 4,500 test questions ranked against the 150 intents, each candidate = the
            intent name + 5 train examples (the first 5 in sha256 order -- deterministic, no cherry-picking), through
            mind.rank itself (learn=False: the static door). Top-1, AURC (risk-coverage area; lower = abstains better)
            with the top score / the margin / a candidate-relative softmax as the confidence, the floor table, and
            off-scope detection on the 1,000 oos_test questions (top score vs p_null). Comparison arms computed in the
            same space: one concatenated string per candidate, rolecall's option_encode, the intent name only, and
            two PLAIN LEXICAL baselines (word Jaccard; TF-IDF word cosine).
  set size  the same, with each question ranked against its truth + k-1 random intents (k = 2, 5, 20, 150): which
            confidence abstains better as the candidate set shrinks.
  learning  prequential through the door: each test question is ranked, then its truth reported with
            decision_outcome -- the rank door's ProtoStore learns (InfoNCE over the call's candidates), its calibrator
            and conformal stream fill. 3 seeded orders. Top-1 by thirds vs the static door on the same thirds; AURC of
            p_correct; ECE; the conformal set's coverage and size at alpha 0.1.
  latency   the 150-candidate CLINC set, the full door (record, null, learn): cold (a fresh mind's first call),
            warm cache on, warm cache OFF (every candidate re-encoded), and the bare ranking (no record, no null) --
            p50 / p95 over >= 1,000 calls where the arm allows it, with /proc/loadavg before and after each arm (the
            same setup took 4.2 s alone and 28 s under swarm load: the load is part of the number).

Run:  PYTHONHASHSEED=0 python tools/bench_rank.py --data <dir with clinc150_full.json> \
          [--json docs/research/evidence/bench_rank.json] [--calls 1000] [--skip-latency] [--skip-quality]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTHONHASHSEED", "0")

N_EXAMPLES = 5
SEEDS = 3
FLOORS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
SET_SIZES = (2, 5, 20, 150)


def loadavg():
    try:
        with open("/proc/loadavg") as f:
            return [float(x) for x in f.read().split()[:3]]
    except OSError:
        return None


def load_clinc(data_dir):
    path = os.path.join(data_dir, "clinc150_full.json")
    with open(path, "rb") as f:
        raw = f.read()
    d = json.loads(raw)
    by = {}
    for t, y in d["train"]:
        by.setdefault(y, []).append(t)
    intents = sorted(by)
    sh = lambda t: hashlib.sha256(t.encode()).hexdigest()
    ex = {y: sorted(by[y], key=sh)[:N_EXAMPLES] for y in intents}
    cands = [{"id": y, "text": y.replace("_", " "), "examples": ex[y]} for y in intents]
    return {"sha256": hashlib.sha256(raw).hexdigest(), "intents": intents, "cands": cands, "ex": ex,
            "test": d["test"], "oos": d["oos_test"]}


def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


# ------------------------------------------------------------------------------------------------- metrics ----

def aurc(conf, correct):
    """Area under the risk-coverage curve: serve in order of confidence, average the error rate over every coverage.
    Lower is better; a perfect ranker scores (1 - acc)^2 / 2 -ish, a random one ~(1 - acc)."""
    conf = np.asarray(conf, float)
    c = np.asarray(correct, float)[np.argsort(-conf, kind="stable")]
    return float((np.cumsum(1.0 - c) / np.arange(1, len(c) + 1)).mean())


def auroc(pos, neg):
    """P(a positive scores above a negative), mid-ranks for ties."""
    s = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    o = np.argsort(s, kind="stable")
    ss = s[o]
    r = np.empty(len(s))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(len(s))
    ranks[o] = r
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece(p, y, bins=10):
    P = np.asarray([a for a in p if a is not None], float)
    Y = np.asarray([b for a, b in zip(p, y) if a is not None], float)
    if not len(P):
        return None
    e = 0.0
    for b in range(bins):
        sel = (P >= b / bins) & ((P < (b + 1) / bins) if b < bins - 1 else (P <= 1.0))
        if sel.any():
            e += sel.mean() * abs(P[sel].mean() - Y[sel].mean())
    return float(e)


def softmax_top(S, tau=0.02):
    z = S / tau
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).max(axis=1)


def _conf_block(S, yi):
    """top-1 + AURC for the three confidences over a score matrix S (rows = questions)."""
    top = S.argmax(1)
    ok = top == yi
    srt = -np.sort(-S, axis=1)
    return {"top1": float(ok.mean()), "aurc_top": aurc(srt[:, 0], ok), "aurc_margin": aurc(srt[:, 0] - srt[:, 1], ok),
            "aurc_softmax_0.02": aurc(softmax_top(S), ok)}


# ------------------------------------------------------------------------------------------------- quality ----

def quality(D, m):
    from holographic.agents_and_reasoning.holographic_ai import unbind
    from holographic.agents_and_reasoning.holographic_rolecall import RoleCodec
    from holographic.unified.holographic_unified_p29_contrastive import _unit_vec
    enc = m._rank_state()["enc"]
    intents, cands, ex = D["intents"], D["cands"], D["ex"]
    yi = np.array([intents.index(y) for _, y in D["test"]])
    out = {}
    # 1) THE DOOR ITSELF (static): mind.rank on every test and oos question
    t0 = time.time()
    rows_in, rows_out = [], []
    for t, _ in D["test"]:
        r = m.rank(t, cands, learn=False, record=False)
        rows_in.append((r["ranked"][0][0], r["top"], r["margin"], r["p_null"], dict(r["ranked"])))
    for t, _ in D["oos"]:
        r = m.rank(t, cands, learn=False, record=False)
        rows_out.append((r["top"], r["p_null"]))
    ok = np.array([a == y for (a, *_), (_, y) in zip(rows_in, D["test"])])
    top = np.array([x[1] for x in rows_in])
    mar = np.array([x[2] for x in rows_in])
    pn_in = np.array([x[3] for x in rows_in])
    top_out = np.array([x[0] for x in rows_out])
    pn_out = np.array([x[1] for x in rows_out])
    S_door = np.array([[x[4][c["id"]] for c in cands] for x in rows_in])
    door = {"top1": float(ok.mean()), "aurc_top": aurc(top, ok), "aurc_margin": aurc(mar, ok),
            "aurc_softmax_0.02": aurc(softmax_top(S_door), ok),
            "oos_auroc_top": auroc(top, top_out), "oos_auroc_p_null": auroc(-pn_in, -pn_out),
            "p_null_le_0.05": {"in_scope": float((pn_in <= 0.05).mean()), "oos": float((pn_out <= 0.05).mean())},
            "p_null_le_0.01": {"in_scope": float((pn_in <= 0.01).mean()), "oos": float((pn_out <= 0.01).mean())},
            "floors": {("%.2f" % f): {"in_scope_kept": float((top >= f).mean()), "right_kept": float((top[ok] >= f).mean()),
                                      "oos_refused": float((top_out < f).mean())} for f in FLOORS},
            "seconds": round(time.time() - t0, 1)}
    out["door_bundle"] = door
    print("door (bundle)   top1 %.4f  AURC top %.4f margin %.4f softmax %.4f | oos AUROC top %.3f p_null %.3f"
          % (door["top1"], door["aurc_top"], door["aurc_margin"], door["aurc_softmax_0.02"], door["oos_auroc_top"],
             door["oos_auroc_p_null"]), flush=True)
    # 2) comparison arms in the SAME space (the door's encoder), scored directly
    Q = np.stack([_unit_vec(enc(t)) for t, _ in D["test"]])
    O = np.stack([_unit_vec(enc(t)) for t, _ in D["oos"]])
    arms = {
        "bundle_direct": np.stack([m._rank_encode(m._rank_state(), c["text"], c["examples"]) for c in cands]),
        "concat": np.stack([_unit_vec(enc(c["text"] + ": " + " | ".join(c["examples"]))) for c in cands]),
        "name_only": np.stack([_unit_vec(enc(c["text"])) for c in cands]),
    }
    rc = RoleCodec()
    arms["rolecall_option_encode"] = np.stack([unbind(rc.encode_option(y, ex[y]), rc.role(rc.TEXT)) for y in intents])
    for name, M in arms.items():
        S = Q @ M.T
        blk = _conf_block(S, yi)
        so = (O @ M.T).max(1)
        blk["oos_auroc_top"] = auroc(S.max(1), so)
        srt = -np.sort(-S, axis=1)
        okk = S.argmax(1) == yi
        blk["floors"] = {("%.2f" % f): {"right_kept": float((srt[okk, 0] >= f).mean()), "oos_refused": float((so < f).mean())}
                         for f in FLOORS}
        out[name] = blk
        print("%-22s top1 %.4f  AURC top %.4f margin %.4f softmax %.4f | oos AUROC %.3f"
              % (name, blk["top1"], blk["aurc_top"], blk["aurc_margin"], blk["aurc_softmax_0.02"], blk["oos_auroc_top"]),
              flush=True)
    # 3) plain lexical baselines
    tok = lambda s: re.findall(r"[a-z0-9']+", s.lower())
    docs = [c["text"] + " " + " ".join(c["examples"]) for c in cands]
    cset = [set(tok(d)) for d in docs]
    J = np.array([[len(set(tok(t)) & c) / max(1, len(set(tok(t)) | c)) for c in cset] for t, _ in D["test"]])
    JO = np.array([[len(set(tok(t)) & c) / max(1, len(set(tok(t)) | c)) for c in cset] for t, _ in D["oos"]])
    blk = _conf_block(J, yi)
    blk["oos_auroc_top"] = auroc(J.max(1), JO.max(1))
    out["lexical_jaccard"] = blk
    vocab = {}
    for d in docs:
        for w in sorted(set(tok(d))):
            vocab.setdefault(w, len(vocab))
    df = np.zeros(len(vocab))
    for d in docs:
        for w in set(tok(d)):
            df[vocab[w]] += 1
    idf = np.log((1 + len(docs)) / (1 + df)) + 1.0

    def tfidf(s):
        v = np.zeros(len(vocab))
        for w in tok(s):
            if w in vocab:
                v[vocab[w]] += 1.0
        return _unit_vec(v * idf)
    T = np.stack([tfidf(d) for d in docs])
    ST = np.stack([tfidf(t) for t, _ in D["test"]]) @ T.T
    SO = np.stack([tfidf(t) for t, _ in D["oos"]]) @ T.T
    blk = _conf_block(ST, yi)
    blk["oos_auroc_top"] = auroc(ST.max(1), SO.max(1))
    out["lexical_tfidf_words"] = blk
    for name in ("lexical_jaccard", "lexical_tfidf_words"):
        b = out[name]
        print("%-22s top1 %.4f  AURC top %.4f margin %.4f | oos AUROC %.3f"
              % (name, b["top1"], b["aurc_top"], b["aurc_margin"], b["oos_auroc_top"]), flush=True)
    # 4) set size: truth + k-1 random intents (seeded) -- which confidence abstains better as sets shrink
    M = arms["bundle_direct"]
    rng = np.random.default_rng(0)
    sizes = {}
    for k in SET_SIZES:
        tops, margs, sms, oks = [], [], [], []
        for i in range(len(Q)):
            others = [j for j in rng.permutation(len(intents))[:k] if j != yi[i]][:k - 1]
            cs = [int(yi[i])] + others
            s = M[cs] @ Q[i]
            srt = -np.sort(-s)
            tops.append(srt[0])
            margs.append(srt[0] - srt[1])
            sms.append(float(softmax_top(s[None, :])[0]))
            oks.append(cs[int(np.argmax(s))] == yi[i])
        sizes[str(k)] = {"top1": float(np.mean(oks)), "aurc_top": aurc(tops, oks), "aurc_margin": aurc(margs, oks),
                         "aurc_softmax_0.02": aurc(sms, oks)}
        print("set size %3d  top1 %.3f  AURC top %.4f margin %.4f softmax %.4f"
              % (k, sizes[str(k)]["top1"], sizes[str(k)]["aurc_top"], sizes[str(k)]["aurc_margin"],
                 sizes[str(k)]["aurc_softmax_0.02"]), flush=True)
    out["set_size"] = sizes
    # 5) p_null on SMALL candidate sets through the door: 4 random sets of k intents, 600 test / 300 oos questions
    rng = np.random.default_rng(1)
    sub_in = rng.permutation(len(D["test"]))[:600]
    sub_out = rng.permutation(len(D["oos"]))[:300]
    small = {}
    for k in (3, 10):
        pin, pout, tin, tout = [], [], [], []
        for s_ in range(4):
            cs = sorted(rng.permutation(len(intents))[:k].tolist())
            cset_ = [cands[j] for j in cs]
            for i in sub_in:
                if yi[i] in cs:
                    r = m.rank(D["test"][i][0], cset_, learn=False, record=False)
                    pin.append(r["p_null"])
                    tin.append(r["top"])
            for i in sub_out:
                r = m.rank(D["oos"][i][0], cset_, learn=False, record=False)
                pout.append(r["p_null"])
                tout.append(r["top"])
        pin, pout = np.array(pin), np.array(pout)
        small[str(k)] = {"n_in": int(len(pin)), "n_oos": int(len(pout)), "oos_auroc_top": auroc(tin, tout),
                         "oos_auroc_p_null": auroc(-pin, -pout), "p_null_le_0.05_in": float((pin <= 0.05).mean()),
                         "p_null_le_0.05_oos": float((pout <= 0.05).mean())}
        print("p_null, sets of %2d: oos AUROC top %.3f p_null %.3f | significant at .05: in %.2f oos %.2f"
              % (k, small[str(k)]["oos_auroc_top"], small[str(k)]["oos_auroc_p_null"], small[str(k)]["p_null_le_0.05_in"],
                 small[str(k)]["p_null_le_0.05_oos"]), flush=True)
    out["p_null_small_sets"] = small
    return out


# ------------------------------------------------------------------------------------------------ learning ----

def learning(D):
    intents, cands = D["intents"], D["cands"]
    out = []
    for seed in range(SEEDS):
        t0 = time.time()
        m = _mind()
        order = np.random.default_rng(seed).permutation(len(D["test"]))
        rec = []
        for i in order:
            q, y = D["test"][int(i)]
            r = m.rank(q, cands, null=False)
            ranked = dict(r["ranked"])
            rec.append({"ok": r["ranked"][0][0] == y, "top": r["top"], "margin": r["margin"], "p": r["p_correct"],
                        "set": r["set"], "truth": y, "static_ok": None})
            m.decision_outcome(r["id"], y)
        # the static door on the same order, for the same thirds
        ms = _mind()
        for j, i in enumerate(order):
            q, y = D["test"][int(i)]
            rec[j]["static_ok"] = ms.rank(q, cands, learn=False, record=False, null=False)["ranked"][0][0] == y
        n = len(rec)
        h = n // 3
        ok = np.array([r["ok"] for r in rec])
        st = np.array([r["static_ok"] for r in rec])
        late = rec[h:]
        lok = ok[h:]
        pc = [r["p"] for r in late]
        sets_ = [r for r in rec[2 * h:] if r["set"] is not None]
        res = {"seed": seed, "top1_all": float(ok.mean()), "top1_thirds": [float(ok[:h].mean()), float(ok[h:2 * h].mean()),
                                                                          float(ok[2 * h:].mean())],
               "static_top1_thirds": [float(st[:h].mean()), float(st[h:2 * h].mean()), float(st[2 * h:].mean())],
               "aurc_last2of3_top": aurc([r["top"] for r in late], lok),
               "aurc_last2of3_margin": aurc([r["margin"] for r in late], lok),
               "aurc_last2of3_p_correct": aurc([(-1.0 if p is None else p) + 1e-9 * r["margin"] for p, r in zip(pc, late)], lok),
               "p_correct_available_last2of3": float(np.mean([p is not None for p in pc])),
               "ece_last2of3": ece(pc, lok),
               "set_coverage_last_third": float(np.mean([r["truth"] in r["set"] for r in sets_])) if sets_ else None,
               "set_mean_size_last_third": float(np.mean([len(r["set"]) for r in sets_])) if sets_ else None,
               "seconds": round(time.time() - t0, 1)}
        out.append(res)
        print("learning seed %d: top1 thirds %s (static %s) | AURC late top %.4f margin %.4f p_correct %.4f | ECE %.3f | "
              "set coverage %.3f size %.2f | %.0fs" % (seed, ["%.3f" % x for x in res["top1_thirds"]],
                                                    ["%.3f" % x for x in res["static_top1_thirds"]], res["aurc_last2of3_top"],
                                                    res["aurc_last2of3_margin"], res["aurc_last2of3_p_correct"],
                                                    res["ece_last2of3"] or -1, res["set_coverage_last_third"] or -1,
                                                    res["set_mean_size_last_third"] or -1, res["seconds"]), flush=True)
    keys = ("top1_all", "aurc_last2of3_top", "aurc_last2of3_margin", "aurc_last2of3_p_correct", "ece_last2of3",
            "set_coverage_last_third", "set_mean_size_last_third")
    summary = {k: {"mean": float(np.mean([r[k] for r in out])), "sd": float(np.std([r[k] for r in out]))} for k in keys}
    summary["top1_last_third"] = {"mean": float(np.mean([r["top1_thirds"][2] for r in out])),
                                  "sd": float(np.std([r["top1_thirds"][2] for r in out]))}
    summary["static_top1_last_third"] = {"mean": float(np.mean([r["static_top1_thirds"][2] for r in out])),
                                         "sd": float(np.std([r["static_top1_thirds"][2] for r in out]))}
    return {"per_seed": out, "summary": summary}


# ------------------------------------------------------------------------------------------------- latency ----

def _pcts(xs):
    a = np.asarray(xs, float) * 1000.0
    return {"n": int(len(a)), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "max_ms": float(a.max()), "total_s": float(a.sum() / 1000.0)}


def latency(D, calls):
    from holographic.unified.holographic_unified_p29_contrastive import _bounded_ngram
    cands = D["cands"]
    qs = [t for t, _ in D["test"]]
    out = {"cpus": os.cpu_count(), "candidates": len(cands), "calls": calls}

    def arm(name, fn, n):
        la0 = loadavg()
        ts = []
        for i in range(n):
            t0 = time.perf_counter()
            fn(i)
            ts.append(time.perf_counter() - t0)
        res = _pcts(ts)
        res["loadavg_before"], res["loadavg_after"] = la0, loadavg()
        out[name] = res
        print("latency %-34s n %4d  p50 %8.2f ms  p95 %8.2f ms  mean %8.2f ms  load %s -> %s"
              % (name, res["n"], res["p50_ms"], res["p95_ms"], res["mean_ms"], la0, res["loadavg_after"]), flush=True)

    # COLD: a fresh mind's FIRST call (its candidate encodings, the atom cache and the null pool all empty); the mind's
    # construction is not timed. 20 minds -- a fresh mind per call is the only honest cold, so n is small (stated).
    minds = []

    def cold(i):
        minds[-1].rank(qs[i], cands)
    la0, ts = loadavg(), []
    for i in range(20):
        minds.append(_mind())
        t0 = time.perf_counter()
        cold(i)
        ts.append(time.perf_counter() - t0)
        minds.clear()
    res = _pcts(ts)
    res["loadavg_before"], res["loadavg_after"] = la0, loadavg()
    out["cold_first_call_full_door"] = res
    print("latency %-34s n %4d  p50 %8.2f ms  p95 %8.2f ms" % ("cold_first_call_full_door", res["n"], res["p50_ms"],
                                                               res["p95_ms"]), flush=True)
    # WARM: one mind, primed with one call per null length bucket (steady state), then `calls` distinct questions
    m = _mind()
    for q in qs[:60]:
        m.rank(q, cands)
    arm("warm_cache_on_full_door", lambda i: m.rank(qs[(60 + i) % len(qs)], cands), calls)
    arm("warm_cache_on_bare_ranking", lambda i: m.rank(qs[(60 + i) % len(qs)], cands, record=False, null=False), calls)
    # CACHE OFF at the default atom cache (4,096 atoms < the 19,921 distinct n-grams of these 900 candidate texts, so
    # the atoms thrash too): n is small because each call costs seconds -- stated, not hidden
    arm("warm_cache_off_default_atom_cap", lambda i: m.rank(qs[(60 + i) % len(qs)], cands, cache=False), 50)
    # CACHE OFF with every candidate n-gram resident (atom cache 24,000): the pure cost of re-encoding the candidates
    m._rank_state()["enc"] = _bounded_ngram(cap=24000)
    for q in qs[:5]:
        m.rank(q, cands, cache=False)
    arm("warm_cache_off_all_atoms_resident", lambda i: m.rank(qs[(60 + i) % len(qs)], cands, cache=False), calls)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="directory holding clinc150_full.json")
    ap.add_argument("--json", default=os.path.join(ROOT, "docs", "research", "evidence", "bench_rank.json"))
    ap.add_argument("--calls", type=int, default=1000)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-learning", action="store_true")
    ap.add_argument("--merge", action="store_true", help="keep the sections of an existing --json that are skipped")
    args = ap.parse_args(argv)
    t0 = time.time()
    D = load_clinc(args.data)
    out = {"config": {"data_sha256": D["sha256"], "intents": len(D["intents"]), "examples_per_candidate": N_EXAMPLES,
                      "test": len(D["test"]), "oos_test": len(D["oos"]), "seeds": SEEDS, "python": sys.version.split()[0]},
           "loadavg_start": loadavg()}
    if args.merge and args.json and os.path.exists(args.json):
        with open(args.json, encoding="utf-8") as f:
            prev = json.load(f)
        prev.update({k: v for k, v in out.items() if k not in ("latency", "quality", "learning")})
        out = prev                                        # sections not re-run keep their earlier numbers

    def save():
        # written after EVERY section: a killed run keeps what it measured
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=1, sort_keys=True)
    if not args.skip_latency:
        out["latency"] = latency(D, args.calls)          # FIRST: a fresh process, before the quality arms fill memory
        save()
    if not args.skip_quality:
        out["quality"] = quality(D, _mind())
        save()
    if not args.skip_learning:
        out["learning"] = learning(D)
    out["loadavg_end"] = loadavg()
    out["seconds"] = round(time.time() - t0, 1)
    save()
    print("done in %.0fs" % out["seconds"])
    return out


if __name__ == "__main__":
    main()

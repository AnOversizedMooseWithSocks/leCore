"""tools/bench_superposed.py -- E6.1: do SUPERPOSED frozen encoders beat their best single channel?

    python3 tools/bench_superposed.py --data DIR [--cache DIR] [--out docs/research/evidence/bench_superposed.json]

DIR holds clinc150_full.json (CLINC150, CC BY 3.0) and banking77_train.csv / banking77_test.csv (Banking77,
CC BY 4.0). They are not vendored; see docs/research/evidence/clm_panel_20260926/README.md for sources.

THE QUESTION. holographic_superposed binds several frozen text channels (character n-grams, word hashing, a
WordNet synonym channel) under unitary role keys and sums them into one hypervector, with a handful of
per-channel relevances learned GRLVQ-style. Does that one vector classify real human wording better than the
best channel alone? The acceptance (docs/BACKLOG_contrastive.md, E6.1) is "beats the best single channel, or
record which channel won" -- a kept negative is a fine outcome and is written as one.

THE PROTOCOL (the panel's w1 protocol, docs/research/evidence/clm_panel_20260926/w1-contrastive/probe_proto.py,
so the numbers sit next to the panel's):
  * every arm is a nearest-prototype classifier on the SHARED ProtoStore rule (holographic_protostore, InfoNCE on
    prototypes, tau 0.05, lr 0.3 -- the store's defaults, untouched here);
  * init: each intent's prototype = unit mean of K=5 training wordings (seeded choice);
  * stream: every other training wording once, shuffled by the seed; decide first (prequential test), then
    update from the true label -- learning from use, one pass, no epochs;
  * test: top-1 on the held-out test split with the final prototypes; AURC (area under the risk-coverage curve,
    lower is better) with the ABSOLUTE top score as the confidence (the panel measured absolute cosine to abstain
    better than any candidate-relative softmax: AURC 0.078 vs 0.094);
  * CLINC150 additionally reports AURC_oos on the E0.4 report split: in-scope test plus the sha256-odd half of
    oos_test, weighted to the 4,500 : 1,000 deployment prior (out-of-scope rows are never "right");
  * 3 seeds (seed = the K-init choice and the stream order; atoms and role keys are frozen, part of the
    encoder's identity); paired bootstrap over test questions, pooled over seeds, for every CI.

THE ARMS
  ngram / word / synonym   one channel alone (its own ProtoStore)
  superposed               uniform superposition of all channels, one ProtoStore
  superposed_rel           the SAME store, query re-weighted by relevances learned GRLVQ-style on the stream
                           (the relevance learning rate is chosen on a validation split, seed 0 only, from a
                           3-value grid; the count of configurations tried is in the JSON)
  fusion_exact             DIAGNOSTIC, not a candidate: mean of the three single-channel stores' cosines -- what
                           the superposition approximates without cross-talk, at 3x the memory

Memory: the 2-core box OOM-kills near 2 GB, so channel matrices are float32, written to --cache as .npy once and
memory-mapped (encoding CLINC150's n-gram channel costs ~70 s and ~530 MB peak; see bag_matrix).

RESULT (2026-09-26, 47 min on a shared 2-core box; every number is in the JSON) -- KEPT NEGATIVE: the n-gram
channel alone won on both datasets. CLINC150 top-1: ngram 0.9029, superposed 0.8999 (-0.30 [-0.76, +0.19]),
superposed_rel 0.9021 (-0.07 [-0.53, +0.43]), fusion_exact 0.9072 (+0.43 [+0.01, +0.95]). Banking77: ngram 0.8813,
superposed 0.8761 (-0.52 [-1.14, +0.09]), superposed_rel 0.8746 (-0.67 [-1.29, -0.08], a LOSS), fusion_exact 0.8794.
AURC ties everywhere. Configurations tried: the 3 relevance steps, nothing else tuned.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from holographic.agents_and_reasoning.holographic_protostore import ProtoStore  # noqa: E402
from holographic.agents_and_reasoning.holographic_superposed import (  # noqa: E402
    CHANNELS, ChannelRelevance, SuperposedEncoder)


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _odd(text):
    """sha256 parity: the E0.4 rule for carving a split without a seed (insertion-stable, reproducible)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 2 == 1


def load(ds, data):
    """-> {train: {intent: [texts]}, val: [(t, i)], test: [(t, i)], oos: [texts] (report half), files: {name: sha}}.
    Banking77 has no official val split: 1 in 10 training rows by sha256(text) % 10 == 0 become val (the E0.1
    'hash-carved val split'), and never appear in init or stream."""
    if ds == "clinc150":
        fn = os.path.join(data, "clinc150_full.json")
        d = json.load(open(fn))
        tr = {}
        for t, i in d["train"]:
            tr.setdefault(i, []).append(t)
        oos = [q for q, _ in d["oos_test"] if _odd(q)]          # the REPORT half (E0.4); even half calibrates
        return {"train": tr, "val": [tuple(x) for x in d["val"]], "test": [tuple(x) for x in d["test"]],
                "oos": oos, "files": {"clinc150_full.json": _sha(fn)}}
    if ds == "banking77":
        def rows(name):
            with open(os.path.join(data, name), newline="") as f:
                return [(r["text"], r["category"]) for r in csv.DictReader(f)]
        tr, val = {}, []
        for t, i in rows("banking77_train.csv"):
            if int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16) % 10 == 0:
                val.append((t, i))
            else:
                tr.setdefault(i, []).append(t)
        return {"train": tr, "val": val, "test": rows("banking77_test.csv"), "oos": [],
                "files": {n: _sha(os.path.join(data, n)) for n in ("banking77_train.csv", "banking77_test.csv")}}
    raise ValueError(ds)


# ---------------------------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------------------------
def aurc(conf, ok, w=None):
    """Area under the risk-coverage curve: serve questions in descending confidence; at every prefix the risk is
    the (weighted) error rate among those served; AURC is the mean risk over the curve. 0 is perfect; a random
    ranking gives about the overall error rate. Stable sort, so ties keep row order (deterministic)."""
    conf, ok = np.asarray(conf, float), np.asarray(ok, bool)
    w = np.ones(len(conf)) if w is None else np.asarray(w, float)
    o = np.argsort(-conf, kind="stable")
    cw = np.cumsum(w[o])
    cr = np.cumsum((ok * w)[o])
    return float(np.sum((1.0 - cr / cw) * w[o]) / cw[-1])


def boot_ci(stat, n, B, seed=0):
    """Percentile bootstrap over question indices: stat(idx) -> float, B resamples, (2.5, 97.5) percentiles."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, size=n)) for _ in range(int(B))])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------------------------------------------------------------------------------------------------------------
# one learning run
# ---------------------------------------------------------------------------------------------------------------
def run_store(M, labels, init, stream, y_stream, dim, per_channel=None, rel_lr=None):
    """Seed a ProtoStore from `init` rows and stream the rest once (decide, then update).

    M           : (N, dim) query matrix for this arm (a bound channel, or the uniform superposition)
    per_channel : optional list of (N, dim) bound channel matrices -> also learn ChannelRelevance on the same
                  verdicts and decide the relevance-weighted arm prequentially (sharing the store)
    Returns (store, relevance or None, prequential accuracies {arm: acc})."""
    st = ProtoStore(dim)
    for lab in labels:
        st.add_option(lab, np.asarray(M[init[lab]], dtype=np.float64))
    rel = ChannelRelevance([str(i) for i in range(len(per_channel))], lr=rel_lr) if per_channel else None
    hit_u = hit_r = 0
    for n, (i, y) in enumerate(zip(stream, y_stream)):
        q = np.asarray(M[i], dtype=np.float64)
        s = st.P @ q
        hit_u += int(np.argmax(s)) == y
        if rel is not None:
            Bq = np.stack([np.asarray(C[i], dtype=np.float64) for C in per_channel])
            S = Bq @ st.P.T                                   # (C, K) per-channel scores, one mat-mul
            r = rel.observe(S, y)                             # decides with the CURRENT lam, then learns
            hit_r += int(r["correct"])
        st.update(q, labels[y])
    k = max(len(stream), 1)
    pre = {"uniform": hit_u / k}
    if rel is not None:
        pre["relevance"] = hit_r / k
    return st, rel, pre


def evaluate(Q, P, y):
    """Top-1 correctness and the absolute top cosine (the AURC confidence) of query rows Q against prototypes P."""
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    S = Qn @ P.T
    pred = np.argmax(S, axis=1)
    return pred == np.asarray(y), S.max(axis=1)


def weighted_query(Bs, idx, lam):
    """sum_c lam_c * B_c[idx] -- the relevance-weighted superposition of rows idx."""
    out = np.zeros((len(idx), Bs[0].shape[1]))
    for w, B in zip(lam, Bs):
        out += float(w) * np.asarray(B[idx], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------------------------------------------
# the bench
# ---------------------------------------------------------------------------------------------------------------
def encode_cached(enc, ds, texts, cache):
    """Bound channel matrices (float32) for every text, cached as .npy keyed by content, then memory-mapped.
    Bound, not raw: a unitary role is an isometry (measured max inner-product error 3e-7 in float32), so a
    single-channel arm gives the same scores on bound rows -- one copy of each channel serves every arm."""
    key = hashlib.sha256(json.dumps([ds, enc.dim, enc.seed, texts]).encode("utf-8")).hexdigest()[:16]
    out = {}
    for ch in enc.channels:
        fn = os.path.join(cache, "%s_%s_%d_%s.npy" % (ds, ch, enc.dim, key))
        if not os.path.exists(fn):
            t0 = time.time()
            M = enc.matrix(ch, texts)
            B = enc.bind_rows(ch, M)
            del M
            np.save(fn + ".tmp.npy", B)
            os.replace(fn + ".tmp.npy", fn)
            print("  encoded %s/%s in %.1fs" % (ds, ch, time.time() - t0), flush=True)
            del B
        out[ch] = np.load(fn, mmap_mode="r")
    fn = os.path.join(cache, "%s_superposed_%d_%s.npy" % (ds, enc.dim, key))
    if not os.path.exists(fn):
        U = np.zeros((len(texts), enc.dim), dtype=np.float32)
        for ch in enc.channels:
            U += out[ch]
        U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        np.save(fn + ".tmp.npy", U)
        os.replace(fn + ".tmp.npy", fn)
        del U
    out["superposed"] = np.load(fn, mmap_mode="r")
    return out


def bench_dataset(ds, args, enc):
    t_ds = time.time()
    D = load(ds, args.data)
    labels = sorted(D["train"])
    if args.max_intents:
        labels = labels[:args.max_intents]
    L = {l: i for i, l in enumerate(labels)}
    train_rows = [(t, l) for l in labels for t in D["train"][l]]
    val = [(t, l) for t, l in D["val"] if l in L]
    test = [(t, l) for t, l in D["test"] if l in L]
    texts = [t for t, _ in train_rows] + [t for t, _ in val] + [t for t, _ in test] + list(D["oos"])
    n_tr, n_va, n_te = len(train_rows), len(val), len(test)
    ix_val = np.arange(n_tr, n_tr + n_va)
    ix_test = np.arange(n_tr + n_va, n_tr + n_va + n_te)
    ix_oos = np.arange(n_tr + n_va + n_te, len(texts))
    y_val = np.array([L[l] for _, l in val])
    y_test = np.array([L[l] for _, l in test])
    print("[%s] %d intents, train %d, val %d, test %d, oos(report half) %d" %
          (ds, len(labels), n_tr, n_va, n_te, len(ix_oos)), flush=True)
    M = encode_cached(enc, ds, texts, args.cache)
    chans = list(enc.channels)
    Bs = [M[c] for c in chans]
    by_label = {}
    for j, (_, l) in enumerate(train_rows):
        by_label.setdefault(l, []).append(j)

    def split(seed):
        """K-init rows per intent and the shuffled stream, from the seed alone."""
        rnd = random.Random(seed)
        init, stream = {}, []
        for l in labels:
            rows = list(by_label[l])
            rnd.shuffle(rows)
            init[l] = rows[:args.k_init]
            stream += rows[args.k_init:]
        rnd.shuffle(stream)
        if args.max_stream:
            stream = stream[:args.max_stream]
        return init, stream, [L[train_rows[j][1]] for j in stream]

    # ---- relevance learning rate: chosen on VAL, seed 0 only (the grid size is reported as configs tried) ----
    init0, stream0, ys0 = split(0)
    lr_val = {}
    for lr in args.rel_lrs:
        st, rel, _ = run_store(M["superposed"], labels, init0, stream0, ys0, enc.dim, per_channel=Bs, rel_lr=lr)
        ok, _ = evaluate(weighted_query(Bs, ix_val, rel.lam), st.P, y_val)
        lr_val[str(lr)] = {"val_top1": float(ok.mean()), "lam": rel.weights}
        print("  [val] rel_lr %g: top-1 %.4f lam %s" % (lr, ok.mean(), {c: round(float(v), 3) for c, v in
                                                                         zip(chans, rel.lam)}), flush=True)
    best_lr = max(args.rel_lrs, key=lambda lr: (lr_val[str(lr)]["val_top1"], -lr))   # tie -> the smaller step

    arms = chans + ["superposed", "superposed_rel", "fusion_exact"]
    per_seed = {a: {"ok": [], "conf": [], "conf_oos": [], "preq": []} for a in arms}
    lams, timing = [], {}
    for seed in args.seeds:
        t0 = time.time()
        init, stream, ys = split(seed)
        stores = {}
        for c, B in zip(chans, Bs):
            st, _, pre = run_store(B, labels, init, stream, ys, enc.dim)
            stores[c] = st
            ok, cf = evaluate(np.asarray(B[ix_test], dtype=np.float64), st.P, y_test)
            per_seed[c]["ok"].append(ok)
            per_seed[c]["conf"].append(cf)
            per_seed[c]["preq"].append(pre["uniform"])
            if len(ix_oos):
                per_seed[c]["conf_oos"].append(evaluate(np.asarray(B[ix_oos], dtype=np.float64), st.P,
                                                        np.zeros(len(ix_oos), int))[1])
        st, rel, pre = run_store(M["superposed"], labels, init, stream, ys, enc.dim, per_channel=Bs, rel_lr=best_lr)
        lams.append({c: float(v) for c, v in zip(chans, rel.lam)})
        for arm, Q, Qo, pq in (
                ("superposed", np.asarray(M["superposed"][ix_test], dtype=np.float64),
                 np.asarray(M["superposed"][ix_oos], dtype=np.float64) if len(ix_oos) else None, pre["uniform"]),
                ("superposed_rel", weighted_query(Bs, ix_test, rel.lam),
                 weighted_query(Bs, ix_oos, rel.lam) if len(ix_oos) else None, pre["relevance"])):
            ok, cf = evaluate(Q, st.P, y_test)
            per_seed[arm]["ok"].append(ok)
            per_seed[arm]["conf"].append(cf)
            per_seed[arm]["preq"].append(pq)
            if Qo is not None:
                per_seed[arm]["conf_oos"].append(evaluate(Qo, st.P, np.zeros(len(ix_oos), int))[1])
        # fusion_exact: mean of the single stores' cosines, no superposition (diagnostic)
        Sf = sum((np.asarray(B[ix_test], dtype=np.float64) @ stores[c].P.T) for c, B in zip(chans, Bs)) / len(chans)
        per_seed["fusion_exact"]["ok"].append(np.argmax(Sf, axis=1) == y_test)
        per_seed["fusion_exact"]["conf"].append(Sf.max(axis=1))
        per_seed["fusion_exact"]["preq"].append(None)
        if len(ix_oos):
            So = sum((np.asarray(B[ix_oos], dtype=np.float64) @ stores[c].P.T) for c, B in zip(chans, Bs)) / len(chans)
            per_seed["fusion_exact"]["conf_oos"].append(So.max(axis=1))
        timing["seed%d_s" % seed] = round(time.time() - t0, 1)
        print("  seed %d done in %.0fs: %s | lam %s" % (
            seed, time.time() - t0,
            " ".join("%s %.4f" % (a, per_seed[a]["ok"][-1].mean()) for a in arms),
            {c: round(v, 3) for c, v in lams[-1].items()}), flush=True)

    # ---- summary with paired bootstrap CIs pooled over seeds ----
    n = len(y_test)
    n_oos = len(ix_oos)
    w_oos = (1000.0 / max(n_oos, 1)) * (4500.0 / n) if n_oos else None   # E0.4 deployment prior 4,500 : 1,000

    def top1(a, idx):
        return float(np.mean([ok[idx].mean() for ok in per_seed[a]["ok"]]))

    def au(a, idx):
        return float(np.mean([aurc(cf[idx], ok[idx]) for ok, cf in zip(per_seed[a]["ok"], per_seed[a]["conf"])]))

    def au_oos(a, idx, jdx):
        vals = []
        for ok, cf, co in zip(per_seed[a]["ok"], per_seed[a]["conf"], per_seed[a]["conf_oos"]):
            conf = np.concatenate([cf[idx], co[jdx]])
            okk = np.concatenate([ok[idx], np.zeros(len(jdx), bool)])
            w = np.concatenate([np.ones(len(idx)), np.full(len(jdx), w_oos)])
            vals.append(aurc(conf, okk, w))
        return float(np.mean(vals))

    full = np.arange(n)
    res = {}
    for a in arms:
        r = {"top1": top1(a, full), "top1_seeds": [float(ok.mean()) for ok in per_seed[a]["ok"]],
             "top1_ci": boot_ci(lambda idx, a=a: top1(a, idx), n, args.boot),
             "aurc": au(a, full), "aurc_seeds": [aurc(cf, ok) for ok, cf in zip(per_seed[a]["ok"], per_seed[a]["conf"])],
             "aurc_ci": boot_ci(lambda idx, a=a: au(a, idx), n, args.boot),
             "prequential": per_seed[a]["preq"]}
        if n_oos:
            r["aurc_oos"] = au_oos(a, full, np.arange(n_oos))
        res[a] = r
    best_single = max(chans, key=lambda c: (res[c]["top1"], -chans.index(c)))
    diffs = {}
    rng_seed = 1
    for a in ("superposed", "superposed_rel", "fusion_exact"):
        d = {"vs": best_single,
             "top1_diff": res[a]["top1"] - res[best_single]["top1"],
             "top1_diff_ci": boot_ci(lambda idx, a=a: top1(a, idx) - top1(best_single, idx), n, args.boot, rng_seed),
             "aurc_diff": res[a]["aurc"] - res[best_single]["aurc"],
             "aurc_diff_ci": boot_ci(lambda idx, a=a: au(a, idx) - au(best_single, idx), n, args.boot, rng_seed)}
        if n_oos:
            rng = np.random.default_rng(rng_seed)
            vals = []
            for _ in range(int(args.boot)):
                idx = rng.integers(0, n, size=n)
                jdx = rng.integers(0, n_oos, size=n_oos)
                vals.append(au_oos(a, idx, jdx) - au_oos(best_single, idx, jdx))
            d["aurc_oos_diff"] = res[a]["aurc_oos"] - res[best_single]["aurc_oos"]
            d["aurc_oos_diff_ci"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        d["verdict_top1"] = ("WIN" if d["top1_diff_ci"][0] > 0 else "LOSS" if d["top1_diff_ci"][1] < 0 else "TIE (CI spans 0)")
        d["verdict_aurc"] = ("WIN" if d["aurc_diff_ci"][1] < 0 else "LOSS" if d["aurc_diff_ci"][0] > 0 else "TIE (CI spans 0)")
        diffs[a] = d
    return {"intents": len(labels), "n_train": n_tr, "n_val": n_va, "n_test": n, "n_oos_report": n_oos,
            "stream_len": len(split(args.seeds[0])[1]), "rel_lr_grid": lr_val, "rel_lr_chosen": best_lr,
            "arms": res, "best_single_channel": best_single, "vs_best_single": diffs, "lam_final": lams,
            "timing_s": dict(timing, total=round(time.time() - t_ds, 1)), "files": D["files"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default=None, help="where channel matrices are cached (default: <data>/sp_cache)")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "research", "evidence", "bench_superposed.json"))
    ap.add_argument("--datasets", nargs="+", default=["clinc150", "banking77"])
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--k-init", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rel-lrs", type=float, nargs="+", default=[0.001, 0.01, 0.05])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--max-intents", type=int, default=0, help="smoke runs only: first N intents")
    ap.add_argument("--max-stream", type=int, default=0, help="smoke runs only: cap the stream")
    args = ap.parse_args(argv)
    args.cache = args.cache or os.path.join(args.data, "sp_cache")
    os.makedirs(args.cache, exist_ok=True)
    enc = SuperposedEncoder(channels=CHANNELS, dim=args.dim, seed=0)
    t0 = time.time()
    out = {"bench": "E6.1 superposed frozen encoders", "command": "python3 tools/bench_superposed.py " + " ".join(argv or sys.argv[1:]),
           "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "machine": {"python": platform.python_version(), "numpy": np.__version__, "cpus": os.cpu_count()},
           "encoder": enc.describe(),
           "protocol": {"k_init": args.k_init, "seeds": args.seeds, "store": "ProtoStore defaults (tau 0.05, lr 0.3)",
                        "confidence": "absolute top cosine", "boot": args.boot,
                        "aurc_oos": "CLINC only: test in-scope + sha256-odd half of oos_test, prior 4500:1000",
                        "configs_tried": {"rel_lr_grid": args.rel_lrs, "note": "rel_lr chosen on val, seed 0"},
                        "smoke": bool(args.max_intents or args.max_stream)},
           "datasets": {}}
    for ds in args.datasets:
        out["datasets"][ds] = bench_dataset(ds, args, enc)
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote", args.out)
    for ds, r in out["datasets"].items():
        print("\n== %s (best single channel: %s)" % (ds, r["best_single_channel"]))
        for a, v in r["arms"].items():
            print("  %-15s top-1 %.4f [%.4f, %.4f]  AURC %.4f [%.4f, %.4f]%s" % (
                a, v["top1"], v["top1_ci"][0], v["top1_ci"][1], v["aurc"], v["aurc_ci"][0], v["aurc_ci"][1],
                ("  AURC_oos %.4f" % v["aurc_oos"]) if "aurc_oos" in v else ""))
        for a, d in r["vs_best_single"].items():
            print("  %s - %s: top-1 %+.4f %s -> %s | AURC %+.4f %s -> %s" % (
                a, d["vs"], d["top1_diff"], d["top1_diff_ci"], d["verdict_top1"], d["aurc_diff"], d["aurc_diff_ci"],
                d["verdict_aurc"]))
    return out


if __name__ == "__main__":
    main()

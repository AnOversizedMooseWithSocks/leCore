#!/usr/bin/env python3
"""bench_curves.py -- E7.1 learning curves: how much does a door learn per verdict, and what does dimension buy?

WHY THIS EXISTS (backlog E7.1, docs/BACKLOG_contrastive.md)
------------------------------------------------------------
CLM reports its scaling as tokens-per-parameter. leCore's analogue is VERDICTS PER ROW: a door's prototypes learn
only from labelled verdicts (a teacher's, a verifier's, an outcome's), so the useful question is how top-1, safe
coverage and model calls move as each row collects 1, 2, 5, 10, 20, 50 verdicts -- and whether a wider hypervector
(d = 1,024 ... 8,192) changes the slope or only the intercept.

PROTOCOL (CLINC150, 150 intents; 3 seeds; everything through the shared E0.4 gate protocol)
  * rows: each intent's row starts from ONE wording (its first verdict creates it -- the meaning loop's cold start);
    wordings per intent are a seeded permutation of its 100 training wordings
  * stream: rounds 2..50, one new verdict per row per round, rounds shuffled (seeded); checkpoints after 1, 2, 5,
    10, 20, 50 verdicts per row
  * arms: "infonce" = the shared rule, ProtoStore(d, tau 0.05, lr 0.3) -- and "centroid" = the positive-only rule
    (every confirmed wording joins its row, MeaningIndex.link's behaviour), the curve InfoNCE must beat
  * encoder: SystemOne's hashed n-grams. One raw encoding at d = 8,192 serves every d: numpy's Generator yields the
    same first d normals whatever length is drawn, so the first d columns ARE the d-dimensional encoding
  * metrics at each checkpoint (report split = CLINC test + sha256 half B of oos_test; calibration = a sha256 half of
    val + oos half A; precision under the 4,500:1,000 prior):
      top1              in-scope top-1 on the full test set (4,500)
      coverage_w1.5     in-scope served-and-right at the threshold calibrated for <= 1.5% served-wrong
      wrong_all / oos   what that threshold realises on the report split (wrong share of all questions, oos served)
      calls_per_1000    the meaning-style loop: every question the gate does not serve goes to the model, so
                        model calls per 1,000 questions = 1000 x (1 - served share) at that threshold
      aurc              threshold-free
  * fit: top-1 ERROR vs verdicts per row, err(n) = c + a * n^(-b), per (arm, d, seed); the seed spread of b is the
    honest error bar on "how fast", and n* = verdicts per row to reach top-1 0.80 is the tokens-per-parameter
    analogue reported per dimension.

WHAT IT MEASURED (2026-09-26, docs/research/evidence/bench_curves.json; means over 3 seeds, d = 2048 unless named)
  * InfoNCE's top-1 error is a clean power law in verdicts per row: b = 0.466 [0.462, 0.469] across seeds, no floor
    within 50 verdicts; top-1 0.346 / 0.623 / 0.750 / 0.829 / 0.885 at 1 / 5 / 10 / 20 / 50. The tokens-per-parameter
    analogue: 15.9 verdicts per row reach top-1 0.80 at d = 2048 (17.6 at 1,024, 14.7 at 4,096, 14.5 at 8,192).
  * Dimension buys a little and saturates: 0.877 -> 0.885 -> 0.889 -> 0.891 at 50 verdicts from d = 1,024 to 8,192.
  * The positive-only centroid (MeaningIndex.link's rule) plateaus: floor ~0.135 error, 0.807 at 50 verdicts, n* 38.
    At 50 verdicts InfoNCE serves 62.9% at <= 1.5% wrong vs 44.1%, i.e. 469 vs 625 model calls per 1,000.
  * KEPT NEGATIVE: below ~10 verdicts per row InfoNCE (lr 0.3) LOSES to the centroid -- 0.623 vs 0.663 top-1 and
    20.7% vs 27.2% safe coverage at 5 verdicts (a one-wording row learns faster by adding the whole wording than by
    lr * (1 - p) of it); the crossover is at ~10. A young-row learning rate, not a curriculum temperature, is the
    lever this points at (untested here).
  * LOUD: the threshold calibrated for <= 1.5% served-wrong realises 1.4%-1.8% on the report split (the val -> test
    gap E0.4 exists to expose); calls_per_1000 is read at that realised operating point.
  * Cost: 9 min 21 s of CPU (23 min 45 s wall on a box loaded 3-5x by other work); the d = 8,192 InfoNCE runs dominate
    because tau 0.05 over 150 rows touches almost every row on every verdict.

Usage (~10 min of CPU on 2 cores, peak ~0.6 GB):
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python3 tools/bench_curves.py --data DIR [--dims 1024,2048,4096,8192]
Writes docs/research/evidence/bench_curves.json. numpy + stdlib only; deterministic.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tools import bench_contrastive as bc  # noqa: E402  (the one harness: loaders, encoder, gate protocol)

OUT_JSON = os.path.join(ROOT, "docs", "research", "evidence", "bench_curves.json")
CHECKPOINTS = (1, 2, 5, 10, 20, 50)
VAL_SALT = "bench_curves/val-half/v1:"


def fit_power(ns, err):
    """err(n) = c + a * n^(-b), least squares in err-space: a grid over the floor c in [0, min err), and for each c
    a linear fit of log(err - c) on log n. -> {a, b, c, r2}. Six points, three parameters: read b's SEED SPREAD,
    not its third digit."""
    ns, err = np.asarray(ns, float), np.asarray(err, float)
    X = np.vstack([np.ones_like(ns), -np.log(ns)]).T
    best = None
    for c in np.linspace(0.0, max(float(err.min()) - 1e-4, 0.0), 200):
        y = np.log(np.maximum(err - c, 1e-9))
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = c + np.exp(coef[0]) * ns ** (-coef[1])
        sse = float(((pred - err) ** 2).sum())
        if best is None or sse < best[0] - 1e-15:
            best = (sse, float(c), float(np.exp(coef[0])), float(coef[1]))
    sse, c, a, b = best
    tot = float(((err - err.mean()) ** 2).sum())
    return {"a": a, "b": b, "c": c, "r2": 1.0 - sse / tot if tot > 0 else 1.0}


def n_to_reach(fit, target_err):
    """Verdicts per row the fitted curve needs to reach an error target (inf if its floor c is above it)."""
    if fit["c"] >= target_err or fit["b"] <= 0:
        return float("inf")
    return float((fit["a"] / (target_err - fit["c"])) ** (1.0 / fit["b"]))


def _eval(Eraw, enorm, d, P, sl, yv, yt, oos_texts):
    """Score the eval block (raw rows, first d columns) against unit prototypes P; -> gate_protocol dict."""
    S = (Eraw[:, :d] @ P.T.astype(np.float32)) / enorm[:, None]
    srt = -np.sort(-S, axis=1)
    g = 2 * srt[:, 0] - srt[:, 1]
    top = np.argmax(S, axis=1)
    v, t, o = sl
    lay = bc.assemble(g[v], top[v] == yv, g[t], top[t] == yt, g[o], oos_texts)
    return bc.gate_protocol(*lay, targets=(0.95,), max_wrong=(0.015,), oracle=False)


def _metrics(gp):
    w = gp["calibrated"]["W0.015"]
    return {"top1": gp["top1"], "aurc": gp["aurc"], "coverage_w1.5": w["coverage"], "wrong_all": w["wrong_all"],
            "oos_served": w["oos_served"], "calls_per_1000": 1000.0 * (1.0 - w["served_all"]),
            "coverage_P0.95": gp["calibrated"]["P0.95"]["coverage"]}


def run(args):
    from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
    t0 = time.time()
    dims = [int(x) for x in args.dims.split(",")]
    dmax = max(dims)
    clinc = bc.load_clinc(args.data)
    labels = sorted(clinc["train"])
    L = {l: i for i, l in enumerate(labels)}
    C = len(labels)
    val = [(t, i) for t, i in clinc["val"] if bc.hash_fraction(t, VAL_SALT) < 0.5]
    test, oos = clinc["test"], clinc["oos_test"]
    eval_texts = [t for t, _ in val] + [t for t, _ in test] + oos
    sl = (np.arange(len(val)), np.arange(len(val), len(val) + len(test)),
          np.arange(len(val) + len(test), len(eval_texts)))
    yv = np.array([L[i] for _, i in val])
    yt = np.array([L[i] for _, i in test])
    # ONE raw encoding at d_max (float32 storage; accumulation is float64 per chunk), every d is a column prefix
    Eraw = bc.encode_texts(eval_texts, dim=dmax, chunk=1500, dtype=np.float32, normalise=False, cache=args.cache)
    enorms = {}
    for d in dims:                                   # row norms of each prefix, in chunks (no float64 copy of Eraw)
        nn = np.empty(len(Eraw), np.float32)
        for a in range(0, len(Eraw), 1000):
            blk = Eraw[a:a + 1000, :d].astype(np.float64)
            nn[a:a + 1000] = np.sqrt((blk * blk).sum(1)) + 1e-12
        enorms[d] = nn
    print("eval encoded: %d texts at d=%d (%.0fs)" % (len(eval_texts), dmax, time.time() - t0), flush=True)
    per = {}            # (arm, d) -> {seed: {n: metrics}}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        words = {l: [clinc["train"][l][j] for j in rng.permutation(len(clinc["train"][l]))[:max(CHECKPOINTS)]]
                 for l in labels}
        rounds = []
        for r in range(1, max(CHECKPOINTS)):
            rr = [(words[l][r], L[l]) for l in labels]
            order = rng.permutation(len(rr))
            rounds.append([rr[k] for k in order])
        seed_texts = [words[l][0] for l in labels]
        stream = [x for rr in rounds for x in rr]
        Sraw = bc.encode_texts(seed_texts + [t for t, _ in stream], dim=dmax, chunk=1500, dtype=np.float32,
                               normalise=False, cache=args.cache)
        print("seed %d stream encoded (%.0fs)" % (seed, time.time() - t0), flush=True)
        ys = np.array([y for _, y in stream])
        for d in dims:
            # the shared rule: ProtoStore normalises every query itself, so raw prefixes go straight in
            st = ProtoStore(d, tau=0.05, lr=0.3, name="curves", mine=False)
            A = np.zeros((C, d))
            for c in range(C):
                st.add_option(labels[c], Sraw[c:c + 1, :d])
                v = Sraw[c, :d].astype(np.float64)
                A[c] = v / np.linalg.norm(v)
            res_inf, res_cen = {}, {}
            done = 0
            for n in CHECKPOINTS:
                upto = (n - 1) * C                      # verdicts after seeding needed for n per row
                for k in range(done, upto):
                    q = Sraw[C + k, :d]
                    st.update(q, labels[int(ys[k])])
                    qq = q.astype(np.float64)
                    A[ys[k]] += qq / np.linalg.norm(qq)   # positive-only: the confirmed wording joins its row
                done = upto
                res_inf[n] = _metrics(_eval(Eraw, enorms[d], d, st.P, sl, yv, yt, oos))
                Pc = A / np.linalg.norm(A, axis=1, keepdims=True)
                res_cen[n] = _metrics(_eval(Eraw, enorms[d], d, Pc, sl, yv, yt, oos))
            per.setdefault(("infonce", d), {})[seed] = res_inf
            per.setdefault(("centroid", d), {})[seed] = res_cen
            print("seed %d d=%d top1 infonce %s | centroid %s (%.0fs)" % (
                seed, d, [round(res_inf[n]["top1"], 3) for n in CHECKPOINTS],
                [round(res_cen[n]["top1"], 3) for n in CHECKPOINTS], time.time() - t0), flush=True)
        del Sraw
    # ---------------------------------------------------------------- aggregate + fits
    curves, fits = {}, {}
    for (arm, d), bys in per.items():
        cd = {}
        for n in CHECKPOINTS:
            cd[str(n)] = {}
            for m in bys[args.seeds[0]][n]:
                vals = [bys[s][n][m] for s in args.seeds]
                cd[str(n)][m] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                                 "max": float(np.max(vals))}
        curves.setdefault(arm, {})[str(d)] = cd
        fs = [fit_power(CHECKPOINTS, [1.0 - bys[s][n]["top1"] for n in CHECKPOINTS]) for s in args.seeds]
        nstar = [n_to_reach(f, 0.20) for f in fs]
        fits.setdefault(arm, {})[str(d)] = {
            "per_seed": fs, "b_mean": float(np.mean([f["b"] for f in fs])),
            "b_min": float(np.min([f["b"] for f in fs])), "b_max": float(np.max([f["b"] for f in fs])),
            "c_mean": float(np.mean([f["c"] for f in fs])),
            "verdicts_per_row_for_top1_0.80": {"per_seed": nstar, "median": float(np.median(nstar))}}
    return {"command": "PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python3 tools/bench_curves.py --data DATA --dims %s"
                       % args.dims,
            "inputs_sha256": bc.pinned_inputs(args.data), "python": sys.version.split()[0], "numpy": np.__version__,
            "protocol": __doc__.split("PROTOCOL", 1)[1].split("Usage", 1)[0].strip(),
            "dims": dims, "verdicts_per_row": list(CHECKPOINTS), "seeds": list(args.seeds),
            "n_eval": {"val_half_in": len(val), "test_in": len(test), "oos_test": len(oos)},
            "curves": curves, "fits": fits, "configs_tried": 2 * len(dims), "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--dims", default="1024,2048,4096,8192")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args(argv)
    args.seeds = [int(x) for x in args.seeds.split(",")]
    res = run(args)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1, default=bc._json_default)
    os.replace(tmp, args.out)
    for arm, byd in res["fits"].items():
        for d, f in byd.items():
            print("%-8s d=%-5s b %.3f [%.3f, %.3f]  floor c %.3f  n*(top1 0.80) %s" % (
                arm, d, f["b_mean"], f["b_min"], f["b_max"], f["c_mean"],
                f["verdicts_per_row_for_top1_0.80"]["median"]))
    print("done in %.0fs -> %s" % (res["seconds"], args.out))


if __name__ == "__main__":
    main()

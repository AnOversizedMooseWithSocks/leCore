"""systemone_stream_bench.py -- honest measurement of the CLOSED decision loop (sweep 172).

Three experiments, all on the cached real datasets (tools/_agnews_cache.json,
tools/_sst2_cache.json -- run tools/systemone_bench.py once with network to build them;
this script REFUSES without them rather than inventing data):

  1. PREQUENTIAL (test-then-train) on AG News: does the AdaptHD miss-update beat frozen
     prototypes when both start from the same k exemplars? lr=0 IS the frozen baseline --
     same code path, learning off -- so the comparison cannot be a strawman.
  2. DRIFT LOCATION: fit on AG News, then stream 150 in-domain states followed by 150
     off-domain SST-2 states (a real covariate shift built from two real datasets, schedule
     disclosed). The label-FREE support channel must place a boundary near row 150 --
     the alarm that fires before any label arrives.
  3. RISK-COVERAGE (the Cranmer seat's ask): a single accuracy number hides what an
     abstaining system buys; sweep the calibrated-p threshold and report
     (coverage, accuracy-on-answered) pairs.

Run:  PYTHONHASHSEED=0 python3 tools/systemone_stream_bench.py [--seeds 3] [--k 32]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HERE = os.path.dirname(os.path.abspath(__file__))
AG = os.path.join(_HERE, "_agnews_cache.json")
SST = os.path.join(_HERE, "_sst2_cache.json")
AG_LABELS = ["world", "sports", "business", "sci-tech"]


def _need(path):
    if not os.path.exists(path):
        raise SystemExit("REFUSED: missing %s -- run tools/systemone_bench.py once with network "
                         "to build the caches; invented data would make every number a lie." % path)
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n-stream", type=int, default=300)
    args = ap.parse_args()

    from holographic.agents_and_reasoning.holographic_systemone import (
        SystemOne, hashed_ngram_encode)
    ag, sst = _need(AG), _need(SST)
    base = hashed_ngram_encode(dim=2048)
    memo = {}
    enc = lambda t: memo.setdefault(t, base(t))   # one encode per distinct text, all experiments

    by = {y: [t for t, yy in ag["train"] if yy == y] for y in range(4)}

    def fitted(seed, min_support=None):
        """Same exemplar budget every condition: k per class, drawn by seed."""
        rng = np.random.default_rng(seed)
        ex = {AG_LABELS[y]: [by[y][i] for i in rng.permutation(len(by[y]))[:args.k]]
              for y in range(4)}
        so = SystemOne(enc, margin=0.02, min_support=min_support)
        so.fit({"topic": {"type": "choice", "options": AG_LABELS, "examples": ex}})
        return so

    # ---- 1. prequential: frozen (lr=0) vs online, same exemplars, same stream order ----
    print("1) PREQUENTIAL on AG News eval, n=%d, k=%d/class, %d seeds "
          "(lr=0 is the frozen baseline -- same code path, learning off):"
          % (args.n_stream, args.k, args.seeds))
    for lr in (0.0, 0.3, 1.0):
        accs = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            rows = [ag["eval"][i] for i in rng.permutation(len(ag["eval"]))[:args.n_stream]]
            so = fitted(seed)
            hit = [so.observe(t, {"topic": AG_LABELS[y]}, lr=lr)["topic"]["was_correct"]
                   for t, y in rows]
            accs.append(float(np.mean(hit)))
        print("   lr=%-4g prequential acc %.3f (spread %.3f)"
              % (lr, np.mean(accs), np.max(accs) - np.min(accs)))

    # ---- 2. drift location: AG in-domain then SST-2 off-domain, support channel ----
    so = fitted(0)
    for t, _ in ag["eval"][:150]:
        so.decide(t)
    for t, _ in sst["eval"][:150]:
        so.decide(t)                         # off-domain: no labels needed, support says it all
    rep = so.drift_report()["topic"]["support"]
    s = so._support["topic"]
    print("2) DRIFT (150 AG News rows then 150 SST-2 rows; true shift at 150):")
    print("   support channel: drift=%s boundaries=%s | segment support means: %s"
          % (rep["drift"], rep["boundaries"],
             ["%.3f" % np.mean(s[a:b]) for a, b in
              zip([0] + rep["boundaries"], rep["boundaries"] + [len(s)])]))

    # ---- 3. risk-coverage: sweep the calibrated-p threshold ----
    rng = np.random.default_rng(7)
    cal = []
    for y in range(4):
        idx = rng.permutation(len(by[y]))[args.k:args.k + 16]
        cal += [(by[y][i], {"topic": AG_LABELS[y]}) for i in idx]
    so = fitted(7)
    so.calibrate(cal)
    ev = ag["eval"][:200]
    ans = so.decide_map([t for t, _ in ev])
    truth = [AG_LABELS[y] for _, y in ev]
    ps = np.array([a["topic"]["p"] for a in ans])
    ok = np.array([a["topic"]["ranked"][0][0] == t for a, t in zip(ans, truth)])
    print("3) RISK-COVERAGE on 200 eval rows (answer iff calibrated p >= threshold):")
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8):
        m_ = ps >= thr
        cov = float(m_.mean())
        acc = float(ok[m_].mean()) if m_.any() else float("nan")
        print("   p>=%.1f  coverage %.3f  accuracy-on-answered %.3f" % (thr, cov, acc))


if __name__ == "__main__":
    main()

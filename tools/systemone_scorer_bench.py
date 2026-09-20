"""systemone_scorer_bench.py -- the nb scorer vs the prototype scorer, measured (sweeps 174-175).

Sweep 175 added Banking77 (77 support intents, the Jev-shaped task; tools/_banking77_cache.json) and the
Rennie transform (nb_transform=True, now the nb default): log(1+tf) * IDF-from-examples, length-normalised.

Same real caches as tools/systemone_bench.py (tools/_agnews_cache.json, tools/_sst2_cache.json;
build them once with network via systemone_bench.py). REFUSES without them: invented data would
make every number a lie.

  1. LADDER    forced accuracy vs examples-per-class, prototype vs nb vs nb+bigrams, 3 seeds.
  2. CALIBRATION POOL   nb posteriors are sharp: how many labeled outcomes before p is honest? (ECE)
  3. PREQUENTIAL   the closed loop with scorer='nb': lr=0 (frozen) vs lr=1 (count every observation).
     For the prototype path, sweep 172 measured high lr LOSING on a stationary stream; a count
     table has no such failure mode -- and that is a claim, so it is measured here, not assumed.

Run:  PYTHONHASHSEED=0 python3 tools/systemone_scorer_bench.py [--seeds 3]
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
B77 = os.path.join(_HERE, "_banking77_cache.json")   # 77 support intents: the Jev-shaped task
LAB = {4: ["world", "sports", "business", "sci-tech"], 2: ["neg", "pos"],
       77: [str(i) for i in range(77)]}


def _need(path):
    if not os.path.exists(path):
        raise SystemExit("REFUSED: missing %s -- run tools/systemone_bench.py once with network." % path)
    return json.load(open(path))


def ece(ps, ok, bins=5):
    """Expected calibration error: bin-mass-weighted |stated p - observed accuracy|."""
    ps, ok = np.asarray(ps, float), np.asarray(ok, float)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (ps >= edges[i]) & ((ps < edges[i + 1]) if i < bins - 1 else (ps <= 1.0))
        if m.any():
            e += m.mean() * abs(ps[m].mean() - ok[m].mean())
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    from holographic.agents_and_reasoning.holographic_systemone import SystemOne, hashed_ngram_encode
    ag, sst, b77 = _need(AG), _need(SST), _need(B77)
    enc = hashed_ngram_encode(dim=2048)
    memo = {}
    E = lambda t: memo.setdefault(t, enc(t))
    seeds = range(args.seeds)

    def fitted(data, C, k, seed, scorer, bigrams, transform=True):
        rng = np.random.default_rng(seed)
        by = {y: [t for t, yy in data["train"] if yy == y] for y in range(C)}
        L = LAB[C]
        perm = {y: rng.permutation(len(by[y])) for y in range(C)}
        ex = {L[y]: [by[y][i] for i in perm[y][:k]] for y in range(C)}
        so = SystemOne(E, margin=0.02, scorer=scorer, nb_bigrams=bigrams, nb_transform=transform)
        so.fit({"q": {"type": "choice", "options": L, "examples": ex}})
        return so, L, by, perm

    # ---- 1. ladder ---------------------------------------------------------------------------
    print("1) LADDER -- forced accuracy, eval 400, mean (spread) over %d seeds; identical examples per row"
          % args.seeds)
    for name, data, C, ks, nev in (("AG News", ag, 4, (32, 128, 300), 400), ("SST-2", sst, 2, (32, 128, 600), 400),
                                   ("Banking77 (77 intents)", b77, 77, (5, 10, 20, 35), 1000)):
        print("   %s" % name)
        for label, scorer, bg, xf in (("prototype", "prototype", False, True), ("nb plain", "nb", False, False),
                                      ("nb+bigrams plain", "nb", True, False), ("nb TRANSFORM", "nb", False, True)):
            cells = []
            for k in ks:
                accs = []
                for s in seeds:
                    so, L, _, _ = fitted(data, C, k, s, scorer, bg, xf)
                    ev = data["eval"][:nev]
                    ans = so.decide_map([t for t, _ in ev])
                    accs.append(np.mean([a["q"]["ranked"][0][0] == L[y] for a, (_, y) in zip(ans, ev)]))
                cells.append("k=%-3d %.3f (%.3f)" % (k, np.mean(accs), np.max(accs) - np.min(accs)))
            print("      %-17s %s%s" % (label, "  ".join(cells),
                                        "   <- sweep-171 default" if scorer == "prototype" else ""))

    # ---- 2. calibration pool -----------------------------------------------------------------
    print("2) CALIBRATION POOL -- nb, SST-2 k=600 (+bigrams): labeled outcomes needed before p is honest")
    C, L = 2, LAB[2]
    for pool in (16, 64, 150):
        rows = []
        for s in seeds:
            so, L, _, _ = fitted(sst, C, 600, s, "nb", True)
            cal = sst["eval"][200:200 + pool]              # never trained on; test on eval 0-199
            so.calibrate([(t, {"q": L[y]}) for t, y in cal])
            ev = sst["eval"][:200]
            ans = so.decide_map([t for t, _ in ev])
            ok = [a["q"]["ranked"][0][0] == L[y] for a, (_, y) in zip(ans, ev)]
            ps = [a["q"]["p"] for a in ans]
            m = np.array(ps) >= 0.7
            rows.append((ece(ps, ok), m.mean(), np.array(ok)[m].mean() if m.any() else float("nan")))
        r = np.array(rows)
        print("      pool=%3d  ECE %.3f | answer iff p>=0.7: coverage %.3f, accuracy-on-answered %.3f"
              % (pool, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean()))

    # ---- 2b. conformal answer sets (sweep 176, H1) --------------------------------------------
    print("2b) CONFORMAL SETS -- Banking77 k=20, calibration ~231 held-out train rows, eval 1000: does the guarantee hold?")
    for scorer in ("nb", "prototype"):
        for alpha in (0.05, 0.10):
            rows = []
            for s in seeds:
                so, L, by, perm = fitted(b77, 77, 20, s, scorer, False)
                cal = [(by[y][i], {"q": L[y]}) for y in range(77) for i in perm[y][20:23]]
                so.calibrate_conformal(cal, alpha=alpha)
                ev = b77["eval"][:1000]
                ans = so.decide_map([t for t, _ in ev])
                sizes = np.array([len(a["q"]["set"]) for a in ans])
                cov = np.mean([L[y] in a["q"]["set"] for a, (_, y) in zip(ans, ev)])
                rows.append((cov, sizes.mean(), (sizes == 1).mean(), (sizes <= 2).mean()))
            r = np.array(rows)
            print("      %-9s nominal %.2f: coverage %.3f (spread %.3f) | mean set %.2f | singletons %.3f | <=2 %.3f"
                  % (scorer, 1 - alpha, r[:, 0].mean(), r[:, 0].max() - r[:, 0].min(), r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean()))

    # ---- 3. prequential ----------------------------------------------------------------------
    print("3) PREQUENTIAL -- AG News, fit on k=32/class then stream 300 labeled rows, decide-then-learn")
    for scorer in ("prototype", "nb"):
        for lr in (0.0, 1.0):
            accs = []
            for s in seeds:
                so, L, _, _ = fitted(ag, 4, 32, s, scorer, False)
                rng = np.random.default_rng(1000 + s)
                rows = [ag["eval"][i] for i in rng.permutation(len(ag["eval"]))[:300]]
                hit = [so.observe(t, {"q": L[y]}, lr=lr)["q"]["was_correct"] for t, y in rows]
                accs.append(np.mean(hit))
            print("      %-10s lr=%-3g prequential accuracy %.3f (spread %.3f)%s"
                  % (scorer, lr, np.mean(accs), np.max(accs) - np.min(accs),
                     "   <- frozen baseline" if lr == 0 else ""))


if __name__ == "__main__":
    main()

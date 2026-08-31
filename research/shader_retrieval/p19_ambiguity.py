"""P1.9 (return the set, not a guess) and P1.6 (an agreement signal), judged against SOTA QPP.

T11 and the containment measurement established that R2 is ill-posed: BM25 sits AT the ceiling,
so the remaining job is not to rank better but to KNOW WHEN RANKING IS MEANINGLESS. That task has
a name and a literature -- Query Performance Prediction -- so the honest bar is the published
unsupervised baselines, not a self-invented score.

BASELINES, as defined in the QPP literature (Shtok et al. 2012; Zhou & Croft 2007;
Perez-Iglesias & Araujo 2010):
  NQC     standard deviation of the top-k retrieval scores, normalised by the corpus score
  WIG     mean of the top-k scores minus the corpus score
  sigma_max  the maximum running standard deviation over prefixes of the ranking
  margin  top1 - top2, the naive predictor everyone reaches for first
DEVIATION DECLARED: the strict definition normalises by the score of the ENTIRE CORPUS treated as
one document, which is degenerate for a single-document idf. The mean score over the full ranking
is used instead and the arms are labelled accordingly -- it is a per-query constant, so it cannot
flip a within-query comparison, but it is not the textbook normaliser and is not claimed to be.

THE CANDIDATE: exact CONTAINMENT SIZE m -- how many passages contain every query term, computed
by intersecting postings. It is not a prediction at all, it is the ground truth of ambiguity read
straight off the index, and intersecting eight postings is cheap. If a free exact signal beats
the literature's estimators, the right conclusion is that this task did not need an estimator.

GROUND TRUTH: 1/m, the Bayes ceiling from T11. Predictors are scored by rank correlation with it
AND by risk-coverage, which is what a system actually does with the number.
"""
import numpy as np
from collections import defaultdict

import hard_corpus as HC
import p01_retriever as P
import holographic.agents_and_reasoning.holographic_hashatom as HA


def kendall_tau(a, b):
    """Kendall tau-b, pure NumPy -- the QPP literature's standard correlation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    conc = disc = ta = tb = 0
    for i in range(n):
        da = a[i + 1:] - a[i]
        db = b[i + 1:] - b[i]
        s = np.sign(da) * np.sign(db)
        conc += int((s > 0).sum()); disc += int((s < 0).sum())
        ta += int(((da == 0) & (db != 0)).sum()); tb += int(((db == 0) & (da != 0)).sum())
    d = math_sqrt((conc + disc + ta) * (conc + disc + tb))
    return (conc - disc) / d if d else 0.0


def math_sqrt(x):
    return float(np.sqrt(x)) if x > 0 else 0.0


def risk_coverage(pred, correct, points=(0.2, 0.4, 0.6, 0.8, 1.0)):
    """Accuracy on the most-confident fraction. A predictor is useful only if accuracy RISES as
    coverage falls; a flat curve means the score carries no information about correctness."""
    order = np.argsort(-np.asarray(pred, float))
    c = np.asarray(correct, float)[order]
    return [float(c[:max(1, int(len(c) * p))].mean()) for p in points]


if __name__ == "__main__":
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    dn = HC.load_passages(target=800)
    docs = [t for _, t in dn]
    sets = [set(t) for t in docs]
    txt = [" ".join(t) for t in docs]
    K, D = len(docs), 1024
    V = np.stack([HA.encode_hash(t, D) for t in docs])
    inv = {}
    for i, s in enumerate(sets):
        for t in s:
            inv.setdefault(t, set()).add(i)

    vocab = sorted(inv)
    dic = {}
    for w in vocab:
        try:
            e = mind.lookup(w)
        except Exception:
            e = None
        if e and e.get("definition"):
            dic[w] = e["definition"]

    def containment(q):
        cand = None
        for t in q:
            s = inv.get(t, set())
            cand = s if cand is None else (cand & s)
            if not cand:
                break
        if cand:
            return cand
        # DEGRADE GRACEFULLY: no passage has every term, so fall back to the largest subset that
        # any passage does contain. This is what turns R3 from "empty" into "explicitly nothing".
        best = set()
        for t in q:
            s = inv.get(t, set())
            if len(s) and (not best or len(s) < len(best)):
                best = s
        return best

    rows = []
    for regime in ("R1", "R2", "R3"):
        for seed in range(3):
            for gold, q in P.build_queries(dn, sets, dic, regime, 40, seed):
                r = mind.bm25_rank(" ".join(q), txt)
                s = np.array([float(x[1]) for x in r])
                top = [int(x[0]) for x in r]
                mu = float(s.mean()) + 1e-9
                k = 50
                nqc = float(np.std(s[:k])) / mu
                wig = float(np.mean(s[:k])) - mu
                smax = max(float(np.std(s[:j])) for j in range(2, 31))
                margin = float(s[0] - s[1])
                cset = containment(q)
                m = max(1, len(cset))
                rows.append(dict(regime=regime, gold=gold, correct=int(top[0] == gold),
                                 nqc=nqc, wig=wig, smax=smax, margin=margin,
                                 inv_m=1.0 / m, m=m,
                                 in_set=int(gold in cset), setsize=len(cset)))

    print("K=%d passages, engine bm25_rank, 40 queries x 3 seeds x 3 regimes = %d queries\n"
          % (K, len(rows)))

    print("P1.6 -- DOES ANY SIGNAL TRACK THE AMBIGUITY GROUND TRUTH (1/m)?  Kendall tau")
    gt = [r["inv_m"] for r in rows]
    for name in ("nqc", "wig", "smax", "margin", "inv_m"):
        t = kendall_tau([r[name] for r in rows], gt)
        print("   %-8s tau vs 1/m = %+0.3f %s" % (name, t, "  <- the exact signal, by construction"
                                                  if name == "inv_m" else ""))

    print("\nRISK-COVERAGE: top-1 accuracy on the most-confident X%% of queries")
    print("   predictor   20%%     40%%     60%%     80%%     100%%")
    corr = [r["correct"] for r in rows]
    for name in ("nqc", "wig", "smax", "margin", "inv_m"):
        rc = risk_coverage([r[name] for r in rows], corr)
        print("   %-11s %s" % (name, "  ".join("%.3f" % v for v in rc)))

    print("\nP1.9 -- RETURN THE SET. Set-recall AND set size, together (either alone is useless).")
    print("   regime  set-recall  median |set|  mean |set|  top-1  Bayes ceiling 1/m")
    for regime in ("R1", "R2", "R3"):
        g = [r for r in rows if r["regime"] == regime]
        sz = np.array([r["setsize"] for r in g])
        print("   %-7s %-11.3f %-13d %-11.1f %-6.3f %.3f"
              % (regime, np.mean([r["in_set"] for r in g]), int(np.median(sz)), sz.mean(),
                 np.mean([r["correct"] for r in g]), np.mean([r["inv_m"] for r in g])))

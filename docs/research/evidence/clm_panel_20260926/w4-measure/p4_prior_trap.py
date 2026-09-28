"""P4: the val->test oos-prior trap. Choose the P=0.97/0.95 threshold on CLINC val WITHOUT re-weighting val's 100 oos
questions to test's prior (3.2% oos in val vs 18% in test), and see what precision test actually gets."""
import numpy as np
from common import *
mi, ri, d, st = load()
X, pn, prow, rids, A = build_index_mats(mi); nr = len(rids); row_int = np.array([ri[r] for r in rids])
cache, d = get_cache(mi)
def sig(si, so):
    out = []
    for s in (si, so):
        Q, qn = cache[s]
        R = row_scores(phrase_cos(Q, qn, X, pn), prow, nr, A["cnorm"])[0]
        t, s1, s2, h = top2(R); out.append((t, np.where(h, 2 * s1 - s2, -9)))
    (ti, gi), (to, go) = out
    ok_in = row_int[ti] == np.array([i for _, i in d[si]])
    return np.concatenate([gi, go]), np.concatenate([ok_in, np.zeros(len(to), bool)]), np.concatenate([np.ones(len(ti), bool), np.zeros(len(to), bool)])
sv, okv, iv = sig("val", "oos_val"); stt, okt, it = sig("test", "oos_test")
for P in (0.97, 0.95):
    _, thr_raw, _ = cov_at_precision(sv, okv, iv, P)            # NO re-weighting
    thr_w = val_threshold(sv, okv, iv, P)                         # re-weighted to the test prior
    for nm, thr in (("unweighted val", thr_raw), ("prior-reweighted val", thr_w)):
        r = apply_threshold(stt, okt, it, thr)
        print("P%.2f %-21s thr %.3f -> test cov %.1f%% realised precision %.3f oos served %.1f%%" % (P, nm, thr, 100 * r["cov"], r["prec"], 100 * r["oos"]))

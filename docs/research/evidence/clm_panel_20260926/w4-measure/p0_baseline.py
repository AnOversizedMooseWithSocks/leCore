"""P0: reproduce the learned index's numbers with the vectorised scorer (must match MeaningIndex.answers exactly),
then report the baseline every probe is compared with. Also audits the stored negatives for false negatives."""
import time, collections
import numpy as np
from common import *

t0 = time.time()
mi, ri, d, st = load()
X, pn, prow, rids, A = build_index_mats(mi)
nr = len(rids)
row_int = np.array([ri[r] for r in rids])
cache, d = get_cache(mi)
print("built in %.1fs; rows %d phrasings %d feats %d" % (time.time() - t0, nr, X.shape[0], X.shape[1]))

# ---- exactness check against the real code path on 300 test + 100 oos questions
Q, qn = cache["test"]
C = phrase_cos(Q[:300], qn[:300], X, pn)
R, _, _ = row_scores(C, prow, nr, A["cnorm"])
top, s1, s2, has = top2(R)
mism = 0
for i in range(300):
    a = mi.answers(d["test"][i][0], k=8)
    if not a:
        mism += int(has[i]); continue
    if rids[top[i]] != a[0][0] or abs(s1[i] - a[0][1]) > 1e-9 or abs(s2[i] - (a[1][1] if len(a) > 1 else 0)) > 1e-9:
        mism += 1
print("exactness vs MeaningIndex.answers on 300 test questions: %d mismatches" % mism)


def evaluate(split_in, split_oos, R_fn):
    Qi, qni = cache[split_in]; Qo, qno = cache[split_oos]
    Ri = R_fn(Qi, qni); Ro = R_fn(Qo, qno)
    ti, s1i, s2i, hi = top2(Ri); to, s1o, s2o, ho = top2(Ro)
    truth = np.array([i for _, i in d[split_in]])
    ok_in = hi & (row_int[ti] == truth)
    sig = np.concatenate([2 * s1i - s2i, 2 * s1o - s2o])
    ok = np.concatenate([ok_in, np.zeros(len(to), bool)])
    is_in = np.concatenate([np.ones(len(ti), bool), np.zeros(len(to), bool)])
    return ok_in, sig, ok, is_in


def base_R(Q, qn):
    return row_scores(phrase_cos(Q, qn, X, pn), prow, nr, A["cnorm"])[0]


okv, sigv, okv_all, inv_ = evaluate("val", "oos_val", base_R)
okt, sigt, okt_all, int_ = evaluate("test", "oos_test", base_R)
print("BASELINE (as learned, BLEND 0.8): top-1 val %.4f  test %.4f" % (okv.mean(), okt.mean()))
for P in (0.97, 0.95):
    c, thr, o = cov_at_precision(sigt, okt_all, int_, P)
    thr_v = val_threshold(sigv, okv_all, inv_, P)
    r = apply_threshold(sigt, okt_all, int_, thr_v)
    print("  P%.2f  oracle-on-test: cov %.1f%% oos %.1f%% (thr %.3f) | val-chosen thr %.3f -> test cov %.1f%% wrong-in %.1f%% "
          "oos %.1f%% realised prec %.3f" % (P, 100 * c, 100 * o, thr, thr_v, 100 * r["cov"], 100 * r["wrong_in"], 100 * r["oos"], r["prec"]))
np.save(OUT + "/base_ok_test.npy", okt)

# ---- the negative store: are any stored negatives same-INTENT rows (false negatives)?
text2int = {normq_t: i for normq_t, i in ((" ".join(q.lower().split()), i) for q, i in d["train"] + [(q, "oos") for q, _ in d["oos_train"]])}
negs = st["meaning"]["neg"]
same = sum(1 for q, rid in negs if rid in ri and text2int.get(q) == ri[rid])
known = sum(1 for q, rid in negs if q in text2int and rid in ri)
print("stored negatives %d, resolvable %d, pointing at a row of the SAME intent (false negatives): %d" % (len(negs), known, same))
# rows per intent (the duplicate-row problem): each row has its own answer key
akeys = len({mi.rows[r].get("akey") for r in rids})
c = collections.Counter(row_int)
print("rows %d, answer keys %d, intents %d; intents split over >1 row: %d; rows that share an intent with another row: %d"
      % (nr, akeys, len(c), sum(1 for k, v in c.items() if v > 1 and k != "oos"), sum(v for k, v in c.items() if v > 1)))
# how often is the runner-up a same-intent duplicate (so the lead is measured against a non-rival)?
Qi, qni = cache["test"]
Ri = base_R(Qi, qni)
o = np.argsort(-Ri, axis=1, kind="stable")[:, :2]
dup = (row_int[o[:, 0]] == row_int[o[:, 1]]).mean()
print("test questions whose runner-up row is the SAME intent as the top row: %.1f%%" % (100 * dup))
print("total %.1fs" % (time.time() - t0))

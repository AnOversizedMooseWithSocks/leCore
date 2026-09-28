"""P2 (backlog E1.1): softmax-weighted (InfoNCE-gradient) prototype learning vs the current best-wording + centroid
blend, and vs miss-only (AdaptHD-style) updates -- on the MeaningIndex's OWN sparse features (meaning_features x idf,
the exact vectors rank() uses; no dense projection).

Settings (argv[1]):
  learned  the saved learned index (602 rows over 151 intents -- 119 intents are split into duplicate rows with
           DIFFERENT answer keys, so a row-level softmax treats same-intent rows as negatives: FALSE NEGATIVES)
  k20      one row per intent, the first 20 CLINC train wordings each (the backlog's E1.1 acceptance setting)
  k100     one row per intent, all 100 train wordings (+ one 'oos' row from oos_train): the dataset-label ceiling

Prototypes live on the unit sphere of the SAME space as the rows (cosine logits, norms tracked exactly), so the
absolute-similarity floor still means something. Update per training wording x (unit) with row label y:
  infonce : P_k += eta/tau * x * (1[k=y] - softmax(cos(x,P)/tau)_k)     (every row, every example)
  adapthd : if argmax_k cos(x,P_k) != y:  P_y += eta*x ; P_yhat -= eta*x  (miss-only)
  masked  : infonce with same-INTENT rows removed from the softmax denominator (ORACLE: uses dataset labels the
            live system does not have -- measures what the false negatives cost)
Readouts: 'blend' = 0.2*best phrasing + 0.8*cos(q, prototype) (the prototype replaces the centroid: how it would land
in MeaningIndex.rank); 'proto' = cos(q, prototype) alone. Confidence g = s1 + (s1 - s2), as the index uses.
Hyper-parameters (tau, eta, epochs, readout) are chosen on CLINC VAL (+oos_val re-weighted to the test prior);
TEST is reported once for the chosen config, over 3 shuffle seeds.
"""
import sys, time, json
import numpy as np
import scipy.sparse as sp
from common import *

SETTING = sys.argv[1] if len(sys.argv) > 1 else "learned"
t0 = time.time()
mi0, ri0, d, st = load()
if SETTING == "learned":
    mi, ri = mi0, ri0
else:
    K = 20 if SETTING == "k20" else 100
    tr = {}
    for t, i in d["train"]:
        tr.setdefault(i, []).append(t)
    if SETTING == "k100":
        tr["oos"] = [t for t, _ in d["oos_train"]]
    mi, ri = MeaningIndex(), {}
    for i in sorted(tr):
        rid = mi.add_row(tr[i][0]); ri[rid] = i
        for q in tr[i][1:K]:
            mi.link(rid, q, learn=False)

X, pn, prow, rids, A = build_index_mats(mi)
nr = len(rids)
row_int = np.array([ri[r] for r in rids])
Xh = (sp.diags(1.0 / np.maximum(pn, 1e-12)) @ X).tocsr()
Mrow = sp.csr_matrix((np.ones(len(prow)), (prow, np.arange(len(prow)))), shape=(nr, len(prow)))
Csum = (Mrow @ Xh).tocsr()
cnorm = np.sqrt(np.asarray(Csum.multiply(Csum).sum(1)).ravel())
C0 = (sp.diags(1.0 / np.maximum(cnorm, 1e-12)) @ Csum).toarray()           # unit centroids (Rocchio) = the init
order = np.argsort(prow, kind="stable"); starts = np.searchsorted(prow[order], np.arange(nr + 1))
F = X.shape[1]

# queries: val / oos_val / test / oos_test, as the index builds them (assoc expansion; total norm incl. unknown words)
if SETTING == "learned":
    QC, _ = get_cache(mi)
else:
    QC = {s: query_mats(mi, [q for q, _ in d[s]]) for s in ("val", "oos_val", "test", "oos_test")}
QH, BEST = {}, {}
for s, (Q, qn) in QC.items():
    Qh = (sp.diags(1.0 / np.maximum(qn, 1e-12)) @ Q).tocsr()
    QH[s] = Qh
    Cp = (Qh @ Xh.T).toarray()[:, order]
    b = np.zeros((Cp.shape[0], nr))
    for r in range(nr):
        if starts[r + 1] > starts[r]:
            b[:, r] = Cp[:, starts[r]:starts[r + 1]].max(1)
    BEST[s] = np.maximum(b, 0.0)
    del Cp
print("[%s] rows %d wordings %d feats %d  setup %.1fs" % (SETTING, nr, X.shape[0], F, time.time() - t0), flush=True)

# training examples: the index's own wordings with their row labels (what the live system actually holds)
tr_f = [Xh.indices[Xh.indptr[i]:Xh.indptr[i + 1]] for i in range(Xh.shape[0])]
tr_v = [Xh.data[Xh.indptr[i]:Xh.indptr[i + 1]] for i in range(Xh.shape[0])]
tr_y = prow.copy()
same_int = {r: np.flatnonzero(row_int == row_int[r]) for r in range(nr)}


def train(rule, tau, eta, epochs, seed, checkpoints):
    """-> {epoch: prototype matrix (F x nr) transposed, unnormalised} at the checkpoints."""
    PT = np.ascontiguousarray(C0.T)                     # F x nr: a wording's features gather contiguous rows
    nrm2 = (PT * PT).sum(0)
    rng = np.random.default_rng(seed)
    out = {}
    for ep in range(1, epochs + 1):
        for i in rng.permutation(len(tr_y)):
            f, v, y = tr_f[i], tr_v[i], tr_y[i]
            if len(f) == 0:
                continue
            blk = PT[f]                                  # len(f) x nr
            cos = (v @ blk) / np.sqrt(np.maximum(nrm2, 1e-12))
            if rule == "adapthd":
                yh = int(np.argmax(cos))
                if yh == y:
                    continue
                g = np.zeros(nr); g[y] = eta; g[yh] = -eta
            else:
                z = cos / tau
                if rule == "masked":
                    m = same_int[y]; z = z.copy(); z[m[m != y]] = -np.inf
                z -= z.max(); p = np.exp(z); p /= p.sum()
                g = -p; g[y] += 1.0; g *= eta / tau
            new = blk + np.outer(v, g)
            nrm2 += (new * new).sum(0) - (blk * blk).sum(0)
            PT[f] = new
        if ep in checkpoints:
            out[ep] = PT / np.sqrt(np.maximum(nrm2, 1e-12))[None, :]
    return out


def readout(PTn, s, mode):
    cos = np.asarray(QH[s] @ PTn)
    return cos if mode == "proto" else 0.2 * BEST[s] + 0.8 * cos


def evaluate(R_in, R_oos, split_in):
    out = []
    for R in (R_in, R_oos):
        o = np.argsort(-R, axis=1, kind="stable")[:, :2]
        sv = np.take_along_axis(R, o, axis=1)
        s1, s2 = sv[:, 0], np.where(sv[:, 1] > 0, sv[:, 1], 0.0)
        out.append((o[:, 0], np.where(s1 > 0, 2 * s1 - s2, -9), s1 > 0))
    (ti, gi, hi), (to, go, ho) = out
    truth = np.array([i for _, i in d[split_in]])
    ok_in = hi & (row_int[ti] == truth)
    return ok_in, np.concatenate([gi, go]), np.concatenate([ok_in, np.zeros(len(to), bool)]), \
        np.concatenate([np.ones(len(ti), bool), np.zeros(len(to), bool)])


import hashlib
HALF_A = np.array([int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 == 0 for q, _ in d["oos_test"]])
N_IN, N_OOS = 4500.0, 1000.0          # the deployment prior every precision below is computed under


def prec_rates(sig, ok, is_in, thr):
    """precision under the fixed prior from RATES: in-scope served/right rates and the oos served rate."""
    s = sig >= thr
    right = (ok & s & is_in).sum() / is_in.sum(); served_in = (s & is_in).sum() / is_in.sum()
    oos = (s & ~is_in).sum() / max((~is_in).sum(), 1)
    return right * N_IN / max(served_in * N_IN + oos * N_OOS, 1e-9), right, oos


def gate_thr(sig, ok, is_in, P):
    """lowest threshold whose prior-weighted precision on the CALIBRATION split is >= P (scan distinct values)."""
    best = np.inf
    for thr in np.unique(sig)[::-1]:
        pr, _, _ = prec_rates(sig, ok, is_in, thr)
        if pr >= P:
            best = thr
    return best


def split_cal(v_in, v_oos_all):
    """calibration = val in-scope + oos_test half A;  report = test in-scope + oos_test half B."""
    return v_in, v_oos_all


def report(name, v, t, extra=""):
    okv, sigv, okva, inv_ = v; okt, sigt, okta, int_ = t
    parts = ["%-26s top1 val %.4f test %.4f" % (name, okv.mean(), okt.mean())]
    res = {"top1_val": float(okv.mean()), "top1_test": float(okt.mean())}
    # t carries the full oos_test block at the end: split it into the calibration half (A) and report half (B)
    n_in = int(int_.sum())
    oo_sig, oo_ok = sigt[n_in:], okta[n_in:]
    cal_sig = np.concatenate([sigv[inv_], oo_sig[HALF_A]]); cal_ok = np.concatenate([okva[inv_], oo_ok[HALF_A]])
    cal_in = np.concatenate([np.ones(int(inv_.sum()), bool), np.zeros(int(HALF_A.sum()), bool)])
    rep_sig = np.concatenate([sigt[:n_in], oo_sig[~HALF_A]]); rep_ok = np.concatenate([okta[:n_in], oo_ok[~HALF_A]])
    rep_in = np.concatenate([np.ones(n_in, bool), np.zeros(int((~HALF_A).sum()), bool)])
    for P in (0.97, 0.95):
        thr = gate_thr(cal_sig, cal_ok, cal_in, P)
        pr, cov, oos = prec_rates(rep_sig, rep_ok, rep_in, thr)
        parts.append("P%.2f cov %.1f%% prec %.3f oos %.1f%%" % (P, 100 * cov, pr, 100 * oos))
        res["P%.2f" % P] = {"cov": float(cov), "prec": float(pr), "oos": float(oos), "thr": float(thr)}
        res["served_ok_P%.2f" % P] = (rep_ok & (rep_sig >= thr))[rep_in]
    print(" | ".join(parts) + extra, flush=True)
    return res


RES = {"setting": SETTING}
# ---- baselines: the index as it scores today (blend with the Rocchio centroid) and centroid alone
C0T = np.ascontiguousarray(C0.T)
for mode, nm in (("blend", "BASE blend0.8 (today)"), ("proto", "centroid only (init)")):
    v = evaluate(readout(C0T, "val", mode), readout(C0T, "oos_val", mode), "val")
    t = evaluate(readout(C0T, "test", mode), readout(C0T, "oos_test", mode), "test")
    RES[nm] = report(nm, v, t)
    RES[nm + "_json"] = {k: x for k, x in RES[nm].items() if not k.startswith("served_ok")}
    if mode == "blend":
        base_t_ok = t[0]; base_t = t; base_v = v; BASE_RES = RES[nm]

GRIDS = {"infonce": [(tau, eta) for tau in (0.02, 0.05) for eta in (0.03, 0.1)],
         "adapthd": [(None, eta) for eta in (0.01, 0.03, 0.1)]}
CK = (2, 4, 8, 16)
if SETTING != "learned":                    # CPU budget: the grid bracketed on the learned index is reused, smaller
    GRIDS = {"infonce": [(0.05, 0.03), (0.05, 0.1), (0.02, 0.1)], "adapthd": [(None, 0.03)]}
    CK = (2, 4, 8)
for rule, grid in GRIDS.items():
    best = None
    for tau, eta in grid:
        tt = time.time()
        snaps = train(rule, tau or 1.0, eta, max(CK), 0, CK)
        for ep, PTn in snaps.items():
            for mode in ("blend", "proto"):
                okv = evaluate(readout(PTn, "val", mode), readout(PTn, "oos_val", mode), "val")
                crit = okv[0].mean()
                if best is None or crit > best[0] + 1e-12:
                    best = (crit, tau, eta, ep, mode)
        print("   %s tau=%s eta=%s: val top1 by epoch %s  (%.1fs)" % (
            rule, tau, eta, {ep: round(float(evaluate(readout(P_, "val", "blend"), readout(P_, "oos_val", "blend"), "val")[0].mean()), 4)
                             for ep, P_ in snaps.items()}, time.time() - tt), flush=True)
    _, tau, eta, ep, mode = best
    seeds = []
    for seed in (0, 1, 2):
        PTn = train(rule, tau or 1.0, eta, ep, seed, (ep,))[ep]
        v = evaluate(readout(PTn, "val", mode), readout(PTn, "oos_val", mode), "val")
        t = evaluate(readout(PTn, "test", mode), readout(PTn, "oos_test", mode), "test")
        dm, lo, hi, gain, loss = paired_boot(base_t_ok, t[0])
        r = report("%s s%d t%s e%s ep%d %s" % (rule, seed, tau, eta, ep, mode), v, t,
                   "  | top-1 d%+.4f CI[%+.4f,%+.4f] fixed %d broke %d" % (dm, lo, hi, gain, loss))
        for P in (0.97, 0.95):
            a, b = BASE_RES["served_ok_P%.2f" % P], r.pop("served_ok_P%.2f" % P)
            cd, clo, chi, _, _ = paired_boot(a, b)
            r["P%.2f" % P]["dcov_ci"] = [cd, clo, chi]
            print("      P%.2f coverage vs base %+.1fpt CI[%+.1f,%+.1f]" % (P, 100 * cd, 100 * clo, 100 * chi))
        r["cfg"] = [tau, eta, ep, mode]
        seeds.append(r)
        if seed == 0 and rule == "infonce":
            # the TRAP, measured: candidate-relative softmax p_max as the confidence signal (cannot say "none")
            sm = []
            for s in (("val", "oos_val"), ("test", "oos_test")):
                pm = []
                for sp_ in s:
                    z = np.asarray(QH[sp_] @ PTn) / (tau or 1.0)
                    z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
                    pm.append(p.max(1))
                sm.append(np.concatenate(pm))
            vv = (v[0], sm[0], v[2], v[3]); tv = (t[0], sm[1], t[2], t[3])
            rr = report("   ...same, signal=softmax p_max", vv, tv)
            rr = {k: x for k, x in rr.items() if not k.startswith("served_ok")}
            RES[rule + "_softmax_pmax_signal"] = rr
    RES[rule] = seeds
    print("%s TEST top-1 over seeds: mean %.4f  range [%.4f, %.4f]  (base %.4f)" % (
        rule, np.mean([r["top1_test"] for r in seeds]), min(r["top1_test"] for r in seeds),
        max(r["top1_test"] for r in seeds), base_t_ok.mean()), flush=True)
json.dump({k: x for k, x in RES.items() if not (isinstance(x, dict) and any(k2.startswith("served_ok") for k2 in x))},
          open(OUT + "/p2b_%s.json" % SETTING, "w"), indent=1, default=float)
print("total %.1fs" % (time.time() - t0))

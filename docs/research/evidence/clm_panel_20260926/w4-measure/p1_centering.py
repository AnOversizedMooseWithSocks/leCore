"""P1 (backlog E1.4): contrastive readout / centering on the LEARNED CLINC150 index.

Four readouts, each with ONE knob chosen on CLINC VAL (+oos_val re-weighted to the test prior), reported on TEST once:
  A  literal E1.4       score(q,r) - lam * cos(q, mu)       mu = mean unit row centroid        (per-QUERY constant)
  A2 literal, local mu  score(q,r) - lam * cos(q, mu_q8)    mu_q8 = mean of the q's 8 nearest rows (per-QUERY)
  B  centering          cos(q - a*mu, v - a*mu) for phrasings (mu_p) and centroids (mu_c)       (changes rank)
  C  all-but-the-top    B at a=1, then remove the top-D principal directions of the row centroids (Mu & Viswanath 2018)
  H  CSLS hub penalty   score(q,r) - lam * h_r, h_r = mean cos of row r's centroid to its 10 nearest phrasings of
                        OTHER rows (a row that sits near everything is penalised; label-free, index-only)
The confidence signal is the index's own g = s1 + (s1 - s2) computed on the transformed scores.
"""
import time
import numpy as np
import scipy.sparse as sp
from common import *

t0 = time.time()
mi, ri, d, st = load()
X, pn, prow, rids, A = build_index_mats(mi)
nr = len(rids)
row_int = np.array([ri[r] for r in rids])
cache, d = get_cache(mi)

Xh = sp.diags(1.0 / np.maximum(pn, 1e-12)) @ X                               # unit phrasings
Mrow = sp.csr_matrix((np.ones(len(prow)), (prow, np.arange(len(prow)))), shape=(nr, len(prow)))
Csum = (Mrow @ Xh).tocsr()
cnorm = np.sqrt(np.asarray(Csum.multiply(Csum).sum(1)).ravel())
assert np.allclose(cnorm, A["cnorm"])
Cm = (sp.diags(1.0 / np.maximum(cnorm, 1e-12)) @ Csum).tocsr()                                 # unit row centroids
mu_c = np.asarray(Cm.mean(0)).ravel()
mu_p = np.asarray(Xh.mean(0)).ravel()
G = (Cm @ Cm.T).toarray()                                                    # centroid Gram (602 x 602)
order = np.argsort(prow, kind="stable"); starts = np.searchsorted(prow[order], np.arange(nr + 1))


def groupmax(Cp):
    Co = Cp[:, order]
    best = np.zeros((Cp.shape[0], nr))
    for r in range(nr):
        a, b = starts[r], starts[r + 1]
        best[:, r] = Co[:, a:b].max(axis=1)
    return np.maximum(best, 0.0)


# hub score for CSLS: each row centroid vs phrasings of OTHER rows, mean of the 10 largest
CX = (Cm @ Xh.T).toarray()                                                   # 602 x 8591
CX[prow, np.arange(len(prow))] = -np.inf                                     # mask each row's OWN phrasings
h_r = np.sort(CX, axis=1)[:, -10:].mean(1)
del CX

# all-but-the-top directions from the (centred) row centroids -- index-only, label-free
Cd = Cm.toarray() - mu_c[None, :]
_, sv, Vt = np.linalg.svd(Cd, full_matrices=False)
del Cd
print("setup %.1fs; top singular values of centred centroids: %s; ||mu_c||=%.3f ||mu_p||=%.3f"
      % (time.time() - t0, np.round(sv[:6], 2), np.linalg.norm(mu_c), np.linalg.norm(mu_p)))

PRE = {}
for s in ("val", "oos_val", "test", "oos_test"):
    Q, qn = cache[s]
    Qh = sp.diags(1.0 / np.maximum(qn, 1e-12)) @ Q
    Cp = (Qh @ Xh.T).toarray()
    Cc = (Qh @ Cm.T).toarray()
    qq = np.asarray(Qh.multiply(Qh).sum(1)).ravel()                           # |q_hat|^2 (<1 with unknown words)
    PRE[s] = dict(Qh=Qh, Cp=Cp, Cc=Cc, qq=qq, qmc=Qh @ mu_c, qmp=Qh @ mu_p, QV=Qh @ Vt[:20].T)
pm_p = Xh @ mu_p; cm_c = Cm @ mu_c
XV = Xh @ Vt[:20].T; CV = Cm @ Vt[:20].T
print("precompute %.1fs" % (time.time() - t0))


def centred(Cx, qq, qm, vm, mm, a, QV=None, VV=None, muV=None, D=0):
    """cos(q - a mu, v - a mu) [then minus the top-D directions] from the plain dot products."""
    num = Cx - a * qm[:, None] - a * vm[None, :] + a * a * mm
    qn2 = qq - 2 * a * qm + a * a * mm
    vn2 = 1.0 - 2 * a * vm + a * a * mm
    if D:
        qp = QV[:, :D] - a * muV[:D][None, :]
        vp = VV[:, :D] - a * muV[:D][None, :]
        num = num - qp @ vp.T
        qn2 = qn2 - (qp * qp).sum(1)
        vn2 = vn2 - (vp * vp).sum(1)
    return num / np.sqrt(np.maximum(qn2, 1e-12))[:, None] / np.sqrt(np.maximum(vn2, 1e-12))[None, :]


def scores(s, kind, k):
    P = PRE[s]
    if kind in ("base", "A", "A2", "H"):
        R = 0.2 * groupmax(P["Cp"]) + 0.8 * P["Cc"]
        if kind == "H":
            R = R - k * h_r[None, :]
        return R
    a, D = (k, 0) if kind in ("B", "U") else (1.0, int(k))
    qtot = P["qq"] if kind == "U" else np.ones_like(P["qq"])      # |q_hat|^2 = 1 incl. unknown-word mass
    Cp = centred(P["Cp"], qtot, P["qmp"], pm_p, float(mu_p @ mu_p), a, P["QV"], XV, Vt[:20] @ mu_p, D)
    Cc = centred(P["Cc"], qtot, P["qmc"], cm_c, float(mu_c @ mu_c), a, P["QV"], CV, Vt[:20] @ mu_c, D)
    R = 0.2 * groupmax(Cp) + 0.8 * Cc
    R[:, ZERO] = 0.0                    # a row with no features is never matched (as in the index)
    return R


ZERO = cnorm <= 1e-12
print("rows with an all-stopword (zero) vector:", int(ZERO.sum()))


def run(split_in, split_oos, kind, k):
    out = []
    for s in (split_in, split_oos):
        R = scores(s, kind, k)
        o = np.argsort(-R, axis=1, kind="stable")[:, :8]
        sv_ = np.take_along_axis(R, o, axis=1)
        s1, s2 = sv_[:, 0], np.where(sv_[:, 1] > 0, sv_[:, 1], 0.0)
        has = s1 > 0
        g = 2 * s1 - s2
        if kind == "A":
            g = g - k * PRE[s]["Cc"].mean(1) / np.linalg.norm(mu_c)      # cos(q, mu_c): q_hat.mu_c = mean_r q_hat.c_r
        if kind == "A2":
            # cos(q, mean of its 8 nearest row centroids) = sum_j q.c_j / |sum_j c_j|
            num = np.take_along_axis(PRE[s]["Cc"], o, axis=1).sum(1)
            key = (s, "den")
            if key not in PRE:
                PRE[key] = np.sqrt(np.array([G[np.ix_(oo, oo)].sum() for oo in o]))
            den = PRE[key]
            g = g - k * num / np.maximum(den, 1e-12)
        out.append((o[:, 0], g, has))
    (ti, gi, hi), (to, go, ho) = out
    truth = np.array([i for _, i in d[split_in]])
    ok_in = hi & (row_int[ti] == truth)
    sig = np.concatenate([np.where(hi, gi, -9), np.where(ho, go, -9)])
    ok = np.concatenate([ok_in, np.zeros(len(to), bool)])
    is_in = np.concatenate([np.ones(len(ti), bool), np.zeros(len(to), bool)])
    return ok_in, sig, ok, is_in


GRID = {"A": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0], "A2": [0.0, 0.1, 0.25, 0.5, 1.0], "B": [0.0, 0.25, 0.5, 0.75, 1.0],
        "C": [1, 2, 5, 10, 20], "H": [0.0, 0.1, 0.2, 0.3, 0.5]}
base_v = run("val", "oos_val", "base", 0); base_t = run("test", "oos_test", "base", 0)


def summary(v, t):
    okv, sigv, okva, inv_ = v; okt, sigt, okta, int_ = t
    row = {"top1_val": okv.mean(), "top1_test": okt.mean()}
    for P in (0.97, 0.95):
        thr = val_threshold(sigv, okva, inv_, P)
        cv, _, _ = cov_at_precision(sigv, okva, inv_, P, w=np.where(inv_, 1.0, (1000 / 4500) / (100 / 3000)))
        r = apply_threshold(sigt, okta, int_, thr)
        co, _, oo = cov_at_precision(sigt, okta, int_, P)
        row.update({"covval%.2f" % P: cv, "cov%.2f" % P: r["cov"], "prec%.2f" % P: r["prec"], "oos%.2f" % P: r["oos"],
                    "oracle%.2f" % P: co})
    return row


def fmt(r):
    return ("top1 val %.4f test %.4f | P.97 val-cov %.1f%% -> test cov %.1f%% prec %.3f oos %.1f%% (oracle %.1f%%) | "
            "P.95 val-cov %.1f%% -> test cov %.1f%% prec %.3f oos %.1f%% (oracle %.1f%%)" % (
                r["top1_val"], r["top1_test"], 100 * r["covval0.97"], 100 * r["cov0.97"], r["prec0.97"], 100 * r["oos0.97"],
                100 * r["oracle0.97"], 100 * r["covval0.95"], 100 * r["cov0.95"], r["prec0.95"], 100 * r["oos0.95"],
                100 * r["oracle0.95"]))


B = summary(base_v, base_t)
print("BASE      ", fmt(B))
okt_base = base_t[0]
W = lambda inv_: np.where(inv_, 1.0, (1000 / 4500) / (100 / 3000))
u = summary(run("val", "oos_val", "U", 0.0), run("test", "oos_test", "U", 0.0))
print("ABLATION U (query norm WITHOUT the unknown-word mass):", fmt(u))


def served_ok(t, thr):
    okt, sigt, okta, int_ = t
    return (okta & (sigt >= thr))[int_]


def pick(kind, P):
    """choose the knob on VAL only: coverage at precision P (oos re-weighted), or top-1 when P is None"""
    best = None
    for k in GRID[kind]:
        v = VCACHE.setdefault((kind, k), run("val", "oos_val", kind, k))
        okv, sigv, okva, inv_ = v
        crit = okv.mean() if P is None else cov_at_precision(sigv, okva, inv_, P, w=W(inv_))[0]
        if best is None or crit > best[0] + 1e-12:
            best = (crit, k)
    return best[1]


VCACHE = {}
for kind, grid in GRID.items():
    for k in grid:
        okv, sigv, okva, inv_ = VCACHE.setdefault((kind, k), run("val", "oos_val", kind, k))
        print("  val %-2s k=%-5s top1 %.4f  cov@.97 %.1f%%  cov@.95 %.1f%%" % (
            kind, k, okv.mean(), 100 * cov_at_precision(sigv, okva, inv_, 0.97, w=W(inv_))[0],
            100 * cov_at_precision(sigv, okva, inv_, 0.95, w=W(inv_))[0]))
    line = [kind]
    if kind in ("B", "C", "H"):
        k = pick(kind, None)
        t = run("test", "oos_test", kind, k)
        dm, lo, hi, gain, loss = paired_boot(okt_base, t[0])
        line.append("top-1(k=%s) test %.4f d%+.4f CI[%+.4f,%+.4f] fixed %d broke %d" % (k, t[0].mean(), dm, lo, hi, gain, loss))
    for P in (0.97, 0.95):
        k = pick(kind, P)
        v = VCACHE[(kind, k)]
        t = run("test", "oos_test", kind, k)
        thr = val_threshold(v[1], v[2], v[3], P)
        r = apply_threshold(t[1], t[2], t[3], thr)
        thr_b = val_threshold(base_v[1], base_v[2], base_v[3], P)
        dm, lo, hi, gain, loss = paired_boot(served_ok(base_t, thr_b), served_ok(t, thr))
        line.append("P%.2f(k=%s) test cov %.1f%% prec %.3f oos %.1f%% d%+.1fpt CI[%+.1f,%+.1f]" % (
            P, k, 100 * r["cov"], r["prec"], 100 * r["oos"], 100 * dm, 100 * lo, 100 * hi))
    print(" | ".join(line))
print("total %.1fs" % (time.time() - t0))

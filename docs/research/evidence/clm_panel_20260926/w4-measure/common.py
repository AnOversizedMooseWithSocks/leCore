"""w4-measure scratch: shared loader + a vectorised, EXACT re-implementation of MeaningIndex.rank()/answers()
on the saved learned CLINC150 index, so probes can swap the readout without touching the repo.

Everything here is deterministic (no RNG unless a probe passes a seed). Nothing is written to the repo.

Matrices (scipy.sparse CSR, scratch only -- core stays NumPy):
  X   phrasing x feature, entries val*idf exactly as MeaningIndex._build() makes them (NOT unit-normalised)
  pn  phrasing norms (|x|), prow phrasing -> row index
  Q   query x feature, from MeaningIndex._query_vec (assoc expansion included), qn = query norm INCLUDING the
      unknown-word mass (so cos(q, p) = (Q @ X.T) / (qn * pn), bit-for-bit the index's formula)
"""
import json, sys, time, os
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "/home/claude/leCore")
from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex, meaning_features  # noqa

S = "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad"
OUT = S + "/swarm/w4-measure"
IDX = S + "/final/default/idx_clinc.json"
DATA = S + "/data/clinc150_full.json"


def load():
    st = json.load(open(IDX))
    ri = st["row_intent"]
    mi = MeaningIndex.from_state(st["meaning"])
    d = json.load(open(DATA))
    return mi, ri, d, st


def build_index_mats(mi):
    A = mi._build()
    assert A["live"].all()
    pid = np.asarray(mi._ent_pid, np.int64)
    fid = np.asarray(mi._ent_fid, np.int64)
    val = np.asarray(mi._ent_val, np.float64)
    w = val * A["idf"][fid]
    n_p, F = len(mi._pid_row), len(mi._fnames)
    X = sp.csr_matrix((w, (pid, fid)), shape=(n_p, F))      # duplicates (same pid,fid) are summed -- as bincount does
    pn = np.asarray(A["norm"], np.float64)
    prow = np.asarray(A["prow"], np.int64)
    rids = list(A["rix"])
    return X, pn, prow, rids, A


def query_mats(mi, texts):
    F = len(mi._fnames)
    rr, cc, vv, qn = [], [], [], np.zeros(len(texts))
    for i, t in enumerate(texts):
        q, n = mi._query_vec(t)
        qn[i] = n
        for f, v in q.items():
            rr.append(i); cc.append(f); vv.append(v)
    Q = sp.csr_matrix((np.asarray(vv, np.float64), (np.asarray(rr, np.int64), np.asarray(cc, np.int64))),
                      shape=(len(texts), F))
    return Q, qn


def phrase_cos(Q, qn, X, pn):
    """dense (n_q x n_p) cosine, the index's per-phrasing score."""
    C = (Q @ X.T).toarray()
    C /= np.maximum(qn, 1e-12)[:, None]
    C /= np.maximum(pn, 1e-12)[None, :]
    return C


def row_scores(C, prow, n_rows, cnorm, blend=0.8):
    """exact MeaningIndex.rank() row score: (1-blend) * best phrasing + blend * cos(q, row centroid)."""
    best = np.zeros((C.shape[0], n_rows))
    # max over phrasings of each row (scores >= 0 so a zero init matches np.maximum.at from zeros)
    order = np.argsort(prow, kind="stable")
    ps = prow[order]
    starts = np.searchsorted(ps, np.arange(n_rows + 1))
    Co = C[:, order]
    cen = np.zeros((C.shape[0], n_rows))
    for r in range(n_rows):
        a, b = starts[r], starts[r + 1]
        if b > a:
            best[:, r] = Co[:, a:b].max(axis=1)
            cen[:, r] = Co[:, a:b].sum(axis=1)
    cen /= np.maximum(cnorm, 1e-12)[None, :]
    return (1.0 - blend) * np.maximum(best, 0) + blend * cen, best, cen


def top2(R):
    """-> top row index, s1, s2 per query, with the index's stable tie rule (lowest row index first) and the
    rule that only s > 0 rows are candidates (rank() stops at s <= 0)."""
    o = np.argsort(-R, axis=1, kind="stable")[:, :2]
    s = np.take_along_axis(R, o, axis=1)
    s = np.where(s > 0, s, 0.0)
    has = s[:, 0] > 0
    return o[:, 0], s[:, 0], s[:, 1], has


# ------------------------------------------------------------------ evaluation
def cov_at_precision(sig, ok, is_in, P, w=None):
    """ORACLE-threshold coverage (the curves.py method): sort by signal, largest prefix with precision >= P.
    ok: served-and-right; OOS rows are never ok. w: optional sample weights (for prior re-weighting).
    -> (in-scope coverage fraction, threshold, oos served fraction)"""
    if w is None:
        w = np.ones(len(sig))
    o = np.argsort(-sig, kind="stable")
    c = np.cumsum((ok * w)[o]); n = np.cumsum(w[o])
    prec = c / n
    idx = np.where(prec >= P)[0]
    if not len(idx):
        return 0.0, np.inf, 0.0
    k = idx.max() + 1
    thr = sig[o][k - 1]
    served = sig >= thr
    return (ok & served & is_in).sum() / is_in.sum(), thr, (served & ~is_in).sum() / max((~is_in).sum(), 1)


def apply_threshold(sig, ok, is_in, thr):
    served = sig >= thr
    n_s = served.sum()
    prec = (ok & served).sum() / max(n_s, 1)
    return dict(cov=(ok & served & is_in).sum() / is_in.sum(), wrong_in=((~ok) & served & is_in).sum() / is_in.sum(),
                oos=(served & ~is_in).sum() / max((~is_in).sum(), 1), prec=prec)


def val_threshold(sig_v, ok_v, in_v, P, n_in_test=4500, n_oos_test=1000):
    """HELD-OUT threshold: chosen on VAL for precision >= P, with val's out-of-scope rows RE-WEIGHTED to the test
    prior (val has 100 oos per 3000 in-scope; test has 1000 per 4500 -- unweighted, val precision is optimistic
    because it has 6.7x fewer out-of-scope questions to serve wrongly)."""
    n_in_v, n_oos_v = in_v.sum(), (~in_v).sum()
    w_oos = (n_oos_test / n_in_test) / (n_oos_v / n_in_v)
    w = np.where(in_v, 1.0, w_oos)
    _, thr, _ = cov_at_precision(sig_v, ok_v, in_v, P, w=w)
    return thr


def paired_boot(a_ok, b_ok, n=2000, seed=0):
    """paired bootstrap CI of mean(b) - mean(a) over the same questions; + exact-ish McNemar counts."""
    rng = np.random.default_rng(seed)
    N = len(a_ok)
    idx = rng.integers(0, N, size=(n, N))
    diff = b_ok[idx].mean(1) - a_ok[idx].mean(1)
    return float(np.mean(b_ok) - np.mean(a_ok)), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)), \
        int((b_ok & ~a_ok).sum()), int((a_ok & ~b_ok).sum())


def get_cache(mi):
    """Build (or load) the query matrices for val / oos_val / test / oos_test."""
    d = json.load(open(DATA))
    f = OUT + "/qcache.npz"
    splits = ["val", "oos_val", "test", "oos_test"]
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        out = {}
        for s in splits:
            out[s] = (sp.csr_matrix((z[s + "_d"], z[s + "_i"], z[s + "_p"]), shape=tuple(z[s + "_sh"])), z[s + "_qn"])
        return out, d
    out, save = {}, {}
    for s in splits:
        Q, qn = query_mats(mi, [q for q, _ in d[s]])
        out[s] = (Q, qn)
        save.update({s + "_d": Q.data, s + "_i": Q.indices, s + "_p": Q.indptr, s + "_sh": np.array(Q.shape), s + "_qn": qn})
    np.savez(f, **save)
    return out, d

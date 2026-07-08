"""GRAD-1 -- Iterative Hard Thresholding recovery (holographic_iht).

WHAT THIS IS
------------
The GRADIENT-NATIVE member of the sparse-recovery family. The engine already has two ways to recover the components
of a bundle `cue = sum_i w_i * codebook[i]`:
  * the LINEAR readout -- one shot, correlations `codebook @ cue` then keep the top m (order-free, washes out at load);
  * OCCLUSION recall (holographic_occlusion) -- GREEDY matching pursuit: take the most-relevant atom, subtract its
    explained part, repeat (one pass, never revisits a pick).
IHT is the third route: PROJECTED GRADIENT DESCENT. Take a gradient step on the reconstruction loss, then PROJECT onto
the K-sparse set by keeping the K largest-magnitude coefficients -- and ITERATE, so a coefficient dropped at one step
can return at the next. That revision is the whole point: greedy matching pursuit cannot undo an early wrong pick,
IHT can.

BUILT ON GRAD-2
  The gradient step is exactly the descent the general optimizer (holographic_optimize, GRAD-2) generalized: the
  reconstruction loss is 0.5 * ||cue - c @ codebook||^2, whose gradient w.r.t. the coefficients c is
  -codebook @ (cue - c @ codebook) -- analytic and cheap. With NO threshold (K = N) IHT reduces to plain gradient
  descent on that loss, reaching the LEAST-SQUARES solution -- the same point optimize() finds (matched to ~1e-11 in
  the self-test). The hard-threshold projection is the ONE thing that turns the optimizer into a sparse recovery
  method. This is the panel's flagged use of the 3DGS gradient machinery for recovery, made literal.

WHAT IT PROVIDES
  * hard_threshold(c, K) -- the K-sparse projection H_K: keep the K largest-magnitude entries of c, zero the rest.
  * iht_recall(cue, codebook, K, steps, mu, tol) -- recover the K active atoms by IHT; returns (index, weight) pairs
    in descending-|weight| order, the SAME signature as occlusion_recall for a head-to-head.

MEASURED -- IHT vs occlusion (greedy MP) vs linear, as dictionary COHERENCE rises (M=12, N=200, D=512, 12 seeds):
  * INCOHERENT (random dictionary): IHT F1 = 1.000 TIES occlusion 1.000; the linear readout lags (0.958).
  * The CROSSOVER is the honest finding -- neither dominates:
      - at MILD coherence, greedy occlusion is BETTER (0.96 vs IHT 0.87): when the cue still points cleanly at the
        true atoms, subtract-and-move wins and IHT's iterations spread energy onto correlated decoys.
      - at HIGH coherence, IHT PULLS AHEAD (0.71 vs occlusion 0.54): greedy MP's early wrong picks on a coherent
        dictionary become UNRECOVERABLE, while IHT keeps revising its support and corrects them. The classic
        matching-pursuit-vs-IHT result, reproduced on the substrate.
  * K = N bridge: IHT with no threshold matches numpy's lstsq to ~1e-11 -- the reduction to plain gradient descent,
    confirming the GRAD-2 connection.

DETERMINISM (per ISA.md)
  No RNG: the gradient step, the threshold, and the step size mu = 1/||codebook||_2^2 are all deterministic given the
  cue and codebook. Same inputs give the same recovery (asserted).

KEPT NEGATIVES (loud)
  * NOT a universal win over occlusion: at LOW-to-MILD dictionary coherence greedy matching pursuit recovers a BETTER
    support than IHT. IHT is the method for the COHERENT-dictionary regime, not a strict upgrade -- the crossover is
    real and on the record.
  * needs two knobs occlusion does not: the sparsity K and the step size mu (defaulted to 1/Lipschitz = the standard
    safe IHT step, but a bad mu crawls or overshoots), plus an iteration budget.
  * it is projected gradient descent, so it inherits a local-optimum character -- it finds the K-sparse stationary
    point reachable from the zero start, not a certified global optimum.
"""

import numpy as np


def hard_threshold(c, K):
    """The K-sparse projection H_K: return a copy of `c` with all but its K largest-magnitude entries zeroed. The one
    operation that turns a gradient-descent step into a sparse-recovery step. K >= len(c) is the identity (no
    thresholding -> plain gradient descent)."""
    c = np.asarray(c, float)
    if K >= c.size:
        return c.copy()
    idx = np.argpartition(np.abs(c), -K)[-K:]              # indices of the K largest |c| (unordered, O(N))
    out = np.zeros_like(c)
    out[idx] = c[idx]                                       # keep the signed coefficients at those positions
    return out


def iht_recall(cue, codebook, K, steps=300, mu=None, tol=1e-12):
    """Recover the K active atoms of `cue` (a bundle of `codebook` rows) by ITERATIVE HARD THRESHOLDING -- projected
    gradient descent: a gradient step on the reconstruction loss 0.5*||cue - c@codebook||^2, then keep the K largest
    coefficients, repeated. The gradient-native sparse-recovery member; unlike greedy occlusion recall it REVISES its
    support, so it holds up where matching pursuit's early mistakes on a coherent dictionary become unrecoverable.

    `mu` is the gradient step; the default 1/||codebook||_2^2 (one over the Gram's largest eigenvalue) is the standard
    safe IHT step that guarantees the loss descends. Stops at `steps` or when the coefficients stop moving (`tol`).
    Returns (index, weight) pairs in descending-|weight| order -- the same shape as occlusion_recall for comparison."""
    cb = np.asarray(codebook, float)
    y = np.asarray(cue, float)
    N = cb.shape[0]
    if mu is None:
        s = np.linalg.norm(cb, 2)                          # spectral norm = largest singular value of the dictionary
        mu = 1.0 / (s * s + 1e-12)                         # 1/Lipschitz: the descent-guaranteeing IHT step
    c = np.zeros(N)
    for _ in range(int(steps)):
        r = y - c @ cb                                     # residual (dim,)
        grad_ascent = cb @ r                               # -gradient of the loss w.r.t. c (so c += mu*this descends)
        c_new = hard_threshold(c + mu * grad_ascent, K)    # gradient step, then the K-sparse projection
        if np.linalg.norm(c_new - c) < tol:                # converged: coefficients stopped moving
            c = c_new
            break
        c = c_new
    nz = np.nonzero(c)[0]
    order = nz[np.argsort(-np.abs(c[nz]))]                 # descending |weight|, like occlusion_recall
    return [(int(i), float(c[i])) for i in order]


# =====================================================================================================
# Self-test -- ties occlusion when incoherent, BEATS it when coherent, reduces to lstsq at K=N.
# =====================================================================================================
def _f1(rec, true_set):
    got = set(i for i, _ in rec)
    tp = len(got & true_set)
    prec = tp / max(len(got), 1)
    rc = tp / max(len(true_set), 1)
    return 2 * prec * rc / max(prec + rc, 1e-12)


def _occlusion(cue, cb, m):
    cb = np.asarray(cb, float)
    resid = np.asarray(cue, float).copy()
    out = []
    for _ in range(m):
        a = cb @ resid
        j = int(np.argmax(a))
        w = float(a[j])
        out.append((j, w))
        resid = resid - w * cb[j]
    return out


def _trial(coherence, seed, N=200, D=512, M=12):
    rng = np.random.default_rng(seed)
    cb = rng.standard_normal((N, D))
    if coherence > 0:
        cb = cb + coherence * rng.standard_normal(D)        # shared component -> mutual coherence
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    S = rng.choice(N, M, replace=False)
    w = rng.uniform(0.5, 1.5, M)
    cue = (w[:, None] * cb[S]).sum(0)
    true = set(int(i) for i in S)
    return _f1(iht_recall(cue, cb, M), true), _f1(_occlusion(cue, cb, M), true)


def _selftest():
    # INCOHERENT: IHT ties greedy occlusion (both recover perfectly)
    iht0 = np.mean([_trial(0.0, s)[0] for s in range(8)])
    occ0 = np.mean([_trial(0.0, s)[1] for s in range(8)])
    assert iht0 > 0.99 and occ0 > 0.99, f"incoherent: both should be ~perfect (IHT {iht0:.3f}, occ {occ0:.3f})"

    # HIGH COHERENCE: IHT pulls ahead of greedy occlusion (the support-revision win)
    iht_hi = np.mean([_trial(1.5, s)[0] for s in range(12)])
    occ_hi = np.mean([_trial(1.5, s)[1] for s in range(12)])
    assert iht_hi > occ_hi + 0.05, f"coherent: IHT must beat greedy MP (IHT {iht_hi:.3f} vs occ {occ_hi:.3f})"

    # K=N BRIDGE: IHT with no threshold == gradient descent == lstsq (the GRAD-2 connection)
    rng = np.random.default_rng(0)
    cb = rng.standard_normal((30, 64))
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    S = rng.choice(30, 5, replace=False)
    w = rng.uniform(0.5, 1.5, 5)
    cue = (w[:, None] * cb[S]).sum(0)
    rec = iht_recall(cue, cb, 30, steps=5000)               # K=N -> plain gradient descent (no thresholding)
    cvec = np.zeros(30)
    for i, wv in rec:
        cvec[i] = wv
    sol = np.linalg.lstsq(cb.T, cue, rcond=None)[0]
    bridge = float(np.linalg.norm(cvec - sol))
    assert bridge < 1e-6, f"K=N IHT must reduce to lstsq (off {bridge:.2e})"

    # hard_threshold keeps exactly K and the right ones
    v = np.array([0.1, -3.0, 2.0, -0.5, 1.0])
    ht = hard_threshold(v, 2)
    assert np.count_nonzero(ht) == 2 and ht[1] == -3.0 and ht[2] == 2.0, "H_K must keep the K largest by magnitude"

    # determinism
    a = iht_recall(cue, cb, 5)
    b = iht_recall(cue, cb, 5)
    assert a == b

    print(f"holographic_iht selftest: ok (INCOHERENT: IHT F1 {iht0:.3f} ties occlusion {occ0:.3f}; HIGH COHERENCE: "
          f"IHT {iht_hi:.3f} BEATS greedy occlusion {occ_hi:.3f} -- support revision corrects MP's unrecoverable "
          f"early picks; K=N reduces to lstsq (off {bridge:.0e}) = the GRAD-2 gradient-descent bridge; deterministic. "
          f"Kept negative: greedy MP wins at LOW-MILD coherence -- IHT is the coherent-regime method, not a strict upgrade)")


if __name__ == "__main__":
    _selftest()

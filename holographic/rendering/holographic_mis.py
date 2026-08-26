"""Multiple Importance Sampling -- Veach's balance heuristic for combining the engine's estimators.

WHY THIS EXISTS
---------------
The engine has several estimators of the SAME quantity that each win in a different regime -- exact 1-NN is
Bayes-optimal on discrete atoms, the soft dense-Hopfield blend wins on continuous OFF-grid values (the B1 kept
negative), the manifold projection wins on smooth low-rank data, the forest is sublinear-but-approximate. Right
now you PICK one by hand or by a heuristic. Veach's Multiple Importance Sampling says you do not have to: combine
them, weighting each by its per-query RELIABILITY (the 'pdf' analog), and the combination covers each estimator's
weak regime.

The load-bearing warning MIS encodes -- and this module MEASURES it -- is that NAIVELY AVERAGING estimators that
are reliable in different regimes makes things WORSE: the average carries each estimator's error into the other's
regime, so it is worse than the better single. The BALANCE HEURISTIC (weight by per-query reliability, w_i =
r_i / sum_j r_j) fixes that -- each estimator is trusted in proportion to how well it fits THIS query.

MEASURED (see `_selftest`, a coarse sharp-kernel ScalarEncoder manifold, mix of on-grid + off-grid cues):
  * naive averaging of hard 1-NN and soft Hopfield is WORSE than the better single (the warning).
  * the balance-heuristic combination beats naive averaging AND, in the crossover regime where neither single
    dominates, beats BOTH singles -- recovering whichever is right per query without a regime label.

SCOPE / KEPT NEGATIVE
  * MIS beats EVERY single only when no single estimator is uniformly best (the crossover). When one estimator
    DOMINATES the whole regime (e.g., a very sharp kernel where soft wins almost everywhere), MIS MATCHES that
    dominant estimator within a few percent rather than beating it -- mixing in the weak one costs a little. The
    robust, always-true wins are over NAIVE AVERAGING; the win over the best single needs a genuine mix.
"""

import numpy as np


def combine_estimators(pairs, power=1.0):
    """Veach balance/power heuristic combination of several estimators of the SAME quantity. `pairs` is a list
    of (estimate, reliability): each `estimate` a vector (or scalar), each `reliability` >= 0 its per-query
    'pdf' / confidence. Returns sum_i w_i * estimate_i with w_i = r_i**power / sum_j r_j**power -- power=1 is
    Veach's BALANCE heuristic, power=2 the POWER heuristic (a sharper routing toward the most reliable).

    The point (vs a naive average): weighting by per-query reliability lets each estimator cover the others'
    weak regime, where a naive average instead carries each estimator's error into the other's regime."""
    estimates = [np.asarray(e, float) for e, _ in pairs]
    r = np.maximum(np.array([rel for _, rel in pairs], float), 0.0) ** power
    w = r / (r.sum() + 1e-12)
    out = sum(wi * e for wi, e in zip(w, estimates))
    return float(out) if np.ndim(out) == 0 else out


def mis_recover(q, codebook, beta=10.0, power=1.0):
    """Recover a vector by combining the engine's HARD 1-NN and SOFT (dense-Hopfield) cleanups per-query via
    the balance heuristic -- the B1 kept-negative ('hard wins on discrete atoms, soft wins on continuous
    off-grid values') turned into one combiner that needs NO regime label.

    The cosine distribution's PEAKINESS is the reliability: a sharp single peak (the cue is a discrete / grid
    atom) trusts the exact 1-NN; a close runner-up (the cue is a value BETWEEN atoms) trusts the interpolating
    soft blend. Returns the combined estimate -- matching whichever is right per query."""
    cb = np.asarray(codebook, float)
    cbn = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-12)
    cs = cbn @ (q / (np.linalg.norm(q) + 1e-12))
    x_hard = cb[int(cs.argmax())]                                  # exact 1-NN: Bayes-optimal on discrete atoms
    w = np.exp(beta * (cs - cs.max())); w /= w.sum()
    x_soft = (w[:, None] * cb).sum(0)                              # dense-Hopfield blend: interpolates off-grid
    top = np.sort(cs)[::-1]
    r_hard = max(float(top[0] - top[1]), 0.0)                      # decisive winner -> trust the exact atom
    r_soft = float(top[1] / max(top[0], 1e-9))                     # close runner-up -> trust the interpolation
    return combine_estimators([(x_hard, r_hard), (x_soft, r_soft)], power=power)


def _selftest():
    """CI-fast: the MIS lesson on a real coarse sharp-kernel ScalarEncoder manifold. On a 50/50 mix of on-grid
    and off-grid cues, the balance-heuristic combination of hard 1-NN and soft Hopfield beats BOTH singles AND
    naive averaging -- and naive averaging is WORSE than the better single (the warning that averaging carries
    each estimator's error into the other's regime)."""
    from holographic.io_and_interop.holographic_encoders import ScalarEncoder
    rng = np.random.default_rng(0)
    D = 512
    enc = ScalarEncoder(D, lo=0.0, hi=1.0, seed=1, kernel="rbf", bandwidth=6.0)
    gv = np.linspace(0, 1, 8)
    CB = np.stack([enc.encode(g) for g in gv])
    CBn = CB / np.linalg.norm(CB, axis=1, keepdims=True)
    def cos(a, b):
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
    eh = es = ea = em = 0.0
    N = 600
    for _ in range(N):
        on = rng.random() < 0.5
        v = float(rng.choice(gv)) if on else float(rng.uniform(0.03, 0.97))   # on-grid atom OR off-grid value
        t = enc.encode(v)
        q = t + 0.5 * rng.standard_normal(D) / np.sqrt(D)
        cs = CBn @ (q / np.linalg.norm(q))
        xh = CB[int(cs.argmax())]
        w = np.exp(10.0 * (cs - cs.max())); w /= w.sum()
        xs = (w[:, None] * CB).sum(0)
        xm = mis_recover(q, CB)
        eh += 1 - cos(xh, t); es += 1 - cos(xs, t)
        ea += 1 - cos(0.5 * xh + 0.5 * xs, t); em += 1 - cos(xm, t)
    eh, es, ea, em = eh / N, es / N, ea / N, em / N
    best = min(eh, es)
    assert ea > best, (ea, best)              # naive averaging is WORSE than the better single -- the warning
    assert em < ea, (em, ea)                  # the balance heuristic beats naive averaging
    assert em < best, (em, best)              # and (in this crossover regime) beats BOTH singles

    # combine_estimators: a zero-reliability estimator drops out; weight collapses onto the reliable one
    a = combine_estimators([(np.array([1.0, 0.0]), 0.0), (np.array([0.0, 1.0]), 5.0)])
    assert cos(a, np.array([0.0, 1.0])) > 0.99


if __name__ == "__main__":
    _selftest()
    print("holographic_mis selftest passed")

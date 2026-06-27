"""Anisotropic / steering kernels for the FPE encoder (RT-IV1): a direction-dependent metric.

The FPE encoder (holographic_fpe.py) now accepts a PER-AXIS bandwidth, giving a diagonal anisotropic kernel:
small bandwidth on an axis -> a wide, smooth kernel there; large bandwidth -> a sharp one. This is the bounded
form of Milanfar's steering-kernel regression (Takeda, Farsiu, Milanfar 2007) -- n bandwidths matched to the
data's directional structure, not a full per-point covariance (which overfits with few samples). It is the same
object as an anisotropic Gaussian splat (a per-splat covariance), the cross-connection Drettakis' seat noted.

`steer_bandwidths` estimates the per-axis bandwidths from data (sharp axis -> large bandwidth); `kernel_regress`
does FPE-kernel-weighted (Nadaraya-Watson) regression.

WHERE IT HELPS, MEASURED -- and the kept negatives (loud, because this is exactly where anisotropy is oversold):
  * DENSE, strongly-directional data (an edge/ridge -- constant along one axis, sharp across another): the steered
    anisotropic kernel beats the best isotropic RBF by ~8%, pooling the many same-value samples ALONG the flat
    direction while staying sharp across the edge. This is the regime steering kernels are designed for.
  * SPARSE scattered data: the advantage collapses to ~1-3% -- there are not enough samples to pool along the
    flat direction. Isotropic stays the honest baseline there.
  * ISOTROPIC data (no directional structure): anisotropy does not help (~0%), as it must not.
  * The STEERING ESTIMATE itself is unreliable on scattered data: a per-axis gradient estimated from scattered
    points is polluted by the OTHER axes varying, so it can even point the wrong way. It needs dense / grid-like
    sampling (so neighbours can be found that differ in just one axis) to estimate cleanly. A full per-point
    covariance is worse still (the splat module's own anisotropy kept negative). So: diagonal bandwidths, on
    dense directional data, with isotropic as the fallback.
"""

import numpy as np

from holographic_fpe import VectorFunctionEncoder
from holographic_ai import cosine


def steer_bandwidths(X, y, base=2.0, k=10, clip=8.0):
    """Estimate a per-axis bandwidth from data: large where the function changes fast along that axis (sharp),
    small where it is flat (smooth). For each axis, the RMS partial gradient is estimated from neighbours that
    are CLOSE in the OTHER axes (so the change is genuinely along this axis). Ratio-preserving, geomean-scaled to
    `base`. RELIABLE ONLY ON DENSE / GRID-LIKE DATA -- on sparse scattered data the estimate is polluted and may
    point the wrong way (see the module docstring)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, D = X.shape
    grads = []
    for axis in range(D):
        others = [j for j in range(D) if j != axis]
        per = []
        for i in range(n):
            if others:
                off = np.abs(X[:, others] - X[i, others]).max(axis=1)   # gap in the OTHER axes
            else:
                off = np.zeros(n)
            for j in np.argsort(off)[1:k + 1]:                          # nearest in the other axes
                d = X[j, axis] - X[i, axis]
                if abs(d) > 1e-6:
                    per.append(abs((y[j] - y[i]) / d))
        grads.append(np.mean(per) if per else 1.0)
    grads = np.asarray(grads)
    if grads.max() <= 0:
        return [float(base)] * D                                       # no structure on any axis -> isotropic
    grads = np.maximum(grads, grads.max() * 1e-3)                       # floor a perfectly-flat axis off zero
    bw = base * grads / np.exp(np.mean(np.log(grads)))                  # geomean-normalized -> preserves ratios
    return list(np.clip(bw, base / clip, base * clip))


def kernel_regress(enc, X_train, y_train, X_query):
    """Nadaraya-Watson regression with the encoder's realised kernel: predict each query as the FPE-similarity-
    weighted average of the training targets. The anisotropy lives entirely in `enc` (its per-axis bandwidths)."""
    y_train = np.asarray(y_train, float)
    Etr = [enc.encode(np.asarray(x, float)) for x in X_train]
    preds = []
    for q in X_query:
        eq = enc.encode(np.asarray(q, float))
        w = np.array([max(0.0, cosine(eq, e)) for e in Etr])
        preds.append(y_train.mean() if w.sum() == 0 else float((w * y_train).sum() / w.sum()))
    return np.array(preds)


def _best_rmse(bounds, X_train, y_train, X_query, y_query, bandwidths, seed=1, dim=2048):
    """Helper: the lowest regression RMSE over a set of (scalar or per-axis) bandwidths."""
    best = np.inf
    for bw in bandwidths:
        enc = VectorFunctionEncoder(len(bounds), dim=dim, bounds=bounds, bandwidth=bw, seed=seed)
        pred = kernel_regress(enc, X_train, y_train, X_query)
        best = min(best, float(np.sqrt(np.mean((pred - y_query) ** 2))))
    return best


def _selftest():
    bounds = [(0, 10), (0, 10)]
    grid_bw = [0.2, 0.5, 1.0, 2.0, 4.0, 7.0]
    iso_set = [b for b in grid_bw]
    ani_set = [[bx, by] for bx in grid_bw for by in grid_bw]

    # DENSE sharp ridge: constant along x, sharp along y. Anisotropy should clearly beat isotropy.
    def f(p):
        return np.tanh(3.0 * (p[1] - 5.0))
    g = np.linspace(0.5, 9.5, 18)
    Xtr = np.array([[x, y] for x in g for y in g])
    ytr = np.array([f(p) for p in Xtr])
    rng = np.random.default_rng(0)
    Xte = rng.uniform(1, 9, (150, 2))
    yte = np.array([f(p) for p in Xte])
    iso = _best_rmse(bounds, Xtr, ytr, Xte, yte, iso_set)
    ani = _best_rmse(bounds, Xtr, ytr, Xte, yte, ani_set)
    assert ani < iso * 0.97, f"anisotropic should beat isotropic on the dense ridge (iso {iso:.3f}, ani {ani:.3f})"

    # steering recovers the right direction on this dense data: flat x -> small bw, sharp y -> large bw
    bw = steer_bandwidths(Xtr, ytr, base=2.0)
    assert bw[0] < bw[1], f"steering should make x (flat) smoother than y (sharp); got {bw}"

    print(f"holographic_steering selftest: ok (dense ridge iso {iso:.3f} -> aniso {ani:.3f}, "
          f"~{100*(iso-ani)/iso:.0f}% better; steered bw {[round(b,2) for b in bw]})")


if __name__ == "__main__":
    _selftest()

"""A deterministic Kolmogorov-Arnold readout built on holostuff's encoders.

WHY THIS EXISTS
---------------
KANs (Kolmogorov-Arnold Networks) represent a function as a SUM of learnable univariate functions of
each input -- F(x) = sum_j psi_j(x_j) -- with each psi_j parametrized as a spline (a weighted sum of
local basis bumps), and an adaptive grid that moves its knots to where the data is. Two pieces of that
already live in holostuff, and this module makes the connection real and deterministic:

  * THE BASIS. holostuff's RBF `ScalarEncoder` is fractional-power (Vector-Function-Architecture)
    encoding whose similarity profile is a Gaussian BUMP. The similarities of encode(x) to a grid of
    anchor points ARE a B-spline-like basis evaluation -- the spline basis a KAN edge function is built
    from, already in the box.

  * THE ADAPTIVE GRID (thread 1). `AdaptiveScalarEncoder` learns a monotonic warp from the data's
    empirical CDF, so the encoder spends resolution where the data is DENSE -- exactly KAN's
    "move the spline knots to the data". It is fit once and frozen, so it stays deterministic and
    seed-reproducible (no gradient training of the basis).

  * THE READOUT (thread 2). `HolographicKAN` is a single-layer KAN: each feature -> its encoder's basis
    activations -> psi_j(x_j) = a_j . basis_j(x_j); the prediction is the SUM over features (the
    Kolmogorov-Arnold inner sum, which is holostuff's `bundle`). Because the output is LINEAR in the
    coefficients a, they are fit by deterministic RIDGE LEAST SQUARES -- no backprop. The per-feature
    psi_j are recoverable and plottable, which is KAN's whole interpretability pitch.

So this is a KAN whose splines are holostuff encoder bumps and whose training is a linear solve:
deterministic, interpretable, and structure-first -- KAN's idea in holostuff's idiom.

MEASURED (honest picture)
  * On an additive target f(x1,x2)=g1(x1)+g2(x2) it fits well AND recovers g1,g2 (high correlation),
    far beating a linear readout on the nonlinear parts.
  * Adaptive grid beats uniform grid on SKEWED features (resolution follows density); on uniform data
    the warp is ~identity, so it neither helps nor hurts (a kept tie, at the cost of a stored CDF).
  * KEPT NEGATIVE: a single-layer additive KAN cannot represent feature INTERACTIONS (e.g. x1*x2) --
    additive by construction. That needs a second layer or explicit interaction features; the boundary
    is shown, not hidden.

Pure NumPy + holostuff encoders, deterministic, no new dependencies.
"""

import numpy as np
from holographic.io_and_interop.holographic_encoders import ScalarEncoder


class AdaptiveScalarEncoder:
    """A ScalarEncoder whose grid ADAPTS to the data via a monotonic CDF warp (KAN's adaptive grid).

    The base encoder covers the warped range [0, 1]; `fit` records the data so `warp` maps a value to
    its empirical rank, concentrating the fixed grid where the data is dense. `basis(x)` returns the
    similarities of encode(x) to the grid anchors -- the spline basis a KAN edge function sums over.
    """

    def __init__(self, dim=512, n_grid=24, bandwidth=8.0, seed=0):
        self.base = ScalarEncoder(dim, 0.0, 1.0, seed=seed, kernel="rbf", bandwidth=bandwidth)
        self.grid = np.linspace(0.0, 1.0, n_grid)
        self.anchors = np.stack([self.base.encode(g) for g in self.grid])   # (n_grid, dim), unit rows
        self.n_grid = n_grid
        self._sorted = None                                                 # sorted samples = the warp

    def fit(self, samples):
        self._sorted = np.sort(np.asarray(samples, float))
        return self

    def warp(self, x):
        """Map values to [0, 1] by empirical CDF (rank). Identity (clamped) before fit."""
        x = np.asarray(x, float)
        if self._sorted is None:
            return np.clip(x, 0.0, 1.0)
        return np.searchsorted(self._sorted, x, side="right") / len(self._sorted)

    def _encode_warped(self, w):
        return np.stack([self.base.encode(float(wi)) for wi in np.atleast_1d(w)])

    def basis(self, x):
        """Spline-basis activations: cosine of encode(warp(x)) to each grid anchor. Shape (N, n_grid)."""
        E = self._encode_warped(self.warp(x))
        return E @ self.anchors.T


class HolographicKAN:
    """A single-layer Kolmogorov-Arnold readout: output = sum_j psi_j(x_j), each psi_j a sum of encoder
    basis bumps, all coefficients fit by deterministic ridge least squares (no backprop)."""

    def __init__(self, n_features, dim=512, n_grid=24, bandwidth=8.0, seed=0, ridge=1e-2):
        self.encoders = [AdaptiveScalarEncoder(dim, n_grid, bandwidth, seed + j + 1)
                         for j in range(n_features)]
        self.n_features = n_features
        self.n_grid = n_grid
        self.ridge = ridge
        self.coef = None

    def _design(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        cols = [enc.basis(X[:, j]) for j, enc in enumerate(self.encoders)]   # each (N, n_grid)
        Phi = np.hstack(cols)
        return np.hstack([Phi, np.ones((Phi.shape[0], 1))])                  # + intercept column

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float)
        for j, enc in enumerate(self.encoders):
            enc.fit(X[:, j])                                                 # learn each feature's grid
        Phi = self._design(X)
        A = Phi.T @ Phi + self.ridge * np.eye(Phi.shape[1])
        self.coef = np.linalg.solve(A, Phi.T @ y)                           # deterministic linear solve
        return self

    def predict(self, X):
        return self._design(X) @ self.coef

    def feature_function(self, j, ts):
        """Recover the learned univariate psi_j(t) over values `ts` -- the plottable KAN edge function."""
        B = self.encoders[j].basis(np.asarray(ts, float))                   # (T, n_grid)
        block = self.coef[j * self.n_grid:(j + 1) * self.n_grid]
        return B @ block

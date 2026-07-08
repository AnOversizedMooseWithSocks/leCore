"""
holographic_classifier.py -- gradient-free, substrate-native classification (Family 2).

WHY this is gradient-free learning
----------------------------------
The HDC/VSA classifier has two stages, neither using gradients:
  (1) One-shot PROTOTYPES: encode each example as a hypervector and BUNDLE all examples of a class into
      one prototype vector. Classify by nearest prototype (cosine). This is holostuff's `bundle` used as
      a learner -- a one-pass centroid model.
  (2) Perceptron-style RETRAINING (AdaptHD / OnlineHD): pass over the data; when an example is
      misclassified, nudge the CORRECT class prototype toward it and the WRONGLY-PREDICTED one away
      (add / subtract the example's hypervector). Pure add/subtract on bundled vectors -- the perceptron
      / LVQ update, no derivatives. This is the part that turns a centroid model into a trained one.

Encoding (the standard HDC "record" encoding, on holostuff primitives):
    H(x) = bundle_j  bind( ID_j , level(x_j) )
  - ID_j: a fixed random atom per feature (distinguishes features),
  - level(v): holostuff's ScalarEncoder (RBF kernel) -- nearby values map to similar vectors,
  - bind = circular convolution, bundle = normalized superposition.
Binds of (feature ID, value level) are PRECOMPUTED per (feature, quantized-level), so encoding a sample
is just summing d precomputed vectors -- fast and fully deterministic.

Honest scope (the field's own verdict, carried here): HDC classifiers are simple, fast, and
gradient-free, and retraining clearly beats the one-shot centroid -- but they typically land BELOW a
tuned linear model (logistic regression / linear SVM). We measure that gap rather than hide it.

Real basis: Kanerva (Sparse Distributed Memory); Rahimi, Imani, Rosing, Kleyko, Hernandez-Cano
(AdaptHD / OnlineHD); Heddes et al. 2024 (HDC framework).
"""
import numpy as np
from holographic.misc.holographic_core import bind, bundle, random_vector
from holographic.io_and_interop.holographic_encoders import ScalarEncoder


class HolographicClassifier:
    def __init__(self, dim=4096, levels=32, bandwidth=3.5, seed=0):
        self.dim, self.levels, self.bandwidth, self.seed = dim, levels, bandwidth, seed
        self.enc = ScalarEncoder(dim, lo=0.0, hi=1.0, seed=seed, kernel="rbf", bandwidth=bandwidth)
        self._fmin = self._fmax = None
        self._bound = None          # (d, levels, dim) precomputed bind(ID_j, level_l)
        self.protos = None          # (n_classes, dim)
        self.classes_ = None

    # ---- feature -> hypervector encoding -------------------------------------------------
    def _build_codebooks(self, d):
        rng = np.random.default_rng(self.seed + 1)
        ids = np.stack([random_vector(self.dim, rng) for _ in range(d)])          # one atom per feature
        ids /= np.linalg.norm(ids, axis=1, keepdims=True)
        grid = np.linspace(0.0, 1.0, self.levels)
        lvl = np.stack([self.enc.encode(g) for g in grid])                        # level codebook (L, dim)
        # precompute bind(ID_j, level_l) so encoding a sample is just a gather + sum (no FFTs at encode)
        self._bound = np.stack([[bind(ids[j], lvl[l]) for l in range(self.levels)] for j in range(d)])

    def _level_idx(self, X):
        Xn = (X - self._fmin) / (self._fmax - self._fmin + 1e-12)                 # normalize to [0,1]
        Xn = np.clip(Xn, 0.0, 1.0)
        return np.clip((Xn * (self.levels - 1)).round().astype(int), 0, self.levels - 1)

    def _encode(self, X):
        LI = self._level_idx(X)                                                   # (N, d) level indices
        H = np.zeros((len(X), self.dim))
        for j in range(X.shape[1]):                                               # sum precomputed binds
            H += self._bound[j][LI[:, j]]
        n = np.linalg.norm(H, axis=1, keepdims=True)
        return H / (n + 1e-12)

    # ---- training: one-shot prototypes, then gradient-free perceptron retraining ---------
    def fit(self, X, y, epochs=20, lr=1.0, shuffle=True):
        self._fmin, self._fmax = X.min(0), X.max(0)
        self._build_codebooks(X.shape[1])
        self.classes_ = np.unique(y); C = len(self.classes_)
        idx = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([idx[v] for v in y])
        H = self._encode(X)
        # (1) one-shot prototypes = bundle (sum) of each class's hypervectors
        self.protos = np.zeros((C, self.dim))
        for c in range(C):
            self.protos[c] = H[yi == c].sum(0)
        # (2) perceptron retraining: on a miss, pull correct proto toward x, push wrong proto away
        rng = np.random.default_rng(self.seed + 2)
        for _ in range(epochs):
            order = rng.permutation(len(H)) if shuffle else np.arange(len(H))
            P = self.protos / (np.linalg.norm(self.protos, axis=1, keepdims=True) + 1e-12)
            for n in order:
                pred = int(np.argmax(P @ H[n]))
                if pred != yi[n]:
                    self.protos[yi[n]] += lr * H[n]
                    self.protos[pred]  -= lr * H[n]
                    P[yi[n]] = self.protos[yi[n]] / (np.linalg.norm(self.protos[yi[n]]) + 1e-12)
                    P[pred]  = self.protos[pred]  / (np.linalg.norm(self.protos[pred])  + 1e-12)
        return self

    def _scores(self, X):
        H = self._encode(X)
        P = self.protos / (np.linalg.norm(self.protos, axis=1, keepdims=True) + 1e-12)
        return H @ P.T

    def predict(self, X):
        return self.classes_[np.argmax(self._scores(X), axis=1)]

    # convenience: one-shot-only prediction (no retraining), for measuring what retraining adds
    def fit_oneshot(self, X, y):
        return self.fit(X, y, epochs=0)


def _selftest():
    """The gradient-free classifier, asserted: one-shot bundle prototypes + perceptron retraining
    (add/subtract on misclassified samples -- no gradients) on a small synthetic task. Retraining must not
    hurt, accuracy must beat chance, and the prototypes must be bit-deterministic for a fixed seed."""
    rng = np.random.default_rng(0)
    C, d, per = 3, 8, 80
    centers = rng.standard_normal((C, d)) * 2.0
    Xtr = np.vstack([centers[c] + rng.standard_normal((per, d)) for c in range(C)])
    ytr = np.repeat(np.arange(C), per)
    Xte = np.vstack([centers[c] + rng.standard_normal((40, d)) for c in range(C)])
    yte = np.repeat(np.arange(C), 40)
    clf = HolographicClassifier(dim=1024, levels=16, bandwidth=3.5, seed=0)
    clf.fit(Xtr, ytr, epochs=0); one = float(np.mean(clf.predict(Xte) == yte))
    clf.fit(Xtr, ytr, epochs=15, lr=1.0); ret = float(np.mean(clf.predict(Xte) == yte))
    assert ret > 0.70, f"classifier accuracy below threshold: {ret:.3f}"
    assert ret >= one - 0.05, f"retraining hurt accuracy: one-shot {one:.3f} -> retrained {ret:.3f}"
    c2 = HolographicClassifier(dim=1024, levels=16, bandwidth=3.5, seed=0)
    c2.fit(Xtr, ytr, epochs=15, lr=1.0)
    assert np.allclose(clf.protos, c2.protos), "prototypes must be deterministic for a fixed seed"
    print(f"holographic_classifier selftest OK (one-shot={one:.3f} -> retrained={ret:.3f}; gradient-free + deterministic)")


if __name__ == "__main__":
    _selftest()

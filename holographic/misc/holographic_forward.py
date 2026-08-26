"""
holographic_forward.py -- the Forward-Forward algorithm: DEPTH from purely LOCAL objectives, no
backpropagation and no settling. Family 4 (the last) of the learning program.

WHY this is the depth corner of the program
--------------------------------------------
The reservoir and prototype classifier are derivative-free but shallow (a linear readout on fixed
features). Equilibrium Propagation learns hidden weights but is a single energy layer that must SETTLE.
Forward-Forward (Hinton, 2022) is the LOCAL-GRADIENT method that stacks MANY layers, each trained by
its OWN objective with no gradient flowing between them -- so it builds depth without a global backward
pass and without relaxation.

The mechanism: replace backprop's forward+backward with TWO forward passes.
  positive pass : real data           -> train each layer to have HIGH "goodness"
  negative pass : data with a WRONG label embedded -> train each layer to have LOW goodness
"Goodness" of a layer is the mean squared activity. Each layer's local loss is a logistic on
(goodness - theta); its weights move by the gradient of THAT loss alone -- local in space, one layer
at a time. Critically, every layer L2-NORMALIZES its output before the next layer sees it, so a later
layer cannot cheat by reading the magnitude an earlier layer already separated -- it must find new
structure. Classification is label-embedded: prepend a one-hot label to the input; at test time try
each label, run forward, and pick the label whose accumulated goodness is highest.

This is LOCAL-GRADIENT (like EP), NOT derivative-free -- each layer follows the gradient of its own
local objective; there is just no backward pass linking the layers. Its niche over EP: arbitrary DEPTH
with no settling, each layer a cheap closed-form local update.

Honest scope kept on the record (MEASURED here, loudly): at the small scale tested this compact FF
is a WORKING but WEAK classifier -- it TRAILS a plain linear / logistic model on every task tried
(two-moons ties ~0.88; overlapping 4-class blobs 0.95 vs 0.99; sklearn digits 0.88 vs logistic 0.97;
a radial task it beats linear ~0.69 vs 0.47 only because linear provably fails there, and even then
weakly). FF's published accuracy (Hinton's ~1.4% MNIST error) needs the full-scale recipe -- many
layers, large width, long training, carefully built negatives -- not reachable in a compact CI-fast
module. What this module DOES demonstrate is the MECHANISM: backprop-free, settling-free DEPTH from
purely local goodness objectives, with positive goodness provably separating from negative. The
contribution is conceptual (the local-objective depth route), not a competitive accuracy number.
Also: goodness-based label inference costs one forward pass per class at test, and FF is sensitive to
the goodness threshold and the negative-data quality.

The stronger refinement, Mono-Forward (2025) -- per-layer LOCAL supervised projections to class logits
instead of the goodness contrast -- is reported to match tuned backprop; it is the natural next step if
a competitive FF-family accuracy is wanted, and is NOT built here.

Real basis: Hinton (2022), *The Forward-Forward Algorithm*; Mono-Forward (2025).
"""
import numpy as np


def _relu(z):
    return np.maximum(z, 0.0)


def _normalize(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)   # L2 per sample: kill the length cue


class ForwardForwardNet:
    """A stack of fully-connected layers trained by Forward-Forward's local goodness objective.
    Each layer: z = h @ W ; a = relu(z) ; goodness = mean(a^2) ; the next layer sees normalize(a)."""

    def __init__(self, n_in, layer_sizes=(64, 64), n_classes=2, theta=0.05,
                 label_scale=3.0, seed=0):
        self.n_in, self.sizes, self.nc = n_in, tuple(layer_sizes), n_classes
        self.theta = np.full(len(self.sizes), float(theta))   # per-layer, adapts to each layer's goodness scale
        self.label_scale = label_scale
        rng = np.random.default_rng(seed)
        self.Ws = []
        din = n_classes + n_in                                   # the label is prepended to the features
        for dout in self.sizes:
            self.Ws.append(rng.standard_normal((din, dout)) * np.sqrt(2.0 / din))
            din = dout
        self._lo = self._hi = None                               # feature standardization (set in fit)

    def _std(self, X):
        X = np.asarray(X, float)
        if self._lo is None:
            return X
        return (X - self._lo) / (self._hi + 1e-8)

    def _embed(self, X, labels):
        oh = np.eye(self.nc)[np.asarray(labels)] * self.label_scale
        return np.concatenate([oh, self._std(X)], axis=1)

    def _goodness(self, inp, accumulate_from=0):
        """Forward `inp` through the stack; return the goodness summed over layers >= accumulate_from.
        All layers count: across label candidates the constant 'a label is present' part cancels in argmax,
        leaving each layer's learned label-vs-input compatibility."""
        h = inp; total = np.zeros(len(inp)); 
        for li, W in enumerate(self.Ws):
            a = _relu(h @ W)
            if li >= accumulate_from:
                total = total + np.mean(a * a, axis=1)
            h = _normalize(a)
        return total

    def goodness_per_label(self, X):
        return np.stack([self._goodness(self._embed(X, np.full(len(X), c))) for c in range(self.nc)], axis=1)

    def predict(self, X):
        return np.argmax(self.goodness_per_label(X), axis=1)

    def fit(self, X, y, epochs=40, lr=0.03, batch=64, seed=0):
        X = np.asarray(X, float); y = np.asarray(y)
        self._lo, self._hi = X.mean(0), X.std(0)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]; Xb, yb = X[b], y[b]; n = len(b)
                wrong = (yb + rng.integers(1, self.nc, size=n)) % self.nc      # a wrong label per sample
                hp = self._embed(Xb, yb); hn = self._embed(Xb, wrong)          # positive / negative inputs
                for li, W in enumerate(self.Ws):
                    zp, zn = hp @ W, hn @ W
                    ap, an = _relu(zp), _relu(zn)
                    H = ap.shape[1]
                    gp, gn = np.mean(ap * ap, axis=1), np.mean(an * an, axis=1)
                    th = self.theta[li]
                    pp, pn = 1.0 / (1.0 + np.exp(-(gp - th))), 1.0 / (1.0 + np.exp(-(gn - th)))
                    # local logistic objective: push pos goodness up (target 1), neg down (target 0).
                    # dL/dg = (p - target); g = mean(a^2) -> dg/da = 2a/H; da/dz = (z>0); dz/dW = h^T
                    dap = ((pp - 1.0)[:, None] * (2.0 / H) * ap) * (zp > 0)
                    dan = ((pn - 0.0)[:, None] * (2.0 / H) * an) * (zn > 0)
                    W -= lr * (hp.T @ dap + hn.T @ dan) / n                     # LOCAL gradient, this layer only
                    self.theta[li] = 0.9 * self.theta[li] + 0.1 * 0.5 * (gp.mean() + gn.mean())  # track scale
                    hp, hn = _normalize(ap), _normalize(an)                     # detached input to the next layer
        return self


def _blobs(n, d, C, sep, rng):
    """A numpy-only multi-class Gaussian-mixture task (C classes in d dims)."""
    centers = rng.standard_normal((C, d)) * sep
    per = n // C
    X = np.vstack([centers[c] + rng.standard_normal((per, d)) for c in range(C)])
    y = np.repeat(np.arange(C), per)
    return X, y


def _selftest():
    """Asserts Forward-Forward's defining behavior -- the MECHANISM, not a competitive accuracy (see the
    module docstring's measured negative): a multi-layer stack trained ONLY by local goodness objectives
    (no backprop, no settling) (1) classifies a separable multi-class task well above chance, and (2) makes
    positive goodness exceed negative goodness on held-out data. Plus determinism for a fixed seed."""
    rng = np.random.default_rng(0)
    C, d = 4, 16
    X, y = _blobs(560, d, C, sep=2.2, rng=rng)
    p = rng.permutation(len(X)); X, y = X[p], y[p]; ntr = 420
    net = ForwardForwardNet(n_in=d, layer_sizes=(100, 100), n_classes=C, theta=0.05, label_scale=4.0, seed=0)
    net.fit(X[:ntr], y[:ntr], epochs=60, lr=0.1, batch=100, seed=0)
    acc = float(np.mean(net.predict(X[ntr:]) == y[ntr:]))
    assert acc > 0.85, f"FF failed to classify separable blobs: {acc:.3f} (chance {1.0 / C:.2f})"
    gap = float(np.mean(net._goodness(net._embed(X[ntr:], y[ntr:])))
                - np.mean(net._goodness(net._embed(X[ntr:], (y[ntr:] + 1) % C))))
    assert gap > 0.0, f"positive goodness does not exceed negative goodness: {gap:.4f}"
    n2 = ForwardForwardNet(n_in=d, layer_sizes=(100, 100), n_classes=C, theta=0.05, label_scale=4.0, seed=0)
    n2.fit(X[:ntr], y[:ntr], epochs=60, lr=0.1, batch=100, seed=0)
    assert np.allclose(net.Ws[0], n2.Ws[0]), "FF weights must be deterministic for a fixed seed"
    print(f"holographic_forward selftest OK (separable-blobs acc={acc:.3f}; pos-neg goodness gap={gap:.3f})")


if __name__ == "__main__":
    _selftest()

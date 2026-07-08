"""
holographic_equilibrium.py -- Equilibrium Propagation: LOCAL-gradient learning for the energy-based
(Hopfield) memory the engine already uses as a fixed cleanup. Family 3 of the learning program.

WHY this is the energy-based learning rule, and how it relates to holostuff
---------------------------------------------------------------------------
holostuff's modern-Hopfield cleanup (B1) is an energy-based associative memory with a FIXED codebook:
it relaxes a query to the nearest stored attractor. Equilibrium Propagation (Scellier & Bengio, 2017)
is the rule that LEARNS those attractors -- the weights of a continuous Hopfield net -- so its energy
minima encode a task rather than being a given codebook.

It uses NO global backpropagation. Relaxations of the same circuit:
  free phase    : clamp the input, relax the state to an energy minimum -> s0  (this is the prediction)
  nudged phases : add a small term (+/- beta) * loss(output, target) to the energy and relax again
The weight update is the DIFFERENCE of the nudged equilibria -- a contrastive Hebbian rule that
Scellier & Bengio proved ESTIMATES the gradient of the loss. We use SYMMETRIC nudging (+beta and -beta,
Laborieux et al. 2021), which cancels the leading finite-beta bias so the estimate matches the true
gradient cleanly. Local in space (each weight uses only the states of the units it connects) and in
time (one circuit, repeated).

So this is the LOCAL-GRADIENT corner of the program -- NOT derivative-free like the reservoir and the
prototype classifier. EP estimates a gradient; it just does so with relaxations instead of a backward
pass. Its payoff over those two: it learns the HIDDEN weights of the energy net, so it fits a NONLINEAR
task (two interleaving moons) that a linear readout on fixed features cannot.

Honest scope, kept on the record: EP needs SYMMETRIC weights; it costs several relaxations (T steps
each) per update, far more than a one-shot rule; the estimate is still biased at finite beta (smaller
beta -> less bias, more noise); and it is validated at small / moderate scale, not frontier scale.

Real basis: Scellier & Bengio (2017), *Equilibrium Propagation*; Laborieux et al. (2021), symmetric
nudging for deep ConvNet EP.
"""
import numpy as np


def _rho(v):
    return np.clip(v, 0.0, 1.0)                       # hard-sigmoid activation (states live in [0,1])


def _rhop(v):
    return ((v >= 0.0) & (v <= 1.0)).astype(float)    # 1 on [0,1] inclusive (so dynamics start from 0)


class EquilibriumNet:
    """A 1-hidden-layer continuous Hopfield net trained by Equilibrium Propagation with symmetric nudging.
    Energy E = 1/2(|h|^2+|o|^2) - rho(h).bh - rho(o).bo - rho(x).Wxh.rho(h) - rho(h).Who.rho(o).
    Weights are symmetric by construction (one matrix per layer pair, used in both directions)."""

    def __init__(self, n_in, n_hidden=64, n_out=2, dt=0.4, t_free=40, t_nudge=12, beta=0.35, seed=0):
        self.ni, self.nh, self.no = n_in, n_hidden, n_out
        self.dt, self.t_free, self.t_nudge, self.beta = dt, t_free, t_nudge, beta
        rng = np.random.default_rng(seed)
        self.Wxh = rng.standard_normal((n_in, n_hidden)) * np.sqrt(1.0 / n_in)
        self.Who = rng.standard_normal((n_hidden, n_out)) * np.sqrt(1.0 / n_hidden)
        self.bh = np.zeros(n_hidden); self.bo = np.zeros(n_out)
        self._lo = self._hi = None                    # input range for [0,1] normalization (set in fit)

    def _norm(self, X):
        X = np.asarray(X, float)
        if self._lo is None:
            return np.clip(X, 0.0, 1.0)
        return np.clip((X - self._lo) / (self._hi - self._lo + 1e-12), 0.0, 1.0)

    def _relax(self, rx, h, o, y=None, beta=0.0, T=30):
        """Relax (h, o) toward an energy minimum; with beta != 0, (weakly) clamp o toward target y."""
        for _ in range(T):
            dh = -h + _rhop(h) * (self.bh + rx @ self.Wxh + _rho(o) @ self.Who.T)     # -dE/dh
            do = -o + _rhop(o) * (self.bo + _rho(h) @ self.Who)                        # -dE/do
            if beta != 0.0 and y is not None:
                do = do - beta * (o - y)                                              # the output clamp
            h = np.clip(h + self.dt * dh, 0.0, 1.0)
            o = np.clip(o + self.dt * do, 0.0, 1.0)
        return h, o

    def free_state(self, X):
        rx = self._norm(np.atleast_2d(X)); N = len(rx)
        return self._relax(rx, np.zeros((N, self.nh)), np.zeros((N, self.no)), T=self.t_free)

    def predict(self, X):
        _, o = self.free_state(X)
        return np.argmax(o, axis=1)

    def _sym_grads(self, rx, h0, o0, yb):
        """Symmetric-nudging EP gradient estimates (dC/dW...) from the +beta and -beta equilibria."""
        hP, oP = self._relax(rx, h0.copy(), o0.copy(), y=yb, beta=+self.beta, T=self.t_nudge)
        hM, oM = self._relax(rx, h0.copy(), o0.copy(), y=yb, beta=-self.beta, T=self.t_nudge)
        n = len(rx); c = 1.0 / (2.0 * self.beta * n)
        gWho = c * (_rho(hM).T @ _rho(oM) - _rho(hP).T @ _rho(oP))          # ~ dC/dWho
        gWxh = c * (rx.T @ _rho(hM) - rx.T @ _rho(hP))                      # ~ dC/dWxh
        gbo = c * (_rho(oM) - _rho(oP)).sum(0)
        gbh = c * (_rho(hM) - _rho(hP)).sum(0)
        return gWxh, gWho, gbh, gbo

    def fit(self, X, y_onehot, epochs=30, lr=0.3, batch=64, seed=0):
        X = np.asarray(X, float); y_onehot = np.asarray(y_onehot, float)
        self._lo, self._hi = X.min(0), X.max(0)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]; rx = self._norm(X[b]); yb = y_onehot[b]; n = len(b)
                h0, o0 = self._relax(rx, np.zeros((n, self.nh)), np.zeros((n, self.no)), T=self.t_free)
                gWxh, gWho, gbh, gbo = self._sym_grads(rx, h0, o0, yb)
                self.Wxh -= lr * gWxh; self.Who -= lr * gWho               # gradient DESCENT on the loss
                self.bh -= lr * gbh; self.bo -= lr * gbo
        return self

    # ---- EP's defining correctness property: its update estimates the true loss gradient ----
    def ep_gradient_Who(self, x, y):
        """EP's symmetric estimate of dLoss/dWho from the nudged equilibria (one sample)."""
        rx = self._norm(np.atleast_2d(x)); yb = np.atleast_2d(y)
        h0, o0 = self._relax(rx, np.zeros((1, self.nh)), np.zeros((1, self.no)), T=self.t_free)
        return self._sym_grads(rx, h0, o0, yb)[1]

    def fd_gradient_Who(self, x, y, eps=1e-4):
        """True dLoss/dWho by central finite differences over the free-phase loss (one sample)."""
        rx = self._norm(np.atleast_2d(x)); yb = np.atleast_2d(y)
        def loss():
            _, o = self._relax(rx, np.zeros((1, self.nh)), np.zeros((1, self.no)), T=self.t_free)
            return 0.5 * float(np.sum((o - yb) ** 2))
        g = np.zeros_like(self.Who)
        for i in range(self.nh):
            for j in range(self.no):
                w = self.Who[i, j]
                self.Who[i, j] = w + eps; cp = loss()
                self.Who[i, j] = w - eps; cm = loss()
                self.Who[i, j] = w; g[i, j] = (cp - cm) / (2 * eps)
        return g


def _moons(n, noise, rng):
    """Two interleaving half-moons (numpy-only) -- a NONLINEAR 2-class task a linear model cannot separate."""
    t = rng.uniform(0, np.pi, n // 2)
    up = np.c_[np.cos(t), np.sin(t)]
    dn = np.c_[1.0 - np.cos(t), 0.5 - np.sin(t)]
    X = np.vstack([up, dn]) + noise * rng.standard_normal((2 * (n // 2), 2))
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    return X, y


def _selftest():
    """Asserts EP's two defining claims: (1) its symmetric contrastive update ESTIMATES the true loss
    gradient (cosine vs finite differences), and (2) it LEARNS a nonlinear task (two moons) past what a
    linear model reaches -- the payoff of learning hidden weights. Plus determinism for a fixed seed."""
    # (1) gradient-matching: EP symmetric estimate vs finite-difference truth (tiny net, one sample)
    net = EquilibriumNet(n_in=2, n_hidden=6, n_out=2, beta=0.1, dt=0.3, t_free=80, t_nudge=40, seed=1)
    x = np.array([0.4, 0.7]); y = np.array([1.0, 0.0])
    g_ep, g_fd = net.ep_gradient_Who(x, y).ravel(), net.fd_gradient_Who(x, y).ravel()
    cos = float(g_ep @ g_fd / (np.linalg.norm(g_ep) * np.linalg.norm(g_fd) + 1e-12))
    assert cos > 0.9, f"EP gradient estimate does not match finite differences: cosine {cos:.3f}"
    # (2) learns two moons (nonlinear); a linear least-squares classifier is the foil
    rng = np.random.default_rng(0)
    X, yy = _moons(360, 0.10, rng)
    perm = rng.permutation(len(X)); X, yy = X[perm], yy[perm]   # shuffle so the split is class-balanced
    Y = np.eye(2)[yy]; ntr = 260
    net = EquilibriumNet(n_in=2, n_hidden=48, n_out=2, beta=0.35, dt=0.35, t_free=45, t_nudge=12, seed=0)
    net.fit(X[:ntr], Y[:ntr], epochs=100, lr=0.3, batch=90, seed=0)
    acc = float(np.mean(net.predict(X[ntr:]) == yy[ntr:]))
    Xa = np.c_[X[:ntr], np.ones(ntr)]; w = np.linalg.lstsq(Xa, Y[:ntr], rcond=None)[0]
    lin = float(np.mean(np.argmax(np.c_[X[ntr:], np.ones(len(X) - ntr)] @ w, 1) == yy[ntr:]))
    assert acc > 0.88, f"EP failed to learn two moons: acc {acc:.3f}"
    assert acc > lin + 0.03, f"EP ({acc:.3f}) did not clearly beat the linear foil ({lin:.3f})"
    n2 = EquilibriumNet(n_in=2, n_hidden=48, n_out=2, beta=0.35, dt=0.35, t_free=45, t_nudge=12, seed=0)
    n2.fit(X[:ntr], Y[:ntr], epochs=100, lr=0.3, batch=90, seed=0)
    assert np.allclose(net.Who, n2.Who), "EP weights must be deterministic for a fixed seed"
    print(f"holographic_equilibrium selftest OK (grad cosine={cos:.3f}; two-moons EP={acc:.3f} vs linear={lin:.3f})")


if __name__ == "__main__":
    _selftest()

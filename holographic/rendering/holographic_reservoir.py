"""
holographic_reservoir.py -- gradient-free sequence learning on the holostuff substrate.

WHY reservoir computing fits holostuff
--------------------------------------
In an Echo-State Network (Jaeger) the recurrent reservoir is FIXED and RANDOM; only a linear
readout is trained, and it is trained by closed-form ridge regression -- no gradients, no
backpropagation-through-time. The single learned object is the readout matrix W_out, solved in
one linear-algebra step. That makes this the cheapest, *truly* derivative-free entry to
substrate-native learning (everything else -- Forward-Forward, Equilibrium Propagation -- needs at
least a local derivative).

The reason it is substrate-native and not just "an ESN bolted on": holostuff's permutation operator
`permute` is a cyclic shift (np.roll), which is norm-preserving (orthogonal -- eigenvalues on the
unit circle). That is precisely the property an echo-state reservoir wants for the echo-state
property and good memory capacity. So the engine's own sequence operator already IS a near-optimal
reservoir recurrence; we scale it below the unit circle (rho < 1) so memory fades, add a leaky
nonlinear update, and read out linearly.

State update (leaky-integrator nonlinear reservoir):
    x_t = (1 - leak) * x_{t-1} + leak * tanh( rho * recur(x_{t-1}) + W_in @ u_t )
where recur() is the FIXED recurrence: by default holostuff's `permute` (cyclic shift), optionally a
fixed random permutation. W_in is a fixed random input projection. NOTHING in the reservoir is
trained.

Readout (the only learned weights):
    y_t = W_out @ [x_t ; 1]
solved by ridge (Tikhonov) regression:  W_out = (XᵀX + λI)^{-1} Xᵀ Y  -- closed form, deterministic.

Real basis: Jaeger (Echo State Networks); Maass (Liquid State Machines); orthogonal / permutation
reservoirs are known to have near-optimal memory capacity. Readout = Tikhonov-regularized regression.
"""
import numpy as np
from holographic.misc.holographic_core import permute        # the engine's real recurrence operator  cyclic shift


class HolographicESN:
    def __init__(self, n_in, dim=600, rho=0.95, leak=0.3, in_scale=0.6, seed=0, recurrence="shift"):
        # rho: spectral-radius-like scaling of the (orthogonal) recurrence -- < 1 gives fading memory.
        # leak: leaky-integrator rate -- smaller = longer memory / smoother state.
        # recurrence: "shift" uses holostuff's permute (cyclic shift); "perm" uses a fixed random
        #             permutation (richer mixing). Both are VSA permutation-family operators.
        self.dim, self.rho, self.leak, self.recurrence = dim, rho, leak, recurrence
        rng = np.random.default_rng(seed)
        self.W_in = rng.standard_normal((dim, n_in)) * in_scale      # FIXED random input projection
        self.perm = rng.permutation(dim)                             # FIXED random permutation (variant)
        self.W_out = None

    def _recur(self, x):
        # The FIXED recurrence operator (norm-preserving). Default: holostuff's permute (cyclic shift).
        return permute(x, 1) if self.recurrence == "shift" else x[self.perm]

    def _step(self, x, u):
        pre = self.rho * self._recur(x) + self.W_in @ u
        return (1.0 - self.leak) * x + self.leak * np.tanh(pre)

    def run(self, U):
        """Drive the reservoir with input sequence U (T, n_in) -> reservoir states X (T, dim)."""
        U = np.atleast_2d(U); U = U if U.shape[0] != 1 or U.shape[1] == self.W_in.shape[1] else U.T
        x = np.zeros(self.dim); X = np.empty((len(U), self.dim))
        for t in range(len(U)):
            x = self._step(x, U[t]); X[t] = x
        self._state = x
        return X

    def fit(self, U, Y, ridge=1e-4, washout=100, noise=0.0):
        """Train ONLY the readout by ridge regression. U:(T,n_in)  Y:(T,) or (T,out).
        noise: std of Gaussian state noise injected before the solve -- the standard ESN trick that
        makes the readout robust to its own drift, which is what keeps closed-loop GENERATION stable."""
        X = self.run(U)
        if noise > 0:
            X = X + noise * np.random.default_rng(12345).standard_normal(X.shape)
        Xa = np.hstack([X, np.ones((len(X), 1))])                    # augment with a bias column
        Y = np.atleast_2d(Y); Y = Y.T if Y.shape[0] == 1 else Y
        A, B = Xa[washout:], Y[washout:]
        # ridge (Tikhonov) closed form -- THE learned step; gradient-free and fully deterministic.
        self.W_out = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ B)
        return self

    def predict(self, U):
        X = self.run(U)
        return np.hstack([X, np.ones((len(X), 1))]) @ self.W_out

    def generate(self, n_steps, warm_U, feedback):
        """Closed-loop autoregressive generation: feed each prediction back as the next input.
        warm_U primes the reservoir; feedback(y_vec) -> next input vector. Returns (n_steps, out)."""
        x = np.zeros(self.dim)
        for u in np.atleast_2d(warm_U): x = self._step(x, u)
        outs = []
        y = np.hstack([x, [1.0]]) @ self.W_out
        for _ in range(n_steps):
            u = np.atleast_1d(feedback(y))
            x = self._step(x, u)
            y = np.hstack([x, [1.0]]) @ self.W_out
            outs.append(y)
        return np.array(outs)


def _selftest():
    """The gradient-free learning mechanism, asserted: a FIXED reservoir (holostuff's permute recurrence)
    plus a single ridge-regression readout -- the ONLY trained weights -- learns one-step prediction of a
    signal, and the readout is bit-deterministic for a fixed seed (the determinism discipline)."""
    t = np.arange(1600); s = np.sin(t / 4.0) + 0.3 * np.sin(t / 9.0)
    esn = HolographicESN(n_in=1, dim=400, rho=0.95, leak=0.5, in_scale=0.6, seed=0)
    esn.fit(s[:1100, None], s[1:1101], ridge=1e-6, washout=100)
    pr = esn.predict(s[1100:1500, None]).ravel(); tgt = s[1101:1501]
    sl = slice(60, None)                                    # skip the test-time reservoir transient
    err = float(np.sqrt(np.mean((tgt[sl] - pr[sl]) ** 2) / (np.var(tgt[sl]) + 1e-12)))
    assert err < 0.4, f"reservoir next-step NRMSE too high: {err:.3f}"
    assert esn.W_out is not None, "readout not learned"
    e2 = HolographicESN(n_in=1, dim=400, rho=0.95, leak=0.5, in_scale=0.6, seed=0)
    e2.fit(s[:1100, None], s[1:1101], ridge=1e-6, washout=100)
    assert np.allclose(esn.W_out, e2.W_out), "readout must be deterministic for a fixed seed"
    print(f"holographic_reservoir selftest OK (next-step NRMSE={err:.3f}; gradient-free + deterministic)")


if __name__ == "__main__":
    _selftest()

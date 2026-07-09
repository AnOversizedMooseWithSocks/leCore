"""Propagator binding -- dynamics as an algebra of binds.

WHY THIS EXISTS
---------------
Represent an evolving state as a vector and learn a fixed operator U so that
    state(t+1) ~ bind(U, state(t)).
In HRR's Fourier domain binding is elementwise complex multiplication (bind = irfft(rfft(a)*rfft(b))),
so a learned bind operator is exactly a PER-FREQUENCY COMPLEX TRANSFER -- the Koopman operator in
Fourier coordinates, the same object Stam's FFT fluid propagation and Puckette's phase vocoder
manipulate, and what Dynamic Mode Decomposition learns. Two payoffs:

  * PREDICTION IS ONE BIND. step(state) = bind(U, state) advances the state in O(n log n).
  * THE TRAJECTORY IS CONTENT-ADDRESSABLE. An inverse operator recovers an earlier state:
    recall_at(state_now, k) = bind(U_inv, ...) applied k times -> "the state k steps ago".

MEASURED (honest picture)
  * On a signal WITH genuine dynamics (a control: decaying sinusoids), the learned propagator beats
    BOTH persistence and a mean predictor at one-step prediction.
  * KEPT NEGATIVE on real SOL returns: it only TIES a trivial mean predictor (and beats persistence,
    a strawman for returns). Near-efficient-market returns have almost no linear structure for a
    fixed operator to exploit -- the correct, expected result, kept on the record.
  * The CONTENT-ADDRESSABLE round-trip is the durable win regardless: forward k then back k returns
    the start at cosine ~1.0.

DESIGN NOTES
  * `step` is literally holostuff's `bind(U, state)` -- the operator is a real hypervector U, so
    "dynamics as an algebra of binds" is exact, not a metaphor.
  * The inverse uses a Wiener-regularised transfer conj(H)/(|H|^2 + eps): exact where the operator
    has energy, bounded where it does not. This is the Plate tradeoff made explicit -- an exact
    deconvolution inverse is precise but amplifies noise at near-null frequencies; the regularised
    inverse trades a hair of round-trip accuracy for robustness.
  * Pure NumPy, deterministic.
"""

import warnings

import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind


class Propagator:
    """A learned dynamics operator: state(t+1) ~ bind(U, state(t)), with content-addressable history."""

    def __init__(self, U, U_inv):
        self.U = U                  # the bind operator as a real hypervector (prediction)
        self.U_inv = U_inv          # the (regularised) inverse operator (history recall)

    @classmethod
    def learn(cls, states, ridge=1e-3):
        """Learn the per-frequency least-squares transfer H[k] = sum_t X1[k] conj(X0[k]) / sum_t |X0[k]|^2
        from a sequence of state rows, then realise it (and its regularised inverse) as bind operators.
        `ridge` regularises both the fit and the inverse against low-energy frequencies."""
        X = np.asarray(states, float)
        n = X.shape[1]
        F = np.fft.rfft(X, axis=1)
        X0, X1 = F[:-1], F[1:]
        num = (X1 * np.conj(X0)).sum(0)
        den = (np.abs(X0) ** 2).sum(0)
        H = num / (den + ridge * (den.max() + 1e-12))          # learned transfer (the bind operator)
        eps = ridge * (np.abs(H) ** 2).mean() + 1e-12
        H_inv = np.conj(H) / (np.abs(H) ** 2 + eps)            # Wiener-regularised inverse
        U = np.fft.irfft(H, n=n)
        U_inv = np.fft.irfft(H_inv, n=n)
        return cls(U, U_inv)

    @classmethod
    def learn_pairs(cls, X0, X1, ridge=1e-3):
        """Learn the transfer from explicit (state -> next_state) PAIRS (X0[i] -> X1[i]) instead of one running
        sequence -- the action-conditioned case, where every successor comes from applying the SAME action. Same
        per-frequency least-squares transfer as learn(), just fed paired rows; this is what lets one Propagator
        model one action (see holographic_condprop.ConditionalPropagator)."""
        X0 = np.asarray(X0, float); X1 = np.asarray(X1, float)
        n = X0.shape[1]
        F0 = np.fft.rfft(X0, axis=1); F1 = np.fft.rfft(X1, axis=1)
        num = (F1 * np.conj(F0)).sum(0)
        den = (np.abs(F0) ** 2).sum(0)
        H = num / (den + ridge * (den.max() + 1e-12))
        eps = ridge * (np.abs(H) ** 2).mean() + 1e-12
        H_inv = np.conj(H) / (np.abs(H) ** 2 + eps)
        return cls(np.fft.irfft(H, n=n), np.fft.irfft(H_inv, n=n))

    def step(self, state):
        """One-step prediction = a single bind with the learned operator."""
        return bind(self.U, np.asarray(state, float))

    def spectral_radius(self):
        """max |eigenvalue| of the operator. The bind operator is DIAGONAL in the Fourier basis, so its eigenvalues
        are just its rfft (holographic_iterate.transfer) -- free, no O(n^3) eigendecomposition. >1 means the iterate
        blows up: k steps will overflow to inf/nan rather than converge."""
        from holographic.misc.holographic_iterate import transfer
        return float(np.abs(transfer(self.U)).max())

    # A learned operator very often sits a hair above 1 (e.g. 1.0005) and is perfectly usable -- 1.0005**5 = 1.003.
    # What actually hurts is GROWTH BIG ENOUGH TO DESTROY THE ANSWER: measured, max|eig|=261 overflows to nan by
    # k=200. So the guard fires on the predicted growth r**k, not on r>1, and stays quiet for benign operators.
    _GROWTH_LIMIT = 1e12

    def _warn_if_divergent(self, k, what):
        """A divergent operator makes k steps silently overflow to inf/nan. We WARN rather than raise, because
        raising would change the behaviour of existing callers -- but the caller now finds out instead of quietly
        reading nans. `jump()`/`limit()` raise, since there the answer is genuinely undefined."""
        r = self.spectral_radius()
        if r <= 1.0:
            return
        with np.errstate(over="ignore"):
            # NB: python's float**k raises OverflowError; numpy returns inf, which is what we want to test for.
            growth = float(np.float_power(np.float64(r), np.float64(k)))   # how much the iterate scales the state
        if not np.isfinite(growth) or growth > self._GROWTH_LIMIT:
            warnings.warn(
                "Propagator.%s(k=%d): operator diverges (max|eigenvalue| = %.4f > 1, growth ~%.1e); the iterate will "
                "overflow to inf/nan. Use jump()/limit() for a clean error, or re-learn with more ridge."
                % (what, k, r, growth), RuntimeWarning, stacklevel=3)

    def rollout(self, state, k):
        """Predict k steps ahead; returns the k predicted states (shape (k, dim)).

        NOTE the contract: this returns the WHOLE TRAJECTORY, not just the k-th state -- compare against
        `rollout(...)[-1]`, never against the array itself. For only the k-th state use `jump()`, which is the
        closed form and is ~50-1300x faster.

        RT-I1: the eigen-decomposition of a bind operator is its rfft, so the i-th state is one inverse FFT of
        `rfft(state) * transfer**i`. We take that transform ONCE and raise the transfer by one power per step,
        instead of paying two FFTs per step inside `bind`. Identical output (it is the same math), just cheaper.
        """
        from holographic.misc.holographic_iterate import transfer
        s = np.asarray(state, float)
        n = s.shape[0]
        self._warn_if_divergent(k, "rollout")
        F = np.fft.rfft(s)
        H = transfer(self.U)
        acc = H.copy()                                        # transfer**(i+1) carried forward, one multiply per step
        out = np.empty((k, n))
        for i in range(k):
            out[i] = np.fft.irfft(F * acc, n=n)
            acc *= H
        return out

    def jump(self, state, k):
        """The state k steps ahead in ONE closed-form evaluation (no loop): `iterate.step_k`, i.e. the transfer
        raised to the k-th power. Matches k sequential `step()` calls to FFT tolerance (measured cos 1.000000) and
        is 50x (k=64) to 1300x (k=4096) faster -- k=1e6 costs the same as k=1. Raises on a divergent operator,
        where the loop would silently overflow to nan. This is what `UnifiedMind.propagator_jump` already used;
        it now lives on the Propagator itself so every caller gets it."""
        from holographic.misc.holographic_iterate import step_k
        r = self.spectral_radius()
        with np.errstate(over="ignore"):
            growth = float(np.float_power(np.float64(r), np.float64(k))) if r > 1.0 else 1.0
        if not np.isfinite(growth) or growth > self._GROWTH_LIMIT:
            raise ValueError("operator diverges (max|eigenvalue| = %.4f > 1, growth ~%.1e over %d steps): the "
                             "k-step iterate has no usable finite value" % (r, growth, k))
        return step_k(np.asarray(state, float), self.U, k)

    def limit(self, state, tol=1e-6):
        """The k -> infinity steady state, in closed form (`iterate.limit`): decaying modes vanish, persistent
        (|eigenvalue| ~ 1) modes remain. No iteration, and no k to choose. Raises if the operator diverges."""
        from holographic.misc.holographic_iterate import limit as _limit
        return _limit(np.asarray(state, float), self.U, tol=tol)

    def persistent_projection(self, state, tol=1e-6):
        """Project a state onto the operator's PERSISTENT (non-decaying) eigenspace: keep the Fourier modes with
        |eigenvalue| ~ 1, drop the rest. Whatever survives infinite iteration lives here.

        P7 -- this is what lets `dynamics` join the `project_onto_constraints` family, and it is NOT `limit()`.
        `limit` keeps the eigenvalue H on the surviving modes, so applying it twice multiplies by H^2: measured
        |P(Px) - Px| = 1.61, i.e. it is not idempotent and therefore not a projection. Masking those modes with 1.0
        instead IS idempotent (measured 2.2e-16) and the subspace is invariant under `step` (2.2e-16), which is
        exactly what a constraint set must be. Use `constraint()` to hand it to the iterated-projection engine.
        """
        from holographic.misc.holographic_iterate import transfer
        x = np.asarray(state, float)
        keep = (np.abs(transfer(self.U)) >= 1.0 - tol).astype(float)
        return np.fft.irfft(np.fft.rfft(x) * keep, n=x.shape[0])

    def constraint(self, tol=1e-6):
        """This propagator as a CONSTRAINT for `denoise.project_onto_constraints`: a callable x -> x' snapping a
        state onto the dynamics' invariant subspace. Compose it with other projections (a codebook cleanup, a data
        term, a collision set) and the shared iterate-a-projection engine solves them together."""
        return lambda x: self.persistent_projection(x, tol=tol)

    def recall_at(self, state_now, k):
        """Recover the state k steps BEFORE `state_now` by applying the inverse operator k times --
        the trajectory is content-addressable, not just forward-runnable.

        RT-I1: k inverse binds ARE the inverse transfer raised to the k-th power, so this is one closed-form
        `iterate.step_k` call against `U_inv` -- same answer (cos 1.000000), ~50-1300x faster, no loop."""
        from holographic.misc.holographic_iterate import step_k
        self._warn_if_divergent(k, "recall_at")
        return step_k(np.asarray(state_now, float), self.U_inv, k)

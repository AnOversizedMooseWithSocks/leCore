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

    def rollout(self, state, k):
        """Predict k steps ahead; returns the k predicted states (shape (k, dim))."""
        s = np.asarray(state, float)
        out = np.empty((k, s.shape[0]))
        for i in range(k):
            s = self.step(s)
            out[i] = s
        return out

    def recall_at(self, state_now, k):
        """Recover the state k steps BEFORE `state_now` by applying the inverse operator k times --
        the trajectory is content-addressable, not just forward-runnable."""
        s = np.asarray(state_now, float)
        for _ in range(k):
            s = bind(self.U_inv, s)
        return s

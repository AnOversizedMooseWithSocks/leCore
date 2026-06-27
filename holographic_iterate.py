"""Spectral iteration of a bind operator (RT-I1): diagonalise once, evaluate any level or the limit in closed form.

The unification: subdivision (Stam's exact eval), the dynamics propagator's k-step rollout, the diffusion
sampler's steady state, and the resonator's fixed points are ALL "iterate a linear operator." And the operator
here is a `bind` (circular convolution), which is DIAGONAL in the Fourier basis: its eigenvalues are simply its
rfft spectrum, its eigenvectors the Fourier modes. So the eigendecomposition is FREE -- it is the FFT, not a
dense O(n^3) decomposition. That is the whole point: live in the Fourier/structured form where the spectrum is
free, never a dense SVD at D=4096 (which is exactly what the topology module timed out on).

Given the bind operator `U` (a real hypervector, e.g. a learned dynamics Propagator's `U`):
  * its k-step iterate is ONE eval -- raise each frequency's transfer to the k-th power -- not k binds;
  * its limit is closed-form -- decaying modes (|eigenvalue|<1) vanish, persistent modes (|.|=1) remain;
  * convergence/stall is READ OFF the spectrum before running -- the regime from max|eigenvalue|, and the
    power-iteration rate from the spectral gap |lambda_2|/|lambda_1|.

KEPT NEGATIVES: only LINEAR operators diagonalise this way; a nonlinear iteration (the true resonator's
alternating projection + cleanup) needs delay-embedding and the spectral prediction is only a heuristic there
(the dynamics module's own nonlinearity negative). The clean, exact results below are for the linear iterate; the
resonator connection is the nonlinear cousin. Dense eigendecomposition is avoided entirely -- everything is the
rfft. Eigenvector sign is pinned for determinism (the ISA-1 fence).
"""

import numpy as np


def transfer(U):
    """The eigenvalues of the bind operator `U`: its rfft. (Circular convolution is diagonal in the Fourier
    basis, so this IS the eigendecomposition -- free, no dense O(n^3) work.)"""
    return np.fft.rfft(np.asarray(U, float))


def step_k(state, U, k):
    """Jump `k` iterations of `x <- bind(U, x)` in ONE eval: raise the transfer to the k-th power. Matches k
    sequential binds to FFT tolerance (~1e-15)."""
    state = np.asarray(state, float)
    return np.fft.irfft(np.fft.rfft(state) * (transfer(U) ** k), n=state.shape[0])


def limit(state, U, tol=1e-6):
    """The closed-form limit of the iterate as k -> infinity. Decaying modes (|eigenvalue| < 1) vanish; persistent
    modes (|.| ~ 1) remain. A purely contractive operator has limit 0 (no iteration needed). Raises if the
    operator diverges (any |eigenvalue| > 1)."""
    state = np.asarray(state, float)
    H = transfer(U)
    mag = np.abs(H)
    if mag.max() > 1.0 + tol:
        raise ValueError(f"operator diverges (max|eigenvalue| = {mag.max():.4f} > 1): no finite limit")
    persistent = np.where(mag >= 1.0 - tol, H, 0.0)          # keep |.|~1 modes, drop the decaying ones
    return np.fft.irfft(np.fft.rfft(state) * persistent, n=state.shape[0])


def dominant_eigenvector(U):
    """The Fourier mode with the largest |eigenvalue| -- the direction power iteration `x <- bind(U,x)/|.|`
    converges to. Sign-pinned (largest-magnitude entry positive) for determinism."""
    U = np.asarray(U, float)
    H = transfer(U)
    j = int(np.argmax(np.abs(H)))
    e = np.zeros_like(H)
    e[j] = 1.0
    v = np.fft.irfft(e, n=U.shape[0])
    if v[int(np.argmax(np.abs(v)))] < 0:                      # sign convention -> deterministic
        v = -v
    return v / (np.linalg.norm(v) + 1e-12)


def spectral_profile(U, tol=1e-6):
    """Read convergence behaviour off the spectrum WITHOUT running. Returns max_magnitude (spectral radius),
    regime (contractive -> decays to 0; marginal -> persists; divergent -> blows up), spectral_gap
    (|lambda_2|/|lambda_1|; small gap -> slow power iteration / near-degenerate stall), and the dominant frequency."""
    mag = np.abs(transfer(U))
    order = np.argsort(mag)[::-1]
    m1 = float(mag[order[0]])
    m2 = float(mag[order[1]]) if mag.size > 1 else 0.0
    regime = "contractive" if m1 < 1.0 - tol else ("divergent" if m1 > 1.0 + tol else "marginal")
    return {"max_magnitude": m1, "regime": regime,
            "spectral_gap": (m2 / m1 if m1 > 0 else 0.0), "dominant_freq": int(order[0])}


def _selftest():
    from holographic_dynamics import Propagator
    from holographic_ai import bind, cosine
    n = 256
    rng = np.random.default_rng(0)

    def make_U(H):
        return np.fft.irfft(H, n=n)

    state = rng.standard_normal(n)

    # (1) the k-step jump matches the k-bind rollout to FFT tolerance
    U = make_U(0.9 * np.exp(1j * rng.uniform(0, 2 * np.pi, n // 2 + 1)))
    P = Propagator(U, U)
    for k in (1, 3, 8, 20):
        assert np.max(np.abs(P.rollout(state, k)[-1] - step_k(state, U, k))) < 1e-9, f"k={k}"

    # (2) regime read off the spectrum matches the actual behaviour, predicted BEFORE running
    contr = make_U(0.85 * np.exp(1j * rng.uniform(0, 2 * np.pi, n // 2 + 1)))
    assert spectral_profile(contr)["regime"] == "contractive"
    assert np.linalg.norm(step_k(state, contr, 60)) < 0.05 * np.linalg.norm(state)   # decays
    assert np.linalg.norm(limit(state, contr)) < 1e-9                                # closed-form limit is 0
    div = make_U(1.08 * np.exp(1j * rng.uniform(0, 2 * np.pi, n // 2 + 1)))
    assert spectral_profile(div)["regime"] == "divergent"
    assert np.linalg.norm(step_k(state, div, 40)) > 5 * np.linalg.norm(state)        # blows up

    # (3) the spectral gap predicts power-iteration convergence speed (linear cousin of resonator stall)
    def power_iter_steps(U, tol=1e-4, maxit=300):
        x = rng.standard_normal(n); x /= np.linalg.norm(x); prev = None
        for it in range(maxit):
            x = bind(U, x); x /= np.linalg.norm(x)
            if prev is not None and 1 - abs(cosine(x, prev)) < tol:
                return it
            prev = x.copy()
        return maxit
    big = np.full(n // 2 + 1, 0.2, dtype=complex); big[3] = 1.0; big[7] = 0.4
    small = np.full(n // 2 + 1, 0.2, dtype=complex); small[3] = 1.0; small[7] = 0.95
    gap_big = spectral_profile(make_U(big))["spectral_gap"]
    gap_small = spectral_profile(make_U(small))["spectral_gap"]
    assert gap_big < gap_small                                                       # smaller gap = slower
    assert power_iter_steps(make_U(big)) < power_iter_steps(make_U(small))

    print(f"holographic_iterate selftest: ok (k-step jump exact; regime + gap read off the free FFT spectrum; "
          f"gaps {gap_big:.2f} < {gap_small:.2f})")


if __name__ == "__main__":
    _selftest()

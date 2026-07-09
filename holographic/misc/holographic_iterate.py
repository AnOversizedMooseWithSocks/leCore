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
    from holographic.simulation_and_physics.holographic_dynamics import Propagator
    from holographic.agents_and_reasoning.holographic_ai import bind, cosine
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


# ---------------------------------------------------------------------------------------------------------------
# RE-ENABLE (adaptive-dispatch audit): the closed-form iterate is EXACT and nearly free -- but ONLY for a LINEAR
# operator that is a circular convolution (a bind), because only then is it diagonal in the Fourier basis. That is
# the kept negative ("only LINEAR operators diagonalise this way"). With adaptive dispatch we can DETECT the regime
# and jump k iterations in one FFT where it holds, and step where it doesn't -- and because the closed form is
# EXACT in its regime, the gate can NEVER do worse than stepping (it either matches it or falls back to it).
#
# THE DETECTOR (decidable, deterministic). An operator `op` is a circular convolution iff op(x) = bind(kernel, x)
# for kernel = op(impulse) (its impulse response). Recover the kernel, then verify op == convolve-by-kernel on a
# few seeded random probes. Pass -> use the closed form; fail -> step. No harm either way.

def bind_kernel_of(op, dim):
    """If `op` is a circular convolution, its kernel is its impulse response op([1,0,0,...]). (Any op's response to
    the unit impulse; only meaningful as a kernel when op turns out to be a convolution -- checked separately.)"""
    import numpy as _np
    delta = _np.zeros(int(dim), float)
    delta[0] = 1.0
    return _np.asarray(op(delta), float)


def is_bind_operator(op, dim, kernel=None, probes=3, seed=0, atol=1e-8):
    """Regime detector for the closed-form iterate: 1.0 if `op` acts as bind(kernel, .) on seeded random probes
    (a circular convolution -- diagonal in Fourier, so the closed form is exact), else 0.0. Deterministic."""
    import numpy as _np
    from holographic.agents_and_reasoning.holographic_ai import bind
    if kernel is None:
        kernel = bind_kernel_of(op, dim)
    rng = _np.random.default_rng(seed)
    for _ in range(int(probes)):
        x = rng.standard_normal(int(dim))
        if not _np.allclose(op(x), bind(kernel, x), atol=atol):
            return 0.0
    return 1.0


def iterate_gated(op, state, k, min_k=8, probes=3, seed=0):
    """Apply `op` to `state` k times, RE-ENABLING the closed-form jump behind its regime detector. If op is a
    circular convolution (a bind) AND k is large enough for the detector to pay for itself (k >= min_k), evaluate
    step_k in ONE FFT -- exact, ~k-fold fewer transforms. Otherwise step k times. Returns (result, info) where info
    records the score / whether the closed form was used / the kernel dim, so the re-enable stays measurable.
    The closed form is EXACT in regime, so this never does worse than stepping."""
    import numpy as _np
    state = _np.asarray(state, float)
    dim = state.shape[0]

    def _step_loop(_op, _state):
        s = _np.asarray(_state, float)
        for _ in range(int(k)):
            s = _np.asarray(_op(s), float)
        return s

    # below min_k, stepping is cheaper than detecting -- don't bother probing.
    if k < int(min_k):
        return _step_loop(op, state), {"gate": "closed_form_iterate", "score": None, "used": "fallback",
                                       "reason": "k<min_k", "k": int(k)}

    kernel = bind_kernel_of(op, dim)
    score = is_bind_operator(op, dim, kernel=kernel, probes=probes, seed=seed)
    if score >= 1.0:
        return step_k(state, kernel, int(k)), {"gate": "closed_form_iterate", "score": score, "used": "superior",
                                               "reason": "linear bind operator", "k": int(k), "dim": dim}
    return _step_loop(op, state), {"gate": "closed_form_iterate", "score": score, "used": "fallback",
                                   "reason": "nonlinear/non-convolution operator", "k": int(k)}

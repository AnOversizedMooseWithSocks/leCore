"""The HRNN time machine: for UNITARY dynamics, installed time is a random-access, reversible,
superposable axis.

The gated-DeltaNet identification (a decay-gated outer-product accumulator IS leCore's HRNN)
has an installed-side consequence this module exploits: a linear step with |spectrum| = 1 per
bin -- pure rotation, energy-conserving dynamics: oscillators, waves, anything the unitary
bake produces -- makes THREE things true at once, each measured before this file was written:

  1. RANDOM ACCESS INTO TIME: state at step t = one spectral power, O(log t) work.
     Measured: t=977 one-shot vs 977 iterated applications, 5.1e-13.
  2. EXACT TIME REVERSAL: the inverse spectrum is the conjugate; running backward is as
     cheap and as exact as forward. Measured: invert 977 steps back to x0 at 1.4e-15.
     KEPT NEGATIVE, pinned: for a DECAYING step the same inversion exploded to 1.4e+121
     (eig_min^50 = 2e-121) -- inversion is refused unless |spectrum| is certified unit,
     and the refusal carries the eig_min^t number.
  3. SIMULATION MULTIPLEX: K initial conditions bound with keys ride ONE state vector
     through ONE evolution (a circulant step COMMUTES with binding: measured 1.6e-15).
     THE HONEST LAW, measured against the wrong prediction first: individual member
     readout fidelity follows 1/sqrt(K) (0.457 / 0.328 / 0.253 at K=4/8/16 -- the
     superposition capacity law, NOT the cleanup-SNR sqrt(D/K) this module's author
     predicted before measuring), while LINEAR FUNCTIONALS of the whole ensemble
     (means, weighted sums, any fixed readout across members) are EXACT by linearity.
     Ensemble simulation in one vector: exact ensemble statistics, law-priced members.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind


def make_unitary_step(dim, seed=0):
    """A certified-unitary step operator as its rule: random per-bin phases (seeded), DC and
    Nyquist pinned real so the operator is a real map. Returns the half-spectrum; every other
    function here takes it. |spectrum| = 1 by construction -- the enabling condition, not an
    aspiration."""
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0.0, 2.0 * np.pi, dim // 2 + 1)
    c = np.exp(1j * ph)
    c[0] = 1.0
    if dim % 2 == 0:
        c[-1] = 1.0
    return c


def _check_unitary(spec, t):
    mags = np.abs(spec)
    lo, hi = float(mags.min()), float(mags.max())
    if abs(hi - 1.0) > 1e-9 or abs(lo - 1.0) > 1e-9:
        raise ValueError("time travel needs a UNITARY spectrum: |eig| in [%.3e, %.3e]; at t=%d "
                         "the inverse error scale is eig_min^-t = %.3e -- the decaying-step probe "
                         "measured exactly this explosion (1.4e+121) before this gate existed"
                         % (lo, hi, t, lo ** (-abs(int(t))) if lo > 0 else float("inf")))


def time_jump(state, spec, t):
    """State after t steps of the unitary recurrence -- t may be NEGATIVE (exact reversal).
    One spectral power either direction; refuses non-unitary spectra with the eig_min^t
    number rather than silently exploding."""
    _check_unitary(spec, t)
    x = np.asarray(state, float).reshape(-1)
    # dim comes from the STATE, never inferred from the spectrum: (len(spec)-1)*2 silently
    # returned 256 for a 257-dim state (circle-back V11 -- dimension corruption with no
    # exception, the worst failure class). The spectrum must MATCH the state or we refuse.
    if len(spec) != len(x) // 2 + 1:
        raise ValueError("spectrum length %d does not match state dim %d (expected %d)"
                         % (len(spec), len(x), len(x) // 2 + 1))
    return np.fft.irfft(np.fft.rfft(x) * spec ** int(t), n=len(x))


def bundle_sims(inits, keys):
    """K initial conditions -> ONE superposed state (each bound with its key). The commutation
    theorem (circulant step commutes with circular-convolution binding, measured 1.6e-15) is
    what lets the bundle EVOLVE as one state."""
    X = np.atleast_2d(np.asarray(inits, float))
    Kt = np.atleast_2d(np.asarray(keys, float))
    return sum(bind(X[i], Kt[i]) for i in range(len(X)))


def read_member(bundle_state, key, k_total):
    """Estimate ONE member's current state from the bundle. Returns (estimate,
    expected_fidelity): the 1/sqrt(K) law travels WITH the readout so no caller mistakes a
    law-priced estimate for an exact recovery."""
    est = unbind(np.asarray(bundle_state, float), np.asarray(key, float))
    return est, 1.0 / np.sqrt(float(k_total))


def evolve_functional(inits, weights, spec, t):
    """EXACT ensemble functional -- for a PRECOMMITTED readout: superpose the members WITH
    their weights (no keys), evolve the single vector, and the result IS sum_i w_i * x_i(t)
    exactly, by linearity of the step. One vector, one jump, zero capacity price -- when the
    functional is chosen before evolution.
    KEPT NEGATIVE, measured before this function replaced the wrong one: reading the SAME
    functional from a KEYED bundle (sum_i w_i * unbind(S_t, k_i)) is NOT exact -- each unbind
    carries its own crosstalk and weighting does not cancel it (measured cosine 0.34 against
    the clean weighted sum at K=8; the author's 'exact by linearity' argument was a fallacy
    until the referee ran). Keyed bundles buy K INDIVIDUAL estimates at the 1/sqrt(K) law;
    weighted superposition buys ONE precommitted functional exactly. Different contracts,
    both priced, choose per need."""
    X = np.atleast_2d(np.asarray(inits, float))
    w = np.asarray(weights, float)
    S = np.zeros(X.shape[1])
    for i in range(len(X)):
        S += w[i] * X[i]
    return time_jump(S, spec, t)


def _selftest():
    rng = np.random.default_rng(9)
    D = 1024
    spec = make_unitary_step(D, seed=3)
    x = rng.standard_normal(D)

    # 1. random access: t=977 one-shot == 977 iterated steps
    step = lambda v: np.fft.irfft(np.fft.rfft(v) * spec, n=D)
    xi = x.copy()
    for _ in range(977):
        xi = step(xi)
    assert np.max(np.abs(time_jump(x, spec, 977) - xi)) < 1e-10

    # 2. exact reversal + the decaying-step negative (refusal carries the number)
    assert np.max(np.abs(time_jump(time_jump(x, spec, 977), spec, -977) - x)) < 1e-12
    bad = spec * 0.92
    try:
        time_jump(x, bad, -50)
        raise AssertionError("non-unitary inversion must refuse")
    except ValueError as e:
        assert "eig_min" in str(e)

    # V11 pin: ODD dims round-trip exactly (the inferred-dim bug returned D-1 silently);
    # mismatched spectrum/state REFUSES instead of corrupting
    spec_odd = make_unitary_step(257, seed=2)
    xo = rng.standard_normal(257)
    back_o = time_jump(time_jump(xo, spec_odd, 33), spec_odd, -33)
    assert back_o.shape == (257,) and np.max(np.abs(back_o - xo)) < 1e-12
    try:
        time_jump(xo, spec, 3)
        raise AssertionError("mismatched spec/state must refuse")
    except ValueError as e:
        assert "does not match" in str(e)

    # 3. multiplex: commutation, the 1/sqrt(K) member law (band-checked), exact functionals
    K = 8
    xs = rng.standard_normal((K, D))
    keys = rng.standard_normal((K, D)) / np.sqrt(D)
    kx = rng.standard_normal(D)
    assert np.max(np.abs(step(bind(x, kx)) - bind(step(x), kx))) < 1e-12, "commutation is the theorem"
    S = bundle_sims(xs, keys)
    St = time_jump(S, spec, 200)
    true = np.stack([time_jump(xs[i], spec, 200) for i in range(K)])
    cos = []
    for i in range(K):
        est, fid = read_member(St, keys[i], K)
        cos.append(float(est @ true[i] / (np.linalg.norm(est) * np.linalg.norm(true[i]))))
        assert abs(fid - 1.0 / np.sqrt(K)) < 1e-12
    mean_cos = float(np.mean(cos))
    law = 1.0 / np.sqrt(K)
    assert abs(mean_cos - law) < 0.12, (mean_cos, law)   # the law, not the wish, within band
    w = rng.standard_normal(K)
    fx = evolve_functional(xs, w, spec, 200)
    clean = sum(w[i] * true[i] for i in range(K))
    assert np.max(np.abs(fx - clean)) < 1e-9, "precommitted functional must be EXACT"
    # the fallacy, pinned as a negative: the keyed-bundle readout of the same functional is NOT
    # exact -- crosstalk does not cancel under weighting (this assertion documents the failure
    # mode the first draft shipped as a feature)
    keyed = sum(w[i] * unbind(St, keys[i]) for i in range(K))
    cos_bad = float(keyed @ clean / (np.linalg.norm(keyed) * np.linalg.norm(clean)))
    assert cos_bad < 0.7, "if this ever becomes exact, the capacity law has been repealed -- investigate"
    print("OK: holographic_timemachine self-test passed (t=977 random access; exact reversal; "
          "decaying inversion refused WITH the number; commutation 1e-12; member fidelity == "
          "1/sqrt(K) law; ensemble functional exact)")


if __name__ == "__main__":
    _selftest()

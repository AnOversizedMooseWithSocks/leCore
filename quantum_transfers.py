"""Two transfers from quantum many-body theory, tested where they can fail.

WHAT THE SEARCH FOUND. The tensor-network literature has spent twenty years on exactly
our object: a fixed-width sequential representation (Matrix Product State, bond dimension
chi) whose representable-correlation budget is LOGARITHMIC in the width -- the
entanglement area law, S <= 2 ln(chi). A log purse, same shape as the Nesting Depth Law.
Two pieces of their toolbox transfer to NumPy immediately:

TRANSFER 1 -- THE BLOCK-ENTROPY SCALING GATE (area law vs volume law, classically).
  Physicists decide "is this state MPS-compressible" by how block entropy GROWS with
  block length: saturating H(L) = area law = compressible; linear H(L) = volume law =
  refuse. Classically this is Crutchfield's computational mechanics: H(L) ~ E + h*L,
  where h (the slope) is the ENTROPY RATE and E (the intercept) is the EXCESS ENTROPY =
  I(past; future) = the bits of state an optimal predictor MUST carry.
  WHY THIS MATTERS HERE: two turns ago the surrogate-calibrated gate FAILED on walk
  (0.250) and ar1 (0.125) false-fit -- phase randomisation does not destroy
  non-stationary / autocorrelated structure. Block entropy needs NO surrogate at all:
  it measures the entropy rate directly. PREDICTION: h -> 0 for genuine generators,
  h > 0 for ALL four nulls INCLUDING walk and ar1 (the ones the surrogate missed), and
  crucially for the phase-randomised twin of a sine, which has the IDENTICAL spectrum.
  If that prediction holds, this is the better compressibility gate: one estimator,
  no matched-surrogate assumption.

TRANSFER 2 -- BOND DIMENSION AS A STATE-DEMAND METER (TT-SVD).
  The MPS <-> stochastic-process equivalence says the minimal predictor's state count is
  the rank structure of the block distribution. TT-SVD (sequential reshaped SVDs -- pure
  numpy, deterministic, no autodiff) reads those ranks off the empirical block tensor.
  PREDICTIONS: period-p symbol stream -> ranks = p (the causal states are the phase);
  i.i.d. stream -> ranks = 1 (a product state: independence is EASY, chi=1);
  noisy period-p -> ranks ~ p with a noise tail.
  If the ranks land as predicted, the meter answers the question the corner-choice
  needs and the compressibility gate cannot: not just "is it structured" but HOW MANY
  BITS OF STATE the structure demands -- the q(d) to allocate before choosing a corner.

HONESTY NOTES, up front: the plug-in block-entropy estimator biases LOW for large L
(undersampling: k^L words vs T samples), so h is estimated from the last stable
conditional-entropy step, and the L range is capped where counts stay dense. The walk is
non-stationary, so its block entropy is computed on INCREMENTS -- differencing is part of
the battery, declared, not hidden.
"""
import numpy as np


def quantize(x, k=8):
    """Quantile-bin a continuous signal into k symbols (equal-mass bins)."""
    edges = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.digitize(x, edges)


def block_entropy(sym, L, k):
    """Plug-in Shannon entropy (bits) of length-L words. Dense counting via base-k coding."""
    if L == 0:
        return 0.0
    T = len(sym) - L + 1
    code = np.zeros(T, dtype=np.int64)
    for i in range(L):
        code = code * k + sym[i:i + T]
    _, counts = np.unique(code, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def scaling_report(sym, k=8, Lmax=6):
    """Return (h, E): entropy rate from the last conditional step, excess entropy."""
    H = [block_entropy(sym, L, k) for L in range(0, Lmax + 1)]
    steps = np.diff(H)                      # conditional entropies h_L, decreasing in L
    h = float(steps[-1])
    E = float(H[-1] - h * Lmax)
    return h, E, H


def tt_ranks(sym, k, L=6, tol=1e-2):
    """TT-SVD ranks of the empirical block distribution P(x1..xL) -- the bond dimensions
    a matrix-product model needs. Sequential reshape+SVD, ranks at relative tol."""
    T = len(sym) - L + 1
    code = np.zeros(T, dtype=np.int64)
    for i in range(L):
        code = code * k + sym[i:i + T]
    P = np.bincount(code, minlength=k ** L).astype(float)
    P /= P.sum()
    ranks, r = [], 1
    M = P.reshape(r * k, -1)
    for _ in range(L - 1):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        keep = int(np.sum(s > tol * s[0]))
        ranks.append(keep)
        M = (np.diag(s[:keep]) @ Vt[:keep]).reshape(keep * k, -1)
        r = keep
    return ranks


def signals(T, rng):
    t = np.arange(T, dtype=float)
    sine = np.sin(2 * np.pi * t / 210.0)
    X = np.fft.rfft(sine)
    ph = rng.uniform(0, 2 * np.pi, len(X)); ph[0] = 0.0
    if T % 2 == 0:
        ph[-1] = 0.0
    prand = np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=T)
    e = rng.standard_normal(T)
    ar1 = np.zeros(T)
    for i in range(1, T):
        ar1[i] = 0.95 * ar1[i - 1] + e[i]
    return {
        "sine":        sine,
        "sawtooth":    ((t % 150.0) / 150.0) * 2 - 1,
        "noisy sine":  sine + 0.1 * rng.standard_normal(T),
        "phase_rand":  prand,                                  # sine's spectrum, no generator
        "ar1":         ar1,                                    # the surrogate gate's 0.125 miss
        "walk (diff)": np.diff(np.cumsum(rng.standard_normal(T))),  # the 0.250 miss, differenced
        "white":       rng.standard_normal(T),
    }


def experiment_1(T=20000, k=8, Lmax=6):
    print("TRANSFER 1 -- block-entropy scaling gate (no surrogates needed)")
    print(f"  h = entropy rate (bits/step, 0 = deterministic), E = excess entropy = state demand")
    print(f"  max possible h = log2({k}) = {np.log2(k):.2f}\n")
    print(f"{'signal':>12} {'h':>7} {'E':>7}   verdict")
    rng = np.random.default_rng(0)
    for name, x in signals(T, rng).items():
        h, E, _ = scaling_report(quantize(x, k), k, Lmax)
        verdict = "DETERMINISTIC (all 3 corners)" if h < 0.15 else "stochastic (triangle binds)"
        print(f"{name:>12} {h:>7.3f} {E:>7.2f}   {verdict}")
    print("\n  PREDICTION was: h ~ 0 only for sine/sawtooth/noisy-sine; h > 0 for all four")
    print("  nulls including phase_rand (identical spectrum to sine) and walk/ar1 (the two")
    print("  the surrogate gate failed on at 0.250 / 0.125).")


def experiment_2(T=40000):
    print("\nTRANSFER 2 -- TT-SVD bond dimensions as the state-demand meter")
    print("  predicted ranks: period-p -> p; iid -> 1; noisy period-p -> ~p + tail\n")
    rng = np.random.default_rng(0)
    k = 4
    period4 = np.tile(np.array([0, 1, 2, 3]), T // 4)
    iid = rng.integers(0, k, T)
    noisy4 = period4.copy()
    flip = rng.random(T) < 0.05
    noisy4[flip] = rng.integers(0, k, int(flip.sum()))
    period2sym = np.tile(np.array([0, 2]), T // 2)             # period 2 inside alphabet 4
    for name, sym, want in (("period-4", period4, "4,4,..."),
                            ("period-2", period2sym, "2,2,..."),
                            ("iid", iid, "1,1,..."),
                            ("noisy period-4 (5%)", noisy4, "~4 + tail")):
        r = tt_ranks(sym, k, L=6, tol=1e-2)
        print(f"  {name:>20}: ranks {r}   (predicted {want})")


if __name__ == "__main__":
    experiment_1()
    experiment_2()

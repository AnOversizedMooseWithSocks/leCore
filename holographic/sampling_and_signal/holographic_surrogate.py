"""holographic_surrogate.py -- the phase-randomized null for CONTINUOUS signals (the honest baseline the panel kept
asking for).

WHY THIS MODULE EXISTS
----------------------
The shuffle-null (a permutation) is the right null for DISCRETE corpora: it destroys order while preserving the
symbol histogram, so structure counts only above chance co-occurrence. But for a CONTINUOUS, autocorrelated
signal a permutation is TOO STRONG a null -- it destroys the autocorrelation that even a trivial forecaster
(persistence, a smoother) exploits, so any such method looks brilliant against it. Both the SETI seat (forecast
abstain gate) and the adaptive pipeline flagged this: the honest continuous baseline must preserve the
autocorrelation and ask whether there is structure BEYOND it.

THE METHOD (Theiler et al. 1992, surrogate data)
  A phase-randomized surrogate has the SAME power spectrum (hence the same autocorrelation, by Wiener-Khinchin) as
  the original, but random phases -- so any DETERMINISTIC / nonlinear structure is destroyed while the linear
  second-order statistics are preserved exactly. Recipe: FFT the signal, keep the magnitudes, replace the phases
  with random ones (kept antisymmetric so the inverse transform is real), inverse-FFT. A statistic that is large
  on the real signal but not on an ensemble of its surrogates reflects structure the power spectrum ALONE does not
  explain -- the honest thing to credit.

WHAT IT IS FOR
  `surrogate_zscore(x, statistic)` measures any structure `statistic` on `x` against an ensemble of phase-
  randomized surrogates and reports a z-score -- the continuous-signal analogue of the shuffle-null z the discrete
  path already uses. Wire this behind a forecast/structure gate for continuous data.

KEPT NEGATIVE: a phase-randomized surrogate assumes the signal's non-Gaussianity is not itself the structure of
interest -- for strongly non-Gaussian amplitudes the amplitude-adjusted variant (AAFT) is stricter. We ship the
basic phase-randomization (the common case) and NAME AAFT as the follow-on, rather than silently over-claiming.

NumPy only. Deterministic given the seed.
"""

import numpy as np


def phase_randomize(x, seed=0):
    """Return a PHASE-RANDOMIZED surrogate of a 1-D real signal `x`: same power spectrum (same autocorrelation) as
    `x`, but random phases -- so any deterministic/nonlinear structure is destroyed while the linear second-order
    statistics are preserved EXACTLY (Theiler et al. 1992). The phases are kept antisymmetric so the inverse
    transform is real, and the DC / Nyquist bins are left real. Deterministic given `seed`. This is the honest
    null for a continuous, autocorrelated signal -- unlike a permutation, it does NOT destroy the autocorrelation
    a trivial forecaster would exploit."""
    x = np.asarray(x, float).ravel()
    n = len(x)
    F = np.fft.rfft(x)
    mag = np.abs(F)
    rng = np.random.default_rng(seed)
    # random phases for the interior bins; DC (0) and, for even n, Nyquist stay at their original (real) phase.
    phases = rng.uniform(0, 2 * np.pi, size=len(F))
    phases[0] = np.angle(F[0])                                 # DC must stay real
    if n % 2 == 0:
        phases[-1] = np.angle(F[-1])                           # Nyquist bin must stay real for even-length signals
    surrogate = np.fft.irfft(mag * np.exp(1j * phases), n=n)
    return surrogate


def surrogate_zscore(x, statistic, n_surrogates=64, seed=0):
    """Measure a structure `statistic(x)` against an ensemble of PHASE-RANDOMIZED surrogates and report how far
    the real value exceeds the surrogate null, in null standard deviations (a z-score). `statistic` is any
    callable signal -> float that is LARGE when there is structure the power spectrum alone does not explain (e.g.
    a forecaster's skill, a nonlinearity measure). Because the surrogates share the real signal's autocorrelation,
    a high z means structure BEYOND linear autocorrelation -- the honest continuous-signal null. Returns a dict:
    `value` (statistic on x), `null_mean`, `null_std`, `z`. Deterministic given `seed`."""
    x = np.asarray(x, float).ravel()
    value = float(statistic(x))
    null = np.empty(n_surrogates)
    for i in range(n_surrogates):
        null[i] = float(statistic(phase_randomize(x, seed=seed + i + 1)))
    null_mean = float(null.mean())
    null_std = float(null.std()) + 1e-12
    return {"value": value, "null_mean": null_mean, "null_std": null_std,
            "z": (value - null_mean) / null_std}


def amplitude_adjusted_surrogate(x, seed=0):
    """AAFT (amplitude-adjusted Fourier transform) surrogate -- the STRICTER null for NON-GAUSSIAN signals
    (Theiler et al. 1992). Basic phase_randomize preserves the power spectrum but GAUSSIANIZES the marginal (it is
    a sum of sinusoids, so the central limit theorem pulls its histogram toward a bell curve) -- which destroys the
    fat tails of, e.g., price returns and makes any tail-sensitive statistic falsely flag the surrogate as
    'different'. AAFT preserves BOTH the exact amplitude DISTRIBUTION and (approximately) the power spectrum.

    Recipe: (1) build a Gaussian signal with the SAME rank-order as x; (2) phase-randomize it (Gaussian marginal,
    so phase-randomization does not distort it); (3) map x's sorted amplitudes onto that surrogate's rank-order --
    so the result is a PERMUTATION of x's own values (exact same histogram) arranged with randomized phase.
    Deterministic given `seed`.

    KEPT NEGATIVE: AAFT's spectrum match is APPROXIMATE (the rank-remapping perturbs it slightly), unlike basic
    phase_randomize which is exact -- a known bias for strongly-coloured non-Gaussian signals (the iterated variant
    IAAFT tightens it, named as the follow-on). Use basic phase_randomize when the signal is ~Gaussian and the
    spectrum must match exactly; use AAFT when the amplitude distribution matters (fat tails)."""
    x = np.asarray(x, float).ravel()
    n = len(x)
    rng = np.random.default_rng(seed)
    # (1) a Gaussian signal with the same rank-order as x: place sorted Gaussian draws at x's ranks.
    ranks = np.argsort(np.argsort(x))                          # rank of each sample of x (0..n-1)
    gauss = np.sort(rng.normal(size=n))
    g = gauss[ranks]                                           # Gaussian marginal, x's ordering
    # (2) phase-randomize the Gaussian version (phase-randomization is faithful for a Gaussian marginal).
    g_surr = phase_randomize(g, seed=seed + 104729)           # a distinct sub-seed for the phase draw
    # (3) give the surrogate x's EXACT amplitude distribution by mapping x's sorted values onto g_surr's ranks.
    x_sorted = np.sort(x)
    surr_ranks = np.argsort(np.argsort(g_surr))
    return x_sorted[surr_ranks]


def iaaft_surrogate(x, n_iter=100, tol=1e-8, seed=0):
    """IAAFT (iterated amplitude-adjusted Fourier transform) surrogate -- the gold-standard null that matches BOTH
    the exact amplitude distribution AND (to convergence) the exact power spectrum (Schreiber & Schmitz 1996).
    AAFT only APPROXIMATES the spectrum; IAAFT fixes that by ITERATING two projections until they agree -- the
    same 'iterate a projection' move guide_structure/IK/PBD/the resonator all share, here alternating between the
    spectrum constraint and the amplitude-distribution constraint:

      (a) SPECTRUM step: FFT the current surrogate, replace its magnitudes with the TARGET (original) magnitudes,
          keep its phases, inverse-FFT -> exact spectrum, but amplitudes drift.
      (b) AMPLITUDE step: rank-order the result and map the original's SORTED amplitudes onto those ranks ->
          exact amplitude distribution, but the spectrum drifts slightly.

    Iterating (a),(b) converges: each step is a projection onto a constraint set, and the fixed point satisfies
    both. Stops when the ranks stop changing (a true fixed point) or `n_iter` runs out. Returns the surrogate.
    Deterministic given `seed`.

    WHY prefer this over AAFT: for strongly-coloured non-Gaussian signals (e.g. fat-tailed price returns with real
    autocorrelation) AAFT's approximate spectrum biases the null; IAAFT removes that bias. It costs the iterations.
    KEPT NEGATIVE: IAAFT can only satisfy BOTH constraints exactly if they are compatible; when they conflict it
    settles at a compromise (the ranks oscillate) -- we return the last amplitude-exact state and report via the
    caller's own spectrum check, never pretending both are perfect."""
    x = np.asarray(x, float).ravel()
    n = len(x)
    target_mag = np.abs(np.fft.rfft(x))
    x_sorted = np.sort(x)
    rng = np.random.default_rng(seed)
    # start from a random permutation of x (exact amplitude distribution, random spectrum/phase).
    surr = x[rng.permutation(n)]
    prev_ranks = None
    for _ in range(n_iter):
        # (a) spectrum step: impose the target magnitudes, keep current phases.
        F = np.fft.rfft(surr)
        phases = np.angle(F)
        surr = np.fft.irfft(target_mag * np.exp(1j * phases), n=n)
        # (b) amplitude step: impose the exact amplitude distribution via rank mapping.
        ranks = np.argsort(np.argsort(surr))
        surr = x_sorted[ranks]
        # converged when the ordering stops changing (a fixed point of the two projections).
        if prev_ranks is not None and np.array_equal(ranks, prev_ranks):
            break
        prev_ranks = ranks
    return surr


def _selftest():
    """Contracts:

    1. A phase-randomized surrogate has (near-)identical power spectrum to the original -- the autocorrelation is
       PRESERVED (this is the whole point; a permutation would fail this).
    2. A permutation, by contrast, DESTROYS the autocorrelation -- demonstrating why the surrogate is the honest
       null for continuous data.
    3. surrogate_zscore: a DETERMINISTIC nonlinear signal (whose structure is not captured by its spectrum) scores
       high against the surrogate null; a linear-Gaussian process (structure fully in its spectrum) scores near 0.
    4. Determinism.
    """
    rng = np.random.default_rng(0)
    n = 1024
    # an autocorrelated signal: a smooth random walk (strong autocorrelation, structure IS its spectrum).
    walk = np.cumsum(rng.normal(size=n))
    walk -= walk.mean()

    # (1) surrogate preserves the power spectrum.
    surr = phase_randomize(walk, seed=1)
    ps_x = np.abs(np.fft.rfft(walk))
    ps_s = np.abs(np.fft.rfft(surr))
    assert np.allclose(ps_x, ps_s, atol=1e-6), "surrogate must preserve the power spectrum"

    # (2) a permutation destroys the autocorrelation (lag-1), the surrogate preserves it.
    def ac1(v):
        v = v - v.mean()
        return float(np.dot(v[:-1], v[1:]) / (np.dot(v, v) + 1e-12))
    perm = walk[rng.permutation(n)]
    assert ac1(walk) > 0.9                                     # the walk is strongly autocorrelated
    assert abs(ac1(perm)) < 0.2                                # permutation kills it
    assert ac1(surr) > 0.7                                     # surrogate keeps most of it

    # (3) a statistic sensitive to DETERMINISTIC structure beyond the spectrum: nonlinear predictability. A
    #     deterministic map (e.g. the logistic/tent map) is highly predictable from its own past even though its
    #     BROADBAND spectrum looks like noise -- so a nearest-neighbour predictor's skill is high on the real
    #     signal but collapses on phase-randomized surrogates (which share the spectrum but destroy the determinism).
    def predictability(v, k=3):
        """One-step nearest-neighbour prediction skill: for each point, find the past delay-vector most similar to
        the current one and predict the next value from it; return correlation of predictions to truth. High for a
        deterministic map, ~0 for linear-stochastic (whose future is not determined by its past pattern)."""
        v = np.asarray(v, float)
        m = len(v)
        emb = np.array([v[i:i + k] for i in range(m - k - 1)])
        nxt = v[k:m - 1]
        if len(emb) < 20:
            return 0.0
        preds, truth = [], []
        # split: build the library on the first half, predict the second (no peeking).
        half = len(emb) // 2
        lib, lib_next = emb[:half], nxt[:half]
        for i in range(half, len(emb)):
            d = np.sum((lib - emb[i]) ** 2, axis=1)
            j = int(np.argmin(d))
            preds.append(lib_next[j]); truth.append(nxt[i])
        preds, truth = np.array(preds), np.array(truth)
        if preds.std() < 1e-9 or truth.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(preds, truth)[0, 1])

    # the logistic map at r=3.9: deterministic chaos -- broadband spectrum but perfectly determined by its past.
    logistic = np.zeros(n)
    logistic[0] = 0.4
    for i in range(1, n):
        logistic[i] = 3.9 * logistic[i - 1] * (1 - logistic[i - 1])
    logistic -= logistic.mean()
    znl = surrogate_zscore(logistic, predictability, n_surrogates=40, seed=2)
    # a linear-Gaussian AR(1) process: predictable only as far as its autocorrelation, which the surrogate KEEPS,
    # so its predictability does NOT stand out above the surrogate null.
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + rng.normal()
    zlin = surrogate_zscore(ar, predictability, n_surrogates=40, seed=3)
    assert znl["z"] > 3.0, znl                                 # deterministic chaos beats its phase-random null
    assert znl["z"] > zlin["z"], (znl["z"], zlin["z"])         # and beats it by more than a linear process does

    # (4) determinism.
    assert np.array_equal(phase_randomize(walk, seed=5), phase_randomize(walk, seed=5))

    # (5) AAFT preserves the EXACT amplitude distribution (fat tails) where basic phase_randomize Gaussianizes it.
    #     Build a fat-tailed signal (Student-t-like: cubed Gaussian) and compare marginals.
    heavy = rng.standard_normal(n) ** 3                       # a fat-tailed signal
    aaft = amplitude_adjusted_surrogate(heavy, seed=6)
    basic = phase_randomize(heavy, seed=6)
    # AAFT's values are a PERMUTATION of the original -> identical sorted amplitudes (exact histogram match).
    assert np.allclose(np.sort(aaft), np.sort(heavy)), "AAFT must preserve the exact amplitude distribution"
    # basic phase-randomization pulls the kurtosis toward Gaussian (3); AAFT keeps the original's high kurtosis.
    def kurt(v):
        v = (v - v.mean()) / (v.std() + 1e-12)
        return float(np.mean(v ** 4))
    k_orig, k_aaft, k_basic = kurt(heavy), kurt(aaft), kurt(basic)
    assert abs(k_aaft - k_orig) < abs(k_basic - k_orig), (k_orig, k_aaft, k_basic)   # AAFT closer to the truth
    # AAFT still approximately preserves the autocorrelation (the point of a surrogate, vs a plain shuffle).
    assert ac1(amplitude_adjusted_surrogate(walk, seed=7)) > 0.4

    # (6) IAAFT matches the spectrum BETTER than AAFT while keeping the EXACT amplitude distribution. Use a
    #     coloured fat-tailed signal (fat tails + autocorrelation), where AAFT's approximate spectrum shows.
    coloured_heavy = np.cumsum(rng.standard_normal(n) ** 3)   # fat-tailed increments + strong autocorrelation
    coloured_heavy -= coloured_heavy.mean()
    ia = iaaft_surrogate(coloured_heavy, n_iter=100, seed=8)
    aa = amplitude_adjusted_surrogate(coloured_heavy, seed=8)
    tgt_mag = np.abs(np.fft.rfft(coloured_heavy))
    err_ia = np.linalg.norm(np.abs(np.fft.rfft(ia)) - tgt_mag) / (np.linalg.norm(tgt_mag) + 1e-12)
    err_aa = np.linalg.norm(np.abs(np.fft.rfft(aa)) - tgt_mag) / (np.linalg.norm(tgt_mag) + 1e-12)
    assert err_ia < err_aa, (err_ia, err_aa)                  # IAAFT's spectrum is closer to the target
    assert np.allclose(np.sort(ia), np.sort(coloured_heavy))  # and it still keeps the EXACT amplitude distribution
    assert np.array_equal(iaaft_surrogate(coloured_heavy, seed=9), iaaft_surrogate(coloured_heavy, seed=9))  # deterministic

    print("holographic_surrogate selftest OK (surrogate preserves the power spectrum exactly; permutation kills "
          "the lag-1 autocorrelation (%.2f->%.2f) while the surrogate keeps it (%.2f); the logistic map's "
          "predictability scores z=%.1f against its phase-randomized null vs z=%.1f for a linear AR(1); AAFT keeps "
          "fat tails (kurtosis %.1f vs basic %.1f, truth %.1f); IAAFT matches the spectrum better than AAFT "
          "(rel err %.3f vs %.3f) with the exact distribution; deterministic)"
          % (ac1(walk), ac1(perm), ac1(surr), znl["z"], zlin["z"], k_aaft, k_basic, k_orig, err_ia, err_aa))


if __name__ == "__main__":
    _selftest()

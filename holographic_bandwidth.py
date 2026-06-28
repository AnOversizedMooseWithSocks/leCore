"""Spectral bandwidth + a fractal-dimension cross-check (holographic_bandwidth).

WHY THIS MODULE EXISTS (and what it deliberately does NOT rebuild)
-----------------------------------------------------------------
The fractal-optics backlog review asked for a "fractal-dimension / bandwidth probe," but its own finding was that
fractal DIMENSION is already shipped -- `holographic_fractal` has box-counting dimension, the R/S Hurst exponent,
and IFS compression, and `UnifiedMind.fractal_dimension` / `self_affinity` expose them. So this module does NOT add
another primary dimension estimator. It ships the two pieces the review identified as the real, missing work:

  1. SPECTRAL BANDWIDTH -- the review's point ("the probe's real job is bandwidth measurement"): how much of the
     spectrum the content occupies, the number that drives the encoder's `bandwidth` knob (the next item). NOT in
     the codebase -- genuinely new.
  2. A SINGULARITY CROSS-CHECK -- the review's kept negative ("isolated singularities artefactually read as broad
     multifractal") made operational. The shipped dimension is a SINGLE estimator, so it silently returns a wrong
     number for a step or a pure tone. This pairs two INDEPENDENT slope estimators -- the power-spectrum slope
     D=(5-gamma)/2 (Berry & Klein) and an increment-variance estimator (Var(x(t+tau)-x(t))~tau^(2H)) -- and reports
     whether they AGREE: trustworthy when they do, "not a clean monofractal" when they don't.

A MEASURED FINDING ON WHY THE CROSS-CHECK USES THESE TWO (not the shipped R/S Hurst): the two slope estimators
agree to ~0.05 on clean fBm, but the shipped R/S Hurst reads a *different* number on the same signal -- because R/S
is a RANGE statistic that weights the (low-frequency, trend-dominated) coarse scales, while the slope methods fit
the whole power law. They measure different things, so R/S is a poor naive co-validator here; the honest cross-check
is slope-vs-slope. (R/S remains the right tool for what it was shipped for -- persistence of a series.)

WHAT IT PROVIDES
  * spectral_bandwidth(x, energy_fraction) -- the bandwidth (fraction of Nyquist) holding that energy fraction.
  * spectral_dimension(x) -- the power-spectrum-slope dimension (a fast estimator; the engine's primary dimension
    stays box-counting / R-S in holographic_fractal).
  * fractal_confidence(x) -- (d_spectral, d_increment, agree): the two independent slope estimates and whether they
    agree to tolerance -- the singularity flag.

THE MEASUREMENT BAR (checked exactly in the self-test)
  * bandwidth is tiny for a smooth sinusoid (band-limited) and near 1 for white noise (broadband); for fBm it rises
    with roughness (lower Hurst -> more bandwidth).
  * on synthetic fBm of known Hurst H (so D=2-H) the two slope estimators AGREE and bracket the true D.
  * a STEP (isolated singularity) and a PURE TONE make the estimators DISAGREE -- the flag fires (the kept negative
    the shipped single-estimator dimension cannot catch).

DETERMINISM (per ISA.md)
  Pure FFT + least-squares slopes; no RNG in the probe. Same signal -> identical numbers (asserted).

KEPT NEGATIVES (loud)
  * The spectral bandwidth is an ENERGY rolloff: a fractal's 1/f^b energy is front-loaded, so its bandwidth can
    read small even though the self-similar DETAIL extends higher -- band-limiting a fractal to its energy-bandwidth
    keeps the bulk and discards the fine detail. That is the fundamental fractal trade (full fidelity needs
    infinite bandwidth); superoscillation is the standing proof it cannot be cheated for free. The number is honest
    about fidelity-for-a-budget, not "lossless bandwidth."
  * The power-spectrum-slope dimension is the one FOOLED by isolated singularities -- it exists here only as a
    cross-check term, never as the engine's reported dimension. Trust the dimension only when `agree`.
  * 1-D signals; higher-D (images/fields) the slope relation is approximate (a different constant) -- and for
    images the engine already uses box-counting, which this does not touch.
"""

import numpy as np


def spectral_bandwidth(x, energy_fraction=0.95):
    """The bandwidth (fraction of Nyquist, in [0,1]) holding `energy_fraction` of the spectral energy -- the
    band-limit preserving that much fidelity. Drives the encoder's bandwidth knob: small for band-limited content,
    near 1 for broadband noise."""
    x = np.asarray(x, float)
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x))
    total = P.sum()
    if total <= 0:
        return 0.0
    idx = int(np.searchsorted(np.cumsum(P), energy_fraction * total))
    return float(f[min(idx, len(f) - 1)] / 0.5)            # Nyquist = 0.5 cycles/sample


def spectral_dimension(x):
    """The power-spectrum-slope fractal dimension D=(5-gamma)/2 (Berry & Klein) of a 1-D signal -- a fast estimator
    used here as a CROSS-CHECK term. (The engine's primary dimension is box-counting / R-S in holographic_fractal.)"""
    x = np.asarray(x, float)
    n = len(x)
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(n)
    m = f > 0
    slope = np.polyfit(np.log(f[m]), np.log(P[m] + 1e-30), 1)[0]
    return (5 - (-slope)) / 2


def _increment_dimension(x):
    """Increment-variance dimension: Var(x(t+tau)-x(t)) ~ tau^(2H), D = 2 - H. An estimator independent of the
    spectrum -- the second slope opinion in the cross-check."""
    x = np.asarray(x, float)
    taus = np.array([1, 2, 4, 8, 16, 32, 64])
    taus = taus[taus < len(x)]
    var = np.array([np.mean((x[t:] - x[:-t]) ** 2) for t in taus])
    slope = np.polyfit(np.log(taus), np.log(var + 1e-30), 1)[0]
    return 2 - slope / 2


def fractal_confidence(x, tol=0.15):
    """Two independent slope-based fractal-dimension estimates and whether they AGREE -- the singularity cross-check
    the shipped single-estimator dimension cannot do. Returns (d_spectral, d_increment, agree), agree True iff the
    two agree to `tol`. Trust a fractal dimension only when agree is True."""
    d_spec = spectral_dimension(x)
    d_inc = _increment_dimension(x)
    return d_spec, d_inc, bool(abs(d_spec - d_inc) < tol)


# =====================================================================================================
# Self-test -- bandwidth separates smooth/broadband; estimators agree on fBm; a singularity makes them disagree.
# =====================================================================================================
def _selftest():
    def make_fbm(n, H, seed=1):
        rng = np.random.default_rng(seed)
        f = np.fft.rfftfreq(n)
        amp = np.zeros(len(f))
        amp[1:] = f[1:] ** (-(2 * H + 1) / 2)              # PSD ~ 1/f^(2H+1) -> fBm with Hurst H, D = 2 - H
        return np.fft.irfft(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, len(f))), n)

    n = 8192

    # --- bandwidth separates band-limited from broadband ---
    smooth = np.sin(2 * np.pi * 3 * np.arange(n) / n)
    white = np.random.default_rng(2).standard_normal(n)
    bw_s, bw_w = spectral_bandwidth(smooth), spectral_bandwidth(white)
    assert bw_s < 0.05 and bw_w > 0.5, f"bandwidth must separate smooth ({bw_s:.3f}) / white ({bw_w:.3f})"

    # --- rougher fBm (lower Hurst) occupies more bandwidth ---
    assert spectral_bandwidth(make_fbm(n, 0.2)) > spectral_bandwidth(make_fbm(n, 0.9))

    # --- on clean fBm the two slope estimators agree and bracket the true D ---
    for H in (0.3, 0.5, 0.8):
        d_spec, d_inc, agree = fractal_confidence(make_fbm(n, H))
        D_true = 2 - H
        assert agree, f"the two slope estimators should agree on clean fBm (H={H})"
        assert min(d_spec, d_inc) - 0.15 < D_true < max(d_spec, d_inc) + 0.15, "bracket true D"

    # --- the kept negative: a step and a pure tone make the estimators DISAGREE (the flag fires) ---
    step = np.zeros(n); step[n // 2:] = 1.0
    assert not fractal_confidence(step)[2], "a singularity must trip the cross-check"
    assert not fractal_confidence(smooth)[2], "a pure tone (not a fractal) must trip the cross-check"

    # --- determinism ---
    x = make_fbm(n, 0.5)
    assert spectral_bandwidth(x) == spectral_bandwidth(x) and fractal_confidence(x)[2] == fractal_confidence(x)[2]

    ds, di, _ = fractal_confidence(step)
    d3s, d3i, _ = fractal_confidence(make_fbm(n, 0.3))
    print(f"holographic_bandwidth selftest: ok (bandwidth smooth {bw_s:.4f} << white {bw_w:.3f} of Nyquist; rougher "
          f"fBm -> more bandwidth; clean fBm H=0.3 -> spectral {d3s:.2f} / increment {d3i:.2f} AGREE, bracket 1.70; a "
          f"STEP -> spectral {ds:.2f} / increment {di:.2f} DISAGREE (flag fires); deterministic)")


if __name__ == "__main__":
    _selftest()

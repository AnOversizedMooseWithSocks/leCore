"""holographic_lombscargle.py -- find the PERIOD of an unevenly-sampled signal (leCore sampling_and_signal).

WHY THIS EXISTS
---------------
Real observations are not evenly spaced: a star is measured whenever the telescope points at it, with gaps for
daylight, weather, and other targets. A plain FFT needs even samples and fails on that. The Lomb-Scargle
periodogram (Lomb 1976; Scargle 1982) is the standard fix -- a least-squares fit of a sinusoid at each trial
frequency, valid for arbitrary sample times. It closes the astro loop: a light curve or radial-velocity series
in, an orbital PERIOD out, which Kepler's third law turns into a semi-major axis for star_system to place.

This is the honest, classical algorithm in pure numpy:
  P(f) = 0.5 * [ (sum y cos w(t-tau))^2 / sum cos^2 w(t-tau) + (sum y sin w(t-tau))^2 / sum sin^2 w(t-tau) ] / var(y)
with w = 2*pi*f and the phase offset tau chosen (Scargle's trick) so the estimator is time-shift invariant and
exact for uneven sampling. Vectorised over the trial-frequency grid, O(N*M).

SIGNIFICANCE (kept honest, Cranmer's discipline): a strong peak is not automatically real -- an unlucky sampling
window can manufacture one. false_alarm_probability re-runs the search on PERMUTED values (times fixed), which
destroys any true period while preserving the sampling window and the value distribution, and reports what
fraction of those nulls beat the observed peak. (We permute rather than phase_randomize because phase_randomize
assumes EVEN sampling; permutation is the right null for an uneven window.)

DIRECTIONS (up/down/sideways)
  DOWN  -- runs on any sub-window of the series (just pass the slice).
  UP    -- lomb_scargle broadcasts the frequency axis; a stack of light curves periodograms in one pass.
  SIDEWAYS
    sequence -- the native costume (a sampled wave over uneven time). field -- the periodogram over the frequency
    grid is a 1-D field. structure -- best_period returns a {period, power, fap} record that feeds star_system.

Determinism: closed-form periodogram; the FAP null uses a seeded default_rng. Same inputs+seed -> identical.
"""

import numpy as np


def lomb_scargle(times, values, freqs):
    """The Lomb-Scargle normalised power at each trial frequency in `freqs` (cycles per unit time). `times` and
    `values` are the (uneven) samples; the mean is removed and the result normalised by the variance, so a pure
    sinusoid peaks near P=(N-1)/2 and noise sits near 1. Field-native over `freqs` -> power array of matching shape.
    Frequencies must be > 0 (a zero frequency has no phase)."""
    t = np.asarray(times, float)
    y = np.asarray(values, float)
    y = y - np.mean(y)
    f = np.asarray(freqs, float)
    if np.any(f <= 0):
        raise ValueError("frequencies must be positive (cycles per unit time)")
    w = 2.0 * np.pi * f                                  # angular frequency (M,)
    # Scargle's time offset tau, per frequency, so the estimator is shift-invariant: tan(2 w tau) = S/ C.
    w2t = 2.0 * np.outer(w, t)                           # (M, N)
    tau = 0.5 * np.arctan2(np.sum(np.sin(w2t), axis=1), np.sum(np.cos(w2t), axis=1)) / w   # (M,)
    arg = np.outer(w, t) - (w * tau)[:, None]            # w (t - tau)  (M, N)
    c = np.cos(arg); s = np.sin(arg)
    yc = np.sum(y * c, axis=1); ys = np.sum(y * s, axis=1)
    cc = np.sum(c * c, axis=1); ss = np.sum(s * s, axis=1)
    var = np.var(y)
    if var <= 0:
        return np.zeros_like(f)
    # guard degenerate cc/ss (can happen at frequencies aliasing the sampling) with a tiny floor
    power = 0.5 * (yc * yc / np.maximum(cc, 1e-300) + ys * ys / np.maximum(ss, 1e-300)) / var
    return power


def freq_grid(times, min_period=None, max_period=None, samples_per_peak=5.0):
    """A sensible trial-frequency grid derived from the sampling itself: from ~1/baseline (the longest period the
    data can constrain) up to a pseudo-Nyquist set by the median spacing, sampled `samples_per_peak` times across
    each intrinsic peak width. Pass min_period/max_period to override. No magic numbers the caller must guess."""
    t = np.sort(np.asarray(times, float))
    baseline = t[-1] - t[0]
    if baseline <= 0:
        raise ValueError("times must span a nonzero range")
    dt = np.median(np.diff(t))
    f_min = 1.0 / max_period if max_period else 1.0 / baseline
    f_max = 1.0 / min_period if min_period else 0.5 / max(dt, baseline / len(t))
    df = 1.0 / (samples_per_peak * baseline)             # peak width ~ 1/baseline; sample it finely
    n = max(int(np.ceil((f_max - f_min) / df)), 4)
    return np.linspace(f_min, f_max, n)


def lomb_scargle_auto(times, values, min_period=None, max_period=None, samples_per_peak=5.0):
    """Convenience: build the frequency grid from the data and return (freqs, power). The one call for 'give me the
    periodogram of this light curve'."""
    f = freq_grid(times, min_period=min_period, max_period=max_period, samples_per_peak=samples_per_peak)
    return f, lomb_scargle(times, values, f)


def best_period(times, values, min_period=None, max_period=None, samples_per_peak=5.0, n_null=0, seed=0):
    """The most likely PERIOD of the series: search the auto grid, take the strongest peak. Returns
    {period, frequency, power, fap} -- fap (false-alarm probability) is filled when n_null>0 (see
    false_alarm_probability). period = 1/frequency, in the same time units as `times`. Feeds star_system via
    Kepler's third law."""
    f, power = lomb_scargle_auto(times, values, min_period=min_period, max_period=max_period, samples_per_peak=samples_per_peak)
    k = int(np.argmax(power))
    out = {"period": float(1.0 / f[k]), "frequency": float(f[k]), "power": float(power[k]), "fap": None}
    if n_null > 0:
        out["fap"] = false_alarm_probability(times, values, out["power"], f, n_null=n_null, seed=seed)
    return out


def false_alarm_probability(times, values, observed_power, freqs, n_null=200, seed=0):
    """The chance a peak this strong arises from the sampling window alone: re-run the periodogram on `n_null`
    random PERMUTATIONS of the values (times fixed) and report the fraction whose peak matches or beats
    `observed_power`. Low fap => the period is real. Seeded, deterministic. This is the honesty gate on a detection
    (Cranmer/Tarter): a beautiful peak is a reason to run the null, not to celebrate."""
    t = np.asarray(times, float); y = np.asarray(values, float); f = np.asarray(freqs, float)
    rng = np.random.default_rng(int(seed))
    beat = 0
    for _ in range(int(n_null)):
        yp = rng.permutation(y)                          # break any real period, keep the window + value set
        if np.max(lomb_scargle(t, yp, f)) >= observed_power:
            beat += 1
    return beat / float(n_null)


def phase_fold(times, values, period, t0=0.0):
    """Fold the series on `period`: return (phase in [0,1), values) sorted by phase. On the TRUE period the folded
    curve is coherent (one clean cycle); on a wrong period it scatters. The standard way to SEE that a recovered
    period is right, and what a downstream plot or a transit fit consumes."""
    t = np.asarray(times, float); y = np.asarray(values, float)
    phase = ((t - t0) / float(period)) % 1.0
    order = np.argsort(phase)
    return phase[order], y[order]


def _selftest():
    """Regression trap: recover a planted period from unevenly-sampled noisy data, fold coherently on it, and show
    the honesty null separates a real signal from noise."""
    rng = np.random.default_rng(0)
    # Uneven sampling with a big gap (like day/night), a planted period, and noise.
    t = np.sort(np.concatenate([rng.uniform(0, 8, 60), rng.uniform(15, 25, 50)]))
    P0 = 2.137                                            # the true period (days)
    y = 1.4 * np.sin(2 * np.pi * t / P0 + 0.7) + 0.3 * rng.standard_normal(t.size)

    bp = best_period(t, y, min_period=0.5, max_period=10.0)
    assert abs(bp["period"] - P0) / P0 < 0.02, "recovered period off: %.4f vs %.4f" % (bp["period"], P0)

    # folding on the recovered period is COHERENT; on a deliberately wrong period it scatters more
    _, yr = phase_fold(t, y, bp["period"])
    _, yw = phase_fold(t, y, bp["period"] * 1.35)
    def fold_scatter(vals):                               # mean abs successive difference in phase order
        return np.mean(np.abs(np.diff(vals)))
    assert fold_scatter(yr) < fold_scatter(yw), "true-period fold should be smoother than a wrong-period fold"

    # even-sampling sanity: LS peak lands where an FFT would
    te = np.linspace(0, 20, 256); ye = np.sin(2 * np.pi * te / 2.0)
    be = best_period(te, ye, min_period=0.5, max_period=6.0)
    assert abs(be["period"] - 2.0) < 0.05, "even-sampling period wrong: %.3f" % be["period"]

    # HONESTY NULL: a real signal is significant (low fap); pure noise is not (high fap)
    f = freq_grid(t, min_period=0.5, max_period=10.0)
    fap_sig = false_alarm_probability(t, y, bp["power"], f, n_null=150, seed=1)
    noise = 0.3 * rng.standard_normal(t.size)
    bn = best_period(t, noise, min_period=0.5, max_period=10.0)
    fap_noise = false_alarm_probability(t, noise, bn["power"], f, n_null=150, seed=1)
    assert fap_sig < 0.05, "real signal should be significant, fap=%.3f" % fap_sig
    assert fap_noise > fap_sig, "noise fap (%.3f) should exceed signal fap (%.3f)" % (fap_noise, fap_sig)

    # determinism
    assert best_period(t, y)["period"] == best_period(t, y)["period"]

    print("holographic_lombscargle selftest OK  |  planted period %.3f recovered to <2%% from uneven+gapped data; "
          "true-period fold coherent; honesty null separates signal (fap %.3f) from noise (fap %.3f)" % (P0, fap_sig, fap_noise))


if __name__ == "__main__":
    _selftest()

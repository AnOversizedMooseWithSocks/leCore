"""DL11 -- canonical affine recovery, and three places the backlog's own reasoning needed correcting.

Translate and scale do not commute, and scale is not diagonal in the linear-frequency basis. The family CLOSES,
though: every order of a chain collapses to some single affine group element, and that element is what is
recoverable. `|FFT|` kills the translation; a log-frequency resample turns the dilation into a SHIFT; the estimator
is then the same cross-correlation-with-a-parabola `est_dx` uses on images.

THREE CORRECTIONS, each measured:

  1. **The SUPPORT BAND is the gate, not log-vs-plain magnitudes.** The backlog's reason -- "dilation scales the
     spectrum amplitude, which tilts plain correlation" -- cannot be right: multiplying one signal by a constant
     scales the entire cross-correlation and leaves the argmax exactly where it was.
  2. **The group law is exact on the PARAMETERS; repeated RESAMPLING is not.** Four interpolated resamples do not
     reproduce one resample by (S, T). The gap converges with sampling density, not at any fixed n.
  3. **State the unit.** The scale recovers to 3.7e-04; the SHIFT recovers to 0.37 SAMPLES, not to 1e-4.
"""

import numpy as np
import pytest

from holographic.sampling_and_signal.holographic_registration import (
    _correlate_1d, _parabolic_peak_1d, affine_compose, alignment, mellin_scale, recover_affine, refine_affine,
    resample_affine, support_band)


def _broadband(n=2048):
    """A chirp plus a narrow bump: SUPPORT ACROSS THE SPECTRUM. A narrowband signal has no log-axis to correlate
    on, and using one as the fixture would test the wrong thing (see the band tests below)."""
    x = np.linspace(0, 1, n)
    return (np.sin(2 * np.pi * (20 * x + 60 * x ** 2)) * np.exp(-((x - 0.5) ** 2) / 0.06)
            + 0.5 * np.sin(2 * np.pi * 180 * x) * np.exp(-((x - 0.3) ** 2) / 0.005))


def _narrowband(n=2048):
    x = np.linspace(0, 1, n)
    return np.exp(-((x - 0.45) ** 2) / 0.01) * np.sin(30 * x)


CHAIN = [(1.03, 4.0), (0.98, -2.5), (1.05, 3.1), (1.02, -3.0)]


def test_selftest_runs():
    from holographic.sampling_and_signal import holographic_registration as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# the group law
# ---------------------------------------------------------------------------------------------------------

def test_the_affine_group_law_is_exact():
    # x -> s2 (s1 x + t1) + t2 = (s2 s1) x + (s2 t1 + t2)
    S, T = affine_compose([(2.0, 1.0), (3.0, 5.0)])
    assert abs(S - 6.0) < 1e-15 and abs(T - 8.0) < 1e-15
    assert affine_compose([]) == (1.0, 0.0)                    # the identity element
    assert affine_compose([(1.0, 0.0)]) == (1.0, 0.0)


def test_order_matters_to_the_result_but_never_breaks_closure():
    S1, T1 = affine_compose(CHAIN)
    S2, T2 = affine_compose(list(reversed(CHAIN)))
    assert abs(S1 - S2) < 1e-12                                # scales commute among themselves
    assert abs(T1 - T2) > 1e-3                                 # ... shifts do not
    for chain in (CHAIN, list(reversed(CHAIN))):
        s, t = affine_compose(chain)
        assert np.isfinite(s) and np.isfinite(t) and s > 0     # every order gives SOME single group element


def test_kept_negative_the_group_law_is_exact_on_parameters_not_on_resampling():
    # Four interpolated resamples do NOT reproduce one resample by (S, T). Interpolation and zero-fill are lossy
    # and they compose lossily. The identity is recovered in the LIMIT of sampling density.
    S, T = affine_compose(CHAIN)
    gaps = []
    for n in (1024, 2048, 8192):
        f = _broadband(n)
        y = f.copy()
        for (s, t) in CHAIN:
            y = resample_affine(y, s, t)
        gaps.append(float(np.abs(y - resample_affine(f, S, T)).max()))
    assert gaps[0] > 0.05                                      # not identical at a fixed n
    assert gaps == sorted(gaps, reverse=True)                  # ... and strictly converging
    assert gaps[-1] < 0.02


# ---------------------------------------------------------------------------------------------------------
# the lift: scale becomes translation
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("s_true", [1.05, 1.2, 1.5])
def test_the_mellin_lift_recovers_a_dilation(s_true):
    f = _broadband()
    g = resample_affine(f, s_true, 17.0)
    assert abs(mellin_scale(f, g) - s_true) < 0.05 * s_true    # translation-invariant: the 17.0 does not matter


def test_the_mellin_scale_ignores_the_translation_entirely():
    f = _broadband()
    a = mellin_scale(f, resample_affine(f, 1.2, 0.0))
    b = mellin_scale(f, resample_affine(f, 1.2, 40.0))
    assert abs(a - b) < 0.02                                   # |FFT| discarded it


# ---------------------------------------------------------------------------------------------------------
# KEPT NEGATIVE 1: the band, not the log
# ---------------------------------------------------------------------------------------------------------

def test_a_constant_amplitude_factor_cannot_move_an_argmax():
    # THE BACKLOG'S STATED REASON, refuted. Scaling one signal scales the whole cross-correlation.
    rng = np.random.default_rng(0)
    a = rng.normal(size=256)
    b = np.roll(a, 7)
    p1 = _parabolic_peak_1d(_correlate_1d(a - a.mean(), b - b.mean()))
    p2 = _parabolic_peak_1d(_correlate_1d(a - a.mean(), 5.0 * (b - b.mean())))
    assert abs(p1 - 7.0) < 1e-9
    assert abs(p1 - p2) < 1e-9


def test_kept_negative_an_unbanded_narrowband_spectrum_recovers_scale_one_for_everything():
    # THE REAL GATE. Outside its support the magnitude spectrum is a noise floor; `log` amplifies it into a
    # structureless signal that dominates the correlation and pins the peak at ZERO shift.
    f = _narrowband()
    for s_true in (1.05, 1.2, 1.5):
        g = resample_affine(f, s_true, 9.0)
        assert abs(mellin_scale(f, g, frac=0.0) - 1.0) < 0.02   # every true scale reads as 1.00


def test_support_band_finds_the_narrow_support():
    F = np.abs(np.fft.rfft(_narrowband()))[1:]
    lo, hi = support_band(F)
    assert lo >= 1 and hi < len(F) // 4                          # genuinely narrow

    Fb = np.abs(np.fft.rfft(_broadband()))[1:]
    lo2, hi2 = support_band(Fb)
    assert hi2 > 4 * hi                                          # and the broadband one is not


def test_banding_rescues_both_plain_and_log_on_a_broadband_signal():
    f = _broadband()
    g = resample_affine(f, 1.5, 17.0)
    for use_log in (True, False):
        assert abs(mellin_scale(f, g, use_log=use_log) - 1.5) < 0.05    # both work, once banded


# ---------------------------------------------------------------------------------------------------------
# THE BAR, and the unit it is measured in
# ---------------------------------------------------------------------------------------------------------

def test_blind_recovery_of_an_exact_single_affine():
    f = _broadband()
    S, T = affine_compose(CHAIN)
    r = recover_affine(f, resample_affine(f, S, T))
    assert abs(r["scale"] - S) < 1e-3
    assert r["alignment"] > 0.999


def test_blind_recovery_of_the_chain_and_state_the_unit():
    f = _broadband()
    S, T = affine_compose(CHAIN)
    y = f.copy()
    for (s, t) in CHAIN:
        y = resample_affine(y, s, t)
    r = recover_affine(f, y)

    assert abs(r["scale"] - S) < 5e-3                           # measured 3.7e-04
    assert abs(r["shift"] - T) < 1.0                            # measured 0.37 SAMPLES -- not 1e-4
    assert r["alignment"] > 0.99


def test_the_refine_carries_the_near_identity_case():
    # KEPT NEGATIVE 2: the coarse Mellin stage alone is not enough when the scale is close to 1 -- the log axis
    # shifts by a fraction of a bin. The refine is what lands it.
    f = _broadband()
    g = resample_affine(f, 1.004, 2.0)
    coarse = recover_affine(f, g, refine=False)
    fine = recover_affine(f, g, refine=True)
    assert fine["alignment"] >= coarse["alignment"]
    assert abs(fine["scale"] - 1.004) <= abs(coarse["scale"] - 1.004) + 1e-9


def test_alignment_is_returned_so_a_bad_answer_can_announce_itself():
    f = _broadband()
    unrelated = np.random.default_rng(1).normal(size=len(f))
    r = recover_affine(f, unrelated)
    assert r["alignment"] < 0.5                                 # the estimator cannot explain this pair, and says so


def test_alignment_peaks_at_the_truth():
    f = _broadband()
    S, T = 1.2, 6.0
    g = resample_affine(f, S, T)
    best = alignment(f, g, S, T)
    for (s, t) in ((S * 1.05, T), (S, T + 4.0), (S * 0.9, T - 3.0)):
        assert alignment(f, g, s, t) < best


# ---------------------------------------------------------------------------------------------------------
# mechanics
# ---------------------------------------------------------------------------------------------------------

def test_resample_affine_is_the_identity_at_s_one_t_zero():
    f = _broadband(512)
    assert np.abs(resample_affine(f, 1.0, 0.0) - f).max() < 1e-12


def test_resample_affine_zero_fills_rather_than_wrapping():
    # An edit history moves content off the end. Pretending it wrapped would make the estimator's job easy.
    f = np.ones(64)
    out = resample_affine(f, 1.0, 30.0)
    assert out[:30].sum() == 0.0 and out[35] == 1.0


def test_recovery_is_deterministic():
    f = _broadband(1024)
    g = resample_affine(f, 1.1, 5.0)
    assert recover_affine(f, g) == recover_affine(f, g)
    assert refine_affine(f, g, 1.1, 5.0) == refine_affine(f, g, 1.1, 5.0)


def test_the_identity_pair_recovers_the_identity():
    f = _broadband(1024)
    r = recover_affine(f, f)
    assert abs(r["scale"] - 1.0) < 1e-2 and abs(r["shift"]) < 1.0 and r["alignment"] > 0.999


def test_a_flat_signal_degrades_gracefully():
    flat = np.zeros(256)
    assert mellin_scale(flat, flat) == 1.0
    assert alignment(flat, flat, 1.0, 0.0) == 0.0               # no direction to correlate


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    assert m.affine_compose([(2.0, 1.0), (3.0, 5.0)]) == (6.0, 8.0)

    f = _broadband()
    S, T = m.affine_compose(CHAIN)
    r = m.recover_affine(f, resample_affine(f, S, T))
    assert abs(r["scale"] - S) < 1e-3 and r["alignment"] > 0.999
    assert abs(m.mellin_scale(f, resample_affine(f, 1.2, 11.0)) - 1.2) < 0.06

    for phrase in ("recover a scale and shift between two signals", "fourier mellin registration",
                   "canonical affine edit"):
        assert "Canonical affine" in str(m.find_capability(phrase)[:3]), phrase

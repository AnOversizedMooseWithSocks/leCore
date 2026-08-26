"""F4 -- the cloud stack, and the two probe bugs that nearly buried the result.

The backlog: *"the photoreal cloud stack is an assembly of shipped parts -- and the honest bar: match a reference
raymarcher's image at equal quality, count steps vs one integral."*

THE ANSWER, measured: the closed form pays on the **shadow ray**, not the view ray. The view integral must march
(its integrand contains the transmittance being accumulated); the shadow ray is a pure line integral of density and
has a closed form. And it is not a speed-for-accuracy trade -- the closed form is the EXACT integral, so the march
is the one carrying error.

    shadow method       density evals   time        max |dI| vs reference
    64 marched (ref)          2,080     36,020 ms       --
    16 marched                  544      9,652 ms     3.94e-06
     8 marched                  288      5,048 ms     1.66e-05
    closed form                  32        687 ms     **3.03e-07**

TWO PROBE BUGS, MINE, PINNED HERE so nobody repeats them:
  * `optical_depth` takes a PER-RAY `L`. My first probe passed the median shadow length for every ray, measured
    3.4e-03 error, and I nearly filed it as the closed form being inaccurate. *Read the signature.*
  * My first probe fired rays from outside the encoder's box, where the `ScalarEncoder` warns that values are not
    distinguishable. Fixing that also fixed `volint`'s own self-test -- whose probe rays left the box -- taking its
    closed-form-vs-marched correlation from 0.9991 to **1.0000**.
"""

import warnings

import numpy as np
import pytest

from holographic.misc.holographic_volint import HolographicVolume
from holographic.rendering.holographic_cloud import (
    cloud_report, phase_hg, single_scatter, transmittance)
from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder


def _volume(n_blobs=20, dim=256, calibration_steps=96, seed=0):
    rng = np.random.default_rng(seed)
    enc = VectorFunctionEncoder(3, dim=dim, bounds=[(-1, 1)] * 3, bandwidth=2.5, seed=0)
    centers = rng.uniform(-0.55, 0.55, size=(n_blobs, 3))
    weights = rng.uniform(0.6, 1.4, size=n_blobs)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)       # calibration must stay inside the encoder's box
        return HolographicVolume.from_blobs(enc, centers, weights, calibration_steps=calibration_steps)


def _rays(R=16, seed=1):
    rng = np.random.default_rng(seed)
    O = np.stack([np.full(R, -0.95), rng.uniform(-0.3, 0.3, R), rng.uniform(-0.3, 0.3, R)], axis=1)
    D = np.tile(np.array([1.0, 0.0, 0.0]), (R, 1))
    return O, D, 1.90


def test_selftest_runs():
    from holographic.rendering import holographic_cloud as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# volint's own defect, fixed
# ---------------------------------------------------------------------------------------------------------

def test_the_calibration_probe_stays_inside_the_encoders_box():
    # It did not: 5 of 6 probe rays started near the low corner with a RANDOM direction and ended outside
    # [-1,1]^3, so the calibration constant was fitted partly on samples the encoder calls meaningless.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _volume()                                            # constructing it runs _calibrate(); no warning = fixed


def test_volints_own_selftest_now_correlates_at_one():
    from holographic.misc import holographic_volint as mod
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mod._selftest()                                       # its probe rays used to leave the box too


def test_more_calibration_steps_buy_a_better_scale():
    # The closed form's SHAPE is exact; its physical SCALE is a fitted constant, and its accuracy is the accuracy
    # of the march it was fitted against. That is the floor on every accuracy number in this module.
    O, D, L = _rays(R=8)
    coarse, fine = _volume(calibration_steps=8), _volume(calibration_steps=192)

    def _march(vol, n):
        acc = np.zeros(len(O))
        for i in range(n):
            acc += np.clip(vol.density(O + (i + 0.5) / n * L * D), 0.0, None) * (L / n)
        return acc

    ref = _march(fine, 512)
    e_coarse = np.abs(coarse.optical_depth(O, D, L) - ref).max() / ref.max()
    e_fine = np.abs(fine.optical_depth(O, D, L) - ref).max() / ref.max()
    assert e_fine < e_coarse


# ---------------------------------------------------------------------------------------------------------
# transmittance, and the per-ray L that nearly fooled me
# ---------------------------------------------------------------------------------------------------------

def test_transmittance_is_exactly_beer_lambert():
    vol = _volume()
    O, D, L = _rays()
    t1 = transmittance(vol, O, D, L, sigma_t=1.0)
    t2 = transmittance(vol, O, D, L, sigma_t=2.0)
    assert np.all((t1 > 0.0) & (t1 <= 1.0))
    assert np.allclose(t2, t1 ** 2, atol=1e-12)              # exp(-2 tau) == exp(-tau)^2, exactly
    assert np.all(t2 <= t1)


def test_kept_negative_the_per_ray_L_is_really_per_ray():
    # PROBE BUG 1. `optical_depth(O, D, L)` accepts `L: scalar or (R,)`. Passing the median for every ray is a
    # DIFFERENT integral, and measuring its error as the closed form's would slander the closed form 1000x.
    vol = _volume()
    O, D, _L = _rays()
    Ls = np.linspace(0.4, 1.9, len(O))
    per_ray = vol.optical_depth(O, D, Ls)
    median = vol.optical_depth(O, D, float(np.median(Ls)))
    assert np.abs(per_ray - median).max() > 1e-2
    assert per_ray[0] < per_ray[-1]                          # a longer ray accumulates more depth


def test_transmittance_of_a_zero_length_ray_is_one():
    vol = _volume()
    O, D, _L = _rays(R=4)
    assert np.allclose(transmittance(vol, O, D, 0.0), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------------------------------------
# THE BAR: the closed-form shadow is cheaper AND more accurate
# ---------------------------------------------------------------------------------------------------------

def test_the_closed_form_shadow_uses_far_fewer_density_evaluations():
    vol = _volume()
    O, D, L = _rays()
    rep = cloud_report(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, view_steps=16, reference_shadow_steps=48)
    assert rep["evals_closed"] == 16                         # one per view step, and nothing more
    assert rep["evals_reference"] == 16 * 49
    assert rep["eval_ratio"] > 20.0


def test_the_closed_form_shadow_is_also_more_accurate_than_a_marched_one():
    # This is the claim worth pinning. It is not a speed-for-accuracy trade: the closed form is the exact integral.
    vol = _volume()
    O, D, L = _rays()
    kw = dict(view_steps=16)
    ref, _ = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), 0.95, shadow_steps=48, **kw)
    closed, c_ev = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), 0.95, shadow_steps=0, **kw)
    m8, m8_ev = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), 0.95, shadow_steps=8, **kw)

    assert c_ev < m8_ev
    assert np.abs(closed - ref).max() < np.abs(m8 - ref).max()
    assert np.abs(closed - ref).max() < 1e-4


def test_a_marched_shadow_converges_toward_the_closed_form():
    # ... which is the other way of saying the closed form is the truth the march is approximating.
    vol = _volume()
    O, D, L = _rays(R=8)
    closed, _ = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), 0.95, view_steps=8, shadow_steps=0)
    errs = []
    for s in (4, 16, 64):
        got, _ev = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), 0.95, view_steps=8, shadow_steps=s)
        errs.append(float(np.abs(got - closed).max()))
    assert errs == sorted(errs, reverse=True)


def test_more_view_steps_do_not_change_the_shadow_cost_per_step():
    vol = _volume()
    O, D, L = _rays(R=8)
    _r1, e1 = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), 0.95, view_steps=8, shadow_steps=0)
    _r2, e2 = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), 0.95, view_steps=32, shadow_steps=0)
    assert e1 == 8 and e2 == 32                              # exactly one density call per view step


# ---------------------------------------------------------------------------------------------------------
# the phase function, and the guards
# ---------------------------------------------------------------------------------------------------------

def test_the_phase_function_is_normalised_forward_peaked_and_isotropic_at_g_zero():
    assert abs(float(phase_hg(1.0, 0.4)[0]) - 1.0) < 1e-12
    assert float(phase_hg(1.0, 0.6)[0]) > float(phase_hg(0.0, 0.6)[0]) > float(phase_hg(-1.0, 0.6)[0])
    for c in (-0.9, 0.0, 0.5):
        assert abs(float(phase_hg(c, 0.0)[0]) - 1.0) < 1e-12  # g=0 is isotropic; normalisation makes it exactly 1
    assert float(phase_hg(-1.0, -0.5)[0]) > float(phase_hg(1.0, -0.5)[0])   # backward scattering


def test_a_horizontal_sun_raises_rather_than_dividing_by_zero():
    vol = _volume()
    O, D, L = _rays(R=4)
    with pytest.raises(ValueError, match="non-zero y"):
        single_scatter(vol, O, D, L, (1.0, 0.0, 0.0), ceiling=0.95)


def test_radiance_grows_with_the_scattering_coefficient_and_dies_with_absorption():
    vol = _volume()
    O, D, L = _rays(R=8)
    lo, _ = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), 0.95, view_steps=8, sigma_s=0.3)
    hi, _ = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), 0.95, view_steps=8, sigma_s=0.9)
    assert np.all(hi > lo)

    thick, _ = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), 0.95, view_steps=8, sigma_t=6.0)
    thin, _ = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), 0.95, view_steps=8, sigma_t=0.5)
    assert thick.max() < thin.max()                          # a thicker medium shadows itself


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    vol = _volume()
    O, D, L = _rays(R=8)
    assert np.all(m.cloud_transmittance(vol, O, D, L) <= 1.0)

    rad, evals = m.cloud_single_scatter(vol, O, D, L, (0, 1, 0), ceiling=0.95, view_steps=8)
    assert evals == 8 and np.all(rad >= 0.0)

    rep = m.cloud_report(vol, O, D, L, (0, 1, 0), ceiling=0.95, view_steps=8, reference_shadow_steps=32)
    assert rep["eval_ratio"] > 20.0 and rep["max_error"] < 1e-4

    for phrase in ("render a cloud", "shadow ray without marching", "participating media"):
        assert "Cloud stack" in str(m.find_capability(phrase)[:3]), phrase


def test_analytic_segment_integral_beats_the_rectangle_rule():
    """The opt-in analytic segment integral (Hillaire, Frostbite SIGGRAPH 2015) integrates each step against its
    OWN extinction instead of assuming constant transmittance across it. Through the mind: it must beat the
    rectangle rule at equal step count, both must converge to the same answer (so the reference is not a strawman),
    and the DEFAULT must remain rect so no recorded decision moves."""
    import warnings
    import numpy as np
    import lecore
    from holographic.misc.holographic_volint import HolographicVolume
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder

    m = lecore.UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(0)
    enc = VectorFunctionEncoder(3, dim=256, bounds=[(-1, 1)] * 3, bandwidth=2.5, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vol = HolographicVolume.from_blobs(enc, rng.uniform(-0.5, 0.5, size=(16, 3)), calibration_steps=96)

    O = np.stack([np.full(8, -0.95), np.zeros(8), np.linspace(-0.2, 0.2, 8)], axis=1)
    D = np.tile([1.0, 0.0, 0.0], (8, 1))
    kw = dict(L=1.9, sun_dir=(0, 1, 0), ceiling=0.95)

    # both modes converge to the same answer -> the reference is trustworthy, not built from the winner
    ref_r, _ = m.cloud_single_scatter(vol, O, D, view_steps=512, integrate="rect", **kw)
    ref_a, _ = m.cloud_single_scatter(vol, O, D, view_steps=512, integrate="analytic", **kw)
    assert np.abs(ref_r - ref_a).max() < 1e-3, np.abs(ref_r - ref_a).max()

    err_rect = np.abs(m.cloud_single_scatter(vol, O, D, view_steps=8, integrate="rect", **kw)[0] - ref_r).mean()
    err_anal = np.abs(m.cloud_single_scatter(vol, O, D, view_steps=8, integrate="analytic", **kw)[0] - ref_r).mean()
    assert err_anal < err_rect, (err_anal, err_rect)

    # the default is rect, exactly -- an existing decision must never flip
    d1, e1 = m.cloud_single_scatter(vol, O, D, view_steps=32, **kw)
    d2, e2 = m.cloud_single_scatter(vol, O, D, view_steps=32, integrate="rect", **kw)
    assert np.array_equal(d1, d2) and e1 == e2

    # an unknown mode is refused, never silently treated as rect
    try:
        m.cloud_single_scatter(vol, O, D, integrate="bogus", **kw)
        assert False, "unknown integrate mode must raise"
    except ValueError:
        pass

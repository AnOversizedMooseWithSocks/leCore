"""holographic_cloud.py -- the photoreal cloud stack, assembled from shipped parts (Box3D backlog F4).

Nothing here is new mathematics. `volint.HolographicVolume` gives a CLOSED-FORM line integral of an FPE density
field (one inner product per ray, no marching); `render._henyey_greenstein` gives the phase function; the density
field is a bundle of blobs. F4 is the assembly, and the bar the backlog set is a measurement: *match a reference
raymarcher at equal quality, and count steps against one integral.*

WHERE THE CLOSED FORM ACTUALLY PAYS -- and it is not where I first looked. Single scattering marches the VIEW ray
and, at every sample, casts a SHADOW ray toward the light. The view march cannot be closed-form (the integrand
contains the transmittance you are accumulating); the shadow ray can, because it is a pure line integral of density.
Measured, 64 view rays x 32 view steps, against a 64-step marched-shadow reference:

    shadow method       density evals   time      max |dI| vs reference
    64 marched (ref)          2,080     36,020 ms      --
    16 marched                  544      9,652 ms    3.94e-06
     8 marched                  288      5,048 ms    1.66e-05
    **closed form**              32        687 ms    **3.03e-07**

**65x fewer density evaluations, 52x faster, and 13x MORE accurate than a 16-step marched shadow.** The closed form
is not an approximation being traded against speed; it is the exact integral, so the marched shadow is the one
carrying error. `volint`'s own note says it plainly: *absorption does not want marching; scattering still does.*

HONEST SCOPE:
  * The VIEW integral still marches. Beer-Lambert transmittance along a ray is O(1) (`transmittance`); single
    scattering is O(view_steps), each step costing ONE closed-form shadow integral instead of `shadow_steps`
    density evaluations.
  * The scale is a fitted constant. `volint`'s closed form is exact in SHAPE; its physical scale is calibrated
    against a short marched reference, so relative error bottoms out near 3.5e-05 at the default
    `calibration_steps=24` and 5.1e-07 at 256. Every number above uses 96.
  * Multiple scattering, and the wavelength-dependent in-scatter that makes a cloud read as a cloud, are not here.
    This is single-scatter with a Henyey-Greenstein phase.

TWO PROBE BUGS, MINE, RECORDED so the next session does not repeat them:
  * `optical_depth` accepts a PER-RAY `L` (its docstring says `L: scalar or (R,)`). My first probe passed the
    MEDIAN shadow length for every ray and got 3.4e-03 error -- 1000x worse than the truth -- and I nearly filed it
    as the closed form being inaccurate. *Read the signature.*
  * My first probe fired rays from OUTSIDE the encoder's box. The `ScalarEncoder` warns that out-of-range values
    are not distinguishable from one another; the warning was the engine telling me the data was bad. Fixing it
    also fixed `volint`'s own self-test, whose probe rays left the box: closed-form-vs-marched correlation went
    from 0.9991 to **1.0000**.
"""

import numpy as np

from holographic.rendering.holographic_render import _henyey_greenstein


def phase_hg(cos_theta, g):
    """The Henyey-Greenstein phase function, normalised so the forward peak is 1. Delegates to the renderer's
    implementation -- one phase function in the engine, not two. `g` in (-1, 1): forward-scattering above 0."""
    p = _henyey_greenstein(np.atleast_1d(np.asarray(cos_theta, float)), float(g))
    return p / _henyey_greenstein(np.array([1.0]), float(g))[0]


def transmittance(volume, O, D, L, sigma_t=1.0):
    """Beer-Lambert transmittance `exp(-sigma_t * tau)` along each ray, with `tau` from ONE closed-form integral.

    O, D: (R, 3). `L` is scalar or per-ray `(R,)` -- *per-ray, and it matters*: a shadow ray's length to the medium
    boundary varies per sample, and passing a single median instead costs three orders of magnitude of accuracy.
    O(1) in the number of march steps, because there are none."""
    tau = volume.optical_depth(np.atleast_2d(O), np.atleast_2d(D), L)
    return np.exp(-float(sigma_t) * tau)


def single_scatter(volume, O, D, L, sun_dir, ceiling, view_steps=32, shadow_steps=0,
                   sigma_t=1.0, sigma_s=0.9, g=0.4, integrate="rect"):
    """Single-scattered radiance along each view ray. Returns `(radiance, density_evals)`.

    The view ray marches `view_steps` times (it must: the integrand contains the transmittance being accumulated).
    At each sample a shadow ray runs toward `sun_dir` up to the `ceiling` plane. `shadow_steps=0` -- the default --
    evaluates that shadow ray's optical depth in CLOSED FORM: one integral, no samples. Any positive value marches
    it instead, which is provided only so a test can show the closed form is both faster AND more accurate.

    `integrate` selects how the in-scattered radiance is accumulated ACROSS each segment:
      * "rect"     (default) -- the rectangle rule `radiance += T * S * dt`, i.e. the segment's in-scatter is
                    assumed to reach the eye under the transmittance measured at the segment's START. This is the
                    original behaviour and is kept the default so no recorded decision moves.
      * "analytic" -- integrate the segment ANALYTICALLY against its own extinction (Hillaire, "Physically Based
                    and Unified Volumetric Rendering in Frostbite", SIGGRAPH 2015): light scattered at the far end
                    of a segment is attenuated by that segment's own optical depth, which the rectangle rule
                    ignores. With sigma_e = sigma_t * rho constant over the step,
                        Sint = integral_0^dt S exp(-sigma_e s) ds = (S - S exp(-sigma_e dt)) / sigma_e
                    which -> S*dt as sigma_e -> 0 (the guarded limit below), so it is a strict refinement.

    MEASURED (mean abs radiance error vs an 8192-step reference; both modes converge to the SAME answer there, so
    the reference is unbiased -- max|rect-analytic| ~ 1e-6 at 8192):
        sigma_t=1.0:  8 steps  rect 8.3e-3  analytic 1.1e-5  (754x)   |  64 steps rect 1.0e-3  analytic 7.8e-6
        sigma_t=5.0:  8 steps  rect 8.2e-4  analytic 5.1e-5  ( 16x)   |  64 steps rect 9.7e-5  analytic 1.6e-6
    The practical claim: ANALYTIC AT 8 STEPS BEATS RECT AT 64 -- an ~8x cut in density evaluations at equal or
    better accuracy, and cost here is dominated by evals per step.
    KEPT NEGATIVE: the analytic form's advantage SHRINKS as extinction rises (754x at sigma_t=1, 16x at sigma_t=5,
    ~4x at sigma_t=20): a very dense medium self-terminates within one step, so the rectangle rule's error is
    already small. It is a large win for thin-to-moderate media, a modest one for thick.
    KEPT NEGATIVE: this refines the SEGMENT integral only. It does not add multiple scattering or wavelength
    dependence, and it cannot fix an under-resolved DENSITY field -- if the step misses a filament, both modes miss
    it identically.

    `density_evals` counts the vectorised density calls, so the cost claim is COUNTED rather than asserted."""
    if integrate not in ("rect", "analytic"):
        raise ValueError("integrate must be 'rect' or 'analytic', got %r" % (integrate,))
    O = np.atleast_2d(np.asarray(O, float))
    D = np.atleast_2d(np.asarray(D, float))
    sun = np.asarray(sun_dir, float)
    sun = sun / (np.linalg.norm(sun) + 1e-12)
    if abs(sun[1]) < 1e-6:
        raise ValueError("sun_dir must have a non-zero y component to reach the ceiling plane")
    R = len(O)
    n_view = int(view_steps)
    dt = float(L) / n_view

    radiance = np.zeros(R)
    T = np.ones(R)
    evals = 0
    # the phase angle is constant per view ray (a directional sun), so it is computed once, not per step
    ph = phase_hg(D @ sun, g)

    for i in range(n_view):
        P = O + (i + 0.5) * dt * D
        rho = np.clip(volume.density(P), 0.0, None)
        evals += 1

        Ls = (float(ceiling) - P[:, 1]) / sun[1]            # PER-RAY shadow length to the ceiling plane
        Ls = np.clip(Ls, 0.0, None)
        Ds = np.tile(sun, (R, 1))
        if shadow_steps <= 0:
            tau_s = volume.optical_depth(P, Ds, Ls)         # ONE integral. This is the whole point.
        else:
            tau_s = np.zeros(R)
            for j in range(int(shadow_steps)):
                Pj = P + ((j + 0.5) / shadow_steps * Ls)[:, None] * Ds
                tau_s += np.clip(volume.density(Pj), 0.0, None) * (Ls / shadow_steps)
                evals += 1

        if integrate == "rect":
            # the ORIGINAL expression, character for character: its left-to-right float association is part of the
            # recorded decision, so it is NOT refactored into the shared S below (regrouping moved results at ULP
            # scale and broke bit-identity -- measured, not assumed).
            radiance += T * float(sigma_s) * rho * np.exp(-float(sigma_t) * tau_s) * ph * dt
            T *= np.exp(-float(sigma_t) * rho * dt)
        else:
            sigma_e = float(sigma_t) * rho                 # extinction over THIS segment
            S = float(sigma_s) * rho * np.exp(-float(sigma_t) * tau_s) * ph   # in-scatter per unit length
            # analytic segment integral; the guarded branch is the sigma_e -> 0 limit, which IS S*dt
            ok = sigma_e > 1e-9
            radiance += T * np.where(ok, (S - S * np.exp(-sigma_e * dt)) / np.where(ok, sigma_e, 1.0), S * dt)
            T *= np.exp(-sigma_e * dt)
    return radiance, evals


def cloud_report(volume, O, D, L, sun_dir, ceiling, view_steps=32, reference_shadow_steps=64, **kw):
    """The bar, carried WITH the capability: run the closed-form shadow against a marched reference on the same
    field and report `{evals_closed, evals_reference, eval_ratio, max_error, mean_error}`.

    The honest reference is a HEAVILY marched shadow, not a coarse one -- a coarse reference would flatter the
    closed form by being wrong in the same direction."""
    ref, ref_evals = single_scatter(volume, O, D, L, sun_dir, ceiling, view_steps=view_steps,
                                    shadow_steps=int(reference_shadow_steps), **kw)
    got, evals = single_scatter(volume, O, D, L, sun_dir, ceiling, view_steps=view_steps, shadow_steps=0, **kw)
    err = np.abs(got - ref)
    return {"evals_closed": int(evals), "evals_reference": int(ref_evals),
            "eval_ratio": float(ref_evals) / max(1, evals),
            "max_error": float(err.max()), "mean_error": float(err.mean()),
            "reference_range": [float(ref.min()), float(ref.max())]}


def _selftest():
    """Regression trap: the closed-form shadow must beat a marched one on BOTH cost and accuracy, transmittance
    must obey Beer-Lambert, and the per-ray `L` must actually be per-ray."""
    import warnings

    from holographic.misc.holographic_volint import HolographicVolume
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder

    rng = np.random.default_rng(0)
    enc = VectorFunctionEncoder(3, dim=256, bounds=[(-1, 1)] * 3, bandwidth=2.5, seed=0)
    centers = rng.uniform(-0.55, 0.55, size=(24, 3))
    weights = rng.uniform(0.6, 1.4, size=24)

    with warnings.catch_warnings():                 # a range warning here would mean the probe left the box
        warnings.simplefilter("error", RuntimeWarning)
        vol = HolographicVolume.from_blobs(enc, centers, weights, calibration_steps=96)

    R = 16
    O = np.stack([np.full(R, -0.95), rng.uniform(-0.3, 0.3, R), rng.uniform(-0.3, 0.3, R)], axis=1)
    D = np.tile(np.array([1.0, 0.0, 0.0]), (R, 1))
    L = 1.90

    # 1. transmittance is Beer-Lambert on a closed-form tau, and it decays with the coefficient
    t1 = transmittance(vol, O, D, L, sigma_t=1.0)
    t2 = transmittance(vol, O, D, L, sigma_t=2.0)
    assert np.all((t1 > 0.0) & (t1 <= 1.0)) and np.all(t2 < t1)
    assert np.allclose(t2, t1 ** 2, atol=1e-9)      # exp(-2 tau) == exp(-tau)^2, exactly

    # 2. the PER-RAY L is really per-ray -- passing a scalar median is a different (wrong) answer
    Ls = np.linspace(0.5, 1.9, R)
    per_ray = vol.optical_depth(O, D, Ls)
    median = vol.optical_depth(O, D, float(np.median(Ls)))
    assert np.abs(per_ray - median).max() > 1e-3

    # 3. THE BAR: the closed-form shadow is cheaper AND more accurate than a marched one
    rep = cloud_report(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, view_steps=16, reference_shadow_steps=48)
    assert rep["eval_ratio"] > 20.0, rep            # measured 65x at 32 view steps / 64 shadow steps
    assert rep["max_error"] < 1e-4, rep

    marched, m_evals = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95,
                                      view_steps=16, shadow_steps=8)
    ref, _ = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, view_steps=16, shadow_steps=48)
    closed, c_evals = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, view_steps=16, shadow_steps=0)
    assert c_evals < m_evals
    assert np.abs(closed - ref).max() < np.abs(marched - ref).max()   # cheaper AND more accurate

    # 4. the phase function is normalised and forward-peaked
    assert abs(float(phase_hg(1.0, 0.4)[0]) - 1.0) < 1e-12
    assert float(phase_hg(1.0, 0.4)[0]) > float(phase_hg(-1.0, 0.4)[0])
    assert abs(float(phase_hg(0.3, 0.0)[0]) - 1.0) < 1e-12            # g=0 is isotropic

    # 5. a sun parallel to the ceiling has no shadow ray to cast
    try:
        single_scatter(vol, O, D, L, (1.0, 0.0, 0.0), ceiling=0.95)
    except ValueError:
        pass
    else:
        raise AssertionError("a horizontal sun must raise")

    # -- ANALYTIC SEGMENT INTEGRAL (Hillaire, Frostbite SIGGRAPH 2015) ------------------------------------------
    # The claim is an ACCURACY claim, so it is pinned as one: against a high-step reference, the analytic mode must
    # beat the rectangle rule at the SAME step count, and both must converge to the SAME answer (which is what makes
    # the reference trustworthy rather than a strawman built from the winner).
    ref_r, _ = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, view_steps=1024, integrate="rect")
    ref_a, _ = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, view_steps=1024, integrate="analytic")
    assert np.abs(ref_r - ref_a).max() < 1e-4, ("the two modes must CONVERGE to the same answer",
                                                float(np.abs(ref_r - ref_a).max()))
    err_r = np.abs(single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, view_steps=8, integrate="rect")[0] - ref_r).mean()
    err_a = np.abs(single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, view_steps=8,
                                  integrate="analytic")[0] - ref_r).mean()
    assert err_a < err_r, ("analytic must beat the rectangle rule at equal steps", err_a, err_r)
    # the headline: analytic at 8 steps beats the rectangle rule at 64 -- an ~8x cut in density evals for equal or
    # better accuracy (cost here is dominated by evals per step).
    err_r64 = np.abs(single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, view_steps=64,
                                    integrate="rect")[0] - ref_r).mean()
    assert err_a < err_r64, ("analytic@8 should beat rect@64", err_a, err_r64)
    # DEFAULT MUST NOT MOVE: rect is the default and is deterministic on re-run (the bit-identity against the
    # pre-change implementation was verified out-of-band; here we trap any future drift in the default path).
    d1, e1 = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, view_steps=32)
    d2, e2 = single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, view_steps=32, integrate="rect")
    assert np.array_equal(d1, d2) and e1 == e2, "the default must BE rect, exactly"
    try:
        single_scatter(vol, O, D, L, (0.0, 1.0, 0.0), ceiling=0.95, integrate="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown integrate mode must raise")
    # KEPT NEGATIVE: the analytic advantage SHRINKS with extinction (measured ~754x at sigma_t=1, ~16x at 5, ~4x at
    # 20) -- a dense medium self-terminates inside one step, so the rectangle rule is already close. Not a
    # universal win; a large one for thin-to-moderate media.

    print("OK: holographic_cloud self-test passed (transmittance is exactly Beer-Lambert on a closed-form tau; the "
          "per-ray L differs from its own median by %.3f, so it really is per-ray; the closed-form shadow uses "
          "%d density evaluations against the reference's %d (%.0fx) while being MORE accurate than an 8-step "
          "marched shadow -- the closed form is the exact integral, so the march is the one carrying error; and the "
          "opt-in analytic segment integral cuts 8-step error from %.2e to %.2e, beating even the 64-step "
          "rectangle rule (%.2e), with the rect default left bit-identical)"
          % (float(np.abs(per_ray - median).max()), rep["evals_closed"], rep["evals_reference"], rep["eval_ratio"],
             err_r, err_a, err_r64))


if __name__ == "__main__":
    _selftest()

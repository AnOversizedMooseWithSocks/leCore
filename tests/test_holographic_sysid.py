"""Tests for holographic_sysid: mass/momentum identification and its gauge freedom.

Pins the physics contracts:
  * THE GAUGE (the theorem): two systems with scaled (mass, force) trace identical
    trajectories, so trajectory-only mass requests are REFUSED (GaugeError).
  * Door 1 (force channel): (m, c, k) recovered to ~1% from a driven oscillator.
  * Door 2 (interaction): the exact mass ratio from momentum conservation.
  * Door 3 (force law): a synthetic Earth orbit weighs the Sun to 1%, in 2-D and in
    an inclined 3-D plane; a partial arc is refused rather than extrapolated.

All arithmetic + least squares on fixed inputs: hard numeric contracts, no RNG.
"""

import numpy as np
import pytest

from holographic.sampling_and_signal.holographic_sysid import (
    GaugeError, derivatives, fit_second_order, mass_ratio_from_interaction,
    central_mass_from_orbit, identify,
)


# ---------------------------------------------------------------------------
# The gauge freedom (the theorem the module is built around)
# ---------------------------------------------------------------------------

def test_scaled_mass_and_force_are_indistinguishable():
    # (m=2,k=8) and (m=5,k=20) share omega^2 = k/m = 4, so their free oscillations
    # are bit-identical: the trajectory exposes only the RATIO. This is why mass is
    # unidentifiable from a path alone -- pinned so the refusal stays justified.
    t = np.arange(0, 4, 0.001)
    x_a = 0.7 * np.cos(np.sqrt(8 / 2) * t)
    x_b = 0.7 * np.cos(np.sqrt(20 / 5) * t)
    assert np.max(np.abs(x_a - x_b)) < 1e-10


def test_trajectory_alone_is_refused_with_the_gauge_statement():
    t = np.arange(0, 2, 0.001)
    with pytest.raises(GaugeError, match="UNIDENTIFIABLE"):
        identify(x=np.cos(2 * t), dt=0.001)


def test_kinematics_is_still_offered():
    # The refusal is about dynamics; kinematics IS observable and derivatives()
    # must deliver it: velocity of cos is -w sin to numerical accuracy.
    dt = 0.001
    t = np.arange(0, 3, dt)
    v, a = derivatives(np.cos(2 * t), dt)
    assert np.max(np.abs(v[100:-100] + 2 * np.sin(2 * t)[100:-100])) < 1e-2
    assert np.max(np.abs(a[100:-100] + 4 * np.cos(2 * t)[100:-100])) < 1e-1


# ---------------------------------------------------------------------------
# Door 1: force channel -> full (m, c, k) identification
# ---------------------------------------------------------------------------

def _simulate_driven(m, c, k, F, dt):
    # Semi-implicit Euler: cheap, stable at small dt; the fit must recover the
    # coefficients the simulator used.
    x = np.zeros_like(F)
    v = 0.0
    for i in range(1, len(F)):
        a_i = (F[i - 1] - c * v - k * x[i - 1]) / m
        v = v + a_i * dt
        x[i] = x[i - 1] + v * dt
    return x


def test_force_channel_identifies_mass_damping_stiffness():
    dt = 0.001
    t = np.arange(0, 4, dt)
    F = 1.5 * np.cos(1.3 * t)
    x = _simulate_driven(2.0, 0.4, 8.0, F, dt)
    fit = fit_second_order(x, F, dt)
    assert abs(fit["mass"] - 2.0) / 2.0 < 0.01
    assert abs(fit["stiffness"] - 8.0) / 8.0 < 0.01
    assert abs(fit["damping"] - 0.4) / 0.4 < 0.05
    assert fit["residual_rms"] < 1e-2            # the model genuinely fits
    assert fit["momentum"].shape == x.shape      # the full observable ledger


def test_residual_flags_a_wrong_model():
    # A system that is NOT (m,c,k)-shaped (a hard cubic stiffness) must show a
    # residual clearly above the linear case's -- the honest "don't trust these
    # coefficients" number, pinned as a CONTRAST not an absolute threshold.
    dt = 0.001
    t = np.arange(0, 4, dt)
    F = 1.5 * np.cos(1.3 * t)
    x_lin = _simulate_driven(2.0, 0.4, 8.0, F, dt)
    lin_resid = fit_second_order(x_lin, F, dt)["residual_rms"]

    x = np.zeros_like(t)
    v = 0.0
    for i in range(1, len(t)):
        a_i = (F[i - 1] - 0.4 * v - 40.0 * x[i - 1] ** 3) / 2.0  # cubic spring
        v = v + a_i * dt
        x[i] = x[i - 1] + v * dt
    cubic_resid = fit_second_order(x, F, dt)["residual_rms"]
    assert cubic_resid > 10 * lin_resid


# ---------------------------------------------------------------------------
# Door 2: interaction -> mass ratio (Mach)
# ---------------------------------------------------------------------------

def test_elastic_collision_returns_exact_mass_ratio():
    m1, m2, u1, u2 = 3.0, 1.0, 1.0, 0.0
    v1 = (m1 - m2) / (m1 + m2) * u1
    v2 = 2 * m1 / (m1 + m2) * u1
    assert abs(mass_ratio_from_interaction(u1, v1, u2, v2) - 3.0) < 1e-9


def test_vector_collision_projects_on_exchange_direction():
    # 2-D: the exchange happens along one line; components off it are unchanged and
    # must not pollute the ratio.
    dv1 = np.array([-0.5, 0.0])
    dv2 = np.array([1.5, 0.0])
    ratio = mass_ratio_from_interaction(np.array([1.0, 0.3]),
                                        np.array([1.0, 0.3]) + dv1,
                                        np.array([0.0, -0.2]),
                                        np.array([0.0, -0.2]) + dv2)
    assert abs(ratio - 3.0) < 1e-9


def test_no_interaction_is_refused():
    with pytest.raises(GaugeError, match="no interaction"):
        mass_ratio_from_interaction(1.0, 1.0, 0.0, 0.0)  # nothing changed


# ---------------------------------------------------------------------------
# Door 3: known force law -> central mass (astronomy)
# ---------------------------------------------------------------------------

def _circular_orbit(R, T, n, frac=1.2, incline_deg=0.0):
    tt = np.linspace(0, frac * T, n)
    c, s = np.cos(2 * np.pi * tt / T), np.sin(2 * np.pi * tt / T)
    if incline_deg:
        inc = np.deg2rad(incline_deg)
        pos = np.stack([R * c, R * s * np.cos(inc), R * s * np.sin(inc)], axis=1)
    else:
        pos = np.stack([R * c, R * s], axis=1)
    return pos, tt[1] - tt[0]


def test_earth_orbit_weighs_the_sun():
    pos, dt = _circular_orbit(1.496e11, 3.156e7, 4000)
    est = central_mass_from_orbit(pos, dt)
    assert abs(est["central_mass"] - 1.989e30) / 1.989e30 < 0.01
    assert abs(est["period"] - 3.156e7) / 3.156e7 < 0.01


def test_inclined_3d_orbit_also_works():
    pos, dt = _circular_orbit(1.496e11, 3.156e7, 4000, incline_deg=30.0)
    est = central_mass_from_orbit(pos, dt)
    assert abs(est["central_mass"] - 1.989e30) / 1.989e30 < 0.01


def test_partial_arc_is_refused():
    pos, dt = _circular_orbit(1.496e11, 3.156e7, 4000, frac=0.6)  # 0.6 orbits
    with pytest.raises(GaugeError, match="full orbit"):
        central_mass_from_orbit(pos, dt)


def test_identify_routes_to_the_right_door():
    # force -> door 1; interaction -> door 2; positions -> door 3.
    dt = 0.001
    t = np.arange(0, 3, dt)
    F = 1.5 * np.cos(1.3 * t)
    x = _simulate_driven(2.0, 0.4, 8.0, F, dt)
    assert identify(x=x, dt=dt, force=F)["door"] == "force_channel"

    inter = {"v1_before": 1.0, "v1_after": 0.5, "v2_before": 0.0, "v2_after": 1.5}
    assert identify(interaction=inter)["door"] == "interaction"

    pos, odt = _circular_orbit(1.496e11, 3.156e7, 2000)
    assert identify(positions=pos, dt=odt)["door"] == "force_law"


def test_selftest_runs():
    from holographic.sampling_and_signal.holographic_sysid import _selftest
    _selftest()

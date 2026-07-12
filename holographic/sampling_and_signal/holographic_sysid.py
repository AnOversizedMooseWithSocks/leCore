"""System identification: mass, momentum, and dynamics from a measurement series.

WHY THIS MODULE EXISTS
----------------------
KINEMATICS (position, velocity, acceleration, frequency, phase) is fully observable
from a trajectory -- it is the geometry of the path, and holographic_analytic
already reads it as rotation. DYNAMICS (mass, momentum, force) is NOT: mass is a
property of how a trajectory RESPONDS to force, and a passive trajectory contains
no force information. The obstruction is a GAUGE FREEDOM, a theorem not a tooling
limit: scale mass and force together (m -> s*m, F -> s*F) and the trajectory is
bit-identical forever (F = m*a exposes only F/m). Measured in this module's
selftest: (m=2, k=8) and (m=5, k=20) oscillators trace the same path.

So this module implements the three honest ways the gauge BREAKS, and a loud
refusal when none of them is available:

  1. A FORCE CHANNEL (an intervention / a driven system). Given both x(t) and the
     applied force F(t), fit the equation of motion m*a + c*v + k*x = F by least
     squares: mass, damping, and stiffness all become identifiable, in the force's
     units. This is how mass is actually measured -- push with a known force, watch
     the response. `fit_second_order`.

  2. AN INTERACTION between two measured objects. Momentum conservation during a
     collision gives the mass RATIO from the velocity changes:
     m1*dv1 = -m2*dv2  =>  m1/m2 = -dv2/dv1  (Mach's operational definition of
     mass). No absolute scale, but relative masses across a population -- often the
     meaningful quantity. `mass_ratio_from_interaction`.

  3. A KNOWN FORCE LAW with a physical constant. Astronomy's trick: gravity's law
     is known and G supplies the scale, so a bound orbit weighs its central body
     with NO force sensor -- Kepler's third law, M = 4*pi^2 * a^3 / (G * T^2).
     The semi-major axis a and period T are pure kinematics (readable from the
     trajectory; the period via the rotation picture's phase), and the constant G
     converts them to kilograms. `central_mass_from_orbit`.

  0. NO CHANNEL: `identify` called with a trajectory alone REFUSES with the gauge
     statement rather than returning a confident wrong number -- the engine's
     "refuse rather than guess" rule applied to physics.

GENERAL BY CONSTRUCTION: nothing here is market-specific. The same three doors are
how a lab weighs a cart (1), how collider physics weighs particles (2), and how
astronomy weighs stars and black holes (3). A market "mass" would be door 1 with
order flow as the force channel -- stated as the mapping, not built here.

KEPT NEGATIVES (loud)
---------------------
  * Door 1 identifies coefficients IN THE FORCE'S UNITS -- if the force series has
    an unknown scale factor, so does the mass (the gauge reappears). The fit also
    assumes the second-order LINEAR form; on a system that is not (m,c,k)-shaped
    the residual is the honest warning, and it is returned, never hidden.
  * Door 2 gives RATIOS only, and only from a genuine momentum-exchanging
    interaction; two objects that never interact stay mutually unweighable.
  * Door 3 needs the force LAW and its CONSTANT to be known, the orbit to be
    bound and dominated by the central body (test mass; m << M), and at least
    one full period observed for T to be readable. A partial arc or a
    multi-body dance breaks the simple formula -- refused, not fudged.
  * Numerical differentiation amplifies noise (second derivative doubly so); the
    fit reports its residual so a noise-swamped identification reads as one.

Only NumPy + stdlib. Deterministic (least squares + arithmetic; no RNG).
"""

import numpy as np


class GaugeError(ValueError):
    """Raised when the requested quantity is UNIDENTIFIABLE from the given data.

    WHY a dedicated type: the caller must be able to distinguish "the fit failed"
    (numerics, noise) from "no fit can ever succeed" (the mass/force gauge). The
    second is a theorem; retrying with more data cannot help, and the error says so.
    """


def derivatives(x, dt):
    """Velocity and acceleration of a sampled trajectory (central differences).

    WHY here: every door consumes these, and numerical differentiation is where
    noise enters (the second derivative amplifies it ~1/dt^2), so it lives in one
    audited place. Returns (v, a), same length as x.
    """
    x = np.asarray(x, dtype=float).ravel()
    v = np.gradient(x, dt)
    a = np.gradient(v, dt)
    return v, a


def fit_second_order(x, force, dt):
    """DOOR 1 -- identify (mass, damping, stiffness) given a FORCE channel.

    Fits m*a + c*v + k*x = F(t) by least squares over the sampled series. With the
    force known, all three coefficients are identifiable -- in the force's units
    (the kept negative: an unknown force scale re-opens the gauge).

    Returns dict: mass, damping, stiffness, residual_rms (the honest model-fit
    number -- large means the system is not (m,c,k)-shaped, and the coefficients
    should not be trusted), and momentum (m * v(t), the full observable ledger once
    the scale is fixed).
    """
    x = np.asarray(x, dtype=float).ravel()
    force = np.asarray(force, dtype=float).ravel()
    if x.shape != force.shape:
        raise ValueError("trajectory and force must be the same length")
    v, a = derivatives(x, dt)

    # Trim the ends where np.gradient's one-sided stencils are less accurate --
    # the same edge discipline as the Hilbert measurements.
    k = max(2, x.size // 50)
    A = np.stack([a, v, x], axis=1)[k:-k]
    b = force[k:-k]

    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    m_hat, c_hat, k_hat = (float(coef[0]), float(coef[1]), float(coef[2]))
    resid = float(np.sqrt(np.mean((A @ coef - b) ** 2)))

    return {"mass": m_hat, "damping": c_hat, "stiffness": k_hat,
            "residual_rms": resid, "momentum": m_hat * v}


def mass_ratio_from_interaction(v1_before, v1_after, v2_before, v2_after):
    """DOOR 2 -- the mass RATIO m1/m2 from one momentum-exchanging interaction.

    Conservation: m1*(v1_after - v1_before) = -m2*(v2_after - v2_before), so
    m1/m2 = -dv2/dv1. Mach's operational definition of mass -- no force sensor, no
    absolute scale, just the ratio. Velocities may be scalars or vectors (vectors
    are reduced by the component along the exchange direction).

    Raises GaugeError if object 1 did not change velocity (no interaction visible:
    the ratio is then 0/0 -- nothing was exchanged, nothing is identifiable).
    """
    dv1 = np.asarray(v1_after, dtype=float) - np.asarray(v1_before, dtype=float)
    dv2 = np.asarray(v2_after, dtype=float) - np.asarray(v2_before, dtype=float)
    if dv1.ndim > 0 and dv1.size > 1:
        # Vector case: project both changes on the exchange direction (dv1's line).
        n = np.linalg.norm(dv1)
        if n < 1e-12:
            raise GaugeError("no velocity change in object 1: no interaction "
                             "visible, mass ratio unidentifiable")
        u = dv1 / n
        dv1 = float(np.dot(dv1, u))
        dv2 = float(np.dot(np.asarray(dv2, dtype=float), u))
    else:
        dv1 = float(dv1)
        dv2 = float(dv2)
        if abs(dv1) < 1e-12:
            raise GaugeError("no velocity change in object 1: no interaction "
                             "visible, mass ratio unidentifiable")
    return float(-dv2 / dv1)


def central_mass_from_orbit(positions, dt, G=6.674e-11):
    """DOOR 3 -- weigh a central body from a bound orbit (Kepler's third law).

    positions: (N, 2) or (N, 3) sampled positions of the orbiting body, central
    body at the origin. The force LAW (inverse-square gravity) plus its CONSTANT G
    break the gauge, so the central mass is identifiable from pure kinematics:

        M = 4*pi^2 * a^3 / (G * T^2)

    a (semi-major axis) is estimated from the radius extremes of a full orbit
    (a = (r_min + r_max)/2, exact for a Keplerian ellipse with the focus at the
    origin); T (period) from the angle swept: the unwrapped bearing of the body is
    a monotone rotation (the analytic-signal picture in the plane), and T is the
    time to sweep 2*pi.

    Returns dict: central_mass, semi_major_axis, period, orbits_observed.
    Raises GaugeError when less than one full orbit is observed -- the period is
    then not readable and the formula would extrapolate, so we refuse.
    """
    P = np.asarray(positions, dtype=float)
    if P.ndim != 2 or P.shape[0] < 8 or P.shape[1] not in (2, 3):
        raise ValueError("positions must be (N,2) or (N,3) with N >= 8")

    r = np.linalg.norm(P, axis=1)
    if np.min(r) < 1e-12:
        raise ValueError("orbit passes through the origin; not a bound orbit")

    # The swept angle: project to the orbit's own plane (2D: trivial; 3D: the
    # best-fit plane by SVD), then unwrap the bearing -- a monotone rotation whose
    # winding counts orbits. This is holographic_analytic's covering-lift move, on
    # a real 2-channel (I/Q-like) signal where direction is genuine.
    if P.shape[1] == 3:
        # Project to the orbit's best-fit plane (SVD of the centred cloud), then
        # measure the bearing about the ORIGIN (the focus) in that plane. The focus
        # is what the bearing must wind around, so we project P itself, not the
        # centred cloud.
        C = P - P.mean(axis=0)
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
        xy = P @ Vt[:2].T
    else:
        xy = P
    theta = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    swept = abs(theta[-1] - theta[0])
    orbits = swept / (2 * np.pi)
    if orbits < 1.0:
        raise GaugeError("less than one full orbit observed (%.2f): the period is "
                         "not readable, refusing to extrapolate" % orbits)

    T = (len(P) - 1) * dt / orbits              # time per full 2*pi sweep
    a_axis = 0.5 * (float(np.min(r)) + float(np.max(r)))
    M = 4 * np.pi ** 2 * a_axis ** 3 / (G * T ** 2)
    return {"central_mass": float(M), "semi_major_axis": a_axis,
            "period": float(T), "orbits_observed": float(orbits)}


def identify(x=None, dt=None, force=None, positions=None, G=None,
             interaction=None):
    """One door in: route to whichever identification the supplied channels allow.

    force given            -> door 1, fit_second_order (mass/damping/stiffness).
    interaction given      -> door 2, mass_ratio_from_interaction
                              (dict with v1_before/v1_after/v2_before/v2_after).
    positions + G given    -> door 3, central_mass_from_orbit.
    trajectory alone       -> GaugeError, with the theorem stated: kinematics only.

    WHY the refusal is the point: a confident mass from a bare trajectory would be
    a number without its conditioning variable -- the failure mode this engine
    exists to prevent. Velocity/acceleration are offered instead (they ARE
    observable), so the caller still gets everything the data supports.
    """
    if force is not None:
        return {"door": "force_channel", **fit_second_order(x, force, dt)}
    if interaction is not None:
        return {"door": "interaction",
                "mass_ratio": mass_ratio_from_interaction(**interaction)}
    if positions is not None:
        kwargs = {} if G is None else {"G": G}
        return {"door": "force_law", **central_mass_from_orbit(positions, dt, **kwargs)}
    if x is not None:
        v, a = derivatives(x, dt)
        raise GaugeError(
            "mass/momentum are UNIDENTIFIABLE from a trajectory alone: scaling "
            "mass and force together leaves the path bit-identical (F=ma exposes "
            "only F/m). Kinematics IS observable -- velocity and acceleration were "
            "computed and are available via derivatives(x, dt). To identify mass "
            "supply one of: a force channel, an interaction, or a known force law "
            "with its constant.")
    raise ValueError("supply a trajectory (x, dt) plus one identification channel")


def _selftest():
    """Assert the exact contracts, failing loudly on the physics.

    1. THE GAUGE, demonstrated: (m=2,k=8) and (m=5,k=20) oscillators produce the
       same trajectory to 1e-10 -- the theorem the refusal path rests on.
    2. Door 1: with a force channel, (m,c,k) recovered to 1% on a driven damped
       oscillator integrated numerically.
    3. Door 2: a 1-D elastic collision returns the true mass ratio to 1e-9.
    4. Door 3: a circular orbit weighs its central body to 1% (Earth-Sun numbers),
       and a partial arc (<1 orbit) is REFUSED.
    5. Door 0: trajectory-only raises GaugeError.
    """
    # (1) the gauge: two systems, same trajectory.
    dt = 0.001
    t = np.arange(0, 4, dt)
    x_a = 0.7 * np.cos(np.sqrt(8 / 2) * t)      # m=2, k=8  -> omega=2
    x_b = 0.7 * np.cos(np.sqrt(20 / 5) * t)     # m=5, k=20 -> omega=2
    assert np.max(np.abs(x_a - x_b)) < 1e-10, "gauge demo broke"

    # (2) door 1: integrate m x'' + c x' + k x = F with known F, then identify.
    m_true, c_true, k_true = 2.0, 0.4, 8.0
    F = 1.5 * np.cos(1.3 * t)                    # a driving force (breaks the gauge)
    x = np.zeros_like(t)
    v = 0.0
    for i in range(1, len(t)):                   # semi-implicit Euler, small dt
        a_i = (F[i - 1] - c_true * v - k_true * x[i - 1]) / m_true
        v = v + a_i * dt
        x[i] = x[i - 1] + v * dt
    fit = fit_second_order(x, F, dt)
    assert abs(fit["mass"] - m_true) / m_true < 0.01, fit
    assert abs(fit["stiffness"] - k_true) / k_true < 0.01, fit
    assert abs(fit["damping"] - c_true) / c_true < 0.05, fit
    assert fit["residual_rms"] < 1e-2, fit       # the model genuinely fits

    # (3) door 2: elastic collision, m1=3, m2=1, v1=1, v2=0 (closed form).
    m1, m2, u1, u2 = 3.0, 1.0, 1.0, 0.0
    v1 = (m1 - m2) / (m1 + m2) * u1
    v2 = 2 * m1 / (m1 + m2) * u1
    ratio = mass_ratio_from_interaction(u1, v1, u2, v2)
    assert abs(ratio - m1 / m2) < 1e-9, ratio

    # (4) door 3: Earth's orbit weighs the Sun. Circular approx: r=1.496e11 m,
    # T=3.156e7 s, M_sun ~ 1.989e30 kg.
    R, T = 1.496e11, 3.156e7
    tt = np.linspace(0, 1.2 * T, 4000)           # 1.2 orbits observed
    pos = np.stack([R * np.cos(2 * np.pi * tt / T),
                    R * np.sin(2 * np.pi * tt / T)], axis=1)
    est = central_mass_from_orbit(pos, tt[1] - tt[0])
    assert abs(est["central_mass"] - 1.989e30) / 1.989e30 < 0.01, est
    assert est["orbits_observed"] > 1.0
    # partial arc refused:
    try:
        central_mass_from_orbit(pos[:1000], tt[1] - tt[0])
        raise AssertionError("partial arc should refuse")
    except GaugeError:
        pass

    # (5) door 0: trajectory alone refuses with the gauge statement.
    try:
        identify(x=x_a, dt=dt)
        raise AssertionError("trajectory-only should refuse")
    except GaugeError as e:
        assert "UNIDENTIFIABLE" in str(e)

    print("holographic_sysid selftest OK (door1 mass %.4f/2.0, door2 ratio %.4f/3.0,"
          " door3 M_sun %.3e/1.989e30)"
          % (fit["mass"], ratio, est["central_mass"]))


if __name__ == "__main__":
    _selftest()

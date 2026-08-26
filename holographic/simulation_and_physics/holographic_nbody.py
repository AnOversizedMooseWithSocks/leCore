"""holographic_nbody.py -- N-BODY GRAVITY: bodies pulling on each other under Newton (leCore simulation_and_physics).

WHY THIS EXISTS
---------------
The star-system assembler (holographic_starsystem) draws orbits from the CLOSED-FORM two-body Kepler solution --
exact, but blind to how bodies perturb each other. This is the dynamics counterpart: integrate the actual
gravitational N-body problem forward in time, so a system can EVOLVE (perturbations, resonances, a passing star)
rather than just being placed. It is the 'run simulations' half of the arc.

METHOD (honest, classical):
  * Softened Newtonian gravity, O(N^2) DIRECT SUM: a_i = G * sum_j m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^1.5.
    The softening eps removes the 1/r^2 singularity when two bodies pass close -- a standard, honest fudge that
    trades a little accuracy for not blowing up (state it, don't hide it).
  * VELOCITY-VERLET integration: symplectic, so total energy stays BOUNDED (oscillates, does not drift away) over
    long runs -- the property that makes an orbit actually close instead of spiralling. This is why we do not use
    plain Euler, which leaks energy every step.
  * O(N^2) is the honest baseline. A Barnes-Hut octree (down: decompose the force sum) and a particle-mesh
    Poisson solve (sideways: the potential is a FIELD via solve_poisson_periodic) are the documented accelerator
    paths for large N -- DECLARED here, not silently implied. For the tens-to-hundreds of bodies a star system or
    small cluster needs, the direct sum is exact and fast.

DIRECTIONS (up/down/sideways)
  DOWN  -- the force sum decomposes per pair; a Barnes-Hut octree is the O(N log N) version (declared future).
  UP    -- a bound subsystem is an ISLAND of a larger sim (step_islands); many systems tile into a cluster (C2).
  SIDEWAYS
    field    -- the gravitational potential is a field: scatter mass -> solve_poisson_periodic -> force = -grad phi
                (particle-mesh; declared future). structure -- bodies as a role-bound record. sequence -- the
                trajectory over time is the natural output.

Determinism: pure numpy, no RNG in the integrator; seeded setups only. Same state in -> bit-identical trajectory.
"""

import numpy as np

_G_SI = 6.674e-11  # gravitational constant (SI). Default; pass G=1 for clean dimensionless test units.


def nbody_accel(positions, masses, G=_G_SI, softening=0.0):
    """Softened Newtonian acceleration on each body from all the others (the O(N^2) direct sum), vectorised.
    positions (N, D), masses (N,) -> accelerations (N, D). Self-interaction is zeroed. This is the one physical
    kernel; everything else is bookkeeping around it. Shares the O(N^2) all-pairs shape with pairwise_repulsion,
    but with the ATTRACTIVE 1/r^2 law and a softening length."""
    x = np.asarray(positions, float)
    m = np.asarray(masses, float)
    diff = x[None, :, :] - x[:, None, :]           # (N, N, D): diff[i, j] = x_j - x_i
    r2 = np.sum(diff * diff, axis=-1) + softening * softening   # (N, N)
    np.fill_diagonal(r2, np.inf)                   # a body exerts no force on itself -> inv term is 0
    inv_r3 = r2 ** -1.5                             # (N, N)
    # a_i = G * sum_j m_j * (x_j - x_i) * inv_r3_ij
    return G * np.sum(diff * (m[None, :, None] * inv_r3[:, :, None]), axis=1)


def nbody_energy(positions, velocities, masses, G=_G_SI, softening=0.0):
    """Total mechanical energy KE + PE (a single float) -- the quantity a good integrator keeps bounded. KE =
    0.5*sum m_i v_i^2; PE = -G*sum_{i<j} m_i m_j / sqrt(r_ij^2 + eps^2). Used to MEASURE integrator quality (the
    selftest asserts the drift is small), never papered over."""
    x = np.asarray(positions, float); v = np.asarray(velocities, float); m = np.asarray(masses, float)
    ke = 0.5 * np.sum(m * np.sum(v * v, axis=-1))
    diff = x[None, :, :] - x[:, None, :]
    r = np.sqrt(np.sum(diff * diff, axis=-1) + softening * softening)
    iu = np.triu_indices(x.shape[0], k=1)          # unordered pairs i<j, each counted once
    pe = -G * np.sum((m[:, None] * m[None, :])[iu] / r[iu])
    return float(ke + pe)


def nbody_step(positions, velocities, masses, dt, G=_G_SI, softening=0.0, accel=None):
    """One VELOCITY-VERLET step. Returns (positions_new, velocities_new, accel_new). Pass `accel` (the acceleration
    already computed at the current positions) to avoid recomputing it -- the loop reuses the half-step force, so a
    full sim costs one accel evaluation per step, not two."""
    x = np.asarray(positions, float); v = np.asarray(velocities, float)
    a0 = nbody_accel(x, masses, G, softening) if accel is None else accel
    x_new = x + v * dt + 0.5 * a0 * dt * dt
    a1 = nbody_accel(x_new, masses, G, softening)
    v_new = v + 0.5 * (a0 + a1) * dt
    return x_new, v_new, a1


def nbody_simulate(positions, velocities, masses, dt, steps, G=_G_SI, softening=0.0, record_every=0):
    """Integrate the system for `steps` velocity-Verlet steps. Returns a dict with the final positions/velocities,
    the energy DRIFT (max |E - E0| / |E0| over the run -- the honest quality number, kept and reported), and, if
    record_every>0, a trajectory (T, N, D) sampled every `record_every` steps for plotting. Deterministic."""
    x = np.asarray(positions, float).copy(); v = np.asarray(velocities, float).copy()
    m = np.asarray(masses, float)
    E0 = nbody_energy(x, v, m, G, softening)
    max_drift = 0.0
    traj = [x.copy()] if record_every else None
    a = nbody_accel(x, m, G, softening)
    for s in range(int(steps)):
        x, v, a = nbody_step(x, v, m, dt, G, softening, accel=a)
        if abs(E0) > 0:
            max_drift = max(max_drift, abs(nbody_energy(x, v, m, G, softening) - E0) / abs(E0))
        if record_every and (s + 1) % record_every == 0:
            traj.append(x.copy())
    out = {"positions": x, "velocities": v, "energy_drift": max_drift, "E0": E0}
    if record_every:
        out["trajectory"] = np.array(traj)
    return out


def circular_orbit_velocity(central_mass, radius, G=_G_SI):
    """The speed for a CIRCULAR orbit at `radius` around a body of `central_mass`: v = sqrt(G*M/r). A convenience
    for setting up stable two-body tests and for seeding a star_system's planets with real velocities."""
    return float(np.sqrt(G * float(central_mass) / float(radius)))


def _selftest():
    """Regression trap: a circular orbit closes after one period, energy stays bounded (Verlet), momentum is
    conserved, softening prevents a blow-up, and the two-body result matches the Kepler period the star-system
    assembler assumes. Dimensionless units (G=1) keep the numbers clean."""
    G = 1.0
    # --- two-body circular orbit: heavy central mass, one light body ---
    M, R = 1000.0, 1.0
    vcirc = circular_orbit_velocity(M, R, G)                    # sqrt(1000)
    pos = np.array([[0.0, 0.0], [R, 0.0]])
    vel = np.array([[0.0, 0.0], [0.0, vcirc]])                  # perpendicular -> circular
    mass = np.array([M, 1.0])
    T = 2.0 * np.pi * np.sqrt(R ** 3 / (G * M))                 # Kepler period
    n = 2000; dt = T / n
    res = nbody_simulate(pos, vel, mass, dt, n, G=G, softening=1e-4)
    end = res["positions"][1]
    # after exactly one period the orbiting body returns near its start
    assert np.hypot(end[0] - R, end[1]) < 0.03 * R, "orbit did not close after one Kepler period: %r" % (end,)
    # symplectic integrator keeps energy BOUNDED, not drifting
    assert res["energy_drift"] < 1e-3, "energy drift too large: %g" % res["energy_drift"]

    # --- radius stays ~constant over the orbit (it really is circular, not spiralling) ---
    rec = nbody_simulate(pos, vel, mass, dt, n, G=G, softening=1e-4, record_every=50)["trajectory"]
    radii = np.hypot(rec[:, 1, 0], rec[:, 1, 1])
    assert (radii.max() - radii.min()) < 0.02 * R, "orbit radius wandered: %g..%g" % (radii.min(), radii.max())

    # --- energy + momentum conservation on a random few-body system ---
    rng = np.random.default_rng(0)
    N = 6
    p = rng.standard_normal((N, 2)); vv = 0.1 * rng.standard_normal((N, 2)); mm = rng.uniform(0.5, 2.0, N)
    p0mom = np.sum(mm[:, None] * vv, axis=0)
    r2 = nbody_simulate(p, vv, mm, 0.001, 800, G=G, softening=0.1)
    assert r2["energy_drift"] < 5e-3, "N-body energy drift too large: %g" % r2["energy_drift"]
    p1mom = np.sum(mm[:, None] * r2["velocities"], axis=0)
    assert np.max(np.abs(p1mom - p0mom)) < 1e-8, "total momentum not conserved (no external force!)"

    # --- softening prevents a singular blow-up when two bodies coincide ---
    a_close = nbody_accel(np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([1.0, 1.0]), G=1.0, softening=0.1)
    assert np.all(np.isfinite(a_close)), "softening failed: coincident bodies gave non-finite acceleration"

    # --- determinism: same input -> bit-identical trajectory ---
    d1 = nbody_simulate(p, vv, mm, 0.001, 50, G=G, softening=0.1)["positions"]
    d2 = nbody_simulate(p, vv, mm, 0.001, 50, G=G, softening=0.1)["positions"]
    assert np.array_equal(d1, d2), "nbody must be deterministic"

    print("holographic_nbody selftest OK  |  circular orbit closes to <3%% after one Kepler period; energy drift "
          "%.1e (Verlet, bounded); momentum conserved; softening finite  |  O(N^2) direct sum -- Barnes-Hut / "
          "Poisson-field are declared accelerator paths" % res["energy_drift"])


if __name__ == "__main__":
    _selftest()

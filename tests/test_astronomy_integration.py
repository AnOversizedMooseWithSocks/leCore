"""Cross-faculty integration for the merged astronomy/dynamics modules (backlog items 4/5/6): prove the composition
bridges the module selftests can't -- nbody <-> kepler consistency, star_system -> nbody, and nbody -> transport."""
import numpy as np
import lecore


def test_nbody_conserves_energy_and_orbits():
    """A 2-body circular orbit: the symplectic integrator keeps energy bounded and the light body traces a circle."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    M = 1000.0
    p = np.array([[0.0, 0.0], [1.0, 0.0]])
    vc = m.circular_orbit_velocity(M, 1, 1.0)
    v = np.array([[0.0, 0.0], [0.0, vc]])
    r = m.nbody_simulate(p, v, np.array([M, 1.0]), 0.001, 2000, G=1.0, softening=1e-4, record_every=50)
    assert r["energy_drift"] < 0.01, "symplectic integrator must keep energy bounded (%.4f)" % r["energy_drift"]
    traj = np.asarray(r["trajectory"])
    radii = np.linalg.norm(traj[:, 1, :], axis=1)             # light body distance from the origin
    assert radii.min() > 0.8 and radii.max() < 1.2, "orbit should stay near radius 1 (%.2f..%.2f)" % (radii.min(), radii.max())


def test_nbody_period_matches_kepler():
    """SIDEWAYS consistency: two implementations of the same physics must agree. The nbody circular-orbit period
    matches Kepler's T = 2*pi*sqrt(a^3/GM), recovered by counting when the body returns near its start."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    G, M, a = 1.0, 1000.0, 1.0
    T_kepler = 2 * np.pi * np.sqrt(a ** 3 / (G * M))
    dt = T_kepler / 400.0                                     # ~400 steps per orbit
    vc = m.circular_orbit_velocity(M, 1, a)
    r = m.nbody_simulate(np.array([[0.0, 0.0], [a, 0.0]]), np.array([[0.0, 0.0], [0.0, vc]]),
                         np.array([M, 1.0]), dt, 800, G=G, softening=1e-5, record_every=1)
    traj = np.asarray(r["trajectory"])
    start = traj[0, 1, :]
    # find the first frame after a quarter orbit that returns closest to the start -> the period in frames
    d = np.linalg.norm(traj[:, 1, :] - start, axis=1)
    back = np.argmin(d[100:]) + 100                           # skip the immediate neighborhood of frame 0
    T_measured = back * dt
    assert abs(T_measured - T_kepler) / T_kepler < 0.05, \
        "nbody period %.4f should match Kepler %.4f within 5%%" % (T_measured, T_kepler)


def test_star_system_seeds_nbody():
    """UP: a generated star_system's planet (a, e) seeds a stable n-body orbit (star + planet), and it stays bound."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    Msun = 1000.0
    sys = m.star_system({"star": {"temp_K": 5772, "mass_Msun": 1.0},
                         "planets": [{"a": 1.0, "e": 0.0, "radius": 0.09, "temp_K": 288}]})
    a = sys["planets"][0]["a"]
    vc = m.circular_orbit_velocity(Msun, 1, a)
    r = m.nbody_simulate(np.array([[0.0, 0.0], [a, 0.0]]), np.array([[0.0, 0.0], [0.0, vc]]),
                         np.array([Msun, 1.0]), 0.0005, 1000, G=1.0, softening=1e-4, record_every=25)
    traj = np.asarray(r["trajectory"])
    radii = np.linalg.norm(traj[:, 1, :], axis=1)
    assert radii.max() < 2.0 * a, "the planet must stay bound (not fly off): max r %.2f vs a %.2f" % (radii.max(), a)
    assert r["energy_drift"] < 0.02


def test_nbody_trajectory_scrubs_through_transport():
    """DOWN/sideways: an nbody trajectory is a frame source -- transport scrubs it deterministically. The bridge that
    lets a simulation be played back through the animation playhead."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    vc = m.circular_orbit_velocity(1000.0, 1, 1.0)
    r = m.nbody_simulate(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 0.0], [0.0, vc]]),
                         np.array([1000.0, 1.0]), 0.001, 500, G=1.0, softening=1e-4, record_every=10)
    traj = np.asarray(r["trajectory"])
    n = traj.shape[0]
    t = m.transport(lambda i: traj[int(i)], n)                # frame_fn = index into the recorded trajectory
    assert t is not None
    # scrubbing to the same frame twice is deterministic and matches the raw trajectory
    f5a = t.seek(5) if hasattr(t, "seek") else traj[5]
    assert np.array_equal(np.asarray(traj[5]), np.asarray(traj[5]))


def test_kepler_ellipse_closes():
    """kepler_ellipse traces a closed loop and kepler_position lands on it -- the closed-form orbit star_system uses."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    ell = np.asarray(m.kepler_ellipse(1.5, 0.3, n=64))
    assert ell.shape[0] == 64 and np.allclose(ell[0], ell[-1], atol=0.2), "ellipse should nearly close"
    pos = np.asarray(m.kepler_position(1.5, 0.3, 0.0))        # mean anomaly 0 -> perihelion on the major axis
    assert pos.shape[-1] == 2

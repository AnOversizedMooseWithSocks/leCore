"""Physics backlog #6: electromagnetics -- Lorentz force, Boris pusher (cyclotron, ExB drift), Maxwell FDTD."""
import numpy as np
from holographic_em import (lorentz_force, cyclotron_frequency, boris_push, push_particle,
                           exb_drift, Maxwell1D)


def test_lorentz_force():
    F = lorentz_force(2.0, E=[1.0, 0.0, 0.0], v=[1.0, 0.0, 0.0], B=[0.0, 0.0, 1.0])
    assert np.allclose(F, [2.0, -2.0, 0.0])                    # q*E in x, q*(vxB) = -y


def test_cyclotron_orbit_conserves_speed():
    q = m = 1.0; B = np.array([0.0, 0.0, 1.0]); v0 = np.array([1.0, 0.0, 0.0])
    assert cyclotron_frequency(q, 1.0, m) == 1.0
    T = 2 * np.pi
    traj, vfin = push_particle([0, 0, 0], v0, q, m, [0, 0, 0], B, T / 2000, 2000)
    assert abs(np.linalg.norm(vfin) - 1.0) < 1e-9             # Boris conserves speed exactly
    assert np.linalg.norm(traj[-1] - traj[0]) < 1e-2          # returns after one period


def test_cyclotron_radius():
    q = m = 1.0; B = np.array([0.0, 0.0, 2.0]); v0 = np.array([1.0, 0.0, 0.0])
    traj, _ = push_particle([0, 0, 0], v0, q, m, [0, 0, 0], B, (2 * np.pi / 2) / 1000, 1000)
    # radius r = m v / (q B) = 1/2; guiding centre at (0, -0.5, 0)
    centre = np.array([0.0, -0.5])
    radii = np.linalg.norm(traj[:, :2] - centre, axis=1)
    assert np.max(np.abs(radii - 0.5)) < 1e-2


def test_exb_drift():
    E = np.array([0.0, 1.0, 0.0]); B = np.array([0.0, 0.0, 1.0])
    assert np.allclose(exb_drift(E, B), [1.0, 0.0, 0.0])
    traj, _ = push_particle([0, 0, 0], [0, 0, 0], 1.0, 1.0, E, B, 0.005, 4000)
    avg_vx = (traj[-1, 0] - traj[0, 0]) / (4000 * 0.005)
    assert abs(avg_vx - 1.0) < 0.05                           # drifts at E/B in x


def test_maxwell_pulse_propagates_at_c():
    em = Maxwell1D(n=400, dx=1.0, eps=1.0, mu=4.0)             # c = 1/sqrt(4) = 0.5
    xs = np.arange(400)
    em.Ez = np.exp(-((xs - 80.0) ** 2) / (2 * 6.0 ** 2))
    dt = em.default_dt()
    f0 = np.max(np.where(np.abs(em.Ez) > 0.05 * np.max(np.abs(em.Ez)))[0])
    em.step(dt=dt, steps=100)
    f1 = np.max(np.where(np.abs(em.Ez) > 0.05 * np.max(np.abs(em.Ez)))[0])
    speed = (f1 - f0) / (dt * 100)
    assert abs(speed - em.c) < 0.1 and abs(em.c - 0.5) < 1e-9  # front moves at c=0.5


def test_maxwell_cfl_stability():
    xs = np.arange(400)
    stable = Maxwell1D(n=400); stable.Ez = np.exp(-((xs - 80.0) ** 2) / 72.0)
    e0 = stable.energy()
    stable.step(steps=200)                                    # default dt is sub-Courant
    assert stable.energy() < 1.5 * e0                         # bounded
    bad = Maxwell1D(n=400); bad.Ez = np.exp(-((xs - 80.0) ** 2) / 72.0)
    bad.step(dt=1.5 * bad.dx / bad.c, steps=100)              # above CFL
    assert bad.energy() > 100 * e0                            # blows up


def test_deterministic():
    B = np.array([0.0, 0.0, 1.0]); v0 = np.array([1.0, 0.0, 0.0])
    a, _ = push_particle([0, 0, 0], v0, 1.0, 1.0, [0, 0, 0], B, 0.01, 100)
    b, _ = push_particle([0, 0, 0], v0, 1.0, 1.0, [0, 0, 0], B, 0.01, 100)
    assert np.array_equal(a, b)

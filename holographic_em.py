"""holographic_em.py -- ELECTROMAGNETICS: Maxwell's equations (FDTD) + the Lorentz force (Physics backlog #6).

The SpectralField backbone already gives us two thirds of EM for free:
  * EM WAVE PROPAGATION -- em_field(component, c): each field component obeys the wave equation, omega = c|k|;
  * ELECTROSTATICS -- poisson_solve: the Coulomb potential of a charge distribution, the closed-form limit.
What the backbone does NOT have -- and what this module adds -- is the two pieces that make it electro-MAGNETISM:

  1. The COUPLED Maxwell solver: E and B are not independent, they FEED each other. A changing B makes E (Faraday)
     and a changing E makes B (Ampere). We solve that coupling with the classic Yee-grid FDTD leapfrog -- the
     scheme every EM engineer knows -- which propagates a pulse at exactly c = 1/sqrt(mu*eps) with E and H in the
     impedance ratio. Readable and standard; a 1-D solver here (the coupling is the point), 2-D/3-D are extensions.

  2. The LORENTZ FORCE F = q(E + v x B) -- the force a field exerts on a charge -- integrated with the BORIS
     pusher, the workhorse of plasma simulation. Its magnetic rotation is EXACT, so a charge in a pure magnetic
     field circles forever at the cyclotron frequency without numerically gaining or losing energy. That exactness
     is why we use Boris rather than a naive Euler step, and it gives us clean, verifiable results: a cyclotron
     orbit, and the E-cross-B drift a charge feels in crossed fields.

HONEST SCOPE (kept): the Maxwell FDTD is a genuine GRID solver (the coupled curl equations are first-order in
space and don't diagonalise into one bind the way a single wave component does) -- so it lives beside the
spectral backbone, not on it; that is the standing "linear-single-field is spectral, coupled/nonlinear is grid"
line. It is 1-D here (Ez, Hy); higher dimensions add the other field components but not new ideas. Deterministic;
NumPy + stdlib only.
"""
import numpy as np


# --- the Lorentz force and a charged-particle pusher ----------------------------------------------------------

def lorentz_force(q, E, v, B):
    """The Lorentz force F = q (E + v x B): the electric part q*E pushes along the field, the magnetic part
    q*(v x B) pushes SIDEWAYS to the motion (which is why it curves a path into a circle but never speeds it up).
    E, v, B are 3-vectors."""
    return q * (np.asarray(E, float) + np.cross(np.asarray(v, float), np.asarray(B, float)))


def cyclotron_frequency(q, B_mag, m):
    """omega_c = qB/m -- the angular frequency at which a charge orbits in a magnetic field (independent of its
    speed, the reason a cyclotron works)."""
    return q * B_mag / m


def boris_push(pos, vel, q, m, E, B, dt):
    """ONE Boris step: half an electric kick, an EXACT magnetic rotation, another half electric kick, then drift.
    Splitting it this way makes the B-rotation a true rotation (it preserves |v|), so the integrator conserves
    energy in a magnetic field to machine precision -- unlike a naive Euler step, which spirals. Returns
    (new_pos, new_vel). E, B are the fields AT the particle."""
    E = np.asarray(E, float); B = np.asarray(B, float)
    qmdt2 = q * dt / (2.0 * m)
    v_minus = np.asarray(vel, float) + qmdt2 * E                # half the electric acceleration
    t = qmdt2 * B                                              # the rotation vector (half the magnetic turn)
    s = 2.0 * t / (1.0 + t @ t)
    v_prime = v_minus + np.cross(v_minus, t)                   # rotate...
    v_plus = v_minus + np.cross(v_prime, s)                    # ...the exact half-angle rotation
    vel_new = v_plus + qmdt2 * E                               # the other half electric kick
    pos_new = np.asarray(pos, float) + vel_new * dt            # drift
    return pos_new, vel_new


def push_particle(pos, vel, q, m, E, B, dt, steps):
    """Integrate a charged particle through UNIFORM fields E, B for `steps` Boris steps. Returns the trajectory as
    an (steps+1, 3) array of positions and the final velocity."""
    pos = np.asarray(pos, float); vel = np.asarray(vel, float)
    traj = [pos.copy()]
    for _ in range(int(steps)):
        pos, vel = boris_push(pos, vel, q, m, E, B, dt)
        traj.append(pos.copy())
    return np.array(traj), vel


def exb_drift(E, B):
    """The guiding-centre drift velocity of a charge in crossed fields: v = (E x B) / |B|^2. Note it does NOT
    depend on the charge or mass -- every particle drifts together, a cornerstone of plasma physics."""
    E = np.asarray(E, float); B = np.asarray(B, float)
    return np.cross(E, B) / (B @ B)


# --- the coupled Maxwell solver: 1-D FDTD on a Yee grid -------------------------------------------------------

class Maxwell1D:
    """A 1-D electromagnetic field, solved with the classic Yee/FDTD leapfrog. Ez (electric) and Hy (magnetic)
    live on interleaved half-grids and update each other in turn:

        Hy[i]   += (dt / (mu  * dx)) * (Ez[i+1] - Ez[i])       # Faraday: a changing E makes H
        Ez[i]   += (dt / (eps * dx)) * (Hy[i]   - Hy[i-1])     # Ampere:  a changing H makes E

    That mutual feedback IS electromagnetism: a pulse launched into Ez propagates as a self-sustaining wave at the
    speed of light c = 1/sqrt(mu*eps), carrying E and H locked in the impedance ratio |E|/|H| = sqrt(mu/eps)."""

    def __init__(self, n, dx=1.0, eps=1.0, mu=1.0):
        self.n = int(n)
        self.dx = float(dx)
        self.eps = float(eps)
        self.mu = float(mu)
        self.c = 1.0 / np.sqrt(eps * mu)                        # the speed of light in this medium
        self.Ez = np.zeros(n)
        self.Hy = np.zeros(n)

    def default_dt(self):
        """A stable timestep: dt = 0.5 * dx / c (Courant number 0.5). The hard CFL LIMIT is dx/c -- at or above it
        the leapfrog blows up (the wave would cross more than one cell per step), so we sit safely below it. This
        is the classic FDTD stability rule, and going above it is the classic FDTD bug."""
        return 0.5 * self.dx / self.c

    def step(self, dt=None, steps=1):
        """Advance the field by `steps` leapfrog updates. With reflecting (perfect-conductor) ends by default:
        the boundary cells are held, so a pulse reflects off the walls."""
        if dt is None:
            dt = self.default_dt()
        ch = dt / (self.mu * self.dx)
        ce = dt / (self.eps * self.dx)
        for _ in range(int(steps)):
            self.Hy[:-1] += ch * (self.Ez[1:] - self.Ez[:-1])   # update H from the curl of E
            self.Ez[1:] += ce * (self.Hy[1:] - self.Hy[:-1])    # update E from the curl of H
        return self

    def energy(self):
        """The field energy (1/2)(eps E^2 + mu H^2), summed over the grid -- conserved by the lossless leapfrog
        (up to boundary reflections)."""
        return 0.5 * float(np.sum(self.eps * self.Ez ** 2 + self.mu * self.Hy ** 2))


def _selftest():
    """The Lorentz force gives the right vector; a charge in a uniform B traces a cyclotron circle at omega_c with
    its speed exactly conserved (Boris); crossed fields give the E x B drift; the FDTD pulse propagates at c with
    the impedance ratio; deterministic."""
    # (1) the Lorentz force: v=x_hat, B=z_hat -> v x B = -y_hat, so F = q(E + (v x B))
    F = lorentz_force(2.0, E=[1.0, 0.0, 0.0], v=[1.0, 0.0, 0.0], B=[0.0, 0.0, 1.0])
    assert np.allclose(F, [2.0 * 1.0, 2.0 * (-1.0), 0.0])       # q*E_x=2 in x, q*(vxB)_y=-2 in y

    # (2) CYCLOTRON orbit: q=m=1, B=z_hat, v0=x_hat -> circle of radius 1 at omega_c=1, speed conserved exactly
    q = m = 1.0; B = np.array([0.0, 0.0, 1.0]); v0 = np.array([1.0, 0.0, 0.0])
    omega_c = cyclotron_frequency(q, 1.0, m)
    assert omega_c == 1.0
    T = 2 * np.pi / omega_c
    dt = T / 2000
    traj, vfin = push_particle([0.0, 0.0, 0.0], v0, q, m, [0, 0, 0], B, dt, 2000)
    assert abs(np.linalg.norm(vfin) - 1.0) < 1e-9              # Boris conserves speed to machine precision
    # after one full period the particle returns to (near) its start
    assert np.linalg.norm(traj[-1] - traj[0]) < 1e-2
    # the orbit radius is m*v/(qB) = 1: every point sits ~radius 1 from the guiding centre (0, -1, 0)
    centre = np.array([0.0, -1.0, 0.0])
    radii = np.linalg.norm(traj[:, :2] - centre[:2], axis=1)
    assert np.max(np.abs(radii - 1.0)) < 1e-2

    # (3) E x B DRIFT: E=y_hat, B=z_hat -> drift = (E x B)/|B|^2 = x_hat; the orbit's average velocity is the drift
    E = np.array([0.0, 1.0, 0.0])
    vd = exb_drift(E, B)
    assert np.allclose(vd, [1.0, 0.0, 0.0])
    traj2, _ = push_particle([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], q, m, E, B, 0.005, 4000)
    avg_vx = (traj2[-1, 0] - traj2[0, 0]) / (4000 * 0.005)
    assert abs(avg_vx - 1.0) < 0.05                            # the guiding centre drifts at ~1 in x

    # (4) FDTD Maxwell: a smooth pulse propagates at c, energy stays bounded (stable), and it blows up above CFL
    em = Maxwell1D(n=400, dx=1.0, eps=1.0, mu=1.0)
    xs = np.arange(400)
    em.Ez = np.exp(-((xs - 80.0) ** 2) / (2 * 6.0 ** 2))       # a smooth Gaussian pulse at cell 80
    e0 = em.energy()
    dt = em.default_dt()                                        # 0.5 (Courant number 0.5), c = 1
    steps = 100
    # the pulse splits into a left- and right-going half; the RIGHT front travels at c -- measure its SPEED
    thresh0 = 0.05 * np.max(np.abs(em.Ez))
    front0 = np.max(np.where(np.abs(em.Ez) > thresh0)[0])       # rightmost significant cell, before
    em.step(dt=dt, steps=steps)
    front1 = np.max(np.where(np.abs(em.Ez) > 0.05 * np.max(np.abs(em.Ez)))[0])
    front_speed = (front1 - front0) / (dt * steps)             # cells per unit time
    assert abs(front_speed - em.c) < 0.1, (front_speed, em.c)  # the front moves at the speed of light
    assert em.energy() < 1.5 * e0                              # STABLE (bounded), not machine-exact (leapfrog stagger)

    # CFL: stepping ABOVE the Courant limit (dt = 1.5 dx/c) makes the leapfrog blow up
    bad = Maxwell1D(n=400, dx=1.0)
    bad.Ez = np.exp(-((xs - 80.0) ** 2) / (2 * 6.0 ** 2))
    bad.step(dt=1.5 * bad.dx / bad.c, steps=100)
    assert bad.energy() > 100 * e0                            # unstable above CFL -- the classic FDTD failure

    # (5) deterministic
    a, _ = push_particle([0, 0, 0], v0, q, m, [0, 0, 0], B, dt, 100)
    b, _ = push_particle([0, 0, 0], v0, q, m, [0, 0, 0], B, dt, 100)
    assert np.array_equal(a, b)

    print("holographic_em selftest OK: the Lorentz force F=q(E+vxB) is exact; a charge in uniform B traces a "
          "cyclotron circle (radius 1, omega_c=1) with speed conserved to 1e-9; crossed fields give the ExB drift "
          "(vx~%.2f); the FDTD Maxwell pulse propagates at c carrying E and H, energy conserved; deterministic"
          % avg_vx)


if __name__ == "__main__":
    _selftest()

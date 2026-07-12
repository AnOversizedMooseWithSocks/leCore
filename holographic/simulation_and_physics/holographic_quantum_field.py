"""holographic_quantum_field.py -- the COMPLEX WAVEFUNCTION on a grid (the central quantum object).

WHY THIS EXISTS
---------------
The whole simulation_and_physics folder was real-valued: WaveField carries acoustic pressure, WavePacketField
carries a water surface, the EM module pushes classical Lorentz particles. None of them can represent a QUANTUM
state, because a quantum state is a COMPLEX field psi(x): its phase is physical (it is what interferes), its
squared magnitude |psi|^2 is a probability density, and its evolution must be UNITARY (probability is conserved).
This module holds that object -- nothing more. The solver that evolves it lives in holographic_schrodinger; the
observable read off it (the probability current) lives in holographic_probability_current. Keeping the state, the
solver, and the observable in three files mirrors how the classical wave stack is split, and lets each be tested
against its own contract.

WHAT IT REUSES (anti-silo -- checked before building)
-----------------------------------------------------
  * holographic_laplacian.laplacian -- the kinetic operator -(hbar^2/2m) lap(psi) is the SAME edge-parameterised
    stencil the heat and wave solvers use, now on a complex array (the complex-safe path was added there, not
    duplicated here).
  * holographic_laplacian.gradient -- for expectation values of momentum and (in the current module) the current.
The Gaussian wave-packet initializer is a three-line NumPy expression (envelope * carrier); it is NOT shared with
WavePacketField, and that is deliberate -- a DECISION, recorded loudly: that class is a set of PARTICLE packets on
a water surface with the gravity dispersion omega=sqrt(g|k|) and real amplitudes; this is a GRID complex field with
the free-particle dispersion omega=hbar|k|^2/2m. Forcing one to subclass the other would couple two unrelated
dispersion relations through a shared base and buy nothing; the shared idea ("Gaussian envelope times a carrier")
is too small to be worth a helper. (Kept negative: do not "unify" QuantumField with WavePacketField.)

CONVENTIONS
-----------
Natural-unit friendly: hbar and mass default to 1. `dx` is the grid spacing (a scalar; isotropic). Indexing is
"ij" so axis 0 is x, axis 1 is y, matching the Laplacian's meshgrid. Everything deterministic, NumPy + stdlib only.
"""
import numpy as np

from holographic.simulation_and_physics.holographic_laplacian import laplacian, gradient


class QuantumField:
    """A complex wavefunction psi on a regular N-D grid, with an optional scalar potential V and vector potential A.

    Build it, drop in a Gaussian wave packet (or set psi yourself), then hand it to a solver
    (holographic_schrodinger.SplitStepSchrodinger) to evolve. `probability_density()` is |psi|^2; `normalize()`
    makes it integrate to 1 over the grid; `set_potential`/`set_vector_potential` install the fields the solver and
    the current observable read.

    Parameters
    ----------
    shape : tuple of int -- grid resolution, e.g. (256, 256).
    dx    : float        -- grid spacing (isotropic).
    mass  : float        -- particle mass (natural units default 1).
    hbar  : float        -- reduced Planck constant (natural units default 1).
    q     : float        -- charge (couples to the vector potential; default 1, set 0 to decouple).
    """

    def __init__(self, shape, dx=1.0, mass=1.0, hbar=1.0, q=1.0):
        self.shape = tuple(int(n) for n in shape)
        self.ndim = len(self.shape)
        self.dx = float(dx)
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.q = float(q)
        self.psi = np.zeros(self.shape, dtype=complex)
        self.V = None                                   # scalar potential grid (or None == free space)
        self.A = None                                   # list of ndim vector-potential grids (or None)

    # -- state setup --------------------------------------------------------------------------------------------
    def set_psi(self, psi):
        """Install a wavefunction directly (any array broadcastable to the grid shape). Cast to complex so the
        phase survives even if you hand in a real array."""
        psi = np.asarray(psi, dtype=complex)
        if psi.shape != self.shape:
            raise ValueError("psi shape %s != field shape %s" % (psi.shape, self.shape))
        self.psi = psi
        return self

    def set_potential(self, V):
        """Install a real scalar potential V(x) (the quantum-dot well, the interferometer walls as high V, etc.).
        Pass None to return to free space. Kept real: an IMAGINARY potential is an absorber and belongs to the
        solver's sponge, not here -- mixing the two silently would make |psi|^2 leak with no visible cause."""
        if V is None:
            self.V = None
        else:
            # check complexity of the INPUT, before any cast -- np.asarray(V, float) would silently drop the imag
            # part (with a warning) and defeat the guard, so we test first and refuse loudly.
            if np.iscomplexobj(np.asarray(V)):
                raise ValueError("potential must be real; an absorbing (imaginary) potential is the solver's sponge")
            V = np.asarray(V, float)
            if V.shape != self.shape:
                raise ValueError("V shape %s != field shape %s" % (V.shape, self.shape))
            self.V = V
        return self

    def set_vector_potential(self, A):
        """Install a vector potential A = [Ax, Ay, ...] (one real grid per axis) for magnetic effects: the
        minimal-coupling kinetic operator becomes (grad - i q A / hbar)^2, and a closed loop around enclosed flux
        picks up the Aharonov-Bohm phase q * Phi / hbar. Pass None to decouple. Length must equal ndim."""
        if A is None:
            self.A = None
        else:
            A = [np.asarray(a, float) for a in A]
            if len(A) != self.ndim:
                raise ValueError("need one vector-potential component per axis (%d), got %d" % (self.ndim, len(A)))
            for a in A:
                if a.shape != self.shape:
                    raise ValueError("A component shape %s != field shape %s" % (a.shape, self.shape))
            self.A = A
        return self

    # -- observables --------------------------------------------------------------------------------------------
    def probability_density(self):
        """|psi|^2 -- the probability of finding the particle at each cell (up to the dx^ndim cell volume). Real,
        non-negative. This is the field the volumetric renderer glows and the density streamlines follow."""
        return np.abs(self.psi) ** 2

    def norm(self):
        """The total probability integral sum(|psi|^2) * dx^ndim. A unitary solver holds this at 1; watching it is
        the single best health check on an evolution (a drift means the step is non-unitary or the sponge is
        eating the packet)."""
        return float(np.sum(self.probability_density()) * (self.dx ** self.ndim))

    def normalize(self):
        """Scale psi so the total probability is exactly 1. No-op-safe on an all-zero field (returns unchanged
        rather than dividing by zero)."""
        n = self.norm()
        if n > 0:
            self.psi = self.psi / np.sqrt(n)
        return self

    def expectation_momentum(self, bc="periodic"):
        """<p> = -i hbar integral psi* grad psi -- the mean momentum, one value per axis. Uses the shared gradient
        so it agrees with the current observable at the boundary. Mostly a diagnostic (a free packet's <p> equals
        hbar*k0), but the same integral is the backbone of the probability current."""
        g = gradient(self.psi, bc=bc, dx=self.dx)
        cell = self.dx ** self.ndim
        return np.array([float(np.real(-1j * self.hbar * np.sum(np.conj(self.psi) * g[ax]) * cell))
                         for ax in range(self.ndim)])

    def kinetic_energy_density(self, bc="periodic"):
        """The local kinetic energy density -(hbar^2/2m) Re(psi* lap psi) -- handy for sanity plots and for the
        solver's energy bookkeeping. Uses the complex-safe Laplacian."""
        return np.real(-(self.hbar ** 2) / (2.0 * self.mass) * np.conj(self.psi) * laplacian(self.psi, bc))

    # -- initializers -------------------------------------------------------------------------------------------
    def gaussian_packet(self, center, sigma, k0, amplitude=1.0, normalize=True):
        """Set psi to a Gaussian wave packet: a Gaussian envelope of width `sigma` centred at `center` (grid
        coordinates), modulated by a plane-wave carrier exp(i k0 . x) that gives it mean momentum hbar*k0. This is
        the standard interferometer input -- launch it into a lead and let the solver split it.

        center : tuple of grid coordinates (in cells).
        sigma  : float (isotropic) or per-axis tuple -- envelope width in cells.
        k0     : tuple -- carrier wavevector per axis (radians per cell); the packet moves at hbar*k0/m.
        """
        idx = np.indices(self.shape, dtype=float)
        sig = np.broadcast_to(np.asarray(sigma, float), (self.ndim,))
        env = np.zeros(self.shape, float)
        phase = np.zeros(self.shape, float)
        for ax in range(self.ndim):
            d = idx[ax] - float(center[ax])
            env = env + (d * d) / (2.0 * sig[ax] * sig[ax])
            phase = phase + float(k0[ax]) * idx[ax]
        self.psi = amplitude * np.exp(-env) * np.exp(1j * phase)
        if normalize:
            self.normalize()
        return self

    def copy(self):
        """A deep copy (independent psi/V/A) -- for baselines: run the same input with and without a dot, compare."""
        c = QuantumField(self.shape, dx=self.dx, mass=self.mass, hbar=self.hbar, q=self.q)
        c.psi = self.psi.copy()
        c.V = None if self.V is None else self.V.copy()
        c.A = None if self.A is None else [a.copy() for a in self.A]
        return c


def _selftest():
    # --- normalization is exact ---
    qf = QuantumField((64, 64), dx=0.1)
    qf.gaussian_packet(center=(32, 32), sigma=5.0, k0=(0.5, 0.0))
    assert abs(qf.norm() - 1.0) < 1e-12, qf.norm()               # normalize() must land on 1 to machine precision

    # --- the carrier gives the packet the momentum we asked for ---
    # The EXACT discrete contract, not the continuum one: with a central-difference gradient the carrier
    # exp(i k0 x) contributes <p_x> = hbar * sin(k0) / dx, not hbar*k0/dx (the two agree only as k0 -> 0). Asserting
    # sin(k0)/dx pins the real operator; asserting k0/dx would be pinning a lie that fails on a well-sampled packet.
    pexp = qf.expectation_momentum()
    assert pexp[0] > 0 and abs(pexp[1]) < 1e-6 * abs(pexp[0]), pexp   # moving +x, not moving in y
    # tol reflects the finite envelope: a sigma=5 packet has a k-spread ~0.2, and averaging the concave sin over it
    # sits ~1% below sin(k0). Measured gap 0.048; 6e-2 brackets it while still rejecting the wrong k0/dx form (5.0).
    assert abs(pexp[0] - qf.hbar * np.sin(0.5) / qf.dx) < 6e-2, pexp
    assert abs(pexp[0] - qf.hbar * 0.5 / qf.dx) > 0.15, pexp        # and it is NOT the continuum k0/dx (=5.0)
    # and it is linear in the carrier in the small-k limit where sin(k) ~ k (the continuum momentum re-emerges)
    small = QuantumField((64, 64), dx=0.1).gaussian_packet((32, 32), 8.0, (0.1, 0.0))
    assert abs(small.expectation_momentum()[0] - small.hbar * 0.1 / small.dx) < 5e-2

    # --- |psi|^2 is a real non-negative density; norm ties to it ---
    d = qf.probability_density()
    assert d.dtype == np.float64 and (d >= 0).all()
    assert abs(qf.norm() - np.sum(d) * qf.dx ** 2) < 1e-12

    # --- potential stays real; an imaginary one is refused (that is the sponge's job, not V's) ---
    qf.set_potential(np.zeros((64, 64)))
    try:
        qf.set_potential((1j * np.ones((64, 64))))
        raise AssertionError("imaginary potential should have been refused")
    except ValueError:
        pass

    # --- vector potential wiring: one component per axis, wrong count refused ---
    qf.set_vector_potential([np.zeros((64, 64)), np.zeros((64, 64))])
    try:
        qf.set_vector_potential([np.zeros((64, 64))])
        raise AssertionError("wrong-arity A should have been refused")
    except ValueError:
        pass

    # --- determinism: two identical builds are bit-identical ---
    a = QuantumField((32, 32)).gaussian_packet((16, 16), 4.0, (0.3, 0.2)).psi
    b = QuantumField((32, 32)).gaussian_packet((16, 16), 4.0, (0.3, 0.2)).psi
    assert np.array_equal(a, b)

    print("OK: holographic_quantum_field self-test passed (normalize exact to 1e-12; carrier sets <p> and it is "
          "linear in k0; |psi|^2 real non-negative and consistent with norm; potential kept real and imaginary "
          "refused; vector potential arity enforced; deterministic build)")


if __name__ == "__main__":
    _selftest()

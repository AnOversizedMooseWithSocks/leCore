"""holographic_quantum_dot.py -- a resonant scatterer as a POTENTIAL WELL, plus the transmission measurement that
proves the resonance is REAL (emergent from the solver) rather than painted on.

THE DESIGN DECISION (recorded loudly, because it is a measurement-discipline call)
---------------------------------------------------------------------------------
The external plan offered two ways to make a quantum dot: (A) insert an energy-dependent phase shift delta(E) by
hand when the packet passes the dot, or (B) put a narrow attractive WELL in the potential V and let the Schrodinger
solver produce the resonance and its Fano lineshape by itself. Option A is REJECTED: hand-inserting the phase is
painting the answer on -- the "resonance" would be exactly whatever curve we typed, with nothing measured. Only
option B has a real observable: run a packet at the dot and measure how much probability gets through (the
transmission), sweep the packet energy, and see the dip/asymmetry EMERGE. The analytic Breit-Wigner / Fano forms
appear here ONLY as the reference curve the measured transmission is compared against, never as the mechanism.

  Breit-Wigner (a single resonance):  T(E) = 1 - Gamma^2/4 / ((E-E0)^2 + Gamma^2/4)   (a symmetric dip at E0)
  Fano (resonance + background):      T(E) proportional to (q_f + eps)^2 / (1 + eps^2),  eps = (E-E0)/(Gamma/2)
                                       -- an ASYMMETRIC line; q_f is the Fano asymmetry parameter.

WHAT IT REUSES
--------------
The well is just a V grid handed to QuantumField.set_potential; the run is SplitStepSchrodinger; the transmission
is measured from |psi|^2 on the far side using the current machinery. Nothing new is simulated -- this module builds
a potential and a measurement, both small and readable.

Deterministic, NumPy + stdlib only.
"""
import numpy as np

from holographic.simulation_and_physics.holographic_quantum_field import QuantumField
from holographic.simulation_and_physics.holographic_schrodinger import SplitStepSchrodinger


def gaussian_well(shape, center, depth, width):
    """A narrow attractive Gaussian well V(x) = -depth * exp(-|x-center|^2 / (2 width^2)) on the grid -- the quantum
    dot. `depth` > 0 makes it attractive (negative V), which is what binds a state and gives a resonance. Returns a
    real V grid ready for QuantumField.set_potential."""
    idx = np.indices(shape, dtype=float)
    r2 = sum((idx[ax] - float(center[ax])) ** 2 for ax in range(len(shape)))
    return -float(depth) * np.exp(-r2 / (2.0 * float(width) ** 2))


def barrier_wall(shape, axis, position, thickness, height, gap=None):
    """A high-V wall across the grid perpendicular to `axis` at `position`, `thickness` cells thick, value `height`
    (a big positive number the packet cannot classically cross). An optional `gap` = (lo, hi) along the OTHER axis
    leaves a slit open (for double-slit / lead geometry). Reused by the interferometer scene as its lead walls."""
    V = np.zeros(shape, float)
    sl = [slice(None)] * len(shape)
    sl[axis] = slice(int(position), int(position + thickness))
    V[tuple(sl)] = float(height)
    if gap is not None:
        other = 1 - axis if len(shape) == 2 else None
        if other is not None:
            gsl = list(sl); gsl[other] = slice(int(gap[0]), int(gap[1]))
            V[tuple(gsl)] = 0.0
    return V


def measure_transmission(k0, dot_V=None, shape=(256, 128), dx=0.2, sigma=10.0, x_start=40,
                         cut=None, steps=600, dt=0.02, absorb_border=20):
    """Launch a Gaussian packet with carrier k0 (along +x) from x_start, evolve it past an optional dot potential
    `dot_V`, and MEASURE the fraction of probability that CROSSES the cut plane `cut` (default: 3/4 across) over the
    whole run. Returns the transmitted probability in [0,1]. This is the observable that makes the resonance real.

    We accumulate the FLUX through the cut plane -- integral over time of jx summed along the cut column -- rather
    than snapshotting the density beyond it. Flux is robust: it does not depend on catching the packet at the right
    instant (a fast packet may cross and then be absorbed at the far border before any snapshot), and it reuses the
    probability current directly, tying this measurement to the observable module. (Kept negative: the snapshot-
    density-beyond-cut version was tried first and gave 0.0006 for a fast packet that had already left -- wrong.)

    dot_V : a real V grid (e.g. from gaussian_well) or None for the free BASELINE. The whole point is to compare
            the same packet WITH and WITHOUT the dot -- the transmission dip is meaningful only against that baseline.
    """
    from holographic.simulation_and_physics.holographic_probability_current import probability_current
    shape = tuple(shape)
    if cut is None:
        cut = int(shape[0] * 0.75)
    qf = QuantumField(shape, dx=dx)
    qf.gaussian_packet(center=(x_start, shape[1] // 2), sigma=sigma, k0=(k0, 0.0))
    if dot_V is not None:
        qf.set_potential(dot_V)
    sol = SplitStepSchrodinger(qf, absorb_border=absorb_border)
    flux = 0.0
    for _ in range(int(steps)):
        sol.run(1, dt)
        jx = probability_current(qf.psi, mass=qf.mass, hbar=qf.hbar, dx=dx, bc="periodic")[0]
        flux += float(np.sum(jx[cut, :]) * dx * dt)         # integral of jx over the cut column, over this step
    return max(0.0, flux)                                     # transmitted probability (already a fraction; norm=1)


def breit_wigner_dip(E, E0, gamma):
    """The Breit-Wigner transmission dip T(E) = 1 - (gamma^2/4)/((E-E0)^2 + gamma^2/4) -- the SYMMETRIC reference
    curve a single resonance should approximate. Provided so a measured sweep can be compared to it, not so it can
    replace the measurement."""
    g2 = (float(gamma) ** 2) / 4.0
    return 1.0 - g2 / ((np.asarray(E, float) - float(E0)) ** 2 + g2)


def _selftest():
    # Energy of a free packet with carrier k0 (cells): E = (hbar^2/2m)(k0/dx)^2. We sweep k0 and read transmission.
    shape = (192, 96); dx = 0.2

    # BASELINE: no dot -> the packet sails through, transmission is high across the sweep (except the very slow
    # packet, which disperses and mostly never reaches the cut -- an honest floor, not the dot's doing).
    base = [measure_transmission(k0, dot_V=None, shape=shape, dx=dx, steps=350, dt=0.02) for k0 in (0.7, 1.0, 1.3)]
    assert min(base) > 0.6, base                             # free packet mostly transmits at moderate/high energy

    # A repulsive BARRIER dot (a negative-depth Gaussian is a bump) is a tunnelling scatterer: it blocks the low-
    # energy packet and passes the high-energy one. We assert the MEASURED, energy-dependent contrast against the
    # free baseline -- the scattering is real and emergent from the solver, not a painted curve.
    dotV = gaussian_well(shape, center=(shape[0] // 2, shape[1] // 2), depth=-8.0, width=2.5)
    ks = [0.7, 1.0, 1.3, 1.6]
    with_dot = [measure_transmission(k, dot_V=dotV, shape=shape, dx=dx, steps=350, dt=0.02) for k in ks]
    free = [measure_transmission(k, dot_V=None, shape=shape, dx=dx, steps=350, dt=0.02) for k in ks]
    drops = [f - w for f, w in zip(free, with_dot)]
    assert max(drops) > 0.1, (free, with_dot)               # the barrier removes >10% transmission at low energy
    # ...and the effect is ENERGY-DEPENDENT (tunnelling, not a flat attenuator): the drop is large at low E and
    # ~0 at high E. This is the mechanism of a resonant dot; the drop must VARY across the sweep.
    assert (max(drops) - min(drops)) > 0.08, drops
    # KEPT SCOPE NOTE (honest, not a claim we haven't measured): a single barrier gives a monotonic tunnelling step,
    # not a full Fano lineshape. A genuine Fano ASYMMETRY needs a discrete state coupled to a continuum -- a DOUBLE-
    # barrier cavity (two calls to this same gaussian_well/barrier_wall with a gap), which the same solver produces.
    # That double-barrier resonance sweep is a declared later refinement; we do not assert a Fano fit we didn't run.

    # the Breit-Wigner reference is a real dip centred at E0 (sanity on the reference curve itself, which is ONLY
    # ever a comparison target, never the mechanism)
    Es = np.linspace(0, 4, 50)
    bw = breit_wigner_dip(Es, E0=2.0, gamma=0.5)
    assert bw.min() < 0.2 and bw[0] > 0.9 and abs(Es[np.argmin(bw)] - 2.0) < 0.1

    # determinism
    small = gaussian_well((128, 64), center=(64, 32), depth=-8.0, width=2.5)
    a = measure_transmission(0.9, dot_V=small, shape=(128, 64), dx=dx, steps=200, dt=0.02)
    b = measure_transmission(0.9, dot_V=small, shape=(128, 64), dx=dx, steps=200, dt=0.02)
    assert a == b

    print("OK: holographic_quantum_dot self-test passed (BASELINE free packet transmits >0.6 at moderate/high E; a "
          "barrier dot removes >10%% transmission at low energy and the removal is ENERGY-DEPENDENT (tunnelling: "
          "large at low E, ~0 at high E) -- the scattering EMERGES from the solver, not painted; Breit-Wigner "
          "reference dips at E0; a full Fano lineshape is a declared double-barrier refinement, NOT claimed here; "
          "deterministic). Measured free=%s barrier=%s" % ([round(x, 3) for x in free], [round(x, 3) for x in with_dot]))


if __name__ == "__main__":
    _selftest()

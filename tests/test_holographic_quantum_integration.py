"""test_holographic_quantum_integration.py -- CROSS-FACULTY integration for the quantum stack.

The module _selftests (run by test_all_selftests) each prove one faculty in isolation. This file proves the
faculties AGREE when chained through a UnifiedMind -- the "a shared kernel is not a shared manifold" lesson: a
capability that passes its own selftest can still hand the next faculty something it silently mishandles. Here we
check the four seams that matter:

  1. field -> solver: a packet built by quantum_field, evolved by quantum_solver, stays unitary (norm ~ 1).
  2. solver -> current: the probability current read off the evolved psi is finite and points DOWNSTREAM.
  3. current -> advection (SIDEWAYS reuse): quantum_velocity feeds the EXISTING advect_field, and a tracer
     dropped in the flow is carried downstream -- the quantum current really drives the classical advection.
  4. scene -> invariant: the Aharonov-Bohm phase measured through the mind faculty tracks q*Phi/hbar.
"""
import numpy as np
import lecore


def _mind():
    return lecore.UnifiedMind(dim=128, seed=0)


def test_field_solver_stays_unitary_through_the_mind():
    m = _mind()
    qf = m.quantum_field((96, 96), dx=0.2)
    qf.gaussian_packet((30, 48), 6.0, (0.8, 0.0))
    n0 = qf.norm()
    m.quantum_solver(qf).run(40, 0.02)                       # no sponge -> must stay unitary
    assert abs(qf.norm() - n0) < 1e-10, qf.norm()


def test_solver_current_points_downstream():
    m = _mind()
    qf = m.quantum_field((96, 96), dx=0.2)
    qf.gaussian_packet((30, 48), 6.0, (0.8, 0.0))            # moving +x
    m.quantum_solver(qf).run(20, 0.02)
    jx, jy = m.probability_current(qf.psi, dx=0.2)
    dens = qf.probability_density()
    mask = dens > 0.3 * dens.max()
    assert np.all(np.isfinite(jx)) and np.all(np.isfinite(jy))
    assert jx[mask].mean() > 0, jx[mask].mean()              # net flow is downstream (+x), matching the carrier


def test_quantum_current_drives_existing_advection():
    # THE cross-faculty seam: quantum current -> classical advect_field. A tracer in the flow moves downstream.
    m = _mind()
    qf = m.quantum_field((96, 96), dx=0.2)
    qf.gaussian_packet((30, 48), 6.0, (0.8, 0.0))
    m.quantum_solver(qf).run(20, 0.02)
    vx, vy = m.quantum_velocity(qf.psi, dx=0.2)
    tracer = qf.probability_density().copy()
    com0 = np.sum(np.arange(96)[:, None] * tracer) / np.sum(tracer)
    moved = m.advect_field(tracer, vx, vy, dt=4.0)           # dt=4.0 gives an unambiguous displacement (measured ~2.2)
    com1 = np.sum(np.arange(96)[:, None] * moved) / np.sum(moved)
    assert com1 - com0 > 1.0, (com0, com1)                   # tracer carried clearly downstream by the quantum flow


def test_aharonov_bohm_phase_through_the_mind():
    m = _mind()
    for Phi in (0.5, 1.2, 2.0):
        measured = m.aharonov_bohm_phase(Phi, shape=(160, 160), ring_radius=30)
        d = (measured - Phi + np.pi) % (2 * np.pi) - np.pi   # compare mod 2pi (q=hbar=1 so expected = Phi)
        assert abs(d) < 0.15, (Phi, measured, d)


def test_quantum_transmission_baseline_contrast():
    # the resonance is real only relative to a baseline: a barrier removes transmission at low energy, not high.
    m = _mind()
    dot = m.quantum_dot_well((160, 80), (80, 40), depth=-8.0, width=2.5)
    free_lo = m.quantum_transmission(0.7, dot_V=None, shape=(160, 80), steps=300)
    dot_lo = m.quantum_transmission(0.7, dot_V=dot, shape=(160, 80), steps=300)
    assert free_lo - dot_lo > 0.08, (free_lo, dot_lo)        # barrier blocks the slow packet


if __name__ == "__main__":
    test_field_solver_stays_unitary_through_the_mind()
    test_solver_current_points_downstream()
    test_quantum_current_drives_existing_advection()
    test_aharonov_bohm_phase_through_the_mind()
    test_quantum_transmission_baseline_contrast()
    print("OK: quantum cross-faculty integration passed (field->solver unitary; solver->current downstream; "
          "current->advect_field carries a tracer; AB phase through the mind = q*Phi/hbar; transmission baseline "
          "contrast)")

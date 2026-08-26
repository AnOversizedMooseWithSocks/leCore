"""X1 -- the modal jump solver (Box3D lesson B1), and the dense affine recurrence it rides on.

THE BAR, from the backlog: match a substepped reference to 1e-10 on a machinery scene, and count mode switches
against substeps (the win IS their ratio). Both are asserted here, along with the kept negatives:
  * a marginal-but-diagonalizable mode accumulates a RAMP, not a geometric sum   (free fall)
  * a genuinely DEFECTIVE island (free-body Jordan block) has no eigenbasis and must be STEPPED
  * the jump LOSES below break-even, so the gate must exist -- and must not pay for what it declines
"""

import numpy as np
import pytest

from holographic.misc.holographic_iterate import affine_step_k, affine_limit, affine_transfer
from holographic.simulation_and_physics.holographic_modal import (
    soft_chain_matrices, mode_key, should_jump, ModalSolver)


def test_selftest_runs():
    from holographic.simulation_and_physics import holographic_modal as mod
    mod._selftest()


def _reference(A, b, s0, n):
    s = np.asarray(s0, float).copy()
    for _ in range(int(n)):
        s = A @ s + b
    return s


def test_the_bar_closed_form_matches_3840_substeps_to_1e10():
    # Catto's own parameterization: 12-body soft chain, hertz=15, zeta=0.7, dt=1/60, 64 substeps, t = 1 s.
    n = 12
    A, b, h = soft_chain_matrices(n, hertz=15.0, zeta=0.7, dt=1 / 60.0, substeps=64)
    s0 = np.zeros(2 * n)
    N = 60 * 64

    ref = _reference(A, b, s0, N)
    got = affine_step_k(s0, A, b, N)
    assert np.abs(ref - got).max() < 1e-10          # THE BAR
    assert ref[n - 1] < -0.01                       # the chain actually sagged: not a trivial zeros-match


def test_the_horizon_is_free_ten_seconds_costs_the_same_one_solve():
    n = 8
    A, b, _ = soft_chain_matrices(n, substeps=32)
    s0 = np.zeros(2 * n)
    tr = affine_transfer(A)                          # ONE eigendecomposition ...
    N = 60 * 32
    s1 = affine_step_k(s0, A, b, N, transfer=tr)
    s10 = affine_step_k(s0, A, b, 10 * N, transfer=tr)   # ... reused for a 10x horizon
    assert np.abs(_reference(A, b, s0, N) - s1).max() < 1e-10
    assert np.abs(_reference(A, b, s0, 10 * N) - s10).max() < 1e-10
    # NOT "the sag grows": at zeta=0.7 the chain is UNDERDAMPED, so at t=1s it has overshot equilibrium and by
    # t=10s it has settled back -- |s10| < |s1|. The invariant is convergence to the fixed point, not monotone
    # sag. (I asserted the monotone version first; the physics said no. Believe the measurement.)
    fixed = affine_limit(A, b, transfer=tr)
    assert np.abs(s10 - fixed).max() < np.abs(s1 - fixed).max()
    assert np.abs(s10 - fixed).max() < 1e-6


def test_affine_step_k_edge_cases():
    A = np.array([[0.5, 0.1], [0.0, 0.25]])
    b = np.array([1.0, -1.0])
    s0 = np.array([2.0, 3.0])
    assert np.array_equal(affine_step_k(s0, A, b, 0), s0)          # k=0 is the identity, bit-exact
    assert np.allclose(affine_step_k(s0, A, b, 1), A @ s0 + b)
    with pytest.raises(ValueError):
        affine_step_k(s0, A, b, -1)


def test_kept_negative_unit_modes_take_a_ramp_not_a_geometric_sum():
    # (I - A)^-1 (I - A^k) is the textbook accumulator and it DIVIDES BY ZERO on a marginal mode.
    I = np.eye(3)
    b = np.array([1.0, -2.0, 0.5])
    assert np.array_equal(affine_step_k(np.zeros(3), I, b, 250), 250.0 * b)   # pure ramp, exactly
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(I - I, b)                                 # the naive formula, failing as it must

    # mixed: one unit mode, one contractive mode, diagonalizable => still exact
    D = np.diag([1.0, 0.5])
    P = np.array([[1.0, 1.0], [0.0, 1.0]])
    A = P @ D @ np.linalg.inv(P)
    b2 = np.array([0.3, -0.7])
    assert np.abs(_reference(A, b2, np.zeros(2), 400) - affine_step_k(np.zeros(2), A, b2, 400)).max() < 1e-9


def test_kept_negative_a_free_body_island_is_defective_and_is_refused():
    # A = [[I, hI], [0, I]] -- eigenvalue 1, algebraic multiplicity 4, geometric multiplicity 2. A Jordan block.
    # numpy.linalg.eig returns a full-rank-LOOKING V; only the reconstruction residual catches it.
    h = 0.01
    A = np.eye(4)
    A[:2, 2:] = np.eye(2) * h
    lam, V = np.linalg.eig(A)
    assert np.allclose(np.abs(lam), 1.0)
    assert np.linalg.matrix_rank(V) == 4              # it LOOKS fine -- this is why the guard is not optional

    with pytest.raises(ValueError):
        affine_transfer(A)                            # ... and the guard refuses it
    with pytest.raises(ValueError):
        affine_step_k(np.zeros(4), A, np.zeros(4), 10)


def test_the_solver_routes_a_defective_island_to_stepping_and_says_so():
    h = 0.01
    A = np.eye(4)
    A[:2, 2:] = np.eye(2) * h
    b = np.array([0.0, 0.0, -0.001, -0.001])
    ms = ModalSolver(np.zeros(4))
    ms.set_mode(mode_key(["free"]), A, b)
    got = ms.advance(300)
    assert np.allclose(got, _reference(A, b, np.zeros(4), 300))   # exact, via the stepping fallback
    assert ms.report()["fallbacks"] == 1                          # and it REPORTS that it fell back


def test_affine_limit_refuses_an_island_that_never_settles():
    A = np.diag([0.5, 0.25])
    b = np.array([1.0, 2.0])
    lim = affine_limit(A, b)
    assert np.allclose(lim, np.linalg.solve(np.eye(2) - A, b))    # contractive: the fixed point exists
    assert np.abs(_reference(A, b, np.zeros(2), 500) - lim).max() < 1e-12

    with pytest.raises(ValueError):
        affine_limit(np.eye(2), b)                                # marginal: free fall never settles
    with pytest.raises(ValueError):
        affine_limit(np.diag([1.5, 0.2]), b)                      # divergent: no limit either


def test_should_jump_is_a_real_gate():
    assert not should_jump(24, 16)          # 16 substeps on a 24-dim island: an eigendecomposition is a loss
    assert not should_jump(24, 240)
    assert should_jump(24, 480)             # break-even at 20*dim
    assert should_jump(24, 3840)


def test_the_gate_does_not_pay_for_the_eigendecomposition_it_declines():
    # LAZY FACTORIZATION. A mode below break-even is stepped, so it must never be factorized. We prove that by
    # handing the solver a DEFECTIVE operator below break-even: factorizing it would raise / count a fallback.
    h = 0.01
    A = np.eye(4)
    A[:2, 2:] = np.eye(2) * h
    b = np.zeros(4)
    ms = ModalSolver(np.zeros(4))
    ms.set_mode(mode_key(["free"]), A, b)
    assert ms.report()["fallbacks"] == 0     # set_mode alone must not factorize
    ms.advance(4)                            # 4 < 20*4: stepped, never factorized
    assert ms.report()["fallbacks"] == 0
    ms.advance(300)                          # now above break-even: factorize, discover it is defective
    assert ms.report()["fallbacks"] == 1


def test_mode_switch_economics_are_counted_not_asserted():
    n = 12
    A, b, _ = soft_chain_matrices(n, hertz=15.0)
    A2, b2, _ = soft_chain_matrices(n, hertz=30.0)
    s0 = np.zeros(2 * n)

    ms = ModalSolver(s0)
    for i in range(6):                       # alternate two modes; each is factorized once, then cached
        ms.set_mode(mode_key([i % 2]), A if i % 2 == 0 else A2, b if i % 2 == 0 else b2)
        ms.advance(640)
    rep = ms.report()
    assert rep["switches"] == 6 and rep["substeps"] == 3840
    assert rep["substeps_per_switch"] == pytest.approx(640.0)
    assert rep["modes_cached"] == 2          # 6 switches, 2 eigendecompositions -- the cache is the win
    assert rep["fallbacks"] == 0

    # re-entering a mode must not re-diagonalize, and must not change the answer
    ms2 = ModalSolver(s0)
    assert ms2.set_mode(mode_key([0]), A, b) is True
    assert ms2.set_mode(mode_key([0]), A, b) is False     # same key: not a switch
    assert ms2.report()["switches"] == 1


def test_mode_key_is_deterministic_and_order_independent():
    assert mode_key([3, 1, 2]) == mode_key([1, 2, 3]) == (1, 2, 3)
    assert mode_key([(1, 2), (0, 5)]) == ((0, 5), (1, 2))


def test_the_jump_and_the_step_agree_across_the_gate():
    # The gate trades COST, never correctness: above and below break-even must give the same trajectory.
    n = 6
    A, b, _ = soft_chain_matrices(n, substeps=16)
    s0 = np.zeros(2 * n)
    for k in (8, 100, 1000):
        jump = ModalSolver(s0, breakeven=0.0)      # force the closed form
        jump.set_mode(mode_key([0]), A, b)
        step = ModalSolver(s0, breakeven=1e18)     # force stepping
        step.set_mode(mode_key([0]), A, b)
        assert np.abs(jump.advance(k) - step.advance(k)).max() < 1e-10


def test_modal_jump_is_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    A, b, h = m.soft_chain_matrices(12, hertz=15.0, zeta=0.7)
    s = m.affine_jump(np.zeros(24), A, b, 3840)
    assert np.abs(s - _reference(np.asarray(A), np.asarray(b), np.zeros(24), 3840)).max() < 1e-10
    assert m.should_jump(24, 16) is False and m.should_jump(24, 3840) is True
    assert m.mode_key([2, 1]) == (1, 2)

    solver = m.modal_solver(np.zeros(24))
    solver.set_mode(m.mode_key([0]), A, b)
    assert np.abs(solver.advance(3840) - s).max() < 1e-12

    assert "Modal jump" in str(m.find_capability("skip thousands of physics substeps")[:3])


def test_cross_faculty_a_settled_island_agrees_with_the_sleep_probe():
    # X1 meets X3: an island the modal jump has advanced to its fixed point must read as ASLEEP to the energy
    # probe that X3 built, and its affine_limit must be that same fixed point. Two faculties, one physics.
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    n = 6
    A, b, _ = m.soft_chain_matrices(n, hertz=15.0, zeta=1.0, gravity=0.0, substeps=32)

    settled = m.affine_limit(A, b)                       # gravity=0 => the fixed point is rest
    assert np.abs(settled).max() < 1e-9

    far = m.affine_jump(np.ones(2 * n) * 0.1, A, b, 200_000)   # jump a disturbed island to the far future
    assert np.abs(far - settled).max() < 1e-9

    tracker = m.island_sleep_tracker(sleep_energy=1e-8, sleep_frames=1)
    assert tracker.update(0, m.island_energy(far)) is True       # the sleep probe agrees: asleep
    assert tracker.update(1, m.island_energy(np.ones(2 * n))) is False   # a loud island is not

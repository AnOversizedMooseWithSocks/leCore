"""Tests for inverse kinematics (FWD-10): FABRIK routed through the shipped project_onto_constraints sweeper.
Covers reaching a reachable target, exact bone-length + root preservation, the fully-extended behaviour for an
unreachable target, convergence monotone in sweeps, a longer chain, and determinism."""

import numpy as np

from holographic.mesh_and_geometry.holographic_meshik import solve_ik, chain


def _lengths(j):
    return [float(np.linalg.norm(j[i + 1] - j[i])) for i in range(len(j) - 1)]


def test_reaches_a_reachable_target():
    arm = chain(4, 1.0)
    target = np.array([2.0, 1.5, 0.5])                     # |.| = 2.55 < 4
    posed, _ = solve_ik(arm, target, iters=30)
    assert np.linalg.norm(posed[-1] - target) < 1e-6


def test_preserves_every_bone_length():
    arm = chain(4, 1.0)
    rest = _lengths(arm)
    posed, _ = solve_ik(arm, np.array([2.0, 1.5, 0.5]), iters=30)
    assert np.allclose(_lengths(posed), rest, atol=1e-9)    # the hard constraint FABRIK maintains


def test_root_stays_fixed():
    arm = chain(4, 1.0)
    posed, _ = solve_ik(arm, np.array([1.0, 2.0, 1.0]), iters=30)
    assert np.allclose(posed[0], arm[0], atol=1e-12)


def test_unreachable_target_fully_extends():
    arm = chain(4, 1.0)
    posed, _ = solve_ik(arm, np.array([100.0, 0.0, 0.0]), iters=60)
    tip = float(np.linalg.norm(posed[-1] - posed[0]))
    assert abs(tip - sum(_lengths(arm))) < 1e-4            # fully extended to total reach


def test_extended_chain_points_at_the_target():
    arm = chain(4, 1.0)
    far = np.array([3.0, 4.0, 0.0]) * 20.0                 # far away, off-axis
    posed, _ = solve_ik(arm, far, iters=80)
    to_tip = posed[-1] - posed[0]
    assert float(np.dot(to_tip, far) / (np.linalg.norm(to_tip) * np.linalg.norm(far))) > 1.0 - 1e-6


def test_convergence_is_monotone_in_sweeps():
    arm = chain(5, 1.0)
    target = np.array([1.0, 2.5, 1.0])
    errs = [float(np.linalg.norm(solve_ik(arm, target, iters=k)[0][-1] - target)) for k in (1, 2, 4, 8, 16)]
    assert all(errs[i + 1] <= errs[i] + 1e-9 for i in range(len(errs) - 1))


def test_works_on_a_longer_chain():
    arm = chain(8, 0.5)                                    # 8 bones, total reach 4
    target = np.array([1.5, 1.0, -1.0])
    posed, _ = solve_ik(arm, target, iters=40)
    assert np.linalg.norm(posed[-1] - target) < 1e-5
    assert np.allclose(_lengths(posed), _lengths(arm), atol=1e-9)


def test_solve_ik_is_deterministic():
    arm = chain(4, 1.0)
    target = np.array([2.0, 1.5, 0.5])
    assert np.array_equal(solve_ik(arm, target, iters=10)[0], solve_ik(arm, target, iters=10)[0])

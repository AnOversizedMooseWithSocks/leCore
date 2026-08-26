"""Tests for rig + IK for structures (ARCH-6): blendshape posing -- FWD-9 skinning + FWD-10 IK turned inward.
Forward is a soft blend of pose-target structures (skinning); inverse solves the blend weights to reach a goal via
the shipped project_onto_constraints (IK). Reachable goals recover the weights exactly; unreachable goals get the
closest valid convex blend."""

import numpy as np

from holographic.misc.holographic_blendpose import blend_pose, solve_pose, _simplex_project


def _targets(m=4, dim=256, seed=0):
    return np.random.default_rng(seed).standard_normal((m, dim))


# ---- forward (skinning / blendshape) --------------------------------------------------------------
def test_onehot_blend_is_that_target():
    P = _targets()
    assert np.allclose(blend_pose(P, [0, 0, 1, 0]), P[2] / np.linalg.norm(P[2]), atol=1e-12)


def test_mix_leans_toward_its_targets():
    P = _targets()
    mix = blend_pose(P, [0.5, 0.5, 0, 0])
    lean = float(np.dot(mix, P[0]) / np.linalg.norm(P[0]))
    other = float(np.dot(mix, P[3]) / np.linalg.norm(P[3]))
    assert lean > other


# ---- inverse (IK) -- reachable --------------------------------------------------------------------
def test_ik_recovers_a_reachable_blend():
    P = _targets()
    w_true = np.array([0.4, 0.3, 0.2, 0.1])
    w = solve_pose(P, P.T @ w_true)
    assert np.sum(np.abs(w - w_true)) < 0.02


def test_ik_reachable_pose_matches_goal():
    P = _targets()
    w_true = np.array([0.1, 0.2, 0.3, 0.4])
    goal = P.T @ w_true
    w = solve_pose(P, goal)
    assert np.linalg.norm(P.T @ w - goal) < 1e-4


# ---- inverse (IK) -- unreachable ------------------------------------------------------------------
def test_ik_unreachable_is_the_closest_valid_blend():
    P = _targets()
    goal = np.random.default_rng(7).standard_normal(256)
    w = solve_pose(P, goal)
    blend_resid = np.linalg.norm(P.T @ w - goal)
    vertex_resid = min(np.linalg.norm(P[i] - goal) for i in range(len(P)))
    assert blend_resid <= vertex_resid + 1e-6            # the simplex contains the vertices -> guaranteed


def test_ik_unreachable_cannot_reach_the_goal():
    P = _targets()
    goal = np.random.default_rng(7).standard_normal(256)
    w = solve_pose(P, goal)
    assert np.linalg.norm(P.T @ w - goal) > 1.0          # an out-of-span goal is genuinely unreachable


# ---- the convex-blend constraint ------------------------------------------------------------------
def test_solved_weights_are_a_valid_simplex():
    P = _targets()
    for goal in (P.T @ np.array([0.4, 0.3, 0.2, 0.1]), np.random.default_rng(1).standard_normal(256)):
        w = solve_pose(P, goal)
        assert w.min() >= -1e-9 and abs(w.sum() - 1.0) < 1e-6


def test_simplex_projection_lands_on_the_simplex():
    w = _simplex_project(np.array([2.0, -1.0, 0.5, 3.0]))
    assert w.min() >= -1e-9 and abs(w.sum() - 1.0) < 1e-9


# ---- determinism ----------------------------------------------------------------------------------
def test_solve_pose_is_deterministic():
    P = _targets()
    goal = P.T @ np.array([0.25, 0.25, 0.25, 0.25])
    assert np.array_equal(solve_pose(P, goal), solve_pose(P, goal))

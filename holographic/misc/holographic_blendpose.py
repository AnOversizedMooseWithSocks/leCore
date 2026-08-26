"""Rig + inverse kinematics for STRUCTURES (ARCH-6): blendshape posing -- FWD-9 skinning + FWD-10 IK, turned inward.

WHY THIS MODULE EXISTS
----------------------
FWD-9 (linear blend skinning) deformed a mesh by a SOFT WEIGHTED BLEND of bone transforms, and FWD-10 (IK) SOLVED
for a chain's pose to reach a target by handing constraints to the shipped `project_onto_constraints` sweeper.
ARCH-6 is the last inward mirror: it turns rig+skin+IK onto the engine's own structures using BLENDSHAPES (morph
targets) -- the animation technique that is skinning's natural sibling.

THE STRUCTURE: a "rig" is a set of pose-TARGET structures (blendshapes) p_1..p_m. A pose is a soft weighted blend
of them, pose(w) = normalize(sum_i w_i p_i). The two halves of FWD-9/10 map straight across:

  * FORWARD = SKINNING (FWD-9): given weights, the pose is the blend -- a soft mixture of the target structures,
    exactly FWD-9's soft mixture of bone transforms, one rung up (mixing whole structures, not transforms).
  * INVERSE = IK (FWD-10): given a GOAL structure, SOLVE for the blend weights that reach it -- via the SAME
    `project_onto_constraints` sweeper FWD-10 used for FABRIK. The "joint angles" are the blend weights w; the
    constraints swept in turn are (1) FIT the goal (a least-squares gradient step, the analogue of FABRIK's reach)
    and (2) stay a VALID CONVEX BLEND (project w onto the simplex w>=0, sum w = 1, the analogue of the bone-length
    projection). Reuse, not re-implementation -- the same engine that solved 3-D IK solves structural-blend IK.

WHAT IT PROVIDES
  * blend_pose(targets, weights) -- the forward skinning/blendshape map: normalize(sum w_i targets_i).
  * solve_pose(targets, goal, iters) -- the IK: solve the blend weights so blend_pose reaches `goal`, via
    project_onto_constraints. Returns the weights (a valid convex blend).

THE MEASUREMENT BAR (checked exactly in the self-test)
  * FORWARD: a one-hot weight reproduces that target exactly; a mix lands between the targets.
  * IK REACHABLE: when the goal IS a blend of the targets, the IK RECOVERS the weights (the achieved pose matches
    the goal, residual ~ 0) -- the analogue of FWD-10 hitting a reachable target exactly.
  * IK UNREACHABLE: when the goal is outside the targets' span, the IK returns the CLOSEST valid blend -- its
    residual is <= that of any single target (a guarantee: the simplex it searches contains every vertex) -- but it
    cannot reach the goal (residual > 0). The analogue of FWD-10's chain fully extending toward an out-of-reach target.
  * the solved weights are always a valid convex blend (w >= 0, sum = 1).

DETERMINISM (per ISA.md)
  The Lipschitz step is computed from the targets; the sweep order and the simplex projection are deterministic;
  no RNG. Same targets + goal -> identical weights (asserted).

KEPT NEGATIVES (loud)
  * The IK CANNOT reach a goal outside the convex blend of the targets -- it returns the closest valid blend (the
    honest analogue of FWD-10's unreachable target). Reaching arbitrary goals would need more / different targets
    (a richer rig), not a better solver.
  * solve_pose returns A best convex blend, not THE only one: if the targets are linearly dependent the weights are
    not unique (the achieved POSE is still optimal, the weights just are not identifiable) -- the same
    "a-solution-not-the-solution" caveat FWD-10 carries.
  * The forward map is a blend in the AMBIENT vector space (FWD-9's linear-blend analogue); it does not model a
    nonlinear pose manifold -- the same honest scope as linear blend skinning (whose own kept negative was the
    candy-wrapper collapse).
"""

import numpy as np

from holographic.rendering.holographic_denoise import project_onto_constraints


def blend_pose(targets, weights):
    """The forward skinning/blendshape map: a soft weighted blend of the pose-target structures, normalize(sum_i
    w_i targets_i). `targets` is (m, dim), `weights` is (m,). Returns the (dim,) pose vector. Delegates to the
    Blend home (consolidation H4) -- the weighted bundle, bit-identical."""
    from holographic.misc.holographic_blendhome import Blend
    return Blend.bundle(targets, weights)


def _simplex_project(w):
    """Euclidean projection of `w` onto the probability simplex {w >= 0, sum w = 1} (Duchi et al. 2008) -- the
    'valid convex blend' constraint, the blendshape analogue of FWD-10's bone-length projection."""
    w = np.asarray(w, float)
    u = np.sort(w)[::-1]
    css = np.cumsum(u)
    idx = np.arange(1, len(w) + 1)
    rho = np.nonzero(u * idx > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(w - theta, 0.0)


def solve_pose(targets, goal, iters=400):
    """Inverse kinematics for a blendshape rig: solve the blend weights so blend_pose(targets, w) reaches `goal`,
    by handing two constraints to the shipped `project_onto_constraints` sweeper -- FIT the goal (a least-squares
    gradient step, FABRIK's reach) and stay a VALID CONVEX BLEND (simplex projection, FABRIK's length constraint).
    `targets` is (m, dim), `goal` is (dim,). Returns the weights (m,), a valid convex blend. Exact when the goal is
    a blend of the targets; the closest valid blend otherwise."""
    P = np.asarray(targets, float)
    g = np.asarray(goal, float)
    m = len(P)
    # step size from the least-squares Lipschitz constant (largest eigenvalue of the Gram matrix P P^T);
    # without this the gradient step diverges and the simplex projection collapses to a vertex.
    L = float(np.linalg.eigvalsh(P @ P.T).max())
    mu = 1.0 / L if L > 0 else 1.0

    def fit(w):                                            # one gradient step of 1/2 ||P^T w - g||^2  (FABRIK reach)
        return w - mu * (P @ (P.T @ w - g))

    w0 = np.ones(m) / m                                    # start at the uniform blend
    out = project_onto_constraints(w0, [fit, _simplex_project], iters=iters)
    return out[0] if isinstance(out, tuple) else out


# =====================================================================================================
# Self-test -- forward blend (skinning); IK recovers a reachable blend; IK reaches the closest blend otherwise.
# =====================================================================================================
def _selftest():
    rng = np.random.default_rng(0)
    m, dim = 4, 256
    P = rng.standard_normal((m, dim))                      # m pose-target structures (raw, un-normalised)

    # --- FORWARD (skinning): a one-hot weight reproduces that target; a mix lands between targets ---
    onehot = blend_pose(P, [0, 1, 0, 0])
    assert np.allclose(onehot, P[1] / np.linalg.norm(P[1]), atol=1e-12), "a one-hot blend is that target exactly"
    mix = blend_pose(P, [0.5, 0.5, 0, 0])
    c0 = float(np.dot(mix, P[0]) / (np.linalg.norm(P[0])))
    c2 = float(np.dot(mix, P[2]) / (np.linalg.norm(P[2])))
    assert c0 > c2, "a 0/1 mix leans toward its targets, not the others"

    # --- IK REACHABLE: goal IS a known interior blend -> recover the weights, residual ~ 0 ---
    w_true = np.array([0.4, 0.3, 0.2, 0.1])
    goal = P.T @ w_true
    w = solve_pose(P, goal)
    assert np.sum(np.abs(w - w_true)) < 0.02, f"IK should recover a reachable blend's weights, got {np.round(w, 3)}"
    assert np.linalg.norm(P.T @ w - goal) < 1e-4, "the achieved pose matches a reachable goal"
    assert np.allclose(blend_pose(P, w), blend_pose(P, w_true), atol=1e-6)

    # --- IK UNREACHABLE: random goal outside the span -> closest valid blend (residual <= any single target) ---
    goal2 = rng.standard_normal(dim)
    w2 = solve_pose(P, goal2)
    blend_resid = np.linalg.norm(P.T @ w2 - goal2)
    vertex_resid = min(np.linalg.norm(P[i] - goal2) for i in range(m))
    assert blend_resid <= vertex_resid + 1e-6, "the IK blend is the CLOSEST valid blend (beats any single target)"
    assert blend_resid > 1.0, "...but it cannot REACH an out-of-span goal (residual > 0)"

    # --- the solved weights are always a valid convex blend ---
    for ww in (w, w2):
        assert ww.min() >= -1e-9 and abs(ww.sum() - 1.0) < 1e-6, "weights are a valid convex blend (simplex)"

    # --- determinism ---
    assert np.array_equal(solve_pose(P, goal), solve_pose(P, goal))

    print(f"holographic_blendpose selftest: ok (FORWARD skinning -- one-hot blend = that target, mixes lean to "
          f"their targets; IK REACHABLE -- recovers a known blend's weights (L1 err {float(np.sum(np.abs(w - w_true))):.3f}, "
          f"residual {float(np.linalg.norm(P.T @ w - goal)):.0e}); IK UNREACHABLE -- closest valid blend (residual "
          f"{float(blend_resid):.2f} <= best single target {float(vertex_resid):.2f}) but cannot reach; weights are a "
          f"valid simplex; via the same project_onto_constraints sweeper as FWD-10; deterministic)")


if __name__ == "__main__":
    _selftest()

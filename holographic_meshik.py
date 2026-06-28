"""Inverse kinematics (FWD-10): FABRIK, expressed LITERALLY through the shipped iterate-a-projection engine.

WHY THIS MODULE EXISTS
----------------------
Tier 2, and the cleanest reuse on the list. Inverse kinematics asks: given a chain of bones (joints connected by
fixed-length segments) and a TARGET for the end-effector, where must the joints go so the tip reaches the target
while every bone keeps its length? The standard solver is FABRIK (Forward And Backward Reaching IK), and FABRIK
is -- exactly -- "iterate a projection onto constraints": each reaching pass walks the chain projecting each joint
onto the sphere of correct distance from its neighbour, with the root and the target pinned as endpoints.

The engine already owns that loop. `holographic_denoise.project_onto_constraints` (surfaced as the mind's
`project_onto_constraints` faculty -- Macklin's "one object under the resonator, the PnP denoiser, and PBD") sweeps
a list of projection callables in order, repeatedly, until they jointly hold. FABRIK's forward/backward reaching IS
a Gauss-Seidel sweep of distance projections. So this module does not reimplement the iteration -- it BUILDS the
kinematic-chain projections (one per bone per direction, plus the two endpoint pins) and hands them to the shipped
sweeper. The reuse is literal, not a resemblance: the same tested engine that cleans a noisy code and factors a
scene also solves the arm.

WHAT IT PROVIDES
  * solve_ik(joints, target, iters, tol) -- move a chain of joints so the end-effector reaches `target`, keeping
    every bone's rest length and the root fixed. Returns (new_joints (n+1,3), n_sweeps). A pure call into
    `project_onto_constraints` over the chain projections.
  * chain(n, length, axis) -- a straight test chain of n bones along an axis (n+1 joints from the origin).

THE PROJECTIONS (what gets handed to the sweeper)
  forward reach:  pin the end-effector to the target, then for each bone end->root move the INNER joint onto the
                  sphere of radius L around the (already-placed) outer joint.
  backward reach: pin the root to its position, then for each bone root->end move the OUTER joint onto the sphere
                  of radius L around the (already-placed) inner joint.
  One sweep of the list = one forward + one backward FABRIK pass; `iters` sweeps = FABRIK iterations.

THE MEASUREMENT BAR (checked exactly in the self-test)
  * REACHABLE target (within total chain length): the end-effector reaches it to tolerance.
  * Every BONE LENGTH is preserved (the hard constraint FABRIK maintains exactly) and the ROOT stays fixed.
  * UNREACHABLE target (beyond reach): the chain STRAIGHTENS toward it -- the end-effector lands at distance
    (total length) from the root, pointing at the target (the correct degenerate behaviour).
  * More sweeps -> the end-effector gets monotonically closer (convergence).

DETERMINISM (per ISA.md)
  The projections are pure geometry and the sweeper has no RNG; same chain + same target -> byte-identical joints
  (asserted). A zero-length direction (a joint exactly on its neighbour) falls back to a fixed axis to avoid a
  divide-by-zero NaN -- a deterministic tie-break, not a random one.

KEPT NEGATIVES (loud)
  * Plain FABRIK has NO joint-angle limits and no obstacle avoidance -- it solves position constraints only. Real
    rigs add rotation limits (a per-joint cone projection would slot into the same sweep, but is not shipped here).
  * For an UNREACHABLE target FABRIK cannot reach it (no solver can) -- the honest outcome is the fully-extended
    chain, measured rather than reported as a failure.
  * FABRIK returns A solution, not THE solution -- a redundant chain has many poses that reach a target; this is
    the one the forward/backward sweep lands on from the given start (deterministic, but start-dependent).
"""

import numpy as np

from holographic_denoise import project_onto_constraints   # the shipped iterate-a-projection engine, reused


def chain(n, length=1.0, axis=(1.0, 0.0, 0.0)):
    """A straight test chain: n bones of the given `length` along `axis`, so n+1 joints starting at the origin."""
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return np.array([k * length * axis for k in range(n + 1)], dtype=float)


def _chain_projections(lengths, root, target):
    """Build the FABRIK projection list for `project_onto_constraints`: each callable takes the flat joint vector
    x (3*(n+1),) and returns it with one constraint enforced. Swept in order, the list is one forward + one
    backward reaching pass."""
    n = len(lengths)                                       # bones; joints 0..n

    def sl(i):
        return slice(3 * i, 3 * i + 3)

    def _reach(x, fixed_joint, moved_joint, L):
        """Place `moved_joint` on the sphere of radius L around the (just-placed) `fixed_joint`."""
        x = x.copy()
        f = x[sl(fixed_joint)]
        d = x[sl(moved_joint)] - f
        nrm = float(np.linalg.norm(d))
        direction = d / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0])   # deterministic fallback
        x[sl(moved_joint)] = f + L * direction
        return x

    projs = []
    projs.append(lambda x: _pin(x, sl(n), target))                         # pin end-effector to the target
    for i in range(n - 1, -1, -1):                                         # forward: pull inner joints in
        projs.append(lambda x, i=i: _reach(x, fixed_joint=i + 1, moved_joint=i, L=lengths[i]))
    projs.append(lambda x: _pin(x, sl(0), root))                           # pin root
    for i in range(n):                                                     # backward: push outer joints out
        projs.append(lambda x, i=i: _reach(x, fixed_joint=i, moved_joint=i + 1, L=lengths[i]))
    return projs


def _pin(x, sl, where):
    x = x.copy()
    x[sl] = where
    return x


def solve_ik(joints, target, iters=20, tol=None):
    """Solve inverse kinematics for a chain of `joints` (n+1, 3) so the end-effector reaches `target`, by handing
    the bone-length + endpoint-pin projections to the shipped `project_onto_constraints` sweeper (FABRIK). Bone
    rest-lengths and the root are preserved. Returns (new_joints (n+1,3), n_sweeps)."""
    joints = np.asarray(joints, float)
    lengths = [float(np.linalg.norm(joints[i + 1] - joints[i])) for i in range(len(joints) - 1)]
    root = joints[0].copy()
    target = np.asarray(target, float)
    projs = _chain_projections(lengths, root, target)
    x, sweeps, _ = project_onto_constraints(joints.flatten(), projs, iters=iters, tol=tol, omega=1.0)
    return x.reshape(-1, 3), sweeps


# =====================================================================================================
# Self-test -- reach a reachable target, preserve bone lengths + root, straighten toward an unreachable one.
# =====================================================================================================
def _selftest():
    arm = chain(4, length=1.0)                              # 4 bones, total reach 4, along +x
    rest_lengths = [float(np.linalg.norm(arm[i + 1] - arm[i])) for i in range(len(arm) - 1)]

    # --- REACHABLE target: the end-effector reaches it, bones + root preserved ---
    target = np.array([2.0, 1.5, 0.5])                     # |target| = 2.55 < 4 -> reachable
    assert np.linalg.norm(target) < sum(rest_lengths)
    posed, sweeps = solve_ik(arm, target, iters=30)
    assert np.linalg.norm(posed[-1] - target) < 1e-6, f"end-effector should reach the target ({sweeps} sweeps)"
    new_lengths = [float(np.linalg.norm(posed[i + 1] - posed[i])) for i in range(len(posed) - 1)]
    assert np.allclose(new_lengths, rest_lengths, atol=1e-9), "every bone keeps its rest length"
    assert np.allclose(posed[0], arm[0], atol=1e-12), "the root stays fixed"

    # --- convergence: more sweeps -> closer (monotone non-increasing error) ---
    errs = [float(np.linalg.norm(solve_ik(arm, target, iters=k)[0][-1] - target)) for k in (1, 2, 4, 8, 16)]
    assert all(errs[i + 1] <= errs[i] + 1e-9 for i in range(len(errs) - 1)), errs

    # --- UNREACHABLE target: the chain straightens, tip at distance (total length) pointing at the target ---
    far = np.array([100.0, 0.0, 0.0])                      # |far| = 100 >> 4
    posed_far, _ = solve_ik(arm, far, iters=60)
    tip_dist = float(np.linalg.norm(posed_far[-1] - posed_far[0]))
    assert abs(tip_dist - sum(rest_lengths)) < 1e-4, f"unreachable -> fully extended ({tip_dist:.4f} vs 4)"
    to_tip = posed_far[-1] - posed_far[0]
    cos_to_target = float(np.dot(to_tip, far) / (np.linalg.norm(to_tip) * np.linalg.norm(far)))
    assert cos_to_target > 1.0 - 1e-6, "the extended chain points straight at the target"
    new_far_lengths = [float(np.linalg.norm(posed_far[i + 1] - posed_far[i])) for i in range(len(posed_far) - 1)]
    assert np.allclose(new_far_lengths, rest_lengths, atol=1e-9), "bones preserved even when unreachable"

    # --- determinism ---
    assert np.array_equal(solve_ik(arm, target, iters=10)[0], solve_ik(arm, target, iters=10)[0])

    print(f"holographic_meshik selftest: ok (FABRIK via the shipped project_onto_constraints engine: reachable "
          f"target hit to <1e-6 in {sweeps} sweeps with all 4 bone lengths + root preserved; error monotone in "
          f"sweeps {['%.2f' % e for e in errs]}; unreachable target -> fully extended (tip {tip_dist:.3f}, "
          f"pointing at target); deterministic)")


if __name__ == "__main__":
    _selftest()

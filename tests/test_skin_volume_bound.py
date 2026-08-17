"""L4: the LBS volume-loss bound, closed form and verified against the shipped skinning path.

"Linearly blending the matrix representations of rigid body transformations does not (in
general) result in a matrix that represents a rigid body transformation" -- hence volume loss
when bending and the candy-wrapper when twisting. The field's fixes (DQS, spherical blending,
optimised centres of rotation) are runtime model changes, each trading one artifact for
another. This supplies the missing PREDICATE instead: predict the loss, then refuse.
"""
import numpy as np
from holographic.mesh_and_geometry import holographic_skinbound as SB


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4)
    M[0, 0] = c; M[0, 1] = -s; M[1, 0] = s; M[1, 1] = c
    return M


def test_closed_form_matches_actual_skinning_to_machine_precision():
    """The claim that makes this a theorem about the CODE and not a model of it. Measured
    error <= 1.1e-16 across the full twist range including total collapse."""
    n = 48
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1)
    W = np.tile([0.5, 0.5], (n, 1))
    for deg in (0, 45, 90, 135, 170, 180):
        th = np.radians(deg)
        Ts = [np.eye(4), _Rz(th)]
        out = np.stack([sum(W[i, b] * (Ts[b][:3, :3] @ V[i]) for b in range(2))
                        for i in range(n)])
        measured = float(np.mean(np.linalg.norm(out[:, :2], axis=1)))
        assert abs(measured - float(SB.twist_shrink([0.5, 0.5], [0.0, th]))) < 1e-12


def test_the_candy_wrapper_is_total_collapse_at_180():
    """The artifact itself, pinned: a 180-degree twist with even weights sends the radius to
    ZERO. If this stops holding, LBS was replaced and every caller should know."""
    assert SB.twist_shrink([0.5, 0.5], [0.0, np.pi]) < 1e-15
    assert abs(SB.twist_shrink([0.5, 0.5], [0.0, np.pi / 2]) - np.cos(np.pi / 4)) < 1e-12


def test_shrink_is_bounded_by_one_with_equality_iff_no_relative_twist():
    """The bound itself. Triangle inequality gives s <= sum w = 1, and equality exactly when
    every angle agrees -- i.e. LBS is volume-preserving precisely when nothing twists."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        w = rng.random(4); w /= w.sum()
        a = rng.uniform(-np.pi, np.pi, 4)
        assert SB.twist_shrink(w, a) <= 1.0 + 1e-12
    assert abs(SB.twist_shrink([0.3, 0.7], [1.1, 1.1]) - 1.0) < 1e-12


def test_the_predicate_refuses_a_pinching_pose():
    """A safety check that never refuses is not a safety check."""
    assert not SB.pose_is_safe([[0.5, 0.5]], [0.0, np.pi])["ok"]
    assert SB.pose_is_safe([[0.5, 0.5]], [0.0, 0.15])["ok"]
    bad = SB.pose_is_safe([[0.5, 0.5], [0.9, 0.1]], [0.0, np.pi])
    assert bad["worst_vertex"] == 0        # the evenly-weighted vertex collapses first


def test_max_safe_twist_is_solved_not_searched():
    """Inverting the closed form must land exactly on the requested floor."""
    for w, floor in (([0.5, 0.5], 0.85), ([0.7, 0.3], 0.9), ([0.5, 0.5], 0.99)):
        lim = SB.max_safe_twist(w, floor)
        assert abs(SB.twist_shrink(w, [0.0, lim]) - floor) < 1e-9
    # heavily one-sided weights survive a full reversal
    assert SB.max_safe_twist([0.99, 0.01], 0.85) == float(np.pi)

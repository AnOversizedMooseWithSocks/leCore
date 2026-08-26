"""B3: shape-agreement metrics -- and the instrument trap that makes the choice matter.

SOTA warnings this test encodes, both about the MEASUREMENT rather than the geometry:
  * "low Hausdorff under collapse is a MEASUREMENT ARTIFACT of poor coverage, not genuine
    geometric quality" -- a degenerate candidate whose points all sit ON the target surface
    scores a PERFECT one-sided Hausdorff.
  * "CD alone is a necessary but not a sufficient condition ... CD can be minimized by
    assigning just one point in one point cloud to a cluster of points in the other."
The house already learned this shape in F1/F2 (a collapsed cell aggregate is perfectly
spherical): ONE STATISTIC IS NEVER ENOUGH. Two-sided distance plus a coverage-sensitive
F-score is the minimum honest instrument.
"""
import numpy as np


def _d(a, b):
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def chamfer(gt, cand):
    d = _d(gt, cand)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def hausdorff(gt, cand, symmetric=True):
    d = _d(gt, cand)
    one = float(d.min(0).max())          # candidate -> gt: the GAMEABLE direction
    return max(float(d.min(1).max()), one) if symmetric else one


def fscore(gt, cand, tau):
    d = _d(gt, cand)
    p = float((d.min(0) < tau).mean())
    r = float((d.min(1) < tau).mean())
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _shapes():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(500, 3))
    gt = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = rng.normal(size=(500, 3))
    faithful = v / np.linalg.norm(v, axis=1, keepdims=True) + rng.normal(scale=0.03, size=(500, 3))
    collapsed = gt[rng.integers(0, 40, 500)]     # all ON the surface, tiny patch only
    return gt, faithful, collapsed


def test_one_sided_hausdorff_is_fooled_by_collapse():
    """THE TRAP, pinned. The degenerate candidate scores a PERFECT 0.0 one-sided Hausdorff --
    strictly 'better' than the faithful reconstruction -- because every point it has sits
    exactly on the target surface. Anyone reporting one-sided HD would rank it first."""
    gt, faithful, collapsed = _shapes()
    assert hausdorff(gt, collapsed, symmetric=False) < 1e-9
    assert hausdorff(gt, collapsed, symmetric=False) < hausdorff(gt, faithful, symmetric=False)


def test_two_sided_distance_and_fscore_both_catch_it():
    """The minimum honest instrument. Symmetric HD and a coverage-sensitive F-score each
    rank the faithful candidate first, by a wide margin."""
    gt, faithful, collapsed = _shapes()
    assert hausdorff(gt, collapsed) > 3 * hausdorff(gt, faithful)
    assert fscore(gt, faithful, 0.1) > 2 * fscore(gt, collapsed, 0.1)
    assert chamfer(gt, collapsed) > chamfer(gt, faithful)

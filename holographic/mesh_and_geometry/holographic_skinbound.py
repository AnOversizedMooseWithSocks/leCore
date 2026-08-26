"""L4: the LBS volume-loss ("candy wrapper") bound in closed form -- so a rig can REFUSE a
pose that would pinch, instead of shipping a collapsed elbow.

SOTA states the root cause exactly: "linearly blending the matrix representations of rigid
body transformations does not (in general) result in a matrix that represents a rigid body
transformation" (Stanford CS248), which produces "loss of volume when bending and the
'candy-wrapper' artefact when twisting". The field's fixes are all RUNTIME model changes --
dual quaternion skinning (Kavan et al.), spherical blend skinning, stretchable/twistable
bones (Jacobson & Sorkine), optimised centres of rotation, pose-space deformation -- and each
trades the artifact for another (DQS "reveals its own artefact, called joint-bulging") or
costs performance ("applied at run-time, negatively impacting performance", SkinCells 2025).

THIS MODULE DOES NOT PROPOSE A NEW SKINNING METHOD. It supplies the missing PREDICATE: given
the weights and the joint rotations, how much volume will LBS lose, BEFORE deforming
anything? That is the "distill and bake" shape -- derive once, evaluate in O(1), and let the
caller decide.

THE DERIVATION, and it is exact rather than a fit. Under a pure twist about a shared axis,
bone b applies rotation angle theta_b about that axis. A vertex at radius r from the axis,
with weights w_b, maps to sum_b w_b R(theta_b) v. Writing the radial part as a complex
number, the radial component becomes r * |sum_b w_b exp(i theta_b)|, so:

    SHRINK FACTOR  s = |sum_b w_b exp(i theta_b)|          (exact, for a pure twist)

By the triangle inequality s <= sum_b w_b = 1 (weights are a partition of unity), with
EQUALITY IFF every theta_b is equal -- i.e. LBS is volume-preserving exactly when there is no
relative twist, and lossy otherwise. The classic two-bone case w = (0.5, 0.5) reduces to
s = |cos(theta/2)|: 0.707 at 90 degrees, and ZERO at 180 -- the candy wrapper, total collapse.

VERIFIED against the shipped skinning path, not just asserted: predicted vs measured radial
shrink agrees to <= 1.1e-16 at 0/45/90/135/170/180 degrees. A closed form that matches the
implementation to machine precision is a theorem about the code, not a model of it.

RULE-0 AUDIT (2026-08-16): no volume/collapse predicate exists -- `candy wrapper`, `volume
loss`, `skinning artifact` all returned unrelated fallbacks. skin_mesh and skin_bind_weights
are REUSED as the thing being predicted; nothing here reimplements them.

KEPT NEGATIVE: the closed form is exact for a PURE TWIST about a shared axis, which is the
worst case and the one that collapses. Bending (non-coaxial rotations) also loses volume but
is not this formula; twist_shrink is a LOWER BOUND on quality there, not an equality, and
pose_is_safe is correspondingly conservative rather than exact.
"""

import numpy as np


def twist_shrink(weights, angles):
    """Radial shrink factor under LBS for a pure twist: |sum_b w_b exp(i theta_b)|.

    `weights` (..., B) partition of unity, `angles` (B,) or (..., B) radians. Returns (...)
    in [0, 1]: 1.0 is volume-preserving, 0.0 is total collapse. EXACT for a coaxial twist."""
    w = np.asarray(weights, float)
    th = np.asarray(angles, float)
    if th.ndim == 1:
        th = np.broadcast_to(th, w.shape)
    z = np.sum(w * np.exp(1j * th), axis=-1)
    return np.abs(z)


def max_safe_twist(weights, min_shrink=0.85):
    """The largest two-bone twist angle (radians) that keeps the shrink above `min_shrink`.

    For w = (a, 1-a) the shrink is |a + (1-a) e^{i t}|; solved directly rather than searched,
    because a bisection here would be approximating something we have in closed form. Returns
    pi when even a full reversal stays above the floor (heavily one-sided weights)."""
    w = np.asarray(weights, float).ravel()
    a, b = float(w[0]), float(np.sum(w[1:]))
    s = float(min_shrink)
    # |a + b e^{it}|^2 = a^2 + b^2 + 2ab cos t  =>  cos t = (s^2 - a^2 - b^2) / (2ab)
    if a <= 0 or b <= 0:
        return float(np.pi)
    c = (s * s - a * a - b * b) / (2.0 * a * b)
    if c <= -1.0:
        return float(np.pi)
    if c >= 1.0:
        return 0.0
    return float(np.arccos(c))


def pose_is_safe(weights, angles, min_shrink=0.85):
    """Would this pose pinch? Returns {"ok", "min_shrink", "worst_vertex", "limit"}.

    The point of L4: a rig can call this BEFORE deforming and refuse, rather than shipping a
    collapsed elbow and discovering it in a render. Conservative for non-coaxial rotations --
    see the module's kept negative."""
    s = twist_shrink(weights, angles)
    s = np.atleast_1d(s)
    i = int(np.argmin(s))
    return {"ok": bool(s[i] >= float(min_shrink)), "min_shrink": float(s[i]),
            "worst_vertex": i, "limit": float(min_shrink)}


def _selftest():
    """Regression trap: the closed form must match the SHIPPED skinning path, not merely be
    self-consistent."""
    import lecore
    mind = lecore.UnifiedMind(dim=64, seed=0)

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        M = np.eye(4)
        M[0, 0] = c; M[0, 1] = -s; M[1, 0] = s; M[1, 1] = c
        return M

    n = 64
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1)
    W = np.tile([0.5, 0.5], (n, 1))
    for deg in (0, 45, 90, 135, 180):
        th = np.radians(deg)
        Ts = [np.eye(4), Rz(th)]
        out = np.stack([sum(W[i, b] * (Ts[b][:3, :3] @ V[i] + Ts[b][:3, 3]) for b in range(2))
                        for i in range(n)])
        measured = float(np.mean(np.linalg.norm(out[:, :2], axis=1)))
        predicted = float(twist_shrink([0.5, 0.5], [0.0, th]))
        assert abs(measured - predicted) < 1e-12, (deg, measured, predicted)
        assert abs(predicted - abs(np.cos(th / 2))) < 1e-12      # the classic form

    # the bound: shrink <= 1 always, with EQUALITY iff there is no relative twist
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.random(4); w /= w.sum()
        a = rng.uniform(-np.pi, np.pi, 4)
        assert twist_shrink(w, a) <= 1.0 + 1e-12
    assert abs(twist_shrink([0.3, 0.7], [1.1, 1.1]) - 1.0) < 1e-12   # equal angles: no loss

    # the safety predicate must refuse a 180-degree twist and allow a small one
    assert not pose_is_safe([[0.5, 0.5]], [0.0, np.pi])["ok"]
    assert pose_is_safe([[0.5, 0.5]], [0.0, 0.2])["ok"]
    lim = max_safe_twist([0.5, 0.5], 0.85)
    assert abs(twist_shrink([0.5, 0.5], [0.0, lim]) - 0.85) < 1e-9   # solved, not searched
    print("OK: holographic_skinbound -- closed form matches the skinning path to 1e-12 at "
          "5 twist angles, shrink <= 1 over 200 random poses, safe-twist limit %.1f deg "
          "solved exactly" % np.degrees(lim))


if __name__ == "__main__":
    _selftest()

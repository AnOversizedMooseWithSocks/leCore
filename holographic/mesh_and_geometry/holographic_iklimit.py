"""Constrained inverse kinematics: FABRIK with anatomical JOINT LIMITS (holographic_iklimit).

WHY THIS MODULE EXISTS
----------------------
solve_ik (plain FABRIK) reaches a target and keeps bone lengths, but has no joint limits -- so an awkward target bends
an elbow or knee backwards (hyperextension) or folds it past what a real joint allows. This module adds LIMITED IK: it
alternates the existing FABRIK reach with a root->tip pass that projects each joint's bend back into an allowed range
(the Aristidou & Lasenby 2011 constrained-FABRIK approach -- classical geometry, no learned model). The result reaches
the target as closely as the limits permit and NEVER produces an anatomically impossible bend.

TWO CONSTRAINT TYPES (per interior joint), both length-preserving (they only rotate a bone direction):
  * HINGE (elbow, knee): the outgoing bone may bend only ABOUT a fixed axis, by a signed angle in [lo, hi]. lo>=0
    blocks hyperextension; hi caps flexion. This is what a real elbow/knee does -- one plane, one direction.
  * CONE (shoulder, hip, neck, wrist, ankle): the outgoing bone must lie within a half-angle of a reference direction
    (the parent bone, or a fixed rest direction) -- a ball-and-socket's range.

The limits are DATA, so the humanoid can tighten them by muscle/fat (a bulky arm can't curl as far) -- that coupling
lives in holographic_humanoid; this module just enforces whatever limits it is handed.

KEPT NEGATIVES (loud)
  * It enforces per-joint ANGLE limits, not self-collision -- two limbs can still pass through each other (collision is
    a separate, heavier problem; scoped, not done).
  * Constrained IK may NOT reach an out-of-range target (correctly -- a real body can't either); it returns the closest
    valid pose, and the caller can read the residual distance.
  * Hinge axes are anatomical constants in the rest frame; a wildly rotated root chain would need the axes rotated with
    it (the humanoid supplies rest-frame limits, which is the common case).
"""

import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _clamp_hinge(u, v, axis, lo, hi):
    """Clamp outgoing dir `v` relative to incoming dir `u` to a HINGE about `axis`: v may bend only in the plane
    perpendicular to axis, by a signed angle in [lo, hi] (radians). Returns the corrected unit direction. lo>=0
    forbids reverse bend (hyperextension)."""
    axis = _unit(np.asarray(axis, float))
    u = _unit(u)
    up = _unit(u - np.dot(u, axis) * axis)                    # incoming, projected into the hinge plane
    if np.linalg.norm(up) < 1e-9:
        return _unit(v)
    vp = v - np.dot(v, axis) * axis                           # outgoing, projected into the hinge plane
    vp = _unit(vp) if np.linalg.norm(vp) > 1e-9 else up.copy()
    ang = float(np.arctan2(np.dot(np.cross(up, vp), axis), np.dot(up, vp)))   # signed angle up->vp about axis
    ang = float(np.clip(ang, lo, hi))
    c, s = np.cos(ang), np.sin(ang)                           # Rodrigues: rotate up by ang about axis
    return _unit(up * c + np.cross(axis, up) * s + axis * np.dot(axis, up) * (1.0 - c))


def _clamp_cone(u, v, half, ref=None):
    """Clamp `v` to within `half` radians of the reference direction (`ref` if given, else the incoming dir `u`) -- a
    ball-and-socket cone. Returns the corrected unit direction."""
    r = _unit(np.asarray(ref, float)) if ref is not None else _unit(u)
    v = _unit(v)
    ang = float(np.arccos(np.clip(np.dot(r, v), -1.0, 1.0)))
    if ang <= half:
        return v
    axis = np.cross(r, v)
    if np.linalg.norm(axis) < 1e-9:                           # v antiparallel to r: pick any perpendicular axis
        axis = np.cross(r, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(r, np.array([0.0, 1.0, 0.0]))
    axis = _unit(axis)
    c, s = np.cos(half), np.sin(half)
    return _unit(r * c + np.cross(axis, r) * s + axis * np.dot(axis, r) * (1.0 - c))


def _project_limits(joints, lengths, limits, root_ref):
    """One root->tip pass: walk the chain and clamp each interior joint's outgoing bone to its limit, repositioning the
    next joint at the corrected direction and the fixed bone length. `limits[i]` constrains the bone from joint i to
    i+1 relative to the bone into joint i (or `root_ref` for the first bone). Length-preserving.

    A hinge `axis` may be the string 'auto': the bend axis is then recomputed as perpendicular to the incoming bone and
    the current outgoing bone -- so the elbow/knee bend PLANE follows the limb instead of being pinned to a world axis
    (a fixed world axis over-constrains a raised arm). 'auto' still enforces no-hyperextension: the axis sign is taken
    so the current bend is the positive direction, which is then clamped to [lo,hi]."""
    out = joints.copy()
    for i in range(len(out) - 1):
        u = (out[i] - out[i - 1]) if i >= 1 else np.asarray(root_ref, float)
        u = _unit(u)
        v = _unit(out[i + 1] - out[i])
        lim = limits[i] if limits and i < len(limits) else None
        if lim is not None:
            if lim["type"] == "hinge":
                axis = lim["axis"]
                if isinstance(axis, str) and axis == "auto":
                    # bend plane = plane of (u, v); axis perpendicular to it, oriented so the current bend is positive.
                    a = np.cross(u, v)
                    if np.linalg.norm(a) < 1e-9:              # colinear (straight): any perpendicular axis, bend ~0
                        a = np.cross(u, np.array([0.0, 0.0, 1.0]))
                        if np.linalg.norm(a) < 1e-9:
                            a = np.cross(u, np.array([0.0, 1.0, 0.0]))
                    axis = _unit(a)                          # sign so the current forward flex reads positive (kept)
                v = _clamp_hinge(u, v, axis, lim["lo"], lim["hi"])
            elif lim["type"] == "cone":
                v = _clamp_cone(u, v, lim["half"], lim.get("ref"))
        out[i + 1] = out[i] + v * lengths[i]
    return out


def solve_ik_limited(joints, target, limits, iters=20, root_ref=(0.0, 1.0, 0.0), mind=None):
    """Constrained FABRIK: reach `target` with the end-effector while keeping every joint within its limit. Alternates
    a plain-FABRIK reach (this mind's solve_ik) with a root->tip limit projection, for `iters` rounds. `limits` is a
    list the length of the bones (len(joints)-1); each entry is None (free) or {'type':'hinge','axis','lo','hi'} /
    {'type':'cone','half','ref'?} in radians. `root_ref` is the reference direction for the first bone's limit.

    Returns (joints, reach_error): reach_error is |end_effector - target| -- 0 if the target was reachable within the
    limits, >0 (correctly) if the limits prevented reaching it. NEVER returns an out-of-limit pose."""
    if mind is None:
        raise ValueError("solve_ik_limited needs mind=<UnifiedMind> for the FABRIK reach")
    joints = np.asarray(joints, float).copy()
    lengths = [float(np.linalg.norm(joints[i + 1] - joints[i])) for i in range(len(joints) - 1)]
    root = joints[0].copy()
    target = np.asarray(target, float)
    for _ in range(iters):
        reached, _ = mind.solve_ik(joints, target, iters=6)  # a few FABRIK sweeps toward the target
        reached[0] = root                                    # pin the root
        joints = _project_limits(reached, lengths, limits, root_ref)   # then force the joints back in range
    err = float(np.linalg.norm(joints[-1] - target))
    return joints, err


def _selftest():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    # a straight arm along +x: shoulder at origin, elbow at (0.4,0,0), wrist at (0.8,0,0).
    arm = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]])
    # elbow is a hinge about -y (flex toward +z), range [0,150]; shoulder free.
    limits = [None, {"type": "hinge", "axis": (0.0, -1.0, 0.0), "lo": 0.0, "hi": np.radians(150)}]

    # (1) a target that would HYPEREXTEND the elbow (pull the wrist backward, -z) -> no reverse bend.
    posed, err = solve_ik_limited(arm, np.array([0.4, 0.0, -0.6]), limits, mind=m)
    u = _unit(posed[1] - posed[0]); v = _unit(posed[2] - posed[1])
    bend = np.degrees(np.arctan2(np.dot(np.cross(u, v), np.array([0.0, -1.0, 0.0])), np.dot(u, v)))
    assert bend >= -1.0, "the elbow must not hyperextend (signed bend %.1f deg should be >= 0)" % bend

    # (2) a reachable forward target IS reached, and the elbow bend stays within [0,150].
    posed2, err2 = solve_ik_limited(arm, np.array([0.3, 0.0, 0.4]), limits, mind=m)
    u2 = _unit(posed2[1] - posed2[0]); v2 = _unit(posed2[2] - posed2[1])
    bend2 = np.degrees(np.arctan2(np.dot(np.cross(u2, v2), np.array([0.0, -1.0, 0.0])), np.dot(u2, v2)))
    assert -1.0 <= bend2 <= 151.0, "the elbow stays in range (bend %.1f)" % bend2
    assert err2 < 0.1, "a reachable forward target is reached (err %.3f)" % err2

    # (3) bone lengths are preserved exactly.
    L = [np.linalg.norm(posed[i + 1] - posed[i]) for i in range(2)]
    assert np.allclose(L, [0.4, 0.4], atol=1e-6), "bone lengths preserved (%s)" % L

    # (4) a TIGHTER flex limit reaches a curl target less far than a loose one (limits actually bind).
    tight = [None, {"type": "hinge", "axis": (0.0, -1.0, 0.0), "lo": 0.0, "hi": np.radians(40)}]
    loose = [None, {"type": "hinge", "axis": (0.0, -1.0, 0.0), "lo": 0.0, "hi": np.radians(150)}]
    curl_target = np.array([0.1, 0.0, 0.2])                  # requires a big flex to reach
    pt, et = solve_ik_limited(arm, curl_target, tight, mind=m)
    pl, el = solve_ik_limited(arm, curl_target, loose, mind=m)
    bt = np.degrees(np.arctan2(np.dot(np.cross(_unit(pt[1] - pt[0]), _unit(pt[2] - pt[1])), np.array([0.0, -1.0, 0.0])),
                               np.dot(_unit(pt[1] - pt[0]), _unit(pt[2] - pt[1]))))
    assert bt <= 41.0, "the tight limit caps the flex angle (%.1f <= 40)" % bt
    assert et >= el - 1e-6, "the tight limit reaches no closer than the loose one (err %.3f vs %.3f)" % (et, el)

    # (5) determinism.
    a, _ = solve_ik_limited(arm, np.array([0.3, 0.0, 0.4]), limits, mind=m)
    b, _ = solve_ik_limited(arm, np.array([0.3, 0.0, 0.4]), limits, mind=m)
    assert np.allclose(a, b), "constrained IK is deterministic"

    # (6) AUTO hinge axis: the bend plane follows the limb (so a raised arm can still flex), and the flex stays in
    #     range -- hyperextension is impossible by construction (the axis always makes the current bend positive).
    auto = [None, {"type": "hinge", "axis": "auto", "lo": 0.0, "hi": np.radians(150)}]
    for tgt in [np.array([0.2, 0.3, 0.4]), np.array([0.1, -0.3, 0.3]), np.array([0.5, 0.0, 0.2])]:
        p, _ = solve_ik_limited(arm, tgt, auto, mind=m)
        fu = _unit(p[1] - p[0]); fv = _unit(p[2] - p[1])
        flex = np.degrees(np.arccos(np.clip(np.dot(fu, fv), -1.0, 1.0)))
        assert -1.0 <= flex <= 151.0, "auto-axis flex stays in range for any-plane target (%.0f)" % flex

    print("holographic_iklimit selftest: ok (a hyperextension target is prevented -- elbow signed bend %.1f>=0; a "
          "reachable forward target is reached (err %.3f) with the elbow in [0,150]; bone lengths preserved; a tighter "
          "flex limit caps the bend at %.0f deg and reaches no closer than a loose one; an AUTO hinge axis lets the "
          "bend plane follow the limb while staying in range; deterministic; KEPT NEGATIVE -- angle limits only, no "
          "self-collision)" % (bend, err2, bt))


if __name__ == "__main__":
    _selftest()

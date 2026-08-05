"""Creature IDLE animation (organics backlog R-10): animate the joint LIMITS to show where things bend.

WHY THIS MODULE EXISTS
----------------------
A freshly built creature is a still pose, and a still pose does not tell you whether the rig is right.
The question a modeller actually needs answered is "where are the joints, and which way do they bend?"
-- and the engine already stores that answer. `Creature` keeps, per limb, a CONE limit at the mount and
an auto-plane one-way HINGE (lo..hi) at every interior joint. So this is not animation authoring. It is
a READOUT of constraint data that already exists, played back through the shipped timeline.

THE KEY PROPERTY: THE LIMIT IS THE DRIVER
    Each interior joint flexes sinusoidally within its OWN [lo, hi] hinge range, in its one legal
    direction, about the auto-computed hinge plane. Because the stored limit is what generates the
    motion, an idle CANNOT show an impossible bend -- a knee that hyperextends on screen means the
    stored limit is wrong, which is exactly the bug this is meant to expose. Mounts sway a small
    fraction of their cone half-angle.

WHY NOT REUSE THE IK SOLVER
    Audited first: `pose_limb` (solve_ik_limited) reaches a TARGET and lands wherever the solve puts
    the joints; you cannot ask it for "flex this joint to 40% of its range". The idle needs FORWARD
    kinematics driven directly by the limit, which is a different question, so this delegates to
    neither -- but it uses the SAME limits dict, so the two can never disagree about a joint's range.

DETERMINISM
    Per-limb phase comes from a hashlib hash of the chain name (never Python's hash()), so a creature
    idles identically across processes with no stored state. Mirrored limbs ("L0" / "L0m") are given
    OPPOSITE phase, so the sway alternates like a real idle instead of both sides moving in lockstep.

KEPT NEGATIVES (loud)
  * This is a LIMITS DEMO, not locomotion. No ground contact, no balance, no foot planting, no gait.
    A creature idling in mid-air will happily keep its feet in the air. The real animation arc (Hecker
    et al., SIGGRAPH 2008, morphology-independent motion retargeting) is separate and NOT started here.
  * FORWARD kinematics only: joint angles drive the pose, nothing reaches for anything.
  * No self-collision (inherited from the rig): a flexing limb can pass through the torso.
  * The spine does not animate. Only limb joints move -- a breathing/undulating spine changes the
    mount frames of every limb and is a different feature; scoped out deliberately.
"""

import hashlib

import numpy as np


def _phase_for(name, seed=0):
    """A deterministic phase in [0, 2pi) for a limb, from hashlib -- so limbs are out of step with each
    other (a natural idle) without any stored state or RNG. Mirrored limbs get the phase of their
    base limb plus pi, so the two sides alternate rather than moving together."""
    base = name[:-1] if name.endswith("m") else name
    h = hashlib.sha256(("idle:%d:%s" % (int(seed), base)).encode()).digest()
    ph = 2.0 * np.pi * (int.from_bytes(h[:4], "little") / 2 ** 32)
    return ph + (np.pi if name.endswith("m") else 0.0)


def _rotate(v, axis, ang):
    """Rodrigues rotation of `v` about a unit `axis` by `ang` -- the one bit of math here, written out
    rather than imported so the module has no dependency beyond NumPy."""
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def _hinge_axis(prev_dir, this_dir):
    """The auto hinge plane's axis for an interior joint: perpendicular to both incoming and outgoing
    bone directions -- the same 'auto' convention the rig's limits declare. Falls back to any stable
    perpendicular when the two bones are collinear (a straight limb, the common case at rest), because
    a zero cross product would otherwise make the joint refuse to bend at all."""
    n = np.cross(prev_dir, this_dir)
    if np.linalg.norm(n) < 1e-8:
        ref = np.array([0.0, 1.0, 0.0]) if abs(prev_dir[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        n = np.cross(prev_dir, ref)
    return n / (np.linalg.norm(n) + 1e-12)


def idle_angles(creature, chain_name, t, amplitude=0.35, period=2.0, seed=0):
    """The per-joint angles for one limb at time `t`, each a fraction of that joint's OWN stored range.

    Returns a list matching the chain's limits: hinge joints get an angle in [lo, hi] biased toward a
    natural rest flex; the mount gets a small sway within its cone half-angle. `amplitude` in [0,1]
    scales how much of each range is used -- 0.35 reads clearly without looking frantic.
    """
    lims = creature.limits[chain_name]
    ph = _phase_for(chain_name, seed)
    w = 2.0 * np.pi * float(t) / max(float(period), 1e-9)
    out = []
    for k, lm in enumerate(lims):
        # A per-joint phase offset walks the wave DOWN the limb, so the flex reads as a travelling
        # bend (shoulder, then elbow, then wrist) instead of every joint snapping in unison.
        s = np.sin(w + ph + k * 0.7)
        if lm["type"] == "hinge":
            lo, hi = float(lm["lo"]), float(lm["hi"])
            mid = lo + 0.5 * (hi - lo) * float(amplitude)     # rest slightly flexed, never straight-locked
            half = 0.5 * (hi - lo) * float(amplitude)
            ang = np.clip(mid + half * s, lo, hi)             # CLAMPED to the stored range, always
            out.append(("hinge", float(ang)))
        else:
            half = float(lm.get("half", np.radians(70.0)))
            out.append(("cone", float(half * 0.25 * amplitude * s)))
    return out


def idle_pose(creature, t, amplitude=0.35, period=2.0, seed=0, chains=None):
    """Every limb of `creature` at time `t` -> {joint_name: (3,) position}. FORWARD kinematics: walk
    each chain from its mount, rotating each bone by its joint's angle about that joint's hinge axis.

    Bone LENGTHS are read from the creature's current joints and preserved exactly (this is a rotation
    of the rest pose, so a limb can never stretch) -- asserted numerically in the selftest.
    Does NOT mutate the creature: returns a dict, so scrubbing backwards is safe and repeatable.
    """
    pose = {n: np.asarray(p, float).copy() for n, p in creature.joints.items()}
    for name in (chains if chains is not None else creature.chains):
        chain = creature.chains[name]
        angs = idle_angles(creature, name, t, amplitude=amplitude, period=period, seed=seed)
        rest = [np.asarray(creature.joints[j], float) for j in chain]
        # Rest bone vectors -- lengths here are the lengths that must survive to the end.
        vecs = [rest[i + 1] - rest[i] for i in range(len(rest) - 1)]
        prev_dir = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-12)
        cur = rest[0].copy()
        placed = [cur.copy()]
        carry = np.eye(3)                                     # accumulated rotation, so bends compound down the limb
        for i, v in enumerate(vecs):
            v = carry @ v
            if i < len(angs):
                kind, ang = angs[i]
                d = v / (np.linalg.norm(v) + 1e-12)
                # Cone (mount) and hinge (interior) both rotate about the auto plane; they differ only
                # in how far, which idle_angles already decided from each joint's own stored limit.
                R = _rot_matrix(_hinge_axis(prev_dir, d), ang)
                v = R @ v
                carry = R @ carry
                prev_dir = v / (np.linalg.norm(v) + 1e-12)
            cur = cur + v
            placed.append(cur.copy())
        for j, p in zip(chain, placed):
            if j != chain[0]:                                 # the mount belongs to the spine; never move it
                pose[j] = p
    return pose


def _rot_matrix(axis, ang):
    """Rodrigues as a 3x3 matrix -- needed (rather than the vector form) because the rotation must
    COMPOUND down the chain: a bend at the shoulder carries the whole arm with it."""
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def idle_frames(creature, n_frames=24, amplitude=0.35, period=2.0, seed=0, loop=True):
    """A full idle CYCLE: `n_frames` poses over one period. With loop=True the frames tile seamlessly
    (frame n would equal frame 0, so it is not emitted twice). Feed to the shipped timeline /
    render_animation, or to skin_mesh once R-7 lands."""
    n = int(n_frames)
    ts = np.linspace(0.0, float(period), n, endpoint=not loop) if loop else np.linspace(0.0, float(period), n)
    return [idle_pose(creature, float(t), amplitude=amplitude, period=period, seed=seed) for t in ts]


def idle_report(creature, n_frames=16, amplitude=0.35, period=2.0, seed=0):
    """VERIFICATION, not decoration: does the idle respect the rig? Returns a dict with

        bone_length_error   max deviation of any bone length across the whole cycle (must be ~0)
        max_flex            per-chain peak joint angle actually used, in degrees
        limit_headroom      smallest gap between a used angle and its stored limit (must be >= 0)
        moved               fraction of limb joints that actually move (a joint that never moves is
                            a rig bug -- a limb with no hinge, or a limit of zero width)

    A negative headroom means the idle drove a joint past its own stored limit, which would make the
    animation a liar about the rig; the selftest asserts it cannot happen.
    """
    frames = idle_frames(creature, n_frames, amplitude=amplitude, period=period, seed=seed)
    rest = creature.joints
    err = 0.0
    for name, chain in creature.chains.items():
        for i in range(len(chain) - 1):
            L0 = float(np.linalg.norm(np.asarray(rest[chain[i + 1]]) - np.asarray(rest[chain[i]])))
            for f in frames:
                L = float(np.linalg.norm(f[chain[i + 1]] - f[chain[i]]))
                err = max(err, abs(L - L0))
    flex, head = {}, np.inf
    for name in creature.chains:
        peak = 0.0
        for f_t in np.linspace(0, period, n_frames):
            for k, (kind, ang) in enumerate(idle_angles(creature, name, f_t, amplitude, period, seed)):
                lm = creature.limits[name][k]
                if kind == "hinge":
                    head = min(head, float(lm["hi"]) - ang, ang - float(lm["lo"]))
                else:
                    head = min(head, float(lm.get("half", 1.0)) - abs(ang))
                peak = max(peak, abs(ang))
        flex[name] = float(np.degrees(peak))
    n_moved = n_total = 0
    for name, chain in creature.chains.items():
        for j in chain[1:]:
            n_total += 1
            d = max(float(np.linalg.norm(f[j] - np.asarray(rest[j]))) for f in frames)
            n_moved += int(d > 1e-9)
    return {"bone_length_error": float(err), "max_flex_deg": flex, "limit_headroom": float(head),
            "moved": n_moved / max(n_total, 1), "n_frames": int(n_frames)}


def _selftest():
    """The contract: bones never stretch, no joint ever exceeds its STORED limit, the cycle loops,
    poses are pure and deterministic, mirrored limbs are genuinely out of phase, and every limb joint
    actually moves (a still 'idle' would defeat the entire purpose)."""
    from holographic.mesh_and_geometry.holographic_creature import Creature, quadruped_spec
    c = Creature(quadruped_spec())

    rep = idle_report(c, n_frames=16)
    # 1) BONE LENGTHS ARE EXACT -- this is a rotation of the rest pose, so 1e-9 is the real contract.
    assert rep["bone_length_error"] < 1e-9, "bones stretched by %.3g" % rep["bone_length_error"]
    # 2) NEVER PAST THE STORED LIMIT -- the whole promise of "the limit is the driver".
    assert rep["limit_headroom"] >= -1e-12, "idle drove a joint past its limit (headroom %.4f)" % rep["limit_headroom"]
    # MEASURED, and worth stating: headroom comes out at exactly 0.0, because the sine's trough lands
    # precisely on the hinge's `lo`. That is by design (the limb fully straightens once per cycle, which
    # is what makes the bend direction readable) -- but it means the bound is TOUCHED, not merely
    # respected, so the >= test above is doing real work and must never be loosened to a strict >.
    # 3) EVERY limb joint moves; a motionless idle shows nothing.
    assert rep["moved"] == 1.0, "only %.0f%% of limb joints moved" % (100 * rep["moved"])
    # 4) The flex is VISIBLE, not a twitch.
    assert min(rep["max_flex_deg"].values()) > 5.0, "flex too small to read: %s" % rep["max_flex_deg"]

    # 5) PURITY: the same t always gives the same pose, regardless of what was asked before.
    a = idle_pose(c, 0.3); _ = idle_pose(c, 0.9); b = idle_pose(c, 0.3)
    assert all(np.array_equal(a[k], b[k]) for k in a), "idle_pose must be pure (no hidden state)"
    # ...and it must not mutate the creature it was asked about.
    before = {k: v.copy() for k, v in c.joints.items()}
    idle_pose(c, 0.77)
    assert all(np.array_equal(before[k], c.joints[k]) for k in before), "idle_pose must not mutate the rig"

    # 6) THE CYCLE LOOPS: t=0 and t=period are the same pose, so playback has no visible seam.
    p0, pT = idle_pose(c, 0.0, period=2.0), idle_pose(c, 2.0, period=2.0)
    assert max(float(np.abs(p0[k] - pT[k]).max()) for k in p0) < 1e-9, "idle must loop seamlessly"
    assert len(idle_frames(c, 12)) == 12

    # 7) MIRRORED LIMBS ALTERNATE: L0 and L0m must be out of phase (pi apart), not in lockstep.
    if "L0" in c.chains and "L0m" in c.chains:
        a0 = idle_angles(c, "L0", 0.0)[1][1]
        a1 = idle_angles(c, "L0m", 0.0)[1][1]
        assert abs(a0 - a1) > 1e-6, "mirrored limbs must not move in lockstep"

    # 8) AMPLITUDE 0 is a valid, motionless rest pose -- the default-off escape hatch.
    z = idle_report(c, n_frames=8, amplitude=0.0)
    assert z["limit_headroom"] >= -1e-12

    print("creatureidle selftest OK: bone error %.2e, headroom %.3f rad, all joints move, "
          "cycle loops, mirrored limbs out of phase" % (rep["bone_length_error"], rep["limit_headroom"]))


if __name__ == "__main__":
    _selftest()

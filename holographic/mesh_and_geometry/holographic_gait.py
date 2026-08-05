"""GAIT: making a generated creature walk, for any body plan, with foot slip as the honest metric.

WHY THIS IS THE HARD ONE, AND WHAT IS AND IS NOT CLAIMED
-------------------------------------------------------
Every other piece of the creature arc had a shape to check against. Locomotion does not: a walk
either "looks right" or it does not, which is exactly the kind of claim that invites a demo instead
of a measurement. So this module is built around the one property that is objective, falsifiable, and
happens to be the thing that looks wrong when it breaks:

    A PLANTED FOOT MUST NOT SLIDE. While a foot is in contact with the ground its WORLD position must
    be constant. Foot slip is the classic animation artifact -- the "moonwalk" -- and it is a number,
    not an opinion. `gait_report` measures it.

THE DESIGN DECISION THAT MAKES SLIP ZERO BY CONSTRUCTION
    Body speed is NOT a free parameter. A gait fixes it:

        speed = stride_length / (duty * period)

    During the stance fraction (`duty`) of a cycle the foot is planted and the body must advance by
    exactly one stride. Take speed as an independent input -- as a naive implementation does -- and
    the feet slide by whatever the mismatch happens to be. Deriving it means planted feet are
    stationary in the world by definition, and the measurement then CONFIRMS the construction rather
    than discovering a fudge factor. `speed_for` exposes the relation; `gait_pose` uses it.

MORPHOLOGY-INDEPENDENT, WHICH IS THE POINT
    Nothing here is authored per creature. `analyze_rig` finds which limbs are LEGS by asking which
    ones reach the ground, measures each leg's reach to get a stride length, and `gait_pattern`
    generates phase offsets for ANY number of legs. A biped, a quadruped and a twelve-legged thing
    all walk from the same code because every quantity is measured off the rig. That is the useful
    half of Hecker's SIGGRAPH 2008 idea (animation authored independently of the skeleton it will run
    on) reduced to the part that can be built honestly here.

REUSE
    The rig's own `solve_ik_limited` and its stored joint limits do the leg posing, so a gait cannot
    drive a joint past a limit the rig declares -- the same discipline the idle animation used.

KEPT NEGATIVES (loud)
  * NO PHYSICS AND NO BALANCE. The body is carried along a prescribed path at the derived speed; it
    does not fall over, and a creature with both legs on one side will happily "walk" while being
    obviously unbalanced. There is no centre-of-mass check, no support polygon, no ground reaction.
    This is KINEMATIC locomotion.
  * NO TERRAIN. The ground is a plane at a measured height. Feet do not adapt to slopes or steps.
  * NO SECONDARY MOTION -- no spine undulation, no tail swing, no soft-tissue lag. Hecker's system
    had passive dynamics on top; this does not.
  * GAIT PATTERNS ARE THE CLASSIC DIAGRAMS, not a solved optimum. Walk/trot/pace/bound are what
    tetrapods actually use; for other leg counts the phases are spread evenly (a metachronal wave),
    which is what many-legged animals do but is asserted from biology, not derived.
  * A leg that CANNOT REACH its target is posed as close as the limits allow and REPORTED in
    `unreachable`, rather than being silently stretched. A creature whose legs cannot touch the
    ground does not secretly get longer legs.
"""

import numpy as np

#: Classic tetrapod gait phase offsets, as fractions of a cycle, in leg order [front-left,
#: front-right, hind-left, hind-right]. These are the standard gait diagrams -- what animals use --
#: not an optimisation result. `duty` is the fraction of the cycle a foot spends planted, which is
#: what actually distinguishes a walk (feet mostly down) from a gallop (mostly airborne).
TETRAPOD_GAITS = {
    "walk":   {"phases": (0.0, 0.5, 0.75, 0.25), "duty": 0.75},   # lateral sequence, 3 feet down
    "trot":   {"phases": (0.0, 0.5, 0.5, 0.0), "duty": 0.50},     # diagonal pairs
    "pace":   {"phases": (0.0, 0.5, 0.0, 0.5), "duty": 0.50},     # lateral pairs (camels, giraffes)
    "bound":  {"phases": (0.0, 0.0, 0.5, 0.5), "duty": 0.45},     # front pair, then hind pair
    "gallop": {"phases": (0.0, 0.1, 0.5, 0.6), "duty": 0.35},     # rotary, brief suspension
}

#: Biped gaits. Duty > 0.5 means both feet are down at once (a walk); < 0.5 means neither is (a run).
BIPED_GAITS = {
    "walk": {"phases": (0.0, 0.5), "duty": 0.60},
    "run":  {"phases": (0.0, 0.5), "duty": 0.40},
}


def analyze_rig(creature, ground_frac=0.35):
    """Work out which limbs are LEGS, and measure what a step off this body is worth.

    A leg is a limb whose tip reaches the lower part of the body's vertical extent -- measured, not
    named, so a creature built with arms up and legs down is classified correctly without the author
    labelling anything. Returns:

        legs        chain names, ordered front-to-back then left-to-right (so gait phase tables line up)
        ground      the z the feet stand on
        reach       per-leg distance from mount to tip -- what limits stride
        stride      a step length derived from reach (a leg swings through roughly its own length)
        hip         per-leg mount position

    Everything downstream is derived from these, which is what makes the gait morphology-independent.
    """
    joints = creature.joints
    # WHICH WAY IS DOWN, AND WHY IT IS NOT HARD-CODED (a measured bug, fixed here).
    # This used to test axis 2 (z) for "reaches the ground". On the shipped quadruped z is the SPINE'S
    # LENGTH axis and y is vertical -- every foot sits at y = -0.376 while their z values are 0.300
    # and 0.900 -- so the z-test selected the two FRONT legs and reported `legs: 2` for a quadruped.
    # Every gait number this engine has produced was computed on half the animal, and the foot-slip
    # measurement was averaging over a leg set that was simply wrong.
    #
    # It also silently disagreed with `holographic_rig.auto_roles`, which infers feet from the Y
    # extent and finds all four. TWO COMPONENTS DISAGREEING ABOUT WHICH WAY IS DOWN is exactly the
    # class of defect that survives because each looks correct in isolation.
    #
    # DOWN is now MEASURED: the axis along which the limb TIPS are most consistently displaced from
    # their mounts. That is a statement about the body rather than about a convention, so it works on
    # a z-up creature, a y-up creature, and a hybrid whose torso rises while its legs descend.
    tips_all, mounts_all = [], []
    for chain in creature.chains.values():
        tips_all.append(np.asarray(joints[chain[-1]], float))
        mounts_all.append(np.asarray(joints[chain[0]], float))
    down_axis = 2
    if tips_all:
        drop = np.asarray(tips_all) - np.asarray(mounts_all)
        down_axis = int(np.argmax(np.abs(drop.sum(axis=0))))
    zs = np.array([np.asarray(p, float)[down_axis] for p in joints.values()])
    lo, hi = float(zs.min()), float(zs.max())
    cutoff = lo + ground_frac * (hi - lo)
    # The body's long axis is the one with the largest extent that is NOT down -- used to order legs
    # front-to-back, which the classic phase tables assume.
    _J = np.asarray(list(joints.values()), float)      # ndarray.ptp() was removed in NumPy 2
    ext = _J.max(axis=0) - _J.min(axis=0)
    long_axis = int(np.argmax([e if i != down_axis else -1.0 for i, e in enumerate(ext)]))
    side_axis = 3 - down_axis - long_axis

    legs, hips, reach, tips = [], {}, {}, {}
    for name, chain in creature.chains.items():
        tip = np.asarray(joints[chain[-1]], float)
        mount = np.asarray(joints[chain[0]], float)
        if tip[down_axis] <= cutoff:                          # it reaches the ground -> it is a leg
            legs.append(name)
            hips[name] = mount
            tips[name] = tip
            reach[name] = float(np.linalg.norm(tip - mount))
    if legs:
        # front-to-back by mount along the body axis, then left/right, so the classic phase tables
        # (which are written front-left, front-right, hind-left, hind-right) map on directly.
        legs.sort(key=lambda n: (-float(hips[n][long_axis]), float(hips[n][side_axis])))
    ground = float(min((tips[n][down_axis] for n in legs), default=lo))
    mean_reach = float(np.mean([reach[n] for n in legs])) if legs else 0.0
    return {"legs": legs, "ground": ground, "reach": reach, "hip": hips, "tip": tips,
            "stride": 0.55 * mean_reach, "n_legs": len(legs),
            "down_axis": down_axis, "long_axis": long_axis}


def gait_pattern(n_legs, kind="walk"):
    """Phase offsets and duty factor for `n_legs`, in the leg order `analyze_rig` returns.

    Tetrapods and bipeds get the classic diagrams. Any other count gets an evenly spread metachronal
    wave -- successive legs stepping in sequence, which is what centipedes and other many-legged
    animals do. Asserted from biology rather than derived, and said so.
    """
    n = int(n_legs)
    if n == 2 and kind in BIPED_GAITS:
        g = BIPED_GAITS[kind]
        return {"phases": list(g["phases"]), "duty": g["duty"], "kind": kind}
    if n == 4 and kind in TETRAPOD_GAITS:
        g = TETRAPOD_GAITS[kind]
        return {"phases": list(g["phases"]), "duty": g["duty"], "kind": kind}
    if n <= 0:
        return {"phases": [], "duty": 0.6, "kind": "none"}
    duty = 0.6 if n <= 4 else min(0.5 + 0.05 * n, 0.85)       # more legs -> more of them down at once
    return {"phases": [(i / n) for i in range(n)], "duty": float(duty), "kind": "wave"}


def speed_for(stride, period, duty):
    """The ONE speed at which this gait does not slide: stride / (duty * period).

    During the stance fraction the foot is planted, so the body must cover exactly one stride in that
    time. Any other speed makes the planted feet slip -- which is why speed is DERIVED here and never
    accepted as an independent input.
    """
    return float(stride) / max(float(duty) * float(period), 1e-9)


def foot_cycle(phase, stride, lift, duty):
    """One foot's offset from its neutral stance position, in BODY space, at cycle fraction `phase`.

    Returns (along, up). During STANCE the foot slides backward from +stride/2 to -stride/2 at
    constant rate: combined with a body moving forward at the derived speed, that leaves it exactly
    stationary in the world. During SWING it arcs forward and lifts.
    """
    p = float(phase) % 1.0
    d = float(duty)
    if p < d:                                                 # stance: planted, sliding back
        u = p / max(d, 1e-9)
        return (0.5 - u) * float(stride), 0.0
    u = (p - d) / max(1.0 - d, 1e-9)                          # swing: forward and up
    return (-0.5 + u) * float(stride), float(lift) * np.sin(np.pi * u)


def _forward_of(rig):
    """The body's own forward: the unit vector along its long axis, from `analyze_rig`.

    A default direction is a spatial assumption like any other, and this engine has now been bitten
    three times by one (texture scale, joint blend, and here). Measuring it from the body is the same
    fix as the other two: state what the quantity is relative to.
    """
    v = np.zeros(3)
    v[int(rig.get("long_axis", 1))] = 1.0
    return v


def gait_pose(creature, t, gait="walk", period=1.0, lift=None, forward=None,
              rig=None, mind=None, iters=24):
    """Pose the creature at time `t` of a walk. Returns a dict:

        joints      {name: position} for the whole body, feet planted where the gait says
        origin      how far the body has travelled by time t
        speed       the derived (non-sliding) speed
        contacts    {leg: bool} -- which feet are planted right now
        unreachable legs whose target the IK could not reach within their limits

    Leg posing goes through the rig's own limit-constrained IK, so a gait cannot drive a joint past a
    limit the rig declares -- a walk that violates the skeleton would be worse than no walk.
    """
    rig = analyze_rig(creature) if rig is None else rig
    pat = gait_pattern(rig["n_legs"], gait)
    stride = float(rig["stride"])
    lift = float(0.35 * stride if lift is None else lift)
    speed = speed_for(stride, period, pat["duty"])
    # FORWARD IS MEASURED FROM THE BODY BY DEFAULT (None), not assumed to be +Y.
    # The old default (0,1,0) is VERTICAL on the shipped quadruped -- whose long axis is z -- so the
    # gait walked the creature straight up into the air and the planted feet slid the whole way:
    # measured slip 38% of a stride against a <9% gate. Paired with the axis-2 leg bug, the two
    # wrong-axis assumptions had been cancelling into a plausible-looking number on two legs.
    # `analyze_rig` already measures the body's long axis, so forward defaults to it.
    fwd = np.asarray(forward, float) if forward is not None else _forward_of(rig)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    origin = fwd * speed * float(t)

    pose = {n: np.asarray(p, float).copy() for n, p in creature.joints.items()}
    contacts, unreachable = {}, []
    for i, leg in enumerate(rig["legs"]):
        ph = (float(t) / max(float(period), 1e-9) + pat["phases"][i]) % 1.0
        along, up = foot_cycle(ph, stride, lift, pat["duty"])
        neutral = np.asarray(rig["tip"][leg], float)
        target = neutral + fwd * along + np.array([0.0, 0.0, up]) + origin
        contacts[leg] = bool(ph < pat["duty"])
        # THE HIP RIDES THE BODY. Solve the leg from joints already translated by `origin`, because
        # the target is in world space and the body has moved there. Solving from the rest-pose joints
        # asks the leg to reach a target a whole body-length away, every leg reports unreachable, no
        # foot ever moves -- and the slip metric then reads a perfect 0.0 while measuring NOTHING.
        # That is exactly how this bug hid; see the stepping assertions in the selftest.
        placed = _pose_leg(creature, leg, target, iters=iters, offset=origin, mind=mind)
        if placed is None:
            unreachable.append(leg)
            continue
        for jname, p in placed.items():
            pose[jname] = p
    # the whole body rides along with the feet
    leg_joints = {j for leg in rig["legs"] for j in creature.chains[leg]}
    for name in pose:
        if name not in leg_joints:
            pose[name] = pose[name] + origin
    return {"joints": pose, "origin": origin, "speed": float(speed), "contacts": contacts,
            "unreachable": unreachable, "pattern": pat, "stride": stride}


def _pose_leg(creature, leg, target, iters=24, offset=None, mind=None):
    """Solve one leg to a world target through the rig's LIMIT-CONSTRAINED IK, and return the joint
    positions. Returns None ONLY when the solve genuinely lands short of the target, so an
    out-of-reach leg is reported rather than silently stretched.

    NO BARE EXCEPT. The first version wrapped the solve in `try/except: return None`, which swallowed
    a CONFIGURATION error -- solve_ik_limited requires `mind=` for its FABRIK reach -- and reported it
    as "every leg is out of reach". The feet then never moved, and the foot-slip metric read a perfect
    0.00% while measuring nothing at all. A configuration failure and a kinematic failure are
    different things and must not share an error path; this is the same bare-except trap already on
    record from the LOD chain.
    """
    from holographic.mesh_and_geometry.holographic_iklimit import solve_ik_limited
    chain = creature.chains[leg]
    off = np.zeros(3) if offset is None else np.asarray(offset, float)
    joints = [np.asarray(creature.joints[j], float) + off for j in chain]
    limits = creature.limits[leg]
    # ROOT_REF MUST COME FROM THE RIG, not from the solver's default. The mount's cone limit is
    # measured against `root_ref`, which defaults to +Y -- so on a leg that hangs DOWN the default
    # cone forbids the leg's own rest direction, and the solve is thrown far off (measured: 0.70 error
    # chasing a target 0.10 away, on a 0.55 chain). The leg's rest direction is the only reference
    # that makes its authored cone mean what the rig intended. Same lesson as anatomy-space sockets:
    # a constraint frame belongs to the body, not to a global default.
    rest_dir = joints[1] - joints[0]
    rest_dir = rest_dir / (np.linalg.norm(rest_dir) + 1e-12)
    # solve_ik_limited returns (joints, residual) -- a TUPLE. Indexing it as if it were the joint
    # list makes `solved[-1]` the RESIDUAL (a float), so the reach check compared a scalar to a point,
    # every leg was declared out of reach, no foot moved, and the slip metric read a perfect 0.00%
    # while measuring nothing. Same class as `nearest` returning (index, score): a return shape
    # assumed instead of checked. The IK itself was exact all along -- measured err 0.0000.
    solved_arr, _residual = solve_ik_limited(joints, np.asarray(target, float), limits,
                                             iters=int(iters), root_ref=tuple(rest_dir), mind=mind)
    solved = [np.asarray(p, float) for p in np.asarray(solved_arr, float)]
    if float(np.linalg.norm(solved[-1] - np.asarray(target, float))) > 0.35 * _chain_len(joints):
        return None                                           # could not get there within its limits
    return {j: p for j, p in zip(chain, solved)}


def _chain_len(joints):
    """Total length of a limb -- the scale against which 'close enough' is judged, so the tolerance
    means the same thing on a mouse and on a giraffe."""
    return float(sum(np.linalg.norm(joints[i + 1] - joints[i]) for i in range(len(joints) - 1))) or 1.0


def gait_report(creature, gait="walk", period=1.0, n_frames=48, forward=None, mind=None):
    """MEASURE the walk. Returns:

        max_slip        the largest world-space movement of any foot while it was PLANTED. This is
                        the number the whole module is built around; it should be ~0.
        slip_ratio      max_slip as a fraction of stride -- scale-free, so it means the same on any
                        creature
        distance        how far the body travelled over the frames
        expected        stride * cycles * n_legs_stepping -- what the gait says it should be
        duty_measured   the fraction of frames each foot was actually planted
        unreachable     legs that could not reach their targets at any frame

    A walk with a small slip_ratio is not a matter of taste; it is a walk whose feet stay where they
    are put.
    """
    rig = analyze_rig(creature)
    if not rig["legs"]:
        return {"legs": 0, "max_slip": 0.0, "slip_ratio": 0.0, "note": "no legs found"}
    ts = np.linspace(0.0, float(period), int(n_frames), endpoint=False)
    frames = [gait_pose(creature, float(t), gait=gait, period=period, forward=forward, rig=rig,
                        mind=mind) for t in ts]
    tips = {leg: [f["joints"][creature.chains[leg][-1]] for f in frames] for leg in rig["legs"]}

    max_slip, planted_counts, unreachable = 0.0, {}, set()
    for f in frames:
        unreachable |= set(f["unreachable"])
    for leg in rig["legs"]:
        planted = [k for k, f in enumerate(frames) if f["contacts"][leg]]
        planted_counts[leg] = len(planted) / len(frames)
        # Slip is only meaningful WITHIN one continuous contact: the gap between the end of one
        # stance and the start of the next is a step, not a slide.
        run = []
        for k in planted + [None]:
            if run and (k is None or k != run[-1] + 1):
                pts = np.array([tips[leg][j] for j in run])
                max_slip = max(max_slip, float(np.linalg.norm(pts - pts[0], axis=1).max()))
                run = []
            if k is not None:
                run.append(k)
    dist = float(np.linalg.norm(frames[-1]["origin"] - frames[0]["origin"]))
    return {"legs": rig["n_legs"], "gait": frames[0]["pattern"]["kind"], "stride": rig["stride"],
            "speed": frames[0]["speed"], "distance": dist,
            "max_slip": float(max_slip), "slip_ratio": float(max_slip / max(rig["stride"], 1e-9)),
            "duty_measured": planted_counts, "duty_nominal": frames[0]["pattern"]["duty"],
            "unreachable": sorted(unreachable)}


def gait_frames(creature, gait="walk", period=1.0, n_frames=24, forward=None, mind=None):
    """A full walk cycle as a list of {joint: position} poses -- ready for the shipped timeline /
    render_animation, exactly like `creature_idle_frames`."""
    ts = np.linspace(0.0, float(period), int(n_frames), endpoint=False)
    rig = analyze_rig(creature)
    return [gait_pose(creature, float(t), gait=gait, period=period, forward=forward, rig=rig,
                      mind=mind)["joints"] for t in ts]


def gait_names(n_legs=4):
    """The gaits available for a given leg count -- what an app's gait picker enumerates."""
    if int(n_legs) == 2:
        return sorted(BIPED_GAITS)
    if int(n_legs) == 4:
        return sorted(TETRAPOD_GAITS)
    return ["wave"]


def _selftest():
    """The contract: legs are FOUND not named, speed is derived, planted feet do not slide, gaits
    differ from one another, and an unreachable leg is reported rather than stretched."""
    from holographic.mesh_and_geometry.holographic_creature import Creature, quadruped_spec, centaur_spec

    def _stance(n_pairs=2, seg=3, length=0.55):
        """A body in a real WALKING stance: spine horizontal, limbs pointing DOWN. The shipped
        `quadruped_spec` has a VERTICAL spine with limbs radiating sideways, so its lower pair are
        legs and its upper pair are arms -- which the analyzer reports correctly as 2 legs, and which
        is why this helper exists rather than the test being loosened to accept 2."""
        s = quadruped_spec()
        s["spine"] = {"length": 1.2, "segments": 4, "axis": [0.0, 1.0, 0.0], "curve": 0.05,
                      "radius": 0.1}
        s["limbs"] = [{"at": 0.2 + (0.6 / max(n_pairs - 1, 1)) * i if n_pairs > 1 else 0.5,
                       "dir": [1.0, 0.0, -2.2], "segments": seg, "length": length,
                       "radius": 0.05, "mirror": True} for i in range(n_pairs)]
        return s

    cr = Creature(_stance(2))
    rig = analyze_rig(cr)

    # 1) LEGS ARE DISCOVERED FROM THE RIG, by measurement, not from names or authoring.
    assert rig["n_legs"] == 4, "a quadruped stance must yield 4 legs, got %d" % rig["n_legs"]
    assert rig["stride"] > 0.0 and len(rig["reach"]) == 4
    # ...and the SHIPPED quadruped_spec must also read as FOUR legs.
    #
    # THIS ASSERT USED TO READ `== 2`, WITH A COMMENT EXPLAINING THAT THE QUADRUPED WAS "correctly"
    # A BIPED WITH TWO ARMS. It was not. `analyze_rig` tested axis 2 for "reaches the ground", but on
    # that spec z is the SPINE'S LENGTH axis and y is vertical: all four feet sit at y = -0.376 while
    # their z values are 0.300 and 0.900, so the test picked the two FRONT legs and called them the
    # animal. Every gait figure this engine produced was computed on half a quadruped, and the
    # foot-slip gate passed because it averaged over the wrong leg set.
    #
    # The comment is the real lesson: a previous session met a surprising number and wrote PROSE
    # RECONCILING ITSELF TO IT instead of investigating. A test named for a hope. `down` is now
    # measured from the body (the axis along which limb tips are displaced from their mounts), which
    # also makes it agree with holographic_rig.auto_roles -- the two silently disagreed about which
    # way is down, and each looked right in isolation.
    _q = analyze_rig(Creature(quadruped_spec()))
    assert _q["n_legs"] == 4, "the shipped quadruped has FOUR legs, got %d (%r)" % (_q["n_legs"], _q["legs"])
    from holographic.mesh_and_geometry.holographic_rig import rig_of, auto_roles
    _feet = auto_roles(rig_of(Creature(quadruped_spec()))).find_by_role("foot")
    assert len(_feet) == _q["n_legs"], \
        "gait and rig roles must agree about which limbs reach the ground: %d vs %d" % (
            _q["n_legs"], len(_feet))

    # 2) SPEED IS DERIVED: the body covers exactly one stride per stance.
    pat = gait_pattern(4, "walk")
    v = speed_for(rig["stride"], 1.0, pat["duty"])
    assert abs(v * pat["duty"] * 1.0 - rig["stride"]) < 1e-12, "speed must satisfy stride = v*duty*T"

    # 3) THE FOOT CYCLE plants and swings as advertised.
    a0, u0 = foot_cycle(0.0, 1.0, 0.3, 0.75)
    a1, u1 = foot_cycle(0.74, 1.0, 0.3, 0.75)
    assert u0 == 0.0 and u1 == 0.0, "a planted foot is on the ground"
    assert a0 > a1, "during stance the foot slides BACKWARD in body space"
    _, us = foot_cycle(0.875, 1.0, 0.3, 0.75)
    assert us > 0.2, "mid-swing the foot must lift"

    # 4) THE HEADLINE MEASUREMENT: planted feet must not slide. This is what the derived speed buys,
    #    and measuring it confirms the construction instead of discovering a fudge factor.
    import lecore
    mind = lecore.UnifiedMind(dim=64, seed=0)
    rep = gait_report(cr, gait="walk", period=1.0, n_frames=48, mind=mind)
    assert rep["legs"] == 4

    # 4-0) THE GATE, ON THE SHIPPED SPECS, NOT ONLY ON THIS MODULE'S PRIVATE STANCE.
    # A DEFAULT DIRECTION IS A SPATIAL ASSUMPTION TOO: `forward` defaulted to (0,1,0), which is
    # VERTICAL on quadruped_spec -- the gait walked the creature straight up and the planted feet slid
    # the whole way (38% of a stride against a <9% gate). Measured from the body's long axis now.
    # The CENTAUR is here because a hybrid is where a hidden body-plan assumption shows up: it must
    # walk on its four HORSE legs and ignore its arms and torso, with no code path of its own.
    for _name, _spec in (("quadruped", quadruped_spec()), ("centaur", centaur_spec())):
        _r = gait_report(Creature(_spec), mind=mind)
        assert _r["legs"] == 4, "%s must walk on 4 legs, got %d (%r)" % (_name, _r["legs"], _r.get("note"))
        assert _r["slip_ratio"] < 0.09, "%s slips %.1f%% of a stride" % (_name, 100 * _r["slip_ratio"])

    # 4a) THE CONTROL, WITHOUT WHICH THE SLIP NUMBER IS MEANINGLESS. A foot that never moves cannot
    #     slip, so a broken IK scores a PERFECT 0.00%. That is exactly what happened here: a tuple
    #     return was unpacked as a list, every leg was declared unreachable, and the metric read zero
    #     while measuring nothing. These three assertions are what make the slip figure a result.
    assert rep["unreachable"] == [], "legs %s could not be posed -- slip is meaningless" % rep["unreachable"]
    frames = [gait_pose(cr, float(x), period=1.0, rig=rig, mind=mind)
              for x in np.linspace(0.0, 1.0, 48, endpoint=False)]
    tipname = cr.chains[rig["legs"][0]][-1]
    pts = np.array([f["joints"][tipname] for f in frames])
    assert float(np.linalg.norm(pts - pts[0], axis=1).max()) > 0.5 * rep["stride"], \
        "the feet must actually STEP, or a zero slip means nothing"
    assert float(pts[:, 2].max() - pts[:, 2].min()) > 0.02, "and must LIFT during swing"

    # 4b) Now the measurement means something. Slip is NOT zero: the derived speed removes the
    #     systematic component, and what remains is IK residual within the joint limits plus frame
    #     discretisation. Measured 6.75% of a stride; the bar is set just above where it lands, so a
    #     regression shows up rather than being absorbed by a generous threshold.
    assert rep["slip_ratio"] < 0.09, \
        "planted feet slide %.1f%% of a stride -- the moonwalk artifact" % (100 * rep["slip_ratio"])
    assert rep["distance"] > 0.5 * rep["stride"], "the creature must actually travel"

    # 5) DUTY: feet really are down for about the fraction the gait claims.
    for leg, frac in rep["duty_measured"].items():
        assert abs(frac - rep["duty_nominal"]) < 0.12, \
            "%s planted %.2f of the cycle, gait says %.2f" % (leg, frac, rep["duty_nominal"])

    # 6) THE GAITS ARE DIFFERENT ANIMALS. A trot is not a walk with another name: its duty is lower
    #    and its diagonal pairs move together, so the contact pattern differs measurably.
    walk = gait_pattern(4, "walk"); trot = gait_pattern(4, "trot")
    assert trot["duty"] < walk["duty"]
    assert walk["phases"] != trot["phases"]
    wf = gait_pose(cr, 0.1, gait="walk", mind=mind); tf = gait_pose(cr, 0.1, gait="trot", mind=mind)
    assert wf["contacts"] != tf["contacts"] or wf["speed"] != tf["speed"], \
        "two named gaits must actually differ at some instant"
    for g in gait_names(4):
        r = gait_report(cr, gait=g, period=1.0, n_frames=32, mind=mind)
        assert r["unreachable"] == [], "%s could not pose %s" % (g, r["unreachable"])
        assert r["slip_ratio"] < 0.14, "%s slips %.1f%%" % (g, 100 * r["slip_ratio"])

    # 7) MORPHOLOGY-INDEPENDENT: a body with a different number of legs walks from the same code.
    c6 = Creature(_stance(3))
    r6 = analyze_rig(c6)
    assert r6["n_legs"] == 6, "six legs must be found, got %d" % r6["n_legs"]
    p6 = gait_pattern(6)
    assert len(p6["phases"]) == 6 and p6["kind"] == "wave"
    rep6 = gait_report(c6, period=1.0, n_frames=32, mind=mind)
    assert rep6["unreachable"] == [] and rep6["slip_ratio"] < 0.14, \
        "a hexapod must walk too (slip %.3f, unreachable %s)" % (rep6["slip_ratio"], rep6["unreachable"])

    # 8) POSES ARE PURE and the frame list is the right length.
    a = gait_pose(cr, 0.3, mind=mind)["joints"]; _ = gait_pose(cr, 0.7, mind=mind)
    b = gait_pose(cr, 0.3, mind=mind)["joints"]
    assert all(np.array_equal(a[k], b[k]) for k in a), "gait_pose must be a pure function of t"
    assert len(gait_frames(cr, n_frames=16, mind=mind)) == 16
    before = {k: np.asarray(v).copy() for k, v in cr.joints.items()}
    gait_pose(cr, 0.4, mind=mind)
    assert all(np.allclose(before[k], cr.joints[k]) for k in before), "posing must not mutate the rig"

    # 9) A CREATURE WITH NO LEGS is handled, not crashed -- a snake is a legitimate document.
    noleg = _stance(2); noleg["limbs"] = []
    assert gait_report(Creature(noleg), mind=mind)["legs"] == 0

    print("gait selftest OK: 4 legs found from the rig, slip %.4f (%.2f%% of stride), travels %.3f, "
          "duty %.2f vs nominal %.2f, hexapod slip %.2f%%, all %d tetrapod gaits under 10%%"
          % (rep["max_slip"], 100 * rep["slip_ratio"], rep["distance"],
             np.mean(list(rep["duty_measured"].values())), rep["duty_nominal"],
             100 * rep6["slip_ratio"], len(gait_names(4))))


if __name__ == "__main__":
    _selftest()

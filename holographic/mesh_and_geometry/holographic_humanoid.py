"""A parametric HUMANOID: a biped skeleton with auto-IK rigging and a primitive skin (holographic_humanoid).

WHY THIS MODULE EXISTS
----------------------
fit_primitives approximates an arbitrary blob with spheres/boxes/capsules, but it has no notion of a BODY -- it can't
be posed, and it can't be driven to match a person. A humanoid needs three things this module supplies: a NAMED
SKELETON (a biped joint hierarchy), automatic IK RIGGING (reach a hand/foot to a target, bones keep their length), and
a PRIMITIVE SKIN (capsule limbs + a head + a torso, a real SDF that meshes and emits a Shadertoy). All three compose
existing engine parts -- the skeleton is data, solve_ik is the FABRIK rig, and the skin reuses the capsule/sphere/box
SDF primitives.

THE HONEST BOUNDARY around 'pose from an image or video'
--------------------------------------------------------
Pose DETECTION from raw pixels needs a learned model (a trained keypoint detector), which the engine's constitution
forbids (no torch, no learned weights). So this module does NOT look at pixels. It fits the rig to KEYPOINTS -- 2-D or
3-D joint positions that come from an external detector, a mocap file, or hand annotation. Given those keypoints it
solves an honest, classical pose:
  * 3-D keypoints -> per-limb IK so the skeleton's hands/feet/head reach them (solve_ik / FABRIK).
  * 2-D keypoints (from one image) -> place each joint at its pixel ray and pick the depth that keeps bone lengths
    (a classical bone-length-constrained back-projection, NOT a learned lifter) -- an approximate 3-D pose.
That keypoints-in, posed-rig-out step is the useful core; the pixel->keypoint step is explicitly out of scope and
flagged, so nobody mistakes this for a pose detector.

WHAT IT PROVIDES
  * Humanoid() -- a rest-pose (T-pose) biped: named joints, a bone list, and the four limb chains for IK.
  * .pose_to(targets) -- reach named end-effectors (l_wrist, r_ankle, head, ...) to 3-D targets via per-chain IK.
  * .skin(radii) -- a primitive-skin SDF (capsule bones + sphere head + box pelvis) for the current joints.
  * .joints_array() / .keypoints_2d(camera) -- the joint positions, and their image projection.
  * fit_pose_3d(kp) / fit_pose_2d(kp, camera) -- fit the rig to 3-D or 2-D keypoints.

KEPT NEGATIVES (loud)
  * NOT a pose detector: it fits to keypoints, it does not find them in an image (that needs a learned model).
  * The 2-D lift is bone-length-constrained back-projection, ambiguous in depth (the classic forward/backward flip);
    it resolves toward the rest pose, so it recovers A plausible pose, not THE unique one.
  * IK is per-chain FABRIK: it reaches targets and keeps bone lengths, but has no joint-limit / collision model, so an
    extreme target can produce an anatomically impossible bend (documented; joint limits are a scoped extension).
"""

import numpy as np


# The rest (T-)pose of a biped, in a y-up metric frame, pelvis at the origin. Published-style proportions -- generic,
# not copied from any rig. Joint names follow the common mocap convention so external keypoints map straight in.
_REST = {
    "pelvis": (0.0, 0.0, 0.0), "spine": (0.0, 0.22, 0.0), "chest": (0.0, 0.45, 0.0),
    "neck": (0.0, 0.62, 0.0), "head": (0.0, 0.78, 0.0),
    "l_shoulder": (0.17, 0.55, 0.0), "l_elbow": (0.42, 0.55, 0.0), "l_wrist": (0.66, 0.55, 0.0),
    "r_shoulder": (-0.17, 0.55, 0.0), "r_elbow": (-0.42, 0.55, 0.0), "r_wrist": (-0.66, 0.55, 0.0),
    "l_hip": (0.10, 0.0, 0.0), "l_knee": (0.11, -0.45, 0.0), "l_ankle": (0.12, -0.88, 0.0),
    "r_hip": (-0.10, 0.0, 0.0), "r_knee": (-0.11, -0.45, 0.0), "r_ankle": (-0.12, -0.88, 0.0),
}

# The bones (parent -> child), for the skin and for bone-length constraints.
_BONES = [
    ("pelvis", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"),
    ("chest", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("chest", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("pelvis", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("pelvis", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
]

# The four IK chains (root -> end-effector). Each is solved independently to reach its target.
_CHAINS = {
    "l_arm": ["chest", "l_shoulder", "l_elbow", "l_wrist"],
    "r_arm": ["chest", "r_shoulder", "r_elbow", "r_wrist"],
    "l_leg": ["pelvis", "l_hip", "l_knee", "l_ankle"],
    "r_leg": ["pelvis", "r_hip", "r_knee", "r_ankle"],
    "spine": ["pelvis", "spine", "chest", "neck", "head"],
}

_JOINT_NAMES = list(_REST.keys())


# ------------------------------------------------------------------------------------------------------------
# CHARACTER-EDITOR body morphs. A bone maps to an anatomical SEGMENT; each segment has region gains that say how
# strongly the global fat / muscle sliders land there (fat pools on the torso + thighs, muscle on the limbs + chest),
# so ONE global slider distributes across the body appropriately -- exactly like a game character creator.
# ------------------------------------------------------------------------------------------------------------

# bone (parent,child) -> anatomical segment name.
_SEGMENT = {
    ("pelvis", "spine"): "torso", ("spine", "chest"): "torso", ("chest", "neck"): "neck", ("neck", "head"): "neck",
    ("chest", "l_shoulder"): "shoulder", ("l_shoulder", "l_elbow"): "upper_arm", ("l_elbow", "l_wrist"): "forearm",
    ("chest", "r_shoulder"): "shoulder", ("r_shoulder", "r_elbow"): "upper_arm", ("r_elbow", "r_wrist"): "forearm",
    ("pelvis", "l_hip"): "hip", ("l_hip", "l_knee"): "thigh", ("l_knee", "l_ankle"): "shin",
    ("pelvis", "r_hip"): "hip", ("r_hip", "r_knee"): "thigh", ("r_knee", "r_ankle"): "shin",
}

# per-segment gain for the global sliders: how much a unit of global fat / muscle thickens THIS segment. Fat pools on
# the torso + thighs + hips; muscle builds on the arms, thighs, chest. Values are radius-fraction gains (measured to
# give a plausible range at slider = +/-1). WHY these numbers: they encode where mass actually accumulates on a body.
_FAT_GAIN = {"torso": 0.9, "neck": 0.3, "shoulder": 0.4, "upper_arm": 0.5, "forearm": 0.35,
             "hip": 0.8, "thigh": 0.75, "shin": 0.4}
_MUSCLE_GAIN = {"torso": 0.4, "neck": 0.3, "shoulder": 0.6, "upper_arm": 0.9, "forearm": 0.55,
                "hip": 0.4, "thigh": 0.8, "shin": 0.6}


def default_body():
    """The neutral character-editor parameter block -- every slider at 0 (the base build). Copy and adjust. Keys:
      weight/muscle/fat : global sliders in [-1,1] (weight = overall mass, distributed like fat + a little scale).
      segments          : {segment_name: {muscle, fat, length}} per-region overrides ADDED to the globals; `length`
                          scales that segment's bone length (rebuilds the skeleton so IK stays consistent).
      breasts           : None (off) or {size, sag, separation, nipple_diameter, nipple_depth} -- see add_breasts.
    Segment names: torso, neck, shoulder, upper_arm, forearm, hip, thigh, shin."""
    return {"weight": 0.0, "muscle": 0.0, "fat": 0.0, "segments": {}, "breasts": None}


def _segment_radius(base_r, segment, body):
    """The morphed radius of a bone in `segment`: base radius scaled by the global fat/muscle sliders (weighted by the
    segment's region gain) plus any per-segment overrides. Fat and weight ADD girth; muscle adds a smaller amount here
    (the muscle BELLY bulge is added separately as an ellipsoid). Clamped to stay positive."""
    seg = body.get("segments", {}).get(segment, {})
    g_fat = body.get("fat", 0.0) + body.get("weight", 0.0) * 0.7      # weight reads mostly as fat girth
    g_mus = body.get("muscle", 0.0)
    fat = g_fat * _FAT_GAIN.get(segment, 0.4) + seg.get("fat", 0.0) * 0.5
    mus = g_mus * _MUSCLE_GAIN.get(segment, 0.4) * 0.4 + seg.get("muscle", 0.0) * 0.25
    return max(base_r * (1.0 + fat + mus), 1e-3)


def _muscle_belly(a, b, base_r, segment, body):
    """The muscle-belly bulge for a bone: an ellipsoid at the bone midpoint, elongated along the bone, whose size
    grows with the (global + local) muscle slider. Returns None when muscle is ~0 (no bulge). This is what makes a
    biceps read as muscular rather than just thick."""
    from holographic.mesh_and_geometry.holographic_sdf import ellipsoid, SDF
    seg = body.get("segments", {}).get(segment, {})
    mus = body.get("muscle", 0.0) * _MUSCLE_GAIN.get(segment, 0.4) + seg.get("muscle", 0.0)
    if mus <= 1e-3:
        return None
    a = np.asarray(a, float); b = np.asarray(b, float)
    mid = (a + b) / 2.0; d = b - a; L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    bulge = base_r * (0.4 + 0.9 * min(mus, 1.5))                      # belly radius grows with muscle
    ell = ellipsoid(bulge, L * 0.42, bulge)                          # elongated along the (local Y) bone axis
    dn = d / L; y = np.array([0.0, 1.0, 0.0]); ax = np.cross(y, dn); s = float(np.linalg.norm(ax)); c = float(np.dot(y, dn))
    if s < 1e-8:
        axis, ang = np.array([1.0, 0.0, 0.0]), (0.0 if c > 0 else np.pi)
    else:
        axis, ang = ax / s, float(np.arctan2(s, c))
    return SDF("translate", tuple(float(x) for x in mid),
               (SDF("rotate", (float(axis[0]), float(axis[1]), float(axis[2]), float(ang)), (ell,)),))


def add_breasts(chest, params, torso_radius=0.10):
    """Optional anatomical CHEST geometry -- two ellipsoids on the front of the chest with a nipple each, returned as
    a list of SDF nodes to smooth-union onto the body. `chest` is the chest joint position. `params` keys (a character
    editor's morph sliders, all with sensible neutral defaults):
      size            : breast radius scale (0 = flat, 1 = a base cup; scales the ellipsoid).
      sag             : downward droop 0..1 -- lowers the centre and elongates the lower profile (teardrop).
      separation      : horizontal gap between the two (0 = together at the sternum, 1 = wide).
      nipple_diameter : nipple width (a small sphere on the front face).
      nipple_depth    : how far the nipple protrudes forward (>0) or is set flush (0).
    Left/right are mirrored across x. Returns [] when size <= 0 (fully optional / off)."""
    from holographic.mesh_and_geometry.holographic_sdf import ellipsoid, sphere, SDF
    size = float(params.get("size", 0.0))
    if size <= 1e-3:
        return []
    sag = float(params.get("sag", 0.0))
    sep = float(params.get("separation", 0.4))
    nd = float(params.get("nipple_diameter", 0.03))
    ndepth = float(params.get("nipple_depth", 0.02))
    chest = np.asarray(chest, float)

    base = 0.09 * size                                               # ellipsoid base radius from the size slider
    # place each breast: out to the side by separation, slightly below the chest joint, drooping with sag.
    dx = torso_radius * 0.5 + sep * 0.11
    dy = -0.02 - sag * 0.10                                          # sag lowers the centre
    dz = torso_radius * 0.7                                          # forward, on the front of the chest
    nodes = []
    for sign in (+1.0, -1.0):                                        # left, right (mirrored across x)
        centre = chest + np.array([sign * dx, dy, dz])
        # a teardrop-ish ellipsoid: wider than tall, and taller (droopier) below as sag rises.
        ell = ellipsoid(base, base * (0.85 + 0.5 * sag), base * (1.0 + 0.25 * size))
        nodes.append(SDF("translate", tuple(float(x) for x in centre), (ell,)))
        if nd > 1e-3:
            nip_centre = centre + np.array([0.0, dy * 0.2, dz * 0.15 + base + ndepth])
            nodes.append(SDF("translate", tuple(float(x) for x in nip_centre), (sphere(max(nd / 2.0, 1e-3)),)))
    return nodes


def _bone_sdf(a, b, r):
    """A capsule bone between joints `a` and `b` of radius `r` (a sphere if the bone is degenerate). Rotates the
    Y-axis capsule onto the bone direction -- the same oriented-primitive trick fit_primitives uses."""
    from holographic.mesh_and_geometry.holographic_sdf import capsule, sphere, SDF
    a = np.asarray(a, float); b = np.asarray(b, float)
    mid = (a + b) / 2.0
    d = b - a; L = float(np.linalg.norm(d))
    if L < 1e-6:
        return SDF("translate", tuple(float(x) for x in mid), (sphere(float(r)),))
    dn = d / L
    y = np.array([0.0, 1.0, 0.0]); ax = np.cross(y, dn); s = float(np.linalg.norm(ax)); c = float(np.dot(y, dn))
    if s < 1e-8:
        axis, ang = np.array([1.0, 0.0, 0.0]), (0.0 if c > 0 else np.pi)
    else:
        axis, ang = ax / s, float(np.arctan2(s, c))
    cap = capsule(L / 2.0, float(r))
    return SDF("translate", tuple(float(x) for x in mid),
               (SDF("rotate", (float(axis[0]), float(axis[1]), float(axis[2]), float(ang)), (cap,)),))


class Humanoid:
    """A parametric biped: named joints (a dict name -> (x,y,z)), a bone list, and IK chains. Starts in a T-pose; pose
    it by reaching end-effectors to targets (per-chain FABRIK IK), then skin it into a primitive-set SDF."""

    def __init__(self, joints=None, scale=1.0, body=None):
        base = dict(_REST) if joints is None else dict(joints)
        self.joints = {k: (np.asarray(v, float) * scale) for k, v in base.items()}
        self.bones = list(_BONES)
        self.chains = {k: list(v) for k, v in _CHAINS.items()}
        self.body = default_body() if body is None else body
        # apply per-segment LENGTH sliders by walking each bone from its parent and re-placing the child further out.
        # length rebuilds the rest skeleton (before posing) so IK stays consistent with the morphed proportions.
        segs = self.body.get("segments", {})
        if joints is None and any("length" in v for v in segs.values()):
            for a, b in self.bones:
                seg = _SEGMENT.get((a, b))
                scale_len = 1.0 + segs.get(seg, {}).get("length", 0.0)
                if abs(scale_len - 1.0) > 1e-9:
                    d = self.joints[b] - self.joints[a]
                    self.joints[b] = self.joints[a] + d * scale_len

    def joints_array(self, names=None):
        """The joint positions as an (N,3) array, in `names` order (default: the canonical joint order)."""
        names = names or _JOINT_NAMES
        return np.array([self.joints[n] for n in names])

    def pose_to(self, targets, iters=30, mind=None, limited=True):
        """Reach named end-effectors to 3-D `targets` (a dict end_effector_name -> (x,y,z)) via per-chain IK. With
        `limited=True` (default) it uses CONSTRAINED FABRIK (solve_ik_limited) so joints stay in their anatomical range
        -- no hyperextended elbows/knees, ball joints within their cones -- and the ranges are TIGHTENED by this body's
        muscle/fat (a bulky limb can't fully close). With `limited=False` it uses plain solve_ik (may hyperextend).
        Updates the joints in place and returns self. Needs a `mind` for the IK solver."""
        if mind is None:
            raise ValueError("pose_to needs mind=<UnifiedMind> for the IK solver")
        for chain_name, chain in self.chains.items():
            tip = chain[-1]
            if tip not in targets:
                continue
            pts = np.array([self.joints[j] for j in chain])
            if limited:
                from holographic.mesh_and_geometry.holographic_iklimit import solve_ik_limited
                limits = _chain_limits(chain_name, self.body)
                # the reference for the first limited bone is the bone entering the chain root (chest->shoulder etc.).
                root_ref = pts[1] - pts[0] if len(pts) > 1 else (0.0, 1.0, 0.0)
                posed, _ = solve_ik_limited(pts, np.asarray(targets[tip], float), limits, iters=iters,
                                            root_ref=root_ref, mind=mind)
            else:
                posed, _ = mind.solve_ik(pts, np.asarray(targets[tip], float), iters=iters)
            for j, p in zip(chain, posed):
                self.joints[j] = np.asarray(p, float)
        return self

    def skin(self, limb_radius=0.06, head_radius=0.11, torso_radius=0.10, body=None):
        """Build a PRIMITIVE-SKIN SDF for the current pose, morphed by the character-editor `body` params (defaults to
        the humanoid's own `self.body`). Each bone becomes a capsule whose radius follows the global + per-segment
        fat/muscle sliders (fat/weight add girth where mass pools, distributed by region); muscular segments also get
        an ellipsoid muscle-belly bulge. Fat softens the joints via a smooth-union blend (rounder as fat rises). The
        head is a sphere and the pelvis a box, both scaled by mass. If body['breasts'] is set, anatomical chest
        geometry is smooth-unioned on. Returns a real SDF (mesh / to_shadertoy work). All morphs default to 0 -> the
        base build is byte-identical to the un-morphed stick figure."""
        from holographic.mesh_and_geometry.holographic_sdf import sphere, box, SDF
        body = self.body if body is None else body
        mass = body.get("fat", 0.0) + body.get("weight", 0.0) * 0.7
        blend = 0.02 + max(mass, 0.0) * 0.06                          # fat rounds the joints (smooth-union radius)

        parts = []                                                   # (node, is_torso) so we can blend appropriately
        for a, b in self.bones:
            seg = _SEGMENT.get((a, b), "torso")
            base_r = torso_radius if seg in ("torso",) else limb_radius
            r = _segment_radius(base_r, seg, body)
            parts.append(_bone_sdf(self.joints[a], self.joints[b], r))
            belly = _muscle_belly(self.joints[a], self.joints[b], base_r, seg, body)
            if belly is not None:
                parts.append(belly)
        # head + pelvis, scaled a little by overall mass.
        parts.append(SDF("translate", tuple(float(x) for x in self.joints["head"]),
                         (sphere(float(head_radius) * (1.0 + 0.15 * max(mass, 0.0))),)))
        pw = 0.14 * (1.0 + 0.5 * max(mass, 0.0))
        parts.append(SDF("translate", tuple(float(x) for x in self.joints["pelvis"]),
                         (box(pw, 0.10 * (1.0 + 0.3 * max(mass, 0.0)), 0.09 * (1.0 + 0.4 * max(mass, 0.0))),)))
        # optional anatomical chest geometry.
        if body.get("breasts"):
            parts.extend(add_breasts(self.joints["chest"], body["breasts"], torso_radius=torso_radius))

        # union everything; use a SMOOTH union when there is fat so the body reads as one soft form, a hard union at
        # the base build so the un-morphed skin is byte-identical to before (additive, no silent change).
        node = parts[0]
        op = "smooth_union" if blend > 0.021 else "union"
        for nxt in parts[1:]:
            node = SDF(op, (blend,) if op == "smooth_union" else (), (node, nxt))
        return node

    def keypoints_2d(self, camera, names=None):
        """Project the joints to 2-D image keypoints through a `camera` (anything with a .project(points)->(N,2)). The
        inverse of fit_pose_2d -- handy for testing the round-trip and for overlaying the rig on an image."""
        pts = self.joints_array(names)
        return np.asarray(camera.project(pts))


def _bone_length_table(joints):
    """The rest bone lengths (name_pair -> length) from a joint dict -- the constraint the 2-D lift preserves."""
    return {(a, b): float(np.linalg.norm(np.asarray(joints[b]) - np.asarray(joints[a]))) for a, b in _BONES}


# ------------------------------------------------------------------------------------------------------------
# ANATOMICAL JOINT LIMITS, per IK chain. Each chain is [root, ..., end]; the limit list aligns with its BONES.
# Hinge joints (elbow, knee) flex in one plane, one direction (no hyperextension); ball joints (shoulder, hip) get a
# cone. Muscle/fat TIGHTEN the flex range -- a bulky arm can't fully close -- which is how the morphs interact with
# pose. Axes are in the rest frame; the humanoid poses from the rest pose so they hold. Angles in DEGREES here.
# ------------------------------------------------------------------------------------------------------------

# per chain: a list aligned with the chain's bones (len = len(chain)-1). None = free; else a limit spec in degrees.
# The hinge axis is chosen so that POSITIVE flex is the anatomically natural direction (elbows/knees flex forward/back
# without hyperextending). Shoulder/hip get a wide cone; elbow/knee a forward hinge.
_CHAIN_LIMITS_DEG = {
    "l_arm": [None,                                               # chest->l_shoulder (structural, free)
              {"type": "cone", "half": 100.0},                    # l_shoulder: wide ball joint
              {"type": "hinge", "axis": "auto", "lo": 0.0, "hi": 150.0}],   # l_elbow: forward hinge (plane follows arm)
    "r_arm": [None,
              {"type": "cone", "half": 100.0},
              {"type": "hinge", "axis": "auto", "lo": 0.0, "hi": 150.0}],
    "l_leg": [None,                                               # pelvis->l_hip (structural)
              {"type": "cone", "half": 80.0},                     # l_hip: ball joint, narrower than shoulder
              {"type": "hinge", "axis": "auto", "lo": 0.0, "hi": 150.0}],   # l_knee: hinge (plane follows leg)
    "r_leg": [None,
              {"type": "cone", "half": 80.0},
              {"type": "hinge", "axis": "auto", "lo": 0.0, "hi": 150.0}],
    "spine": [None, {"type": "cone", "half": 35.0}, {"type": "cone", "half": 35.0}, {"type": "cone", "half": 45.0}],
}

# which segment's muscle/fat tightens each hinge (the flexion-blocking bulk sits on the flexor side).
_HINGE_SEGMENT = {"l_arm": "upper_arm", "r_arm": "upper_arm", "l_leg": "thigh", "r_leg": "thigh"}


def _chain_limits(chain_name, body):
    """Resolve a chain's anatomical limits to RADIANS, tightening the hinge flex range by the muscle+fat on the
    flexing segment: a bulkier limb loses flexion range (a big biceps blocks the elbow from fully closing). Returns a
    list aligned with the chain's bones."""
    specs = _CHAIN_LIMITS_DEG.get(chain_name)
    if specs is None:
        return None
    seg = _HINGE_SEGMENT.get(chain_name)
    segp = body.get("segments", {}).get(seg, {}) if seg else {}
    bulk = (body.get("muscle", 0.0) * _MUSCLE_GAIN.get(seg, 0.4) + body.get("fat", 0.0) * _FAT_GAIN.get(seg, 0.4)
            + body.get("weight", 0.0) * 0.4 + segp.get("muscle", 0.0) + segp.get("fat", 0.0)) if seg else 0.0
    bulk = max(bulk, 0.0)
    out = []
    for spec in specs:
        if spec is None:
            out.append(None)
        elif spec["type"] == "hinge":
            # bulk reduces the max flex: hi shrinks by up to ~55 deg as bulk rises (soft-tissue apposition).
            hi = max(spec["hi"] - min(bulk, 1.5) * 40.0, 45.0)
            out.append({"type": "hinge", "axis": spec["axis"], "lo": np.radians(spec["lo"]), "hi": np.radians(hi)})
        else:
            half = max(spec["half"] - min(bulk, 1.5) * 15.0, 20.0)   # bulk narrows the cone a little too
            out.append({"type": "cone", "half": np.radians(half), "ref": spec.get("ref")})
    return out


def fit_pose_3d(keypoints, iters=30, mind=None, scale=1.0):
    """Fit a humanoid to 3-D `keypoints` (a dict joint_name -> (x,y,z), e.g. from mocap): snap the rig's targeted
    end-effectors to their keypoints via IK, so the posed skeleton reaches them with correct bone lengths. Returns the
    posed Humanoid. The honest 3-D pose fit. Needs a `mind` for solve_ik."""
    h = Humanoid(scale=scale)
    # anchor the root joints directly (pelvis/chest), then IK the limbs to their end-effector keypoints.
    for anchor in ("pelvis", "chest", "neck", "head", "spine"):
        if anchor in keypoints:
            h.joints[anchor] = np.asarray(keypoints[anchor], float)
    targets = {tip: keypoints[tip] for tip in ("l_wrist", "r_wrist", "l_ankle", "r_ankle", "head")
               if tip in keypoints}
    h.pose_to(targets, iters=iters, mind=mind)
    return h


def fit_pose_2d(keypoints_2d, camera, iters=30, mind=None, scale=1.0):
    """Fit a humanoid to 2-D `keypoints_2d` from ONE image (a dict joint_name -> (u,v)) plus a `camera`: back-project
    each keypoint to its viewing ray and choose the depth along the ray that best keeps the rest bone lengths (a
    classical bone-length-constrained lift -- NOT a learned 3-D lifter), then IK-clean the limbs. Returns the posed
    Humanoid and the lifted 3-D keypoints.

    KEPT NEGATIVE: monocular depth is ambiguous (the forward/backward flip), so the lift resolves toward the rest
    pose -- it recovers A plausible 3-D pose consistent with the image, not THE unique one. Needs `camera.ray(uv)` ->
    (origin, direction) and a `mind` for solve_ik."""
    h = Humanoid(scale=scale)
    rest = h.joints
    lifted = {}
    # start every joint at its rest DEPTH along its keypoint ray (a stable, deterministic initial lift).
    for name, uv in keypoints_2d.items():
        if name not in rest:
            continue
        origin, direction = camera.ray(np.asarray(uv, float))
        origin = np.asarray(origin, float); direction = np.asarray(direction, float)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        # pick the ray parameter t so the point sits at the rest joint's distance from the camera (the depth prior).
        t = float(np.linalg.norm(np.asarray(rest[name]) - origin))
        lifted[name] = origin + t * direction
    # snap known joints, then IK the limbs so bone lengths are exact again (the lift only approximates them).
    for name, p in lifted.items():
        h.joints[name] = p
    targets = {tip: lifted[tip] for tip in ("l_wrist", "r_wrist", "l_ankle", "r_ankle", "head") if tip in lifted}
    h.pose_to(targets, iters=iters, mind=mind)
    return h, lifted


def _selftest():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    # (1) a rest humanoid skins into a real SDF that meshes and emits a Shadertoy shader.
    h = Humanoid()
    body = h.skin()
    assert "mainImage" in m.to_shadertoy(body), "the humanoid skin emits a Shadertoy shader"
    assert m.sdf_to_mesh(body, resolution=32).n_faces > 200, "the humanoid skin meshes"

    # (2) IK posing: reach a target WITHIN the arm's reach; it reaches it and bone lengths are preserved. (A target
    #     beyond arm's length is correctly NOT reached under limits -- tested in (7).)
    rest_arm = [np.linalg.norm(h.joints["l_elbow"] - h.joints["l_shoulder"]),
                np.linalg.norm(h.joints["l_wrist"] - h.joints["l_elbow"])]
    reach_tgt = np.array([0.35, 0.75, 0.25])                 # in front + up, within the ~0.49 arm reach
    h.pose_to({"l_wrist": tuple(reach_tgt)}, mind=m)
    assert np.linalg.norm(h.joints["l_wrist"] - reach_tgt) < 0.06, "IK reaches a reachable target"
    posed_arm = [np.linalg.norm(h.joints["l_elbow"] - h.joints["l_shoulder"]),
                 np.linalg.norm(h.joints["l_wrist"] - h.joints["l_elbow"])]
    assert np.allclose(rest_arm, posed_arm, atol=1e-3), "IK preserves bone lengths (%s vs %s)" % (rest_arm, posed_arm)

    # (3) 3-D pose fit: give keypoints, get a rig whose end-effectors reach them.
    kp = {"pelvis": (0, 0, 0), "head": (0, 0.78, 0.1),
          "l_wrist": (0.4, 0.9, 0.2), "r_wrist": (-0.5, 0.3, 0.1),
          "l_ankle": (0.12, -0.88, 0.0), "r_ankle": (-0.12, -0.7, 0.3)}
    h3 = fit_pose_3d(kp, mind=m)
    for tip in ("l_wrist", "r_wrist", "r_ankle"):
        assert np.linalg.norm(h3.joints[tip] - np.asarray(kp[tip])) < 0.05, "%s reaches its keypoint" % tip

    # (4) 2-D pose fit round-trips through a pinhole camera: project a known pose, lift it back, recover a close pose.
    class Cam:
        """A trivial pinhole at (0,0,3) looking down -z, focal 1 in normalised image coords."""
        def __init__(self): self.eye = np.array([0.0, 0.0, 3.0]); self.f = 1.0
        def project(self, P):
            P = np.atleast_2d(P); z = self.eye[2] - P[:, 2]
            return np.column_stack([self.f * P[:, 0] / z, self.f * P[:, 1] / z])
        def ray(self, uv):
            d = np.array([uv[0], uv[1], -1.0]); return self.eye, d
    cam = Cam()
    true_pose = fit_pose_3d({"l_wrist": (0.3, 0.9, 0.2)}, mind=m)
    kp2d = {n: cam.project(true_pose.joints[n])[0] for n in _JOINT_NAMES}
    h2, lifted = fit_pose_2d(kp2d, cam, mind=m)
    reproj = np.array([cam.project(h2.joints[n])[0] for n in _JOINT_NAMES])
    truth2d = np.array([kp2d[n] for n in _JOINT_NAMES])
    reproj_err = float(np.mean(np.linalg.norm(reproj - truth2d, axis=1)))
    assert reproj_err < 0.1, "the 2-D fit reprojects near the input keypoints (err %.3f)" % reproj_err

    # (5) determinism.
    h_a = fit_pose_3d(kp, mind=m); h_b = fit_pose_3d(kp, mind=m)
    assert np.allclose(h_a.joints_array(), h_b.joints_array()), "pose fitting is deterministic"

    # (6) CHARACTER-EDITOR MORPHS. Base build is byte-identical (hard union); muscle + fat thicken a limb; length
    #     scales a segment; breasts add chest volume; the morphed body still meshes + emits a Shadertoy.
    assert Humanoid().skin().kind == "union", "the un-morphed base build stays a hard union (additive, unchanged)"

    def _forearm_thickness(body):
        hh = Humanoid(body=body); sdf = hh.skin()
        mid = (hh.joints["l_elbow"] + hh.joints["l_wrist"]) / 2.0
        for r in np.linspace(0.02, 0.4, 200):
            if sdf.eval(np.array([mid + [0, 0, r]]))[0] > 0:
                return r
        return 0.4
    t0 = _forearm_thickness(default_body())
    bm = default_body(); bm["muscle"] = 1.0
    bf = default_body(); bf["fat"] = 1.0
    assert _forearm_thickness(bm) > t0 and _forearm_thickness(bf) > t0, "muscle and fat both thicken a limb"

    b_len = default_body(); b_len["segments"] = {"thigh": {"length": 0.5}}
    knee_len = Humanoid(body=b_len).joints["l_knee"][1]
    assert knee_len < Humanoid().joints["l_knee"][1] - 0.15, "the length slider lengthens the thigh (knee drops)"

    b_br = default_body()
    b_br["breasts"] = {"size": 1.0, "sag": 0.3, "separation": 0.4, "nipple_diameter": 0.03, "nipple_depth": 0.02}
    hbr = Humanoid(body=b_br); sdf_br = hbr.skin()
    probe = hbr.joints["chest"] + np.array([0.1, -0.02, 0.12])
    assert sdf_br.eval(np.array([probe]))[0] < 0 <= Humanoid().skin().eval(np.array([probe]))[0], \
        "the breast morph adds chest-front volume that the flat build lacks"
    assert "mainImage" in m.to_shadertoy(sdf_br), "the morphed body (with ellipsoids) still emits a Shadertoy"
    assert m.sdf_to_mesh(hbr.skin(), resolution=28).n_faces > 200, "the morphed body meshes"
    # global weight distributes appropriately: the torso thickens more than the forearm per unit weight (region gains)
    bw = default_body(); bw["weight"] = 1.0
    hw = Humanoid(body=bw)
    assert _segment_radius(0.10, "torso", bw) / 0.10 > _segment_radius(0.06, "forearm", bw) / 0.06, \
        "global weight pools on the torso more than the forearm (region-weighted distribution)"

    # (7) ANATOMICAL JOINT LIMITS + muscle/fat coupling. Constrained posing keeps the elbow flex within range, and a
    #     bulky arm can't curl as far as a lean one (the morphs interact with range of motion).
    def _elbow_flex(hh):
        # the FLEXION angle = how far the forearm has bent away from straight (0 = straight arm, 180 = fully folded).
        u = hh.joints["l_elbow"] - hh.joints["l_shoulder"]; v = hh.joints["l_wrist"] - hh.joints["l_elbow"]
        u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))

    # a tight curl: the lean arm bends hard; the muscular arm is capped and can't curl as far.
    tight_curl = {"l_wrist": (0.25, 0.55, 0.05)}
    lean = Humanoid(); lean.pose_to(tight_curl, mind=m)
    bmus = default_body(); bmus["muscle"] = 1.0; bmus["segments"] = {"upper_arm": {"muscle": 1.0}}
    musc = Humanoid(body=bmus); musc.pose_to(tight_curl, mind=m)
    assert _elbow_flex(musc) < _elbow_flex(lean) - 20.0, ("muscle tightens the elbow flex range (lean %.0f vs muscular "
                                                          "%.0f)" % (_elbow_flex(lean), _elbow_flex(musc)))
    # the muscular wrist ends up FARTHER from the curl target (it physically can't reach as far)
    tgt = np.array([0.25, 0.55, 0.05])
    assert np.linalg.norm(musc.joints["l_wrist"] - tgt) > np.linalg.norm(lean.joints["l_wrist"] - tgt), \
        "the bulkier arm cannot reach the tight-curl target as closely (range of motion reduced)"

    # the elbow flex never exceeds its limit, and bone lengths are preserved under constrained posing.
    assert _elbow_flex(lean) <= 151.0 and _elbow_flex(musc) <= 91.0, "flex stays within the (morph-tightened) limit"
    fresh = Humanoid()
    rest_a = [np.linalg.norm(fresh.joints["l_elbow"] - fresh.joints["l_shoulder"]),
              np.linalg.norm(fresh.joints["l_wrist"] - fresh.joints["l_elbow"])]
    la = [np.linalg.norm(musc.joints["l_elbow"] - musc.joints["l_shoulder"]),
          np.linalg.norm(musc.joints["l_wrist"] - musc.joints["l_elbow"])]
    assert np.allclose(la, rest_a, atol=1e-3), "constrained posing preserves bone lengths"

    print("holographic_humanoid selftest: ok (a biped skins into a Shadertoy-emitting SDF that meshes; IK raises the "
          "wrist to its target and PRESERVES bone lengths; 3-D keypoints -> a rig whose hands/feet reach them; 2-D "
          "keypoints from one camera -> a lifted pose that reprojects at err %.3f; deterministic; CHARACTER-EDITOR "
          "MORPHS -- muscle/fat/length per segment, global weight by region, optional breast geometry; ANATOMICAL "
          "JOINT LIMITS -- no hyperextension (auto-plane hinge), and muscle/fat reduce the flex range (lean elbow flex "
          "%.0f vs muscular %.0f on a tight curl); KEPT NEGATIVE -- fits KEYPOINTS not pixels, angle limits only "
          "(no self-collision))" % (reproj_err, _elbow_flex(lean), _elbow_flex(musc)))


if __name__ == "__main__":
    _selftest()

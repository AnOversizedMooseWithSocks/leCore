"""Spore-style CREATURE builder: a spine with attachable limbs, bilateral symmetry, constraints (holographic_creature).

WHY THIS MODULE EXISTS
----------------------
The humanoid is a FIXED biped. Non-humanoid organic lifeforms -- quadrupeds, hexapods, tentacled things, serpents --
need an ARBITRARY body plan: a spine (backbone) with limbs attached at points along it, each limb a segmented chain,
with bilateral symmetry (a limb placed off-centre mirrors to the other side, as in Spore's creature creator). This
module builds that from a declarative SPEC, reusing the humanoid's bone-SDF, morph radii, and CONSTRAINED IK so the
limbs pose with joint limits and never hyperextend.

The body plan is DATA (a spec dict), so it is deterministic, serialisable, and callable over /invoke.

THE SPEC
  {
    "spine":  {"length": 1.2, "segments": 4, "axis": (0,0,1), "curve": 0.0},
    "limbs":  [ {"at": 0.3, "dir": (1,-0.4,0), "segments": 3, "length": 0.6, "radius": 0.05,
                 "mirror": True, "cone_deg": 70, "hinge_deg": 140} , ... ],
    "head":   {"at": 1.0, "radius": 0.14},          # optional sphere at a spine fraction
    "body":   <a humanoid.default_body() morph block: global muscle/fat + smooth-union blend>,
  }
  `at` is a fraction along the spine (0 = tail, 1 = head). `mirror` adds the x-mirrored twin automatically.

JOINT CONSTRAINTS, "the best we can" (honest)
  We cannot know an arbitrary creature's real anatomy, so we impose SENSIBLE ORGANIC DEFAULTS: each limb's MOUNT is a
  ball joint (a cone, default 70 deg) and every interior joint is an auto-plane HINGE that flexes one way only (no
  hyperextension), capped at `hinge_deg`. Muscle/fat tighten the flex range, same coupling as the humanoid. The tip is
  free (a foot / grasper). These are defaults you can override per limb, not a claim of species-correct anatomy.

WHAT IT PROVIDES
  * Creature(spec) -- builds joints, bones, per-limb IK chains + limits.
  * .skin(...) -- a morph-aware primitive-skin SDF (spine + limb capsules + optional head), meshes + emits Shadertoy.
  * .pose_limb(i, target) / .pose(targets) -- constrained IK on a limb (or several), joints stay in range.

KEPT NEGATIVES (loud)
  * Default constraints are GENERIC organic limits, not species-correct anatomy -- the caller tunes cone_deg/hinge_deg.
  * Angle limits only, NO self-collision (a limb can pass through the body) -- inherited from solve_ik_limited.
  * Bilateral symmetry only (one mirror plane); radial/other symmetries are a scoped extension.
  * No procedural gait / animation and no feature-parts (mouths, eyes) yet -- this is the structural rig, scoped so.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_humanoid import _bone_sdf, _muscle_belly, default_body


def _mirror(p):
    """Mirror a point across the sagittal (x=0) plane -- bilateral symmetry."""
    return np.array([-p[0], p[1], p[2]])


class Creature:
    """A procedural creature from a body-plan spec: a spine chain with limbs attached at fractional positions, with
    bilateral symmetry and per-limb organic joint constraints. Generalises the humanoid to arbitrary body plans."""

    def __init__(self, spec):
        spec = dict(spec or {})
        self.body = spec.get("body") or default_body()
        sp = dict(spec.get("spine") or {})
        length = float(sp.get("length", 1.2))
        nseg = int(sp.get("segments", 4))
        axis = np.asarray(sp.get("axis", (0.0, 0.0, 1.0)), float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        curve = float(sp.get("curve", 0.0))                  # a gentle backbone bend (arch), in the x-axis
        self.spine_radius = float(sp.get("radius", 0.08))

        # spine nodes s0..sN along the axis, optionally arched by `curve`.
        self.joints = {}
        self.spine_nodes = []
        for i in range(nseg + 1):
            t = i / nseg
            p = axis * (t * length)
            if abs(curve) > 1e-9:
                p = p + np.array([0.0, np.sin(t * np.pi) * curve, 0.0])   # arch upward toward the middle
            name = "s%d" % i
            self.joints[name] = p
            self.spine_nodes.append(name)

        self.bones = [(self.spine_nodes[i], self.spine_nodes[i + 1]) for i in range(nseg)]
        self.chains = {}                                     # limb IK chains: name -> [mount, seg0, seg1, ...]
        self.limits = {}                                    # chain name -> per-bone limit list
        self.limb_radius = {}                               # chain name -> radius
        self._limb_count = 0

        for limb in spec.get("limbs", []):
            self._add_limb(limb, length, nseg)

        head = spec.get("head")
        if head:
            self.head = {"node": self._node_at(float(head.get("at", 1.0)), nseg),
                         "radius": float(head.get("radius", 0.14))}
        else:
            self.head = None

    def _node_at(self, frac, nseg):
        """The spine node nearest a fraction `frac` in [0,1] along the backbone."""
        return self.spine_nodes[int(round(np.clip(frac, 0.0, 1.0) * nseg))]

    def _add_limb(self, limb, length, nseg):
        """Attach a limb chain at a spine fraction, in a direction, with `segments` bones; mirror it if requested."""
        at = float(limb.get("at", 0.5))
        mount_name = self._node_at(at, nseg)
        mount = self.joints[mount_name]
        d = np.asarray(limb.get("dir", (1.0, 0.0, 0.0)), float)
        d = d / (np.linalg.norm(d) + 1e-12)
        segs = int(limb.get("segments", 3))
        llen = float(limb.get("length", 0.5))
        radius = float(limb.get("radius", 0.05))
        cone = np.radians(float(limb.get("cone_deg", 70.0)))
        hinge = np.radians(float(limb.get("hinge_deg", 140.0)))
        seglen = llen / segs

        def build(direction, tag):
            chain = [mount_name]
            prev = mount
            for j in range(segs):
                p = prev + direction * seglen
                nm = "%s_%d" % (tag, j)
                self.joints[nm] = p
                self.bones.append((chain[-1], nm))
                chain.append(nm)
                prev = p
            # limits: mount = cone (ball), interior joints = auto hinge (no hyperextension), tip free.
            lim = [{"type": "cone", "half": cone}]
            for _ in range(segs - 1):
                lim.append({"type": "hinge", "axis": "auto", "lo": 0.0, "hi": hinge})
            self.chains[tag] = chain
            self.limits[tag] = lim
            self.limb_radius[tag] = radius

        idx = self._limb_count
        build(d, "L%d" % idx)
        if limb.get("mirror", True) and abs(d[0]) > 1e-6:
            # the spine lies on x=0, so a mirrored limb shares the mount and mirrors only the x-component of direction.
            build(_mirror(d), "L%dm" % idx)
            self.chains["L%dm" % idx][0] = mount_name
        self._limb_count += 1

    def joints_array(self, names=None):
        names = names or list(self.joints.keys())
        return np.array([self.joints[n] for n in names])

    def pose_limb(self, chain_name, target, iters=30, mind=None):
        """Reach a limb's tip to `target` via CONSTRAINED IK (joint limits from the limb's spec, tightened by muscle/
        fat). Updates the limb joints in place. Needs a `mind`."""
        if mind is None:
            raise ValueError("pose_limb needs mind=<UnifiedMind> for the IK solver")
        from holographic.mesh_and_geometry.holographic_iklimit import solve_ik_limited
        chain = self.chains[chain_name]
        pts = np.array([self.joints[j] for j in chain])
        # tighten the hinge range by this creature's global muscle/fat bulk (same coupling as the humanoid).
        bulk = max(self.body.get("muscle", 0.0) * 0.6 + self.body.get("fat", 0.0) * 0.6
                   + self.body.get("weight", 0.0) * 0.4, 0.0)
        limits = []
        for lm in self.limits[chain_name]:
            if lm["type"] == "hinge":
                hi = max(lm["hi"] - min(bulk, 1.5) * np.radians(40.0), np.radians(45.0))
                limits.append({"type": "hinge", "axis": lm["axis"], "lo": lm["lo"], "hi": hi})
            else:
                limits.append(lm)
        root_ref = pts[1] - pts[0] if len(pts) > 1 else (0.0, 1.0, 0.0)
        posed, err = solve_ik_limited(pts, np.asarray(target, float), limits, iters=iters, root_ref=root_ref, mind=mind)
        for j, p in zip(chain, posed):
            self.joints[j] = np.asarray(p, float)
        return err

    def pose(self, targets, iters=30, mind=None):
        """Pose several limbs at once: `targets` is {chain_name: (x,y,z)}. Returns self."""
        for name, tgt in targets.items():
            if name in self.chains:
                self.pose_limb(name, tgt, iters=iters, mind=mind)
        return self

    def skin(self, body=None):
        """Build a morph-aware primitive-skin SDF: a capsule for every bone (spine thicker, limbs by their radius),
        muscle bellies where muscle is high, an optional head sphere. Fat softens joints via a smooth union. Returns a
        real SDF (meshes, emits Shadertoy). Reuses the humanoid morph helpers, so muscle/fat behave consistently."""
        from holographic.mesh_and_geometry.holographic_sdf import sphere, SDF
        body = self.body if body is None else body
        mass = body.get("fat", 0.0) + body.get("weight", 0.0) * 0.7
        blend = 0.02 + max(mass, 0.0) * 0.06

        spine_set = set(self.spine_nodes)
        parts = []
        for a, b in self.bones:
            on_spine = a in spine_set and b in spine_set
            if on_spine:
                r = self.spine_radius * (1.0 + max(mass, 0.0) * 0.6 + body.get("muscle", 0.0) * 0.2)
                seg = "torso"
            else:
                # find which limb chain this bone belongs to for its radius.
                r = 0.05
                for nm, chain in self.chains.items():
                    if b in chain:
                        r = self.limb_radius.get(nm, 0.05) * (1.0 + max(mass, 0.0) * 0.5 + body.get("muscle", 0.0) * 0.3)
                        break
                seg = "upper_arm"
            parts.append(_bone_sdf(self.joints[a], self.joints[b], max(r, 1e-3)))
            belly = _muscle_belly(self.joints[a], self.joints[b], r, seg, body)
            if belly is not None:
                parts.append(belly)
        if self.head:
            parts.append(SDF("translate", tuple(float(x) for x in self.joints[self.head["node"]]),
                             (sphere(float(self.head["radius"]) * (1.0 + 0.15 * max(mass, 0.0))),)))
        node = parts[0]
        op = "smooth_union" if blend > 0.021 else "union"
        for nxt in parts[1:]:
            node = SDF(op, (blend,) if op == "smooth_union" else (), (node, nxt))
        return node


def quadruped_spec(body=None):
    """A ready-made body plan: a quadruped -- a spine with two pairs of legs (front + back) and a head. A concrete
    starting point that shows the spec shape."""
    return {
        "spine": {"length": 1.2, "segments": 4, "axis": (0.0, 0.0, 1.0), "curve": 0.12, "radius": 0.10},
        "limbs": [
            {"at": 0.25, "dir": (1.0, -1.2, 0.0), "segments": 3, "length": 0.6, "radius": 0.05, "mirror": True},
            {"at": 0.80, "dir": (1.0, -1.2, 0.0), "segments": 3, "length": 0.6, "radius": 0.05, "mirror": True},
        ],
        "head": {"at": 1.0, "radius": 0.16},
        "body": body or default_body(),
    }


def _selftest():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    # (1) a quadruped builds: a spine + 4 legs (2 mirrored pairs) + a head; the skin meshes + emits a Shadertoy.
    cre = Creature(quadruped_spec())
    assert len([n for n in cre.joints if n.startswith("s")]) == 5, "the spine has its nodes"
    assert len(cre.chains) == 4, "two mirrored pairs -> four leg chains, got %d" % len(cre.chains)
    body = cre.skin()
    assert "mainImage" in m.to_shadertoy(body), "the creature skin emits a Shadertoy shader"
    assert m.sdf_to_mesh(body, resolution=36).n_faces > 300, "the creature skin meshes"

    # (2) bilateral symmetry: a mirrored leg's tip is the x-mirror of its twin's tip.
    l0 = cre.chains["L0"][-1]; l0m = cre.chains["L0m"][-1]
    assert np.allclose(cre.joints[l0][0], -cre.joints[l0m][0], atol=1e-6), "mirrored legs are x-symmetric"
    assert np.allclose(cre.joints[l0][1:], cre.joints[l0m][1:], atol=1e-6), "mirrored legs share y,z"

    # (3) constrained IK on a leg: reach a target; joints stay in range (no hyperextension), bones preserved.
    leg = cre.chains["L0"]
    rest_lens = [np.linalg.norm(cre.joints[leg[i + 1]] - cre.joints[leg[i]]) for i in range(len(leg) - 1)]
    err = cre.pose_limb("L0", cre.joints[leg[0]] + np.array([0.4, -0.3, 0.2]), mind=m)
    posed_lens = [np.linalg.norm(cre.joints[leg[i + 1]] - cre.joints[leg[i]]) for i in range(len(leg) - 1)]
    assert np.allclose(rest_lens, posed_lens, atol=1e-4), "IK preserves the leg's bone lengths"
    # interior hinge flex stays within [0,140] (no hyperextension)
    for i in range(1, len(leg) - 1):
        u = cre.joints[leg[i]] - cre.joints[leg[i - 1]]; v = cre.joints[leg[i + 1]] - cre.joints[leg[i]]
        u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
        flex = np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))
        assert -1.0 <= flex <= 141.0, "leg joint flex stays in range (%.0f)" % flex

    # (4) muscle thickens the creature; determinism.
    bmus = default_body(); bmus["muscle"] = 1.0
    thick = Creature(quadruped_spec(bmus)).skin()
    mid = (cre.joints["s2"] + cre.joints["s3"]) / 2.0
    def spine_thickness(sdf):
        for r in np.linspace(0.05, 0.6, 200):
            if sdf.eval(np.array([mid + [r, 0, 0]]))[0] > 0:
                return r
        return 0.6
    assert spine_thickness(thick) > spine_thickness(Creature(quadruped_spec()).skin()), "muscle thickens the body"
    a = Creature(quadruped_spec()); b = Creature(quadruped_spec())
    assert np.allclose(a.joints_array(list(a.joints)), b.joints_array(list(b.joints))), "creature build is deterministic"

    # (5) a NON-quadruped plan works too (a hexapod-ish: three leg pairs) -- arbitrary body plans.
    hexspec = {"spine": {"length": 1.0, "segments": 3},
               "limbs": [{"at": f, "dir": (1.0, -1.0, 0.0), "segments": 2, "length": 0.4, "mirror": True}
                         for f in (0.2, 0.5, 0.8)]}
    hexc = Creature(hexspec)
    assert len(hexc.chains) == 6, "three mirrored pairs -> six legs, got %d" % len(hexc.chains)

    print("holographic_creature selftest: ok (a quadruped builds a spine + 4 legs + head that meshes + emits a "
          "Shadertoy; bilateral symmetry mirrors legs across x; constrained IK poses a leg keeping bone lengths and "
          "joint flex in [0,140] (no hyperextension); muscle thickens the body; a hexapod plan gives 6 legs; "
          "deterministic; KEPT NEGATIVE -- generic organic joint limits not species anatomy, no self-collision)")


if __name__ == "__main__":
    _selftest()

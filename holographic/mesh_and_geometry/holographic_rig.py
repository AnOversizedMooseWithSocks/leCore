"""ONE RIG TYPE -- the shared segment/joint/chain view over ANY skeleton, plus role tags.

Backlog D-1/R-2/R-3/R-5. The decision this module enforces: there must not be a creature rig and a
humanoid rig. A humanoid is a creature whose spec is bipedal, and a hybrid (centaur, minotaur) must
be a SPEC, not a code path -- because anything that walks a rig otherwise has to branch on which kind
it is, and the branch is where hybrids die.

WHY A VIEW AND NOT A REWRITE. The audit found `Creature.bones` and `Humanoid.bones` were ALREADY the
same structure -- an explicit list of (parent, child) joint pairs, one per rigid segment, alongside
`joints` and `chains`. The backlog assumed a rewrite ("give the creature bones"); it was already
there. So the honest work is not a new representation but a shared VIEW plus the invariant that keeps
them agreeing, which is additive by construction: neither class changes, and neither one's emitted
bytes move.

THE TAG RULE, stated once for the whole engine: a segment is named `"<chain>#<index>"`. Creature skin
provenance (`bone_of`), the readability reports, and role tags all use it, so provenance from any of
them joins to any other without a translation table.

ROLE TAGS (R-5) are HOLOGRAPHIC, and that is the point rather than a flourish: a role assignment is
`bind(segment_atom, role_atom)`, all assignments superpose into ONE vector, and "which segments are
feet" is one unbind plus a cleanup sweep instead of a dict walk. Gait and animation can then find
parts BY ROLE on any body plan -- which is precisely what lets one authored behaviour drive a
quadruped, a biped and a centaur without branching.

KEPT NEGATIVE, MEASURED, LOUD -- AND IT CHANGED THE API. The superposed role vector is capacity-
limited like every bundle, and the measurement was bad enough to demote it: recall 0.94 / 0.66 / 0.19
/ 0.04 at 16 / 32 / 64 / 128 segments (dim=512, 4 roles). TWO RESCUES TESTED AND BOTH FAILED -- a
load-aware largest-gap cut instead of a fixed threshold (0.94 / 0.72 / 0.17 / 0.17) and raising the
dimension (at 64 segments recall went 0.19 -> 0.05 -> 0.00 across dim 512 -> 2048 -> 8192, the higher
dimensions merely removing the noise that had been accidentally lifting items over an absolute
threshold). So `find_by_role` is the EXACT dict and is the AUTHORITY; `find_by_role_holographic` is
the VSA path, exact on the shipped 16-segment rigs, with its cliff pinned by a test that fails if the
measured table ever goes stale. A role query that silently drops a foot is worse than no query at
all: the gait would not find that leg and would report a clean result for a creature limping on
three.
"""

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind, cosine, random_vector


class Rig:
    """The shared skeleton view: joints, per-segment bones, chains, canonical segment tags, roles.

    Built from anything that carries `joints`, `bones` and `chains` -- today a `Creature` or a
    `Humanoid`, tomorrow a rig fitted from an image (backlog D-6/L-1), with no new code path.
    """

    def __init__(self, joints, bones, chains, source=None):
        self.joints = {k: np.asarray(v, float) for k, v in dict(joints).items()}
        self.bones = [(str(a), str(b)) for a, b in bones]
        self.chains = {str(k): [str(x) for x in v] for k, v in dict(chains).items()}
        self.source = source
        self.tags = _segment_tags(self.bones, self.chains)
        self.roles = {}                      # segment tag -> set of role names (the GROUND TRUTH)
        self._role_vec = None                # the superposed holographic index (a cache)
        self._atoms = None

    # -- geometry ---------------------------------------------------------------------------
    def segment(self, tag):
        """The (start, end) world endpoints of the segment named `tag`."""
        a, b = self.bones[self.tags.index(tag)]
        return self.joints[a], self.joints[b]

    def segment_length(self, tag):
        """Rest length of one segment -- the quantity a bone must preserve under any pose."""
        a, b = self.segment(tag)
        return float(np.linalg.norm(b - a))

    def extent(self):
        """The rig's bounding extent (lo, hi). THE REFERENCE LENGTH (backlog D-7): any spatial
        frequency expressed relative to the body must declare what it is relative to, and this is
        that declaration for everything downstream of a rig."""
        P = np.stack(list(self.joints.values()))
        return P.min(axis=0), P.max(axis=0)

    def reference_length(self):
        """One scalar body scale -- the diagonal of the rig's extent. Texture frequency, marching
        resolution and limb thickness have each been wrong for the same reason (an absolute where a
        body-relative quantity belonged); this is the number they should be expressed against."""
        lo, hi = self.extent()
        return float(np.linalg.norm(hi - lo))

    # -- roles (R-5) ------------------------------------------------------------------------
    def tag_role(self, tag, role):
        """Assign a capability role (`foot`, `grasper`, `mouth`, `eye`, `torso`, `head`) to a
        segment. Invalidates the holographic index so it is rebuilt on next query."""
        if tag not in self.tags:
            raise KeyError("no such segment tag %r" % tag)
        self.roles.setdefault(tag, set()).add(str(role))
        self._role_vec = None
        return self

    def role_atoms(self, dim=512, seed=0):
        """The codebook: one deterministic atom per segment tag and per role name. Seeded and sorted,
        so the same rig yields the same atoms in any process (PYTHONHASHSEED-independent)."""
        if self._atoms is not None and self._atoms["dim"] == dim and self._atoms["seed"] == seed:
            return self._atoms
        rng = np.random.default_rng(seed)
        names = list(self.tags) + sorted({r for rs in self.roles.values() for r in rs})
        book = {n: random_vector(dim, rng) for n in names}      # sorted-order draw -> reproducible
        self._atoms = {"book": book, "dim": dim, "seed": seed}
        self._role_vec = None
        return self._atoms

    def role_vector(self, dim=512, seed=0):
        """ALL role assignments as ONE vector: sum of bind(segment, role). This is the holographic
        index -- the whole point being that adding an assignment does not grow a structure, and a
        query is an unbind rather than a scan."""
        atoms = self.role_atoms(dim=dim, seed=seed)["book"]
        if self._role_vec is not None:
            return self._role_vec
        acc = np.zeros(dim)
        for tag in sorted(self.roles):
            for role in sorted(self.roles[tag]):
                acc = acc + bind(atoms[tag], atoms[role])
        self._role_vec = acc
        return acc

    def find_by_role(self, role):
        """Segments carrying `role` -- THE AUTHORITATIVE ANSWER, from the exact assignment dict.

        This is deliberately NOT the holographic path. `find_by_role_holographic` was measured and
        drops assignments well before a realistic rig is large (see its docstring), and a role query
        that silently loses a foot is worse than no query at all: the gait would simply not find that
        leg and would report a clean result for a creature limping on three.
        """
        return sorted(t for t, rs in self.roles.items() if str(role) in rs)

    def find_by_role_holographic(self, role, dim=512, seed=0, threshold=0.15):
        """The VSA query: unbind `role` out of the superposed index, keep segment atoms above
        `threshold`. One unbind instead of a scan, and adding an assignment does not grow a structure.

        MEASURED CAPACITY (dim=512, threshold=0.15, synthetic rigs, 4 roles):

            segments   16     32     64    128
            recall    0.94   0.66   0.19   0.04

        It is EXACT on the shipped quadruped and biped (16 segments, few per role) and unusable past
        roughly 32. TWO RESCUES TESTED AND REFUTED, so nobody re-tries them: a load-aware largest-gap
        cut instead of a fixed threshold changed almost nothing (0.94 / 0.72 / 0.17 / 0.17), and
        RAISING THE DIMENSION did not buy the capacity back -- at 64 segments recall went 0.19 (512)
        -> 0.05 (2048) -> 0.00 (8192), i.e. the higher dimensions were merely removing the noise that
        had been accidentally lifting some items over an absolute threshold. The honest reading is
        that an absolute cosine cut is the wrong instrument for a bundle whose per-item score falls
        as ~1/sqrt(load), AND that the bundle itself is loaded past its capacity -- two separate
        problems, neither of which a threshold tweak fixes.

        Kept as the measured VSA path for small rigs and as an honest record of the limit; use
        `find_by_role` for anything that must be correct, and `role_recall` to see the gap.
        """
        if not self.roles:
            return []
        atoms = self.role_atoms(dim=dim, seed=seed)["book"]
        if str(role) not in atoms:
            return []
        probe = unbind(self.role_vector(dim=dim, seed=seed), atoms[str(role)])
        hits = [(t, float(cosine(probe, atoms[t]))) for t in self.tags]
        return [t for t, s in sorted(hits, key=lambda r: (-r[1], r[0])) if s >= float(threshold)]

    def role_recall(self, dim=512, seed=0, threshold=0.15):
        """HONEST SCORE for the holographic path: precision/recall of `find_by_role` against the exact
        dict, per role. Reported rather than assumed because a bundle has a capacity cliff and a query
        that silently drops a foot is worse than no query at all."""
        exact = {}
        for tag, rs in self.roles.items():
            for r in rs:
                exact.setdefault(r, set()).add(tag)
        out = {}
        for role, want in exact.items():
            got = set(self.find_by_role_holographic(role, dim=dim, seed=seed, threshold=threshold))
            tp = len(want & got)
            out[role] = {"recall": tp / max(len(want), 1),
                         "precision": tp / max(len(got), 1),
                         "expected": len(want), "returned": len(got)}
        return out


def _segment_tags(bones, chains):
    """Canonical `"<chain>#<index>"` name for every bone, in bone order.

    A bone belongs to the chain in which its two joints appear consecutively. A bone in no chain gets
    `"free#<i>"` rather than being dropped -- a silently unnamed segment is a segment no report,
    weight or role can ever address, which is the burial failure this whole module exists to prevent.
    """
    pos = {}
    for cname, chain in chains.items():
        for i in range(len(chain) - 1):
            pos[(chain[i], chain[i + 1])] = "%s#%d" % (cname, i)
            pos[(chain[i + 1], chain[i])] = "%s#%d" % (cname, i)
    tags, seen = [], {}
    for i, (a, b) in enumerate(bones):
        t = pos.get((a, b), "free#%d" % i)
        if t in seen:                       # two bones claiming one name would silently merge them
            t = "%s.%d" % (t, seen[t])
        seen[t] = seen.get(t, 0) + 1
        tags.append(t)
    return tags


def rotation_invariance_probe(build, measure, axes=((0, 0, 1), (1, 0, 0), (0, 0, -1), (1, 0, 1)),
                              tol=0.0):
    """DOES THIS SURVIVE A CHANGE OF BODY ORIENTATION? -- the directional twin of
    `scale_invariance_probe`.

    That one exists because the SAME absolute-vs-relative bug appeared four times in DISTANCES. It has
    now appeared three times in DIRECTIONS as well:

        mirror plane    reflected across world x=0, so a spine along +x mirrored every limb onto
                        itself and the creature came out with 2 legs instead of 4 -- silently
        limb `dir`      a world vector, so rotating the spine axis left the legs pointing where the
                        ground no longer was (2 feet instead of 4)
        spine `curve`   arches along world +y regardless of which way the body faces

    `build(axis)` makes the thing with its body axis pointed that way; `measure(obj)` returns a
    quantity that SHOULD NOT depend on orientation (a leg count, a segment count, a webbing score).
    Returns {'ok', 'values', 'reference', 'varied', 'axes'} -- a quantity that changes when only the
    ORIENTATION changed is written in world terms where body terms belong.

    Reports rather than raises, for the same reason as the scale probe: the caller usually wants to
    see the broken AND fixed paths side by side, and that comparison is the evidence.

    IT CANNOT KNOW WHICH QUANTITIES SHOULD ROTATE, AND THAT IS NOT A WEAKNESS OF THE PROBE -- it is
    the actual content of the problem. Running this sweep produced a variation that turned out to be
    CORRECT: with a vertical spine the quadruped reports 2 feet instead of 4, and the geometry says
    why -- the body is REARING, its front tips sit at y=0.524 while the back pair sits at y=-0.076, so
    only two limbs touch the ground. `auto_roles` defines a foot by height against the GROUND, and the
    ground does not rotate with the body.

    So the arc's rule is NOT "make everything body-relative". It is: EVERY QUANTITY MUST BE RELATIVE
    TO THE RIGHT FRAME -- the BODY for shape (symmetry, limb direction, blend radii, texture scale)
    and the WORLD for gravity (what counts as a foot, which way is down, standing). The five bugs
    fixed this arc were all shape quantities written in world terms; a naive sweep that "fixed" every
    variation this probe reports would have broken foot detection, which was right all along.
    """
    vals = []
    for ax in axes:
        try:
            vals.append(measure(build(tuple(ax))))
        except Exception as exc:                      # a build that FAILS under rotation is a finding
            vals.append("error: %s" % type(exc).__name__)
    ref = vals[0]
    if isinstance(ref, (int, float)) and not isinstance(ref, bool):
        varied = [v for v in vals if not (isinstance(v, (int, float))
                                          and abs(float(v) - float(ref)) <= float(tol))]
    else:
        varied = [v for v in vals if v != ref]
    return {"ok": not varied, "values": vals, "reference": ref, "varied": varied,
            "axes": [tuple(a) for a in axes]}


def rig_from_primitives(fit, min_length=1e-6):
    """L-1, THE LOOP CLOSED: turn `fit_primitives` output into the SHARED Rig type.

    THE CLAIM THIS MAKES REAL (backlog D-6): generate and observe are the SAME pipeline run in
    opposite directions --

        generate:  spec  -> rig -> tissue -> skin
        observe:   image -> cloud -> rig -> skin

    `fit_primitives` already emits capsules, and a capsule between two points IS a bone segment. It
    was never converted, so the fit path and the creature path were two unrelated worlds that happened
    to have similar shapes -- the exact silent orphaning D-6 warns about, which no existing audit
    catches because the signatures still match.

    Accepts the dict `fit_primitives` returns (or its `parts` list). Each CAPSULE becomes a bone; its
    endpoints come from rotating the capsule's local +Y axis by its stored (axis, angle) and stepping
    +/- the half-height from the centre. SPHERES ARE SKIPPED, not faked into zero-length bones: a
    blob is not a segment, and a degenerate bone would fail `rig_invariant` -- correctly.

    Returns a `Rig` whose segments are tagged `fit#<i>`, so everything downstream (creature_tree,
    tissue_fields, the reports, role tags) works on it with no new code path.
    """
    parts = fit["parts"] if isinstance(fit, dict) else list(fit)
    joints, bones = {}, []
    n = 0
    for kind, params in parts:
        if kind != "capsule":
            continue
        centre = np.asarray(params[0], float)
        half = float(params[1])
        axis, ang = params[3]
        d = _rotate_vector(np.array([0.0, 1.0, 0.0]), np.asarray(axis, float), float(ang))
        a, b = centre - d * half, centre + d * half
        if float(np.linalg.norm(b - a)) < float(min_length):
            continue
        ja, jb = "f%d_a" % n, "f%d_b" % n
        joints[ja], joints[jb] = a, b
        bones.append((ja, jb))
        n += 1
    if not bones:
        raise ValueError("no capsules in the fit -- a rig needs at least one segment "
                         "(spheres are blobs, not bones, and are skipped by design)")
    # One chain per fitted capsule: a point-cloud fit has no parent/child knowledge, and INVENTING a
    # hierarchy here would be a guess dressed as structure. Separate chains is the honest topology,
    # and the canonical tag rule then names them fit0#0, fit1#0, ... via the shared namer.
    chains = {"fit%d" % i: [a, b] for i, (a, b) in enumerate(bones)}
    return Rig(joints, bones, chains, source=fit if isinstance(fit, dict) else None)


def _mesh_of(sdf, lo, hi, res=32):
    """March an SDF to a mesh -- the small bridge the inverse-pipeline selftest needs to hand a
    GENERATED body to the OBSERVING half, which is the only honest way to test a recoverer."""
    # PROBED, NOT REMEMBERED: the marcher lives in holographic_meshbridge as sample_field +
    # marching_tetrahedra_vec (there is no holographic_marching module -- I assumed one and it did
    # not exist).
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
    values, axes = sample_field(sdf, (tuple(lo), tuple(hi)), int(res))
    return marching_tetrahedra_vec(values, axes, level=0.0)


def rig_from_mesh(mesh, res=28, pad=0.1, nbins=12, min_length=1e-6):
    """L-2, THE OTHER HALF OF THE LOOP: recover a SPINED rig from a mesh, plus per-segment thickness.

    `rig_from_primitives` recovers segments but no backbone, so a fitted body had unparented chains
    (nothing to blend to, no articulation) and no anatomy space (hence no organs). This recovers the
    BACKBONE instead: the medial-axis centerline of the mesh, which is a chain, ordered, with the
    medial radius at each node -- i.e. the shape's own measurement of how thick it is there.

    Returns (rig, thickness) where `thickness` maps each segment tag to the MEDIAL RADIUS along it.
    The rig has a single `spine` chain, so its segments are tagged `spine#0`, `spine#1`, ... exactly
    like an authored creature's -- which is what lets the shared joint-blending, tissue and report
    machinery treat a scanned body and a designed one identically.

    KEPT NEGATIVE, INHERITED AND NOT HIDDEN: `skeleton_curve` is SINGLE-BRANCH. It collapses the
    medial ridge along one PCA axis, so it recovers the TORSO of a limbed creature and cuts the
    corner on the limbs -- they are not separate chains here, they are absent. Recovering limbs needs
    branch segmentation of the ridge, which is not built. This is a spine recoverer, not a full rig
    recoverer, and it is named for the part it actually does.
    """
    from holographic.mesh_and_geometry.holographic_skeleton import skeleton_curve
    sc = skeleton_curve(mesh, res=int(res), pad=float(pad), nbins=int(nbins))
    C = np.asarray(sc["curve"], float)
    depth = np.asarray(sc["depth"], float)
    if len(C) < 2:
        raise ValueError("the medial curve collapsed to %d point(s) -- no spine to recover" % len(C))

    joints, bones, chain, thickness = {}, [], [], {}
    for i, p in enumerate(C):
        joints["s%d" % i] = p
        chain.append("s%d" % i)
    kept = 0
    for i in range(len(C) - 1):
        if float(np.linalg.norm(C[i + 1] - C[i])) < float(min_length):
            continue
        bones.append(("s%d" % i, "s%d" % (i + 1)))
        # The medial radius of a SEGMENT is the smaller of its endpoints' -- taking the larger would
        # let a segment claim thickness it does not have along its whole length.
        thickness["spine#%d" % kept] = float(min(depth[i], depth[i + 1]))
        kept += 1
    if not bones:
        raise ValueError("every medial segment was degenerate")
    return Rig(joints, bones, {"spine": chain}), thickness


def infer_tissue_fractions(rig, thickness, bone_frac=0.45, skin_frac=0.06):
    """L-3: infer MUSCLE and FAT thickness from the gap between a fitted bone and the observed skin.

    The anatomy literature derives soft-tissue thickness exactly this way -- the skin is measured, the
    skeleton is known or estimated, and what lies between is apportioned. Here the observed surface is
    the medial radius at each segment (the shape's own half-thickness) and the bone is `bone_frac` of
    it; the remainder, minus a thin skin, is split between muscle and fat.

    Returns {'radii': {tag: observed half-thickness}, 'body': body_params-shaped dict}. The radii are
    the load-bearing output -- feed them to `tissue_fields(..., radii=...)` and the layers are grown
    to the OBSERVED body. The body_params modifiers carry the per-segment DEVIATION from the animal's
    own average thickness, so a barrel-chested body reads positive at the chest and negative at the
    tail, and an inferred body drives `tissue_fields` through the identical control surface an
    authored one does. Observe and generate meet at the same struct, not at a converter.

    THE FIRST VERSION OF THIS WAS A CONSTANT DRESSED AS A MEASUREMENT. It computed the soft-tissue
    modifier as `0.6 * (r * (1 - bone - skin)) / (r * 0.35) - 1`, in which `r` CANCELS -- so every
    segment of every creature returned exactly -0.16 muscle and -0.02 fat, carrying none of the
    observation it claimed to derive from. It looked like inference and was arithmetic. The fix is
    that a per-segment number must be relative to the OTHER SEGMENTS, since a proportion is the only
    thing a single silhouette can actually tell you.

    KEPT NEGATIVE: the muscle/fat SPLIT is not observable from a surface at all. A fat animal and a
    muscular one with the same silhouette are indistinguishable here, so the deviation is applied to
    both in a fixed 60/40 ratio; the honest content is TOTAL soft tissue, not the division.
    """
    vals = {t: float(r) for t, r in thickness.items()}
    if not vals:
        return {"radii": {}, "body": {"muscle": 0.0, "fat": 0.0, "segments": {}}}
    mean_r = float(np.mean(list(vals.values()))) or 1.0
    segs = {}
    for tag, r in vals.items():
        dev = r / max(mean_r, 1e-9) - 1.0        # 0 on an average segment, +ve where the body is thick
        segs[tag] = {"muscle": 0.6 * dev, "fat": 0.4 * dev}
    soft = max(1.0 - float(bone_frac) - float(skin_frac), 0.0)
    return {"radii": {t: r for t, r in vals.items()},
            "soft_fraction": soft,
            "body": {"muscle": 0.0, "fat": 0.0, "segments": segs}}


def _rotate_vector(v, axis, angle):
    """Rodrigues rotation of `v` about a unit `axis` by `angle` -- the inverse of the axis/angle a
    fitted capsule stores, so the fit's own convention is honoured rather than re-derived."""
    axis = np.asarray(axis, float)
    nrm = float(np.linalg.norm(axis))
    if nrm < 1e-12:
        return np.asarray(v, float)
    k = axis / nrm
    v = np.asarray(v, float)
    return (v * np.cos(angle) + np.cross(k, v) * np.sin(angle)
            + k * float(k @ v) * (1.0 - np.cos(angle)))


def scale_invariance_probe(build, reference_length, measure, factor=3.0, tol=0.02):
    """X-3, THE GENERALISED RULE: does a spatial quantity survive a change of body size?

    Backlog D-7 says every spatial frequency must declare a reference length. That rule has now been
    broken FOUR TIMES in this codebase for the same reason, each time discovered separately:

        cell_scale     a raw world texture frequency  -- a 3x creature grew 2x finer skin
        marching res   cells-across set by the whole body, so a thin limb got no say (beading)
        joint blend    an absolute blend radius, so a fat blend crossed the gap between two legs
        organ blend    an absolute metaball k ten times the organ radius -- one giant organ

    A fifth rediscovery is not a matter of remembering harder, so this is the check. `build(scale)`
    returns the thing under test at that body scale, `reference_length(obj)` its declared reference,
    and `measure(obj, L)` the quantity that must stay constant. Returns {'ok', 'at_1x', 'at_Nx',
    'relative_error', 'factor'} -- a quantity that changes by more than `tol` when only the SIZE
    changed is expressed in the wrong units, and the report says so rather than raising, because the
    caller usually wants to see BOTH the broken and the fixed path (that comparison is the evidence).

    This is a probe, not a gate: it cannot know which quantities SHOULD scale with the body. A limb
    length must scale; a scale-count must not. It answers "does this change with size", and the
    caller states which answer is correct.
    """
    a = build(1.0)
    b = build(float(factor))
    La, Lb = float(reference_length(a)), float(reference_length(b))
    ma, mb = float(measure(a, La)), float(measure(b, Lb))
    rel = abs(mb - ma) / max(abs(ma), 1e-12)
    return {"ok": rel <= float(tol), "at_1x": ma, "at_Nx": mb, "relative_error": rel,
            "factor": float(factor), "length_1x": La, "length_Nx": Lb}


def rig_of(source):
    """THE ONE DOOR: a `Rig` view over any skeleton carrying `joints`, `bones` and `chains`.

    Works on a `Creature` and a `Humanoid` unchanged (verified in the selftest), which is what makes
    D-1 real rather than aspirational: downstream code takes a Rig and never asks which kind it was.
    A creature's spine bones live outside `chains`, so a synthetic `spine` chain is derived from
    `spine_nodes` -- giving `spine#0`, the same spelling the skin's `bone_of` uses.
    """
    joints = dict(getattr(source, "joints"))
    bones = list(getattr(source, "bones"))
    chains = dict(getattr(source, "chains", {}))
    nodes = getattr(source, "spine_nodes", None)
    if nodes and "spine" not in chains:
        chains["spine"] = list(nodes)
    return Rig(joints, bones, chains, source=source)


def rig_invariant(source):
    """R-2, THE PINNED INVARIANT: a bone is a single rigid segment between two joints, 1:1 with the
    rig, and it cannot bend in the middle (D-2).

    Returns a report dict and RAISES on violation, because a rig that fails this produces skin weights
    coarser than the rig and limbs that deform as one blended unit -- the B-1 defect, which shipped
    for six sessions precisely because nothing counted. Checks: every bone joint exists; every bone
    has a unique canonical tag; no zero-length segment; tag count == segment count.
    """
    rig = source if isinstance(source, Rig) else rig_of(source)
    missing = [j for a, b in rig.bones for j in (a, b) if j not in rig.joints]
    if missing:
        raise ValueError("bones reference %d joint(s) that do not exist: %r" % (len(missing), missing[:4]))
    if len(set(rig.tags)) != len(rig.bones):
        raise ValueError("segment tags are not 1:1 with bones: %d tags for %d bones"
                         % (len(set(rig.tags)), len(rig.bones)))
    degenerate = [t for t in rig.tags if rig.segment_length(t) < 1e-9]
    return {"segments": len(rig.bones), "tags": len(set(rig.tags)),
            "joints": len(rig.joints), "chains": len(rig.chains),
            "degenerate": degenerate, "reference_length": rig.reference_length()}


def auto_roles(source, ground_frac=0.35):
    """Infer capability roles from the rig's own geometry -- no authoring, any body plan (R-5).

    `foot` : the TIP segment of any chain whose end joint sits in the lowest `ground_frac` of the
             rig's vertical extent. This is how the gait already decides which limbs are legs, reused
             rather than reinvented so the two can never disagree about what a leg is.
    `tip`  : the last segment of every chain (where a hand, mouth or claw would socket).
    `torso`: the spine chain's segments.
    Returns the Rig with roles assigned, so a caller can add or override before querying.
    """
    rig = source if isinstance(source, Rig) else rig_of(source)
    lo, hi = rig.extent()
    span = float(hi[1] - lo[1])
    cut = lo[1] + float(ground_frac) * (span if span > 1e-9 else 1.0)
    for cname, chain in rig.chains.items():
        if len(chain) < 2:
            continue
        tip = "%s#%d" % (cname, len(chain) - 2)
        if tip not in rig.tags:
            continue
        if cname == "spine":
            for i in range(len(chain) - 1):
                t = "%s#%d" % (cname, i)
                if t in rig.tags:
                    rig.tag_role(t, "torso")
            continue
        rig.tag_role(tip, "tip")
        if float(rig.joints[chain[-1]][1]) <= cut:
            rig.tag_role(tip, "foot")
    return rig


def _selftest():
    from holographic.mesh_and_geometry.holographic_creature import Creature, quadruped_spec
    from holographic.mesh_and_geometry.holographic_humanoid import Humanoid
    from holographic.mesh_and_geometry.holographic_creatureskin import creature_metaballs
    import numpy as np

    # 1) ONE TYPE, BOTH SKELETONS (D-1). The same door opens a creature and a humanoid, and the R-2
    # invariant holds on BOTH -- the backlog explicitly requires the invariant on both, because an
    # invariant enforced on one rig is a convention, not a contract.
    spec = quadruped_spec()
    cr = Creature(spec)
    h = Humanoid()
    rc, rh = rig_of(cr), rig_of(h)
    inv_c, inv_h = rig_invariant(cr), rig_invariant(h)
    assert inv_c["segments"] == inv_c["tags"] == len(cr.bones), inv_c
    assert inv_h["segments"] == inv_h["tags"] == len(h.bones), inv_h
    assert not inv_c["degenerate"] and not inv_h["degenerate"], "zero-length segment in a rest rig"

    # 2) THE TAG RULE JOINS TO SKIN PROVENANCE. This is the whole reason the spine spelling was
    # unified: the rig's names and the skin's `bone_of` must be the SAME strings, or every downstream
    # join needs a translation table. Exact set equality, not "mostly overlaps".
    _, _, B = creature_metaballs(cr, spec)
    skin_tags = set(b for b in B if b != "head")
    assert skin_tags == set(rc.tags), \
        "rig tags and skin provenance must be identical: rig-only %r skin-only %r" % (
            sorted(set(rc.tags) - skin_tags)[:3], sorted(skin_tags - set(rc.tags))[:3])

    # 3) A FREE BONE IS NAMED, NOT DROPPED. A segment in no chain must still get a tag, or it becomes
    # unaddressable by every report, weight and role downstream.
    lone = Rig({"a": (0, 0, 0), "b": (1, 0, 0)}, [("a", "b")], {})
    assert lone.tags == ["free#0"], lone.tags

    # 4) ROLES ARE INFERRED FROM GEOMETRY ON ANY BODY PLAN, and a quadruped has four feet -- the
    # number is asserted, because "some feet were found" is exactly the shape of a measurement that
    # passes while the claim is false.
    auto_roles(rc)
    feet = [t for t, rs in rc.roles.items() if "foot" in rs]
    assert len(feet) == 4, "a quadruped has four feet, found %d: %r" % (len(feet), feet)
    n_spine = len(rc.chains["spine"]) - 1
    assert all("torso" in rc.roles.get("spine#%d" % i, set()) for i in range(n_spine)), \
        "every spine segment must be tagged torso"

    # 5) THE HOLOGRAPHIC QUERY AGREES WITH THE DICT -- measured, not assumed, and reported as a score
    # so the capacity cliff is visible rather than trusted.
    rec = rc.role_recall(dim=512, seed=0)
    assert rec["foot"]["recall"] == 1.0 and rec["foot"]["precision"] == 1.0, rec["foot"]
    holo = rc.find_by_role_holographic("foot")
    assert set(holo) == set(feet), "unbind must return exactly the four feet: %r" % holo
    assert set(rc.find_by_role("foot")) == set(feet), "the exact path is the authority and must agree"

    # 5b) THE CAPACITY CLIFF IS REAL AND PINNED, so the limit cannot quietly be forgotten and the
    # holographic path cannot quietly be promoted to authority. A 64-segment rig at 4 roles must
    # still lose most of its assignments -- if this ever starts passing at full recall, the bundle
    # got better and the docstring's measured table is stale.
    big = Rig({"j%d" % i: (float(i), 0.0, 0.0) for i in range(65)},
              [("j%d" % i, "j%d" % (i + 1)) for i in range(64)],
              {"c": ["j%d" % i for i in range(65)]})
    for i in range(64):
        big.tag_role("c#%d" % i, "role%d" % (i % 4))
    big_rec = big.role_recall(dim=512, seed=0)
    mean_recall = float(np.mean([v["recall"] for v in big_rec.values()]))
    assert mean_recall < 0.5, \
        "the measured capacity cliff is gone (recall %.2f) -- re-measure the table" % mean_recall
    assert set(big.find_by_role("role0")) == {"c#%d" % i for i in range(0, 64, 4)}, \
        "the exact path must be unaffected by the bundle's capacity"

    # 6) THE HOLOGRAPHIC PATH IS DETERMINISTIC ACROSS INSTANCES (a fresh rig, same seed, same answer).
    rc2 = auto_roles(rig_of(Creature(spec)))
    assert rc2.find_by_role_holographic("foot") == holo, "role query must be reproducible across rigs"

    # 7) ROLES WORK UNCHANGED ON THE HUMANOID -- the D-1 proof: no branch, no humanoid special case.
    auto_roles(rh)
    hfeet = rh.find_by_role("foot")          # authoritative path, on a rig it has never seen
    assert len(hfeet) == 2, "a biped has two feet, found %r" % hfeet

    # 8) X-3 THE GENERALISED SCALE-INVARIANCE PROBE, demonstrated on the defect it generalises. The
    # probe must SEE the broken absolute frequency and PASS the body-relative one -- a probe that only
    # ever returns ok is the "tool that cannot fail" failure this codebase already found once.
    from holographic.materials_and_texture.holographic_creaturematerial import structure_field

    def _build(s):
        sp = dict(quadruped_spec())
        sp["spine"] = dict(sp["spine"])
        sp["spine"]["length"] *= s
        sp["spine"]["radius"] *= s
        return Creature(sp)

    def _cells(obj, L, **kw):
        f = structure_field("insect", seed=0, **kw)
        n = int(2400 * max(L / 1.5087276941843604, 1.0))
        P = np.stack([np.linspace(0.0, L, n), np.zeros(n), np.zeros(n)], axis=1)
        return int((np.diff((np.asarray(f(P), float) > 0.5).astype(int)) != 0).sum())

    _ref = lambda o: rig_of(o).reference_length()
    broken = scale_invariance_probe(_build, _ref, lambda o, L: _cells(o, L))
    fixed = scale_invariance_probe(_build, _ref, lambda o, L: _cells(o, L, body_length=L))
    assert not broken["ok"] and broken["relative_error"] > 0.5, \
        "the probe must SEE the absolute-frequency defect: %r" % broken
    assert fixed["ok"] and fixed["relative_error"] < 1e-9, \
        "the probe must pass a body-relative quantity: %r" % fixed

    # 9) L-2/L-3 THE INVERSE PIPELINE: recover a SPINED rig from a mesh, and infer tissue from it.
    from holographic.mesh_and_geometry.holographic_creaturetree import creature_tree as _ctree
    from holographic.mesh_and_geometry.holographic_sdf import SDF as _SDF

    _lo, _hi = rc.extent()
    _mesh = _mesh_of(_ctree(rc), _lo - 0.1, _hi + 0.1)
    r_fit, thick = rig_from_mesh(_mesh, res=24)
    inv_fit = rig_invariant(r_fit)
    assert inv_fit["segments"] >= 2 and not inv_fit["degenerate"], inv_fit
    # A RECOVERED SPINE IS A SPINE: its segments must be tagged like an authored one, or none of the
    # shared machinery (joint blending, anatomy space, the reports) treats it as a body.
    assert all(t.startswith("spine#") for t in r_fit.tags), r_fit.tags
    assert set(thick) == set(r_fit.tags), "every recovered segment needs a measured thickness"

    # THE INFERENCE MUST CARRY THE OBSERVATION. The first version computed a modifier in which the
    # measured radius CANCELLED, so every segment of every creature returned the same -0.16 -- an
    # arithmetic identity wearing the costume of a measurement. Requiring VARIATION, and requiring the
    # thickest segment to read highest, is what catches that class.
    inf = infer_tissue_fractions(r_fit, thick)
    muscles = {t: s["muscle"] for t, s in inf["body"]["segments"].items()}
    assert len(set(round(v, 6) for v in muscles.values())) > 1, \
        "inferred tissue must VARY with the observed body, got %r" % muscles
    fattest = max(thick, key=lambda t: thick[t])
    assert muscles[fattest] == max(muscles.values()), \
        "the thickest measured segment must infer the most soft tissue"

    # 10) THE ROTATION PROBE, demonstrated on the defect it generalises AND on a variation that is
    # CORRECT -- both directions, because a probe that only ever fires is as useless as one that never
    # does. World-space limb dirs must SHOW the defect (a rotated spine loses limbs); body-space dirs
    # must be invariant; and the vertical-spine foot count must still vary, because a reared body
    # genuinely has fewer limbs on the ground.
    def _mkb(space):
        def _b(axis):
            _s = quadruped_spec()
            _s["spine"] = dict(_s["spine"]); _s["spine"]["axis"] = axis
            _s["limbs"] = [dict(_l) for _l in _s["limbs"]]
            for _l in _s["limbs"]:
                _l["dir_space"] = space
            return Creature(_s)
        return _b
    _segs = lambda c: len(c.bones)
    _flat = ((0, 0, 1), (1, 0, 0), (1, 0, 1))
    assert not rotation_invariance_probe(_mkb("world"), _segs, axes=_flat)["ok"],         "world-space dirs must still show the rotation defect, or this gate measures nothing"
    assert rotation_invariance_probe(_mkb("body"), _segs, axes=_flat)["ok"],         "body-space dirs must be orientation-invariant"
    _feet = lambda c: len(auto_roles(rig_of(c)).find_by_role("foot"))
    assert not rotation_invariance_probe(_mkb("body"), _feet,
                                         axes=((0, 0, 1), (0, 1, 0)))["ok"],         "a REARED body must have fewer feet on the ground -- gravity is a world quantity, and a "         "sweep that made this invariant would have broken foot detection"

    print("rig selftest OK: creature %d segs / humanoid %d segs both invariant, tags == skin "
          "provenance (%d), %d quadruped feet + %d biped feet by unbind, role recall %.2f, "
          "reference length %.3f"
          % (inv_c["segments"], inv_h["segments"], len(skin_tags), len(feet), len(hfeet),
             rec["foot"]["recall"], inv_c["reference_length"]))


if __name__ == "__main__":
    _selftest()

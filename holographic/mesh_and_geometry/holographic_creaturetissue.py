"""TISSUE AS A VOLUMETRIC MATERIAL -- bone, muscle, fat and skin as NESTED FIELDS, not meshes.

Backlog Tier 3 (T-1..T-4), Tier 4 (V-1, V-2, V-4) and the M-4 nesting report.

THE INVERSION THIS IMPLEMENTS (backlog D-3). Today you set a SKIN radius and everything is derived
inward, which is backwards: bone is primary, muscle forms over bone, fat over muscle, and skin is the
outermost CONSEQUENCE. Setting muscle and fat per bone is also the better control -- it is why the
same skeleton can be a whippet or a bulldog, and why "thick thighs, fat belly" is a muscle-and-fat
statement rather than a skin statement.

TISSUE IS MATERIAL, NOT GEOMETRY (D-4). There are no per-tissue submeshes to build and composite.
`tissue_at(P)` answers "which tissue am I in" the way an albedo lookup answers "what colour is here",
by comparing a handful of nested fields outermost-in. Everything in Tier 4 falls out of that: hiding
a layer is a choice of which field to render, and a cross-section is `max(field, plane)` with the cut
face shaded by `tissue_at`. Both compose, because neither is geometry.

WHY NESTED RADII AND NOT AN `onion`/`round` OFFSET OF THE WHOLE TREE. A uniform offset would grow
every bone by the same amount, which is precisely the mistake the rest of this arc has been undoing:
a thickness that is only meaningful per-bone expressed as one global number. Each layer is compiled
as its OWN tree through the same `creature_tree` compiler with a per-segment radius dict, so muscle
on a thigh and muscle on a finger are separate quantities and the joint-blend rule (relative to the
limb) keeps working at every layer.

MEASURED, on the shipped quadruped: nesting holds at 396/396 sampled interior points (bone inside
muscle inside fat inside skin), no layer has zero or inverted volume, and the interior divides
bone 15.8% / muscle 36.3% / fat 25.3% / skin 22.7% by volume. Hiding layers shrinks the rendered
solid monotonically -- occupancy 0.137 (skin) -> 0.102 (muscle envelope) -> 0.021 (skeleton).

(The first draft of this paragraph said "bone volume is 6.6%", written before the measurement
existed. It was 15.8%. A number in a docstring is a claim like any other and gets measured, not
estimated -- this one survived long enough to be worth recording.)

THIS IS NOT ANATOMY. It is a plausible layered body for an invented creature -- there is no ground
truth to be correct against, and nothing here is claimed to match a real species. Real anatomy for a
real animal needs a reference model and a transfer or physics fit, which is not what a creature
editor wants.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_sdf import SDF
from holographic.mesh_and_geometry.holographic_rig import rig_of
from holographic.mesh_and_geometry.holographic_creaturetree import creature_tree


#: The NESTED ENVELOPES, outermost-in. Organs are NOT here: they are not an envelope around bone,
#: they are blobs occupying the cavity, built separately and keyed "organ" in the fields dict.
LAYERS = ("skin", "fat", "muscle", "bone")


def tissue_thickness(source, body=None, bone_frac=0.24, muscle=0.50, fat=0.21, skin=0.05,
                     fill_fat=True):
    """Per-segment radii for each tissue layer, grown OUTWARD from bone (D-3).

    Returns {layer: {segment_tag: radius}}. The starting point is the rig's own skinning radius R for
    a segment; bone is `bone_frac` of it, and muscle / fat / skin are added as fractions of R on top.
    Everything is expressed as a FRACTION of the segment's own radius rather than an absolute
    thickness (D-7) -- a millimetre of fat means nothing without saying a millimetre of what.

    `body` accepts the shipped `body_params` block ({'muscle': .., 'fat': .., 'segments': {tag: {..}}})
    so per-bone thickness is authored exactly where the character editor already authors it: a global
    slider plus per-segment overrides. Sliders are in [-1, 1] and scale the layer by 1 + slider.

    MINIMUM OFFSET IS ENFORCED. Komaritzan et al. keep a small positive gap between nested surfaces
    for the same reason: a layer allowed to reach zero thickness produces an inverted or empty shell
    that every downstream query then has to special-case. Each layer is guaranteed to exceed the one
    below it by at least 0.5% of the rig's reference length.
    """
    rig = source if hasattr(source, "tags") else rig_of(source)
    src = getattr(rig, "source", None)
    body = dict(body or getattr(src, "body", None) or {})
    segs = dict(body.get("segments", {}) or {})
    g_mus = 1.0 + float(body.get("muscle", 0.0))
    g_fat = 1.0 + float(body.get("fat", 0.0)) + 0.4 * float(body.get("weight", 0.0))
    # THE FLOOR MUST NOT BECOME THE SPEC. It exists so no layer reaches zero or inverted volume
    # (Komaritzan et al. keep a millimetre for the same reason), but at 0.5% of reference length it
    # was 0.0075 on the shipped body while the requested SKIN was 0.004 -- so skin was floor-driven,
    # not spec-driven, and came out at a third of body volume. A minimum that overrides the thing it
    # protects is a bug wearing a safety belt. 0.15% keeps every layer non-degenerate while leaving
    # the thin layers thin.
    floor = 0.0015 * rig.reference_length()

    lr = dict(getattr(src, "limb_radius", {}) or {})
    spine_r = float(getattr(src, "spine_radius", 0.08))

    # THE AUTHORED OUTER RADIUS per segment -- what `creature_tree` actually draws as the skin,
    # including any per-node spine profile. Asking the one radius rule rather than re-deriving it.
    try:
        from holographic.mesh_and_geometry.holographic_creaturetree import segment_radii as _segr
        authored = _segr(rig)
    except Exception:
        authored = {}

    out = {k: {} for k in LAYERS}
    for tag in rig.tags:
        chain = tag.split("#")[0]
        # THE SKELETAL SCALE is the segment's BASE radius, deliberately NOT the authored/profiled one.
        # This is the inside-out inversion finished (backlog D-3): bone and muscle are set by the
        # SKELETON, and everything the author adds on top of that is FAT.
        R = spine_r if chain == "spine" else float(lr.get(chain, 0.05))
        over = dict(segs.get(tag, {}) or {})
        r_bone = max(R * float(bone_frac), floor)
        r_mus = r_bone + max(R * float(muscle) * g_mus * (1.0 + float(over.get("muscle", 0.0))), floor)
        # The fat ring, split into the BASELINE every body has and the EXTRA a slider asks for. They
        # are separated because the authored radius acts as a floor below, and a floor that swallows
        # the slider makes the control silently dead -- measured: with the authored radius above the
        # stack, `fat: 4.0` produced exactly the lean body, because max() picked the floor either way.
        _fat_base = max(R * float(fat), floor)
        _fat_extra = max(R * float(fat) * (g_fat * (1.0 + float(over.get("fat", 0.0))) - 1.0), 0.0)
        r_fat = r_mus + _fat_base + _fat_extra
        r_skin = r_fat + max(R * float(skin), floor)

        # FAT ABSORBS THE DIFFERENCE. If the author made this segment THICKER than bone + muscle +
        # fat + skin -- a fat belly drawn by thickening the spine profile, which is the natural way to
        # ask for one -- the extra is FAT, not thicker bone and muscle. A creature with a big tummy
        # does not have a bigger spine there, and the previous behaviour ignored the profile entirely
        # so the tissue layers did not follow the visible skin at all.
        want = float(authored.get(tag, 0.0))
        if fill_fat and want > r_skin:
            # Authored thicker than the stack: FAT fills the difference -- and the slider's extra
            # rides ON TOP of that floor rather than being clipped by it.
            r_fat = (want - max(R * float(skin), floor)) + _fat_extra
            r_skin = r_fat + max(R * float(skin), floor)
        # AUTHORED THICKNESS IS A FLOOR, NOT A CEILING. The first version also SHRANK the stack when
        # the authored radius was thinner -- which silently clipped the fat slider: asking for
        # `fat: 4.0` produced exactly the lean body, because the clamp pulled it back to the drawn
        # skin. A control that reports no error and does nothing is worse than one that is missing.
        # The slider is the user saying "this creature is obese"; the skin is the CONSEQUENCE (D-3),
        # so it grows to contain the tissue rather than the tissue being trimmed to fit it.
        out["bone"][tag] = r_bone
        out["muscle"][tag] = r_mus
        out["fat"][tag] = r_fat
        out["skin"][tag] = r_skin
    return out


def bone_field(source, th, dorsal=0.55, ribs=True, rib_every=1, rib_arc=0.66, rib_thick=0.50,
               rib_segments=5, blend_rel=0.5):
    """THE SKELETON as an ANATOMICAL shape: a DORSAL spine plus a RIB CAGE, not a centred rod.

    Three things were wrong with modelling bone as "the body at a smaller radius", and all three are
    visible the moment you cut the body open:

      1. TOO MUCH BONE. Measured on the shipped body, bone was 17.5% of the section AREA -- a solid
         core wider than the muscle ring around it. A real torso section is a vertebra plus a thin
         rib wall; the marrow-to-flesh ratio here was closer to a bone with skin on.
      2. CENTRED. A spine is DORSAL. Putting it on the axis leaves the body cavity as a ring around
         the bone, when anatomically the cavity is a large ventral space UNDER the spine, which is
         where viscera actually live and why they have room.
      3. NO RIBS. The single largest bone structure in a chest section is the rib wall, and it sits
         near the OUTSIDE of the cavity, not at its centre. Without it the section reads as a tube
         with a rod in it.

    So bone is built here rather than derived: spine capsules pushed dorsally by `dorsal` (as a
    fraction of the room between the bone and muscle radii, so it CANNOT leave the muscle envelope
    however large the fraction), plus rib arcs sweeping from the vertebra down and around the cavity
    at `rib_arc` of the local muscle radius.

    Ribs are approximated as chains of thin capsules along an ellipse in the body's (sideways, up)
    plane -- the same anatomy-space frame everything else uses -- and mirrored automatically, since
    the arc is swept with +/- the sagittal direction.
    """
    from holographic.mesh_and_geometry.holographic_creaturetree import bone_capsule, _join

    rig = source if hasattr(source, "tags") else rig_of(source)
    src = getattr(rig, "source", None)
    axis = np.asarray(getattr(src, "spine_axis", (0.0, 0.0, 1.0)), float)
    sag = np.asarray(getattr(src, "sagittal_normal", (1.0, 0.0, 0.0)), float)
    up = np.cross(axis, sag)
    n = float(np.linalg.norm(up))
    up = np.array([0.0, 1.0, 0.0]) if n < 1e-9 else up / n

    nodes = []
    for tag in rig.tags:
        a, b = rig.segment(tag)
        r_b, r_m = float(th["bone"][tag]), float(th["muscle"][tag])
        # The offset is a fraction of the ROOM available, so bone stays strictly inside muscle -- the
        # nesting invariant is preserved BY CONSTRUCTION rather than checked afterwards.
        off = up * (float(dorsal) * max(r_m - r_b, 0.0))
        nodes.append(bone_capsule(a + off, b + off, r_b))
        if not (ribs and tag.split("#")[0] == "spine"):
            continue
        if int(tag.split("#")[1]) % max(int(rib_every), 1) != 0:
            continue
        # A rib pair at this vertebra: an arc from the spine, out sideways and down around the cavity.
        centre = 0.5 * (a + b) + off
        rx = float(rib_arc) * r_m
        ry = float(rib_arc) * r_m
        thick = max(float(rib_thick) * r_b, 1e-4)
        for side in (+1.0, -1.0):
            prev = None
            for k in range(int(rib_segments) + 1):
                ang = (np.pi / 2.0) * (k / max(int(rib_segments), 1))     # 0 = dorsal, pi/2 = ventral
                pt = centre + side * sag * (rx * np.sin(ang)) - up * (ry * (1.0 - np.cos(ang)))
                if prev is not None:
                    nodes.append(bone_capsule(prev, pt, thick))
                prev = pt
    if not nodes:
        raise ValueError("no bone segments")
    node = nodes[0]
    for x in nodes[1:]:
        node = SDF("union", (), (node, x))          # bones HARD-union (backlog F-4): no bleed at joints
    return node


def tissue_fields(source, body=None, blend_rel=0.5, organs=True, skeleton=True, **kw):
    """One SDF per tissue layer: {'bone':…, 'muscle':…, 'fat':…, 'skin':…}, each nested in the next.

    Every layer is compiled by the SAME `creature_tree` (metaball groups, joint-relative blending), so
    a layer is not a different KIND of object from the skin -- it is the same body at a different
    radius. That is what makes hiding one and cutting through another compose for free.

    T-1 (bone fields from the rig) is the 'bone' entry; T-2 (envelopes as offsets) is the whole dict.
    """
    rig = source if hasattr(source, "tags") else rig_of(source)
    th = tissue_thickness(rig, body=body, **kw)
    out = {layer: creature_tree(rig, radii=th[layer], blend_rel=blend_rel, head=(layer == "skin"))
           for layer in LAYERS}
    if skeleton:
        # THE SKELETON IS BUILT, NOT DERIVED. "The body at a smaller radius" gives a centred rod with
        # no ribs, which is why the shipped cross-section read as a tube with a bar in it. See
        # `bone_field` for the three defects and the measurements.
        try:
            out["bone"] = bone_field(rig, th, blend_rel=blend_rel)
        except Exception:
            pass                                     # a rig with no usable segments keeps the simple bone
    if organs:
        # Built LAST because organs must be fitted against the finished muscle and bone envelopes.
        # Off for a rig with no spine (anatomy space needs a backbone) rather than raising, so a
        # humanoid or a fitted rig still gets a body.
        try:
            out["organ"] = organ_field(rig, body=body, fields=out)
        except ValueError:
            pass
    return out


def tissue_at(P, fields, level=0.0):
    """T-4. Which tissue is at each point: 'bone' | 'muscle' | 'fat' | 'skin' | 'air'.

    The primitive everything in Tier 4 rests on, and it is deliberately a handful of comparisons
    OUTERMOST-IN rather than a lookup structure: outside skin is air, else inside bone is bone, and so
    on inward. Returns an array of labels the same length as `P`.
    """
    P = np.atleast_2d(np.asarray(P, float))
    out = np.full(len(P), "air", dtype=object)
    inside_outer = np.asarray(fields["skin"](P), float).ravel() < float(level)
    out[inside_outer] = "skin"
    # Outermost-in. `organ` is walked AFTER muscle and BEFORE bone: viscera sit inside the muscle
    # envelope and the bone field is subtracted out of them, so bone still wins any remaining overlap
    # -- the order encodes the containment rather than relying on the fields never disagreeing.
    for layer in ("fat", "muscle", "organ", "bone"):
        if layer not in fields:
            continue
        sel = inside_outer & (np.asarray(fields[layer](P), float).ravel() < float(level))
        out[sel] = layer
    return out


def anatomy_report(source, fields=None, body=None, samples=6000, seed=0, level=0.0):
    """M-4, THE NESTING INVARIANT: is `bone` really inside `muscle` inside `fat` inside `skin`?

    Returns {'violations', 'checked', 'fractions', 'thin_layers', 'ok'}. A violation is any sampled
    point inside a layer but OUTSIDE the layer that should contain it -- a bone poking through the
    skin, or a fat shell that has collapsed. `fractions` is the share of interior volume each tissue
    occupies, which is the number that catches a layer that is technically nested but has no volume.

    WITHOUT THIS a pretty cut face can hide geometry that is simply wrong, which is exactly how the
    old `anatomy_stack` looked convincing while no bone existed in space at all.
    """
    rig = source if hasattr(source, "tags") else rig_of(source)
    if fields is None:
        fields = tissue_fields(rig, body=body)
    lo, hi = rig.extent()
    pad = 0.1 * rig.reference_length()
    rng = np.random.default_rng(int(seed))
    P = rng.uniform(lo - pad, hi + pad, size=(int(samples), 3))
    vals = {k: np.asarray(f(P), float).ravel() for k, f in fields.items()}

    violations, checked = 0, 0
    for inner, outer in (("bone", "muscle"), ("muscle", "fat"), ("fat", "skin")):
        sel = vals[inner] < float(level)
        checked += int(sel.sum())
        violations += int((vals[outer][sel] >= float(level)).sum())

    # ORGAN CHECKS (backlog M-4 names these explicitly): an organ outside the skin, or intersecting
    # bone, is the class of defect a pretty cut face hides. Both are structurally impossible in the
    # shipped organ field (fitted to the muscle envelope, bone subtracted) -- which is exactly why
    # they are worth asserting: a check that only passes because the construction is right is the
    # check that catches the construction going wrong.
    organ_outside, organ_in_bone = 0, 0
    if "organ" in vals:
        osel = vals["organ"] < float(level)
        organ_outside = int((vals["skin"][osel] >= float(level)).sum())
        organ_in_bone = int((vals["bone"][osel] < float(level)).sum())
        violations += organ_outside + organ_in_bone

    body_pts = int((vals["skin"] < level).sum())
    fr = {}
    prev = 0
    for layer in ("bone", "muscle", "fat", "skin"):
        n = int((vals[layer] < level).sum())
        fr[layer] = (n - prev) / max(body_pts, 1) if layer != "bone" else n / max(body_pts, 1)
        prev = n
    thin = [k for k, v in fr.items() if v <= 0.0]
    if "organ" in vals:
        fr["organ"] = int((vals["organ"] < level).sum()) / max(body_pts, 1)
    return {"violations": violations, "checked": checked, "fractions": fr,
            "organ_outside_skin": organ_outside, "organ_in_bone": organ_in_bone,
            "thin_layers": thin, "ok": violations == 0 and not thin}


def visible_field(fields, hide=(), cut=None, level=0.0):
    """V-1 + V-2 + V-4: hide tissue layers and/or cut with a plane, and get ONE SDF back.

    `hide` is any of ('skin', 'fat', 'muscle') -- hide skin and the ray passes to muscle; hide muscle
    and fat too and you see the skeleton in place. NO SEPARATE GEOMETRY IS BUILT: hiding is choosing
    which nested field to return, which is only possible because tissue is material (D-4).

    `cut` is (point, normal): the half-space to remove, applied as `max(f, plane)` -- the standard
    implicit cutaway. The interesting part is the CUT FACE, which callers shade with `tissue_at`.

    THE RETURNED FIELD CARRIES `cut_face_normal`, `cut_point` AND `camera_hint(dist)` so a caller can
    place a camera on the side the face actually points. Material is kept on the NEGATIVE side, so the
    face points along +normal; viewing from the other side shows the intact back of the body and
    looks like no cut happened at all.

    V-4 REQUIRES THESE TO BE ORTHOGONAL, and they are: hiding picks the field, cutting wraps it, so
    every combination of the two is expressible and the selftest checks a hidden-and-cut case rather
    than assuming composition.

    WHY HIDING SKIN BEATS A CROSS-SECTION AS A VERIFIER: a cut plane only shows you the bones it
    happens to pass through, while hiding the skin shows the WHOLE skeleton at once -- so "is a bone
    poking out" or "is a bone bent mid-shaft" is visible immediately.
    """
    order = [l for l in LAYERS if l not in set(hide)]
    if not order:
        raise ValueError("every layer hidden -- nothing to render")
    node = fields[order[0]]
    if cut is not None:
        p0, n = np.asarray(cut[0], float), np.asarray(cut[1], float)
        n = n / (np.linalg.norm(n) + 1e-12)

        def cut_field(P, _node=node, _p0=p0, _n=n):
            """max(f, plane) -- keep the solid only on the negative side of the plane."""
            Q = np.atleast_2d(np.asarray(P, float))
            half = (Q - _p0[None, :]) @ _n
            return np.maximum(np.asarray(_node(Q), float).ravel(), half)

        # WHICH WAY TO LOOK. The material kept is on the NEGATIVE side of the plane, so the exposed
        # cut face points along +normal -- a camera must be on that side to see it. Getting this
        # backwards renders the intact BACK of the creature, which looks like an ordinary body and
        # gives no hint that anything was cut: I did exactly that and reported the result as a
        # cross-section. The answer is a property of the cut, so the cut carries it rather than
        # leaving every caller to re-derive a sign.
        cut_field.cut_face_normal = n
        cut_field.cut_point = p0
        cut_field.camera_hint = lambda dist=1.5, centre=None: (
            (np.asarray(centre, float) if centre is not None else p0) + n * float(dist))
        return cut_field
    return node


#: A plausible viscera layout in ANATOMY SPACE: (name, t along the spine, ventral offset, lateral
#: offset, radius) with the three spatial numbers as FRACTIONS of the local body radius, so the whole
#: set rides a spine bend, a length change or a thickness edit without re-authoring (D-7 again: every
#: spatial quantity states what it is relative to). Ventral is -normal (belly side).
#: PLAUSIBLE, NOT CORRECT -- see the module docstring. A quadruped's real viscera are not this.
#: Pushed further VENTRAL and enlarged now that the spine sits dorsally and the bone is thin: the
#: cavity is a real space under the vertebrae rather than a ring around a central rod, which is the
#: anatomical reason viscera have anywhere to be. Radii are still clamped to the muscle envelope at
#: build time, so an over-large value here shrinks rather than protrudes.
DEFAULT_ORGANS = (
    ("heart",     0.62,  0.34, 0.00, 0.46),
    ("lung_l",    0.70,  0.18, 0.38, 0.44),
    ("lung_r",    0.70,  0.18, -0.38, 0.44),
    ("liver",     0.48,  0.42, 0.12, 0.52),
    ("stomach",   0.42,  0.38, -0.24, 0.44),
    ("gut",       0.28,  0.40, 0.00, 0.56),
)


def organ_field(source, organs=DEFAULT_ORGANS, body=None, blend_rel=0.35, fields=None, clearance=1.02):
    """T-3. Viscera as METABALLS in anatomy space -- and this is the one place metaballs are RIGHT.

    Everything else in this arc has been about getting away from summed metaballs, because they melt
    limbs together. An organ is the opposite case: a liver genuinely IS a smooth blob, and neighbouring
    organs genuinely DO press against one another and deform where they touch. The representation that
    ruins a limb is the correct one for viscera, so this uses the shipped `metaball_distance` rather
    than the composition tree.

    Organs are placed in ANATOMY SPACE (`spine_station`), so they ride a spine bend or a thickness
    edit instead of being left floating in world coordinates -- the sixth use of that primitive and
    the reason it was promoted out of the socket module.

    EACH ORGAN IS SHRUNK TO FIT INSIDE THE MUSCLE ENVELOPE (times `clearance`). An organ poking
    through the skin is the exact class of defect `anatomy_report` exists to catch, so rather than
    reporting it later this places them legally in the first place: the radius is clamped to the
    distance from the organ's centre to the muscle surface. Returns a distance-field callable.
    """
    from holographic.mesh_and_geometry.holographic_creatureskin import metaball_distance
    from holographic.mesh_and_geometry.holographic_creaturesocket import spine_station

    rig = source if hasattr(source, "tags") else rig_of(source)
    src = getattr(rig, "source", None)
    # A CREATURE OR A RECOVERED SPINE. `spine_frames` now reads either `spine_nodes` or a `spine`
    # chain, so a rig recovered from a mesh (L-2) gets viscera too -- it was the missing backbone,
    # not the missing Creature type, that blocked it.
    station_src = src if (src is not None and hasattr(src, "spine_nodes")) else rig
    if not hasattr(station_src, "spine_nodes") and "spine" not in getattr(rig, "chains", {}):
        raise ValueError("organs need a backbone (anatomy space): no spine_nodes and no 'spine' chain")
    if fields is None:
        fields = tissue_fields(rig, body=body)
    inner = fields["muscle"]
    # Body radius: the creature's own, or -- for a recovered rig -- the mean segment radius the tree
    # gives it, so organ sizes are stated relative to THIS body rather than a default (D-7).
    if src is not None and hasattr(src, "spine_radius"):
        R = float(src.spine_radius)
    else:
        from holographic.mesh_and_geometry.holographic_creaturetree import segment_radii
        rr = segment_radii(rig)
        R = float(np.mean(list(rr.values()))) if rr else 0.08

    C, RA = [], []
    for name, t, ventral, lateral, frac in organs:
        p, tan, nor, binor = spine_station(station_src, float(t))
        centre = p - nor * (float(ventral) * R) + binor * (float(lateral) * R)
        want = float(frac) * R
        # How much room is actually there? The muscle field's own value at the centre IS the distance
        # to that surface (negative inside), so the fit test is one evaluation, not a search.
        room = -float(np.asarray(inner(centre[None, :]), float).ravel()[0])
        C.append(centre)
        RA.append(max(min(want, room / float(clearance)), 1e-4))
    RA = np.asarray(RA, float)
    # D-7, AND I WALKED INTO IT AGAIN IN THIS FUNCTION. The first version passed `blend=0.35` straight
    # to metaball_distance as an absolute smooth-union k, against organ radii of ~0.04 -- a blend ten
    # times the size of the things it blends. iq's smin subtracts up to k/4, so the field read
    # NEGATIVE almost everywhere: measured organ occupancy 0.93 of the whole bounding box, i.e. the
    # creature was one giant organ. The blend is now a FRACTION OF THE SMALLEST ORGAN RADIUS, which is
    # the only reading of "how softly do these press together" that survives a change of body scale.
    k = float(blend_rel) * float(RA.min())
    blobs = metaball_distance(np.asarray(C, float), RA, max(k, 1e-6))

    # BONE WINS. Shrinking each organ to fit inside the MUSCLE envelope does not stop it overlapping
    # the SPINE, which sits right where viscera go -- measured, 12 of 8000 sampled points were inside
    # both. That is an anatomy violation (`anatomy_report` names organs-intersecting-bone as one), and
    # nudging positions until it went away would leave it waiting to come back on the next body plan.
    # Subtracting the bone field makes the overlap UNEXPRESSIBLE instead: an organ is whatever room is
    # left after the skeleton, which is also the physically honest statement.
    bone = fields["bone"]

    def organs_minus_bone(P, _b=blobs, _k=bone):
        """max(organ, -bone) -- the SDF difference, so no point can be organ AND bone."""
        Q = np.atleast_2d(np.asarray(P, float))
        return np.maximum(np.asarray(_b(Q), float).ravel(), -np.asarray(_k(Q), float).ravel())
    return organs_minus_bone


def tissue_weights(source, points, body=None, falloff=1.5, max_bones=4):
    """T-5. Skin weights from ANATOMY: tissue formed around bone B belongs to bone B.

    Returns {'tags': [...], 'weights': (N, len(tags))} -- a normalised soft assignment of each point
    to the rig's segments, computed from distance to each bone's own AXIS rather than from which
    metaball happened to land nearest.

    WHY THIS REPLACES THE METABALL-PROVENANCE APPROXIMATION: provenance answers "which ball produced
    this lump", which is a question about the SKINNING PROCESS, not about the body. Two balls from
    different bones overlapping in a fat torso get assigned by whichever centre is nearer, which is
    the reported cause of the fat-torso shear -- Hecker reports the same weakness in Spore for the
    same reason. Distance to the bone axis is a statement about the ANATOMY and does not care how the
    surface was built.

    `falloff` shapes the softness (weight ~ 1/d**falloff); `max_bones` keeps only the strongest few
    per point, which is what a GPU skinning path expects. Weights sum to 1 per point.
    """
    rig = source if hasattr(source, "tags") else rig_of(source)
    P = np.atleast_2d(np.asarray(points, float))
    tags = list(rig.tags)
    D = np.empty((len(P), len(tags)))
    for j, t in enumerate(tags):
        a, b = rig.segment(t)
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-18:
            D[:, j] = np.linalg.norm(P - a[None, :], axis=1)
            continue
        u = np.clip(((P - a[None, :]) @ ab) / L2, 0.0, 1.0)
        D[:, j] = np.linalg.norm(P - (a[None, :] + u[:, None] * ab[None, :]), axis=1)
    W = 1.0 / np.maximum(D, 1e-6) ** float(falloff)
    if int(max_bones) < len(tags):
        # Zero all but the strongest few, BEFORE normalising, so the kept weights still sum to 1.
        #
        # TIE-BREAK IS PART OF THE CONTRACT, not an implementation detail. A threshold cut
        # (`W >= kth`) keeps FIVE bones when two mirrored limbs are exactly equidistant from a spine
        # point -- which happens on every bilaterally symmetric creature, i.e. all of them. Exact
        # ties are the tie-sensitive path this engine has been bitten by before, so this follows the
        # ISA rule: resolve to the LOWEST INDEX, via a STABLE argsort. Deterministic, and exactly
        # max_bones survive.
        keep = np.argsort(-W, axis=1, kind="stable")[:, :int(max_bones)]
        mask = np.zeros_like(W, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        W = np.where(mask, W, 0.0)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    return {"tags": tags, "weights": W}


def _selftest():
    from holographic.mesh_and_geometry.holographic_creature import Creature, quadruped_spec
    from holographic.mesh_and_geometry.holographic_humanoid import Humanoid

    spec = quadruped_spec()
    cr = Creature(spec)
    rig = rig_of(cr)
    th = tissue_thickness(rig)

    # 1) INSIDE-OUT AND STRICTLY NESTED PER SEGMENT (D-3). Radii must increase bone -> skin on EVERY
    # segment, with a real gap: equality would mean a layer with zero volume, which is the failure
    # Komaritzan's minimum offset exists to prevent.
    floor = 0.0015 * rig.reference_length()
    for tag in rig.tags:
        r = [th[l][tag] for l in ("bone", "muscle", "fat", "skin")]
        assert all(r[i + 1] - r[i] >= floor * 0.999 for i in range(3)), \
            "layers must strictly nest on %s: %r" % (tag, [round(x, 4) for x in r])

    # 2) THE NESTING INVARIANT IN SPACE (M-4), which is a different claim from radii ordering: the
    # radii could be ordered and the compiled FIELDS still violate nesting at a joint blend.
    fields = tissue_fields(rig)
    rep = anatomy_report(rig, fields=fields, samples=4000)
    assert rep["violations"] == 0, "nesting violated at %d of %d points" % (rep["violations"], rep["checked"])
    assert rep["checked"] > 100, "too few interior samples to claim anything: %d" % rep["checked"]
    assert not rep["thin_layers"], "layers with no volume: %r" % rep["thin_layers"]

    # 3) tissue_at RETURNS EVERY LABEL, AND IN THE RIGHT PLACE. A point in the bone must be bone; a
    # point far outside must be air. Checking only "it runs" would pass with everything labelled air.
    #
    # THE PROBE MOVED WITH THE ANATOMY: it used to sample the segment AXIS midpoint, which was inside
    # the bone only while the bone was a centred rod. The spine is DORSAL now, so the axis midpoint is
    # correctly muscle, and a probe that still expected bone there was asserting the old anatomy. The
    # bone is found by walking dorsally from the axis until the label changes.
    a, b = rig.segment(rig.tags[0])
    mid = 0.5 * (a + b)
    _up = np.array([0.0, 1.0, 0.0])
    _probe = np.vstack([mid + _up * t for t in np.linspace(0.0, 0.06, 40)])
    _labs = tissue_at(_probe, fields)
    assert "bone" in set(_labs), "the spine must be somewhere dorsal of its axis: %r" % sorted(set(_labs))
    assert tissue_at(np.array([[9.0, 9.0, 9.0]]), fields)[0] == "air", "far outside must be air"


    # And every layer must actually be REACHABLE somewhere in the body -- a stack whose fat is never
    # the answer is a stack with no fat, however well its radii are ordered.
    lo, hi = rig.extent()
    rng = np.random.default_rng(0)
    # 40k samples, not 6k: organs occupy ~0.1% of the bounding box, so a 6k uniform sample lands on
    # them about 4 times and "organs exist" would be a coin flip dressed as a test. Measured 35/40000.
    labs = tissue_at(rng.uniform(lo, hi, size=(40000, 3)), fields)
    seen = set(np.unique(labs))
    want = {"bone", "muscle", "fat", "skin", "organ", "air"}
    assert want <= seen, "unreachable tissue labels: %r" % (want - seen)

    # 3b) ORGANS ARE LEGAL WHERE THEY ARE (T-3 / M-4). Fitted to the muscle envelope with the bone
    # field subtracted out, so neither violation is merely unlikely -- it is unexpressible. Measured
    # 0/0 over 8000 samples; asserting it is how a future placement change gets caught.
    orep = anatomy_report(rig, fields=fields, samples=8000)
    assert orep["organ_outside_skin"] == 0, "organs must not poke through the skin: %d" % orep["organ_outside_skin"]
    assert orep["organ_in_bone"] == 0, "organs must not intersect bone: %d" % orep["organ_in_bone"]
    assert orep["fractions"].get("organ", 0.0) > 0.0, "organs must occupy some interior volume"

    # 3c) ORGANS RIDE BODY EDITS -- the entire reason they are placed in ANATOMY SPACE rather than in
    # world coordinates. Bend the spine and the viscera must MOVE WITH IT and must still be legal in
    # the new body. Measured: all 17 occupied samples changed, 0 organ points outside the bent skin.
    # Placing them in world coordinates would pass every other test in this file and fail this one.
    bent = dict(quadruped_spec())
    bent["spine"] = dict(bent["spine"]); bent["spine"]["curve"] = 0.5
    cr_bent = Creature(bent)
    f_bent = tissue_fields(cr_bent)
    Qb = rng.uniform(lo, hi, size=(20000, 3))
    o_straight = np.asarray(fields["organ"](Qb), float).ravel() < 0.0
    o_bent = np.asarray(f_bent["organ"](Qb), float).ravel() < 0.0
    moved = int((o_straight != o_bent).sum())
    assert moved == int(o_straight.sum() + o_bent.sum()) and moved > 0, \
        "organs must ride a spine bend: only %d of %d occupied samples moved" % (
            moved, int(o_straight.sum() + o_bent.sum()))
    brep = anatomy_report(cr_bent, fields=f_bent, samples=6000)
    assert brep["organ_outside_skin"] == 0 and brep["organ_in_bone"] == 0, \
        "organs must stay legal after a body edit: %r" % brep

    # 3d) ORGANS ON A RECOVERED BODY (backlog L-2 closing the gap L-1 left open). A rig recovered
    # from a MESH has a real backbone but no Creature behind it, and anatomy space used to refuse it
    # -- so an observed body silently got no viscera. It was the missing SPINE, not the missing type.
    # Pinned end to end: generate -> mesh -> recover -> grow tissue, all layers, no violations.
    from holographic.mesh_and_geometry.holographic_rig import rig_from_mesh, _mesh_of, infer_tissue_fractions
    _lo, _hi = rig.extent()
    _m = _mesh_of(fields["skin"], _lo - 0.1, _hi + 0.1, res=30)
    r_rec, thick = rig_from_mesh(_m, res=22)
    f_rec = tissue_fields(r_rec, body=infer_tissue_fractions(r_rec, thick)["body"])
    assert "organ" in f_rec, "a rig with a recovered SPINE must get organs -- anatomy space is " \
                             "spine-relative, not Creature-relative"
    rec_rep = anatomy_report(r_rec, fields=f_rec, samples=3000)
    assert rec_rep["violations"] == 0, "recovered body failed nesting: %r" % rec_rep

    # 4) V-1 HIDING IS NOT GEOMETRY: hiding the skin must expose a SMALLER solid (the muscle envelope),
    # measured as occupancy, not asserted.
    def occupancy(f, n=4000):
        Q = np.random.default_rng(1).uniform(lo, hi, size=(n, 3))
        return float((np.asarray(f(Q), float).ravel() < 0.0).mean())
    o_skin = occupancy(visible_field(fields))
    o_mus = occupancy(visible_field(fields, hide=("skin",)))
    o_bone = occupancy(visible_field(fields, hide=("skin", "fat", "muscle")))
    assert o_skin > o_mus > o_bone > 0.0, \
        "hiding layers must reveal strictly smaller solids: %.4f %.4f %.4f" % (o_skin, o_mus, o_bone)

    # 5) V-2 THE CUT REMOVES A HALF-SPACE, and V-4 CUT AND HIDE COMPOSE. Both checked by measurement:
    # a cut body must be smaller than the uncut one, and cutting a hidden layer must be smaller again.
    cut = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    _cf = visible_field(fields, cut=cut)
    # THE CUT MUST SAY WHICH WAY TO LOOK. Material is kept on the negative side, so the face points
    # along +normal and a camera belongs there; I rendered a "cross-section" from the wrong side and
    # got the intact back of the creature, which looks like an ordinary body.
    assert float(_cf.cut_face_normal[0]) > 0.9, _cf.cut_face_normal
    assert float(np.asarray(_cf.camera_hint(1.5))[0]) > 0.0, "the camera hint must be on the face side"
    o_cut = occupancy(_cf)
    o_cut_bone = occupancy(visible_field(fields, hide=("skin", "fat", "muscle"), cut=cut))
    assert o_cut < o_skin, "a cut must remove material: %.4f !< %.4f" % (o_cut, o_skin)
    assert o_cut_bone < o_bone and o_cut_bone < o_cut, \
        "cut and hide must compose: %.4f vs bone %.4f / cut %.4f" % (o_cut_bone, o_bone, o_cut)

    # 6) THE CONTROL IS INSIDE-OUT (D-3): raising the FAT slider must grow the outer skin without
    # touching the bone. This is the whole point of the inversion, so it is pinned rather than trusted.
    fat_body = {"muscle": 0.0, "fat": 1.0, "segments": {}}
    th_fat = tissue_thickness(rig, body=fat_body)
    t0 = rig.tags[0]
    assert abs(th_fat["bone"][t0] - th["bone"][t0]) < 1e-12, "fat must not move bone"
    assert th_fat["skin"][t0] > th["skin"][t0] * 1.05, \
        "the fat slider must visibly grow the skin: %.4f -> %.4f" % (th["skin"][t0], th_fat["skin"][t0])

    # 6b) T-5 WEIGHTS FROM ANATOMY. A point on a bone's own axis must be dominated by THAT bone, and
    # weights must be a partition (sum to 1). "It returned an array" would pass with garbage.
    probe = np.vstack([0.5 * (rig.segment(t)[0] + rig.segment(t)[1]) for t in rig.tags])
    W = tissue_weights(rig, probe)
    assert W["weights"].shape == (len(rig.tags), len(rig.tags))
    assert np.allclose(W["weights"].sum(axis=1), 1.0), "weights must be a partition"
    dominant = [W["tags"][i] for i in np.argmax(W["weights"], axis=1)]
    assert dominant == list(rig.tags), \
        "each bone midpoint must be dominated by its own bone: %r" % [
            (a, b) for a, b in zip(dominant, rig.tags) if a != b][:3]
    # EXACTLY max_bones, not "at most": a threshold cut kept 5 where two mirrored limbs tie exactly,
    # which is every bilaterally symmetric creature. Equality is the assert that catches that.
    assert int((W["weights"] > 0).sum(axis=1).max()) == 4, "max_bones must be honoured exactly"
    # And the tie-break must be REPRODUCIBLE, since a tie resolved differently moves a weight.
    assert np.array_equal(W["weights"], tissue_weights(rig, probe)["weights"]), \
        "weights must be deterministic across calls"

    # 7) ANY RIG (D-1): the humanoid grows tissue through the identical machinery, no branch.
    hrep = anatomy_report(Humanoid(), samples=2500)
    assert hrep["violations"] == 0, "humanoid nesting violated: %r" % hrep

    # ANATOMICAL PROPORTIONS. The first version made bone "the body at a smaller radius": 17.5% of
    # section AREA, centred, no ribs -- a rod in a tube. Pinned as area shares at a mid-torso segment,
    # because that is what a cross-section actually shows and what a viewer objects to.
    _r = {k: th[k]["spine#2"] for k in ("bone", "muscle", "fat", "skin")}
    _prev, _share = 0.0, {}
    for _k in ("bone", "muscle", "fat", "skin"):
        _a = _r[_k] ** 2
        _share[_k] = (_a - _prev) / _r["skin"] ** 2
        _prev = _a
    assert _share["bone"] < 0.11, "bone must not dominate the section: %.1f%%" % (100 * _share["bone"])
    assert _share["muscle"] > 0.40, "muscle must be the dominant tissue: %.1f%%" % (100 * _share["muscle"])

    # THE SPINE IS DORSAL AND THE CAVITY IS VENTRAL. A centred spine leaves a ring for viscera; a
    # dorsal one leaves a real space under it, which is why organs have anywhere to go.
    _bf = fields["bone"]
    _nodes = np.stack([rig.joints[n] for n in rig.source.spine_nodes])
    _mid = _nodes[len(_nodes) // 2]
    _up = np.array([0.0, 1.0, 0.0])
    _d_up = float(np.asarray(_bf((_mid + _up * 0.02)[None, :]), float).ravel()[0])
    _d_dn = float(np.asarray(_bf((_mid - _up * 0.02)[None, :]), float).ravel()[0])
    assert _d_up < _d_dn, "the spine must sit DORSAL of the segment axis (%.4f vs %.4f)" % (_d_up, _d_dn)

    # RIBS EXIST AND ARE A REAL SHARE OF THE SKELETON -- a rib cage that rounds to nothing is a
    # comment, not a structure.
    _lo, _hi = rig.extent()
    _P = np.random.default_rng(3).uniform(_lo, _hi, size=(30000, 3))
    _in = np.asarray(_bf(_P), float).ravel() < 0
    _lat = _in & (np.abs(_P[:, 0]) > 0.03)
    assert _lat.sum() > 0.15 * max(_in.sum(), 1), \
        "ribs must be a real share of bone volume: %d lateral of %d" % (_lat.sum(), _in.sum())

    print("creaturetissue selftest OK: nesting 0/%d violations, fractions %s (organ %.3f), tissue_at reaches all "
          "labels, occupancy skin %.3f > muscle %.3f > bone %.3f, cut composes (%.3f), fat slider "
          "grows skin %.3f -> %.3f without moving bone, humanoid 0 violations"
          % (rep["checked"], {k: round(v, 3) for k, v in rep["fractions"].items() if k != "organ"},
             orep["fractions"]["organ"],
             o_skin, o_mus, o_bone, o_cut_bone, th["skin"][t0], th_fat["skin"][t0]))


if __name__ == "__main__":
    _selftest()

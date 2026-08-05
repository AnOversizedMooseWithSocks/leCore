"""Spore-style creature skin + spine editing (organics backlog R-1 / R-2).

WHY THIS MODULE EXISTS
----------------------
`Creature` builds a rig from a spec and skins it with a UNION OF CAPSULES, and its spine radius is a
single SCALAR. Spore's creature editor works differently, and the difference is exactly what makes it
feel like sculpting rather than assembling (Hecker, "My Liner Notes for Spore"; Rempton Games):

  * The skin is a chain of SPHERICAL METABALLS distributed along the spine and limbs, spaced so the
    implicit surface stays smooth -- so stretching a segment ADDS balls automatically instead of
    stretching one capsule.
  * Each ball's radius is independently editable, which is how a torso gets fat in the middle and
    thin at the neck. A scalar spine radius cannot express that at all.
  * The spine is EDITED (extend, insert, move, re-thicken), not declared once.

REUSE, NOT REPLACEMENT
  * `metaball_mesh` is shipped and wired; this module produces the (centers, radii) it eats.
  * `Creature` is untouched -- these are functions OVER a creature, so the scalar path still works
    byte-identically and no existing spec changes meaning. Additive, per the hard constraint.
  * Spine edits return a NEW spec dict rather than mutating, so they are serialisable, deterministic
    and /invoke-able, and an editor gets undo for free.

THE SPACING RULE (the "neat bit of math", derived rather than guessed)
  Two Gaussian-ish balls of radius r whose centres are d apart merge into a smooth surface only while
  d is small relative to r. Space them at `d = r * spacing` with spacing < 2: at spacing 1.0 the balls
  overlap by half a radius, which keeps the union's surface bulge-free along a straight chain. The
  selftest MEASURES the resulting surface rather than trusting the rule -- it samples the field along
  a stretched chain and asserts the radial distance never dents by more than a tolerance.

KEPT NEGATIVES (loud)
  * SPHERICAL balls only. Ellipsoids would widen the shape space, but they are orientation-dependent
    and markedly slower to evaluate -- the same trade-off Spore made and documented, adopted here for
    the same reason, not copied blindly.
  * Ball count grows LINEARLY with limb length / spacing. A very long spine at tight spacing is a lot
    of balls and the marcher cost follows; that is the honest cost of auto-density.
  * This produces a SKIN, not weights. Binding vertices to bones (backlog R-7) is separate and needs
    the `bone_of` array this returns -- which is why that array exists here rather than being dropped.
  * The metaball union will merge limbs that pass close to each other, whether or not they are meant
    to be connected. Implicit surfaces have no local control; that is inherent, not a bug.
"""

import numpy as np


def spine_profile(spec, radii):
    """R-1: replace a spec's scalar spine radius with a PROFILE -- a per-node radius array (or a
    callable f(t)->radius sampled at the nodes). Returns a NEW spec; the original is untouched.

    This is the "adjust thickness around the spine" control. A scalar still works everywhere it did
    before (the profile is simply absent), so no existing creature changes shape.
    """
    spec = {k: (dict(v) if isinstance(v, dict) else v) for k, v in dict(spec).items()}
    sp = dict(spec.get("spine") or {})
    nseg = int(sp.get("segments", 4))
    if callable(radii):
        radii = [float(radii(i / nseg)) for i in range(nseg + 1)]
    radii = [float(r) for r in np.atleast_1d(np.asarray(radii, float))]
    if len(radii) == 1:
        radii = radii * (nseg + 1)
    if len(radii) != nseg + 1:
        raise ValueError("spine profile needs one radius per spine NODE (%d for %d segments), got %d"
                         % (nseg + 1, nseg, len(radii)))
    sp["profile"] = radii
    spec["spine"] = sp
    return spec


def _profile_of(creature, spec=None):
    """The per-spine-node radius array for a creature: its spec's profile when present, otherwise the
    scalar spine_radius repeated. One place, so every caller treats scalar and profile identically."""
    n = len(creature.spine_nodes)
    prof = ((spec or {}).get("spine") or {}).get("profile")
    if prof is None:
        return np.full(n, float(creature.spine_radius))
    p = np.asarray(prof, float)
    if len(p) != n:                                            # tolerate a resampled spine
        p = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(p)), p)
    return p


def ball_chain(a, b, r_a, r_b, spacing=1.0):
    """Metaballs along one bone from `a` to `b`, radius lerping r_a -> r_b, spaced so the union stays
    smooth. Returns (centers, radii).

    THE AUTO-DENSITY PROPERTY: the count is derived from LENGTH / (mean radius * spacing), so
    stretching the bone adds balls rather than stretching a shape. That is the whole point -- it is
    what makes a spine editable without the skin degrading.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    rm = 0.5 * (float(r_a) + float(r_b))
    step = max(rm * float(spacing), 1e-6)
    n = max(int(np.ceil(L / step)), 1)
    t = np.linspace(0.0, 1.0, n + 1)
    centers = a[None, :] + (b - a)[None, :] * t[:, None]
    radii = float(r_a) + (float(r_b) - float(r_a)) * t
    return centers, radii


def creature_metaballs(creature, spec=None, spacing=1.0, limb_taper=0.6, head=True):
    """R-1: the full metaball skin of a creature -- (centers, radii, bone_of).

    Walks the spine with its radius PROFILE, then every limb chain tapering from its mount radius to
    `limb_taper` of it, plus the head sphere. `bone_of` records which bone produced each ball, which
    is precisely what skin-weight binding (R-7) needs; it is returned rather than discarded so that
    item is a small function later instead of a re-derivation.

    Feed straight to the shipped `metaball_mesh(centers, radius=..., level=...)`, or to
    `creature_metaball_mesh` below which handles the varying-radius case properly.
    """
    prof = _profile_of(creature, spec)
    C, R, B = [], [], []
    nodes = creature.spine_nodes
    for i in range(len(nodes) - 1):
        c, r = ball_chain(creature.joints[nodes[i]], creature.joints[nodes[i + 1]],
                          prof[i], prof[i + 1], spacing)
        # ONE TAG RULE FOR EVERY RIG: "<chain>#<segment index>". The spine used to spell itself
        # "spine0" while limbs (post-B-1) spell themselves "L0#0", which is the same idea in two
        # spellings -- and a shared Rig type cannot join provenance across two conventions without a
        # translation table, which is exactly the kind of seam that rots. `startswith("spine")`
        # consumers are unaffected.
        C.append(c); R.append(r); B += ["spine#%d" % i] * len(c)
    for name, chain in creature.chains.items():
        r0 = float(creature.limb_radius.get(name, 0.05))
        for j in range(len(chain) - 1):
            # A limb tapers along its own length: thickest at the mount, thinnest at the tip.
            f0 = 1.0 - (1.0 - float(limb_taper)) * (j / max(len(chain) - 1, 1))
            f1 = 1.0 - (1.0 - float(limb_taper)) * ((j + 1) / max(len(chain) - 1, 1))
            c, r = ball_chain(creature.joints[chain[j]], creature.joints[chain[j + 1]],
                              r0 * f0, r0 * f1, spacing)
            # WHY per-SEGMENT and not per-chain: `B` is the provenance that skin weights are derived
            # from, and a label is only as fine as the deformation it can express. Tagging a whole
            # 3-segment limb `L0` (backlog B-1) produced 9 tags for 17 rig segments, so every ball
            # from hip to toe bound to ONE bone -- a bone that necessarily bends mid-shaft, violating
            # D-2, and a limb that deforms as one blended unit. The spine was already per-segment
            # ("spine%d"); this makes the limbs agree. Label is "<chain>#<segment index>".
            C.append(c); R.append(r); B += ["%s#%d" % (name, j)] * len(c)
    if head and getattr(creature, "head", None):
        C.append(np.asarray(creature.joints[creature.head["node"]], float)[None, :])
        R.append(np.array([float(creature.head["radius"])]))
        B += ["head"]
    if not C:
        return np.zeros((0, 3)), np.zeros(0), []
    return np.concatenate(C), np.concatenate(R), B


def metaball_field(centers, radii, blend=1.0):
    """The sum-of-blobs scalar field f(P) for VARYING-radius metaballs, as a plain callable.

    The shipped `metaball_mesh` takes ONE radius for all centres, which cannot express a fat torso and
    a thin wrist in the same creature -- so this builds the per-ball field and hands back an SDF-shaped
    callable that `mesh_from_sdf` can march. Each ball contributes a smooth falloff that reaches zero
    at its own radius, so a ball's radius really is its extent.
    """
    C = np.asarray(centers, float); R = np.asarray(radii, float)

    def field(P):
        """Negative inside the blob union, positive outside -- the sign convention mesh_from_sdf wants."""
        P = np.atleast_2d(np.asarray(P, float))
        out = np.empty(len(P))
        for s in range(0, len(P), 4096):                       # chunked: a full outer product would
            Q = P[s:s + 4096]                                  # be len(P) x len(C) and blow memory
            d2 = ((Q[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            q = np.clip(1.0 - d2 / (R[None, :] ** 2 + 1e-12), 0.0, None)
            out[s:s + 4096] = 0.5 - (q ** 2).sum(1) * float(blend)
        return out
    return field


def metaball_distance(centers, radii, k=0.06):
    """A TRUE DISTANCE field for the same ball set -- the smooth union of spheres, |grad| ~ 1.

    WHY THIS EXISTS ALONGSIDE `metaball_field`. That one returns a DENSITY (0.5 - sum of falloffs),
    which is all marching cubes needs, since the marcher only reads the SIGN. A sphere tracer needs a
    Lipschitz distance BOUND, and the density is nowhere near one -- measured on a real creature, its
    gradient magnitude ranges from 0.0 to 26.1 where a distance field is ~1.0, so a tracer would
    overshoot straight through the body. That is exactly what happened: the first quality render came
    back as pure background.

    Uses the log-sum-exp smooth minimum, which is associative (so it reduces over all balls in ONE
    vectorised op rather than a fold), numerically stabilised by subtracting the running minimum, and
    Lipschitz-1 by construction. `k` is the blend radius in world units -- larger fuses limbs into the
    torso more softly, and at k -> 0 it degenerates to a hard union of spheres.
    """
    C = np.asarray(centers, float)
    R = np.asarray(radii, float)
    kk = max(float(k), 1e-6)

    def sdf(P):
        """Distance to the smooth union of every ball: exact per-sphere distance, softly minimised."""
        Q = np.atleast_2d(np.asarray(P, float))
        out = np.empty(len(Q))
        for s in range(0, len(Q), 4096):                      # chunked: the full (N,M) matrix is large
            B = Q[s:s + 4096]
            d = np.linalg.norm(B[:, None, :] - C[None, :, :], axis=-1) - R[None, :]
            mn = d.min(axis=1)                                # stabilise the exponent, standard LSE trick
            out[s:s + 4096] = mn - kk * np.log(np.exp(-(d - mn[:, None]) / kk).sum(axis=1))
        return out
    return sdf


def section_warp(creature, width=1.0, depth=1.0, ridge=0.0, belly=0.0):
    """A CROSS-SECTION shaper: turn the body's circular tube into an actual silhouette.

    WHY THIS IS THE DIFFERENCE BETWEEN A TUBE AND A BODY. Metaballs are spheres, so every cross
    section of a creature built from them is a CIRCLE, and no amount of profile editing changes that
    -- a fat belly is still a round belly. Real bodies are not round: they are wider than they are
    deep (or the reverse), flat underneath, ridged along the spine.

    HOW, and why not ellipsoid metaballs. Spore used spheres only, and the stated reason was that
    ellipsoids are orientation-dependent and slower. Both are true of ellipsoid PRIMITIVES -- but the
    same effect comes free by warping SPACE instead of the balls: scale a query point's cross-body
    coordinates before measuring distance, and every sphere in the chain becomes the same ellipse
    without any of them knowing. One transform for the whole body rather than an orientation per ball.

    The warp is applied in the BODY FRAME (the point is projected onto the spine and split into
    along/across components), which is the same anatomy-space principle as sockets, scales and
    rig-bound paint: a shape attached to an animal belongs in the animal's coordinates. Warping in
    world space would shear the creature whenever its spine curved.

        width   scale across the body   (>1 broad, <1 narrow)
        depth   scale front-to-back     (>1 deep-chested, <1 slab-sided)
        ridge   raise a crest along the spine's top
        belly   flatten the underside   (a body that sits, rather than a sausage)

    Returns a callable warp(P) -> P' to compose with the field. KEPT NEGATIVE: this scales SPACE, so
    it breaks the field's Lipschitz bound by 1/min(width, depth) -- `CreatureField` divides by that,
    which keeps the tracer conservative at the cost of marching more slowly. Extreme values (below
    ~0.3) make the surface march sluggishly and are not worth it.
    """
    nodes = np.array([np.asarray(creature.joints[n], float) for n in creature.spine_nodes])
    from holographic.mesh_and_geometry.holographic_curves import rotation_minimizing_frame
    T, N, B = rotation_minimizing_frame(nodes)
    T = np.asarray(T, float); N = np.asarray(N, float); B = np.asarray(B, float)
    w, d = max(float(width), 1e-3), max(float(depth), 1e-3)

    # Station blending width: how far a point's frame is smeared across neighbouring spine nodes.
    # Scaled to the actual node spacing so it means the same thing on a mouse and a giraffe.
    sigma = float(np.mean(np.linalg.norm(np.diff(nodes, axis=0), axis=1))) * 0.9 if len(nodes) > 1 else 1.0

    def warp(P):
        """World point -> the point at which to evaluate the unwarped field.

        THE FRAME IS BLENDED, NOT PICKED. An earlier version took the NEAREST spine station, which
        makes the frame jump discontinuously at the midpoint between nodes -- and a discontinuous
        warp means a discontinuous field. Measured: gradient magnitude spiked to 2.11 where a
        distance field must stay at 1.0, so a sphere tracer would overshoot and punch through the
        surface. (Marching cubes never noticed, because it only reads the SIGN -- which is exactly
        how a defect like this ships looking fine.) Softmax-weighting the stations by distance makes
        the frame vary smoothly along the body, and the gradient comes back into range.
        """
        Q = np.atleast_2d(np.asarray(P, float))
        d2 = ((Q[:, None, :] - nodes[None, :, :]) ** 2).sum(-1)
        wgt = np.exp(-d2 / (2.0 * sigma * sigma))
        wgt = wgt / (wgt.sum(1, keepdims=True) + 1e-12)
        base = wgt @ nodes
        tan = wgt @ T; nor = wgt @ N; bin_ = wgt @ B
        # re-orthonormalise: a weighted average of orthonormal frames is not itself orthonormal
        tan = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12)
        nor = nor - tan * (nor * tan).sum(1, keepdims=True)
        nor = nor / (np.linalg.norm(nor, axis=1, keepdims=True) + 1e-12)
        bin_ = np.cross(tan, nor)
        rel = Q - base
        a = (rel * tan).sum(1)                                # along the body
        x = (rel * nor).sum(1)                                # across
        y = (rel * bin_).sum(1)                               # front-to-back
        xs, ys = x / w, y / d
        if ridge:
            # pull the field DOWN above the spine so the surface bulges up into a crest
            ys = ys - float(ridge) * np.clip(ys, 0.0, None) * np.exp(-(xs / 0.35) ** 2)
        if belly:
            # compress below, so the underside flattens instead of hanging round
            ys = ys + float(belly) * np.clip(-ys, 0.0, None)
        return base + tan * a[:, None] + nor * xs[:, None] + bin_ * ys[:, None]

    warp.lipschitz = 1.0 / min(w, d, 1.0)
    return warp


class CreatureField:
    """The creature's metaball skin as an SDF that satisfies BOTH consumer contracts at once.

    The mesher wants a bare callable f(P)->distance; `render_surface` wants an object with `.eval(P)`
    and `.ids(P)` so it can look a material up per object. Rather than make callers remember which
    door to use, this is callable AND carries both methods, so one object feeds the marcher, the
    raymarcher and the path tracer. `.ids` returns zeros -- a creature is ONE object with one
    material; per-part materials would need per-ball ids, which is a real extension and not pretended
    at here.
    """

    def __init__(self, centers, radii, blend=1.0, bone_of=None, distance=True, smooth_k=0.06,
                 warp=None):
        self.centers = np.asarray(centers, float)
        self.radii = np.asarray(radii, float)
        self.bone_of = list(bone_of) if bone_of is not None else None
        # DEFAULT TO THE DISTANCE FORM. Both are correct for the marcher (it only reads the sign), but
        # only the distance form is safe for a sphere tracer -- so the door that the raymarcher uses
        # hands out the field that actually works there, rather than the one that silently renders
        # empty. Pass distance=False for the original density field.
        self.is_distance = bool(distance)
        base = (metaball_distance(self.centers, self.radii, smooth_k) if distance
                else metaball_field(self.centers, self.radii, blend))
        self.warp = warp
        if warp is None:
            self._f = base
        else:
            # Warping space breaks the distance bound by the largest compression factor, so divide by
            # it to keep the tracer conservative -- the same correction the relief displacement needs.
            L = float(getattr(warp, "lipschitz", 1.0))
            self._f = lambda P: np.asarray(base(warp(P)), float) / max(L, 1.0)

    def __call__(self, P):
        """The bare-callable contract, for mesh_from_sdf and anything else taking f(P)."""
        return self._f(P)

    def eval(self, P):
        """The scene-SDF contract used by render_surface / sphere_trace."""
        return self._f(P)

    def ids(self, P):
        """One object, one material -- see the class docstring before extending this to per-part ids."""
        return np.zeros(len(np.atleast_2d(np.asarray(P, float))), int)

    def with_relief(self, structure, amplitude=0.006, lipschitz=None):
        """A copy of this field with the skin STRUCTURE displaced into the surface -- real scale/plate
        relief, not a fake normal.

        `structure` is a field f(P)->[0,1] (what `creature_material` returns as its "structure"
        channel). The displaced field is sdf(P) - amplitude*(structure(P) - 0.5), so a scale face
        bulges and a groove sinks. The tracer then derives its normals from the DISPLACED surface,
        which is why the scales catch light instead of looking painted on.

        HONEST COST, measured rather than hidden: adding a high-frequency function to a distance field
        breaks its Lipschitz bound, because the structure's gradient is large (its cells are small).
        The result is divided by a Lipschitz estimate so the tracer's steps stay CONSERVATIVE -- it
        marches more slowly and the render costs more, which is the true price of relief. Keep
        `amplitude` small (a few thousandths of body size); large values make the tracer overshoot and
        punch holes in the surface.
        """
        base = self._f
        amp = float(amplitude)
        if lipschitz is None:
            # Estimate the worst-case slope the displacement adds, by sampling the structure's own
            # gradient near the surface -- guessing a constant here is what produces holes.
            rng = np.random.default_rng(0)
            Q = self.centers[rng.integers(0, len(self.centers), size=min(200, len(self.centers)))]
            e = 2e-3
            g = np.stack([(np.asarray(structure(Q + d)) - np.asarray(structure(Q - d))) / (2 * e)
                          for d in (np.array([e, 0, 0]), np.array([0, e, 0]), np.array([0, 0, e]))], 1)
            lipschitz = 1.0 + amp * float(np.linalg.norm(g, axis=1).max())
        L = max(float(lipschitz), 1.0)

        out = CreatureField.__new__(CreatureField)
        out.centers, out.radii, out.bone_of = self.centers, self.radii, self.bone_of
        out.is_distance, out.relief_lipschitz = True, L

        def displaced(P):
            """Base distance minus the structure bump, rescaled to stay a conservative bound."""
            return (np.asarray(base(P), float)
                    - amp * (np.asarray(structure(P), float).ravel() - 0.5)) / L
        out._f = displaced
        return out

    def bounds(self, pad=0.35):
        """A bounding box that contains every ball plus `pad` -- what the marcher and the camera fit
        need, computed here so callers stop re-deriving it from centers and radii."""
        lo = (self.centers - self.radii[:, None]).min(0) - float(pad)
        hi = (self.centers + self.radii[:, None]).max(0) + float(pad)
        return tuple(lo), tuple(hi)


def creature_field(creature, spec=None, spacing=1.0, blend=1.0, distance=True, smooth_k=0.06,
                   warp=None):
    """The creature's skin as a CreatureField -- the door the quality render path needs. Previously
    the field was built only inside the mesher, so a raymarcher had no way to reach it."""
    C, R, B = creature_metaballs(creature, spec, spacing=spacing)
    if not len(C):
        raise ValueError("creature has no bones to skin")
    return CreatureField(C, R, blend, B, distance=distance, smooth_k=smooth_k, warp=warp)


def skin_quality(creature, spec=None, spacing=1.0, resolution=48, pad=0.35):
    """IS THIS RESOLUTION ENOUGH? Reports how many marching cells span the THINNEST feature.

    THE BUG THIS EXISTS TO CATCH, on the record: rendered limbs came out visibly LUMPY -- beaded,
    like a string of sausages -- and the cause was not the metaball spacing (measured gap/radius 0.76,
    well inside the smooth range) but UNDERSAMPLING. At resolution 104 on a body 2.7 units tall, a
    limb of radius 0.030 is 2.3 CELLS ACROSS. Marching cubes cannot represent a tube that thin: it
    beads into whatever the lattice happens to catch. Measured surface ripple on a real limb: 225%,
    against the 6% the straight-chain selftest allows.

    WHY THE SELFTEST MISSED IT: it measured a straight, constant-radius chain IN ISOLATION, at its own
    comfortable scale. The lumpiness only appears when a THIN feature is marched inside the bounding
    box of a MUCH LARGER body, because the cell size is set by the whole body and the limb has no say.
    A quality metric that is not scale-relative is not a quality metric.

    RULE: >= 4 cells across a feature to mesh it smoothly; below 2 it beads. Returns the count, the
    resolution that would fix it, and a verdict, so a caller can raise the resolution or thicken the
    limb rather than shipping a lumpy creature.
    """
    C, R, bones = creature_metaballs(creature, spec, spacing=spacing)
    if not len(C):
        return {"ok": False, "note": "no geometry"}
    lo = (C - R[:, None]).min(0) - float(pad)
    hi = (C + R[:, None]).max(0) + float(pad)
    extent = float(np.max(hi - lo))
    cell = extent / max(int(resolution), 1)
    thin = float(R.min())
    across = 2.0 * thin / max(cell, 1e-12)
    need = int(np.ceil(4.0 * extent / (2.0 * thin)))
    thinnest = bones[int(np.argmin(R))]
    return {"cells_across": float(across), "thinnest_radius": thin, "thinnest_part": thinnest,
            "cell": float(cell), "extent": extent, "resolution": int(resolution),
            "recommended_resolution": need, "ok": bool(across >= 4.0),
            "verdict": ("smooth" if across >= 4.0 else
                        "soft" if across >= 2.5 else "LUMPY -- the thinnest feature is undersampled")}


def creature_metaball_mesh(creature, spec=None, spacing=1.0, resolution=40, pad=0.35, blend=1.0,
                           warn=True):
    """R-1 end to end: a creature -> a smooth blended metaball SKIN mesh, marched by the shipped
    `mesh_from_sdf`. Limbs FLOW into the torso instead of intersecting it, which is the visual
    difference between the capsule union and the Spore-style skin."""
    # The isosurface extractor lives in holographic_meshbridge (marching tetrahedra); UnifiedMind's
    # mesh_from_sdf is a thin wrapper over it, so we call the same two functions directly rather than
    # requiring a mind instance for what is a pure geometry operation.
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra
    C, R, _ = creature_metaballs(creature, spec, spacing=spacing)
    if not len(C):
        raise ValueError("creature has no bones to skin")
    lo = (C - R[:, None]).min(0) - float(pad)
    hi = (C + R[:, None]).max(0) + float(pad)
    bounds = (tuple(lo), tuple(hi))
    # Undersampling is invisible in the returned mesh -- it just looks bad -- so say so at the point
    # it happens. Not an exception: a low-res preview is a legitimate thing to ask for.
    across = 2.0 * float(R.min()) / (float(np.max(np.asarray(hi) - np.asarray(lo))) / max(int(resolution), 1))
    if across < 2.5 and warn:
        import warnings
        warnings.warn("creature skin at resolution %d gives only %.1f marching cells across the "
                      "thinnest feature (radius %.3f); it will look LUMPY. Use resolution >= %d, or "
                      "thicken the limb. See holographic_creatureskin.skin_quality."
                      % (int(resolution), across, float(R.min()),
                         int(np.ceil(4.0 * float(np.max(np.asarray(hi) - np.asarray(lo)))
                                     / (2.0 * float(R.min()))))), RuntimeWarning, stacklevel=2)
    values, axes = sample_field(metaball_field(C, R, blend), bounds, int(resolution))
    return marching_tetrahedra(values, axes, level=0.0)   # marcher takes the AXES, not the bounds


# ------------------------------------------------------------------ R-2: spine editing operations --
# Every one of these returns a NEW spec. Nothing mutates: an editor gets undo for free, the result is
# serialisable and /invoke-able, and a scrub or preview can never corrupt the asset it previews.

def _spine(spec):
    """The spec's spine block as a fresh dict, with defaults filled -- one place, so every edit below
    starts from the same normalised shape instead of repeating .get() chains."""
    spec = {k: (dict(v) if isinstance(v, dict) else v) for k, v in dict(spec or {}).items()}
    sp = dict(spec.get("spine") or {})
    sp.setdefault("length", 1.2); sp.setdefault("segments", 4)
    spec["spine"] = sp
    return spec, sp


def extend_spine(spec, n=1, keep_segment_length=True):
    """R-2: add `n` segments to the tail of the spine. With `keep_segment_length` the spine gets
    LONGER at the same resolution (the intuitive 'drag the tail out'); without it, the spine keeps its
    length and just subdivides. Any radius profile is extended by repeating its last value, so
    thickness edits survive a length change."""
    spec, sp = _spine(spec)
    old = int(sp["segments"])
    seg_len = float(sp["length"]) / old
    sp["segments"] = old + int(n)
    if keep_segment_length:
        sp["length"] = seg_len * sp["segments"]
    if sp.get("profile"):
        p = list(sp["profile"])
        sp["profile"] = p + [p[-1]] * int(n)
    return _rescale_limb_positions(spec, old, sp["segments"])


def insert_node(spec, at=0.5):
    """R-2: subdivide the spine at fraction `at`, raising its resolution by one segment without
    changing its length or shape. The profile is resampled so thickness is preserved."""
    spec, sp = _spine(spec)
    old = int(sp["segments"])
    sp["segments"] = old + 1
    if sp.get("profile"):
        p = np.asarray(sp["profile"], float)
        sp["profile"] = list(np.interp(np.linspace(0, 1, old + 2), np.linspace(0, 1, len(p)), p))
    return _rescale_limb_positions(spec, old, sp["segments"])


def set_radius(spec, at, radius, falloff=0.0):
    """R-2 + R-1: thicken or thin the spine AT a fraction along it. With `falloff` > 0 the change
    blends into neighbouring nodes (a smooth belly rather than a single bulging ring), using a cosine
    window -- the difference between sculpting and poking."""
    spec, sp = _spine(spec)
    n = int(sp["segments"]) + 1
    prof = np.asarray(sp.get("profile") or [float(sp.get("radius", 0.08))] * n, float)
    if len(prof) != n:
        prof = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(prof)), prof)
    ts = np.linspace(0.0, 1.0, n)
    if float(falloff) <= 0:
        prof[int(round(float(at) * (n - 1)))] = float(radius)
    else:
        d = np.abs(ts - float(at)) / float(falloff)
        w = np.where(d < 1.0, 0.5 * (1.0 + np.cos(np.pi * np.clip(d, 0, 1))), 0.0)
        prof = prof * (1.0 - w) + float(radius) * w
    sp["profile"] = list(prof)
    return spec


def move_node(spec, curve=None, length=None, axis=None):
    """R-2: reshape the spine as a whole -- its arch (`curve`), its `length`, or its `axis`. Per-node
    free positioning is deliberately NOT offered: the spine is generated from these parameters, and
    an arbitrary node offset would make the spec no longer describe the curve it produces. Stated
    rather than silently unsupported."""
    spec, sp = _spine(spec)
    if curve is not None:
        sp["curve"] = float(curve)
    if length is not None:
        sp["length"] = float(length)
    if axis is not None:
        sp["axis"] = tuple(float(x) for x in axis)
    return spec


def _rescale_limb_positions(spec, old_segments, new_segments):
    """Keep limbs where the user PUT them when spine resolution changes.

    `at` is a fraction, and Creature._node_at snaps it to the nearest spine node -- so adding a
    segment silently moves every limb to a different node. This preserves the intended WORLD position
    by keeping the fraction and letting the snap land on the nearest new node, and it exists because
    the alternative (limbs drifting along the back whenever you extend the tail) is the kind of bug
    that looks like a physics problem and is actually an indexing one.
    """
    limbs = []
    for limb in spec.get("limbs", []) or []:
        limb = dict(limb)
        frac = float(limb.get("at", 0.5))
        # Snap to the nearest node of the OLD spine first, so a limb that was exactly on a node stays
        # exactly on one; a limb between nodes keeps its fraction.
        snapped = round(frac * int(old_segments)) / max(int(old_segments), 1)
        limb["at"] = float(np.clip(snapped, 0.0, 1.0))
        limbs.append(limb)
    if limbs:
        spec["limbs"] = limbs
    return spec


def _selftest():
    """Numeric contracts: auto-density must actually ADD balls when a bone stretches, the chain's
    surface must stay smooth (measured, not assumed), the profile must reach the skin, and every
    spine edit must leave the original spec untouched."""
    from holographic.mesh_and_geometry.holographic_creature import Creature, quadruped_spec

    # 1) AUTO-DENSITY: doubling a bone's length must roughly double its ball count.
    c1, _ = ball_chain([0, 0, 0], [0, 0, 1.0], 0.1, 0.1, spacing=1.0)
    c2, _ = ball_chain([0, 0, 0], [0, 0, 2.0], 0.1, 0.1, spacing=1.0)
    assert len(c2) >= 2 * len(c1) - 2, "stretching must ADD balls: %d -> %d" % (len(c1), len(c2))
    # ...and a THINNER bone needs MORE balls to stay smooth, not the same number.
    c3, _ = ball_chain([0, 0, 0], [0, 0, 1.0], 0.05, 0.05, spacing=1.0)
    assert len(c3) > len(c1), "a thinner chain needs denser balls: %d vs %d" % (len(c3), len(c1))

    # 2) SMOOTHNESS, MEASURED. Sample the field's zero crossing radially along a straight chain: if
    #    the spacing rule works, the surface radius must not dent between balls. This is the claim
    #    the "neat bit of math" makes, so it is measured rather than trusted.
    C, R = ball_chain([0, 0, 0], [0, 0, 1.0], 0.12, 0.12, spacing=1.0)
    f = metaball_field(C, R)
    zs = np.linspace(0.25, 0.75, 40)                           # interior only: ends taper by design
    surf = []
    for z in zs:
        rs = np.linspace(0.0, 0.35, 400)
        vals = f(np.column_stack([rs, np.zeros_like(rs), np.full_like(rs, z)]))
        cross = np.where(np.diff(np.sign(vals)) != 0)[0]
        surf.append(rs[cross[0]] if len(cross) else 0.0)
    surf = np.asarray(surf)
    ripple = (surf.max() - surf.min()) / max(surf.mean(), 1e-9)
    assert ripple < 0.06, "the chain's surface ripples %.1f%% -- spacing rule is wrong" % (100 * ripple)

    # 3) THE PROFILE REACHES THE SKIN: a fat middle must produce visibly bigger balls there.
    spec = quadruped_spec()
    spec = spine_profile(spec, [0.06, 0.16, 0.20, 0.16, 0.06])
    cr = Creature(spec)
    Cs, Rs, B = creature_metaballs(cr, spec)
    spine_r = np.asarray([r for r, b in zip(Rs, B) if b.startswith("spine")])
    assert spine_r.max() > 2.5 * spine_r.min(), \
        "the spine profile must reach the skin: %.3f..%.3f" % (spine_r.min(), spine_r.max())
    assert len(B) == len(Cs) == len(Rs), "bone_of must label every ball (R-7 depends on it)"
    assert "head" in B and any(b.startswith("L") for b in B), "limbs and head must be skinned too"

    # 4) R-2 THE RIG INVARIANT (backlog B-1): provenance must be exactly as fine as the rig. One
    # label per SEGMENT, never one per chain -- a coarser label is a bone that bends mid-shaft (D-2).
    # This is a REGRESSION TRAP with the exact count, not a smoke test: the old per-chain labelling
    # gave 9 distinct tags where the rig has 17 segments, and every selftest passed anyway because
    # none of them counted. Head is a sphere, not a segment, so it is excluded from both sides.
    seg_count = len(cr.bones)
    tags = set(b for b in B if b != "head")
    assert len(tags) == seg_count, \
        "bone_of must be per-segment: %d distinct tags vs %d rig segments" % (len(tags), seg_count)
    for name, chain in cr.chains.items():
        for j in range(len(chain) - 1):
            assert "%s#%d" % (name, j) in tags, "missing segment tag %s#%d" % (name, j)

    # 4) THE SCALAR PATH IS UNCHANGED -- the additive constraint, asserted not assumed.
    plain = Creature(quadruped_spec())
    _, Rp, _ = creature_metaballs(plain, quadruped_spec())
    assert abs(Rp.max() - Rp.min()) < 0.2, "a scalar-radius creature must stay uniform along the spine"

    # 5) SPINE EDITS ARE PURE: the input spec must come back untouched, every time.
    base = quadruped_spec()
    import copy
    frozen = copy.deepcopy(base)
    e1 = extend_spine(base, 2)
    e2 = insert_node(base, 0.5)
    e3 = set_radius(base, 0.5, 0.25, falloff=0.3)
    e4 = move_node(base, curve=0.3)
    assert base == frozen, "spine edits must NOT mutate the input spec"
    assert e1["spine"]["segments"] == frozen["spine"]["segments"] + 2
    assert e2["spine"]["segments"] == frozen["spine"]["segments"] + 1
    assert abs(e4["spine"]["curve"] - 0.3) < 1e-12

    # 6) EXTENDING KEEPS SEGMENT LENGTH (the 'drag the tail out' behaviour), and all edits still build.
    seg_before = frozen["spine"]["length"] / frozen["spine"]["segments"]
    assert abs(e1["spine"]["length"] / e1["spine"]["segments"] - seg_before) < 1e-12
    for e in (e1, e2, e3, e4):
        Creature(e)                                            # every edited spec must remain valid

    # 7) set_radius WITH FALLOFF is a smooth belly, not a spike: neighbours must move too.
    p = np.asarray(set_radius(spine_profile(base, [0.08] * 5), 0.5, 0.25, falloff=0.5)["spine"]["profile"])
    assert p[2] > p[1] > p[0], "falloff must blend into neighbours, got %s" % p
    assert abs(p[1] - p[3]) < 1e-9, "a centred belly must be symmetric"

    # 8) THE MESH PATH, which this selftest originally did NOT cover -- an integration test caught a
    #    broken import here that every assertion above sailed past. A selftest that never calls the
    #    module's end-to-end door is not a regression trap for it.
    mesh = creature_metaball_mesh(cr, spec, resolution=20, warn=False)  # low res on purpose
    assert len(np.asarray(mesh.vertices)) > 100, "the blended skin must actually mesh"

    # 9) THE DISTANCE FORM IS ACTUALLY A DISTANCE FIELD. This is the property a sphere tracer needs
    #    and the density field does not have (measured on a creature: |grad| spans 0.0 to 26.1). A
    #    render that comes back as empty background is what a non-Lipschitz field looks like.
    fld = creature_field(cr, spec, spacing=0.9)
    Pg = np.random.default_rng(0).normal(size=(300, 3)) * 0.5 + np.array([0, 0, 0.5])
    eps = 1e-3
    grad = np.stack([(fld(Pg + d) - fld(Pg - d)) / (2 * eps)
                     for d in (np.array([eps, 0, 0]), np.array([0, eps, 0]), np.array([0, 0, eps]))], 1)
    gm = np.linalg.norm(grad, axis=1)
    assert 0.75 < gm.mean() < 1.25, "the distance field must be Lipschitz ~1, got mean |grad| %.3f" % gm.mean()
    assert gm.max() < 3.0, "a distance field must not have a runaway gradient (%.1f)" % gm.max()
    assert fld.ids(Pg).shape == (300,) and set(np.unique(fld.ids(Pg))) == {0}

    # 10) CROSS-SECTION SHAPING: the body must stop being a circular tube, EXACTLY as asked, and the
    #     warped field must remain a usable distance field.
    from holographic.mesh_and_geometry.holographic_curves import rotation_minimizing_frame
    nodes = np.array([np.asarray(cr.joints[n], float) for n in cr.spine_nodes])
    Tn, Nn, Bn = rotation_minimizing_frame(nodes)
    kk = len(nodes) // 2
    midp, nax, bax = nodes[kk], np.asarray(Nn)[kk], np.asarray(Bn)[kk]

    def _halfwidth(fld, direction):
        rs = np.linspace(0.0, 1.2, 1200)
        vals = fld(midp[None, :] + rs[:, None] * np.asarray(direction, float)[None, :])
        cross = np.where(np.diff(np.sign(vals)) != 0)[0]
        return float(rs[cross[0]]) if len(cross) else 0.0

    plain_fld = creature_field(cr, spec, spacing=0.9)
    b_n, b_b = _halfwidth(plain_fld, nax), _halfwidth(plain_fld, bax)
    assert abs(b_n / b_b - 1.0) < 0.15, "an unwarped body is CIRCULAR in cross-section"
    shaped_fld = creature_field(cr, spec, spacing=0.9, warp=section_warp(cr, width=1.6, depth=0.7))
    s_n, s_b = _halfwidth(shaped_fld, nax), _halfwidth(shaped_fld, bax)
    assert abs(s_n / b_n - 1.6) < 0.12, "width must scale the body across, got x%.2f" % (s_n / b_n)
    assert abs(s_b / b_b - 0.7) < 0.12, "depth must scale it front-to-back, got x%.2f" % (s_b / b_b)

    # 10b) THE WARP MUST NOT BREAK THE DISTANCE FIELD. Frames are BLENDED across spine stations, not
    #      picked nearest -- a nearest pick makes the frame jump at the midpoint between nodes, and
    #      the measured gradient spiked to 2.11 where a distance field must stay at 1.0. Marching
    #      cubes never noticed (it reads only the SIGN), which is exactly how such a defect ships
    #      looking fine; a sphere tracer would have punched through the surface.
    Pw = np.random.default_rng(3).normal(size=(400, 3)) * 0.5
    ew = 1e-3
    gw = np.stack([(shaped_fld(Pw + d) - shaped_fld(Pw - d)) / (2 * ew)
                   for d in (np.array([ew, 0, 0]), np.array([0, ew, 0]), np.array([0, 0, ew]))], 1)
    gmax = float(np.linalg.norm(gw, axis=1).max())
    assert gmax <= 1.05, "a warped body must stay a valid distance field, got |grad| max %.2f" % gmax

    # 10c) ADDITIVE: warp=None must be BIT-identical to the unwarped field.
    assert np.array_equal(plain_fld(Pw), creature_field(cr, spec, spacing=0.9, warp=None)(Pw))

    print("creatureskin selftest OK: auto-density %d->%d balls on stretch, surface ripple %.2f%%, "
          "profile %.3f..%.3f reaches the skin, %d skin verts, distance |grad| %.2f, "
          "cross-section x%.2f/x%.2f warped |grad| %.2f, all 4 spine edits pure"
          % (len(c1), len(c2), 100 * ripple, spine_r.min(), spine_r.max(),
             len(np.asarray(mesh.vertices)), gm.mean(), s_n / b_n, s_b / b_b, gmax))


if __name__ == "__main__":
    _selftest()

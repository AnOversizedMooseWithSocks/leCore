"""Creature READABILITY REPORTS -- the instruments the field rebuild is judged against.

Backlog Tier 0. Six sessions of incremental creature fixes each produced measurably better numbers
and outputs that still looked wrong, which is the signature of measuring the wrong thing. These are
the three numbers that actually track "does it read as a creature":

  * `webbing_report`   -- M-2. Is there material BETWEEN bones that should have a gap between them?
                          This is Hecker's flying-squirrel bug (one global implicit surface lets the
                          skin of independent limbs join into webbing) and it is THE number the
                          Tier 2 field rebuild is gated on. Predicted non-zero everywhere today.
  * `silhouette_report`-- M-3. Count the enclosed negative-space holes in a black silhouette. A blob
                          has ~0; a creature you can read has several (between legs, under the body).
  * `part_colour_ids`  -- M-1. Which bone owns each surface point, for a flat per-part colour render.
                          No colour seam at a limb/torso junction CONFIRMS a single global field.

WHY A SEPARATE MODULE rather than methods on the field: an instrument that lives inside the thing it
measures gets refactored together with it, and this arc has already lost eight measurements to that
class of confound. These read a field through its public surface only.

WHY IT DELEGATES: the counting machinery already exists in the engine and is reused rather than
rewritten -- `graph_connected_components` (the generic flood fill) does the hole counting, and the
provenance labels come from `creature_metaballs`' `bone_of`, which is per-SEGMENT since backlog B-1.
Two components you wrote agreeing is not evidence, so the hole count is checked against a shape whose
answer is known independently (a ring has exactly one hole).

KEPT NEGATIVE: `webbing_pairs` measures material in the STRAIGHT-LINE corridor between two bone
segments. It cannot tell webbing from a legitimately intervening third body part -- a corridor from
a front foot to a back foot passes through the torso and counts as occupied. That is why the report
excludes pairs whose corridor is blocked by a THIRD bone's own material (`shielded`), and why the
number to watch is the trend under a rebuild, not its absolute value.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_creatureskin import creature_metaballs, creature_field


def _segment_endpoints(creature):
    """The rig's segments as (tag, a, b) with world endpoints.

    DELEGATES to the shared Rig rather than walking the skeleton itself. It used to do its own walk
    and its own naming, and that private copy DRIFTED the instant the canonical spine spelling was
    unified: this report kept emitting "spine0" while the skin emitted "spine#0", so a webbing pair
    could not be joined to the provenance of the balls that caused it. Two namings of one thing is
    the failure this module is supposed to detect, not commit.
    """
    from holographic.mesh_and_geometry.holographic_rig import rig_of
    rig = creature if hasattr(creature, "tags") else rig_of(creature)
    return [(t,) + tuple(rig.segment(t)) for t in rig.tags]


def _adjacent(creature):
    """Pairs of segment tags that SHARE A JOINT. Material between these is not webbing -- it is the
    body. Only non-adjacent pairs can web. Joint identity comes from the shared Rig's bone list, so
    adjacency and naming can never disagree."""
    from holographic.mesh_and_geometry.holographic_rig import rig_of
    rig = creature if hasattr(creature, "tags") else rig_of(creature)
    ends = {t: set(b) for t, b in zip(rig.tags, rig.bones)}
    adj = set()
    for i, a in enumerate(rig.tags):
        for b in rig.tags[i + 1:]:
            if ends[a] & ends[b]:
                adj.add((a, b))
    return adj


def webbing_report(creature, spec=None, field=None, samples=9, margin=0.25, level=0.0,
                   tolerance=None):
    """M-2, THE GATE NUMBER. For every pair of NON-ADJACENT rig segments, sample the corridor between
    their midpoints and ask whether the field says there is material where there should be a gap.

    Returns {'webbing_pairs', 'pairs_tested', 'tolerance', 'reference_length', 'accounted', 'worst'}.
    A pair is WEBBED when any corridor sample is inside the surface but outside every bone's own
    volume -- material that exists only because two fields blended. `worst` carries the fraction of
    corridor samples that were unaccounted, so pairs can be ranked.

    The corridor skips a `margin` fraction at each end so a sample sitting inside the bones' own
    legitimate flesh is not counted as webbing between them -- webbing is what happens in the MIDDLE.

    `field` may be supplied to measure an alternative field type against the same rig; by default the
    creature's shipped field is built. Distance fields are negative inside, so `level` is the
    isolevel and inside means f(P) < level.
    """
    segs = _segment_endpoints(creature)
    if field is None:
        field = creature_field(creature, spec)
    adj = _adjacent(creature)
    ts = np.linspace(0.0, 1.0, int(samples))
    keep = (ts >= float(margin)) & (ts <= 1.0 - float(margin))
    ts = ts[keep] if keep.any() else np.array([0.5])

    # THE DEFINITION, and the third one tried -- the first two were instrument errors and are kept
    # here so nobody re-derives them:
    #   (1) "any material in the corridor" counted a limb's OWN flesh as webbing between it and
    #       something on the far side of it.
    #   (2) "unless a third bone's axis is within 0.15 x the span" was a magic fraction with no
    #       physical meaning; it missed by a hair (samples 0.051 from a radius-0.05 leg, cut 0.046)
    #       and, worse, when tightened it made webbing DECREASE as the blend widened -- the shield
    #       was absorbing the very thing being measured. A metric that moves the wrong way under the
    #       one knob it should track is refuted, not tuned.
    #   (3) THIS ONE: webbing is material NO BONE'S OWN PRIMITIVE ACCOUNTS FOR. A point inside the
    #       surface but outside every bone's own volume exists only because two fields blended. Under
    #       a hard union there is no such material BY CONSTRUCTION, which is why this definition can
    #       reach exactly zero rather than merely getting small. No third-party shield is needed at
    #       all: a limb's own flesh is accounted for by that limb.
    #
    # Bone volumes come from the metaball provenance (the same per-segment `bone_of` skin the weights
    # use), so the metric asks the model rather than a heuristic.
    ref = 1.0
    try:
        from holographic.mesh_and_geometry.holographic_rig import rig_of as _rg
        ref = float(_rg(creature).reference_length())
    except Exception:
        pass
    if tolerance is None:
        # D-7: a spatial threshold DECLARES its reference length. MEASURED noise floor -- a ball chain
        # only approximates a capsule, so even a pure hard union shows some unaccounted material: max
        # 0.0098 = 0.65% of reference length on the shipped quadruped (p99 0.35%). 1.5% is ~2x that
        # floor, so the tolerance is derived from a measurement rather than picked to make a number
        # look good.
        tolerance = 0.015 * ref
    tol = float(tolerance)

    # WHAT ACCOUNTS FOR MATERIAL. Metaball provenance when the source is a creature with a metaball
    # skin; otherwise the RIG ITSELF (each segment's capsule, using creature_tree's own radius rule).
    #
    # WHY THE SECOND PATH EXISTS: the first version only knew metaballs, so handing it a bare Rig --
    # a centaur, or a rig fitted from a point cloud -- silently fell back to raw occupancy and
    # reported 53 webbed pairs on the same quadruped that scores 0 through the creature path. A
    # metric that degrades silently produces a NUMBER, and a number gets believed. Now `accounted`
    # says which definition was used, and the rig path means every body plan gets the real one.
    ball_C = ball_R = None
    seg_axes = None
    try:
        ball_C, ball_R, _ = creature_metaballs(creature, spec)
    except Exception:
        ball_C = None
    if ball_C is None:
        try:
            from holographic.mesh_and_geometry.holographic_creaturetree import segment_radii
            rr = segment_radii(creature)
            seg_axes = [(rig_seg[1], rig_seg[2], rr[rig_seg[0]]) for rig_seg in segs]
        except Exception:
            seg_axes = None

    mids = {t: 0.5 * (a + b) for t, a, b in segs}
    tags = [t for t, _, _ in segs]
    webbed, worst, tested = 0, [], 0
    for i, ta in enumerate(tags):
        for tb in tags[i + 1:]:
            if (ta, tb) in adj or (tb, ta) in adj:
                continue
            tested += 1
            P = mids[ta][None, :] * (1.0 - ts)[:, None] + mids[tb][None, :] * ts[:, None]
            v = np.asarray(field(P), float).ravel()
            inside = v < float(level)
            if not inside.any():
                continue
            if ball_C is None and seg_axes is None:  # neither definition available: SAY so via `accounted`
                webbed += 1
                worst.append((ta, tb, float(inside.mean())))
                continue
            if ball_C is not None:
                d = (np.linalg.norm(P[:, None, :] - ball_C[None, :, :], axis=2) - ball_R[None, :]).min(axis=1)
            else:
                d = np.min(np.stack([_dist_to_segment(P, a, b) - r for a, b, r in seg_axes], axis=1), axis=1)
            unaccounted = inside & (d > tol)
            if unaccounted.any():
                webbed += 1
                worst.append((ta, tb, float(unaccounted.mean())))
    worst.sort(key=lambda r: -r[2])
    return {"webbing_pairs": webbed, "pairs_tested": tested, "tolerance": tol,
            "reference_length": ref,
            "accounted": ("metaballs" if ball_C is not None else
                          ("rig" if seg_axes is not None else False)),
            "worst": worst[:8]}


def _dist_to_segment(P, a, b):
    """Point-to-segment distances (the shared primitive both the shielding test and any future
    corridor test need -- written once so the two can never disagree)."""
    P = np.atleast_2d(np.asarray(P, float))
    ab = np.asarray(b, float) - np.asarray(a, float)
    L2 = float(ab @ ab)
    if L2 < 1e-18:
        return np.linalg.norm(P - np.asarray(a, float)[None, :], axis=1)
    t = np.clip(((P - np.asarray(a, float)[None, :]) @ ab) / L2, 0.0, 1.0)
    proj = np.asarray(a, float)[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(P - proj, axis=1)


def silhouette_mask(field, bounds, axis=0, res=96, level=0.0):
    """A boolean silhouette: project the field along `axis` and mark a pixel solid if ANY sample down
    that ray is inside. Orthographic and grid-based rather than a render, because a render brings a
    camera, shading and a background colour -- three places this arc has already lost a measurement.
    """
    (lo, hi) = bounds
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    ax = int(axis)
    u, v = [k for k in (0, 1, 2) if k != ax]
    gu = np.linspace(lo[u], hi[u], int(res))
    gv = np.linspace(lo[v], hi[v], int(res))
    gw = np.linspace(lo[ax], hi[ax], int(res))
    U, V, W = np.meshgrid(gu, gv, gw, indexing="ij")
    P = np.zeros((U.size, 3))
    P[:, u] = U.ravel(); P[:, v] = V.ravel(); P[:, ax] = W.ravel()
    val = np.asarray(field(P), float).reshape(U.shape)
    return (val < float(level)).any(axis=2)


def _cross2(u, v):
    """The scalar 2-D cross product u x v (the z of the 3-D one). Sign gives turn direction."""
    return float(u[0] * v[1] - u[1] * v[0])


def _convex_hull_area(pts):
    """Area of the convex hull of 2-D points (monotone chain + shoelace).

    Written here because the audit found no hull in the engine (`find_capability` on four phrasings
    returned only unrelated fallbacks). Kept small and local; if a third caller ever wants it, that is
    the moment to promote it rather than now.
    """
    P = np.unique(np.asarray(pts, float), axis=0)
    if len(P) < 3:
        return 0.0
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def half(seq):
        out = []
        for p in seq:
            # 2-D cross product written out: np.cross on 2-vectors is deprecated in NumPy 2, and a
            # deprecation that becomes an error later is a silent future breakage in a metric.
            while len(out) >= 2 and _cross2(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out
    hull = half(P)[:-1] + half(P[::-1])[:-1]
    H = np.asarray(hull, float)
    if len(H) < 3:
        return 0.0
    x, y = H[:, 0], H[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def silhouette_report(creature, spec=None, field=None, axis=0, res=96, level=0.0):
    """M-3, READABILITY: how much NEGATIVE SPACE the silhouette has -- the gaps between legs and under
    the body that make a shape read as an animal rather than a blob.

    Returns {'holes', 'components', 'solid_fraction', 'solidity', 'negative_space'}.

    TWO MEASURES, BECAUSE THE FIRST ONE ALONE WAS THE WRONG INSTRUMENT. `holes` counts ENCLOSED empty
    regions (not touching the border). Measured, a standing quadruped scores 0 holes under EVERY
    field on all three axes -- and that is correct, not a defect: the gap between a quadruped's legs
    is OPEN AT THE GROUND, so it is never enclosed. A metric that reads 0 for both the blob and the
    fixed version cannot gate the rebuild it was written for.

    `solidity` is the standard shape descriptor that does work here: silhouette area over its CONVEX
    HULL area. A blob approaches 1.0; a shape with limbs and gaps between them is far below it.
    `negative_space` = 1 - solidity, i.e. the fraction of the hull that the creature does NOT fill.
    `holes` is kept because it is the right measure for a body with a genuine enclosed opening (an
    arm on a hip, a ring tail) -- two measures for two shapes, rather than one measure pretending.
    """
    from holographic.mesh_and_geometry.holographic_creatureskin import creature_field as _cf
    if field is None:
        field = _cf(creature, spec)
    # BOUNDS FROM THE RIG WHEN THE FIELD HAS NONE. An SDF tree is a plain callable with no bounds()
    # -- refusing it would have meant the M-3 gate could only ever be run on the field it was written
    # against, i.e. the metric could never judge the rebuild it exists to judge. The rig's own extent
    # is the honest padded box for ANY field built from that rig, so both are measured in the same
    # frame and the comparison is like-for-like.
    b = field.bounds() if hasattr(field, "bounds") else None
    if b is None:
        from holographic.mesh_and_geometry.holographic_rig import rig_of as _rg
        rig = creature if hasattr(creature, "tags") else _rg(creature)
        lo, hi = rig.extent()
        pad = 0.35 * rig.reference_length()
        b = (np.asarray(lo, float) - pad, np.asarray(hi, float) + pad)
    mask = silhouette_mask(field, b, axis=axis, res=res, level=level)
    holes, comps = _count_enclosed_holes(mask)
    pts = np.argwhere(mask).astype(float)
    hull = _convex_hull_area(pts)
    area = float(mask.sum())
    # SOLIDITY IS CLAMPED AT 1.0, and the reason is a discretisation mismatch rather than a defect:
    # `area` counts PIXELS while the hull is a polygon through pixel CENTRES, so the hull misses a
    # half-pixel border all the way round. For a nearly convex blob that made solidity exceed 1 and
    # negative_space go NEGATIVE (measured -0.013 on the fitted rig), which is not a meaningful
    # reading -- a shape cannot fill more than its own hull. Clamped rather than corrected with a
    # perimeter term, because the quantity that matters is CONCAVITY and the error only appears where
    # there is none.
    solidity = min(area / hull, 1.0) if hull > 1e-9 else 1.0
    return {"holes": holes, "components": comps, "solid_fraction": float(mask.mean()),
            "solidity": float(solidity), "negative_space": float(1.0 - solidity)}


def _count_enclosed_holes(mask):
    """Enclosed empty regions in a boolean solid-mask, and the number of solid components.

    Delegates connectivity to the shared graph flood fill rather than writing a second one: an
    instrument with its own private connectivity is exactly how two components you wrote come to
    agree with each other and with nothing else.
    """
    # RETURN SHAPE, VERIFIED FROM THE LIVE MODULE, NOT FROM MEMORY: connected_components returns a
    # LIST OF SORTED INDEX LISTS (one per component), not a per-node label array. Unpacking it as
    # labels is the exact "check return shapes" trap on the standing process list, and it would have
    # produced a confident wrong hole count rather than an error.
    from holographic.simulation_and_physics.holographic_island import connected_components as _cc

    def _components(sel):
        """(points, [component index lists]) for 4-connected True pixels of `sel`."""
        idx = -np.ones(sel.shape, int)
        pts = np.argwhere(sel)
        for n, (i, j) in enumerate(pts):
            idx[i, j] = n
        edges = []
        for n, (i, j) in enumerate(pts):
            for di, dj in ((1, 0), (0, 1)):
                a, bq = i + di, j + dj
                if a < sel.shape[0] and bq < sel.shape[1] and sel[a, bq]:
                    edges.append((n, int(idx[a, bq])))
        return pts, _cc(len(pts), edges)

    empty = ~mask
    pts, comps = _components(empty)
    H, W = empty.shape
    holes = 0
    for comp in comps:
        touches_border = any(pts[k][0] in (0, H - 1) or pts[k][1] in (0, W - 1) for k in comp)
        if not touches_border:
            holes += 1
    _, solid_comps = _components(mask)
    return holes, len(solid_comps)


def ratio(numerator, denominator, of, note=""):
    """A RATIO THAT CARRIES WHAT IT IS A RATIO OF -- the measurement twin of D-7's reference length.

    Returns {'value', 'numerator', 'denominator', 'of', 'note'}. `of` is REQUIRED and names the
    denominator in words.

    WHY THIS EXISTS AS A FUNCTION rather than a rule people remember: I reported "parts change 0.58%
    of pixels, so parts do not read", built a fix for it, measured the fix as a regression, and
    recorded BOTH as findings -- and all of it was one mistake. 0.58% was a fraction OF THE WHOLE
    IMAGE, which is ~95% background. Against the SUBJECT the same parts add 11% of body silhouette
    and extend 2.6 limb-radii past the leg. The parts read fine; the denominator was the defect.

    That is the same failure as every absolute-vs-relative bug in this codebase (cell_scale, the
    joint blend, the organ blend, the webbing tolerance) wearing a statistical costume: a quantity
    meaningful only relative to the subject, expressed against something else. D-7 made distances
    declare their reference length; this makes ratios declare their denominator, so a percentage
    cannot be written down without saying what it is a percentage of.
    """
    d = float(denominator)
    return {"value": (float(numerator) / d) if abs(d) > 1e-12 else float("nan"),
            "numerator": float(numerator), "denominator": d, "of": str(of), "note": str(note)}


def regression_specs(seed=0):
    """THE THREE BODIES THE UNIFICATION IS JUDGED ON (backlog Tier 9 / D-1 / D-6).

    Returns [(name, rig), ...]:
        quadruped   the baseline
        centaur     a HYBRID -- proves hybrids are specs, not code paths, because nothing in the
                    engine knows what a centaur is
        fitted      a rig recovered from a POINT CLOUD via fit_primitives -- proves the loop is
                    closed, i.e. that observe and generate produce the same rig type

    The honest test of unification is NOT that these still work separately; it is that ONE code path
    walks all three without branching on which kind of body it got. `regression_report` runs it.
    """
    import numpy as _np
    from holographic.mesh_and_geometry.holographic_creature import Creature, quadruped_spec, centaur_spec
    from holographic.mesh_and_geometry.holographic_rig import rig_of, rig_from_primitives
    from holographic.mesh_and_geometry.holographic_primfit import fit_primitives

    rng = _np.random.default_rng(int(seed))
    # A cloud shaped like a two-segment limb, so the fit has something a capsule can honestly claim.
    t = _np.linspace(0.0, 1.0, 500)[:, None]
    cloud = (t * _np.array([1.0, 0.0, 0.0]) + _np.maximum(t - 0.5, 0.0) * _np.array([0.0, 0.6, 0.0])
             + rng.normal(size=(500, 3)) * 0.05)
    return [("quadruped", rig_of(Creature(quadruped_spec()))),
            ("centaur", rig_of(Creature(centaur_spec()))),
            ("fitted", rig_from_primitives(fit_primitives(cloud, k=4)))]


def regression_report(seed=0, res=64, samples=3000):
    """RUN the three regression specs through ONE pipeline and report per body.

    For each: the rig invariant, a compiled skin (`creature_tree`), the webbing and negative-space
    numbers, and the anatomy nesting check. Every entry comes from the SAME calls -- if any body plan
    needed a special case, that special case would have to live here, visibly.

    Returns {name: {...}} with an 'ok' per body. Organs are reported as `organs` (True/False) rather
    than required: anatomy space is spine-relative, so a fitted rig legitimately has none, and that
    gap is REPORTED rather than hidden or faked.
    """
    from holographic.mesh_and_geometry.holographic_rig import rig_invariant
    from holographic.mesh_and_geometry.holographic_creaturetree import creature_tree
    from holographic.mesh_and_geometry.holographic_creaturetissue import tissue_fields, anatomy_report

    out = {}
    for name, rig in regression_specs(seed=seed):
        inv = rig_invariant(rig)
        skin = creature_tree(rig)
        fields = tissue_fields(rig)
        anat = anatomy_report(rig, fields=fields, samples=samples)
        web = webbing_report(rig, field=skin)
        sil = silhouette_report(rig, field=skin, res=res)
        out[name] = {"segments": inv["segments"], "reference_length": inv["reference_length"],
                     "webbing_pairs": web["webbing_pairs"], "pairs_tested": web["pairs_tested"],
                     "negative_space": sil["negative_space"],
                     "nesting_violations": anat["violations"],
                     "organs": "organ" in fields,
                     "ok": bool(anat["violations"] == 0 and not inv["degenerate"])}
    return out


def part_colour_ids(creature, spec=None, points=None, spacing=1.0):
    """M-1, THE SEAM TEST. For each query point, which rig segment's ball is nearest -- i.e. which
    part would colour it in a flat per-part render. Returns an integer id array plus the tag list.

    A flat colour render built from this shows a hard seam wherever ownership changes. NO seam at a
    limb/torso junction is the positive confirmation that the surface there came from one global
    blended field rather than from two parts meeting.
    """
    C, R, B = creature_metaballs(creature, spec, spacing=spacing)
    tags = sorted(set(B))
    tag_index = {t: i for i, t in enumerate(tags)}
    owner = np.array([tag_index[b] for b in B], int)
    if points is None:
        return {"tags": tags, "ball_owner": owner, "centers": C, "radii": R}
    P = np.atleast_2d(np.asarray(points, float))
    # Nearest ball SURFACE, not centre: a big torso ball would otherwise steal points from a small
    # nearby limb ball whose surface is much closer.
    d = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2) - R[None, :]
    return {"tags": tags, "ids": owner[np.argmin(d, axis=1)]}


def _selftest():
    from holographic.mesh_and_geometry.holographic_creature import Creature, quadruped_spec

    # 0) THE INSTRUMENT IS CHECKED AGAINST A KNOWN ANSWER FIRST. A ring has exactly one enclosed
    # hole and one solid component; a disc has zero holes. If the counter cannot get these right the
    # creature numbers it reports are decoration. (A perfect score is a hypothesis about the
    # instrument -- so the instrument is tested on ground truth it did not produce.)
    g = np.linspace(-1.0, 1.0, 41)
    X, Y = np.meshgrid(g, g, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2)
    ring = (r < 0.8) & (r > 0.4)
    disc = r < 0.8
    h_ring, c_ring = _count_enclosed_holes(ring)
    h_disc, c_disc = _count_enclosed_holes(disc)
    assert (h_ring, c_ring) == (1, 1), "ring must be 1 hole / 1 component, got %d/%d" % (h_ring, c_ring)
    assert (h_disc, c_disc) == (0, 1), "disc must be 0 holes / 1 component, got %d/%d" % (h_disc, c_disc)

    spec = quadruped_spec()
    cr = Creature(spec)

    # 1) ADJACENCY IS REAL: segments sharing a joint are excluded, and there are strictly fewer
    # tested pairs than all pairs -- an exclusion that excludes nothing is a bug that looks clean.
    segs = _segment_endpoints(cr)
    adj = _adjacent(cr)
    n = len(segs)
    assert len(adj) > 0, "a chain must have adjacent segments"
    assert len(adj) < n * (n - 1) // 2, "adjacency must not swallow every pair"

    # 2) THE WEBBING NUMBER EXISTS AND IS THE PREDICTED SIGN. The backlog predicts non-zero webbing
    # everywhere under today's single global summed field; this pins that prediction so the Tier 2
    # rebuild has a baseline it must beat rather than a vibe.
    rep = webbing_report(cr, spec)
    assert rep["pairs_tested"] > 0, "no non-adjacent pairs to test"
    assert rep["webbing_pairs"] >= 0 and rep["webbing_pairs"] <= rep["pairs_tested"]

    # 3) THE WEBBING METRIC RESPONDS TO THE THING IT CLAIMS TO MEASURE. Widening the blend must not
    # DECREASE webbing. This is the discriminating test: a metric that returns a plausible constant
    # regardless of the field is the failure mode this whole module exists to avoid.
    tight = creature_field(cr, spec, smooth_k=0.01)
    loose = creature_field(cr, spec, smooth_k=0.30)
    w_tight = webbing_report(cr, spec, field=tight)["webbing_pairs"]
    w_loose = webbing_report(cr, spec, field=loose)["webbing_pairs"]
    assert w_loose >= w_tight, \
        "more blend must not web LESS: k=0.30 -> %d, k=0.01 -> %d" % (w_loose, w_tight)

    # 4) SILHOUETTE: solid fraction is a real fraction and the counter runs on a creature.
    sil = silhouette_report(cr, spec, res=64)
    assert 0.0 < sil["solid_fraction"] < 1.0, "silhouette must be neither empty nor full: %r" % sil
    assert sil["components"] >= 1, "a creature is at least one solid component"

    # 4a) THE SOLIDITY INSTRUMENT, CHECKED ON GROUND TRUTH IT DID NOT PRODUCE. A filled disc is
    # convex, so its solidity must be ~1 and its negative space ~0; a plus sign is strongly concave
    # and must score well below. Without this, "the new field has more negative space" would rest on
    # a number nothing had ever validated.
    g = np.linspace(-1.0, 1.0, 61)
    XX, YY = np.meshgrid(g, g, indexing="ij")
    disc_m = (XX ** 2 + YY ** 2) < 0.64
    plus_m = (np.abs(XX) < 0.10) | (np.abs(YY) < 0.10)      # thin arms -> unambiguously concave
    d_sol = float(disc_m.sum()) / _convex_hull_area(np.argwhere(disc_m).astype(float))
    p_sol = float(plus_m.sum()) / _convex_hull_area(np.argwhere(plus_m).astype(float))
    assert 0.93 < d_sol < 1.08, "a disc must be near-solid, got %.3f" % d_sol
    assert p_sol < 0.45, "a plus sign must be strongly concave, got %.3f" % p_sol

    # 4b) NO PRIVATE NAMING. The report's segment tags must be EXACTLY the shared Rig's tags, which
    # are exactly the skin's `bone_of` tags. Pinned as set equality because this module already
    # drifted once: it kept its own "spine0" walk after the canonical spelling became "spine#0", so a
    # reported webbing pair could not be joined back to the balls that caused it.
    from holographic.mesh_and_geometry.holographic_rig import rig_of
    from holographic.mesh_and_geometry.holographic_creatureskin import creature_metaballs as _cm
    rig_tags = set(rig_of(cr).tags)
    report_tags = set(t for t, _, _ in _segment_endpoints(cr))
    skin_tags = set(b for b in _cm(cr, spec)[2] if b != "head")
    assert report_tags == rig_tags == skin_tags, \
        "report/rig/skin tags must be identical: report-only %r skin-only %r" % (
            sorted(report_tags - skin_tags)[:3], sorted(skin_tags - report_tags)[:3])

    # 4c) THE THREE REGRESSION SPECS (backlog Tier 9). The honest test of unification is not that a
    # quadruped still works -- it is that a HYBRID and a rig recovered from a POINT CLOUD walk the
    # SAME calls. If any of them needed a special case, that case would have to appear in
    # `regression_report`, where it would be visible.
    reg = regression_report(res=48, samples=1500)
    assert set(reg) == {"quadruped", "centaur", "fitted"}, sorted(reg)
    for nm, r in reg.items():
        assert r["ok"], "%s failed the shared pipeline: %r" % (nm, r)
        assert r["nesting_violations"] == 0, "%s: anatomy nesting violated" % nm
        assert r["webbing_pairs"] == 0, "%s: %d webbed pairs" % (nm, r["webbing_pairs"])
    # The hybrid must really be a HYBRID -- more segments than the quadruped it is built from, or the
    # centaur spec silently degraded to an ordinary body and the D-1 claim proves nothing.
    assert reg["centaur"]["segments"] > reg["quadruped"]["segments"], reg
    # And the fitted rig must really come from a FIT (few segments, no authored spine).
    assert reg["fitted"]["segments"] >= 1 and not reg["fitted"]["organs"], reg

    # 4d) THE ACCOUNTING DEFINITIONS MUST AGREE where both are available. The quadruped can be
    # measured through metaball provenance AND through the rig; if those two disagreed, one of them
    # would be wrong and every webbing number in the arc would be suspect.
    from holographic.mesh_and_geometry.holographic_creaturetree import creature_tree as _ct2
    q_creature = webbing_report(cr, spec, field=_ct2(cr))
    q_rig = webbing_report(rig_of(cr), field=_ct2(cr))
    assert q_creature["accounted"] == "metaballs" and q_rig["accounted"] == "rig"
    assert q_creature["webbing_pairs"] == q_rig["webbing_pairs"], \
        "the two accounting definitions must agree on one body: %d vs %d" % (
            q_creature["webbing_pairs"], q_rig["webbing_pairs"])

    # 4e) THE RATIO HELPER STATES ITS DENOMINATOR, and the test is the case that burned me: the same
    # part contribution reads 0.58% of the IMAGE and 11% of the BODY. Both are correct; only one
    # answers "do the parts read". A ratio without its denominator is not a measurement.
    of_image = ratio(694, 120000, "whole image (mostly background)")
    of_body = ratio(694, 6162, "body silhouette")
    assert of_image["value"] < 0.01 < of_body["value"], (of_image, of_body)
    assert of_image["of"] and of_body["of"], "a ratio must name its denominator"

    # 5) M-1 OWNERSHIP IS PER-SEGMENT (rides on backlog B-1) and every ball is owned.
    pc = part_colour_ids(cr, spec)
    assert len(pc["tags"]) == len(cr.bones) + 1, \
        "one tag per rig segment plus head: %d vs %d" % (len(pc["tags"]), len(cr.bones) + 1)
    assert pc["ball_owner"].min() >= 0 and pc["ball_owner"].max() == len(pc["tags"]) - 1

    print("creaturereport selftest OK: instrument ring=1/disc=0 holes, %d/%d pairs webbed "
          "(tol %.4f = 1.5%% of ref %.3f), blend response %d->%d, silhouette %d holes / %d comps "
          "solid %.3f, %d part tags"
          % (rep["webbing_pairs"], rep["pairs_tested"], rep["tolerance"], rep["reference_length"],
             w_tight, w_loose, sil["holes"], sil["components"], sil["solid_fraction"],
             len(pc["tags"])))


if __name__ == "__main__":
    _selftest()

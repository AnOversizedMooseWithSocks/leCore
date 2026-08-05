"""CONVOLUTION SURFACES over contiguous skeletons -- the right tool for hands, feet and digits.

WHY THIS EXISTS, and it is a correction to how the creature parts were being built. Feet and hands
were assembled by SMOOTH-UNIONING capsules, which bulges at every joint: a visible collar where a
limb meets the body, a melted lump where an ankle meets a sole, and toes that lose their separation.
The literature has a direct answer.

    Bloomenthal & Shoemake 1991, "Convolution Surfaces" (SIGGRAPH '91)
    Bloomenthal, "Hand Crafted" -- a hand as 85 convolution primitives

Bloomenthal models a human hand as CONTIGUOUS skeletal primitives -- a palm of 15 triangles, fingers
of 48, plus line segments for tendons and veins -- and states the property that matters here:

    "Each convolution surface is evaluated independently, and, because of the superposition property
     of convolution, the sum of individual convolution surfaces does not produce unwanted bulging...
     When the primitives are contiguous, the resulting implicit surface contains no bulge."

THAT IS THE WHOLE POINT. A smooth-union of two capsules adds material at the joint (the bulge is the
blend). A SUM of convolutions over a contiguous skeleton does not -- the field is already continuous
along the skeleton, so a joint is just a place where the skeleton bends. No blend parameter, nothing
to tune, and no collar.

ANISOTROPY, because a sole is not round:

    Fuentes Suarez, Hubert & Zanni 2019, "Anisotropic convolution surfaces" (Computers & Graphics 82)

They note convolution surfaces "have been limited to close-to-circular normal sections" and extend
them to ELLIPSOIDAL normal sections given by a rotation and three radii per extremity, which "creates
smooth shapes that previously required tweaking the skeleton or supplementing it with 2D pieces". A
foot is exactly that case: wide, long and flat. Here the anisotropy is applied as a per-segment
diagonal warp of the query point, which is the cheap form of the same idea and keeps the field a
plain NumPy expression.

KEPT NEGATIVE, NAMED BY BLOOMENTHAL HIMSELF and inherited by this module: convolution SUMS
everything, so primitives that merely come NEAR each other still blend -- "when two fingers approach
each other, they should not blend". Contiguity fixes joints, not proximity. Separate digits therefore
have to be kept apart by GROUPING (evaluating them as separate fields and taking a hard union), which
is the same metaball-groups rule the creature body already uses.
"""

import numpy as np


def _seg_convolution(P, a, b, r, aniso=None, samples=24, kernel=2.2):
    """Convolution of one skeletal SEGMENT with a Gaussian-like kernel, evaluated at points `P`.

    Numerically integrated along the segment rather than solved in closed form: the closed forms in
    the literature are per-kernel and per-primitive, and a quadrature keeps this module honest about
    what it computes while staying pure NumPy. `samples` is the quadrature resolution.

    `aniso` is a (3,) scale on the segment's own frame -- (along, side, up) -- which is the cheap form
    of Fuentes Suarez et al.'s ellipsoidal normal sections: a sole is wide and flat, so its section is
    an ellipse, not a circle.
    """
    P = np.atleast_2d(np.asarray(P, float))
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-12:
        pts = a[None, :]
        w = np.array([1.0])
    else:
        t = (np.arange(samples) + 0.5) / samples
        pts = a[None, :] + t[:, None] * d[None, :]
        w = np.full(samples, L / samples)

    Q = P[:, None, :] - pts[None, :, :]
    if aniso is not None and L > 1e-12:
        # Build the segment's own frame and squash the offset in it -- an ellipsoidal section.
        e1 = d / L
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(e1 @ up)) > 0.95:
            up = np.array([0.0, 1.0, 0.0])
        e2 = np.cross(up, e1)
        e2 /= np.linalg.norm(e2) + 1e-12
        e3 = np.cross(e1, e2)
        M = np.stack([e1, e2, e3])                    # rows: along, side, up
        s = np.asarray(aniso, float)
        Q = (Q @ M.T) / s[None, None, :]
    d2 = np.einsum("ijk,ijk->ij", Q, Q)
    return (np.exp(-float(kernel) * d2 / (float(r) ** 2)) * w[None, :]).sum(axis=1)


def convolution_field(segments, iso=0.35, samples=24, kernel=2.2):
    """A field from a CONTIGUOUS skeleton: sum the convolution of every segment, then subtract `iso`.

    `segments` is a list of (a, b, radius) or (a, b, radius, aniso). Returns a callable f(P) that is
    NEGATIVE inside, so it drops straight into the same pipeline as every other field here.

    THE CONTIGUITY IS THE CONTRACT, not a suggestion: joints between consecutive segments come out
    bulge-free BECAUSE the sum is continuous along the skeleton. Feed it disjoint pieces that happen
    to be near each other and it will blend them, which is the kept negative above.
    """
    segs = [(np.asarray(s[0], float), np.asarray(s[1], float), float(s[2]),
             (np.asarray(s[3], float) if len(s) > 3 else None)) for s in segments]
    if not segs:
        raise ValueError("a convolution field needs at least one segment")
    scale = float(np.mean([s[2] for s in segs]))

    def f(P, _s=tuple(segs), _iso=float(iso), _sc=scale):
        Q = np.atleast_2d(np.asarray(P, float))
        acc = np.zeros(len(Q))
        for a, b, r, an in _s:
            acc += _seg_convolution(Q, a, b, r, aniso=an, samples=samples, kernel=kernel)
        # Normalised so `iso` means the same thing regardless of how many segments contributed, and
        # negated so the result reads as a distance-like field (negative inside).
        return (_iso - acc / max(_sc, 1e-9)) * _sc

    return f


def digit_skeleton(base, direction, length, joints=3, curl=0.35, radius=0.02, taper=0.55):
    """A CONTIGUOUS chain of segments for one finger or toe, bending by `curl` at each joint.

    Returns [(a, b, r), ...] sharing endpoints, so the convolution of the chain has no bulge at the
    knuckles -- the knuckle is where the skeleton BENDS, not where two blobs are glued.
    """
    base = np.asarray(base, float)
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(d, up)
    if np.linalg.norm(side) < 1e-9:
        side = np.array([1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    bend = np.cross(side, d)
    bend /= np.linalg.norm(bend) + 1e-12

    out = []
    p = base
    seg = float(length) / max(int(joints), 1)
    for i in range(int(joints)):
        ang = float(curl) * (i + 1) / max(int(joints), 1)
        step = (d * np.cos(ang) - bend * np.sin(ang)) * seg
        q = p + step
        r = float(radius) * (1.0 - (1.0 - float(taper)) * (i / max(int(joints) - 1, 1)))
        out.append((p, q, r))
        p = q
    return out


def foot_skeleton(size=1.0, digits=3, spread=0.7, toe_len=0.9, sole_flat=0.45):
    """A foot as Bloomenthal builds one: a CONTIGUOUS sole chain with digit chains growing off it.

    Returns (groups, sole_aniso) where `groups` is a list of segment-lists -- the sole-plus-ankle as
    one contiguous group, and EACH TOE as its own group. The grouping is deliberate: contiguity kills
    the joint bulge, but convolution still sums, so toes evaluated together would web. Rendering the
    groups with a hard union between them keeps the toes separate, which is the exact problem
    Bloomenthal flags as unsolved in his own hand.

    The sole carries an ANISOTROPIC section (wide and flat), which is what Fuentes Suarez et al.'s
    ellipsoidal normal sections are for -- otherwise a sole has to be faked with extra primitives.
    """
    s = float(size)
    r = 0.055 * s
    heel = np.array([0.0, -0.95 * s * 0.09, 0.95 * r])
    arch = np.array([0.0, -0.15 * s * 0.09, 0.95 * r])
    ball = np.array([0.0, 0.80 * s * 0.09, 0.95 * r])
    ankle = np.array([0.0, -0.55 * s * 0.09, 2.35 * r])
    aniso = np.array([1.0, 1.5, float(sole_flat)])          # along, wide, FLAT

    sole = [(heel, arch, 0.85 * r, aniso),
            (arch, ball, 0.95 * r, aniso),
            (ball, heel, 0.0 * r + 1e-9, aniso)][:2]
    sole.append((arch, ankle, 0.70 * r, np.array([1.0, 1.0, 1.0])))   # ankle rises, round section
    groups = [sole]
    for a in np.linspace(-1.0, 1.0, int(np.clip(digits, 2, 6))) * float(spread):
        d = np.array([np.sin(a), np.cos(a), -0.05])
        groups.append(digit_skeleton(ball + np.array([np.sin(a) * 0.55 * r, 0.25 * r, -0.15 * r]),
                                     d, float(toe_len) * s * 0.10, joints=3, curl=0.30,
                                     radius=0.30 * r))
    return groups, aniso


def grouped_field(groups, iso=0.35, samples=20, kernel=2.2):
    """Each group becomes its own convolution field; the groups HARD-UNION.

    Contiguity removes the bulge WITHIN a group; the hard union between groups removes the blending
    BETWEEN groups. That is Bloomenthal's open problem ("when two fingers approach each other, they
    should not blend") answered with the same metaball-groups rule the creature body already uses.
    """
    fields = [convolution_field(g, iso=iso, samples=samples, kernel=kernel) for g in groups]

    def f(P, _f=tuple(fields)):
        Q = np.atleast_2d(np.asarray(P, float))
        out = None
        for g in _f:
            v = np.asarray(g(Q), float).ravel()
            out = v if out is None else np.minimum(out, v)
        return out
    return f


def skeleton_mesh(groups, res=56, pad=0.35, iso=0.35, samples=16, kernel=2.2):
    """March a grouped convolution field into a MESH -- the bridge from field to the parts pipeline.

    The parts pipeline consumes meshes, and a convolution field is a DENSITY, not a distance field:
    it does not satisfy the Lipschitz bound the sphere tracer and the scaffold projector assume. So
    it is polygonised here, where only the SIGN matters, and the mesh is what travels onward.

    `pad` is a fraction of the skeleton's extent, so the sampling box holds the surface at any size
    (D-7: the box is stated relative to the thing it must contain, not as an absolute).
    """
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec

    pts = np.array([p for g in groups for s in g for p in (s[0], s[1])], float)
    rad = float(max(s[2] for g in groups for s in g))
    lo = pts.min(axis=0) - rad * (1.0 + float(pad)) * 2.0
    hi = pts.max(axis=0) + rad * (1.0 + float(pad)) * 2.0
    f = grouped_field(groups, iso=iso, samples=samples, kernel=kernel)
    values, axes = sample_field(f, (tuple(lo), tuple(hi)), int(res))
    return marching_tetrahedra_vec(values, axes, level=0.0)


def foot_mesh(size=1.0, digits=3, spread=0.7, toe_len=0.9, sole_flat=0.45, res=56):
    """A FOOT MESH built the way the convolution-surface literature builds one.

    Replaces "a pad mesh plus N toe meshes merged into one vertex array": those were separate shells
    that intersected each other and were never joined, which is what a close-up showed. Here the sole
    and ankle are ONE contiguous convolution (no joint bulge) and each toe is its own group (so they
    do not web), and the whole thing is polygonised as a single surface.
    """
    groups, _ = foot_skeleton(size=size, digits=digits, spread=spread, toe_len=toe_len,
                              sole_flat=sole_flat)
    return skeleton_mesh(groups, res=int(res))


def creature_groups(source, radii=None, mount_flare=0.55):
    """The whole creature as CONVOLUTION GROUPS: the spine as one contiguous chain, each limb chain as
    another. Returns groups for `grouped_field`.

    WHY THE BODY, NOT JUST THE FEET. A hip is a joint like a knuckle, and it had the same defect for
    the same reason: `smooth_union` between a limb and the torso ADDS material at the mount, which is
    the "collar" or skirt a viewer sees where a leg meets the body. Contiguous convolution does not.
    MEASURED at a hip, distance from the joint to the surface along the bisector:

        smooth_union tree   0.1375
        convolution         0.1052      23% less, and no blend parameter at all

    Each limb is its own GROUP, so limbs still hard-union with each other and cannot web -- the same
    metaball-groups rule, now the only blending control the body needs.
    """
    from holographic.mesh_and_geometry.holographic_creaturetree import segment_radii

    rig = source if hasattr(source, "tags") else rig_of_lazy(source)
    rr = dict(radii) if radii else segment_radii(rig, mount_flare=float(mount_flare))
    groups = []
    spine = sorted([t for t in rig.tags if t.split("#")[0] == "spine"],
                   key=lambda t: int(t.split("#")[1]))
    if spine:
        chain = [(rig.segment(t)[0], rig.segment(t)[1], rr[t]) for t in spine]
        # THE HEAD IS PART OF THE SPINE CHAIN, not a sphere stuck on the end.
        #
        # It was simply MISSING: this function built spine and limb chains and nothing else, so a
        # convolution body had NO HEAD -- the eyes and mouth floated in front of a headless neck,
        # which is exactly what the render showed. Appending it to the spine group rather than
        # unioning a sphere is also what makes the neck read: a contiguous chain that swells at the
        # end has no seam, where a smooth-unioned sphere adds a bulge at the join (the same defect
        # this module exists to remove, and it would have been the one place still showing it).
        src = getattr(rig, "source", None)
        head = getattr(src, "head", None) if src is not None else None
        if head and head.get("node") in rig.joints:
            hp = np.asarray(rig.joints[head["node"]], float)
            prev = np.asarray(chain[-1][0], float) if chain else hp
            d = hp - prev
            n = float(np.linalg.norm(d))
            d = d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
            hr = float(head.get("radius", 0.12))
            # A short segment carrying the head's radius, so the skull is a THICKENING of the body
            # axis. Set slightly back from the joint so the head sits ON the neck, not beyond it.
            chain.append((hp - d * hr * 0.35, hp + d * hr * 0.30, hr * 0.92))
        groups.append(chain)
    for cname in sorted(rig.chains):
        if cname == "spine":
            continue
        segs = sorted([t for t in rig.tags if t.split("#")[0] == cname],
                      key=lambda t: int(t.split("#")[1]))
        if segs:
            groups.append([(rig.segment(t)[0], rig.segment(t)[1], rr[t]) for t in segs])
    return groups


def rig_of_lazy(source):
    """Deferred `rig_of`, so this module does not import the rig at module load and create a cycle."""
    from holographic.mesh_and_geometry.holographic_rig import rig_of
    return rig_of(source)


def creature_field(source, radii=None, mount_flare=0.55, iso=0.32, samples=18):
    """The creature's skin as a grouped convolution surface -- bulge-free joints, grouped limbs."""
    return grouped_field(creature_groups(source, radii=radii, mount_flare=mount_flare),
                         iso=float(iso), samples=int(samples))


def _selftest():
    # 1) THE BULGE TEST, which is the entire reason this module exists. Two segments meeting at a
    # right angle: a smooth-union of capsules ADDS material at the corner, a contiguous convolution
    # does not. Measured as the surface's distance from the joint along the corner bisector -- a
    # bulge pushes it OUT past the segment radius.
    a = np.array([-0.30, 0.0, 0.0])
    j = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.30, 0.0])
    r = 0.06
    conv = convolution_field([(a, j, r), (j, b, r)])
    from holographic.mesh_and_geometry.holographic_creaturetree import bone_capsule
    from holographic.mesh_and_geometry.holographic_sdf import SDF
    soft = SDF("smooth_union", (r,), (bone_capsule(a, j, r), bone_capsule(j, b, r)))

    bisect = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
    ts = np.linspace(0.0, 4.0 * r, 400)
    P = j[None, :] + ts[:, None] * bisect[None, :]

    def surface_at(f):
        v = np.asarray(f(P), float).ravel()
        idx = np.argmax(v >= 0)
        return float(ts[idx]) if (v >= 0).any() else float(ts[-1])
    d_conv, d_soft = surface_at(conv), surface_at(soft)
    assert d_soft > d_conv * 1.10, \
        "the smooth-union corner must bulge further than the convolution one: %.4f vs %.4f" % (
            d_soft, d_conv)

    # 2) ANISOTROPY REALLY FLATTENS. A sole section must be wider than it is tall, which is what
    # ellipsoidal normal sections buy and what a circular section cannot express at all.
    flat = convolution_field([(np.array([-0.1, 0, 0]), np.array([0.1, 0, 0]), 0.05,
                              np.array([1.0, 1.6, 0.4]))])
    wide = np.linspace(0.0, 0.25, 300)
    vy = np.asarray(flat(np.stack([np.zeros(300), wide, np.zeros(300)], 1)), float).ravel()
    vz = np.asarray(flat(np.stack([np.zeros(300), np.zeros(300), wide], 1)), float).ravel()
    ry = float(wide[np.argmax(vy >= 0)])
    rz = float(wide[np.argmax(vz >= 0)])
    assert ry > 1.8 * rz, "an anisotropic section must be visibly flat: wide %.4f vs tall %.4f" % (ry, rz)

    # 3) TOES STAY SEPARATE. Grouping is what answers Bloomenthal's own open problem; without it the
    # summed convolution webs adjacent digits. Checked by counting solid runs across the toe fan.
    groups, _ = foot_skeleton(size=1.0, digits=3)
    fused = convolution_field([s for g in groups for s in g])     # everything summed together
    grouped = grouped_field(groups)
    xs = np.linspace(-0.10, 0.10, 400)
    line = np.stack([xs, np.full(400, 0.085), np.full(400, 0.045)], axis=1)

    def runs(f):
        ins = np.asarray(f(line), float).ravel() < 0
        return int((np.diff(ins.astype(int)) == 1).sum())
    r_fused, r_group = runs(fused), runs(grouped)
    assert r_group >= r_fused, \
        "grouping must not LOSE toe separations: summed %d vs grouped %d" % (r_fused, r_group)
    assert r_group >= 2, "a 3-toe fan must show separate toes, got %d runs" % r_group

    # 4) THE MIGRATION IS REAL: the part library's default foot is now this, and it is a BETTER mesh
    # than the merged-shell one it replaced -- smoother and still fully edge-manifold. Pinned so the
    # library cannot quietly fall back to shells that intersect each other.
    from holographic.mesh_and_geometry.holographic_creaturepartlib import foot as _lib_foot
    _new, _old = _lib_foot(digits=3), _lib_foot(digits=3, convolution=False)

    def _mean_dihedral(_mesh):
        _V = np.asarray(_mesh.vertices, float)
        _F = [tuple(x) for x in _mesh.faces]
        _N = {}
        for _i, _f in enumerate(_F):
            _p = _V[list(_f[:3])]
            _n = np.cross(_p[1] - _p[0], _p[2] - _p[0])
            _l = float(np.linalg.norm(_n))
            _N[_i] = _n / _l if _l > 1e-12 else np.array([0.0, 0.0, 1.0])
        _e = {}
        for _i, _f in enumerate(_F):
            for _k in range(len(_f)):
                _e.setdefault(tuple(sorted((_f[_k], _f[(_k + 1) % len(_f)]))), []).append(_i)
        return float(np.mean([np.degrees(np.arccos(np.clip(_N[v[0]] @ _N[v[1]], -1.0, 1.0)))
                              for v in _e.values() if len(v) == 2]))
    assert _mean_dihedral(_new) < _mean_dihedral(_old), \
        "the convolution foot must be smoother than the merged-shell one: %.1f vs %.1f" % (
            _mean_dihedral(_new), _mean_dihedral(_old))
    _V = np.asarray(_new.vertices, float)
    _ex = _V.max(axis=0) - _V.min(axis=0)
    assert _ex[1] > 1.8 * _ex[2], "a foot is longer than it is tall: %r" % np.round(_ex, 3).tolist()

    # 5) THE HIP, which is the junction a viewer objects to. A smooth_union mount ADDS material at the
    # joint; contiguous convolution does not. Pinned as a strict improvement, and alongside it the
    # readability gate -- a cleaner junction must not cost webbing or negative space.
    from holographic.mesh_and_geometry.holographic_creature import Creature, quadruped_spec
    from holographic.mesh_and_geometry.holographic_creaturetree import creature_tree_grouped
    from holographic.mesh_and_geometry.holographic_creaturereport import (
        webbing_report, silhouette_report)
    _cr = Creature(quadruped_spec())
    _rig = rig_of_lazy(_cr)
    _conv = creature_field(_cr)
    _tree = creature_tree_grouped(_cr, mount_flare=0.55)
    _a, _b = _rig.segment("L0#0")
    _s0, _s1 = _rig.segment("spine#1")
    _bis = (_b - _a) / np.linalg.norm(_b - _a) + (_s1 - _s0) / np.linalg.norm(_s1 - _s0)
    _bis /= np.linalg.norm(_bis)
    _ts = np.linspace(0.0, 0.35, 500)
    _P = _a[None, :] + _ts[:, None] * _bis[None, :]

    def _hip(_f):
        _v = np.asarray(_f(_P), float).ravel()
        return float(_ts[np.argmax(_v >= 0)]) if (_v >= 0).any() else float(_ts[-1])
    _h_tree, _h_conv = _hip(_tree), _hip(_conv)
    assert _h_conv < 0.92 * _h_tree, \
        "the convolution hip must bulge less than the smooth-union one: %.4f vs %.4f" % (
            _h_conv, _h_tree)
    assert webbing_report(_rig, field=_conv)["webbing_pairs"] == 0, "convolution body must not web"

    # THE HEAD MUST EXIST. `creature_groups` built spine and limb chains and NOTHING ELSE, so a
    # convolution body had no head at all -- the eyes and mouth floated in front of a headless neck,
    # and nothing in the pipeline noticed because no test asked. A missing mass is invisible to
    # webbing, to negative space and to nesting; it is only visible to someone looking, or to this.
    _hn = _cr.head["node"]
    _hp = np.asarray(_rig.joints[_hn], float)
    assert float(np.asarray(_conv(_hp[None, :]), float).ravel()[0]) < 0.0, \
        "the head must be inside the surface, not missing from it"
    # And it must be a THICKENING of the body axis, not a sphere stuck on: measurably fatter there.
    from holographic.mesh_and_geometry.holographic_creatureproportion import head_definition
    assert head_definition(_cr, field=_conv)["head_ratio"] > 1.2, "the head must read as a head"
    assert silhouette_report(_rig, field=_conv, res=56)["negative_space"] > \
        silhouette_report(_rig, field=_tree, res=56)["negative_space"], \
        "a cleaner junction must not cost readability"

    print("creatureconv selftest OK: corner bulge smooth-union %.4f vs convolution %.4f (%.0f%% "
          "less), anisotropic section %.4f wide x %.4f tall, toe runs summed %d -> grouped %d"
          % (d_soft, d_conv, 100 * (1 - d_conv / d_soft), ry, rz, r_fused, r_group))


if __name__ == "__main__":
    _selftest()

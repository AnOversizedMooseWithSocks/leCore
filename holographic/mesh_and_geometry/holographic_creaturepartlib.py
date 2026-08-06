"""Parametric creature PARTS -- eyes, mouths, feet with digits, claws, horns, fins, antennae, ears.

WHY PROCEDURAL RATHER THAN AUTHORED MESHES
------------------------------------------
Spore's parts were hand-modelled "rigblocks", each carrying a few pre-defined deformation handles.
The handles are the important half: a part is not a fixed mesh, it is a small FAMILY of shapes its
author sanctioned. Generating the family from parameters gets the same result with three advantages
that matter for this engine specifically:

    no asset pipeline   leCore has no modelling tool and no binary assets; a part that is a function
                        is a part that ships in the source tree
    determinism         the same parameters give the same bytes forever, with nothing stored -- the
                        determinism-instead-of-storage lever, applied to art
    real handle ranges  `digits=3` versus `digits=5` is a genuinely different foot, not a scale on
                        one mesh. A parametric part can express that; a deformed mesh cannot

Every part here is built from ONE workhorse (`sweep_profile`, a swept ring with per-station radii)
plus the shipped mesh primitives. Nothing in this module needs the SDF marcher, because parts are
small and a swept surface is exact, cheap and watertight where a marched one is approximate and
expensive at this scale.

HANDLES ARE DECLARED, NOT IMPLIED
    `PART_HANDLES` records each part's authored ranges, and `library()` registers them with the
    PartLibrary so `clamp()` enforces them. A caller asking for a horn ten times longer than its
    author allowed gets the author's maximum, which is what makes this a library rather than a pile
    of meshes.

KEPT NEGATIVES (loud)
  * These are SHAPES, not anatomy. A mouth is a modelled opening, not a jaw that articulates; an eye
    does not look at anything. Rigging parts to the creature's skeleton is a separate arc.
  * No UVs and no per-part materials: parts come out as plain geometry and take the colour the
    caller gives them. The material layer works on the body, not on sockets, so far.
  * Watertight but NOT manifold-checked at the seams where sub-pieces meet (a foot's toes meet its
    pad by overlap, not by a boolean). Fine for rendering, not fine as CAD input -- run the shipped
    mesh repair first if a part must be solid.
  * Everything is built +Z up, origin at the attachment point, because that is what the socket frame
    expects. A part authored in another orientation will attach sideways.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh

#: Authored deformation ranges per part -- the rigblock contract. Registered with a PartLibrary so
#: out-of-range requests are clamped to what the part's author sanctioned rather than honoured.
PART_HANDLES = {
    "eye":     {"size": (0.4, 2.0), "stalk": (0.0, 3.0)},
    "mouth":   {"width": (0.5, 2.0), "opening": (0.0, 1.0)},
    "foot":    {"size": (0.5, 2.0), "digits": (2.0, 6.0), "spread": (0.2, 1.5)},
    "hand":    {"size": (0.5, 2.0), "digits": (2.0, 6.0), "spread": (0.2, 1.5)},
    "claw":    {"length": (0.4, 2.5), "curl": (0.0, 1.5)},
    "horn":    {"length": (0.4, 3.0), "curl": (0.0, 1.5)},
    "spike":   {"length": (0.3, 2.0)},
    "fin":     {"span": (0.5, 2.5), "height": (0.4, 2.0)},
    "antenna": {"length": (0.5, 3.0), "bulb": (0.0, 2.0)},
    "ear":     {"size": (0.5, 2.5), "fold": (0.0, 1.0)},
}


def sweep_profile(path, radii, sides=10, cap_start=True, cap_end=True):
    """THE WORKHORSE: sweep a circular cross-section of varying radius along a 3-D path.

    Every tapered part here -- horn, claw, toe, antenna, ear rim -- is this function with a different
    path and radius curve. The shipped `sweep_tube` takes ONE profile for the whole tube and so
    cannot taper, which is the one thing every organic appendage does.

    Uses a rotation-minimizing frame so the ring does not spin around a curved path (the same reason
    sockets use one). Returns a watertight Mesh with optional end caps.
    """
    from holographic.mesh_and_geometry.holographic_curves import rotation_minimizing_frame
    P = np.asarray(path, float)
    r = np.broadcast_to(np.asarray(radii, float), (len(P),)).astype(float)
    n = len(P)
    if n < 2:
        raise ValueError("a swept part needs at least two path points")
    T, N, B = rotation_minimizing_frame(P)
    th = np.linspace(0.0, 2.0 * np.pi, int(sides), endpoint=False)
    cs, sn = np.cos(th), np.sin(th)
    # ring i = P[i] + r[i] * (cos*N[i] + sin*B[i]) -- one vectorised outer product, no Python loop
    V = (P[:, None, :]
         + r[:, None, None] * (cs[None, :, None] * np.asarray(N)[:, None, :]
                               + sn[None, :, None] * np.asarray(B)[:, None, :])).reshape(-1, 3)
    F = []
    s = int(sides)
    for i in range(n - 1):
        for j in range(s):
            a = i * s + j; b = i * s + (j + 1) % s
            c = (i + 1) * s + j; d = (i + 1) * s + (j + 1) % s
            F.append([a, b, d]); F.append([a, d, c])
    V = list(V)
    if cap_start:
        ci = len(V); V.append(P[0])
        F += [[ci, (j + 1) % s, j] for j in range(s)]
    if cap_end:
        ci = len(V); V.append(P[-1])
        base = (n - 1) * s
        F += [[ci, base + j, base + (j + 1) % s] for j in range(s)]
    return Mesh(np.asarray(V, float), np.asarray(F, int))


def _arc(length, curl, n=14, lean=0.0):
    """A path curving forward by `curl` over `length`, standing on +Z at the origin -- the spine of a
    horn, claw or toe. curl=0 is straight; higher curls sweep further over."""
    t = np.linspace(0.0, 1.0, int(n))
    ang = float(curl) * t ** 1.4
    x = np.cumsum(np.sin(ang)) / max(n, 1) * float(length)
    z = np.cumsum(np.cos(ang)) / max(n, 1) * float(length)
    y = t * float(lean) * float(length)
    return np.stack([x - x[0], y, z - z[0]], axis=1)


def horn(length=1.0, curl=0.6, base=0.075, sides=12, taper=0.06):
    """A tapered, curving horn standing on +Z. `curl` sweeps it forward; `taper` is the tip radius as
    a fraction of the base."""
    P = _arc(float(length) * 0.3, float(curl), n=16)
    t = np.linspace(0.0, 1.0, len(P))
    r = float(base) * ((1.0 - t) ** 1.3 * (1.0 - float(taper)) + float(taper))
    return sweep_profile(P, r, sides=sides)


def claw(length=1.0, curl=1.1, base=0.035, sides=10):
    """A claw: shorter, thinner and more strongly curved than a horn, coming to a near-point."""
    P = _arc(float(length) * 0.14, float(curl), n=14)
    t = np.linspace(0.0, 1.0, len(P))
    r = float(base) * (1.0 - t) ** 1.6 + 0.002
    return sweep_profile(P, r, sides=sides)


def spike(length=1.0, base=0.03, sides=8):
    """A straight spike -- the cheapest part, for dorsal ridges where there will be many."""
    P = np.stack([np.zeros(6), np.zeros(6), np.linspace(0, 0.16 * float(length), 6)], axis=1)
    t = np.linspace(0.0, 1.0, 6)
    return sweep_profile(P, float(base) * (1.0 - t) ** 1.5 + 0.0015, sides=sides)


def antenna(length=1.0, bulb=1.0, base=0.014, sides=8):
    """A thin antenna with an optional bulb at the tip -- the radius curve swells at the end rather
    than tapering, which is the whole difference from a spike."""
    P = _arc(float(length) * 0.22, 0.35, n=16, lean=0.15)
    t = np.linspace(0.0, 1.0, len(P))
    r = float(base) * (1.0 - 0.5 * t) + float(bulb) * 0.02 * np.exp(-((t - 1.0) / 0.16) ** 2)
    return sweep_profile(P, r, sides=sides)


def eye(size=1.0, stalk=0.0, sides=14, rings=10):
    """An eyeball, optionally raised on a stalk. Built as a swept sphere so it shares the one
    workhorse rather than pulling in the SDF marcher for a ball."""
    R = 0.05 * float(size)
    parts = []
    z0 = 0.0
    if float(stalk) > 1e-6:
        h = 0.06 * float(stalk)
        P = np.stack([np.zeros(6), np.zeros(6), np.linspace(0, h, 6)], axis=1)
        parts.append(sweep_profile(P, np.full(6, R * 0.35), sides=max(6, sides // 2)))
        z0 = h
    phi = np.linspace(-np.pi / 2 * 0.98, np.pi / 2 * 0.98, int(rings))
    P = np.stack([np.zeros(rings), np.zeros(rings), z0 + R + R * np.sin(phi)], axis=1)
    parts.append(sweep_profile(P, R * np.cos(phi), sides=sides))
    return _merge(parts)


def mouth(width=1.0, opening=0.5, sides=16):
    """A mouth: a shallow lens-shaped opening, its lips parting by `opening`. A SHAPE, not a jaw --
    see the module's kept negative before expecting it to articulate."""
    w = 0.10 * float(width)
    gap = 0.055 * float(opening) * float(width)
    parts = []
    for sign in (+1.0, -1.0):                                 # upper and lower lip
        u = np.linspace(-1.0, 1.0, int(sides))
        P = np.stack([u * w, np.zeros_like(u), sign * (gap * 0.5 + 0.012) * (1.0 - u ** 2)], axis=1)
        r = 0.016 * float(width) * (1.0 - 0.6 * u ** 2) + 0.002
        parts.append(sweep_profile(P, r, sides=8))
    return _merge(parts)


def digit(length=1.0, curl=0.5, base=0.022, with_claw=True, sides=8, knuckles=3):
    """One finger or toe: a JOINTED segment with knuckle bulges and an optional claw.

    KNUCKLES ARE WHAT MAKE A TOE READ AS A TOE. A smooth taper is a cone, and a row of cones is what
    made the feet look like a pile of spikes -- there is nothing in a cone for the eye to name. A real
    digit is a chain of phalanges: it swells at each joint and narrows between them, so even in
    silhouette you can count the segments. `knuckles` sets how many.
    """
    P = _arc(float(length) * 0.11, float(curl), n=16)
    t = np.linspace(0.0, 1.0, len(P))
    # Taper along the digit, with a cosine bulge at each joint riding on top of it.
    taper = 1.0 - 0.42 * t
    bulge = 1.0 + 0.22 * np.cos(t * float(max(int(knuckles), 1)) * 2.0 * np.pi - np.pi)
    parts = [sweep_profile(P, float(base) * taper * bulge, sides=sides)]
    if with_claw:
        c = claw(length=float(length) * 0.62, curl=1.3, base=float(base) * 0.62, sides=sides)
        parts.append(_translate(c, P[-1]))
    return _merge(parts)


def foot(size=1.0, digits=3, spread=0.9, claws=True, sides=10, convolution=True, res=56):
    """A foot. `convolution=True` (default) builds it as a CONVOLUTION SURFACE over a contiguous
    skeleton, which is how the literature builds hands and feet -- see holographic_creatureconv for
    the sources and the measurements. The old construction (a pad mesh plus N toe meshes MERGED into
    one vertex array) is reachable with `convolution=False`, and is kept only because a shipped shape
    should stay reachable: those sub-meshes were separate shells that intersected each other and were
    never joined, which is exactly what a close-up render showed.

    MEASURED, old vs convolution: mean dihedral between adjacent faces 21.9 -> 15.7 degrees, p95
    60.5 -> 55.4, both 100% edge-manifold. The convolution foot is smoother AND has a real sole
    profile (bbox length/height 2.27 rather than a plate).
    """
    if convolution:
        from holographic.mesh_and_geometry.holographic_creatureconv import foot_mesh
        return foot_mesh(size=float(size), digits=int(digits), spread=float(spread) * 0.78,
                         res=int(res))
    return _foot_merged(size=size, digits=digits, spread=spread, claws=claws, sides=sides)


def _foot_merged(size=1.0, digits=3, spread=0.9, claws=True, sides=10):
    """A foot: a pad with `digits` toes fanned out by `spread`.

    DIGIT COUNT IS THE POINT. A three-toed foot and a five-toed foot are genuinely different shapes,
    not one mesh scaled -- which is exactly what a parametric part can express and a deformed
    authored mesh cannot. This is the handle that makes the library feel like Spore's.
    """
    n = int(np.clip(round(float(digits)), 2, 6))
    s = float(size)
    pad_r = 0.055 * s

    # A FOOT IS NOT A STARFISH. The previous build swept the pad ALONG +Z for 0.028 and laid every
    # toe flat in the same plane, so the whole part measured 0.211 x 0.171 x 0.040 -- a plate with
    # spokes. Rendered at limb scale that reads as smashed geometry with no definition, which is
    # exactly what it is: no heel, no arch, no ankle, nothing for the eye to name.
    #
    # Backlog P-3 lists the three things a foot needs and this builds all three:
    #   SOLE PLANE   the pad is swept HEEL-TO-TOE along +Y with a varying radius, so it has a heel,
    #                a narrower arch and a wider ball -- a recognisable footprint rather than a disc.
    #   TOE SILHOUETTE  toes leave the BALL of the foot (not the pad's centre) and are thicker, so
    #                the gaps between them survive at render scale.
    #   ANKLE JOIN   a stub rising in +Z where the leg lands, so the foot MEETS the limb instead of
    #                being a plate stuck across its end. +Z is up once `ground_frame` orients it.
    # THE SOLE SITS ON THE SOCKET PLANE. A part's origin is its socket and it grows along +Z, which
    # `ground_frame` maps to world UP -- so for a foot the sole belongs at z ~ 0 and the ANKLE rises
    # from it. My first version centred the sole ON the plane, so half the pad hung below the socket
    # and the part-library selftest caught it ("should stand on its socket, not sink below it").
    # HEEL, ARCH, BALL. A foot is not a lozenge: it is wide at the heel, PINCHED at the arch and wide
    # again at the ball, and that double-bulge is most of what makes a footprint recognisable in
    # silhouette. The profile below is sampled finely enough for the arch to actually appear rather
    # than being averaged away between two control points.
    heel, ball = -0.95 * s * 0.09, 0.85 * s * 0.09
    sole_y = np.linspace(heel, ball, 9)
    sole_z = 0.95 * pad_r
    P = np.stack([np.zeros(9), sole_y, np.full(9, sole_z)], axis=1)
    parts = [sweep_profile(P, pad_r * np.array(
        [0.72, 0.92, 0.86, 0.66, 0.58, 0.64, 0.82, 0.98, 0.88]), sides=sides)]

    # ANKLE: a short column over the rear third, tapering up to meet the leg.
    ank = np.stack([np.zeros(4), np.full(4, heel * 0.35),
                    np.linspace(sole_z, sole_z + 1.35 * pad_r, 4)], axis=1)
    parts.append(sweep_profile(ank, pad_r * np.array([0.85, 0.72, 0.62, 0.55]), sides=sides))

    fan = np.linspace(-1.0, 1.0, n) * float(spread)
    for a in fan:
        d = digit(length=s * 0.85 * (1.0 - 0.16 * abs(a)), curl=0.30,
                  base=0.026 * s, with_claw=bool(claws), sides=max(6, sides - 2))
        d = _rotate_x(d, np.pi / 2)                      # lay the toe forward
        d = _rotate_z(d, float(a) * 0.55)                # fan it out
        parts.append(_translate(d, np.array([np.sin(a * 0.55) * pad_r * 0.85,
                                             ball + np.cos(a * 0.55) * pad_r * 0.25,
                                             sole_z])))
    return _merge(parts)


def hand(size=1.0, digits=5, spread=1.1, claws=False, sides=10):
    """A hand -- the same construction as a foot with more, longer digits and usually no claws.
    Kept as its own name because an app's part picker needs both words."""
    return foot(size=size, digits=digits, spread=spread, claws=claws, sides=sides)


def fin(span=1.0, height=1.0, rays=5, sides=6):
    """A fin: a membrane spanned by `rays` tapered spines. Built as the rays plus a thin web between
    them, so it reads as a fin from any angle instead of vanishing edge-on like a single quad."""
    parts = []
    w = 0.09 * float(span)
    h = 0.11 * float(height)
    xs = np.linspace(-w, w, int(rays))
    tip = []
    for x in xs:
        hh = h * (1.0 - 0.55 * (abs(x) / max(w, 1e-9)) ** 2)
        P = np.stack([np.full(6, x), np.zeros(6), np.linspace(0.0, hh, 6)], axis=1)
        t = np.linspace(0.0, 1.0, 6)
        parts.append(sweep_profile(P, 0.010 * float(span) * (1.0 - 0.7 * t) + 0.0015, sides=sides))
        tip.append([x, 0.0, hh])
    # the web: one thin quad strip joining the ray tips to the base line
    tip = np.asarray(tip, float)
    base = np.stack([xs, np.zeros(len(xs)), np.zeros(len(xs))], axis=1)
    V = np.vstack([base, tip])
    m = len(xs)
    F = []
    for i in range(m - 1):
        F += [[i, i + 1, m + i + 1], [i, m + i + 1, m + i]]
        F += [[i, m + i + 1, i + 1], [i, m + i, m + i + 1]]   # both windings: a membrane is two-sided
    parts.append(Mesh(V, np.asarray(F, int)))
    return _merge(parts)


def ear(size=1.0, fold=0.5, sides=8):
    """An ear: a tapered cone leaning back, thinned by `fold` so it reads as a flap rather than a
    horn. The difference between the two is entirely in the radius curve."""
    P = _arc(0.13 * float(size), 0.5 + 0.6 * float(fold), n=12, lean=-0.25)
    t = np.linspace(0.0, 1.0, len(P))
    r = 0.05 * float(size) * (1.0 - t) ** 0.8 * (1.0 - 0.55 * float(fold)) + 0.003
    return sweep_profile(P, r, sides=sides)


#: How a part sits relative to its socket. "standing" parts rise from the surface point (a horn, a
#: claw); "centred" parts STRADDLE it, because they are apertures rather than protrusions -- a mouth
#: is a hole in the skin, and pushing it above the surface so it stood on the body would look exactly
#: as wrong as it sounds. Declared rather than inferred, so the selftest can hold each kind to the
#: right contract instead of one loose rule that fits neither.
PART_ORIGIN = {"mouth": "centred"}


def part_origin(name):
    """Whether a part stands on its socket or straddles it -- what a placement routine needs to know
    if it ever wants to sink or float a part deliberately."""
    return PART_ORIGIN.get(name, "standing")


#: name -> builder. A dict rather than an if-chain so `part_names()` can list them and adding a part
#: is one entry, not a new branch in three places.
BUILDERS = {"eye": eye, "mouth": mouth, "foot": foot, "hand": hand, "claw": claw, "horn": horn,
            "spike": spike, "fin": fin, "antenna": antenna, "ear": ear, "digit": digit}


def part_names():
    """Every part this library can build -- what an app's part picker enumerates."""
    return sorted(BUILDERS)


def build_part(name, **params):
    """Build one part by name with its parameters. Unknown names raise rather than returning an empty
    mesh, so a typo in a saved document surfaces immediately instead of as an invisible part."""
    if name not in BUILDERS:
        raise ValueError("unknown part %r; one of %s" % (name, part_names()))
    return BUILDERS[name](**params)


def library(dim=1024, seed=0, params=None):
    """A PartLibrary pre-loaded with every part, its authored handle ranges, and its default geometry.

    This is the one call an app makes to get a working part palette. `params` overrides the defaults
    used to build each part's representative mesh.
    """
    from holographic.mesh_and_geometry.holographic_creatureparts import PartLibrary
    lib = PartLibrary(dim=int(dim), seed=int(seed))
    over = dict(params or {})
    for name in part_names():
        lib.define(name, handles=PART_HANDLES.get(name, {}),
                   geometry=build_part(name, **over.get(name, {})))
    return lib


# ------------------------------------------------------------------ small mesh helpers --

def _merge(meshes):
    """Concatenate sub-pieces into one mesh. Overlap rather than boolean union -- see the kept
    negative about manifoldness before feeding a part to CAD."""
    V, F, off = [], [], 0
    for m in meshes:
        v = np.asarray(m.vertices, float); f = np.asarray(m.faces, int)
        V.append(v); F.append(f + off); off += len(v)
    return Mesh(np.concatenate(V), np.concatenate(F))


def _translate(m, offset):
    """Move a mesh. Local helper rather than a matrix, because parts are assembled from a handful of
    rigid moves and building 4x4s for each would be noise."""
    return Mesh(np.asarray(m.vertices, float) + np.asarray(offset, float)[None, :],
                np.asarray(m.faces, int))


def _rotate_x(m, a):
    """Rotate about +X (lay a part forward)."""
    c, s = np.cos(a), np.sin(a)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    return Mesh(np.asarray(m.vertices, float) @ R.T, np.asarray(m.faces, int))


def _rotate_z(m, a):
    """Rotate about +Z (fan a digit out)."""
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
    return Mesh(np.asarray(m.vertices, float) @ R.T, np.asarray(m.faces, int))


def _selftest():
    """Contracts: every part builds and is non-degenerate, handles genuinely change the SHAPE (not
    just the scale), digit count changes topology, everything stands on +Z at the origin, and the
    library registers ranges that clamp."""
    # 1) Every part builds, has area, and is finite.
    for name in part_names():
        m = build_part(name)
        V = np.asarray(m.vertices, float); F = np.asarray(m.faces, int)
        assert len(V) > 3 and len(F) > 3, "%s is degenerate" % name
        assert np.isfinite(V).all(), "%s has non-finite vertices" % name
        assert F.max() < len(V), "%s has an out-of-range face index" % name
        # Parts are authored +Z up from the origin, which is what the socket frame expects. An
        # APERTURE (a mouth) legitimately straddles the surface instead of standing on it, so each
        # kind is held to its own contract rather than to one rule that fits neither.
        assert V[:, 2].max() > 0.01, "%s has no height" % name
        if part_origin(name) == "standing":
            assert V[:, 2].min() > -0.03, "%s should stand on its socket, not sink below it" % name
        else:
            assert V[:, 2].min() < 0.0 < V[:, 2].max(), \
                "%s is declared an aperture, so it must straddle the surface" % name

    # 2) DIGIT COUNT CHANGES TOPOLOGY, not just size -- the handle that a deformed authored mesh
    #    could not express, and the reason these are parametric.
    f3, f5 = foot(digits=3), foot(digits=5)
    assert len(np.asarray(f5.faces)) > len(np.asarray(f3.faces)), \
        "a five-toed foot must have MORE geometry than a three-toed one (%d vs %d)" % (
            len(np.asarray(f5.faces)), len(np.asarray(f3.faces)))
    assert len(np.asarray(foot(digits=99).faces)) == len(np.asarray(foot(digits=6).faces)), \
        "digit count must clamp to the authored maximum"

    # 3) HANDLES CHANGE THE SHAPE. A longer horn is longer; a curled one leans further forward; a
    #    stalked eye is taller. Each is measured, not assumed.
    assert horn(length=2.0).vertices[:, 2].max() > 1.6 * horn(length=1.0).vertices[:, 2].max()
    straight = np.asarray(horn(curl=0.0).vertices, float)[:, 0].max()
    curled = np.asarray(horn(curl=1.4).vertices, float)[:, 0].max()
    assert curled > straight + 0.02, "curl must sweep the horn forward (%.3f vs %.3f)" % (curled, straight)
    assert np.asarray(eye(stalk=2.0).vertices)[:, 2].max() > \
        np.asarray(eye(stalk=0.0).vertices)[:, 2].max() + 0.05, "a stalk must raise the eye"
    def _extent(m, ax):
        v = np.asarray(m.vertices, float)[:, ax]
        return float(v.max() - v.min())                       # np 2.x removed ndarray.ptp()
    assert _extent(mouth(opening=1.0), 2) > _extent(mouth(opening=0.0), 2), "opening must part the lips"
    assert _extent(fin(span=2.0), 0) > 1.6 * _extent(fin(span=1.0), 0), "span must widen the fin"

    # 4) THE SWEEP is watertight in the sense that matters here: every ring is closed, and the caps
    #    add exactly one vertex each.
    P = np.stack([np.zeros(5), np.zeros(5), np.linspace(0, 1, 5)], axis=1)
    s = sweep_profile(P, np.linspace(0.1, 0.02, 5), sides=8)
    assert len(np.asarray(s.vertices)) == 5 * 8 + 2, "5 rings of 8 plus two cap centres"
    assert len(np.asarray(s.faces)) == 4 * 8 * 2 + 2 * 8, "quads between rings, fans on the caps"
    # a tapered sweep must actually taper
    V = np.asarray(s.vertices, float)
    r0 = np.linalg.norm(V[:8, :2], axis=1).mean()
    r1 = np.linalg.norm(V[32:40, :2], axis=1).mean()
    assert r0 > 4 * r1, "the sweep must taper along the path (%.3f -> %.3f)" % (r0, r1)

    # 5) DETERMINISM: same parameters, same bytes.
    assert np.array_equal(np.asarray(foot(digits=4).vertices), np.asarray(foot(digits=4).vertices))

    # 6) THE LIBRARY registers every part with its ranges, and clamps out-of-range requests.
    lib = library(dim=256, seed=0)
    assert set(lib.parts) == set(part_names())
    assert abs(lib.clamp("horn", "length", 99.0) - 3.0) < 1e-9, "clamp to the authored maximum"
    assert abs(lib.clamp("horn", "length", 0.0) - 0.4) < 1e-9, "and to the authored minimum"
    assert all(lib.parts[n]["geometry"] is not None for n in part_names()), \
        "every registered part must carry geometry an editor can place"

    print("creaturepartlib selftest OK: %d parts all build +Z-up and non-degenerate, digits 3->5 "
          "changes topology (%d -> %d faces), curl sweeps %.3f -> %.3f, handles clamp 99 -> 3.0"
          % (len(part_names()), len(np.asarray(f3.faces)), len(np.asarray(f5.faces)),
             straight, curled))


if __name__ == "__main__":
    _selftest()

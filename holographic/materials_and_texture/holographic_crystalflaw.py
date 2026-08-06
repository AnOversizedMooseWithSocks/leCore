"""CRYSTAL IMPERFECTIONS -- inclusions, cloudiness, phantoms, fractures and chipped faces.

A perfect crystal reads as GLASS. What makes a render look like a mineral specimen rather than a
prop is that real crystals are flawed in specific, structured ways, and every one of these is an
optical variation INSIDE one solid rather than a change to its outline:

    CLOUDINESS  milky quartz is not a different material, it is the same quartz full of sub-micron
                fluid inclusions that scatter. Optically that is high absorption with LOW colour
                selectivity plus a rough interior -- white haze, not a tint.
    INCLUSIONS  rutile needles, chlorite blebs, hematite flecks: small foreign bodies with their own
                colour, suspended in the host.
    PHANTOMS    a growth pause leaves a dusty layer that the crystal then grows over, so a ghost of
                the earlier, smaller crystal floats inside the finished one. Concentric with the
                habit, because it IS the habit at an earlier size.
    FRACTURES   internal cleavage planes that catch light as bright sheets.
    CHIPS       real specimens are damaged: faces are dinged and edges knocked off.

The first four are MATERIAL modifiers -- they change absorption, albedo and roughness as a function
of position, and leave the geometry alone. Only chipping touches the SDF. That split is the design:
a modifier composes with any habit, any growth mode and any host material, so imperfections do not
have to be re-implemented per crystal type.

KEPT NEGATIVE, stated up front: this is phenomenological. Real inclusions are minerals with their own
refractive indices, and light entering a rutile needle refracts at ITS boundary; here they are
absorption and albedo variations within the host's index. That is a good approximation for small
scattered flecks and a poor one for a large included crystal.
"""

import numpy as np


def _hash_noise(P, freq, seed, octaves=3):
    """Deterministic value noise from summed sinusoids -- NumPy-only and reproducible, no library.

    Sinusoids rather than a hash grid because the result must be C-infinity: absorption is sampled
    along a ray at arbitrary points, and a discontinuous field would show as banding in the
    transmitted light rather than as cloud.
    """
    rng = np.random.default_rng(int(seed))
    Q = np.atleast_2d(np.asarray(P, float))
    out = np.zeros(len(Q))
    amp, f = 1.0, float(freq)
    total = 0.0
    for _ in range(int(octaves)):
        d = rng.normal(0.0, 1.0, (3, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        ph = rng.uniform(0.0, 2 * np.pi, 3)
        v = np.ones(len(Q))
        for k in range(3):
            v = v * np.sin(f * (Q @ d[k]) + ph[k])
        out += amp * v
        total += amp
        amp *= 0.55
        f *= 2.1
    return 0.5 + 0.5 * (out / max(total, 1e-9))


def cloudiness(strength=1.0, freq=9.0, seed=0, threshold=0.45, sharp=5.0):
    """MILKY zones: a field in [0,1] that is high where the crystal should be clouded.

    Returns a callable P -> [0,1]. Multiply a material's absorption by it (and desaturate) to get
    milky quartz: real cloudiness is scattering by trapped fluid, so it whitens and dims WITHOUT
    tinting -- absorption that is equal across RGB, not more of the crystal's own colour.
    """
    def f(P, _s=float(strength), _f=float(freq), _sd=int(seed), _t=float(threshold), _k=float(sharp)):
        n = _hash_noise(P, _f, _sd)
        return _s / (1.0 + np.exp(-_k * (n - _t)))
    return f


def phantom(habit_sdf, fractions=(0.45, 0.72), width=0.035, strength=1.0):
    """PHANTOMS: ghost outlines of the crystal at earlier, smaller sizes.

    A growth pause dusts the surface, then growth resumes and buries it -- so the ghost is the SAME
    habit scaled down, which is why phantoms are always concentric with the crystal that contains
    them and never an arbitrary blob. Built by evaluating the habit's own field at a scaled position
    and lighting up a thin shell where it crosses zero.

    Returns P -> [0,1], high on the ghost surfaces.
    """
    fr = tuple(float(x) for x in fractions)
    w = float(width)

    def f(P, _h=habit_sdf, _fr=fr, _w=w, _s=float(strength)):
        Q = np.atleast_2d(np.asarray(P, float))
        acc = np.zeros(len(Q))
        for a in _fr:
            d = np.asarray(_h(Q / max(a, 1e-6)), float).ravel() * a
            acc = np.maximum(acc, np.exp(-(d / max(_w, 1e-6)) ** 2))
        return _s * acc
    return f


def inclusions(count=40, radius=0.035, extent=1.0, seed=0, elongation=1.0):
    """Suspended foreign bodies -- blebs, or NEEDLES when `elongation` > 1 (rutile).

    Returns P -> [0,1], high inside an inclusion. Deliberately a FIELD rather than geometry: an
    inclusion sits inside the host and is seen THROUGH it, so what matters optically is that light
    passing through that region is absorbed and coloured differently, not that the surface changes.
    """
    rng = np.random.default_rng(int(seed))
    C = rng.uniform(-float(extent), float(extent), (int(count), 3))
    A = rng.normal(0.0, 1.0, (int(count), 3))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    R = float(radius) * rng.uniform(0.5, 1.5, int(count))
    e = max(float(elongation), 1.0)

    def f(P, _C=C, _A=A, _R=R, _e=e):
        Q = np.atleast_2d(np.asarray(P, float))
        best = np.zeros(len(Q))
        for i in range(len(_C)):
            d = Q - _C[i][None, :]
            along = d @ _A[i]
            perp = d - along[:, None] * _A[i][None, :]
            # squash along the needle axis -> a capsule-ish ellipsoid without a second primitive
            r2 = (np.linalg.norm(perp, axis=1) / _R[i]) ** 2 + (along / (_R[i] * _e)) ** 2
            best = np.maximum(best, np.exp(-3.0 * r2))
        return best
    return f


def fractures(count=6, seed=0, width=0.02, extent=1.0):
    """Internal cleavage planes -- thin sheets that catch light inside the crystal."""
    rng = np.random.default_rng(int(seed))
    N = rng.normal(0.0, 1.0, (int(count), 3))
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    O = rng.uniform(-float(extent) * 0.6, float(extent) * 0.6, (int(count), 3))
    w = float(width)

    def f(P, _N=N, _O=O, _w=w):
        Q = np.atleast_2d(np.asarray(P, float))
        acc = np.zeros(len(Q))
        for i in range(len(_N)):
            d = np.abs((Q - _O[i][None, :]) @ _N[i])
            acc = np.maximum(acc, np.exp(-(d / max(_w, 1e-6)) ** 2))
        return acc
    return f


def chipped(sdf, count=10, radius=0.06, extent=0.6, seed=0):
    """Knock chips off a solid -- the only imperfection here that changes GEOMETRY.

    Real specimens are damaged; a crystal with mathematically perfect edges reads as CGI. Subtracts
    small spheres placed near the surface, so edges and corners take the damage (they are where a
    random nearby sphere is most likely to bite), which is also where real specimens chip.
    """
    rng = np.random.default_rng(int(seed))
    tries = rng.uniform(-float(extent), float(extent), (int(count) * 12, 3))
    d = np.asarray(sdf(tries), float).ravel()
    near = tries[np.abs(d) < float(radius) * 1.2][:int(count)]
    if len(near) == 0:
        return sdf
    R = float(radius) * rng.uniform(0.5, 1.2, len(near))

    def f(P, _s=sdf, _C=near, _R=R):
        Q = np.atleast_2d(np.asarray(P, float))
        out = np.asarray(_s(Q), float).ravel()
        for i in range(len(_C)):
            bite = np.linalg.norm(Q - _C[i][None, :], axis=1) - _R[i]
            out = np.maximum(out, -bite)                      # subtract the sphere
        return out
    f.eval = f
    return f


def flawed_material(gem, cloud=None, incl=None, phan=None, frac=None,
                    incl_color=(0.20, 0.14, 0.10), absorb_scale=1.0):
    """Compose a gem material with imperfection fields into the path tracer's 8-channel callback.

    Each field modifies the OPTICS, not the shape:
      cloudiness  raises absorption EQUALLY across RGB (white haze, not more colour) and roughens
      inclusions  swap in their own dark albedo and absorb hard -- they are opaque specks
      phantom     a dusty veil: mild neutral absorption on a thin concentric shell
      fractures   bright sheets: low absorption and a lifted albedo, so they catch light

    Returns a callback ready for `path_trace(material=...)`.
    """
    from holographic.materials_and_texture.holographic_matlib import material

    mat = material(gem)
    alb = np.array(mat.base_color[:3], float)
    sig = np.array(getattr(mat, "absorption", (0.05,) * 3), float) * float(absorb_scale)
    ior = float(getattr(mat, "ior", 1.55))
    rough = float(getattr(mat, "roughness", 0.05))
    inc_a = np.array(incl_color, float)

    def cb(P, _a=alb, _s=sig, _i=ior, _r=rough, _ia=inc_a):
        Q = np.atleast_2d(np.asarray(P, float))
        n = len(Q)
        A = np.tile(_a, (n, 1))
        S = np.tile(_s, (n, 1))
        RG = np.full(n, _r)
        if cloud is not None:
            c = np.asarray(cloud(Q), float).ravel()[:, None]
            # NEUTRAL absorption: milkiness is scattering, so it whitens rather than saturating.
            S = S + c * 2.4
            A = A * (1 - 0.55 * c) + np.array([[0.88, 0.88, 0.90]]) * (0.55 * c)
            RG = RG + 0.35 * c.ravel()
        if phan is not None:
            p = np.asarray(phan(Q), float).ravel()[:, None]
            S = S + p * 1.6
            A = A * (1 - 0.4 * p) + np.array([[0.80, 0.78, 0.76]]) * (0.4 * p)
        if frac is not None:
            f = np.asarray(frac(Q), float).ravel()[:, None]
            A = A * (1 - 0.6 * f) + np.array([[0.95, 0.96, 1.00]]) * (0.6 * f)
            S = S * (1 - 0.7 * f)
        if incl is not None:
            g = np.asarray(incl(Q), float).ravel()[:, None]
            A = A * (1 - g) + _ia[None, :] * g
            S = S + g * 14.0                                  # specks read as opaque
        return (A, np.zeros(n), np.clip(RG, 0.0, 1.0), np.zeros((n, 3)),
                np.full(n, _i), np.zeros(n), np.zeros(n), np.clip(S, 0.0, None))
    return cb


def _selftest():
    from holographic.mesh_and_geometry.holographic_crystalgrow import habit_sdf

    rng = np.random.default_rng(0)
    Q = rng.uniform(-0.6, 0.6, size=(6000, 3))

    # 1) EVERY FIELD IS IN [0,1] AND ACTUALLY VARIES. A modifier that returns a constant is a no-op
    # dressed as a feature -- the spread is what makes it an imperfection rather than a tint.
    for name, f in (("cloud", cloudiness(freq=8.0, seed=1)),
                    ("inclusions", inclusions(count=25, radius=0.05, extent=0.5, seed=2)),
                    ("fractures", fractures(count=5, seed=3, extent=0.5))):
        v = np.asarray(f(Q), float).ravel()
        assert v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9, "%s out of [0,1]: %.3f..%.3f" % (
            name, v.min(), v.max())
        assert v.std() > 0.01, "%s must VARY, std %.4f" % (name, v.std())

    # 2) A PHANTOM IS CONCENTRIC WITH ITS HABIT, which is what distinguishes it from a random blob:
    # it must light up INSIDE the crystal, near the scaled-down copy of the same form.
    h = habit_sdf("quartz", 0.5)
    ph = phantom(h, fractions=(0.55,), width=0.03)
    inside = np.asarray(h(Q), float).ravel() < 0
    v = np.asarray(ph(Q), float).ravel()
    assert v[inside].max() > 0.5, "the phantom must appear inside the crystal: %.3f" % v[inside].max()
    # and it must sit near the SCALED habit surface, not at the outer wall
    dscaled = np.abs(np.asarray(h(Q / 0.55), float).ravel() * 0.55)
    hot = v > 0.5
    assert dscaled[hot].mean() < 0.05, \
        "phantom must hug the scaled habit surface: mean |d| %.4f" % dscaled[hot].mean()

    # 3) CLOUDINESS WHITENS RATHER THAN SATURATES. This is the physical claim -- milky quartz is
    # scattering, not more pigment -- so the clouded albedo must be LESS saturated, and absorption
    # must rise about EQUALLY across RGB rather than in the gem's own colour ratio.
    clean = flawed_material("amethyst")
    milky = flawed_material("amethyst", cloud=cloudiness(strength=1.0, freq=8.0, seed=4, threshold=0.2))
    Ac, _, _, _, _, _, _, Sc = clean(Q)
    Am, _, _, _, _, _, _, Sm = milky(Q)
    sat = lambda A: float(np.mean((A.max(axis=1) - A.min(axis=1)) / np.maximum(A.max(axis=1), 1e-9)))
    assert sat(Am) < sat(Ac), "cloudiness must DESATURATE: %.3f vs %.3f" % (sat(Am), sat(Ac))
    added = (Sm - Sc).mean(axis=0)
    assert added.max() > 0.2, "cloudiness must raise absorption: %s" % np.round(added, 3)
    assert (added.max() - added.min()) / max(added.max(), 1e-9) < 0.25, \
        "cloud absorption must be NEUTRAL across RGB (scattering, not pigment): %s" % np.round(added, 3)

    # 4) INCLUSIONS ARE DARK AND OPAQUE where they are, and absent elsewhere.
    inc = inclusions(count=30, radius=0.06, extent=0.5, seed=5)
    withinc = flawed_material("amethyst", incl=inc)
    Ai, _, _, _, _, _, _, Si = withinc(Q)
    g = np.asarray(inc(Q), float).ravel()
    hot = g > 0.5
    assert hot.sum() > 5, "the test needs some inclusion samples, got %d" % hot.sum()
    assert Ai[hot].mean() < Ai[~hot].mean(), "inclusions must be DARKER than the host"
    assert Si[hot].mean() > Si[~hot].mean() * 3.0, "inclusions must absorb hard (opaque specks)"

    # 5) CHIPPING REMOVES MATERIAL and only material -- a chip cannot ADD solid.
    base = habit_sdf("quartz", 0.5)
    ch = chipped(base, count=8, radius=0.07, extent=0.5, seed=6)
    b = np.asarray(base(Q), float).ravel()
    c = np.asarray(ch(Q), float).ravel()
    assert np.all(c >= b - 1e-9), "chipping must never add solid"
    assert (c > 0).sum() > (b > 0).sum(), "chipping must remove some solid"

    print("crystalflaw selftest OK: fields vary in [0,1], phantom hugs the scaled habit (|d| %.4f), "
          "cloud desaturates %.3f -> %.3f with NEUTRAL absorption %s, inclusions darker and %0.f x "
          "more absorbing, chips only subtract"
          % (dscaled[hot if False else (v > 0.5)].mean(), sat(Ac), sat(Am), np.round(added, 2),
             Si[g > 0.5].mean() / max(Si[g <= 0.5].mean(), 1e-9)))


if __name__ == "__main__":
    _selftest()

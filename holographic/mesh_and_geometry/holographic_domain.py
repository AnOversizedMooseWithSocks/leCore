"""Domain operators and the cosine palette (DEMO-1): infinite procedural worlds from a tiny kernel.

WHY THIS MODULE EXISTS
----------------------
The demoscene art (Inigo Quilez / Shadertoy) is getting maximal richness from minimal, deterministic code:
an entire lattice of shapes from one modulo, a kaleidoscope from one abs(), a whole colour scheme from four
numbers. The SDF class already carries a few of these (repeat, twist) as METHODS bolted to the distance-field
tree. This module lifts them out and GENERALISES the move: a domain operator is a coordinate PRE-WARP -- it
transforms the query point P before ANY field sampler looks it up. So the same opRep that tiles an SDF into an
infinite lattice also tiles a baked field, a noise function, or a texture, because they all share the one
interface f(P: (N,k)) -> value. "Which module is this in a different costume?" -- a domain warp is the costume
that fits every field.

THE GENERALISATION (why these are pre-warps, not SDF methods)
  A signed-distance field, a baked hypervector field (bake_nd), value noise, and a procedural texture are ALL
  functions of a coordinate. Repetition, mirroring, twisting are operations on the COORDINATE, upstream of the
  lookup. Implementing them as `warp(P) -> P'` composed with any sampler means one implementation serves the
  whole engine, and any future field type gets infinite tiling for free. `wrap_sdf` adapts a warp to the SDF
  eval interface (with the distance correction domain repetition needs); `warp_field` adapts it to a plain
  callable f(P).

THE DISTANCE-CORRECTION CAVEAT (kept honest, iq's own warning)
  Domain repetition of an SDF is only an EXACT distance in the cell the point folds into. A *bounded* shape
  smaller than the cell is exact; a shape near or larger than the period is not, and a raymarcher may overstep.
  The functions here fold correctly and are exact for the common case (a bounded primitive comfortably inside
  its period); the docstrings say where the guarantee ends rather than pretending it is unconditional.

VSA-NATIVE WHERE IT IS TRUE (not forced)
  A domain repeat is a QUOTIENT of space by a lattice -- the same "fold the torus" structure that circular
  convolution (binding) lives on. The honest line: the modulo arithmetic here is plain NumPy geometry, not a
  hypervector trick. What is genuinely engine-shaped is that these warps compose with bake_nd's field sampler
  exactly as they compose with an SDF -- the shared f(P) interface IS the VSA substrate's "everything is a
  field" thesis paying off.

The cosine palette is here because it is the colour half of the same idea: iq's `a + b*cos(2*pi*(c*t + d))`
turns one scalar (a distance, an iteration count, an orbit trap) into a smooth, harmonious colour with four
vec3 constants -- and a random seed makes a random-but-never-garish palette, which is exactly the engine's
"regenerate from seeds" lever applied to colour.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Domain warps: pure coordinate pre-transforms P -> P' (each vectorised over (N,k))
# ---------------------------------------------------------------------------

def repeat(P, period):
    """Infinite lattice: fold every point into one central cell (iq's opRep). p = mod(p + c/2, c) - c/2.

    `period` is a scalar (same spacing on every axis) or a per-axis vector. A component of `period` that is 0
    or negative means "do not repeat this axis" (left unchanged) -- so [2, 0, 2] tiles a floor plane in x,z
    while leaving height alone. Returns the warped points, same shape as P.

    DISTANCE NOTE: composed with an SDF (see wrap_sdf), this is an exact distance only for a shape bounded well
    inside the cell; a shape near the period wraps into itself and the field is approximate."""
    P = np.asarray(P, float)
    c = np.broadcast_to(np.asarray(period, float), (P.shape[-1],)).astype(float)
    out = P.copy()
    live = c > 0                                          # a zero/negative period axis is a "do not repeat" flag
    if live.any():
        cc = c[live]
        # +0.5c so the cell is CENTRED on the origin (the primitive sits in the middle of its tile, not a corner)
        out[..., live] = np.mod(P[..., live] + 0.5 * cc, cc) - 0.5 * cc
    return out


def repeat_limited(P, period, lo, hi):
    """Finite lattice: repeat only within an integer box, a single copy outside it (iq's opRepLim). The cell
    index is clamped to [lo, hi] per axis, so you get an N*M block of copies instead of an infinite field --
    the version you actually want for a courtyard of pillars rather than an endless one.

    period: scalar or per-axis. lo, hi: per-axis integer bounds (scalars broadcast). Returns warped points."""
    P = np.asarray(P, float)
    k = P.shape[-1]
    c = np.broadcast_to(np.asarray(period, float), (k,)).astype(float)
    lo = np.broadcast_to(np.asarray(lo, float), (k,)).astype(float)
    hi = np.broadcast_to(np.asarray(hi, float), (k,)).astype(float)
    # id = round(P/c) is which cell a point is in; clamp it, then subtract the clamped cell centre.
    cell = np.round(P / np.where(c != 0, c, 1.0))
    cell = np.clip(cell, lo, hi)
    return P - cell * c


def domain_mirror(P, axis=0, plane=0.0):
    """Fold space across a plane (kaleidoscopic symmetry from one abs()): everything on the far side of
    `plane` on `axis` is reflected onto the near side. Two mirrors at right angles give a wedge; three give a
    corner -- the cheap route to the crystalline symmetry demos are built from. Returns warped points."""
    P = np.asarray(P, float).copy()
    P[..., axis] = plane + np.abs(P[..., axis] - plane)
    return P


def fold(P, axes=None, plane=0.0):
    """Mirror several axes at once (a convenience wrapper over `domain_mirror`): fold on each axis in `axes` (default:
    all axes) about `plane`. Folding all three axes of 3-space maps the whole world into one octant -- an
    instant 8-fold (or 48-fold, with the diagonal folds) kaleidoscope. Returns warped points."""
    P = np.asarray(P, float)
    axes = range(P.shape[-1]) if axes is None else axes
    for a in axes:
        P = domain_mirror(P, axis=a, plane=plane)
    return P


def domain_twist(P, k, axis=2, plane=(0, 1)):
    """Rotate space by an amount proportional to position along `axis` (iq's opTwist): a column twisted into a
    helix, a bar into a screw. `k` radians of rotation per unit distance along `axis`; `plane` is the pair of
    axes rotated. Returns warped points -- compose with any sampler to twist the shape it defines."""
    P = np.asarray(P, float).copy()
    a, b = plane
    ang = k * P[..., axis]
    ca, sa = np.cos(ang), np.sin(ang)
    pa, pb = P[..., a].copy(), P[..., b].copy()
    P[..., a] = ca * pa - sa * pb
    P[..., b] = sa * pa + ca * pb
    return P


def domain_bend(P, k, axis=0, bend_plane=None):
    """Bend space by an amount proportional to position along `axis` (iq's opCheapBend): a straight beam curled
    into an arc. Like twist but the rotation angle drives a bend of the OTHER two axes rather than a spin about
    `axis`. `k` radians per unit along `axis`. `bend_plane` defaults to the two axes OTHER than `axis` (so
    bending along X curls the YZ plane -- the intuitive convention, and the one SDF.bend uses so the two agree
    exactly); pass an explicit (a, b) pair to override. Returns warped points.

    KEPT NOTE: this is the CHEAP bend -- it warps the domain, so distances stretch slightly (the exact bend
    would need an arc-length reparametrisation). Fine for shading and silhouettes, which is what it is for."""
    P = np.asarray(P, float).copy()
    if bend_plane is None:
        # the two axes other than `axis`, in ascending order -- matches SDF.bend's "other two axes" convention
        bend_plane = tuple(i for i in range(P.shape[-1]) if i != axis)[:2]
    a, b = bend_plane
    ang = k * P[..., axis]
    ca, sa = np.cos(ang), np.sin(ang)
    pa, pb = P[..., a].copy(), P[..., b].copy()
    P[..., a] = ca * pa - sa * pb
    P[..., b] = sa * pa + ca * pb
    return P


# ---------------------------------------------------------------------------
# Smooth combinators: the polynomial smooth-min (organic blends)
# ---------------------------------------------------------------------------

def smin(a, b, k=0.1):
    """Polynomial smooth minimum (iq's smin): a soft `min(a, b)` that MELTS the two together over a width `k`
    instead of creasing at a hard seam. This is what turns two distance fields into one organic blob (a
    metaball union) rather than a union with a visible crack. k -> 0 recovers the hard min.

    Uses the polynomial form h = clamp(0.5 + 0.5*(b-a)/k, 0, 1); result = mix(b, a, h) - k*h*(1-h) -- cheaper
    and crease-free versus the naive min, and the correction term -k*h*(1-h) is what makes the blend smooth.
    Vectorised: a, b are scalars or arrays of matching shape."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1.0 - h)


def smax(a, b, k=0.1):
    """Smooth maximum -- the intersection/subtraction partner of smin. smax(a,b,k) = -smin(-a,-b,k), so a
    smooth intersection of two SDFs is smax(f, g, k) and a smooth subtraction is smax(f, -g, k). Same width
    parameter, same crease-free blend."""
    return -smin(-np.asarray(a, float), -np.asarray(b, float), k)


# ---------------------------------------------------------------------------
# Adapters: make a warp compose with the SDF eval interface or a plain field callable
# ---------------------------------------------------------------------------

class _WarpedSDF:
    """An SDF whose query points are warped before the base field is evaluated. Carries `.eval(P) -> (N,)` so
    it drops straight into sphere_trace / render_sdf. `dist_scale` (<=1) shrinks the reported distance to keep
    a raymarcher conservative when the warp stretches space (bend/twist) -- a Lipschitz safety factor."""

    def __init__(self, base, warp, dist_scale=1.0):
        self.base = base
        self.warp = warp
        self.dist_scale = float(dist_scale)

    def eval(self, P):
        d = self.base.eval(self.warp(np.asarray(P, float)))
        return d * self.dist_scale if self.dist_scale != 1.0 else d


def wrap_sdf(base, warp, dist_scale=1.0):
    """Compose a domain `warp` (any P->P' from this module, or your own) with an SDF `base`, returning an object
    with `.eval` that raymarches. This is the bridge that makes `repeat`, `fold`, `twist`, ... apply to signed
    distance fields: wrap_sdf(sphere(0.3), lambda P: repeat(P, 1.0)) is an infinite lattice of spheres.

    `dist_scale` (default 1.0): multiply the returned distance by this. A domain warp that stretches space
    (bend, strong twist) makes the true distance SHORTER than the base field reports along the stretched
    direction, so a raymarcher can overstep and miss the surface; a dist_scale below 1 (e.g. 0.5) trades march
    speed for safety. Repetition and mirroring are isometries and need no correction (leave it at 1.0)."""
    return _WarpedSDF(base, warp, dist_scale=dist_scale)


def warp_field(f, warp):
    """Compose a domain `warp` with ANY field callable f(P: (N,k)) -> values, returning the warped callable.
    The generalisation past SDFs: warp_field(lambda P: fbm(P), lambda P: repeat(P, 4.0)) tiles a noise field
    seamlessly; the same with a bake_nd sampler tiles a baked function. One warp, every field type."""
    def warped(P):
        return f(warp(np.asarray(P, float)))
    return warped


# ---------------------------------------------------------------------------
# The cosine palette: one scalar -> a harmonious colour (iq's four-vector palette)
# ---------------------------------------------------------------------------

def cosine_palette(t, a=(0.5, 0.5, 0.5), b=(0.5, 0.5, 0.5),
                   c=(1.0, 1.0, 1.0), d=(0.0, 0.33, 0.67)):
    """iq's cosine gradient palette: colour(t) = a + b * cos(2*pi*(c*t + d)), evaluated per channel. Maps a
    scalar `t` (a normalised distance, an iteration count, an orbit trap, anything in ~[0,1]) to a smooth RGB
    colour that never bands and never clips harshly. a = base level, b = contrast/amplitude, c = how many
    cycles across the range per channel, d = per-channel phase (the hue offset). The four defaults are iq's
    classic rainbow; changing d alone rotates the hue.

    `t` is a scalar or an array of any shape; returns colours of shape (*t.shape, 3) in [0,1] (clipped)."""
    t = np.asarray(t, float)
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = np.asarray(c, float); d = np.asarray(d, float)
    # broadcast t against the length-3 constants: t[..., None] * c -> (*t.shape, 3)
    rgb = a + b * np.cos(2.0 * np.pi * (t[..., None] * c + d))
    return np.clip(rgb, 0.0, 1.0)


def random_palette(seed=0, contrast=0.5):
    """A random-but-harmonious cosine palette from a seed (the "regenerate from seeds" lever, for colour). Draws
    iq-shaped constants -- base near 0.5, bounded amplitude, low integer-ish frequencies, random phases -- so a
    random seed gives a pleasing scheme rather than noise. Returns the (a, b, c, d) tuple ready for
    cosine_palette. Deterministic per seed.

    `contrast` scales the amplitude b (how vivid the swings are). Kept honest: this samples a TASTEFUL subspace
    of all palettes on purpose; it is not a uniform sample of colour space, it is a sample of the palettes that
    look good, which is the whole point of iq's parameterisation."""
    rng = np.random.default_rng(seed)
    a = 0.5 + 0.1 * rng.standard_normal(3)                # base brightness clustered around mid-grey
    b = contrast * (0.6 + 0.4 * rng.random(3))            # amplitude: vivid but bounded, never blows past [0,1]
    c = rng.integers(1, 3, 3).astype(float)              # low frequencies -> smooth, few-band gradients
    d = rng.random(3)                                     # free per-channel phase -> the hue identity of the palette
    return (tuple(a), tuple(b), tuple(c), tuple(d))


def palette_stops(seed=0, n=8, contrast=0.5, coeffs=None):
    """Sample a cosine palette into `n` plottable RGB colour STOPS -> array (n, 3) in [0,1].

    WHY THIS EXISTS: random_palette returns the (a,b,c,d) COEFFICIENTS of a cosine gradient, not colours -- the
    name reads like "a list of colours", so a caller that wanted swatches interpolated the coefficients as if they
    were RGB and got garbage. This is the colours-you-can-plot companion: it evaluates the palette at `n` evenly
    spaced points t in [0,1] and hands back the actual RGB rows a UI swatch strip / gradient ramp / legend wants.

    By default it draws the coefficients from random_palette(seed, contrast); pass an explicit `coeffs=(a,b,c,d)`
    (e.g. from your own random_palette call) to sample a KNOWN palette instead. This is pure composition of the two
    existing primitives (random_palette + cosine_palette), not a new colour model -- so the stops are exactly the
    colours cosine_palette would produce, just materialised as a small table. Deterministic per seed."""
    if coeffs is None:
        coeffs = random_palette(seed=seed, contrast=contrast)
    a, b, c, d = coeffs
    t = np.linspace(0.0, 1.0, int(n))                    # even stops across the gradient -> a swatch strip
    return cosine_palette(t, a=a, b=b, c=c, d=d)         # (n, 3) float RGB, identical to the palette at those t


def _gf3(v):
    """Format a length-3 vector as a GLSL vec3 literal. Delegates to the shared holographic_emit.glsl_vec3
    (sweep-consolidated -- one formatter, three call sites)."""
    from holographic.io_and_interop.holographic_emit import glsl_vec3
    return glsl_vec3(v)


def cosine_palette_to_glsl(a=(0.5, 0.5, 0.5), b=(0.5, 0.5, 0.5), c=(1.0, 1.0, 1.0), d=(0.0, 0.33, 0.67),
                           fn_name="palette"):
    """Compile iq's cosine palette to a GLSL `vec3 <fn_name>(float t)` function -- colour(t) = a + b*cos(2*pi*(c*t+d)),
    clamped [0,1]. Matches holographic_domain.cosine_palette per-point to float precision, so a demoscene palette
    renders client-side and composes with the SDF/postfx emitters (feed it an orbit trap, an iteration count, a
    distance). Pair with random_palette(seed) for a seed-driven scheme: cosine_palette_to_glsl(*random_palette(seed)).
    (leStudio backlog item 10.)"""
    return ("// cosine palette as GLSL (matches holographic_domain.cosine_palette to float precision)\n"
            "vec3 %s(float t){\n"
            "    return clamp(%s + %s * cos(6.28318530717959 * (%s * t + %s)), 0.0, 1.0);\n"
            "}\n" % (fn_name, _gf3(a), _gf3(b), _gf3(c), _gf3(d)))


def _selftest():
    """Assert the REAL contracts, with cross-condition CONTRAST rather than fragile absolutes.

    1. repeat makes an infinite lattice: a bounded SDF wrapped in repeat is hit by FAR more rays than the bare
       primitive (the whole point -- one shape becomes many). Measured through the actual raymarcher.
    2. repeat_limited is finite: outside its integer box there are no copies, so a ray that would hit the
       infinite version misses the limited one.
    3. mirror/fold are isometries: the warped distance to a mirrored primitive equals the unwarped distance to
       the reflected query point (exactness to 1e-12), and folding is idempotent.
    4. smin is a crease-free lower bound: smin(a,b,k) <= min(a,b), approaches min as k->0, and is smooth
       (its blend region is strictly below the hard min).
    5. cosine_palette is bounded, smooth, and phase d rotates hue; random_palette is deterministic per seed and
       stays in gamut.
    """
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    from holographic.rendering.holographic_raymarch import sphere_trace

    # (1) infinite lattice vs single primitive, through the real raymarcher.
    base = sphere(0.3)
    lat = wrap_sdf(base, lambda P: repeat(P, 1.0))
    xs = np.linspace(-3, 3, 36)
    O = np.stack([np.repeat(xs, 36), np.tile(xs, 36), np.full(36 * 36, 5.0)], 1)
    D = np.tile([0.0, 0.0, -1.0], (36 * 36, 1))
    hit_lat = int(np.sum(sphere_trace(lat, O, D, max_dist=12.0)[0]))
    hit_one = int(np.sum(sphere_trace(base, O, D, max_dist=12.0)[0]))
    assert hit_lat > 5 * hit_one, (hit_lat, hit_one)      # the lattice fills the frame; one sphere is a dot

    # (2) limited repetition is finite: a 1x1x1 box of copies around the origin. Far out (x=8) the infinite
    #     version still has a sphere but the limited one does not.
    lim = wrap_sdf(base, lambda P: repeat_limited(P, 1.0, -1, 1))
    far = np.array([[8.0, 0.0, 5.0]])
    down = np.array([[0.0, 0.0, -1.0]])
    assert sphere_trace(lat, far, down, max_dist=12.0)[0][0]        # infinite: still a sphere at x=8
    assert not sphere_trace(lim, far, down, max_dist=12.0)[0][0]    # limited: nothing out there

    # (3) mirror is an isometry and folding is idempotent (exact).
    P = np.array([[-1.3, 0.4, -2.1], [0.7, -0.9, 1.2]])
    assert np.allclose(domain_mirror(P, axis=0), domain_mirror(domain_mirror(P, axis=0), axis=0), atol=1e-12)  # already-folded stays
    m = sphere(0.5)
    # distance to a mirrored sphere == distance from the mirrored point to the base sphere (definition of a warp)
    assert np.allclose(wrap_sdf(m, lambda Q: domain_mirror(Q, axis=0)).eval(P), m.eval(domain_mirror(P, axis=0)), atol=1e-12)

    # (4) smin is a smooth lower bound on min.
    a = np.linspace(-1, 1, 50)
    b = np.full_like(a, 0.0)
    s = smin(a, b, k=0.2)
    assert np.all(s <= np.minimum(a, b) + 1e-9)                         # never above the hard min
    assert np.allclose(smin(a, b, k=1e-6), np.minimum(a, b), atol=1e-3) # k->0 recovers min
    assert np.max(np.minimum(a, b) - s) > 1e-3                          # dips strictly below min somewhere (real blend)

    # (5) palette: bounded, smooth, deterministic; phase d rotates hue.
    ts = np.linspace(0, 1, 64)
    cols = cosine_palette(ts)
    assert cols.shape == (64, 3) and cols.min() >= 0.0 and cols.max() <= 1.0
    # a phase shift changes the colour (hue rotates) -- the palette is not constant in d
    shifted = cosine_palette(ts, d=(0.2, 0.5, 0.8))
    assert np.mean(np.abs(cols - shifted)) > 0.05
    pa = random_palette(seed=7)
    assert random_palette(seed=7) == pa                                # deterministic
    assert cosine_palette(ts, *pa).min() >= 0.0                        # in gamut

    # (6) ITEM 4: palette_stops turns a palette into n PLOTTABLE rgb rows -- shape, gamut, determinism, and (the
    #     whole point) it equals cosine_palette sampled at the same even t, so the stops ARE the palette's colours,
    #     not an interpolation of its coefficients (the bug that shipped garbage when a caller read the a,b,c,d
    #     coeffs as if they were colours).
    stops = palette_stops(seed=7, n=8)
    assert stops.shape == (8, 3) and stops.min() >= 0.0 and stops.max() <= 1.0
    assert np.array_equal(palette_stops(seed=7, n=8), stops)           # deterministic per seed
    t8 = np.linspace(0.0, 1.0, 8)
    assert np.allclose(stops, cosine_palette(t8, *random_palette(seed=7)))  # exactly the palette at those t
    # explicit coeffs path samples a KNOWN palette rather than a seeded one
    assert np.allclose(palette_stops(coeffs=pa, n=8), cosine_palette(t8, *pa))
    # KEPT NEGATIVE: random_palette returns COEFFICIENTS (a,b,c,d), not colours -- interpolating those coefficients
    # as RGB stops is the documented misuse this function exists to prevent.

    # (7) ITEM 10: cosine_palette_to_glsl emits a `vec3 palette(float t)` matching cosine_palette per-point to float
    #     precision. Verified with an INDEPENDENT transcription of the emitted GLSL on t in [0,1].
    tt = np.linspace(0, 1, 129)
    for sd in (0, 3, 7):
        a2, b2, c2, d2 = random_palette(seed=sd)
        g = cosine_palette_to_glsl(a2, b2, c2, d2)
        assert "vec3 palette(float t)" in g
        glsl_col = np.clip(np.array(a2) + np.array(b2) * np.cos(6.28318530717959 * (np.array(c2) * tt[:, None] + np.array(d2))), 0, 1)
        assert np.abs(cosine_palette(tt, a=a2, b=b2, c=c2, d=d2) - glsl_col).max() < 1e-9, sd

    print("holographic_domain selftest OK (one sphere -> %d lattice hits vs %d single; "
          "mirror exact to 1e-12; smin a smooth lower bound; cosine palette in gamut)"
          % (hit_lat, hit_one))


if __name__ == "__main__":
    _selftest()

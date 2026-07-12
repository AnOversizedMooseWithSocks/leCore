"""holographic_pattern.py -- deterministic PROCEDURAL PATTERN FIELDS for material/parameter maps.

WHY THIS EXISTS
---------------
`holographic_param` gave every parameter a SOCKET: a channel can be a constant, a baked `map`, a wired `source`,
or a `field` -- a callable f(points (M,D)) -> (M,) values. `noise_field` gave ONE band-limited noise field. What
was missing is the small library of NAMED procedural patterns every material system exposes (Substance, Blender's
texture nodes, RenderMan patterns): checker, stripes, gradient, dots, value-noise, fbm. This module supplies them
as plain callables so they drop straight into a `Param(field=...)` and get resolved per point by `resolve_param` --
which means ANY faculty that reads a Param (the region field's material/reflect/roughness, an emitter's rate, a
material channel) can be driven by a procedural texture, not just a number.

DESIGN (kept honest and on-thesis)
----------------------------------
  * DETERMINISTIC. The value-noise uses an INTEGER lattice hash (pure arithmetic, PYTHONHASHSEED-independent) -- not
    Python's `hash()` and not `np.random` inside the field -- so the same point always yields the same value, run to
    run, exactly like the planet's fBm. Seedable via an integer that perturbs the hash.
  * Every generator returns a callable that maps world points -> a scalar field in [0, 1]. Compose them (a channel
    LERPs between lo..hi by the pattern) with `field_lerp`. NumPy / stdlib only; no learned weights.
  * These are FIELDS over space (3-D world position), so a pattern on a curved surface is a genuine solid texture --
    it wraps around the object without a UV unwrap, the field-native way.
"""
import numpy as np


# ---------------------------------------------------------------------------------------------------------------
# Deterministic value noise: an integer-lattice hash (arithmetic, seed-perturbed) trilinearly interpolated. Same
# family as the planet's fBm -- reproducible to the bit, independent of PYTHONHASHSEED.
# ---------------------------------------------------------------------------------------------------------------
def _hash01(ix, iy, iz, seed):
    """A deterministic hash of an integer lattice cell -> a float in [0,1). Pure int64 arithmetic (wraps), so it is
    reproducible run-to-run and does NOT use Python's salted hash()."""
    h = (ix.astype(np.int64) * np.int64(73856093)) ^ (iy.astype(np.int64) * np.int64(19349663)) \
        ^ (iz.astype(np.int64) * np.int64(83492791)) ^ (np.int64(seed) * np.int64(2654435761))
    h = (h ^ (h >> np.int64(13))) * np.int64(1274126177)
    h = h ^ (h >> np.int64(16))
    return (h & np.int64(0xFFFFFF)).astype(np.float64) / float(0x1000000)     # 24 bits -> [0,1)


def _smooth(t):
    """Quintic smoothstep (Perlin's) -- C2 interpolation weight, so the noise has no visible lattice creases."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise(scale=3.0, seed=0):
    """A callable f(P (M,3)) -> [0,1] value noise: hash the 8 lattice corners around each point and trilinearly blend
    with a quintic weight. `scale` sets the feature frequency (cells per unit)."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float)) * float(scale)
        i = np.floor(P).astype(np.int64); f = P - i
        w = _smooth(f)
        ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
        def corner(dx, dy, dz):
            return _hash01(ix + dx, iy + dy, iz + dz, seed)
        c000, c100 = corner(0, 0, 0), corner(1, 0, 0)
        c010, c110 = corner(0, 1, 0), corner(1, 1, 0)
        c001, c101 = corner(0, 0, 1), corner(1, 0, 1)
        c011, c111 = corner(0, 1, 1), corner(1, 1, 1)
        x00 = c000 * (1 - w[:, 0]) + c100 * w[:, 0]
        x10 = c010 * (1 - w[:, 0]) + c110 * w[:, 0]
        x01 = c001 * (1 - w[:, 0]) + c101 * w[:, 0]
        x11 = c011 * (1 - w[:, 0]) + c111 * w[:, 0]
        y0 = x00 * (1 - w[:, 1]) + x10 * w[:, 1]
        y1 = x01 * (1 - w[:, 1]) + x11 * w[:, 1]
        return y0 * (1 - w[:, 2]) + y1 * w[:, 2]
    return field


def fbm(scale=2.0, octaves=4, seed=0, gain=0.5, lacunarity=2.0):
    """Fractal (fBm) noise: sum octaves of value_noise at rising frequency and falling amplitude -- clouds, marble,
    weathering. Returns f(P) -> [0,1]."""
    bands = [value_noise(scale * lacunarity ** k, seed + 17 * k) for k in range(octaves)]
    amps = [gain ** k for k in range(octaves)]
    norm = sum(amps)
    def field(P):
        acc = np.zeros(len(np.atleast_2d(P)))
        for b, a in zip(bands, amps):
            acc = acc + a * b(P)
        return acc / norm
    return field


def domain_warped_fbm(scale=2.0, octaves=4, seed=0, warp=0.4, warp_scale=1.0, gain=0.5, lacunarity=2.0):
    """DOMAIN-WARPED fBm (iq's "warped noise" / dFBM): sample fbm at a point that has itself been DISPLACED by a
    vector of other fbm fields. f(P) = fbm(P + warp * [fbm_x(P), fbm_y(P), fbm_z(P)]). The displacement swirls the
    iso-contours into the flowing, turbulent, marbled look plain fbm cannot make -- smoke, magma, wood grain,
    weather fronts. `warp` is the displacement strength; `warp_scale` the frequency of the warp field. Returns
    f(P) -> [0,1]. This is the single most demoscene-recognisable noise (iq's "clouds" and "warping" articles).

    WHY three warp fields: each output axis needs its OWN noise or the displacement is diagonal (all axes move
    together) and the swirl collapses to a shear. Independent seeds give a true 3-D flow."""
    base = fbm(scale, octaves, seed, gain, lacunarity)
    # three INDEPENDENT warp fields (different seeds) so the displacement is a real 3-D vector, not a shear.
    wx = fbm(scale * warp_scale, max(1, octaves - 1), seed + 101, gain, lacunarity)
    wy = fbm(scale * warp_scale, max(1, octaves - 1), seed + 211, gain, lacunarity)
    wz = fbm(scale * warp_scale, max(1, octaves - 1), seed + 331, gain, lacunarity)

    def field(P):
        P = np.atleast_2d(np.asarray(P, float))
        # centre the warp fields to ~[-0.5, 0.5] so the displacement pushes both ways, not just positive.
        disp = np.stack([wx(P) - 0.5, wy(P) - 0.5, wz(P) - 0.5], axis=1)
        return base(P + warp * disp)
    return field


# ---------------------------------------------------------------------------------------------------------------
# Exact (non-noise) patterns: checker, stripes, gradient, dots. Deterministic by construction.
# ---------------------------------------------------------------------------------------------------------------
def checker(scale=2.0):
    """3-D checkerboard: parity of the summed floored coordinates -> {0,1}. A solid texture (wraps any surface)."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float)) * float(scale)
        s = np.floor(P[:, 0]) + np.floor(P[:, 1]) + np.floor(P[:, 2])
        return (np.mod(s, 2.0) < 1.0).astype(float)
    return field


def stripes(scale=3.0, axis=1, sharp=False):
    """Sinusoidal (or hard) stripes along an axis, in [0,1]. `sharp=True` -> square-wave bands."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float))
        v = 0.5 + 0.5 * np.sin(P[:, int(axis)] * float(scale) * 2.0 * np.pi)
        return (v > 0.5).astype(float) if sharp else v
    return field


def gradient(axis=1, lo=-1.0, hi=1.0):
    """A linear ramp along an axis, normalised to [0,1] over [lo,hi] -- a height/position gradient."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float))
        return np.clip((P[:, int(axis)] - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return field


def dots(scale=3.0, radius=0.3):
    """Round dots on a lattice: 1 near each cell centre, 0 between -> polka / rivets. In [0,1]."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float)) * float(scale)
        d = np.abs(P - np.round(P))
        r = np.linalg.norm(d, axis=1)
        return np.clip(1.0 - r / max(radius, 1e-3), 0.0, 1.0)
    return field


# named registry so a UI / serialized spec can ask for a pattern by name
PATTERNS = {
    "noise": value_noise, "fbm": fbm, "checker": checker,
    "stripes": stripes, "gradient": gradient, "dots": dots,
}


def make_pattern(name, **params):
    """Build a pattern FIELD by name (for a serialized map spec). Unknown name -> a constant 0.5 field."""
    fn = PATTERNS.get(name)
    if fn is None:
        return lambda P: np.full(len(np.atleast_2d(P)), 0.5)
    return fn(**params)


def field_lerp(pattern, lo, hi):
    """Wrap a [0,1] pattern into a channel field that LERPs between `lo` and `hi` (scalars OR (3,) colours). This is
    how a pattern DRIVES a material channel: roughness lo..hi, or colour A..B, by the pattern value."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    def field(P):
        w = np.asarray(pattern(P), float)
        if lo.ndim == 0:
            return lo + (hi - lo) * w
        return lo[None, :] + (hi - lo)[None, :] * w[:, None]        # colour / vector channel
    return field


def _selftest():
    """Determinism + range: every pattern is in [0,1], and the noise is identical across two evaluations and two
    process-independent calls (integer hash, not salted hash())."""
    P = np.random.default_rng(0).uniform(-2, 2, (2000, 3))
    for name in PATTERNS:
        f = make_pattern(name)
        v = np.asarray(f(P), float)
        assert v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9, (name, v.min(), v.max())
    n = value_noise(3.0, seed=7)
    assert np.allclose(n(P), n(P))                                   # deterministic
    # a channel driven lo..hi by a pattern stays within [lo,hi]
    g = field_lerp(make_pattern("checker", scale=2.0), 0.1, 0.9)
    gv = g(P); assert gv.min() >= 0.1 - 1e-9 and gv.max() <= 0.9 + 1e-9

    # W11 dFBM: domain-warped fbm stays in [0,1], DIFFERS from plain fbm (the warp swirled it), reduces EXACTLY to
    # fbm at warp=0 (clean degradation), and is deterministic.
    plain = fbm(scale=2.0, seed=0)(P)
    warped = domain_warped_fbm(scale=2.0, seed=0, warp=0.5)(P)
    assert warped.min() >= -1e-9 and warped.max() <= 1.0 + 1e-9
    assert np.abs(warped - plain).mean() > 0.01                       # the warp actually changed the field
    assert np.allclose(domain_warped_fbm(scale=2.0, seed=0, warp=0.0)(P), plain, atol=1e-9)   # warp=0 == fbm
    assert np.allclose(domain_warped_fbm(scale=2.0, seed=0)(P), domain_warped_fbm(scale=2.0, seed=0)(P))
    print("holographic_pattern selftest OK (dFBM in [0,1], swirls vs fbm, warp=0 reduces to fbm, deterministic)")


if __name__ == "__main__":
    _selftest()

"""holographic_grainmat.py -- M1: procedural GRAIN and STACKED-SUBSTRATE material sockets (wood, plywood, sediment).

WHY THIS MODULE EXISTS (Material Structure backlog, item M1)
-----------------------------------------------------------
The engine could already paint a "rust look" and blend two colours by a noise field, but it had no dedicated
*grain* primitive: real wood is concentric growth RINGS around an axis, plus FIBRE streaks running along that
axis, plus TURBULENCE that makes knots and wavy figure. This module supplies that as a socket -- a callable
f(points (M,3)) -> value -- exactly what a `holographic_param.Param(field=...)` / SurfaceMaterial channel accepts,
so grain drops straight onto a material's colour or roughness and is resolved per hit by render_surface.

The key on-thesis property: grain is evaluated in OBJECT SPACE from the 3-D position, so it is VOLUMETRIC -- cut
the board and the rings continue through the cut, because the field is defined everywhere, not painted on a UV
sheet. Same "socket into a region" the planet biomes use, one scale down (as above, so below).

DESIGN (deterministic, readable, NumPy-only)
--------------------------------------------
  * Rings: distance from the grain AXIS, banded by a triangle wave -> concentric year-rings. A fBm DOMAIN WARP
    on that distance bends the rings into knots and natural wander (the standard procedural-wood trick,
    Perlin 1985 / domain-warped noise), so straight rings become believable figure.
  * Fibre: noise sampled with the along-axis coordinate STRETCHED, so features smear into long streaks parallel
    to the axis -- the fine lengthwise fibre over the coarse rings.
  * Determinism: reuses holographic_pattern.value_noise / fbm, whose integer-lattice hash is
    PYTHONHASHSEED-independent, so the same board looks the same run to run.

HONEST SCOPE (kept negative): this is APPEARANCE grain -- believable and art-directable -- NOT a xylem /
cell-growth simulation. Ring spacing, warp and fibre are tuning parameters, not a biological growth model.
"""
import numpy as np
from holographic.misc.holographic_pattern import value_noise, fbm


def _axis_frame(axis):
    """Return (unit axis a, two unit vectors u,v spanning the plane perpendicular to it). Rings live in that
    perpendicular plane; the axis is the direction the trunk/grain runs."""
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    # pick any vector not parallel to a, then Gram-Schmidt out the axis component -> u; v = a x u
    seed_vec = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed_vec - a * float(seed_vec @ a); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(a, u)
    return a, u, v


def _tri(x):
    """A triangle wave in [0,1] with period 1 -- sharper, more ring-like bands than a sine (year-rings have a
    fast latewood transition), and cheap."""
    f = x - np.floor(x)
    return 1.0 - np.abs(2.0 * f - 1.0)


def wood_grain(axis=(0, 1, 0), ring_scale=8.0, fibre=0.35, warp=0.6, seed=0, center=(0.0, 0.0, 0.0)):
    """A wood-grain SCALAR socket f(points (M,3)) -> [0,1] (0 = earlywood/light, 1 = latewood/dark), volumetric in
    object space. `ring_scale` = rings per unit radius; `fibre` = strength of the lengthwise streaks; `warp` =
    fBm domain-warp that bends rings into knots/figure; `center` = a point the axis passes through.

    Drive a colour channel with holographic_pattern.field_lerp(this, light_rgb, dark_rgb), or a roughness channel
    lo..hi (latewood reads slightly rougher). Deterministic."""
    a, u, v = _axis_frame(axis)
    c = np.asarray(center, float)
    warp_fld = fbm(scale=2.0, octaves=3, seed=seed + 11)          # low-freq wander that bends the rings
    fibre_fld = value_noise(scale=ring_scale * 2.0, seed=seed + 23)  # fine noise, stretched into streaks below

    def _socket(points):
        P = np.atleast_2d(np.asarray(points, float)) - c
        # decompose each point into (distance from axis, height along axis)
        along = P @ a                                             # signed distance along the grain axis
        pu = P @ u; pv = P @ v
        radius = np.sqrt(pu * pu + pv * pv)                       # distance from the axis line -> the ring radius
        # DOMAIN WARP: perturb the radius by low-freq fBm so rings wander and knot instead of being perfect circles
        w = (warp_fld(P + c) - 0.5) * 2.0 * warp                  # [-warp, warp]
        rings = _tri((radius + w) * ring_scale)                  # concentric banded rings
        # FIBRE: sample noise with the ALONG-axis coordinate compressed -> features smear into lengthwise streaks
        stretch = np.stack([pu, pv, along * 0.15], axis=1)       # squash the axis coord -> long streaks
        streak = fibre_fld(stretch)
        g = np.clip((1.0 - fibre) * rings + fibre * streak, 0.0, 1.0)
        return g
    return _socket


def wood_albedo(axis=(0, 1, 0), light=(0.72, 0.52, 0.32), dark=(0.40, 0.26, 0.14),
                ring_scale=8.0, fibre=0.35, warp=0.6, seed=0, center=(0.0, 0.0, 0.0)):
    """A wood COLOUR socket f(points)->(M,3) rgb: the grain scalar lerped between earlywood `light` and latewood
    `dark`. Drops straight into SurfaceMaterial(color=Param(field=wood_albedo(...)))."""
    g = wood_grain(axis, ring_scale, fibre, warp, seed, center)
    lo = np.asarray(light, float); hi = np.asarray(dark, float)

    def _socket(points):
        t = g(points)[:, None]
        return (1.0 - t) * lo + t * hi
    return _socket


def substrate_layers(axis=(0, 1, 0), layers=None, center=(0.0, 0.0, 0.0), blend=0.02):
    """A STACKED-SUBSTRATE colour socket: bands stacked ALONG an axis, each a (thickness, colour-or-rgb) -- plywood
    plies, sedimentary strata, laminated stock. `blend` softens the seam between bands. Returns f(points)->(M,3).

    The band a point falls in is decided by its along-axis coordinate; because that is a field over 3-D space, a
    cut through the stack shows the strata continuing (volumetric, like the wood rings)."""
    a, _, _ = _axis_frame(axis)
    c = np.asarray(center, float)
    layers = layers or [(0.3, (0.72, 0.52, 0.32)), (0.05, (0.30, 0.20, 0.12)),
                        (0.3, (0.66, 0.46, 0.28)), (0.05, (0.30, 0.20, 0.12))]
    cols = [np.asarray(rgb, float) for _, rgb in layers]
    thick = np.array([t for t, _ in layers], float)
    edges = np.concatenate([[0.0], np.cumsum(thick)])
    period = float(edges[-1])                                    # the stack repeats with this period

    def _socket(points):
        P = np.atleast_2d(np.asarray(points, float)) - c
        s = np.mod(P @ a, period)                                # position within one repeat of the stack
        out = np.zeros((len(P), 3))
        for i in range(len(cols)):
            lo, hi = edges[i], edges[i + 1]
            # soft membership in band i: 1 inside [lo,hi], ramped over `blend` at each seam
            m = np.clip((s - lo) / blend, 0, 1) * np.clip((hi - s) / blend, 0, 1)
            out += m[:, None] * cols[i]
        # normalise the (softly overlapping) memberships so seams blend instead of darkening
        wsum = np.zeros(len(P))
        for i in range(len(cols)):
            lo, hi = edges[i], edges[i + 1]
            wsum += np.clip((s - lo) / blend, 0, 1) * np.clip((hi - s) / blend, 0, 1)
        return out / (wsum[:, None] + 1e-9)
    return _socket


def _selftest():
    """Grain is deterministic, in range, VOLUMETRIC (rings continue across a cut), and axis-aligned; substrate
    bands land in the right colours."""
    rng = np.random.default_rng(0)
    P = rng.uniform(-1, 1, (2000, 3))
    g = wood_grain(axis=(0, 1, 0), ring_scale=8.0, seed=1)
    a = g(P); b = g(P)
    assert np.array_equal(a, b)                                   # deterministic (integer-hash noise)
    assert a.min() >= -1e-9 and a.max() <= 1.0 + 1e-9            # in range

    # VOLUMETRIC: the grain depends only on distance-from-axis and along-axis position, so two points that differ
    # ONLY in the axis coordinate by a whole ring-period share the same radius -> same ring value. Concretely,
    # slide a set of points along the axis and the RING component (fibre=0) is unchanged.
    gp = wood_grain(axis=(0, 1, 0), ring_scale=8.0, fibre=0.0, warp=0.0, seed=1)
    base = np.array([[0.3, 0.0, 0.2], [0.6, 0.0, -0.1], [0.1, 0.0, 0.5]])
    shifted = base + np.array([0.0, 1.7, 0.0])                    # move only along the axis
    assert np.allclose(gp(base), gp(shifted))                    # rings continue along the axis (volumetric)

    # a colour socket returns rgb in range
    col = wood_albedo(seed=2)(P)
    assert col.shape == (2000, 3) and col.min() >= -1e-9 and col.max() <= 1.0 + 1e-9

    # substrate: a point deep inside band 0 reads band-0 colour
    lay = substrate_layers(axis=(0, 1, 0), layers=[(0.4, (0.9, 0.1, 0.1)), (0.4, (0.1, 0.1, 0.9))], blend=0.02)
    c0 = lay(np.array([[0.0, 0.2, 0.0]]))[0]                      # s=0.2, inside band 0
    assert np.allclose(c0, (0.9, 0.1, 0.1), atol=1e-3)
    print("holographic_grainmat selftest OK: deterministic, in-range, VOLUMETRIC rings (continue along the axis), "
          "colour + substrate sockets land correctly")


if __name__ == "__main__":
    _selftest()

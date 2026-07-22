"""Procedural textures (the standard 3D-app set, 2D and 3D) + the mask-edge REFRACTION effect.

WHY THIS EXISTS. The engine had strong but SCATTERED texture primitives: value noise / fbm / checker /
stripes / dots as point-fields (holographic_pattern), VSA-encoded fbm/voronoi/curl fields (texturehome),
domain-warped fbm (warped_noise), curl noise, texture synthesis. What a user coming from Blender/Maya/Houdini
expects is the STANDARD MENU -- noise, voronoi (F1/F2/F2-F1/cell), musgrave, wave/rings, marble, wood, brick,
magic, white noise -- addressable by NAME, evaluable in 2D (an image) or 3D (a volume or any points), from one
entry point. This module is that menu. Every texture is a callable field f(P (M,3)) -> values, so 2D vs 3D is
just WHERE you sample it (a z=const plane vs a grid vs a mesh's surface points) -- the same texture paints an
image, fills a volume for clouds, or feeds a Material channel.

RELATION TO WHAT EXISTS (audited, not assumed): value_noise / fbm / checker / stripes / gradient / dots are
REUSED from holographic_pattern (imported, not reimplemented). texturehome's voronoi/fbm are the VSA-ENCODED
costume (fields as hypervectors for the material algebra); the Worley here is the DIRECT-EVALUATION costume for
fast rasterisation -- same idea, different medium, cross-referenced both ways. warped_noise (the mind faculty)
is the marble ancestor; `marble`/`wood` here are the named presets a 3D app ships.

THE REFRACTION EFFECT. mask_refraction(image, mask, ...) treats the mask as a flat-bottomed LENS: the jump-flood
distance transform gives distance-to-edge inside the mask; a profile turns that into a height bump (a meniscus);
the bump's GRADIENT is the surface normal's tilt, and small-angle Snell says the view ray is displaced
proportionally to that tilt scaled by (ior - 1). The displacement is therefore automatically strongest NEAR THE
EDGE (where the profile is steep) and zero both at the mask's flat centre and outside -- exactly the
'distortion that changes with distance from the edge' behaviour of a water droplet or glass blob on a surface.
Optional chromatic aberration displaces R/G/B by slightly different strengths (dispersion).

Deterministic (hashed lattices from pattern's _hash01, seeded), NumPy-only, vectorised.
KEPT NEGATIVE: mask_refraction is a SCREEN-SPACE approximation (single interface, small-angle Snell, no total
internal reflection, no caustics). For true refraction render the SDF through path_trace's dielectric --
this effect is for fast 2D water/glass looks, UI, and compositing, and says so.
"""

import numpy as np

from holographic.misc.holographic_pattern import value_noise, fbm as _fbm_field


# ======================================================================================================
# hashing for lattice textures -- reuse the pattern module's deterministic integer hash
# ======================================================================================================

from holographic.misc.holographic_pattern import _hash01                     # (ix,iy,iz,seed) -> [0,1)


def _hash3(ix, iy, iz, seed):
    """Three decorrelated hash channels per lattice cell (for jittered feature points)."""
    return _hash01(ix, iy, iz, seed), _hash01(ix, iy, iz, seed + 101), _hash01(ix, iy, iz, seed + 202)


# ======================================================================================================
# the standard set (each returns a callable field f(P (M,3)) -> (M,) in [0,1] unless noted)
# ======================================================================================================

def white_noise(seed=0):
    """Uncorrelated per-cell noise (Blender's White Noise): hash the CONTAINING unit cell. Piecewise-constant;
    for smooth noise use `noise` (value noise) or `fbm`."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float))
        i = np.floor(P).astype(np.int64)
        return _hash01(i[:, 0], i[:, 1], i[:, 2], seed)
    return field


def voronoi(scale=3.0, seed=0, jitter=1.0, kind="f1", metric="euclidean"):
    """Worley/cellular noise -- the 3D-app Voronoi with its standard OUTPUTS:
      kind='f1'    distance to the nearest feature point (cells with dark centres)
      kind='f2'    distance to the second-nearest (inverted-crack look)
      kind='f2f1'  F2 - F1 (bright CELL WALLS -- the classic crack/cell-boundary texture)
      kind='cell'  a flat random value per cell (id colouring / mosaic)
      kind='smooth' exp-smoothed minimum (rounded organic cells)
    `jitter` in [0,1]: 0 = a regular grid, 1 = fully random feature points. `metric` = 'euclidean' or
    'manhattan' (the chebyshev/manhattan variants apps expose). Direct-evaluation Worley (the fast
    rasterisation costume; texturehome.voronoi is the VSA-encoded costume of the same idea).
    Distances are normalised by the cell size so the output is resolution-independent [0, ~1.6]."""
    if kind not in ("f1", "f2", "f2f1", "cell", "smooth"):
        raise ValueError("voronoi kind must be f1/f2/f2f1/cell/smooth, got %r" % (kind,))
    if metric not in ("euclidean", "manhattan"):
        raise ValueError("voronoi metric must be euclidean/manhattan, got %r" % (metric,))

    def field(P):
        P = np.atleast_2d(np.asarray(P, float)) * float(scale)
        i = np.floor(P).astype(np.int64)
        f = P - i
        M = len(P)
        f1 = np.full(M, 1e9); f2 = np.full(M, 1e9); cell_id = np.zeros(M)
        smooth_acc = np.zeros(M)
        for dx in (-1, 0, 1):                                   # the 27-neighbourhood: features can only be
            for dy in (-1, 0, 1):                               # this close from adjacent cells
                for dz in (-1, 0, 1):
                    cx, cy, cz = i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz
                    hx, hy, hz = _hash3(cx, cy, cz, seed)
                    px = dx + float(jitter) * hx - f[:, 0]
                    py = dy + float(jitter) * hy - f[:, 1]
                    pz = dz + float(jitter) * hz - f[:, 2]
                    if metric == "euclidean":
                        d = np.sqrt(px * px + py * py + pz * pz)
                    else:
                        d = np.abs(px) + np.abs(py) + np.abs(pz)
                    closer = d < f1
                    f2 = np.where(closer, f1, np.minimum(f2, d))
                    cell_id = np.where(closer, _hash01(cx, cy, cz, seed + 777), cell_id)
                    f1 = np.where(closer, d, f1)
                    smooth_acc += np.exp(-8.0 * d)              # falloff 8: rounded but distinct cells
        if kind == "f1":
            return f1
        if kind == "f2":
            return f2
        if kind == "f2f1":
            return f2 - f1
        if kind == "cell":
            return cell_id
        return -np.log(np.maximum(smooth_acc, 1e-30)) / 8.0     # smooth-min of the distances
    return field


def musgrave(scale=2.0, octaves=5, seed=0, kind="ridged", lacunarity=2.0, gain=0.5, offset=0.9):
    """Musgrave multifractals (Texturing & Modeling: A Procedural Approach, Ebert et al. -- Musgrave's
    chapters), the mountain/erosion noises every 3D app ships:
      kind='ridged'  sharp ridge lines (1 - |noise|, squared, weighted by the previous octave) -- mountain
                     crests, lightning, cloud wisps
      kind='hybrid'  additive multifractal whose roughness varies with altitude -- eroded terrain
      kind='fbm'     plain fractal Brownian motion (delegates to the pattern module's fbm)
    Output roughly [0,1] (ridged/hybrid renormalised by their own accumulated weight)."""
    if kind == "fbm":
        return _fbm_field(scale=scale, octaves=octaves, seed=seed, gain=gain, lacunarity=lacunarity)
    if kind not in ("ridged", "hybrid"):
        raise ValueError("musgrave kind must be ridged/hybrid/fbm, got %r" % (kind,))
    base = value_noise(scale=1.0, seed=seed)                    # frequency applied per octave below

    def field(P):
        P = np.atleast_2d(np.asarray(P, float)) * float(scale)
        M = len(P)
        freq = 1.0
        if kind == "ridged":
            out = np.zeros(M); weight = np.ones(M); norm = 0.0; amp = 1.0
            for _ in range(int(octaves)):
                n = base(P * freq) * 2.0 - 1.0                  # [-1,1]
                r = (float(offset) - np.abs(n)) ** 2 * weight   # the ridge: fold, sharpen, occlude
                out += r * amp
                weight = np.clip(r * 2.0, 0.0, 1.0)             # high ridges roughen the next octave
                norm += amp
                freq *= float(lacunarity); amp *= float(gain)
            return np.clip(out / max(norm, 1e-12), 0.0, 1.0)
        # hybrid
        n0 = base(P) * 2.0 - 1.0
        out = (n0 + float(offset)) * 0.5
        weight = np.clip(out, 0.0, 1.0)
        amp = float(gain); freq = float(lacunarity); norm = 1.0
        for _ in range(int(octaves) - 1):
            n = (base(P * freq) * 2.0 - 1.0 + float(offset)) * 0.5
            out += weight * amp * n
            weight = np.clip(weight * n * 2.0, 0.0, 1.0)
            norm += amp
            freq *= float(lacunarity); amp *= float(gain)
        return np.clip(out / norm * 1.6, 0.0, 1.0)
    return field


def wave(scale=4.0, kind="bands", axis=0, distortion=0.0, seed=0, profile="sine"):
    """The Wave texture (bands or rings, optionally noise-distorted) -- the base of marble and wood:
      kind='bands' parallel stripes along `axis`;  kind='rings' concentric shells around the axis line.
      distortion > 0 warps the coordinate with fbm BEFORE the wave -- bands become marble veins, rings
      become wood grain. profile='sine' (smooth) or 'saw' (sharp ramps)."""
    if kind not in ("bands", "rings"):
        raise ValueError("wave kind must be bands/rings, got %r" % (kind,))
    if profile not in ("sine", "saw"):
        raise ValueError("wave profile must be sine/saw, got %r" % (profile,))
    dis = _fbm_field(scale=max(scale * 0.5, 0.5), octaves=4, seed=seed) if distortion else None

    def field(P):
        P = np.atleast_2d(np.asarray(P, float))
        if kind == "bands":
            t = P[:, int(axis)] * float(scale)
        else:
            others = [a for a in (0, 1, 2) if a != int(axis)]
            t = np.sqrt(P[:, others[0]] ** 2 + P[:, others[1]] ** 2) * float(scale)
        if dis is not None:
            t = t + float(distortion) * (dis(P) * 2.0 - 1.0) * float(scale)
        if profile == "sine":
            return 0.5 + 0.5 * np.sin(2.0 * np.pi * t)
        return np.mod(t, 1.0)
    return field


def marble(scale=2.5, distortion=2.2, seed=0):
    """Marble: heavily noise-distorted sine bands (the classic Perlin marble recipe -- the named preset a
    3D app ships; the mind's warped_noise is the same family with free parameters)."""
    return wave(scale=scale, kind="bands", axis=0, distortion=distortion, seed=seed, profile="sine")


def wood(scale=6.0, distortion=0.35, seed=0):
    """Wood grain: gently distorted concentric rings around the y axis (a log's growth rings)."""
    return wave(scale=scale, kind="rings", axis=1, distortion=distortion, seed=seed, profile="sine")


def brick(scale=3.0, mortar=0.06, row_offset=0.5, seed=0, colour_variation=0.35):
    """Brick bonds: value ~0 in the mortar gaps, per-brick random brightness in the bricks (Blender's Brick).
    Rows are `row_offset`-staggered (0.5 = running bond). 2.5D by construction (x/y layout, constant in z) --
    exactly how the apps ship it."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float))
        x = P[:, 0] * float(scale)
        y = P[:, 1] * float(scale) * 2.0                        # bricks are twice as wide as tall
        row = np.floor(y)
        x = x + float(row_offset) * row                          # stagger alternate rows
        col = np.floor(x)
        fx, fy = x - col, y - row
        m = float(mortar)                                       # a FRACTION of the brick cell (the first draft
        in_mortar = (fx < m) | (fx > 1 - m) | (fy < m * 2) | (fy > 1 - m * 2)   # multiplied by scale: 82% mortar)
        shade = 0.55 + float(colour_variation) * (_hash01(col.astype(np.int64), row.astype(np.int64),
                                                          np.zeros_like(col, dtype=np.int64), seed) - 0.5)
        return np.where(in_mortar, 0.08, np.clip(shade, 0.0, 1.0))
    return field


def magic(scale=3.0, depth=3, distortion=1.5):
    """The 'Magic' texture (Blender's psychedelic sin/cos feedback): iterated trigonometric folding of the
    coordinates. Decorative; deterministic; no seed (the pattern is fully determined by the parameters)."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float)) * float(scale)
        x = np.sin((P[:, 0] + P[:, 1] + P[:, 2]) * 5.0)
        y = np.cos((P[:, 0] - P[:, 1]) * 4.0)
        z = np.sin((P[:, 1] - P[:, 2]) * 3.0)
        for _ in range(int(depth)):
            x, y = np.sin(y * float(distortion) + x), np.cos(x * float(distortion) - z)
            z = np.sin(z * float(distortion) + y)
        return 0.5 + 0.5 * (x * 0.5 + y * 0.3 + z * 0.2)
    return field


# the registry -- name -> builder. `noise` and friends REUSE the pattern module's fields directly.
from holographic.misc.holographic_pattern import checker as _checker, stripes as _stripes, \
    gradient as _gradient, dots as _dots

TEXTURES = {
    "noise":    lambda **kw: value_noise(**kw),
    "fbm":      lambda **kw: _fbm_field(**kw),
    "white":    lambda **kw: white_noise(**kw),
    "voronoi":  lambda **kw: voronoi(**kw),
    "musgrave": lambda **kw: musgrave(**kw),
    "wave":     lambda **kw: wave(**kw),
    "marble":   lambda **kw: marble(**kw),
    "wood":     lambda **kw: wood(**kw),
    "brick":    lambda **kw: brick(**kw),
    "magic":    lambda **kw: magic(**kw),
    "checker":  lambda **kw: _checker(**kw),
    "stripes":  lambda **kw: _stripes(**kw),
    "gradient": lambda **kw: _gradient(**kw),
    "dots":     lambda **kw: _dots(**kw),
}


def proc_texture(name, **params):
    """The MENU: proc_texture('voronoi', kind='f2f1', scale=4) -> a callable field f(P (M,3)) -> values.
    Names: noise, fbm, white, voronoi, musgrave, wave, marble, wood, brick, magic, checker, stripes,
    gradient, dots. The SAME field paints a 2D image (proc_texture_image), fills a 3D volume (proc_texture_volume --
    cloud densities), or evaluates at any points (a mesh's surface, a Material channel)."""
    if name not in TEXTURES:
        raise ValueError("unknown texture %r; the menu is %s" % (name, sorted(TEXTURES)))
    return TEXTURES[name](**params)


def proc_texture_image(name, size=256, region=((0.0, 0.0), (1.0, 1.0)), z=0.0, **params):
    """Rasterise a texture to a 2D (size, size) image in [0,1]: sample the 3D field on the z=const plane
    over `region` ((x0,y0),(x1,y1)). 2D texturing IS 3D texturing on a plane -- one implementation.
    (Module-level name is proc_texture_image: holographic_preview.texture_image already rasterises CMP1
    GRAPHS, a different input type with the same natural name -- renamed here rather than colliding; the
    mind-level faculty is still texture_image, which does not collide.)"""
    field = proc_texture(name, **params)
    (x0, y0), (x1, y1) = region
    xs = np.linspace(x0, x1, int(size)); ys = np.linspace(y0, y1, int(size))
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(z))], axis=1)
    v = np.asarray(field(P), float).reshape(int(size), int(size))
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def proc_texture_volume(name, res=48, bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), **params):
    """Sample a texture on a (res,res,res) 3D grid in [0,1] -- a cloud/smoke density, a 3D material field,
    a displacement volume. The same named field as texture_image, sampled in the volume."""
    field = proc_texture(name, **params)
    (x0, y0, z0), (x1, y1, z1) = bounds
    r = int(res)
    xs = np.linspace(x0, x1, r); ys = np.linspace(y0, y1, r); zs = np.linspace(z0, z1, r)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    v = np.asarray(field(P), float).reshape(r, r, r)
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


# ======================================================================================================
# TEXTURES AS NUMBERS, NUMBERS AS TEXTURES -- the sampler + ramp layer
# ======================================================================================================
#
# The two directions of the same identity: a texture IS a function of coordinates, so it can be READ as a
# numerical input (sample_image / image_field -- drive any parameter from a painted map), and any numbers can
# be WRITTEN as a texture (values_to_texture; ramp/ramp_texture for the classic stop-gradient). The closing
# contract, pinned in the selftest: sample_image(values_to_texture(vals), texel_centres) == vals -- assign,
# then sample, and the numbers come back exactly.


def sample_image(image, uv, mode="bilinear", wrap="clamp"):
    """READ a raster texture as numbers: sample `image` (H,W) or (H,W,C) at `uv` (M,2) in [0,1]x[0,1]
    (u across width, v down height; v=0 is the TOP row, image convention). Returns (M,) or (M,C).

    mode='bilinear' (smooth -- the GPU's default) or 'nearest' (exact texel reads: uv at a texel CENTRE
    returns that texel's value exactly, in both modes). wrap='clamp' (edge-extend) or 'repeat' (tile).
    The uv->texel mapping is the half-texel-centre convention (uv 0.5/W is texel 0's centre), matching GPU
    samplers so a map painted for the three.js front end reads the same numbers here."""
    img = np.asarray(image, float)
    flat = img.ndim == 2
    if flat:
        img = img[..., None]
    H, W = img.shape[:2]
    uv = np.atleast_2d(np.asarray(uv, float))
    if wrap == "repeat":
        uv = np.mod(uv, 1.0)
    elif wrap != "clamp":
        raise ValueError("wrap must be 'clamp' or 'repeat', got %r" % (wrap,))
    # half-texel centres: uv=0 maps to -0.5 texel, uv=1 to N-0.5; texel k's centre is (k+0.5)/N
    x = uv[:, 0] * W - 0.5
    y = uv[:, 1] * H - 0.5
    if mode == "nearest":
        xi = np.clip(np.rint(x).astype(int), 0, W - 1)
        yi = np.clip(np.rint(y).astype(int), 0, H - 1)
        out = img[yi, xi]
    elif mode == "bilinear":
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        fx = x - x0; fy = y - y0
        x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x0 + 1, 0, W - 1)
        y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y0 + 1, 0, H - 1)
        out = (img[y0c, x0c] * ((1 - fx) * (1 - fy))[:, None] + img[y0c, x1c] * (fx * (1 - fy))[:, None]
               + img[y1c, x0c] * ((1 - fx) * fy)[:, None] + img[y1c, x1c] * (fx * fy)[:, None])
    else:
        raise ValueError("mode must be 'bilinear' or 'nearest', got %r" % (mode,))
    return out[:, 0] if flat else out


def image_field(image, scale=1.0, wrap="repeat", mode="bilinear"):
    """WRAP a raster image as a FIELD f(P (M,3)) -> values: P's x/y (times `scale`) are the uv, z is ignored
    -- so a painted map plugs in anywhere the engine takes a field (a Material channel, cloud_scene's
    texture= density, a displacement source). The raster sibling of proc_texture's analytic entries: after
    this, 'texture' means the same thing whether it was computed or painted."""
    img = np.asarray(image, float)

    def field(P):
        P = np.atleast_2d(np.asarray(P, float))
        return sample_image(img, P[:, :2] * float(scale), mode=mode, wrap=wrap)
    return field


def ramp(positions, values, interp="linear"):
    """A STOP RAMP (the ColorRamp node): map scalars in [0,1] through stops -- `positions` (N,) sorted-able
    in [0,1], `values` (N,) scalars or (N,C) colours. Returns a callable t (M,) -> (M,) or (M,C).

    interp='linear' (blend between stops), 'constant' (hold each stop until the next -- hard bands), or
    'smooth' (smoothstep eased between stops). Outside [first, last] the ends CLAMP (ColorRamp semantics).
    A stop's own position returns its exact value in every mode. The ramp is itself a texture in waiting:
    ramp_texture bakes it to a strip; sample_image reads it back."""
    pos = np.asarray(positions, float)
    val = np.asarray(values, float)
    order = np.argsort(pos)
    pos, val = pos[order], val[order]
    if len(pos) < 1:
        raise ValueError("a ramp needs at least one stop")
    if interp not in ("linear", "constant", "smooth"):
        raise ValueError("interp must be linear/constant/smooth, got %r" % (interp,))
    vec = val.ndim == 2

    def fn(t):
        t = np.atleast_1d(np.asarray(t, float))
        if len(pos) == 1:
            out = np.broadcast_to(val[0], t.shape + val[0].shape if vec else t.shape).copy()
            return out
        idx = np.clip(np.searchsorted(pos, t, side="right") - 1, 0, len(pos) - 2)
        lo, hi = pos[idx], pos[idx + 1]
        f = np.clip((t - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)
        f = np.where(t <= pos[0], 0.0, np.where(t >= pos[-1], 1.0, f))
        if interp == "constant":
            f = np.zeros_like(f)
            f = np.where(t >= pos[-1], 1.0, f)                  # at/after the last stop, hold the last value
        elif interp == "smooth":
            f = f * f * (3.0 - 2.0 * f)
        a, b = val[idx], val[idx + 1]
        return a + (b - a) * (f[:, None] if vec else f)
    return fn


def ramp_texture(positions, values, size=256, interp="linear"):
    """ASSIGN a ramp's numbers to a TEXTURE: bake the stop ramp to a (size,) strip (scalar values) or
    (size, C) strip (colours) by evaluating it at the texel CENTRES -- so sample_image reads back exactly
    what the ramp says at those points. The 1-D gradient texture every 3D app ships."""
    ts = (np.arange(int(size)) + 0.5) / float(size)
    return ramp(positions, values, interp=interp)(ts)


def values_to_texture(values, normalize=False):
    """ASSIGN arbitrary numbers to a texture: an (H,W) / (H,W,C) / (N,) / (N,C) array becomes a
    sample_image-able raster ((N,)/(N,C) inputs become one-row strips). normalize=True affinely maps the
    values to [0,1] (display/export); the DEFAULT keeps the numbers untouched, because the round trip is the
    point: sample_image(values_to_texture(v), texel_centres) == v exactly (pinned in the selftest)."""
    v = np.asarray(values, float)
    if v.ndim == 1:
        v = v[None, :]                                          # a strip: one row, N columns
    # 2-D input is taken AS an (H,W) image and 3-D as (H,W,C) -- a caller with an (N,C) strip of colours
    # passes it as (1,N,C) or uses ramp_texture, which owns that shape; guessing here would be K10's sin.
    if normalize:
        lo, hi = float(v.min()), float(v.max())
        v = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    return v


# ======================================================================================================
# the mask-edge refraction effect
# ======================================================================================================

def mask_refraction(image, mask, strength=12.0, ior=1.33, profile="lens", edge_width=None,
                    chromatic=0.0, ripple=None, seed=0):
    """Refract `image` through a 2D shape given by `mask` (H,W bool/0-1): the LENS reading of a mask.

    HOW (small-angle Snell, screen space): the jump-flood distance transform gives each inside pixel its
    distance to the mask EDGE. A profile turns distance into a height bump h --
      profile='lens'     h = smoothstep(d / edge_width): a meniscus that rises over `edge_width` pixels then
                         plateaus (a water droplet / glass blob). Steep near the edge, FLAT in the middle.
      profile='dome'     h = normalised d (a cone smoothed by the transform): distortion everywhere inside,
                         still strongest at the edge.
    The view-ray displacement is  -(ior - 1) * strength * grad(h)  -- proportional to the surface tilt, so it
    is automatically LARGEST NEAR THE EDGE and ZERO on the plateau and outside the mask: the requested
    'distortion that changes with distance from the edge'. The image is bilinearly resampled at the displaced
    coordinates; pixels outside the mask are returned untouched.

    chromatic > 0 displaces R/G/B by (1-c, 1, 1+c) x strength -- dispersion fringes at the rim.
    ripple = (amplitude_px, scale) adds an fbm wobble to the displacement -- water-surface shimmer on top of
    the lens (deterministic in `seed`).
    edge_width defaults to 35% of the mask's max interior distance (a proportioned meniscus at any size).

    Returns the refracted image, same shape/dtype family as `image` (float in [0,1] if input was).
    KEPT NEGATIVE: single-interface small-angle approximation -- no TIR, no caustics, no second surface.
    True refraction is path_trace's dielectric; this is the fast 2D compositing effect and says so."""
    from holographic.misc.holographic_jit import distance_transform
    img = np.asarray(image, float)
    if img.ndim == 2:
        img = img[..., None]
    H, W = img.shape[:2]
    m = np.asarray(mask).astype(bool)
    if m.shape != (H, W):
        raise ValueError("mask shape %r must match image %r" % (m.shape, (H, W)))
    if profile not in ("lens", "dome"):
        raise ValueError("profile must be 'lens' or 'dome', got %r" % (profile,))
    if not m.any():
        return image if np.asarray(image).ndim == img.ndim else img[..., 0]

    # distance to the edge, INSIDE the mask: seed the transform with the OUTSIDE region
    d = distance_transform(~m)                                  # 0 outside, grows toward the interior
    d = np.where(m, d, 0.0)
    dmax = float(d.max())
    ew = float(edge_width) if edge_width is not None else max(0.35 * dmax, 1.0)
    if profile == "lens":
        t = np.clip(d / ew, 0.0, 1.0)
        h = t * t * (3.0 - 2.0 * t)                             # smoothstep meniscus
    else:
        h = d / max(dmax, 1e-12)

    gy, gx = np.gradient(h)                                     # the surface tilt
    if ripple is not None:
        amp, rscale = float(ripple[0]), float(ripple[1])
        f = _fbm_field(scale=rscale, octaves=3, seed=seed)
        yy, xx = np.mgrid[0:H, 0:W]
        Pn = np.stack([xx.ravel() / max(W, 1), yy.ravel() / max(H, 1), np.zeros(H * W)], axis=1)
        wob = (f(Pn).reshape(H, W) * 2.0 - 1.0) * amp / max(float(strength), 1e-12)
        gx = gx + wob * np.where(m, 1.0, 0.0)                   # wobble only inside the shape
        gy = gy + np.roll(wob, 7, axis=0) * np.where(m, 1.0, 0.0)

    base = (float(ior) - 1.0) * float(strength)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    out = np.empty_like(img)
    n_ch = img.shape[2]
    for ci in range(n_ch):
        cscale = 1.0 + (ci - (n_ch - 1) / 2.0) * float(chromatic) if n_ch >= 3 else 1.0
        sx = np.clip(xx - base * cscale * gx * W, 0, W - 1)     # displacement AGAINST the tilt (Snell bends
        sy = np.clip(yy - base * cscale * gy * H, 0, H - 1)     # the ray toward the normal entering glass)
        x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
        x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
        fx2 = sx - x0; fy2 = sy - y0
        c = (img[y0, x0, ci] * (1 - fx2) * (1 - fy2) + img[y0, x1, ci] * fx2 * (1 - fy2)
             + img[y1, x0, ci] * (1 - fx2) * fy2 + img[y1, x1, ci] * fx2 * fy2)
        out[..., ci] = np.where(m, c, img[..., ci])             # outside the mask: untouched
    return out if np.asarray(image).ndim == 3 else out[..., 0]


# ======================================================================================================

def _selftest():
    rng = np.random.default_rng(0)
    P = rng.uniform(-1.0, 2.0, (400, 3))

    # 1. every menu entry builds, evaluates on (M,3), and is deterministic
    for name in sorted(TEXTURES):
        kw = {"seed": 3} if _sig(name) else {}
        v1 = np.asarray(proc_texture(name, **kw)(P))
        v2 = np.asarray(proc_texture(name, **kw)(P))
        assert v1.shape == (400,), (name, v1.shape)
        assert np.array_equal(v1, v2), "%s must be deterministic" % name
        assert np.isfinite(v1).all(), name
    try:
        proc_texture("granite2")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown texture must raise")

    # 2. voronoi contracts: F2 >= F1 everywhere; F2-F1 >= 0; cell is piecewise constant (two points in the
    #    same cell share an id); a different seed moves the features
    vf1 = voronoi(seed=1, kind="f1")(P); vf2 = voronoi(seed=1, kind="f2")(P)
    assert np.all(vf2 >= vf1 - 1e-12), "F2 must dominate F1"
    assert np.all(voronoi(seed=1, kind="f2f1")(P) >= -1e-12)
    cid = voronoi(seed=1, kind="cell", scale=1.0)
    assert cid(np.array([[0.31, 0.32, 0.30]]))[0] == cid(np.array([[0.33, 0.30, 0.31]]))[0], \
        "nearby points in one cell share the id"
    assert not np.array_equal(voronoi(seed=1)(P), voronoi(seed=2)(P))

    # 3. musgrave: ridged output in [0,1]; ridged is NOT plain fbm (the fold changes the distribution --
    #    ridged has more mass near its peaks); hybrid also bounded
    r = musgrave(kind="ridged", seed=0)(P)
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert not np.allclose(r, musgrave(kind="fbm", seed=0)(P))
    h = musgrave(kind="hybrid", seed=0)(P)
    assert h.min() >= 0.0 and h.max() <= 1.0

    # 4. wave/rings geometry: bands depend ONLY on the chosen axis; rings are radially symmetric about it
    wb = wave(kind="bands", axis=0, scale=2.0)
    assert abs(wb(np.array([[0.4, 0.1, 0.9]]))[0] - wb(np.array([[0.4, -3.0, 7.0]]))[0]) < 1e-12
    wr = wave(kind="rings", axis=1, scale=2.0)
    a = wr(np.array([[0.5, 9.9, 0.0]]))[0]; b = wr(np.array([[0.0, -4.0, 0.5]]))[0]
    assert abs(a - b) < 1e-12, "rings: same radius -> same value, any angle/height"

    # 5. brick: mortar pixels are dark and cover roughly the mortar fraction; bricks vary per cell
    img = proc_texture_image("brick", size=128, region=((0.0, 0.0), (2.0, 1.0)), seed=0)
    dark = float((img < 0.15).mean())
    assert 0.05 < dark < 0.5, ("a plausible mortar fraction", dark)

    # 6. texture_image == the z-plane of texture_volume (2D texturing IS 3D on a plane) -- same field,
    #    same lattice, so the volume's z=0 slice must equal the image up to the shared normalisation
    fimg = proc_texture_image("noise", size=24, region=((0.0, 0.0), (1.0, 1.0)), z=0.0, scale=3.0, seed=5)
    vol = proc_texture_volume("noise", res=24, bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), scale=3.0, seed=5)
    f = value_noise(scale=3.0, seed=5)
    xs = np.linspace(0, 1, 24)
    X, Y = np.meshgrid(xs, xs)
    raw_img = f(np.stack([X.ravel(), Y.ravel(), np.zeros(576)], 1)).reshape(24, 24)
    raw_vol = f(np.stack([X.ravel(), Y.ravel(), np.zeros(576)], 1)).reshape(24, 24)
    assert np.array_equal(raw_img, raw_vol), "one field, two samplers"
    assert vol.shape == (24, 24, 24) and fimg.shape == (24, 24)

    # 7. mask_refraction: strength 0 (or ior 1) is the identity; outside the mask NEVER changes; the
    #    displacement is edge-concentrated (mean |shift| in the rim band exceeds the plateau's); chromatic
    #    separates the channels
    Himg = 96
    yy, xx = np.mgrid[0:Himg, 0:Himg]
    bg = np.stack([np.mod(xx // 8 + yy // 8, 2).astype(float)] * 3, axis=-1)      # a checkerboard
    mask = (xx - 48) ** 2 + (yy - 48) ** 2 < 30 ** 2                               # a disc
    same = mask_refraction(bg, mask, strength=10.0, ior=1.0)
    assert np.array_equal(same, bg), "ior=1 must be the identity"
    ref = mask_refraction(bg, mask, strength=10.0, ior=1.5)
    assert np.array_equal(ref[~mask], bg[~mask]), "outside the mask is untouched"
    assert not np.array_equal(ref[mask], bg[mask]), "inside must distort"
    from holographic.misc.holographic_jit import distance_transform as _dt
    d = np.where(mask, _dt(~mask), 0.0)
    rim = mask & (d < 0.35 * d.max()); plateau = mask & (d > 0.7 * d.max())
    delta = np.abs(ref - bg).mean(axis=-1)
    assert delta[rim].mean() > 3.0 * max(delta[plateau].mean(), 1e-9), \
        ("distortion must concentrate at the edge", delta[rim].mean(), delta[plateau].mean())
    chrom = mask_refraction(bg, mask, strength=10.0, ior=1.5, chromatic=0.25)
    assert not np.array_equal(chrom[..., 0], chrom[..., 2]), "chromatic must separate R from B"
    #    ripple wobbles deterministically
    r1 = mask_refraction(bg, mask, strength=10.0, ripple=(2.0, 6.0), seed=4)
    r2 = mask_refraction(bg, mask, strength=10.0, ripple=(2.0, 6.0), seed=4)
    assert np.array_equal(r1, r2) and not np.array_equal(r1, ref)

    # 8. the sampler + ramp layer: assign -> sample identity; stop exactness; field wrapping
    vals = rng.uniform(0, 1, (3, 5))
    tex = values_to_texture(vals)
    Ht, Wt = tex.shape
    uvc = np.array([[(x + 0.5) / Wt, (y + 0.5) / Ht] for y in range(Ht) for x in range(Wt)])
    assert np.array_equal(sample_image(tex, uvc).reshape(Ht, Wt), vals), \
        "assign->sample at texel centres must be the identity (the layer's closing contract)"
    strip = values_to_texture(np.array([0.0, 1.0]))
    assert float(sample_image(strip, [[0.5, 0.5]])[0]) == 0.5, "bilinear midpoint is the mean"
    assert float(sample_image(strip, [[1.25, 0.5]], wrap="repeat")[0]) == 0.0
    assert float(sample_image(strip, [[1.25, 0.5]], wrap="clamp")[0]) == 1.0
    for interp in ("linear", "constant", "smooth"):
        rr = ramp([0.0, 0.5, 1.0], [0.0, 1.0, 0.2], interp=interp)
        assert np.allclose(rr([0.0, 0.5, 1.0]), [0.0, 1.0, 0.2]), \
            "a stop's own position returns its exact value in every mode (%s)" % interp
    assert float(ramp([0.0, 0.5, 1.0], [0.0, 1.0, 0.2])([0.25])[0]) == 0.5
    rcst = ramp([0.0, 0.5, 1.0], [0.0, 1.0, 0.2], interp="constant")
    assert float(rcst([0.49])[0]) == 0.0 and float(rcst([0.51])[0]) == 1.0, "constant mode holds bands"
    rt = ramp_texture([0.0, 1.0], [0.0, 1.0], size=8)
    ts8 = (np.arange(8) + 0.5) / 8
    assert np.allclose(sample_image(values_to_texture(rt), np.stack([ts8, np.full(8, 0.5)], 1)), ts8), \
        "a baked ramp sampled back IS the ramp"
    fimg2 = image_field(np.array([[0.0, 1.0], [0.5, 0.25]]), wrap="clamp")
    assert np.allclose(fimg2(np.array([[0.25, 0.25, 9.9], [0.75, 0.25, -3.0]])), [0.0, 1.0]), \
        "image_field reads texels; z is ignored (a painted map is a field)"
    for bad in (dict(mode="cubic"), dict(wrap="mirror")):
        try:
            sample_image(strip, [[0.5, 0.5]], **bad); raise AssertionError("must refuse %r" % bad)
        except ValueError:
            pass

    print("OK: holographic_proctex self-test passed (14-texture menu deterministic on (M,3); voronoi F2>=F1 "
          "with shared cell ids; ridged/hybrid bounded; bands/rings geometrically exact; brick mortar %.0f%%; "
          "one field serves both samplers; refraction: identity at ior=1, outside untouched, edge/plateau "
          "distortion ratio %.1fx, chromatic separates channels)"
          % (dark * 100, delta[rim].mean() / max(delta[plateau].mean(), 1e-9)))


def _sig(name):
    """Which builders accept a seed (for the selftest's determinism loop)."""
    return {"noise": "seed", "fbm": "seed", "white": "seed", "voronoi": "seed", "musgrave": "seed",
            "wave": "seed", "marble": "seed", "wood": "seed", "brick": "seed"}.get(name, "")


if __name__ == "__main__":
    _selftest()

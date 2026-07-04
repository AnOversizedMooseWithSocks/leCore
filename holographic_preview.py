"""holographic_preview.py -- SEE what you composed: a flat swatch for a texture graph, a shaded ball for a material.

The composability stack (CMP1-CMP5) builds things you sample with .sample(uv) -- texture graphs, multi/layered
materials. This module renders them to a small RGB image you can save and look at, which is the missing last step
between "I composed a material" and "does it look right?". Two previews, the two every DCC tool gives you:

  * texture_image(graph)   -- a flat SWATCH of a CMP1 texture graph (its colour over the UV square). Instant.
  * material_ball(material) -- the classic MATERIAL BALL: the material shaded on a sphere with a Cook-Torrance BRDF
                               and one light, so its roughness / metallic / albedo read the way they will on a
                               curved surface.

Both return a float image in [0,1], shape (res, res, 3) -- save it with any image writer (PIL, imageio) or hand it
straight to a viewer. Deterministic; plain NumPy.

Reuses: holographic_texturegraph.sample_grid (the swatch is a graph evaluated on a grid), holographic_brdf
(cook_torrance -- the same BRDF the real renderer uses, so a preview matches the render). The only per-pixel loop is
sampling the material over the sphere's visible pixels; everything else is vectorised.
"""
import numpy as np


def texture_image(graph, res=256, lo=0.0, hi=1.0):
    """A CMP1 texture graph as a flat RGB SWATCH: evaluate it over the UV square and return a (res,res,3) image in
    [0,1]. A colour graph shows its rgb; a scalar graph shows as greyscale. Values are CLAMPED to [0,1] for display
    (composition can legitimately push a value out of range -- see the CMP1 'saturate' op if you want that baked in)."""
    from holographic_texturegraph import sample_grid
    grid = np.asarray(sample_grid(graph, res=res, lo=lo, hi=hi), float)
    if grid.ndim == 2:                                      # scalar field -> greyscale
        grid = np.repeat(grid[:, :, None], 3, axis=2)
    grid = grid[:, :, :3]                                   # drop an alpha channel if the graph produced rgba
    return np.clip(grid, 0.0, 1.0)


def _channel_names(material):
    """The channels a material exposes, whether it's a plain Material (.channels dict) or a CMP2/CMP3 material
    (.channel_names())."""
    if hasattr(material, "channel_names"):
        return set(material.channel_names())
    return set(getattr(material, "channels", {}))


def _sample_scalar(material, name, uv, default):
    """Sample a scalar channel at uv, clamped to [0,1]; `default` if the material doesn't have that channel. Handles a
    channel that returns a colour by taking its mean (so a preview never crashes on an unexpected shape)."""
    if name not in _channel_names(material):
        return float(default)
    v = np.asarray(material.sample(name, uv), float)
    return float(np.clip(v if v.ndim == 0 else v.mean(), 0.0, 1.0))


def material_ball(material, res=192, base_color=(0.82, 0.80, 0.78), light_dir=(0.6, 0.7, 0.5),
                  background=0.14, ambient=0.06):
    """Render `material` on a preview SPHERE -- the standard 'material ball'. Works on a plain Material or a CMP2/CMP3
    layered/multi material (anything with .sample(channel, uv) + channels). Uses the material's `roughness` and
    `metallic` channels where present (else sensible defaults), and modulates `base_color` by an `albedo` channel if
    there is one. Shades with the same Cook-Torrance BRDF the real renderer uses, so the ball matches a render.
    Returns a (res, res, 3) float image in [0,1].

    Orthographic camera down -z onto a unit sphere at the origin, one directional light. The only loop is sampling the
    material at each visible pixel's UV; the shading is vectorised."""
    from holographic_brdf import cook_torrance

    base_color = np.asarray(base_color, float)
    L = np.asarray(light_dir, float)
    L = L / (np.linalg.norm(L) + 1e-12)
    V = np.array([0.0, 0.0, 1.0])                           # orthographic view direction (toward the camera)

    # image-plane coords in [-1.2, 1.2]; a pixel hits the sphere when x^2 + y^2 <= 1 (front hemisphere)
    xs = np.linspace(-1.2, 1.2, res)
    X, Y = np.meshgrid(xs, xs)
    Y = -Y                                                  # image row 0 at top
    r2 = X * X + Y * Y
    hit = r2 <= 1.0
    Z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))               # front-surface z on the unit sphere

    img = np.empty((res, res, 3), float)
    img[:] = _background(res, background)                   # neutral vertical gradient behind the ball

    ph, pw = np.where(hit)                                  # the pixels that land on the sphere
    P = np.stack([X[ph, pw], Y[ph, pw], Z[ph, pw]], axis=1)   # surface points (M,3)
    N = P                                                   # on a unit sphere the point IS the normal
    # spherical UVs: u around the equator, v from pole to pole -- a stable, seam-simple mapping for a preview
    u = np.arctan2(P[:, 0], P[:, 2]) / (2.0 * np.pi) + 0.5
    v = np.arccos(np.clip(P[:, 1], -1.0, 1.0)) / np.pi

    # sample the material per visible pixel (the one loop) -> albedo tint, roughness, metallic
    has_albedo = "albedo" in _channel_names(material)
    alb = np.empty((len(P), 3))
    rough = np.empty(len(P))
    metal = np.empty(len(P))
    for i in range(len(P)):
        uv = (float(u[i]), float(v[i]))
        rough[i] = _sample_scalar(material, "roughness", uv, 0.5)
        metal[i] = _sample_scalar(material, "metallic", uv, 0.0)
        # albedo channel (scalar) modulates the base tint; if the material has none, the tint is used as-is
        alb[i] = base_color * (_sample_scalar(material, "albedo", uv, 1.0) if has_albedo else 1.0)

    # shade all sphere pixels at once with Cook-Torrance + a touch of ambient, tone-mapped into [0,1]
    Nv = np.repeat(N[None, :, :], 1, axis=0)[0]            # (M,3) already
    shaded = cook_torrance(Nv, np.broadcast_to(V, N.shape), np.broadcast_to(L, N.shape), alb, metal, rough)
    shaded = shaded + ambient * alb                        # small ambient so shadowed side isn't pure black
    shaded = shaded / (1.0 + shaded)                       # Reinhard tone-map -> [0,1)
    img[ph, pw] = np.clip(shaded, 0.0, 1.0)
    return img


def _background(res, level):
    """A neutral vertical gradient behind the ball, so it reads as sitting in a soft studio rather than on flat grey."""
    col = np.linspace(level * 1.6, level * 0.7, res)[:, None]
    return np.repeat(np.repeat(col[:, :, None], res, axis=1), 3, axis=2)


def _selftest():
    from holographic_texturegraph import Map, Const, field_leaf
    from holographic_fpe import VectorFunctionEncoder
    from holographic_material import Material, texture_field

    # texture swatch: a colour graph -> a viewable rgb image in [0,1]
    g = Map("mix", a=Const("red"), b=Const("blue"), t=field_leaf("fbm", n_dims=2, seed=0))
    swatch = texture_image(g, res=64)
    assert swatch.shape == (64, 64, 3)
    assert swatch.min() >= 0.0 and swatch.max() <= 1.0
    # a scalar graph -> greyscale (all three channels equal)
    grey = texture_image(Map("scale", x=field_leaf("fbm", n_dims=2, seed=1), k=Const(1.0)), res=32)
    assert np.allclose(grey[:, :, 0], grey[:, :, 1]) and np.allclose(grey[:, :, 1], grey[:, :, 2])

    # material ball: a material with a roughness pattern -> a shaded sphere image
    enc = VectorFunctionEncoder(2, dim=512, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)
    grid = [(a, b) for a in np.linspace(0.05, 0.95, 6) for b in np.linspace(0.05, 0.95, 6)]
    mat = Material(enc, {"roughness": texture_field(enc, grid, [a for (a, b) in grid]),
                         "metallic": texture_field(enc, grid, [0.0 for _ in grid])})
    ball = material_ball(mat, res=96)
    assert ball.shape == (96, 96, 3) and ball.min() >= 0.0 and ball.max() <= 1.0
    # the ball's centre (on the sphere) differs from a background corner (the sphere was actually shaded)
    assert not np.allclose(ball[48, 48], ball[0, 0])

    # a CMP2/CMP3 material also previews (has .sample(channel, uv) + channel_names)
    from holographic_layeredmaterial import Layer, LayeredMaterial
    stack = LayeredMaterial([Layer("base", mat), Layer("coat", mat, alpha=0.3)])
    ball2 = material_ball(stack, res=64)
    assert ball2.shape == (64, 64, 3)

    print("OK: holographic_preview self-test passed (texture swatch is a %s rgb image in [0,1]; scalar graph -> "
          "greyscale; material ball shades a sphere with the real BRDF (centre != background); a CMP2 layered "
          "material previews too)" % (swatch.shape,))


if __name__ == "__main__":
    _selftest()

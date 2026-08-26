"""holographic_atmosphere.py -- depth fog and volumetric light shafts / god rays (W16).

WHY THIS MODULE EXISTS
----------------------
The engine had `holographic_volint.render_fog` (closed-form optical depth over a HolographicVolume), but no
LIGHT SHAFTS / god rays, and no simple DEPTH fog that works straight off a render's depth buffer without building
a volume first. iq's D2 "Orbit-Trap Cathedral" wants exactly those: dusty air and shafts of light stabbing
through it. This module adds both as cheap screen-space passes over an image + its depth buffer.

THE TWO EFFECTS
  * depth_fog: exponential (Beer-Lambert) fog -- out = lerp(color, fog_color, 1 - exp(-density * depth)). The
    farther a pixel, the more it fades to the fog colour. One line of physics, the atmosphere of every scene.
  * light_shafts: the RADIAL-BLUR god-ray trick (Kenny Mitchell, GPU Gems 3). Build a bright-pass mask (the sky
    / light source), then blur it radially OUTWARD from the light's screen position, accumulating with decay.
    The streaks that result ARE the shafts -- volumetric scattering approximated in screen space, no marching.

DESIGN NOTES (negatives)
  * The god-ray pass is SCREEN-SPACE: it can only shaft light from a source that is ON screen (or just off the
    edge). A light behind the camera gives nothing -- that is a property of the trick, named here so no one files
    it as a bug. For an off-screen source, place the sample origin at the clamped screen projection.
  * Fog is applied in LINEAR space then the caller gamma-corrects, same as every other render path -- fog on
    gamma-encoded colour looks milky and wrong.

NumPy only. Deterministic. Both take/return (H, W, 3) float images in [0, 1].
"""

import numpy as np


def depth_fog(color, depth, density=0.15, fog_color=(0.55, 0.65, 0.82), start=0.0):
    """Exponential DEPTH FOG (Beer-Lambert): fade each pixel toward `fog_color` by 1 - exp(-density * (depth -
    start)). `color` (H,W,3) linear, `depth` (H,W) per-pixel distance (e.g. the raymarch `t`, inf/large for the
    background). `start` delays the fog to begin at a distance. Returns (H,W,3). The atmosphere of a scene in one
    pass -- apply BEFORE gamma. A larger density = thicker air."""
    color = np.asarray(color, float)
    depth = np.asarray(depth, float)
    d = np.clip(depth - start, 0.0, None)
    f = 1.0 - np.exp(-density * d)                              # fog fraction in [0,1)
    fog = np.asarray(fog_color, float)
    return color * (1.0 - f[..., None]) + fog * f[..., None]


def _bright_pass(color, threshold=0.7):
    """Keep only the bright pixels (the light source / sky) as a luminance mask -- the god-ray source. (H,W)."""
    lum = color @ np.array([0.2126, 0.7152, 0.0722])           # Rec.709 luminance
    return np.clip((lum - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0)


def light_shafts(color, light_uv=(0.5, 0.2), threshold=0.7, density=0.9, decay=0.92,
                 weight=0.5, exposure=0.35, samples=48):
    """Volumetric LIGHT SHAFTS / god rays by radial blur (Mitchell, GPU Gems 3). Build a bright-pass mask of the
    light/sky, then march each pixel's sample position TOWARD the light's screen UV, accumulating the mask with
    per-step `decay` so brightness streaks outward from the source. `light_uv` is the light's (u, v) in [0,1]
    screen coords (u=0 left, v=0 top). Returns the shaft glow (H,W,3) to ADD to the scene (out = scene +
    shafts). `density` scales the step length toward the light; `weight`/`exposure` scale the glow.

    SCREEN-SPACE: only shafts from a source on (or near) screen -- a light behind the camera yields nothing (see
    the module note). Cheap: no volume marching, just `samples` texture-space taps per pixel."""
    color = np.asarray(color, float)
    H, W, _ = color.shape
    mask = _bright_pass(color, threshold)                      # (H, W)
    lu, lv = light_uv
    lx, ly = lu * (W - 1), lv * (H - 1)

    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    # vector from each pixel toward the light, stepped `samples` times with decreasing weight
    dx = (lx - xs) * (density / samples)
    dy = (ly - ys) * (density / samples)
    accum = np.zeros((H, W))
    illum_decay = 1.0
    sx, sy = xs.copy(), ys.copy()
    for _ in range(samples):
        sx += dx; sy += dy
        ix = np.clip(sx, 0, W - 1).astype(int)
        iy = np.clip(sy, 0, H - 1).astype(int)
        accum += mask[iy, ix] * (illum_decay * weight)
        illum_decay *= decay
    glow = np.clip(accum * exposure, 0.0, 1.0)
    # tint the shafts warm (sunlight); a caller can recolour by scaling channels
    return glow[..., None] * np.array([1.0, 0.95, 0.82])


def _selftest():
    """Contracts as properties:

    1. Depth fog fades FAR pixels toward the fog colour and leaves NEAR pixels almost untouched.
    2. Zero density = no change; larger density = more fog on the same pixel (monotone).
    3. Light shafts are BRIGHTEST near the light's screen position and fall off away from it (the streak).
    4. A dark scene (no bright pass) yields ~no shafts (nothing to scatter).
    5. Determinism.
    """
    H = W = 48
    # a scene: a dark foreground object (near) and a bright sky (far)
    color = np.full((H, W, 3), 0.2)
    color[:, :] = 0.15
    depth = np.full((H, W), 1.0)
    depth[:H // 2] = 12.0                                       # top half is far away

    # (1) far pixels fog more than near pixels.
    fogged = depth_fog(color, depth, density=0.3, fog_color=(0.9, 0.9, 1.0))
    near_change = np.abs(fogged[H - 1] - color[H - 1]).mean()
    far_change = np.abs(fogged[0] - color[0]).mean()
    assert far_change > near_change * 3                        # the far half fogged much more

    # (2) monotone in density.
    light = depth_fog(color, depth, density=0.1)
    heavy = depth_fog(color, depth, density=0.6)
    assert np.abs(heavy - color).mean() > np.abs(light - color).mean()
    assert np.allclose(depth_fog(color, depth, density=0.0), color)   # zero density = identity

    # (3) shafts brightest near the light. Put a bright disc at the top-centre.
    scene = np.full((H, W, 3), 0.1)
    yy, xx = np.mgrid[0:H, 0:W]
    disc = ((xx - W // 2) ** 2 + (yy - 6) ** 2) < 25
    scene[disc] = 1.0
    shafts = light_shafts(scene, light_uv=(0.5, 6.0 / H), samples=40)
    top_glow = shafts[:H // 3].mean()
    bot_glow = shafts[2 * H // 3:].mean()
    assert top_glow > bot_glow                                 # brighter near the light

    # (4) a dark scene yields ~no shafts.
    dark = np.full((H, W, 3), 0.05)
    assert light_shafts(dark).max() < 0.05

    # (5) determinism.
    assert np.array_equal(light_shafts(scene), light_shafts(scene))

    print("holographic_atmosphere selftest OK (depth fog: far/near %.3f>%.3f, monotone in density, zero=identity; "
          "light shafts brightest near source %.3f>%.3f, dark scene yields none; deterministic)"
          % (far_change, near_change, top_glow, bot_glow))


if __name__ == "__main__":
    _selftest()

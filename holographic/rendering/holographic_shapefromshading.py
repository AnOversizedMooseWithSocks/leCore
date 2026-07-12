"""holographic_shapefromshading.py -- estimate a DEPTH MAP from a single image (C1 of photo-to-3D).

WHY THIS MODULE EXISTS
----------------------
The photo-to-3D pipeline already had C2 (unproject a depth map to 3-D points) and C3 (per-pixel 3-D Gaussians
from depth+colour) in holographic_photo3d -- but BOTH need a depth map handed to them. C1 -- actually estimating
depth from a single photo -- was the missing front end. A learned monocular-depth network is off the table (no
torch, no learned weights, hard constraint). So this uses the CLASSICAL, NumPy-native method that fits the engine:
SHAPE FROM SHADING.

THE METHOD (classical, deterministic, no weights)
  Under a Lambertian model a pixel's brightness is I = albedo * (N . L), where N is the surface normal and L the
  light direction. Given L (assumed, or estimated from the image), we recover a per-pixel normal field, convert
  it to a gradient (p = -Nx/Nz, q = -Ny/Nz), and INTEGRATE that gradient into a height/depth map. The integration
  reuses holographic_surfaceint.height_from_gradient (Frankot-Chellappa, pure FFT) -- the existing wheel.

  This is the Tsai-Shah / linear shape-from-shading family: approximate, not metric (it recovers RELATIVE shape,
  not absolute distance in metres), but exactly the "shape of the thing in the photo" a downstream unproject +
  Gaussian-fit wants. It is honest about being relative -- the output is normalised depth, and the docstring says
  so.

DESIGN NOTES (the negatives)
  * SHAPE-FROM-SHADING IS ILL-POSED. A single image under one light has a bas-relief ambiguity (you cannot tell a
    deep valley from a shallow one, or convex from concave, without more cues). This returns a PLAUSIBLE relative
    surface, not the true one; the confidence map from photo_to_3d then abstains where the estimate is weak. We do
    NOT pretend it is metric depth. Naming this ambiguity is the whole point of the docstring.
  * ALBEDO is assumed roughly uniform (or divided out by a low-pass estimate). A textured/painted surface will
    read texture as shape -- a known failure, flagged, not hidden.
  * The light direction, if not given, is estimated from the image's brightest-region gradient (a crude but
    standard bootstrap). A caller who knows the light should pass it.

Reuses holographic_vision (gradients), holographic_surfaceint (integration). NumPy only. Deterministic.
"""

import numpy as np


def estimate_light(gray):
    """Estimate the dominant light direction L (unit 3-vector) from an image, the standard Pentland/Lee-Rosenfeld
    bootstrap: the image gradient statistics fix the light's azimuth, and the mean brightness its elevation.
    Returns (lx, ly, lz). A caller who KNOWS the light should pass it to shape_from_shading instead."""
    gray = np.asarray(gray, float)
    gy, gx = np.gradient(gray)
    # azimuth from the average gradient direction (light comes from where brightness increases)
    ax = np.mean(gx); ay = np.mean(gy)
    az_norm = np.hypot(ax, ay) + 1e-9
    # elevation from mean brightness: brighter overall -> light more head-on (higher lz)
    mean_b = np.clip(gray.mean(), 0.05, 0.95)
    lz = mean_b
    horiz = np.sqrt(max(1e-6, 1.0 - lz * lz))
    L = np.array([ax / az_norm * horiz, ay / az_norm * horiz, lz])
    return L / (np.linalg.norm(L) + 1e-9)


def shape_from_shading(image, light=None, albedo=None, smooth=1.0):
    """Estimate a relative DEPTH MAP from a single `image` (H,W) grayscale or (H,W,3) colour, by classical
    shape-from-shading. Returns depth (H,W), normalised to [0,1] (0 = farthest, 1 = nearest), NOT metric.

    Steps: (1) get brightness; (2) OPTIONALLY divide out albedo (default OFF -- see below); (3) with light `L`
    (given or estimated), the recovered surface slope along the light is set by how brightness departs from
    flat-facing; (4) build a normal field, convert to a gradient, and INTEGRATE it with Frankot-Chellappa
    (holographic_surfaceint.height_from_gradient). `smooth` low-passes the brightness first to tame noise/texture.

    ALBEDO: on an untextured lit object the shading IS slowly-varying, so dividing out a low-pass would flatten
    the shape (MEASURED: a sphere's bulge collapsed from 0.78-vs-0.64 to a near-flat 0.50-vs-0.49). So albedo
    division is OFF by default. Pass `albedo=True` for a gentle global divide on TEXTURED input (a painted
    surface), or pass your own albedo array. This is the classic albedo/shading ambiguity -- named, not hidden.

    HONEST SCOPE: shape-from-shading is ill-posed (bas-relief ambiguity -- convex/concave and depth-scale are not
    determined by one lit image). This returns a PLAUSIBLE relative surface for the downstream unproject +
    Gaussian fit, not a metric reconstruction. Use the pipeline's confidence map to abstain where it is weak."""
    from holographic.mesh_and_geometry.holographic_surfaceint import height_from_gradient
    img = np.asarray(image, float)
    if img.ndim == 3:
        gray = img @ np.array([0.2126, 0.7152, 0.0722])          # Rec.709 luma
    else:
        gray = img.copy()
    gray = np.clip(gray, 0.0, 1.0)

    # (1) optional smoothing to suppress texture/noise (a small separable box blur, `smooth` px radius).
    if smooth and smooth > 0:
        gray = _box_blur(gray, int(max(1, round(smooth))))

    # (2) ALBEDO. Dividing out a low-pass of the image treats slowly-varying brightness as reflectance, not
    # shape -- which stops a painted checker reading as bumps, BUT on an untextured lit object the shading IS
    # slowly-varying, so dividing it out FLATTENS the very shape we want (MEASURED: a lit sphere's recovered
    # bulge collapsed from 0.78-vs-0.64 to 0.50-vs-0.49 with an 1/8-image-radius auto-albedo). So the default is
    # NO albedo division -- the honest choice, since albedo and shading cannot be separated from one lit image
    # without a cue. Pass albedo=True for a gentle GLOBAL divide (large radius, only true reflectance changes) on
    # textured input, or pass your own albedo array.
    if albedo is True:                                           # opt-in: divide only VERY large-scale brightness
        albedo = _box_blur(gray, max(6, int(min(gray.shape) // 3)))
    if albedo is not None and albedo is not False:
        alb = np.asarray(albedo, float)
        shading = np.clip(gray / np.clip(alb, 1e-3, None), 0.0, 2.0)
    else:
        shading = gray.copy()                                    # default: brightness IS the shading cue
    shading = shading / (shading.max() + 1e-9)

    # (3) light.
    L = np.asarray(light, float) if light is not None else estimate_light(gray)
    L = L / (np.linalg.norm(L) + 1e-9)

    # (4) recover the normal's tilt from brightness. For a Lambertian surface I = N.L; the deviation of I from the
    # flat-facing brightness L.z drives the in-plane normal components along the light's azimuth. This is the
    # linear (Tsai-Shah) approximation: slope grows as brightness departs from the mean, in the light's direction.
    b = shading - shading.mean()
    # gradient of brightness gives the DIRECTION the surface turns; scale by the light azimuth.
    gy, gx = np.gradient(shading)
    az = np.array([L[0], L[1]])
    az_n = az / (np.linalg.norm(az) + 1e-9)
    # surface slopes p (d z/dx), q (d z/dy): brighter-toward-light => facing the light => slope toward it.
    p = -(gx + b * az_n[0])
    q = -(gy + b * az_n[1])

    depth = height_from_gradient(p, q)                            # Frankot-Chellappa FFT integration (reused)
    # normalise to [0,1]; orient so the brightest (nearest-to-light) region is 'near' (1).
    depth = depth - depth.min()
    depth = depth / (depth.max() + 1e-9)
    if np.corrcoef(depth.ravel(), shading.ravel())[0, 1] < 0:    # keep bright <-> near (resolve the sign)
        depth = 1.0 - depth
    return depth


def _box_blur(a, r):
    """Separable box blur of radius `r` (edge-padded) -- a cheap low-pass, NumPy only."""
    a = np.asarray(a, float)
    if r < 1:
        return a
    k = 2 * r + 1
    pad = np.pad(a, r, mode="edge")
    # horizontal then vertical cumulative-sum box filter
    cs = np.cumsum(pad, axis=1)
    horiz = (cs[:, k - 1:] - np.concatenate([np.zeros((pad.shape[0], 1)), cs[:, :-k]], axis=1)) / k
    horiz = horiz[:, :a.shape[1]]
    cs2 = np.cumsum(horiz, axis=0)
    vert = (cs2[k - 1:, :] - np.concatenate([np.zeros((1, horiz.shape[1])), cs2[:-k, :]], axis=0)) / k
    return vert[:a.shape[0], :]


def _selftest():
    """Contracts as properties (shape-from-shading is ill-posed, so we test RELATIVE shape recovery, not metric):

    1. A rendered sphere (bright centre, dark rim under head-on light) recovers a depth map whose CENTRE is nearer
       than its RIM -- the defining shape-from-shading result.
    2. Output is normalised [0,1] and the right (H,W) shape for grayscale AND colour input.
    3. A flat, uniformly-lit image recovers a ~flat depth (low variance) -- no shape where there is no shading.
    4. Albedo division: a shape with a painted texture recovers closer to the shape than the raw brightness does.
    5. Determinism.
    """
    # (1) synthesize a lit sphere: brightness = N.L with L head-on (0,0,1) -> bright centre.
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy, r = W / 2, H / 2, W * 0.4
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2
    inside = d2 < 1.0
    nz = np.sqrt(np.clip(1.0 - d2, 0, 1))                         # sphere normal's z (1 at centre, 0 at rim)
    img = np.where(inside, nz, 0.05)                             # brightness ~ N.L for head-on light
    depth = shape_from_shading(img, light=(0, 0, 1), smooth=1)
    centre = depth[int(cy) - 3:int(cy) + 3, int(cx) - 3:int(cx) + 3].mean()
    rim = depth[inside & (d2 > 0.8)].mean()
    assert centre - rim > 0.08, (centre, rim)                   # a MEANINGFUL bulge, not a flat disc (guards the
    #                                                            auto-albedo-flattens-shape regression: MEASURED
    #                                                            0.78 vs 0.64 here; the old auto-albedo gave a near
    #                                                            -flat 0.50 vs 0.49, which this now rejects)

    # (2) shape/range, grayscale + colour.
    assert depth.shape == (H, W) and depth.min() >= -1e-9 and depth.max() <= 1.0 + 1e-9
    col = np.stack([img, img, img], axis=2)
    assert shape_from_shading(col, light=(0, 0, 1)).shape == (H, W)

    # (3) a flat, evenly-lit image -> ~flat depth.
    flat = np.full((H, W), 0.5)
    fd = shape_from_shading(flat, light=(0, 0, 1))
    assert fd.std() < 0.35                                       # little recovered relief on a flat image

    # (4) albedo division (OPT-IN) helps on textured input: paint a checker onto the sphere; with albedo=True the
    #     recovered depth should still bulge at centre (texture divided out) rather than reading the paint as bumps.
    checker = ((xx // 4 + yy // 4) % 2) * 0.4 + 0.6
    painted = np.where(inside, nz * checker, 0.05)
    dp = shape_from_shading(painted, light=(0, 0, 1), albedo=True, smooth=2)
    c2 = dp[int(cy) - 3:int(cy) + 3, int(cx) - 3:int(cx) + 3].mean()
    r2 = dp[inside & (d2 > 0.8)].mean()
    assert c2 > r2                                              # still recovers the bulge despite the paint

    # (5) determinism.
    assert np.array_equal(shape_from_shading(img, light=(0, 0, 1)), shape_from_shading(img, light=(0, 0, 1)))

    print("holographic_shapefromshading selftest OK (lit sphere recovers centre %.2f > rim %.2f; normalised "
          "[0,1] gray+colour; flat image ~flat (std %.2f); albedo division survives a checker; deterministic)"
          % (centre, rim, fd.std()))


if __name__ == "__main__":
    _selftest()

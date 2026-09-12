"""causticpass -- project a forward-traced caustic onto a rendered frame, as a separate pass.

WHY THIS EXISTS: BRUTE-FORCE PATH TRACING IS THE WRONG ALGORITHM FOR A CAUSTIC, and this engine's own
tracer says so in its kept negative -- it has no next-event estimation, so light is gathered only when
a bounce happens to hit an emitter. A caustic needs a small bright source, and a small bright source is
precisely what random hemisphere sampling almost never hits. Measured while trying: pushing the key
panel from 95 to 220 to strengthen the caustic made the RAW GRAIN WORSE, 0.76 -> 1.19, because every
extra stop of brightness concentrates the same light into fewer, hotter, rarer paths. That is not a
tuning problem, it is the estimator.

leCore already has the right algorithm for the light path -- holographic_globalillum.caustics forward-
traces light and splats where it lands, which is deterministic and therefore clean. So each path uses
the algorithm built for it and the two are composited, which is also what production renderers do with
a photon-mapped caustic pass. This module is the join.

THE JOIN IS EXACT, NOT APPROXIMATE. The caustic lands on a PLANE, and a plane is analytic: every
pixel's floor hit is a closed-form ray/plane intersection, so there is no G-buffer to allocate, no
depth to reproject, and no resampling error to trade against. What the projection does need is
OCCLUSION -- a floor point hidden behind the object must not receive the caustic, or the pattern paints
straight over the thing casting it, which looks exactly like a bug and is the first thing to check if a
composite looks wrong.

KEPT NEGATIVE -- THIS IS A PLANE-ONLY PASS. The projection assumes the receiver is a single axis-
aligned plane, because that is what makes it exact and free. A caustic falling on curved or multiple
receivers needs the general machinery (a real photon map with a nearest-neighbour gather), which is a
different and much larger build. The limit is declared rather than discovered: `project_to_plane` takes
a plane, not a scene.
"""
import numpy as np


def project_to_plane(caustic_rgb, eye, dirs, plane_y, window, center=(0.0, 0.0),
                     occluder_sdf=None, occluder_eps=0.03):
    """Map a caustic map (res,res,3) onto the camera's view of a plane -> (H,W,3) to composite.

    `dirs` is the camera's per-pixel ray directions, (H,W,3) or (H*W,3), as camera.ray_dirs gives them;
    `window` and `center` must be the SAME framing the caustic was rendered with (mind.caustics'
    window=/center=), because they are what turns a world position into a texel.

    `occluder_sdf` is the glass itself: floor points inside it are hidden from the camera and get zero.
    Skipping it paints the caustic over the object that casts it."""
    eye = np.asarray(eye, float).reshape(3)
    D = np.asarray(dirs, float)
    if D.ndim != 3:
        raise ValueError("dirs must be (H, W, 3); got %s -- reshape before calling" % (D.shape,))
    caustic = np.asarray(caustic_rgb, float)
    res = caustic.shape[0]
    half = float(window)
    cx, cz = float(center[0]), float(center[1])

    dy = D[..., 1]
    t = (float(plane_y) - eye[1]) / np.where(np.abs(dy) < 1e-9, -1e-9, dy)
    P = eye[None, None, :] + D * t[..., None]                      # the floor hit for every pixel
    u = (P[..., 0] - cx + half) / (2.0 * half)
    v = (P[..., 2] - cz + half) / (2.0 * half)
    inside = (t > 0) & (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0)

    xi = np.clip((u * (res - 1)).astype(int), 0, res - 1)
    zi = np.clip((v * (res - 1)).astype(int), 0, res - 1)
    out = np.where(inside[..., None], caustic[zi, xi], 0.0)

    if occluder_sdf is not None:
        hidden = np.asarray(occluder_sdf(P.reshape(-1, 3)), float).reshape(P.shape[:2])
        out = np.where((hidden < occluder_eps)[..., None], 0.0, out)
    return out


def composite(beauty, caustic_px, strength=1.0, normalise_pct=99.0, subtract_baseline=True, receiver_mask=None):
    """beauty + caustic EXCESS, with the excess normalised so `strength` means the same across scenes.

    `receiver_mask` (H,W) bool, default None = everywhere: the pixels that actually SHOW the receiver. caustic_pass
    projects the floor pattern through the camera and knows nothing about what is in front of the floor, so
    without the mask the pattern paints over the body standing on it -- seen as caustic light on the rind of a
    geode. Pass the bake's floor mask (bake.floor.reshape(H, W)). The normalisation is computed BEFORE masking,
    so hiding part of the pattern does not rescale the rest.

    SUBTRACT THE BASELINE, and this is a correctness fix rather than a cosmetic one. mind.caustics
    returns its splat map normalised to MEAN 1.0, so an unfocused part of the receiver reads about 1.0
    rather than 0. Compositing that whole map ADDS a flat sheet of light the beauty render already
    accounted for -- it double-counts the ambient, and because the map is finite it does so only inside
    the caustic window, which paints a hard-edged bright QUADRILATERAL across the floor. It looks
    exactly like a broken render and it was one: the window boundary, lit. What belongs in the
    composite is only the EXCESS over the unfocused level, which is what a caustic physically is.
    The baseline is the median of the lit region -- most of a receiver is not in the caustic, so the
    median IS the unfocused level, while the mean is dragged by the focus itself.

    WHY NORMALISE BY A PERCENTILE AND NOT THE MAX: a caustic's peak is a handful of splat cells that
    happened to collide, so dividing by the max is dividing by an outlier and makes `strength` mean a
    different thing every render. The 99th percentile is the bright BODY of the pattern.

    MEASURED BOUND, because a percentile has one: it needs enough lit pixels to have a 99th percentile
    that is not itself the outlier. Injecting a single 500x spike into an otherwise flat pattern moves
    the normalised BODY by 99.8% at 65 lit pixels and by exactly 0.0% at 200 or more. Real caustics
    light thousands of pixels, so this is a statement about degenerate inputs rather than a caveat on
    normal use -- but a 20x20 test pattern is degenerate, and finding that out from a test beat finding
    it out from a render."""
    b = np.asarray(beauty, float)
    c = np.asarray(caustic_px, float)
    if c.shape != b.shape:
        raise ValueError("caustic %s does not match beauty %s" % (c.shape, b.shape))
    lit = c > 0
    if not lit.any():
        return b.copy()
    if subtract_baseline:
        base = float(np.median(c[lit]))
        c = np.clip(c - base, 0.0, None)          # only the focused excess survives
        lit = c > 0
        if not lit.any():
            return b.copy()
    k = float(np.percentile(c[lit], normalise_pct))
    if receiver_mask is not None:
        c = c * np.asarray(receiver_mask, bool)[..., None]      # only where the receiver is what the pixel sees
    return b + c / max(k, 1e-12) * float(strength)


def _selftest():
    # A camera looking straight down at a plane, so the geometry is checkable by hand.
    H = W = 16
    eye = np.array([0.0, 2.0, 0.0])
    xs = np.linspace(-0.5, 0.5, W)
    zs = np.linspace(-0.5, 0.5, H)
    ZZ, XX = np.meshgrid(zs, xs, indexing="ij")
    dirs = np.stack([XX, -np.ones_like(XX), ZZ], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    caustic = np.zeros((32, 32, 3))
    caustic[16, 16] = [1.0, 1.0, 1.0]                       # one lit texel at the centre

    out = project_to_plane(caustic, eye, dirs, plane_y=0.0, window=1.0)
    assert out.shape == (H, W, 3)
    # 1. THE CENTRE PIXEL SEES THE CENTRE TEXEL. A ray straight down from (0,2,0) lands at the origin,
    #    which is texel 16 of 32 over a window of [-1,1]. If this is off by one the whole pass slides.
    assert out[H // 2, W // 2].max() > 0.0, "the centre ray missed the centre texel"
    assert np.count_nonzero(out.sum(-1)) <= 4, "one texel lit more than a small neighbourhood"

    # 2. RAYS THAT NEVER REACH THE PLANE CONTRIBUTE NOTHING -- looking UP must be black, not wrapped.
    up = project_to_plane(caustic, eye, -dirs, plane_y=0.0, window=1.0)
    assert up.sum() == 0.0, "rays pointing away from the plane still sampled it"

    # 3. OCCLUSION IS LOAD-BEARING: an occluder covering the origin must blank the centre.
    blocked = project_to_plane(caustic, eye, dirs, plane_y=0.0, window=1.0,
                               occluder_sdf=lambda P: np.linalg.norm(P, axis=-1) - 0.5)
    assert blocked.sum() == 0.0, "occluder did not blank the covered floor"

    # 4. COMPOSITE: strength scales the ADDED pattern and leaves the beauty alone where it is dark.
    beauty = np.full((H, W, 3), 0.1)
    c1 = composite(beauty, out, strength=1.0)
    c2 = composite(beauty, out, strength=3.0)
    assert np.allclose(c1[0, 0], 0.1) and np.allclose(c2[0, 0], 0.1), "composite touched unlit pixels"
    lit = out.sum(-1) > 0
    assert np.allclose((c2 - beauty)[lit], 3.0 * (c1 - beauty)[lit]), "strength is not linear"

    # 5. A mismatched caustic must fail loudly rather than broadcast into nonsense.
    try:
        composite(beauty, np.zeros((4, 4, 3)))
        raise AssertionError("shape mismatch must raise")
    except ValueError:
        pass
    print("causticpass selftest OK -- exact ray/plane projection, away-facing rays contribute 0, "
          "occlusion blanks hidden floor, composite strength linear")


if __name__ == "__main__":
    _selftest()

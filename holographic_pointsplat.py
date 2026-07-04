"""holographic_pointsplat -- render a cloud of 3D points (particles) into a camera image.

The pipeline SIMULATES particles (holographic_integrate.ParticleSim advances an (N,3) position array and
stashes it in the frame buffer), but nothing ever drew those points to a picture. This module is that missing
renderer: project each world point through the camera to a pixel, then paint it as a small soft round dot
(a Gaussian falloff, so points read as glowing sparks/dust rather than single hard pixels), compositing
front-to-back so nearer points cover farther ones.

The output matches the volume renderer's contract -- (image (H,W,3), alpha (H,W)) -- so the render pipeline can
over-composite a particle layer onto the surface render exactly the way it composites a smoke volume:
    out = points + surface * (1 - alpha)

Design notes / WHY:
  * We reuse the Camera's own view_matrix() and projection_matrix() (OpenGL convention, camera looks down -z),
    so particles line up with the ray-traced surfaces to the pixel. No separate camera math to drift out of sync.
  * Splatting is done with numpy scatter-add over a small stamp per point, not a Python loop over pixels --
    the hot path stays vectorised and cache-friendly.
  * Nearer points are painted LAST (painter's algorithm, far-to-near) so a near spark correctly occludes a far
    one; alpha accumulates with the standard over-operator.
  * Deterministic: no RNG here. Given the same points/colours/camera you get the same image, bit-for-bit.

Everything is NumPy + stdlib. No learned weights, no external deps.
"""

import numpy as np


def _project(points, camera, width, height):
    """Project world points (N,3) to pixel coordinates.

    Returns (px, py, depth, visible):
      px, py  -- pixel centres (float, N,), origin top-left, +x right, +y down
      depth   -- view-space distance in front of the camera (larger = farther); used for painter ordering
      visible -- boolean (N,) mask: in front of the camera AND inside the frame
    The maths is the standard model->clip->NDC->screen chain using the camera's own matrices, so points land
    exactly where the ray tracer would put the same surface.
    """
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    V = camera.view_matrix()                                  # world -> camera (view) space
    PR = camera.projection_matrix()                           # view -> clip space

    hom = np.concatenate([P, np.ones((len(P), 1))], axis=1)   # (N,4) homogeneous world points
    view = hom @ V.T                                          # (N,4) in camera space; camera looks down -z
    depth = -view[:, 2]                                       # distance in FRONT of the camera (view -z is forward)
    clip = view @ PR.T                                        # (N,4) clip space
    w = clip[:, 3]
    in_front = w > 1e-6                                       # only points genuinely in front project sensibly
    w_safe = np.where(in_front, w, 1.0)                       # avoid divide-by-zero for the culled ones
    ndc = clip[:, :3] / w_safe[:, None]                       # perspective divide -> [-1,1] cube

    # NDC (-1..1) -> pixel centres. +x is right; NDC +y is up, so screen y flips.
    px = (ndc[:, 0] * 0.5 + 0.5) * (width - 1)
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (height - 1)

    inside = (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    visible = in_front & inside
    return px, py, depth, visible


def splat_points(points, camera, width, height, colors=None, radius_px=2.0, intensity=1.0,
                 depth_fade=None, background=(0.0, 0.0, 0.0)):
    """Render an (N,3) point cloud to (image (H,W,3), alpha (H,W)).

    Parameters
      points     : (N,3) world positions of the particles.
      camera     : a holographic_render.Camera (uses its view/projection matrices).
      width,height : output pixel size.
      colors     : (N,3) per-point rgb, or a single (3,) rgb for all, or None -> white.
      radius_px  : soft dot radius in pixels (the Gaussian's ~1-sigma; the stamp spans a few sigma).
      intensity  : overall brightness multiplier on each dot's contribution.
      depth_fade : None, or (near, far) -- linearly fade a point's alpha from 1 at `near` to ~0 at `far`
                   view distance, so a receding spark dims (a cheap fog-on-particles). Off by default.
      background : rgb the image is cleared to before compositing (usually black; the caller composites over it).

    Returns image and alpha ready for the standard over-operator against a surface render.
    """
    H, W = int(height), int(width)
    img = np.zeros((H, W, 3), float) + np.asarray(background, float)
    alpha = np.zeros((H, W), float)

    P = np.asarray(points, float)
    if len(P) == 0:
        return img, alpha

    # --- colours: broadcast a single rgb, default to white ---
    if colors is None:
        col = np.ones((len(P), 3), float)
    else:
        col = np.asarray(colors, float)
        if col.ndim == 1:
            col = np.tile(col, (len(P), 1))

    px, py, depth, visible = _project(P, camera, W, H)
    if not visible.any():
        return img, alpha

    # keep only the visible points, and their colours/depths
    px, py, depth, col = px[visible], py[visible], depth[visible], col[visible]

    # --- per-point alpha, optionally faded by depth so far sparks dim ---
    pt_alpha = np.full(len(px), float(intensity))
    if depth_fade is not None:
        near, far = float(depth_fade[0]), float(depth_fade[1])
        fade = 1.0 - np.clip((depth - near) / max(far - near, 1e-6), 0.0, 1.0)
        pt_alpha = pt_alpha * fade

    # --- painter's algorithm: paint FAR points first so NEAR ones composite on top ---
    order = np.argsort(-depth)                                # descending depth = far -> near
    px, py, col, pt_alpha = px[order], py[order], col[order], pt_alpha[order]

    # --- the soft round stamp: a small Gaussian dot, computed once and reused for every point ---
    r = max(1, int(np.ceil(radius_px * 2.5)))                 # stamp half-size: a few sigma of the Gaussian
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    sigma = max(radius_px, 0.5)
    stamp = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))   # (2r+1, 2r+1) falloff, 1.0 at the centre

    # --- composite each point's stamp with the over-operator, near points last ---
    # This loop is over POINTS (usually a few hundred), not pixels; each step is a vectorised stamp blend.
    ix = np.round(px).astype(int)
    iy = np.round(py).astype(int)
    for k in range(len(ix)):
        cx, cy = ix[k], iy[k]
        # the stamp's footprint, clipped to the image bounds
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        # matching sub-window of the stamp
        sx0, sx1 = x0 - (cx - r), (x0 - (cx - r)) + (x1 - x0)
        sy0, sy1 = y0 - (cy - r), (y0 - (cy - r)) + (y1 - y0)
        s = stamp[sy0:sy1, sx0:sx1] * pt_alpha[k]             # this dot's coverage in the footprint
        a_dst = alpha[y0:y1, x0:x1]                           # coverage already there (nearer? no -- we go far->near)
        # over-operator: new = src + dst*(1-src_alpha); here src is this dot (nearer, painted later)
        keep = 1.0 - s
        img[y0:y1, x0:x1] = col[k] * s[..., None] + img[y0:y1, x0:x1] * keep[..., None]
        alpha[y0:y1, x0:x1] = s + a_dst * keep

    return img, np.clip(alpha, 0.0, 1.0)


def _selftest():
    """Points project where expected, occlude by depth, and the (image,alpha) contract holds."""
    from holographic_render import Camera
    cam = Camera(eye=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), fov_deg=45.0, aspect=1.0)

    # a single point at the origin should land at the image centre
    img, alpha = splat_points(np.array([[0.0, 0.0, 0.0]]), cam, 64, 64, colors=(1.0, 1.0, 1.0), radius_px=2.0)
    cy, cx = np.unravel_index(np.argmax(alpha), alpha.shape)
    assert abs(cy - 32) <= 2 and abs(cx - 32) <= 2, (cy, cx)
    assert 0.0 <= alpha.min() and alpha.max() <= 1.0

    # a point behind the camera must not draw
    img2, alpha2 = splat_points(np.array([[0.0, 0.0, 5.0]]), cam, 64, 64)
    assert alpha2.max() == 0.0

    # nearer point (brighter/on top): two points same screen spot, near one red, far one blue -> centre reads red
    pts = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])       # first is nearer the eye at z=3
    cols = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    img3, a3 = splat_points(pts, cam, 64, 64, colors=cols, radius_px=2.0)
    centre = img3[32, 32]
    assert centre[0] > centre[2], centre                     # red (near) dominates blue (far)

    # depth fade dims a far point
    img4, a4 = splat_points(np.array([[0.0, 0.0, -2.0]]), cam, 64, 64, depth_fade=(1.0, 4.0))
    img5, a5 = splat_points(np.array([[0.0, 0.0, 0.0]]), cam, 64, 64, depth_fade=(1.0, 4.0))
    assert a4.max() < a5.max()                                # farther point is fainter

    print("OK: holographic_pointsplat self-test passed")


if __name__ == "__main__":
    _selftest()

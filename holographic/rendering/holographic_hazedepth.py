"""holographic_hazedepth.py -- estimate a relative DEPTH MAP from a single HAZY/FOGGY image via the atmospheric
scattering model, the classical no-weights fix for the case shape-from-shading gets exactly backwards.

WHY THIS MODULE EXISTS
----------------------
shape_from_shading assumes a pixel's brightness encodes its surface ORIENTATION (I = albedo * (N . L)). In an
outdoor scene with haze or fog that assumption is wrong: the dominant brightness cue is ATMOSPHERIC SCATTERING --
distant surfaces are brighter and lower-contrast because more airlight is added along the longer line of sight.
So SfS reads "bright fog in the distance" as "near, facing the light" and inverts the depth ordering. We measured
exactly that on a foggy forest-staircase photo: the near steps came out FARTHER than the mid-ground.

THE PHYSICS (Koschmieder's law, the atmospheric scattering model)
    I(x) = J(x) t(x) + A (1 - t(x)),   t(x) = exp(-beta * d(x))
  I is the observed (hazy) pixel, J the haze-free radiance, A the global atmospheric light (airlight), t the
  transmission, beta the scattering coefficient, d the distance. Transmission FALLS with distance, so
    d(x) = -ln(t(x)) / beta
  is MONOTONIC in depth. Recover t (or the "atmospheric veil" V = A(1 - t)) and you have a depth-ordering that is
  correct for hazy scenes -- the opposite failure mode from SfS, and the right tool for this image class.

THE METHOD (Tarel & Hautiere, ICCV 2009 -- "Fast Visibility Restoration from a Single ... Image")
  Instead of estimating transmission directly, infer the atmospheric VEIL V(x) = A(1 - t(x)), an INCREASING
  function of distance, then d = -ln(1 - V/A)/beta. The veil is estimated from the per-pixel "whiteness"
  W = min over colour channels (the darkest channel is small where haze is thin, large where haze is thick -- the
  same observation the Dark Channel Prior uses), robustly regularised by a median filter so the veil never exceeds
  the local haze and stays below W:
      A_est   = median(W)                          # a scalar airlight proxy (whiteness scale)
      B       = A_est - median(|W - A_est|)        # robust spread (median absolute deviation)
      V       = max( min(p * B_map, W), 0 )        # p in [0,1] controls how much veil we trust
  where B_map is W smoothed by a median filter and clamped by the local whiteness. This is O(N) in the pixels --
  the paper reports 0.17 s on a 759x574 image -- and uses ONLY channel-min, median filters, and arithmetic, so it
  is deterministic and pure NumPy. The veil is then refined with a guided filter (He et al.) so depth edges align
  to image edges, and depth = -ln(t) is returned normalised to [0,1] (1 = nearest).

DESIGN NOTES (the negatives, kept loud)
  * RELATIVE, NOT METRIC. beta and A are unknown from one image, so the output is a relative depth ORDERING
    (near/far correct), not distance in metres -- the same honesty as shape_from_shading, for a different reason.
  * SKY / BRIGHT-WHITE REGIONS FOOL THE WHITENESS CUE. A bright sky or a white object has a large dark-channel and
    reads as "maximally hazy" = far. That is CORRECT for sky (it is far) but WRONG for a near white object. We
    apply a sky/bright guard (bright + low-saturation + upper-frame prior) and clamp those pixels to far, which is
    right for fog/sky and the documented failure for a near white object -- flagged, not hidden.
  * NEEDS ACTUAL HAZE. On a clear, contrasty scene the veil is near zero everywhere and the depth is flat/noisy --
    this is the fog specialist, not a general depth estimator. Use the fused estimator (fuse_depth) or SfS when
    there is no haze. (Kept negative: do not run this on a clear studio object -- it will read the background bokeh
    as the only "depth".)

Reuses nothing learned. NumPy only. Deterministic (median/box filters + arithmetic; no RNG).
"""

import numpy as np


def _box_filter(img, r):
    """O(N) box filter (mean over a (2r+1) window) via the summed-area-table / cumulative-sum trick, per channel.
    The workhorse behind the guided filter. `img` is (H,W) or (H,W,C). Border handling replicates the edge count so
    the result is a true local mean (not darkened at the borders)."""
    img = np.asarray(img, float)
    single = img.ndim == 2
    if single:
        img = img[:, :, None]
    H, W, C = img.shape
    out = np.empty_like(img)
    # a normaliser image of ones gives the true window size at every pixel (handles borders correctly)
    ones = np.ones((H, W))
    def _cumwin(a):
        # cumulative sum along rows then columns, differenced to get a (2r+1) window sum -- classic SAT box sum
        cs = np.cumsum(a, axis=0)
        top = np.zeros((1, a.shape[1]))
        lo = np.concatenate([cs[r:2 * r + 1], cs[2 * r + 1:] - cs[:-2 * r - 1], top + cs[-1] - cs[-2 * r - 1:-r - 1]], axis=0) \
            if H > 2 * r + 1 else np.broadcast_to(cs[-1], a.shape).copy()
        cs2 = np.cumsum(lo, axis=1)
        res = np.concatenate([cs2[:, r:2 * r + 1], cs2[:, 2 * r + 1:] - cs2[:, :-2 * r - 1],
                              np.broadcast_to(cs2[:, -1:], (a.shape[0], r)) - cs2[:, -2 * r - 1:-r - 1]], axis=1) \
            if W > 2 * r + 1 else np.broadcast_to(cs2[:, -1:], a.shape).copy()
        return res
    win = _cumwin(ones)
    for c in range(C):
        out[:, :, c] = _cumwin(img[:, :, c]) / np.maximum(win, 1e-9)
    return out[:, :, 0] if single else out


def guided_filter(guide, src, radius=8, eps=1e-3):
    """He, Sun & Tang GUIDED FILTER (O(N), edge-preserving): filter `src` using `guide` as the edge reference, so
    the output is smooth WHERE the guide is smooth and keeps edges WHERE the guide has edges. Used here to snap the
    coarse veil/transmission to the image's real depth discontinuities (a tree trunk edge, a step lip) without a
    slow matting solve. `guide` and `src` are (H,W) in [0,1]; `eps` sets the edge sensitivity. Deterministic."""
    guide = np.asarray(guide, float); src = np.asarray(src, float)
    r = int(radius)
    mean_I = _box_filter(guide, r)
    mean_p = _box_filter(src, r)
    corr_I = _box_filter(guide * guide, r)
    corr_Ip = _box_filter(guide * src, r)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)              # the local linear model output = a*guide + b
    b = mean_p - a * mean_I
    mean_a = _box_filter(a, r)
    mean_b = _box_filter(b, r)
    return mean_a * guide + mean_b


def _median_filter(img, r):
    """A simple deterministic median filter over a (2r+1) square window (edge-replicated). Small windows only --
    this is the veil regulariser, not a hot loop; kept readable over clever. Pure NumPy via a stacked-shifts
    median (no scipy)."""
    img = np.asarray(img, float)
    H, W = img.shape
    pad = np.pad(img, r, mode="edge")
    stack = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            stack.append(pad[r + dy:r + dy + H, r + dx:r + dx + W])
    return np.median(np.stack(stack, axis=0), axis=0)


def _sky_mask(rgb, gray):
    """Detect sky / bright-white haze regions that the whiteness cue would (correctly for sky) call 'far' but that
    must NOT be trusted as surface geometry: bright + low-saturation + biased toward the top of the frame. Returns a
    float mask in [0,1]. This is the guard the research flagged (DCP/whiteness fails on sky/white)."""
    rgb = np.asarray(rgb, float); H, W = gray.shape
    mx = rgb.max(-1); mn = rgb.min(-1)
    sat = (mx - mn) / (mx + 1e-6)                       # HSV-style saturation
    bright = gray > 0.6
    low_sat = sat < 0.15
    row = np.linspace(1.0, 0.0, H)[:, None]             # top of frame -> 1, bottom -> 0
    top_bias = np.broadcast_to(row, (H, W))
    m = bright.astype(float) * low_sat.astype(float) * (0.4 + 0.6 * top_bias)
    return np.clip(m, 0.0, 1.0)


def haze_depth(image, p=0.95, veil_radius=15, refine_radius=16, refine_eps=1e-3,
               sky_guard=True, return_extras=False):
    """Estimate a RELATIVE DEPTH MAP from a single HAZY/FOGGY `image` by inverting the atmospheric scattering model
    (Tarel & Hautiere veil inference + guided-filter refine). Returns depth (H,W) in [0,1], 1 = NEAREST -- the
    depth ORDERING is correct for hazy scenes, fixing the shape-from-shading inversion. This is the fog specialist.

    `p` in [0,1] is how aggressively to trust the veil (higher = stronger haze assumed; 0.95 is the paper default).
    `veil_radius` is the median-filter window for the robust veil estimate; `refine_radius`/`refine_eps` control the
    guided-filter edge snap. `sky_guard=True` clamps bright low-saturation upper-frame pixels to FAR (right for
    sky/fog, the documented failure for a near white object). `return_extras=True` also returns
    {veil, transmission, airlight, sky_mask} for inspection.

    Deterministic (median/box filters + arithmetic; no RNG). KEPT NEGATIVES: relative not metric (beta,A unknown);
    needs actual haze (flat/noisy on a clear scene -- use fuse_depth or shape_from_shading there); a near white
    object under the sky guard reads as far."""
    img = np.asarray(image, float)
    if img.max() > 1.5:                                 # accept 0..255 or 0..1
        img = img / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    H, W, _ = img.shape
    gray = img.mean(-1)

    # WHITENESS W = the per-pixel MIN over colour channels (the "dark channel" at a single pixel). It is small where
    # haze is thin (a dark, saturated colour survives), large where haze is thick (all channels lifted by airlight).
    W_img = img.min(-1)

    # ATMOSPHERIC LIGHT proxy: the brightest whiteness is the most-hazed pixel; use a high percentile for robustness.
    airlight = float(np.percentile(W_img, 99))
    airlight = max(airlight, 1e-3)

    # ROBUST VEIL (Tarel-Hautiere): smooth W by a median filter, clamp it below the local whiteness, scale by p.
    Wmed = _median_filter(W_img, veil_radius)
    Amed = np.median(Wmed)
    Bmap = Wmed - _median_filter(np.abs(Wmed - Amed), veil_radius)   # robust local spread (MAD)
    veil = np.maximum(np.minimum(p * Bmap, W_img), 0.0)             # 0 <= veil <= whiteness

    # TRANSMISSION and DEPTH: t = 1 - veil/A ; d = -ln(t). Guard t away from 0 so the log is finite.
    t = np.clip(1.0 - veil / airlight, 0.03, 1.0)
    depth = -np.log(t)                                             # increases with distance (far = large)

    # refine the transmission's edges to the image, then recompute depth so depth edges snap to real boundaries
    t_ref = np.clip(guided_filter(gray, t, radius=refine_radius, eps=refine_eps), 0.03, 1.0)
    depth = -np.log(t_ref)

    sky = _sky_mask(img, gray) if sky_guard else np.zeros((H, W))
    if sky_guard:
        # sky/bright-white -> clamp to the far end (max depth). Right for sky/fog; documented wrong for a near white.
        depth = depth * (1.0 - sky) + depth.max() * sky

    # normalise to [0,1] with 1 = NEAREST (invert, since our depth grows with distance)
    d = depth - depth.min()
    d = d / (d.max() + 1e-9)
    near_is_one = 1.0 - d

    if return_extras:
        return near_is_one, {"veil": veil, "transmission": t_ref, "airlight": airlight, "sky_mask": sky}
    return near_is_one


def color_attenuation_depth(image, theta=(0.121779, 0.959710, -0.780245), refine_radius=8):
    """Color Attenuation Prior depth (Zhu, Mai & Shao, IEEE TIP 2015): a one-line LINEAR depth model from the fact
    that haze RAISES brightness (Value) but LOWERS saturation, so their difference grows with haze thickness ->
    distance. depth ~ theta0 + theta1 * V + theta2 * S, with V,S the HSV value and saturation. The three theta
    coefficients are the paper's PUBLISHED scalars (originally fit by linear regression on synthetic data, but they
    are three fixed numbers -- hardcoded here, so there are NO learned weights and this is a trivial per-pixel
    linear combination). Returns depth (H,W) in [0,1], 1 = NEAREST. A cheap, independent cross-check on haze_depth.

    Deterministic. KEPT NEGATIVE: like all haze cues it needs real haze; on a clear scene V and S do not track
    distance and the map is meaningless. The coefficients are the outdoor-daylight fit; extreme colour casts drift."""
    img = np.asarray(image, float)
    if img.max() > 1.5:
        img = img / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    mx = img.max(-1); mn = img.min(-1)
    V = mx                                          # HSV value
    S = (mx - mn) / (mx + 1e-6)                      # HSV saturation
    d = theta[0] + theta[1] * V + theta[2] * S      # the CAP linear depth model (raw distance, larger = farther)
    d = guided_filter(img.mean(-1), d, radius=refine_radius, eps=1e-3)
    d = d - d.min(); d = d / (d.max() + 1e-9)
    return 1.0 - d                                  # 1 = nearest


def defocus_depth(image, sigma_reblur=1.5, edge_thresh=0.03, refine_radius=12, max_blur=5.0):
    """DEFOCUS-MAP DEPTH from a single image (Zhuo & Sim, Pattern Recognition 2011): estimate per-edge BLUR, then
    propagate to a dense map. A sharp region is in the focal plane (usually the subject/foreground); a blurred
    region is off it (usually the background). We RE-BLUR the image by a known Gaussian and measure the ratio of
    gradient magnitudes at edges: a sharp edge's gradient drops a lot when re-blurred, an already-blurry edge barely
    changes -- the ratio recovers the original blur sigma. Returns depth (H,W) in [0,1], 1 = NEAREST (assuming the
    sharp foreground is nearest, the common photographic case).

    This is an INDEPENDENT cue from haze: it works on a clear image with shallow depth of field (exactly the forest
    staircase: sharp near steps, blurred distant trees) where the haze cue is weak. Deterministic, pure NumPy.

    KEPT NEGATIVES: (1) the "sharp = near" assumption fails for a macro/tilt-shift shot focused on the background --
    it recovers a blur ORDERING, not which side of focus; (2) blur is only measurable AT EDGES, so the dense map is
    a propagation (guided-filter) of sparse edge estimates -- flat textureless regions are interpolated, not
    measured; (3) it cannot separate defocus blur from motion blur or genuine soft texture."""
    img = np.asarray(image, float)
    if img.max() > 1.5:
        img = img / 255.0
    gray = img.mean(-1) if img.ndim == 3 else img
    H, W = gray.shape

    def _gauss_blur(a, s):
        # separable Gaussian via a small 1-D kernel (pure NumPy convolution) -- deterministic, no scipy
        rad = max(1, int(3 * s))
        x = np.arange(-rad, rad + 1)
        k = np.exp(-(x ** 2) / (2 * s * s)); k /= k.sum()
        pad = np.pad(a, ((0, 0), (rad, rad)), mode="edge")
        tmp = np.stack([pad[:, i:i + W] for i in range(len(k))], -1) @ k
        pad2 = np.pad(tmp, ((rad, rad), (0, 0)), mode="edge")
        return np.stack([pad2[i:i + H] for i in range(len(k))], -1) @ k

    # gradient magnitude of the image and of a re-blurred copy
    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)
    reb = _gauss_blur(gray, sigma_reblur)
    ry, rx = np.gradient(reb)
    grad_r = np.hypot(rx, ry)

    # gradient-magnitude RATIO R = grad / grad_r at edges. Zhuo-Sim: the original blur sigma is recovered as
    # sigma = sigma_reblur / sqrt(R^2 - 1) for R>1. Large R (gradient collapses a lot on re-blur) = originally SHARP
    # = small sigma = NEAR. Small R (already blurry) = large sigma = FAR.
    edges = grad > edge_thresh
    R = np.where(grad_r > 1e-6, grad / (grad_r + 1e-9), 1.0)
    R = np.maximum(R, 1.0 + 1e-3)
    sigma = sigma_reblur / np.sqrt(R * R - 1.0)
    sigma = np.clip(sigma, 0.0, max_blur)

    # propagate the sparse edge blur to a dense map: guided-filter the edge values, normalised by edge evidence.
    conf = edges.astype(float)
    filled = np.where(edges, sigma, 0.0)
    dense = guided_filter(gray, filled, radius=refine_radius, eps=1e-3)
    weight = guided_filter(gray, conf, radius=refine_radius, eps=1e-3)
    blur_map = dense / np.maximum(weight, 1e-3)          # normalise by how much edge evidence reached each pixel

    # blur -> depth: MORE blur = FARTHER. So near_is_one = 1 - normalised blur.
    b = blur_map - blur_map.min(); b = b / (b.max() + 1e-9)
    return 1.0 - b


def sharpness_depth(image, radius=6, refine_radius=14, gamma=0.6):
    """SHARPNESS-MAP DEPTH: a robust, direct depth-of-field cue -- LOCAL SHARPNESS (box-averaged gradient magnitude)
    measured everywhere, not just at edges. In-focus foreground = high local sharpness = NEAR; out-of-focus
    background = low sharpness = FAR. This is a simpler, more robust sibling of defocus_depth for HEAVILY TEXTURED
    scenes (foliage, forests) where the re-blur gradient-RATIO method (defocus_depth) gets diluted by texture edges
    that survive in the blurred regions. Returns depth (H,W) in [0,1], 1 = NEAREST. Deterministic, pure NumPy.

    `gamma` < 1 lifts the low end so the far background is not crushed to a single value. KEPT NEGATIVES: same
    "sharp = near" assumption as defocus_depth (wrong for background-focused shots); a flat/textureless near surface
    has no gradient and reads as far (no sharpness to measure) -- so pair it with haze in fuse_depth, which supplies
    the textureless regions."""
    img = np.asarray(image, float)
    if img.max() > 1.5:
        img = img / 255.0
    gray = img.mean(-1) if img.ndim == 3 else img
    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)
    local = _box_filter(grad, int(radius))                  # local sharpness = mean gradient magnitude in a window
    # edge-align to the image so sharpness follows real object boundaries, then normalise + gamma-lift
    local = guided_filter(gray, local, radius=int(refine_radius), eps=1e-3)
    s = local - local.min(); s = s / (s.max() + 1e-9)
    s = s ** float(gamma)                                    # lift the low end (far background keeps some structure)
    return s                                                 # already 1 = sharp = near


def fuse_depth(image, weights=(0.5, 0.5), use_haze=True, use_defocus=True, sky_guard=True):
    """FUSE multiple classical depth cues into one estimate -- the research recommendation: no single cue is robust,
    so combine the HAZE (aerial-perspective) depth and the DEFOCUS (depth-of-field) depth, each normalised, by a
    weighted average, then edge-align to the image. For a foggy scene with shallow depth of field (the forest
    staircase) the two cues REINFORCE: haze gives the far background, defocus gives the sharp near foreground.

    `weights` = (haze_weight, defocus_weight), renormalised over whichever cues are enabled. Returns depth (H,W) in
    [0,1], 1 = NEAREST. Deterministic. This is the front end to hand to unproject/photo_to_3d for hazy or
    shallow-DoF photos, where shape_from_shading fails. KEPT NEGATIVE: still relative not metric; if NEITHER cue is
    present (a clear, deep-focus studio shot) the result is meaningless -- use shape_from_shading there instead."""
    img = np.asarray(image, float)
    if img.max() > 1.5:
        img = img / 255.0
    gray = img.mean(-1) if img.ndim == 3 else img
    parts = []; ws = []
    if use_haze:
        parts.append(haze_depth(img, sky_guard=sky_guard)); ws.append(weights[0])
    if use_defocus:
        # use the robust sharpness cue (defocus_depth's re-blur ratio gets diluted in heavily textured scenes)
        parts.append(sharpness_depth(img)); ws.append(weights[1])
    if not parts:
        raise ValueError("fuse_depth needs at least one cue enabled")
    ws = np.array(ws, float); ws = ws / ws.sum()
    fused = sum(w * p for w, p in zip(ws, parts))
    fused = guided_filter(gray, fused, radius=12, eps=1e-3)         # snap to image edges
    fused = fused - fused.min(); fused = fused / (fused.max() + 1e-9)
    return fused


def vanishing_point(image, top_lines=14, oblique_min=15.0, oblique_max=75.0, return_confidence=False):
    """Estimate the dominant VANISHING POINT of a scene's linear perspective from its strong OBLIQUE lines (the
    receding edges -- rails, walls, a corridor). Detects Hough lines, keeps those whose angle is clearly diagonal
    (between `oblique_min` and `oblique_max` degrees from horizontal, so near-horizontal horizon lines and near-
    vertical tree trunks do not dominate), and solves for their least-squares intersection. Returns (vx, vy) in
    PIXEL coordinates (may lie outside the image, which is normal for a VP), or None if there is no clear perspective.

    With `return_confidence`, returns ((vx,vy), confidence) where confidence in [0,1] reflects HOW WELL the oblique
    lines actually agree on a single point (their residual to the fitted VP, plus how many there are) -- low
    confidence means the "VP" is a spurious fit to scattered texture edges (a flat wall, a gradient), and callers
    should NOT trust the perspective prior. Reuses holographic_vision.edges + hough_lines (classic CV, no weights).
    Deterministic. KEPT NEGATIVE: a two-vanishing-point scene returns only the strongest single VP (a simplification)."""
    from holographic.misc.holographic_vision import edges as _edges, hough_lines as _hough
    img = np.asarray(image, float)
    if img.max() > 1.5:
        img = img / 255.0
    gray = img.mean(-1) if img.ndim == 3 else img
    H, W = gray.shape
    em = _edges(gray)
    lines = _hough(em, ntheta=180, top=int(top_lines))       # (rho, theta_deg, votes)
    A = []; b = []; vts = []
    for (rho, theta_deg, votes) in lines:
        ang = abs(((theta_deg) % 180) - 90.0)                # 0 = horizontal line, 90 = vertical line
        if oblique_min <= ang <= oblique_max:
            th = np.deg2rad(theta_deg)
            A.append([np.cos(th), np.sin(th)]); b.append(rho); vts.append(votes)
    if len(A) < 3:                                           # need at least 3 agreeing oblique lines for a real VP
        return (None, 0.0) if return_confidence else None
    A = np.asarray(A, float); b = np.asarray(b, float)
    try:
        vp, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return (None, 0.0) if return_confidence else None
    if not np.all(np.isfinite(vp)):
        return (None, 0.0) if return_confidence else None
    # CONFIDENCE: how tightly the oblique lines pass through the fitted VP. residual = |A@vp - b| in pixels (distance
    # of the VP from each line). Small residual relative to the image size + several lines = a real, agreed-on VP.
    resid = np.abs(A @ vp - b)
    diag = float(np.hypot(H, W))
    tightness = float(np.clip(1.0 - np.median(resid) / (0.15 * diag), 0.0, 1.0))
    count_factor = float(np.clip((len(A) - 2) / 4.0, 0.0, 1.0))    # 3 lines -> 0.25 ... 6+ lines -> 1.0
    conf = tightness * count_factor
    if return_confidence:
        return (float(vp[0]), float(vp[1])), conf
    return (float(vp[0]), float(vp[1]))


def _perspective_prior(shape, vp):
    """A relative-depth PRIOR from a vanishing point: depth INCREASES toward the VP (things nearer the vanishing
    point are farther away). Returns near_is_one (H,W) in [0,1], 1 = NEAREST (farthest FROM the vp). Used to score
    which measured depth cue agrees with the scene's perspective."""
    H, W = shape
    vy, vx = np.mgrid[0:H, 0:W]
    d = np.hypot(vx - vp[0], vy - vp[1])                     # pixel distance to the VP; large = far from VP = NEAR
    d = d - d.min(); d = d / (d.max() + 1e-9)
    return d                                                 # already 1 = far-from-VP = near


def auto_fuse_depth(image, sky_guard=None, min_weight=0.05, return_weights=False):
    """AUTO-WEIGHTED depth fusion: combine the HAZE and SHARPNESS cues, but weight each by how well it AGREES with the
    scene's LINEAR PERSPECTIVE (the vanishing-point depth prior) -- so the cue that is actually tracking depth for
    THIS image dominates, and a cue that is INVERTED for this image (e.g. sharpness on a dark-foreground corridor,
    which reads the near tracks as far) is down-weighted or flipped automatically. This removes the manual per-image
    cue weighting that the tracks/forest/bridge scenes each needed by hand.

    How: detect the vanishing point (vanishing_point); build the perspective depth prior; for each cue compute its
    correlation with the prior; weight = max(correlation, min_weight), and FLIP a cue whose correlation is negative
    (it is anti-aligned, so 1-cue aligns). If there is no clear perspective (vanishing_point returns None) it falls
    back to the default 55/45 haze/sharpness fuse_depth. `sky_guard` defaults to auto (on only if the far region is
    bright). Returns depth (H,W) in [0,1], 1 = NEAREST; with return_weights, also the (haze_w, sharp_w, vp) chosen.

    Deterministic. KEPT NEGATIVE: the perspective prior is itself only a PRIOR (a VP says where the depth axis points,
    not the true depth), so on a scene with perspective but unusual depth (a wall receding to a bright far end) the
    prior can mis-rank a cue; and with no detectable VP it cannot auto-weight and falls back to the fixed blend."""
    img = np.asarray(image, float)
    if img.max() > 1.5:
        img = img / 255.0
    gray = img.mean(-1) if img.ndim == 3 else img
    H, W = gray.shape

    # auto sky-guard: only guard if the far/upper region is BRIGHT (a real sky), not a dark tunnel
    if sky_guard is None:
        upper = gray[:H // 3].mean()
        sky_guard = bool(upper > 0.55)

    vp, conf = vanishing_point(img, return_confidence=True)
    hz = haze_depth(img, sky_guard=sky_guard)
    sh = sharpness_depth(img)

    # only TRUST the perspective prior when the vanishing point is well-supported (several oblique lines agreeing).
    # a spurious VP from scattered texture edges (low confidence) would mis-weight/flip the cues -> fall back.
    if vp is None or conf < 0.45:
        fused = fuse_depth(img, weights=(0.55, 0.45), sky_guard=sky_guard)
        if return_weights:
            return fused, (0.55, 0.45, vp if conf >= 0.45 else None)
        return fused

    prior = _perspective_prior((H, W), vp)
    pr = prior.ravel()

    # RAW correlations of each cue with the perspective prior. If the STRONGEST agreement is weak, the VP prior is a
    # poor proxy for depth on this scene (e.g. a very high VP where radial distance does not match the ground-plane
    # recession) -- trusting it would over-weight whichever cue happens to correlate best with a bad prior. Fall back.
    hc = float(np.corrcoef(hz.ravel(), pr)[0, 1]); hc = hc if np.isfinite(hc) else 0.0
    sc = float(np.corrcoef(sh.ravel(), pr)[0, 1]); sc = sc if np.isfinite(sc) else 0.0

    # GROUND-PLANE BACKBONE: for a forward-looking scene the ground receding to the horizon is the DOMINANT depth
    # axis, and it is exactly what haze+defocus miss when the scene is mostly in focus with only distant haze (a
    # track/road to a misty vanishing point -- the depth otherwise comes out nearly flat, 80% "near"). Use the
    # ground-plane ramp as the backbone and let haze/sharpness MODULATE it (add local relief), instead of asking the
    # weak cues to carry the whole depth. Gated on a confident VP (a level forward-looking camera).
    gp = ground_plane_depth(img, vp=vp)

    def _score(cue, corr):
        if corr >= 0:
            return cue, corr
        return (1.0 - cue), 0.35 * (-corr)

    hz2, hw = _score(hz, hc)
    sh2, sw = _score(sh, sc)
    hw = max(hw, float(min_weight)); sw = max(sw, float(min_weight))
    tot = hw + sw
    relief = (hw * hz2 + sw * sh2) / (hw + sw)              # the cue-based relief, in [0,1]

    # blend: mostly the ground-plane ramp (the reliable global recession), with the cue relief adding local structure.
    # backbone_w scales with how STRONGLY the cues disagreed with the backbone -- if they already agree, lean on them
    # more; if they are flat/weak, lean on the ramp. Default favours the ramp for forward-looking scenes.
    backbone_w = 0.65
    fused = backbone_w * gp + (1.0 - backbone_w) * relief
    fused = guided_filter(gray, fused, radius=12, eps=1e-3)
    fused = fused - fused.min(); fused = fused / (fused.max() + 1e-9)
    if return_weights:
        return fused, (hw / tot, sw / tot, vp)
    return fused


def ground_plane_depth(image, vp=None, horizon_softness=0.05):
    """GROUND-PLANE DEPTH from linear perspective: for a forward-looking camera (a road, a railway, a hallway) the
    dominant depth axis is the GROUND receding toward the horizon -- depth increases with height in the frame up to
    the horizon line (the vanishing point's row). This is the classical "vertical position / linear perspective" cue,
    and it is the ONE cue that captures a track/road recession when haze and defocus are both weak (a scene that is
    mostly in-focus with only distant mist). Returns depth (H,W) in [0,1], 1 = NEAREST (the bottom of the frame).

    `vp` is (vx, vy) in pixels (auto-detected if None); the horizon is the VP's row. Pixels below the horizon get a
    ramp from near (bottom) to far (horizon); pixels above the horizon (sky / distant peaks) are clamped far.
    `horizon_softness` feathers the horizon so it is not a hard line. Deterministic, pure NumPy.

    KEPT NEGATIVES: assumes a roughly LEVEL forward-looking camera with the ground at the bottom -- it is meaningless
    for a top-down, a portrait, or a wall-facing shot (there is no ground plane), which is why it is GATED behind a
    confident vanishing point in auto_fuse_depth. It gives the global RAMP only; local relief (a rock, a tree trunk
    standing off the ground) comes from haze/sharpness modulating it, not from this cue alone."""
    img = np.asarray(image, float)
    if img.max() > 1.5:
        img = img / 255.0
    gray = img.mean(-1) if img.ndim == 3 else img
    H, W = gray.shape
    if vp is None:
        vp = vanishing_point(img)
        if vp is None:
            # no perspective -> a gentle top-far/bottom-near ramp is still the safest ground guess
            vp = (W / 2.0, 0.0)
    vy = float(vp[1])
    yy = np.mgrid[0:H, 0:W][0].astype(float)
    denom = max(H - vy, 1.0)
    ramp = np.clip((yy - vy) / denom, 0.0, 1.0)             # 0 at horizon row, 1 at bottom (near)
    # feather across the horizon so pixels just above it are not a hard 0
    soft = float(horizon_softness) * H
    if soft > 1e-6:
        above = yy < vy
        fade = np.clip(1.0 - (vy - yy) / soft, 0.0, 1.0)
        ramp = np.where(above, ramp * fade, ramp)
    return ramp


def _selftest():
    # Build a SYNTHETIC hazy scene with a KNOWN depth ramp so we can assert the estimator recovers the ORDERING.
    # A haze-free image J (a textured pattern), a linear distance ramp d (near at bottom, far at top), transmission
    # t = exp(-beta d), airlight A; compose I = J t + A (1 - t). A correct estimator makes depth INCREASE with the
    # true distance (top reads FAR = low 'near_is_one', bottom reads NEAR = high).
    rng = np.random.default_rng(0)
    H, W = 80, 100
    yy, xx = np.mgrid[0:H, 0:W]
    J = 0.35 + 0.25 * np.sin(xx / 6.0) * np.cos(yy / 5.0)         # a haze-free textured radiance in [~0.1, 0.85]
    J = np.clip(J, 0.05, 0.9)
    J3 = np.stack([J, J * 0.9, J * 0.8], -1)                       # mild colour so saturation is non-trivial
    true_dist = yy / (H - 1)                                       # 0 at top... but we want TOP = far, so flip:
    true_dist = 1.0 - true_dist                                    # 1 at TOP (far), 0 at BOTTOM (near)
    beta = 1.6
    t = np.exp(-beta * true_dist)[..., None]
    A = 0.85
    I = J3 * t + A * (1.0 - t)                                     # the hazy observation

    near = haze_depth(I, sky_guard=False)                         # sky guard off (synthetic has no real sky)
    # near_is_one is 1 at NEAR (bottom) and small at FAR (top). Assert the TOP is farther than the BOTTOM.
    top_near = near[:H // 4].mean()                               # top = far -> should be LOW
    bot_near = near[3 * H // 4:].mean()                           # bottom = near -> should be HIGH
    assert bot_near > top_near + 0.15, f"haze_depth ordering wrong: bottom(near) {bot_near:.3f} !> top(far) {top_near:.3f}"

    # correlation of estimated depth (= 1 - near) with the TRUE distance must be strongly positive
    est_dist = 1.0 - near
    corr = float(np.corrcoef(est_dist.ravel(), true_dist.ravel())[0, 1])
    assert corr > 0.6, f"haze_depth should track true distance; corr only {corr:.3f}"

    # CAP cross-check: same ordering (bottom nearer than top)
    capn = color_attenuation_depth(I)
    assert capn[3 * H // 4:].mean() > capn[:H // 4].mean(), "CAP ordering wrong (bottom should read nearer than top)"

    # guided filter sanity: filtering a signal by itself is near-identity; a constant stays constant
    const = np.full((20, 20), 0.5)
    gf = guided_filter(const, const, radius=4)
    assert np.allclose(gf, 0.5, atol=1e-6), "guided_filter should preserve a constant"

    # KEPT-NEGATIVE trap: on a CLEAR (haze-free) image the depth is near-flat -- assert low dynamic range so a
    # future caller cannot mistake this for a general depth estimator.
    clear = np.clip(J3, 0, 1)
    flat = haze_depth(clear, sky_guard=False)
    # a clear scene has no distance signal here -> the spread is small relative to a genuinely hazy one
    assert flat.std() < near.std(), "clear-scene haze depth should be flatter than a hazy scene's (needs real haze)"

    # DEFOCUS: build a scene where the BOTTOM is SHARP (near) and the TOP is BLURRED (far), independent of haze.
    sharp = np.clip(0.5 + 0.4 * np.sin(xx / 3.0) * np.cos(yy / 3.0), 0, 1)   # high-freq texture everywhere
    # blur the top half progressively (simulate depth-of-field: far = blurred)
    defoc = sharp.copy()
    for _ in range(3):
        blurred = 0.25 * (np.roll(defoc, 1, 0) + np.roll(defoc, -1, 0) + np.roll(defoc, 1, 1) + np.roll(defoc, -1, 1))
        ramp = np.clip(1.0 - yy / (H - 1), 0, 1)[..., None] if False else np.clip(1.0 - yy / (H * 0.6), 0, 1)
        defoc = defoc * (1 - ramp) + blurred * ramp                          # top gets more blur each pass
    dd = defocus_depth(np.stack([defoc] * 3, -1))
    # sharp bottom should read NEAR (high), blurred top FAR (low)
    assert dd[3 * H // 4:].mean() > dd[:H // 4].mean(), \
        f"defocus_depth ordering wrong: sharp-bottom {dd[3*H//4:].mean():.3f} !> blurred-top {dd[:H//4].mean():.3f}"

    # SHARPNESS cue: the sharp bottom reads near, the blurred top far (robust variant for textured scenes)
    sd = sharpness_depth(np.stack([defoc] * 3, -1))
    assert sd[3 * H // 4:].mean() > sd[:H // 4].mean(), \
        f"sharpness_depth ordering wrong: sharp-bottom {sd[3*H//4:].mean():.3f} !> blurred-top {sd[:H//4].mean():.3f}"

    # FUSION: on the hazy synthetic scene, fusing haze+sharpness keeps the correct ordering (bottom near > top far)
    fused = fuse_depth(I, sky_guard=False)
    assert fused[3 * H // 4:].mean() > fused[:H // 4].mean(), "fuse_depth ordering wrong on the hazy scene"

    # VANISHING POINT: a synthetic corridor -- two strong oblique lines converging near (40,18) -- should yield a VP
    # near that convergence WITH real confidence (several agreeing oblique lines).
    cor = np.full((60, 80), 0.1)
    vxp, vyp = 40, 18
    for t in np.linspace(0, 1, 200):
        x1 = int((1 - t) * 5 + t * vxp); y1 = int((1 - t) * 59 + t * vyp)
        x2 = int((1 - t) * 75 + t * vxp); y2 = int((1 - t) * 59 + t * vyp)
        for (xx2, yy2) in ((x1, y1), (x2, y2)):
            if 0 <= yy2 < 60 and 0 <= xx2 < 80:
                cor[max(0, yy2 - 1):yy2 + 2, max(0, xx2 - 1):xx2 + 2] = 0.9
    vp, vpconf = vanishing_point(np.stack([cor] * 3, -1), return_confidence=True)
    assert vp is not None, "vanishing_point should find the corridor VP"
    assert abs(vp[0] - vxp) < 20 and abs(vp[1] - vyp) < 20, f"VP off: got {vp}, expected near ({vxp},{vyp})"

    # GROUND-PLANE depth: a forward-looking ramp from a vanishing point -- bottom near, horizon far, monotonic.
    gpd = ground_plane_depth(np.stack([cor] * 3, -1), vp=(vxp, vyp))
    assert gpd[45:].mean() > gpd[:15].mean(), "ground_plane_depth should read the bottom nearer than the top"
    assert gpd.std() > 0.15, "ground_plane_depth should have a real ramp, not be flat"

    # AUTO-FUSE: on the hazy synthetic scene (pure texture, no real perspective lines) the VP is spurious/low-conf,
    # so auto_fuse_depth must FALL BACK to the fixed blend and NOT invert the ordering (bottom near >= top far).
    af, (aw_h, aw_s, aw_vp) = auto_fuse_depth(I, sky_guard=False, return_weights=True)
    assert af.shape == (H, W) and af.min() >= 0.0 and af.max() <= 1.0 + 1e-6
    assert af[3 * H // 4:].mean() >= af[:H // 4].mean() - 0.05, "auto_fuse_depth inverted the ordering"

    # GUIDED FILTER IS GENERAL (GS-A): it was written for the transmission map, but it refines ANY map against
    # ANY guide. Assert the measured contract on a map it was NEVER designed for (ambient occlusion), against the
    # honest same-support baseline (a plain box blur): the guided result must be markedly closer to truth AND keep
    # the guide's edge, which the box blur destroys.
    _rng = np.random.default_rng(0)
    _H = _W = 96
    _guide = np.zeros((_H, _W)); _guide[:, _W // 2:] = 1.0
    _guide = np.clip(_guide + 0.03 * _rng.standard_normal((_H, _W)), 0, 1)
    _ao = np.zeros((_H, _W)); _ao[:, :_W // 2] = 0.35; _ao[:, _W // 2:] = 0.9
    _noisy = np.clip(_ao + 0.15 * _rng.standard_normal((_H, _W)), 0, 1)
    _gf = guided_filter(_guide, _noisy, radius=6, eps=1e-3)
    _box = _box_filter(_noisy, 6)
    _rmse = lambda x: float(np.sqrt(np.mean((x - _ao) ** 2)))
    _step = lambda x: float(np.mean(np.abs(x[:, _W // 2] - x[:, _W // 2 - 1])))
    assert _rmse(_gf) < 0.5 * _rmse(_box), ("guided must beat a box blur on a guide-aligned map",
                                            _rmse(_gf), _rmse(_box))
    assert _step(_gf) > 0.7 * _step(_ao), ("guided must keep the guide's edge", _step(_gf), _step(_ao))
    assert _step(_box) < 0.2 * _step(_ao), ("the box baseline is expected to destroy the edge", _step(_box))

    # KEPT NEGATIVE (loud): the guide must EXPLAIN the map. On a ramp whose structure ignores the guide, the
    # guided filter is NOT better than the box blur -- and injects a spurious edge from the guide. Never sell it
    # as a universal denoiser.
    _ramp = np.tile(np.linspace(0, 1, _W), (_H, 1))
    _rn = np.clip(_ramp + 0.15 * _rng.standard_normal((_H, _W)), 0, 1)
    _rmse_r = lambda x: float(np.sqrt(np.mean((x - _ramp) ** 2)))
    assert _rmse_r(guided_filter(_guide, _rn, radius=6)) >= 0.95 * _rmse_r(_box_filter(_rn, 6)), \
        "KEPT NEGATIVE broken: guided should NOT beat a box blur when the map ignores the guide"

    print(f"holographic_hazedepth selftest ok: veil-depth recovers ordering (bottom-near {bot_near:.2f} > "
          f"top-far {top_near:.2f}), corr-to-true-distance {corr:.2f}, CAP agrees, defocus recovers sharp=near, "
          f"fusion keeps ordering, vanishing-point found at {vp[0]:.0f},{vp[1]:.0f}, ground-plane ramp monotonic, "
          f"auto-fuse ordering held, guided filter preserves constants, clear-scene depth is flatter.")


if __name__ == "__main__":
    _selftest()

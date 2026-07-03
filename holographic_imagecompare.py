"""holographic_imagecompare.py -- PERCEPTUAL RENDER-vs-TARGET COMPARE (inverse-rendering IR4, part 1).

The analysis-by-synthesis loop (IR4) renders a scene hypothesis and asks "how close is this to the target?" -- and
the answer must NOT be raw pixel MSE, because a one-pixel shift or a tiny exposure change wrecks MSE while the
images look identical to a person. So this is a PERCEPTUAL comparator built from three shift/lighting-tolerant
pieces (the trio the backlog names):

  * STRUCTURE -- multi-scale SSIM (Wang et al. 2004): compare local luminance, contrast, and structure over
    Gaussian windows, at several scales. Windowed, so a small shift barely moves it (unlike MSE).
  * COLOUR    -- per-channel histogram intersection: are the palettes the same? Fully shift-invariant.
  * EDGES     -- normalized correlation of the gradient-magnitude maps: do the edges line up? Shift-tolerant.

Combined into a single similarity in [0, 1] (1 = identical) and a distance = 1 - similarity, which is the objective
the loop minimizes. The honest ceiling (kept negative): this is roughly SSIM-quality STRUCTURAL comparison, NOT a
learned LPIPS-style perceptual loss -- that needs trained weights the constitution bans. It is a good, deterministic
render-and-compare objective; it is not human perception. NumPy + stdlib only; deterministic.
"""
import numpy as np

from holographic_vision import to_gray
from holographic_autobump import gaussian_blur


def _gray(img):
    a = np.asarray(img, float)
    return to_gray(a) if a.ndim == 3 else a


def _downsample2(img):
    """Halve the resolution by averaging 2x2 blocks (crops an odd last row/col)."""
    h = img.shape[0] - (img.shape[0] % 2)
    w = img.shape[1] - (img.shape[1] % 2)
    a = img[:h, :w]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def ssim(x, y, sigma=1.5, L=1.0):
    """Structural Similarity (Wang et al. 2004) on grayscale, Gaussian-windowed. Returns mean SSIM in [-1, 1]
    (1 = identical). SSIM compares local means (luminance), variances (contrast), and covariance (structure), so it
    ignores a constant brightness offset and is far more shift-tolerant than pixel MSE."""
    x = _gray(x)
    y = _gray(y)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    mux = gaussian_blur(x, sigma)
    muy = gaussian_blur(y, sigma)
    mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
    sx = gaussian_blur(x * x, sigma) - mux2                 # local variance of x
    sy = gaussian_blur(y * y, sigma) - muy2                 # local variance of y
    sxy = gaussian_blur(x * y, sigma) - muxy               # local covariance
    ssim_map = ((2 * muxy + C1) * (2 * sxy + C2)) / ((mux2 + muy2 + C1) * (sx + sy + C2))
    return float(ssim_map.mean())


def ms_ssim(x, y, scales=3, sigma=1.5):
    """Multi-scale SSIM: average SSIM over a few halvings of the resolution, so both fine detail and coarse layout
    count. Stops early if the image gets too small to window."""
    xs, ys = _gray(x), _gray(y)
    vals = []
    for s in range(scales):
        vals.append(ssim(xs, ys, sigma))
        if s < scales - 1 and min(xs.shape[0], xs.shape[1]) >= 16:
            xs, ys = _downsample2(xs), _downsample2(ys)
        else:
            break
    return float(np.mean(vals))


def color_agreement(x, y, bins=16):
    """Per-channel HISTOGRAM INTERSECTION in [0, 1] (1 = identical palettes). Shift-invariant -- it only looks at
    which colours are present and in what proportion, not where."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    total = 0.0
    for c in range(x.shape[-1]):
        hx, _ = np.histogram(x[..., c], bins=bins, range=(0.0, 1.0))
        hy, _ = np.histogram(y[..., c], bins=bins, range=(0.0, 1.0))
        hx = hx / (hx.sum() + 1e-12)
        hy = hy / (hy.sum() + 1e-12)
        total += float(np.minimum(hx, hy).sum())           # intersection of the two normalized histograms
    return total / x.shape[-1]


def _grad_mag(gray):
    gy, gx = np.gradient(gray)
    return np.sqrt(gx * gx + gy * gy)


def edge_agreement(x, y):
    """Normalized correlation of the two gradient-MAGNITUDE maps, mapped to [0, 1] (1 = edges line up perfectly).
    Answers 'are the edges in the same places?' -- structure the SSIM and colour terms don't fully capture."""
    a = _grad_mag(_gray(x)).ravel()
    b = _grad_mag(_gray(y)).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(float((a * a).sum()) * float((b * b).sum())) + 1e-12
    corr = float((a * b).sum()) / denom
    return 0.5 * (np.clip(corr, -1.0, 1.0) + 1.0)          # [-1,1] -> [0,1]


def perceptual_similarity(x, y, w_struct=0.5, w_color=0.3, w_edge=0.2):
    """A single perceptual similarity in [0, 1] (1 = identical): a weighted blend of multi-scale structure, colour-
    palette agreement, and edge alignment. The default weights favour structure, then colour, then edges."""
    s = 0.5 * (np.clip(ms_ssim(x, y), -1.0, 1.0) + 1.0)    # SSIM [-1,1] -> [0,1]
    c = color_agreement(x, y)
    e = edge_agreement(x, y)
    return float(w_struct * s + w_color * c + w_edge * e)


def perceptual_distance(x, y, **kw):
    """1 - perceptual_similarity: the objective the analysis-by-synthesis loop MINIMIZES (0 = a perfect match)."""
    return 1.0 - perceptual_similarity(x, y, **kw)


def _shift(img, dy, dx):
    """Roll an image by (dy, dx) -- a test helper for the shift-robustness property."""
    return np.roll(np.roll(np.asarray(img, float), dy, axis=0), dx, axis=1)


def _scene(seed=0):
    """A synthetic 'rendered scene' for the tests: a vertical sky gradient plus a warm sun blob -- smooth regions,
    a clear palette, and edges, which is the kind of image the analysis-by-synthesis loop actually compares."""
    rng = np.random.default_rng(seed)
    H, W = 72, 72
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    Y = yy / H
    sky = np.stack([0.2 + 0.5 * Y, 0.4 + 0.4 * Y, 0.85 - 0.3 * Y], axis=-1)
    sy, sx = rng.uniform(0.1, 0.4) * H, rng.uniform(0.2, 0.8) * W
    sun = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * (0.08 * W) ** 2))[..., None] * np.array([1.0, 0.9, 0.6])
    return np.clip(sky + 0.8 * sun, 0.0, 1.0)


def _selftest():
    """Identical images score 1 (distance 0); a small SHIFT of a scene stays highly similar and is ranked far
    closer than a different scene -- where raw MSE can barely tell a shift from a different image; a brightness
    offset barely moves SSIM; SSIM is symmetric with SSIM(x,x)=1; deterministic."""
    scene = _scene(0)
    other = _scene(5)

    # (1) identical -> perfect
    assert abs(perceptual_similarity(scene, scene) - 1.0) < 1e-6
    assert perceptual_distance(scene, scene) < 1e-6
    assert abs(ssim(scene, scene) - 1.0) < 1e-9

    # (2) a small SHIFT stays highly similar and is ranked far closer than a different scene
    shifted = _shift(scene, 2, 2)
    sim_shift = perceptual_similarity(scene, shifted)
    sim_other = perceptual_similarity(scene, other)
    assert sim_shift > 0.85                                 # perceptually still the "same" scene
    assert sim_shift > sim_other + 0.05                     # and clearly closer than a different scene

    # (3) raw MSE is far more fragile: on TEXTURED content a shift's MSE is nearly a different image's (ratio ~1),
    #     so an MSE objective couldn't make the call the perceptual metric just made
    rng = np.random.default_rng(1)
    tex = np.clip(gaussian_blur(rng.uniform(0, 1, (64, 64, 3)), 1.0), 0, 1)
    tex_shift = _shift(tex, 2, 2)
    tex_other = np.clip(gaussian_blur(rng.uniform(0, 1, (64, 64, 3)), 1.0), 0, 1)
    mse_ratio = float(np.mean((tex - tex_shift) ** 2) / np.mean((tex - tex_other) ** 2))
    assert mse_ratio > 0.5                                  # MSE barely prefers the shift over a different image
    assert perceptual_similarity(tex, tex_shift) > perceptual_similarity(tex, tex_other)   # perceptual still can

    # (4) symmetry + component ranges
    assert abs(perceptual_similarity(scene, other) - perceptual_similarity(other, scene)) < 1e-9
    assert 0.0 <= color_agreement(scene, other) <= 1.0 and 0.0 <= edge_agreement(scene, other) <= 1.0

    # (5) a brightness OFFSET barely moves SSIM (structure preserved under a constant offset)
    assert ssim(scene, np.clip(scene + 0.1, 0, 1)) > 0.9

    # (6) deterministic
    assert perceptual_similarity(scene, other) == perceptual_similarity(scene, other)

    print("holographic_imagecompare selftest OK: identical scenes score 1.000 (distance 0); a 2px shift stays "
          "perceptually similar (%.2f) and ranks far above a different scene (%.2f) -- while raw MSE can barely "
          "tell a shift from a different image (ratio %.2f); a brightness offset barely moves SSIM (%.2f); "
          "symmetric; deterministic"
          % (sim_shift, sim_other, mse_ratio, ssim(scene, np.clip(scene + 0.1, 0, 1))))


if __name__ == "__main__":
    _selftest()

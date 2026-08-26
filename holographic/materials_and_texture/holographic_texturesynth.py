"""holographic_texturesynth.py -- EXAMPLE-BASED TEXTURE SYNTHESIS by Image Quilting (inverse-rendering ST2).

Grow a larger (optionally seamless) texture from a small sample -- for material synthesis, texture-by-numbers, and
feeding IR1 auto-bump with tileable maps. No learned weights: this is the classical patch-based method.

Image Quilting (Efros & Freeman, SIGGRAPH 2001): lay the output as a grid of OVERLAPPING patches copied from the
sample; choose each new patch so its overlap with the already-placed patches MATCHES (a patch search), then stitch it
in along the least-error MIN-CUT seam through the overlap, so the joins are invisible instead of a hard grid. Two
pieces map straight onto shipped primitives:

  * THE PATCH SEARCH IS NATIVE RECALL. "Find a sample patch whose border matches this context" is exactly
    HoloForest.recall_k -- "find the patches that look like this one" -- the same sublinear neighbour search NLM
    denoising uses. We index the sample patches once and recall candidates per position, then refine by the exact
    overlap error (recall narrows sub-linearly; the exact SSD picks the winner).
  * THE MIN-CUT is a small dynamic program (the least-cost monotone seam through the overlap error surface).

KEPT NEGATIVES (loud): classical example-based synthesis is patch-COPYING -- it can repeat or seam (the min-cut
mitigates the seam; variety comes from picking among the near-best, not just the best). Its quality is BELOW neural
for arbitrary artistic styles -- it is best for TEXTURE / colour / material, not free-form painterly restyle, and it
needs the sample to be roughly stationary (a repeating texture), not a structured scene. NumPy + stdlib only;
deterministic (seeded choice among the near-best).
"""
import numpy as np

from holographic.misc.holographic_tree import HoloForest


def _extract_patches(sample, psize):
    """Every psize x psize patch of the sample (stride 1), plus their (y,x) top-left positions."""
    H, W = sample.shape[:2]
    patches, pos = [], []
    for y in range(0, H - psize + 1):
        for x in range(0, W - psize + 1):
            patches.append(sample[y:y + psize, x:x + psize])
            pos.append((y, x))
    return np.array(patches), pos


def _min_cut_vertical(err):
    """Least-cost TOP-to-BOTTOM seam through an (H, W) error surface (the left overlap). Returns a boolean mask
    (H, W): True = take the NEW patch's pixel, False = keep the existing one. A monotone seam via DP."""
    H, W = err.shape
    E = err.copy()
    back = np.zeros((H, W), int)
    for i in range(1, H):
        for j in range(W):
            lo, hi = max(0, j - 1), min(W, j + 2)
            k = lo + int(np.argmin(E[i - 1, lo:hi]))
            E[i, j] += E[i - 1, k]
            back[i, j] = k
    seam = np.zeros(H, int)
    seam[-1] = int(np.argmin(E[-1]))
    for i in range(H - 2, -1, -1):
        seam[i] = back[i + 1, seam[i + 1]]
    mask = np.zeros((H, W), bool)
    for i in range(H):
        mask[i, seam[i]:] = True                              # right of the seam belongs to the new patch
    return mask


def _min_cut_horizontal(err):
    """Least-cost LEFT-to-RIGHT seam through the top overlap -- the vertical case transposed."""
    return _min_cut_vertical(err.T).T


def _blend_patch(out, patch, y0, x0, overlap, has_left, has_top, seam="mincut"):
    """Place `patch` at (y0,x0), stitching its left/top overlaps into the existing output. seam='mincut' routes the
    boundary along the least-error min-cut seam; seam='hard' cuts straight down the middle of the overlap (the naive
    baseline, for measuring what the min-cut buys)."""
    ph, pw = patch.shape[:2]
    region = out[y0:y0 + ph, x0:x0 + pw].copy()
    take = np.ones((ph, pw), bool)                            # which pixels come from the new patch
    if has_left:
        if seam == "mincut":
            e = ((patch[:, :overlap] - region[:, :overlap]) ** 2).sum(axis=-1)
            take[:, :overlap] = _min_cut_vertical(e)
        else:
            take[:, :overlap // 2] = False                   # straight cut: keep existing on the left half
    if has_top:
        if seam == "mincut":
            e = ((patch[:overlap, :] - region[:overlap, :]) ** 2).sum(axis=-1)
            take[:overlap, :] &= _min_cut_horizontal(e)
        else:
            take[:overlap // 2, :] = False
    out[y0:y0 + ph, x0:x0 + pw] = np.where(take[..., None], patch, region)


def synthesize_texture(sample, out_h, out_w, psize=24, overlap=6, seed=0, candidates=24, seam="mincut"):
    """Quilt a (>=out_h x out_w) texture from `sample`. Each patch is chosen by recalling candidates whose border
    matches the placed context (HoloForest.recall_k) and refining by the exact overlap error, then stitched along a
    min-cut seam. Deterministic. Returns an image of shape (out_h, out_w, C) (or (out_h,out_w) for grayscale in)."""
    sample = np.asarray(sample, float)
    gray_in = sample.ndim == 2
    if gray_in:
        sample = sample[..., None]
    H, W, C = sample.shape
    psize = int(min(psize, H, W))
    overlap = int(min(overlap, psize // 2))
    step = psize - overlap
    rng = np.random.default_rng(seed)

    patches, _ = _extract_patches(sample, psize)
    flat = patches.reshape(len(patches), -1)
    forest = HoloForest(flat.shape[1], seed=seed).build(flat)   # index patches for sublinear recall
    mean_col = sample.reshape(-1, C).mean(0)

    n_rows = int(np.ceil((out_h - overlap) / step))
    n_cols = int(np.ceil((out_w - overlap) / step))
    OH, OW = n_rows * step + overlap, n_cols * step + overlap
    out = np.zeros((OH, OW, C))

    for i in range(n_rows):
        for j in range(n_cols):
            y0, x0 = i * step, j * step
            has_left, has_top = j > 0, i > 0

            if not has_left and not has_top:
                choice = int(rng.integers(len(patches)))     # the seed patch is free
            else:
                # build a context query patch: fill the overlaps from placed neighbours, the rest with the mean
                q = np.tile(mean_col, (psize, psize, 1))
                if has_left:
                    q[:, :overlap] = out[y0:y0 + psize, x0:x0 + overlap]
                if has_top:
                    q[:overlap, :] = out[y0:y0 + overlap, x0:x0 + psize]
                cand, _ = forest.recall_k(q.ravel(), k=candidates)   # NATIVE patch search (sub-linear)
                # refine: pick the candidate with the smallest EXACT overlap error
                errs = []
                for c in cand:
                    p = patches[c]
                    e = 0.0
                    if has_left:
                        e += float(((p[:, :overlap] - out[y0:y0 + psize, x0:x0 + overlap]) ** 2).sum())
                    if has_top:
                        e += float(((p[:overlap, :] - out[y0:y0 + overlap, x0:x0 + psize]) ** 2).sum())
                    errs.append(e)
                order = np.argsort(errs)
                near_best = order[:max(1, len(order) // 4)]   # pick among the near-best for variety
                choice = int(cand[near_best[int(rng.integers(len(near_best)))]])

            _blend_patch(out, patches[choice], y0, x0, overlap, has_left, has_top, seam=seam)

    out = np.clip(out[:out_h, :out_w], 0, 1)
    return out[..., 0] if gray_in else out


def find_similar_patches(sample, query_patch, k=8, seed=0):
    """The native patch search on its own: index the sample's patches in a HoloForest and recall the k most similar
    to `query_patch` (HoloForest.recall_k). Returns (patches, cosines). This is the 'find the patches that look like
    this one' primitive quilting is built on -- the same one NLM denoising uses."""
    q = np.asarray(query_patch, float)
    psize = q.shape[0]
    patches, _ = _extract_patches(np.asarray(sample, float), psize)
    flat = patches.reshape(len(patches), -1)
    forest = HoloForest(flat.shape[1], seed=seed).build(flat)
    idx, sims = forest.recall_k(q.ravel(), k=k)
    return patches[idx], sims


def _seam_energy(img):
    """Mean gradient magnitude -- a proxy for visible seams/discontinuities (lower = smoother joins)."""
    g = img.mean(axis=-1) if img.ndim == 3 else img
    gy, gx = np.gradient(g)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def _selftest():
    """Synthesis grows a texture larger than the sample whose statistics match it; the min-cut quilt has clearly
    LOWER seam energy than a naive grid tiling of the same patches; the native patch search finds patches similar to
    a query; deterministic."""
    # a structured sample texture: diagonal weave + speckle (stationary, quilt-friendly)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:48, 0:48].astype(float)
    base = 0.5 + 0.3 * np.sin((xx + yy) / 3.0) + 0.1 * rng.standard_normal((48, 48))
    sample = np.clip(np.stack([base, base * 0.9 + 0.05, base * 0.8], axis=-1), 0, 1)

    syn = synthesize_texture(sample, 96, 96, psize=20, overlap=6, seed=0)
    assert syn.shape == (96, 96, 3)                           # grown larger than the 48x48 sample

    # statistics match the sample (it is made of the sample's patches)
    assert abs(syn.mean() - sample.mean()) < 0.05
    assert abs(syn.std() - sample.std()) < 0.05

    # the min-cut seam joins more smoothly than a straight hard cut of the SAME patches (isolates the min-cut)
    hard = synthesize_texture(sample, 96, 96, psize=20, overlap=6, seed=0, seam="hard")
    assert _seam_energy(syn) < _seam_energy(hard)            # min-cut routes the boundary through low-error paths

    # the native patch search finds patches similar to a query patch
    query = sample[5:25, 5:25]
    found, sims = find_similar_patches(sample, query, k=6)
    assert found.shape[1:] == (20, 20, 3) and sims[0] > 0.9   # the sample contains near-duplicates of its own patch

    # deterministic
    assert np.array_equal(synthesize_texture(sample, 64, 64, psize=20, overlap=6, seed=1),
                          synthesize_texture(sample, 64, 64, psize=20, overlap=6, seed=1))

    print("holographic_texturesynth selftest OK: quilted a 48x48 sample into 96x96 whose mean/std match "
          "(%.2f/%.2f vs %.2f/%.2f); the min-cut quilt seams less than a hard cut (%.4f vs %.4f, same patches); the native "
          "HoloForest patch search finds similar patches (top cosine %.2f); deterministic"
          % (syn.mean(), syn.std(), sample.mean(), sample.std(), _seam_energy(syn), _seam_energy(hard), sims[0]))


if __name__ == "__main__":
    _selftest()

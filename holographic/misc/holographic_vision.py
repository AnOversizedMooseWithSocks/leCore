"""
holographic_vision.py -- seeing with arithmetic.

The whole premise of this file is the one in the title of the conversation that
spawned it: an image is just numbers.  A picture is an H x W x 3 grid of bytes,
and every classical "computer vision" idea -- colour, edges, corners, lines,
circles, shape, and even unsupervised *classification* -- falls out of plain
arithmetic on that grid.  No OpenCV, no scikit-image, no learned weights from a
2 GB checkpoint.  Just numpy, written so you can read every step.

The module is organised as a pipeline, bottom to top:

    colour        rgb_to_hsv / hsv_to_rgb / hue_histogram / dominant_colours
    gradients     to_gray / sobel / gradient / edges
    descriptors   orientation_histogram / harris / corners
    shapes        hough_lines / hough_circles / shape_stats / classify_shape
    patterns      describe  (one feature vector per image)
    emergence     kmeans / emergent_classes / cluster_purity
    holographic   vsa_encode / vsa_prototypes / vsa_classify
                  (encode a descriptor as a weighted superposition of random
                   basis vectors, bundle members into a class prototype, and
                   classify new images by cleanup -- i.e. cosine to the nearest
                   prototype.  This is the bridge back to the rest of leOS.)

Everything works in float64 internally to dodge the uint8 / NEP-50 overflow
traps, and every nontrivial claim in the docstrings is checked in the test
suite and the __main__ demo below.
"""

import numpy as np


# ======================================================================
# colour  --  RGB is one basis for colour; HSV is a more perceptual one.
# ======================================================================

def _as_float(rgb):
    """Accept uint8 [0..255] or float [0..1] (RGB or RGBA) and return float
    RGB in [0, 1]."""
    a = np.asarray(rgb)
    if a.dtype == np.uint8:
        a = a.astype(np.float64) / 255.0
    else:
        a = a.astype(np.float64)
    return a[..., :3]


def rgb_to_hsv(rgb):
    """Vectorised RGB->HSV.  Returns H in [0, 360), S and V in [0, 1].

    HSV separates *what* colour (hue) from *how vivid* (saturation) and *how
    bright* (value).  That separation is exactly what makes "find everything
    reddish regardless of lighting" a one-liner later on.
    """
    r, g, b = _as_float(rgb)[..., 0], _as_float(rgb)[..., 1], _as_float(rgb)[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn
    # Value is just the brightest channel; saturation is the spread / value.
    v = mx
    s = np.where(mx <= 0, 0.0, df / np.where(mx <= 0, 1.0, mx))
    # Hue: which channel is on top, and by how much, sets the angle on the wheel.
    safe = np.where(df <= 0, 1.0, df)                 # avoid 0/0; masked out below
    h = np.zeros_like(r)
    h = np.where(mx == r, ((g - b) / safe) % 6.0, h)
    h = np.where(mx == g, ((b - r) / safe) + 2.0, h)
    h = np.where(mx == b, ((r - g) / safe) + 4.0, h)
    h = (h * 60.0) % 360.0
    h = np.where(df <= 0, 0.0, h)                      # greys have no hue
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv):
    """Inverse of rgb_to_hsv, used mainly to prove the forward transform is
    faithful (round-trip error ~1e-12 in the tests)."""
    h = np.asarray(hsv, float)[..., 0] / 60.0
    s = np.asarray(hsv, float)[..., 1]
    v = np.asarray(hsv, float)[..., 2]
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.clip(np.stack([r, g, b], -1), 0, 1)


def hue_histogram(rgb, bins=12, sat_min=0.20, val_min=0.20):
    """A normalised histogram of hue over the *colourful* pixels (greys and
    near-black/near-white pixels carry no reliable hue, so we drop them).  This
    is a compact, lighting-tolerant colour fingerprint."""
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    keep = (s >= sat_min) & (v >= val_min)
    if not np.any(keep):
        return np.zeros(bins)
    hist, _ = np.histogram(h[keep], bins=bins, range=(0, 360))
    total = hist.sum()
    return hist / total if total else hist.astype(float)


def dominant_colours(rgb, k=4, seed=0, sample=2000):
    """The k most common colours, by clustering the pixels (k-means in RGB).
    Returns (centres_uint8[k,3], weights[k]) sorted most-common first."""
    px = _as_float(rgb).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    if len(px) > sample:                               # subsample big images
        px = px[rng.choice(len(px), sample, replace=False)]
    labels, centres = kmeans(px, k, seed=seed, iters=25)
    counts = np.bincount(labels, minlength=k).astype(float)
    order = np.argsort(-counts)
    w = counts[order] / counts.sum()
    return (np.clip(centres[order] * 255, 0, 255).astype(np.uint8), w)


# ======================================================================
# gradients  --  edges are just *where the numbers change fast*.
# ======================================================================

def to_gray(rgb):
    """Perceptual luma (Rec. 601 weights).  Returns float [H,W] in [0,1]."""
    a = _as_float(rgb)
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def _conv3(a, k):
    """Convolve a 2-D array with a 3x3 kernel, edge-padded.  Written as nine
    shifted, weighted adds so there is nothing hidden."""
    p = np.pad(a, 1, mode="edge")
    out = np.zeros_like(a, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if k[i, j]:
                out += k[i, j] * p[i:i + a.shape[0], j:j + a.shape[1]]
    return out


_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
_SOBEL_Y = _SOBEL_X.T


def sobel(gray):
    """Horizontal and vertical Sobel derivatives (gx, gy)."""
    return _conv3(gray, _SOBEL_X), _conv3(gray, _SOBEL_Y)


def gradient(gray):
    """Returns (magnitude, orientation_deg).  Magnitude says *how strong* an
    edge is; orientation says *which way it points* (0..360)."""
    gx, gy = sobel(gray)
    mag = np.hypot(gx, gy)
    ori = np.rad2deg(np.arctan2(gy, gx)) % 360.0
    return mag, ori


def edges(gray, quantile=0.85):
    """Boolean edge map: keep pixels whose gradient magnitude is in the top
    (1 - quantile) fraction.  Relative thresholding means it adapts to the
    image instead of needing a hand-tuned constant."""
    mag, _ = gradient(gray)
    thr = np.quantile(mag, quantile)
    return mag > max(thr, 1e-9)


# ======================================================================
# descriptors  --  summarise the patterns in an image as a short vector.
# ======================================================================

def orientation_histogram(gray, bins=9):
    """HOG-lite: a magnitude-weighted histogram of *unsigned* edge orientation
    (0..180).  Edges going the same way pile into the same bin, so this captures
    texture/structure direction independent of contrast sign."""
    gx, gy = sobel(gray)
    mag = np.hypot(gx, gy).ravel()
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0).ravel()
    hist, _ = np.histogram(ang, bins=bins, range=(0, 180), weights=mag)
    total = hist.sum()
    return hist / total if total else hist


def harris(gray, k=0.04, win=3):
    """Harris corner response R.  Corners are points where the image changes in
    *two* directions at once, which the structure tensor (gradient outer
    products, locally averaged) detects."""
    gx, gy = sobel(gray)
    box = np.ones((win, win)) / (win * win)
    Sxx = _conv3(gx * gx, box) if win == 3 else gx * gx
    Syy = _conv3(gy * gy, box) if win == 3 else gy * gy
    Sxy = _conv3(gx * gy, box) if win == 3 else gx * gy
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    return det - k * trace * trace


def corners(gray, n=12, rel=0.05, min_dist=4):
    """Top-n Harris corners as (x, y), greedily spaced at least min_dist apart."""
    R = harris(gray)
    thr = rel * R.max()
    ys, xs = np.nonzero(R > thr)
    if len(xs) == 0:
        return []
    order = np.argsort(-R[ys, xs])
    picked = []
    for idx in order:
        x, y = int(xs[idx]), int(ys[idx])
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist * min_dist for px, py in picked):
            picked.append((x, y))
        if len(picked) >= n:
            break
    return picked


# ======================================================================
# shapes  --  lines and circles by voting; shape class by geometry.
# ======================================================================

def hough_lines(edge_mask, ntheta=180, top=5, nms=10):
    """Classic Hough line transform.  Every edge pixel votes for all the lines
    that could pass through it (one per angle); real lines collect many votes.
    Returns up to `top` lines as (rho, theta_deg, votes)."""
    ys, xs = np.nonzero(edge_mask)
    if len(xs) == 0:
        return []
    H, W = edge_mask.shape
    thetas = np.deg2rad(np.arange(ntheta))
    cos, sin = np.cos(thetas), np.sin(thetas)
    diag = int(np.ceil(np.hypot(H, W)))
    rho = xs[:, None] * cos[None, :] + ys[:, None] * sin[None, :]   # (P, ntheta)
    rho_idx = np.round(rho).astype(int) + diag                       # shift to >= 0
    nrho = 2 * diag + 1
    acc = np.zeros((nrho, ntheta), dtype=np.int32)
    np.add.at(acc, (rho_idx, np.broadcast_to(np.arange(ntheta), rho_idx.shape)), 1)
    # Greedy peak picking with a little non-maximum suppression.
    out = []
    work = acc.copy()
    for _ in range(top):
        r, t = np.unravel_index(np.argmax(work), work.shape)
        if work[r, t] == 0:
            break
        out.append((int(r - diag), float(t), int(acc[r, t])))
        work[max(0, r - nms):r + nms + 1, max(0, t - nms):t + nms + 1] = 0
    return out


def hough_circles(gray, radii, top=5, quantile=0.88, nms=6):
    """Gradient-guided Hough circle transform.  An edge pixel's gradient points
    toward (or away from) a circle's centre, so each edge pixel votes for two
    candidate centres per radius -- far cheaper than the brute-force version.
    Returns up to `top` circles as (cx, cy, r, votes)."""
    H, W = gray.shape
    mag, _ = gradient(gray)
    gx, gy = sobel(gray)
    thr = np.quantile(mag, quantile)
    ys, xs = np.nonzero(mag > max(thr, 1e-9))
    if len(xs) == 0:
        return []
    nx = gx[ys, xs] / (mag[ys, xs] + 1e-9)             # unit gradient direction
    ny = gy[ys, xs] / (mag[ys, xs] + 1e-9)
    best = []
    for r in radii:
        acc = np.zeros((H, W), dtype=np.int32)
        for sign in (+1, -1):                          # centre is r along +/- grad
            cx = np.round(xs + sign * r * nx).astype(int)
            cy = np.round(ys + sign * r * ny).astype(int)
            ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            np.add.at(acc, (cy[ok], cx[ok]), 1)
        cy, cx = np.unravel_index(np.argmax(acc), acc.shape)
        best.append((int(cx), int(cy), int(r), int(acc[cy, cx])))
    best.sort(key=lambda c: -c[3])
    # suppress near-duplicate centres
    out = []
    for c in best:
        if all((c[0] - o[0]) ** 2 + (c[1] - o[1]) ** 2 >= nms * nms for o in out):
            out.append(c)
        if len(out) >= top:
            break
    return out


def shape_stats(mask):
    """Geometric summary of a single filled blob (boolean mask):
      area        pixels inside
      perimeter   boundary pixels (have a background 4-neighbour)
      circularity 4*pi*area / perimeter^2  (1.0 for a perfect disk, less for
                  anything with corners or elongation)
      extent      area / bounding-box area  (1.0 fills its box -> rectangle-ish)
      aspect      long side / short side of the bounding box
    """
    m = np.asarray(mask, bool)
    area = int(m.sum())
    if area == 0:
        return dict(area=0, perimeter=0, circularity=0.0, extent=0.0, aspect=1.0)
    p = np.pad(m, 1)
    boundary = m & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    perim = int(boundary.sum())
    ys, xs = np.nonzero(m)
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    circ = float(4 * np.pi * area / (perim * perim)) if perim else 0.0
    return dict(area=area, perimeter=perim, circularity=min(circ, 1.0),
                extent=area / float(bh * bw), aspect=max(bh, bw) / float(min(bh, bw)))


def classify_shape(mask):
    """A small, honest, *rule-based* shape labeller built on shape_stats.  It is
    not learned and not magic -- it just encodes the obvious geometry:
        very elongated      -> 'line'
        nearly round        -> 'circle'
        fills its box       -> 'rectangle'
        otherwise           -> 'triangle'
    Good enough to recover clean shapes; deliberately simple."""
    st = shape_stats(mask)
    if st["area"] == 0:
        return "empty"
    if st["aspect"] >= 4.0 or st["extent"] <= 0.32:
        return "line"
    if st["extent"] >= 0.85:                # fills its bounding box -> rectangle
        return "rectangle"
    if st["circularity"] >= 0.88:           # round and compact -> circle
        return "circle"
    return "triangle"


# ======================================================================
# patterns  --  one descriptor vector per image (the input to clustering).
# ======================================================================

def describe(rgb):
    """Turn an image into a single fixed-length feature vector by stacking the
    cheap descriptors above:
        12 hue-histogram bins      (colour)
         2 mean saturation/value   (colourfulness/brightness)
         1 edge density            (how busy)
         9 orientation bins        (structure direction)
    -> a 24-D vector, L2-normalised.  Similar-looking images land near each
    other; that is the whole basis for the clustering and classification below.
    """
    rgb = np.asarray(rgb)
    g = to_gray(rgb)
    hsv = rgb_to_hsv(rgb)
    feats = np.concatenate([
        hue_histogram(rgb, bins=12),
        [hsv[..., 1].mean(), hsv[..., 2].mean()],
        [edges(g).mean()],
        orientation_histogram(g, bins=9),
    ]).astype(np.float64)
    n = np.linalg.norm(feats)
    return feats / n if n else feats


# ======================================================================
# emergence  --  let categories fall out of the data, unsupervised.
# ======================================================================

def _kmeanspp_init(X, k, rng):
    """k-means++ seeding: spread the initial centres out so a run does not
    collapse.  First centre is random; each next is chosen with probability
    proportional to its squared distance from the nearest centre so far."""
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        s = d2.sum()
        probs = d2 / s if s > 0 else np.full(len(X), 1.0 / len(X))
        centres.append(X[rng.choice(len(X), p=probs)])
    return np.array(centres)


def _one_kmeans(X, k, rng, iters):
    centres = _kmeanspp_init(X, k, rng)
    labels = np.zeros(len(X), int)
    for step in range(iters):
        d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if step > 0 and np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = X[labels == c]
            centres[c] = members.mean(0) if len(members) else X[d.min(1).argmax()]
    inertia = ((X - centres[labels]) ** 2).sum()
    return labels, centres, inertia


def kmeans(X, k, seed=0, iters=50, n_init=5):
    """Lloyd's k-means with k-means++ seeding and a few random restarts; the run
    with the lowest within-cluster spread (inertia) wins.  Restarts are what make
    'emergent' clustering reproducible instead of init-luck.  Returns (labels,
    centres)."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        labels, centres, inertia = _one_kmeans(X, k, rng, iters)
        if best is None or inertia < best[2]:
            best = (labels, centres, inertia)
    return best[0], best[1]


def emergent_classes(images, k, seed=0, standardize=False):
    """Describe every image, then cluster the descriptors.  No labels go in;
    the groups that come out are 'emergent classes'.  Returns (labels,
    descriptors).

    `standardize` z-scores each feature dimension and drops constant ones before
    clustering.  This is *not* a universal win: it helps when the discriminating
    signal lives in low-variance dimensions (e.g. clustering same-coloured shapes
    by structure), but it can hurt when a high-variance feature like colour is
    exactly the signal you want (e.g. clustering sprites by character).  Left off
    by default; flip it on when the data calls for it.
    """
    X = np.stack([describe(im) for im in images])
    M = X
    if standardize:
        mu, sd = X.mean(0), X.std(0)
        keep = sd > 1e-9
        M = (X[:, keep] - mu[keep]) / sd[keep]
    labels, _ = kmeans(M, k, seed=seed)
    return labels, X


def cluster_purity(labels, truth):
    """Fraction of points that agree with the majority true label of their
    cluster.  1.0 means the unsupervised clusters perfectly match the real
    categories.  This is how we keep ourselves honest about 'emergence'."""
    labels, truth = np.asarray(labels), np.asarray(truth)
    correct = 0
    for c in np.unique(labels):
        members = truth[labels == c]
        if len(members):
            vals, counts = np.unique(members, return_counts=True)
            correct += counts.max()
    return correct / len(labels)


# ======================================================================
# holographic  --  encode descriptors as hypervectors and classify by cleanup.
# ======================================================================

def _basis(n_features, dim, seed):
    """One random unit hypervector per feature dimension."""
    rng = np.random.default_rng(seed)
    from holographic.agents_and_reasoning.holographic_ai import random_vector
    return np.stack([random_vector(dim, rng) for _ in range(n_features)])


def vsa_encode(descriptors, dim=2048, seed=0):
    """Encode each descriptor as a *weighted superposition* of the basis
    hypervectors: hv = normalise( sum_i  d[i] * B[i] ).  This is exactly the
    bundle operation from holographic_ai, with the feature values as weights --
    so two similar descriptors produce two similar hypervectors."""
    X = np.atleast_2d(np.asarray(descriptors, float))
    B = _basis(X.shape[1], dim, seed)
    hv = X @ B                                          # weighted bundle
    norms = np.linalg.norm(hv, axis=1, keepdims=True)
    return hv / np.where(norms == 0, 1, norms)


def vsa_prototypes(descriptors, labels, dim=2048, seed=0):
    """Bundle the member hypervectors of each class into one prototype vector
    (the VSA way to form a concept from examples).  Returns
    {label: prototype_hv} plus the encoder settings so new images use the same
    basis."""
    hv = vsa_encode(descriptors, dim=dim, seed=seed)
    protos = {}
    for c in np.unique(labels):
        from holographic.agents_and_reasoning.holographic_ai import bundle
        protos[int(c)] = bundle(hv[np.asarray(labels) == c])
    return protos, dict(dim=dim, seed=seed)


def vsa_classify(descriptor, protos, enc):
    """Classify one descriptor by cleanup: encode it, then return the label of
    the nearest prototype by cosine similarity."""
    from holographic.agents_and_reasoning.holographic_ai import cosine
    hv = vsa_encode(descriptor, dim=enc["dim"], seed=enc["seed"])[0]
    best, lab = -2.0, None
    for c, p in protos.items():
        s = cosine(hv, p)
        if s > best:
            best, lab = s, c
    return lab, best


# ======================================================================
# drawing helpers  --  synthetic shapes, used by tests, demo and the UI.
# ======================================================================

def make_shape(kind, S=64, seed=0, bg=(14, 22, 38), fg=(45, 212, 191)):
    """Draw one clean filled shape on a dark background.  Returns
    (rgb_uint8[S,S,3], mask_bool[S,S]).  Pure numpy, no PIL."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    cx, cy = S / 2 + rng.uniform(-4, 4), S / 2 + rng.uniform(-4, 4)
    if kind == "circle":
        r = S * rng.uniform(0.28, 0.38)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    elif kind == "rectangle":
        hw, hh = S * rng.uniform(0.24, 0.36), S * rng.uniform(0.24, 0.36)
        mask = (np.abs(xx - cx) <= hw) & (np.abs(yy - cy) <= hh)
    elif kind == "triangle":
        r = S * rng.uniform(0.34, 0.42)
        pts = np.array([[cx + r * np.cos(a), cy + r * np.sin(a)]
                        for a in (-np.pi / 2, -np.pi / 2 + 2 * np.pi / 3, -np.pi / 2 + 4 * np.pi / 3)])
        mask = _in_triangle(xx, yy, pts)
    elif kind == "line":
        ang = rng.uniform(0, np.pi)
        dx, dy = np.cos(ang), np.sin(ang)
        L, w = S * 0.42, 1.5
        # distance from each pixel to the line through the centre
        dist = np.abs((xx - cx) * (-dy) + (yy - cy) * dx)
        along = np.abs((xx - cx) * dx + (yy - cy) * dy)
        mask = (dist <= w) & (along <= L)
    else:
        raise ValueError(kind)
    img = np.empty((S, S, 3), np.uint8)
    img[:] = np.array(bg, np.uint8)
    img[mask] = np.array(fg, np.uint8)
    return img, mask


def _in_triangle(xx, yy, pts):
    """Boolean mask of points inside triangle pts[3,2] via half-plane signs."""
    def side(a, b):
        return (xx - a[0]) * (b[1] - a[1]) - (yy - a[1]) * (b[0] - a[0])
    d1, d2, d3 = side(pts[0], pts[1]), side(pts[1], pts[2]), side(pts[2], pts[0])
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


# ======================================================================
# demo
# ======================================================================

def _demo():
    import colorsys
    print("holographic_vision -- seeing with arithmetic\n" + "-" * 52)

    # colour: prove the HSV transform is exact against the stdlib
    rng = np.random.default_rng(0)
    px = rng.random((1000, 3))
    ours = rgb_to_hsv(px.reshape(1, -1, 3))[0]
    ref = np.array([colorsys.rgb_to_hsv(*p) for p in px])
    err = np.abs(np.stack([ours[:, 0] / 360, ours[:, 1], ours[:, 2]], 1) - ref)
    print(f"RGB->HSV vs colorsys : max error {err.max():.2e}")

    # shapes: classify clean synthetic shapes
    kinds = ["circle", "rectangle", "triangle", "line"]
    n_each, ok = 25, 0
    truth, imgs = [], []
    for s in range(n_each):
        for ki, k in enumerate(kinds):
            img, mask = make_shape(k, 64, seed=100 * s + ki)
            ok += classify_shape(mask) == k
            truth.append(ki); imgs.append(img)
    print(f"rule-based shape ID  : {ok}/{n_each*len(kinds)} correct "
          f"({100*ok/(n_each*len(kinds)):.0f}%)")

    # hough: a board with one strong horizontal line and one disk
    board, _ = make_shape("line", 80, seed=3)
    lines = hough_lines(edges(to_gray(board)), top=2)
    print(f"hough lines found    : {len(lines)} (top votes {lines[0][2] if lines else 0})")
    disk, _ = make_shape("circle", 80, seed=4)
    circ = hough_circles(to_gray(disk), radii=range(15, 35, 2), top=1)
    print(f"hough circle found   : {circ[0] if circ else None}")

    # emergence: cluster the mixed shapes unsupervised, measure purity
    labels, X = emergent_classes(imgs, k=4, seed=0, standardize=True)
    print(f"emergent clusters    : purity {cluster_purity(labels, truth):.0%} "
          f"(unsupervised; rotated lines scatter, so ~70% is the honest ceiling)")

    # holographic: train prototypes on half, classify the other half by cleanup
    X = np.stack([describe(im) for im in imgs]); truth = np.array(truth)
    tr = rng.random(len(imgs)) < 0.5
    protos, enc = vsa_prototypes(X[tr], truth[tr], dim=2048, seed=1)
    acc = np.mean([vsa_classify(X[i], protos, enc)[0] == truth[i]
                   for i in np.nonzero(~tr)[0]])
    print(f"VSA cleanup classify : {acc:.0%} on held-out shapes "
          f"(bundle prototypes + cosine)")


if __name__ == "__main__":
    _demo()

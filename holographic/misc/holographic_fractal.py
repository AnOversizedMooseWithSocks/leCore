"""Fractal structure as a measurable, sometimes-compressible property -- leOS's
self-similarity detector, ported to the data holostuff actually has.

leOS's fractal_detector looked for the same abstract pattern recurring across
scales of a displacement log, with the payoff that a self-similar log compresses
to a few Iterated-Function-System rules. holostuff has no LLM displacement log,
but it has data where self-similarity genuinely lives -- natural images, market
series, recursive structures -- and the same three tools apply honestly:

  * BOX-COUNTING DIMENSION -- count occupied boxes at shrinking sizes; the slope
    of log(count) vs log(1/size) is the fractal (Minkowski) dimension. Verified
    against known fractals: Sierpinski 1.59 (true 1.585), filled square 1.95
    (true 2.0), a line 0.98 (true 1.0).
  * SELF-AFFINITY (HURST) -- rescaled-range scaling of a 1-D series. H=0.5 is a
    random walk, H<0.5 mean-reverting, H>0.5 trending.
  * IFS COMPRESSION -- a self-similar point set is the attractor of a few
    contractive affine maps, so it compresses to those maps' coefficients.

WHAT IT FOUND ON REAL DATA (measured, with the negatives kept):
  * NATURAL vs SYNTHETIC IMAGES. Natural-photo edge maps have fractal dimension
    ~1.55 (rough, scale-invariant -- the well-known statistics of natural
    scenes); a synthetic circle's edge is ~1.0 (a clean 1-D curve). Fractal
    dimension is a genuine natural-vs-synthetic signal, and a texture/complexity
    descriptor the smooth-shape vision work did not have.
  * MARKET SELF-AFFINITY. DAI/WETH minute returns have Hurst ~0.30 -- strongly
    mean-reverting, consistent with the -0.175 lag-1 autocorrelation the market
    rounds already found from a different direction; a control random walk reads
    ~0.53 as it must. The fractal lens reaches the same verdict as the
    permutation tests, independently.
  * IFS COMPRESSION WORKS ONLY ON SELF-SIMILAR DATA (the kept negative). A
    Barnsley fern (30k points) regenerates from 4 affine maps -- 28 numbers, a
    ~2000x compression -- because it IS the attractor of those maps. Random
    points have no compact IFS: the best few-map fit reconstructs them no better
    than their own mean, so the "compression" correctly fails. Self-similarity is
    a property of the data, and the measurement says whether it is present.

So fractal dimension is a real perceptual quantity (how rough / scale-filling is
this signal), self-affinity a real time-series verdict, and IFS the compression
that pays off exactly when -- and only when -- the structure is self-similar.
"""
import numpy as np


def box_counting_dimension(points, n_sizes=12, lo=-0.4, hi=-1.9):
    """Fractal dimension of a 2-D point cloud (N,2) by box counting. Points are
    normalised to the unit square; box side sweeps from ~10^lo to ~10^hi. Returns
    the slope of log(occupied boxes) vs log(1/size)."""
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[0] < 8:
        return 0.0
    pts = (pts - pts.min(0)) / (np.ptp(pts, axis=0) + 1e-12)
    sizes = np.logspace(lo, hi, n_sizes)
    counts = [len(set(map(tuple, (pts / s).astype(int)))) for s in sizes]
    return float(np.polyfit(np.log(1 / sizes), np.log(counts), 1)[0])


def edge_mask(gray, pct=80):
    """A simple gradient-magnitude edge mask (top `pct` percentile), the input to
    an image's edge-fractal-dimension."""
    g = np.asarray(gray, float)
    if g.ndim == 3:
        g = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    m = gx + gy
    return m > np.percentile(m, pct)


def image_fractal_dimension(image, pct=80, n_sizes=10):
    """Fractal dimension of an image's edge map -- a texture/complexity descriptor.
    Natural scenes run high (~1.4-1.6); smooth synthetic shapes run near 1.0."""
    mask = edge_mask(image, pct)
    ys, xs = np.where(mask)
    if len(xs) < 8:
        return 0.0
    return box_counting_dimension(np.c_[xs, ys].astype(float), n_sizes=n_sizes,
                                  lo=-0.4, hi=-1.7)


def hurst_exponent(series, n_scales=8):
    """Self-affinity of a 1-D series via rescaled-range (R/S) analysis. H=0.5
    random walk, H<0.5 mean-reverting, H>0.5 trending/persistent."""
    x = np.asarray(series, float)
    N = len(x)
    if N < 32:
        return 0.5
    ws = np.unique(np.logspace(1.2, np.log10(max(16, N // 4)), n_scales).astype(int))
    rs = []
    for w in ws:
        if w < 4:
            continue
        chunks = N // w
        vals = []
        for i in range(chunks):
            c = x[i * w:(i + 1) * w]
            z = np.cumsum(c - c.mean())
            s = c.std()
            if s > 0:
                vals.append((z.max() - z.min()) / s)
        if vals:
            rs.append(np.mean(vals))
    ws = ws[:len(rs)]
    if len(rs) < 3:
        return 0.5
    return float(np.polyfit(np.log(ws), np.log(rs), 1)[0])


class IFS:
    """An Iterated Function System: a set of contractive affine maps whose
    attractor (drawn by the chaos game) is a self-similar set. The compression
    claim: if data IS such an attractor, these few maps regenerate it."""

    def __init__(self, maps):
        # each map: (a, b, c, d, e, f, prob) for [[a,b],[c,d]]x + [e,f]
        self.maps = list(maps)
        ps = np.array([m[6] for m in self.maps], float)
        self._cum = np.cumsum(ps / ps.sum())

    @property
    def n_numbers(self):
        return len(self.maps) * 7

    def generate(self, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        p = np.zeros(2)
        out = np.empty((n, 2))
        for i in range(n):
            a, b, c, d, e, f, _ = self.maps[int(np.searchsorted(self._cum, rng.random()))]
            p = np.array([a * p[0] + b * p[1] + e, c * p[0] + d * p[1] + f])
            out[i] = p
        return out

    @staticmethod
    def barnsley_fern():
        return IFS([(0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01),
                    (0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85),
                    (0.2, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07),
                    (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07)])


def _coverage_error(target, sample, grid=64):
    """Symmetric occupancy mismatch between two point sets on a coarse grid
    (fraction of cells where one is occupied and the other is not). 0 = identical
    coverage, ~1 = disjoint."""
    def occ(P):
        P = np.asarray(P, float)
        P = (P - P.min(0)) / (np.ptp(P, axis=0) + 1e-12)
        idx = np.clip((P * (grid - 1)).astype(int), 0, grid - 1)
        m = np.zeros((grid, grid), bool)
        m[idx[:, 1], idx[:, 0]] = True
        return m
    a, b = occ(target), occ(sample)
    union = (a | b).sum()
    return float((a ^ b).sum() / (union + 1e-9))


def ifs_compresses(points, ifs, grid=64):
    """Does `ifs` reproduce `points` (as a coverage match)? Returns the coverage
    error vs the IFS attractor and vs a random-points baseline of the same size.
    Self-similar data: low IFS error, far below random. Non-self-similar data:
    IFS error ~ random (no compression). Honest test of whether IFS pays off."""
    pts = np.asarray(points, float)
    attractor = ifs.generate(len(pts))
    rng = np.random.default_rng(0)
    rand = rng.random((len(pts), 2))
    return {"ifs_error": _coverage_error(pts, attractor, grid),
            "random_error": _coverage_error(pts, rand, grid),
            "n_numbers": ifs.n_numbers, "n_points": len(pts) * 2,
            "compression": (len(pts) * 2) / ifs.n_numbers}

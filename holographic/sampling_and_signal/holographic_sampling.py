"""Blue-noise / Poisson-disk point sampling -- the EXCLUSION principle, done right.

WHY THIS MODULE EXISTS
----------------------
A recurring idea in the engine (and the spark for it here) is "instances can't coincide, so they push apart
and self-organise." Taken literally as a relaxation (iterated pairwise repulsion) it under-converges and
trades local separation against global uniformity -- a measured kept-negative. The CORRECT realisation of the
same exclusion principle is Bridson's Poisson-disk sampling: grow a point set by dart-throwing, accepting a
candidate only if no existing point is within `radius`. That yields a MAXIMAL point set with a hard
min-distance guarantee AND the blue-noise spectrum (suppressed low frequencies + a ring at the sample
spacing) that makes blue noise the gold standard for sampling -- stippling, splat / particle initialisation,
anti-aliased Monte Carlo, dithering.

It is pure NumPy and deterministic (seeded). The dart-throwing loop is inherently sequential (an active list),
which is fine: sampling is a one-off setup cost, not a per-frame inner loop. The neighbour test uses a
background grid (cell = radius / sqrt(D)), so each candidate is checked against O(1) nearby cells, not all
points -- the same "cull, don't batch" spatial-grid idea the rest of the engine leans on.
"""

import numpy as np


def poisson_disk_sample(radius, bounds, k=30, seed=0):
    """Bridson's Poisson-disk (blue-noise) sampling. Returns an (M, D) point set in which every pair of points
    is at least `radius` apart and the empty space is filled as densely as that allows (a *maximal* set), with
    the blue-noise spectral signature. `bounds` = (min_corner, max_corner) of any dimension D. `k` is the
    candidates tried per active point before it is retired (Bridson's default 30). Deterministic in `seed`.

    This is the honest realisation of the exclusion principle: a candidate is ACCEPTED only if no existing
    point lies within `radius` of it (checked against a background grid, so O(1) per candidate). It does NOT
    validate any cosmology -- it is a sampling algorithm -- but it is the genuinely useful thing the principle
    points at."""
    lo = np.asarray(bounds[0], float)
    hi = np.asarray(bounds[1], float)
    D = lo.shape[0]
    extent = hi - lo
    rng = np.random.default_rng(seed)

    cell = radius / np.sqrt(D)                          # at most one point per cell (Bridson's grid)
    dims = np.maximum(np.ceil(extent / cell).astype(int), 1)
    grid = {}                                           # cell-tuple -> point index (sparse, any dimension)

    def cell_of(p):
        return tuple(np.minimum(((p - lo) / cell).astype(int), dims - 1))

    def fits(p):
        c = np.array(cell_of(p))
        # scan the neighbouring cells (within 2 along each axis -- a point in them could be within radius)
        for off in _neighbour_offsets(D):
            key = tuple(c + off)
            j = grid.get(key)
            if j is not None and float(np.sum((p - pts[j]) ** 2)) < radius * radius:
                return False
        return True

    first = lo + rng.random(D) * extent                # seed the set with one random point
    pts = [first]
    grid[cell_of(first)] = 0
    active = [0]

    while active:
        ai = int(rng.integers(len(active)))
        i = active[ai]
        base = pts[i]
        placed = False
        for _ in range(k):                              # try k candidates in the annulus [radius, 2*radius]
            cand = _annulus_point(base, radius, rng)
            if np.any(cand < lo) or np.any(cand >= hi):
                continue
            if fits(cand):
                pts.append(cand)
                grid[cell_of(cand)] = len(pts) - 1
                active.append(len(pts) - 1)
                placed = True
                break
        if not placed:                                 # exhausted -> retire this point (swap-remove)
            active[ai] = active[-1]
            active.pop()
    return np.array(pts, float)


def _annulus_point(center, radius, rng):
    """A uniform random point in the spherical shell [radius, 2*radius] around `center` (any dimension)."""
    D = center.shape[0]
    direction = rng.standard_normal(D)
    direction /= (np.linalg.norm(direction) + 1e-12)
    r = radius * (1.0 + rng.random())                  # uniform-ish in [r, 2r]
    return center + r * direction


def _neighbour_offsets(D):
    """All integer offsets in [-2, 2]^D -- the cells a within-radius point could occupy (grid cell = r/sqrt D
    means a neighbour within `radius` is at most 2 cells away on each axis)."""
    import itertools
    return [np.array(o) for o in itertools.product(range(-2, 3), repeat=D)]


def radial_power_spectrum(points, bounds, grid=64):
    """The radially-averaged power spectrum of a point set, the blue-noise diagnostic. Rasterise the points to a
    `grid`^2 histogram (2-D), remove the mean, take |FFT|^2, and average over frequency rings. Normalised so the
    mean of the nonzero rings is 1. Blue noise shows a SUPPRESSED low-frequency region (ratio < 1 near bin 0)
    and a PEAK/ring at the sample-spacing frequency; white noise is flat. Returns the 1-D radial profile."""
    pts = np.asarray(points, float)
    lo = np.asarray(bounds[0], float); hi = np.asarray(bounds[1], float)
    g = np.zeros((grid, grid))
    idx = np.minimum(((pts[:, :2] - lo[:2]) / (hi[:2] - lo[:2]) * grid).astype(int), grid - 1)
    np.add.at(g, (idx[:, 0], idx[:, 1]), 1.0)
    g -= g.mean()
    P = np.fft.fftshift(np.abs(np.fft.fft2(g)) ** 2)
    c = grid // 2
    Y, X = np.indices((grid, grid))
    r = np.sqrt((X - c) ** 2 + (Y - c) ** 2).astype(int)
    rad = np.bincount(r.ravel(), P.ravel()) / (np.bincount(r.ravel()) + 1e-9)
    return rad / (rad[1:].mean() + 1e-9)


def _selftest():
    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    pts = poisson_disk_sample(0.04, bounds, seed=0)
    # min distance respected
    d = pts[:, None, :] - pts[None, :, :]
    dd = np.sqrt((d ** 2).sum(-1)); np.fill_diagonal(dd, np.inf)
    assert dd.min() >= 0.04 - 1e-9, dd.min()
    # blue-noise low-frequency suppression vs white noise
    white = np.random.default_rng(0).uniform(0, 1, (len(pts), 2))
    sb = radial_power_spectrum(pts, bounds); sw = radial_power_spectrum(white, bounds)
    assert np.mean(sb[1:4]) < np.mean(sw[1:4]), (np.mean(sb[1:4]), np.mean(sw[1:4]))
    print(f"sampling selftest ok: {len(pts)} points, min-dist {dd.min():.4f} >= 0.04, "
          f"low-freq blue {np.mean(sb[1:4]):.2f} < white {np.mean(sw[1:4]):.2f}")


if __name__ == "__main__":
    _selftest()

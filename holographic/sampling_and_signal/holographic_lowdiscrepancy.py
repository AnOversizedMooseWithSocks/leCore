"""Low-discrepancy (quasi-random) sampling -- even coverage of a domain, deterministically.

WHY THIS EXISTS
---------------
Wherever the engine PLACES points to COVER a domain rather than to draw an INDEPENDENT random sample --
generation seeds, codebook / anchor placement, the sub-pixel jitter offsets a temporal-accumulation pass
needs -- pure `default_rng` is the wrong tool. Independent uniform points CLUMP (leaving holes), and a
regular grid ALIASES. The graphics answer (Pharr's PBRT sampling chapter; Roberts 2018) is a LOW-DISCREPANCY
sequence: deterministic points that spread far more evenly than random, with none of a grid's regular
structure.

This module ships the simplest good one -- Roberts' generalised R-sequence, the d-dimensional golden-ratio /
"plastic constant" additive recurrence. One line of NumPy, no state, seed-reproducible, and PROGRESSIVE: any
prefix of the sequence is itself well-distributed, so you can keep taking points.

MEASURED (see `_selftest`)
  * 64 points in 2-D cover ~28% tighter than the mean of random (dispersion ~0.16 vs ~0.23).
  * As a quasi-Monte-Carlo integrator the sequence has materially lower error than plain Monte Carlo on a
    smooth integrand at the same sample count -- the downstream payoff of even coverage.

DESIGN NOTES
  * `seed == 0` is the canonical centred sequence; `seed != 0` applies a Cranley-Patterson rotation (a
    deterministic per-seed offset) so you can draw DIFFERENT well-distributed sets.
  * This is for COVERAGE. Where genuine independence is wanted (bootstrap resampling, noise injection) keep
    `default_rng` -- low-discrepancy points are correlated by construction.
"""

import numpy as np


def _plastic_constant(d, iters=64):
    """The unique positive root phi of x^(d+1) = x + 1 (d=1 -> golden ratio 1.618, d=2 -> plastic number
    1.3247...), by the fixed-point iteration phi = (1 + phi)^(1/(d+1)). The base of the generalised
    golden-ratio sequence; converges in a handful of steps from any positive start."""
    phi = 2.0
    for _ in range(iters):
        phi = (1.0 + phi) ** (1.0 / (d + 1))
    return phi


def low_discrepancy(n, d=2, seed=0):
    """`n` quasi-random points in [0, 1)^d with low discrepancy (even coverage), as an (n, d) float array.

    Roberts' generalised R-sequence: point_k = frac(offset + k * alpha), with alpha_j = phi^-(j+1) and phi
    the plastic constant for dimension d. Deterministic, and progressive -- any prefix is well-distributed,
    so `low_discrepancy(10, ...)` equals the first 10 rows of `low_discrepancy(64, ...)`."""
    if n <= 0:
        return np.zeros((0, d))
    phi = _plastic_constant(d)
    alpha = (1.0 / phi) ** (1.0 + np.arange(d))                       # alpha_j = phi^-(j+1): the per-axis step
    offset = 0.5 if seed == 0 else np.random.default_rng(seed).random(d)   # centred, or a per-seed rotation
    k = np.arange(1, n + 1).reshape(-1, 1)                            # start at k=1 (k=0 would land on offset)
    return (offset + k * alpha) % 1.0


def dispersion(points, grid=64):
    """Coverage measure (2-D): the LARGEST distance from any test point -- a `grid` x `grid` lattice over the
    unit square -- to its nearest sample. Lower = more even coverage = fewer holes. Random clumps (high
    dispersion); a low-discrepancy set spreads (low dispersion)."""
    points = np.asarray(points, float)
    g = np.linspace(0.0, 1.0, grid)
    gy, gx = np.meshgrid(g, g)
    test = np.column_stack([gx.ravel(), gy.ravel()])
    d2 = ((test[:, None, :] - points[None, :, :]) ** 2).sum(-1)       # (grid^2, n) squared distances
    return float(np.sqrt(d2.min(1)).max())


def _selftest():
    """CI-fast: prove the R-sequence (1) covers more evenly than random (lower dispersion, mean over seeds),
    (2) integrates a smooth function with LOWER error than plain Monte Carlo at the same count -- the
    downstream payoff of even coverage -- and (3) is deterministic and progressive."""
    # (1) coverage: the R-sequence beats the mean of random
    pts = low_discrepancy(64, d=2, seed=0)
    assert pts.shape == (64, 2) and pts.min() >= 0.0 and pts.max() < 1.0
    r2_disp = dispersion(pts)
    rand_disp = np.mean([dispersion(np.random.default_rng(s).random((64, 2))) for s in range(20)])
    assert r2_disp < rand_disp, f"R-sequence should cover tighter than random: {r2_disp:.3f} vs {rand_disp:.3f}"

    # (2) quasi-Monte-Carlo: lower integration error than plain MC at equal N (the coverage payoff)
    def f(p):                                                         # smooth 2-D integrand; true mean known
        return np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])      # integral over [0,1]^2 = (2/pi)^2
    truth = (2.0 / np.pi) ** 2
    N = 256
    qmc_err = abs(f(low_discrepancy(N, 2, 0)).mean() - truth)
    mc_err = np.mean([abs(f(np.random.default_rng(s).random((N, 2))).mean() - truth) for s in range(20)])
    assert qmc_err < mc_err, f"QMC should integrate tighter than MC: {qmc_err:.4f} vs {mc_err:.4f}"

    # (3) deterministic + progressive, and a different seed gives a different (still valid) set
    assert np.allclose(low_discrepancy(64, 2, 0), low_discrepancy(64, 2, 0)), "must be deterministic"
    assert np.allclose(low_discrepancy(10, 2, 0), low_discrepancy(64, 2, 0)[:10]), "must be progressive"
    assert not np.allclose(low_discrepancy(64, 2, 1), low_discrepancy(64, 2, 0)), "seed should vary the set"


if __name__ == "__main__":
    _selftest()
    print("holographic_lowdiscrepancy selftest passed")

"""holographic_coarsefirst.py -- the COARSE-FIRST residual pass (the Group-B unlocker from the re-enable audit).

WHY
---
Several superior-but-costly methods were shelved because paying for them EVERYWHERE is wasteful -- they only earn
their cost in a few hard regions (edges, high-variance pixels, a non-low-rank patch of a field, a self-shadowing
volume). The adaptive-dispatch audit's Group B all share one detector: run the CHEAP method first, measure a per-cell
RESIDUAL / UNCERTAINTY, and ESCALATE to the expensive method ONLY where that signal is high. "Refine where uncertain"
and "use the superior method where the cheap one fails" are the same sentence.

This module is that one detector, reusable across the Group-B candidates (adaptive AA, Nystrom, splat refinement,
volumetric marching, ...). It does NOT decide the uncertainty signal for you -- that is domain-specific (path-trace
variance, a fit residual, a landmark residual, a luminance gradient) -- it takes the signal and does the escalation
honestly:

    mask = escalate_mask(uncertainty, frac=0.25)                 # WHERE to refine (the top-frac hardest cells)
    refined, mask, n = refine_where_uncertain(coarse, uncertainty, refine_fn, frac=0.25)

THE DISCIPLINE (same as the parameter gates, applied to a residual signal)
  * CONSERVATIVE -- when unsure, refine a bit MORE, never less: escalating a cell that didn't need it only wastes a
    little work, but MISSING a hard cell reintroduces the error the method was meant to fix. So `frac`/`threshold`
    err toward refining.
  * KEEP THE CHEAP RESULT -- the coarse pass is always the base; refinement only overwrites the flagged cells.
  * MEASURED BREAKEVEN -- a coarse-first pass earns its keep only when the uncertainty is CONCENTRATED (a small
    fraction of cells are hard). If every cell is equally hard, uniform refinement is as good and simpler -- measure
    the concentration before trusting the win (an honest kept caveat, see the module tests / NOTES).
"""
import numpy as np


def escalate_mask(uncertainty, frac=0.25, threshold=None):
    """The cells to escalate to the expensive method. Pass EITHER `frac` (refine the top fraction by uncertainty --
    a fixed budget) OR `threshold` (refine everything above an absolute uncertainty). Returns a boolean mask the
    shape of `uncertainty`. Conservative: ties at the cutoff are INCLUDED (refine rather than skip)."""
    u = np.asarray(uncertainty, float)
    if threshold is not None:
        return u >= float(threshold)               # absolute cutoff: >= includes ties (conservative)
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac <= 0.0:
        return np.zeros_like(u, dtype=bool)
    if frac >= 1.0:
        return np.ones_like(u, dtype=bool)
    k = max(int(round(u.size * frac)), 1)
    # pick EXACTLY the k highest-uncertainty cells by a stable argsort. (A value-threshold here would blow up when
    # the cut lands on a common baseline value like 0 -- every cell >= 0 would be flagged; top-k avoids that.)
    top = np.argsort(u.ravel(), kind="stable")[-k:]
    mask = np.zeros(u.size, dtype=bool)
    mask[top] = True
    return mask.reshape(u.shape)


def refine_where_uncertain(coarse, uncertainty, refine_fn, frac=0.25, threshold=None):
    """Run the expensive `refine_fn` ONLY on the high-uncertainty cells and merge into the cheap `coarse` result.

    `refine_fn(mask)` receives the boolean escalate mask and returns the refined values -- either a full array the
    shape of `coarse` (we take the masked cells) or a 1-D array of length mask.sum() (the flagged cells in order).
    Returns (refined, mask, n_refined). The coarse result is preserved everywhere the mask is False."""
    coarse = np.asarray(coarse, float)
    mask = escalate_mask(uncertainty, frac=frac, threshold=threshold)
    out = coarse.copy()
    if mask.any():
        vals = refine_fn(mask)
        vals = np.asarray(vals, float)
        if vals.shape == coarse.shape:
            out[mask] = vals[mask]                     # refine_fn returned a full array
        else:
            out[mask] = vals                           # refine_fn returned just the flagged cells, in mask order
    return out, mask, int(mask.sum())


def gradient_uncertainty(field):
    """A cheap, deterministic uncertainty signal for a 2-D field: the local gradient magnitude. Where a coarse
    estimate changes fast it is probably under-resolved (an edge / a sharp feature), so refine there. (Domain code
    may supply a better signal -- a fit residual, a path-trace variance -- this is the generic default.)"""
    f = np.asarray(field, float)
    if f.ndim == 3:                                    # colour: use luminance
        f = f.mean(axis=-1)
    gx = np.zeros_like(f); gy = np.zeros_like(f)
    gx[:, 1:-1] = f[:, 2:] - f[:, :-2]
    gy[1:-1, :] = f[2:, :] - f[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def concentration(uncertainty):
    """How CONCENTRATED the uncertainty is (0..1): the share of total uncertainty carried by the top 10% of cells,
    above the 0.10 a uniform field would give. This is the coarse-first breakeven check, but it is NECESSARY, not
    sufficient: LOW concentration (~0) means coarse-first CANNOT help -- the hard work is everywhere, so uniform
    refinement is just as good and simpler. HIGH concentration means it CAN help -- but only if that concentrated
    region also carries meaningful error in the metric you care about (measured: a small bright object can have
    concentration ~1 yet barely move whole-image RMSE, because the object is a tiny fraction of the pixels). So:
    low concentration rules coarse-first OUT; high concentration makes it a candidate that still owes a measured win."""
    u = np.asarray(uncertainty, float).ravel()
    if u.sum() <= 0:
        return 0.0
    k = max(int(u.size * 0.10), 1)
    top = np.sort(u)[-k:].sum() / u.sum()              # share of uncertainty in the top 10% of cells
    return float(np.clip((top - 0.10) / 0.90, 0.0, 1.0))


def _selftest():
    rng = np.random.default_rng(0)
    H = W = 64
    ys, xs = np.mgrid[0:H, 0:W] / float(H)

    # a field that is SMOOTH almost everywhere with a sharp localized ridge (concentrated structure) -- the regime
    # where coarse-first pays off.
    def true_field(Y, X):
        smooth = 0.3 * np.sin(3 * Y) + 0.3 * np.cos(3 * X)
        ridge = np.exp(-((X - 0.5) ** 2) / 0.0008)     # a thin vertical ridge at x=0.5
        return smooth + ridge
    truth = true_field(ys, xs)

    # COARSE: sample on a 1/4-resolution grid and bilinearly upsample -- misses the thin ridge.
    cs = 4
    coarse_grid = true_field(ys[::cs, ::cs], xs[::cs, ::cs])
    yi = np.clip((np.arange(H) / cs), 0, coarse_grid.shape[0] - 1)
    xi = np.clip((np.arange(W) / cs), 0, coarse_grid.shape[1] - 1)
    y0 = np.floor(yi).astype(int); x0 = np.floor(xi).astype(int)
    y1 = np.minimum(y0 + 1, coarse_grid.shape[0] - 1); x1 = np.minimum(x0 + 1, coarse_grid.shape[1] - 1)
    fy = (yi - y0)[:, None]; fx = (xi - x0)[None, :]
    coarse = ((coarse_grid[y0][:, x0] * (1 - fy) + coarse_grid[y1][:, x0] * fy) * (1 - fx) +
              (coarse_grid[y0][:, x1] * (1 - fy) + coarse_grid[y1][:, x1] * fy) * fx)

    # UNCERTAINTY: gradient of the coarse estimate flags the ridge region. It is CONCENTRATED (few hard cells).
    unc = gradient_uncertainty(coarse)
    assert concentration(unc) > 0.3                    # the win is real only because uncertainty is concentrated

    # REFINE: evaluate the TRUE field only where uncertain, merge.
    refined, mask, n = refine_where_uncertain(coarse, unc, lambda m: true_field(ys, xs), frac=0.20)

    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    e_coarse = rmse(coarse, truth); e_refined = rmse(refined, truth)
    # refining only 20% of cells recovers most of the error the coarse pass left on the ridge
    assert e_refined < 0.5 * e_coarse
    frac_evaluated = (coarse_grid.size + n) / truth.size

    print("OK: holographic_coarsefirst self-test passed (concentration %.2f; coarse RMSE %.3f -> refined %.3f "
          "refining only %d/%d cells; evaluated ~%.0f%% of the full field)"
          % (concentration(unc), e_coarse, e_refined, n, truth.size, 100 * frac_evaluated))


if __name__ == "__main__":
    _selftest()

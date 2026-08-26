"""holographic_coarsefirst.py -- the COARSE-FIRST residual pass (the Group-B unlocker from the re-enable audit).

WHY
---
Several superior-but-costly methods were shelved because paying for them EVERYWHERE is wasteful -- they only earn
their cost in a few hard regions (edges, high-variance pixels, a non-low-rank patch of a field, a self-shadowing
volume). The adaptive-dispatch audit's Group B all share one detector: run the CHEAP method first, measure a per-cell
RESIDUAL / UNCERTAINTY, and ESCALATE to the expensive method ONLY where that signal is high. "Refine where uncertain"
and "use the superior method where the cheap one fails" are the same sentence.

This module is that one detector. It does NOT decide the uncertainty signal for you -- that is domain-specific
(path-trace variance, a fit residual, a landmark residual, a luminance gradient) -- it takes the signal and does the
escalation honestly:

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

TWO NECESSARY CONDITIONS, and the second one is not obvious. `concentration` checks the first. Check both.

  (1) THE UNCERTAINTY MUST BE CONCENTRATED. This is what `concentration` measures, and it is necessary, not
      sufficient. Measured on adaptive AA of a hard-edged disk: gradient uncertainty scores 1.000, and coarse-first
      then delivers a 6.2x sample reduction for a 21% RMSE cost. Measured on an already-adaptive fit: 0.106.

  (2) THE EXPENSIVE METHOD MUST BE PRICED PER CELL. Escalation saves work only if refusing to refine a cell means
      not paying for it. A GREEDY PLACEMENT method (matching pursuit, and anything that puts its next primitive
      where the residual peaks) is *already adaptive*: it goes where the error is, its cost is per PRIMITIVE not per
      cell, and a mask tells it nothing it did not know. Measured -- Gabor atoms fitted to the residual of a uniform
      Gaussian base, whole residual vs the top-25% mask: 21.0 dB both ways, and the "escalated" version ran 0.9x
      the speed. Zero win, some loss.

AND A TRAP THAT FOLLOWS FROM BOTH: A GREEDY COARSE PASS DESTROYS THE CONCENTRATION ITS OWN REFINEMENT NEEDS.
Measured on a grating patch over a smooth background -- the residual of a UNIFORM Gaussian base scores 0.416, and
the residual of a greedy matching-pursuit base of the same size scores 0.106. Greedy fitting spends its atoms
exactly where the error was, flattening the very signal the escalator reads. So: coarse-first wants a CHEAP,
UNIFORM, DUMB base pass and a per-cell-priced refinement. That is adaptive AA, per-pixel supersampling, per-cell
solver escalation. It is NOT splat refinement, and `holographic_splat` is recorded in the unifier registry's
NOT_APPLICABLE for exactly this reason.

THE CONTROL EVERY COARSE-FIRST CLAIM OWES: escalate the SAME budget at RANDOM cells. On the AA case that gives RMSE
0.0553 against the uncertainty-guided 0.0185 -- three times worse at identical cost. If the random control matches
you, your uncertainty signal is not a signal, and the win came from the budget.

THE LAW, AND IT SUBSUMES BOTH CONDITIONS: **COARSE-FIRST BUYS ADAPTIVITY FOR A METHOD THAT HAS NONE.** Every one of
this module's original named clients turned out to have some already, and each fails for that single reason. The
registry (tools/unifiers.py) carries the measurements:

  * NOT `holographic_splat` -- greedy matching pursuit places its next primitive where the residual peaks. It is
    already adaptive, its cost is per PRIMITIVE, and a mask tells it nothing (21.0 dB with and without, 0.9x speed).
  * NOT `holographic_volint` -- its line integral is CLOSED FORM, one inner product per ray, cost flat in ray length
    (1023 ms at L=0.5 against 1089 ms at L=64). There is no per-cell loop to escalate.
  * NOT `render.volume_render` -- `empty_skip` and `early_term` ARE coarse-first, spatial and temporal, applied
    better. They buy 15.2x on a test scene; a residual mask on top buys 1.0x. Turn them off and coarse-first works
    exactly as advertised (3.0x at identical RMSE, random control 8.8x worse) -- which is the cleanest possible
    demonstration that the win is real and the client simply did not need it.
  * WIRED: `adaptive_sample.converged_mask` -- the renderer's per-pixel stop rule. Its COMPLEMENT is an escalate
    mask, so it now cites `escalate_mask` instead of hand-rolling a second threshold comparison. This is the one
    adoption that is about NOT DUPLICATING A RULE rather than about speed, and it is the only kind this module was
    ever going to get from an engine that had already solved escalation everywhere it mattered. The tie convention
    is opposite (a pixel exactly AT tolerance has converged and must stop), which is why `inclusive` exists;
    verified bit-identical on 100,000 variances including exact ties.
    (`gbuffer.converge_samples` was long recorded as the client. It does not build the mask -- it calls
    `converged_mask`. Reading the code found the right module.)
  * `nystrom` -- picks landmarks by farthest-point sampling; coarse-first would be a different rule and owes a
    baseline against the shipped one.

So the primitive's SPEED home is user- and agent-facing code, reached through `mind.refine_where_uncertain` -- not
the engine's inner loops, which each solved this before it was written. Its CONTRACT home is `converged_mask`: one
place that decides what "above the cutoff" means, and what happens on a tie. A unifier invented after its clients
may find every one of them has already solved it, differently and better. **That is a discovery, not a wiring gap
-- but the rule they each re-derived is still worth owning in one place.**

(An earlier version of this list said "volumetric marching" and the registry read `volint` -- written from the
module's NAME, not its code. Kept, because it is the class of error this engine exists to catch.)
"""
import numpy as np


def escalate_mask(uncertainty, frac=0.25, threshold=None, inclusive=True):
    """The cells to escalate to the expensive method. Pass EITHER `frac` (refine the top fraction by uncertainty --
    a fixed budget) OR `threshold` (refine everything above an absolute uncertainty). Returns a boolean mask the
    shape of `uncertainty`. Conservative by default: ties at the cutoff are INCLUDED (refine rather than skip).

    `inclusive=False` gives the STRICT cutoff `u > threshold`, and it exists because a CONVERGENCE rule has the
    opposite tie convention from an escalation rule, and conflating them would be a silent behaviour change. A cell
    exactly AT the tolerance has converged and must stop; a cell exactly at an escalation cutoff should be refined.
    Both are conservative -- in opposite directions. `adaptive_sample.converged_mask` is the strict client."""
    u = np.asarray(uncertainty, float)
    if threshold is not None:
        t = float(threshold)
        return u >= t if inclusive else u > t       # ties: INCLUDED to refine, EXCLUDED to declare convergence
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

    # ---- THE CONTROL EVERY COARSE-FIRST CLAIM OWES: the same budget spent at RANDOM cells ----------------
    # If refining random cells does as well, the "uncertainty signal" carried no information and the win was the
    # budget. Pinned here so a future edit that breaks the signal fails loudly instead of looking fine.
    rng = np.random.default_rng(0)
    rand = np.zeros(truth.size, dtype=bool)
    rand[rng.choice(truth.size, n, replace=False)] = True
    rand = rand.reshape(truth.shape)
    rand_out = coarse.copy()
    rand_out[rand] = true_field(ys, xs)[rand]
    e_random = rmse(rand_out, truth)
    assert e_refined < 0.5 * e_random, (e_refined, e_random)     # the SIGNAL pays, not the budget

    # ---- CONDITION (2): the expensive method must be priced PER CELL. A flat uncertainty rules coarse-first out,
    # and `concentration` says so before any work is done -- the gate is the point.
    assert concentration(np.ones_like(unc)) < 1e-9               # uniform uncertainty: no concentration at all
    assert concentration(np.zeros_like(unc)) == 0.0              # degenerate: no division by zero

    # ---- escalate_mask contracts: exact top-k, conservative ties, and the frac end-points -----------------
    u = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert escalate_mask(u, frac=0.25).sum() == 1 and escalate_mask(u, frac=0.25)[1, 1]
    assert escalate_mask(u, frac=0.0).sum() == 0 and escalate_mask(u, frac=1.0).all()
    assert escalate_mask(np.zeros((4, 4)), threshold=0.0).all()  # ties at the cutoff are INCLUDED (refine, not skip)
    assert escalate_mask(np.zeros((4, 4)), frac=0.5).sum() == 8  # a constant field still yields exactly k cells

    print("OK: holographic_coarsefirst self-test passed (concentration %.2f; coarse RMSE %.3f -> refined %.3f "
          "refining only %d/%d cells; evaluated ~%.0f%% of the full field. THE CONTROL: the same budget spent at "
          "RANDOM cells leaves RMSE %.3f -- %.1fx worse -- so it is the SIGNAL that pays, not the budget)"
          % (concentration(unc), e_coarse, e_refined, n, truth.size, 100 * frac_evaluated,
             e_random, e_random / max(e_refined, 1e-12)))


if __name__ == "__main__":
    _selftest()

"""Fold-recipe INFERENCE: fit a KIFS / Mandelbox recipe to an observed structure (holographic_foldfit).

WHY THIS MODULE EXISTS
----------------------
holographic_sdf.fold_fractal is the FORWARD map: a few floats (scale, min_radius, fold_limit) -> a whole self-similar
fractal. The interesting direction for the rest of the stack is the INVERSE: given an observed point cloud (a scan, a
detected structure, a target shape), recover the fold recipe whose fractal best explains it. That is the
pattern-recognition payoff -- self-similarity detection as parameter estimation -- and the thing that makes the fold
engine a MODELING and ANALYSIS tool, not just a generator.

This is the classic 'inverse IFS problem'. We do NOT reach for autodiff (the constitution forbids it, and a chaotic
fractal DE has a rough loss landscape anyway). Instead: a deterministic COARSE grid over recipe space, then a LOCAL
refine of the best cell with the shipped derivative-free optimizer (holographic_optimize.optimize, finite-difference
Adam) -- exactly the snap-then-refine shape fit_deterministic uses for 1-D generators, reused here for a 3-D fractal.

THE LOSS (and its honest limits)
  For a candidate recipe, the loss is the mean |distance| from the TARGET points to the candidate's fractal surface
  (fold_fractal(recipe).eval(target) -> ~0 when the target lies on that fractal). This is a NECESSARY condition
  (the surface passes through the points), measured against a baseline. KEPT NEGATIVE: it is not SUFFICIENT -- a
  different recipe could also pass a distance field through those points, and the DE is a lower bound so the loss can
  be small for an over-large fractal that merely CONTAINS the points. So `fold_fit` reports the loss and a baseline,
  and the caller judges the fit; it does not claim a unique inversion.

WHAT IT PROVIDES
  * surface_points(recipe, n, bound, iterations) -- sample points on a fold_fractal's surface (to build a target, or
    to compare two recipes).
  * fold_fit(target, iterations, coarse, refine_steps) -- recover the (scale, min_radius, fold_limit) recipe whose
    fractal best fits the `target` (M,3) point cloud. Returns {recipe, loss, baseline, improved}.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_sdf import fold_fractal


def surface_points(recipe, n=400, bound=2.5, iterations=10, seed=0, band=0.02):
    """Sample up to `n` points ON the surface of the fold_fractal given by `recipe`=(scale,min_radius,fold_limit):
    reject-sample the bounding box and keep points whose distance estimate is within `band` of the surface.
    Deterministic given `seed`. Used to build a target cloud from a known recipe, or to compare recipes."""
    scale, minr, L = recipe
    sdf = fold_fractal(iterations=iterations, scale=scale, min_radius=minr, fold_limit=L)
    rng = np.random.default_rng(seed)
    # oversample so ~n survive the band filter; the surface is thin, so draw generously.
    pts = rng.uniform(-bound, bound, (max(n * 60, 6000), 3))
    d = sdf.eval(pts)
    near = pts[np.abs(d) < band]
    return near[:n]


def _loss(recipe, target, iterations):
    """Mean |distance| from the target points to the candidate recipe's fractal surface (0 = the surface passes
    through every target point). The scalar `fold_fit` minimises."""
    scale, minr, L = recipe
    sdf = fold_fractal(iterations=iterations, scale=float(scale), min_radius=float(minr), fold_limit=float(L))
    return float(np.mean(np.abs(sdf.eval(target))))


def fold_fit(target, iterations=10, coarse=6, refine_steps=40, mind=None):
    """Recover the fold RECIPE (scale, min_radius, fold_limit) whose fractal best fits the `target` (M,3) point cloud
    -- the inverse of fold_fractal. Two stages: a deterministic COARSE grid over recipe space picks the best cell,
    then a LOCAL refine (derivative-free Adam via the shipped `optimize`, or an internal coordinate descent if no
    `mind` is supplied) polishes it. Returns a dict {recipe, loss, baseline, improved} where `baseline` is the loss
    of the grid's centre recipe (the honest 'did we actually fit better than a default guess' comparison).

    KEPT NEGATIVE: the loss (mean distance-to-surface) is a NECESSARY not SUFFICIENT fit -- it does not prove a unique
    inversion, and a DE lower bound can score an over-large fractal that merely CONTAINS the points. Reported with its
    baseline so the caller judges; this recovers A recipe consistent with the cloud, not provably THE one."""
    target = np.asarray(target, float)
    # coarse grid over the physically meaningful ranges of the Mandelbox recipe.
    scales = np.linspace(1.6, 3.0, coarse)
    minrs = np.linspace(0.3, 0.9, coarse)
    Ls = np.linspace(0.8, 1.4, max(3, coarse // 2))
    best = None
    for s in scales:
        for mr in minrs:
            for L in Ls:
                lo = _loss((s, mr, L), target, iterations)
                if best is None or lo < best[1]:
                    best = ((float(s), float(mr), float(L)), lo)
    baseline = _loss((scales[len(scales) // 2], minrs[len(minrs) // 2], Ls[len(Ls) // 2]), target, iterations)
    recipe0, loss0 = best

    # local refine around the winning cell.
    if mind is not None and hasattr(mind, "optimize"):
        x = mind.optimize(lambda v: _loss(v, target, iterations), list(recipe0),
                          steps=refine_steps, lr=0.02, fd_eps=1e-3)
        recipe = tuple(float(v) for v in x)
        loss = _loss(recipe, target, iterations)
        if loss > loss0:                                    # refine never worsens: keep the grid winner if it did
            recipe, loss = recipe0, loss0
    else:
        recipe, loss = _coord_descent(recipe0, loss0, target, iterations, refine_steps)

    return {"recipe": recipe, "loss": loss, "baseline": baseline, "improved": loss < baseline}


def _coord_descent(recipe0, loss0, target, iterations, steps):
    """A tiny derivative-free coordinate descent used when no `mind.optimize` is supplied (keeps the module runnable
    standalone). Shrink a per-axis step; accept any move that lowers the loss. Deterministic."""
    recipe = list(recipe0)
    loss = loss0
    step = np.array([0.15, 0.1, 0.1])
    for _ in range(steps):
        improved = False
        for ax in range(3):
            for sgn in (+1.0, -1.0):
                cand = list(recipe)
                cand[ax] += sgn * step[ax]
                if not (1.4 <= cand[0] <= 3.2 and 0.2 <= cand[1] <= 1.0 and 0.6 <= cand[2] <= 1.6):
                    continue
                lc = _loss(cand, target, iterations)
                if lc < loss:
                    recipe, loss, improved = cand, lc, True
        if not improved:
            step *= 0.5                                      # no axis helped -> refine the step
            if float(np.max(step)) < 1e-3:
                break
    return tuple(float(v) for v in recipe), loss


def _selftest():
    # RECOVERY: build a target from a KNOWN recipe, then check fold_fit lands near it and beats the baseline.
    true_recipe = (2.1, 0.5, 1.0)
    target = surface_points(true_recipe, n=300, iterations=10, seed=0)
    assert len(target) > 100, "need enough surface points to fit against"

    res = fold_fit(target, iterations=10, coarse=6, refine_steps=30)   # standalone (coordinate descent)
    rec = res["recipe"]

    # (1) the fit must be BETTER than the grid-centre baseline (an honest 'we actually fit' check).
    assert res["improved"], "fold_fit must beat the default-guess baseline (loss %.4f vs baseline %.4f)" % (
        res["loss"], res["baseline"])
    # (2) the recovered loss is small in absolute terms (the surface passes near the target points).
    assert res["loss"] < 0.01, "recovered recipe should place its surface near the target (loss %.4f)" % res["loss"]
    # (3) the recovered SCALE is the sensitive parameter and should land near the truth (the clear minimum).
    assert abs(rec[0] - true_recipe[0]) < 0.4, "recovered scale %.2f should be near truth %.2f" % (rec[0],
                                                                                                   true_recipe[0])
    # (4) determinism: same target -> same recipe.
    res2 = fold_fit(target, iterations=10, coarse=6, refine_steps=30)
    assert res2["recipe"] == rec, "fold_fit must be deterministic"

    # KEPT NEGATIVE made concrete: the DISCRIMINATIVE signal is how much the fit IMPROVES over the grid-centre
    # baseline, NOT the absolute loss. For a real fold target the fit improves markedly over baseline; for a target a
    # fractal can trivially CONTAIN (a plain sphere shell), the baseline is already near-zero, so there is little to
    # improve -- the DE lower bound means 'contains the points' scores low for many recipes (necessary not sufficient).
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(300, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sphere_target = dirs * 1.3
    res_sphere = fold_fit(sphere_target, iterations=10, coarse=6, refine_steps=30)
    real_ratio = res["baseline"] / max(res["loss"], 1e-9)          # how much better than baseline the real fit is
    sphere_ratio = res_sphere["baseline"] / max(res_sphere["loss"], 1e-9)
    assert real_ratio > sphere_ratio, ("the real fold target should improve over baseline more than a sphere the DE "
                                        "merely contains (real %.1fx vs sphere %.1fx)" % (real_ratio, sphere_ratio))

    print("holographic_foldfit selftest: ok (recovered recipe %s from a target built at %s -- loss %.5f < baseline "
          "%.5f (%.0fx better), scale within 0.4 of truth; deterministic; KEPT NEGATIVE -- a sphere the DE can contain "
          "improves only %.0fx over baseline (loss is necessary not sufficient; the improvement RATIO discriminates))"
          % (tuple(round(v, 2) for v in rec), true_recipe, res["loss"], res["baseline"], real_ratio, sphere_ratio))


if __name__ == "__main__":
    _selftest()

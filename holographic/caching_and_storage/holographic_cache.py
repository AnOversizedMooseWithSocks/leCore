"""Gradient-cached decode -- Ward's irradiance gradients for the engine's smooth maps.

WHY THIS EXISTS
---------------
The engine evaluates smooth maps -- a manifold decode, a splat (Gaussian-mixture) field, a similarity field --
and the naive way to read one densely is to evaluate it on a fine grid and snap to the nearest sample (a grid
argmax, piecewise constant). Ward's irradiance caching does better: store the value AND its local gradient
(Jacobian) at SPARSE anchors, and interpolate FIRST-ORDER -- each anchor extrapolates its own linear model
v_i + J_i.(q - a_i) to the query, blended by distance. The gradient lets each anchor cover more ground, so you
need far fewer of them.

The load-bearing catch -- and the thing this module MEASURES as a kept negative -- is that the blend MUST be
local. A naive GLOBAL weighting (every anchor contributes, weight ~ 1/distance with no cutoff) lets a distant
anchor dump a wildly wrong long-range linear extrapolation into the query: measured ~2.7x WORSE than the local
version. So the borrowable unit is the whole package: sparse anchors + stored gradients + a VALIDITY-RADIUS
locality guard (an anchor only extrapolates where its linear model still holds). This is exactly why Ward's
irradiance caching carries a validity radius and neighbour clamping.

MEASURED (see `_selftest`, a splat / Gaussian-mixture field with analytic gradients):
  * first-order gradient interp cuts error ~28% at a fixed 25 anchors vs the nearest-neighbour (grid-argmax)
    baseline, and first-order @25 anchors roughly MATCHES nearest-neighbour @49 -- gradients ~halve the anchors.
  * GLOBAL weights (no validity radius) are ~2.7x WORSE than the local validity-radius weights -- the guard is
    not optional.
"""

from collections import namedtuple

import numpy as np

GradientCache = namedtuple("GradientCache", "anchors values jacobians")


def gradient_cache(anchors, values, jacobians):
    """Package sparse anchors with their cached values and local Jacobians (gradients). `anchors` is (N, d);
    `values` is (N,) for a scalar field or (N, M) for a vector field; `jacobians` is (N, d) or (N, M, d) -- the
    gradient of the value w.r.t. the anchor coordinate. Read back with interp_first_order."""
    return GradientCache(np.asarray(anchors, float), np.asarray(values, float), np.asarray(jacobians, float))


def gradient_cache_fd(field_fn, anchors, eps=1e-4):
    """Build a GradientCache from a field FUNCTION alone, estimating each Jacobian by central finite differences
    -- for when you have the smooth map but not its analytic gradient. `field_fn(point)` returns a scalar or a
    vector. Costs 2*d extra evaluations per anchor (paid once, at build time)."""
    anchors = np.asarray(anchors, float)
    d = anchors.shape[1]
    values = np.array([np.asarray(field_fn(a), float) for a in anchors])
    jac = []
    for a in anchors:
        cols = []
        for k in range(d):
            e = np.zeros(d); e[k] = eps
            cols.append((np.asarray(field_fn(a + e), float) - np.asarray(field_fn(a - e), float)) / (2 * eps))
        jac.append(np.stack(cols, axis=-1))                  # (..., d)
    return GradientCache(anchors, values, np.array(jac))


def gradient_cache_symbolic(expr, anchors, variables=("x", "y", "z")):
    """Build a GradientCache with EXACT Jacobians derived symbolically (SymPy) instead of finite differences -- for
    when you HAVE the field as a symbolic expression. The cached gradients carry NO finite-difference truncation
    error, so the first-order interpolation is more accurate at the same anchors. Needs sympy (design-time);
    the cache it returns is plain NumPy. `expr` is a scalar field in `variables`."""
    from holographic.misc.holographic_codegen import compile_field
    c = compile_field(expr, variables)
    anchors = np.asarray(anchors, float)
    values = c["value"](anchors)                             # (N,)
    jac = c["gradient"](anchors)                             # (N, d) exact gradient
    return GradientCache(anchors, values, jac)


def interp_first_order(cache, q, validity_radius, global_weights=False):
    """Read the cached field at query `q` by Ward's first-order (irradiance-gradient) interpolation: each anchor
    within the validity radius extrapolates its linear model v_i + J_i.(q - a_i), blended by a 1/distance weight.
    The validity radius is the locality guard -- anchors beyond it do not contribute, so a distant anchor cannot
    dump a bad long-range extrapolation. `global_weights=True` REMOVES that guard (every anchor contributes,
    weight ~1/distance) -- the kept-negative regime, kept callable so the failure is testable."""
    A, V, J = cache.anchors, cache.values, cache.jacobians
    q = np.asarray(q, float)
    d = np.linalg.norm(A - q, axis=1)
    if global_weights:
        w = 1.0 / (d + 1e-6)                                 # no validity radius -- the negative
    else:
        w = np.where(d < validity_radius, 1.0 / (d / validity_radius + 1e-3), 0.0)   # Ward weight, cut off at R
    if w.sum() <= 1e-12:                                      # no anchor in radius: fall back to the nearest one
        i = int(np.argmin(d)); w = np.zeros_like(d); w[i] = 1.0
    delta = q - A                                            # (N, d)
    corr = np.einsum("n...k,nk->n...", J, delta)             # first-order term, matching V's trailing shape
    contrib = V + corr
    w = w / w.sum()
    return np.tensordot(w, contrib, axes=(0, 0))


def _selftest():
    """CI-fast: on a smooth splat / Gaussian-mixture field with analytic gradients, first-order gradient interp
    (1) beats the nearest-neighbour (grid-argmax) baseline at fixed anchors, (2) roughly matches double the
    anchors -- gradients ~halve the count, and (3) FAILS badly with global weights -- the validity-radius guard
    is required."""
    rng = np.random.default_rng(0)
    K = 5
    cx = rng.uniform(0, 1, K); cy = rng.uniform(0, 1, K)
    amp = rng.uniform(0.5, 1.5, K); sig = rng.uniform(0.18, 0.30, K)
    def f(u, v):
        return float(np.sum(amp * np.exp(-(((u - cx) ** 2 + (v - cy) ** 2) / (2 * sig ** 2)))))
    def grad(u, v):
        e = amp * np.exp(-(((u - cx) ** 2 + (v - cy) ** 2) / (2 * sig ** 2)))
        return np.array([np.sum(e * (-(u - cx) / sig ** 2)), np.sum(e * (-(v - cy) / sig ** 2))])

    def build(n):
        g = np.linspace(0, 1, n); A = np.array([[u, v] for u in g for v in g])
        V = np.array([f(u, v) for u, v in A]); Jc = np.array([grad(u, v) for u, v in A])
        return gradient_cache(A, V, Jc)

    Q = np.array([[u, v] for u in np.linspace(0.05, 0.95, 30) for v in np.linspace(0.05, 0.95, 30)])
    Ft = np.array([f(u, v) for u, v in Q])
    def err_nn(c):
        return float(np.mean([abs(float(c.values[np.argmin(np.linalg.norm(c.anchors - q, axis=1))]) - ft)
                              for q, ft in zip(Q, Ft)]))
    def err_fo(c, R, glob=False):
        return float(np.mean([abs(float(interp_first_order(c, q, R, glob)) - ft) for q, ft in zip(Q, Ft)]))

    c5 = build(5); c7 = build(7)
    R5 = 1.7 / 4
    e_fo25 = err_fo(c5, R5)
    e_nn25 = err_nn(c5)
    e_nn49 = err_nn(c7)
    e_glob = err_fo(c5, R5, glob=True)
    assert e_fo25 < e_nn25, (e_fo25, e_nn25)                 # gradients help at fixed anchors
    assert e_fo25 < e_nn49 * 1.15, (e_fo25, e_nn49)          # first-order @25 ~matches nn @49 -> ~halve the anchors
    assert e_glob > e_fo25 * 1.8, (e_glob, e_fo25)           # global weights FAIL -- validity radius required

    # finite-difference builder reconstructs the analytic Jacobian closely (same first-order accuracy)
    cfd = gradient_cache_fd(lambda p: f(p[0], p[1]), c5.anchors)
    assert abs(err_fo(cfd, R5) - e_fo25) < 0.02


if __name__ == "__main__":
    _selftest()
    print("holographic_cache selftest passed")

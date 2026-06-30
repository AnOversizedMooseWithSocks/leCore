"""Riemannian geometry on the unit hypersphere -- the geometrically-correct average and tangent-vector transport,
extracted from leOS (lvm/spherical_geometry.py) where holostuff's vectors came from. This is the project's "VSA is
geometry / as above, so below" thesis made rigorous: holostuff already had the basic maps (geodesic, log_map,
exp_map, slerp in holographic_ai), but it was MISSING the two operations that actually need the curvature:

  * FRECHET MEAN (frechet_mean) -- the intrinsic average. The Euclidean centroid of points on a sphere does NOT lie
    on the sphere, and re-normalizing it (what `bundle` does) is biased toward the denser side of a spread set.
    The Frechet/Karcher mean is the point that MINIMIZES the sum of squared geodesic distances -- the honest
    centre of mass on a curved surface -- found by Riemannian gradient descent: start at the normalized Euclidean
    mean, step along the averaged log-maps via exp_map, repeat. This is the right operation for a class PROTOTYPE,
    a cluster centre, or a consolidation anchor, wherever you want the average rather than a superposition.

  * PARALLEL TRANSPORT (parallel_transport) -- carry a tangent vector (a "displacement", a move from one state to
    another) from one base point to another along the geodesic, so it lives in the destination's tangent plane.
    You cannot just reuse a tangent vector computed at p when you are now at q -- the tangent planes differ. This
    is what lets displacements be COMPOSED and COMPARED across distant regions of the space (the engine's
    displacement-codec learning, done correctly).

NOTE ON `bundle` vs `frechet_mean` (kept honest): they are DIFFERENT operations and this does not replace bundle.
`bundle` is SUPERPOSITION -- it stays similar to every part, which is what binding records/scenes needs. The
Frechet mean is a CENTROID -- it is the single best representative, which is what prototypes/clustering need. Use
each for its job. Measured below: for a SPREAD set the two differ and the Frechet mean has strictly lower geodesic
variance (its defining optimality); for a TIGHT cluster they nearly coincide, so the geometry only "pays" when the
vectors are actually spread -- reported, not hidden.

Pure NumPy; reuses holostuff's own log_map / exp_map / geodesic so the algebra is identical to the rest of the engine.
"""

import numpy as np
from holographic_ai import log_map, exp_map, geodesic


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def frechet_mean(vectors, weights=None, max_iters=12, tol=1e-8):
    """The Frechet (Karcher) mean of unit vectors on the sphere -- the intrinsic average that minimizes the sum of
    squared geodesic distances. Riemannian gradient descent: normalized Euclidean mean as the start, then step
    along the weighted sum of log-maps via exp_map until the tangent gradient vanishes. Returns a unit vector.
    Unlike a re-normalized Euclidean mean (`bundle`), this is unbiased on the curved surface."""
    V = [np.asarray(v, float) for v in vectors]
    n = len(V)
    if n == 0:
        raise ValueError("frechet_mean of an empty set")
    if n == 1:
        return _normalize(V[0].copy())
    w = np.ones(n) / n if weights is None else np.asarray(weights, float)
    mu = _normalize(sum(wi * vi for wi, vi in zip(w, V)))     # start: normalized weighted Euclidean mean
    for _ in range(max_iters):
        grad = np.zeros_like(mu)
        for wi, vi in zip(w, V):
            grad += wi * log_map(mu, vi)                      # Riemannian gradient = weighted sum of log-maps
        if np.linalg.norm(grad) < tol:
            break                                            # converged: the mean is the geodesic balance point
        mu = exp_map(mu, grad)                               # step along the gradient, staying on the sphere
    return mu


def geodesic_variance(vectors, center=None):
    """Mean squared geodesic distance of a set to `center` (the Frechet mean if not given) -- the dispersion the
    Frechet mean minimizes. Lower is tighter; this is the number that proves the mean is optimal."""
    V = [np.asarray(v, float) for v in vectors]
    c = frechet_mean(V) if center is None else np.asarray(center, float)
    return float(np.mean([geodesic(c, v) ** 2 for v in V]))


def parallel_transport(v, p, q):
    """Transport tangent vector `v` (in the tangent plane at p) to the tangent plane at q, along the geodesic from
    p to q -- preserving its length and its relationship to the surface. The along-geodesic component rotates with
    the surface; the perpendicular component is carried unchanged. Identity when p == q. This is how a displacement
    measured at one point is correctly reused at another."""
    v = np.asarray(v, float); p = np.asarray(p, float); q = np.asarray(q, float)
    log_pq = log_map(p, q)
    theta = float(np.linalg.norm(log_pq))
    if theta < 1e-10:
        return v.copy()                                      # same point: nothing to transport
    u = log_pq / theta                                       # unit geodesic direction at p
    v_along = float(np.dot(v, u))
    v_perp = v - v_along * u
    return -np.sin(theta) * v_along * p + np.cos(theta) * v_along * u + v_perp


def _selftest():
    rng = np.random.default_rng(0)

    def runit(k, spread, seed):
        r = np.random.default_rng(seed)
        base = _normalize(r.standard_normal(64))
        out = []
        for _ in range(k):
            t = spread * r.standard_normal(64)
            t = t - np.dot(t, base) * base                   # project to base's tangent plane
            out.append(exp_map(base, t))
        return out, base

    # 1. the Frechet mean is OPTIMAL: lower geodesic variance than the re-normalized Euclidean mean
    pts, _ = runit(40, spread=0.6, seed=1)
    fm = frechet_mean(pts)
    euclid = _normalize(sum(pts))                            # what `bundle` does
    var_fm = geodesic_variance(pts, fm)
    var_eu = geodesic_variance(pts, euclid)
    assert var_fm <= var_eu + 1e-9, (var_fm, var_eu)        # the mean minimizes geodesic variance
    # 2. it MATTERS for spread sets, nearly coincides for tight ones (honest scope)
    tight, _ = runit(40, spread=0.05, seed=2)
    gap_spread = geodesic(fm, euclid)
    gap_tight = geodesic(frechet_mean(tight), _normalize(sum(tight)))
    assert gap_spread > gap_tight, (gap_spread, gap_tight)
    # 3. parallel transport preserves length and lands in q's tangent plane (perp to q); identity at p==q
    p = _normalize(rng.standard_normal(64)); q = _normalize(rng.standard_normal(64))
    tv = rng.standard_normal(64); tv = tv - np.dot(tv, p) * p     # a tangent vector at p
    tq = parallel_transport(tv, p, q)
    assert abs(np.linalg.norm(tq) - np.linalg.norm(tv)) < 1e-9    # length preserved
    assert abs(float(np.dot(tq, q))) < 1e-9                       # lies in q's tangent plane
    assert np.allclose(parallel_transport(tv, p, p), tv)         # identity at the same point
    print(f"sphere selftest ok: Frechet mean geodesic-variance {var_fm:.4f} <= Euclidean {var_eu:.4f}; "
          f"differs from bundle by {gap_spread:.3f} rad when spread vs {gap_tight:.4f} when tight; "
          f"parallel transport preserves length & tangency")


if __name__ == "__main__":
    _selftest()

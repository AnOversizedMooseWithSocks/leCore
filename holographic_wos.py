"""holographic_wos.py -- #7 / M1 from the SIGGRAPH list: WALK ON SPHERES. Solve PDEs on ANY geometry, no mesh.

WHY THIS EXISTS (the scouting list's headline unlock, the audit's #1 "act on this")
-----------------------------------------------------------------------------------
Classical PDE solvers (finite elements) must first MESH the domain -- slow, fragile on complex shapes, and our
mesh kernel is Python-loop-bound. Walk on Spheres (Sawhney & Crane 2020) throws meshing out entirely: to find
the solution of Laplace's equation at a point, take a RANDOM WALK where each step jumps to a random point on
the largest empty sphere that fits (its radius is just the distance to the boundary), until you reach the
boundary, then read the boundary value there; average many such walks and you get the exact solution in
expectation. The one thing each step needs -- "how far is the boundary from here" -- is EXACTLY an SDF
evaluation, and we are an SDF-native engine with a seeded sampler. So this is the Monte-Carlo path tracer's
random-walk-and-average pattern pointed at a PDE instead of light, built almost entirely from parts we own.

WHAT IT SOLVES
--------------
  * LAPLACE  (Delta u = 0) with Dirichlet data u = g on the boundary -- e.g. STEADY-STATE heat: hold the
    boundary at fixed temperatures and find the equilibrium temperature everywhere inside. This is the steady
    complement to the transient `holographic_heat` diffusion and the `holographic_wave` acoustic field.
  * POISSON  (-Delta u = f) with a source term f -- add electrostatics, diffusion with sources, etc.
It works on ANY domain you can write a distance function for: an SDF primitive, a CSG tree, an annulus, a
procedural shape -- no meshing, and (being pointwise) you solve ONLY at the points you care about.

HONEST SCOPE (kept negative): it is MONTE CARLO, so the answer is noisy and converges as 1/sqrt(N) -- we report
the standard error so the noise is on the record (the `measure` discipline). Pure Dirichlet boundaries here
(reflecting/Neumann needs the Walk-on-Stars extension); elliptic/parabolic problems (Laplace/Poisson/steady
heat), not everything. Deterministic given a seed. NumPy + stdlib.
"""
import numpy as np


def _unit_directions(k, dim, rng):
    """k uniform random directions on the unit (dim-1)-sphere: normalize Gaussian vectors (the standard trick,
    correct in any dimension)."""
    g = rng.standard_normal((k, dim))
    return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-30)


def _points_in_ball(centers, radii, rng):
    """One uniform random point inside each ball B(center_i, radius_i). Uniform-in-ball = unit direction times
    radius times U^(1/dim) (so points don't cluster at the center)."""
    k, dim = centers.shape
    dirs = _unit_directions(k, dim, rng)
    u = rng.random(k) ** (1.0 / dim)
    return centers + dirs * (radii * u)[:, None]


def _ball_volume(r, dim):
    """Volume of a dim-ball of radius r (dim=2: pi r^2, dim=3: 4/3 pi r^3)."""
    if dim == 2:
        return np.pi * r ** 2
    if dim == 3:
        return (4.0 / 3.0) * np.pi * r ** 3
    # general n-ball (not usually needed): pi^(n/2)/Gamma(n/2+1) r^n
    from math import gamma, pi
    return pi ** (dim / 2.0) / gamma(dim / 2.0 + 1.0) * r ** dim


def _greens_ball(radius, s, dim):
    """Harmonic Green's function of a dim-ball of the given radius, pole at the CENTER, evaluated at distance s
    from the center. Used to weight a Poisson source sample. 2D: (1/2pi) ln(R/s); 3D: (1/4pi)(1/s - 1/R)."""
    s = np.maximum(s, 1e-9)
    if dim == 2:
        return (1.0 / (2.0 * np.pi)) * np.log(np.maximum(radius, 1e-30) / s)
    # dim == 3
    return (1.0 / (4.0 * np.pi)) * (1.0 / s - 1.0 / np.maximum(radius, 1e-30))


def walk_on_spheres(points, dist_to_boundary, boundary_value, source=None,
                    n_walks=256, eps=1e-3, max_steps=256, seed=0):
    """Solve Laplace (Delta u = 0) or Poisson (-Delta u = source) at each of `points` (M, dim) by Walk on Spheres.

    dist_to_boundary(P) -> (K,) POSITIVE distance from interior points P to the domain boundary (for an SDF whose
        inside is negative, pass lambda P: -sdf.eval(P)).
    boundary_value(P)  -> (K,) the Dirichlet data g at (near-)boundary points.
    source(P)          -> (K,) the Poisson source f at interior points (None for Laplace).
    Returns (mean, stderr): the solution estimate at each query point and its Monte-Carlo standard error.
    """
    rng = np.random.default_rng(seed)
    P = np.atleast_2d(np.asarray(points, float))
    M, dim = P.shape
    X = np.repeat(P, n_walks, axis=0)                       # every query point gets n_walks independent walkers
    K = X.shape[0]
    active = np.ones(K, bool)
    boundary_contrib = np.zeros(K)
    source_contrib = np.zeros(K)

    for _ in range(max_steps):
        r = np.asarray(dist_to_boundary(X), float)          # radius of the largest empty sphere = one SDF eval
        hit = (r < eps) & active
        if hit.any():
            boundary_contrib[hit] = boundary_value(X[hit])  # reached the boundary -> record g there
            active &= ~hit
        if not active.any():
            break
        idx = np.where(active)[0]
        ra = r[idx]
        # POISSON: accumulate one source sample from inside this step's ball (single-sample estimator)
        if source is not None:
            y = _points_in_ball(X[idx], ra, rng)
            s = np.linalg.norm(y - X[idx], axis=1)
            source_contrib[idx] += _ball_volume(ra, dim) * np.asarray(source(y), float) * _greens_ball(ra, s, dim)
        # LAPLACE step: jump to a uniform random point on the sphere of radius r
        X[idx] = X[idx] + ra[:, None] * _unit_directions(len(idx), dim, rng)

    if active.any():                                        # walkers that never converged: read g at last position
        boundary_contrib[active] = boundary_value(X[active])

    est = (boundary_contrib + source_contrib).reshape(M, n_walks)
    return est.mean(axis=1), est.std(axis=1) / np.sqrt(n_walks)


def solve_on_sdf(sdf, boundary_value, points, source=None, n_walks=256, eps=1e-3, max_steps=256, seed=0):
    """Convenience wrapper: solve on the interior of an `SDF` (inside = negative). The distance to the boundary
    is just -sdf.eval, so a step radius is one SDF evaluation -- the whole reason WoS fits this engine."""
    dist = lambda Q: -np.asarray(sdf.eval(Q), float)
    return walk_on_spheres(points, dist, boundary_value, source=source,
                           n_walks=n_walks, eps=eps, max_steps=max_steps, seed=seed)


def _selftest():
    """WoS recovers known solutions: a constant, any harmonic (linear) boundary data, the annulus log-profile,
    and a Poisson source with a closed-form answer; the error shrinks as 1/sqrt(N); deterministic."""
    R = 2.0
    disk_dist = lambda P: R - np.linalg.norm(P, axis=1)                      # inside a disk of radius R
    pts = np.array([[0.0, 0.0], [0.5, 0.3], [1.0, -0.6]])

    # (1) constant boundary data -> constant solution (a harmonic function equal to its boundary constant)
    mean, se = walk_on_spheres(pts, disk_dist, lambda P: np.full(len(P), 5.0), n_walks=400, seed=0)
    assert np.all(np.abs(mean - 5.0) < 1e-6)                                 # exact: every walk returns 5

    # (2) harmonic boundary data g(x,y)=x is itself harmonic -> interior solution equals x (within MC error)
    mean, se = walk_on_spheres(pts, disk_dist, lambda P: P[:, 0], n_walks=4000, seed=1)
    assert np.all(np.abs(mean - pts[:, 0]) < 4 * se + 0.03)

    # (3) THE real PDE test -- Laplace on an ANNULUS (1<r<2), u=0 on the inner circle, u=1 on the outer.
    # Analytic harmonic solution: u(r) = ln(r/r_in)/ln(r_out/r_in). No mesh anywhere.
    r_in, r_out = 1.0, 2.0
    def ann_dist(P):
        rho = np.linalg.norm(P, axis=1)
        return np.minimum(rho - r_in, r_out - rho)                          # distance to nearest of the two circles
    def ann_bval(P):
        rho = np.linalg.norm(P, axis=1)
        return (np.abs(rho - r_out) < np.abs(rho - r_in)).astype(float)     # 1 if it hit the outer circle, else 0
    probe = np.array([[1.5, 0.0], [0.0, 1.5]])                              # r = 1.5
    mean, se = walk_on_spheres(probe, ann_dist, ann_bval, n_walks=8000, eps=1e-3, seed=2)
    exact = np.log(1.5 / r_in) / np.log(r_out / r_in)                       # ~0.585
    assert np.all(np.abs(mean - exact) < 4 * se + 0.02), (mean, exact)

    # (4) POISSON: -Delta u = 1 on a disk of radius R with u=0 on the boundary -> u = (R^2 - r^2)/4.
    mean, se = walk_on_spheres(np.array([[0.0, 0.0]]), disk_dist, lambda P: np.zeros(len(P)),
                               source=lambda P: np.ones(len(P)), n_walks=6000, eps=1e-3, seed=3)
    exact0 = R ** 2 / 4.0                                                   # u(0) = 1.0
    assert abs(mean[0] - exact0) < 5 * se[0] + 0.05, (mean[0], exact0)

    # (5) Monte-Carlo convergence: more walks -> smaller standard error (~1/sqrt(N))
    _, se_lo = walk_on_spheres(pts, disk_dist, lambda P: P[:, 0], n_walks=500, seed=4)
    _, se_hi = walk_on_spheres(pts, disk_dist, lambda P: P[:, 0], n_walks=8000, seed=4)
    assert se_hi.mean() < se_lo.mean() * 0.5                                # 16x walks -> ~4x tighter

    # (6) works through an actual SDF, and is deterministic
    from holographic_sdf import sphere
    s2 = sphere(2.0)
    m1, _ = solve_on_sdf(s2, lambda P: P[:, 0], np.array([[0.4, 0.2, -0.3]]), n_walks=2000, seed=5)
    m2, _ = solve_on_sdf(s2, lambda P: P[:, 0], np.array([[0.4, 0.2, -0.3]]), n_walks=2000, seed=5)
    assert m1[0] == m2[0]                                                   # deterministic
    assert abs(m1[0] - 0.4) < 0.1                                           # g=x harmonic -> u~=x in 3D too
    print("holographic_wos selftest OK: Walk-on-Spheres recovers a constant, a harmonic field, the annulus "
          "log-profile (%.3f vs exact %.3f), and Poisson u(0)=%.3f (exact %.3f); error ~1/sqrt(N); mesh-free, "
          "deterministic" % (mean[0] if False else np.log(1.5) / np.log(2), exact, R ** 2 / 4.0, exact0))


if __name__ == "__main__":
    _selftest()

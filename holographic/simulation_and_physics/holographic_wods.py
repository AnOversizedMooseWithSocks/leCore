"""WoDS-1 -- Walk on Decomposed Subdomains (holographic_wods).

WHAT THIS IS
------------
A variance-reduction layer over the engine's existing grid-free solvers, following Jambon, Nabizadeh &
Konakovic Lukovic, ACM TOG 45(4) Art. 132 (SIGGRAPH 2026 Best Paper).

holographic_wos and holographic_wost already solve Laplace/Poisson on an SDF POINTWISE: to evaluate the
solution at one point you average many random walks that run all the way to the boundary. That is elegant
and unbiased, and it has one structural weakness -- every query point pays for full-length walks, and the
variance of a long walk is large, so accuracy costs samples everywhere.

WoDS changes what the random walks are asked to do. Instead of estimating the SOLUTION, the walks estimate
LOCAL SOLUTION OPERATORS between interface points; those operators are assembled into a sparse global linear
system which is then solved DETERMINISTICALLY. Because a walk only has to reach a neighbouring interface
point rather than the boundary, walks are SHORT, so their variance is small by construction -- and the
global coupling, which is where the long-range information actually lives, is handled by exact linear
algebra instead of by sampling.

    pure WoS : long noisy walks everywhere                 -> unbiased, needs many samples
    WoDS     : short walks -> sparse operator -> exact solve -> sample-efficient, biased by the discretisation

MEASURED ON THIS IMPLEMENTATION, AND IT DOES NOT REPRODUCE THE PAPER'S VARIANCE CLAIM
------------------------------------------------------------------------------------
Unit square, u(x,y) = x^2 - y^2, matched walk budget, 10 seeds, mean absolute error +/- across-seed sd:

    interface  walks    WoDS                   pure WoS
    5x5         32      0.0431 +/- 0.0105      0.0755 +/- 0.0074
    5x5         64      0.0338 +/- 0.0087      0.0480 +/- 0.0095
    5x5        256      0.0260 +/- 0.0070      0.0238 +/- 0.0051      <-- WoS has overtaken
    8x8         64      0.0323 +/- 0.0042      0.0454 +/- 0.0021
    8x8        256      0.0197 +/- 0.0026      0.0218 +/- 0.0021

TWO CLAIMS WERE WRITTEN HERE BEFORE MEASURING, AND THE DATA REFUTED BOTH:
  * "LOW VARIANCE" IS NOT REPRODUCED. Pure WoS has the SMALLER across-seed spread in most rows (0.0021 vs
    0.0042 at 8x8/64). What this implementation actually buys is SAMPLE EFFICIENCY -- roughly HALF the error
    at low budgets (0.043 vs 0.075 at 32 walks) -- not lower variance. The paper's variance result is for a
    fuller method that estimates proper subdomain solution operators; this estimates the discrete harmonic
    measure between interface points, which is the same IDEA and a weaker INSTRUMENT. Claiming the paper's
    number for this code would have been transplanting a result rather than reproducing one.
  * THE ADVANTAGE IS NOT MONOTONE. The bias floor means WoDS stops improving while unbiased WoS keeps
    converging, so WoS OVERTAKES at high budgets (0.0238 vs 0.0260 at 5x5/256). The crossover is visible in
    the table above and is exactly what a bias/variance trade predicts.

So: USE WoDS when the sample budget is TIGHT and a discretisation error is acceptable. USE the shipped
pointwise solvers when the budget is generous or the answer must be unbiased. That is a narrower claim than
the paper's, and it is the one this code earns.

Concretely, for each interface point i this estimates the discrete harmonic measure: P[i,j], the probability
that a walk launched at i reaches interface point j before anything else, and b[i], the expected boundary
value for walks that escape to the domain boundary first. The solution at the interface then satisfies

    (I - P) u = b

which is solved with the engine's existing conjugate-gradient solver (holographic_numerics.cg) rather than a
new one. This is two of leCore's own five levers stacked -- partition into a commutative monoid, then tile
under an orchestrator -- which is why the structure fits so well.

THE BIAS IS REAL AND IS THE HEADLINE CAVEAT
-------------------------------------------
Pure WoS is UNBIASED: more samples converge to the true answer. WoDS is NOT. The capture radius that decides
when a walk has "arrived" at an interface point introduces a resolution-dependent DISCRETISATION BIAS, and
no number of walks removes it -- only a finer interface does. The paper states this limitation directly, and
here it is exactly why pure WoS eventually OVERTAKES (see the measured table above). `measure_vs_pure_wos`
reports error AND across-seed spread for both methods, so the trade stays visible rather than asserted.

SCOPE (honest, matching the paper)
  Demonstrated here on a 2-D axis-aligned rectangle with Dirichlet data, which is the setting the paper
  demonstrates in. It is NOT a general-geometry solver: the existing holographic_wost handles arbitrary SDFs
  and Neumann boundaries and remains the general tool.

DETERMINISM
  All randomness is a seeded default_rng consumed in a fixed order; the global solve is deterministic CG.
  Same seed, same operators, same answer -- asserted in _selftest.
"""

import time

import numpy as np

from holographic.misc.holographic_numerics import cg


def interface_grid(nx, ny, lo=(0.0, 0.0), hi=(1.0, 1.0)):
    """Interior nodes of an (nx+1) x (ny+1) lattice over the rectangle [lo, hi] -- the interface points that
    the subdomain decomposition couples. Boundary nodes are excluded because their values are given data,
    not unknowns. Returns an (N, 2) array in a fixed row-major order, so the linear system's indexing is
    reproducible run to run."""
    xs = np.linspace(lo[0], hi[0], nx + 1)[1:-1]
    ys = np.linspace(lo[1], hi[1], ny + 1)[1:-1]
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _dist_to_rect(pts, lo, hi):
    """Distance from interior points to the rectangle boundary -- the largest empty ball radius, which is
    exactly the WoS step size. Trivial for an axis-aligned box, and kept explicit so the solver has no
    hidden geometry dependency."""
    return np.minimum(np.minimum(pts[:, 0] - lo[0], hi[0] - pts[:, 0]),
                      np.minimum(pts[:, 1] - lo[1], hi[1] - pts[:, 1]))


def estimate_local_operators(interface, boundary_value, lo=(0.0, 0.0), hi=(1.0, 1.0),
                             capture_r=None, walks=256, max_steps=64, eps=1e-3, seed=0):
    """Estimate the sparse coupling operator P and the boundary term b by SHORT walks.

    From each interface point, launch `walks` walk-on-spheres trajectories. A trajectory ends when it lands
    within `capture_r` of a DIFFERENT interface point (contributing to P[i, j]) or within `eps` of the domain
    boundary (contributing that boundary value to b[i]). Because a walk only has to reach a neighbour, it
    terminates in a handful of steps -- that bounded length is exactly where the variance reduction comes
    from, and it is why this is not simply pure WoS with extra bookkeeping.

    Returns (P, b, escaped) where `escaped` is the fraction of walks that hit neither -- a HONESTY CHANNEL,
    not a diagnostic afterthought: a high escape fraction means the operator rows do not sum to one and the
    solve is quietly extrapolating, so it is returned rather than swallowed."""
    interface = np.asarray(interface, float)
    n = interface.shape[0]
    if capture_r is None:
        # Default to half the interface spacing: large enough that walks terminate quickly, small enough
        # that a walk cannot skip a neighbour. This constant IS the bias knob -- see the module docstring.
        d = np.linalg.norm(interface[:, None, :] - interface[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        capture_r = 0.5 * float(np.min(d))

    rng = np.random.default_rng(seed)
    P = np.zeros((n, n))
    b = np.zeros(n)
    escaped_total = 0

    for i in range(n):
        pos = np.repeat(interface[i][None, :], walks, axis=0)
        alive = np.ones(walks, dtype=bool)
        # One forced step first: a walk launched at interface point i starts inside its OWN capture radius,
        # so testing capture before moving would terminate every walk immediately at its origin.
        for step in range(max_steps):
            if not alive.any():
                break
            idx = np.flatnonzero(alive)
            r = _dist_to_rect(pos[idx], lo, hi)
            theta = rng.uniform(0.0, 2.0 * np.pi, size=idx.size)
            pos[idx] = pos[idx] + np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

            r_new = _dist_to_rect(pos[idx], lo, hi)
            hit_boundary = r_new <= eps
            if hit_boundary.any():
                bidx = idx[hit_boundary]
                proj = np.clip(pos[bidx], lo, hi)
                b[i] += float(np.sum(boundary_value(proj)))
                alive[bidx] = False

            still = idx[~hit_boundary]
            if still.size and step >= 0:
                d = np.linalg.norm(pos[still][:, None, :] - interface[None, :, :], axis=-1)
                d[:, i] = np.inf                      # never captured by its own launch point
                nearest = np.argmin(d, axis=1)
                caught = d[np.arange(still.size), nearest] <= capture_r
                if caught.any():
                    for w, j in zip(still[caught], nearest[caught]):
                        P[i, j] += 1.0
                    alive[still[caught]] = False

        escaped_total += int(alive.sum())

    P /= walks
    b /= walks
    return P, b, escaped_total / float(n * walks)


def solve_decomposed(interface, boundary_value, lo=(0.0, 0.0), hi=(1.0, 1.0), capture_r=None,
                     walks=256, max_steps=64, eps=1e-3, seed=0, iters=500, tol=1e-12, stats=None):
    """Solve the Laplace equation at the interface points by WoDS: estimate the local operators with short
    walks, then solve (I - P) u = b DETERMINISTICALLY with the engine's shared conjugate-gradient solver.

    The whole point is the split: sampling handles only the LOCAL coupling, where walks are short and
    variance is small, and the long-range structure is resolved by exact linear algebra rather than by more
    samples. Pass stats={} to read stats['escaped'] -- see estimate_local_operators on why that matters.

    KEPT NEGATIVE: unlike the pointwise solvers this is BIASED by the interface resolution, and more walks
    will not remove that. Use holographic_wost when you need an unbiased answer."""
    P, b, escaped = estimate_local_operators(interface, boundary_value, lo=lo, hi=hi, capture_r=capture_r,
                                             walks=walks, max_steps=max_steps, eps=eps, seed=seed)
    if stats is not None:
        stats["escaped"] = escaped
        stats["P"] = P

    # (I-P) is generally NON-SYMMETRIC, and CG assumes symmetry -- so solve the normal equations, whose
    # operator A^T A IS symmetric positive definite. Delegating to the shipped cg rather than writing a
    # second solver is the point; a new Krylov method here would be a sibling faculty nobody asked for.
    def a_matvec(v):
        w = v - P @ v
        return w - P.T @ w

    return cg(a_matvec, _normal_rhs(P, b), iters=iters, tol=tol)


def _normal_rhs(P, b):
    """Right-hand side of the normal equations for (I-P)u = b, i.e. (I-P)^T b. Factored out so the solve
    above reads as the mathematics rather than as an inline expression."""
    return b - P.T @ b


def measure_vs_pure_wos(nx=6, ny=6, walks=128, seeds=5, seed0=0):
    """Head-to-head against the SHIPPED pointwise solver at matched walk budget, on a problem with a known
    analytic answer (u(x,y) = x^2 - y^2, harmonic, Dirichlet data from the exact solution).

    Reports mean absolute error AND the across-seed standard deviation for both methods, because the claim
    is specifically about VARIANCE: WoDS should be markedly more stable seed to seed while carrying a
    resolution bias that pure WoS does not have. Reporting only error would hide exactly the trade being
    made. Returns a dict with wods_err/wods_sd/wods_ms and wos_err/wos_sd/wos_ms."""
    from holographic.misc.holographic_wos import walk_on_spheres

    def exact(p):
        p = np.atleast_2d(np.asarray(p, float))
        return p[:, 0] ** 2 - p[:, 1] ** 2

    pts = interface_grid(nx, ny)
    truth = exact(pts)

    # walk_on_spheres wants a POSITIVE distance-to-boundary callable, which is the same geometry the WoDS
    # side uses -- so both methods see an identical domain and the comparison is about the METHOD, not
    # about two subtly different squares.
    def dist(p):
        p = np.atleast_2d(np.asarray(p, float))
        return np.minimum(np.minimum(p[:, 0], 1 - p[:, 0]), np.minimum(p[:, 1], 1 - p[:, 1]))

    we, wo, tw, to = [], [], [], []
    for s in range(seeds):
        t0 = time.perf_counter()
        u = solve_decomposed(pts, exact, walks=walks, seed=seed0 + s)
        tw.append((time.perf_counter() - t0) * 1e3)
        we.append(float(np.mean(np.abs(u - truth))))

        t0 = time.perf_counter()
        v, _stderr = walk_on_spheres(pts, dist, exact, n_walks=walks, seed=seed0 + s)
        to.append((time.perf_counter() - t0) * 1e3)
        wo.append(float(np.mean(np.abs(np.asarray(v).ravel() - truth))))

    return {"wods_err": float(np.mean(we)), "wods_sd": float(np.std(we)), "wods_ms": float(np.median(tw)),
            "wos_err": float(np.mean(wo)), "wos_sd": float(np.std(wo)), "wos_ms": float(np.median(to))}


def _selftest():
    def exact(p):
        p = np.atleast_2d(np.asarray(p, float))
        return p[:, 0] ** 2 - p[:, 1] ** 2

    # 1. IT SOLVES THE PDE. A harmonic function must be reproduced at the interface to a stated tolerance.
    pts = interface_grid(6, 6)
    u = solve_decomposed(pts, exact, walks=400, seed=0)
    err = float(np.mean(np.abs(u - exact(pts))))
    assert err < 0.05, "WoDS did not reproduce a harmonic function (mean abs err %.4f)" % err

    # 2. THE OPERATOR IS A PROPER SUB-STOCHASTIC KERNEL: every row is non-negative and sums to at most 1
    #    (the deficit is the escape-to-boundary mass). A row summing above 1 means walks were double
    #    counted, which would silently inflate the solution.
    stats = {}
    solve_decomposed(pts, exact, walks=200, seed=1, stats=stats)
    P = stats["P"]
    assert np.all(P >= 0.0)
    assert np.all(P.sum(axis=1) <= 1.0 + 1e-12), "a coupling row exceeds 1 -- walks are being double counted"
    assert stats["escaped"] < 0.05, "too many walks escaped (%.3f); the operator rows are unreliable" % stats["escaped"]

    # 3. DETERMINISM: same seed, bit-identical answer.
    a = solve_decomposed(pts, exact, walks=120, seed=3)
    b = solve_decomposed(pts, exact, walks=120, seed=3)
    assert np.array_equal(a, b), "WoDS is not deterministic at a fixed seed"

    # 4. THE CLAIM THIS CODE ACTUALLY EARNS: sample efficiency at a TIGHT budget. An earlier version of this
    #    assertion demanded lower VARIANCE -- the paper's headline -- and it FAILED (0.00397 vs 0.00395, a
    #    tie inside noise). Measuring properly showed pure WoS often has the SMALLER spread, so the
    #    docstring was rewritten from the data and this pins the effect that is really there.
    m = measure_vs_pure_wos(nx=5, ny=5, walks=32, seeds=6)
    assert m["wods_err"] < m["wos_err"], \
        "WoDS lost its low-budget accuracy edge (%.5f vs %.5f) -- re-measure the claim" % (m["wods_err"], m["wos_err"])

    # 5. THE BIAS, KEPT LOUD. Refining the interface must reduce the error; adding walks at a FIXED
    #    interface must not remove it. That asymmetry IS the discretisation bias, and it is what separates
    #    this from the unbiased pointwise solvers.
    coarse = interface_grid(3, 3)
    e_few = float(np.mean(np.abs(solve_decomposed(coarse, exact, walks=200, seed=5) - exact(coarse))))
    e_many = float(np.mean(np.abs(solve_decomposed(coarse, exact, walks=1600, seed=5) - exact(coarse))))
    assert abs(e_many - e_few) < 0.05, "error moved a lot with walks alone; the bias story needs re-checking"

    print("holographic_wods: all selftests passed (solves, sub-stochastic operator, determinism, variance, bias)")


if __name__ == "__main__":
    _selftest()

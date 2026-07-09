"""holographic_wost.py -- Walk on Spheres / Walk on *Stars*: a grid-free Laplace/Poisson solver on an SDF.

WHY THIS FITS leCore SO WELL
----------------------------
Classical PDE solvers need a mesh, a global linear system, and a basis. Walk on Spheres (Muller 1956; Sawhney &
Crane, SIGGRAPH 2020) needs none of them. To evaluate the solution at ONE point:

    repeat: jump to a uniformly random point on the largest ball that fits inside the domain;
    stop when you are within eps of the boundary; take the boundary value there.
    Average over many walks -> the solution at that point.

**The only geometry query is distance-to-boundary -- which is exactly what a signed distance function returns.**
leCore is SDF-native, so the solver is a few lines and needs no meshing at all (which also sidesteps the mesh
kernel's documented Python-loop cost for PDE work).

The properties that matter here:
  * POINTWISE. Evaluate the solution where you want it; no global solve, no grid to refine.
  * EMBARRASSINGLY PARALLEL, with **no seed coordination**: every random number is `hash_unit(point, walk, step, ...)`
    -- a pure function of where you are and which walk you're on (A4/D1). Any node computes any walk, in any order,
    and gets the same answer. A stateful `default_rng` could not do this.
  * PROGRESSIVE. Error falls as 1/sqrt(N); stop when it's good enough.

WALK ON *STARS*, NOT VANILLA WoS
--------------------------------
Vanilla WoS handles DIRICHLET boundaries (the value is pinned on the wall). leCore's `heat` and `wave` solvers use
REFLECTING (Neumann / zero-flux) boundaries -- and vanilla WoS cannot do those. Walk on Stars (Sawhney, Miller,
Gkioulekas & Crane, TOG 2023) extends the walk to Neumann by REFLECTING the walk off the insulating part of the
boundary instead of absorbing it there. That is implemented below (`neumann=True`), and it is why this module can
serve the solvers `holographic_laplacian` describes.

HONEST SCOPE (measured; read before using)
------------------------------------------
  * Convergence is Monte-Carlo: 1/sqrt(N). Getting one more digit costs 100x the walks. This is a POINTWISE,
    progressive solver -- it is not competing with a good multigrid solve on a dense grid.
  * The reflecting (Neumann) treatment here is the simple star-shaped variant: reflect the step about the boundary
    normal. It is correct for the zero-flux case that `heat`/`wave` use; a general Robin/flux boundary is NOT done.
  * `source` (Poisson, non-zero right-hand side) is estimated with a single-sample ball integral per step, which is
    unbiased but noisier than the Dirichlet-only case. Kept honest: measured below.

numpy + the SDF + `determinism.hash_unit`. No mesh, no linear algebra, no learned anything.
"""
import numpy as np

from holographic.misc.holographic_determinism import hash_direction, hash_unit


def _ball_radius(dist, r_min):
    """The radius of the largest empty ball at a point: its distance to the boundary (never below r_min, so a walk
    that grazes the wall still makes progress)."""
    return np.maximum(np.abs(dist), r_min)


def solve_laplace(sdf_eval, points, boundary_value, walks=256, max_steps=64, eps=1e-3,
                  seed=0, source=None, dirichlet_sdf=None, dim=3):
    """Solve the Laplace (or Poisson) equation at `points`, grid-free, by Walk on Spheres / Stars.

        laplacian(u) = -source   inside the domain (source=None -> Laplace)
        u = boundary_value       on the DIRICHLET (absorbing) boundary
        du/dn = 0                on the NEUMANN (reflecting) boundary

    Arguments
      sdf_eval(P)        -> signed distance to the WHOLE domain boundary; NEGATIVE inside (leCore's convention).
      points             -> (N, dim) evaluation points, inside the domain.
      boundary_value(P)  -> the Dirichlet value at boundary points P:(M, dim).
      dirichlet_sdf(P)   -> OPTIONAL signed distance to the ABSORBING part of the boundary only. Give this and the
                            rest of the boundary becomes reflecting (zero-flux) -- that is Walk on *Stars*. Omit it
                            and every wall absorbs -- vanilla Walk on Spheres.
      walks / max_steps / eps / source / seed -- see the module docstring.

    WHY `dirichlet_sdf` RATHER THAN a "which walls are Neumann" predicate (this is the crux of WoSt, and I got it
    wrong first): if the ball radius is the distance to the NEAREST boundary, then a walk beside a reflecting wall
    can only take vanishing steps -- it crawls along the wall, exhausts its step budget, and never reaches the
    absorbing boundary at all. **That is exactly why vanilla WoS cannot do Neumann.** Walk on Stars fixes it by
    sizing the ball by the distance to the ABSORBING boundary only. The ball may then stick out through a reflecting
    wall; when a jump lands outside, it is MIRRORED back in across that wall (which is the zero-flux condition:
    the reflected domain is the image problem). Measured: with a poisoned reflecting wall, the correct scheme gives
    RMSE 0.0091 (clean 1/sqrt(N)) while a merely-absorbing one gives 55.1 -- it reads the wall it must ignore.

    Returns u:(N,) the solution estimate. Deterministic: same inputs -> same output, on any node, in any order.
    """
    P0 = np.atleast_2d(np.asarray(points, float))
    n = P0.shape[0]
    total = np.zeros(n)
    absorb_sdf = dirichlet_sdf if dirichlet_sdf is not None else sdf_eval

    for w in range(int(walks)):
        x = P0.copy()
        alive = np.ones(n, bool)
        contrib = np.zeros(n)                         # the Poisson source term accumulated along each walk

        for step in range(int(max_steps)):
            live = np.where(alive)[0]
            if live.size == 0:
                break

            da = np.asarray(absorb_sdf(x[live])).ravel()      # distance to the ABSORBING boundary
            done = np.abs(da) <= eps
            if done.any():
                hit = live[done]
                total[hit] += np.asarray(boundary_value(x[hit])).ravel() + contrib[hit]
                alive[hit] = False
                live = live[~done]
                da = da[~done]
                if live.size == 0:
                    break

            r = _ball_radius(da, eps)                          # the WoSt ball: sized by the absorbing boundary

            # Poisson source: one unbiased sample of the ball integral, r^2/(2*dim) * f(y), y uniform in the ball.
            if source is not None:
                y = x[live] + r[:, None] * _uniform_in_ball(x[live], w, step, seed, dim)
                contrib[live] += (r ** 2) / (2.0 * dim) * np.asarray(source(y)).ravel()

            # jump to a uniform point on the sphere of radius r (the mean-value property)
            y = x[live] + r[:, None] * _walk_direction(x[live], w, step, seed, dim)

            # ...the ball may poke through a REFLECTING wall. Mirror any escapee back inside (zero flux).
            if dirichlet_sdf is not None:
                y = _reflect_inside(sdf_eval, y, dim)
            x[live] = y

        # walks that never reached the absorbing boundary: use its nearest value (a bounded, documented bias)
        if alive.any():
            stuck = np.where(alive)[0]
            total[stuck] += np.asarray(boundary_value(_project_to_boundary(absorb_sdf, x[stuck], dim))).ravel() \
                + contrib[stuck]

    return total / float(walks)


def _reflect_inside(sdf_eval, pts, dim, tries=3):
    """Mirror any point that has left the domain back across the boundary it crossed: x <- x - 2*sdf(x)*n(x).
    For a true SDF this is the exact reflection (the image point), which IS the zero-flux (Neumann) condition.
    Repeated a few times because a corner can need more than one bounce."""
    out = pts.copy()
    for _ in range(tries):
        d = np.asarray(sdf_eval(out)).ravel()
        outside = d > 0.0
        if not outside.any():
            break
        nrm = _sdf_normal(sdf_eval, out[outside], dim)
        out[outside] = out[outside] - 2.0 * d[outside][:, None] * nrm
    return out


def _walk_direction(x, w, step, seed, dim):
    """A uniform direction, keyed by WHERE we are and WHICH walk/step -- stateless, so any node reproduces it."""
    keys = ("wost_dir", seed, w, step) + tuple(x[:, k] for k in range(dim))
    return hash_direction(*keys, dim=dim)


def _uniform_in_ball(x, w, step, seed, dim):
    """A uniform point in the unit ball (direction * radius^(1/dim)), keyed statelessly."""
    u = hash_unit("wost_ball", seed, w, step, *[x[:, k] for k in range(dim)])
    d = hash_direction("wost_balldir", seed, w, step, *[x[:, k] for k in range(dim)], dim=dim)
    return d * (u ** (1.0 / dim))[:, None]


def _sdf_normal(sdf_eval, pts, dim, h=1e-4):
    """Outward normal by central differences on the SDF (its gradient is the unit normal)."""
    g = np.zeros_like(pts)
    for k in range(dim):
        off = np.zeros(dim); off[k] = h
        g[:, k] = (np.asarray(sdf_eval(pts + off)).ravel() - np.asarray(sdf_eval(pts - off)).ravel()) / (2 * h)
    nrm = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(nrm, 1e-12)


def _project_to_boundary(sdf_eval, pts, dim):
    """Closest boundary point: walk down the SDF gradient by its own value (one Newton step -- exact for a true SDF)."""
    d = np.asarray(sdf_eval(pts)).ravel()[:, None]
    return pts - d * _sdf_normal(sdf_eval, pts, dim)


def _selftest():
    # ------------------------------------------------------------------------------------------------------
    # (1) DIRICHLET: u(x,y) = x is harmonic on the unit disk, so WoS at any interior point must converge to that
    #     point's own x. Measure RMSE over many points (mean|err| on a handful of points is itself too noisy to
    #     show a rate) and check the Monte-Carlo 1/sqrt(N) law.
    # ------------------------------------------------------------------------------------------------------
    def sdf(P):
        return np.linalg.norm(np.atleast_2d(P), axis=1) - 1.0

    def bval(P):
        return np.atleast_2d(P)[:, 0]

    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, 40)
    rad = 0.8 * np.sqrt(rng.uniform(0, 1, 40))
    pts = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    truth = pts[:, 0]

    errs = {}
    for w in (64, 1024):
        u = solve_laplace(sdf, pts, bval, walks=w, seed=0, dim=2)
        errs[w] = float(np.sqrt(np.mean((u - truth) ** 2)))
    assert errs[1024] < 0.03, errs
    assert errs[1024] < 0.55 * errs[64], errs          # 16x the walks -> ~4x less error

    # (2) DETERMINISM: stateless coordinate-keyed hashing means the answer cannot depend on run order, or on which
    #     OTHER points are being solved alongside it. A stateful rng would fail the permutation check.
    a = solve_laplace(sdf, pts, bval, walks=32, seed=0, dim=2)
    assert np.array_equal(a, solve_laplace(sdf, pts, bval, walks=32, seed=0, dim=2))
    perm = rng.permutation(len(pts))
    assert np.allclose(solve_laplace(sdf, pts[perm], bval, walks=32, seed=0, dim=2), a[perm])

    # (3) POISSON: laplacian(u) = -1 with u = -(|x|^2)/4 on the boundary gives u(0) = 0.
    up = solve_laplace(sdf, np.array([[0.0, 0.0]]), lambda P: -(np.linalg.norm(np.atleast_2d(P), axis=1) ** 2) / 4.0,
                       walks=2048, seed=1, dim=2, source=lambda P: np.ones(len(np.atleast_2d(P))))
    assert abs(float(up[0])) < 0.05, up

    # ------------------------------------------------------------------------------------------------------
    # (4) NEUMANN, tested so it CANNOT pass by accident. Upper half-disk; u(x,y)=x is harmonic and has zero flux
    #     across the flat edge (du/dy = 0), so the exact answer is u = x. We POISON the flat edge with u = 99: a
    #     correct reflecting walk never reads it, a merely-absorbing one does and is wrong by ~55.
    # ------------------------------------------------------------------------------------------------------
    def sdf_half(P):
        P = np.atleast_2d(P)
        return np.maximum(np.linalg.norm(P, axis=1) - 1.0, -P[:, 1])

    def arc_only(P):                                   # distance to the ABSORBING part (the curved arc)
        return np.linalg.norm(np.atleast_2d(P), axis=1) - 1.0

    def poisoned(P):
        P = np.atleast_2d(P)
        v = P[:, 0].copy()
        flat = (np.abs(P[:, 1]) < 5e-3) & (np.linalg.norm(P, axis=1) < 0.97)
        v[flat] = 99.0
        return v

    ang = rng.uniform(0.2, np.pi - 0.2, 30)
    rad = 0.75 * np.sqrt(rng.uniform(0.05, 1, 30))
    hp = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    htruth = hp[:, 0]

    wost = solve_laplace(sdf_half, hp, poisoned, walks=1024, seed=0, dim=2,
                         dirichlet_sdf=arc_only, max_steps=128)
    vanilla = solve_laplace(sdf_half, hp, poisoned, walks=1024, seed=0, dim=2)
    e_wost = float(np.sqrt(np.mean((wost - htruth) ** 2)))
    e_van = float(np.sqrt(np.mean((vanilla - htruth) ** 2)))
    assert e_wost < 0.05, e_wost                       # reflects: never reads the poisoned wall
    assert e_van > 10.0, e_van                         # absorbs: reads it, and is catastrophically wrong

    print("OK: holographic_wost self-test passed (Dirichlet: harmonic u=x on the disk, RMSE %.4f at 1024 walks vs "
          "%.4f at 64 -- the 1/sqrt(N) law; bit-identical run-to-run and invariant to point ORDER, so the walks are "
          "farm-parallel with no seed coordination; Poisson source recovered; NEUMANN: with the reflecting wall "
          "poisoned, WoSt scores %.4f while vanilla WoS reads the wall and scores %.1f)"
          % (errs[1024], errs[64], e_wost, e_van))


if __name__ == "__main__":
    _selftest()

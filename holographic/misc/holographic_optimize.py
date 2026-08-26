"""GRAD-2 -- a general gradient-descent optimizer (holographic_optimize).

WHY THIS EXISTS
---------------
The 3D-Gaussian-splatting work brought a real optimizer into the engine: the anisotropic splat fit
(`holographic_splat._aniso_optimize`) runs Adam with HAND-DERIVED analytic gradients -- gradient descent with no
autodiff framework, inside the NumPy-only rule. And the cache module already carries finite-difference gradients
(`gradient_cache_fd`). So both halves of a general gradient-descent capability -- a gradient source and an optimizer --
were in the box, but SILOED: the optimizer woven into the splat-specific gradients, the FD gradient specialized to
field-maps-at-anchors. This module promotes them to ONE reusable faculty, so "gradients on the fly" is first-class for
the whole engine: minimize any scalar loss from any start, with an analytic gradient where you have one and finite
differences where you do not. (The splat module's embedded Adam is left UNCHANGED -- it stays specialized for speed;
this is the general twin, the same update rule extracted.)

It is also the prerequisite the occlusion-speed panel flagged: Iterative Hard Thresholding -- the gradient-native
member of the sparse-recovery family (the M-factor fix, GRAD-1) -- is a gradient step plus a threshold, so it wants
exactly this optimizer underneath.

WHAT IT PROVIDES
  * fd_gradient(f, x, eps) -- the central finite-difference gradient of a scalar function f: R^n -> R at x (2*n
    evaluations). The general scalar-loss version of the engine's field-map FD helper.
  * optimize(loss, x0, grad, steps, lr, ...) -- minimize loss(x) from x0 by Adam (with bias correction, the exact
    update the splat fit uses). `grad(x)` supplies the analytic gradient (fast); omit it for the finite-difference
    fallback. Optional convergence-gated early stop. Returns the optimized x; pass stats={} for the loss trajectory.

THE MEASUREMENT BAR (checked exactly in the self-test)
  * CONVEX: minimizes a quadratic to its known minimum, and a least-squares loss to the lstsq solution.
  * NON-CONVEX: drives Rosenbrock close to its (1, 1) optimum.
  * FD == ANALYTIC: the finite-difference fallback reaches the same optimum as the analytic-gradient run (the
    "gradients on the fly" guarantee), and fd_gradient matches a known analytic gradient to FD tolerance.

DETERMINISM (per ISA.md)
  No RNG: Adam and central finite differences are deterministic given x0. Same loss and x0 give the same result
  (asserted). The Adam update is identical in form to the splat fit's (`_aniso_optimize`), just generalized.

KEPT NEGATIVES (loud)
  * NO AUTODIFF (the constraint): an analytic `grad` must be supplied for speed; the FD fallback costs 2*n loss
    evaluations PER STEP, which is fine for small n and expensive for large n -- supply the gradient where it matters.
  * GENERAL gradient descent inherits gradient descent's limits: a poor `lr` diverges or crawls, and on a non-convex
    loss it finds a LOCAL minimum from the given start (Rosenbrock gets CLOSE, not exact, in a fixed budget). It is a
    workhorse, not a global solver.
  * the splat module keeps its OWN embedded Adam (specialized, with the splat gradients inlined for speed) -- this is
    the general extraction beside it, not a replacement; refactoring the splat fit to call through here is a separate,
    optional step.
"""

import numpy as np


def fd_gradient(f, x, eps=1e-5):
    """Central finite-difference gradient of a scalar function f: R^n -> R at x. Perturbs a COPY of x one coordinate
    at a time (so f sees x with a single entry nudged, then restored), costing 2*n evaluations of f. Returns an array
    shaped like x. For when you have the loss but not its analytic gradient."""
    base = np.array(x, float)
    flat = base.ravel()
    g = np.zeros_like(base)
    gflat = g.ravel()
    for i in range(flat.size):
        o = flat[i]
        flat[i] = o + eps
        fp = float(f(base))
        flat[i] = o - eps
        fm = float(f(base))
        flat[i] = o
        gflat[i] = (fp - fm) / (2.0 * eps)
    return g


def optimize(loss, x0, grad=None, steps=200, lr=0.05, b1=0.9, b2=0.999, eps=1e-8,
             tol=0.0, patience=10, min_steps=20, fd_eps=1e-5, stats=None):
    """Minimize `loss(x)` (a scalar) from `x0` by Adam -- the exact bias-corrected update the splat fit uses, now
    general. Supply `grad(x)` for the analytic gradient (fast); omit it to use central finite differences (2*n loss
    evaluations per step). With `tol > 0`, stop early once the loss improvement over the last `patience` steps falls
    below `tol` * the initial loss (past a `min_steps` warm-up). Returns the optimized x; pass stats={} to read
    stats['steps'] and stats['loss'] (the trajectory)."""
    x = np.array(x0, float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    hist = []
    step = 0
    for step in range(1, steps + 1):
        g = grad(x) if grad is not None else fd_gradient(loss, x, eps=fd_eps)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
        hist.append(float(loss(x)))
        if tol > 0.0 and step >= min_steps and len(hist) > patience:
            improvement = hist[-patience - 1] - hist[-1]   # gain over the last `patience` steps
            if improvement <= tol * abs(hist[0]) + 1e-15:  # below tol of the initial loss -> converged
                break
    if stats is not None:
        stats["steps"] = int(step)
        stats["loss"] = hist
    return x


# =====================================================================================================
# Self-test -- convex to the known min, least-squares to lstsq, non-convex close, FD == analytic.
# =====================================================================================================
def _selftest():
    rng = np.random.default_rng(0)

    # --- CONVEX quadratic: minimize ||x - target||^2 -> target (analytic gradient) ---
    target = rng.standard_normal(8)
    x = optimize(lambda z: float(((z - target) ** 2).sum()), np.zeros(8),
                 grad=lambda z: 2 * (z - target), steps=400, lr=0.1)
    assert np.linalg.norm(x - target) < 1e-3, f"quadratic must reach its minimum, off by {np.linalg.norm(x - target):.3e}"

    # --- LEAST SQUARES: minimize ||A x - b||^2 -> the lstsq solution ---
    A = rng.standard_normal((20, 6))
    b = rng.standard_normal(20)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    xls = optimize(lambda z: float(((A @ z - b) ** 2).sum()), np.zeros(6),
                   grad=lambda z: 2 * A.T @ (A @ z - b), steps=2000, lr=0.02)
    assert np.linalg.norm(xls - sol) < 1e-2, f"least-squares must reach the lstsq solution, off by {np.linalg.norm(xls - sol):.3e}"

    # --- FD == ANALYTIC: the finite-difference fallback reaches the same optimum (gradients on the fly) ---
    xfd = optimize(lambda z: float(((z - target) ** 2).sum()), np.zeros(8), steps=400, lr=0.1)  # no grad -> FD
    assert np.linalg.norm(xfd - target) < 1e-3, "FD fallback must also reach the minimum"
    # and fd_gradient matches the analytic gradient on a test point
    z0 = rng.standard_normal(8)
    g_fd = fd_gradient(lambda z: float(((z - target) ** 2).sum()), z0)
    g_an = 2 * (z0 - target)
    assert np.max(np.abs(g_fd - g_an)) < 1e-5, "fd_gradient must match the analytic gradient"

    # --- NON-CONVEX: Rosenbrock driven close to (1, 1) ---
    def rosen(z):
        return float((1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2)

    def rosen_grad(z):
        return np.array([-2 * (1 - z[0]) - 400 * z[0] * (z[1] - z[0] ** 2), 200 * (z[1] - z[0] ** 2)])

    st = {}
    xr = optimize(rosen, np.array([-1.2, 1.0]), grad=rosen_grad, steps=8000, lr=0.002, tol=1e-9, stats=st)
    assert np.linalg.norm(xr - np.array([1.0, 1.0])) < 0.1, f"Rosenbrock must get close to (1,1), got {xr}"

    # --- determinism ---
    a = optimize(lambda z: float(((z - target) ** 2).sum()), np.zeros(8), grad=lambda z: 2 * (z - target), steps=50)
    b2 = optimize(lambda z: float(((z - target) ** 2).sum()), np.zeros(8), grad=lambda z: 2 * (z - target), steps=50)
    assert np.array_equal(a, b2)

    print(f"holographic_optimize selftest: ok (CONVEX quadratic -> min (off {np.linalg.norm(x - target):.1e}); "
          f"LEAST-SQUARES -> lstsq (off {np.linalg.norm(xls - sol):.1e}); FD fallback reaches the same min and "
          f"fd_gradient matches analytic to {np.max(np.abs(g_fd - g_an)):.0e}; NON-CONVEX Rosenbrock -> {xr.round(3)} "
          f"near (1,1) in {st['steps']} steps; deterministic. The splat Adam, generalized -- gradients on the fly)")


if __name__ == "__main__":
    _selftest()

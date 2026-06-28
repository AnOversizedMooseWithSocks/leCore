"""Tests for GRAD-2 the general gradient-descent optimizer (holographic_optimize): the 3DGS splat-fit Adam machinery
promoted to a reusable faculty -- minimize any scalar loss, analytic gradient where supplied, finite differences
where not ("gradients on the fly")."""

import numpy as np

from holographic_optimize import optimize, fd_gradient


def test_convex_quadratic_reaches_minimum():
    rng = np.random.default_rng(0)
    target = rng.standard_normal(8)
    x = optimize(lambda z: float(((z - target) ** 2).sum()), np.zeros(8),
                 grad=lambda z: 2 * (z - target), steps=400, lr=0.1)
    assert np.linalg.norm(x - target) < 1e-3


def test_least_squares_reaches_lstsq_solution():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((20, 6))
    b = rng.standard_normal(20)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    x = optimize(lambda z: float(((A @ z - b) ** 2).sum()), np.zeros(6),
                 grad=lambda z: 2 * A.T @ (A @ z - b), steps=2000, lr=0.02)
    assert np.linalg.norm(x - sol) < 1e-2


def test_fd_fallback_matches_analytic_run():
    rng = np.random.default_rng(2)
    target = rng.standard_normal(8)
    loss = lambda z: float(((z - target) ** 2).sum())
    x_an = optimize(loss, np.zeros(8), grad=lambda z: 2 * (z - target), steps=400, lr=0.1)
    x_fd = optimize(loss, np.zeros(8), steps=400, lr=0.1)  # no grad -> finite differences
    assert np.linalg.norm(x_fd - x_an) < 1e-4


def test_fd_gradient_matches_analytic():
    rng = np.random.default_rng(3)
    target = rng.standard_normal(8)
    z0 = rng.standard_normal(8)
    g_fd = fd_gradient(lambda z: float(((z - target) ** 2).sum()), z0)
    g_an = 2 * (z0 - target)
    assert np.max(np.abs(g_fd - g_an)) < 1e-5


def test_fd_gradient_preserves_input():
    # fd_gradient must not mutate the array it is handed
    x = np.array([1.0, 2.0, 3.0])
    x_before = x.copy()
    fd_gradient(lambda z: float((z ** 2).sum()), x)
    assert np.array_equal(x, x_before)


def test_rosenbrock_gets_close():
    def rosen(z):
        return float((1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2)

    def rosen_grad(z):
        return np.array([-2 * (1 - z[0]) - 400 * z[0] * (z[1] - z[0] ** 2), 200 * (z[1] - z[0] ** 2)])

    x = optimize(rosen, np.array([-1.2, 1.0]), grad=rosen_grad, steps=8000, lr=0.002, tol=1e-9)
    assert np.linalg.norm(x - np.array([1.0, 1.0])) < 0.1


def test_early_stop_reports_fewer_steps():
    rng = np.random.default_rng(4)
    target = rng.standard_normal(6)
    st = {}
    optimize(lambda z: float(((z - target) ** 2).sum()), np.zeros(6),
             grad=lambda z: 2 * (z - target), steps=5000, lr=0.1, tol=1e-9, stats=st)
    assert st["steps"] < 5000  # converged before the budget
    assert len(st["loss"]) == st["steps"]


def test_deterministic():
    rng = np.random.default_rng(5)
    target = rng.standard_normal(8)
    loss = lambda z: float(((z - target) ** 2).sum())
    g = lambda z: 2 * (z - target)
    a = optimize(loss, np.zeros(8), grad=g, steps=50)
    b = optimize(loss, np.zeros(8), grad=g, steps=50)
    assert np.array_equal(a, b)

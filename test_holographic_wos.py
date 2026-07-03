"""Walk on Spheres (#7): mesh-free Monte Carlo Laplace/Poisson on any SDF; recovers known solutions."""
import numpy as np
from holographic_wos import walk_on_spheres, solve_on_sdf


DISK_R = 2.0
disk_dist = lambda P: DISK_R - np.linalg.norm(P, axis=1)


def test_constant_boundary_gives_constant():
    mean, se = walk_on_spheres(np.array([[0.0, 0.0], [0.5, 0.3]]), disk_dist,
                               lambda P: np.full(len(P), 5.0), n_walks=400, seed=0)
    assert np.all(np.abs(mean - 5.0) < 1e-6)


def test_harmonic_boundary_recovered():
    pts = np.array([[0.0, 0.0], [0.5, 0.3], [1.0, -0.6]])
    mean, se = walk_on_spheres(pts, disk_dist, lambda P: P[:, 0], n_walks=4000, seed=1)
    assert np.all(np.abs(mean - pts[:, 0]) < 4 * se + 0.03)


def test_annulus_log_profile():
    r_in, r_out = 1.0, 2.0
    def ann_dist(P):
        rho = np.linalg.norm(P, axis=1); return np.minimum(rho - r_in, r_out - rho)
    def ann_bval(P):
        rho = np.linalg.norm(P, axis=1); return (np.abs(rho - r_out) < np.abs(rho - r_in)).astype(float)
    probe = np.array([[1.5, 0.0], [0.0, 1.5]])
    mean, se = walk_on_spheres(probe, ann_dist, ann_bval, n_walks=8000, eps=1e-3, seed=2)
    exact = np.log(1.5 / r_in) / np.log(r_out / r_in)
    assert np.all(np.abs(mean - exact) < 4 * se + 0.02)


def test_poisson_source():
    mean, se = walk_on_spheres(np.array([[0.0, 0.0]]), disk_dist, lambda P: np.zeros(len(P)),
                               source=lambda P: np.ones(len(P)), n_walks=6000, eps=1e-3, seed=3)
    assert abs(mean[0] - DISK_R ** 2 / 4.0) < 5 * se[0] + 0.05


def test_monte_carlo_convergence():
    pts = np.array([[0.0, 0.0], [0.5, 0.3]])
    _, se_lo = walk_on_spheres(pts, disk_dist, lambda P: P[:, 0], n_walks=500, seed=4)
    _, se_hi = walk_on_spheres(pts, disk_dist, lambda P: P[:, 0], n_walks=8000, seed=4)
    assert se_hi.mean() < se_lo.mean() * 0.5


def test_on_sdf_and_deterministic():
    from holographic_sdf import sphere
    s = sphere(2.0)
    m1, _ = solve_on_sdf(s, lambda P: P[:, 0], np.array([[0.4, 0.2, -0.3]]), n_walks=2000, seed=5)
    m2, _ = solve_on_sdf(s, lambda P: P[:, 0], np.array([[0.4, 0.2, -0.3]]), n_walks=2000, seed=5)
    assert m1[0] == m2[0] and abs(m1[0] - 0.4) < 0.1

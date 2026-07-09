import numpy as np
from holographic.sampling_and_signal.holographic_sampling import poisson_disk_sample, radial_power_spectrum


def test_poisson_disk_respects_min_distance():
    """Every pair of sampled points is at least `radius` apart (the hard Poisson-disk guarantee)."""
    b = (np.array([0., 0]), np.array([1., 1.]))
    pts = poisson_disk_sample(0.05, b, seed=0)
    d = pts[:, None, :] - pts[None, :, :]
    dd = np.sqrt((d ** 2).sum(-1)); np.fill_diagonal(dd, np.inf)
    assert dd.min() >= 0.05 - 1e-9 and len(pts) > 50           # maximal-ish fill


def test_poisson_disk_has_blue_noise_spectrum():
    """Low-frequency power is suppressed vs white noise -- the blue-noise signature."""
    b = (np.array([0., 0]), np.array([1., 1.]))
    pts = poisson_disk_sample(0.03, b, seed=1)
    white = np.random.default_rng(1).uniform(0, 1, (len(pts), 2))
    sb = radial_power_spectrum(pts, b); sw = radial_power_spectrum(white, b)
    assert np.mean(sb[1:4]) < 0.6 * np.mean(sw[1:4])           # clear low-freq dip


def test_poisson_disk_deterministic_and_3d():
    """Deterministic in the seed, and works in 3-D."""
    b2 = (np.array([0., 0]), np.array([1., 1.]))
    assert np.array_equal(poisson_disk_sample(0.06, b2, seed=3), poisson_disk_sample(0.06, b2, seed=3))
    b3 = (np.array([0., 0, 0]), np.array([1., 1, 1.]))
    pts = poisson_disk_sample(0.12, b3, seed=0)
    assert pts.shape[1] == 3 and len(pts) > 20

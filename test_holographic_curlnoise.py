"""Curl noise (#1): divergence-free by construction, turbulent, flows around obstacles."""
import numpy as np
from holographic_curlnoise import (curl_noise, curl_noise_3d, divergence, streamfunction, curl_of_streamfunction)


def test_divergence_free_by_construction():
    u, v = curl_noise(64, octaves=4, seed=0)
    speed = np.sqrt(u ** 2 + v ** 2)
    assert np.abs(divergence(u, v)).max() < 1e-9 * max(speed.max(), 1e-9) + 1e-9   # machine-zero divergence
    assert speed.mean() > 0.0 and speed.std() > 0.1 * speed.mean()                 # real turbulent structure


def test_flows_around_obstacle():
    cx = cy = 4.0; R = 1.5
    disk = lambda p: np.sqrt((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2) - R
    u, v = curl_noise(96, octaves=4, seed=0, obstacle_sdf=disk, ramp=1.0)
    xs = np.linspace(0, 8, 96); ys = np.linspace(0, 8, 96); gx, gy = np.meshgrid(xs, ys)
    dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    sp = np.sqrt(u ** 2 + v ** 2)
    assert sp[dist < R * 0.8].mean() < 0.2 * sp[dist > R * 1.5].mean()             # no penetration
    assert np.abs(divergence(u, v)).max() < 1e-6                                    # still divergence-free


def test_3d_divergence_free():
    u, v, w = curl_noise_3d(24, octaves=2, seed=1)
    div3 = np.gradient(u)[0] + np.gradient(v)[1] + np.gradient(w)[2]
    assert np.abs(div3).max() < 1e-9 * np.sqrt(u ** 2 + v ** 2 + w ** 2).max() + 1e-8


def test_streamfunction_curl_roundtrip_and_determinism():
    psi = streamfunction(40, seed=3)
    u, v = curl_of_streamfunction(psi)
    assert u.shape == psi.shape and v.shape == psi.shape
    a, b = curl_noise(32, seed=2); c, d = curl_noise(32, seed=2)
    assert np.array_equal(a, c) and np.array_equal(b, d)

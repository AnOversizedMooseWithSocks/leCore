"""Tests for holographic_denoisehome -- the Denoise home (R5: image/sharpen/signal denoise, one home)."""
import numpy as np
from holographic_denoisehome import Denoise, denoise_backends


def _noisy_image(H=24, W=24, seed=0):
    rng = np.random.default_rng(seed)
    clean = np.ones((H, W, 3)) * 0.5
    noisy = clean + 0.1 * rng.standard_normal((H, W, 3))
    normal = np.tile([0., 0., 1.], (H, W, 1))
    albedo = np.ones((H, W, 3)) * 0.5
    depth = np.ones((H, W))
    return clean, noisy, normal, albedo, depth


def test_image_svgf_cleans_and_routes_bit_identical():
    from holographic_svgf import atrous_bilateral
    clean, noisy, N, A, D = _noisy_image()
    den = Denoise.image(noisy, N, A, D, method="svgf", levels=4)
    assert np.abs(den - clean).mean() < np.abs(noisy - clean).mean()
    assert np.array_equal(den, atrous_bilateral(noisy, N, A, D, levels=4, variance=None))


def test_image_demodulated_routes():
    from holographic_modulate import denoise_demodulated
    _, noisy, N, A, D = _noisy_image()
    a = Denoise.image(noisy, N, A, D, method="demodulated", levels=4)
    b = denoise_demodulated(noisy, N, A, D, variance=None, levels=4)
    assert np.array_equal(a, b)


def test_image_unknown_method_raises():
    _, noisy, N, A, D = _noisy_image()
    try:
        Denoise.image(noisy, N, A, D, method="nope")
        assert False
    except ValueError as e:
        assert "svgf" in str(e)


def test_signal_trajectory_prior_free():
    t = np.linspace(0, 4 * np.pi, 256)
    sig = np.sin(t)
    noisy = sig + 0.3 * np.random.default_rng(1).standard_normal(256)
    clean = Denoise.signal(noisy, method="trajectory", rank=4)
    assert np.abs(clean - sig).mean() < np.abs(noisy - sig).mean()


def test_signal_manifold_routes_bit_identical():
    from holographic_denoise import fit_manifold, manifold_denoise
    rng = np.random.default_rng(2)
    samples = rng.standard_normal((40, 8)) @ rng.standard_normal((8, 32))
    x = samples[0] + 0.05 * rng.standard_normal(32)
    basis, mean = fit_manifold(samples, rank=8)
    assert np.array_equal(Denoise.signal(x, samples=samples, method="manifold", rank=8),
                          manifold_denoise(x, basis, mean))


def test_signal_auto_and_missing_prior():
    # auto with no prior -> trajectory (works on a raw signal)
    assert Denoise.signal(np.sin(np.linspace(0, 6, 128)), method="auto") is not None
    try:
        Denoise.signal(np.zeros(16), method="manifold")             # needs samples
        assert False
    except ValueError:
        pass


def test_sharpen_routes():
    from holographic_sharpen import sharpen_loop
    x = np.sin(np.linspace(0, 6, 96))
    assert np.array_equal(Denoise.sharpen(x, sigma=2.0, iters=15),
                          sharpen_loop(np.asarray(x, float), blur=None, sigma=2.0, lam=1.0, iters=15, noise_level=0.0))


def test_backends_listed():
    assert "image:svgf" in denoise_backends() and "signal:trajectory" in denoise_backends()

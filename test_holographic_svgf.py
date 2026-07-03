"""Render-speed E: holographic bilateral SVGF denoise (feature-cosine edge-stopping)."""
import numpy as np
from holographic_svgf import atrous_bilateral, plain_blur, _psnr


def _scene(seed=0):
    rng = np.random.default_rng(seed); H = W = 64
    clean = np.zeros((H, W, 3)); normal = np.zeros((H, W, 3)); albedo = np.zeros((H, W, 3)); depth = np.zeros((H, W))
    clean[:, :W // 2] = [0.8, 0.2, 0.2]; clean[:, W // 2:] = [0.2, 0.3, 0.8]
    normal[:, :W // 2] = [0, 0, 1]; normal[:, W // 2:] = [1, 0, 0]
    albedo[:, :W // 2] = [0.8, 0.2, 0.2]; albedo[:, W // 2:] = [0.2, 0.3, 0.8]
    depth[:, :W // 2] = 1.0; depth[:, W // 2:] = 3.0
    noisy = np.clip(clean + 0.15 * rng.standard_normal((H, W, 3)), 0, 1)
    return clean, noisy, normal, albedo, depth


def test_beats_plain_blur_on_psnr():
    clean, noisy, n, a, z = _scene()
    den = atrous_bilateral(noisy, n, a, z, levels=5)
    blur = plain_blur(noisy, levels=5)
    assert _psnr(den, clean) > _psnr(noisy, clean)      # denoises
    assert _psnr(den, clean) > _psnr(blur, clean)       # beats the edge-blind blur


def test_edge_stays_sharp():
    clean, noisy, n, a, z = _scene(); W = 64; e = W // 2
    den = atrous_bilateral(noisy, n, a, z, levels=5)
    blur = plain_blur(noisy, levels=5)
    de = float(np.mean((den[:, e - 1:e + 1] - clean[:, e - 1:e + 1]) ** 2))
    be = float(np.mean((blur[:, e - 1:e + 1] - clean[:, e - 1:e + 1]) ** 2))
    assert de < be


def test_deterministic():
    _, noisy, n, a, z = _scene()
    assert np.array_equal(atrous_bilateral(noisy, n, a, z, levels=3),
                          atrous_bilateral(noisy, n, a, z, levels=3))

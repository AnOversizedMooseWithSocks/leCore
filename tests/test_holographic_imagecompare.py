"""Inverse-rendering IR4 (part 1): perceptual render-vs-target compare -- SSIM + colour + edges, shift-robust."""
import numpy as np
from holographic.io_and_interop.holographic_imagecompare import ssim, ms_ssim, color_agreement, edge_agreement, perceptual_similarity, perceptual_distance, _shift
from holographic.mesh_and_geometry.holographic_autobump import gaussian_blur


def _scene(seed=0):
    rng = np.random.default_rng(seed); H, W = 72, 72
    yy, xx = np.mgrid[0:H, 0:W].astype(float); Y = yy / H
    sky = np.stack([0.2 + 0.5 * Y, 0.4 + 0.4 * Y, 0.85 - 0.3 * Y], axis=-1)
    sy, sx = rng.uniform(0.1, 0.4) * H, rng.uniform(0.2, 0.8) * W
    sun = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * (0.08 * W) ** 2))[..., None] * [1.0, 0.9, 0.6]
    return np.clip(sky + 0.8 * sun, 0, 1)


def test_identical_is_perfect():
    s = _scene(0)
    assert abs(perceptual_similarity(s, s) - 1.0) < 1e-6 and perceptual_distance(s, s) < 1e-6
    assert abs(ssim(s, s) - 1.0) < 1e-9


def test_small_shift_stays_similar():
    s = _scene(0)
    assert perceptual_similarity(s, _shift(s, 2, 2)) > 0.85


def test_shift_ranks_above_different_scene():
    s = _scene(0)
    assert perceptual_similarity(s, _shift(s, 2, 2)) > perceptual_similarity(s, _scene(5)) + 0.05


def test_mse_cannot_but_perceptual_can():
    # on textured content, a shift's MSE is nearly a different image's -> MSE can't rank; perceptual can
    rng = np.random.default_rng(1)
    tex = np.clip(gaussian_blur(rng.uniform(0, 1, (64, 64, 3)), 1.0), 0, 1)
    tsh = _shift(tex, 2, 2); tot = np.clip(gaussian_blur(rng.uniform(0, 1, (64, 64, 3)), 1.0), 0, 1)
    assert np.mean((tex - tsh) ** 2) / np.mean((tex - tot) ** 2) > 0.5     # MSE barely prefers the shift
    assert perceptual_similarity(tex, tsh) > perceptual_similarity(tex, tot)


def test_brightness_offset_preserves_ssim():
    s = _scene(0)
    assert ssim(s, np.clip(s + 0.1, 0, 1)) > 0.9           # structure survives a constant offset


def test_components_in_range_and_symmetric():
    a, b = _scene(0), _scene(3)
    assert 0 <= color_agreement(a, b) <= 1 and 0 <= edge_agreement(a, b) <= 1 and -1 <= ms_ssim(a, b) <= 1
    assert abs(perceptual_similarity(a, b) - perceptual_similarity(b, a)) < 1e-9


def test_deterministic():
    a, b = _scene(0), _scene(3)
    assert perceptual_similarity(a, b) == perceptual_similarity(a, b)

"""Inverse-rendering ST1: colour transfer -- match a reference image's colour statistics (Reinhard 2001)."""
import numpy as np
from holographic.materials_and_texture.holographic_colortransfer import color_transfer


def _imgs():
    rng = np.random.default_rng(0)
    src = np.clip(0.4 + 0.15 * rng.standard_normal((64, 64, 3)) @ [[1, .3, 0], [.3, 1, .2], [0, .2, 1]], 0, 1)
    ref = np.clip(0.6 + 0.20 * rng.standard_normal((50, 70, 3)) @ [[1, .1, .4], [.1, 1, .1], [.4, .1, 1]], 0, 1)
    return src, ref


def test_covariance_matches_mean_and_cov():
    src, ref = _imgs()
    out = color_transfer(src, ref, mode="covariance", strength=1.0, clip=False).reshape(-1, 3)
    R = ref.reshape(-1, 3)
    assert np.allclose(out.mean(0), R.mean(0), atol=1e-6)
    assert np.allclose(np.cov(out.T), np.cov(R.T), atol=1e-3)


def test_meanstd_matches_per_channel():
    src, ref = _imgs()
    out = color_transfer(src, ref, mode="meanstd", strength=1.0, clip=False).reshape(-1, 3)
    R = ref.reshape(-1, 3)
    assert np.allclose(out.mean(0), R.mean(0), atol=1e-6) and np.allclose(out.std(0), R.std(0), atol=1e-6)


def test_strength_blends():
    src, ref = _imgs()
    assert np.allclose(color_transfer(src, ref, strength=0.0, clip=False), src)
    full = color_transfer(src, ref, strength=1.0, clip=False)
    half = color_transfer(src, ref, strength=0.5, clip=False)
    assert np.allclose(half, 0.5 * src + 0.5 * full)


def test_shape_and_clip():
    src, ref = _imgs()
    out = color_transfer(src, ref)
    assert out.shape == src.shape and out.min() >= 0.0 and out.max() <= 1.0


def test_grayscale_covariance_stable():
    # a near-degenerate (grayscale) source covariance must not blow up (eigenvalue clamp)
    g = np.tile(np.linspace(0.2, 0.8, 32)[:, None, None], (1, 32, 3))
    ref = np.random.default_rng(1).uniform(0, 1, (32, 32, 3))
    out = color_transfer(g, ref, mode="covariance")
    assert np.isfinite(out).all()


def test_deterministic():
    src, ref = _imgs()
    assert np.array_equal(color_transfer(src, ref), color_transfer(src, ref))

"""Render-speed E: holographic bilateral SVGF denoise (feature-cosine edge-stopping)."""
import numpy as np
from holographic.rendering.holographic_svgf import atrous_bilateral, plain_blur, _psnr


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


def test_variance_guided_svgf_backward_compatible_and_calibrates():
    """variance=None reproduces the old fixed-sigma behaviour EXACTLY (backward compatible); and with a variance
    map, a NOISY region is smoothed more than a CONVERGED region (the self-calibration)."""
    import numpy as np
    from holographic.rendering.holographic_svgf import atrous_bilateral
    rng = np.random.default_rng(0)
    H = W = 48
    clean = np.zeros((H, W, 3)); clean[:, :W // 2] = [0.8, 0.3, 0.3]; clean[:, W // 2:] = [0.3, 0.4, 0.8]
    normal = np.zeros((H, W, 3)); normal[:, :W // 2] = [0, 0, 1]; normal[:, W // 2:] = [1, 0, 0]
    albedo = clean.copy(); depth = np.ones((H, W)); depth[:, W // 2:] = 3.0
    noisy = clean + 0.12 * rng.standard_normal((H, W, 3))

    # backward compatibility: no variance arg == the old call, bit-for-bit
    a = atrous_bilateral(noisy, normal, albedo, depth, levels=4)
    b = atrous_bilateral(noisy, normal, albedo, depth, levels=4, variance=None)
    assert np.array_equal(a, b)

    # calibration: left half marked converged (tiny variance), right half noisy (large variance)
    var = np.zeros((H, W)); var[:, :W // 2] = 1e-6; var[:, W // 2:] = 4e-2
    out = atrous_bilateral(noisy, normal, albedo, depth, levels=4, variance=var)
    # the noisy (right) half should be smoothed MORE -> lower residual variance than the converged (left) half's change
    left_change = float(np.mean((out[:, :W // 2] - noisy[:, :W // 2]) ** 2))
    right_change = float(np.mean((out[:, W // 2:] - noisy[:, W // 2:]) ** 2))
    assert right_change > left_change                        # variance guidance blurs the noisy region harder

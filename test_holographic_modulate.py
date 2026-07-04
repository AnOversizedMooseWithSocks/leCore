"""Tests for holographic_modulate -- the modulate/demodulate primitive (bind/unbind) and demodulated denoising (M4)."""
import numpy as np
from holographic_modulate import demodulate, remodulate, denoise_demodulated
from holographic_svgf import atrous_bilateral


def test_roundtrip_exact_where_carrier_known():
    rng = np.random.default_rng(0)
    x = rng.random((10, 10, 3)); c = 0.2 + rng.random((10, 10, 3))     # non-zero carrier
    assert np.allclose(remodulate(demodulate(x, c, eps=0.0), c), x, atol=1e-9)


def test_demodulate_is_elementwise_unbind():
    # demodulate divides, remodulate multiplies -- the FFT/HRR unbind/bind pair, elementwise here
    x = np.array([[[0.6, 0.6, 0.6]]]); c = np.array([[[0.3, 0.3, 0.3]]])
    assert np.allclose(demodulate(x, c, eps=0.0), 2.0)
    assert np.allclose(remodulate(np.array([[[2.0, 2.0, 2.0]]]), c), 0.6)


def _textured_scene(rng, H=96, W=96, noise=0.06):
    yy, xx = np.mgrid[0:H, 0:W]
    checker = (((xx // 8) + (yy // 8)) % 2).astype(float) * 0.6 + 0.3   # crisp albedo (texture)
    albedo = np.stack([checker, checker, checker], axis=2)
    irr = (0.4 + 0.5 * (xx / W))[..., None] * np.ones(3)               # smooth irradiance (lighting)
    clean = albedo * irr
    noisy = clean + rng.normal(0, noise, clean.shape)
    normal = np.zeros((H, W, 3)); normal[..., 2] = 1.0
    depth = np.ones((H, W))
    return noisy, clean, albedo, normal, depth


def test_m4_cleaner_than_guide_on_textured():
    # on a textured diffuse surface, demodulated denoise leaves LESS error than filtering colour directly
    rng = np.random.default_rng(0)
    noisy, clean, albedo, normal, depth = _textured_scene(rng)
    guide = atrous_bilateral(noisy, normal, albedo, depth, levels=5)
    demod = denoise_demodulated(noisy, normal, albedo, depth, levels=5)
    assert np.abs(demod - clean).mean() < np.abs(guide - clean).mean()


def test_m4_preserves_texture_edges():
    rng = np.random.default_rng(1)
    noisy, clean, albedo, normal, depth = _textured_scene(rng)
    guide = atrous_bilateral(noisy, normal, albedo, depth, levels=5)
    demod = denoise_demodulated(noisy, normal, albedo, depth, levels=5)
    def edge(im):
        return float(np.abs(np.diff(im[48].mean(1))).max())
    assert edge(demod) >= edge(guide) * 0.95                           # edges comparable, not smeared


def test_m4_masks_black_background():
    # near-black background (albedo ~ 0) must NOT explode under division -> the mask keeps it at carrier 1
    rng = np.random.default_rng(2)
    H = W = 48
    albedo = np.full((H, W, 3), 0.8)
    albedo[:, :W // 2] = 0.01                                          # left half = near-black "background"
    irr = np.full((H, W, 3), 0.5)
    noisy = albedo * irr + rng.normal(0, 0.05, (H, W, 3))
    normal = np.zeros((H, W, 3)); normal[..., 2] = 1.0; depth = np.ones((H, W))
    out = denoise_demodulated(noisy, normal, albedo, depth, levels=4, carrier_floor=0.05)
    assert np.isfinite(out).all() and out.max() < 5.0                 # no blow-up in the near-black region


def test_m4_neutral_on_uniform_albedo():
    # honest kept-negative: on UNIFORM albedo there's nothing to separate, so M4 is ~the same as guide-only
    rng = np.random.default_rng(3)
    H = W = 64
    albedo = np.full((H, W, 3), 0.7)
    irr = (0.4 + 0.5 * (np.mgrid[0:H, 0:W][1] / W))[..., None] * np.ones(3)
    clean = albedo * irr
    noisy = clean + rng.normal(0, 0.05, clean.shape)
    normal = np.zeros((H, W, 3)); normal[..., 2] = 1.0; depth = np.ones((H, W))
    guide = atrous_bilateral(noisy, normal, albedo, depth, levels=5)
    demod = denoise_demodulated(noisy, normal, albedo, depth, levels=5)
    eg = np.abs(guide - clean).mean(); ed = np.abs(demod - clean).mean()
    assert abs(ed - eg) < eg * 0.3                                    # within 30% -- comparable, no big loss


def test_m5_superres_beats_plain_upscale_on_texture():
    # M5: render lighting low-res, upscale the smooth irradiance, remodulate the crisp high-res albedo -> recovers
    # texture a plain colour upscale blurs away
    from holographic_modulate import superres_demodulated
    from holographic_fsr import easu_upscale
    rng = np.random.default_rng(0)
    noisy, clean, albedo, normal, depth = _textured_scene(rng, noise=0.0)   # clean textured+lit target
    low = clean[::2, ::2]                                              # a 2x-smaller "low-res" render
    m5 = superres_demodulated(low, albedo)                            # demod upscale (carrier from high albedo)
    plain = easu_upscale(low, 2.0)[:clean.shape[0], :clean.shape[1]]  # naive colour upscale
    assert np.abs(m5 - clean).mean() < np.abs(plain - clean).mean()


def test_m5_carrier_from_downsampled_high_albedo():
    # the carrier is derived from the DOWNSAMPLED high albedo (anti-aliased), not a point-sampled low albedo, so
    # high-frequency texture doesn't alias the carrier
    from holographic_modulate import _downsample_to
    a = np.arange(8 * 8 * 3, dtype=float).reshape(8, 8, 3)
    d = _downsample_to(a, 4, 4)
    assert d.shape == (4, 4, 3)
    # each output cell is the average of its 2x2 block
    assert np.allclose(d[0, 0], a[0:2, 0:2].mean(axis=(0, 1)))


def test_m5_neutral_on_uniform_albedo():
    # honest kept-negative: uniform albedo has no texture to restore, so M5 ~ a plain upscale (no worse)
    from holographic_modulate import superres_demodulated
    from holographic_fsr import easu_upscale
    H = W = 64
    albedo = np.full((H, W, 3), 0.7)
    irr = (0.3 + 0.6 * (np.mgrid[0:H, 0:W][1] / W))[..., None] * np.ones(3)
    clean = albedo * irr
    low = clean[::2, ::2]
    m5 = superres_demodulated(low, albedo)
    plain = easu_upscale(low, 2.0)[:H, :W]
    em = np.abs(m5 - clean).mean(); ep = np.abs(plain - clean).mean()
    assert abs(em - ep) < ep * 0.5 + 1e-3                            # comparable to plain (no big loss)

"""Engine defects leStudio (the image editor built on leCore) documented in its LECORE_* backlogs and worked around
app-side. Each test pins the fix so the workaround can be deleted downstream; each names its source so the next
sweep knows why it exists. Sweep 162."""
import numpy as np
import pytest

import lecore


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_pattern_image_renders_the_texture_menu_and_refuses_unknown_names(mind):
    """LECORE_UPDATE_R8: '7 of 12 pattern kinds return flat zeros at its defaults (marble/wood/brick/voronoi/
    musgrave/wave/magic)'. Those names live in holographic_proctex; pattern_image now routes to them, and a name in
    NEITHER menu raises instead of returning a flat image."""
    for k in ("marble", "wood", "brick", "voronoi", "musgrave", "wave", "magic"):
        im = mind.pattern_image(k, 24, 16)
        assert im.shape == (16, 24) and im.std() > 0.05, k
    assert mind.pattern_image("fbm", 16, 16).std() > 0.05                 # the old menu still works
    with pytest.raises(ValueError):
        mind.pattern_image("marbel", 8, 8)


def test_sky_model_sun_intensity_changes_the_sky_and_default_is_bit_stable(mind):
    """LECORE_027_BACKLOG: 'sun_intensity has NO effect on the sampled sky -- byte-identical output from 0 to 40'
    on a 200x120 grid (the dial scaled only the 1.4-degree disk). The glow scales with it now, relative to the
    default so renders at 18.0 are unchanged."""
    d = np.random.default_rng(0).normal(size=(200 * 120, 3)); d[:, 1] = np.abs(d[:, 1]); d /= np.linalg.norm(d, axis=1, keepdims=True)
    lo, hi = mind.sky_model(hour=10, sun_intensity=1.0)(d), mind.sky_model(hour=10, sun_intensity=30.0)(d)
    assert not np.allclose(lo, hi) and hi.mean() > lo.mean()
    assert np.array_equal(mind.sky_model(hour=10)(d), mind.sky_model(hour=10, sun_intensity=18.0)(d))


def test_sharpen_image_accepts_2d_and_rgb(mind):
    """APP_BACKLOG SS-D: 'sharpen_image is 1-D only (rfft with a 1-D kernel: fails on any 2-D array, not just
    RGB)'. Both axes are sharpened and channels never mix; 1-D output is bit-identical to before."""
    truth = np.zeros((24, 32)); truth[8:16, 10:22] = 1.0
    from holographic.rendering.holographic_sharpen import _gauss_blur
    blurred = _gauss_blur(truth, 2.0)
    sharp = mind.sharpen_image(blurred, sigma=2.0, lam=0.9, iters=40)
    assert sharp.shape == truth.shape and np.linalg.norm(sharp - truth) < 0.65 * np.linalg.norm(blurred - truth)
    rgb = np.stack([truth, 0.5 * truth, np.zeros_like(truth)], -1)
    s3 = mind.sharpen_image(_gauss_blur(rgb, 2.0), sigma=2.0, lam=0.9, iters=10)
    assert s3.shape == rgb.shape and np.abs(s3[..., 2]).max() < 1e-9
    x1 = np.random.default_rng(3).normal(size=50)
    ref = np.fft.irfft(np.fft.rfft(x1) * np.exp(-0.5 * (2 * np.pi * np.fft.rfftfreq(50) * 3.0) ** 2), n=50)
    assert np.array_equal(_gauss_blur(x1, 3.0), ref)


def test_morph_scene_accepts_non_square_images(mind):
    """APP_BACKLOG SS-D: 'morph_scene requires SQUARE images (its DCT matrix is one-dimensional)'. A separable DCT
    is two matrices; endpoints are exact and shapes are kept."""
    A = np.random.default_rng(0).random((16, 24)); B = np.random.default_rng(1).random((16, 24))
    fr = mind.morph_scene(A, B, steps=5)
    assert len(fr) == 5 and fr[2].shape == (16, 24)
    assert np.abs(fr[0] - A).max() < 1e-9 and np.abs(fr[-1] - B).max() < 1e-9
    with pytest.raises(ValueError):
        mind.morph_scene(A, B[:8], steps=3)


def test_make_cloud_accepts_a_dict_camera(mind):
    """APP_BACKLOG SS-D: 'make_cloud(camera=...) rejects dict cameras' while render_mesh accepted the same dict."""
    img = mind.make_cloud(camera={"eye": [0, 0, 4], "target": [0, 0, 0], "up": [0, 1, 0], "fov_deg": 40},
                          width=24, height=16, steps=6, grid=8)
    assert np.asarray(img).shape == (16, 24, 3)

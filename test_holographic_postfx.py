"""Tests for the composable post-processing pipeline (PostChain program + effects)."""
import numpy as np
from holographic_postfx import (PostChain, default_chain, cinematic_chain, EFFECTS,
                                aces, reinhard, gamma, vignette, bloom, dof, film_grain,
                                resample, supersample, chromatic_aberration, _fft_blur)


def _hdr():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (48, 48, 3))
    img[20:28, 20:28] = 4.0
    return img


def test_tonemap_compresses_highlights():
    img = _hdr()
    assert aces(img).max() <= 1.0 and reinhard(img).max() < 1.0


def test_gamma_brightens_linear_midtone():
    assert gamma(np.full((2, 2, 3), 0.25))[0, 0, 0] > 0.25         # display-encode lifts midtones


def test_vignette_darkens_corners():
    v = vignette(np.ones((40, 40, 3)), strength=0.6)
    assert v[0, 0].mean() < v[20, 20].mean()


def test_bloom_adds_glow_energy():
    img = _hdr()
    out = bloom(img, threshold=1.0, sigma=4.0, intensity=0.8)
    assert out[16:32, 16:32].sum() > img[16:32, 16:32].sum()


def test_blur_is_energy_preserving_dc():
    img = _hdr()
    b = _fft_blur(img, 3.0)
    assert abs(b.sum() - img.sum()) / img.sum() < 1e-6            # circular Gaussian conserves total energy


def test_film_grain_deterministic_per_seed():
    img = _hdr()
    assert np.array_equal(film_grain(img, seed=1), film_grain(img, seed=1))
    assert not np.array_equal(film_grain(img, seed=1), film_grain(img, seed=2))


def test_dof_needs_depth_and_blurs_far():
    img = np.tile(np.linspace(0, 1, 48), (48, 1))[:, :, None].repeat(3, 2)
    depth = np.zeros((48, 48)); depth[:, 24:] = 8.0
    out = dof(img, depth=depth, focus=0.0, aperture=2.0, max_sigma=5.0)
    assert out.shape == img.shape
    assert dof(img, depth=None) is not None                       # no depth -> passthrough, no crash


def test_resample_roundtrip_shape():
    img = _hdr()
    assert resample(img, 2.0).shape[0] == 96
    assert supersample(resample(img, 2.0), 2).shape[0] == 48


def test_postchain_is_serializable_program():
    ch = default_chain()
    assert PostChain.from_list(ch.to_list()).to_list() == ch.to_list()
    assert (ch + PostChain().then("sharpen")).steps[-1][0] == "sharpen"
    assert "->" in repr(ch)


def test_postchain_runs_end_to_end_clamped():
    out = default_chain().apply(_hdr())
    assert out.shape == (48, 48, 3) and out.min() >= 0.0 and out.max() <= 1.0


def test_unknown_effect_raises():
    try:
        PostChain().then("does_not_exist")
        assert False
    except KeyError:
        pass


def test_render_post_integration():
    """post= on render_sdf composes the chain onto the frame (and changes it)."""
    import numpy as np
    from holographic_render import Camera
    from holographic_raymarch import render_sdf

    class Ball:
        def eval(self, P): return np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    cam = Camera(eye=(0, 0, 3.0)); W = H = 48
    raw = render_sdf(Ball(), cam, width=W, height=H, reflect=0.0)
    graded, depth = render_sdf(Ball(), cam, width=W, height=H, reflect=0.0,
                               post=default_chain(), return_depth=True)
    assert graded.shape == (H, W, 3) and depth.shape == (H, W)
    assert not np.allclose(raw, graded)                           # grading actually changed the frame

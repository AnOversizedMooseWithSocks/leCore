"""Tests for the coarse-first splat aniso-refine RE-ENABLE (fit_coarse_first / splat_refine_residual)."""
import numpy as np
from holographic_splat import splat_fit, splat_render, fit_coarse_first, splat_refine_residual, psnr

H = W = 64
_ys, _xs = np.mgrid[0:H, 0:W].astype(float)


def _sharp():
    t = (_ys > _xs).astype(float) * 0.9 + 0.1 + 0.4 * np.exp(-(((_ys - 45) ** 2 + (_xs - 20) ** 2) / 50.0))
    return t / t.max()


def _smooth():
    t = np.exp(-(((_ys - 20) ** 2 + (_xs - 20) ** 2) / 40.0)) + 0.8 * np.exp(-(((_ys - 45) ** 2 + (_xs - 42) ** 2) / 60.0))
    return t / t.max()


def test_coarse_first_beats_iso_baseline_on_sharp_edge():
    t = _sharp()
    iso = psnr(splat_render(splat_fit(t, 30), (H, W)), t)
    combined, _, _ = fit_coarse_first(t, K_iso=30, K_aniso=8)
    assert psnr(combined, t) > iso + 2.0                       # big win on anisotropic content


def test_no_harm_mode_across_targets():
    # aniso-refining the residual is never WORSE than the isotropic baseline
    for t in [_sharp(), _smooth(), np.random.default_rng(0).random((H, W))]:
        t = t / t.max()
        iso_splats = splat_fit(t, 30)
        iso = psnr(splat_render(iso_splats, (H, W)), t)
        combined, _ = splat_refine_residual(t, iso_splats, K_aniso=8, steps=120)
        assert psnr(combined, t) >= iso - 0.05                 # strictly >= baseline (no harm mode)


def test_refine_returns_render_and_splats():
    t = _sharp()
    iso_splats = splat_fit(t, 20)
    combined, aniso_splats = splat_refine_residual(t, iso_splats, K_aniso=6, steps=80)
    assert combined.shape == (H, W) and len(aniso_splats) == 6


def test_concentration_is_the_wrong_detector_here():
    # documents the honest finding: point-concentration is BACKWARDS for anisotropy (a spread edge vs a peaked blob)
    from holographic_coarsefirst import concentration
    rs = _sharp() - splat_render(splat_fit(_sharp(), 30), (H, W))
    rb = _smooth() - splat_render(splat_fit(_smooth(), 30), (H, W))
    assert concentration(np.abs(rs)) < concentration(np.abs(rb))   # sharp edge LESS concentrated than smooth blob

"""Tests for the Cook-Torrance/GGX physically-based BRDF (BRDF-1)."""
import numpy as np
from holographic_brdf import (fresnel_schlick, d_ggx, cook_torrance, sample_ggx,
                              directional_albedo, sample_brdf)


def test_white_furnace_energy_conserving():
    # a smooth dielectric in a unit environment reflects close to (not above) 1.0
    a = directional_albedo(metallic=0.0, roughness=0.1, n=8000)
    assert a <= 1.03


def test_single_scatter_loss_grows_with_roughness_kept_negative():
    a_smooth = directional_albedo(metallic=1.0, roughness=0.1, base_color=(1, 1, 1), n=8000)
    a_rough = directional_albedo(metallic=1.0, roughness=0.95, base_color=(1, 1, 1), n=8000)
    assert a_rough < a_smooth                                      # the documented GGX energy loss


def test_fresnel_rises_at_grazing():
    f0 = np.array([0.04, 0.04, 0.04])
    assert fresnel_schlick(1.0, f0).mean() < fresnel_schlick(0.05, f0).mean()


def test_ggx_importance_sampler_unbiased():
    rng = np.random.default_rng(0)
    N = np.tile([0.0, 0.0, 1.0], (40000, 1)).astype(float)
    V = np.tile([0.3, 0.0, 0.954], (40000, 1)).astype(float)
    V /= np.linalg.norm(V, axis=-1, keepdims=True)
    L, pdf = sample_ggx(N, V, 0.4, rng)
    ndl = np.clip(np.sum(N * L, -1), 0, 1); ok = (ndl > 0) & (pdf > 1e-6)
    fr = cook_torrance(N[ok], V[ok], L[ok], (1, 1, 1), 0.0, 0.4)
    est = float(np.mean(fr.sum(-1) / 3 / pdf[ok]))
    ref = directional_albedo(0.0, 0.4, view_cos=0.954, n=120000)
    assert abs(est - ref) < 0.06                                   # estimator matches brute force


def test_metal_has_no_diffuse():
    # a metal keeps only its (tinted) specular; a dielectric of the same dark base also has the diffuse lobe,
    # so over the whole hemisphere the dielectric reflects more
    metal = directional_albedo(metallic=1.0, roughness=1.0, base_color=(0.2, 0.2, 0.2), n=20000)
    diel = directional_albedo(metallic=0.0, roughness=1.0, base_color=(0.2, 0.2, 0.2), n=20000)
    assert diel > metal                                            # dielectric has the extra diffuse lobe


def test_sample_brdf_returns_finite_weight():
    rng = np.random.default_rng(1)
    N = np.tile([0.0, 0.0, 1.0], (500, 1)).astype(float)
    V = np.tile([0.0, 0.0, 1.0], (500, 1)).astype(float)
    L, w = sample_brdf(N, V, np.tile([0.7, 0.7, 0.7], (500, 1)), 0.0, 0.5, rng)
    assert np.isfinite(w).all() and (w >= 0).all()


def test_fresnel_dielectric_glass():
    """Dielectric Fresnel: ~0.04 reflectance at normal incidence for glass, rising to ~1 at grazing."""
    import numpy as np
    from holographic_brdf import fresnel_dielectric
    assert abs(float(fresnel_dielectric(1.0, 1.5)) - 0.04) < 0.005     # normal incidence
    assert float(fresnel_dielectric(0.02, 1.5)) > 0.7                  # grazing -> mostly reflective

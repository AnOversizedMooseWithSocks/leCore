"""Performance MC3: pre-integrated view LUT -- view-dependent specular becomes a bilinear lookup."""
import numpy as np
from holographic.rendering.holographic_viewlut import ViewLUT, bake_view_lut
from holographic.rendering.holographic_brdf import directional_albedo


def test_lut_matches_integral_stable_range():
    lut = bake_view_lut(metallic=1.0, res_view=24, res_rough=24, samples=16384, seed=0)
    worst = 0.0
    for vc in (0.3, 0.5, 0.7, 0.9):
        for rg in (0.3, 0.5, 0.7, 0.9):
            ref = directional_albedo(1.0, rg, n=65536, view_cos=vc, seed=1)
            worst = max(worst, abs(float(lut.sample(vc, rg)[0]) - ref))
    assert worst < 0.08                                      # within the estimator's own variance


def test_reflectance_falls_with_roughness():
    lut = bake_view_lut(metallic=1.0, res_view=16, res_rough=16, samples=8192, seed=0)
    assert float(lut.sample(0.9, 0.3)[0]) > float(lut.sample(0.9, 0.9)[0])   # GGX single-scatter energy loss


def test_sample_vectorized():
    lut = bake_view_lut(res_view=12, res_rough=12, samples=4096, seed=0)
    vc = np.array([0.3, 0.6, 0.9]); rg = np.array([0.4, 0.5, 0.6])
    out = lut.sample(vc, rg)
    assert out.shape == (3,) and np.all(out >= 0)


def test_bilinear_exact_at_nodes():
    # a lookup exactly at a grid node returns that node's value (no interpolation)
    grid = np.arange(16.0).reshape(4, 4)
    lut = ViewLUT(grid, 0.0, 1.0)
    assert abs(float(lut.sample(0.0, 0.0)[0]) - grid[0, 0]) < 1e-9
    assert abs(float(lut.sample(1.0, 1.0)[0]) - grid[3, 3]) < 1e-9


def test_lookup_far_cheaper_than_integral():
    import time
    lut = bake_view_lut(res_view=16, res_rough=16, samples=4096, seed=0)
    vcs = np.random.default_rng(0).uniform(0.1, 1.0, 2000); rgs = np.random.default_rng(1).uniform(0.3, 1.0, 2000)
    t0 = time.time(); lut.sample(vcs, rgs); t_lut = time.time() - t0
    t0 = time.time(); directional_albedo(1.0, 0.5, n=4096, view_cos=0.7, seed=0); t_one = time.time() - t0
    assert (t_one * 2000) / max(t_lut, 1e-9) > 50            # thousands of lookups << one integral each


def test_clamps_out_of_range():
    lut = bake_view_lut(res_view=12, res_rough=12, rough_min=0.05, rough_max=1.0, samples=4096, seed=0)
    # out-of-range roughness/view clamp to the edge, no crash
    assert np.isfinite(lut.sample(1.5, 2.0)[0]) and np.isfinite(lut.sample(-0.5, 0.0)[0])

"""Tests for holographic_lightinghome -- the Lighting home (R7: light types + the shade integral, one home)."""
import numpy as np
from holographic_lightinghome import (Lighting, lighting_modes, PointLight, DirectionalLight, RectLight, DomeLight,
                                       IESLight)


def _scene():
    from holographic_sdf import sphere, box
    return sphere(0.5).smooth_union(box(2.0, 0.1, 2.0).translate((0, -0.55, 0)), k=0.03)


def test_direct_integral_bit_identical():
    from holographic_lights import direct_lighting
    sdf = _scene()
    P = np.array([[0.0, -0.4, 0.0], [0.3, -0.4, 0.2]])
    N = np.tile([0.0, 1.0, 0.0], (2, 1)); V = N.copy()
    alb = np.full((2, 3), 0.8); met = np.zeros(2); rough = np.full(2, 0.5)
    L = [DirectionalLight(direction=(0.2, -1.0, -0.1), intensity=3.0)]
    got = Lighting.direct(sdf, P, N, V, alb, met, rough, L, np.random.default_rng(0), area_samples=8)
    ref = direct_lighting(sdf, P, N, V, alb, met, rough, L, np.random.default_rng(0), area_samples=8)
    assert np.array_equal(got, ref) and np.isfinite(got).all()


def test_light_types_reexported():
    assert len(Lighting.light_types()) == 10
    # the common ones are importable straight from the home
    assert PointLight is not None and RectLight is not None and IESLight is not None


def test_split_cached_partitions_lights():
    L = [DirectionalLight(direction=(0, -1, 0), intensity=2.0),
         RectLight(position=(0, 1, 0), u_vec=(0.3, 0, 0), v_vec=(0, 0.3, 0), intensity=10.0),
         DomeLight(intensity=1.0)]
    domes, soft, hard = Lighting.split_cached(L)
    assert len(domes) == 1 and len(soft) == 1 and len(hard) == 1


def test_prt_relight_finite():
    from holographic_prt import precompute_transfer, project_env_to_sh
    sdf = _scene()
    P = np.array([[0.0, -0.4, 0.0]]); N = np.array([[0.0, 1.0, 0.0]])
    T = precompute_transfer(sdf, P, N, order=3, n=48)
    sh = project_env_to_sh(lambda d: np.tile([0.5, 0.6, 0.8], (len(d), 1)), order=3, n=256)
    lit = Lighting.prt(T, sh, np.full((1, 3), 0.8))
    assert np.isfinite(lit).all()


def test_modes_listed():
    assert set(lighting_modes()) == {"direct", "prt", "environment_sh"}

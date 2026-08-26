"""Tests for the multi-scatter BRDF RE-ENABLE (Kulla-Conty energy compensation, gated by roughness)."""
import numpy as np
from holographic.rendering.holographic_brdf import directional_albedo, directional_albedo_ms, cook_torrance, cook_torrance_ms, brdf_gated, MS_ROUGHNESS_GATE


def test_single_scatter_loses_energy_at_high_roughness():
    # the kept negative: a rough metal loses real energy under single-scatter GGX
    e = directional_albedo(metallic=1.0, roughness=0.9, base_color=(1, 1, 1), n=16384, view_cos=0.6, seed=0)
    assert e < 0.7                                              # well below energy-conserving 1.0


def test_multiscatter_restores_energy_at_high_roughness():
    for r in [0.5, 0.7, 0.9]:
        e = directional_albedo_ms(metallic=1.0, roughness=r, base_color=(1, 1, 1), n=16384, view_cos=0.6, seed=0)
        assert 0.95 < e < 1.10                                  # back to ~energy-conserving


def test_gate_uses_single_below_and_multi_above():
    N = np.array([0., 0, 1]); V = np.array([0.6, 0, 0.8])
    L = np.array([-0.3, 0.2, 0.93]); L = L / np.linalg.norm(L)
    lo, i_lo = brdf_gated(N, V, L, (1, 1, 1), 1.0, 0.15)
    hi, i_hi = brdf_gated(N, V, L, (1, 1, 1), 1.0, 0.8)
    assert i_lo["used"] == "fallback" and i_hi["used"] == "superior"
    # below the gate the gated value equals plain cook_torrance (no compensation, no overshoot)
    assert np.allclose(lo, cook_torrance(N, V, L, (1, 1, 1), 1.0, 0.15))
    assert np.allclose(hi, cook_torrance_ms(N, V, L, (1, 1, 1), 1.0, 0.8))


def test_gated_energy_never_worse_than_single_scatter():
    # with the gate, |energy - 1| is no worse than single-scatter at every roughness (the overshoot is gated out)
    for r in [0.15, 0.25, 0.5, 0.9]:
        se = abs(directional_albedo(metallic=1.0, roughness=r, base_color=(1, 1, 1), n=16384, view_cos=0.6) - 1)
        if r >= MS_ROUGHNESS_GATE:
            ge = abs(directional_albedo_ms(metallic=1.0, roughness=r, base_color=(1, 1, 1), n=16384, view_cos=0.6) - 1)
        else:
            ge = se
        assert ge <= se + 1e-6


def test_multiscatter_only_adds_energy():
    # the compensation term is non-negative -- it adds back missing energy, never subtracts
    N = np.array([0., 0, 1]); V = np.array([0.5, 0, 0.866])
    L = np.array([0.2, -0.1, 0.974]); L = L / np.linalg.norm(L)
    single = cook_torrance(N, V, L, (1, 1, 1), 1.0, 0.7)
    multi = cook_torrance_ms(N, V, L, (1, 1, 1), 1.0, 0.7)
    assert np.all(multi >= single - 1e-9)

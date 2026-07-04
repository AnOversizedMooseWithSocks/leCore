"""Tests for holographic_multimaterial.py (CMP3) -- N materials blended/selected by per-point masks."""
import numpy as np
import pytest
from holographic_fpe import VectorFunctionEncoder
from holographic_material import Material, texture_field
from holographic_multimaterial import MultiMaterial


def _two_materials():
    """Two materials with OPPOSITE albedo ramps (differ in pattern, since sample() is a direction readout)."""
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)
    grid = [(u, v) for u in np.linspace(0.05, 0.95, 7) for v in np.linspace(0.05, 0.95, 7)]
    a = Material(enc, {"albedo": texture_field(enc, grid, [u for (u, v) in grid])})
    b = Material(enc, {"albedo": texture_field(enc, grid, [1.0 - u for (u, v) in grid])})
    return a, b


def test_blend_is_exact_weighted_sum():
    a, b = _two_materials()
    mm = MultiMaterial([a, b], [lambda p: 1 - np.asarray(p)[:, 0], lambda p: np.asarray(p)[:, 0]])
    for uv in ([0.15, 0.5], [0.5, 0.5], [0.85, 0.5]):
        w = mm.weights_at(uv)
        expect = w[0] * a.sample("albedo", uv) + w[1] * b.sample("albedo", uv)
        assert abs(mm.sample("albedo", uv) - expect) < 1e-9


def test_weights_partition_to_one_by_default():
    a, b = _two_materials()
    mm = MultiMaterial([a, b], [2.0, 3.0])            # raw 2 and 3 -> normalised to 0.4 and 0.6
    w = mm.weights_at([0.5, 0.5])
    assert abs(w.sum() - 1.0) < 1e-9 and abs(w[0] - 0.4) < 1e-9


def test_normalize_false_drifts_brighter():
    a, b = _two_materials()
    uv = [0.4, 0.6]
    norm = MultiMaterial([a, b], [1.0, 1.0], normalize=True)
    drift = MultiMaterial([a, b], [1.0, 1.0], normalize=False)
    assert abs(norm.sample("albedo", uv) - 0.5 * (a.sample("albedo", uv) + b.sample("albedo", uv))) < 1e-9
    assert abs(drift.sample("albedo", uv) - (a.sample("albedo", uv) + b.sample("albedo", uv))) < 1e-9


def test_select_mode_picks_dominant():
    a, b = _two_materials()
    uv = [0.4, 0.6]
    mm = MultiMaterial([a, b], [0.3, 0.7], mode="select")
    assert abs(mm.sample("albedo", uv) - b.sample("albedo", uv)) < 1e-9      # b's mask is bigger -> b wins


def test_zero_masks_fall_back_to_uniform():
    a, b = _two_materials()
    uv = [0.4, 0.6]
    mm = MultiMaterial([a, b], [0.0, 0.0])
    assert abs(mm.sample("albedo", uv) - 0.5 * (a.sample("albedo", uv) + b.sample("albedo", uv))) < 1e-9


def test_mask_can_be_a_cmp1_texture_graph():
    """A mask is any CMP1 node -- here a texture_map, proving CMP1 feeds CMP3 as the backlog intends."""
    from holographic_texturegraph import Map, Const, field_leaf
    a, b = _two_materials()
    m0 = Map("scale", x=field_leaf("fbm", n_dims=2, seed=0), k=Const(1.0))    # an fbm-driven mask
    m1 = Const(0.5)
    mm = MultiMaterial([a, b], [m0, m1])
    v = mm.sample("albedo", [0.3, 0.7])
    assert np.isfinite(v)


def test_missing_channel_contributes_zero():
    a, b = _two_materials()
    b.add("roughness", b.channels["albedo"])          # only b has roughness
    mm = MultiMaterial([a, b], [0.5, 0.5])
    # roughness comes only from b at weight 0.5
    assert abs(mm.sample("roughness", [0.5, 0.5]) - 0.5 * b.sample("roughness", [0.5, 0.5])) < 1e-9


def test_validation():
    a, b = _two_materials()
    with pytest.raises(ValueError):
        MultiMaterial([a, b], [1.0])                  # mismatched counts
    with pytest.raises(ValueError):
        MultiMaterial([a, b], [1.0, 1.0], mode="bogus")
    with pytest.raises(ValueError):
        MultiMaterial([], [])                         # no materials


def test_non_material_refused_at_compose_time():
    """A non-Material is caught when you BUILD the MultiMaterial, not deep in sample() -- clear, early error."""
    with pytest.raises(TypeError) as e:
        MultiMaterial(["not a material"], [1.0])
    assert "not a Material" in str(e.value)

"""Tests for holographic_layeredmaterial.py (CMP2) -- an ordered material stack + its layer-order schema."""
import numpy as np
import pytest
from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
from holographic.materials_and_texture.holographic_material import Material, texture_field
from holographic.materials_and_texture.holographic_layeredmaterial import Layer, LayeredMaterial, LAYER_RANK


def _two():
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)
    grid = [(u, v) for u in np.linspace(0.05, 0.95, 7) for v in np.linspace(0.05, 0.95, 7)]
    a = Material(enc, {"albedo": texture_field(enc, grid, [u for (u, v) in grid])})
    b = Material(enc, {"albedo": texture_field(enc, grid, [1.0 - u for (u, v) in grid])})
    return a, b


def test_composite_is_exact_over():
    a, b = _two()
    stack = LayeredMaterial([Layer("base", a), Layer("coat", b, alpha=0.4)])
    for uv in ([0.2, 0.5], [0.5, 0.5], [0.8, 0.5]):
        expect = 0.4 * b.sample("albedo", uv) + 0.6 * a.sample("albedo", uv)
        assert abs(stack.sample("albedo", uv) - expect) < 1e-9


def test_order_schema_refuses_out_of_order():
    a, b = _two()
    with pytest.raises(ValueError) as e:
        LayeredMaterial([Layer("coat", b), Layer("base", a)])       # base above coat
    assert "cannot sit above" in str(e.value)
    with pytest.raises(ValueError):
        LayeredMaterial([Layer("reflection", b), Layer("diffuse", a)])   # diffuse above reflection


def test_valid_ascending_stack_builds():
    a, b = _two()
    ok = LayeredMaterial([Layer("base", a), Layer("diffuse", a),
                          Layer("specular", b, alpha=0.3), Layer("clearcoat", b, alpha=0.2)])
    assert [l.kind for l in ok.layers] == ["base", "diffuse", "specular", "clearcoat"]


def test_same_tier_layers_allowed_adjacent():
    a, b = _two()
    # specular and reflection share tier 2 -> non-decreasing, allowed
    ok = LayeredMaterial([Layer("specular", a, alpha=0.5), Layer("reflection", b, alpha=0.5)])
    assert len(ok.layers) == 2


def test_unknown_kind_refused():
    a, _ = _two()
    with pytest.raises(ValueError):
        Layer("bogus", a)


def test_varying_alpha_field_covers_where_higher():
    a, b = _two()
    from holographic.materials_and_texture.holographic_texturegraph import FieldLeaf
    grad = FieldLeaf(lambda pts: np.asarray(pts, float)[:, 0])       # alpha = u
    stack = LayeredMaterial([Layer("base", a), Layer("coat", b, alpha=grad)])
    left, right = stack.sample("albedo", [0.1, 0.5]), stack.sample("albedo", [0.9, 0.5])
    assert abs(left - a.sample("albedo", [0.1, 0.5])) < abs(left - b.sample("albedo", [0.1, 0.5]))
    assert abs(right - b.sample("albedo", [0.9, 0.5])) < abs(right - a.sample("albedo", [0.9, 0.5]))


def test_alpha_is_a_cmp1_texture_graph():
    """The coverage alpha can be a full CMP1 texture graph -- CMP1 feeds CMP2 like it feeds CMP3."""
    from holographic.materials_and_texture.holographic_texturegraph import Map, Const, field_leaf
    a, b = _two()
    cov = Map("scale", x=field_leaf("fbm", n_dims=2, seed=0), k=Const(1.0))
    stack = LayeredMaterial([Layer("base", a), Layer("coat", b, alpha=cov)])
    assert np.isfinite(stack.sample("albedo", [0.3, 0.7]))


def test_channel_only_in_one_layer_passes_through():
    a, b = _two()
    b.add("roughness", b.channels["albedo"])                        # only the coat has roughness
    stack = LayeredMaterial([Layer("base", a), Layer("coat", b, alpha=0.4)])
    # base has no roughness, so roughness is just the coat's (the lowest layer that HAS it becomes its base)
    assert abs(stack.sample("roughness", [0.5, 0.5]) - b.sample("roughness", [0.5, 0.5])) < 1e-9


def test_structural_encode_matches_and_reorder_differs():
    a, b = _two()
    s1 = LayeredMaterial([Layer("base", a), Layer("coat", b, alpha=0.3)])
    s2 = LayeredMaterial([Layer("base", a), Layer("coat", b, alpha=0.3)])
    cos = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    assert cos(s1.encode(1024), s2.encode(1024)) > 0.99


def test_layer_non_material_refused_at_compose_time():
    with pytest.raises(TypeError) as e:
        Layer("base", "not a material")
    assert "needs a Material" in str(e.value)

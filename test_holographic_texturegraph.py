"""Tests for holographic_texturegraph.py (CMP1) -- the composable texture map graph + its compose-time schema."""
import numpy as np
import pytest
from holographic_texturegraph import (Const, FieldLeaf, Map, field_leaf, sample_grid, encode, to_expr,
                                       NUMBER, COLOR, FIELD, MAP)


def test_leaf_kinds():
    assert Const(0.5).kind == NUMBER
    assert Const([1, 0, 0]).kind == COLOR
    assert field_leaf("fbm", n_dims=2).kind == FIELD
    with pytest.raises(TypeError):
        Const([1, 2])                          # length-2 is neither a number nor a 3/4 color


def test_nested_map_samples():
    base = Map("mix", a=Const([1.0, 0, 0]), b=Const([0, 0, 1.0]), t=field_leaf("fbm", n_dims=2, seed=0))
    top = Map("multiply", a=base, b=Const([1.0, 1.0, 1.0]))     # a map whose input is another map
    assert top.kind == MAP
    out = top.sample([0.3, 0.7])
    assert out.shape == (3,)


def test_sampling_is_deterministic():
    m = Map("mix", a=Const(0.0), b=Const(1.0), t=field_leaf("fbm", n_dims=2, seed=1))
    assert np.allclose(m.sample([0.2, 0.9]), m.sample([0.2, 0.9]))


def test_schema_rejects_color_as_weight():
    with pytest.raises(TypeError) as e:
        Map("mix", a=Const([1, 0, 0]), b=Const([0, 1, 0]), t=Const([0, 0, 1]))   # color in the weight slot
    assert "accepts" in str(e.value)


def test_schema_rejects_missing_and_extra_inputs():
    with pytest.raises(TypeError):
        Map("mix", a=Const(0.0), b=Const(1.0))                 # missing t
    with pytest.raises(TypeError):
        Map("multiply", a=Const(0.0), b=Const(1.0), c=Const(2.0))   # extra c


def test_unknown_op_refused():
    with pytest.raises(ValueError):
        Map("bogus", a=Const(0.0), b=Const(1.0))


def test_ops_evaluate():
    assert np.isclose(Map("add", a=Const(2.0), b=Const(3.0)).sample([0, 0]), 5.0)
    assert np.isclose(Map("multiply", a=Const(2.0), b=Const(3.0)).sample([0, 0]), 6.0)
    assert np.isclose(Map("mix", a=Const(0.0), b=Const(10.0), t=Const(0.25)).sample([0, 0]), 2.5)
    assert np.isclose(Map("scale", x=Const(4.0), k=Const(0.5)).sample([0, 0]), 2.0)
    assert np.isclose(Map("over", a=Const(1.0), b=Const(0.0), alpha=Const(0.75)).sample([0, 0]), 0.75)
    assert np.isclose(Map("remap", x=Const(0.5), lo=Const(2.0), hi=Const(4.0)).sample([0, 0]), 3.0)


def test_grid_bake_shapes():
    scalar = Map("scale", x=field_leaf("fbm", n_dims=2), k=Const(1.0))
    assert sample_grid(scalar, res=6).shape == (6, 6)
    color = Map("mix", a=Const([1.0, 0, 0]), b=Const([0, 1.0, 0]), t=Const(0.5))
    assert sample_grid(color, res=6).shape == (6, 6, 3)


def test_structural_encode_matches_and_differs():
    g1 = Map("mix", a=Const([1.0, 0, 0]), b=Const([0, 0, 1.0]), t=field_leaf("fbm", n_dims=2, seed=0))
    g2 = Map("mix", a=Const([1.0, 0, 0]), b=Const([0, 0, 1.0]), t=field_leaf("fbm", n_dims=2, seed=0))
    g3 = Map("multiply", a=Const([1.0, 0, 0]), b=Const([0, 0, 1.0]))
    v1, v2, v3 = encode(g1, 1024), encode(g2, 1024), encode(g3, 1024)
    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos(v1, v2) > 0.99                  # same structure -> same code (identical)
    # g3 (multiply of the two colours) shares the colour leaves but differs in op + arity, so it is clearly
    # SEPARABLE from g1 (well below the identical score) without being orthogonal -- shared structure shows.
    assert cos(v1, v3) < 0.9


def test_to_expr_shape():
    g = Map("mix", a=Const(0.0), b=Const(1.0), t=field_leaf("fbm", n_dims=2))
    expr = to_expr(g)
    assert expr[0] == "mix" and expr[3].startswith("field:")


def test_const_accepts_color_names():
    from holographic_semantic import COLORS
    c = Const("red")
    assert c.kind == COLOR
    assert np.allclose(c.sample([0, 0]), COLORS["red"])       # resolves to the scene system's tuned rgb
    assert c.sample([0, 0])[0] > c.sample([0, 0])[1]          # red channel dominant, sanity


def test_const_bad_string_gives_clear_error():
    with pytest.raises(TypeError) as e:
        Const("chartreuse")                      # not a known colour name
    assert "colour name" in str(e.value) and "chartreuse" in str(e.value)


def test_named_color_flows_through_map():
    g = Map("mix", a=Const("red"), b=Const("blue"), t=Const(0.5))
    out = g.sample([0.3, 0.7])
    assert out.shape == (3,)


def test_field_leaf_unknown_source_lists_available():
    with pytest.raises(ValueError) as e:
        field_leaf("perlin")
    msg = str(e.value)
    assert "available sources" in msg and "fbm" in msg and "voronoi" in msg


def test_saturate_and_clamp_keep_values_in_range():
    # saturate: an out-of-range value comes back inside [0,1]
    hot = Map("scale", x=Const([1.0, 1.0, 1.0]), k=Const(2.0))       # -> 2.0 per channel (too bright)
    sat = Map("saturate", x=hot)
    assert np.allclose(sat.sample([0, 0]), [1.0, 1.0, 1.0])
    # clamp to a custom range
    c = Map("clamp", x=Const(5.0), lo=Const(0.0), hi=Const(3.0))
    assert np.isclose(c.sample([0, 0]), 3.0)
    c2 = Map("clamp", x=Const(-2.0), lo=Const(-1.0), hi=Const(1.0))
    assert np.isclose(c2.sample([0, 0]), -1.0)

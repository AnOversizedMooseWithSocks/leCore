"""Tests for holographic_preview.py -- texture swatch + material ball previews of the composability stack."""
import numpy as np
from holographic.materials_and_texture.holographic_texturegraph import Map, Const, field_leaf
from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
from holographic.materials_and_texture.holographic_material import Material, texture_field
from holographic.misc.holographic_preview import texture_image, material_ball


def _colorgraph():
    return Map("mix", a=Const("red"), b=Const("blue"), t=field_leaf("fbm", n_dims=2, seed=0))


def _mat(**channels):
    enc = VectorFunctionEncoder(2, dim=512, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)
    grid = [(a, b) for a in np.linspace(0.05, 0.95, 6) for b in np.linspace(0.05, 0.95, 6)]
    return Material(enc, {name: texture_field(enc, grid, [fn(a, b) for (a, b) in grid]) for name, fn in channels.items()})


def test_texture_swatch_shape_and_range():
    img = texture_image(_colorgraph(), res=48)
    assert img.shape == (48, 48, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_scalar_graph_is_greyscale():
    g = Map("scale", x=field_leaf("fbm", n_dims=2, seed=2), k=Const(1.0))
    img = texture_image(g, res=32)
    assert np.allclose(img[:, :, 0], img[:, :, 1]) and np.allclose(img[:, :, 1], img[:, :, 2])


def test_swatch_clamps_out_of_range_values():
    hot = Map("scale", x=Const([1.0, 1.0, 1.0]), k=Const(3.0))     # 3.0 per channel -> must clamp to 1.0
    img = texture_image(hot, res=8)
    assert img.max() <= 1.0


def test_material_ball_shape_range_and_shaded():
    mat = _mat(roughness=lambda a, b: a, metallic=lambda a, b: 0.0)
    ball = material_ball(mat, res=96)
    assert ball.shape == (96, 96, 3) and ball.min() >= 0.0 and ball.max() <= 1.0
    assert not np.allclose(ball[48, 48], ball[0, 0])              # the sphere was actually shaded vs background


def test_material_ball_uses_metallic_and_roughness():
    shiny = _mat(roughness=lambda a, b: 0.1, metallic=lambda a, b: 1.0)
    rough = _mat(roughness=lambda a, b: 0.9, metallic=lambda a, b: 0.0)
    b1 = material_ball(shiny, res=80)
    b2 = material_ball(rough, res=80)
    assert not np.allclose(b1, b2)                               # different material -> different ball


def test_layered_material_previews():
    from holographic.materials_and_texture.holographic_layeredmaterial import Layer, LayeredMaterial
    mat = _mat(roughness=lambda a, b: a, metallic=lambda a, b: 0.5)
    stack = LayeredMaterial([Layer("base", mat), Layer("coat", mat, alpha=0.3)])
    ball = material_ball(stack, res=64)
    assert ball.shape == (64, 64, 3)


def test_multi_material_previews():
    from holographic.materials_and_texture.holographic_multimaterial import MultiMaterial
    a = _mat(roughness=lambda x, y: x)
    b = _mat(roughness=lambda x, y: 1.0 - x)
    mm = MultiMaterial([a, b], [0.5, 0.5])
    ball = material_ball(mm, res=64)
    assert ball.shape == (64, 64, 3)

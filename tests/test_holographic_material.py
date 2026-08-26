"""Tests for G2 the holographic material/texture stack (holographic_material): a texture is an FPE function
over UV; a material is a role-filler HRR record sum_r bind(role_r, channel_r). sample() is exact (stored
field), the bare record recovers channels with measured (balanced) crosstalk, blend is linear, transform_uv
re-UVs every channel with one bind, and geometry+appearance compose into one vector."""

import numpy as np

from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
from holographic.agents_and_reasoning.holographic_ai import unbind, cosine
from holographic.materials_and_texture.holographic_material import Material, texture_field, sample_texture, compose_object, _selftest


def _setup():
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], bandwidth=3.0, seed=1)
    grid = [(u, v) for u in np.linspace(0.05, 0.95, 9) for v in np.linspace(0.05, 0.95, 9)]
    alb = texture_field(enc, grid, [u for (u, v) in grid])
    rough = texture_field(enc, grid, [0.5] * len(grid))
    return enc, grid, alb, rough


def test_texture_tracks_values():
    enc, grid, alb, _ = _setup()
    us = np.linspace(0.2, 0.8, 20)
    read = np.array([sample_texture(enc, alb, [u, 0.5]) for u in us])
    assert np.corrcoef(read, us)[0, 1] > 0.95


def test_sample_is_exact():
    enc, grid, alb, rough = _setup()
    mat = Material(enc, {"albedo": alb, "roughness": rough})
    uv = [0.4, 0.5]
    assert abs(mat.sample("albedo", uv) - sample_texture(enc, alb, uv)) < 1e-9


def test_record_recovery_balanced():
    enc, grid, alb, rough = _setup()
    height = texture_field(enc, grid, [np.exp(-((u-.5)**2+(v-.5)**2)/.05) for (u, v) in grid])
    mat = Material(enc, {"albedo": alb, "roughness": rough, "height": height})
    recalls = [cosine(mat.channel(n), Material._unit(mat.channels[n])) for n in mat.channels]
    assert min(recalls) > 0.45            # no channel swamped by the others


def test_blend_is_linear():
    enc, grid, alb, rough = _setup()
    m1 = Material(enc, {"albedo": alb})
    m2 = Material(enc, {"albedo": rough})
    mix = m1.blend(m2, 0.7)
    uv = [0.4, 0.5]
    want = 0.7 * m1.sample("albedo", uv) + 0.3 * m2.sample("albedo", uv)
    assert abs(mix.sample("albedo", uv) - want) < 0.05


def test_transform_uv_shifts_all_channels():
    enc, grid, alb, rough = _setup()
    mat = Material(enc, {"albedo": alb})
    d = np.array([0.15, 0.0])
    uv = np.array([0.55, 0.5])
    assert abs(mat.transform_uv(d).sample("albedo", uv) - mat.sample("albedo", uv - d)) < 0.05


def test_compose_object_recovers_both_sides():
    enc, grid, alb, rough = _setup()
    mat = Material(enc, {"albedo": alb, "roughness": rough})
    rng = np.random.default_rng(0)
    geom = rng.standard_normal(1024); geom /= np.linalg.norm(geom)
    obj, roles = compose_object(geom, mat)
    rec_unit = mat.record / np.linalg.norm(mat.record)
    app = unbind(obj, roles["APPEARANCE"])
    assert cosine(app, rec_unit) > 0.45 and cosine(app, rec_unit) > 3 * abs(cosine(app, geom))


def test_selftest_runs():
    _selftest()

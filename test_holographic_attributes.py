"""Tests for G6 the attribute channel (holographic_attributes): a per-vertex/texel attribute as a
RESOLUTION-INDEPENDENT FPE field (bake at any density, shared points keep their values), plus a light
additive raster store (.data dict) for hard masks."""

import numpy as np

from holographic_fpe import VectorFunctionEncoder
from holographic_mesh import box
from holographic_attributes import (attribute_field, sample_attribute, bake_to_vertices,
                                    attach_attribute, get_attribute, _selftest)


def _field():
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], bandwidth=3.0, seed=1)
    grid = [(u, v) for u in np.linspace(0.05, 0.95, 9) for v in np.linspace(0.05, 0.95, 9)]
    return enc, attribute_field(enc, grid, [u for (u, v) in grid])


def test_attribute_tracks_values():
    enc, field = _field()
    us = np.linspace(0.2, 0.8, 20)
    read = np.array([sample_attribute(enc, field, [u, 0.5]) for u in us])
    assert np.corrcoef(read, us)[0, 1] > 0.95


def test_resolution_independent():
    enc, field = _field()
    coarse = np.array([[u, 0.5] for u in np.linspace(0.2, 0.8, 7)])
    dense = np.array([[u, 0.5] for u in np.linspace(0.2, 0.8, 13)])
    assert np.allclose(bake_to_vertices(enc, field, coarse), bake_to_vertices(enc, field, dense)[::2], atol=1e-9)


def test_raster_store_roundtrip():
    cube = box(1, 1, 1)
    attach_attribute(cube, "wear", np.linspace(0, 1, cube.n_vertices))
    assert len(get_attribute(cube, "wear")) == cube.n_vertices
    assert get_attribute(cube, "missing", default=-1) == -1


def test_raster_rejects_wrong_length():
    cube = box(1, 1, 1)
    try:
        attach_attribute(cube, "bad", np.zeros(cube.n_vertices + 3))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_selftest_runs():
    _selftest()

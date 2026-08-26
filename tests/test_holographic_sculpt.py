"""Tests for FS-1 implicit-field sculpt brushes (holographic_sculpt): falloff-weighted LOCAL edits of a field
function (inflate, carve, smooth, grab, flatten, pinch). Each changes the surface only inside the brush ball, the
re-extracted mesh stays watertight/manifold, and the same operator reshapes any field (e.g. a value landscape)."""

import numpy as np

from holographic.mesh_and_geometry.holographic_sculpt import falloff, apply_brush, brush_inflate, brush_carve, brush_smooth, brush_grab, brush_flatten, brush_pinch
from holographic.mesh_and_geometry.holographic_meshbridge import metaball_field, sample_field, marching_tetrahedra


def _setup():
    fn = metaball_field(np.array([[0.0, 0.0, 0.0]]), radius=0.4)
    bounds = ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))
    res, level = 28, 0.5
    (x0, y0, z0), (x1, y1, z1) = bounds
    xs = np.linspace(x0, x1, res); ys = np.linspace(y0, y1, res); zs = np.linspace(z0, z1, res)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    return fn, bounds, res, level, grid


def test_inflate_grows_carve_shrinks():
    fn, _b, _r, level, grid = _setup()
    p = np.zeros(3)
    base = int((fn(grid) > level).sum())
    inf = apply_brush(fn, "inflate", p, 1.0, s=0.4)
    car = apply_brush(fn, "carve", p, 1.0, s=0.4)
    assert int((inf(grid) > level).sum()) > base
    assert int((car(grid) > level).sum()) < base


def test_all_brushes_are_local():
    fn, _b, _r, _level, grid = _setup()
    p = np.zeros(3); r = 1.0
    far = grid[np.linalg.norm(grid - p, axis=1) > r + 1e-9]
    edits = [
        brush_inflate(fn, p, r, s=0.4),
        brush_carve(fn, p, r, s=0.4),
        brush_smooth(fn, p, r, s=0.6),
        brush_grab(fn, p, r, np.array([0.2, 0.0, 0.0])),
        brush_flatten(fn, p, r, 0.7, s=0.5),
        brush_pinch(fn, p, r, s=0.5),
    ]
    for e in edits:
        assert np.max(np.abs(e(far) - fn(far))) < 1e-12     # bit-identical outside the ball


def test_reextract_stays_manifold():
    fn, bounds, res, level, _grid = _setup()
    edited = apply_brush(fn, "inflate", np.zeros(3), 1.0, s=0.4)
    vals, axes = sample_field(edited, bounds, res)
    mesh = marching_tetrahedra(vals, axes, level=level)
    assert mesh.is_manifold()
    assert len(mesh.faces) > 0


def test_grab_displaces_inside_only():
    fn, _b, _r, _level, grid = _setup()
    p = np.array([0.4, 0.0, 0.0]); r = 0.6
    grabbed = brush_grab(fn, p, r, np.array([0.15, 0.0, 0.0]))
    near = grid[np.linalg.norm(grid - p, axis=1) < r * 0.5]
    # at least some near-centre points change (the domain was dragged there)
    assert np.max(np.abs(grabbed(near) - fn(near))) > 1e-6


def test_brush_reshapes_a_value_field():
    # "works on any field": inflate raises a reward landscape locally (reward shaping)
    def value_field(P):
        P = np.asarray(P, float)
        return np.exp(-np.sum((P - np.array([0.5, 0.5, 0.5])) ** 2, axis=1))
    shaped = apply_brush(value_field, "inflate", np.array([0.5, 0.5, 0.5]), 0.4, s=1.0)
    q = np.array([[0.5, 0.5, 0.5]])
    assert shaped(q)[0] > value_field(q)[0]
    far = np.array([[-1.0, -1.0, -1.0]])                    # outside the ball -> unchanged
    assert abs(shaped(far)[0] - value_field(far)[0]) < 1e-12


def test_falloff_matches_shapes_and_is_zero_beyond_radius():
    # centre weight 1, beyond radius exactly 0; smoothstep and linear at the shipped shapes
    assert abs(falloff(0.0, 1.0, "smooth") - 1.0) < 1e-12
    assert abs(falloff(0.0, 1.0, "linear") - 1.0) < 1e-12
    assert falloff(1.5, 1.0, "smooth") == 0.0
    assert falloff(1.5, 1.0, "linear") == 0.0
    t = 0.5
    assert abs(falloff(0.5, 1.0, "linear") - (1 - t)) < 1e-12
    assert abs(falloff(0.5, 1.0, "smooth") - (1 - (3 * t ** 2 - 2 * t ** 3))) < 1e-12


def test_apply_brush_dispatch_and_unknown():
    fn, _b, _r, _level, grid = _setup()
    p = np.zeros(3)
    # flatten needs level, grab needs drag -- dispatch routes the kwargs
    assert apply_brush(fn, "flatten", p, 1.0, level=0.6, s=0.5)(grid).shape == (len(grid),)
    assert apply_brush(fn, "grab", p, 1.0, drag=np.array([0.1, 0.0, 0.0]))(grid).shape == (len(grid),)
    try:
        apply_brush(fn, "nonsense", p, 1.0)
        assert False
    except ValueError:
        pass


def test_deterministic():
    fn, _b, _r, _level, grid = _setup()
    a = apply_brush(fn, "inflate", np.zeros(3), 1.0, s=0.4)(grid)
    b = apply_brush(fn, "inflate", np.zeros(3), 1.0, s=0.4)(grid)
    assert np.array_equal(a, b)

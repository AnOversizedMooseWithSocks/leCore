"""Tests for holographic_sdfscene -- the SDFScene base class an app subclasses to get .eval/.part_ids/.ids."""
import numpy as np
from holographic.mesh_and_geometry.holographic_sdfscene import SDFScene


def _sphere(center, radius):
    """A signed-distance sphere: negative inside, zero on the surface, positive outside (engine convention)."""
    c = np.asarray(center, float)
    return lambda P: np.linalg.norm(np.atleast_2d(P) - c, axis=1) - radius


class _ThreeSpheres(SDFScene):
    def parts(self):
        return [(_sphere((0, 0, 0), 1.0), "red"),
                (_sphere((3, 0, 0), 1.0), "blue"),
                (_sphere((0, 3, 0), 0.5), "green")]

    def bounds(self):
        return [((0, 0, 0), 1.0), ((3, 0, 0), 1.0), ((0, 3, 0), 0.5)]


def test_eval_is_min_over_parts():
    s = _ThreeSpheres()
    P = np.array([[0, 0, 0], [3, 0, 0], [1.5, 0, 0], [0, 3, 0]], float)
    # inside sphere 0 (-1), inside sphere 1 (-1), midway between 0 and 1 (0.5), inside sphere 2 (-0.5).
    assert np.allclose(s.eval(P), [-1.0, -1.0, 0.5, -0.5])


def test_part_ids_is_argmin_and_ids_is_alias():
    s = _ThreeSpheres()
    P = np.array([[0, 0, 0], [3, 0, 0], [1.5, 0, 0], [0, 3, 0]], float)
    assert list(s.part_ids(P)) == [0, 1, 0, 2]
    assert np.array_equal(s.ids(P), s.part_ids(P))            # exporters call .ids -- must be the same thing


def test_material_at_follows_the_owning_part():
    s = _ThreeSpheres()
    P = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]], float)
    assert list(s.material_at(P)) == ["red", "blue", "green"]


def test_parts_near_matches_brute_force():
    """The SpatialGrid cull returns exactly the parts a brute-force sphere test would -- no false prunes."""
    s = _ThreeSpheres()
    q, r = (0.2, 0.1, 0.0), 0.5
    near = s.parts_near(q, r)
    centers, radii = [(0, 0, 0), (3, 0, 0), (0, 3, 0)], [1.0, 1.0, 0.5]
    brute = sorted(i for i, c in enumerate(centers)
                   if np.linalg.norm(np.array(q, float) - c) <= r + max(radii))
    assert near == brute


def test_parts_near_without_bounds_returns_all():
    """A scene that doesn't supply bounds() gets no culling -- parts_near yields every part, so callers can
    always rely on it. (Kept negative: no bounds means no acceleration.)"""
    class NoBounds(SDFScene):
        def parts(self):
            return [(_sphere((0, 0, 0), 1.0), "a"), (_sphere((5, 0, 0), 1.0), "b")]
    s = NoBounds()
    assert s.parts_near((0, 0, 0), 0.1) == [0, 1]


def test_empty_scene_is_well_behaved():
    class Empty(SDFScene):
        def parts(self):
            return []
    s = Empty()
    P = np.zeros((3, 3))
    assert np.all(np.isinf(s.eval(P)))                        # nothing anywhere -> +inf distance
    assert list(s.part_ids(P)) == [-1, -1, -1]                # ...and no owning part


def test_selftest_runs():
    from holographic.mesh_and_geometry.holographic_sdfscene import _selftest
    _selftest()

"""Tests for holographic_blendhome -- the Blend home (H4: superpose/interpolate/average/composite/merge)."""
import numpy as np
from holographic_blendhome import Blend, blend_backends


def test_weighted_bundle_equals_blend_pose():
    from holographic_blendpose import blend_pose
    targets = np.random.default_rng(0).standard_normal((3, 32)); w = np.array([0.5, 0.3, 0.2])
    assert np.array_equal(Blend.bundle(targets, w), blend_pose(targets, w))


def test_unweighted_bundle_routes_to_ai():
    from holographic_ai import bundle
    V = np.random.default_rng(1).standard_normal((4, 16))
    assert np.array_equal(Blend.bundle(V), bundle(V))


def test_slerp_equals_ai_slerp_and_stays_on_sphere():
    from holographic_ai import slerp
    a = np.zeros(4); a[0] = 1.0; b = np.zeros(4); b[1] = 1.0
    assert np.array_equal(Blend.slerp(a, b, 0.5), slerp(a, b, 0.5))
    assert abs(np.linalg.norm(Blend.slerp(a, b, 0.5)) - 1.0) < 1e-9


def test_lerp_is_the_chord():
    a = np.array([1.0, 0.0]); b = np.array([0.0, 1.0])
    assert np.allclose(Blend.lerp(a, b, 0.25), [0.75, 0.25])


def test_alpha_composite_front_hides_back():
    col, acc = Blend.alpha_composite(np.array([[1., 0, 0], [0, 1., 0]]), np.array([1.0, 1.0]))
    assert np.allclose(col, [1., 0, 0]) and abs(acc - 1.0) < 1e-9
    # a half-transparent front over an opaque back mixes them
    col2, _ = Blend.alpha_composite(np.array([[1., 0, 0], [0, 0, 1.]]), np.array([0.5, 1.0]))
    assert np.allclose(col2, [0.5, 0, 0.5])


def test_merge_conflict_policies():
    assert Blend.merge({"a": 1, "b": 2}, {"b": 9, "c": 3}, policy="prefer_a") == {"a": 1, "b": 2, "c": 3}
    assert Blend.merge({"b": 2}, {"b": 9}, policy="prefer_b") == {"b": 9}
    assert Blend.merge({"b": 2.0}, {"b": 4.0}, policy="average") == {"b": 3.0}


def test_frechet_mean_on_sphere():
    a = np.zeros(4); a[0] = 1.0; b = np.zeros(4); b[1] = 1.0
    m = Blend.mean(np.array([a, b]))
    assert abs(np.linalg.norm(m) - 1.0) < 1e-6


def test_backends_listed():
    assert set(blend_backends()) == {"bundle", "lerp", "slerp", "mean", "alpha_composite", "merge"}

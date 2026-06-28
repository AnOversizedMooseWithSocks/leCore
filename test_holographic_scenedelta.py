"""Tests for holographic_scenedelta: the explicit diff/transmission + dedup-measurement layer over the AUTOMATIC
content-addressed sharing of scene components. The dedup itself is free (content-hashed atoms); this adds the
transmittable delta and the saving measurement."""

from holographic_scenegraph import SceneNode, translation
from holographic_mesh import box
from holographic_scenedelta import scene_components, scene_delta, apply_scene_delta, scene_dedup_saving

_CUBE = box()
_OTHER = box(2, 1, 1)


def _variant(i, changed=True):
    children = [SceneNode(translation([0, 0, 0]), mesh=_CUBE),
                SceneNode(translation([2, 0, 0]), mesh=_CUBE),
                SceneNode(translation([0, 2, 0]), mesh=_OTHER),
                SceneNode(translation([2, 2, 0]), mesh=_CUBE)]
    if changed:
        children[i % 4] = SceneNode(translation([float(i), 5, 0]), mesh=_CUBE)
    return SceneNode(children=children)


_BASE = _variant(0, changed=False)


def test_one_subtree_change_is_a_small_delta():
    var = _variant(1)
    d = scene_delta(_BASE, var)
    assert len(d["added"]) + len(d["removed"]) < len(scene_components(var))


def test_reconstruction_is_exact():
    var = _variant(2)
    rebuilt = apply_scene_delta(scene_components(_BASE), scene_delta(_BASE, var))
    assert rebuilt == scene_components(var)


def test_identical_scene_gives_empty_delta():
    d = scene_delta(_BASE, _variant(0, changed=False))
    assert not d["added"] and not d["removed"]


def test_dedup_saving_above_one():
    scenes = [_BASE] + [_variant(i) for i in range(8)]
    assert scene_dedup_saving(scenes)["saving_x"] > 2.0


def test_dedup_accounting_consistent():
    scenes = [_BASE] + [_variant(i) for i in range(4)]
    sav = scene_dedup_saving(scenes)
    assert sav["naive"] >= sav["unique"] and sav["unique"] > 0


def test_variants_share_most_components():
    a, b = scene_components(_variant(0)), scene_components(_variant(1))
    assert len(a & b) >= len(a) - 3                        # differ in only a couple of components


def test_apply_delta_round_trip_on_added_and_removed():
    var = _variant(3)
    d = scene_delta(_BASE, var)
    assert d["added"] and d["removed"]                    # this variant both adds and removes
    assert apply_scene_delta(scene_components(_BASE), d) == scene_components(var)


def test_deterministic():
    var = _variant(1)
    assert scene_delta(_BASE, var) == scene_delta(_BASE, var)
    scenes = [_BASE] + [_variant(i) for i in range(4)]
    assert scene_dedup_saving(scenes) == scene_dedup_saving(scenes)

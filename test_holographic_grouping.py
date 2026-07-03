"""Modeling-app feature layer: grouping (a bundle) and instancing (a bind)."""
import numpy as np
from holographic_scene_doc import Scene
from holographic_grouping import (group_objects, ungroup, group_members, is_group, group_bundle,
                                  instance, instance_source, resolve_geometry, instances_of)
from holographic_ai import cosine


def _scene():
    s = Scene(dim=256, seed=0)
    a = s.add(name="wheel_l", geometry=np.zeros((4, 3)))
    b = s.add(name="wheel_r", geometry=np.ones((4, 3)))
    c = s.add(name="body", geometry=np.full((4, 3), 2.0))
    return s, a, b, c


def test_group_parents_members_one_undo():
    s, a, b, c = _scene()
    n = len(s._undo)
    g = group_objects(s, [a, b], name="wheels")
    assert set(group_members(s, g)) == {a, b} and is_group(s, g)
    assert s.parent_of(a) == g and len(s._undo) == n + 1     # ONE undo step


def test_group_bundle_recognizes_member():
    s, a, b, c = _scene()
    g = group_objects(s, [a, b])
    gb = group_bundle(s, g)
    assert cosine(gb, s.handle_vector(a)) > cosine(gb, s.handle_vector(c))


def test_ungroup_frees_members():
    s, a, b, c = _scene()
    g = group_objects(s, [a, b])
    ungroup(s, g)
    assert g not in s.objects and s.parent_of(a) is None and s.parent_of(b) is None


def test_instance_shares_geometry():
    s, a, b, c = _scene()
    T = np.eye(4); T[0, 3] = 5.0
    inst = instance(s, c, transform=T)
    assert instance_source(s, inst) == c
    assert np.allclose(resolve_geometry(s, inst), s.get(c).geometry)   # shared
    assert np.allclose(s.get(inst).transform, T)                       # own transform
    assert inst in instances_of(s, c)


def test_instance_follows_source_edit():
    s, a, b, c = _scene()
    inst = instance(s, c)
    s.edit(c, geometry=np.full((4, 3), 9.0))
    assert np.allclose(resolve_geometry(s, inst), 9.0)      # nothing copied -> instance follows

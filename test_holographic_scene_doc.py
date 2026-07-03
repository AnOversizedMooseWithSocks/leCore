"""Modeling-app backlog item 0: the canonical Scene document + stable handles (B) + change notification (E)."""
import numpy as np
from holographic_scene_doc import Scene, SceneObject, _content_hash


def test_add_returns_stable_handle_and_fires_event():
    s = Scene(dim=128, seed=0)
    events = []; s.on_change(lambda k, h: events.append((k, h)))
    a = s.add(name="wheel", geometry=np.zeros((4, 3)))
    assert a in s.objects and s.get(a).name == "wheel"
    assert ("add", a) in events


def test_handle_survives_edit_identity_vs_content():
    """The B guarantee: an edit changes the content hash but NOT the handle or its identity atom."""
    s = Scene(dim=128, seed=0)
    a = s.add(name="wheel", geometry=np.zeros((4, 3)))
    key0 = s._content_key[a]; id0 = s.handle_vector(a).copy()
    s.select([a])
    s.edit(a, geometry=np.full((4, 3), 9.0))
    assert s._content_key[a] != key0                 # content changed
    assert np.array_equal(s.handle_vector(a), id0)   # identity unchanged
    assert a in s.selection                          # selection still resolves


def test_change_events_for_all_mutations():
    s = Scene(dim=128, seed=0)
    seen = []; s.on_change(lambda k, h: seen.append(k))
    a = s.add(); s.edit(a, name="x"); s.select([a]); s.remove(a)
    for kind in ("add", "edit", "select", "remove"):
        assert kind in seen


def test_undo_redo_edit_preserves_identity():
    s = Scene(dim=128, seed=0)
    a = s.add(name="wheel"); id0 = s.handle_vector(a).copy()
    s.edit(a, name="front-wheel")
    assert s.undo() and s.get(a).name == "wheel"
    assert np.array_equal(s.handle_vector(a), id0)
    assert s.redo() and s.get(a).name == "front-wheel"


def test_undo_add_and_remove():
    s = Scene(dim=128, seed=0)
    a = s.add(name="a")
    s.undo(); assert a not in s.objects              # undo an add -> removed
    idc = None
    s.redo(); assert a in s.objects                  # redo -> back with same handle
    s.remove(a)
    s.undo(); assert a in s.objects and s.get(a).name == "a"   # undo a remove -> restored


def test_hierarchy_parenting():
    s = Scene(dim=128, seed=0)
    a = s.add(name="child"); b = s.add(name="parent")
    s.set_parent(a, b)
    assert a in s.children_of(b)


def test_content_hash_is_deterministic_and_geometry_sensitive():
    assert _content_hash(np.zeros((3, 3))) == _content_hash(np.zeros((3, 3)))
    assert _content_hash(np.zeros((3, 3))) != _content_hash(np.ones((3, 3)))


def test_deterministic_identity_atoms():
    s1 = Scene(dim=128, seed=7); a1 = s1.add()
    s2 = Scene(dim=128, seed=7); a2 = s2.add()
    assert np.array_equal(s1.handle_vector(a1), s2.handle_vector(a2))


def test_transaction_groups_into_one_undo():
    """A drag = one undo: many mutations inside a group coalesce into a single undo step."""
    s = Scene(dim=128, seed=0)
    a = s.add(name="a"); b = s.add(name="b"); c = s.add(name="c")
    with s.group("Move all"):
        s.edit(a, name="a2"); s.edit(b, name="b2"); s.edit(c, name="c2")
    assert s.history()[-1] == "Move all"                  # one labelled step for the whole batch
    s.undo()                                              # a single undo reverts all three
    assert s.get(a).name == "a" and s.get(b).name == "b" and s.get(c).name == "c"
    s.redo()
    assert s.get(a).name == "a2" and s.get(b).name == "b2" and s.get(c).name == "c2"


def test_history_labels():
    s = Scene(dim=128, seed=0)
    a = s.add(name="wheel")
    s.edit(a, name="wheel2")
    assert s.history() == ["Add wheel", "Edit wheel2"]    # the Edit menu / history panel content


def test_nested_groups_commit_once():
    s = Scene(dim=128, seed=0)
    a = s.add(name="a"); b = s.add(name="b")
    with s.group("Outer"):
        s.edit(a, name="a2")
        with s.group("Inner"):
            s.edit(b, name="b2")
    assert s.history()[-1] == "Outer"                     # nested -> ONE step, the outer label
    s.undo()
    assert s.get(a).name == "a" and s.get(b).name == "b"  # one undo reverts both


def test_depth_cap():
    s = Scene(dim=128, seed=0); s._max_undo = 5
    for i in range(20):
        s.add(name="o%d" % i)
    assert len(s._undo) == 5                              # only the most recent 5 steps kept


def test_can_undo_redo_and_redo_invalidation():
    s = Scene(dim=128, seed=0)
    assert not s.can_undo() and not s.can_redo()
    a = s.add(name="a")
    assert s.can_undo() and not s.can_redo()
    s.undo()
    assert s.can_redo()
    s.add(name="b")                                       # a fresh edit invalidates redo
    assert not s.can_redo()


def test_empty_group_records_nothing():
    s = Scene(dim=128, seed=0)
    s.add(name="a")
    before = len(s._undo)
    with s.group("Nothing"):
        pass
    assert len(s._undo) == before                        # an empty transaction adds no step

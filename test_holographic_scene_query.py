"""Modeling-app feature layer: selection, search, and tagging over the Scene document."""
from holographic_scene_doc import Scene
from holographic_scene_query import (select, select_fuzzy, select_by_tag, tag, untag, Selection)
from holographic_ai import cosine


def _scene():
    s = Scene(dim=512, seed=0)
    w1 = s.add(name="wheel_front", material="metal", tags={"kind": "wheel"})
    w2 = s.add(name="wheel_rear", material="metal", tags={"kind": "wheel"})
    body = s.add(name="body", material="paint", tags={"kind": "panel"})
    glass = s.add(name="windscreen", material="glass", tags={"kind": "panel"})
    return s, w1, w2, body, glass


def test_exact_select():
    s, w1, w2, body, glass = _scene()
    assert select(s, material="metal") == {w1, w2}
    assert select_by_tag(s, "kind", "panel") == {body, glass}
    assert select(s, name_contains="wheel") == {w1, w2}
    assert select(s, where=lambda o: o.material == "glass") == {glass}


def test_set_algebra():
    s, w1, w2, body, glass = _scene()
    metal = select(s, material="metal"); panels = select_by_tag(s, "kind", "panel")
    assert Selection.minus(metal, {w1}) == {w2}
    assert Selection.union(metal, panels) == {w1, w2, body, glass}
    assert Selection.intersect(select_by_tag(s, "kind", "wheel"), metal) == {w1, w2}
    assert Selection(s).invert(metal) == {body, glass}


def test_named_sets_and_apply():
    s, w1, w2, body, glass = _scene()
    sel = Selection(s)
    sel.save("drivetrain", {w1, w2})
    assert sel.get("drivetrain") == {w1, w2} and "drivetrain" in sel.names()
    sel.apply(sel.get("drivetrain"))
    assert s.selection == {w1, w2}


def test_fuzzy_select_carries_confidence():
    s, w1, w2, body, glass = _scene()
    hits = select_fuzzy(s, "material", "metal")
    assert {w1, w2} <= {h for h, c in hits}
    assert all(0.0 <= c <= 1.0 for _, c in hits)          # calibrated confidence surfaced


def test_tag_untag_through_document():
    s, w1, w2, body, glass = _scene()
    events = []; s.on_change(lambda k, h: events.append(k))
    tag(s, [body], "review", "pending")
    assert select_by_tag(s, "review") == {body} and "edit" in events
    s.undo()                                              # tagging was recorded -> undoable
    assert select_by_tag(s, "review") == set()
    untag(s, [w1], "kind")
    assert "kind" not in s.get(w1).tags


def test_selection_bundle_membership():
    s, w1, w2, body, glass = _scene()
    sel = Selection(s)
    b = sel.as_bundle({w1, w2})
    assert cosine(b, s.handle_vector(w1)) > cosine(b, s.handle_vector(glass))
    assert sel.as_bundle(set()) is None


def test_deterministic():
    s, w1, w2, body, glass = _scene()
    assert select(s, material="metal") == select(s, material="metal")

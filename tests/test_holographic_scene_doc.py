"""Modeling-app backlog item 0: the canonical Scene document + stable handles (B) + change notification (E)."""
import numpy as np
from holographic.scene_and_pipeline.holographic_scene_doc import Scene, SceneObject, _content_hash


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


# ---------------------------------------------------------------------------
# J-3D-15: scene_info -- the read side of the document.
# ---------------------------------------------------------------------------

def test_scene_info_through_the_mind_is_json_safe():
    """Cross-faculty, and JSON-safety is the load-bearing half. This crosses POST /invoke, where an
    np.float64 is not serialisable -- the exact 'works in-process, an agent cannot call it' split that this
    whole backlog exists to close. A test that only checked the values would pass on a broken surface."""
    import json
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    sc.add(name="ball", geometry=m.sdf_parse("(sphere 0.6)"), material="copper")
    info = m.scene_info(sc)
    json.dumps(info)                                   # raises if any numpy scalar leaked through
    o = info["objects"][0]
    assert type(o["scale"]) is float and type(o["handle"]) is str
    assert o["geometry"] == "sphere" and info["n_objects"] == 1 and info["empty"] is False


def test_empty_scene_says_so():
    """'Never assume the scene is empty' is the guidance that makes this call worth making at all, so the
    empty case is a first-class answer rather than an edge case."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    info = m.scene_info(m.new_scene())
    assert info["empty"] is True and info["n_objects"] == 0 and info["problems"] == []


def test_preflight_catches_the_bad_material_before_the_render_does():
    """The strongest single reason to call this. A material typo is accepted silently by scene.add and
    raises at RENDER time -- after the whole scene is built and a trace has been paid for. Here it costs
    milliseconds, and the 'did you mean' arrives while it is still cheap to act on."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    sc.add(name="cube", geometry=m.sdf_parse("(box 0.4 0.4 0.4)"), material="oak")
    problems = " | ".join(m.scene_info(sc)["problems"])
    assert "oak" in problems and "wood_oak" in problems


def test_dropped_rotation_is_reported_not_silent():
    """scene_to_render honours translation + uniform scale and DROPS rotation -- documented in its own
    docstring and invisible to a caller, so a rotated object renders unrotated and nothing says so. This is
    the one problem class with no downstream error at all; without this line it is undetectable.

    KEPT NEGATIVE: reporting is not fixing. J-3D-16 (full affine placement) is still open, and when it
    lands this assertion should be inverted rather than deleted."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    R = np.eye(4)
    R[0, 0] = R[2, 2] = np.cos(0.6); R[0, 2] = np.sin(0.6); R[2, 0] = -np.sin(0.6)
    sc.add(name="spun", geometry=m.sdf_parse("(torus 0.4 0.15)"), material="copper", transform=R)
    info = m.scene_info(sc)
    assert info["objects"][0]["rotated"] is True
    assert any("ROTATION" in p for p in info["problems"])


def test_a_clean_scene_reports_nothing():
    """A checker that always complains gets ignored, which makes it worse than no checker."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    sc.add(name="ball", geometry=m.sdf_parse("(sphere 0.6)"), material="copper")
    sc.add(name="floor", geometry=m.sdf_parse("(plane 0.0)"), material="matte_gray")
    assert m.scene_info(sc)["problems"] == []
    assert "What is in my scene" in str(m.find_capability("is my scene empty")[0])


# ---------------------------------------------------------------------------
# scene_set_texture: the JSON-safe door to the albedo socket (Blender-parity item).
# ---------------------------------------------------------------------------

def test_texture_by_name_changes_the_render_and_none_removes_it():
    """The round trip that matters: texture on -> image changes; texture None -> image restored EXACTLY.
    The remove path must be exact because the socket goes through set_override, and a remove that left
    residue would mean the override system leaked state into the record."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    h = sc.add(name="ball", geometry=m.sdf_parse("(sphere 0.6)"), material="matte_gray")
    sc.add(name="floor", geometry=m.sdf_parse("(plane -0.7)"), material="matte_gray")
    cam = m.camera(eye=(0.0, 0.6, 2.4), target=(0.0, 0.0, 0.0), fov_deg=40.0, aspect=4 / 3.)
    L = [m.scene_light("dome", intensity=1.4)]
    kw = dict(width=32, height=24, lights=L, seed=0)
    a = np.asarray(m.render_preview(sc, cam, **kw), float)
    m.scene_set_texture(sc, h, "checker", scale=2.5)
    b = np.asarray(m.render_preview(sc, cam, **kw), float)
    assert np.abs(a - b).mean() > 1e-3, "a named texture must actually change the render"
    m.scene_set_texture(sc, h, None)
    c = np.asarray(m.render_preview(sc, cam, **kw), float)
    assert np.array_equal(a, c), "removing the texture must restore the untextured render EXACTLY"


def test_an_image_texture_arrives_as_a_json_list():
    """THE reason the faculty exists: a callable cannot cross POST /invoke, so the JSON shapes -- a texture
    NAME, or a plain nested list image -- must be the complete interface. This feeds the image exactly as
    json.loads would deliver it."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    h = sc.add(name="floor", geometry=m.sdf_parse("(plane 0.0)"), material="matte_gray")
    img = [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]   # 2x2, pure JSON shape
    m.scene_set_texture(sc, h, img, scale=1.0)
    obj = sc.get(h)
    socket = obj.overrides["albedo_socket"]
    rgb = socket(np.array([[0.25, 0.0, 0.25], [0.75, 0.0, 0.25]]))
    assert rgb.shape == (2, 3) and float(rgb.max()) <= 1.0
    assert not np.allclose(rgb[0], rgb[1]), "two points half a tile apart must sample different texels"


def test_wrong_shapes_raise_with_directions():
    """A (H,W) grey array is a plausible mistake; the error must say what was expected AND point at the
    named-texture path, because 'wrong shape' without a route forward is a dead end for an agent."""
    import pytest
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    h = sc.add(name="b", geometry=m.sdf_parse("(sphere 0.5)"), material="matte_gray")
    with pytest.raises(ValueError, match="H,W,3"):
        m.scene_set_texture(sc, h, [[0.5, 0.5], [0.5, 0.5]])
    assert "Texture a scene object" in str(m.find_capability("wood grain texture on my object")[0])


# ---------------------------------------------------------------------------
# describe_to_scene: the join between the semantic scene and the Scene document.
# ---------------------------------------------------------------------------

def test_words_become_document_objects_with_handles():
    """The join's basic promise: text in, canonical Scene document out, one handle per grounded object --
    the handles being what every other parity faculty (texture, place, animate, info) operates on."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    r = m.describe_to_scene("a red cube on the left and a green sphere on the right")
    assert set(r["handles"]) == {"red box", "green sphere"}
    info = m.scene_info(r["scene"])
    assert info["n_objects"] == 2 and info["problems"] == []


def test_unknown_words_are_reported_not_swallowed():
    """'a purple wombat' quietly becoming an empty scene sends an agent debugging its camera. The parser's
    unknown list must surface through the bridge untouched."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    r = m.describe_to_scene("a purple wombat")
    assert "wombat" in r["unknown"]
    assert r["handles"] == {}, "an ungrounded description must not invent objects"


def test_described_objects_join_the_full_pipeline():
    """THE REASON THE BRIDGE EXISTS, and the defect it flushed out, both pinned in one test. A described
    object must accept a texture AND move under keyframes. The first version of this chain produced frames
    with mean delta 0.0000: realize_scene's SDFs are eval-only, and _place's hasattr guards silently
    SKIPPED the transform -- rendered fine, ignored every placement, said nothing. _PlacedEval now wraps
    eval-only geometry, so if this test ever reads 0.0 again, the silent-skip came back."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    r = m.describe_to_scene("a green sphere")
    sc, h = r["scene"], r["handles"]["green sphere"]
    m.scene_set_texture(sc, h, "checker", scale=1.5)
    cam = m.camera(eye=(0.0, 1.0, 3.5), target=(0.0, 0.0, 0.0), fov_deg=40.0, aspect=4 / 3.)
    frames = m.render_animation(sc, cam, {h: {"position": [[0.0, [0, 0, 0]], [1.0, [0, 1.0, 0]]]}},
                                n_frames=3, fps=3, width=32, height=24, seed=0)
    assert np.abs(frames[0] - frames[-1]).mean() > 1e-3, \
        "a described object ignored its keyframes -- the eval-only silent-skip regression"
    assert "Describe" in str(m.find_capability("turn a text description into scene document objects")[0]) \
        or "describe_to_scene" in str(m.find_capability("turn a text description into scene document objects")[:3])


def test_adding_into_an_existing_scene():
    """scene= must ADD, not replace -- describing furniture into a scene that already has a floor."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    sc.add(name="floor", geometry=m.sdf_parse("(plane 0.0)"), material="matte_gray")
    r = m.describe_to_scene("a blue cone", scene=sc)
    assert r["scene"] is sc
    assert m.scene_info(sc)["n_objects"] == 2

"""Tests for holographic_scene_semantic (describe -> build -> adjust named objects -> render/simulate)."""
import numpy as np
from holographic_scene_semantic import scene_from_description, SemanticScene, parse_adjust


def test_describe_builds_named_objects():
    sc = scene_from_description("a big red metal sphere and a small blue glass box on a sunny day")
    assert len(sc.objects) == 2
    assert sc.environment["sun"] == "bright"
    names = " ".join(sc.names())
    assert "sphere" in names and "box" in names


def test_parse_adjust_disambiguation():
    # a colour that a shape FOLLOWS is a selector; otherwise it's the change
    ref, ch, all_ = parse_adjust("change the red box to metal")
    assert ref == {"shape": "box", "color": "red"} and ch == {"material": "metal"} and not all_
    ref, ch, all_ = parse_adjust("make the sphere red")
    assert ref == {"shape": "sphere"} and ch == {"color": "red"}
    ref, ch, all_ = parse_adjust("make everything glass")
    assert ref == {} and ch == {"material": "glass"} and all_


def test_adjust_bigger_and_material():
    sc = scene_from_description("a red sphere and a blue box")
    sc.adjust("make the sphere bigger")
    assert sc.get({"shape": "sphere"})[0]["size"] == "large"
    sc.adjust("change the box to metal")
    assert sc.get({"shape": "box"})[0]["material"] == "metal"


def test_adjust_everything():
    sc = scene_from_description("a red sphere and a blue box")
    sc.adjust("make everything matte")
    assert all(o["material"] == "matte" for o in sc.objects)


def test_unknown_reference_is_a_noop():
    sc = scene_from_description("a red sphere and a blue box")
    before = [dict(o) for o in sc.objects]
    sc.adjust("make the pyramid golden")          # no pyramid, and 'pyramid' isn't a known shape
    assert [dict(o) for o in sc.objects] == before


def test_set_by_description():
    sc = scene_from_description("a red metal sphere and a blue box")
    sc.set("the red sphere", material="glass")
    assert sc.get({"shape": "sphere"})[0]["material"] == "glass"


def test_semantic_scene_from_existing_objects():
    sc = SemanticScene([{"shape": "sphere", "color": "red", "material": "metal", "size": "big"},
                        {"shape": "box", "color": "blue", "material": "glass", "size": "small"}])
    assert len(sc.names()) == 2
    sc.adjust("make everything gold")
    assert all(o["material"] == "gold" for o in sc.objects)


def test_simulate_settles_on_ground():
    sc = scene_from_description("a red sphere and a blue box")
    frames = sc.simulate(steps=25)
    assert len(frames) == 25
    y0 = list(frames[0].values())[0][1]
    y1 = list(frames[-1].values())[0][1]
    assert y1 < y0 and y1 > 0.0                   # fell under gravity, rests above the ground


def test_render_produces_image():
    sc = scene_from_description("a red metal sphere")
    img = np.asarray(sc.render(width=48, height=36, quality="fast"))
    assert img.shape == (36, 48, 3) and np.all(np.isfinite(img))


def test_adjust_leaves_helpful_feedback_on_unknown():
    sc = scene_from_description("a red sphere and a blue box")
    sc.adjust("make the pyramid golden")               # unknown target
    fb = sc.feedback
    assert not fb["applied"] and fb["suggestions"]
    assert any("pyramid" in s for s in fb["suggestions"])
    # scene unchanged
    assert sc.get({"shape": "sphere"})[0].get("material") in (None,)


def test_interpret_previews_without_applying():
    sc = scene_from_description("a red sphere")
    rep = sc.interpret("make the sphere bigger")
    assert rep["applied"] and rep["understood"]["changes"] == {"size": "large"}
    assert sc.get({"shape": "sphere"})[0].get("size") in (None,)   # interpret applied nothing


def test_adjust_resolves_synonyms():
    sc = scene_from_description("a blue box")
    rep = sc.interpret("make it crimson")
    assert rep["read_as"].get("crimson") == "color=red"
    sc.adjust("make it crimson")
    assert sc.objects[0]["color"] == "red"


def test_understood_target_but_not_change_suggests():
    sc = scene_from_description("a red sphere")
    rep = sc.interpret("do something wibbly to the sphere")
    assert not rep["applied"] and any("change" in s.lower() for s in rep["suggestions"])


def test_options_palette():
    sc = scene_from_description("a red sphere and a blue box")
    opt = sc.options()
    assert "sphere" in opt["shapes"] and "metal" in opt["materials"]
    assert "bigger" in opt["sizes"] and opt["objects"]


def test_description_with_no_objects_gives_help():
    sc = scene_from_description("a wibbly wobbly thing")
    assert sc.objects == []
    assert sc.feedback and any("object" in s.lower() for s in sc.feedback["suggestions"])


def test_build_does_its_best_with_synonyms():
    sc = scene_from_description("a shiny crimson orb")
    o = sc.objects[0]
    assert o["shape"] == "sphere" and o["color"] == "red" and o["material"] == "mirror"

"""Tests for holographic_scene_semantic (describe -> build -> adjust named objects -> render/simulate)."""
import numpy as np
from holographic.simulation_and_physics.holographic_scene_semantic import scene_from_description, SemanticScene, parse_adjust


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
    # NOTE: "gold" is now BOTH a colour and a material in the vocabulary, and colours are resolved first, so
    # "make everything gold" reads as colour=gold. "golden" is material-only -- the unambiguous way to ask for
    # the MATERIAL, which is what this test is about.
    sc.adjust("make everything golden")
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
    # "pyramid" is now a KNOWN shape (it resolves to cone), so an absent-but-known target is answered with a
    # QUESTION, not a suggestion. To test the UNKNOWN-WORD path we need a word that is genuinely out of vocabulary.
    sc.adjust("make the dodecahedron golden")          # unknown target
    fb = sc.feedback
    assert not fb["applied"] and fb["suggestions"]
    assert any("dodecahedron" in s for s in fb["suggestions"])
    # scene unchanged
    assert sc.get({"shape": "sphere"})[0].get("material") in (None,)


def test_interpret_previews_without_applying():
    sc = scene_from_description("a red sphere")
    rep = sc.interpret("make the sphere bigger")
    assert rep["applied"] and rep["understood"]["changes"] == {"size": "large"}
    assert sc.get({"shape": "sphere"})[0].get("size") in (None,)   # interpret applied nothing


def test_adjust_resolves_synonyms():
    sc = scene_from_description("a blue box")
    # "crimson" is now a first-class colour (it has its own RGB), so it is NOT resolved to red any more.
    # "scarlet" is still a synonym of red -- which is the synonym-resolution path this test is for.
    rep = sc.interpret("make it scarlet")
    assert rep["read_as"].get("scarlet") == "color=red"
    sc.adjust("make it scarlet")
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
    # "scarlet" (a synonym of red), not "crimson" -- crimson is now a colour in its own right.
    sc = scene_from_description("a shiny scarlet orb")
    o = sc.objects[0]
    assert o["shape"] == "sphere" and o["color"] == "red" and o["material"] == "mirror"


# ---- named objects + scene textures (naming / reference-by-nickname / paint / render routing) --------------
def _scene():
    from holographic.simulation_and_physics.holographic_scene_semantic import scene_from_description
    return scene_from_description("a big red metal sphere and a small blue box")


def test_name_and_reference_by_nickname():
    sc = _scene()
    sc.name("the red sphere", "hero")
    assert sc.labels()["hero"].endswith("sphere")
    assert "hero" in sc.names()                       # names() shows the nickname
    sc.adjust("make hero glass")                       # reference by nickname in a command
    assert sc.get("hero")[0]["material"] == "glass"
    # the nickname still resolves after the object's description changed
    assert sc.select("hero") == sc.select({"shape": "sphere"})


def test_rename_and_name_via_command():
    sc = _scene()
    sc.name("the sphere", "hero")
    sc.adjust("rename hero to champion")
    assert "champion" in sc.labels() and "hero" not in sc.labels()
    sc.adjust("call the box crate")
    assert "crate" in sc.labels()


def test_label_is_unique():
    sc = _scene()
    sc.name("the sphere", "thing")
    sc.name("the box", "thing")                        # reusing a label moves it
    assert sc.labels() == {"thing": sc.get({"shape": "box"})[0]["name"]}


def test_paint_named_texture_and_render_routes():
    import numpy as np
    sc = _scene()
    sc.adjust("give the sphere a rusty texture")
    assert sc.get({"shape": "sphere"})[0]["texture"] is not None
    sc.paint("the box", "marbled")
    assert sc.get({"shape": "box"})[0]["texture"] is not None
    img = np.asarray(sc.render(width=48, height=40))
    assert img.shape == (40, 48, 3) and img.std() > 0.02


def test_unknown_texture_is_refused_helpfully():
    sc = _scene()
    sc.paint("the sphere", "zebra")
    assert not sc.feedback["applied"] and sc.feedback["suggestions"]
    assert sc.get({"shape": "sphere"})[0]["texture"] is None


def test_named_texture_library():
    from holographic.simulation_and_physics.holographic_scene_semantic import named_texture, texture_names
    for name in texture_names():
        g = named_texture(name)
        assert g is not None and hasattr(g, "sample")
    assert named_texture("nonsense") is None


def test_make_object_texture_word_only():
    """'make the box mossy' (a texture word with no 'texture' trigger) still paints it."""
    sc = _scene()
    sc.adjust("make the box mossy")
    assert sc.get({"shape": "box"})[0]["texture"] is not None


# ---- external texture files on scene objects (asset library + resolve/relink through the scene) ------------
def _make_png(path, tint=(220, 60, 60)):
    import numpy as np
    from PIL import Image
    img = np.zeros((16, 16, 3), np.uint8)
    img[::4] = tint
    img[:, ::4] = tint[::-1]
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


def test_attach_external_texture_renders(tmp_path):
    import numpy as np
    from holographic.simulation_and_physics.holographic_scene_semantic import scene_from_description
    p = str(tmp_path / "project" / "textures" / "checker.png")
    _make_png(p)
    sc = scene_from_description("a big sphere")
    sc.attach_texture_file("the sphere", p)
    assert len(sc.missing_assets()) == 0
    img = np.asarray(sc.render(width=64, height=48))
    assert img.shape == (48, 64, 3) and img.std() > 0.02        # the image texture shows


def test_moved_file_falls_back_then_resolves(tmp_path):
    import os, shutil, numpy as np
    from holographic.simulation_and_physics.holographic_scene_semantic import scene_from_description
    p = str(tmp_path / "Documents" / "project" / "textures" / "checker.png")
    _make_png(p)
    sc = scene_from_description("a big sphere")
    sc.attach_texture_file("the sphere", p, with_hash=True)

    # move the project -> the file is missing, but render must NOT crash (falls back to colour)
    shutil.move(str(tmp_path / "Documents" / "project"), str(tmp_path / "Projects_project"))
    assert len(sc.missing_assets()) == 1
    img = np.asarray(sc.render(width=48, height=40))            # graceful, no exception
    assert img.shape == (40, 48, 3)

    # point the scene at the new root and resolve -> the file is re-found and the texture returns
    sc.set_asset_roots([str(tmp_path / "Projects_project")])
    sc.resolve_assets()
    assert len(sc.missing_assets()) == 0
    assert np.asarray(sc.render(width=48, height=40)).std() > 0.02


def test_relink_one_updates_the_scene(tmp_path):
    import shutil
    from holographic.simulation_and_physics.holographic_scene_semantic import scene_from_description
    a = str(tmp_path / "old" / "tex" / "a.png")
    b = str(tmp_path / "old" / "tex" / "b.png")
    _make_png(a); _make_png(b, tint=(60, 200, 60))
    sc = scene_from_description("a big sphere and a small box")
    sc.attach_texture_file("the sphere", a)
    sc.attach_texture_file("the box", b)
    shutil.move(str(tmp_path / "old"), str(tmp_path / "new"))
    assert len(sc.missing_assets()) == 2
    # relink ONE; the other is found by the shared moved-parent
    sc.relink(sc.assets.assets[0].path, str(tmp_path / "new" / "tex" / "a.png"))
    assert len(sc.missing_assets()) == 0


def test_check_assets_report(tmp_path):
    import os
    from holographic.simulation_and_physics.holographic_scene_semantic import scene_from_description
    p = str(tmp_path / "t" / "x.png")
    _make_png(p)
    sc = scene_from_description("a sphere")
    sc.attach_texture_file("the sphere", p)
    assert sc.check_assets()["counts"].get("ok") == 1
    os.remove(p)
    assert sc.check_assets()["counts"].get("missing") == 1


def test_procedural_scene_makes_no_asset_library():
    from holographic.simulation_and_physics.holographic_scene_semantic import scene_from_description
    sc = scene_from_description("a red sphere")
    sc.adjust("give the sphere a rusty texture")               # procedural, not a file
    assert sc._assets is None                                  # no AssetLibrary created for a purely procedural scene

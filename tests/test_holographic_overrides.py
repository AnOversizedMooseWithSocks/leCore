"""Modeling-app feature layer: render overrides -- a bound role with fallback."""
from holographic.scene_and_pipeline.holographic_scene_doc import Scene
from holographic.misc.holographic_overrides import set_override, clear_override, resolve, effective_settings, overridden_props

DEF = {"samples": 64, "denoise": "svgf", "visible": True}


def test_fallback_and_override():
    s = Scene(dim=128, seed=0)
    a = s.add(name="hero")
    b = s.add(name="prop", overrides={"samples": 256})
    assert resolve(s, a, "samples", DEF) == 64               # inherited default
    assert resolve(s, b, "samples", DEF) == 256              # overridden wins
    assert resolve(s, b, "denoise", DEF) == "svgf"           # partial: inherits the rest


def test_only_deltas_stored():
    s = Scene(dim=128, seed=0)
    a = s.add(name="a"); b = s.add(name="b", overrides={"samples": 256})
    assert overridden_props(s, a) == {} and overridden_props(s, b) == {"samples": 256}


def test_set_clear_undoable():
    s = Scene(dim=128, seed=0); a = s.add(name="a")
    set_override(s, a, "samples", 512); assert resolve(s, a, "samples", DEF) == 512
    clear_override(s, a, "samples"); assert resolve(s, a, "samples", DEF) == 64
    s.undo(); assert resolve(s, a, "samples", DEF) == 512    # the clear was recorded


def test_material_tier():
    s = Scene(dim=128, seed=0)
    c = s.add(name="glass", material={"denoise": "bilateral"})
    assert resolve(s, c, "denoise", DEF) == "bilateral"      # material beats scene default
    s.edit(c, overrides={"denoise": "off"})
    assert resolve(s, c, "denoise", DEF) == "off"            # object beats material


def test_effective_settings():
    s = Scene(dim=128, seed=0); b = s.add(name="b", overrides={"samples": 256})
    assert effective_settings(s, b, ["samples", "denoise", "visible"], DEF) == \
        {"samples": 256, "denoise": "svgf", "visible": True}

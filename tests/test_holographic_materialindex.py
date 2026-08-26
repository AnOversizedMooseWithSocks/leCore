"""Tests for holographic_materialindex (the bridge/discovery layer over the render + physical material libraries)."""
import numpy as np
import pytest
import holographic.materials_and_texture.holographic_materialindex as mi


def test_summary_counts_both_libraries():
    s = mi.summary()
    assert s["render_presets"] >= 100 and s["physical_materials"] >= 30 and s["in_both"] >= 10


def test_material_info_unified_view():
    gold = mi.material_info("gold")
    assert gold["in_render"] and gold["in_physical"]
    assert gold["render"]["class"] == "metal" and gold["render"]["metallic"] == 1.0
    assert gold["physical"]["density"] == 19300


def test_render_only_and_physical_only():
    assert mi.has_render("chrome") and not mi.has_physical("chrome")     # render preset, no physical def
    assert mi.has_physical("mercury") and not mi.has_render("mercury")   # physical only (science)
    assert mi.material_info("chrome")["render"]["metallic"] == 1.0
    assert mi.material_info("mercury")["physical"]["phase"] == "liquid"


def test_unknown_material_raises():
    with pytest.raises(KeyError):
        mi.material_info("unobtainium_xyz")


def test_render_material_is_pbrmaterial():
    from holographic.materials_and_texture.holographic_materialio import PBRMaterial
    assert isinstance(mi.render_material("copper"), PBRMaterial)


def test_physical_properties_for_scientist():
    assert abs(mi.physical_properties("water")["refractive"] - 1.333) < 1e-6
    assert mi.physical_properties("gold")["density"] == 19300
    with pytest.raises(KeyError):
        mi.physical_properties("chrome")                                 # no physical def


def test_find_across_both_libraries():
    assert any(r["name"] == "water" for r in mi.find_materials("water"))
    assert any(r["name"] == "diamond" for r in mi.find_materials("gem crystal"))
    metals = mi.find_materials("metal", k=40)
    assert any(r["name"] == "gold" for r in metals)


def test_all_materials_is_union():
    names = {r["name"] for r in mi.all_materials()}
    assert {"gold", "mercury", "chrome"} <= names                        # both-libs, physical-only, render-only


def test_render_material_feeds_cook_torrance():
    # the bridge's render material actually shades through the BRDF
    from holographic.rendering.holographic_brdf import cook_torrance
    m = mi.render_material("gold")
    base = np.array(m.base_color[:3])
    N = np.array([0, 0, 1.0]); V = np.array([0, 0, 1.0]); L = np.array([0.3, 0.0, 0.95]); L /= np.linalg.norm(L)
    rad = cook_torrance(N, V, L, base, m.metallic, m.roughness)
    assert rad.shape == (3,) and np.all(np.isfinite(rad)) and np.all(rad >= 0)


def test_physical_categories_and_units():
    cats = mi.physical_categories()
    assert "metal" in cats and "gas" in cats and "polymer" in cats
    assert len(mi.physical_by_category("metal")) >= 15
    units = mi.physical_units()
    assert units["density"][0] == "kg/m^3" and units["youngs"][0] == "GPa"


def test_material_info_carries_units():
    info = mi.material_info("gold")
    assert info["physical_units"]["density"] == "kg/m^3"
    assert info["physical_units"]["melting_point"] == "K"


def test_validate_physical_clean():
    assert mi.validate_physical() == []


def test_expanded_library_size():
    s = mi.summary()
    assert s["physical_materials"] >= 100 and len(s["physical_categories"]) >= 10

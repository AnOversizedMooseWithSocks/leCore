"""Tests for holographic_materialdata (the comprehensive physical material database)."""
import pytest
import holographic_materialdata as md


def test_comprehensive_and_categorized():
    assert len(md.PHYSICAL_MATERIALS) >= 85
    assert set(md.categories()) <= set(md.CATEGORIES)
    # every listed category is non-empty
    for cat in md.categories():
        assert md.by_category(cat), cat


def test_validate_is_clean():
    issues = md.validate()
    assert issues == [], issues[:5]


def test_every_entry_has_core_fields():
    for name, e in md.PHYSICAL_MATERIALS.items():
        assert "density" in e and e["density"] > 0, name
        assert e["phase"] in ("solid", "liquid", "gas"), name
        assert e["category"] in md.CATEGORIES, name


def test_units_cover_all_fields():
    used = set()
    for e in md.PHYSICAL_MATERIALS.values():
        used |= set(e)
    used -= {"category"}                                     # category is organizational, not a measured field
    assert used <= set(md.UNITS), used - set(md.UNITS)


def test_spot_check_real_values():
    assert md.PHYSICAL_MATERIALS["tungsten"]["melting_point"] == 3695     # highest-melting common metal
    assert md.PHYSICAL_MATERIALS["helium"]["density"] == 0.1786
    assert abs(md.PHYSICAL_MATERIALS["diamond"]["thermal_conductivity"] - 2200) < 1  # diamond conducts heat superbly
    assert md.PHYSICAL_MATERIALS["mercury"]["phase"] == "liquid"          # the liquid metal
    assert md.PHYSICAL_MATERIALS["balsa"]["density"] < md.PHYSICAL_MATERIALS["oak_wood"]["density"]


def test_field_coverage_reports_partial_honesty():
    cov = md.field_coverage()
    n = len(md.PHYSICAL_MATERIALS)
    assert cov["density"] == n and cov["phase"] == n and cov["category"] == n
    assert cov["refractive"] < n                             # only transparent media -- honest partial coverage


def test_add_material_and_category_guard():
    md.add_material("test_x", "metal", "solid", 5000, youngs=100.0)
    assert "test_x" in md.PHYSICAL_MATERIALS
    del md.PHYSICAL_MATERIALS["test_x"]
    with pytest.raises(ValueError):
        md.add_material("bad", "not_a_category", "solid", 100)


def test_merged_into_definitions_preserves_legacy():
    # the merge into holographic_definitions must NOT change legacy densities (resolve_scenario depends on them)
    from holographic_definitions import MATERIALS
    assert MATERIALS["wood"]["density"] == 650 and MATERIALS["steel"]["density"] == 7850
    assert MATERIALS["mercury"]["density"] == 13534
    # ... and legacy entries got ENRICHED with new fields
    assert "thermal_conductivity" in MATERIALS["gold"] and "category" in MATERIALS["gold"]
    # ... and new materials are present
    assert "tungsten" in MATERIALS and "polycarbonate" in MATERIALS and "helium" in MATERIALS

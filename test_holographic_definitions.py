"""Tests for holographic_definitions.py -- the definition library and scenario resolver.

The suite pins the three things that matter and keeps the two honest negatives visible:
  * the physical CONSTANTS are the real reference values, and the buoyancy PREDICATES match a
    hand-checked truth table (steel and lead float on mercury, gold sinks -- real density facts);
  * the VSA encoding recalls CATEGORICAL properties exactly and the scenario's single-slot facts
    (relation, medium) clean up to the right symbol;
  * (kept negative) the DENSITY scalar decodes only approximately from the bundled record, and the
    multi-entity scenario set degrades -- both asserted as bounded, not exact.
"""
import math
import numpy as np
import pytest

from holographic_definitions import (
    build_standard_library, resolve_scenario, MATERIALS, GEOMETRY,
    rel_float, rel_sink, rel_rise, _density_class,
)
from holographic_core import unbind


# ---- fixtures -------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lib():
    return build_standard_library(dim=1024, seed=0)


# ---- the library exists and is populated across every category -----------------------------------

def test_library_categories(lib):
    assert len(lib) >= 80
    for kind in ("material", "geometry", "phenomenon", "texture", "operator", "signal"):
        assert len(lib.by_kind(kind)) > 0, "every promised category is populated"
    assert len(lib.by_kind("material")) >= 30


def test_real_physical_constants(lib):
    # the constants are the actual textbook values, not placeholders
    assert lib.get("water").props["density"] == 1000
    assert lib.get("water").props["viscosity"] == 0.001
    assert lib.get("air").props["density"] == pytest.approx(1.225)
    assert lib.get("steel").props["density"] == 7850
    assert lib.get("gold").props["density"] == 19300
    assert lib.get("mercury").props["density"] == 13534
    assert lib.get("steel").props["youngs"] == 200.0
    assert lib.get("water").props["refractive"] == pytest.approx(1.333)


# ---- the VSA encoding: categorical recall is exact -----------------------------------------------

def test_categorical_recall_exact(lib):
    for name in lib.by_kind("material"):
        kind, kc = lib.query_property(name, "KIND")
        assert kind == "material" and kc > 0.3
        if "phase" in lib.get(name).props:
            phase, pc = lib.query_property(name, "PHASE")
            assert phase == lib.get(name).props["phase"], "phase must decode exactly"


def test_density_class_query(lib):
    # a holographic query -- 'find the heavy things' -- via the density-class role
    heavy = [n for n in lib.by_kind("material")
             if lib.query_property(n, "DENSITY_CLASS")[0] in ("heavy", "very_heavy")]
    assert {"steel", "iron", "copper", "lead", "gold", "mercury"} <= set(heavy)
    assert "wood" not in heavy and "air" not in heavy


def test_density_scalar_decode_is_approximate(lib):
    # KEPT NEGATIVE: the density scalar rides in a bundle with the categorical roles, so unbinding it
    # and decoding gives an APPROXIMATE density (grid + crosstalk). Bounded, not exact; the exact value
    # lives in .props. Assert the mean log10 error is small-ish but nonzero (worse than categorical).
    errs = []
    for n in lib.by_kind("material"):
        true_log = math.log10(lib.get(n).props["density"])
        dec_log = math.log10(max(lib.query_density(n), 1e-6))
        errs.append(abs(true_log - dec_log))
    mean_err = float(np.mean(errs))
    assert 0.0 < mean_err < 0.35, "density decode is approximate (kept negative), within ~factor 2"


def test_similarity_carries_physical_meaning(lib):
    # steel's nearest material neighbours should be other dense metals, not a gas
    from holographic_core import cosine
    steel = lib.get("steel").vector
    assert cosine(steel, lib.get("iron").vector) > cosine(steel, lib.get("helium").vector)
    assert cosine(steel, lib.get("aluminum").vector) > cosine(steel, lib.get("water").vector)
    neighbours = [n for n, _ in lib.similar("steel", k=4, kind="material")]
    assert "iron" in neighbours  # steel and iron have near-identical density -> near-identical record


# ---- the physics predicates match a hand-checked truth table (real density facts) ----------------

@pytest.mark.parametrize("obj,medium,floats", [
    ("wood", "water", True),      # ~650 < 1000
    ("oak_wood", "water", True),  # 700 < 1000
    ("ice", "water", True),       # 917 < 1000 -- ice floats (the classic)
    ("steel", "water", False),    # 7850 > 1000
    ("concrete", "water", False), # 2400 > 1000
    ("vegetable_oil", "water", True),  # 920 < 1000 -- oil floats on water
    ("steel", "mercury", True),   # 7850 < 13534 -- steel FLOATS on mercury
    ("lead", "mercury", True),    # 11340 < 13534 -- lead floats on mercury too
    ("gold", "mercury", False),   # 19300 > 13534 -- gold sinks in mercury
])
def test_buoyancy_truth_table(lib, obj, medium, floats):
    holds, _, _ = rel_float(lib.get(obj).props, lib.get(medium).props)
    assert holds == floats


def test_helium_rises_in_air(lib):
    holds, why, _ = rel_rise(lib.get("helium").props, lib.get("air").props)
    assert holds and "rises" in why


def test_submerged_fraction_is_density_ratio(lib):
    holds, _, d = rel_float(lib.get("wood").props, lib.get("water").props, obj_volume=0.002)
    assert holds
    assert d["submerged_fraction"] == pytest.approx(650 / 1000)
    # at equilibrium a floating body's buoyant force equals its weight
    assert d["buoyant_force_N"] == pytest.approx(d["weight_N"], rel=1e-6)


def test_sink_reports_net_force(lib):
    holds, _, d = rel_sink(lib.get("steel").props, lib.get("water").props, obj_volume=0.001)
    assert holds
    assert d["net_downward_force_N"] == pytest.approx((7850 - 1000) * 0.001 * 9.81)


# ---- the resolver: description -> validated scenario ---------------------------------------------

def test_resolve_wood_floats(lib):
    sc = resolve_scenario("a block of wood floating in water", lib=lib)
    assert sc.understood and sc.consistent
    assert sc.entities[0]["shape"] == "box" and sc.entities[0]["material"] == "wood"
    assert sc.medium["name"] == "water" and sc.relation == "float"
    assert sc.results[0]["derived"]["submerged_fraction"] == pytest.approx(0.65)


def test_resolve_steel_flagged_inconsistent(lib):
    sc = resolve_scenario("a steel ball floating in water", lib=lib)
    assert sc.understood            # it parsed fine
    assert not sc.consistent        # but the physics says no: steel sinks
    assert "sinks" in sc.results[0]["why"]


def test_resolve_material_shape_order(lib):
    # "<material> <shape>" and "<shape> of <material>" both parse
    a = resolve_scenario("a wooden cube sinking in water", lib=lib)
    assert a.entities[0]["shape"] == "box" and a.entities[0]["material"] == "wood"
    b = resolve_scenario("a sphere of steel resting on the table", lib=lib)
    assert b.entities[0]["shape"] == "sphere" and b.entities[0]["material"] == "steel"
    assert b.relation == "rest_on" and b.medium["role"] == "surface"


def test_resolve_multiple_entities(lib):
    sc = resolve_scenario("a cork and a steel ball floating in water", lib=lib)
    assert len(sc.entities) == 2
    mats = {e["material"] for e in sc.entities}
    assert mats == {"cork", "steel"}
    verdicts = {r["entity_id"]: r["holds"] for r in sc.results}
    # cork floats, steel does not -- the scenario is only 'consistent' if ALL hold
    assert list(verdicts.values()).count(True) == 1
    assert not sc.consistent


def test_resolve_unknown_word_is_loud_not_guessed(lib):
    sc = resolve_scenario("a blorptangle of quffium floating in water", lib=lib)
    assert not sc.understood
    assert sc.unresolved, "unknown words are reported, not silently guessed"


def test_build_spec_names_solver(lib):
    sc = resolve_scenario("a block of wood floating in water", lib=lib)
    spec = sc.build_spec()
    assert spec["phenomenon"] == "buoyancy"
    assert spec["bodies"][0]["material"] == "wood"
    assert spec["medium"]["name"] == "water"
    assert spec["gravity_m_s2"] == pytest.approx(9.81)


# ---- the scenario as a VSA structure: single-slot recall is clean --------------------------------

def test_scenario_vector_recall(lib):
    sc = resolve_scenario("a block of wood floating in water", lib=lib)
    v = sc.to_vector(lib)
    rel, _ = lib.names.cleanup(unbind(v, lib.roles.get("RELATION")))
    med, _ = lib.names.cleanup(unbind(v, lib.roles.get("MEDIUM")))
    assert rel == "float" and med == "water", "single-slot facts recall to the right symbol"


# ---- geometry formulas are correct ---------------------------------------------------------------

def test_geometry_volume_formulas():
    sph = GEOMETRY["sphere"]["volume"](radius=1.0)
    assert sph == pytest.approx(4.0 / 3.0 * math.pi)
    box = GEOMETRY["box"]["volume"](lx=2.0, ly=3.0, lz=4.0)
    assert box == pytest.approx(24.0)
    cyl = GEOMETRY["cylinder"]["volume"](radius=1.0, height=2.0)
    assert cyl == pytest.approx(2.0 * math.pi)


# ---- the operator vocabulary ties calculus to the engine's own ops -------------------------------

def test_convolution_is_binding(lib):
    # the gem: convolution and correlation ARE the core bind/unbind
    from holographic_definitions import OPERATORS
    assert "bind" in OPERATORS["convolution"]["realized_by"]
    assert "unbind" in OPERATORS["correlation"]["realized_by"]


# ---- determinism ---------------------------------------------------------------------------------

def test_determinism():
    a = build_standard_library(dim=512, seed=7)
    b = build_standard_library(dim=512, seed=7)
    for name in ("water", "steel", "gold"):
        assert np.array_equal(a.get(name).vector, b.get(name).vector)
    va = resolve_scenario("a block of wood floating in water", lib=a).to_vector(a)
    vb = resolve_scenario("a block of wood floating in water", lib=b).to_vector(b)
    assert np.array_equal(va, vb)


def test_density_class_boundaries():
    assert _density_class(50) == "ultralight"
    assert _density_class(650) == "light"
    assert _density_class(2400) == "medium"
    assert _density_class(7850) == "heavy"
    assert _density_class(19300) == "very_heavy"

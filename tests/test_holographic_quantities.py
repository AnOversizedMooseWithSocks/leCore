"""Tests for holographic_quantities.py -- the dimensional grammar and the composition recipes.

Pins the three claims: (1) units convert and compose with the dimension tracked exactly; (2) an illegal
operation (adding unlike dimensions) is refused, not silently coerced; (3) a bill of materials composes
into mass / cost / embodied carbon by reusing the density ingredients from holographic_definitions.
The kept negative -- that the cost/carbon FACTORS are sample placeholders -- is a data statement, not a
machinery one, so the tests check the machinery and treat the numbers as illustrative.
"""
import math
import pytest

from holographic.misc.holographic_quantities import Quantity, unit, register_unit, _dim, _dim_str, BASE, body_mass, bill_mass, bill_cost, bill_embodied_carbon, SAMPLE_PRICE_USD_PER_KG, SAMPLE_CARBON_KG_PER_KG
from holographic.misc.holographic_definitions import build_standard_library


@pytest.fixture(scope="module")
def lib():
    return build_standard_library(dim=256, seed=0)


# ---- conversion + dimensional composition --------------------------------------------------------

def test_unit_conversion_within_and_across_systems():
    assert Quantity(1.0, "km").to("m") == pytest.approx(1000.0)
    assert Quantity(1.0, "m3").to("L") == pytest.approx(1000.0)
    assert Quantity(1.0, "kg").to("g") == pytest.approx(1000.0)
    assert Quantity(1.0, "ft").to("m") == pytest.approx(0.3048)   # imperial -> SI
    assert Quantity(1.0, "t").to("kg") == pytest.approx(1000.0)


def test_temperature_is_affine():
    # 0 degC == 273.15 K, and the offset is handled on conversion (not in products)
    assert Quantity(0.0, "degC").to("K") == pytest.approx(273.15)
    assert Quantity(100.0, "degC").to("K") == pytest.approx(373.15)


def test_products_compose_the_dimension():
    accel = Quantity(9.81, "m/s") / Quantity(1.0, "s")           # m/s^2
    force = Quantity(10.0, "kg") * accel
    assert force.to("N") == pytest.approx(98.1)
    assert force.dim == _dim(mass=1, length=1, time=-2)
    # density * volume = mass
    m = Quantity(1000.0, "kg/m3") * Quantity(0.002, "m3")
    assert m.dim == _dim(mass=1) and m.to("kg") == pytest.approx(2.0)
    # price_per_kg * mass = money
    cost = Quantity(1.0, "USD/kg") * Quantity(2.0, "kg")
    assert cost.dim == _dim(currency=1) and cost.to("USD") == pytest.approx(2.0)


def test_adding_unlike_dimensions_is_refused():
    with pytest.raises(ValueError):
        _ = Quantity(1.0, "kg") + Quantity(1.0, "m")
    with pytest.raises(ValueError):
        _ = Quantity(1.0, "USD") + Quantity(1.0, "kg")
    # converting across dimensions is also refused
    with pytest.raises(ValueError):
        Quantity(1.0, "kg").to("m")


def test_unknown_unit_is_loud():
    with pytest.raises(KeyError):
        unit("furlong_per_fortnight")


def test_uncertainty_propagates_in_quadrature():
    d = Quantity(1000.0, "kg/m3", uncertainty=10.0)   # 1% relative
    v = Quantity(0.002, "m3", uncertainty=0.00002)    # 1% relative
    m = d * v
    val, unc = m.value_unc("kg")
    assert val == pytest.approx(2.0)
    # product of two 1% relatives -> ~sqrt(2)% relative
    assert unc / val == pytest.approx(math.sqrt(2) * 0.01, rel=1e-3)


def test_register_unit_is_additive():
    register_unit("stone", _dim(mass=1), 6.35029318)
    assert Quantity(1.0, "stone").to("kg") == pytest.approx(6.35029318)


# ---- the recipes: a bill of materials composes to mass / cost / carbon ---------------------------

def test_body_mass_reuses_density_ingredient(lib):
    # 18 m^3 of concrete at 2400 kg/m^3 = 43.2 tonnes -- density comes from the definition library
    m = body_mass(lib, "concrete", 18.0)
    assert m.to("t") == pytest.approx(43.2)
    # the density INGREDIENT carries its provenance (a product of two sourced quantities drops the single
    # source, honestly -- so we check the leaf, which is where provenance lives)
    from holographic.misc.holographic_quantities import quantity_from_definition
    rho = quantity_from_definition(lib, "concrete", "density", "kg/m3")
    assert rho.source == "holographic_definitions.MATERIALS"


def test_bill_mass_totals_and_flags_unknowns(lib):
    bill = [("concrete", 18.0), ("wood", 12.0), ("steel", 0.6), ("glass", 0.4)]
    total, unknown = bill_mass(lib, bill)
    assert unknown == []
    # 43200 + 7800 + 4710 + 1000 kg
    assert total.to("kg") == pytest.approx(43200 + 7800 + 4710 + 1000)
    # an unknown material is reported, not silently dropped
    total2, unknown2 = bill_mass(lib, [("adamantium", 1.0), ("wood", 1.0)])
    assert unknown2 == ["adamantium"] and total2.to("kg") == pytest.approx(650.0)


def test_house_cost_and_carbon_compose(lib):
    bill = [("concrete", 18.0), ("wood", 12.0), ("steel", 0.6), ("glass", 0.4)]
    cost, miss_c = bill_cost(lib, bill, SAMPLE_PRICE_USD_PER_KG)
    carbon, miss_k = bill_embodied_carbon(lib, bill, SAMPLE_CARBON_KG_PER_KG)
    assert miss_c == [] and miss_k == []
    # cost is money-dimensioned, carbon is mass-dimensioned (kgCO2e)
    assert cost.dim == _dim(currency=1)
    assert carbon.dim == _dim(mass=1)
    # hand-check one term: 43200 kg concrete * $0.10/kg = $4320 of the total
    assert cost.to("USD") > 4320.0
    # carbon total is positive and dominated by steel+concrete
    assert carbon.to("kgCO2e") > 0


def test_missing_price_is_reported_not_zeroed(lib):
    bill = [("titanium", 0.1)]   # no sample price for titanium
    cost, missing = bill_cost(lib, bill, SAMPLE_PRICE_USD_PER_KG)
    assert missing == ["titanium"] and cost.to("USD") == pytest.approx(0.0)


def test_base_dimensions_include_currency():
    # currency is the extra base axis the cost recipe needs (not SI, no FX conversion)
    assert "currency" in BASE and "mass" in BASE
    assert _dim_str(_dim(mass=1, length=-3)) == "mass * length^-3"

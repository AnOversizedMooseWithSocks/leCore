"""holographic_quantities.py -- the GRAMMAR OF QUANTITIES: a value is a sentence in a dimensional language.

WHY THIS MODULE EXISTS (Moose's point, made mechanical)
------------------------------------------------------
The external science databases -- Materials Project, the Particle Data Group, BioNumbers, CODATA/IAU,
USGS commodity prices, the ICE embodied-carbon inventory -- are the INGREDIENTS. The composability thesis
says we should REFERENCE them and COMPOSE derived facts from RECIPES, not duplicate them. But composition
across sources only works if every number carries its UNIT and its DIMENSION, so that

    density * volume            -> a mass          (M/L^3 * L^3 = M)
    mass * gravity              -> a force         (M * L/T^2 = M L/T^2 = newtons)
    price_per_kg * mass         -> a cost          ($/M * M = $)
    carbon_factor * mass        -> embodied carbon (kgCO2e)

are CHECKED, not hoped for -- and adding a length to a mass is caught at the '+' as the grammar error it
is. This module is that grammar, built FROM SCRATCH. The constraint is the point: we implement the small
dimensional algebra ourselves (the QUDT / UCUM / astropy-Quantity model boiled down to an exponent vector
plus an SI multiplier) rather than vendoring a units library, because the engine's rule is NumPy + stdlib,
and because the algebra is tiny once you see it.

THE 'AS ABOVE, SO BELOW' PARALLEL
---------------------------------
A quantity decomposes into base dimensions exactly the way a VSA structure decomposes into role-filler
atoms. Density's dimension is the exponent vector (mass=1, length=-3); that vector IS the quantity's
grammatical parse. Multiplication ADDS exponent vectors (bind-like); a legal addition requires EQUAL
exponent vectors (a cleanup-like identity check); conversion between units is a single multiply. So the
same decompose/compose discipline the rest of holostuff runs on quantities too.

PROVENANCE + UNCERTAINTY RIDE ALONG
-----------------------------------
The astropy-Constant and BioNumbers lesson, kept: a number without a SOURCE and a RANGE is not yet a
scientific number. A Quantity carries an absolute uncertainty and a source string; uncertainty propagates
(first-order: quadrature through multiply/divide, linear through add/subtract).

HONEST SCOPE (kept loud)
------------------------
  * CURRENCY is modelled as an extra base dimension with NO exchange-rate conversion -- USD only. FX is
    out of scope; mixing currencies raises, it does not silently convert.
  * The demo cost and carbon FACTORS are SAMPLE placeholders (order-of-magnitude, clearly labelled), NOT
    the real USGS / ICE numbers -- the sandbox cannot reach those APIs (network-restricted), so the
    INGEST adapters are an interface (see holostuff_scientific_databases_backlog.md), and only the
    composition MACHINERY is exercised here. The grammar and the recipe are real and tested; the specific
    dollars and kilograms-of-CO2 are illustrative until a real ingest.
  * Affine units (degrees Celsius) can be CONVERTED but not multiplied -- an affine offset has no place in
    a product. Arithmetic requires offset == 0 (convert to kelvin first).
  * Uncertainty propagation is first-order and assumes INDEPENDENT inputs (no covariance).

Pure NumPy + stdlib; deterministic.
"""

import math

# The base dimensions, in fixed order. Seven SI base quantities + currency (money), which is not SI but
# is exactly the extra axis the cost recipe needs. An exponent vector over these eight IS a "dimension".
BASE = ("mass", "length", "time", "current", "temperature", "amount", "luminous", "currency")
_N = len(BASE)
_ZERO = (0,) * _N


def _dim(**kw):
    """Build a dimension exponent tuple from keyword powers, e.g. _dim(mass=1, length=-3) for density."""
    return tuple(kw.get(name, 0) for name in BASE)


def _dim_add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _dim_sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dim_scale(a, k):
    return tuple(x * k for x in a)


def _dim_str(d):
    """Human-readable dimension, e.g. 'mass * length^-3'. '1' for dimensionless."""
    parts = []
    for name, p in zip(BASE, d):
        if p == 0:
            continue
        parts.append(name if p == 1 else "%s^%g" % (name, p))
    return " * ".join(parts) if parts else "1"


class Unit:
    """A unit of measure: a symbol, the dimension it measures (an exponent vector over BASE), the factor
    that converts a value in this unit to SI base units, and an optional affine offset (for Celsius).

        value_in_SI = value * factor + offset

    Example: kilometre = Unit('km', _dim(length=1), 1000.0); gram = Unit('g', _dim(mass=1), 1e-3)."""

    def __init__(self, symbol, dim, factor=1.0, offset=0.0):
        self.symbol = symbol
        self.dim = dim
        self.factor = float(factor)
        self.offset = float(offset)

    def __repr__(self):
        return "Unit(%r, %s)" % (self.symbol, _dim_str(self.dim))


# ------------------------------------------------------------------------------------------------------
# The unit registry. Small on purpose -- the base SI units, the derived units the physics needs, a few
# imperial units to prove conversion, and the money units the cost recipe needs. Extend by REGISTERING,
# never by editing an existing entry (backward-compatible, the engine's rule).
# ------------------------------------------------------------------------------------------------------
REGISTRY = {}


def register_unit(symbol, dim, factor=1.0, offset=0.0):
    REGISTRY[symbol] = Unit(symbol, dim, factor, offset)
    return REGISTRY[symbol]


def unit(symbol):
    """Look a unit up by symbol; raise a clear error naming what is missing (never guess)."""
    if symbol not in REGISTRY:
        raise KeyError("unknown unit %r -- register it or use a known one (%s ...)"
                       % (symbol, ", ".join(sorted(list(REGISTRY)[:8]))))
    return REGISTRY[symbol]


# --- dimensionless ---
register_unit("", _ZERO, 1.0)
register_unit("1", _ZERO, 1.0)
# --- mass ---
register_unit("kg", _dim(mass=1), 1.0)
register_unit("g",  _dim(mass=1), 1e-3)
register_unit("mg", _dim(mass=1), 1e-6)
register_unit("t",  _dim(mass=1), 1e3)            # tonne (metric)
register_unit("lb", _dim(mass=1), 0.45359237)     # pound (imperial) -- proves cross-system conversion
# --- length / area / volume ---
register_unit("m",  _dim(length=1), 1.0)
register_unit("cm", _dim(length=1), 1e-2)
register_unit("mm", _dim(length=1), 1e-3)
register_unit("km", _dim(length=1), 1e3)
register_unit("in", _dim(length=1), 0.0254)
register_unit("ft", _dim(length=1), 0.3048)
register_unit("m2", _dim(length=2), 1.0)
register_unit("m3", _dim(length=3), 1.0)
register_unit("L",  _dim(length=3), 1e-3)         # litre
register_unit("cm3", _dim(length=3), 1e-6)
# --- time ---
register_unit("s",   _dim(time=1), 1.0)
register_unit("min", _dim(time=1), 60.0)
register_unit("hr",  _dim(time=1), 3600.0)
register_unit("day", _dim(time=1), 86400.0)
# --- velocity ---
register_unit("m/s", _dim(length=1, time=-1), 1.0)
# --- force / pressure (stiffness) ---
register_unit("N",   _dim(mass=1, length=1, time=-2), 1.0)
register_unit("Pa",  _dim(mass=1, length=-1, time=-2), 1.0)
register_unit("kPa", _dim(mass=1, length=-1, time=-2), 1e3)
register_unit("MPa", _dim(mass=1, length=-1, time=-2), 1e6)
register_unit("GPa", _dim(mass=1, length=-1, time=-2), 1e9)
# --- energy / power ---
register_unit("J",  _dim(mass=1, length=2, time=-2), 1.0)
register_unit("kJ", _dim(mass=1, length=2, time=-2), 1e3)
register_unit("MJ", _dim(mass=1, length=2, time=-2), 1e6)
register_unit("W",  _dim(mass=1, length=2, time=-3), 1.0)
# --- temperature (K is base; degC is affine) ---
register_unit("K",    _dim(temperature=1), 1.0)
register_unit("degC", _dim(temperature=1), 1.0, offset=273.15)
# --- material intensive quantities ---
register_unit("kg/m3", _dim(mass=1, length=-3), 1.0)          # density
register_unit("g/cm3", _dim(mass=1, length=-3), 1000.0)       # density (common in datasets) -> kg/m3
register_unit("Pa*s",  _dim(mass=1, length=-1, time=-1), 1.0) # dynamic viscosity
register_unit("J/(kg K)", _dim(length=2, time=-2, temperature=-1), 1.0)   # specific heat capacity
register_unit("W/(m K)",  _dim(mass=1, length=1, time=-3, temperature=-1), 1.0)  # thermal conductivity
register_unit("N/m",   _dim(mass=1, time=-2), 1.0)            # surface tension
# --- money + the intensive cost/carbon factors the recipes consume ---
register_unit("USD",       _dim(currency=1), 1.0)
register_unit("USD/kg",    _dim(currency=1, mass=-1), 1.0)
register_unit("USD/t",     _dim(currency=1, mass=-1), 1e-3)   # $/tonne (USGS unit value) -> $/kg SI
register_unit("kgCO2e",    _dim(mass=1), 1.0)                 # CO2-equivalent is a mass
register_unit("kgCO2e/kg", _dim(), 1.0)                       # embodied-carbon factor: a mass ratio


class Quantity:
    """A number with a unit, an uncertainty, and a source -- the atom the recipes compose.

    Arithmetic works in SI internally and tracks the dimension exactly:
        * / multiply/divide the SI values and add/subtract the dimension exponent vectors;
        + - require the SAME dimension and add/subtract the SI values.
    Uncertainty propagates first-order: relative-quadrature through * and /, absolute-quadrature through
    + and -. The result's unit is the coherent SI unit for its dimension (call .to(sym) to re-express)."""

    def __init__(self, value, unit_sym, uncertainty=0.0, source=None):
        u = unit(unit_sym) if isinstance(unit_sym, str) else unit_sym
        # store canonically in SI base units so composition never has to think about the display unit
        if u.offset and (value is not None):
            self.si = value * u.factor + u.offset
            self.si_unc = abs(uncertainty) * abs(u.factor)
        else:
            self.si = value * u.factor
            self.si_unc = abs(uncertainty) * abs(u.factor)
        self.dim = u.dim
        self.source = source

    # -- construction from an already-SI value (used internally by arithmetic) -------------------------
    @classmethod
    def _from_si(cls, si_value, dim, si_unc=0.0, source=None):
        q = cls.__new__(cls)
        q.si = si_value
        q.si_unc = abs(si_unc)
        q.dim = dim
        q.source = source
        return q

    # -- read-out --------------------------------------------------------------------------------------
    def to(self, unit_sym):
        """Return the plain float value expressed in `unit_sym`. Raises if the dimension does not match
        (converting a mass to a length is a grammar error, refused loudly)."""
        u = unit(unit_sym) if isinstance(unit_sym, str) else unit_sym
        if u.dim != self.dim:
            raise ValueError("dimension mismatch: this quantity is [%s], cannot express in %r [%s]"
                             % (_dim_str(self.dim), getattr(u, "symbol", u), _dim_str(u.dim)))
        return (self.si - u.offset) / u.factor

    def value_unc(self, unit_sym):
        """(value, uncertainty) in the requested unit."""
        u = unit(unit_sym) if isinstance(unit_sym, str) else unit_sym
        return self.to(unit_sym), self.si_unc / abs(u.factor)

    # -- dimensional arithmetic ------------------------------------------------------------------------
    def _check_addable(self, other):
        if self.dim != other.dim:
            raise ValueError("cannot add/subtract [%s] and [%s] -- different dimensions (grammar error)"
                             % (_dim_str(self.dim), _dim_str(other.dim)))

    def __add__(self, other):
        self._check_addable(other)
        unc = math.hypot(self.si_unc, other.si_unc)
        return Quantity._from_si(self.si + other.si, self.dim, unc)

    def __sub__(self, other):
        self._check_addable(other)
        unc = math.hypot(self.si_unc, other.si_unc)
        return Quantity._from_si(self.si - other.si, self.dim, unc)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quantity._from_si(self.si * other, self.dim, self.si_unc * abs(other), self.source)
        val = self.si * other.si
        # relative uncertainties add in quadrature for a product
        rel = math.hypot(self._rel(), other._rel())
        return Quantity._from_si(val, _dim_add(self.dim, other.dim), abs(val) * rel)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Quantity._from_si(self.si / other, self.dim, self.si_unc / abs(other), self.source)
        val = self.si / other.si
        rel = math.hypot(self._rel(), other._rel())
        return Quantity._from_si(val, _dim_sub(self.dim, other.dim), abs(val) * rel)

    def _rel(self):
        return (self.si_unc / abs(self.si)) if self.si else 0.0

    def is_dimensionless(self):
        return self.dim == _ZERO

    def __repr__(self):
        # pick a readable unit: prefer an exact registry match for this dimension
        sym = None
        for s, u in REGISTRY.items():
            if u.dim == self.dim and u.factor == 1.0 and u.offset == 0.0 and s not in ("", "1"):
                sym = s
                break
        if sym is None:
            return "Quantity(%g SI [%s])" % (self.si, _dim_str(self.dim))
        v = self.to(sym)
        tail = ""
        if self.si_unc:
            tail = " +/- %.3g" % (self.si_unc / REGISTRY[sym].factor)
        src = (" [%s]" % self.source) if self.source else ""
        return "%.6g %s%s%s" % (v, sym, tail, src)


# ======================================================================================================
# RECIPES: derived quantities composed from base ingredients + a rule. A recipe is a small parse tree
# over Quantities -- the same compose discipline as a StructureRecipe, one domain up. These three answer
# Moose's "render the building cost/carbon of a house": give a bill of materials, get mass, cost, carbon.
# ======================================================================================================

def quantity_from_definition(lib, name, prop, unit_sym):
    """Lift a raw number out of the definition library into a dimensioned Quantity -- the bridge from the
    ingredient tables (holographic_definitions.MATERIALS) to the grammar. Reuses the density that already
    lives there; does not duplicate it. Returns None if the thing or the property is unknown (loud)."""
    d = lib.get(name)
    if d is None or prop not in d.props:
        return None
    return Quantity(d.props[prop], unit_sym, source="holographic_definitions.MATERIALS")


def body_mass(lib, material, volume_m3):
    """mass = density * volume -- the first recipe, and the one every other one leans on. Returns a
    Quantity in kg (dimension mass), or None if the material's density is unknown."""
    rho = quantity_from_definition(lib, material, "density", "kg/m3")
    if rho is None:
        return None
    return rho * Quantity(volume_m3, "m3")


def bill_mass(lib, bill):
    """Total mass of a bill of materials: [(material, volume_m3), ...] -> Quantity(kg). Unknown materials
    are skipped and returned in a second list (loud, not silently dropped)."""
    total = Quantity(0.0, "kg")
    unknown = []
    for material, vol in bill:
        m = body_mass(lib, material, vol)
        if m is None:
            unknown.append(material)
        else:
            total = total + m
    return total, unknown


def bill_cost(lib, bill, price_per_kg):
    """Total cost = sum over the bill of (mass * price_per_kg[material]). price_per_kg maps material ->
    a Quantity in USD/kg (or USD/t). Returns Quantity(USD) plus the list of materials with no price."""
    total = Quantity(0.0, "USD")
    missing = []
    for material, vol in bill:
        m = body_mass(lib, material, vol)
        p = price_per_kg.get(material)
        if m is None or p is None:
            missing.append(material)
            continue
        total = total + (m * p)                     # (kg) * (USD/kg) = USD, checked by the grammar
    return total, missing


def bill_embodied_carbon(lib, bill, carbon_factor):
    """Total embodied carbon = sum of (mass * carbon_factor[material]). carbon_factor maps material -> a
    Quantity in kgCO2e/kg (a dimensionless mass ratio). Returns Quantity(kgCO2e) + missing list."""
    total = Quantity(0.0, "kg")                     # CO2e is a mass
    missing = []
    for material, vol in bill:
        m = body_mass(lib, material, vol)
        f = carbon_factor.get(material)
        if m is None or f is None:
            missing.append(material)
            continue
        total = total + (m * f)                     # (kg) * (dimensionless) = kg (of CO2e)
    return total, missing


# ------------------------------------------------------------------------------------------------------
# SAMPLE cost / carbon factors -- ORDER-OF-MAGNITUDE PLACEHOLDERS, clearly labelled, NOT the real USGS /
# ICE data. They exist only to exercise the recipe machinery end to end. Replace via a real ingest (see
# the backlog doc); the grammar guarantees the composition is correct whatever the numbers are.
# ------------------------------------------------------------------------------------------------------
SAMPLE_PRICE_USD_PER_KG = {   # rough construction-material unit costs, USD/kg (SAMPLE)
    "concrete": Quantity(0.10, "USD/kg", source="SAMPLE (pending USGS/RSMeans ingest)"),
    "steel":    Quantity(1.00, "USD/kg", source="SAMPLE (pending USGS ingest)"),
    "wood":     Quantity(0.50, "USD/kg", source="SAMPLE"),
    "glass":    Quantity(1.50, "USD/kg", source="SAMPLE"),
    "aluminum": Quantity(2.50, "USD/kg", source="SAMPLE"),
    "copper":   Quantity(9.00, "USD/kg", source="SAMPLE (cf. USGS ~$9,000/t)"),
}
SAMPLE_CARBON_KG_PER_KG = {   # rough embodied-carbon factors, kgCO2e/kg (SAMPLE, cf. ICE ranges)
    "concrete": Quantity(0.13, "kgCO2e/kg", source="SAMPLE (cf. ICE ~0.10-0.16)"),
    "steel":    Quantity(1.55, "kgCO2e/kg", source="SAMPLE (cf. ICE ~1.4-2.9)"),
    "wood":     Quantity(0.45, "kgCO2e/kg", source="SAMPLE (cf. ICE ~0.3-0.5)"),
    "glass":    Quantity(0.85, "kgCO2e/kg", source="SAMPLE"),
    "aluminum": Quantity(9.00, "kgCO2e/kg", source="SAMPLE (cf. ICE ~9 virgin)"),
}


# ======================================================================================================
# SELF TEST
# ======================================================================================================

def _selftest():
    # --- the grammar: conversion, dimensional products, and the caught grammar error -----------------
    length = Quantity(1.0, "km")
    assert abs(length.to("m") - 1000.0) < 1e-9
    assert abs(length.to("ft") - 3280.8398950131) < 1e-6      # cross-system conversion
    print("conversion: 1 km = %.1f m = %.1f ft" % (length.to("m"), length.to("ft")))

    # mass * gravity = force (newtons), dimensions composed, not asserted
    g = Quantity(9.81, "m/s")                                   # (we only need m/s here; see note below)
    # build acceleration properly as m / s^2 via division
    accel = Quantity(9.81, "m/s") / Quantity(1.0, "s")         # m/s^2
    force = Quantity(10.0, "kg") * accel
    assert abs(force.to("N") - 98.1) < 1e-6
    print("F = m*g: 10 kg * 9.81 m/s^2 = %.2f N (dimension %s)" % (force.to("N"), _dim_str(force.dim)))

    # the grammar error: adding a length to a mass is refused
    try:
        _ = Quantity(1.0, "kg") + Quantity(1.0, "m")
        raise AssertionError("should have refused to add mass + length")
    except ValueError as e:
        print("grammar error caught: %s" % str(e)[:70])

    # uncertainty propagates (the astropy/BioNumbers lesson)
    d = Quantity(1000.0, "kg/m3", uncertainty=10.0)            # water density +/- 10
    v = Quantity(0.002, "m3", uncertainty=0.0001)
    m = d * v
    mv, mu = m.value_unc("kg")
    print("mass with uncertainty: %.3f +/- %.3f kg (relative %.1f%%)" % (mv, mu, 100 * mu / mv))
    assert mu > 0

    # --- the recipe: render the mass, cost and carbon of a small house -------------------------------
    from holographic_definitions import build_standard_library
    lib = build_standard_library(dim=256, seed=0)             # small dim: we only use the density table
    # a toy bill of materials (material, volume in m^3)
    house = [("concrete", 18.0),   # foundation + floor slab
             ("wood", 12.0),       # framing
             ("steel", 0.6),       # rebar + fasteners
             ("glass", 0.4)]       # windows
    mass, unknown_m = bill_mass(lib, house)
    cost, miss_c = bill_cost(lib, house, SAMPLE_PRICE_USD_PER_KG)
    carbon, miss_k = bill_embodied_carbon(lib, house, SAMPLE_CARBON_KG_PER_KG)
    print("\nHOUSE 'render' from a bill of materials (densities reused from the definition library):")
    for material, vol in house:
        bm = body_mass(lib, material, vol)
        print("  %-9s %5.1f m3 -> %8.0f kg" % (material, vol, bm.to("kg")))
    print("  TOTAL MASS   : %.1f t"  % mass.to("t"))
    print("  TOTAL COST   : $%.0f  (SAMPLE factors -- pending USGS/RSMeans ingest)" % cost.to("USD"))
    print("  EMBODIED CO2 : %.0f kgCO2e  (SAMPLE factors -- pending ICE ingest)" % carbon.to("kgCO2e"))
    assert unknown_m == [] and miss_c == [] and miss_k == []
    assert mass.dim == _dim(mass=1) and cost.dim == _dim(currency=1)
    assert carbon.dim == _dim(mass=1)
    # sanity: concrete dominates the mass (18 m3 * 2400 kg/m3 = 43.2 t of the total)
    assert abs(body_mass(lib, "concrete", 18.0).to("t") - 43.2) < 1e-6

    print("\nOK: holographic_quantities self-test passed")


if __name__ == "__main__":
    _selftest()

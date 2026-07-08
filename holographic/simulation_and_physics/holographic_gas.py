"""holographic_gas.py -- T1: the GAS STATE. Pressure, volume, temperature and density tied by the ideal gas law.

WHY THIS EXISTS (thermodynamics foundation, item T1)
----------------------------------------------------
Gases connect pressure, temperature and density, and one downstream fact matters most for the process layer:
a liquid's BOILING POINT depends on pressure. Water boils at 100 degrees C at sea level but cooler up a
mountain, because the boiling point tracks the ambient pressure (Clausius-Clapeyron). So M5 (phase change)
needs a gas-state model to know WHEN water turns to steam. This module supplies the ideal gas law, adiabatic
compression/expansion, the speed of sound, and the boiling-point-vs-pressure curve.

THE PHYSICS (readable, first-principles)
----------------------------------------
  * IDEAL GAS LAW, mass form:  P V = m R_specific T, where R_specific = R / M (R = 8.314 J/mol/K, M = molar
    mass kg/mol). Equivalently density rho = P / (R_specific T). Using the SPECIFIC gas constant keeps everything
    in mass units the rest of the engine speaks -- no moles needed.
  * ADIABATIC process (no heat exchanged): P V^gamma = const and T V^(gamma-1) = const, gamma the adiabatic
    index. This is why a compressed gas heats up (a bike pump, a diesel cylinder) and an expanding one cools.
  * SPEED OF SOUND:  a = sqrt(gamma R_specific T). A satisfying cross-check: this module DERIVES ~343 m/s for
    air at 20 C from gamma and molar mass alone -- and the material definitions independently list air's
    sound_speed as 343. Two roads, same number.
  * BOILING POINT vs PRESSURE (Clausius-Clapeyron):  ln(P2/P1) = -(L/R_specific)(1/T2 - 1/T1). Anchored at a
    known point (water: 373.15 K at 101325 Pa, latent heat L), it gives the boiling temperature at any pressure.

HONEST SCOPE (kept negative): the IDEAL gas law (no Van der Waals corrections -- fine well away from
condensation), constant gamma and constant latent heat over the range used (Clausius-Clapeyron's standard
approximation), and no humidity/mixture chemistry. Correct and sufficient for driving phase change and buoyant
plumes; not a real-gas equation of state. NumPy + stdlib; deterministic. Molar mass / gamma are the new gas
data columns; density is reused from the material definitions where a cross-check is wanted.
"""
import numpy as np

_R = 8.314462618          # universal gas constant, J/mol/K (CODATA)

# The new gas data columns: molar mass (kg/mol) and adiabatic index gamma (dimensionless). Real values;
# monatomic gamma=5/3, diatomic ~7/5, triatomic lower. These belong in the definition data layer long-term.
GAS_PROPERTIES = {
    "air":            {"molar_mass": 0.028964, "gamma": 1.40},   # ~78% N2 + 21% O2
    "nitrogen":       {"molar_mass": 0.028013, "gamma": 1.40},   # diatomic
    "oxygen":         {"molar_mass": 0.031998, "gamma": 1.40},   # diatomic
    "carbon_dioxide": {"molar_mass": 0.044010, "gamma": 1.289},  # triatomic (linear)
    "helium":         {"molar_mass": 0.004003, "gamma": 1.667},  # monatomic (5/3)
    "hydrogen":       {"molar_mass": 0.002016, "gamma": 1.405},  # diatomic
    "water_vapor":    {"molar_mass": 0.018015, "gamma": 1.33},   # steam
}


def molar_mass_of(name):
    """Molar mass (kg/mol) of a gas. Prefers the GAS_PROPERTIES table (fast path), but if a gas is not listed it
    is DERIVED from the material's elemental composition (holographic_elements) -- so a gas the engine knows the
    formula of gets its molar mass from its makeup, not a hand-set number. This is the composition grammar feeding
    the gas law: molar mass = sum(count * atomic_mass)."""
    if name in GAS_PROPERTIES:
        return GAS_PROPERTIES[name]["molar_mass"]
    try:
        from holographic.simulation_and_physics.holographic_elements import material_elemental
        me = material_elemental(name)
        if me is not None:
            return me["molar_mass"] / 1000.0                        # g/mol -> kg/mol
    except Exception:
        pass
    raise KeyError("unknown gas %r and no elemental composition to derive its molar mass" % name)


def specific_gas_constant(name):
    """R_specific = R / M (J/kg/K) for a named gas. 287 for air, 2077 for helium (light gas -> big R_specific).
    Molar mass comes from the table or, failing that, from the elemental composition (molar_mass_of)."""
    return _R / molar_mass_of(name)


def gas_density(pressure_Pa, temp_K, name="air"):
    """rho = P / (R_specific T): the density (kg/m^3) of a gas at a given pressure and temperature."""
    return float(pressure_Pa) / (specific_gas_constant(name) * float(temp_K))


def gas_pressure(density, temp_K, name="air"):
    """P = rho R_specific T: the pressure (Pa) of a gas at a given density and temperature."""
    return float(density) * specific_gas_constant(name) * float(temp_K)


def speed_of_sound(temp_K, name="air"):
    """a = sqrt(gamma R_specific T) (m/s). ~343 for air at 293 K -- matches the definitions' sound_speed."""
    g = GAS_PROPERTIES[name]["gamma"]
    return float(np.sqrt(g * specific_gas_constant(name) * float(temp_K)))


def adiabatic(p1_Pa, t1_K, volume_ratio, name="air"):
    """Compress or expand a gas with NO heat exchange by `volume_ratio` = V2/V1 (<1 compress, >1 expand). Returns
    (P2, T2) from P V^gamma = const and T V^(gamma-1) = const. Compressing heats the gas; expanding cools it."""
    g = GAS_PROPERTIES[name]["gamma"]
    p2 = float(p1_Pa) * volume_ratio ** (-g)
    t2 = float(t1_K) * volume_ratio ** (-(g - 1.0))
    return p2, t2


# Water's vaporization anchor for Clausius-Clapeyron (the common case M5 needs). L is the latent heat of
# vaporization (J/kg); the reference point is the normal boiling point at one standard atmosphere.
_WATER_L_VAPOR = 2.257e6        # J/kg
_WATER_BOIL_REF_T = 373.15      # K (100 C)
_WATER_BOIL_REF_P = 101325.0    # Pa (1 atm)


def boiling_point(pressure_Pa, latent_heat=_WATER_L_VAPOR, molar_mass=0.018015,
                  ref_T=_WATER_BOIL_REF_T, ref_P=_WATER_BOIL_REF_P):
    """The boiling temperature (K) at a given pressure, from Clausius-Clapeyron anchored at (ref_T, ref_P).
    Defaults are water. Lower pressure -> lower boiling point (why water boils cooler up a mountain), which is
    the fact M5 (phase change) consumes. `latent_heat` J/kg, `molar_mass` kg/mol."""
    R_spec = _R / float(molar_mass)
    # ln(P/ref_P) = -(L/R_spec)(1/T - 1/ref_T)  ->  solve for T
    inv_T = 1.0 / float(ref_T) - (R_spec / float(latent_heat)) * np.log(float(pressure_Pa) / float(ref_P))
    return 1.0 / inv_T


class IdealGas:
    """A parcel of gas in a definite state (pressure, temperature) -- query its density and speed of sound, or
    push it through an adiabatic volume change. The simplest useful gas object; M5 reads its boiling-point curve
    and a buoyant plume reads its density-vs-temperature."""

    def __init__(self, name="air", temp_K=293.15, pressure_Pa=101325.0):
        self.name = name
        self.T = float(temp_K)
        self.P = float(pressure_Pa)

    def density(self):
        """Current density (kg/m^3) from the ideal gas law."""
        return gas_density(self.P, self.T, self.name)

    def sound_speed(self):
        """Current speed of sound (m/s)."""
        return speed_of_sound(self.T, self.name)

    def adiabatic_change(self, volume_ratio):
        """Change volume with no heat exchange; updates P and T in place, returns (P, T)."""
        self.P, self.T = adiabatic(self.P, self.T, volume_ratio, self.name)
        return self.P, self.T


def _selftest():
    """The gas law reproduces air's density and speed of sound (cross-checked against the material definitions),
    adiabatic compression heats the gas, and the boiling point falls with pressure."""
    # (1) air density at 20 C, 1 atm ~ 1.204 kg/m^3 (definitions list ~1.225 at 15 C -- consistent)
    rho = gas_density(101325.0, 293.15, "air")
    assert abs(rho - 1.204) < 0.02, rho

    # (2) speed of sound in air at 20 C ~ 343 m/s -- and the definitions independently say 343. Two roads, one number.
    a = speed_of_sound(293.15, "air")
    assert abs(a - 343.0) < 3.0, a
    from holographic.misc.holographic_definitions import MATERIALS
    assert abs(a - MATERIALS["air"]["sound_speed"]) < 5.0          # derived vs tabulated agree

    # helium (light, monatomic) carries sound much faster than air
    assert speed_of_sound(293.15, "helium") > 2.5 * a

    # (3) adiabatic compression to half volume HEATS the gas and raises pressure (a diesel/bike-pump)
    p2, t2 = adiabatic(101325.0, 293.15, volume_ratio=0.5, name="air")
    assert t2 > 293.15 and p2 > 101325.0
    # expansion cools it
    _, t3 = adiabatic(101325.0, 293.15, volume_ratio=2.0, name="air")
    assert t3 < 293.15

    # (4) boiling point: 100 C at 1 atm, and LOWER at altitude (70 kPa ~ 3000 m) -- the fact M5 needs
    assert abs(boiling_point(101325.0) - 373.15) < 0.5
    high_alt = boiling_point(70000.0)
    assert high_alt < 373.15 and high_alt > 355.0                  # ~90 C on a mountain
    # higher pressure (a pressure cooker, ~2 atm) raises it above 100 C
    assert boiling_point(2 * 101325.0) > 373.15

    # (5) deterministic
    assert speed_of_sound(300.0, "air") == speed_of_sound(300.0, "air")
    print("holographic_gas selftest OK: air rho=%.3f kg/m3, a=%.0f m/s (matches definitions 343); adiabatic "
          "compression heats; water boils %.0f C @1atm, %.0f C @70kPa" %
          (rho, a, boiling_point(101325.0) - 273.15, boiling_point(70000.0) - 273.15))


if __name__ == "__main__":
    _selftest()

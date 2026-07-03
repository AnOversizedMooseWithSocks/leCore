"""holographic_acoustic.py -- A2: acoustic IMPEDANCE and what sound does at a boundary (reflect / transmit / absorb).

WHY THIS EXISTS (Acoustics & Cymatics backlog, item A2)
------------------------------------------------------
The engine's reflection code is all OPTICAL (Fresnel/BRDF for light). Sound needs its own version, and the data
for it is already on every material: the characteristic acoustic impedance is Z = rho * c (density times speed
of sound), both of which the definitions already carry (the speed of sound we even cross-checked against the gas
law). From Z, everything follows -- this is the acoustic twin of the light BRDF:

  * at a boundary between media, the pressure reflection coefficient is r = (Z2 - Z1) / (Z2 + Z1); the fraction
    of ENERGY reflected is R = r^2 and transmitted is T = 1 - R (energy conserved at a lossless interface). A big
    impedance MISMATCH (air <-> water, air <-> steel) reflects almost everything -- which is why you hear so
    little of the world above when your ears are underwater.
  * a SURFACE also ABSORBS some sound (turns it to heat in the material); the absorption coefficient alpha (0 =
    perfect reflector, 1 = anechoic) is the one new data column, and the reflected fraction off a wall is 1 - alpha.

These feed geometric room acoustics (A6): rays reflect with (1 - alpha) and lose energy to alpha, building a
room's reverberation.

HONEST SCOPE (kept negative): normal-incidence, single-frequency (real) impedance to start -- angle- and
frequency-dependent absorption is a later refinement. Interface R/T is for a plane wave hitting a flat boundary
head-on. Deterministic; NumPy + stdlib. Impedance reuses the definitions' density x sound_speed (not restated);
absorption is the one added column.
"""
import numpy as np


# The one new data column: the Sabine absorption coefficient alpha per material (fraction of incident sound
# ENERGY absorbed at a surface, ~mid-frequency). 0 = perfect reflector (hard, shiny), 1 = anechoic. Reference
# values used in room-acoustics practice. Extensible; belongs in the material definition data layer long-term.
ABSORPTION = {
    "concrete": 0.02, "granite": 0.02, "marble": 0.01, "glass": 0.03, "steel": 0.02, "iron": 0.02,
    "brick": 0.03, "ceramic": 0.02, "water": 0.01, "ice": 0.02,
    "wood": 0.10, "oak_wood": 0.09, "pine_wood": 0.11, "plaster": 0.05,
    "carpet": 0.30, "cork": 0.25, "rubber": 0.20,
    "curtain": 0.50, "foam": 0.70, "acoustic_foam": 0.85, "fiberglass": 0.90,
    None: 0.05,                                                    # a mild default for un-tabled materials
}


def _rho_c(material):
    """Density (kg/m3) and speed of sound (m/s) of a material, reused from the definition library. Falls back to a
    gas-model speed of sound for named gases, and to air-ish defaults if truly unknown."""
    from holographic_definitions import MATERIALS
    if material in MATERIALS:
        p = MATERIALS[material]
        rho = float(p["density"])
        c = float(p.get("sound_speed", 0.0))
        if c <= 0.0:                                                # no tabulated c: derive it for a gas, else guess
            try:
                from holographic_gas import speed_of_sound
                c = speed_of_sound(293.15, material)
            except Exception:
                c = 1000.0
        return rho, c
    # not a known material -- assume air
    return 1.225, 343.0


def impedance(material):
    """Characteristic acoustic impedance Z = rho * c (Pa*s/m, i.e. the rayl). Air ~415, water ~1.48e6, steel
    ~4.7e7 -- the huge spread is why sound barely crosses an air/solid boundary. Reused from the material data."""
    rho, c = _rho_c(material)
    return rho * c


def interface(mat_a, mat_b):
    """Sound going from medium A into medium B at a flat boundary: returns (R, T) = the fractions of ENERGY
    reflected and transmitted. r = (Zb - Za)/(Zb + Za); R = r^2; T = 1 - R. A large mismatch -> R near 1."""
    za, zb = impedance(mat_a), impedance(mat_b)
    r = (zb - za) / (zb + za)
    R = float(r * r)
    return R, float(1.0 - R)


def wall_absorption(material):
    """The absorption coefficient alpha of a surface material (0 reflector .. 1 anechoic). The reflected fraction
    off that wall is 1 - alpha -- what a sound ray keeps after a bounce (feeds room acoustics A6)."""
    return float(ABSORPTION.get(material, ABSORPTION[None]))


def reflect_absorb(material):
    """A wall's split of incident sound ENERGY into (reflected, absorbed) = (1 - alpha, alpha). Conserved: sums to
    1. This is the acoustic analog of a surface's diffuse reflectance vs. what it soaks up."""
    a = wall_absorption(material)
    return 1.0 - a, a


def _selftest():
    """Impedances have the right huge spread; a big mismatch reflects almost everything; energy is conserved;
    absorption is a sane split. Deterministic."""
    # (1) impedance ordering + rough magnitudes (rho*c): air << water << steel
    za, zw, zs = impedance("air"), impedance("water"), impedance("steel")
    assert za < zw < zs
    assert 380 < za < 460                                          # air ~415 rayl
    assert 1.3e6 < zw < 1.6e6                                      # water ~1.48e6
    assert zs > 3e7                                                # steel ~4.7e7

    # (2) air<->water and air<->steel reflect almost all the sound (textbook near-total reflection)
    R_aw, T_aw = interface("air", "water")
    R_as, T_as = interface("air", "steel")
    assert R_aw > 0.99 and R_as > 0.999                            # you hear almost nothing across the boundary
    assert abs(R_aw + T_aw - 1.0) < 1e-12 and abs(R_as + T_as - 1.0) < 1e-12   # energy conserved

    # (3) a small mismatch (water <-> ice, similar Z) transmits a lot more than air<->water
    R_wi, T_wi = interface("water", "ice")
    assert T_wi > T_aw                                             # closer impedances -> more gets through

    # (4) absorption: a hard wall reflects almost all, acoustic foam soaks most up; each split sums to 1
    hard_r, hard_a = reflect_absorb("concrete")
    soft_r, soft_a = reflect_absorb("acoustic_foam")
    assert hard_r > 0.95 and soft_a > 0.8
    assert abs(hard_r + hard_a - 1.0) < 1e-12 and abs(soft_r + soft_a - 1.0) < 1e-12

    # (5) deterministic
    assert interface("air", "steel") == interface("air", "steel")
    print("holographic_acoustic selftest OK: Z=rho*c spans air %.0f -> water %.2e -> steel %.2e; air/solid "
          "reflects >99%% (energy conserved); foam absorbs >80%%" % (za, zw, zs))


if __name__ == "__main__":
    _selftest()

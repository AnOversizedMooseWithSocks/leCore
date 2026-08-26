"""holographic_burn.py -- M7: object BURN / DECAY over time. A log goes wood -> char -> ash and loses its mass.

WHY THIS EXISTS (Material Structure & Process backlog, item M7, the last one)
-----------------------------------------------------------------------------
M6 gave a fire its flame, smoke and heat; this closes the loop at the OBJECT level: a burning object is CONSUMED
over time. Its mass drops as fuel turns to smoke and ash, and its appearance darkens from the pristine material to
CHAR (blackened) and finally to grey ASH -- the visible history of a fire. This ties the pieces together: it
drives an M6 Fire for the combustion, tracks the mass the fire eats, and uses the blend/interpolation capability
to move the surface colour base -> char -> ash as the burn fraction climbs. (The same object-consumption idea
covers evaporation -- a puddle heated past boiling loses mass to vapour via M5 -- exposed here as a `decay` mode.)

THE MODEL (readable)
--------------------
A `BurningObject` holds an initial mass and an M6 `Fire`. Each `step(dt)`:
  * advance the Fire -> it reports fuel burned this step, whether it is lit, the smoke it made, the flame colour;
  * the remaining mass IS the remaining fuel, so burn_fraction = 1 - mass/initial_mass climbs from 0 to 1;
  * the appearance is a two-stage blend: pristine base -> char over the first stretch, then char -> ash near the
    end -- so an object looks scorched before it looks like ash.
When the fuel is spent the object is ASH (burn_fraction ~ 1), cold, and emits no more smoke. Monotonic: an object
does not un-burn.

HONEST SCOPE (kept negative): object-level bookkeeping (mass, burn fraction, appearance) driving the M6 fire --
NOT a resolved char-layer pyrolysis model or shrinking-geometry simulation (the mesh/SDF does not physically
shrink here; the appearance and mass do). Ash is an end colour, not separate ash particles. Deterministic;
NumPy + stdlib. Char/ash colours are art-directable data.
"""
import numpy as np


# char (blackened) and ash (pale grey) end colours; base comes from the material's own albedo (reused).
_CHAR = np.array([0.09, 0.07, 0.06])          # scorched black-brown
_ASH = np.array([0.62, 0.60, 0.58])           # pale grey ash
_CHAR_UNTIL = 0.7                              # burn fraction at which the surface is fully charred (then -> ash)


def _base_albedo(material):
    """Pristine colour of the material (from the matlib catalog if present, else a wood-ish default)."""
    try:
        import holographic.materials_and_texture.holographic_matlib as _ml
        if material in _ml.RENDER_MATERIALS:
            return _ml.albedo(material)
    except Exception:
        pass
    return np.array([0.55, 0.40, 0.24])


def char_color(material, burn_fraction):
    """The surface colour at a burn fraction 0..1: pristine base -> char (blackened) over the first stretch, then
    char -> ash (grey) as it finishes. Two blends chained -- the interpolation capability applied to burning."""
    bf = float(np.clip(burn_fraction, 0.0, 1.0))
    base = _base_albedo(material)
    if bf <= _CHAR_UNTIL:
        t = bf / _CHAR_UNTIL                                        # base -> char
        return (1.0 - t) * base + t * _CHAR
    t = (bf - _CHAR_UNTIL) / (1.0 - _CHAR_UNTIL)                    # char -> ash
    return (1.0 - t) * _CHAR + t * _ASH


class BurningObject:
    """An object being consumed by fire. Drives an M6 Fire, tracks mass loss and the base->char->ash appearance,
    and reports the smoke/flame each step. `light()` ignites it; `step(dt)` advances the burn."""

    def __init__(self, material, mass_kg, temp_K=293.15):
        from holographic.simulation_and_physics.holographic_combustion import Fire
        self.material = material
        self.mass0 = float(mass_kg)
        self.fire = Fire(material, mass_kg, temp_K=temp_K)

    def light(self, spark_temp_K=1000.0):
        """Apply a spark: raise the object to a temperature that will ignite it (if it is flammable)."""
        self.fire.T = max(self.fire.T, float(spark_temp_K))
        return self

    @property
    def mass(self):
        """Remaining mass (kg) -- the fire has eaten the rest."""
        return self.fire.fuel

    def burn_fraction(self):
        return float(np.clip(1.0 - self.mass / max(self.mass0, 1e-12), 0.0, 1.0))

    def is_ash(self, tol=0.98):
        return self.burn_fraction() >= tol

    def step(self, dt, ambient_K=293.15):
        """Advance the burn by dt. Returns {mass, burn_fraction, burning, appearance, smoke_color, flame_color,
        is_ash}. The appearance blends base->char->ash by how much has burned."""
        s = self.fire.step(dt, ambient_K=ambient_K)
        bf = self.burn_fraction()
        return {"mass": self.mass, "burn_fraction": bf, "burning": s["burning"],
                "appearance": char_color(self.material, bf), "smoke_color": s["smoke_color"],
                "flame_color": s["flame_color"], "is_ash": self.is_ash()}


def evaporate(material, mass_kg, temp_K, energy_per_step, steps, pressure_Pa=101325.0):
    """DECAY by evaporation (the M5 analog of burning): heat a liquid parcel and let it boil away to vapour, losing
    liquid mass over time. Returns the list of remaining-liquid masses per step. Ties M7's 'consumed over time'
    idea to phase change -- a puddle drying up. Uses holographic_phase (M5)."""
    from holographic.misc.holographic_phase import PhaseState
    ps = PhaseState(material, mass_kg, temp_K=temp_K, pressure_Pa=pressure_Pa)
    liquid = []
    for _ in range(int(steps)):
        ps.add_heat(float(energy_per_step))
        liquid.append(ps.liquid)
    return liquid


def _selftest():
    """A lit wood object loses mass over time, its appearance marches base->char->ash, it emits smoke while
    burning and ends as ash; nothing burns unlit; evaporation drains a puddle. Deterministic."""
    # (1) an unlit object at room temperature does not burn
    cold = BurningObject("wood", 1.0)
    r = cold.step(0.5)
    assert not r["burning"] and r["mass"] == 1.0 and r["burn_fraction"] == 0.0

    # (2) a lit wood object burns down: mass falls monotonically, burn fraction rises, ends as ash
    obj = BurningObject("wood", 1.0).light()
    masses, fracs, appearances = [], [], []
    smoked = False
    for _ in range(80):
        s = obj.step(0.5)
        masses.append(s["mass"]); fracs.append(s["burn_fraction"]); appearances.append(s["appearance"])
        if s["burning"] and s["smoke_color"].mean() > 0.3:
            smoked = True
    assert masses[0] < 1.0 and masses[-1] < 0.05                    # burned down to almost nothing
    assert all(masses[i + 1] <= masses[i] + 1e-12 for i in range(len(masses) - 1))   # mass monotonic down
    assert fracs[-1] > 0.95 and obj.is_ash()                        # ended as ash
    assert smoked                                                   # emitted (pale wood) smoke while burning

    # (3) appearance really marched base -> char -> ash: it got darker mid-burn, then paler (ash) at the end
    base = _base_albedo("wood")
    mid = char_color("wood", 0.5); end = char_color("wood", 1.0)
    assert mid.mean() < base.mean()                                 # charring darkens it
    assert end.mean() > mid.mean()                                  # ash is paler than char
    assert np.allclose(char_color("wood", 0.0), base)               # unburned = pristine

    # (4) evaporation (M5 decay analog): a water puddle boils away, liquid mass drops
    liq = evaporate("water", 1.0, temp_K=355.0, energy_per_step=2.0e5, steps=15)
    assert liq[-1] < liq[0]                                          # the puddle shrank

    # (5) deterministic
    a = BurningObject("wood", 1.0).light(); b = BurningObject("wood", 1.0).light()
    ra = a.step(0.5); rb = b.step(0.5)
    assert ra["mass"] == rb["mass"] and np.array_equal(ra["appearance"], rb["appearance"])
    print("holographic_burn selftest OK: a lit object loses mass to almost nothing (monotonic), appearance marches "
          "base->char->ash, emits smoke while burning, ends as ash; evaporation drains a puddle; deterministic")


if __name__ == "__main__":
    _selftest()

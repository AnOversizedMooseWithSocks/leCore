"""holographic_combustion.py -- M6: MATERIAL-SPECIFIC combustion. Wood smoke and plastic smoke actually differ.

WHY THIS EXISTS (Material Structure & Process backlog, item M6)
--------------------------------------------------------------
The fluid solver (holographic_fluid.StableFluid) already turns fuel above an ignition temperature into heat +
smoke -- but with ONE global ignition/burn_rate/smoke_yield for the whole solver, so everything burns the same.
Real materials do not: wood lights at ~300 C and makes pale-grey woodsmoke; PVC needs far more heat and makes
dense, sooty black smoke; gasoline lights easily and burns hot and fast. This module adds the missing DATA
(per-material autoignition, heat of combustion, smoke colour, soot yield, burn rate) and the COUPLINGS that feed
those numbers to the two things that render a fire:
  * the fluid solver's combustion (configure it from the material, then inject fuel), and
  * the surface emitter (spawn smoke particles coloured by the material), with the flame/ember COLOUR coming from
    the blackbody radiator (T3) at the flame temperature.
Nothing here is a new solver -- it is the data layer + couplings the backlog describes, standing on the heat
model (T4) that supplies the temperature which decides whether a thing is hot enough to ignite.

THE PHYSICS / DATA (readable)
-----------------------------
  * AUTOIGNITION temperature: the surface temperature at which a material spontaneously catches fire. Below it,
    nothing happens; at or above it, combustion begins. (Real fire-safety data; approximate, art-directable.)
  * HEAT OF COMBUSTION (J/kg): the energy released per kilogram burned -- this is what raises the flame's (and
    the surroundings') temperature, via the T4 heat model's Q = m c dT run backwards (energy -> temperature).
  * SOOT YIELD / SMOKE COLOUR: incomplete combustion makes soot; more soot -> darker, denser smoke. Wood is
    moderately sooty and grey; PVC and heavy plastics are very sooty and black; clean fuels (alcohol) are pale.
  * BURN RATE: the fraction of remaining fuel consumed per second -- how fast it goes.

HONEST SCOPE (kept negative): data-driven combustion PRODUCTS (heat, smoke colour, soot, rate), NOT chemical
reaction kinetics; flame SPREAD is left to the fluid solver / a reaction-diffusion front, not modelled here as
detailed flame chemistry. The numbers are plausible fire-safety values, art-directable, not a combustion-lab
dataset. Deterministic; NumPy + stdlib. These belong in the material definition data layer long-term (they are
the combustion columns); they live here for now beside the physics that uses them.
"""
import numpy as np

# The combustion data columns, per material:
#   autoignition_K   -- surface temperature at which it catches fire (K)
#   heat_of_combustion -- energy released per kg burned (J/kg)
#   smoke_color      -- rgb of the smoke it makes (sooty -> dark; clean -> pale)
#   soot_yield       -- 0..1, how sooty/opaque the smoke is (incomplete combustion)
#   burn_rate        -- fraction of remaining fuel consumed per second (1/s)
COMBUSTION = {
    # material          autoign(K)  HoC(J/kg)   smoke_color            soot   burn_rate  flame_temp(K)
    "wood":         dict(autoignition_K=573.0, heat_of_combustion=1.8e7, smoke_color=(0.55, 0.53, 0.50), soot_yield=0.55, burn_rate=0.15, flame_temp_K=1400.0),
    "paper":        dict(autoignition_K=506.0, heat_of_combustion=1.6e7, smoke_color=(0.70, 0.68, 0.64), soot_yield=0.35, burn_rate=0.45, flame_temp_K=1300.0),
    "pvc_plastic":  dict(autoignition_K=727.0, heat_of_combustion=1.8e7, smoke_color=(0.10, 0.10, 0.11), soot_yield=0.90, burn_rate=0.10, flame_temp_K=1500.0),
    "abs_plastic":  dict(autoignition_K=689.0, heat_of_combustion=3.6e7, smoke_color=(0.12, 0.12, 0.13), soot_yield=0.85, burn_rate=0.12, flame_temp_K=1550.0),
    "gasoline":     dict(autoignition_K=553.0, heat_of_combustion=4.4e7, smoke_color=(0.20, 0.19, 0.18), soot_yield=0.70, burn_rate=0.60, flame_temp_K=1600.0),
    "ethanol":      dict(autoignition_K=636.0, heat_of_combustion=2.7e7, smoke_color=(0.85, 0.85, 0.86), soot_yield=0.05, burn_rate=0.40, flame_temp_K=1500.0),
    "coal":         dict(autoignition_K=723.0, heat_of_combustion=3.0e7, smoke_color=(0.08, 0.08, 0.08), soot_yield=0.95, burn_rate=0.05, flame_temp_K=1600.0),
    "methane":      dict(autoignition_K=853.0, heat_of_combustion=5.0e7, smoke_color=(0.90, 0.90, 0.92), soot_yield=0.02, burn_rate=0.90, flame_temp_K=1950.0),
}


def is_flammable(material):
    """Does this material have combustion data (can it burn)?"""
    return material in COMBUSTION


def ignites(material, temperature_K):
    """True if `material` is at or above its autoignition temperature -- i.e. hot enough to catch fire. Below its
    threshold, nothing ignites (the honest gate the bar checks)."""
    if material not in COMBUSTION:
        return False
    return float(temperature_K) >= COMBUSTION[material]["autoignition_K"]


def flame_color(temperature_K, material=None, normalize="hue"):
    """The colour of flame/embers. The thermal glow comes from the blackbody radiator (T3) at `temperature_K` (a
    cool smoulder dull red, a hot flame yellow-white). If a `material` is given and its elemental makeup has a
    characteristic flame-test colour (holographic_elements: copper green, sodium yellow, ...), that emission-LINE
    colour is blended in -- the very thing the blackbody continuum alone cannot produce. So a copper-bearing fire
    tints green over its thermal glow. Default (material=None) is the pure thermal colour, unchanged."""
    from holographic_blackbody import blackbody_rgb
    base = blackbody_rgb(temperature_K, normalize=normalize)
    if material is not None:
        try:
            from holographic_elements import material_elemental
            me = material_elemental(material)
            if me is not None and me["flame_color"] is not None:
                tint = np.asarray(me["flame_color"], float)
                w = 0.6                                              # emission lines dominate a flame test visually
                base = np.clip((1.0 - w) * base + w * tint, 0.0, 1.0)
        except Exception:
            pass                                                    # no composition -> just the thermal glow
    return base


def combustion_products(material, burned_kg):
    """What burning `burned_kg` of `material` produces: {heat_J, smoke_color, soot_mass, smoke_mass}. Heat is
    burned_mass * heat_of_combustion; soot_mass scales with the soot yield (the dark, opaque fraction)."""
    c = COMBUSTION[material]
    heat = float(burned_kg) * c["heat_of_combustion"]
    soot = float(burned_kg) * c["soot_yield"]
    return {"heat_J": heat, "smoke_color": np.asarray(c["smoke_color"], float),
            "soot_mass": soot, "smoke_mass": float(burned_kg)}


class Fire:
    """A burning body, material-aware. Holds remaining fuel and a temperature. A fire LATCHES: once its
    temperature reaches the material's autoignition point it stays lit (a real flame sustains itself) and consumes
    fuel at the material's burn rate until the fuel runs out; then it goes cold. While burning, the temperature
    climbs toward the material's flame temperature (so the flame colour reads hot); when starved it cools toward
    ambient. Each step reports the smoke it made (material colour + soot) and the flame colour (blackbody at the
    current temperature). This is the object M7 (burn/decay) will drive to consume an object over time."""

    def __init__(self, material, fuel_kg, temp_K=293.15, specific_heat=None, heat_retention=0.02):
        if material not in COMBUSTION:
            raise KeyError("%r has no combustion data -- known: %s" % (material, sorted(COMBUSTION)))
        self.material = material
        self.fuel = float(fuel_kg)
        self.T = float(temp_K)
        if specific_heat is None:                                   # reuse the material definition's specific heat
            try:
                from holographic_heat import material_thermal
                specific_heat = material_thermal(material)["specific_heat"]
            except Exception:
                specific_heat = 1700.0
        self.c = float(specific_heat)
        self.retention = float(heat_retention)
        self.ignited = False                                        # the latch: has it caught fire yet?
        self.burning = False

    def step(self, dt, ambient_K=293.15, cool_rate=0.4):
        """Advance the fire by dt seconds. Returns a dict: whether it is burning, fuel burned this step, heat
        released, the smoke (colour + soot) and the flame colour. Deterministic."""
        c = COMBUSTION[self.material]
        # LATCH: catches fire the first time it is hot enough, then stays lit while fuel remains
        if not self.ignited and self.fuel > 1e-9 and self.T >= c["autoignition_K"]:
            self.ignited = True
        self.burning = self.ignited and self.fuel > 1e-9
        burned = 0.0
        if self.burning:
            burned = min(self.fuel, c["burn_rate"] * dt * self.fuel)   # fraction of remaining fuel this step
            self.fuel -= burned
            # a lit fire is HOT: temperature relaxes toward the material's flame temperature (fast)
            self.T += (c["flame_temp_K"] - self.T) * min(1.0, 3.0 * dt)
            if self.fuel <= 1e-9:                                   # fuel just ran out -> the fire dies
                self.ignited = False
        else:
            self.T += -cool_rate * (self.T - ambient_K) * dt        # not burning -> cool toward ambient
        prod = combustion_products(self.material, burned) if burned > 0 else \
            {"heat_J": 0.0, "smoke_color": np.asarray(c["smoke_color"], float), "soot_mass": 0.0}
        return {"burning": self.burning, "burned_kg": burned, "heat_J": prod["heat_J"],
                "smoke_color": prod["smoke_color"], "soot_mass": prod["soot_mass"],
                "fuel_left": self.fuel, "temperature_K": self.T,
                "flame_color": flame_color(self.T, material=self.material)}


# --------------------------------------------------------------------------------------------------------------
# COUPLINGS: feed the material's numbers to the two things that render a fire (the fluid solver + the emitter).
# --------------------------------------------------------------------------------------------------------------
def configure_fluid(fluid, material):
    """Set a StableFluid's global combustion parameters FROM a material, so its volumetric fire behaves like that
    material burning (ignition temperature, burn rate, smoke produced per unit fuel from the soot yield). Returns
    the fluid for chaining. This is the coupling to holographic_fluid's combust()."""
    c = COMBUSTION[material]
    fluid.ignition = c["autoignition_K"]
    fluid.burn_rate = c["burn_rate"]
    fluid.smoke_yield = 0.2 + 0.6 * c["soot_yield"]                 # sootier material -> more visible smoke
    return fluid


def emit_smoke(sdf_eval, material, n, bounds, speed=1.0, seed=0):
    """Spawn `n` smoke particles from an SDF surface (holographic_emitter.emit_from_surface), coloured by the
    material's smoke colour and carrying its soot yield -- so wood emits pale-grey particles and PVC emits dense
    black ones from the same geometry. Returns (positions, velocities, colors, soot). The coupling to the emitter."""
    from holographic_emitter import emit_from_surface
    c = COMBUSTION[material]
    pos, nrm, vel = emit_from_surface(sdf_eval, n, bounds, speed=speed, seed=seed)
    colors = np.repeat(np.asarray(c["smoke_color"], float)[None, :], len(pos), axis=0)
    return pos, vel, colors, float(c["soot_yield"])


def _selftest():
    """Ignition respects each material's own threshold, wood and PVC make visibly different smoke, a lit fire
    sustains then burns out as fuel runs low, and everything is deterministic."""
    # (1) ignition gate: wood lights at 300 C, PVC does not until much hotter; nothing lights when cold
    assert ignites("wood", 600.0) and not ignites("wood", 500.0)
    assert not ignites("pvc_plastic", 600.0) and ignites("pvc_plastic", 750.0)
    assert not ignites("wood", 293.15)                              # room temperature: nothing burns

    # (2) wood smoke vs PVC smoke are visibly different: PVC is far darker (sootier) and burns slower
    wood = combustion_products("wood", 1.0); pvc = combustion_products("pvc_plastic", 1.0)
    assert wood["smoke_color"].mean() > 0.4 and pvc["smoke_color"].mean() < 0.2   # pale grey vs black
    assert pvc["soot_mass"] > wood["soot_mass"]                      # PVC much sootier
    assert COMBUSTION["pvc_plastic"]["burn_rate"] < COMBUSTION["wood"]["burn_rate"]

    # (3) flame colour comes from the blackbody: a hot flame is not red-dominant like a cool smoulder
    cool = flame_color(900.0); hot = flame_color(1600.0)
    assert cool[0] > cool[2] and hot[2] >= cool[2]                   # blue rises with flame temperature

    # (4) a lit wood fire sustains itself then burns out as fuel depletes; nothing lights below threshold
    cold = Fire("wood", fuel_kg=1.0, temp_K=293.15)
    r = cold.step(1.0); assert not r["burning"] and r["burned_kg"] == 0.0   # too cold -> no burn
    lit = Fire("wood", fuel_kg=1.0, temp_K=900.0)                    # lit with a match (already hot)
    fuel_trace = []
    for _ in range(60):
        s = lit.step(0.5)
        fuel_trace.append(s["fuel_left"])
    assert fuel_trace[0] < 1.0                                       # it burned
    assert fuel_trace[-1] < fuel_trace[0] * 0.2                      # fuel mostly consumed over time
    assert all(fuel_trace[i + 1] <= fuel_trace[i] + 1e-12 for i in range(len(fuel_trace) - 1))  # monotonic burn

    # (5) deterministic
    a = Fire("gasoline", 1.0, 900.0).step(0.3); b = Fire("gasoline", 1.0, 900.0).step(0.3)
    assert a["burned_kg"] == b["burned_kg"] and np.array_equal(a["smoke_color"], b["smoke_color"])
    print("holographic_combustion selftest OK: per-material ignition thresholds hold; wood smoke pale/grey vs PVC "
          "black+sooty; flame colour from blackbody; a lit fire sustains then burns out; deterministic")


if __name__ == "__main__":
    _selftest()

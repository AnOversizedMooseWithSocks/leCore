"""holographic_phase.py -- M5: PHASE CHANGE. Water <-> steam <-> ice, with latent heat and the boiling plateau.

WHY THIS EXISTS (Material Structure & Process backlog, item M5)
--------------------------------------------------------------
The engine has a `phase` LABEL (solid/liquid/gas) on each material but no DYNAMICS -- nothing melts, boils,
freezes, or condenses. This module adds that, standing on the two thermodynamics pieces just built: the heat
model (T4) supplies the energy that changes temperature, and the gas model (T1) supplies the pressure-dependent
boiling point (so water boils cooler up a mountain). The result is the textbook behaviour: pour heat into ice
and it warms, then HOLDS at 0 C while it melts (paying the latent heat of fusion), warms as liquid, then HOLDS
at 100 C while it boils (paying the latent heat of vaporization), and the reverse on cooling.

THE PHYSICS (readable, first-principles)
----------------------------------------
  * SENSIBLE heat changes TEMPERATURE:  dT = Q / (m c), with c the specific heat of the CURRENT phase (ice 2090,
    liquid water 4186, steam ~2010 J/kg/K). This is just the T4 heat model per phase.
  * LATENT heat changes PHASE at constant temperature:  to melt takes L_fusion J/kg, to vaporize L_vapor J/kg.
    While a transition is in progress the temperature PLATEAUS -- every joule goes into converting mass, not into
    raising temperature -- which is why a pot of boiling water stays at 100 C until it has boiled away.
  * The boiling point is not fixed: it TRACKS PRESSURE (Clausius-Clapeyron, from the gas model). Lower pressure
    -> lower boiling point. Melting point is treated as fixed (its pressure dependence is tiny).

The bookkeeping is lumped: a parcel holds mass in each phase and a temperature; `add_heat(Q)` walks the energy
through warming and transitions in order, moving mass between phases and holding temperature during each change.
Vapour produced can be handed to the fluid solver's smoke/temperature field to rise and diffuse (a coupling, not
re-modelled here).

HONEST SCOPE (kept negative): lumped latent-heat BOOKKEEPING with a moving-fraction rule, NOT a molecular-
dynamics or free-surface two-phase flow; constant specific heats and latent heats over the range; boiling point
tracks pressure but nucleation / superheating are out of scope. Correct for the melt/boil/freeze/condense
behaviour and the plateau; not a CFD two-phase solver. Deterministic; NumPy + stdlib. Latent heats and transition
points are the new phase data columns.
"""
import numpy as np

# The phase data columns, per material: transition temperatures (K) and latent heats (J/kg), plus the specific
# heat of each phase (J/kg/K). Water is the fully-populated case; a couple of metals show the pattern generalises.
PHASE_DATA = {
    "water": dict(melt_point_K=273.15, boil_point_K=373.15, latent_fusion=3.34e5, latent_vapor=2.257e6,
                  c_solid=2090.0, c_liquid=4186.0, c_gas=2010.0),
    "iron":  dict(melt_point_K=1811.0, boil_point_K=3134.0, latent_fusion=2.47e5, latent_vapor=6.09e6,
                  c_solid=449.0, c_liquid=820.0, c_gas=450.0),
    "aluminum": dict(melt_point_K=933.0, boil_point_K=2792.0, latent_fusion=3.97e5, latent_vapor=1.05e7,
                     c_solid=897.0, c_liquid=1180.0, c_gas=900.0),
}


def has_phase_data(material):
    """Does this material carry the melt/boil/latent-heat columns needed to change phase?"""
    return material in PHASE_DATA


def boiling_point_at(material, pressure_Pa=101325.0):
    """The boiling temperature (K) of `material` at a given pressure. Uses the gas model's Clausius-Clapeyron
    curve (T1) anchored at this material's normal boiling point + latent heat -- so it falls with pressure."""
    d = PHASE_DATA[material]
    if pressure_Pa == 101325.0:
        return d["boil_point_K"]
    from holographic_gas import boiling_point
    # anchor Clausius-Clapeyron at THIS material's normal boiling point; molar mass ~ water default is fine for
    # the shape, but for water we pass its own molar mass for accuracy
    mm = 0.018015 if material == "water" else 0.05
    return float(boiling_point(pressure_Pa, latent_heat=d["latent_vapor"], molar_mass=mm,
                               ref_T=d["boil_point_K"], ref_P=101325.0))


class PhaseState:
    """A parcel of one material with mass split across solid / liquid / gas and a single temperature. `add_heat(Q)`
    pushes energy in (or out, if negative) and lets it warm the current phase and drive transitions, holding the
    temperature flat during each melt/boil while the latent heat is paid. Boiling point can track pressure."""

    def __init__(self, material, mass_kg, temp_K=293.15, pressure_Pa=101325.0):
        if material not in PHASE_DATA:
            raise KeyError("%r has no phase data -- known: %s" % (material, sorted(PHASE_DATA)))
        self.material = material
        self.d = PHASE_DATA[material]
        self.P = float(pressure_Pa)
        self.T = float(temp_K)
        # start all the mass in whichever phase matches the starting temperature
        m = float(mass_kg)
        self.solid = m if temp_K < self.d["melt_point_K"] else 0.0
        self.gas = m if temp_K >= boiling_point_at(material, self.P) else 0.0
        self.liquid = m - self.solid - self.gas

    def total_mass(self):
        return self.solid + self.liquid + self.gas

    def _c_current(self):
        """Specific heat of the phase the mass is currently mostly in (for the sensible-heat warming step)."""
        if self.solid > max(self.liquid, self.gas):
            return self.d["c_solid"]
        if self.gas > max(self.solid, self.liquid):
            return self.d["c_gas"]
        return self.d["c_liquid"]

    def add_heat(self, Q, max_iter=1000):
        """Add Q joules (negative removes heat). Walks the energy through: warm the current phase up to the next
        transition, then spend energy converting mass at constant temperature until that transition completes,
        then continue. Handles both heating (melt then boil) and cooling (condense then freeze)."""
        melt = self.d["melt_point_K"]
        boil = boiling_point_at(self.material, self.P)
        Lf, Lv = self.d["latent_fusion"], self.d["latent_vapor"]
        Q = float(Q)
        heating = Q >= 0.0
        it = 0
        while abs(Q) > 1e-9 and it < max_iter:
            it += 1
            m = self.total_mass()
            if m <= 1e-12:
                break
            c = self._c_current()
            if heating:
                # 1) if sitting exactly at melt with solid left, spend energy MELTING (T held)
                if abs(self.T - melt) < 1e-6 and self.solid > 1e-12:
                    need = self.solid * Lf                          # energy to melt all remaining solid
                    spend = min(Q, need)
                    frac = spend / need if need > 0 else 1.0
                    moved = self.solid * frac
                    self.solid -= moved; self.liquid += moved       # solid -> liquid, temperature UNCHANGED
                    Q -= spend
                    continue
                # 2) if sitting exactly at boil with liquid left, spend energy BOILING (T held)
                if abs(self.T - boil) < 1e-6 and self.liquid > 1e-12:
                    need = self.liquid * Lv
                    spend = min(Q, need)
                    frac = spend / need if need > 0 else 1.0
                    moved = self.liquid * frac
                    self.liquid -= moved; self.gas += moved         # liquid -> gas, temperature UNCHANGED
                    Q -= spend
                    continue
                # 3) otherwise warm toward the next transition above the current temperature
                target = melt if (self.T < melt and self.solid > 1e-12) else \
                         (boil if (self.T < boil and self.liquid > 1e-12) else np.inf)
                if target == np.inf:
                    self.T += Q / (m * c); Q = 0.0                  # nothing left to transition -> all sensible
                else:
                    dT_need = target - self.T
                    q_to_target = m * c * dT_need
                    if Q >= q_to_target:
                        self.T = target; Q -= q_to_target           # reach the transition, loop handles latent
                    else:
                        self.T += Q / (m * c); Q = 0.0
            else:  # cooling (Q negative): condense then freeze, mirror image
                if abs(self.T - boil) < 1e-6 and self.gas > 1e-12:
                    need = self.gas * Lv
                    spend = min(-Q, need); frac = spend / need if need > 0 else 1.0
                    moved = self.gas * frac
                    self.gas -= moved; self.liquid += moved         # gas -> liquid, T held
                    Q += spend
                    continue
                if abs(self.T - melt) < 1e-6 and self.liquid > 1e-12:
                    need = self.liquid * Lf
                    spend = min(-Q, need); frac = spend / need if need > 0 else 1.0
                    moved = self.liquid * frac
                    self.liquid -= moved; self.solid += moved       # liquid -> solid (freeze), T held
                    Q += spend
                    continue
                target = boil if (self.T > boil and self.gas > 1e-12) else \
                         (melt if (self.T > melt and self.liquid > 1e-12) else -np.inf)
                if target == -np.inf:
                    self.T += Q / (m * c); Q = 0.0
                else:
                    q_to_target = m * c * (target - self.T)         # negative
                    if Q <= q_to_target:
                        self.T = target; Q -= q_to_target
                    else:
                        self.T += Q / (m * c); Q = 0.0
        return self

    def phase_fractions(self):
        """Mass fraction in each phase right now."""
        m = self.total_mass() or 1.0
        return {"solid": self.solid / m, "liquid": self.liquid / m, "gas": self.gas / m}

    def dominant_phase(self):
        return max(("solid", "liquid", "gas"), key=lambda p: getattr(self, p))


def _selftest():
    """The boiling PLATEAU is the headline: heat liquid water past 100 C and the temperature holds there until the
    latent heat is paid while mass moves liquid->gas. Melting, freezing and pressure-dependent boiling also work."""
    # (1) THE PLATEAU. 1 kg of water at 99 C; add heat in steps and watch the temperature.
    ps = PhaseState("water", 1.0, temp_K=372.15)                    # just below boiling
    temps, gas = [], []
    for _ in range(40):
        ps.add_heat(1.0e5)                                          # 100 kJ per step
        temps.append(ps.T); gas.append(ps.gas)
    temps = np.array(temps)
    at_boil = np.abs(temps - 373.15) < 0.2
    assert at_boil.sum() >= 15                                      # temperature HOLDS at 100 C for many steps
    assert gas[-1] > gas[0]                                         # liquid turned to steam during the hold
    # total latent heat to boil 1 kg is ~2.257 MJ; the plateau should span roughly that much energy
    boil_steps = int(at_boil.sum())
    assert 2.0e6 < boil_steps * 1.0e5 < 2.6e6                       # ~22-23 steps of 100 kJ

    # (2) MELTING plateau at 0 C: heat ice through the melt point, temperature holds while it melts
    ice = PhaseState("water", 1.0, temp_K=272.15)                   # just below freezing point (solid)
    assert ice.dominant_phase() == "solid"
    ice.add_heat(3.34e5 * 0.5 + 2090.0 * 1.0)                        # enough to reach 0C then melt ~half
    assert abs(ice.T - 273.15) < 0.5 and 0.2 < ice.phase_fractions()["liquid"] < 0.8   # mid-melt, T held at 0C

    # (3) FREEZING reverses it: remove heat from liquid at 0 C -> it turns to solid at constant temperature
    water = PhaseState("water", 1.0, temp_K=273.15)
    water.liquid = 1.0; water.solid = 0.0
    water.add_heat(-3.34e5 * 0.5)                                   # remove half the fusion heat
    assert abs(water.T - 273.15) < 0.5 and water.solid > 0.3        # partly frozen, temperature held

    # (4) boiling point TRACKS PRESSURE (from the gas model): lower pressure -> boils cooler
    assert abs(boiling_point_at("water", 101325.0) - 373.15) < 0.5
    assert boiling_point_at("water", 70000.0) < 373.15

    # (5) deterministic
    a = PhaseState("water", 1.0, 372.15); a.add_heat(5e5)
    b = PhaseState("water", 1.0, 372.15); b.add_heat(5e5)
    assert abs(a.T - b.T) < 1e-12 and abs(a.gas - b.gas) < 1e-12
    print("holographic_phase selftest OK: boiling temperature HOLDS at 100 C while liquid->steam (latent-heat "
          "plateau ~%.1f MJ/kg); melt/freeze hold at 0 C; boiling point falls with pressure; deterministic" %
          (boil_steps * 1.0e5 / 1e6))


if __name__ == "__main__":
    _selftest()

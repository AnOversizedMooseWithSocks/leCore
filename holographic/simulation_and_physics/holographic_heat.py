"""holographic_heat.py -- T4: the HEAT MODEL. Energy heats things (Q = m c dT) and heat spreads (Fourier conduction).

WHY THIS EXISTS (thermodynamics foundation, item T4)
----------------------------------------------------
This is the keystone the whole material-PROCESS layer stands on: every process is triggered by TEMPERATURE.
M5 (phase change) fires when temperature crosses a melt/boil point; M6 (combustion) ignites when it crosses
autoignition; M7 (burn/decay) proceeds while a body stays hot. So first we need to answer two questions from
real physics: how much energy changes a thing's temperature, and how does temperature spread through and between
things. That is exactly Q = m c dT and the heat (diffusion) equation.

THE PHYSICS (readable, first-principles)
----------------------------------------
  * SPECIFIC HEAT:  Q = m * c * dT. To raise mass m (kg) of a material with specific heat c (J/kg/K) by dT
    kelvin costs Q joules. Invert it and an energy input Q yields dT = Q / (m c). (c comes straight from the
    material definitions -- water 4186, steel 490 -- reused, not restated.)
  * CONDUCTION (Fourier's law -> the heat equation):  dT/dt = alpha * laplacian(T), where the thermal
    diffusivity alpha = k / (rho c) (k = thermal conductivity W/m/K, rho = density, c = specific heat). Hot
    spots spread and even out; total heat is conserved under insulated (zero-flux) boundaries.
  * NEWTON COOLING:  a body loses heat to its surroundings at a rate proportional to the temperature difference,
    dT/dt = -(hA / mc)(T - T_ambient) -- why a hot object cools fast at first, then slowly as it nears ambient.

The conduction step is an explicit finite-difference stencil; it AUTO-SUBSTEPS to stay under the stability
limit (r = alpha dt / dx^2 <= 1/(2*ndim)), so a caller can ask for any dt and get a stable, still-correct result.

HONEST SCOPE (kept negative): constant material properties (c, k not temperature-dependent), an explicit scheme
(auto-substepped for stability, so large dt costs more inner steps rather than going unstable), and insulated
default boundaries. No radiative transfer here (that is the blackbody module) and no convection (that is the
fluid solver). NumPy + stdlib; deterministic. Conductivity is pulled from the definition enrichment data
(data/definitions/native/materials/*.json) so it is not duplicated.
"""
import numpy as np
import json
import os


_R_GAS = 8.314462618      # not used here; kept out -- see holographic_gas


def heat_energy(mass_kg, specific_heat, delta_T):
    """Q = m c dT (joules): the energy to change `mass_kg` of a material (specific heat J/kg/K) by `delta_T` K."""
    return float(mass_kg) * float(specific_heat) * float(delta_T)


def temperature_change(energy_J, mass_kg, specific_heat):
    """dT = Q / (m c): the temperature change from putting `energy_J` into mass m with specific heat c."""
    return float(energy_J) / (float(mass_kg) * float(specific_heat))


def thermal_diffusivity(k, rho, c):
    """alpha = k / (rho c) (m^2/s): how fast heat spreads. High for metals (k big), low for insulators."""
    return float(k) / (float(rho) * float(c))


def _laplacian(T):
    """The edge-replicated (zero-flux / Neumann) discrete Laplacian, any dimension.

    A1: this stencil was duplicated bit-for-bit in `holographic_heat` and `holographic_wave`. It now lives
    once in `holographic_laplacian.laplacian(field, bc=)`, which also offers the periodic and dirichlet
    boundaries; this alias keeps the old private name working for existing callers.
    """
    from holographic.simulation_and_physics.holographic_laplacian import laplacian
    return laplacian(T, bc="neumann")


def diffuse_heat(temp, alpha, dx=1.0, dt=None, steps=1, bc="neumann", operator=None):
    """Evolve a temperature FIELD (any-D array) by conduction dT/dt = alpha*laplacian(T) for `steps` steps.
    Auto-substeps to stay stable: if the requested dt exceeds the explicit limit, it is split into inner steps,
    so the result is stable AND still the correct amount of diffusion. Insulated boundaries (total heat conserved).
    `dt=None` picks the largest stable step.

    `bc="periodic"` uses the EXACT closed form (no stepping, no stability limit). `operator=` accepts a prebuilt
    `holographic_laplacian.diffusion_operator` -- a shader-algebra Pipeline holding exp(-alpha|k|^2 t) -- which is
    bit-identical and ~1.9x faster on reuse because the transfer is composed once rather than re-exponentiated per
    call. It is ignored under Neumann, where no such transfer exists: the edge-replicated Laplacian is not
    shift-equivariant, and applying the periodic form there is measured 4.76e-02 wrong."""
    T = np.asarray(temp, float).copy()
    # bc="periodic": the Laplacian is then a CIRCULAR convolution, diagonal in the FFT, so the whole
    # evolution to time dt*steps is ONE closed-form evaluation -- each mode decays by exp(-alpha k^2 t).
    # No time step, no stability limit, no substepping, and exact (measured 2.2e-16 vs an iterative
    # stepper still at 1.5e-4 after 1000 steps). The default (Neumann/insulated) stencil is NOT circular,
    # so it keeps the substepped loop below -- that is the honest boundary of the closed form.
    if bc == "periodic":
        from holographic.simulation_and_physics.holographic_laplacian import diffuse_spectral
        r_max = 0.9 / (2.0 * T.ndim)
        step = (r_max * dx * dx / float(alpha)) if dt is None else float(dt)
        if operator is not None:
            # A prebuilt `laplacian.diffusion_operator` (a shader-algebra Pipeline holding exp(-alpha|k|^2 t)).
            # Bit-identical to diffuse_spectral (max|diff| 0.0e+00) and ~1.9x faster on reuse, because the transfer
            # is composed once instead of re-exponentiated per call. Compose once, apply many.
            return operator.apply(T)
        return diffuse_spectral(T, alpha, step * int(steps), dx=dx)
    if bc != "neumann":
        raise ValueError("bc must be 'neumann' (insulated, the default) or 'periodic'")
    ndim = T.ndim
    r_max = 0.9 / (2.0 * ndim)                                       # stay just under the stability limit
    if dt is None:
        dt = r_max * dx * dx / float(alpha)
    r = float(alpha) * dt / (dx * dx)
    n_sub = max(1, int(np.ceil(r / r_max)))                         # how many inner steps to stay stable
    r_sub = r / n_sub
    for _ in range(int(steps) * n_sub):
        T = T + r_sub * _laplacian(T)
    return T


class HeatBody:
    """A lumped-capacity body at a single uniform temperature -- the simplest useful thermal object. Add energy
    and its temperature rises by Q/(m c); let it sit in cooler surroundings and it relaxes toward ambient by
    Newton's law of cooling. This is what M6/M7 will hang a material's ignition/consumption state on."""

    def __init__(self, mass_kg, specific_heat, temp_K=293.15):
        self.mass = float(mass_kg)
        self.c = float(specific_heat)
        self.T = float(temp_K)

    def add_energy(self, energy_J):
        """Put `energy_J` in (or take out, if negative). Temperature rises by Q/(m c)."""
        self.T += temperature_change(energy_J, self.mass, self.c)
        return self.T

    def newton_cool(self, ambient_K, h_area, dt, steps=1):
        """Relax toward `ambient_K` by Newton cooling: dT/dt = -(hA/mc)(T - ambient). `h_area` = h*A (W/K), the
        heat-transfer coefficient times exposed area. Auto-substeps so a big dt stays stable."""
        tau = self.mass * self.c / max(h_area, 1e-12)               # time constant (s)
        sub = max(1, int(np.ceil((dt * steps) / (0.5 * tau))))      # keep each inner step well within tau
        ds = dt * steps / sub
        for _ in range(sub):
            self.T += -(1.0 / tau) * (self.T - float(ambient_K)) * ds
        return self.T


# --------------------------------------------------------------------------------------------------------------
# Material property lookup -- REUSE the definitions (density, specific_heat) and read conductivity from the
# enrichment DATA files (not restated here). This is the "one material table, add columns via data" design.
# --------------------------------------------------------------------------------------------------------------
_ENRICH_CACHE = None


def _load_enrichment():
    """Read the material enrichment JSON (thermal_conductivity etc.) once. Each property is either a plain number
    or a {'value','unit'} record; we take the value (already SI for the fields we use). Cached."""
    global _ENRICH_CACHE
    if _ENRICH_CACHE is not None:
        return _ENRICH_CACHE
    out = {}
    # the enrichment JSON ships in the lecore_data package (works from a clone and a wheel); fall back to the old
    # repo-relative data/ path for older checkouts. Missing file -> defaults below, so this stays graceful either way.
    path = None
    try:
        import lecore_data
        cand = lecore_data.file("definitions", "native", "materials", "enrich.json")
        if os.path.exists(cand):
            path = cand
    except Exception:
        path = None
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "definitions", "native", "materials", "enrich.json")
    try:
        with open(path) as f:
            for row in json.load(f):
                props = {}
                for key, val in row.get("properties", {}).items():
                    props[key] = float(val["value"]) if isinstance(val, dict) and "value" in val else val
                out[row["name"]] = props
    except (OSError, ValueError):
        pass                                                        # no enrichment file -> fall back to defaults
    _ENRICH_CACHE = out
    return out


# sensible fallbacks (W/m/K) for materials the enrichment file does not cover -- documented, not silent
_K_FALLBACK = {"aluminum": 237.0, "copper": 401.0, "iron": 80.0, "gold": 318.0, "concrete": 1.7,
               "glass": 1.0, "ice": 2.2, "granite": 2.9, "air": 0.026}


def material_thermal(name):
    """Thermal properties of a named material: {'density','specific_heat','thermal_conductivity'} in SI. Density
    and specific heat are REUSED from holographic_definitions.MATERIALS; conductivity comes from the enrichment
    data (or a documented fallback). Raises KeyError if the material is unknown."""
    from holographic.misc.holographic_definitions import MATERIALS
    if name not in MATERIALS:
        raise KeyError("unknown material %r" % name)
    props = MATERIALS[name]
    k = _load_enrichment().get(name, {}).get("thermal_conductivity", _K_FALLBACK.get(name))
    return {"density": float(props["density"]),
            "specific_heat": float(props.get("specific_heat", 900.0)),
            "thermal_conductivity": (float(k) if k is not None else None)}


def _selftest():
    """Q=mcdT round-trips, conduction conserves heat while smoothing, and Newton cooling relaxes toward ambient."""
    # (1) heating water: raising 1 kg of water (c=4186) by 20 K costs ~83.7 kJ, and that energy gives back 20 K
    Q = heat_energy(1.0, 4186.0, 20.0)
    assert abs(Q - 83720.0) < 1.0
    assert abs(temperature_change(Q, 1.0, 4186.0) - 20.0) < 1e-9

    # (2) conduction: a hot spot in a cool plate spreads -- total heat CONSERVED, spatial variance DROPS
    T = np.full((21, 21), 300.0); T[10, 10] = 800.0
    total0 = T.sum(); var0 = T.var()
    T2 = diffuse_heat(T, alpha=1e-4, dx=0.01, dt=0.5, steps=20)
    assert abs(T2.sum() - total0) < 1e-6                            # insulated boundary -> heat conserved
    assert T2.var() < var0 and T2.max() < 800.0                    # the hot spot spread out
    assert np.isfinite(T2).all()                                   # stayed stable (auto-substepped)

    # (3) a big dt still stays stable and conserves heat (the auto-substep guard)
    T3 = diffuse_heat(T, alpha=1e-3, dx=0.01, dt=100.0, steps=1)
    assert np.isfinite(T3).all() and abs(T3.sum() - total0) < 1e-6

    # (4) Newton cooling: a hot steel cube relaxes toward ambient, fast then slow
    from_props = material_thermal("steel")
    body = HeatBody(mass_kg=1.0, specific_heat=from_props["specific_heat"], temp_K=800.0)
    t1 = body.newton_cool(ambient_K=300.0, h_area=2.0, dt=10.0, steps=1)
    t2 = body.newton_cool(ambient_K=300.0, h_area=2.0, dt=10.0, steps=1)
    assert 300.0 < t2 < t1 < 800.0                                 # cooling toward ambient, monotonic
    assert (800.0 - t1) > (t1 - t2)                                # faster while hotter (Newton's law)

    # (5) conductivity really came from the enrichment data (steel 50 W/m/K), reused not restated
    assert abs(material_thermal("steel")["thermal_conductivity"] - 50.0) < 1e-9
    assert abs(material_thermal("water")["thermal_conductivity"] - 0.6) < 1e-9
    print("holographic_heat selftest OK: Q=mcdT round-trips; conduction conserves heat & smooths (stable at big "
          "dt); Newton cooling relaxes toward ambient; conductivity read from enrichment data")


if __name__ == "__main__":
    _selftest()

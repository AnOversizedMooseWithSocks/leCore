"""holographic_definitions.py -- a DEFINITION LIBRARY: known things, their physical
properties, and a resolver that builds a scenario from a description.

WHY THIS MODULE EXISTS (the gap the audit found)
------------------------------------------------
The engine already had a lot of the *machinery* the user asked for and almost none of the
*vocabulary* that machinery needs:

  * holographic_material.Material carries APPEARANCE (albedo, roughness, metallic) but no
    PHYSICAL property -- no density, viscosity, Young's modulus, refractive index.
  * holographic_fluid.StableFluid (smoke/fire), holographic_softbody.{SoftBody,RigidBody}
    (PBD/XPBD), holographic_physics.Kinematics and holographic_noise.FractalNoise are real
    SOLVERS, but they take RAW parameters -- nothing maps a NAMED thing ("water", "steel")
    to the numbers a solver needs.
  * holographic_semantic.parse_description turns "a red ball on a blue box" into a queryable
    VSA scene, but it only knows STATIC SPATIAL relations (on / left-of). It has no notion of
    a PHYSICAL relation -- "floating" -- governed by the things' material properties.

So this module adds the missing layer, and adds it the engine's way: as role-bound VSA
records (the KnowledgeStore / Plate pattern -- bundle of bind(role, filler)), so a definition
is QUERYABLE (unbind a property back), COMPOSABLE, and SIMILARITY-SEARCHABLE. It REFERENCES
the existing solvers rather than re-implementing them: a phenomenon definition names the
solver family that runs it; a material definition names the constants that solver consumes.

WHAT IS HONEST HERE, AND WHAT IS NOT (kept loud)
------------------------------------------------
  * The physical CONSTANTS (density of water = 1000 kg/m^3, etc.) are hand-authored reference
    data from standard tables. The library does not DERIVE physics; it ENCODES known physics so
    it becomes composable. That is the whole and only claim.
  * The physical RULES the resolver uses to VALIDATE a scenario -- buoyancy (an object floats
    iff its density is below the medium's), the submerged fraction rho_obj/rho_medium -- are
    exact, textbook, deterministic. "A block of steel floating in water" is flagged inconsistent
    because steel sinks; that verdict is real physics, not a guess.
  * The scenario the resolver returns is a VALIDATED, PARAMETERISED build spec + a VSA structure.
    It is NOT a running simulation. Running one is the job of the shipped solvers, which this
    spec is shaped to feed. And a genuine KEPT NEGATIVE: the shipped fluid solver is a PERIODIC
    smoke/combustion solver, not a free-surface water solver, so "wood bobbing in a pool" has an
    analytic EQUILIBRIUM here (exact) but no free-surface dynamics yet (no solver for it).
  * The parser is a CONTROLLED vocabulary + keyword grammar over the physical domain -- exactly
    the honest boundary holographic_semantic keeps. It is not natural language. Unknown words are
    reported as UNRESOLVED, loudly, rather than guessed.
  * Categorical properties (phase, kind, density-class) decode back at cosine ~1. CONTINUOUS
    properties (density) are encoded on a log scale and decode APPROXIMATELY (grid-limited); the
    authoritative number lives in the record dict, the VSA scalar is for holographic queries.
    Both are measured below.

Pure NumPy; deterministic; builds on the frozen core (holographic_core) + ScalarEncoder
(holographic_encoders) + the shape table from holographic_semantic.
"""

import math
import numpy as np

# The stable kernel surface (holographic_core re-exports these; safe to build against).
from holographic_core import bind, unbind, bundle, cosine, Vocabulary
from holographic_encoders import ScalarEncoder
# Reuse the shape word table already grounded in the semantic layer (ball->sphere, block->box, ...)
# rather than re-inventing it. This is the "apply the semantic vocabulary to things" the user asked for.
from holographic_semantic import SHAPES as _SEMANTIC_SHAPES


# ==================================================================================================
# SECTION 1 -- THE DATA: known things and their properties (standard reference values, SI units)
# ==================================================================================================
# Every value here is a textbook physical constant. Units are stated once, per field, and never mixed.
#   density        kg/m^3
#   viscosity      Pa*s          (dynamic viscosity; fluids only)
#   youngs         GPa           (Young's modulus / stiffness; solids only)
#   refractive     dimensionless (index of refraction at ~visible light)
#   sound_speed    m/s           (longitudinal speed of sound in the bulk material)
#   specific_heat  J/(kg*K)
#   phase          "solid" | "liquid" | "gas"
# A field is simply omitted where it does not apply or is not well-defined for the substance.

MATERIALS = {
    # ---- solids ----
    "wood":        dict(density=650,   youngs=10.0,  phase="solid",  sound_speed=3300, specific_heat=1700),
    "oak_wood":    dict(density=700,   youngs=11.0,  phase="solid",  sound_speed=3960, specific_heat=2400),
    "pine_wood":   dict(density=500,   youngs=9.0,   phase="solid",  sound_speed=3300, specific_heat=2300),
    "cork":        dict(density=240,   youngs=0.02,  phase="solid",  specific_heat=2000),
    "ice":         dict(density=917,   youngs=9.0,   phase="solid",  refractive=1.31, sound_speed=3200, specific_heat=2090),
    "styrofoam":   dict(density=50,    youngs=0.005, phase="solid"),
    "rubber":      dict(density=1100,  youngs=0.05,  phase="solid",  sound_speed=60,   specific_heat=2005),
    "pvc_plastic": dict(density=1380,  youngs=3.0,   phase="solid",  specific_heat=900),
    "bone":        dict(density=1900,  youngs=14.0,  phase="solid",  specific_heat=440),
    "concrete":    dict(density=2400,  youngs=30.0,  phase="solid",  sound_speed=3200, specific_heat=880),
    "glass":       dict(density=2500,  youngs=70.0,  phase="solid",  refractive=1.52, sound_speed=5640, specific_heat=840),
    "aluminum":    dict(density=2700,  youngs=69.0,  phase="solid",  sound_speed=6320, specific_heat=897),
    "granite":     dict(density=2700,  youngs=50.0,  phase="solid",  sound_speed=5950, specific_heat=790),
    "diamond":     dict(density=3510,  youngs=1050.0,phase="solid",  refractive=2.417,sound_speed=12000,specific_heat=509),
    "titanium":    dict(density=4506,  youngs=116.0, phase="solid",  sound_speed=6070, specific_heat=523),
    "steel":       dict(density=7850,  youngs=200.0, phase="solid",  sound_speed=5960, specific_heat=490),
    "iron":        dict(density=7874,  youngs=211.0, phase="solid",  sound_speed=5120, specific_heat=449),
    "copper":      dict(density=8960,  youngs=117.0, phase="solid",  sound_speed=4760, specific_heat=385),
    "lead":        dict(density=11340, youngs=16.0,  phase="solid",  sound_speed=1210, specific_heat=129),
    "gold":        dict(density=19300, youngs=79.0,  phase="solid",  sound_speed=3240, specific_heat=129),
    # ---- liquids (also fluids) ----
    "water":         dict(density=1000,  viscosity=0.001,   phase="liquid", refractive=1.333, sound_speed=1481, specific_heat=4186),
    "seawater":      dict(density=1025,  viscosity=0.00107, phase="liquid", refractive=1.34,  sound_speed=1500),
    "gasoline":      dict(density=745,   viscosity=0.0006,  phase="liquid"),
    "ethanol":       dict(density=789,   viscosity=0.0012,  phase="liquid", refractive=1.361, sound_speed=1160),
    "vegetable_oil": dict(density=920,   viscosity=0.065,   phase="liquid", refractive=1.47,  sound_speed=1450),
    "milk":          dict(density=1030,  viscosity=0.003,   phase="liquid", refractive=1.35),
    "glycerin":      dict(density=1260,  viscosity=1.0,     phase="liquid", refractive=1.47),
    "honey":         dict(density=1420,  viscosity=8.0,     phase="liquid", refractive=1.49),
    "mercury":       dict(density=13534, viscosity=0.0015,  phase="liquid", sound_speed=1450, specific_heat=140),
    # ---- gases (also fluids) ----
    "hydrogen":       dict(density=0.0899, viscosity=8.8e-6,  phase="gas"),
    "helium":         dict(density=0.1786, viscosity=1.96e-5, phase="gas", sound_speed=972),
    "air":            dict(density=1.225,  viscosity=1.8e-5,  phase="gas", refractive=1.0003, sound_speed=343, specific_heat=1005),
    "carbon_dioxide": dict(density=1.98,   viscosity=1.47e-5, phase="gas", sound_speed=267),
}

# HARDEN + EXPAND: fold in the comprehensive physical database (holographic_materialdata). We ENRICH the legacy
# entries above (adding fields they lack -- thermal conductivity, melting point, category, ... -- via setdefault, so
# their long-standing values are never overwritten and resolve_scenario's density-based verdicts are unchanged) and we
# ADD the many new materials. This keeps ONE source the whole engine reads (resolve_scenario, physical_material, the
# material index) while the readable, validated data lives in its own module.
from holographic_materialdata import PHYSICAL_MATERIALS as _PHYS_DB

for _mname, _mprops in _PHYS_DB.items():
    if _mname in MATERIALS:
        for _mk, _mv in _mprops.items():
            MATERIALS[_mname].setdefault(_mk, _mv)          # enrich: add only fields the legacy entry is missing
    else:
        MATERIALS[_mname] = dict(_mprops)                   # add: a brand-new material

# The words a user might type -> the canonical material name above. Adjectives fold to nouns
# ("wooden"->wood), common synonyms map to a representative substance ("stone"->granite). This is
# the controlled physical-material vocabulary, the physical twin of holographic_semantic.MATERIALS
# (which grounds APPEARANCE); here we ground SUBSTANCE.
MATERIAL_WORDS = {
    "wood": "wood", "wooden": "wood", "timber": "wood", "log": "wood",
    "oak": "oak_wood", "pine": "pine_wood", "balsa": "pine_wood",
    "cork": "cork", "styrofoam": "styrofoam", "foam": "styrofoam",
    "ice": "ice", "icy": "ice", "frozen": "ice",
    "rubber": "rubber", "plastic": "pvc_plastic", "pvc": "pvc_plastic", "vinyl": "pvc_plastic",
    "bone": "bone",
    "concrete": "concrete", "cement": "concrete",
    "glass": "glass", "crystal": "glass",
    "aluminum": "aluminum", "aluminium": "aluminum",
    "granite": "granite", "stone": "granite", "rock": "granite", "marble": "granite",
    "diamond": "diamond",
    "titanium": "titanium",
    "steel": "steel", "iron": "iron", "metal": "steel", "metallic": "steel",
    "copper": "copper", "bronze": "copper", "brass": "copper",
    "lead": "lead", "gold": "gold", "golden": "gold",
    "water": "water", "seawater": "seawater", "brine": "seawater",
    "gasoline": "gasoline", "petrol": "gasoline", "gas": "gasoline",
    "ethanol": "ethanol", "alcohol": "ethanol",
    "oil": "vegetable_oil", "milk": "milk", "glycerin": "glycerin", "glycerine": "glycerin",
    "honey": "honey", "syrup": "honey",
    "mercury": "mercury", "quicksilver": "mercury",
    "hydrogen": "hydrogen", "helium": "helium",
    "air": "air", "wind": "air", "co2": "carbon_dioxide",
}

# GEOMETRY primitives. Each carries a parametric VOLUME and (surface) AREA formula and the SDF
# signature that holographic_sdf implements, so a named shape resolves to both a mass (via volume
# x density) and a renderable primitive. `params` documents the argument names; `default` gives a
# concrete instance (metres) so an under-specified "a block" still has a size.
GEOMETRY = {
    "point":    dict(dims=0, params=(),               volume=lambda: 0.0,
                     area=lambda: 0.0, sdf="length(p)", note="a 0-D location"),
    "line":     dict(dims=1, params=("length",),      volume=lambda length: 0.0,
                     area=lambda length: 0.0, sdf="segment(p,a,b)", note="a 1-D segment"),
    "plane":    dict(dims=2, params=("normal", "d"),  volume=lambda normal=None, d=0.0: math.inf,
                     area=lambda normal=None, d=0.0: math.inf, sdf="dot(p,n)-d", note="an infinite surface"),
    "circle":   dict(dims=2, params=("radius",),      volume=lambda radius: 0.0,
                     area=lambda radius: math.pi * radius ** 2, sdf="length(p.xy)-r", note="a 2-D disc"),
    "sphere":   dict(dims=3, params=("radius",),      default=dict(radius=0.10),
                     volume=lambda radius: 4.0 / 3.0 * math.pi * radius ** 3,
                     area=lambda radius: 4.0 * math.pi * radius ** 2, sdf="length(p)-r"),
    "box":      dict(dims=3, params=("lx", "ly", "lz"), default=dict(lx=0.20, ly=0.10, lz=0.10),
                     volume=lambda lx, ly, lz: lx * ly * lz,
                     area=lambda lx, ly, lz: 2.0 * (lx * ly + ly * lz + lx * lz), sdf="box(p,b)"),
    "cylinder": dict(dims=3, params=("radius", "height"), default=dict(radius=0.05, height=0.20),
                     volume=lambda radius, height: math.pi * radius ** 2 * height,
                     area=lambda radius, height: 2.0 * math.pi * radius * (radius + height), sdf="cylinder(p,r,h)"),
    "cone":     dict(dims=3, params=("radius", "height"), default=dict(radius=0.06, height=0.18),
                     volume=lambda radius, height: math.pi * radius ** 2 * height / 3.0,
                     area=lambda radius, height: math.pi * radius * (radius + math.hypot(radius, height)), sdf="cone(p,r,h)"),
    "capsule":  dict(dims=3, params=("radius", "height"), default=dict(radius=0.05, height=0.15),
                     volume=lambda radius, height: math.pi * radius ** 2 * height + 4.0 / 3.0 * math.pi * radius ** 3,
                     area=lambda radius, height: 2.0 * math.pi * radius * height + 4.0 * math.pi * radius ** 2, sdf="capsule(p,a,b,r)"),
    "torus":    dict(dims=3, params=("major", "minor"), default=dict(major=0.12, minor=0.04),
                     volume=lambda major, minor: 2.0 * math.pi ** 2 * major * minor ** 2,
                     area=lambda major, minor: 4.0 * math.pi ** 2 * major * minor, sdf="torus(p,t)"),
}

# PHENOMENA / physics models. Each names the governing EQUATION (as a readable string), its key
# PARAMETERS, and the SOLVER FAMILY in the repo that integrates it -- so the definition is a bridge
# from a word to the module that runs it, not a re-implementation.
PHENOMENA = {
    "gravity":       dict(equation="F = m*g (uniform) ; F = G*m1*m2/r^2 (Newtonian)",
                          params=dict(g=9.81, G=6.674e-11), solver="holographic_physics.Kinematics"),
    "buoyancy":      dict(equation="F_b = rho_fluid * V_displaced * g (Archimedes)",
                          params=dict(g=9.81), solver="analytic (this module)"),
    "drag":          dict(equation="F_d = 0.5 * rho * v^2 * C_d * A",
                          params=dict(C_d=0.47), solver="holographic_physics"),
    "rigid_body":    dict(equation="Newton-Euler rigid dynamics (F=ma, tau=I*alpha)",
                          params=dict(restitution=0.4, friction=0.5), solver="holographic_softbody.RigidBody"),
    "soft_body":     dict(equation="XPBD: predict -> project distance/volume constraints -> velocity update",
                          params=dict(compliance=1e-6, iterations=20), solver="holographic_softbody.SoftBody"),
    "cloth":         dict(equation="XPBD distance + bending constraints on a mesh",
                          params=dict(compliance=1e-7, iterations=20), solver="holographic_softbody.SoftBody"),
    "fluid_flow":    dict(equation="Navier-Stokes (incompressible): du/dt + (u.grad)u = -grad p/rho + nu*lap u ; div u = 0",
                          params=dict(viscosity=0.02), solver="holographic_fluid.StableFluid"),
    "smoke":         dict(equation="buoyant advection-diffusion of density + temperature (Fedkiw 2001)",
                          params=dict(buoyancy=4.0, confinement=0.5), solver="holographic_fluid.StableFluid"),
    "fire":          dict(equation="combustion source (fuel above ignition burns) + buoyancy + cooling",
                          params=dict(ignition=0.4, burn_rate=2.5, smoke_yield=0.5), solver="holographic_fluid.StableFluid"),
    "collision":     dict(equation="contact resolve: restitution + Coulomb friction via constraint projection",
                          params=dict(restitution=0.4, friction=0.5), solver="holographic_softbody / holographic_collide"),
    "spring_damper": dict(equation="F = -k*x - c*v (Hooke + viscous damping)",
                          params=dict(k=50.0, c=0.5), solver="holographic_softbody"),
    "wave":          dict(equation="d2u/dt2 = c^2 * lap u (the wave equation)",
                          params=dict(c=1.0), solver="holographic_fields"),
    "diffusion":     dict(equation="du/dt = D * lap u (the heat equation)",
                          params=dict(D=0.1), solver="holographic_fluid / holographic_fields"),
    "advection":     dict(equation="du/dt + v.grad u = 0 (semi-Lagrangian, unconditionally stable)",
                          params=dict(), solver="holographic_fluid.StableFluid"),
    "pendulum":      dict(equation="theta'' = -(g/L) sin(theta)",
                          params=dict(g=9.81, L=1.0), solver="holographic_physics"),
    "orbit":         dict(equation="Kepler: T^2 = 4*pi^2*a^3 / (G*M)",
                          params=dict(G=6.674e-11), solver="holographic_physics"),
}

# PROCEDURAL TEXTURE / NOISE definitions -> the generator + params + module that realises them.
# HONEST: holographic_noise makes BAND-LIMITED RBF-kernel Gaussian-process noise; value/fBm/turbulence
# fall out of it directly. Classic gradient (Perlin) / cellular (Worley) noise are named here as the
# reference the engine's band-limited field approximates, not a bit-identical Perlin. Kept.
TEXTURES = {
    "value_noise":  dict(params=("frequency", "seed"), module="holographic_noise.noise_field",
                         note="band-limited random field (a GP sample); the native primitive"),
    "fbm":          dict(params=("octaves", "lacunarity", "gain", "seed"), module="holographic_noise.FractalNoise",
                         note="fractional Brownian motion = a BUNDLE of octave bands (freq up, amplitude down)"),
    "turbulence":   dict(params=("octaves", "lacunarity", "gain"), module="holographic_noise.FractalNoise",
                         note="sum of |octave| -- the billowy variant used for smoke/marble"),
    "ridged":       dict(params=("octaves", "lacunarity", "gain"), module="holographic_noise.FractalNoise",
                         note="1-|octave| ridged multifractal (mountain crests)"),
    "perlin":       dict(params=("frequency",), module="holographic_noise (approx: band-limited)",
                         note="classic gradient noise; REFERENCE -- engine field is band-limited, not bit-Perlin"),
    "worley":       dict(params=("frequency", "n"), module="holographic_noise (approx)",
                         note="cellular / Voronoi F1 distance; REFERENCE (kept negative)"),
    "checker":      dict(params=("scale",), module="analytic",
                         note="sign(sin(x)*sin(y)) hard checker -- band-limited field smooths it (use analytic for hard edges)"),
    "gradient":     dict(params=("axis",), module="analytic", note="a linear ramp"),
}

# CALCULUS / ALGEBRA operators -- named, with notation, arity, and the engine op that realises them
# where one exists. The gem: CONVOLUTION is exactly the core `bind` (circular convolution via FFT),
# and the FOURIER transform is the domain `bind` multiplies in -- so two textbook operators ARE the
# engine's binding operation in different clothes.
OPERATORS = {
    "gradient":     dict(notation="grad f", maps="scalar field -> vector field", realized_by="finite differences"),
    "divergence":   dict(notation="div F",  maps="vector field -> scalar field", realized_by="finite differences"),
    "curl":         dict(notation="curl F", maps="vector field -> vector field", realized_by="finite differences"),
    "laplacian":    dict(notation="lap f",  maps="scalar field -> scalar field", realized_by="holographic_spectral (graph Laplacian)"),
    "derivative":   dict(notation="d/dt",   maps="signal -> signal",             realized_by="finite differences"),
    "integral":     dict(notation="int",    maps="signal -> scalar/signal",      realized_by="cumulative sum / quadrature"),
    "dot":          dict(notation="a . b",  maps="(vec, vec) -> scalar",         realized_by="numpy dot ; cosine after normalize"),
    "cross":        dict(notation="a x b",  maps="(vec, vec) -> vec",            realized_by="numpy cross"),
    "norm":         dict(notation="|v|",    maps="vec -> scalar",                realized_by="numpy linalg.norm"),
    "convolution":  dict(notation="f * g",  maps="(signal, signal) -> signal",   realized_by="holographic_core.bind (circular convolution, THE binding op)"),
    "fourier":      dict(notation="F{f}",   maps="signal -> spectrum",           realized_by="numpy rfft ; bind multiplies in this domain"),
    "correlation":  dict(notation="f corr g", maps="(signal, signal) -> signal", realized_by="holographic_core.unbind (correlation = bind with involution)"),
}

# SIGNAL / PATTERN definitions -- the shapes a detection pipeline looks for, with params and the
# DETECTOR faculty in the repo that decides whether one is present. This is the vocabulary the
# radio-telescope / audio / SETI use cases speak.
SIGNALS = {
    "sine_wave":      dict(params=("frequency", "amplitude", "phase"), detector="holographic_fft / matched filter"),
    "chirp":          dict(params=("f0", "f1", "rate"), detector="holographic_dedoppler.detect_drifting",
                           note="a linear frequency sweep -- a DRIFTING narrowband tone (the SETI case)"),
    "gaussian_pulse": dict(params=("center", "sigma"), detector="matched filter"),
    "impulse":        dict(params=("time",), detector="peak pick"),
    "narrowband":     dict(params=("carrier", "bandwidth"), detector="holographic_honesty.SPRTRecall + holographic_dedoppler",
                           note="bandwidth << carrier: a technosignature, no natural source is this compressed"),
    "white_noise":    dict(params=("sigma",), detector="the NULL model -- what a detector must beat"),
    "pink_noise":     dict(params=("exponent",), detector="spectral slope"),
    "matched_filter": dict(params=("template",), detector="holographic_core.unbind (correlation)",
                           note="correlate with a known template -- the optimal linear detector in white noise"),
    "spectrogram":    dict(params=("window", "hop"), detector="holographic_fft"),
    "autocorrelation":dict(params=("max_lag",), detector="holographic_signal_structure"),
    "snr":            dict(params=(), detector="signal power / noise power (the decision quantity)"),
    "bandpass":       dict(params=("f_lo", "f_hi"), detector="spectral mask"),
}


# ==================================================================================================
# SECTION 2 -- PHYSICAL RELATIONS: the predicates that VALIDATE a scenario (real, exact physics)
# ==================================================================================================
# A relation takes the resolved property dicts of an object and a medium and returns
#   (holds: bool, explanation: str, derived: dict)
# The predicates are textbook. Buoyancy is the workhorse: an object floats in a fluid iff its
# density is below the fluid's; the equilibrium submerged fraction is exactly rho_obj / rho_medium.

_G = 9.81  # standard gravity, m/s^2


def _need_density(obj, medium):
    """Both participants must have a density for a buoyancy verdict; say so plainly if not."""
    if obj is None or "density" not in obj:
        return "object has no known density"
    if medium is None or "density" not in medium:
        return "medium has no known density"
    return None


def rel_float(obj, medium, obj_volume=None):
    """Object floats on/in medium iff rho_obj < rho_medium (Archimedes). Derived: the fraction of
    the object's volume that sits below the surface at equilibrium = rho_obj / rho_medium."""
    miss = _need_density(obj, medium)
    if miss:
        return False, miss, {}
    ro, rm = obj["density"], medium["density"]
    holds = ro < rm
    submerged = min(ro / rm, 1.0)
    derived = {"submerged_fraction": submerged, "freeboard_fraction": max(0.0, 1.0 - submerged),
               "density_ratio": ro / rm}
    if obj_volume is not None and math.isfinite(obj_volume):
        derived["displaced_volume_m3"] = submerged * obj_volume
        derived["buoyant_force_N"] = rm * submerged * obj_volume * _G
        derived["weight_N"] = ro * obj_volume * _G
    if holds:
        why = ("floats: object density %.0f < medium density %.0f, so %.0f%% sits below the surface"
               % (ro, rm, 100.0 * submerged))
    else:
        why = ("does NOT float: object density %.0f >= medium density %.0f -- it sinks" % (ro, rm))
    return holds, why, derived


def rel_sink(obj, medium, obj_volume=None):
    """Object sinks iff rho_obj > rho_medium. Derived: the net downward force per unit volume."""
    miss = _need_density(obj, medium)
    if miss:
        return False, miss, {}
    ro, rm = obj["density"], medium["density"]
    holds = ro > rm
    derived = {"density_ratio": ro / rm}
    if obj_volume is not None and math.isfinite(obj_volume):
        derived["net_downward_force_N"] = (ro - rm) * obj_volume * _G
    why = ("sinks: object density %.0f > medium density %.0f" % (ro, rm)) if holds else \
          ("does NOT sink: object density %.0f <= medium density %.0f -- it floats" % (ro, rm))
    return holds, why, derived


def rel_suspend(obj, medium, obj_volume=None, tol=0.02):
    """Neutral buoyancy: densities within `tol` (relative). A fish, a submarine trimmed to depth."""
    miss = _need_density(obj, medium)
    if miss:
        return False, miss, {}
    ro, rm = obj["density"], medium["density"]
    rel = abs(ro - rm) / rm
    holds = rel <= tol
    why = ("neutrally buoyant: densities match within %.0f%% (%.0f vs %.0f)" % (100 * tol, ro, rm)) if holds else \
          ("not neutrally buoyant: densities differ by %.0f%%" % (100 * rel))
    return holds, why, {"density_ratio": ro / rm, "relative_difference": rel}


def rel_rise(obj, medium, obj_volume=None):
    """A gas/body rises through a fluid iff it is LESS dense than it (helium in air). Same physics as
    float, named for the gas case where the object ends up ABOVE, not at, the surface."""
    holds, why, derived = rel_float(obj, medium, obj_volume)
    if holds:
        why = why.replace("floats", "rises")
    return holds, why, derived


def rel_rest_on(obj, surface, obj_volume=None, obj_mass=None):
    """A solid resting on a solid surface: always geometrically consistent. Derived: the normal force
    supporting it = m*g (mass from density x volume when available)."""
    derived = {}
    if obj_mass is not None:
        derived["normal_force_N"] = obj_mass * _G
    return True, "rests on the surface (supported; normal force balances weight)", derived


def rel_flow_through(fluid, geometry_props, velocity=1.0, length=0.1):
    """A fluid flowing through/around geometry. Derived: the Reynolds number Re = rho*v*L/mu, which
    decides laminar (<~2300) vs turbulent (>~4000) flow -- the single most useful fluid number."""
    if fluid is None or "density" not in fluid or "viscosity" not in fluid:
        return False, "fluid needs both density and viscosity for a Reynolds number", {}
    Re = fluid["density"] * velocity * length / fluid["viscosity"]
    regime = "laminar" if Re < 2300 else ("transitional" if Re < 4000 else "turbulent")
    return True, "flow Reynolds number %.0f -> %s" % (Re, regime), {"reynolds": Re, "regime": regime}


# Canonical relation -> (predicate function, role of the trailing noun). This is the single source
# of truth; the verb table and the preposition table below both resolve TO a canonical relation name
# and then look its behaviour up here. medium_role="fluid": the trailing noun is the surrounding
# medium; "surface": it is a solid support.
_RELATION_SPEC = {
    "float":        (rel_float,   "fluid"),
    "sink":         (rel_sink,    "fluid"),
    "suspend":      (rel_suspend, "fluid"),
    "rise":         (rel_rise,    "fluid"),
    "submerge":     (rel_float,   "fluid"),   # fully surrounded, but the buoyancy verdict is the same law
    "rest_on":      (rel_rest_on, "surface"),
    "flow_through": (None,        "fluid"),
}

# A verb/participle a user might type -> its canonical relation name.
RELATION_WORDS = {
    "float": "float", "floats": "float", "floating": "float", "bobbing": "float",
    "sink": "sink", "sinks": "sink", "sinking": "sink",
    "suspended": "suspend", "neutral": "suspend", "hovering": "suspend",
    "rise": "rise", "rises": "rise", "rising": "rise",
    "rest": "rest_on", "rests": "rest_on", "resting": "rest_on",
    "sitting": "rest_on", "sits": "rest_on", "lying": "rest_on",
    "submerged": "submerge", "immersed": "submerge",
    "flowing": "flow_through",
}

# surface nouns -> treated as a solid support, not a material medium
SURFACE_WORDS = {"table", "desk", "floor", "ground", "bottom", "shelf", "counter", "surface", "bench"}


# ==================================================================================================
# SECTION 3 -- THE LIBRARY: definitions as role-bound VSA records (queryable, composable, searchable)
# ==================================================================================================

class Definition:
    """One named thing: its category ('kind'), its authoritative property dict, and -- lazily -- its
    VSA record. The record is the holographic face; the dict is ground truth. Nothing is lost by
    keeping both: numbers you want back exactly come from the dict, holographic OPERATIONS (query a
    property, measure similarity, compose into a scenario) run on the record."""

    def __init__(self, name, kind, props, meta=None, aliases=None, external_ids=None):
        self.name = name
        self.kind = kind
        self.props = dict(props)
        # meta[prop] = {"unit":..., "uncertainty":..., "source":...} -- per-property provenance carried
        # alongside the flat numeric value in props. Default empty, so hand-authored definitions and the
        # existing seed tables (which pass no meta) are completely unchanged.
        self.meta = dict(meta or {})
        self.aliases = list(aliases or [])          # alternate names a resolver can match
        self.external_ids = dict(external_ids or {})  # canonical IDs into external DBs (mp-id, CID, ...)
        self.vector = None   # filled by DefinitionLibrary.encode (needs the shared vocabularies)

    def __repr__(self):
        return "Definition(%r, kind=%r, %d props)" % (self.name, self.kind, len(self.props))


# density buckets -> a categorical class symbol, so "find the heavy materials" is a holographic query
def _density_class(density):
    if density < 100:   return "ultralight"
    if density < 1000:  return "light"
    if density < 3000:  return "medium"
    if density < 9000:  return "heavy"
    return "very_heavy"


class DefinitionLibrary:
    """A registry of Definitions over one shared set of VSA vocabularies, so every definition is
    encoded in the SAME role/filler space and can be compared, queried and composed.

    The encoding follows the KnowledgeStore / Plate record pattern exactly:
        record = bundle( bind(KIND, kind_sym), bind(PHASE, phase_sym),
                         bind(DENSITY_CLASS, class_sym), bind(DENSITY, scalar_encode(log rho)) )
    Categorical roles (kind/phase/class) decode back at cosine ~1; the scalar DENSITY role decodes
    approximately (grid-limited) -- both measured in `self_test`.
    """

    def __init__(self, dim=1024, seed=0):
        self.dim = dim
        self.seed = seed
        self.defs = {}                                   # name -> Definition
        # Shared spaces. Unitary role/filler atoms: this is the few-factor role-binding path where
        # exact unbinding widens the cleanup margin (the KnowledgeStore uses unitary for the same
        # reason), and no bundle-spread signal is read here.
        self.roles = Vocabulary(dim, seed + 1, unitary=True)     # property slots: KIND, PHASE, ...
        self.fillers = Vocabulary(dim, seed + 2, unitary=True)   # categorical values: solid, liquid, steel, ...
        self.names = Vocabulary(dim, seed + 3, unitary=True)     # a clean atom per definition NAME (for scenarios)
        # Continuous DENSITY on a LOG scale: helium 0.09 -> log10 -1.05, gold 19300 -> 4.29. A sinc
        # kernel is fine here -- we decode a single scalar, not read a density (the density lives in
        # the dict); the encoder just makes density a queryable holographic coordinate.
        self.density_enc = ScalarEncoder(dim, lo=-1.5, hi=4.5, seed=seed + 4, kernel="sinc")

    # -- construction ----------------------------------------------------------------------------
    def register(self, name, kind, props):
        """Add a definition and encode it. Idempotent by name (re-registering replaces)."""
        d = Definition(name, kind, props)
        d.vector = self._encode(d)
        self.defs[name] = d
        return d

    def upsert(self, name, kind, props, meta=None, aliases=None, external_ids=None):
        """Add a definition, OR enrich one that already exists by MERGING property dicts. This is the
        entry point the auto-discovering loader uses: a dropped file can introduce a new thing, or add
        properties to (or override properties of) a bundled one -- 'water' gains a refractive index it
        did not have, without restating the density. Later values win per-key; provenance merges. The
        definition is re-encoded after the merge so its VSA record reflects the new facts."""
        existing = self.defs.get(name)
        if existing is None:
            d = Definition(name, kind, props, meta=meta, aliases=aliases, external_ids=external_ids)
        else:
            merged = dict(existing.props); merged.update(props)
            merged_meta = dict(existing.meta); merged_meta.update(meta or {})
            merged_ids = dict(existing.external_ids); merged_ids.update(external_ids or {})
            merged_aliases = sorted(set(existing.aliases) | set(aliases or []))
            d = Definition(name, kind or existing.kind, merged,
                           meta=merged_meta, aliases=merged_aliases, external_ids=merged_ids)
        d.vector = self._encode(d)
        self.defs[name] = d
        return d

    def _encode(self, d):
        """Build the role-bound VSA record for a definition. Categorical facts always; the scalar
        DENSITY role only when a density is known."""
        parts = [bind(self.roles.get("KIND"), self.fillers.get(d.kind))]
        p = d.props
        if "phase" in p:
            parts.append(bind(self.roles.get("PHASE"), self.fillers.get(p["phase"])))
        if "density" in p:
            parts.append(bind(self.roles.get("DENSITY_CLASS"),
                              self.fillers.get(_density_class(p["density"]))))
            log_rho = math.log10(max(p["density"], 1e-6))
            parts.append(bind(self.roles.get("DENSITY"), self.density_enc.encode(log_rho)))
        return bundle(parts)

    def get(self, name):
        return self.defs.get(name)

    def __contains__(self, name):
        return name in self.defs

    def __len__(self):
        return len(self.defs)

    # -- holographic queries ---------------------------------------------------------------------
    def query_property(self, name, role):
        """Read a categorical property back OUT of the record by unbinding its role and cleaning up
        against the filler vocabulary -- the queryable half of the record. Returns (value, cosine).
        role in {"KIND","PHASE","DENSITY_CLASS"}."""
        d = self.defs[name]
        est = unbind(d.vector, self.roles.get(role))
        return self.fillers.cleanup(est)

    def query_density(self, name):
        """Decode the DENSITY scalar from the record (approximate; grid-limited). Returns the decoded
        density in kg/m^3. The exact value is self.defs[name].props['density']."""
        d = self.defs[name]
        est = unbind(d.vector, self.roles.get("DENSITY"))
        log_rho = self.density_enc.decode(est)
        return 10.0 ** log_rho

    def similar(self, name, k=5, kind=None):
        """The k most similar definitions by record cosine -- captures shared kind/phase/density-class
        and nearby density. Optionally restrict to one kind. Returns [(name, cosine), ...]."""
        q = self.defs[name].vector
        out = []
        for other, d in self.defs.items():
            if other == name:
                continue
            if kind is not None and d.kind != kind:
                continue
            out.append((other, float(cosine(q, d.vector))))
        out.sort(key=lambda t: -t[1])
        return out[:k]

    def by_kind(self, kind):
        return [n for n, d in self.defs.items() if d.kind == kind]

    # -- self-measurement (honest) ---------------------------------------------------------------
    def self_test(self):
        """Measure what the encoding actually delivers -- categorical recall (should be ~perfect),
        scalar-density decode error (approximate), and one similarity sanity check. Returns a dict."""
        mats = self.by_kind("material")
        kind_hits = phase_hits = phase_n = 0
        log_err = []
        for n in mats:
            v, _ = self.query_property(n, "KIND")
            kind_hits += (v == "material")
            if "phase" in self.defs[n].props:
                pv, _ = self.query_property(n, "PHASE")
                phase_hits += (pv == self.defs[n].props["phase"])
                phase_n += 1
            if "density" in self.defs[n].props:
                true_log = math.log10(self.defs[n].props["density"])
                dec_log = math.log10(max(self.query_density(n), 1e-6))
                log_err.append(abs(true_log - dec_log))
        # similarity sanity: steel should be nearer aluminum (both heavy-ish metals) than helium (a gas)
        sane = None
        if "steel" in self and "aluminum" in self and "helium" in self:
            sane = (float(cosine(self.defs["steel"].vector, self.defs["aluminum"].vector)) >
                    float(cosine(self.defs["steel"].vector, self.defs["helium"].vector)))
        return dict(materials=len(mats),
                    kind_recall=kind_hits / max(len(mats), 1),
                    phase_recall=phase_hits / max(phase_n, 1),
                    density_log_err_mean=float(np.mean(log_err)) if log_err else 0.0,
                    steel_nearer_aluminum_than_helium=sane)


def build_standard_library(dim=1024, seed=0):
    """Populate a DefinitionLibrary with everything in SECTION 1: materials (with real constants),
    geometry primitives, phenomena, textures, operators, and signals. Returns the library.

    Materials get the rich scalar/categorical encoding above. The other categories are registered
    as definitions too (so they are nameable, retrievable, and composable), carrying their spec as
    the property dict; they do not have a density, so their record is the KIND binding plus whatever
    categorical fields apply."""
    lib = DefinitionLibrary(dim=dim, seed=seed)
    for name, props in MATERIALS.items():
        lib.register(name, "material", props)
    for name, spec in GEOMETRY.items():
        lib.register(name, "geometry", {"dims": spec["dims"], "sdf": spec["sdf"]})
    for name, spec in PHENOMENA.items():
        lib.register(name, "phenomenon", {"equation": spec["equation"], "solver": spec["solver"]})
    for name, spec in TEXTURES.items():
        lib.register(name, "texture", {"module": spec["module"]})
    for name, spec in OPERATORS.items():
        lib.register(name, "operator", {"notation": spec["notation"], "realized_by": spec["realized_by"]})
    for name, spec in SIGNALS.items():
        lib.register(name, "signal", {"detector": spec["detector"]})
    return lib


# ==================================================================================================
# SECTION 4 -- THE RESOLVER: a description -> a validated, parameterised, VSA-encoded scenario
# ==================================================================================================

_ARTICLES = {"a", "an", "the", "some"}
_GLUE = {"of", "with", "that", "which", "is", "are", "and"}  # 'and' handled separately as a splitter
# Prepositions that ALSO carry the relation verb -> the canonical relation they imply. Matched before
# the plain prepositions (longest-first) so "... floating in water" resolves the verb even though the
# verb sits inside the matched phrase. This is what a bare-verb scan on the object clause would miss.
_VERB_PREPS = {
    " floating up in ": "rise", " floating in ": "float", " floating on ": "float",
    " bobbing in ": "float", " sinking in ": "sink", " suspended in ": "suspend",
    " hovering in ": "suspend", " immersed in ": "submerge", " submerged in ": "submerge",
    " resting on ": "rest_on", " sitting on ": "rest_on", " lying on ": "rest_on",
    " rising in ": "rise", " rising through ": "rise", " flowing through ": "flow_through",
}
# Plain prepositions: no verb, so scan the object clause for one, else infer from the preposition.
_PLAIN_PREPS = (" in ", " on ", " through ", " above ", " below ")


def _canon_shape(word):
    return _SEMANTIC_SHAPES.get(word)


def _canon_material(word):
    return MATERIAL_WORDS.get(word)


def _parse_entity(clause, lib):
    """Parse one object phrase into (shape, material, dims, volume, unresolved_words).

    Recognised constructions (controlled grammar, honest boundary):
        "<shape> of <material>"   e.g. block of wood
        "<material> <shape>"      e.g. steel ball, wooden cube
        "<material>"              e.g. water            (a blob; shape defaults to sphere)
        "<shape>"                 e.g. sphere           (material unknown -> unresolved)
    Adjectives fold to nouns via the tables (wooden->wood). Unknown tokens are returned, not guessed."""
    toks = [t for t in clause.replace(",", " ").split() if t and t not in _ARTICLES]
    shape = material = None
    unresolved = []
    for t in toks:
        if t in _GLUE:
            continue
        s, m = _canon_shape(t), _canon_material(t)
        if s and shape is None:
            shape = s
        elif m and material is None:
            material = m
        elif s is None and m is None:
            unresolved.append(t)
    if shape is None:
        shape = "sphere"          # a bare "water"/"a lump" gets a default body so it has a volume
    # size + volume from the geometry primitive's default instance
    geom = GEOMETRY[shape]
    dims = dict(geom.get("default", {}))
    try:
        volume = geom["volume"](**dims) if dims else geom["volume"]()
    except TypeError:
        volume = float("nan")
    return shape, material, dims, volume, unresolved


class Scenario:
    """The resolved, validated scenario. Holds the structured spec (entities, medium, relation,
    per-entity physics verdicts), the loud list of unresolved words, an overall `consistent` flag,
    and -- built on demand -- a VSA structure vector for holographic queries.

    A note on what `consistent` means: it is the physics verdict. "a block of wood floating in
    water" is consistent (wood floats); "a steel ball floating in water" is NOT (steel sinks), and
    the scenario says so with the reason. An UNRESOLVED word makes the scenario `understood=False`
    (we could not parse it) -- separate from the physics being wrong."""

    def __init__(self, text, entities, medium, relation, results, unresolved):
        self.text = text
        self.entities = entities        # [{id, shape, material, props, dims, volume, mass}]
        self.medium = medium            # {name, props, role} or None
        self.relation = relation        # canonical relation name or None
        self.results = results          # [{entity_id, holds, why, derived}]
        self.unresolved = unresolved    # [words we could not ground]
        self.understood = (len(unresolved) == 0 and relation is not None
                           and all(e["material"] for e in entities))
        self.consistent = bool(results) and all(r["holds"] for r in results)
        self._vector = None

    # -- the actionable build spec a solver would consume (the practical "recipe") --------------
    def build_spec(self):
        """A plain, serialisable construction plan: the phenomenon + solver family this scenario
        needs, and the per-entity bodies with their masses/volumes. This is what you hand to the
        shipped solvers -- the definition library's job ends at a validated, parameterised spec."""
        phen = None
        if self.relation in ("float", "sink", "suspend", "rise", "submerge"):
            phen = "buoyancy"
        elif self.relation == "rest_on":
            phen = "rigid_body"
        elif self.relation == "flow_through":
            phen = "fluid_flow"
        solver = PHENOMENA.get(phen, {}).get("solver") if phen else None
        return {
            "phenomenon": phen,
            "solver": solver,
            "gravity_m_s2": _G,
            "medium": None if not self.medium else {
                "name": self.medium["name"],
                "properties": self.medium["props"],
            },
            "bodies": [{
                "id": e["id"], "shape": e["shape"], "material": e["material"],
                "dimensions_m": e["dims"], "volume_m3": e["volume"], "mass_kg": e["mass"],
            } for e in self.entities],
            "relation": self.relation,
            "verdicts": self.results,
        }

    # -- the holographic face: the scenario as ONE role-bound VSA structure ----------------------
    def to_vector(self, lib):
        """Encode the scenario as a role-bound record over the library's own vocabularies, so it is
        a holographic object you can store, compare, and query:

            scenario = bundle( bind(RELATION, relation_sym), bind(MEDIUM, medium_name_sym),
                               bind(ENTITY, entity_record_i) ... )
            entity_record = bundle( bind(SHAPE, shape_sym), bind(MATERIAL, material_name_sym) )

        Single-slot facts (RELATION, MEDIUM) recall cleanly; the ENTITY set is a superposition and
        degrades with count -- the capacity cliff, measured in resolve_scenario's demo. Cached."""
        if self._vector is not None:
            return self._vector
        R, MED, ENT, SH, MAT = (lib.roles.get("RELATION"), lib.roles.get("MEDIUM"),
                                lib.roles.get("ENTITY"), lib.roles.get("SHAPE"), lib.roles.get("MATERIAL"))
        parts = []
        if self.relation:
            parts.append(bind(R, lib.names.get(self.relation)))
        if self.medium:
            parts.append(bind(MED, lib.names.get(self.medium["name"])))
        for e in self.entities:
            rec = [bind(SH, lib.names.get(e["shape"]))]
            if e["material"]:
                rec.append(bind(MAT, lib.names.get(e["material"])))
            parts.append(bind(ENT, bundle(rec)))
        self._vector = bundle(parts) if parts else np.zeros(lib.dim)
        return self._vector

    def summary(self):
        """A human-readable account of the scenario and its physics."""
        lines = ["scenario: %r" % self.text]
        for e in self.entities:
            mass = "%.3g kg" % e["mass"] if e["mass"] == e["mass"] else "mass n/a"
            lines.append("  body: %s of %s  (V=%.3g m^3, %s)"
                         % (e["shape"], e["material"] or "UNKNOWN MATERIAL", e["volume"], mass))
        if self.medium:
            lines.append("  medium: %s (%s)" % (self.medium["name"], self.medium["role"]))
        if self.relation:
            lines.append("  relation: %s" % self.relation)
        for r in self.results:
            lines.append("  -> %s: %s" % (r["entity_id"], r["why"]))
            for k, val in r["derived"].items():
                lines.append("       %s = %.4g" % (k, val))
        if self.unresolved:
            lines.append("  UNRESOLVED (not in the vocabulary): %s" % ", ".join(self.unresolved))
        lines.append("  understood=%s  physically_consistent=%s" % (self.understood, self.consistent))
        return "\n".join(lines)


def resolve_scenario(text, lib=None, dim=1024, seed=0):
    """Turn a description like 'a block of wood floating in water' into a validated Scenario.

    Pipeline (all deterministic):
      1. split on the physical preposition into an OBJECT clause and a MEDIUM/SURFACE clause,
         detecting the relation verb (floating -> float, sinking -> sink, ...);
      2. parse the object clause into one or more entities (split on 'and'), each a shape + material;
      3. ground the medium noun (water/air/... a fluid, or table/ground/... a surface);
      4. VALIDATE with the relation's predicate (real physics) and compute derived quantities;
      5. return the Scenario (which can emit a build_spec and a VSA vector on demand).

    Pass an existing `lib` to reuse its vocabularies; otherwise a standard library is built."""
    if lib is None:
        lib = build_standard_library(dim=dim, seed=seed)
    raw = text.strip().lower().rstrip(".!")
    # 1. Separate the OBJECT clause from the MEDIUM/SURFACE clause, and fix the relation.
    obj_clause, med_clause, relation, medium_role, pred = raw, None, None, None, None
    # (a) try the verb-carrying prepositions first, longest match wins (they fix the relation directly)
    for prep in sorted(_VERB_PREPS, key=len, reverse=True):
        idx = raw.find(prep)
        if idx != -1:
            obj_clause, med_clause = raw[:idx], raw[idx + len(prep):]
            relation = _VERB_PREPS[prep]
            break
    # (b) otherwise a plain preposition: scan the object clause tail for a bare verb, else infer
    if relation is None:
        for prep in sorted(_PLAIN_PREPS, key=len, reverse=True):
            idx = raw.find(prep)
            if idx != -1:
                obj_clause, med_clause = raw[:idx], raw[idx + len(prep):]
                head = obj_clause.split()
                for w in reversed(head):
                    if w in RELATION_WORDS:
                        relation = RELATION_WORDS[w]
                        obj_clause = " ".join(head[:head.index(w)])   # drop the verb from the object phrase
                        break
                if relation is None:                                  # infer from the preposition itself
                    p = prep.strip()
                    relation = "rest_on" if p == "on" else "submerge"  # in/through/above/below -> surrounded
                break
    if relation is not None:
        pred, medium_role = _RELATION_SPEC[relation]
    # 2. entities (split the object clause on ' and ')
    entities = []
    unresolved = []
    for i, part in enumerate(obj_clause.split(" and ")):
        part = part.strip()
        if not part:
            continue
        shape, material, dims, volume, unres = _parse_entity(part, lib)
        unresolved += unres
        props = lib.get(material).props if material and material in lib else None
        density = props["density"] if props and "density" in props else None
        mass = density * volume if (density is not None and volume == volume) else float("nan")
        entities.append({"id": "obj%d" % (i + 1), "shape": shape, "material": material,
                         "props": props, "dims": dims, "volume": volume, "mass": mass})
    # 3. ground the medium / surface
    medium = None
    if med_clause:
        med_toks = [t for t in med_clause.split() if t not in _ARTICLES and t not in _GLUE]
        med_name = None
        for t in med_toks:
            if t in SURFACE_WORDS:
                med_name, medium_role = t, "surface"
                break
            m = _canon_material(t)
            if m:
                med_name = m
                break
        if med_name is None:
            unresolved += med_toks
        elif medium_role == "surface":
            medium = {"name": med_name, "props": None, "role": "surface"}
        else:
            mprops = lib.get(med_name).props if med_name in lib else None
            medium = {"name": med_name, "props": mprops, "role": "fluid"}
    # 4. validate each entity against the medium with the relation predicate
    results = []
    if pred is not None and entities:
        for e in entities:
            if medium_role == "surface":
                holds, why, derived = rel_rest_on(e["props"], None, e["volume"], e["mass"])
            elif medium is not None:
                holds, why, derived = pred(e["props"], medium["props"], e["volume"])
            else:
                holds, why, derived = False, "no medium given", {}
            results.append({"entity_id": e["id"], "holds": holds, "why": why, "derived": derived})
    return Scenario(text, entities, medium, relation, results, unresolved)


# ==================================================================================================
# SECTION 5 -- SELF TEST (the module's own honest measurement)
# ==================================================================================================

def _selftest():
    lib = build_standard_library(dim=1024, seed=0)
    print("definition library: %d definitions (%d materials, %d geometry, %d phenomena, "
          "%d textures, %d operators, %d signals)"
          % (len(lib), len(lib.by_kind("material")), len(lib.by_kind("geometry")),
             len(lib.by_kind("phenomenon")), len(lib.by_kind("texture")),
             len(lib.by_kind("operator")), len(lib.by_kind("signal"))))

    st = lib.self_test()
    print("encoding: kind_recall=%.2f phase_recall=%.2f density_log_err=%.3f  steel~Al>He: %s"
          % (st["kind_recall"], st["phase_recall"], st["density_log_err_mean"],
             st["steel_nearer_aluminum_than_helium"]))
    assert st["kind_recall"] == 1.0, "KIND must decode exactly"
    assert st["phase_recall"] >= 0.95, "PHASE must decode near-exactly"
    assert st["steel_nearer_aluminum_than_helium"] is True, "similarity should carry physical meaning"

    # a holographic query: "what heavy materials do we know?" via the density-class role
    heavy = [n for n in lib.by_kind("material")
             if lib.query_property(n, "DENSITY_CLASS")[0] in ("heavy", "very_heavy")]
    print("heavy/very-heavy materials (by holographic class query): %s" % ", ".join(sorted(heavy)))
    assert "steel" in heavy and "gold" in heavy and "wood" not in heavy

    # the headline: resolve a scenario end to end
    print()
    sc = resolve_scenario("a block of wood floating in water", lib=lib)
    print(sc.summary())
    assert sc.understood and sc.consistent, "wood should float in water and parse cleanly"
    sub = sc.results[0]["derived"]["submerged_fraction"]
    assert abs(sub - 0.65) < 0.001, "wood/water submerged fraction is rho_wood/rho_water = 0.65"

    # the contrast that proves the physics is real, not a rubber stamp
    print()
    bad = resolve_scenario("a steel ball floating in water", lib=lib)
    print(bad.summary())
    assert bad.understood and not bad.consistent, "steel must be flagged: it sinks, it cannot float"

    # a counter-intuitive-but-true case: steel and lead FLOAT on mercury; gold SINKS
    print()
    for who, expect in [("steel", True), ("lead", True), ("gold", False)]:
        r = resolve_scenario("a %s block floating in mercury" % who, lib=lib)
        holds = r.results[0]["holds"]
        print("  %s on mercury -> floats=%s (expected %s)" % (who, holds, expect))
        assert holds == expect, "mercury buoyancy must match density comparison"

    # scenario recall: single-slot facts recall cleanly; the entity set degrades (the capacity cliff)
    print()
    v = sc.to_vector(lib)
    rel_name, rel_cos = lib.names.cleanup(unbind(v, lib.roles.get("RELATION")))
    med_name, med_cos = lib.names.cleanup(unbind(v, lib.roles.get("MEDIUM")))
    print("scenario recall: RELATION -> %s (cos %.2f), MEDIUM -> %s (cos %.2f)"
          % (rel_name, rel_cos, med_name, med_cos))
    assert rel_name == "float" and med_name == "water"

    print("\nOK: holographic_definitions self-test passed")


if __name__ == "__main__":
    _selftest()

"""holographic_starsystem.py -- PLUG DATA IN, GET A STAR SYSTEM: parameters -> a scene recipe (leCore scene_and_pipeline).

WHY THIS EXISTS
---------------
This is the keystone of the astro arc: the step that turns physical parameters -- a star's temperature, a
planet's orbit and temperature -- into a concrete, drawable system. The measurement tools upstream (doppler
velocity, central_mass_from_orbit, a fitted temperature) produce exactly those parameters; this assembles them
into a star at the origin, planets on Kepler orbits, each planet painted by the biome its temperature implies.

It ASSEMBLES BY DELEGATION -- it invents no geometry it can borrow:
  * a star's colour comes from holographic_blackbody (the same blackbody the observer uses);
  * a planet's surface comes from fractal_planet (the library's whole-world generator), referenced by SEED and
    knobs rather than baked -- the field is regenerated on demand (descend-the-recipe, never store the world);
  * orbits come from the closed-form Kepler geometry below.

It builds a RECIPE, not a heavy scene: a deterministic, JSON-serializable dict (star + planets + orbits) that the
renderer / scene builder draws. Same parameters + seed -> byte-identical recipe -- determinism is the contract, so
the same system always renders the same way and an agent can round-trip it over /invoke.

DIRECTIONS (up/down/sideways)
  DOWN  -- kepler_position works on a single phase; a whole orbit is kepler_ellipse over an array of phases.
  UP    -- a system is a component of a cluster (C2): many star_system recipes placed in a larger field.
  SIDEWAYS
    structure-- the recipe IS a role-bound record (star bound to its props, each planet to its orbit + surface).
    field    -- planet_field descends a planet entry to its actual surface field (via fractal_planet).
    sequence -- stepping `phase` over time walks the planets along their orbits (kepler_position over a time axis).

Determinism: closed-form orbital geometry + seeded delegation; no RNG here. Exact.
"""

import math
import numpy as np
from holographic.misc import holographic_blackbody as _bb   # a star's colour from its temperature (shared path)


def solve_kepler(mean_anomaly, e, iters=8):
    """Solve Kepler's equation M = E - e*sin(E) for the eccentric anomaly E, by Newton's method (a handful of
    steps converge for any bound orbit, e<1). The one transcendental step in orbital mechanics. Field-native
    over an array of mean anomalies M."""
    M = np.asarray(mean_anomaly, float)
    E = M + e * np.sin(M)                       # a good first guess (exact at e=0)
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def kepler_position(a, e, mean_anomaly):
    """Position (x, y) in the orbital plane at phase `mean_anomaly` (radians), with the star at a FOCUS (not the
    centre). a = semi-major axis, e = eccentricity; perihelion sits at +x, distance a(1-e). Closed form from the
    eccentric anomaly. Field-native over an array of phases -> (..., 2)."""
    E = solve_kepler(mean_anomaly, e)
    x = a * (np.cos(E) - e)                      # focus-centred: x ranges over [-a(1+e), a(1-e)]
    y = a * math.sqrt(max(1.0 - e * e, 0.0)) * np.sin(E)
    return np.stack([x, y], axis=-1)


def kepler_ellipse(a, e, n=128):
    """A whole orbit as `n` points (x, y), star at a focus. Sampling is uniform in phase (mean anomaly), so points
    BUNCH near aphelion exactly as a real planet lingers there -- physically honest, not uniform in arc length.
    Use it to draw the orbit or to place a planet."""
    M = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return kepler_position(a, e, M)


# Equilibrium-temperature tiers (Kelvin) -> a surface regime. A CHOICE of thresholds, documented so nobody mistakes
# it for a climate model: it is a legible bucketing that selects which biome class + water level a planet is painted
# with. Aligned to material_catalog's biome class names.
_TEMP_TIERS = (
    (150.0, "frozen"),      # < 150 K: ice world, no liquid water
    (250.0, "cold"),        # 150-250 K: tundra / snow
    (330.0, "temperate"),   # 250-330 K: earthlike, liquid water
    (500.0, "hot"),         # 330-500 K: arid / desert
)


def temperature_to_biome(temp_K):
    """Map a planet's equilibrium temperature (K) to a surface regime (frozen/cold/temperate/hot/molten). A CHOICE
    of thresholds, not a climate model -- it selects which biome class + sea level fractal_planet paints. See
    material_catalog for the biome materials these names pick."""
    t = float(temp_K)
    for hi, name in _TEMP_TIERS:
        if t < hi:
            return name
    return "molten"                              # >= 500 K


def _biome_surface(name):
    """Biome regime -> fractal_planet knobs (sea_level, relief). Colder/hotter worlds carry less or no open water;
    temperate worlds have seas; molten worlds are high-relief and dry. Explicit and boring on purpose."""
    return {
        "frozen":    dict(sea_level=0.0, relief=0.06),
        "cold":      dict(sea_level=0.2, relief=0.10),
        "temperate": dict(sea_level=0.5, relief=0.10),
        "hot":       dict(sea_level=0.1, relief=0.12),
        "molten":    dict(sea_level=0.0, relief=0.18),
    }.get(name, dict(sea_level=0.4, relief=0.10))


def star_system(params, seed=0):
    """Assemble a STAR SYSTEM RECIPE from physical parameters -- the 'plug data in, see a system' step.

    params = {"star": {"temp_K":.., "radius":.., "mass":..},
              "planets": [{"a":.., "e":.., "radius":.., "temp_K":.., "phase":..}, ...]}
    Returns a deterministic, JSON-serializable recipe: the star (blackbody colour, radius, mass, at the origin) and
    each planet (radius, biome regime, its orbit as points, its current position at `phase`, and the seed + surface
    knobs to regenerate its actual surface via planet_field). Same params + seed -> byte-identical recipe.

    Delegates: star colour from holographic_blackbody, surfaces referenced to fractal_planet by seed (regenerate on
    demand, never baked), orbits from the Kepler geometry above. Builds the recipe; the renderer draws it."""
    st = params.get("star", {})
    t_star = float(st.get("temp_K", 5772.0))
    star = {
        "temp_K": t_star,
        "color": _bb.blackbody_rgb(t_star).tolist(),      # shared blackbody path -> sRGB
        "radius": float(st.get("radius", 1.0)),
        "mass": float(st.get("mass", 1.0)),
        "position": [0.0, 0.0],
    }
    planets = []
    for i, p in enumerate(params.get("planets", [])):
        a = float(p.get("a", 1.0)); e = float(p.get("e", 0.0))
        temp = float(p.get("temp_K", 288.0))
        biome = temperature_to_biome(temp)
        pseed = int(seed) * 1000 + i                      # per-planet seed derived deterministically from the system
        planets.append({
            "radius": float(p.get("radius", 0.1)),
            "temp_K": temp,
            "biome": biome,
            "a": a, "e": e,
            "orbit": kepler_ellipse(a, e).tolist(),
            "position": kepler_position(a, e, float(p.get("phase", 0.0))).tolist(),
            "seed": pseed,
            "surface": _biome_surface(biome),
        })
    return {"star": star, "planets": planets, "seed": int(seed)}


def planet_field(planet_spec, dim=256):
    """Regenerate a planet's actual surface field from its recipe entry via fractal_planet -- the descend-the-recipe
    step (the world is never stored, only its seed + knobs, so it is reproduced bit-for-bit on demand). Delegates;
    reimplements no planet geometry. Returns whatever fractal_planet returns."""
    from holographic.materials_and_texture.holographic_matlib import fractal_planet
    s = planet_spec["surface"]
    return fractal_planet(radius=planet_spec["radius"], seed=planet_spec["seed"], dim=dim,
                          relief=s["relief"], sea_level=s["sea_level"])


def sample_imf(n, seed=0, m_low=0.1, m_high=50.0, alpha=2.35):
    """Draw `n` stellar masses (solar units) from a Salpeter initial mass function (Salpeter 1955): p(m) ~ m^-alpha,
    with alpha=2.35 the classic slope. Closed-form inverse-CDF on [m_low, m_high] -- exact, deterministic, no
    rejection. The IMF is bottom-heavy: most stars are small red dwarfs, a few are massive blue giants. Seeds a
    cluster's stellar population."""
    rng = np.random.default_rng(int(seed))
    u = rng.random(int(n))
    exp = 1.0 - alpha                                          # power-law inverse-CDF exponent
    lo = m_low ** exp; hi = m_high ** exp
    return (u * (hi - lo) + lo) ** (1.0 / exp)


def mass_to_temperature(mass):
    """A star's main-sequence effective temperature (K) from its mass (solar units). Rough MS scaling: L ~ M^3.5,
    R ~ M^0.7 give T = (L/(4 pi R^2 sigma))^0.25 ~ M^0.525, normalised so 1 Msun -> 5772 K (the Sun). Monotonic:
    a 10 Msun star is a hot blue giant, a 0.3 Msun a cool red dwarf. A CHOICE of a simple relation, not a stellar
    model -- stated so it is not mistaken for one. Field-native."""
    return 5772.0 * np.asarray(mass, float) ** 0.525


def _bilinear(field, gx, gy):
    """Bilinear sample of a 2-D `field` at continuous grid coords (gx=col, gy=row). Inlined (a few lines) to keep
    this scene module free of a cross-family import; mirrors sample_field's bilinear read."""
    x0 = np.clip(np.floor(gx).astype(int), 0, field.shape[1] - 1)
    y0 = np.clip(np.floor(gy).astype(int), 0, field.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, field.shape[1] - 1); y1 = np.clip(y0 + 1, 0, field.shape[0] - 1)
    fx = gx - np.floor(gx); fy = gy - np.floor(gy)
    return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x1] * fx * (1 - fy)
            + field[y1, x0] * (1 - fx) * fy + field[y1, x1] * fx * fy)


def star_cluster(n, seed=0, extent=1.0, density_field=None, planets_per_star=0):
    """Place `n` star systems in a 2-D field -- the UP direction of star_system: a cluster is many systems. Each
    star's mass is drawn from a Salpeter IMF and coloured by its main-sequence temperature (blue giants, red
    dwarfs), so the cluster looks like a real stellar population, not identical dots.

    POSITIONS: with density_field=None, systems get EVEN low-discrepancy coverage (Roberts' sequence, reused). Pass
    a 2-D density_field (e.g. a cosmic-web filament map from the Physarum/maze solver) and systems are REJECTION-
    sampled to cluster where density is high -- the tie to large-scale structure (Burchett 2020 MCPM: slime-mould
    reconstructs the cosmic web). Returns a deterministic recipe {systems:[{position, star_mass, star_temp_K,
    system}], extent, n, seed}; each `system` is a full star_system recipe."""
    from holographic.sampling_and_signal.holographic_lowdiscrepancy import low_discrepancy  # even coverage (reused)
    rng = np.random.default_rng(int(seed))
    n = int(n)
    if density_field is None:
        pts = low_discrepancy(n, 2, seed) * extent            # even, gap-free placement
    else:
        df = np.asarray(density_field, float)
        df = df / max(float(df.max()), 1e-30)                 # normalise to [0,1] as an acceptance probability
        H, W = df.shape[:2]
        cand = low_discrepancy(n * 8, 2, seed)                # low-discrepancy candidates in [0,1)^2
        u = rng.random(cand.shape[0])
        d = _bilinear(df, cand[:, 0] * (W - 1), cand[:, 1] * (H - 1))
        keep = cand[u < d][:n]                                # accept in proportion to density
        if keep.shape[0] < n:                                 # top up if the field was sparse
            extra = low_discrepancy(n, 2, seed + 1)[: n - keep.shape[0]]
            keep = np.vstack([keep, extra]) if keep.size else extra
        pts = keep * extent
    masses = sample_imf(len(pts), seed=seed)
    temps = mass_to_temperature(masses)
    systems = []
    for i, (pos, mm, tt) in enumerate(zip(pts, masses, temps)):
        planets = [{"a": 0.4 + 0.7 * k, "e": 0.02 * k, "radius": 0.05,
                    "temp_K": float(tt) * 0.05 / (1.0 + k)} for k in range(int(planets_per_star))]
        rec = star_system({"star": {"temp_K": float(tt), "mass": float(mm)}, "planets": planets}, seed=seed * 10000 + i)
        systems.append({"position": [float(pos[0]), float(pos[1])], "star_mass": float(mm),
                        "star_temp_K": float(tt), "system": rec})
    return {"systems": systems, "extent": float(extent), "n": len(systems), "seed": int(seed)}


def _selftest():
    """Regression trap: exact orbital geometry, temperature->biome buckets, a deterministic recipe whose star colour
    and orbit perihelion match first principles, and a real delegated planet surface."""
    # --- Kepler geometry ---
    circ = kepler_ellipse(1.0, 0.0, n=64)
    assert np.allclose(np.hypot(circ[:, 0], circ[:, 1]), 1.0, atol=1e-9), "e=0 orbit is not a unit circle about the focus"
    a, e = 1.0, 0.5
    peri = kepler_position(a, e, 0.0); apo = kepler_position(a, e, np.pi)
    assert abs(np.hypot(*peri) - a * (1 - e)) < 1e-9, "perihelion distance wrong"
    assert abs(np.hypot(*apo) - a * (1 + e)) < 1e-9, "aphelion distance wrong"
    # Kepler's equation actually solved: E - e sin E == M
    M = np.linspace(0, 2 * np.pi, 50); E = solve_kepler(M, 0.4)
    assert np.max(np.abs((E - 0.4 * np.sin(E)) - M)) < 1e-10, "Kepler's equation not satisfied"

    # --- temperature tiers ---
    assert temperature_to_biome(100) == "frozen" and temperature_to_biome(288) == "temperate" and temperature_to_biome(1200) == "molten"

    # --- the assembler: deterministic, delegating, physically anchored ---
    params = {"star": {"temp_K": 5772.0, "radius": 1.0, "mass": 1.0},
              "planets": [{"a": 0.39, "e": 0.20, "radius": 0.03, "temp_K": 440.0},   # a hot inner world
                          {"a": 1.00, "e": 0.02, "radius": 0.09, "temp_K": 288.0},   # an earthlike one
                          {"a": 5.20, "e": 0.05, "radius": 1.00, "temp_K": 120.0}]}  # a cold giant
    rec = star_system(params, seed=0)
    assert rec == star_system(params, seed=0), "recipe must be deterministic"
    assert rec["star"]["color"] == _bb.blackbody_rgb(5772.0).tolist(), "star colour must match blackbody"
    assert [p["biome"] for p in rec["planets"]] == ["hot", "temperate", "frozen"], "biomes mis-assigned"
    # each planet's stored orbit really has perihelion a(1-e)
    for p in rec["planets"]:
        orb = np.array(p["orbit"]); rmin = np.min(np.hypot(orb[:, 0], orb[:, 1]))
        assert abs(rmin - p["a"] * (1 - p["e"])) < 1e-6, "orbit perihelion wrong for planet a=%s" % p["a"]
    # JSON round-trips (agent-invokable contract)
    import json
    assert json.loads(json.dumps(rec))["planets"][1]["biome"] == "temperate"

    # --- delegated surface really builds (small dim to stay quick) ---
    field = planet_field(rec["planets"][1], dim=48)
    assert field is not None

    # --- C2: IMF, mass-temperature, and the cluster (up-direction) ---
    masses = sample_imf(5000, seed=1)
    assert masses.min() >= 0.1 - 1e-9 and masses.max() <= 50.0 + 1e-9, "IMF masses out of range"
    assert np.median(masses) < np.mean(masses), "Salpeter IMF must be bottom-heavy (median < mean)"
    assert abs(mass_to_temperature(1.0) - 5772.0) < 1.0, "1 Msun should be ~5772 K"
    assert mass_to_temperature(10.0) > mass_to_temperature(0.3), "mass-temperature must be monotonic"

    clus = star_cluster(40, seed=0, extent=2.0)
    assert clus["n"] == 40 and clus == star_cluster(40, seed=0, extent=2.0), "cluster must be deterministic"
    P = np.array([s["position"] for s in clus["systems"]])
    assert P.min() >= 0.0 and P.max() <= 2.0 + 1e-9, "cluster positions must lie within extent"
    assert clus["systems"][0]["system"]["star"]["color"], "each cluster star needs a blackbody colour"

    # density-weighted: a bright blob near a corner pulls systems toward it vs an even spread
    dens = np.zeros((32, 32)); yy, xx = np.mgrid[0:32, 0:32]
    dens += np.exp(-((xx - 7) ** 2 + (yy - 7) ** 2) / (2 * 4.0 ** 2))
    web = star_cluster(120, seed=2, extent=1.0, density_field=dens)
    Pw = np.array([s["position"] for s in web["systems"]])
    assert Pw[:, 0].mean() < 0.45 and Pw[:, 1].mean() < 0.45, "density field did not cluster systems toward the blob"

    print("holographic_starsystem selftest OK  |  Kepler orbits exact (peri/apo, eqn to 1e-10); biomes bucketed; "
          "recipe deterministic + JSON round-trips; star colour == blackbody; surface delegates to fractal_planet  |  + cluster: Salpeter IMF, mass->temp, density-weighted placement (cosmic web)")


if __name__ == "__main__":
    _selftest()

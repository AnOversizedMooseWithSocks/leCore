"""Tests for holographic_matlib.py -- the render material library, plain diffuse -> fractal planet.

Catalog + factory are cheap and exhaustively checked; the fractal-planet tests use small noise settings
(low dim, few octaves, coarse raster) so they stay fast while still proving surface biomes, interior
layers, ore deposits, and determinism.
"""
import numpy as np
import pytest

import holographic_matlib as ml
from holographic_matlib import material, RENDER_MATERIALS, by_class, biome_at, fractal_planet


# ---- catalog + factory ---------------------------------------------------------------------------

def test_catalog_spans_the_range():
    assert len(RENDER_MATERIALS) >= 120
    for cls in ("diffuse", "metal", "wood", "stone", "glass", "gem", "emissive",
                "biome", "layer", "deposit", "liquid", "fabric", "organic"):
        assert len(by_class(cls)) >= 1, "class %s should be populated" % cls


def test_material_factory_returns_gltf_pbr():
    m = material("matte_white")
    assert m.metallic == 0.0 and m.roughness >= 0.5           # a plain diffuse
    d = m.to_gltf_dict()                                       # round-trips to the glTF factor set
    assert "pbrMetallicRoughness" in d


def test_metals_are_metallic_and_emissives_emit():
    assert material("gold").metallic == 1.0
    assert material("chrome").roughness < 0.2
    assert material("lava").emissive[0] > 0.5
    assert material("neon_pink").emissive[0] > 0.5


def test_dielectric_transmission_carried_as_alpha():
    assert material("glass_clear").base_color[3] < 0.2        # transmits
    assert material("matte_white").base_color[3] == 1.0       # opaque


def test_unknown_material_is_loud():
    with pytest.raises(KeyError):
        material("unobtanium_shiny")


# ---- procedural + blend --------------------------------------------------------------------------

def test_noise_blend_socket_returns_per_point_rgb():
    from holographic_noise import FractalNoise
    n = FractalNoise(3, dim=96, bounds=[(-1, 1)] * 3, octaves=2, base_bandwidth=3.0, seed=1)
    sock = ml.noise_blend_albedo("marble", "slate", n)
    pts = np.random.default_rng(0).uniform(-1, 1, (12, 3))
    cols = sock(pts)
    assert cols.shape == (12, 3) and cols.min() >= 0 and cols.max() <= 1


def test_blend_presets_lerps_factors():
    worn = ml.blend_presets("gold", "rust", 0.5)
    g, r = material("gold"), material("rust")
    assert abs(worn.metallic - 0.5 * (g.metallic + r.metallic)) < 1e-6


# ---- biome classifier ----------------------------------------------------------------------------

def test_biome_cells():
    assert biome_at(-0.3, 0.8, 0.5) == "ocean_deep"
    assert biome_at(-0.05, 0.8, 0.5) == "ocean"
    assert biome_at(0.01, 0.8, 0.5) == "beach"
    assert biome_at(0.2, 0.9, 0.1) == "desert"
    assert biome_at(0.2, 0.9, 0.9) == "rainforest"
    assert biome_at(0.2, 0.5, 0.9) == "forest"
    assert biome_at(0.2, 0.05, 0.5) == "polar_ice"
    # every biome the classifier can emit is a real catalog entry
    for e in (-0.5, -0.05, 0.01, 0.2, 0.7):
        for t in (0.05, 0.3, 0.5, 0.9):
            for mo in (0.1, 0.5, 0.9):
                assert biome_at(e, t, mo) in RENDER_MATERIALS


# ---- the fractal planet (small + fast) -----------------------------------------------------------

@pytest.fixture(scope="module")
def planet():
    return fractal_planet(radius=1.0, seed=5, dim=96, octaves=2, relief=0.12)


def test_planet_surface_has_multiple_biomes(planet):
    hist = planet.biome_histogram(n=120)
    assert len(hist) >= 2
    assert all(name in RENDER_MATERIALS for name in hist)
    assert abs(sum(hist.values()) - 1.0) < 1e-6


def test_planet_interior_is_layered(planet):
    core = planet.material_at([[0.0, 0.0, 0.0]])[0]
    shallow = planet.material_at([[0.0, 0.0, 0.5]])[0]
    assert not np.allclose(core, shallow)                     # different layers at different depths
    # centre matches the innermost layer colour
    assert np.allclose(core, ml.albedo(planet.layer_names[-1]))


def test_planet_has_deposit_pockets(planet):
    sec = planet.cross_section(res=40)
    dep_cols = np.array([ml.albedo(nm) for nm in by_class("deposit")])
    flat = sec.reshape(-1, 3)
    is_dep = np.array([np.any(np.all(np.abs(c - dep_cols) < 1e-6, axis=1)) for c in flat])
    assert is_dep.sum() > 0                                   # visible ore/mineral pockets in a slice


def test_planet_is_deterministic():
    a = fractal_planet(radius=1.0, seed=7, dim=96, octaves=2, relief=0.1)
    b = fractal_planet(radius=1.0, seed=7, dim=96, octaves=2, relief=0.1)
    q = [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.6, 0.0, 0.0]]
    assert np.allclose(a.material_at(q), b.material_at(q))


def test_cross_section_shape_and_range(planet):
    img = planet.cross_section(res=24)
    assert img.shape == (24, 24, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_png_writer_roundtrips(tmp_path):
    img = np.zeros((4, 4, 3)); img[1, 1] = (1.0, 0.5, 0.25)
    path = ml.write_png(str(tmp_path / "t.png"), img)
    with open(path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"              # valid PNG signature


def test_material_carries_physical_dielectric_and_fiber_properties():
    import holographic_matlib as L
    # transmissive classes get a real IOR + transmission + attenuation tint
    glass = L.material("glass_clear")
    assert glass.transmission == 1.0 and abs(glass.ior - 1.50) < 1e-9
    assert L.material("diamond").ior > 2.0                      # diamond bends hard
    water = L.material("water")
    assert water.transmission == 1.0 and abs(water.ior - 1.33) < 1e-9
    # opaque metals are untouched (no transmission)
    assert L.material("gold").transmission == 0.0 and L.material("gold").metallic == 1.0
    # fibers carry strand params
    fur = L.material("fur_brown")
    assert fur.fiber and fur.fiber_roughness > 0


def test_shade_adapter_routes_transmission_vs_opaque():
    import numpy as np, holographic_matlib as L
    a, m, r, e, ior = L.shade(L.material("glass_clear"), 6)
    assert a.shape == (6, 3) and (ior > 1.0).all() and (m == 0.0).all()
    assert np.allclose(a[0], L.material("glass_clear").attenuation_color)   # glass 'albedo' is the tint
    a2, m2, r2, e2, ior2 = L.shade(L.material("copper"), 6)
    assert (ior2 == 0.0).all() and (m2 == 1.0).all()           # opaque metal: no refraction, metallic on


def test_fiber_params_reads_hair_and_refuses_non_fiber():
    import holographic_matlib as L
    fp = L.fiber_params(L.material("hair_black"))
    assert set(fp) == {"hair_color", "roughness", "tilt_deg"} and fp["roughness"] > 0
    try:
        L.fiber_params(L.material("gold")); assert False, "gold is not a fiber"
    except ValueError:
        pass


def test_thermal_emission_from_temperature():
    # backlog H2: a hot material EMITS blackbody radiation derived from its temperature (matlib.heat -> shade)
    import numpy as np
    import holographic_matlib as ML
    cold = ML.shade(ML.material("iron"), 1)[3][0]
    assert np.allclose(cold, 0.0)                                # a cold material does not glow
    lums = []
    for T in (900, 1400, 2200):
        e = ML.shade(ML.heat("iron", T), 1)[3][0]
        assert e[0] >= e[1] >= e[2]                              # blackbody: red >= green >= blue (warm glow)
        lums.append(float(e.mean()))
    assert lums[0] < lums[1] < lums[2]                          # hotter -> brighter emission (monotonic)


def test_heat_accepts_name_or_material():
    import holographic_matlib as ML
    a = ML.heat("gold", 1500)
    b = ML.heat(ML.material("gold"), 1500)
    assert a.temperature_K == 1500.0 and b.temperature_K == 1500.0
    assert ML.material("gold").temperature_K == 0.0             # heating a copy doesn't mutate the library


def test_iridescent_presets_carry_film():
    import holographic_matlib as ML
    for name in ("soap_bubble", "oil_slick", "beetle_shell"):
        m = ML.material(name)
        assert m.iridescence_nm > 0.0                            # the preset carries a film thickness
        out = ML.shade(m, 3)
        assert len(out) == 7                                     # shade emits the 7th (iridescence) value
    # a non-iridescent material stays a 5-tuple
    assert len(ML.shade(ML.material("steel"), 3)) == 5


def test_iridesce_helper():
    import holographic_matlib as ML
    m = ML.iridesce("steel", 350.0)
    assert m.iridescence_nm == 350.0 and len(ML.shade(m, 3)) == 7
    assert ML.material("steel").iridescence_nm == 0.0           # library material unchanged

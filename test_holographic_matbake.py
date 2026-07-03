"""Performance MC2: bake view-independent material channels into trilinear field lookups."""
import numpy as np
from holographic_surface import SurfaceMaterial
from holographic_param import Param
from holographic_matbake import BakedField, bake_field, bake_material


def _mat():
    rough = lambda P, **k: 0.3 + 0.2 * np.sin(np.asarray(P)[:, 0] * 2.0)
    col = lambda P, **k: np.stack([0.5 + 0.4 * np.asarray(P)[:, 1], np.full(len(P), 0.3), np.full(len(P), 0.6)], axis=1)
    return SurfaceMaterial(color=Param(field=col), roughness=Param(field=rough), reflect=0.1, emission=0.0)


LO, HI = (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)


def test_bakedfield_trilinear_matches_at_grid_nodes():
    # a linear field is reproduced EXACTLY by trilinear interpolation (no interpolation error for linear data)
    f = lambda P, **k: np.asarray(P)[:, 0] + 2 * np.asarray(P)[:, 1]
    bf = bake_field(f, "roughness", LO, HI, res=16)
    pts = np.random.default_rng(0).uniform(-0.9, 0.9, size=(50, 3))
    assert np.abs(bf.sample(pts) - f(pts)).max() < 1e-10


def test_baked_material_matches_resolve():
    mat = _mat()
    pts = np.random.default_rng(0).uniform(-0.9, 0.9, size=(200, 3))
    got, ref = bake_material(mat, LO, HI, res=48)(pts), mat.resolve(pts)
    assert np.abs(got["roughness"] - ref["roughness"]).max() < 0.02
    assert np.abs(got["color"] - ref["color"]).max() < 0.02


def test_constants_folded_fields_baked():
    shade = bake_material(_mat(), LO, HI, res=16)
    assert set(shade.const_names) == {"reflect", "emission", "opacity"}
    assert set(shade.baked_names) == {"color", "roughness"}


def test_finer_grid_more_accurate():
    mat = _mat()
    pts = np.random.default_rng(1).uniform(-0.9, 0.9, size=(200, 3))
    ref = mat.resolve(pts)["roughness"]
    err8 = np.abs(bake_material(mat, LO, HI, res=8)(pts)["roughness"] - ref).max()
    err64 = np.abs(bake_material(mat, LO, HI, res=64)(pts)["roughness"] - ref).max()
    assert err64 < err8                                        # the resolution/accuracy trade


def test_lookup_is_o1_reusable():
    # once baked, sampling doesn't call the field again -- prove by counting field calls
    calls = {"n": 0}
    def rough(P, **k):
        calls["n"] += 1
        return 0.3 + 0.1 * np.asarray(P)[:, 0]
    mat = SurfaceMaterial(color=(0.5, 0.5, 0.5), roughness=Param(field=rough))
    shade = bake_material(mat, LO, HI, res=16)                 # bakes -> some field calls here
    baked_calls = calls["n"]
    for _ in range(10):                                        # ten frames of lookups
        shade(np.random.default_rng(0).uniform(-0.9, 0.9, size=(20, 3)))
    assert calls["n"] == baked_calls                          # no further field evaluation after the bake


def test_color_grid_shape():
    bf = bake_field(lambda P, **k: np.tile([0.2, 0.4, 0.6], (len(P), 1)), "color", LO, HI, res=8)
    assert bf.grid.shape == (8, 8, 8, 3)
    out = bf.sample(np.zeros((5, 3)))
    assert out.shape == (5, 3) and np.allclose(out, [0.2, 0.4, 0.6])

"""Performance MC1: compile+fuse materials -- one cached kernel per material, constants folded."""
import numpy as np
from holographic_surface import SurfaceMaterial
from holographic_param import Param
from holographic_compile import CompileCache
from holographic_matcompile import compiled_shader, material_spec, CHANNELS, _is_const


def _mat():
    rough = lambda P, **k: 0.2 + 0.3 * (np.asarray(P)[:, 0] > 0)      # position-dependent socket
    return SurfaceMaterial(color=(0.8, 0.3, 0.2), roughness=Param(field=rough), reflect=0.1, emission=0.0)


def test_compiled_matches_resolve():
    mat = _mat()
    pts = np.random.default_rng(0).uniform(-1, 1, size=(40, 3))
    got, ref = compiled_shader(mat, cache=CompileCache())(pts), mat.resolve(pts)
    for name in CHANNELS:
        assert np.allclose(got[name], ref[name]), name


def test_constants_folded_only_sockets_resolve():
    shade = compiled_shader(_mat(), cache=CompileCache())
    assert shade.socket_names == ["roughness"]
    assert set(shade.const_names) == {"color", "reflect", "emission", "opacity"}


def test_built_once_and_reused():
    mat = _mat(); cache = CompileCache()
    compiled_shader(mat, cache=cache)
    compiled_shader(mat, cache=cache)
    compiled_shader(mat, cache=cache)
    assert cache.stats["compiles"] == 1 and cache.stats["hits"] == 2   # one build, two reuses


def test_changed_constant_is_a_fresh_compile():
    cache = CompileCache()
    rough = lambda P, **k: np.asarray(P)[:, 0] * 0
    compiled_shader(SurfaceMaterial(color=(0.8, 0.3, 0.2), roughness=Param(field=rough)), cache=cache)
    compiled_shader(SurfaceMaterial(color=(0.1, 0.1, 0.9), roughness=Param(field=rough)), cache=cache)
    assert cache.stats["compiles"] == 2                               # different color spec -> recompile


def test_is_const_detection():
    assert _is_const("roughness", 0.5)                               # a scalar is constant
    assert _is_const("color", (0.5, 0.2, 0.1))                       # an rgb triple is constant
    assert not _is_const("roughness", Param(field=lambda P, **k: np.asarray(P)[:, 0]))   # a field is not


def test_all_constant_material_has_no_sockets():
    shade = compiled_shader(SurfaceMaterial(color=(0.5, 0.5, 0.5), roughness=0.3, reflect=0.0), cache=CompileCache())
    assert shade.socket_names == []                                  # nothing re-resolves per hit
    out = shade(np.zeros((5, 3)))
    assert np.allclose(out["roughness"], 0.3) and out["color"].shape == (5, 3)


def test_empty_cache_is_respected():
    # regression for the empty-CompileCache-is-falsy gotcha: a fresh cache must actually be used
    cache = CompileCache()
    compiled_shader(_mat(), cache=cache)
    assert cache.stats["compiles"] == 1                              # went into OUR cache, not the default

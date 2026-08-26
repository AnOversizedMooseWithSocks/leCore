"""Tests for holographic_texturehome -- the Texture home (R6: detail fields feeding Material channels)."""
import numpy as np
from holographic.materials_and_texture.holographic_texturehome import Texture, texture_backends


def test_voronoi_field_vectorised_and_nonneg():
    crack = Texture.voronoi(n_seeds=12, seed=0, kind="edge")
    pts = np.random.default_rng(0).uniform(-1, 1, (80, 3))
    v = crack(pts)
    assert v.shape == (80,) and np.isfinite(v).all() and (v >= 0).all()
    ids = Texture.voronoi(n_seeds=12, seed=0, kind="id")(pts)
    assert ids.shape == (80,) and ids.min() >= 0


def test_material_channel_sourced_through_texture():
    # the R6 done-when: a Material channel pulls from a Texture field
    from holographic.mesh_and_geometry.holographic_surface import SurfaceMaterial
    from holographic.misc.holographic_param import Param
    crack = Texture.voronoi(n_seeds=10, seed=1, kind="edge")
    rough = lambda P, **k: 0.25 + 0.5 * np.clip(crack(P) * 4.0, 0, 1)
    mat = SurfaceMaterial(color=(0.6, 0.6, 0.62), roughness=Param(field=rough), reflect=0.1, emission=0.0)
    pts = np.random.default_rng(2).uniform(-1, 1, (16, 3))
    r = np.asarray(mat.resolve(pts)["roughness"], float)
    assert r.shape == (16,) and (r >= 0.25).all() and np.isfinite(r).all()


def test_fbm_field():
    fb = Texture.fbm(n_dims=3, bounds=[(-1, 1)] * 3, octaves=3, seed=0)
    v = fb(np.random.default_rng(3).uniform(-1, 1, (20, 3)))
    assert v.shape == (20,) and np.isfinite(v).all()
    assert hasattr(fb, "generator")                                  # underlying FractalNoise exposed


def test_curl_is_nearly_divergence_free():
    u, v = Texture.curl(res=24, seed=0)
    div = np.abs(np.gradient(u, axis=1) + np.gradient(v, axis=0)).mean()
    assert div < 0.5


def test_voronoi_deterministic():
    a = Texture.voronoi(n_seeds=8, seed=5, kind="edge")
    b = Texture.voronoi(n_seeds=8, seed=5, kind="edge")
    pts = np.random.default_rng(0).uniform(-1, 1, (30, 3))
    assert np.array_equal(a(pts), b(pts))


def test_backends_listed():
    assert set(texture_backends()) == {"fbm", "voronoi", "curl", "synth"}

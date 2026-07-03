"""SurfaceMaterial: every channel a socket, resolved per hit; from_name uses the one canonical table."""
import numpy as np
from holographic_surface import SurfaceMaterial, render_surface
from holographic_param import Param
from holographic_pattern import make_pattern, field_lerp
from holographic_render import Camera


class _Balls:
    cs = np.array([[0.0, 0, 0], [1.9, 0, 0]])
    def eval(s, P): return np.min(np.stack([np.linalg.norm(P - c, axis=1) - 0.85 for c in s.cs]), axis=0)
    def ids(s, P): return np.argmin(np.stack([np.linalg.norm(P - c, axis=1) for c in s.cs]), axis=0)


def test_channels_resolve_per_hit_from_any_socket_kind():
    pts = np.random.default_rng(0).standard_normal((50, 3))
    m = SurfaceMaterial(color=Param(field=field_lerp(make_pattern("checker"), (1, 0, 0), (0, 0, 1))),
                        roughness=make_pattern("gradient", axis=1),   # a bare callable field
                        reflect=0.3, opacity=np.float64(0.8))
    ch = m.resolve(pts)
    assert ch["color"].shape == (50, 3) and ch["roughness"].shape == (50,)
    assert ch["roughness"].std() > 0                                 # the field really varies per point
    assert np.allclose(ch["reflect"], 0.3) and np.allclose(ch["opacity"], 0.8)


def test_from_name_uses_canonical_material_table():
    from holographic_semantic import MATERIAL_RENDER
    m = SurfaceMaterial.from_name("metal")
    assert abs(float(np.mean(m.resolve(np.zeros((3, 3)))["reflect"])) - MATERIAL_RENDER["metal"]["reflect"]) < 1e-9
    g = SurfaceMaterial.from_name("glass")
    assert float(np.mean(g.resolve(np.zeros((3, 3)))["opacity"])) < 0.5   # refractive -> mostly transparent


def test_render_surface_pattern_is_solid_texture():
    cam = Camera(eye=(0.9, 1.0, 4.6), target=(0.9, 0, 0), fov_deg=52)
    tex = SurfaceMaterial(color=Param(field=field_lerp(make_pattern("checker", scale=2.5), (0.9, 0.2, 0.1), (0.95, 0.9, 0.85))))
    flat = SurfaceMaterial(color=(0.9, 0.5, 0.5))
    other = SurfaceMaterial.from_name("metal", color=(0.8, 0.8, 0.85))
    img_tex = render_surface(_Balls(), cam, 56, 56, {0: tex, 1: other})
    img_flat = render_surface(_Balls(), cam, 56, 56, {0: flat, 1: other})
    assert img_tex.std() > img_flat.std() * 1.02                     # pattern adds real variation on the surface


def test_opacity_composites_one_layer():
    cam = Camera(eye=(0.9, 1.0, 4.6), target=(0.9, 0, 0), fov_deg=52)
    solid = SurfaceMaterial(color=(0.3, 0.5, 0.9), opacity=1.0)
    clear = SurfaceMaterial(color=(0.3, 0.5, 0.9), opacity=0.4)
    m1 = SurfaceMaterial(color=(0.7, 0.7, 0.7))
    a = render_surface(_Balls(), cam, 48, 48, {0: solid, 1: m1})
    b = render_surface(_Balls(), cam, 48, 48, {0: clear, 1: m1})
    assert not np.allclose(a, b)                                     # transparency changes the composite

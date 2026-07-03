"""RenderSession ties preview + progressive final + splat proxy to ONE scene; an edit shows in both; path_trace's
progressive callback is backward-compatible."""
import numpy as np
from holographic_session import RenderSession, sdf_surface_points, _pathtrace_material
from holographic_surface import SurfaceMaterial
from holographic_param import Param
from holographic_pattern import make_pattern, field_lerp
from holographic_render import Camera


class _Two:
    cs = np.array([[0.0, 0, 0], [1.9, 0, 0]])
    def eval(s, P): return np.min(np.stack([np.linalg.norm(P - c, axis=1) - 0.85 for c in s.cs]), axis=0)
    def ids(s, P): return np.argmin(np.stack([np.linalg.norm(P - c, axis=1) for c in s.cs]), axis=0)


def _session():
    m0 = SurfaceMaterial(color=Param(field=field_lerp(make_pattern("checker", scale=2.5), (0.9, 0.2, 0.1), (0.95, 0.9, 0.85))))
    m1 = SurfaceMaterial.from_name("metal", color=(0.8, 0.8, 0.85))
    cam = Camera(eye=(0.9, 1.0, 4.6), target=(0.9, 0, 0), fov_deg=52)
    return RenderSession(_Two(), {0: m0, 1: m1}, cam, width=48, height=48)


def test_preview_renders_the_scene():
    img = _session().preview()
    assert img.shape == (48, 48, 3) and np.isfinite(img).all()


def test_edit_shows_in_preview_and_final_share_materials():
    s = _session()
    a = s.preview()
    s.edit_channel(1, "color", (0.1, 0.9, 0.3))
    b = s.preview()
    assert not np.allclose(a, b)                                     # preview reflects the edit
    # the final's material adapter reads the SAME edited material
    mat = _pathtrace_material(s.sdf, s.materials)
    alb, met, rough, emis, ior = mat(np.array([[1.9, 0, 0.85]]))    # a point on object 1
    assert np.allclose(alb[0], (0.1, 0.9, 0.3), atol=0.05)          # edited colour flows to the path tracer too


def test_progressive_final_fires_and_returns():
    s = _session()
    fired = []
    fin = s.render_final(spp=6, on_progress=lambda im, d, t: fired.append(d), progress_every=2,
                         width=32, height=32, sky=lambda D: np.ones((len(D), 3)) * 0.9)
    assert fin.shape == (32, 32, 3) and len(fired) >= 1 and fired[0] == 2


def test_pathtrace_progress_is_backward_compatible():
    """progress_every=0 (default) is byte-identical to the old path_trace -- the hook is additive."""
    from holographic_pathtrace import path_trace, constant_material
    cam = Camera(eye=(0, 0, 4), target=(0, 0, 0), fov_deg=45)
    white = lambda D: np.ones((len(D), 3)) * 0.8
    a = path_trace(_Two(), cam, 28, 28, spp=6, material=constant_material(), sky=white, seed=0)
    b = path_trace(_Two(), cam, 28, 28, spp=6, material=constant_material(), sky=white, seed=0, progress_every=0)
    assert np.array_equal(a, b)


def test_surface_points_land_on_the_surface():
    pts = sdf_surface_points(_Two(), (np.full(3, -4.0), np.full(3, 4.0)), n=400, seed=0)
    assert len(pts) > 50                                            # found the surface
    assert np.abs(_Two().eval(pts)).max() < 0.05                    # every point is ON it


def test_to_splats_makes_coloured_proxy():
    s = _session()
    splats, js = s.to_splats(n=300)
    assert len(splats) > 0 and isinstance(js, str) and len(js) > 2

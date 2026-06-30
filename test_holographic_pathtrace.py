"""Tests for the Monte-Carlo path tracer (PATHTRACE-1)."""
import numpy as np
from holographic_pathtrace import path_trace, constant_material


class _Sphere:
    def eval(self, P):
        return np.linalg.norm(P, axis=-1) - 1.0


class _Cam:
    eye = np.array([0.0, 0.0, 3.0])
    def ray_dirs(self, w, h):
        ys, xs = np.mgrid[0:h, 0:w]
        u = (xs / (w - 1) - 0.5) * 1.4; v = -(ys / (h - 1) - 0.5) * 1.4
        d = np.stack([u, v, -np.ones_like(u)], -1)
        return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)


def test_white_furnace_converges_to_albedo():
    white = lambda D: np.ones((len(D), 3))
    mat = constant_material(albedo=(0.6, 0.6, 0.6), metallic=0.0, roughness=1.0)
    img = path_trace(_Sphere(), _Cam(), 40, 40, spp=48, max_bounce=3, material=mat, sky=white, seed=0)
    sphere = img.reshape(-1, 3)[img.reshape(-1, 3).mean(1) < 0.95]
    assert 0.5 < sphere.mean() < 0.62                              # ~ albedo (unbiased, single-scatter slack)


def test_noise_falls_with_more_samples():
    white = lambda D: np.ones((len(D), 3))
    mat = constant_material(albedo=(0.6, 0.6, 0.6), metallic=0.0, roughness=1.0)
    ref = path_trace(_Sphere(), _Cam(), 32, 32, spp=160, max_bounce=3, material=mat, sky=white, seed=9)
    lo = path_trace(_Sphere(), _Cam(), 32, 32, spp=8, max_bounce=3, material=mat, sky=white, seed=1)
    hi = path_trace(_Sphere(), _Cam(), 32, 32, spp=64, max_bounce=3, material=mat, sky=white, seed=1)
    m = ref.mean(-1) < 0.95
    err_lo = np.sqrt(((lo - ref)[m] ** 2).mean()); err_hi = np.sqrt(((hi - ref)[m] ** 2).mean())
    assert err_hi < err_lo


def test_multibounce_adds_indirect_light():
    # a sphere on a bright red floor: more bounces -> red bleeds onto the sphere underside
    class Scene:
        def eval(self, P):
            return np.minimum(np.linalg.norm(P, axis=-1) - 1.0, P[..., 1] + 1.0)
    def material(P):
        n = len(P); floor = P[..., 1] < -0.985
        alb = np.where(floor[:, None], np.array([0.85, 0.08, 0.06]), np.array([0.85, 0.85, 0.85]))
        return alb, np.zeros(n), np.full(n, 0.7), np.zeros((n, 3))
    from holographic_raymarch import sky_dome
    sky = lambda D: 1.6 * np.clip(sky_dome(D), 0, None)

    class Cam:
        eye = np.array([0.0, 0.5, 3.4])
        def ray_dirs(self, w, h):
            fwd = np.array([0.0, -0.16, -1.0]); fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
            ys, xs = np.mgrid[0:h, 0:w]; u = (xs / (w - 1) - 0.5) * 1.4; v = -(ys / (h - 1) - 0.5) * 1.4
            d = fwd[None, None] + u[..., None] * right[None, None] + v[..., None] * up[None, None]
            return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)
    sc, cam = Scene(), Cam()
    direct = path_trace(sc, cam, 56, 56, spp=48, max_bounce=2, material=material, sky=sky, seed=0)
    gi = path_trace(sc, cam, 56, 56, spp=48, max_bounce=5, material=material, sky=sky, seed=0)
    band = np.zeros((56, 56), bool); band[32:42, 18:38] = True     # lower sphere patch
    hit = gi.mean(-1) > 0.02; band &= hit
    rg = lambda im: im[..., 0][band].mean() / (im[..., 1][band].mean() + 1e-6)
    assert rg(gi) > rg(direct) + 0.1                              # indirect red bleed present


def test_path_trace_variance_and_active_mask():
    """return_variance gives a per-pixel noise map; active restricts which pixels are traced (others stay 0)."""
    import numpy as np
    from holographic_pathtrace import path_trace, constant_material
    from holographic_render import Camera

    class Ball:
        def eval(self, P):
            return np.linalg.norm(P, axis=1) - 1.0
    cam = Camera(eye=(0, 0, 3.5), target=(0, 0, 0), fov_deg=45.0)
    img, var = path_trace(Ball(), cam, width=16, height=16, spp=6, material=constant_material((0.7, 0.3, 0.3)),
                          return_variance=True)
    assert img.shape == (16, 16, 3) and var.shape == (16, 16) and var.min() >= 0.0
    mask = np.zeros(16 * 16, bool); mask[:128] = True
    sub = path_trace(Ball(), cam, width=16, height=16, spp=4, material=constant_material(), active=mask)
    assert sub.reshape(-1, 3)[128:].sum() == 0.0          # untraced pixels stay black
    assert sub.reshape(-1, 3)[:128].sum() > 0.0           # traced pixels are rendered

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


def test_low_discrepancy_antialias_optin():
    # backlog H1: antialias defaults OFF (byte-identical), and ON it changes edges without breaking the image
    import numpy as np
    from holographic_pathtrace import path_trace
    from holographic_render import Camera
    class S:
        def eval(self, P):
            return np.min(np.linalg.norm(P[..., None, :] - np.array([[-0.6, 0, 0], [0.6, 0, 0]], float),
                                         axis=-1) - 0.55, axis=-1)
    def mat(P):
        n = len(P); return np.tile([.8, .4, .3], (n, 1)).astype(float), np.zeros(n), np.full(n, .5), np.zeros((n, 3))
    cam = Camera(eye=(0, 0, 3.2), target=(0, 0, 0), fov_deg=40, aspect=1.0)
    a = path_trace(S(), cam, 40, 40, spp=6, max_bounce=2, material=mat, seed=0)
    b = path_trace(S(), cam, 40, 40, spp=6, max_bounce=2, material=mat, seed=0, antialias=False)
    assert np.array_equal(a, b)                                   # default OFF == explicit OFF, byte-identical
    c = path_trace(S(), cam, 40, 40, spp=6, max_bounce=2, material=mat, seed=0, antialias=True)
    assert not np.allclose(a, c) and np.isfinite(c).all() and c.min() >= 0   # ON differs, stays a valid image


def test_subsurface_glow_optin():
    # backlog H2: SSS defaults OFF (byte-identical), and ON a translucent material glows -> brighter, thin edges most
    import numpy as np
    from holographic_pathtrace import path_trace
    from holographic_render import Camera
    from holographic_sdf import sphere
    scene = sphere(0.8)
    def plain(P):
        n = len(P); return np.tile([.2, .55, .38], (n, 1)).astype(float), np.zeros(n), np.full(n, .4), np.zeros((n, 3))
    def translucent(P):   # 6-tuple with sss=1
        n = len(P); return (np.tile([.2, .55, .38], (n, 1)).astype(float), np.zeros(n), np.full(n, .4),
                            np.zeros((n, 3)), np.zeros(n), np.full(n, 1.0))
    cam = Camera(eye=(0, 0, 3.2), target=(0, 0, 0), fov_deg=40)
    sky = lambda D: np.tile([0.05, 0.06, 0.08], (len(D), 1))
    # sss_dir=None => the 6-tuple material still renders, and equals... not necessarily plain (glow needs a dir).
    a = path_trace(scene, cam, 40, 40, spp=6, max_bounce=2, material=plain, sky=sky, seed=0)
    b = path_trace(scene, cam, 40, 40, spp=6, max_bounce=2, material=plain, sky=sky, seed=0, sss_dir=None)
    assert np.array_equal(a, b)                                   # sss_dir off = byte-identical
    lit = path_trace(scene, cam, 40, 40, spp=6, max_bounce=2, material=translucent, sky=sky, seed=0,
                     sss_dir=(-0.3, 0.4, -0.9), sss_depth=1.4, sss_sigma=2.5)
    assert lit.mean() > a.mean() * 1.05 and np.isfinite(lit).all()   # the glow adds light


def test_sss_march_dither_reduces_banding():
    # the SSS march quantizes thickness into depth/steps levels -> contour BANDS on smooth objects. More steps +
    # per-point dither must yield strictly more distinct transmit levels across a smooth ramp; jitter=None must
    # stay byte-identical to the old sampling.
    import numpy as np
    from holographic_raymarch import subsurface
    from holographic_sdf import sphere
    s = sphere(1.0)
    xs = np.linspace(-0.95, 0.95, 300)
    P = np.stack([xs, np.zeros_like(xs), np.sqrt(np.clip(1 - xs ** 2, 0, None))], 1)
    N = P.copy()
    L = np.array([0.0, 0.0, -1.0])
    old = subsurface(s, P, N, L, depth=1.3, steps=10, sigma=2.8)
    assert np.array_equal(old, subsurface(s, P, N, L, depth=1.3, steps=10, sigma=2.8, jitter=None))
    jit = np.abs(np.modf(np.sin(P @ np.array([12.9898, 78.233, 37.719])) * 43758.5453)[0] % 1.0)
    fine = subsurface(s, P, N, L, depth=1.3, steps=37, sigma=2.8, jitter=jit)
    levels = lambda v: len(np.unique(np.round(v, 5)))
    assert levels(fine) > 2 * levels(old)                        # dithered fine march resolves far more levels


def test_iridescence_adds_view_dependent_colour():
    # a 7-tuple material with a film thickness tints the reflection by view angle -> more hue variation across a
    # curved surface than a plain material (backlog: thin-film iridescence).
    import numpy as np
    from holographic_pathtrace import path_trace
    from holographic_render import Camera
    from holographic_sdf import sphere
    scene = sphere(0.8)
    cam = Camera(eye=(0, 0, 3.0), target=(0, 0, 0), fov_deg=40, aspect=1.0)
    sky = lambda D: np.tile([0.6, 0.65, 0.75], (len(D), 1))
    def plain(P):
        n = len(P); return (np.tile([0.15, 0.15, 0.18], (n, 1)).astype(float), np.full(n, 0.1),
                            np.full(n, 0.25), np.zeros((n, 3)))
    def irid(P):
        n = len(P)
        return (np.tile([0.6, 0.6, 0.6], (n, 1)).astype(float), np.full(n, 0.2), np.full(n, 0.15),
                np.zeros((n, 3)), np.zeros(n), np.zeros(n), np.full(n, 320.0))     # 7-tuple: 320 nm film
    def hue_spread(im):
        px = im.reshape(-1, 3); px = px[px.sum(1) > 0.1]
        return (px / (px.sum(1, keepdims=True) + 1e-6)).std(0).mean()
    a = path_trace(scene, cam, 40, 40, spp=6, max_bounce=2, material=plain, sky=sky, seed=0)
    b = path_trace(scene, cam, 40, 40, spp=6, max_bounce=2, material=irid, sky=sky, seed=0)
    assert hue_spread(b) > hue_spread(a) * 1.5                   # iridescence adds hue variation
    assert np.isfinite(b).all()


def test_next_event_estimation_lights_a_dark_scene():
    # NEE: a placed light lets a small lamp light a scene that a dark environment alone cannot (backlog: lights).
    import numpy as np
    from holographic_pathtrace import path_trace
    from holographic_render import Camera
    from holographic_lights import PointLight
    from holographic_sdf import sphere, box
    scene = sphere(0.6).smooth_union(box(2.0, 0.1, 2.0).translate((0, -0.7, 0)), k=0.05)
    cam = Camera(eye=(0, 0.6, 3.0), target=(0, -0.2, 0), fov_deg=45, aspect=1.0)
    dark = lambda D: np.tile([0.02, 0.02, 0.03], (len(D), 1))
    def mat(P):
        n = len(P); return np.tile([0.7, 0.6, 0.5], (n, 1)).astype(float), np.zeros(n), np.full(n, 0.6), np.zeros((n, 3))
    light = PointLight(position=(1.5, 2.5, 1.0), color=(1, 0.95, 0.85), intensity=12.0)
    no_light = path_trace(scene, cam, 48, 48, spp=8, max_bounce=3, material=mat, sky=dark, seed=0)
    with_light = path_trace(scene, cam, 48, 48, spp=8, max_bounce=3, material=mat, sky=dark, seed=0, lights=[light])
    assert with_light.mean() > no_light.mean() * 3                    # the lamp lights the scene
    assert np.isfinite(with_light).all()
    assert (with_light.mean(2) < 0.05).mean() > 0.1                   # and casts shadows (dark regions remain)


def test_lights_none_is_backward_compatible():
    # not passing lights leaves the render byte-identical to before (environment-only path).
    import numpy as np
    from holographic_pathtrace import path_trace
    from holographic_render import Camera
    from holographic_sdf import sphere
    scene = sphere(0.7)
    cam = Camera(eye=(0, 0, 3.0), target=(0, 0, 0), fov_deg=40, aspect=1.0)
    sky = lambda D: np.tile([0.6, 0.65, 0.75], (len(D), 1))
    def mat(P):
        n = len(P); return np.tile([0.6, 0.6, 0.6], (n, 1)).astype(float), np.zeros(n), np.full(n, 0.5), np.zeros((n, 3))
    a = path_trace(scene, cam, 40, 40, spp=6, max_bounce=2, material=mat, sky=sky, seed=0)
    b = path_trace(scene, cam, 40, 40, spp=6, max_bounce=2, material=mat, sky=sky, seed=0, lights=None)
    assert np.array_equal(a, b)                                       # lights=None changes nothing

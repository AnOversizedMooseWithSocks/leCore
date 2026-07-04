"""Hair shading (H4/H5/H6): Kajiya-Kay anisotropy, Marschner R/TT/TRT, strand render."""
import numpy as np
from holographic_hairshade import kajiya_kay, marschner, marschner_lobes, absorption_from_color, render_hair, _unit


T = np.array([0.0, 1.0, 0.0])


def test_kajiya_diffuse_and_specular_anisotropic():
    perp = kajiya_kay(T, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    para = kajiya_kay(T, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert perp.sum() > para.sum()
    ldir = _unit(np.array([1.0, 0.5, 0.0])); mirror = _unit(np.array([-1.0, 0.5, 0.0])); off = _unit(np.array([1.0, -0.9, 0.3]))
    assert kajiya_kay(T, ldir, mirror, shininess=40.0)[0] > kajiya_kay(T, ldir, off, shininess=40.0)[0]


def test_marschner_blonde_brighter_and_colored_trt():
    l = _unit(np.array([0.4, 0.3, 0.7])); v = _unit(np.array([-0.3, 0.2, 0.8]))
    assert marschner(T, l, v, hair_color=(0.85, 0.7, 0.4)).sum() > marschner(T, l, v, hair_color=(0.05, 0.04, 0.03)).sum()
    R, TT, TRT = marschner_lobes(T, l, v, hair_color=(0.7, 0.4, 0.2))
    assert np.sum(TRT) > 0.0 and not np.allclose(TRT / (TRT.max() + 1e-9), 1.0)
    assert np.allclose(R / (R.max() + 1e-9), 1.0)                 # R is white


def test_absorption_monotone():
    assert absorption_from_color((0.9, 0.9, 0.9)).mean() < absorption_from_color((0.1, 0.1, 0.1)).mean()


def test_render_and_lod():
    from holographic_groom import groom
    from holographic_render import Camera
    from holographic_sdf import sphere
    s = sphere(1.0)
    strands = groom(s.eval, 200, ([-1.6] * 3, [1.6] * 3), length=0.5, n_pts=6, curl=0.5, seed=0)
    cam = Camera(eye=(0.0, 0.0, 3.2), target=(0.0, 0.0, 0.0), fov_deg=45.0)
    full = render_hair(strands, cam, width=120, height=120, smooth_levels=1)
    coarse = render_hair(strands, cam, width=120, height=120, smooth_levels=1, lod_stride=3)
    assert full.std() > 0.0
    cf = (full.sum(axis=2) > 0.25).astype(float); cc = (coarse.sum(axis=2) > 0.25).astype(float)
    assert (cf * cc).sum() / (cc.sum() + 1e-9) > 0.6
    assert np.array_equal(full, render_hair(strands, cam, width=120, height=120, smooth_levels=1))


def test_render_hair_return_alpha_backward_compatible():
    # backlog H4: return_alpha adds a coverage mask WITHOUT changing the default image (byte-identical).
    import numpy as np
    from holographic_groom import groom
    from holographic_hairshade import render_hair
    from holographic_render import Camera
    from holographic_sdf import sphere
    strands = groom(sphere(0.6).eval, 120, ((-0.8, -0.8, -0.8), (0.8, 0.8, 0.8)), length=0.3, n_pts=6, curl=0.2, seed=0)
    cam = Camera(eye=(0, 0, 2.5), target=(0, 0, 0), fov_deg=45, aspect=1.0)
    img = render_hair(strands, cam, width=80, height=80, hair_color=(0.6, 0.4, 0.2))
    img2, alpha = render_hair(strands, cam, width=80, height=80, hair_color=(0.6, 0.4, 0.2), return_alpha=True)
    assert np.array_equal(img, img2)                             # image unchanged by asking for alpha
    assert alpha.shape == (80, 80)
    assert 0.0 < (alpha > 0).mean() < 1.0                        # some pixels covered, some background
    # alpha is 1 exactly where a strand painted: those pixels differ from the black background in img2-on-black
    black_bg = render_hair(strands, cam, width=80, height=80, hair_color=(0.6, 0.4, 0.2),
                           background=(0.0, 0.0, 0.0))
    painted = black_bg.sum(axis=2) > 1e-6
    assert np.array_equal(painted, alpha > 0)                    # coverage mask matches the painted pixels

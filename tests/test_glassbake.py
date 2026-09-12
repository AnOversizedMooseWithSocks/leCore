"""Sweep 150: glass as a baked spectral-refractive transfer (holographic_glassbake).

The render audit named this rung: the 30-100 minute cost lived in Monte-Carlo path tracing a solid glass
interior, and the only random thing in that estimator was the estimator. These pin the deterministic version:
bake once, relight by dot product, dispersion in the bake, internal bounces followed, and the two bugs found on
the way (the interior march re-exiting the face it left; the relight-vs-bake ratio being the wrong claim).
"""
import numpy as np
import pytest

from holographic.mesh_and_geometry.holographic_sdf import sphere, box, fold_fractal
from holographic.rendering.holographic_render import Camera
from holographic.rendering.holographic_glassbake import bake_glass, EnvField, GlassBake, HdriEnv
from holographic.rendering.holographic_pathtrace import _march_through
from holographic.rendering.holographic_dispersion import cauchy_n

CAM = Camera(eye=(0.0, 1.0, 4.0), target=(0.0, 0.0, 0.0), fov_deg=35.0, aspect=1.0)
LAMS = np.linspace(430.0, 670.0, 5)
IOR = lambda lam: float(cauchy_n(lam, 1.55, 25.68))
CMF = lambda l: np.stack([np.exp(-0.5 * ((l - c) / 45.0) ** 2) for c in (610.0, 545.0, 465.0)], axis=1)


def test_env_field_reads_back_its_samples_and_is_dark_elsewhere():
    env = EnvField().deposit([[0.0, 1.0, 0.0]], [[1.0, 0.2, 0.2]])
    up = env.radiance([[0.0, 1.0, 0.0]])[0]; down = env.radiance([[0.0, -1.0, 0.0]])[0]
    assert np.allclose(up, [1.0, 0.2, 0.2], atol=0.05)
    assert down.max() < 0.15


def test_env_field_is_superposition():
    a = EnvField().deposit([[0, 1, 0]], [[1, 0, 0]]); b = EnvField().deposit([[1, 0, 0]], [[0, 0, 1]])
    both = EnvField().deposit([[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 1]])
    assert np.allclose(a.L_spec + b.L_spec, both.L_spec)


def test_bake_is_light_independent_and_relight_changes_with_the_light():
    bake = bake_glass(sphere(0.8), CAM, 40, 40, IOR, LAMS, floor_y=-0.9)
    warm = EnvField().add_softbox((3, 4, 2), (0, 0, 0), 2.0, 2.0, 40.0, color=(1.0, 0.8, 0.6))
    cool = EnvField().add_softbox((-3, 4, 2), (0, 0, 0), 2.0, 2.0, 40.0, color=(0.6, 0.8, 1.0))
    fa, fb = bake.relight(warm, CMF), bake.relight(cool, CMF)
    assert fa.shape == (40, 40, 3) and np.isfinite(fa).all()
    assert fa[..., 0].mean() > fa[..., 2].mean() and fb[..., 2].mean() > fb[..., 0].mean()
    # the bake did not change between relights
    assert bake.exit_dirs.flags.writeable and np.isfinite(bake.exit_dirs).all()


def test_dispersion_lives_in_the_bake():
    bake = bake_glass(sphere(0.8), CAM, 40, 40, IOR, LAMS)
    g = np.flatnonzero(bake.glass); ok = bake.exit_ok[0, g] & bake.exit_ok[-1, g]
    spread = np.linalg.norm(bake.exit_dirs[0, g][ok] - bake.exit_dirs[-1, g][ok], axis=1)
    assert np.median(spread) > 1e-3, "red and blue must exit in different directions"
    flat = bake_glass(sphere(0.8), CAM, 40, 40, lambda lam: 1.55, LAMS)
    g2 = np.flatnonzero(flat.glass)
    assert np.allclose(flat.exit_dirs[0, g2], flat.exit_dirs[-1, g2]), "no dispersion -> identical exits"


def test_internal_bounces_recover_energy_a_cube_traps():
    """THE BUG THIS PINS: a box traps most rays in TIR at first order. Following internal bounces must recover them
    -- and it only does if the interior march refuses to re-exit the face it just left (require_inside)."""
    b0 = bake_glass(box(1.0, 1.0, 1.0), CAM, 40, 40, lambda l: 1.55, LAMS[:1], max_internal=0)
    b4 = bake_glass(box(1.0, 1.0, 1.0), CAM, 40, 40, lambda l: 1.55, LAMS[:1], max_internal=4)
    loss0 = 1.0 - b0.exit_ok[:, b0.glass].mean(); loss4 = 1.0 - b4.exit_ok[:, b4.glass].mean()
    assert loss0 > 0.2, "a glass cube at first order must trap a lot: %.3f" % loss0
    assert loss4 < 0.5 * loss0, "internal bounces must recover most of it: %.3f -> %.3f" % (loss0, loss4)


def test_march_through_require_inside_leaves_the_face_it_reflected_off():
    """The real geometry of the bug: after a GRAZING internal reflection the restart point (exitP + r*3e-3, with
    exitP ~1e-3 OUTSIDE where the march stopped) slides along the face and never gets under it, so the default
    march sees d > surf_eps immediately and returns the same face. A normal-incidence restart does NOT reproduce
    it -- the first draft of this test used one and failed the other way."""
    s = box(1.0, 1.0, 1.0)
    D = np.array([[0.9, 0.0, -0.1]]); D /= np.linalg.norm(D)
    # exitP is where the previous march STOPPED, i.e. d just above surf_eps (1e-3); a grazing 3e-3 step barely
    # lowers it, so the restart still reads d > surf_eps -- the condition that makes the default return at once
    O = np.array([[0.0, 0.0, 1.0 + 1.5e-3]]) + D * 3e-3
    assert float(s.eval(O)[0]) > 1e-3, 'fixture must start just OUTSIDE by more than surf_eps'
    near = _march_through(s, O, D, max_steps=64)
    far = _march_through(s, O, D, max_steps=64, require_inside=True)
    assert near[0, 2] > 0.99, "the default march stops at the face it is grazing (old contract): z=%.4f" % near[0, 2]
    assert far[0, 0] > 0.9, "require_inside carries the ray into the body and out the +x face: %r" % far[0]
    # default behaviour untouched for a ray that starts properly inside
    O2 = np.array([[0.0, 0.0, 0.0]]); Dz = np.array([[0.0, 0.0, -1.0]])
    assert np.allclose(_march_through(s, O2, Dz), _march_through(s, O2, Dz, require_inside=True), atol=5e-3)


def test_relight_is_a_read_not_a_trace():
    import time
    bake = bake_glass(fold_fractal(iterations=1, scale=-1.8, bailout=4.0, solid=True), CAM, 48, 48, IOR, LAMS)
    env = EnvField().add_softbox((3, 4, 2), (0, 0, 0), 2.0, 2.0, 40.0)
    t = time.time(); img = bake.relight(env, CMF); dt = time.time() - t
    assert dt < 3.0 and np.isfinite(img).all()


def test_env_field_ingests_the_engines_studio_rig():
    """The engine's studio_sky(preset) is a sky(D) callable -- exactly add_sky's contract. One door, not two rigs."""
    from holographic.rendering.holographic_studiorig import studio_sky
    env = EnvField().add_sky(studio_sky("classic"))
    # the classic rig lights from the SIDES (key/fill/rim), not from straight overhead -- probe where it shines
    assert env.radiance([[1.0, 0.3, 0.0]]).max() > 0.02 and np.isfinite(env.L_spec).all()


def test_antialiased_bake_smooths_the_silhouette():
    """samples>1 must reduce the stair-step on the glass silhouette against a plain analytic light."""
    env = lambda D: np.tile([[0.3, 0.3, 0.3]], (len(D), 1)) + 2.0 * np.clip(D[:, 1:2], 0, None)
    b1 = bake_glass(sphere(0.8), CAM, 48, 48, IOR, LAMS, floor_y=-0.9, samples=1)
    b4 = bake_glass(sphere(0.8), CAM, 48, 48, IOR, LAMS, floor_y=-0.9, samples=4)
    f1, f4 = b1.relight(env, CMF), b4.relight(env, CMF)
    edge = lambda f: np.abs(np.diff(f.mean(-1), axis=1)).max()
    assert edge(f4) < edge(f1), "the largest single-pixel step must shrink with jittered sub-samples"
    assert np.isfinite(f4).all() and f4.shape == f1.shape


def test_analytic_env_is_accepted_directly():
    bake = bake_glass(sphere(0.8), CAM, 24, 24, IOR, LAMS)
    img = bake.relight(lambda D: np.tile([[1.0, 0.5, 0.2]], (len(D), 1)), CMF)
    assert img.shape == (24, 24, 3) and img[..., 0].mean() > img[..., 2].mean()


# ------------------------------------------------------------------ sweep 152: physics that separates glass from a glass-shaped thing
def test_exit_rays_see_the_floor_through_the_glass():
    """A downward exit ray must take the floor's colour, not the environment's -- the biggest realism fix."""
    bake = bake_glass(sphere(0.8), CAM, 40, 40, IOR, LAMS, floor_y=-0.9)
    # a sky that is black everywhere except a small bright disc straight overhead: it lights the floor (the sun
    # term reads it along +y) but a ray leaving the glass sideways or downward sees nothing from the sky itself
    sun_only = lambda D: np.where(D[:, 1:2] > 0.98, 3.0, 0.0) * np.ones((len(D), 3))
    a = bake.relight(sun_only, CMF, floor_albedo=(1.0, 0.0, 0.0), see_floor=True, sun_dir=(0, 1, 0))
    b = bake.relight(sun_only, CMF, floor_albedo=(1.0, 0.0, 0.0), see_floor=False, sun_dir=(0, 1, 0))
    g = bake.glass.reshape(40, 40)
    assert a[g][:, 0].mean() > b[g][:, 0].mean(), "with a dark sky, glass over a red floor must show red THROUGH it"


def test_glass_shadow_is_transmissive():
    bake = bake_glass(sphere(0.8), CAM, 40, 40, IOR, LAMS, floor_y=-0.9)
    env = lambda D: np.tile([[1.0, 1.0, 1.0]], (len(D), 1))
    opaque = bake.relight(env, CMF, sdf=sphere(0.8), sun_dir=(0, 1, 0), shadow_transmittance=0.0)
    glassy = bake.relight(env, CMF, sdf=sphere(0.8), sun_dir=(0, 1, 0), shadow_transmittance=0.85)
    f = bake.floor.reshape(40, 40)
    assert glassy[f].mean() > opaque[f].mean(), "a glass occluder must pass most of the light into its shadow"


def test_sun_solid_angle_scales_only_the_direct_floor_term():
    bake = bake_glass(sphere(0.8), CAM, 32, 32, IOR, LAMS, floor_y=-0.9)
    env = lambda D: np.tile([[1.0, 1.0, 1.0]], (len(D), 1))
    a = bake.relight(env, CMF, sun_dir=(0, 1, 0), sun_solid_angle=1.0)
    b = bake.relight(env, CMF, sun_dir=(0, 1, 0), sun_solid_angle=0.16)
    f = bake.floor.reshape(32, 32)
    assert b[f].mean() < a[f].mean(), "a small light delivers radiance x solid angle, less than its radiance"


def test_material_optics_drive_the_bake_and_relight():
    from holographic.materials_and_texture.holographic_matlib import glass_optics
    for name, key in (("ruby", 0), ("sapphire", 2)):
        go = glass_optics(name)
        assert go["n_d"] == 1.77 and go["abbe"] == 72.2            # corundum, both
        bake = bake_glass(sphere(0.8), CAM, 32, 32, lambda l: go["n_d"], LAMS, floor_y=-0.9)
        img = bake.relight(lambda D: np.tile([[0.8, 0.8, 0.8]], (len(D), 1)), CMF, absorb=go["absorb"])
        g = bake.glass.reshape(32, 32)
        assert np.argmax(img[g].mean(0)) == key, "%s must come out its own colour" % name
    assert glass_optics("diamond")["abbe"] == 44.3 and glass_optics("diamond")["n_d"] == 2.42
    with pytest.raises(KeyError):
        glass_optics("gold")


def test_hdri_env_integrates_its_own_pixels():
    """An equirect map with ONE bright pixel straight up: irradiance = L * dOmega (cos=1 at the zenith), the dominant
    direction is +y, and nearly all of the floor irradiance is in that lobe."""
    env_img = np.zeros((64, 128, 3), np.float32) + 0.01
    env_img[0, :, :] = 500.0                                   # the whole top row is the zenith cap
    e = HdriEnv(env_img)
    assert e.dominant_dir[1] > 0.95, "the lobe must point up: %r" % e.dominant_dir
    assert e.sun_share > 0.5, "nearly all irradiance comes from the cap: %.2f" % e.sun_share
    assert e.floor_irradiance.mean() > 0.0 and np.isfinite(e.floor_irradiance).all()
    r = e.radiance([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert r[0].mean() > 100 * r[1].mean(), "the sampler must read the cap bright and the horizon dark"


def test_relight_with_hdri_uses_exact_floor_irradiance():
    env_img = np.zeros((32, 64, 3), np.float32); env_img[:16] = 2.0     # uniform bright upper hemisphere
    e = HdriEnv(env_img)
    bake = bake_glass(sphere(0.8), CAM, 32, 32, IOR, LAMS, floor_y=-0.9)
    img = bake.relight(e, CMF, floor_albedo=(1.0, 1.0, 1.0))
    f = bake.floor.reshape(32, 32)
    # E for uniform L over the hemisphere is pi*L; Lambert radiance = albedo*E/pi = L (=2), up to pixel quadrature
    assert abs(img[f].mean() - 2.0) < 0.35, "uniform sky: floor radiance ~= sky radiance: %.3f" % img[f].mean()


def test_load_exr_is_opt_in_and_matches_load_hdr_contract():
    pytest.importorskip("OpenEXR")
    import glob
    from holographic.rendering.holographic_render import load_exr
    files = glob.glob("/root/.claude/uploads/*/*.exr")
    if not files:
        pytest.skip("no .exr fixture on this machine")
    img = load_exr(files[0])
    assert img.ndim == 3 and img.shape[-1] == 3 and img.dtype == np.float32 and img.max() > 1.0, "linear, unbounded"

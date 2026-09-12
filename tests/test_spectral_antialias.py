"""Sweep 156: the two cures for spectral confetti (lam_jitter stratification, footprint_filter ray differential), the
fan-averaged lobe read, and the druse habit's wall coverage."""
import numpy as np
import pytest

import lecore
import holographic.mesh_and_geometry.holographic_crystalgrow as cg
from holographic.rendering.holographic_glassbake import HdriEnv


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_lam_jitter_shifts_wavelengths_per_subbake_default_identical(mind):
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    cam = mind.camera(eye=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), fov_deg=30.0, aspect=1.0)
    a = mind.bake_glass(sphere(0.6), cam, 12, 12, n_lams=5, samples=3)
    b = mind.bake_glass(sphere(0.6), cam, 12, 12, n_lams=5, samples=3, lam_jitter=True)
    assert all(np.array_equal(s.lams, a.subs[0].lams) for s in a.subs)              # default: same grid everywhere
    lam_sets = [tuple(np.round(s.lams, 6)) for s in b.subs]
    assert len(set(lam_sets)) == 3 and np.allclose(b.subs[0].lams, a.subs[0].lams)  # jitter: distinct shifted grids
    assert np.allclose(b.subs[1].lams - b.subs[0].lams, (a.subs[0].lams[1] - a.subs[0].lams[0]) / 3)


def test_hdri_env_blur_conserves_lobe_energy_and_lowers_peak():
    img = np.zeros((16, 32, 3), np.float32); img[:8] = 0.1
    env = HdriEnv(img); d = np.array([0.0, 1.0, 0.0]); env.add_light(d, radiance=100.0, sigma=0.05)
    peak = env.radiance(d[None, :])[0, 0]
    blurred = env.radiance(d[None, :], blur=np.array([0.1]))[0, 0]
    assert blurred < peak and blurred > 0.1
    # energy: integrate L over a fine polar grid around the lobe
    th = np.linspace(0, 0.6, 400); dth = th[1] - th[0]
    dirs = np.column_stack([np.sin(th), np.cos(th), np.zeros_like(th)])
    E0 = ((env.radiance(dirs)[:, 0] - 0.1) * np.sin(th) * 2 * np.pi * dth).sum()
    E1 = ((env.radiance(dirs, blur=np.full(len(th), 0.1))[:, 0] - 0.1) * np.sin(th) * 2 * np.pi * dth).sum()
    assert abs(E1 - E0) / E0 < 0.05


def test_footprint_filter_only_changes_dispersive_pixels(mind):
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    cam = mind.camera(eye=(0.0, 1.2, 3.0), target=(0.0, 0.0, 0.0), fov_deg=35.0, aspect=1.0)
    ck = lambda P: np.where(((np.floor(P[:, 0] / 0.2) + np.floor(P[:, 2] / 0.2)).astype(int) % 2 == 0)[:, None], [[0.9] * 3], [[0.1] * 3])
    env = lambda D: np.ones((len(D), 3))
    b0 = mind.bake_glass(sphere(0.6), cam, 32, 32, n_d=1.5, abbe=1e9, n_lams=5, floor_y=-0.7)   # no dispersion
    p0 = mind.relight_glass(b0, env, floor_albedo=ck); f0 = mind.relight_glass(b0, env, floor_albedo=ck, footprint_filter=True)
    assert np.allclose(p0, f0)                                                        # zero spread: identical
    b1 = mind.bake_glass(sphere(0.6), cam, 32, 32, n_d=2.4, abbe=10.0, n_lams=5, floor_y=-0.7)  # very dispersive
    p1 = mind.relight_glass(b1, env, floor_albedo=ck); f1 = mind.relight_glass(b1, env, floor_albedo=ck, footprint_filter=True)
    assert not np.allclose(p1, f1)
    g = b1.glass.reshape(32, 32)
    assert f1[g].std() <= p1[g].std()                                                  # the filter smooths, never sharpens


def test_druse_habit_paves_a_cavity_wall():
    GR, GT = 0.9, 0.18
    _, lin = cg.geode(radius=GR, shell=GT, count=900, habit="druse", size=0.16, size_jitter=0.3, seed=1, cull=True,
                      parts=True, clip_to_skin=True, skin_margin=0.06)
    _, thin = cg.geode(radius=GR, shell=GT, count=560, habit="quartz", size=0.115, size_jitter=0.35, seed=1, cull=True,
                       parts=True, clip_to_skin=True, skin_margin=0.06)
    rng = np.random.default_rng(0); D = rng.normal(size=(4000, 3)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    P = D * (GR - GT - 0.03)
    cov_druse = (lin(P) < 0).mean(); cov_thin = (thin(P) < 0).mean()
    assert cov_druse > 0.95 and cov_thin < 0.5                                         # 99% vs 32% measured


def test_habit_proportions_pyramidal_points_measured():
    """Sweep 157: the (1,0,1) form caps c at the same z whatever the prism radius -- so 'druse'/'amethyst_druse' are
    drums, not pyramids (kept negatives). quartz_point / amethyst_pyramid put the pyramid distance below the prism's."""
    g = np.linspace(-2, 2, 81); G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    def extents(name):
        ins = G[cg.habit_sdf(name, 1.0)(G) < 0]
        r0 = np.linalg.norm(ins[np.abs(ins[:, 2]) < 0.05][:, :2], axis=1).max()
        zmax = np.abs(ins[:, 2]).max()
        near_tip = ins[np.abs(ins[:, 2]) > 0.85 * zmax]
        rtip = np.linalg.norm(near_tip[:, :2], axis=1).max() if len(near_tip) else 0.0
        return r0, zmax, rtip / r0
    r_d, z_d, tip_d = extents("druse"); r_q, z_q, tip_q = extents("quartz_point"); r_a, z_a, tip_a = extents("amethyst_pyramid")
    assert abs(z_d - 1.5) < 0.1 and z_d > 1.3 * r_d                # the drum: the pyramid cap sits at 1.5 whatever the prism
    assert abs(z_q - 0.9) < 0.1 and abs(r_q - 0.9) < 0.1 and tip_q < 0.45   # a point: narrows toward the tip
    assert z_a < 0.85 and tip_a < 0.5                              # pyramidal


def test_directional_haze_has_a_lit_and_a_dark_side(mind):
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    cam = mind.camera(eye=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), fov_deg=30.0, aspect=1.0)
    b = mind.bake_glass(sphere(0.6), cam, 32, 32, n_lams=3, max_internal=2)
    img = np.zeros((16, 32, 3), np.float32); img[:8] = 0.05
    env = HdriEnv(img); env.add_light((1.0, 0.3, 0.3), irradiance=5.0, sigma=0.2)         # key from the RIGHT
    fv = mind.gem_flaw_volume(field=lambda P: np.ones(len(P)), bounds=[(-0.7, 0.7)] * 3, res=8, dim=1024)
    out = mind.relight_glass(b, env, sdf=sphere(0.6), sun_dir=tuple(env.dominant_dir), flaw_volume=fv, flaw_sigma=6.0,
                             flaw_albedo=(0.8, 0.8, 0.8)).reshape(32, 32, 3)
    g = b.glass.reshape(32, 32)
    right = out[:, 16:][g[:, 16:]].mean(); left = out[:, :16][g[:, :16]].mean()
    assert right > 1.3 * left                                       # the haze is brighter on the lit side

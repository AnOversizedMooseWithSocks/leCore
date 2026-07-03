"""Sweep 3 item 8: one shared spherical-harmonic primitive for directional sound AND light."""
import numpy as np
from holographic_spharm import sphere_dirs, sh_project, sh_reconstruct


def test_light_and_sound_same_primitive():
    dirs = sphere_dirs(400)
    ld = np.array([0.3, 0.5, 0.8]); ld /= np.linalg.norm(ld)
    radiance = np.clip(dirs @ ld, 0, None) ** 2
    rl = sh_reconstruct(sh_project(dirs, radiance, 4), dirs, 4)
    assert np.sqrt(np.mean((rl - radiance) ** 2)) / (radiance.std() + 1e-9) < 0.25
    sd = np.array([-0.6, 0.2, 0.7]); sd /= np.linalg.norm(sd)
    gain = np.clip(dirs @ sd, 0, None)
    rs = sh_reconstruct(sh_project(dirs, gain, 4), dirs, 4)
    assert np.sqrt(np.mean((rs - gain) ** 2)) / (gain.std() + 1e-9) < 0.25


def test_higher_order_sharpens():
    dirs = sphere_dirs(400)
    ld = np.array([0.3, 0.5, 0.8]); ld /= np.linalg.norm(ld)
    tight = np.clip(dirs @ ld, 0, None) ** 8
    e2 = np.sqrt(np.mean((sh_reconstruct(sh_project(dirs, tight, 2), dirs, 2) - tight) ** 2))
    e4 = np.sqrt(np.mean((sh_reconstruct(sh_project(dirs, tight, 4), dirs, 4) - tight) ** 2))
    assert e4 < e2


def test_rgb_and_deterministic():
    dirs = sphere_dirs(300)
    rgb = np.abs(np.stack([dirs[:, 0], dirs[:, 1], dirs[:, 2]], axis=1))
    cc = sh_project(dirs, rgb, 3)
    assert cc.shape == (9, 3) and sh_reconstruct(cc, dirs, 3).shape == (len(dirs), 3)
    assert np.allclose(sh_project(dirs, rgb[:, 0], 3), sh_project(dirs, rgb[:, 0], 3))

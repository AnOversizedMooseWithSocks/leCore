"""Inverse-rendering ST2: example-based texture synthesis (Image Quilting) -- patch search = HoloForest recall_k."""
import numpy as np
from holographic.materials_and_texture.holographic_texturesynth import synthesize_texture, find_similar_patches, _seam_energy


def _sample():
    rng = np.random.default_rng(0); yy, xx = np.mgrid[0:48, 0:48].astype(float)
    base = 0.5 + 0.3 * np.sin((xx + yy) / 3.0) + 0.1 * rng.standard_normal((48, 48))
    return np.clip(np.stack([base, base * 0.9 + 0.05, base * 0.8], axis=-1), 0, 1)


def test_grows_larger():
    assert synthesize_texture(_sample(), 96, 96, psize=20, overlap=6, seed=0).shape == (96, 96, 3)


def test_statistics_match_sample():
    s = _sample(); syn = synthesize_texture(s, 96, 96, psize=20, overlap=6, seed=0)
    assert abs(syn.mean() - s.mean()) < 0.05 and abs(syn.std() - s.std()) < 0.05


def test_mincut_beats_hardcut():
    s = _sample()
    mc = synthesize_texture(s, 96, 96, psize=20, overlap=6, seed=0, seam="mincut")
    hd = synthesize_texture(s, 96, 96, psize=20, overlap=6, seed=0, seam="hard")
    assert _seam_energy(mc) < _seam_energy(hd)


def test_grayscale_in_grayscale_out():
    g = _sample().mean(axis=-1)
    syn = synthesize_texture(g, 80, 80, psize=20, overlap=6, seed=0)
    assert syn.ndim == 2 and syn.shape == (80, 80)


def test_native_patch_search():
    s = _sample()
    found, sims = find_similar_patches(s, s[5:25, 5:25], k=6)
    assert found.shape[1:] == (20, 20, 3) and sims[0] > 0.9   # sample contains near-duplicates of its own patch


def test_deterministic():
    s = _sample()
    a = synthesize_texture(s, 64, 64, psize=20, overlap=6, seed=1)
    b = synthesize_texture(s, 64, 64, psize=20, overlap=6, seed=1)
    assert np.array_equal(a, b)

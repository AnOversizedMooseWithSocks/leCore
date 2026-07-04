"""Tests for the 2D / text / learning / utility tool wiring + catalog discoverability (gap closure)."""
import numpy as np
from holographic_unified import UnifiedMind
from holographic_catalog import default_catalog, seed_from_modules


def _mind():
    return UnifiedMind(dim=128, seed=0)


def test_recolor_image_faculty():
    m = _mind()
    a = np.random.default_rng(0).random((12, 12, 3))
    ref = np.random.default_rng(1).random((12, 12, 3))
    out = m.recolor_image(a, ref)
    assert out.shape == a.shape and np.all(np.isfinite(out))


def test_blend_images_faculty():
    m = _mind()
    a = np.random.default_rng(0).random((12, 12, 3))
    b = np.random.default_rng(1).random((12, 12, 3))
    frames = m.blend_images(a, b, steps=7)
    assert len(frames) == 7
    assert np.allclose(frames[0], a) and np.allclose(frames[-1], b)     # endpoints are the inputs


def _cat():
    return seed_from_modules(default_catalog())


def test_2d_family_discoverable():
    cat = _cat()
    for q in ["draw a picture", "make a 2d drawing", "paint on a canvas", "edit an image",
              "recolor an image", "crossfade two images"]:
        hits = cat.find_capability(q, k=3)
        assert any("2d image" in h.name.lower() for h in hits), (q, [h.name for h in hits])


def test_text_generation_discoverable():
    cat = _cat()
    for q in ["generate text", "write a sentence", "write a paragraph"]:
        assert any("text generation" in h.name.lower() for h in cat.find_capability(q, k=3)), q


def test_language_learning_discoverable():
    cat = _cat()
    for q in ["learn from a corpus", "language curriculum", "learn word meanings"]:
        assert any("language learning" in h.name.lower() for h in cat.find_capability(q, k=3)), q


def test_utilities_discoverable():
    cat = _cat()
    # unambiguously-utility queries land on the Utilities home
    for q in ["verify data integrity", "erasure code for reliability"]:
        assert any("utilit" in h.name.lower() for h in cat.find_capability(q, k=3)), q
    # content-addressing legitimately overlaps Utilities / Compression / the render farm -- any curated home is fine
    hits = cat.find_capability("content address a file", k=3)
    assert any(not h.name.startswith("holographic_") for h in hits)


def test_gap_tool_reports_zero_for_fixed_families():
    import importlib.util, os
    path = os.path.join(os.path.dirname(__file__), "tools", "catalog_gaps.py")
    spec = importlib.util.spec_from_file_location("catalog_gaps", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.report() == 0                                            # all four families now have curated homes

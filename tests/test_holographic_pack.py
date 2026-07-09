"""Tests for the lossless delta set packer."""
import numpy as np
import pytest
from holographic.misc.holographic_pack import pack, unpack, packed_bytes, benchmark, _suite, _ramp


def _exact(images):
    back = unpack(pack(images))
    return all(np.array_equal(np.asarray(images[i], np.uint8), np.asarray(back[i], np.uint8))
               for i in range(len(images)))


class TestPack:
    def test_lossless_roundtrip_color(self):
        assert _exact(_suite())

    def test_lossless_roundtrip_gradients(self):
        assert _exact(_ramp())

    def test_lossless_roundtrip_grayscale(self):
        rng = np.random.default_rng(0)
        base = rng.integers(0, 256, (40, 40), np.uint8)
        imgs = [np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
                for _ in range(5)]
        assert _exact(imgs)

    def test_beats_per_file_png_on_shared_structure(self):
        rows = dict((nm.split(" [")[0], b) for nm, b, _ in benchmark(_suite()))
        # a set with big identical regions should pack well under per-file PNG
        assert rows["set-pack (delta)"] < 0.6 * rows["per-file PNG"]

    def test_handles_single_image(self):
        img = _suite()[0]
        assert _exact([img])

    def test_rejects_foreign_blob(self):
        with pytest.raises(ValueError):
            unpack(b"not a pack blob at all")

    def test_size_helper_matches(self):
        imgs = _suite()
        assert packed_bytes(imgs) == len(pack(imgs))


if __name__ == "__main__":
    import sys; sys.exit(pytest.main([__file__, "-v"]))


# ======================================================================================================
# The packer has a door now -- and its baseline is the engine's own encoder, not Pillow's.
# ======================================================================================================
def test_the_benchmark_needs_no_image_library_in_core():
    """`benchmark` used to import PIL to build its own PNG baseline: an image library in the core, against the
    rules, duplicating a pure-stdlib encoder the engine already shipped. The PNG rows are now the engine's."""
    import ast
    import pathlib
    src = pathlib.Path("holographic/misc/holographic_pack.py").read_text()
    tree = ast.parse(src)
    pil_lines = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("PIL")]
    assert len(pil_lines) <= 1, "PIL may appear ONCE, guarded, for the optional lossy JPEG row"
    assert "except ImportError" in src, "the one PIL import must be optional"
    names = [r[0] for r in benchmark(_suite())]
    assert any("PNG" in n for n in names) and any("set-pack" in n for n in names)


def test_the_packer_wins_on_shared_structure_and_loses_on_gradients():
    """Both claims, against the engine's own (now competitive) PNG encoder. The negative matters more than the
    positive: this codec is 16x WORSE than per-file PNG on the wrong content."""
    def row(rows, key):
        return next(b for n, b, _ in rows if key in n)

    suite = benchmark(_suite())
    assert row(suite, "set-pack") < 0.6 * row(suite, "per-file PNG")       # measured 1,744 vs 3,553
    assert row(suite, "set-pack") < row(suite, "gzip the PNGs together")   # ...and beats gzip-the-set too

    ramp = benchmark(_ramp())
    assert row(ramp, "set-pack") > 10.0 * row(ramp, "per-file PNG")        # measured 32,274 vs 1,987 -- LOUD


def test_pack_round_trips_bit_exactly_through_the_mind():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    imgs = _suite()
    blob = m.pack_images(imgs)
    back = m.unpack_images(blob)
    assert len(back) == len(imgs)
    for a, b in zip(imgs, back):
        assert np.array_equal(np.asarray(a, np.uint8), np.asarray(b, np.uint8))   # byte for byte
    assert len(blob) < sum(a.size for a in imgs) // 10                             # and it is actually small
    names = [r[0] for r in m.pack_benchmark(imgs)]
    assert any("set-pack" in n for n in names)
    assert any("set-pack" in c.name.lower() or "set-packing" in c.name.lower()
               for c in m.find_capability("lossless set packer"))

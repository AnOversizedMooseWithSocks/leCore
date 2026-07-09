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

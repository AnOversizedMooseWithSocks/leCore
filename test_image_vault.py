"""Tests for the format-agnostic ImageVault: relate, compress, retrieve."""
import numpy as np, pytest
from PIL import Image
from image_vault import ImageVault, to_rgba


def _palette_set(n=10, S=24):
    rng = np.random.default_rng(0)
    pal = rng.integers(0, 256, (12, 4), np.uint8); pal[:, 3] = 255
    base = pal[rng.integers(0, 12, (S, S))]
    out = []
    for k in range(n):
        a = base.copy(); a[6:12, 6:12] = pal[rng.integers(0, 12, (6, 6))]
        out.append(a.astype(np.uint8))
    return out


def _roundtrip(items):
    v = ImageVault()
    for i, a in enumerate(items):
        v.add(a, f"x{i}")
    back = ImageVault.load(v.pack())
    return v, back


class TestVault:
    def test_roundtrip_palette_set(self):
        v, back = _roundtrip(_palette_set())
        assert all(np.array_equal(v.images[i], back.images[i]) for i in range(len(v)))
        assert len(v.pack()) < sum(a.size for a in v.images)        # actually compresses

    def test_roundtrip_mixed_sizes_and_colours(self):
        rng = np.random.default_rng(1)
        items = [rng.integers(0, 256, sz, np.uint8) for sz in [(40, 30, 3), (16, 64, 3), (50, 50, 3)]]
        v, back = _roundtrip(items)
        assert all(np.array_equal(to_rgba(v.images[i]), back.images[i]) for i in range(len(v)))

    def test_normalises_any_input(self):
        rng = np.random.default_rng(2)
        v = ImageVault()
        v.add(rng.integers(0, 256, (20, 20, 3), np.uint8), "nd_rgb")       # numpy RGB
        v.add(Image.fromarray(rng.integers(0, 256, (20, 20), np.uint8)), "pil_gray")  # PIL grayscale
        assert all(im.shape[-1] == 4 for im in v.images)                  # everything -> RGBA
        back = ImageVault.load(v.pack())
        assert all(np.array_equal(v.images[i], back.images[i]) for i in range(len(v)))

    def test_query_by_example(self):
        items = _palette_set()
        v = ImageVault()
        for i, a in enumerate(items):
            v.add(a, f"x{i}")
        top = v.most_similar(items[3], k=1)
        assert top[0][0] == "x3" and top[0][1] > 0.99                      # finds itself

    def test_clusters_group_similar(self):
        base = _palette_set(1)[0]
        v = ImageVault()
        for k in range(5):
            v.add(base, f"same{k}")                                        # identical -> one cluster
        v.add(np.zeros_like(base), "diff")
        cl = v.clusters(0.9)
        assert max(len(c) for c in cl) >= 5

    def test_lossy_beats_lossless_on_photos(self):
        # smooth RGB "photos": a lossy codec should be much smaller and decode faithfully
        imgs = []
        for k in range(4):
            yy, xx = np.mgrid[0:64, 0:64] / 64.0
            g = (np.clip(0.5 + 0.4 * np.sin(3 * xx + k) + 0.2 * np.cos(4 * yy), 0, 1) * 255).astype(np.uint8)
            imgs.append(np.dstack([g, g, g]))
        v = ImageVault()
        for i, a in enumerate(imgs):
            v.add(a, f"p{i}")
        lossy, lossless = v.pack(lossy=True, quality=85), v.pack()
        assert len(lossy) < len(lossless)
        back = ImageVault.load(lossy)
        assert len(back) == len(v) and all(back.images[i].shape[:2] == v.images[i].shape[:2] for i in range(len(v)))
        import image_vault as m
        assert np.mean([m._psnr(v.images[i], back.images[i]) for i in range(len(v))]) > 25

    def test_rejects_foreign_blob(self):
        with pytest.raises(ValueError):
            ImageVault.load(b"nope")


if __name__ == "__main__":
    import sys; sys.exit(pytest.main([__file__, "-v"]))

"""Tests for the palette-indexed sprite packer."""
import numpy as np, zlib, pytest
import pack_sprites as ps


def _sprites(n=12, S=32):
    """Synthetic palette sprites sharing a small colour set (like a sprite sheet)."""
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 256, (16, 4), np.uint8); palette[..., 3] = 255
    base = palette[rng.integers(0, 16, (S, S))]
    items = []
    for k in range(n):
        a = base.copy()
        a[10:16, 10:16] = palette[rng.integers(0, 16, (6, 6))]   # small per-frame change
        items.append((f"spr_{k}.gif", a.astype(np.uint8)))
    return items


class TestPackSprites:
    def test_roundtrip_exact(self):
        items = _sprites()
        back = ps.unpack(ps.pack(items))
        assert all(np.array_equal(items[i][1], back[i][1]) and items[i][0] == back[i][0]
                   for i in range(len(items)))

    def test_smaller_than_loose(self):
        items = _sprites()
        loose = sum(a.size for _, a in items)
        assert len(ps.pack(items)) < loose // 2

    def test_rejects_too_many_colours(self):
        rng = np.random.default_rng(1)
        big = [("x.gif", rng.integers(0, 256, (40, 40, 4), np.uint8))]   # ~all unique colours
        with pytest.raises(ValueError):
            ps.pack(big)

    def test_rejects_foreign_blob(self):
        with pytest.raises(ValueError):
            ps.unpack(b"nope")


if __name__ == "__main__":
    import sys; sys.exit(pytest.main([__file__, "-v"]))

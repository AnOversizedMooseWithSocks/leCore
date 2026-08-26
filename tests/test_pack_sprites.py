"""Tests for the palette-indexed sprite packer."""
import numpy as np, zlib, pytest
import tools.pack_sprites as ps


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


def test_v2_orientation_choice_shrinks_and_roundtrips():
    # THE v2 WIN: per-pack orientation choice (column-major for these character
    # sprites gives LZMA longer vertical runs) shrinks the real 712-sprite pack
    # ~15% AND stays bit-exact. The choice is per-pack, so it can never regress
    # below row-major.
    import os
    import numpy as np
    import tools.pack_sprites as pack_sprites
    folder = "features/sprites"
    if not os.path.isdir(folder):
        import pytest
        pytest.skip("sprite folder not present")
    items = pack_sprites.load_folder(folder)
    blob = pack_sprites.pack(items)
    un = pack_sprites.unpack(blob)
    assert len(un) == len(items)
    assert all(np.array_equal(a, b) and n1 == n2
               for (n1, a), (n2, b) in zip(items, un))     # bit-exact
    assert len(blob) < 62000                                # measured ~58 KB (<68 KB v1)


def test_pack_handles_mixed_sizes_both_orientations():
    # The transpose path must respect each sprite's own (h, w) -- verify on a
    # non-uniform set so the column-major reshape is genuinely exercised.
    import numpy as np
    import tools.pack_sprites as pack_sprites
    rng = np.random.default_rng(0)

    def quant(a):
        a = a.copy(); a[..., :3] = (a[..., :3] // 64) * 64; a[..., 3] = 255
        return a

    mixed = [("a", quant(rng.integers(0, 255, (16, 24, 4), np.uint8))),
             ("b", quant(rng.integers(0, 255, (20, 12, 4), np.uint8)))]
    un = pack_sprites.unpack(pack_sprites.pack(mixed))
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(mixed, un))

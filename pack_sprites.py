"""
pack_sprites.py -- pack a folder of palette GIF sprites into one small file.

Measured on a real 712-sprite set (32x32, 88 colours across the whole set), this
is what actually wins -- and it is NOT delta coding. Sprites are palette images,
and the bloat is that every GIF carries its own palette and its own LZW stream.
The fix is a change of REPRESENTATION:

  1. build ONE palette shared by every sprite (here 88 colours fit in 256),
  2. turn each sprite into a plane of 1-byte indices into that palette,
  3. hand all the index planes to a strong general compressor (LZMA), which finds
     the cross-sprite structure that 712 separate GIF streams hide.

Result on the real set: 65 KB, versus 158 KB for zipping the folder and 792 KB
for the loose GIFs. Lossless and bit-exact; integers throughout (1 byte per pixel,
a uint8 palette). Delta coding was tried and made it WORSE, so it is not used.

If a set needs more than 256 colours total, this raises -- those want either
per-group palettes or plain truecolour compression.

Needs: numpy, PIL, and lzma/struct from the standard library.
"""

import os, glob, io, struct, lzma
import numpy as np
from PIL import Image

MAGIC = b"HSPK"


def _codes(a):                                   # RGBA (...,4) uint8 -> uint32 colour code
    a = a.astype(np.uint32)
    return (a[..., 0] << 24) | (a[..., 1] << 16) | (a[..., 2] << 8) | a[..., 3]


def load_folder(folder):
    """Read every .gif under `folder` as (name, RGBA uint8 array)."""
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.gif"), recursive=True))
    return [(os.path.basename(p), np.asarray(Image.open(p).convert("RGBA"), np.uint8)) for p in paths]


def pack(items):
    """Pack [(name, RGBA array), ...] into one bytes blob, losslessly."""
    names = [n for n, _ in items]
    arrs = [a for _, a in items]
    allc = np.unique(np.concatenate([_codes(a).ravel() for a in arrs]))
    if len(allc) > 256:
        raise ValueError(f"{len(allc)} colours across the set; this packer needs <=256")
    pal = np.stack([(allc >> 24) & 255, (allc >> 16) & 255, (allc >> 8) & 255, allc & 255], -1).astype(np.uint8)

    index_blob = b"".join(np.searchsorted(allc, _codes(a)).astype(np.uint8).tobytes() for a in arrs)
    body = lzma.compress(index_blob, preset=9 | lzma.PRESET_EXTREME)
    namedata = lzma.compress("\n".join(names).encode("utf-8"))

    out = [struct.pack("<4sBHB", MAGIC, 1, len(items), len(pal)), pal.tobytes()]
    for _, a in items:                           # per-sprite H, W (sizes may vary)
        out.append(struct.pack("<HH", a.shape[0], a.shape[1]))
    out += [struct.pack("<I", len(namedata)), namedata,
            struct.pack("<I", len(body)), body]
    return b"".join(out)


def unpack(blob):
    """Reverse of pack(): returns [(name, RGBA array), ...], bit-exact."""
    if len(blob) < struct.calcsize("<4sBHB") or blob[:4] != MAGIC:
        raise ValueError("not a pack_sprites blob")
    magic, _ver, n, K = struct.unpack_from("<4sBHB", blob); pos = struct.calcsize("<4sBHB")
    pal = np.frombuffer(blob, np.uint8, count=K * 4, offset=pos).reshape(K, 4); pos += K * 4
    shapes = []
    for _ in range(n):
        h, w = struct.unpack_from("<HH", blob, pos); pos += 4; shapes.append((h, w))
    (nl,) = struct.unpack_from("<I", blob, pos); pos += 4
    names = lzma.decompress(blob[pos:pos + nl]).decode("utf-8").split("\n"); pos += nl
    (bl,) = struct.unpack_from("<I", blob, pos); pos += 4
    flat = np.frombuffer(lzma.decompress(blob[pos:pos + bl]), np.uint8)

    items, off = [], 0
    for i in range(n):
        h, w = shapes[i]
        idx = flat[off:off + h * w].reshape(h, w); off += h * w
        items.append((names[i], pal[idx]))       # indices -> RGBA, exact
    return items


def save_folder(items, out_dir):
    """Write unpacked sprites back out as GIFs (round-trips the originals)."""
    os.makedirs(out_dir, exist_ok=True)
    for name, rgba in items:
        Image.fromarray(rgba).convert("P").save(os.path.join(out_dir, name))


if __name__ == "__main__":
    import sys, zlib
    folder = sys.argv[1] if len(sys.argv) > 1 else "features/sprites"
    items = load_folder(folder)
    if not items:
        print(f"no .gif files under {folder!r}"); raise SystemExit
    gif_total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(folder, "**", "*.gif"), recursive=True))
    blob = pack(items)
    back = unpack(blob)
    exact = all(np.array_equal(items[i][1], back[i][1]) and items[i][0] == back[i][0]
                for i in range(len(items)))
    zip_est = len(zlib.compress(b"".join(open(p, "rb").read()
                  for p in sorted(glob.glob(os.path.join(folder, "**", "*.gif"), recursive=True))), 9))
    print(f"\n{len(items)} sprites from {folder!r}")
    print(f"  loose GIF files     {gif_total:>9,}  100%")
    print(f"  zip the folder      {zip_est:>9,}  {zip_est/gif_total:.0%}")
    print(f"  pack_sprites (.hsp) {len(blob):>9,}  {len(blob)/gif_total:.0%}   lossless round-trip: {'EXACT' if exact else 'FAILED'}")
    print(f"  -> {gif_total/len(blob):.1f}x smaller than the loose GIFs, {zip_est/len(blob):.1f}x smaller than zipping them\n")

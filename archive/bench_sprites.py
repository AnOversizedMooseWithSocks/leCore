# This one-off research script lives in archive/; put the library at the repo
# root on the path so its imports keep working when run from anywhere.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""bench_sprites.py -- measure the sprite packer on a real sprite set.

    python bench_sprites.py [folder-of-GIFs]

With no argument it runs on the bundled set (features/sprites.hsp, the real 712
sprites).  It compares the *latest* canonical packer -- pack_sprites.pack, which
maps the whole set onto one shared palette and LZMA-compresses the 1-byte index
planes -- against honest baselines:

    raw RGBA           every sprite as a full uint8 array (the floor to beat)
    PNG per file       optimised PNG each, summed
    zip the set        gzip over the concatenated PNGs (~zipping the folder)
    pack_sprites       shared palette + LZMA on the index planes  <-- the winner

Everything is lossless; the packed blob is verified to round-trip bit-exactly.
The lesson the numbers teach: for sprites the win is the *representation*
(a shared palette) far more than any clever diff.
"""
import io, os, sys, zlib
import numpy as np
from PIL import Image
import tools.pack_sprites as ps


def _load():
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        print(f"loading GIFs from {folder}")
        items = ps.load_folder(folder)
        loose = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)
                    if f.lower().endswith(".gif"))
        return items, loose
    hsp = os.path.join(os.path.dirname(__file__), "features", "sprites.hsp")
    print(f"no folder given -> bundled {os.path.basename(hsp)}")
    return ps.unpack(open(hsp, "rb").read()), None


def _png(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, "PNG", optimize=True); return b.getvalue()


def main():
    items, loose = _load()
    n = len(items)
    raw = sum(a.size for _, a in items)
    pngs = [_png(a) for _, a in items]
    png_total = sum(len(p) for p in pngs)
    zip_total = len(zlib.compress(b"".join(pngs), 9))
    blob = ps.pack(items)
    back = ps.unpack(blob)
    exact = len(back) == n and all(np.array_equal(a, b) for (_, a), (_, b) in zip(items, back))

    print(f"\n{n} sprites\n" + "-" * 44)
    rows = [("raw RGBA", raw), ("PNG per file", png_total), ("zip the set", zip_total),
            ("pack_sprites (palette+LZMA)", len(blob))]
    if loose:
        rows.insert(0, ("loose GIFs on disk", loose))
    base = png_total
    for name, size in rows:
        print(f"  {name:30s} {size / 1024:8.1f} KB   {base / size:4.1f}x vs PNG")
    print("-" * 44)
    print(f"  round-trip bit-exact: {exact}")


if __name__ == "__main__":
    main()

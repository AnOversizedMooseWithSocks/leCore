"""
holographic_pack.py -- a lossless delta "set packer" for families of images.

Single-file codecs (PNG, JPEG) compress every image on its own. So when a SET of
images shares structure -- a logo suite with one background and ring, sprite
variants, UI frames, scanned pages -- everything they have in common is paid for
again in every file. That repeated, shared content is the file bloat.

This packs a whole set as ONE reference image plus per-image deltas, entropy-coded
with zlib, so the shared part is stored once and each image costs only what makes
it different. It is the old demoscene move: store the diff, not the frame. The
residual is taken modulo 256, which makes the round trip bit-exact -- an 8-bit
image plus its mod-256 delta reconstructs to the original byte for byte. Integer
in, integer out; no floats anywhere.

The reference is chosen automatically (first image / per-pixel mean / median --
whichever packs smallest), because that single choice drives the residual size.

WHEN THIS HELPS, AND WHEN IT DOES NOT (measured -- see benchmark()):
  * It wins big when images share large BIT-IDENTICAL regions and differ in
    localized spots: the delta is itself sparse and zlib crushes it. On a 6-logo
    suite it packs to 1,744 B against 3,553 B of per-file PNG (49%) and 3,162 B
    of gzip-the-whole-set (55%) -- it beats both.
  * It LOSES, and badly, when each image is already highly compressible on its
    own (smooth gradients, photographs): per-file PNG/JPEG already exploit that,
    and the inter-image delta is not sparse. On the gradient ramp it packs to
    32,274 B against per-file PNG's 1,987 B -- SIXTEEN TIMES WORSE. For those,
    just PNG each image (and, if you like, zip the PNGs together).
    The benchmark shows both so the choice is clear. Run it; do not guess.

  Both baselines are the ENGINE's own pure-stdlib PNG encoder, not Pillow's.
  Measuring that swap is what exposed the encoder's missing scanline filters
  (it was 43x larger than Pillow on a gradient, and nobody had checked). A
  baseline you have not measured is not a baseline.

A lossy tier (integer Walsh-Hadamard on the residual) was tried and dropped: it
never beat JPEG, so shipping it would have been misleading.

Needs: numpy, plus zlib and struct from the standard library. (PIL is imported
only inside benchmark(), to compare against PNG/JPEG.)
"""

import numpy as np
import zlib
import struct

MAGIC = b"HPK1"


def _z(raw):
    return zlib.compress(raw, 9)


def _as_stack(images):
    """Stack a list of same-shaped uint8 images into (n, H, W, C); grayscale gets
    a trailing channel of 1 so the rest of the code is uniform."""
    stack = np.stack([np.asarray(im, dtype=np.uint8) for im in images])
    if stack.ndim == 3:
        stack = stack[..., None]
    return stack


def _references(stack):
    """The reference candidates we try; whichever packs smallest is kept. 'first'
    is ideal when one image already holds the shared structure; 'mean'/'median'
    centre the set so every image's delta is small."""
    return [(0, stack[0]),
            (1, np.round(stack.mean(0)).astype(np.uint8)),
            (2, np.round(np.median(stack, 0)).astype(np.uint8))]


def pack(images):
    """Pack a list of uint8 images (all the same shape) into one bytes blob,
    losslessly. Returns the blob; feed it to unpack() to get the images back."""
    stack = _as_stack(images)
    n, H, W, C = stack.shape

    best = None
    for refcode, ref in _references(stack):
        refp = _z(ref.astype(np.uint8).tobytes())
        # residual mod 256 -> bit-exact reconstruction; shared pixels become 0 and
        # vanish under zlib, so only the genuine differences cost anything.
        deltas = [_z(((im.astype(np.int16) - ref.astype(np.int16)) % 256)
                     .astype(np.uint8).tobytes()) for im in stack]
        size = len(refp) + sum(len(d) for d in deltas)
        if best is None or size < best[0]:
            best = (size, refcode, refp, deltas)

    _, refcode, refp, deltas = best
    head = struct.pack("<4sBHHHHB", MAGIC, 1, n, H, W, C, refcode)
    out = [head, struct.pack("<I", len(refp)), refp]
    for d in deltas:
        out += [struct.pack("<I", len(d)), d]
    return b"".join(out)


def unpack(blob):
    """Reverse of pack(): returns the original list of uint8 images, bit-exact."""
    fmt = "<4sBHHHHB"
    magic, _ver, n, H, W, C, _ref = struct.unpack_from(fmt, blob)
    if magic != MAGIC:
        raise ValueError("not a holographic_pack blob")
    pos = struct.calcsize(fmt)

    def take():
        nonlocal pos
        (ln,) = struct.unpack_from("<I", blob, pos); pos += 4
        chunk = blob[pos:pos + ln]; pos += ln
        return chunk

    ref = np.frombuffer(zlib.decompress(take()), np.uint8).reshape(H, W, C).astype(np.int16)
    images = []
    for _ in range(n):
        d = np.frombuffer(zlib.decompress(take()), np.uint8).reshape(H, W, C).astype(np.int16)
        im = ((ref + d) % 256).astype(np.uint8)        # exact round trip
        images.append(im if C > 1 else im[..., 0])
    return images


def packed_bytes(images):
    return len(pack(images))


# ---------------------------------------------------------------------------
# honest benchmark: the packer vs the alternatives you would actually reach for
# ---------------------------------------------------------------------------
def _psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def benchmark(images):
    """Returns rows (name, bytes, psnr) comparing per-file PNG, gzip-the-PNGs,
    the set packer, and -- only if Pillow happens to be installed -- per-file
    JPEG as a lossy reference point.

    The PNG baseline is the ENGINE's own encoder (holographic_render.png_bytes),
    not Pillow's. This module used to import PIL to make its own baseline, which
    (a) put an image library in the core, against the rules, and (b) duplicated a
    pure-stdlib PNG encoder the engine already shipped. Measuring that swap is
    what turned up the encoder's missing scanline filters -- it was 43x larger
    than Pillow on a gradient. Filtered, it is within 20% of Pillow there and
    beats it 2x on flat art, so it is now an honest baseline as well as the
    right dependency."""
    import io
    from holographic.rendering.holographic_render import png_bytes    # the shared stdlib encoder; no PIL in core
    stack = _as_stack(images)
    n = stack.shape[0]
    rgb = stack.shape[-1] == 3
    pick = lambda im: im if rgb else im[..., 0]

    def png_blob(im):
        a = pick(im)
        if a.ndim == 2:
            a = np.repeat(a[:, :, None], 3, axis=2)                   # the encoder is 8-bit RGB
        return png_bytes(a.astype(float) / 255.0, level=6)

    pngs = [png_blob(im) for im in stack]
    rows = [("raw uint8", stack.size, float("inf")),
            ("per-file PNG", sum(len(p) for p in pngs), float("inf")),
            ("gzip the PNGs together", len(_z(b"".join(pngs))), float("inf"))]

    blob = pack(images)
    back = unpack(blob)
    exact = all(np.array_equal(np.asarray(images[i], np.uint8), np.asarray(back[i], np.uint8))
                for i in range(n))
    rows.append((f"set-pack (delta) {'[exact]' if exact else '[BROKEN]'}", len(blob), float("inf")))

    # The lossy reference point needs a DCT encoder, which the engine does not ship and will not add to core.
    # It is genuinely optional: without Pillow the comparison is simply reported as unavailable, not faked.
    try:
        from PIL import Image                                          # OPTIONAL, and only for the lossy baseline
    except ImportError:
        rows.append(("per-file JPEG (lossy) -- skipped, Pillow not installed", 0, float("nan")))
        return rows
    for q in (75, 90):
        tot = 0; ps = []
        for im in stack:
            b = io.BytesIO(); Image.fromarray(pick(im)).save(b, "JPEG", quality=q)
            data = b.getvalue(); tot += len(data)
            r = np.asarray(Image.open(io.BytesIO(data)).convert("RGB" if rgb else "L"), np.uint8)
            ps.append(_psnr(np.asarray(images[len(ps)]), r))
        rows.append((f"per-file JPEG q{q} (lossy)", tot, float(np.mean(ps))))
    return rows


# ---------------------------------------------------------------------------
# self-contained demos: one set where the packer wins, one where it does not
# ---------------------------------------------------------------------------
def _suite(S=96, n=6):
    """Six logos sharing a dark background and a teal ring, differing only in a
    small central mark -- big bit-identical regions, the packer's best case."""
    yy, xx = np.mgrid[0:S, 0:S] / S
    r = np.sqrt((xx - .5) ** 2 + (yy - .5) ** 2)
    base = np.zeros((S, S, 3), np.uint8); base[...] = (14, 22, 38)
    base[np.abs(r - 0.42) < 0.03] = (101, 231, 197)
    marks = [(252, 155, 113), (99, 220, 190), (240, 153, 123),
             (255, 209, 102), (160, 90, 200), (101, 231, 197)]
    out = []
    for k in range(n):
        im = base.copy()
        cx, cy = .5 + .05 * np.sin(k), .5 + .05 * np.cos(k)
        inner = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < 0.16 + 0.02 * (k % 3)
        im[inner] = marks[k]; out.append(im)
    return out


def _ramp(S=96, n=6):
    """One smooth gradient at several brightness levels -- already trivially
    PNG-compressible, so the packer should and does lose here."""
    yy, xx = np.mgrid[0:S, 0:S] / S
    g = np.stack([xx, 1 - xx, np.abs(yy - .5) * 2], -1)
    return [(np.clip(g * (0.6 + 0.08 * k) + 0.05 * k, 0, 1) * 255).astype(np.uint8) for k in range(n)]


def _show(title, images):
    rows = benchmark(images)
    png = next(b for nm, b, _ in rows if nm == "per-file PNG")
    print(f"\n{title}  ({len(images)} images, {np.asarray(images[0]).shape})")
    print(f"  {'method':28s}{'bytes':>9}{'vs PNG':>9}   fidelity")
    print("  " + "-" * 58)
    for nm, b, ps in rows:
        rel = "--" if nm.startswith("raw") else f"{b/png:.0%}"
        fid = "lossless" if ps == float("inf") else f"{ps:.0f} dB"
        print(f"  {nm:28s}{b:9d}{rel:>9}   {fid}")


if __name__ == "__main__":
    print("=" * 64)
    print("holographic_pack -- lossless delta set packer for related images")
    print("=" * 64)
    print("\nStore a set as one reference + per-image deltas: shared structure is")
    print("kept once, not in every file. Bit-exact, uint8 throughout.")
    _show("A) logos -- big shared regions, small local marks (packer's case)", _suite())
    _show("B) smooth gradients -- already PNG-friendly (packer should lose)", _ramp())
    print("\n  Read it honestly: the packer wins when images share large identical")
    print("  regions (logos, sprites, UI frames) -- there the delta is sparse and")
    print("  zlib eats it, beating both per-file PNG and gzip-the-set. When each")
    print("  image is already compressible on its own (gradients, photos), reach")
    print("  for per-file PNG/JPEG instead. The delta-between-files lever is the")
    print("  one no single-file codec pulls; that is the whole trick.\n")

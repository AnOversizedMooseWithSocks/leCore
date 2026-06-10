"""
image_vault.py -- a format-agnostic store that can RELATE, COMPRESS and RETRIEVE
a collection of images.

The whole journey to here taught one lesson: there is no single best trick. Single
-image holographic coding loses to JPEG; delta coding wins only when images share
bit-identical regions; a shared palette + LZMA crushes sprite sets. So this vault
does not commit to a method. It:

  1. NORMALISES any input to RGBA uint8 -- any file format (PNG/GIF/JPEG/BMP/WebP/
     TIFF), any size, any colour mode, numpy arrays or PIL images alike. Nothing
     downstream is format-, size- or codec-specific.
  2. RELATES the images by a small size-invariant perceptual fingerprint, which
     gives pairwise similarity, clustering, and an ordering that puts similar
     images next to each other (which in turn helps general compressors).
  3. COMPRESSES adaptively: it actually measures a few lossless encoders
     (shared-palette + LZMA, LZMA over the related-ordered pixels, per-image PNG)
     and keeps whichever is smallest for THIS set, reporting the comparison
     honestly. Round-trips bit-exact.
  4. RETRIEVES: pull any image back exactly by index or name, or query by example
     -- hand it an image and get the nearest stored ones by fingerprint. Query
     still works after save/load, because fingerprints are recomputed from the
     decoded pixels rather than stored.

Needs: numpy, PIL, and zlib/lzma/struct from the standard library.
"""

import os, io, glob, struct, zlib, lzma
import numpy as np
from PIL import Image

MAGIC = b"IVLT"
IMAGE_EXTS = (".png", ".gif", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm")


# ---------------------------------------------------------------------------
# normalisation: anything -> RGBA uint8
# ---------------------------------------------------------------------------
def to_rgba(src):
    if isinstance(src, np.ndarray):
        a = src.astype(np.uint8)
        if a.ndim == 2:
            a = np.dstack([a, a, a, np.full(a.shape, 255, np.uint8)])
        elif a.shape[-1] == 3:
            a = np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])
        return np.ascontiguousarray(a[..., :4])
    if isinstance(src, Image.Image):
        return np.asarray(src.convert("RGBA"), np.uint8)
    if isinstance(src, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(src)).convert("RGBA"), np.uint8)
    return np.asarray(Image.open(src).convert("RGBA"), np.uint8)


def _codes(a):
    a = a.astype(np.uint32)
    return (a[..., 0] << 24) | (a[..., 1] << 16) | (a[..., 2] << 8) | a[..., 3]


def _png(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, "PNG", optimize=True); return b.getvalue()


from PIL import features as _pilfeat
_HAS_WEBP = _pilfeat.check("webp")


def _enc(a, fmt, quality=None):
    """Encode one RGBA array to a codec's bytes. JPEG has no alpha (RGB only);
    PNG and WebP keep it."""
    b = io.BytesIO()
    if fmt == "PNG":
        Image.fromarray(a).save(b, "PNG", optimize=True)
    elif fmt == "JPEG":
        Image.fromarray(a[..., :3]).save(b, "JPEG", quality=quality)
    elif fmt == "WEBP":
        Image.fromarray(a).save(b, "WEBP", quality=quality)
    return b.getvalue()


def _dec(data):
    """Decode any codec bytes (PIL detects the format) back to RGBA uint8."""
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"), np.uint8)


def _psnr(a, b):
    a = a[..., :3].astype(np.float64); b = b[..., :3].astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


class ImageVault:
    def __init__(self):
        self.names, self.images, self._fp = [], [], None

    # ---- ingest -----------------------------------------------------------
    def add(self, src, name=None):
        self.images.append(to_rgba(src))
        self.names.append(name if name is not None else f"img{len(self.images)-1}")
        self._fp = None
        return self

    def add_folder(self, folder):
        for p in sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True)):
            if p.lower().endswith(IMAGE_EXTS):
                self.add(p, os.path.relpath(p, folder))
        return self

    def __len__(self):
        return len(self.images)

    # ---- relate -----------------------------------------------------------
    @staticmethod
    def _fingerprint(a):
        """A 16x16 RGB thumbnail, mean-removed and L2-normalised -- size- and
        format-invariant, so cosine of two fingerprints is a perceptual match."""
        t = np.asarray(Image.fromarray(a).convert("RGB").resize((16, 16), Image.BOX), np.float32)
        v = t.ravel() - t.mean()
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def fingerprints(self):
        if self._fp is None:
            self._fp = (np.stack([self._fingerprint(a) for a in self.images])
                        if self.images else np.zeros((0, 768), np.float32))
        return self._fp

    def most_similar(self, query, k=5):
        """Query by example: returns [(name, similarity), ...] for the closest stored
        images to `query` (any format/size)."""
        q = self._fingerprint(to_rgba(query))
        sims = self.fingerprints() @ q
        order = np.argsort(-sims)[:k]
        return [(self.names[i], float(sims[i])) for i in order]

    def clusters(self, threshold=0.9):
        """Group images whose fingerprints are within `threshold` cosine, via
        connected components. Returns a list of index lists."""
        n = len(self.images)
        if n == 0:
            return []
        S = self.fingerprints() @ self.fingerprints().T
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(n):
            for j in np.where(S[i, i+1:] >= threshold)[0] + i + 1:
                parent[find(i)] = find(int(j))
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return sorted(groups.values(), key=len, reverse=True)

    def _order(self, threshold=0.9):
        """An ordering with related images adjacent -- helps general compressors."""
        order = []
        for c in self.clusters(threshold):
            order += c
        return order

    # ---- compress (adaptive, lossless) ------------------------------------
    def _candidates(self, order):
        imgs = [self.images[i] for i in order]
        out = {}
        # (1) per-image PNG -- the general lossless baseline
        out["png"] = [ _png(a) for a in imgs ]
        # (2) LZMA over the related-ordered raw pixels -- wins on highly similar sets
        out["lzma"] = lzma.compress(b"".join(a.tobytes() for a in imgs), preset=6)
        # (3) shared palette + LZMA on index planes -- wins on low-colour sets (sprites)
        allc = np.unique(np.concatenate([_codes(a).ravel() for a in imgs]))
        if len(allc) <= 256:
            pal = np.stack([(allc >> 24) & 255, (allc >> 16) & 255,
                            (allc >> 8) & 255, allc & 255], -1).astype(np.uint8)
            idx = b"".join(np.searchsorted(allc, _codes(a)).astype(np.uint8).tobytes() for a in imgs)
            out["palette"] = (pal, lzma.compress(idx, preset=9 | lzma.PRESET_EXTREME))
        return out

    def _sizeof(self, method, data):
        if method == "png":
            return sum(len(p) for p in data)
        if method == "lzma":
            return len(data)
        pal, body = data
        return pal.nbytes + len(body)

    def _candidates_lossy(self, order, quality):
        """Per-image lossy codecs. Returns {name: (total_bytes, blobs, psnr, fmt)}.
        Intended for photographs, where JPEG/WebP beat any lossless option."""
        imgs = [self.images[i] for i in order]
        out = {}
        for fmt in (["JPEG", "WEBP"] if _HAS_WEBP else ["JPEG"]):
            blobs = [_enc(a, fmt, quality) for a in imgs]
            recon = [_dec(b) for b in blobs]
            psnr = float(np.mean([_psnr(imgs[i], recon[i]) for i in range(len(imgs))]))
            out[f"{fmt.lower()} q{quality}"] = (sum(len(b) for b in blobs), blobs, psnr, fmt)
        return out

    def report(self, threshold=0.9, lossy_quality=85):
        """Honest size/fidelity comparison of every candidate encoder on this set.
        Lossless rows read 'inf' PSNR; lossy rows carry their measured PSNR."""
        order = self._order(threshold)
        rows = [(m, self._sizeof(m, d), float("inf")) for m, d in self._candidates(order).items()]
        if lossy_quality:
            for name, (sz, _b, psnr, _f) in self._candidates_lossy(order, lossy_quality).items():
                rows.append((name, sz, psnr))
        rows.append(("raw RGBA", sum(a.size for a in self.images), float("inf")))
        return sorted(rows, key=lambda r: r[1])

    def pack(self, threshold=0.9, lossy=None, quality=85):
        """Serialise the vault to one bytes blob. lossy=None -> the smallest LOSSLESS
        encoder (bit-exact). lossy=True (or 'jpeg'/'webp') -> per-image lossy codec at
        `quality` (smaller, approximate -- for photographs)."""
        order = self._order(threshold)
        if lossy:
            cl = self._candidates_lossy(order, quality)
            key = (next(k for k in cl if k.startswith(str(lossy).lower())) if isinstance(lossy, str)
                   else min(cl, key=lambda k: cl[k][0]))
            _sz, blobs, _ps, fmt = cl[key]
            method_code, kind, payload = {"JPEG": 3, "WEBP": 4}[fmt], "perimage", blobs
        else:
            cands = self._candidates(order)
            m = min(cands, key=lambda mm: self._sizeof(mm, cands[mm]))
            method_code = {"png": 0, "lzma": 1, "palette": 2}[m]
            kind = {"png": "perimage", "lzma": "lzma", "palette": "palette"}[m]
            payload = cands[m]

        names = lzma.compress("\n".join(self.names).encode("utf-8"))
        sizes = zlib.compress(np.array([self.images[i].shape[:2] for i in order], np.uint16).tobytes(), 9)
        perm = np.array(order, np.uint32).tobytes()
        head = struct.pack("<4sBHI", MAGIC, 1, method_code, len(self.images))
        parts = [head]
        for blob in (names, perm, sizes):
            parts += [struct.pack("<I", len(blob)), blob]
        if kind == "perimage":
            for p in payload:
                parts += [struct.pack("<I", len(p)), p]
        elif kind == "lzma":
            parts += [struct.pack("<I", len(payload)), payload]
        else:
            pal, body = payload
            parts += [struct.pack("<H", len(pal)), pal.tobytes(), struct.pack("<I", len(body)), body]
        return b"".join(parts)

    def save(self, path, threshold=0.9):
        with open(path, "wb") as f:
            f.write(self.pack(threshold))

    @classmethod
    def load(cls, path_or_bytes):
        blob = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
        if len(blob) < 11 or blob[:4] != MAGIC:
            raise ValueError("not an image_vault blob")
        _magic, _ver, method, n = struct.unpack_from("<4sBHI", blob, 0)
        pos = struct.calcsize("<4sBHI")

        def take():
            nonlocal pos
            (ln,) = struct.unpack_from("<I", blob, pos); pos += 4
            chunk = blob[pos:pos + ln]; pos += ln
            return chunk

        names = lzma.decompress(take()).decode("utf-8").split("\n")
        perm = np.frombuffer(take(), np.uint32)
        sizes = np.frombuffer(zlib.decompress(take()), np.uint16).reshape(n, 2).astype(np.int64)

        imgs_in_order = []
        if method in (0, 3, 4):                          # per-image codec (png / jpeg / webp)
            for _ in range(n):
                imgs_in_order.append(_dec(take()))
        elif method == 1:                                # lzma raw
            flat = np.frombuffer(lzma.decompress(take()), np.uint8); off = 0
            for h, w in sizes:
                imgs_in_order.append(flat[off:off + h*w*4].reshape(h, w, 4)); off += h*w*4
        else:                                            # palette
            (K,) = struct.unpack_from("<H", blob, pos); pos += 2
            pal = np.frombuffer(blob[pos:pos + K*4], np.uint8).reshape(K, 4); pos += K*4
            flat = np.frombuffer(lzma.decompress(take()), np.uint8); off = 0
            for h, w in sizes:
                idx = flat[off:off + h*w].reshape(h, w); off += h*w
                imgs_in_order.append(pal[idx])

        v = cls()
        v.images = [None] * n; v.names = [None] * n
        for slot, src_i in enumerate(perm):             # undo the compression ordering
            v.images[src_i] = np.ascontiguousarray(imgs_in_order[slot])
            v.names[src_i] = names[src_i]
        return v

    # ---- retrieve ---------------------------------------------------------
    def get(self, key):
        i = key if isinstance(key, int) else self.names.index(key)
        return self.images[i]


METHOD_NAME = {"png": "per-image PNG", "lzma": "LZMA (related order)",
               "palette": "shared palette + LZMA", "raw RGBA": "raw RGBA"}


if __name__ == "__main__":
    import sys, time
    folder = sys.argv[1] if len(sys.argv) > 1 else "features/sprites"
    v = ImageVault().add_folder(folder)
    if len(v) == 0:
        print(f"no images under {folder!r}"); raise SystemExit
    sizes = {a.shape[:2] for a in v.images}
    print(f"\n{len(v)} images from {folder!r}  ({len(sizes)} distinct size(s), any format)")
    cl = v.clusters(0.9)
    print(f"related into {len(cl)} clusters (largest {len(cl[0])} images)")
    print("\ncompression candidates (lossless):")
    for m, b in v.report():
        print(f"  {METHOD_NAME.get(m, m):26s}{b:>10,}")
    t = time.perf_counter(); blob = v.pack(); dt = time.perf_counter() - t
    back = ImageVault.load(blob)
    exact = all(np.array_equal(v.images[i], back.images[i]) and v.names[i] == back.names[i]
                for i in range(len(v)))
    print(f"\npacked: {len(blob):,} bytes in {dt:.1f}s   round-trip: {'EXACT' if exact else 'FAILED'}")
    # query by example: use the first image as the query
    print("\nquery-by-example (using image 0 as the probe):")
    for name, sim in v.most_similar(v.images[0], k=4):
        print(f"  {sim:5.2f}  {name}")
    print()

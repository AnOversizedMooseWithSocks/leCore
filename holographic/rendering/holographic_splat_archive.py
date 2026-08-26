"""A splat-bundle image archive -- store a gallery as Gaussian-splat codes BESIDE the WHT plates.

WHY THIS EXISTS
---------------
holographic_archive stores images as superposed Walsh-Hadamard plates: EXACT, content-addressable,
damage-tolerant. holographic_splat shows that a 2-D field is a SUPERPOSITION of Gaussian primitives --
a bundle. This is the matching archive MODE the splat work implied: each image is stored as its splat
code (cy, cx, amplitude, sigma) per channel, and reconstructed by rendering that superposition. Because
matching pursuit places splats in DECREASING residual-energy order, the stored list is already sorted by
importance, which buys three things the plates do not have:

  * PROGRESSIVE REFINEMENT for free -- rendering a PREFIX of k splats is a coarser-but-valid preview,
    and quality rises monotonically with k (3DGS's densify move, already baked into the stored order).
  * EXACT REGION QUERY -- the splats whose centre lies in a box ARE what is there; no decode needed.
  * a FIXED, TUNABLE byte budget -- K splats x 4 params per channel.

MEASURED (the real _gallery, at a matched LOW byte budget -- see test_splat_archive): the splat archive
matches or beats the WHT-plate archive's reconstruction quality in the low-byte regime (where the plates
can keep only a few DCT coefficients), and always adds region-query + progressive refinement.

KEPT NEGATIVE / SCOPE (the honest boundary, not hidden):
  * LOSSY. The WHT archive is EXACT on an undamaged read; this never is. At a HIGH byte budget the plates'
    exact coefficients win outright -- the splat archive's edge is the low-byte / region / refine regime.
  * ISOTROPIC splats only (the honest matching-pursuit baseline). Anisotropic covariances and gradient
    refinement -- full 3D Gaussian Splatting -- are out of scope here.
  * NO damage-tolerant joint recovery (the plates' real strength under erasure). Different tool, different
    tradeoff. This sits BESIDE the plates; it does not replace them.

Pure NumPy, deterministic (matching pursuit's order is fixed by the residual).
"""

import numpy as np

from holographic.rendering.holographic_splat import splat_fit, splat_render


class SplatArchive:
    """A gallery of images stored as Gaussian-splat codes -- a bundle of primitives per image."""

    def __init__(self, shape, keep=40, scales=(1.0, 2.0, 3.5, 6.0), thumb=12):
        self.S = shape[0]
        self.nchan = shape[2] if len(shape) == 3 else 1
        self.K = keep
        self.scales = scales
        self.thumb = thumb
        self.codes = []          # [image][channel] -> list of (cy, cx, amp, sigma), importance-ordered
        self.fingerprints = []   # normalised thumbnails, for content recall (same idea as the WHT archive)
        self.n = 0

    def _channels(self, img):
        return [img[..., c] for c in range(self.nchan)] if img.ndim == 3 else [img]

    def _fingerprint(self, img):
        from holographic.misc.holographic_archive import box_resize
        t = box_resize(img, self.thumb).ravel()
        return t / (np.linalg.norm(t) + 1e-9)

    def add(self, image):
        """Fit and store one image as K splats per channel (matching pursuit -> importance order)."""
        image = np.asarray(image, float)
        self.codes.append([splat_fit(ch, self.K, self.scales) for ch in self._channels(image)])
        self.fingerprints.append(self._fingerprint(image))
        self.n += 1
        return self

    def recover(self, i, k=None):
        """Reconstruct image i by rendering its splats (their superposition). k < K renders only the
        most important k -- a progressively-refined preview, since the stored list is in decreasing
        residual-energy order. k=None renders all K."""
        out = []
        for c in range(self.nchan):
            out.append(splat_render(self.codes[i][c][:(k or self.K)], (self.S, self.S)))
        img = np.stack(out, -1) if self.nchan > 1 else out[0]
        return np.clip(img, 0, 1)

    def recall(self, query):
        """Identify which stored image the query is (by thumbnail fingerprint) and return
        (index, reconstruction) -- content-addressable, like the WHT archive's recall."""
        fq = self._fingerprint(np.asarray(query, float))
        i = int(np.argmax([fq @ fp for fp in self.fingerprints]))
        return i, self.recover(i)

    def region(self, i, box):
        """EXACT region query: the splats of image i whose centre lies in box=(y0, y1, x0, x1), plus a
        render of just those primitives -- 'what is HERE'. This is the precise, per-splat complement to the
        holographic splat_bundle / recall_region (which is content-addressable and reliable but COARSE --
        a quantised per-region occupancy, not the individual splats)."""
        y0, y1, x0, x1 = box
        here = [[(cy, cx, a, s) for (cy, cx, a, s) in self.codes[i][c] if y0 <= cy < y1 and x0 <= cx < x1]
                for c in range(self.nchan)]
        out = [splat_render(here[c], (self.S, self.S)) for c in range(self.nchan)]
        patch = np.stack(out, -1) if self.nchan > 1 else out[0]
        return here, np.clip(patch, 0, 1)

    def stored_bytes(self, param_bytes=4):
        """Honest on-disk size: K splats x 4 params x nchan x N at param_bytes per value (float32 = 4),
        plus the thumbnail fingerprints (the only side data, exactly as the WHT archive counts its)."""
        splat = self.n * self.nchan * self.K * 4 * param_bytes
        fp = self.n * (self.thumb ** 2 * self.nchan * 8)
        return int(splat + fp)

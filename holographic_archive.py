"""
holographic_archive.py -- a content-addressable holographic image memory.

This ties the whole thread together. A gallery of images is superposed into a
few damage-tolerant plates (one per colour channel). Each image is given a
DISJOINT pool of Walsh-Hadamard key slots, so the combined keys across images
are exactly orthonormal -- which means any stored image can be pulled back out
by index with no crosstalk, up to a hard capacity of capacity*keep <= dim.

On top of that storage sits associative recall. A small normalised thumbnail of
each image is kept as a content fingerprint. Hand the archive a *degraded* query
-- noisy, blurred, partially occluded -- and it matches the query's fingerprint
to the nearest stored one and reconstructs the clean original. Because the
fingerprints live outside the plate, recognition keeps working even when much of
the plate is destroyed; the plate's redundancy is what carries the pixels back.

It's the capability holographic memory has always been pitched on -- recall the
whole from a corrupted part -- built from the pieces we validated along the way:
WHT structured keys, disjoint-slot multiplexing, and joint masked recovery.

Needs: numpy, and holographic_image.py (for the fast transform and CG solver).
"""

import numpy as np
from holographic_image import _fwht, _dct_matrix, _psnr, _cg, _lloyd_max
from holographic_ai import Vocabulary, bind, bundle, cosine
from holographic_encoders import ScalarEncoder


def box_resize(img, n):
    """Average-pool down to n x n (per channel for colour). Downsample only."""
    h, w = img.shape[:2]
    ys = np.linspace(0, h, n + 1).astype(int)
    xs = np.linspace(0, w, n + 1).astype(int)
    if img.ndim == 3:
        return np.stack([np.array([[img[ys[i]:ys[i+1], xs[j]:xs[j+1], c].mean()
                for j in range(n)] for i in range(n)]) for c in range(img.shape[2])], -1)
    return np.array([[img[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean()
                      for j in range(n)] for i in range(n)])


class HolographicArchive:
    """A damage-tolerant, content-addressable store for a gallery of images."""

    def __init__(self, shape, capacity, keep=2000, dim=32768, seed=0, thumb=12):
        self.S = shape[0]
        self.nchan = shape[2] if len(shape) == 3 else 1
        self.K = keep
        self.dim = dim
        self.thumb = thumb
        if capacity * keep > dim:
            raise ValueError(f"capacity*keep ({capacity*keep}) must be <= dim ({dim})")
        rng = np.random.default_rng(seed)
        self.signs = rng.choice([-1.0, 1.0], size=dim)
        self.perm = rng.permutation(dim)            # disjoint slot pools, sliced per image
        self._scale = 1.0 / np.sqrt(dim)
        self.M = _dct_matrix(self.S)
        self.plates = [np.zeros(dim) for _ in range(self.nchan)]
        self._idx = []                              # [image][channel] -> kept DCT indices
        self.fingerprints = []                      # normalised thumbnails
        self.n = 0
        self.bits = None                            # plate quantisation (None = float)
        # cross-modal addressing: a small VSA space so an image can be recalled by
        # word/number tags, not just by a degraded picture of itself.
        self.addr_dim = 4096
        self.vocab = Vocabulary(self.addr_dim, seed=seed + 1)
        self.scalar = ScalarEncoder(self.addr_dim, lo=0.0, hi=1.0, seed=seed + 2)
        self.addresses = []                         # one hypervector tag-address per image (or None)

    # --- the WHT key operator for image i (disjoint slots => orthonormal across images) ---
    def _pos(self, i):
        return self.perm[i * self.K:(i + 1) * self.K]

    def _apply(self, i, v):
        x = np.zeros(self.dim); x[self._pos(i)] = v
        return _fwht(x * self.signs) * self._scale

    def _adjoint(self, i, y):
        return (_fwht(y) * self._scale * self.signs)[self._pos(i)]

    def _fingerprint(self, img):
        t = box_resize(img, self.thumb).ravel()
        return t / (np.linalg.norm(t) + 1e-9)

    def _channels(self, img):
        return [img[..., c] for c in range(self.nchan)] if img.ndim == 3 else [img]

    def _address(self, words=None, nums=None):
        """Build a hypervector content-address from word tags and/or numeric
        attributes. Words become clean atoms; a numeric attribute k=v becomes
        bind(atom('attr:'+k), scalar.encode(v)). The bundle is the address."""
        parts = []
        for w in (words or []):
            parts.append(self.vocab.get(w))
        for k, v in (nums or {}).items():
            parts.append(bind(self.vocab.get("attr:" + k), self.scalar.encode(float(v))))
        return bundle(parts) if parts else None

    def add(self, image, tags=None, nums=None):
        """Superpose one image into the plates and remember its fingerprint.
        Optionally attach word tags / numeric attributes for cross-modal recall."""
        image = np.asarray(image, dtype=float)
        i = self.n
        per_chan_idx = []
        for c, ch in enumerate(self._channels(image)):
            flat = (self.M @ ch @ self.M.T).ravel()
            idx = np.argpartition(np.abs(flat), -self.K)[-self.K:]
            per_chan_idx.append(idx)
            self.plates[c] += self._apply(i, flat[idx])
        self._idx.append(per_chan_idx)
        self.fingerprints.append(self._fingerprint(image))
        self.addresses.append(self._address(tags, nums))
        self.n += 1
        return self

    def quantize(self, bits):
        """Compress every plate to `bits` per value with a Lloyd-Max codebook.
        Recovery still works (the quantisation just adds a noise floor); measured
        to keep content recall intact down to ~4 bits."""
        self.bits = bits
        self.plates = [_lloyd_max(p, bits)[1][_lloyd_max(p, bits)[0]] for p in self.plates]
        return self

    def stored_bytes(self):
        """Honest on-disk size: plates (at the current bit-depth) + per-image
        coefficient index maps + thumbnail fingerprints."""
        npix = self.S * self.S
        if self.bits is None:
            plate = self.nchan * self.dim * 8
        else:
            plate = self.nchan * (self.bits * self.dim / 8 + (2 ** self.bits) * 8)
        index = self.n * self.nchan * np.ceil(npix / 8)
        fp = self.n * (self.thumb ** 2 * self.nchan * 8)
        return int(plate + index + fp)

    def _joint_recover(self, c, mask):
        """Recover ALL images' coefficients for channel c at once under a damage
        mask (erasure breaks the per-image orthogonality, so we must solve jointly)."""
        N = self.n
        def app(V): return sum(self._apply(n, V[n*self.K:(n+1)*self.K]) for n in range(N))
        def adj(y): return np.concatenate([self._adjoint(n, y) for n in range(N)])
        Vf = _cg(lambda V: adj(mask * app(V)) + 1e-3 * V, adj(mask * self.plates[c]), 250)
        return [Vf[n*self.K:(n+1)*self.K] for n in range(N)]

    def recover(self, i, mask=None):
        """Reconstruct stored image i. Undamaged: exact, by a single adjoint per
        channel. Damaged (mask given): joint masked recovery, graceful until the
        survivors drop below the total coefficient count."""
        joints = [self._joint_recover(c, mask) for c in range(self.nchan)] if mask is not None else None
        out = []
        for c in range(self.nchan):
            v = self._adjoint(i, self.plates[c]) if mask is None else joints[c][i]
            flat = np.zeros(self.S * self.S); flat[self._idx[i][c]] = v
            out.append(self.M.T @ flat.reshape(self.S, self.S) @ self.M)
        img = np.stack(out, -1) if self.nchan > 1 else out[0]
        return np.clip(img, 0, 1)

    def recall(self, query, mask=None):
        """Identify which stored image the (possibly degraded) query is, and
        return (index, clean reconstruction)."""
        fq = self._fingerprint(np.asarray(query, dtype=float))
        i = int(np.argmax([fq @ fp for fp in self.fingerprints]))
        return i, self.recover(i, mask)

    def recall_by_tags(self, words=None, nums=None, mask=None):
        """Cross-modal recall: describe what you want with word/number tags and
        get back the best-matching stored image -- no picture needed. Returns
        (index, reconstruction, confidence)."""
        q = self._address(words, nums)
        if q is None:
            raise ValueError("give some words or numeric attributes to match")
        sims = [cosine(q, a) if a is not None else -1.0 for a in self.addresses]
        i = int(np.argmax(sims))
        return i, self.recover(i, mask), float(sims[i])

    def damage_mask(self, destroy_fraction, seed=0):
        rng = np.random.default_rng(seed)
        keep = np.ones(self.dim)
        keep[rng.permutation(self.dim)[:int(self.dim * destroy_fraction)]] = 0
        return keep


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
def _gallery(S=128):
    yy, xx = np.mgrid[0:S, 0:S] / S
    imgs = []
    g = np.zeros((S, S, 3)); h = S // 2
    g[:h, :h] = [1, 0, 0]; g[:h, h:] = [0, 0, 1]; g[h:, :h] = [1, 1, 0]; g[h:, h:] = [0, 1, 0]; imgs.append(g)
    g = np.zeros((S, S, 3)); g[:S//3] = [1, .1, .1]; g[S//3:2*S//3] = [.1, 1, .1]; g[2*S//3:] = [.1, .1, 1]; imgs.append(g)
    imgs.append(np.stack([xx, 1 - xx, np.abs(yy - .5) * 2], -1))
    r = np.sqrt((xx - .5)**2 + (yy - .5)**2); imgs.append(np.clip(np.stack([1 - r, r, np.abs(.5 - r) * 2], -1), 0, 1))
    g = np.zeros((S, S, 3)); g[..., 0] = np.sin(xx*12)*.5+.5; g[..., 1] = np.cos(yy*12)*.5+.5; g[..., 2] = .5; imgs.append(g)
    g = (((np.floor(xx*4)+np.floor(yy*4)) % 2)[..., None]) * np.array([1, .4, .7]); imgs.append(g)
    return [np.clip(im, 0, 1) for im in imgs]


def demo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = 128
    imgs = _gallery(S)
    arc = HolographicArchive((S, S, 3), capacity=len(imgs), keep=2000, dim=32768, seed=0)
    for im in imgs:
        arc.add(im)

    rng = np.random.default_rng(1)
    noisy = lambda im: np.clip(im + 0.5 * rng.standard_normal(im.shape), 0, 1)
    blur = lambda im: np.repeat(np.repeat(box_resize(im, 16), S // 16, 0), S // 16, 1)
    def occl(im):
        g = im.copy(); g[20:80, 20:80] = 0; return g

    mask = arc.damage_mask(0.4, seed=7)
    rows = [0, 1, 3, 5]
    fig, ax = plt.subplots(len(rows), 6, figsize=(15, 2.5 * len(rows)))
    cols = ["stored", "noisy query", "blurred query", "occluded query", "recalled", "recalled\n(plate 40% gone)"]
    for r, i in enumerate(rows):
        q_noise, q_blur, q_occ = noisy(imgs[i]), blur(imgs[i]), occl(imgs[i])
        j, clean = arc.recall(q_noise)
        jd, dmg = arc.recall(q_noise, mask=mask)
        panels = [imgs[i], q_noise, q_blur, q_occ, clean, dmg]
        for cidx, (img, title) in enumerate(zip(panels, cols)):
            ax[r, cidx].imshow(img); ax[r, cidx].axis("off")
            if r == 0:
                ax[r, cidx].set_title(title, fontsize=10)
        ax[r, 4].set_title(("recalled" if r else "recalled") +
                            (" OK" if j == i else " MISS"), fontsize=9)
    fig.suptitle("Content-addressable holographic memory: any degraded query recalls the clean original, even with 40% of the plate destroyed", y=1.0, fontsize=11)
    fig.tight_layout(); fig.savefig("archive.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    for name, deg in [("noise", noisy), ("blur", blur), ("occlusion", occl)]:
        hits = sum(arc.recall(deg(imgs[i]))[0] == i for i in range(arc.n))
        print(f"recall from {name:9s}: {hits}/{arc.n}")
    dmg_hits = sum(arc.recall(noisy(imgs[i]), mask=mask)[0] == i for i in range(arc.n))
    dmg_psnr = np.mean([_psnr(imgs[i], arc.recall(noisy(imgs[i]), mask=mask)[1]) for i in range(arc.n)])
    print(f"recall w/ 40% plate destroyed: {dmg_hits}/{arc.n}, recon {dmg_psnr:.1f} dB")
    print("wrote archive.png")


if __name__ == "__main__":
    demo()

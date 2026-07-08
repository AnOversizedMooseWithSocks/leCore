"""Regression tests for the hardened holographic image storage."""
import numpy as np
import pytest
from holographic.io_and_interop.holographic_image import Hologram, HolographicImage, _demo_image, _psnr


def box_gray(n):
    g = _demo_image(160).mean(2)
    h, w = g.shape
    ys = np.linspace(0, h, n + 1).astype(int)
    xs = np.linspace(0, w, n + 1).astype(int)
    return np.array([[g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
                      for j in range(n)] for i in range(n)])


class TestHologram:
    def test_cg_near_exact_below_capacity(self):
        n = 40; g = box_gray(n); npix = n * n
        h = Hologram(npix, npix * 2, 0).store(g.ravel())     # load 0.5
        rec = h.recall(method="iterative", lam=0, iters=400).reshape(n, n)
        assert _psnr(g, np.clip(rec, 0, 1)) > 45             # essentially exact

    def test_cg_beats_matched_by_a_mile(self):
        n = 40; g = box_gray(n); npix = n * n
        h = Hologram(npix, npix * 3, 0).store(g.ravel())
        cg = _psnr(g, np.clip(h.recall(method="iterative", lam=0, iters=400).reshape(n, n), 0, 1))
        mf = _psnr(g, np.clip(h.recall(method="matched").reshape(n, n), 0, 1))
        assert cg > mf + 20

    def test_mask_aware_tolerates_half_erased(self):
        n = 32; g = box_gray(n); npix = n * n
        h = Hologram(npix, npix * 4, 0).store(g.ravel())     # cliff ~0.75
        rec = h.recall(h.damage_mask(0.5, 3), method="iterative").reshape(n, n)
        assert _psnr(g, np.clip(rec, 0, 1)) > 30

    def test_more_regularisation_helps_under_heavy_noise(self):
        n = 32; g = box_gray(n); npix = n * n; D = npix * 4
        h = Hologram(npix, D, 0).store(g.ravel())
        noise = np.random.default_rng(1).standard_normal(D) * 0.4
        hh = Hologram(npix, D, 0); hh.P = h.P; hh.plate = h.plate + noise
        low = _psnr(g, np.clip(hh.recall(method="iterative", lam=0.002, iters=400).reshape(n, n), 0, 1))
        high = _psnr(g, np.clip(hh.recall(method="iterative", lam=0.5, iters=400).reshape(n, n), 0, 1))
        assert high > low


class TestHolographicImage:
    def test_colour_image_stored_with_good_fidelity(self):
        img = _demo_image(96)
        hi = HolographicImage(img.shape, keep=2000, dim=8192, seed=0).store(img)
        assert _psnr(img, hi.reconstruct()) > 25

    def test_colour_image_degrades_gracefully(self):
        img = _demo_image(96)
        hi = HolographicImage(img.shape, keep=2000, dim=16384, seed=0).store(img)  # cliff ~0.88
        intact = _psnr(img, hi.reconstruct())
        half = _psnr(img, hi.reconstruct(hi.damage_mask(0.5, 1)))
        assert intact > 25
        assert half > intact - 3        # destroying half the plate costs almost nothing


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


class TestWalshHadamard:
    def test_fwht_is_orthonormal_involution(self):
        from holographic.io_and_interop.holographic_image import _fwht
        x = np.random.default_rng(0).standard_normal(256)
        # (1/N) * WHT(WHT(x)) == x  (Hadamard squared = N*I)
        assert np.allclose(_fwht(_fwht(x)) / 256, x, atol=1e-9)

    def test_wht_keys_exact_isometry(self):
        from holographic.io_and_interop.holographic_image import WHTKeys
        keys = WHTKeys(3000, 8192, seed=0)
        v = np.random.default_rng(1).standard_normal(3000)
        assert np.allclose(keys.adjoint(keys.apply(v)), v, atol=1e-9)  # A^T A = I

    def test_wht_backend_is_tiny_in_memory(self):
        img = _demo_image(96)
        hi = HolographicImage(img.shape, keep=2000, dim=8192, backend="wht").store(img)
        dense_equiv = hi.K * hi.dim * 8
        assert hi.key_bytes() < dense_equiv / 100        # >100x smaller key storage

    def test_wht_full_resolution_capability(self):
        img = _demo_image(240)
        hi = HolographicImage(img.shape, keep=6000, dim=16384, backend="wht").store(img)
        assert _psnr(img, hi.reconstruct()) > 28          # large colour image, good fidelity

    def test_wht_matches_dense_fidelity(self):
        img = _demo_image(96)
        d = HolographicImage(img.shape, keep=2000, dim=8192, backend="dense").store(img)
        w = HolographicImage(img.shape, keep=2000, dim=8192, backend="wht").store(img)
        pd = _psnr(img, d.reconstruct())
        pw = _psnr(img, w.reconstruct())
        assert pw > pd - 1.0                              # structured is at least as faithful


class TestQuantizedPlate:
    def test_lloyd_quantized_plate_near_float(self):
        img = _demo_image(120)
        f = HolographicImage(img.shape, keep=2500, dim=8192, seed=0).store(img)
        q = HolographicImage(img.shape, keep=2500, dim=8192, seed=0).store(img, bits=4)
        assert _psnr(img, q.reconstruct()) > _psnr(img, f.reconstruct()) - 3.0

    def test_quantized_plate_is_smaller(self):
        img = _demo_image(120)
        f = HolographicImage(img.shape, keep=2500, dim=8192, seed=0).store(img)
        q = HolographicImage(img.shape, keep=2500, dim=8192, seed=0).store(img, bits=3)
        assert q.stored_bytes() < f.stored_bytes() / 2

    def test_quantized_damage_tolerance_survives(self):
        img = _demo_image(120)
        q = HolographicImage(img.shape, keep=2500, dim=16384, seed=0).store(img, bits=4)
        intact = _psnr(img, q.reconstruct())
        half = _psnr(img, q.reconstruct(q.damage_mask(0.4, 1)))
        assert intact > 25 and half > intact - 3        # graceful even when quantized

    def test_scratch_as_good_as_random(self):
        import numpy as np
        img = _demo_image(120)
        q = HolographicImage(img.shape, keep=2500, dim=16384, seed=0).store(img, bits=4)
        scratch = np.ones(q.dim); scratch[:int(q.dim * 0.4)] = 0   # contiguous gouge
        p_scratch = _psnr(img, q.reconstruct(scratch))
        p_random = _psnr(img, q.reconstruct(q.damage_mask(0.4, 1)))
        assert abs(p_scratch - p_random) < 2.0           # no spatial weak spot


class TestSharedIndexAndAccounting:
    def test_shared_index_smaller_and_valid(self):
        img = _demo_image(120)
        per = HolographicImage(img.shape, keep=2500, dim=8192, seed=0).store(img, bits=4)
        sh = HolographicImage(img.shape, keep=2500, dim=8192, seed=0).store(img, bits=4, shared_index=True)
        assert sh.stored_bytes() < per.stored_bytes()        # one index map, not three
        assert _psnr(img, sh.reconstruct()) > _psnr(img, per.reconstruct()) - 2.5

    def test_stored_bytes_counts_index_map(self):
        import numpy as np
        img = _demo_image(120); npix = 120 * 120
        q = HolographicImage(img.shape, keep=2500, dim=8192, seed=0).store(img, bits=4)
        # total must exceed keys + plate alone by at least the (3) index bitmaps
        plate = 3 * (4 * q.dim / 8 + 16 * 8)
        assert q.stored_bytes() >= q.key_bytes() + plate + 3 * np.ceil(npix / 8)

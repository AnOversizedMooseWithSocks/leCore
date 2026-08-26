"""Regression tests for the content-addressable holographic archive."""
import numpy as np
import pytest
from holographic.misc.holographic_archive import HolographicArchive, _gallery
from holographic.io_and_interop.holographic_image import _psnr


def build():
    imgs = _gallery(128)
    tags = [["quadrants", "red", "blue"], ["bands", "horizontal", "green"],
            ["gradient", "diagonal"], ["radial", "rings", "pink"],
            ["ripples", "waves", "sine"], ["checker", "squares", "pink"]]
    arc = HolographicArchive((128, 128, 3), capacity=len(imgs), keep=2000, dim=32768, seed=0)
    for im, t in zip(imgs, tags):
        arc.add(im, tags=t)
    return imgs, arc


class TestArchive:
    def test_exact_recovery_no_crosstalk(self):
        imgs, arc = build()                      # disjoint slots -> orthonormal -> no crosstalk
        assert all(_psnr(imgs[i], arc.recover(i)) > 40 for i in range(arc.n))

    def test_recall_from_heavy_noise(self):
        imgs, arc = build(); rng = np.random.default_rng(1)
        hits = sum(arc.recall(np.clip(imgs[i] + 0.5 * rng.standard_normal(imgs[i].shape), 0, 1))[0] == i
                   for i in range(arc.n))
        assert hits == arc.n

    def test_recall_from_occlusion(self):
        imgs, arc = build()
        occ = lambda im: (lambda g: (g.__setitem__((slice(20, 80), slice(20, 80)), 0), g)[1])(im.copy())
        assert sum(arc.recall(occ(imgs[i]))[0] == i for i in range(arc.n)) == arc.n

    def test_recall_survives_40pct_plate_damage(self):
        imgs, arc = build(); m = arc.damage_mask(0.4, 7); rng = np.random.default_rng(2)
        i = 2; q = np.clip(imgs[i] + 0.4 * rng.standard_normal(imgs[i].shape), 0, 1)
        j, rec = arc.recall(q, mask=m)
        assert j == i and _psnr(imgs[i], rec) > 30

    def test_capacity_guard(self):
        with pytest.raises(ValueError):
            HolographicArchive((128, 128, 3), capacity=100, keep=2000, dim=32768)

    def test_cross_modal_recall_by_tags(self):
        # describe an image in words; the right one comes back from the address space
        imgs, arc = build()
        cases = [(["quadrants"], 0), (["radial", "pink"], 3), (["checker"], 5),
                 (["horizontal"], 1), (["ripples"], 4), (["diagonal"], 2)]
        assert all(arc.recall_by_tags(words=w)[0] == truth for w, truth in cases)

    def test_quantized_plates_keep_recall(self):
        # 4-bit Lloyd-Max plates: store shrinks ~8x, content recall still perfect
        imgs, arc = build()
        big = arc.stored_bytes()
        arc.quantize(4)
        assert arc.stored_bytes() < big / 4
        assert sum(arc.recall(imgs[i])[0] == i for i in range(arc.n)) == arc.n
        assert all(_psnr(imgs[i], arc.recover(i)) > 25 for i in range(arc.n))

    def test_cross_modal_survives_quantization(self):
        imgs, arc = build(); arc.quantize(4)     # addresses live outside the plates
        assert arc.recall_by_tags(words=["radial", "pink"])[0] == 3


if __name__ == "__main__":
    import sys; sys.exit(pytest.main([__file__, "-v"]))


def test_verify_confirms_exact_recall_on_the_hardest_images():
    # verify() checks the disjoint-slot exact-recall property on THIS build: reconstruct the
    # most collision-prone stored images and confirm each recalls back to its own index.
    from holographic.misc.holographic_archive import HolographicArchive, _gallery
    imgs = _gallery(64)
    a = HolographicArchive(shape=(64, 64, 3), capacity=len(imgs), seed=0)
    for im in imgs:
        a.add(im)
    checked, exact = a.verify()
    assert checked == len(imgs) and exact == checked          # every hard case recalls as itself
    # the identity survives the lossy plate quantisation too (its whole point)
    a.quantize(4)
    c2, e2 = a.verify()
    assert e2 == c2


# ---- vectorized recall (the above/below sweep: the cleanup matvec pattern applied here) --------------

def test_archive_recall_is_vectorized_and_matches_loop():
    """recall and recall_by_tags scan stored images with ONE matrix-vector product (cached fingerprint /
    address matrices) instead of a per-image Python loop -- the same move the core Vocabulary.cleanup uses.
    The result is the SAME index the loop returned, and the matrices rebuild after a new image is added."""
    from holographic.agents_and_reasoning.holographic_machine import cosine
    imgs, arc = build()
    for k, im in enumerate(imgs):                       # image recall == the old fingerprint loop
        i_vec, _ = arc.recall(im)
        fq = arc._fingerprint(np.asarray(im, float))
        assert i_vec == int(np.argmax([fq @ fp for fp in arc.fingerprints])) == k
    q = arc._address(["red"], None)                     # tag recall == the old cosine loop
    i_vec, _, _ = arc.recall_by_tags(["red"])
    assert i_vec == int(np.argmax([cosine(q, a) if a is not None else -1.0 for a in arc.addresses]))
    n0 = arc.n                                          # cache invalidation: a newly added image is recallable
    new = np.random.default_rng(7).random(imgs[0].shape)
    arc.add(new, tags=["fresh"])
    assert arc.recall(new)[0] == n0


def test_scalar_decode_cached_matrix_matches_loop():
    """ScalarEncoder.decode caches the grid encodings as a unit-normalized matrix and reads back a number
    with one matvec (~200x faster than re-encoding the grid each call), giving the SAME value a per-grid
    cosine scan would. Repeated decodes reuse the cache; a different `steps` builds its own."""
    from holographic.io_and_interop.holographic_encoders import ScalarEncoder
    from holographic.agents_and_reasoning.holographic_machine import cosine
    e = ScalarEncoder(dim=1024, lo=0.0, hi=1.0, seed=0)
    for x in (0.0, 0.23, 0.5, 0.91, 1.0):
        v = e.encode(x)
        grid = np.linspace(e.lo, e.hi, 200)
        loop = float(grid[int(np.argmax([cosine(v, e.encode(g)) for g in grid]))])
        assert e.decode(v, steps=200) == loop           # bit-identical argmax
    assert 200 in e._grid_cache                          # cached
    e.decode(v, steps=64)
    assert 64 in e._grid_cache and 200 in e._grid_cache  # per-steps caches coexist

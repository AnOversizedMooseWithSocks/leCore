"""Regression tests for the content-addressable holographic archive."""
import numpy as np
import pytest
from holographic_archive import HolographicArchive, _gallery
from holographic_image import _psnr


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

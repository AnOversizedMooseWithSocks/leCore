"""Fractal structure: box-counting recovers known dimensions, distinguishes
natural from synthetic images, reads market self-affinity, and IFS compresses
self-similar data only (the kept negative)."""
import glob
import os

import numpy as np

from holographic_fractal import (box_counting_dimension, image_fractal_dimension,
                                  hurst_exponent, IFS, ifs_compresses)


def test_box_counting_recovers_known_dimensions():
    # The instrument must be correct before it is trusted: a line is ~1-D, a
    # filled square ~2-D, a Sierpinski gasket ~1.585.
    rng = np.random.default_rng(0)
    line = np.c_[np.linspace(0, 1, 20000), np.linspace(0, 1, 20000)]
    square = rng.random((20000, 2))
    verts = np.array([[0, 0], [1, 0], [0.5, 0.866]])
    p = np.array([0.5, 0.3])
    sier = []
    for _ in range(20000):
        p = (p + verts[rng.integers(3)]) / 2
        sier.append(p.copy())
    assert abs(box_counting_dimension(line) - 1.0) < 0.2
    assert abs(box_counting_dimension(square) - 2.0) < 0.2
    assert abs(box_counting_dimension(np.array(sier)) - 1.585) < 0.15


def test_natural_images_are_rougher_than_synthetic():
    # Fractal dimension of the edge map: natural photos run high (scale-invariant
    # natural-scene statistics); a smooth synthetic circle runs near 1.0.
    if not glob.glob("features/photo_sample/*.npy"):
        import pytest
        pytest.skip("photo samples not present")
    photos = [np.load(f) for f in sorted(glob.glob("features/photo_sample/*.npy"))]
    nat = np.mean([image_fractal_dimension(p) for p in photos])
    S = 96
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    circ = (((xx - 48) ** 2 + (yy - 48) ** 2) < 30 ** 2).astype(float) * 255
    assert nat > image_fractal_dimension(circ) + 0.2     # natural clearly rougher
    assert nat > 1.3


def test_hurst_separates_random_walk_from_mean_reversion():
    # A random walk reads H~0.5; a mean-reverting (anti-correlated) series reads
    # below 0.5. The fractal lens on a 1-D series.
    rng = np.random.default_rng(0)
    walk = rng.standard_normal(2000)
    assert abs(hurst_exponent(walk) - 0.5) < 0.15
    # anti-correlated series: flip sign each step around a mean
    mr = np.array([(-1) ** i + 0.3 * rng.standard_normal() for i in range(2000)])
    assert hurst_exponent(mr) < 0.5


def test_ifs_compresses_self_similar_data_only():
    # THE COMPRESSION PAYOFF AND ITS HONEST LIMIT: a fern fits its own IFS
    # (coverage error ~0, far below random) at a huge compression ratio; random
    # points fit that IFS no better than random -- no compression.
    fern = IFS.barnsley_fern()
    pts = fern.generate(8000)
    fit = ifs_compresses(pts, fern)
    assert fit["ifs_error"] < 0.1                        # fern matches its own maps
    assert fit["ifs_error"] < fit["random_error"] - 0.3  # far better than random
    assert fit["compression"] > 100                      # 28 numbers vs thousands of points
    rng = np.random.default_rng(0)
    rand = rng.random((8000, 2))
    rfit = ifs_compresses(rand, fern)
    assert rfit["ifs_error"] > 0.3                        # random does NOT compress to the fern


def test_brain_reads_fractal_dimension():
    # Wired to the brain: it reads an image's roughness and a series' affinity
    # directly as perceptual quantities.
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=512, seed=0)
    S = 96
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    circ = np.stack([(((xx - 48) ** 2 + (yy - 48) ** 2) < 30 ** 2).astype(np.uint8) * 200] * 3, -1)
    assert m.fractal_dimension(circ, modality="image") < 1.3      # smooth
    rng = np.random.default_rng(0)
    assert abs(m.self_affinity(rng.standard_normal(1500)) - 0.5) < 0.15

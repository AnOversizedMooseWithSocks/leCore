"""Tests for holographic_vision: colour, gradients, shapes, emergence, VSA.

Every assertion here corresponds to a claim made in the module's docstrings,
so the honest numbers in the demo stay honest as the code changes.
"""
import colorsys
import numpy as np
import holographic.misc.holographic_vision as hv


# ---- colour -------------------------------------------------------------
def test_rgb_to_hsv_matches_stdlib():
    rng = np.random.default_rng(0)
    px = rng.random((500, 3))
    ours = hv.rgb_to_hsv(px.reshape(1, -1, 3))[0]
    ref = np.array([colorsys.rgb_to_hsv(*p) for p in px])
    err = np.abs(np.stack([ours[:, 0] / 360.0, ours[:, 1], ours[:, 2]], 1) - ref)
    assert err.max() < 1e-6


def test_hsv_roundtrip():
    rng = np.random.default_rng(1)
    rgb = rng.random((8, 8, 3))
    back = hv.hsv_to_rgb(hv.rgb_to_hsv(rgb))
    assert np.abs(rgb - back).max() < 1e-6


def test_grey_has_no_hue_and_zero_saturation():
    grey = np.full((4, 4, 3), 0.5)
    hsv = hv.rgb_to_hsv(grey)
    assert np.allclose(hsv[..., 1], 0.0)


def test_hue_histogram_picks_dominant_colour():
    img = np.zeros((20, 20, 3), np.uint8); img[..., 0] = 255   # pure red
    h = hv.hue_histogram(img, bins=12)
    assert h.argmax() == 0 and abs(h.sum() - 1.0) < 1e-9


def test_dominant_colours_finds_two_blocks():
    img = np.zeros((20, 20, 3), np.uint8)
    img[:, :10] = [200, 30, 30]; img[:, 10:] = [30, 30, 200]
    centres, w = hv.dominant_colours(img, k=2, seed=0)
    assert centres.shape == (2, 3) and abs(w.sum() - 1.0) < 1e-6


def test_segment_image_max_dim_bounds_resolution_through_the_mind():
    """ITEM 5: max_dim bounds the segmentation sweep. Default (None) is byte-identical to the historic call; with
    max_dim set, masks come back FULL-SIZE (nearest-upsampled) with stats recomputed on the original image."""
    from holographic.misc.holographic_unified import UnifiedMind
    um = UnifiedMind(dim=64, seed=0)
    H = W = 128
    img = np.zeros((H, W, 3)); img[:, :, 2] = 1.0
    yy, xx = np.mgrid[0:H, 0:W]
    img[(yy - 50) ** 2 + (xx - 44) ** 2 <= 24 ** 2] = (1.0, 0.0, 0.0)
    base = um.segment_image(img, k=2, seed=0)
    same = um.segment_image(img, k=2, seed=0, max_dim=None)
    assert len(base) == len(same)
    for a, b in zip(base, same):
        assert np.array_equal(a["mask"], b["mask"]) and a["bbox"] == b["bbox"]      # None == historic
    bounded = um.segment_image(img, k=2, seed=0, max_dim=48)
    assert all(r["mask"].shape == (H, W) for r in bounded)                          # full-size masks
    cols = [r["mean_color"] for r in bounded]
    assert any(c[0] > c[2] for c in cols) and any(c[2] > c[0] for c in cols)        # red + blue recovered


# ---- gradients / edges --------------------------------------------------
def test_sobel_detects_vertical_edge():
    g = np.zeros((16, 16)); g[:, 8:] = 1.0          # vertical step
    gx, gy = hv.sobel(g)
    assert np.abs(gx).max() > np.abs(gy).max() * 5   # edge is horizontal-gradient


def test_edges_are_boolean_and_sparse():
    img, _ = hv.make_shape("rectangle", 48, seed=2)
    e = hv.edges(hv.to_gray(img))
    assert e.dtype == bool and 0 < e.mean() < 0.5


# ---- shapes -------------------------------------------------------------
def test_hough_finds_a_line():
    img, _ = hv.make_shape("line", 80, seed=3)
    lines = hv.hough_lines(hv.edges(hv.to_gray(img)), top=3)
    assert len(lines) >= 1 and lines[0][2] >= 20    # a real line gets many votes


def test_hough_finds_a_circle_near_centre():
    img, mask = hv.make_shape("circle", 80, seed=4)
    ys, xs = np.nonzero(mask); tcx, tcy = xs.mean(), ys.mean()
    c = hv.hough_circles(hv.to_gray(img), radii=range(15, 35, 2), top=1)
    assert c and abs(c[0][0] - tcx) < 8 and abs(c[0][1] - tcy) < 8


def test_shape_stats_separate_circle_from_rectangle():
    cimg, cmask = hv.make_shape("circle", 64, seed=5)
    rimg, rmask = hv.make_shape("rectangle", 64, seed=5)
    assert hv.shape_stats(cmask)["circularity"] > hv.shape_stats(rmask)["circularity"]
    assert hv.shape_stats(rmask)["extent"] > hv.shape_stats(cmask)["extent"]


def test_classify_shape_clean_shapes():
    kinds = ["circle", "rectangle", "triangle", "line"]
    ok = total = 0
    for s in range(15):
        for ki, k in enumerate(kinds):
            _, mask = hv.make_shape(k, 64, seed=100 * s + ki)
            ok += hv.classify_shape(mask) == k; total += 1
    assert ok / total >= 0.95


# ---- patterns / emergence ----------------------------------------------
def test_describe_shape_and_norm():
    img, _ = hv.make_shape("circle", 48, seed=6)
    d = hv.describe(img)
    assert d.shape == (24,) and abs(np.linalg.norm(d) - 1.0) < 1e-9


def test_kmeans_recovers_two_blobs():
    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal(0, 0.1, (30, 2)), rng.normal(5, 0.1, (30, 2))])
    labels, _ = hv.kmeans(X, 2, seed=0)
    assert hv.cluster_purity(labels, np.r_[np.zeros(30), np.ones(30)]) > 0.98


def test_emergent_classes_recover_shape_kinds():
    # Same-coloured shapes: structure is the only signal, so standardize=True
    # (drops the constant colour dims). Unsupervised, no labels used.
    kinds = ["circle", "rectangle", "triangle", "line"]
    imgs, truth = [], []
    for s in range(20):
        for ki, k in enumerate(kinds):
            img, _ = hv.make_shape(k, 64, seed=100 * s + ki)
            imgs.append(img); truth.append(ki)
    labels, _ = hv.emergent_classes(imgs, k=4, seed=0, standardize=True)
    # ~70% is the honest ceiling: randomly rotated lines scatter across
    # orientation bins, so the four kinds don't separate cleanly.
    assert hv.cluster_purity(labels, truth) >= 0.6


# ---- holographic classification ----------------------------------------
def test_vsa_prototype_classification():
    kinds = ["circle", "rectangle", "triangle", "line"]
    X, truth = [], []
    for s in range(20):
        for ki, k in enumerate(kinds):
            img, _ = hv.make_shape(k, 64, seed=100 * s + ki)
            X.append(hv.describe(img)); truth.append(ki)
    X, truth = np.stack(X), np.array(truth)
    rng = np.random.default_rng(0); tr = rng.random(len(X)) < 0.5
    protos, enc = hv.vsa_prototypes(X[tr], truth[tr], dim=2048, seed=1)
    acc = np.mean([hv.vsa_classify(X[i], protos, enc)[0] == truth[i]
                   for i in np.nonzero(~tr)[0]])
    assert acc >= 0.7


def test_vsa_encoding_preserves_similarity():
    # two similar descriptors -> similar hypervectors; a different one -> less
    from holographic.agents_and_reasoning.holographic_ai import cosine
    a, _ = hv.make_shape("circle", 64, seed=1)
    b, _ = hv.make_shape("circle", 64, seed=2)
    c, _ = hv.make_shape("line", 64, seed=3)
    H = hv.vsa_encode(np.stack([hv.describe(a), hv.describe(b), hv.describe(c)]), dim=2048, seed=0)
    assert cosine(H[0], H[1]) > cosine(H[0], H[2])


def test_tighten_selection_fixes_the_rotate_pivot():
    """The reported bug: a small drawing inside a big transparent rectangle selection rotated about the RECTANGLE's
    centre (empty space) instead of the drawing's centre. tighten_selection shrinks the marquee to the opaque
    pixels so the pivot lands on the content. Through the mind (the /invoke path the canvas uses)."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)

    # a 100x100 transparent layer, a 10x10 opaque drawing off-centre
    alpha = np.zeros((100, 100), float)
    alpha[20:30, 60:70] = 1.0

    # the user drags a big marquee over the whole thing; without tightening, its centre is (49.5, 49.5) -- empty
    r = m.tighten_selection(alpha, bbox=(0, 0, 99, 99))
    assert not r["empty"]
    assert r["bbox"] == (20, 60, 29, 69)                 # tightened exactly to the drawing
    assert r["centre"] == (24.5, 64.5)                   # pivot is the DRAWING centre, not the marquee's (49.5,49.5)
    assert r["area"] == 100

    # bbox arrives as a JSON list over /invoke, not a tuple -- must still work
    r_list = m.tighten_selection(alpha, bbox=[0, 0, 99, 99])
    assert r_list["bbox"] == (20, 60, 29, 69)

    # a marquee over blank space is a REAL answer: signal empty so the caller keeps the original selection
    blank = m.tighten_selection(alpha, bbox=(0, 0, 10, 10))
    assert blank["empty"] and blank["bbox"] is None

    # RGBA input uses the alpha channel; a threshold drops near-transparent anti-aliased fringe
    rgba = np.zeros((100, 100, 4), np.uint8)
    rgba[20:30, 60:70, 3] = 255
    rgba[50, 50, 3] = 40                                  # a faint stray pixel outside the drawing
    assert m.tighten_selection(rgba, threshold=0.5)["bbox"] == (20, 60, 29, 69)   # fringe excluded

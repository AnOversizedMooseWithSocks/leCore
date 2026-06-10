"""Tests for holographic_scene: DCT/colour/shape tagging, compositional encoding,
and resonator factoring (single object and multi-object scenes)."""
import numpy as np
import holographic_scene as sc
import holographic_vision as hv

_key = lambda d: (str(d["colour"]), str(d["shape"]), str(d["texture"]))


# ---- automatic tagging --------------------------------------------------
def test_colour_tag_primaries_and_grey():
    for name, rgb in [("red", (255, 0, 0)), ("green", (0, 255, 0)), ("blue", (0, 0, 255))]:
        img = np.zeros((16, 16, 3), np.uint8); img[:] = rgb
        assert sc.colour_tag(img) == name
    assert sc.colour_tag(np.full((16, 16, 3), 120, np.uint8)) == "grey"


def test_texture_tag_from_dct_canonical():
    N = 48; yy, xx = np.mgrid[0:N, 0:N] / N
    cases = {"smooth": xx,
             "horizontal": 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 5 * yy)),
             "vertical": 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 5 * xx)),
             "busy": 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 6 * xx) * np.sin(2 * np.pi * 6 * yy))}
    assert all(sc.texture_tag(g) == k for k, g in cases.items())


def test_dct_features_smooth_vs_busy():
    N = 48; yy, xx = np.mgrid[0:N, 0:N] / N
    smooth = sc.dct_features(xx)
    busy = sc.dct_features(0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 6 * xx) * np.sin(2 * np.pi * 6 * yy)))
    assert smooth["low"] > 0.6 and busy["off"] > max(busy["col"], busy["row"])


def test_auto_tags_red_circle():
    img, _ = hv.make_shape("circle", 64, seed=1, fg=(235, 70, 70))
    t = sc.auto_tags(img)
    assert set(t) == {"colour", "shape", "texture"}
    assert t["colour"] == "red" and t["shape"] == "circle"


# ---- segmentation -------------------------------------------------------
def test_segment_finds_two_objects():
    img = sc.make_scene([("circle", "red"), ("rectangle", "blue")], S=96, seed=1)
    assert len(sc.segment(img)) == 2


# ---- compositional encode + resonator ----------------------------------
def test_resonator_single_object_roundtrip():
    coder = sc.SceneCoder(dim=1024, seed=0); rng = np.random.default_rng(0); ok = 0
    for _ in range(60):
        t = {"colour": str(rng.choice(sc.COLOURS)), "shape": str(rng.choice(sc.SHAPES)),
             "texture": str(rng.choice(sc.TEXTURES))}
        ok += sc.SceneCoder.factor(coder, coder.encode(t)) == t
    assert ok >= 59                                     # ~100% within capacity


def test_two_object_scene_decomposition():
    coder = sc.SceneCoder(dim=1024, seed=0); rng = np.random.default_rng(1); ok = 0; T = 60
    for _ in range(T):
        ts = [{"colour": str(rng.choice(sc.COLOURS)), "shape": str(rng.choice(sc.SHAPES)),
               "texture": str(rng.choice(sc.TEXTURES))} for _ in range(2)]
        got = coder.factor_scene(coder.encode_scene(ts), 2)
        ok += {_key(ts[0]), _key(ts[1])} == {_key(got[0]), _key(got[1])}
    assert ok / T >= 0.9                                # 2 objects: reliable


def test_four_object_scene_decomposition():
    # the old ceiling was ~50% at THREE objects; sum-scene + coordinate-descent
    # sweeps now recover four reliably.
    coder = sc.SceneCoder(dim=1024, seed=0); rng = np.random.default_rng(2); ok = 0; T = 20
    for _ in range(T):
        ts = [{"colour": str(rng.choice(sc.COLOURS)), "shape": str(rng.choice(sc.SHAPES)),
               "texture": str(rng.choice(sc.TEXTURES))} for _ in range(4)]
        got = coder.factor_scene(coder.encode_scene(ts), 4, sweeps=2)
        ok += {_key(t) for t in ts} == {_key(g) for g in got}
    assert ok / T >= 0.9


def test_compositional_separates_what_holistic_cannot():
    # the headline: a 2-object image where one global tag loses an object but
    # segmentation + the resonator keep both.
    img = sc.make_scene([("circle", "red"), ("rectangle", "blue")], S=96, seed=2)
    objs = sc.tag_objects(img)
    assert {(o["colour"], o["shape"]) for o in objs} == {("red", "circle"), ("blue", "rectangle")}
    coder = sc.SceneCoder(dim=2048, seed=0)
    rec = coder.factor_scene(coder.encode_scene(objs), 2)
    assert {(o["colour"], o["shape"]) for o in rec} == {("red", "circle"), ("blue", "rectangle")}
    # one holistic colour label can only name one of them
    assert sc.colour_tag(img) in ("red", "blue")

"""Tests for holographic_uri: deterministic keys, S3-style prefix listing,
address-from-content via the resonator, and the key-depth balance lever."""
import numpy as np
import holographic.misc.holographic_vision as hv
import holographic.io_and_interop.holographic_uri as uri
from holographic.scene_and_pipeline.holographic_scene import auto_tags, SceneCoder
from holographic.agents_and_reasoning.holographic_ai import random_vector as htree_random


def test_make_parse_key_roundtrip():
    tags = {"colour": "red", "shape": "circle", "texture": "smooth"}
    assert uri.make_key(tags) == "red/circle/smooth"
    assert uri.parse_key("red/circle/smooth") == tags
    assert uri.make_key(tags, kv=True) == "colour=red/shape=circle/texture=smooth"
    assert uri.parse_key("colour=red/shape=circle/texture=smooth", kv=True) == tags


def test_put_list_and_bucket():
    s = uri.FacetStore()
    s.put(1, {"colour": "red", "shape": "circle", "texture": "smooth"})
    s.put(2, {"colour": "red", "shape": "line", "texture": "busy"})
    s.put(3, {"colour": "blue", "shape": "circle", "texture": "smooth"})
    assert {r["id"] for r in s.list("red/")} == {1, 2}
    assert {r["id"] for r in s.list("")} == {1, 2, 3}
    assert [r["id"] for r in s.bucket("blue/circle/smooth")] == [3]


def test_common_prefixes_like_s3():
    s = uri.FacetStore()
    for i, t in enumerate([("red", "circle"), ("red", "line"), ("blue", "circle")]):
        s.put(i, {"colour": t[0], "shape": t[1], "texture": "smooth"})
    assert set(s.common_prefixes("")) == {"red/", "blue/"}        # top-level "folders"
    assert set(s.common_prefixes("red/")) == {"red/circle/", "red/line/"}
    assert s.common_prefixes("")["red/"] == 2                     # roll-up counts


def test_address_from_content_via_resonator():
    coder = SceneCoder(dim=1024, seed=0)
    rng = np.random.default_rng(0); ok = 0; n = 40
    pal = {"red": (235, 70, 70), "blue": (80, 120, 235), "green": (70, 200, 110)}
    for i in range(n):
        shp = ["circle", "rectangle", "triangle", "line"][rng.integers(4)]
        col = list(pal)[rng.integers(len(pal))]
        img, _ = hv.make_shape(shp, 64, seed=i, fg=pal[col])
        tags = auto_tags(img)
        key = uri.make_key(tags)
        ok += uri.address_from_content(coder.encode(tags), coder) == key
    assert ok / n >= 0.95                                          # resonator recovers the URI


def test_key_depth_controls_bucket_size():
    rng = np.random.default_rng(1)
    items = [{"colour": c, "shape": s, "texture": t}
             for c in ["red", "blue", "green"] for s in ["circle", "line"]
             for t in ["smooth", "busy"] for _ in range(4)]
    shallow = uri.FacetStore(order=("colour",))
    deep = uri.FacetStore(order=("colour", "shape", "texture"))
    for i, it in enumerate(items):
        shallow.put(i, it); deep.put(i, it)
    assert deep.stats()["max_bucket"] < shallow.stats()["max_bucket"]
    assert deep.stats()["buckets"] > shallow.stats()["buckets"]


def test_nearest_scoped_to_prefix():
    coder = SceneCoder(dim=1024, seed=0)
    s = uri.FacetStore()
    a = {"colour": "red", "shape": "circle", "texture": "smooth"}
    b = {"colour": "red", "shape": "line", "texture": "busy"}
    s.put(1, a, vector=coder.encode(a)); s.put(2, b, vector=coder.encode(b))
    got = s.nearest("red/", coder.encode(a))
    assert got["id"] == 1


def test_bilevel_hot_bucket_is_sublinear():
    # a single hot prefix with many items: the in-bucket forest finds the match
    # touching far fewer than all of them (the bi-level fix for skew).
    coder = SceneCoder(dim=512, seed=0)
    s = uri.FacetStore()
    tag = {"colour": "red", "shape": "circle", "texture": "busy"}
    rng = np.random.default_rng(0)
    vecs = {}
    for i in range(800):                       # all land in ONE bucket: red/circle/busy
        v = htree_random(512, rng); vecs[i] = v
        s.put(i, tag, vector=v)
    key = uri.make_key(tag)
    s.build_indexes(threshold=128)
    assert key in s._idx
    i = 7; got = s.nearest(key, vecs[i], beam=6)
    assert got["id"] == i and s.last_comparisons < 800     # correct AND sub-linear


def test_facetstore_inner_index_migration_is_byte_identical():
    """PARITY: FacetStore's hot-bucket content search now delegates to StructuredIndex (normalize=False)
    instead of a bespoke HoloForest. This proves nearest() returns the SAME record the bare forest would --
    the content store is now literally 'this index at its at-scale operating point', not just by claim."""
    from holographic.misc.holographic_tree import HoloForest
    rng = np.random.default_rng(0)
    tag = {"colour": "red", "shape": "circle", "texture": "busy"}
    s = uri.FacetStore()
    vecs = {}
    for i in range(200):                                   # all land in ONE hot bucket
        v = htree_random(256, rng) * float(rng.uniform(0.5, 2.0))   # deliberately NOT unit-norm
        vecs[i] = v
        s.put(i, tag, vector=v)
    key = uri.make_key(tag)
    s.build_indexes(threshold=128, n_trees=4, leaf_size=64, dim=256)
    assert key in s._idx

    # the bare forest the migration replaced, built identically over the same (insertion-ordered) stack
    bare = HoloForest(256, n_trees=4, leaf_size=64, seed=0).build(np.stack([vecs[i] for i in range(200)]))
    for _ in range(100):
        q = rng.standard_normal(256)
        assert s.nearest(key, q)["id"] == int(bare.recall(q))   # same record as the raw forest, every query

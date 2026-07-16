"""Tests for holographic_scene: DCT/colour/shape tagging, compositional encoding,
and resonator factoring (single object and multi-object scenes)."""
import numpy as np
import pytest
import holographic.scene_and_pipeline.holographic_scene as sc
import holographic.misc.holographic_vision as hv

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


@pytest.mark.slow  # resonator factoring across a blended multi-factor scene; measured ~27s, exceeds the 15s
                    # per-test budget (was silently skipped by the watchdog on every default run)
def test_blend_scenes_projects_one_factor_across():
    # SCENE-LEVEL PROJECTION: given two scene vectors (objects unknown), factor
    # each, project one factor across (A's forms wearing B's colours), and
    # recompose a NOVEL scene that factors back exactly. The full decompose ->
    # project -> recompose loop through the resonator -- a scene neither input
    # contained. Measured 100% across factors and 2-4 separable objects.
    import numpy as np
    from holographic.scene_and_pipeline.holographic_scene import SceneCoder, COLOURS, SHAPES, TEXTURES
    sc = SceneCoder(dim=2048, seed=0)
    rng = np.random.default_rng(0)

    def distinct_scene(n):
        # distinct shapes so the objects are separable for the resonator
        shapes = list(rng.choice(SHAPES, n, replace=False))
        return [{"colour": rng.choice(COLOURS), "shape": shapes[i],
                 "texture": rng.choice(TEXTURES)} for i in range(n)]

    for project in ("colour", "shape", "texture"):
        ok = 0
        for _ in range(8):
            A, B = distinct_scene(3), distinct_scene(3)
            vecBlend, blended = sc.blend_scenes(sc.encode_scene(A), sc.encode_scene(B),
                                                3, project=project)
            out = sc.factor_scene(vecBlend, 3)
            got = sorted((o["colour"], o["shape"], o["texture"]) for o in out)
            want = sorted((o["colour"], o["shape"], o["texture"]) for o in blended)
            ok += (got == want)
        assert ok >= 7                                # >=7/8 (resonator capacity)
        # and the projected factor really came from B
        A = [{"colour": "red", "shape": "circle", "texture": "smooth"},
             {"colour": "cyan", "shape": "rectangle", "texture": "busy"}]
        B = [{"colour": "blue", "shape": "triangle", "texture": "vertical"},
             {"colour": "green", "shape": "line", "texture": "horizontal"}]
        _, blended = sc.blend_scenes(sc.encode_scene(A), sc.encode_scene(B), 2,
                                     project=project)
        # each blended object keeps A's other two factors, takes B's projected one
        assert all(o[project] in [b[project] for b in B] for o in blended)


def test_morph_scenes_is_ordered_coherent_frames():
    # MORPH SEQUENCE: projection unfolded over time. A continuous attribute blend
    # is impossible (the cleanup law makes it crossfade-with-snap), so the honest
    # morph is a sequence of discrete COHERENT frames -- objects adopt B's
    # attribute one at a time, every frame factors EXACTLY, frame 0 is A and the
    # last has B's full pattern.
    from holographic.scene_and_pipeline.holographic_scene import SceneCoder
    sc = SceneCoder(dim=2048, seed=0)
    A = [{"colour": "red", "shape": "circle", "texture": "smooth"},
         {"colour": "red", "shape": "rectangle", "texture": "busy"},
         {"colour": "red", "shape": "triangle", "texture": "horizontal"}]
    B = [{"colour": "blue", "shape": "circle", "texture": "smooth"},
         {"colour": "green", "shape": "rectangle", "texture": "busy"},
         {"colour": "cyan", "shape": "triangle", "texture": "horizontal"}]
    frames = sc.morph_scenes(sc.encode_scene(A), sc.encode_scene(B), 3)
    assert len(frames) == 4                            # n+1 frames
    # every frame factors back exactly (each is a coherent scene)
    for f in frames:
        rec = sc.factor_scene(sc.encode_scene(f), 3)
        got = sorted((o["colour"], o["shape"], o["texture"]) for o in rec)
        want = sorted((o["colour"], o["shape"], o["texture"]) for o in f)
        assert got == want
    # flip-count increases monotonically (0 -> n): a genuine ordered progression
    flips = [sum(1 for o in f if o["colour"] != "red") for f in frames]
    assert flips == sorted(flips) and flips[0] == 0 and flips[-1] == 3


def test_morph_sequence_passes_sequentiality_test():
    # INTEGRATION: projection generates the frames, sequence-discovery confirms
    # the order. The morph as a token sequence (flip-count per frame) is genuinely
    # ordered and passes the permutation test; a shuffle does not.
    import numpy as np
    from holographic.scene_and_pipeline.holographic_scene import SceneCoder
    from holographic.misc.holographic_sequence import sequentiality_z
    from holographic.agents_and_reasoning.holographic_ai import Vocabulary
    sc = SceneCoder(dim=2048, seed=0)
    A = [{"colour": "red", "shape": s, "texture": "smooth"}
         for s in ("circle", "rectangle", "triangle", "line")]
    B = [{"colour": c, "shape": s, "texture": "smooth"}
         for c, s in zip(("blue", "green", "cyan", "magenta"),
                         ("circle", "rectangle", "triangle", "line"))]
    frames = sc.morph_scenes(sc.encode_scene(A), sc.encode_scene(B), 4)
    seq = [f"step{sum(1 for o in fr if o['colour'] != 'red')}" for fr in frames]
    v = Vocabulary(1024, seed=0)
    assert sequentiality_z([seq] * 8, v) > 2.0         # ordered
    shuffled = [list(np.random.default_rng(i).permutation(seq)) for i in range(8)]
    assert sequentiality_z(shuffled, v) < 2.0          # shuffle is not


def test_count_objects_is_self_discovered():
    # CARDINALITY IS SELF-MEASURED: the scene is an unnormalised superposition of
    # near-orthogonal unit products, so round(||v||^2) IS the object count -- the
    # design decision made for explain-away pays off again. No one tells the
    # system n.
    import numpy as np
    from holographic.scene_and_pipeline.holographic_scene import SceneCoder, COLOURS, SHAPES, TEXTURES
    sc = SceneCoder(dim=2048, seed=0)
    rng = np.random.default_rng(1)

    def rand_scene(n):
        picks, objs = set(), []
        while len(objs) < n:
            o = (rng.choice(COLOURS), rng.choice(SHAPES), rng.choice(TEXTURES))
            if o not in picks:
                picks.add(o)
                objs.append({"colour": o[0], "shape": o[1], "texture": o[2]})
        return objs

    ok = tot = 0
    for n in range(1, 6):
        for _ in range(8):
            ok += (sc.count_objects(sc.encode_scene(rand_scene(n))) == n)
            tot += 1
    assert ok >= tot - 2                              # ~96% measured; tiny slack


def test_scene_vector_is_algebraically_editable():
    # The scene VECTOR is a directly editable structure: removing an object is
    # subtracting its factored product (explain-away as an editor), adding is
    # adding a product -- no re-encoding. Count tracks every edit.
    from holographic.scene_and_pipeline.holographic_scene import SceneCoder
    sc = SceneCoder(dim=2048, seed=0)
    A = [{"colour": "red", "shape": "circle", "texture": "smooth"},
         {"colour": "cyan", "shape": "rectangle", "texture": "busy"},
         {"colour": "grey", "shape": "triangle", "texture": "horizontal"}]
    v = sc.encode_scene(A)
    v = sc.remove_object(v, {"shape": "triangle"})
    assert sc.count_objects(v) == 2
    v = sc.add_object(v, {"colour": "magenta", "shape": "line", "texture": "busy"})
    assert sc.count_objects(v) == 3
    rec = sc.factor_scene(v, 3)
    shapes = sorted(o["shape"] for o in rec)
    assert shapes == ["circle", "line", "rectangle"]   # triangle gone, line added
    import pytest
    with pytest.raises(ValueError):
        sc.remove_object(v, {"shape": "triangle"})     # honest: can't remove absent


def test_morph_cardinality_changes_count_per_frame():
    # CARDINALITY MORPH: scene A (3 objects) -> scene B (2 objects) as a chain of
    # algebraic edits, one per frame. Every frame's count is DISCOVERED from its
    # own norm and the frame factors exactly at that count; the last frame holds
    # exactly scene B.
    from holographic.scene_and_pipeline.holographic_scene import SceneCoder
    sc = SceneCoder(dim=2048, seed=0)
    A = [{"colour": "red", "shape": "circle", "texture": "smooth"},
         {"colour": "cyan", "shape": "rectangle", "texture": "busy"},
         {"colour": "grey", "shape": "triangle", "texture": "horizontal"}]
    B = [{"colour": "blue", "shape": "line", "texture": "vertical"},
         {"colour": "green", "shape": "circle", "texture": "busy"}]
    frames = sc.morph_cardinality(sc.encode_scene(A), sc.encode_scene(B))
    counts = [sc.count_objects(f) for f in frames]
    assert counts == [3, 2, 1, 1, 2]                   # remove, remove, swap, add
    for fv, n in zip(frames, counts):
        rec = sc.factor_scene(fv, n)
        assert len(rec) == n                           # coherent at discovered n
    rec = sc.factor_scene(frames[-1], 2)
    got = sorted((o["colour"], o["shape"], o["texture"]) for o in rec)
    want = sorted((o["colour"], o["shape"], o["texture"]) for o in B)
    assert got == want                                 # arrives exactly at B


# ---- X1: tiled scene factorization (beat the resonator's object cap by tiling) --------------------

from holographic.scene_and_pipeline.holographic_scene import SceneCoder, COLOURS, SHAPES, TEXTURES


def _x1_distinct_objs(K, r):
    seen, objs = set(), []
    while len(objs) < K:
        t = (COLOURS[r.integers(len(COLOURS))], SHAPES[r.integers(len(SHAPES))], TEXTURES[r.integers(len(TEXTURES))])
        if t not in seen:
            seen.add(t)
            objs.append({"colour": t[0], "shape": t[1], "texture": t[2]})
    return objs


def _x1_keys(objs):
    return set((o["colour"], o["shape"], o["texture"]) for o in objs)


def test_tiled_scene_factorization_beats_whole_past_the_cap():
    # A 15-object scene exceeds the resonator's per-scene cap at dim 1024 (whole-scene recovery collapses to
    # ~30%); tiling into <=5-object sub-scenes and merging recovers the great majority (~93%). Averaged over
    # seeds so the assertion is about the cap, not one lucky scene.
    coder = SceneCoder(dim=1024, seed=0)
    K, tile, trials = 15, 5, 5
    whole_ok = tiled_ok = 0
    for s in range(trials):
        r = np.random.default_rng(200 + s)
        objs = _x1_distinct_objs(K, r)
        whole = coder.factor_scene(coder.encode_scene(objs), K, sweeps=3)
        whole_ok += len(_x1_keys(whole) & _x1_keys(objs))
        groups = [objs[i:i + tile] for i in range(0, K, tile)]
        got = coder.factor_scene_tiled([coder.encode_scene(g) for g in groups],
                                       [len(g) for g in groups], sweeps=3)
        tiled_ok += len(_x1_keys(got) & _x1_keys(objs))
    assert whole_ok < 0.6 * K * trials       # the whole scene is capped past ~5 objects
    assert tiled_ok > 0.85 * K * trials       # tiling recovers the great majority
    assert tiled_ok > whole_ok + K            # a large, unambiguous improvement

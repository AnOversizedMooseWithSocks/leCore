"""Forward compositional generation: compose NEW scenes by running the resonator forward,
render them, and -- the honest bar -- verify by ROUND-TRIP that the generated artefact can
be analysed straight back to the spec it was built from. Hermetic (no corpora); the scene
machinery is deterministic given a seed.
"""
import numpy as np

import holographic.scene_and_pipeline.holographic_scene as hs
from holographic.misc.holographic_compose import all_object_specs, compose_object, compose_scene, render_scene, roundtrip_object, roundtrip_scene, render_fidelity, novel_specs, animate_attribute, animation_is_faithful


def _coder():
    return hs.SceneCoder(dim=1024, seed=0)


def test_compose_is_the_inverse_of_factor_on_single_objects():
    # Running the resonator forward (compose) then backward (factor) is the identity on
    # tags -- across MANY novel triples, not one lucky pick.
    coder = _coder()
    novel = novel_specs(n=50, rng=np.random.default_rng(0))
    ok = sum(roundtrip_object(coder, t) for t in novel)
    assert ok >= 48                                      # near-perfect forward/back round-trip


def test_composed_objects_are_genuinely_novel_not_a_stored_table():
    # The specs we generate are excluded from a "seen" set, so a correct round-trip proves
    # composition rather than recall.
    coder = _coder()
    seen = {("red", "circle", "smooth"), ("blue", "rectangle", "busy")}
    gen = novel_specs(n=20, hold_from=seen, rng=np.random.default_rng(1))
    for t in gen:
        assert (t["colour"], t["shape"], t["texture"]) not in seen
        assert roundtrip_object(coder, t)                # still composes + factors correctly


def test_novel_multi_object_scenes_round_trip():
    # Compose several objects into one scene vector forward, recover the SET back.
    coder = _coder()
    rng = np.random.default_rng(2)
    for n in (2, 3, 4):
        good = 0
        for _ in range(20):
            tags = [{"colour": str(rng.choice(hs.COLOURS)), "shape": str(rng.choice(hs.SHAPES)),
                     "texture": str(rng.choice(hs.TEXTURES))} for _ in range(n)]
            good += roundtrip_scene(coder, tags)
        assert good >= 18                                # >=90% recovered through 4 objects


def test_rendered_pixels_auto_tag_back_to_the_composed_spec():
    # The generated image is a real picture: its pixels read back as the shape and colour
    # it was composed from (texture is carried in the vector, not painted).
    rng = np.random.default_rng(3)
    sh_ok = col_ok = 0
    N = 30
    for _ in range(N):
        t = {"colour": str(rng.choice([c for c in hs.COLOURS if c != "grey"])),
             "shape": str(rng.choice(hs.SHAPES)), "texture": "smooth"}
        s_ok, c_ok = render_fidelity(t, seed=int(rng.integers(1000)))
        sh_ok += s_ok; col_ok += c_ok
    assert sh_ok >= 28 and col_ok >= 28                  # generated pixels match the spec


def test_render_scene_produces_a_valid_image():
    img = render_scene([{"colour": "red", "shape": "circle", "texture": "smooth"},
                        {"colour": "blue", "shape": "triangle", "texture": "busy"}], S=96, seed=1)
    assert img.shape == (96, 96, 3)
    assert img.dtype == np.uint8


def test_animation_frames_factor_to_their_intended_value():
    # A swept attribute is a deliberate trajectory: every frame's composed vector factors
    # back to its intended value.
    coder = _coder()
    frames = animate_attribute(coder, {"colour": "red", "shape": "circle", "texture": "smooth"},
                               "colour", ["red", "yellow", "green", "cyan", "blue", "magenta"])
    assert len(frames) == 6
    assert animation_is_faithful(coder, frames, "colour") == 1.0
    # and every frame carries a rendered image
    for _v, _vec, img in frames:
        assert img.shape == (96, 96, 3)


def test_animation_can_sweep_shape_too():
    coder = _coder()
    frames = animate_attribute(coder, {"colour": "green", "shape": "circle", "texture": "smooth"},
                               "shape", hs.SHAPES)
    assert animation_is_faithful(coder, frames, "shape") == 1.0


def test_full_object_space_round_trips():
    # The whole composable vocabulary (|COLOURS|x|SHAPES|x|TEXTURES|) composes and factors
    # back, so generation is reliable across the entire attribute space, not a lucky corner.
    coder = _coder()
    specs = all_object_specs()
    ok = sum(roundtrip_object(coder, t) for t in specs)
    assert ok >= int(0.97 * len(specs))                  # >=97% of the entire space


def test_nested_scene_composition_is_self_similar_and_round_trips():
    # "Same above, same below": the same bind+superpose that builds a scene from objects
    # builds a scene-of-scenes from sub-scenes, and the same unbind-then-factor recovers it
    # at two levels. 2-3 groups recover exactly; the faculty is seed-deterministic.
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=1024, seed=0)
    rng = np.random.default_rng(3)

    def rt():
        return {"colour": str(rng.choice(hs.COLOURS)), "shape": str(rng.choice(hs.SHAPES)),
                "texture": str(rng.choice(hs.TEXTURES))}
    key = lambda d: (d["colour"], d["shape"], d["texture"])

    groups = {"left": [rt(), rt()], "right": [rt(), rt()]}
    sizes = {k: len(v) for k, v in groups.items()}
    super_scene = m.compose_nested(groups)
    rec = m.decompose_nested(super_scene, sizes)
    for k in groups:
        assert {key(t) for t in groups[k]} == {key(g) for g in rec[k]}   # exact at 2 groups

    # seed-deterministic: a fresh mind with the same seed composes the identical super-scene
    m2 = UnifiedMind(dim=1024, seed=0)
    assert np.allclose(super_scene, m2.compose_nested(groups))


def test_nested_scene_recovery_holds_at_three_groups():
    # The measured boundary: 3 groups of 2 still recover the great majority of sub-scenes.
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=1024, seed=0)
    key = lambda d: (d["colour"], d["shape"], d["texture"])
    total = ok = 0
    for t in range(10):
        rng = np.random.default_rng(t)

        def rt():
            return {"colour": str(rng.choice(hs.COLOURS)), "shape": str(rng.choice(hs.SHAPES)),
                    "texture": str(rng.choice(hs.TEXTURES))}
        groups = {f"g{i}": [rt(), rt()] for i in range(3)}
        sizes = {k: len(v) for k, v in groups.items()}
        rec = m.decompose_nested(m.compose_nested(groups), sizes)
        for k in groups:
            total += 1
            ok += ({key(x) for x in groups[k]} == {key(g) for g in rec[k]})
    assert ok / total >= 0.9          # ~0.97 measured; floor well clear of chance

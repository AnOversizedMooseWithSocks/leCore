"""Tests for the controlled semantic scene layer: parse -> VSA encode -> bidirectional query -> batch -> control."""
import numpy as np
from holographic.misc.holographic_unified import UnifiedMind
from holographic.simulation_and_physics.holographic_semantic import parse_description, encode_objects, encode_scene, decode_object, query_scene, batch_set, find_objects, control_spec, realize_scene, schema_codebooks, render_scene

_DESC = ("A red ball sitting inside of a box with a glass material, with a metallic elongated box leaning on the "
         "glass box diagonally. The sun is bright in the sky, which is partly cloudy")


def _mind():
    return UnifiedMind(dim=1024, seed=0)


def test_parses_example_into_three_objects_and_environment():
    scene = parse_description(_DESC)
    objs = scene["objects"]
    assert len(objs) == 3
    assert objs[0]["shape"] == "sphere" and objs[0]["color"] == "red"
    assert objs[0]["relation"] == ("inside", 1)               # ball inside the box (index 1)
    assert objs[1]["shape"] == "box" and objs[1]["material"] == "glass"
    assert objs[2]["material"] == "metal" and objs[2]["size"] == "elongated"
    assert objs[2]["relation"] == ("leaning", 1)              # metal box leans on the glass box (index 1)
    # the parser now ALSO derives a `lighting` hint from the sky clause, so check the two keys this test is about
    # rather than pinning the whole dict (which would break every time the environment vocabulary grows).
    env = scene["environment"]
    assert env["sun"] == "bright" and env["sky"] == "partly"


def test_synonyms_fold_to_canonical():
    o = parse_description("a metallic cube and a glassy orb")["objects"]
    assert o[0]["shape"] == "box" and o[0]["material"] == "metal"
    assert o[1]["shape"] == "sphere" and o[1]["material"] == "glass"


def test_object_record_decodes_bidirectionally():
    mind = _mind()
    objs = parse_description("a blue metallic box")["objects"]
    rec = encode_objects(objs, mind)[0]
    d = decode_object(rec, mind)
    assert d["shape"] == "box" and d["color"] == "blue" and d["material"] == "metal"


def test_bundled_scene_recovers_all_attributes():
    """Every attribute of every object decodes from the single superposed scene vector (the headline VSA win)."""
    mind = _mind()
    objs = parse_description(_DESC)["objects"]
    sv, recs, roles = encode_scene(objs, mind)
    ok = tot = 0
    for i, o in enumerate(objs):
        q = query_scene(sv, roles, mind, i)
        for f in ("shape", "color", "material", "size"):
            truth = o[f] if o[f] is not None else "none"
            tot += 1; ok += (q[f] == truth)
    assert ok == tot                                          # 12/12 recovered through the superposition


def test_batch_edit_and_find():
    objs = parse_description(_DESC)["objects"]
    assert find_objects(objs, material="glass") == [1]
    mirrored = batch_set(objs, "material", "mirror")
    assert all(o["material"] == "mirror" for o in mirrored)
    assert objs[1]["material"] == "glass"                     # original untouched (immutable input)


def test_realize_positions_inside_at_container_center():
    objs = parse_description(_DESC)["objects"]
    r = realize_scene(objs)
    assert len(r) == 3
    # the ball (inside the box) sits at the box's centre and is shrunk
    ball, box = r[0]["sdf"], r[1]["sdf"]
    assert np.allclose(ball.c, box.c) and ball.r < 0.5


def test_control_spec_maps_keywords_to_widgets():
    spec = control_spec("control the ball size and how metallic it is")
    params = [c["param"] for c in spec["controls"]]
    assert spec["target"] == "ball" and "size" in params and "metallic" in params
    assert all("min" in c or "options" in c for c in spec["controls"])   # each is a renderable widget


def test_control_spec_all_target():
    spec = control_spec("make all materials reflective")
    assert spec["target"].startswith("all")
    assert any(c["param"] == "reflect" for c in spec["controls"])


# ---- SEMANTIC-2: synonym grounding + single-pass material-id render --------------------------------------------
def test_synonym_table_resolves_common_synonyms():
    from holographic.simulation_and_physics.holographic_semantic import SynonymResolver
    r = SynonymResolver()                                     # table-based, deterministic
    # NOTE: "crimson" used to live here, but it is now a first-class colour (its own RGB), so it is not a synonym
    # any more. The others below are still out-of-vocabulary words the table resolves.
    # "giant"/"miniature" were also promoted to first-class SIZES, so they are no longer out-of-vocabulary either.
    # "petite" and "stretched" are the size words the table still resolves.
    cases = [("scarlet", "red"), ("azure", "blue"), ("emerald", "green"),
             ("spherical", "sphere"), ("cubic", "box"), ("petite", "small"), ("stretched", "elongated")]
    for w, want in cases:
        assert (r.classify_unknown(w) or (None, None))[1] == want, (w, r.classify_unknown(w))


def test_parse_with_resolver_maps_synonyms_and_no_double_object():
    from holographic.simulation_and_physics.holographic_semantic import SynonymResolver
    r = SynonymResolver()
    objs = parse_description("a scarlet spherical ball beside a petite chrome cube", resolver=r)["objects"]
    assert len(objs) == 2                                     # 'spherical' is an adjective, not its own object
    assert objs[0]["shape"] == "sphere" and objs[0]["color"] == "red"
    assert objs[1]["shape"] == "box" and objs[1]["material"] == "metal" and objs[1]["size"] == "small"


def test_learned_resolver_is_optional_and_weak_on_tiny_corpus():
    """The learned path exists but is honestly weak on a tiny corpus -- the table is what carries it."""
    from holographic.simulation_and_physics.holographic_semantic import SynonymResolver
    r = SynonymResolver(learned=True)                         # builds the random-indexing encoder
    assert r.enc is not None
    # table still wins for known synonyms regardless of the weak encoder
    assert r.classify_unknown("scarlet") == ("color", "red")   # crimson is a real colour now; scarlet still resolves


def test_single_pass_render_sees_through_glass():
    """A red ball behind a glass panel shows red THROUGH the glass (the see-through secondary ray)."""
    import numpy as np
    import holographic.simulation_and_physics.holographic_semantic as HS
    from holographic.simulation_and_physics.holographic_semantic import render_scene, _SphereSDF, _BoxSDF, COLORS, MATERIAL_RENDER
    from holographic.rendering.holographic_render import Camera
    rs = [{"sdf": _SphereSDF((0, 0, 0), 0.95), "color": COLORS["red"], "material": MATERIAL_RENDER[None],
           "mat_name": None, "name": "red ball"},
          {"sdf": _BoxSDF((0, 0, 1.25), (1.2, 1.2, 0.12)), "color": (0.8, 0.8, 0.8),
           "material": MATERIAL_RENDER["glass"], "mat_name": "glass", "name": "glass"}]
    orig = HS.realize_scene
    HS.realize_scene = lambda objs, **k: rs
    try:
        frame = render_scene([0, 1], Camera(eye=(0, 0, 4.2), target=(0, 0, 0), fov_deg=42.0), width=120, height=120)
    finally:
        HS.realize_scene = orig
    patch = frame[45:75, 45:75]
    assert patch[:, :, 0].mean() > patch[:, :, 1].mean() and patch[:, :, 0].mean() > patch[:, :, 2].mean()


def test_render_scene_from_description_runs():
    objs = parse_description("a red ball beside a blue box")["objects"]
    from holographic.rendering.holographic_render import Camera
    frame = render_scene(objs, Camera(eye=(2, 1.4, 4.0), target=(0, 0, 0)), width=64, height=64)
    assert frame.shape == (64, 64, 3) and frame.std() > 0.03


def test_supersampling_reduces_aliasing():
    """SSAA (ss>1) averages multiple rays per output pixel -> measurably less edge grain than 1 ray/pixel."""
    import numpy as np
    from holographic.rendering.holographic_render import Camera
    from holographic.rendering.holographic_postfx import _fft_blur
    objs = parse_description("a big red ball")["objects"]
    cam = Camera(eye=(0, 0.4, 4.0), target=(0, 0, 0), fov_deg=40.0)
    grain = lambda f: float(np.abs(f - _fft_blur(f, 1.0)).mean())
    g1 = grain(render_scene(objs, cam, width=120, height=120, ss=1, ground=False, dither=0.0))
    g3 = grain(render_scene(objs, cam, width=120, height=120, ss=3, ground=False, dither=0.0))
    assert g3 < g1                                            # supersampling is smoother


def test_ground_plane_catches_objects():
    """ground=True adds a floor below the objects, so the lower frame fills with surface instead of sky."""
    import numpy as np
    from holographic.rendering.holographic_render import Camera
    objs = parse_description("a red ball")["objects"]
    cam = Camera(eye=(0, 1.2, 4.0), target=(0, -0.2, 0), fov_deg=45.0)
    no_ground = render_scene(objs, cam, width=64, height=64, ss=1, ground=False, dither=0.0)
    with_ground = render_scene(objs, cam, width=64, height=64, ss=1, ground=True, dither=0.0)
    # bottom rows differ once a floor is present
    assert not np.allclose(no_ground[-12:], with_ground[-12:])


def test_adaptive_aa_matches_brute_at_far_fewer_rays():
    """Edge-adaptive supersampling (default) matches brute SSAA quality using a fraction of the rays."""
    import numpy as np
    from holographic.rendering.holographic_render import Camera
    objs = parse_description("a big red ball beside a blue box")["objects"]
    cam = Camera(eye=(0.4, 1.6, 5.0), target=(0, 0, 0), fov_deg=42.0)
    sa, sb = {}, {}
    fa = render_scene(objs, cam, width=120, height=120, ss=3, adaptive=True, dither=0.0, stats=sa)
    fb = render_scene(objs, cam, width=120, height=120, ss=3, adaptive=False, dither=0.0, stats=sb)
    assert sa["rays"] < sb["rays"] * 0.6                      # adaptive casts far fewer rays
    assert np.abs(fa - fb).mean() < 0.002                    # but the result is essentially identical


def test_incremental_rerender_only_touches_changed_object():
    """A material edit through SceneRenderer re-renders only the changed object's pixels, not the whole frame."""
    import numpy as np
    from holographic.rendering.holographic_render import Camera
    from holographic.simulation_and_physics.holographic_semantic import SceneRenderer
    objs = parse_description("a green ball beside a blue box")["objects"]
    cam = Camera(eye=(0.3, 1.7, 6.2), target=(0, 0, 0), fov_deg=50.0)
    r = SceneRenderer(cam, width=120, height=120, ss=2)
    before = r.render(objs).copy()
    # pick a visible non-ground object
    vals, counts = np.unique(r.idbuf, return_counts=True)
    ng = r.idbuf.max()
    cand = sorted([(int(v), int(c)) for v, c in zip(vals, counts) if 0 <= v < ng], key=lambda x: -x[1])
    target = cand[0][0]
    after, st = r.set_attr(target, "material", "mirror")
    assert 0 < st["rerendered_pixels"] < st["full_pixels"]   # only a subset re-rendered
    # pixels NOT belonging to the changed object are byte-identical
    untouched = r.idbuf != target
    assert np.array_equal(before[untouched], after[untouched])


# ---- HYPERREAL: PBR path tracing + volumetric objects ---------------------------------------------------------
def test_pbr_materials_resolve():
    from holographic.simulation_and_physics.holographic_semantic import _pbr_props
    g = _pbr_props({"material": "gold", "color": None})
    assert g[1] == 1.0 and tuple(round(float(x), 2) for x in g[0]) == (1.0, 0.78, 0.34)   # gold metal + tint
    e = _pbr_props({"material": "emissive", "color": "blue"})
    assert e[3].sum() > 0                                     # emissive objects emit light
    m = _pbr_props({"material": "matte", "color": "red"})
    assert m[1] == 0.0 and m[2] > 0.5                         # matte: dielectric, rough


def test_path_traced_render_runs():
    import numpy as np
    from holographic.rendering.holographic_render import Camera
    from holographic.simulation_and_physics.holographic_semantic import parse_description, render_scene_pbr
    objs = parse_description("a gold ball beside a matte red box")["objects"]
    cam = Camera(eye=(0.2, 1.6, 5.4), target=(0, 0.1, 0), fov_deg=44.0)
    f = render_scene_pbr(objs, cam, width=48, height=48, spp=4, max_bounce=3, dither=0.0)
    assert f.shape == (48, 48, 3) and f.std() > 0.03


def test_volumetric_materials_parse_and_render():
    import numpy as np
    from holographic.rendering.holographic_render import Camera
    from holographic.simulation_and_physics.holographic_semantic import parse_description, render_scene, _VOLUMETRIC
    # "cloud" is now a volumetric material in its own right, so "a smoke cloud" would parse as TWO blobs.
    objs = parse_description("a red ball beside a smoke")["objects"]
    assert len(objs) == 2 and objs[1]["material"] == "smoke" and "smoke" in _VOLUMETRIC
    # a bare volumetric word creates its own blob object (no shape noun needed)
    fire = parse_description("a fire")["objects"]
    assert len(fire) == 1 and fire[0]["material"] == "fire"
    cam = Camera(eye=(0.3, 1.6, 5.4), target=(0, 0.2, 0), fov_deg=46.0)
    f = render_scene(objs, cam, width=64, height=64, ss=1, dither=0.0)
    assert f.shape == (64, 64, 3) and f.std() > 0.03


def test_cloud_word_does_not_break_object_clause():
    """'cloud' next to a shape must NOT misroute the clause to environment."""
    from holographic.simulation_and_physics.holographic_semantic import parse_description
    objs = parse_description("a blue ball beside a smoke cloud")["objects"]
    assert any(o["shape"] == "sphere" and o["color"] == "blue" for o in objs)


def test_glass_is_refractive_in_pbr():
    from holographic.simulation_and_physics.holographic_semantic import _pbr_props
    g = _pbr_props({"material": "glass", "color": None})
    assert g[4] > 1.0                                    # glass carries an ior > 1 (refractive dielectric)
    assert _pbr_props({"material": "matte", "color": "red"})[4] == 0.0   # opaque has ior 0


def test_adaptive_sampling_uses_fewer_samples():
    import numpy as np
    from holographic.rendering.holographic_render import Camera
    from holographic.simulation_and_physics.holographic_semantic import parse_description, render_scene_pbr
    objs = parse_description("a gold ball beside a matte red box")["objects"]
    cam = Camera(eye=(0.2, 1.6, 5.4), target=(0, 0.1, 0), fov_deg=44.0)
    st = {}
    render_scene_pbr(objs, cam, width=40, height=40, spp=4, adaptive_spp=8, noise_pct=70, dither=0.0, stats=st)
    assert st["total_samples"] < st["uniform_equiv_samples"]      # adaptive spends fewer than uniform (spp+extra)
    assert 0.0 < st["noisy_fraction"] < 1.0                       # only some pixels flagged noisy

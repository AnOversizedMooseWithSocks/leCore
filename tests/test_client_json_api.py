"""C2 + C3 from the comfy-lecore client audit: make the engine callable by a JSON client.

The audit's headline was that `render_mesh` -- the flagship mesh->image path -- "cannot be called by ANY JSON
client today, including your own POST /invoke". True, and reproduced. But Rule 0 corrected the diagnosis: the
report also said the constructors were missing ("make_box, box_mesh, cube, primitive -- all absent"), and they
were not. `m.render_mesh(m.mesh_box(), m.camera(...))` already rendered, in-process, with no class imports.

So there were three separate bugs wearing one costume:
  * C2(a) DISCOVERABILITY -- `find_capability("make a box")` answered "Catmull-Clark subdivision", never
    `mesh_box`. A capability find_capability cannot surface does not exist; the vocabulary was the bug.
  * C2(b) COERCION -- dict args raised `AttributeError: 'dict' object has no attribute 'faces'` from deep inside
    the rasteriser, naming neither the caller nor the cure.
  * C2(c) THE TWO-CAMERA TRAP -- `CameraController` has no `projection_matrix()`, so it failed deep in the MVP
    build ("I fell in it during the audit"). It had carried `to_camera()` the whole time; nothing called it.
  * C3 -- dispatch existed ONLY inside Service._invoke, so every client re-implemented it and each copy could
    drift.

These tests pin the client's own acceptance criteria, verbatim where possible.
"""
import numpy as np
import pytest

from holographic.io_and_interop.holographic_coerce import as_camera, as_mesh
from holographic.mesh_and_geometry.holographic_mesh import Mesh
from holographic.misc.holographic_unified import UnifiedMind
from holographic.rendering.holographic_camera import CameraController
from holographic.rendering.holographic_render import Camera

_V = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_F = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]


@pytest.fixture(scope="module")
def mind():
    return UnifiedMind(dim=64, seed=0)


def test_the_flagship_path_needs_no_class_imports(mind):
    """The premise Rule 0 rescued: this ALWAYS worked. If it ever breaks, the audit's "constructors are absent"
    reading becomes true and C2 regresses to a real capability gap."""
    img = mind.render_mesh(mind.mesh_box(), mind.camera(eye=(2, 2, 2), target=(0, 0, 0)), width=16, height=16)
    assert img.shape == (16, 16, 3)


def test_render_mesh_is_json_drivable(mind):
    """C2's acceptance: plain JSON in, image out -- what a node pack or POST /invoke can actually send."""
    img = mind.render_mesh({"vertices": _V, "faces": _F}, camera={"eye": [2, 2, 2], "target": [0, 0, 0]},
                           width=16, height=16)
    assert img.shape == (16, 16, 3)


def test_the_look_spelling_from_the_audit_works(mind):
    """The auditor wrote camera={'eye':..., 'look':...}. Accept their wording rather than punish it."""
    img = mind.render_mesh({"vertices": _V, "faces": _F}, camera={"eye": [2, 2, 2], "look": [0, 0, 0]},
                           width=8, height=8)
    assert img.shape == (8, 8, 3)


def test_the_two_camera_trap_is_closed(mind):
    """C2(c): a CameraController handed to the rasteriser used to fail deep inside the MVP build. The premise is
    asserted too -- if CameraController ever grows projection_matrix directly, this coercion is dead weight."""
    cc = mind.camera_controller(eye=(2, 2, 2), target=(0, 0, 0))
    assert not hasattr(cc, "projection_matrix"), "premise changed: revisit whether coercion is still needed"
    assert mind.render_mesh(mind.mesh_box(), cc, width=8, height=8).shape == (8, 8, 3)


def test_real_objects_pass_through_by_identity():
    """Additive-only: coercion must not copy or rebuild a real object, or a caller's edits would silently vanish
    and existing renders could shift."""
    m = Mesh(np.array(_V), np.array(_F))
    c = Camera(eye=(1.0, 1.0, 1.0), target=(0.0, 0.0, 0.0))
    assert as_mesh(m) is m
    assert as_camera(c) is c


def test_coercion_errors_name_the_fix():
    """KEPT NEGATIVE: the original complaint was an AttributeError from 200 lines deeper that named neither the
    caller nor the cause. A bad payload must fail AT THE EDGE with a message a client can act on."""
    with pytest.raises(TypeError, match="vertices"):
        as_mesh({"vertices": _V})                      # no faces
    with pytest.raises(TypeError, match="mesh"):
        as_mesh(42)
    with pytest.raises(TypeError, match="camera"):
        as_camera(42)
    with pytest.raises(TypeError):
        as_camera({"eye": [0, 0, 1], "zoom": 2})       # unknown field, not silently dropped


@pytest.mark.parametrize("phrasing", ["make a box", "make_box", "cube", "primitive", "box mesh"])
def test_the_constructors_are_findable_by_the_words_a_client_types(mind, phrasing):
    """C2(a): these are the audit's OWN words. Before the aliases every one of them returned Catmull-Clark
    subdivision or a scene describer -- which is why a working capability was reported as absent."""
    assert any(c.name == "mesh_box" for c in mind.find_capability(phrasing)[:3]), phrasing


@pytest.mark.parametrize("phrasing", ["make a camera", "make_camera", "camera from eye and target"])
def test_the_camera_is_findable(mind, phrasing):
    assert any(c.name in ("camera", "camera_controller") for c in mind.find_capability(phrasing)[:3]), phrasing


def test_invoke_dispatches_by_name(mind):
    """C3's acceptance, their exact shape."""
    assert mind.invoke("semantic_tag_coverage", {})["tagged"] > 500
    assert mind.invoke("infer_semantic_tag", {"name": "render_scene"}) == "render/raster"
    assert mind.invoke("infer_semantic_tag", ["render_scene"]) == "render/raster"      # positional form


@pytest.mark.parametrize("bad", ["_job_manager", "_editor", "nope", "", None])
def test_invoke_refuses_private_and_unknown(mind, bad):
    """It must RAISE, never return something a caller could mistake for a result."""
    with pytest.raises(ValueError):
        mind.invoke(bad, {})


def test_invoke_refuses_a_non_callable_attribute(mind):
    """`dim` is a public attribute but not a faculty -- the callable check is load-bearing."""
    assert isinstance(mind.dim, int)
    with pytest.raises(ValueError):
        mind.invoke("dim", {})


def test_service_delegates_to_mind_invoke():
    """The point of C3: ONE set of dispatch rules. If the service ever re-grows its own copy, they can drift --
    which is the bug the client hit from the other side."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "holographic_service.py").read_text(encoding="utf-8")
    assert "self.mind.invoke(name, args)" in src, "Service._invoke must delegate, not re-implement dispatch"


def test_c2_and_c3_compose(mind):
    """The whole point, end to end: a JSON client calls the flagship by name with plain JSON."""
    img = mind.invoke("render_mesh", {"mesh": {"vertices": _V, "faces": _F},
                                      "camera": {"eye": [2, 2, 2], "target": [0, 0, 0]},
                                      "width": 12, "height": 12})
    assert img.shape == (12, 12, 3)


# ---------------------------------------------------------------- C9 (structured args on skill cards)

def test_skill_cards_expose_params_as_data(mind):
    """C9: the card served `signature` as PROSE, so every client wrote a parser to turn it back into fields. The
    data is available from `inspect` -- which also knows *args/**kwargs, as a naive split does not."""
    card = mind.describe_skill("render_mesh")
    names = [p["name"] for p in card["params"]]
    assert names[:2] == ["mesh", "camera"] and "self" not in names
    by = {p["name"]: p for p in card["params"]}
    assert by["mesh"]["required"] is True and by["mesh"]["default"] is None
    assert by["width"]["required"] is False and by["width"]["default"] == "512"


def test_cards_are_json_safe(mind):
    """A tuple default like (0.8, 0.8, 0.8) is not a JSON scalar. `default` is repr'd so a client that dumps the
    card is not handed a landmine."""
    import json
    for n in ("render_mesh", "image_to_3d", "invoke"):
        json.dumps(mind.describe_skill(n))


def test_primary_and_produces_are_served(mind):
    """The two fields the audit said kill client guessing: which param takes the piped input, and what comes out.
    A client that guesses `primary` wrong wires the wrong socket."""
    r = mind.describe_skill("render_mesh")
    assert (r["primary"], r["consumes"], r["produces"]) == ("mesh", ["mesh"], ["image"])
    i = mind.describe_skill("image_to_3d")
    assert (i["primary"], i["consumes"], i["produces"]) == ("image", ["image"], ["mesh"])


def test_both_card_kinds_answer_the_same_question(mind):
    """describe_skill("render_mesh") resolved through the METHOD index and served params; "image_to_3d" is ALSO a
    catalog entry, so it returned through the CAPABILITY branch and served none -- same question, two answers,
    decided by which index happened to hold the name. `method` is now present on both kinds and always means the
    same thing."""
    for n in ("render_mesh", "image_to_3d", "invoke", "Semantic action menu coverage (verb tags)"):
        card = mind.describe_skill(n)
        assert "method" in card, n
        if card["method"]:
            assert callable(getattr(mind, card["method"], None)), (n, card["method"])
            assert "params" in card and "primary" in card, n


def test_module_entries_never_claim_a_method_even_in_an_unverified_catalog(mind):
    """FIFTH instance of the one-source shape, fixed at the SOURCE instead of per-consumer. holographic_skills
    builds its own catalog WITHOUT seed_from_mind, so it never ran the verification pass that nulls liars -- and
    it served method="holographic_rayindex", a call that does not exist. _derive_method now refuses a
    module-shaped name outright, so every consumer is honest whether it verifies or not."""
    for n in ("holographic_rayindex", "holographic_archive", "holographic_pivot"):
        card = mind.describe_skill(n)
        if card:
            assert card.get("method") is None, (n, card.get("method"))


# ---------------------------------------------------------------- C14/C15/C16 (P2 footguns)

def test_features_answers_the_clients_preflight_in_one_call(mind):
    """C14: the node pack hardcoded a list of faculty names to preflight and noted "that list will rot". It rots
    SILENTLY -- a missing faculty and a renamed one both look like an absent attribute from outside."""
    want = ["pipeline_map", "suggest_pipeline", "io_kinds", "resolve_capability_uri", "job_list",
            "browse_capabilities", "set_file_root", "invoke", "job_submit", "features"]
    got = mind.features(want)
    assert got == {n: True for n in want}, got
    assert mind.features("invoke") == {"invoke": True}, "a bare string is a convenience, not an error"
    assert mind.features(["definitely_not_a_faculty"]) == {"definitely_not_a_faculty": False}


def test_features_never_advertises_a_private_faculty(mind):
    """Private names are not part of the contract; a client that discovers one has found a footgun."""
    assert mind.features(["_job_manager", "_editor"]) == {"_job_manager": False, "_editor": False}
    assert not any(n.startswith("_") for n in mind.features())


def test_version_identifies_the_build(mind):
    v = mind.version()
    assert set(v) == {"engine", "capabilities_schema", "dim", "seed"}
    assert v["dim"] == mind.dim and isinstance(v["seed"], int)


def test_io_kinds_publishes_its_contract(mind):
    """C15: a client hardcoded these as a dropdown fallback and asked what it can rely on. The STABLE set must
    stay in the vocabulary -- renaming one would break every tag at once."""
    kinds = mind.io_kinds()
    for stable in ("mesh", "points", "sdf", "sdf_scene", "field", "image", "hypervector",
                   "transform", "selection", "scalar"):
        assert stable in kinds, (stable, "a STABLE io kind vanished -- this breaks every tag using it")
    doc = type(mind).io_kinds.__doc__ or ""
    assert "STABLE" in doc and "PROVISIONAL" in doc, "the contract must be stated where the client reads it"


def test_provisional_kinds_are_honest_about_their_gap(mind):
    """curve/skeleton have no tagged producer, and the docstring says so. If someone later tags a producer, this
    test fails LOUDLY -- prompting the doc to be corrected rather than quietly going stale."""
    src = mind.pipeline_map()["gaps"]["source_only"]
    assert set(src) == {"curve", "skeleton"}, (src, "source_only changed -- update io_kinds' PROVISIONAL note")


def test_resolve_capability_uri_says_it_is_uri_only(mind):
    """C16a: the obvious reading is "resolve any name", and an integrator read it that way. The BEHAVIOUR is
    unchanged (still [] for a method name) -- only the docstring stops implying otherwise."""
    assert mind.resolve_capability_uri("render_mesh") == []
    doc = type(mind).resolve_capability_uri.__doc__ or ""
    assert doc.strip().startswith("URI-ONLY"), "the caveat must lead, not hide in paragraph three"


def test_render_mesh_dtype_is_exact_and_default_off(mind):
    """C16b: saves the client a full-image copy per frame. It must not move a single pixel -- the cast happens at
    the EXIT, after the shading maths, so the default path is byte-identical to before."""
    import numpy as np

    cam = mind.camera(eye=(2, 2, 2), target=(0, 0, 0))
    a = mind.render_mesh(mind.mesh_box(), cam, width=12, height=12)
    b = mind.render_mesh(mind.mesh_box(), cam, width=12, height=12, dtype="float32")
    assert a.dtype == np.float64, "the default must not change"
    assert b.dtype == np.float32
    assert np.array_equal(a.astype("float32"), b), "dtype= must be a cast, not a different render"

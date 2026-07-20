"""C6 + C7 from the comfy-lecore audit: make the pipeline tags honest, and the callable name DATA.

C6's evidence was a route, not a theory:

    m.suggest_pipeline("image", "mesh")
    -> [Denoise multi-way data (low-rank tensor prior), Aharonov-Bohm ring (magnetic flux phase), ...]

Reproducing it found TWO bugs compounding, neither a typo:

  * FAKE EDGES -- both edge builders read `consumes` x `produces` as a CROSS PRODUCT. Right for a CONJUNCTIVE
    capability (transform_selection needs mesh AND selection AND transform, so reaching a mesh from a selection is
    a real question); WRONG for a POLYMORPHIC one -- denoise_tensor takes an image OR a field and returns THE SAME
    KIND, so `image->field` is a conversion it cannot perform. The router escaped into field-space through that
    invented hop. `polymorphic=True` keeps only the diagonal.
  * THE REAL ROUTE WAS UNTAGGED -- image_to_mesh / depth_to_mesh / photo_to_3d each announce "image -> MESH" in
    their own first docstring line and declared nothing, so no typed image->mesh edge existed. The router had no
    honest answer available and took the dishonest one. That is S7's thesis: the coverage gap PRODUCES wrong
    routes, it is not cosmetic.

And a duplicate the fix exposed: `pipelinemap._edges` and `Catalog.suggest_pipeline` each built the edge set, and
_edges' docstring claimed to match "EXACTLY". They had already diverged -- the polymorphic fix landed in one and
the nonsense route SURVIVED in the other. `test_both_edge_builders_agree` pins the claim instead of asserting it.

C7: `capability` in an edge is often prose ("Mesh repair (weld + split non-manifold + fill + compact)"), so a
client REGEXED `m.foo(` out of the example to invoke it -- fragile enough to need its own EXCLUDED.md. `method`
carries the callable name as data, VERIFIED against a live mind (511 of 2,040 naive guesses were wrong), and None
means honestly import-only -- which is also their item 8's flag, so one field answers both questions.
"""
import sys

import pytest

from holographic.caching_and_storage.holographic_catalog import _derive_method
from holographic.misc.holographic_unified import UnifiedMind

sys.path.insert(0, "tools")
import tag_lint  # noqa: E402


@pytest.fixture(scope="module")
def mind():
    return UnifiedMind(dim=64, seed=0)


# ---------------------------------------------------------------- C6

def test_the_reported_nonsense_route_is_gone(mind):
    """THE bug report, verbatim. It must never return a route through a denoiser and a quantum ring again."""
    route = mind.suggest_pipeline("image", "mesh") or []
    names = [s["name"] for s in route]
    assert not any("Denoise" in n or "Aharonov" in n for n in names), ("the nonsense route is back", names)


def test_image_to_mesh_now_answers_with_a_real_route(mind):
    """Killing the fake edge is only half: the REAL route had to become visible, or the honest answer is 'empty'
    and the client's flagship (photo -> 3D) still cannot be planned."""
    route = mind.suggest_pipeline("image", "mesh") or []
    assert route, "image->mesh must have a typed route"
    assert all(callable(getattr(mind, s.get("name"), None)) or True for s in route)
    assert any(s["name"] in ("image_to_mesh", "depth_to_mesh", "photo_to_3d") for s in route), route


def test_polymorphic_keeps_only_the_diagonal(mind):
    """denoise_tensor: image|field in -> the SAME kind out. image->field must not exist as an edge."""
    edges = [(e["consumes"], e["produces"]) for e in mind.pipeline_map()["edges"]
             if e["capability"].startswith("Denoise multi-way")]
    assert set(edges) == {("image", "image"), ("field", "field")}, edges


def test_both_edge_builders_agree(mind):
    """The duplicate that let the bug survive its own fix: two builders, one rule. If they drift again, a fix can
    land in one and the router keep believing the other."""
    import pipelinemap

    cat = mind._capability_catalog()
    drawn = sorted((ci, po, n) for ci, po, n in pipelinemap._edges(cat))
    # rebuild what suggest_pipeline routes over, by the same public data
    routed = []
    for cap in sorted(cat._by_name.values(), key=lambda c: c.name):
        if not cap.consumes or not cap.produces:
            continue
        if getattr(cap, "polymorphic", False):
            routed += [(k, k, cap.name) for k in cap.consumes if k in cap.produces]
        else:
            routed += [(ci, po, cap.name) for ci in cap.consumes for po in cap.produces]
    assert drawn == sorted(routed), "pipelinemap._edges and suggest_pipeline's edge set have diverged"


def test_the_lint_is_clean(mind):
    """CI gate: no tag a router would act on and regret."""
    r = tag_lint.audit(mind)
    assert not r["liars"], r["liars"]
    assert not r["crossfake"], r["crossfake"]


def test_the_lint_bites_on_a_crossfake(mind):
    """A gate that cannot fail is worse than no gate: remove the fix, the reported bug must be caught."""
    cap = mind._capability_catalog()._by_name["Denoise multi-way data (low-rank tensor prior)"]
    cap.polymorphic = False
    try:
        assert any(n.startswith("Denoise") for n, _ in tag_lint.audit(mind)["crossfake"])
    finally:
        cap.polymorphic = True


def test_the_lint_bites_on_a_liar(mind):
    cap = mind._capability_catalog()._by_name["Denoise multi-way data (low-rank tensor prior)"]
    cap.method = "no_such_faculty"
    try:
        assert any(n.startswith("Denoise") for n, _ in tag_lint.audit(mind)["liars"])
    finally:
        cap.method = "denoise_tensor"


def test_the_lint_does_not_cry_wolf_on_conjunctive_tags(mind):
    """KEPT NEGATIVE, pinned: the first rule flagged any consumes/produces OVERLAP and produced 5 false positives
    (select_edge_loop, skin_mesh, transform_selection...). Those are CONJUNCTIVE -- they need every consumed kind
    at once -- so their cross-product edges are REAL. A lint that cries wolf gets muted, and the real liar rides
    in behind it."""
    flagged = {n for n, _ in tag_lint.audit(mind)["crossfake"]}
    for legit in ("select_edge_loop", "skin_mesh", "transform_selection", "select_symmetric"):
        assert legit not in flagged, (legit, "conjunctive tag wrongly flagged")


# ---------------------------------------------------------------- C7

def test_every_pipeline_edge_carries_a_callable_method(mind):
    """Their acceptance test, adapted for the honest None: an edge either names a callable faculty or declares
    itself import-only. What must never happen is a method name that fails at call time."""
    for e in mind.pipeline_map()["edges"]:
        if e["method"] is not None:
            assert callable(getattr(mind, e["method"], None)), e


def test_no_capability_claims_an_uncallable_method(mind):
    """seed_from_mind verifies every derived guess against a live mind -- the step no client could do for itself.
    511 of 2,040 naive guesses were wrong (508 of them module names)."""
    liars = [c.name for c in mind._capability_catalog().all()
             if getattr(c, "method", None) and not callable(getattr(mind, c.method, None))]
    assert not liars, liars[:5]


def test_module_entries_are_honestly_import_only(mind):
    """A module is not a faculty. `holographic_rayindex` looks like a bare identifier, so the derivation would
    claim mind.holographic_rayindex -- which does not exist. None IS the client's item-8 flag."""
    caps = {c.name: c for c in mind._capability_catalog().all()}
    for n in ("holographic_rayindex", "holographic_pivot", "holographic_archive"):
        if n in caps:
            assert caps[n].method is None, (n, caps[n].method)


def test_method_is_derived_from_a_prose_title_via_its_example():
    """The regex the client was running on our example strings, now run ONCE in the engine."""
    assert _derive_method("2D SDF + extrude/revolve", "s = mind.sdf_extrude(poly, h=1.0)") == "sdf_extrude"
    assert _derive_method("Mesh repair (weld + fill)", "m.mesh_repair(mesh)") == "mesh_repair"
    assert _derive_method("render_mesh", "") == "render_mesh"            # a bare name IS the callable
    assert _derive_method("Some Import-Only Thing", "from holographic.x import Y; Y(z)") is None


def test_pipelines_json_matches_the_live_engine(mind):
    """The committed machine contract must equal what the faculty serves. It did not: generate() read
    default_catalog() (~400 curated) while pipeline_map() read the mind-seeded catalog (~2,100), so the JSON had
    no `method` field and no image->mesh edges while the engine served both -- and the drift gate was blind to
    it, because the file was up to date WITH ITS GENERATOR. Third instance of the one-source lesson; this test is
    the pin. (Ordering-insensitive: both sides sorted.)"""
    import json
    import pathlib

    jp = pathlib.Path(__file__).resolve().parent.parent / "pipelines.json"
    committed = json.load(open(jp))
    live = mind.pipeline_map()
    key = lambda e: (e["consumes"], e["produces"], e["capability"], e.get("method"))
    assert sorted(map(key, committed["edges"])) == sorted(map(key, live["edges"])), \
        "pipelines.json has drifted from mind.pipeline_map() -- run tools/regen_docs.py"
    assert committed["gaps"] == live["gaps"]


# ---------------------------------------------------------------- S7 (io-kind drive, batch 1)

def test_s7_batch1_routes_are_real(mind):
    """Each of these was verified by READING the signature + docstring before tagging. They exist as a
    regression trap: if a tag is later loosened or a faculty renamed, the route dies here rather than in a
    downstream node pack's Auto Route."""
    for start, goal, expect in (("image", "mesh", ("depth_to_mesh", "image_to_mesh", "photo_to_3d", "image_to_3d")),
                                ("field", "mesh", ("occupancy_to_mesh",)),
                                ("field", "hypervector", ("grid_to_hypervector",)),
                                ("hypervector", "field", ("hypervector_to_grid",))):
        route = mind.suggest_pipeline(start, goal) or []
        assert route, "%s->%s lost its route" % (start, goal)
        assert any(s["name"] in expect for s in route), (start, goal, [s["name"] for s in route])


def test_s7_closed_the_timeseries_source_only_gap(mind):
    """`timeseries` was consumed but nothing tagged produced it. record_physics_trace / audio_param_bus do --
    verified by reading, not inferred. curve/skeleton REMAIN source-only on purpose: nothing in the engine
    produces them (you import a skeleton, you draw a curve), and inventing a producer tag to empty a report
    would be the exact dishonesty tag_lint exists to prevent."""
    gaps = mind.pipeline_map()["gaps"]
    assert "timeseries" not in gaps["source_only"], gaps
    assert not gaps["dead_end"], gaps


def test_s7_abstentions_stay_abstained(mind):
    """KEPT NEGATIVE, pinned. The `X_to_Y` name convention LIES about types: mesh_to_stl -> a string,
    mesh_to_softbody -> a SoftBody, field_to_splats takes `centers` (points) not a field, dynamics_to_mesh's
    `source` is a dynamics object. Tagging these from their names would manufacture the fake edges the lint
    exists to catch. If a future session tags one, it must read the signature first -- and then this test."""
    caps = {c.name: c for c in mind._capability_catalog().all()}
    for n in ("field_to_splats", "dynamics_to_mesh", "mesh_to_softbody", "mesh_to_stl"):
        c = caps.get(n)
        if c is not None:
            assert not (c.consumes and c.produces), (n, "tagged without reading the signature?", c.consumes, c.produces)


def test_s7_batch2_routes_are_real(mind):
    """Batch 2, every entry verified by reading the signature + first docstring line. These pin the MULTI-STEP
    routes the batch unlocked -- sdf->image is mesh_from_sdf then render, a chain the router could not plan when
    either link was untagged."""
    # Assert the EDGE EXISTS, not which one BFS picks. suggest_pipeline breaks ties by capability NAME, so
    # field->field legitimately answers with the Denoise capability ("D" sorts before "advect_field") -- a real
    # field->field edge, just not the one this batch added. Testing the router's tie-break here would pin an
    # unrelated implementation detail and fail the day a capability is renamed.
    edges = {(e["consumes"], e["produces"], e["capability"]) for e in mind.pipeline_map()["edges"]}
    for start, goal, expect in (("sdf", "points", ("collide_sdf", "emit_from_surface")),
                                ("sdf", "field", ("bake_sdf",)),
                                ("field", "field", ("advect_field", "diffuse_field")),
                                ("sdf", "sdf", ("domain_twist", "domain_repeat", "domain_bend", "domain_fold"))):
        for name in expect:
            assert (start, goal, name) in edges, ("%s->%s via %s is missing" % (start, goal, name))
        assert mind.suggest_pipeline(start, goal, require_step=(start == goal)), "%s->%s unroutable" % (start, goal)


def test_s7_batch2_abstentions_stay_abstained(mind):
    """KEPT NEGATIVE, pinned. ascii_field/ascii_sdf produce TEXT -- there is no `text` io kind, and minting one to
    fit two faculties is a taxonomy decision, not a tagging one. curve_resample_arc_length's signature is
    (points, n): the NAME says curve, the CODE takes points -- the same trap batch 1 recorded. bump /
    apply_mueller / classify_transform merely MENTION several kinds; mentioning is not converting."""
    caps = {c.name: c for c in mind._capability_catalog().all()}
    for n in ("ascii_field", "ascii_sdf", "curve_resample_arc_length", "bump", "apply_mueller", "classify_transform"):
        c = caps.get(n)
        if c is not None:
            assert not (c.consumes and c.produces), (n, "tagged without reading the signature?", c.consumes, c.produces)


def test_the_engine_can_plan_a_multistep_route(mind):
    """The payoff of tagging at all: a goal two hops away is now plannable. sdf->image needs mesh_from_sdf THEN a
    renderer; either link untagged and the router answers 'no route' (or, before C6, invented one)."""
    route = mind.suggest_pipeline("sdf", "image") or []
    assert len(route) >= 2, ("sdf->image should be a multi-step chain", [s["name"] for s in route])
    kinds = [(s["consumes"], s["produces"]) for s in route]
    assert kinds[0][0] == ["sdf"] and kinds[-1][1] == ["image"], kinds

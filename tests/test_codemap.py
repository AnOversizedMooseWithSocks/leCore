"""Traps for the holographic code map -- and above all for its KEPT NEGATIVE.

The hypervector encoding LOST to token-set Jaccard on the same features. These tests pin that the losing
representation stayed opt-in, and that the comparison which produced that verdict still runs.
"""
import numpy as np
import pytest

import lecore
import holographic.io_and_interop.holographic_codemap as cm


def test_role_binding_actually_separates_roles():
    """If the same token in NAME and in CALLS produced the same vector, the binding would be decorative and
    this would be a bag of words with extra steps."""
    a = cm.encode_features({"NAME": ["mesh"], "DOC": [], "CALLS": [], "MODULE": [], "SHAPE": [], "BAND": []})
    b = cm.encode_features({"NAME": [], "DOC": [], "CALLS": ["mesh"], "MODULE": [], "SHAPE": [], "BAND": []})
    assert abs(float(a @ b)) < 0.2, "roles are not separated (cos=%.3f)" % float(a @ b)


def test_encoder_is_deterministic_and_unit_norm():
    f = {"NAME": ["catmull", "clark"], "DOC": ["subdivide"], "CALLS": ["bundle"],
         "MODULE": ["meshsubdiv"], "SHAPE": ["shape:abc"], "BAND": ["cc4"]}
    v1, v2 = cm.encode_features(f), cm.encode_features(f)
    assert np.array_equal(v1, v2), "encoder is not deterministic"
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-9


def test_the_see_reference_is_stripped_from_docstrings():
    """Faculty docstrings end with 'See holographic_x.y', which IS the answer in the delegate-retrieval
    evaluation. Indexing it would score a cheat rather than a retrieval."""
    import ast
    src = 'def f():\n    """Do a thing. See holographic_meshtools.reproject_uv."""\n    pass\n'
    node = ast.parse(src).body[0]
    assert "meshtools" not in cm.features(node, "m")["DOC"]
    assert "reproject" not in cm.features(node, "m")["DOC"]


def test_search_and_similar_return_plausible_neighbours():
    m = lecore.UnifiedMind(dim=128, seed=0)
    hits = [l for l, _s in m.code_search("subdivide a mesh", k=8)]
    assert hits and any("mesh" in h for h in hits), "mesh query returned nothing about meshes: %s" % hits[:3]
    sim = [l for l, _s in m.code_similar("catmull_clark", k=8)]
    # The query function itself must not be returned. Its DELEGATING FACULTY (mesh_catmull_clark) legitimately
    # may be -- that is the correct answer to "what else is like this", not a leak. An earlier version of this
    # assertion used a substring test and wrongly rejected it; it went unnoticed because the 15s watchdog was
    # skipping this test rather than running it.
    assert sim, "code_similar returned nothing"
    assert all(s.split(".")[-1] != "catmull_clark" for s in sim), "the query leaked into its own results"


def test_jaccard_is_the_default_and_the_vector_path_stays_opt_in():
    """THE KEPT NEGATIVE, pinned. The hypervector path measured recall@1 0.175 against Jaccard's 0.542 on
    identical features. If the default ever silently flips to the prettier representation, this fails."""
    import inspect
    for fn in (cm.search_source, cm.similar):
        assert inspect.signature(fn).parameters["method"].default == "jaccard", (
            "%s stopped defaulting to the measured winner" % fn.__name__)
    a = cm.similar("catmull_clark", k=5)
    b = cm.similar("catmull_clark", k=5, method="holographic")
    assert a and b, "one of the two retrieval paths returned nothing"
    assert [l for l, _ in a] != [l for l, _ in b], "the two methods are indistinguishable -- one is not running"


@pytest.mark.slow
def test_the_baseline_comparison_still_runs_and_still_says_the_same_thing():
    """The verdict must be re-derivable, not a number pasted in a docstring. Loose gates: this asserts the
    ORDERING of the three methods, which is the claim, not the exact figures on shared hardware."""
    r = cm.evaluate_retrieval(k=10, limit=40)
    h, j, rnd = r["holographic"], r["jaccard"], r["random"]
    assert h["recall@10"] > rnd["recall@10"] * 5, "the encoding carries no signal above random"
    assert j["recall@1"] > h["recall@1"], (
        "Jaccard no longer beats the hypervectors -- the kept negative may be stale, re-measure before "
        "changing the default (was 0.542 vs 0.175)")

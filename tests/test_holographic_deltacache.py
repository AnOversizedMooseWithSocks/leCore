"""C1/C4 -- the dependency-key layer, and the cache that is more correct than the thing it caches.

Part C: every triangle is THE canonical triangle plus a recognised delta chain. A computation runs on the canonical
once and its RESULT is transformed through the deltas; deltas the computation never reads never enter the key.

    policy        computes (400 tris)   speedup   bit-identical
    brute                400              1x          --
    read_set              50              8x         True      <- the material never enters the key
    equivariant            1            400x         FALSE     <- and the CACHE is the one that is right

THE KEPT NEGATIVE, and it is the interesting one. `area` is invariant under rotation in exact arithmetic. In
floating point, rotating a triangle and re-integrating accumulates ~1e-17 of round-off that the canonical
evaluation never incurs. So the cached answer differs from the brute answer at machine epsilon -- and **the brute
answer is the one carrying the error.** `max_abs_diff` is reported rather than a boolean, and `equivariant` is
opt-in, because this engine's constitution says a change at 1e-12 has still flipped a creature's trajectory.

C4: **a cache is only sound over a deterministic evaluator.** One drawing from a global RNG stream returns 0.4019
then 0.3188 for the same input. The cache stores the first and serves it forever while the uncached path keeps
drawing -- and the cache gets blamed. `DeltaCache` refuses it.
"""

import numpy as np
import pytest

from holographic.caching_and_storage.holographic_deltacache import (
    DeltaCache, cache_report, delta_id, evaluate_elements, family_verdict, is_deterministic)
from holographic.mesh_and_geometry.holographic_equivariance import apply_affine, area, centroid, sample_transform
from holographic.misc.holographic_determinism import hash_unit


CANON = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.9, 0.0]])
QUAD = np.random.default_rng(0).dirichlet(np.ones(3), size=64)     # a FIXED quadrature: deterministic by design


def _scene(n_shapes=64, n_mats=8, n=400, family="rotate"):
    deltas = [sample_transform(family, np.random.default_rng(i)) for i in range(n_shapes)]
    scene = [(s, m) for s in range(n_shapes) for m in range(n_mats)][:n]
    return scene, deltas


def _expensive(tri):
    """A real geometric integral -- reads GEOMETRY only, never the material. That is why the material delta can be
    dropped from the key, and it is a fact about the operator, not a convention."""
    return float(((QUAD @ tri) ** 2).sum(axis=1).mean() * area(tri))


def test_selftest_runs():
    from holographic.caching_and_storage import holographic_deltacache as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# the three policies
# ---------------------------------------------------------------------------------------------------------

def test_the_brute_baseline_recomputes_everything():
    scene, deltas = _scene()
    c = DeltaCache(_expensive, CANON, policy="brute")
    c.evaluate(scene, deltas)
    assert c.computes == 400 and c.hits == 0


def test_the_read_set_policy_drops_the_material_and_is_exact():
    scene, deltas = _scene()
    rep = cache_report(_expensive, CANON, scene, deltas, op_name="area", transform_family="rotate")
    assert rep["read_set"]["computes"] == 50                # 50 distinct shapes in the first 400 entries
    assert rep["read_set"]["bit_identical"] is True
    assert rep["read_set"]["max_abs_diff"] == 0.0
    assert rep["read_set"]["speedup"] == 8.0


def test_the_material_delta_genuinely_never_enters_the_key():
    # Same shape, eight materials -> ONE compute and seven hits. The operator does not read the material, so the
    # cache does not either.
    deltas = [sample_transform("rotate", np.random.default_rng(0))]
    scene = [(0, m) for m in range(8)]
    c = DeltaCache(_expensive, CANON, policy="read_set")
    out = c.evaluate(scene, deltas)
    assert c.computes == 1 and c.hits == 7
    assert len(set(out)) == 1                                # ... and every material got the same answer


def test_the_equivariant_policy_collapses_the_whole_scene_to_one_compute():
    scene, deltas = _scene()
    rep = cache_report(_expensive, CANON, scene, deltas, op_name="area", transform_family="rotate")
    assert rep["equivariant"]["computes"] == 1
    assert rep["equivariant"]["speedup"] == 400.0
    assert rep["equivariant"]["hit_rate"] > 0.99


def test_kept_negative_the_equivariant_path_is_not_bit_identical_and_the_cache_is_right():
    # Rotating a triangle and re-integrating accumulates round-off the canonical evaluation never incurs.
    scene, deltas = _scene()
    rep = cache_report(_expensive, CANON, scene, deltas, op_name="area", transform_family="rotate")
    assert rep["equivariant"]["bit_identical"] is False
    assert 0.0 < rep["equivariant"]["max_abs_diff"] < 1e-12

    # ... and the drift is in the BRUTE path: every rotated triangle's integral wobbles around the canonical value.
    canon_val = _expensive(CANON)
    rotated = [_expensive(apply_affine(A, b, CANON)) for (A, b) in deltas[:16]]
    assert len(set(rotated)) > 1                             # the brute values are not all equal ...
    assert max(abs(v - canon_val) for v in rotated) < 1e-12  # ... but they should be


def test_the_equivariant_policy_refuses_to_guess_the_verdict():
    scene, deltas = _scene()
    with pytest.raises(ValueError, match="MEASURED"):
        DeltaCache(_expensive, CANON, policy="equivariant")   # no op_name / transform_family


def test_the_equivariant_policy_does_not_fire_when_the_operator_is_not_invariant():
    # `centroid` is EQUIVARIANT under rotation, not invariant -- the delta may not be dropped, and the measurement
    # is what says so. This is the cell where a hand-written table would quietly lose correctness.
    scene, deltas = _scene(n_shapes=8, n=32)
    n_distinct = len({s for s, _m in scene})                 # 8 shapes x 8 materials, first 32 -> 4 shapes
    assert n_distinct == 4
    c = DeltaCache(centroid, CANON, policy="equivariant", op_name="centroid", transform_family="rotate")
    assert c._invariant is False
    c.evaluate(scene, deltas)
    assert c.computes == n_distinct                          # falls back to the read_set key, not to 1


def test_the_report_carries_every_policy_and_its_baseline():
    scene, deltas = _scene(n_shapes=8, n=32)
    rep = cache_report(_expensive, CANON, scene, deltas, op_name="area", transform_family="rotate")
    assert set(rep) == {"brute", "read_set", "equivariant"}
    for pol in rep:
        assert {"computes", "speedup", "max_abs_diff", "bit_identical"} <= set(rep[pol])
    assert rep["brute"]["speedup"] == 1.0


def test_the_report_omits_the_equivariant_policy_rather_than_guessing():
    scene, deltas = _scene(n_shapes=4, n=8)
    rep = cache_report(_expensive, CANON, scene, deltas)      # no names given
    assert "equivariant" not in rep
    assert rep["read_set"]["bit_identical"] is True


# ---------------------------------------------------------------------------------------------------------
# C4 -- the cache is only sound over a deterministic evaluator
# ---------------------------------------------------------------------------------------------------------

def test_a_global_rng_evaluator_is_not_deterministic_and_is_refused():
    stream = np.random.default_rng(0)

    def sampled_global(tri):
        return float(((stream.dirichlet(np.ones(3), size=32) @ tri) ** 2).sum(axis=1).mean())

    assert not is_deterministic(sampled_global, CANON)
    with pytest.raises(ValueError, match="non-deterministic"):
        DeltaCache(sampled_global, CANON)


def test_a_coordinate_keyed_sampler_is_deterministic_and_accepted():
    def sampled_keyed(tri):
        seed = int(hash_unit(*np.round(np.asarray(tri).ravel(), 12)) * (2 ** 32))
        b = np.random.default_rng(seed).dirichlet(np.ones(3), size=32)
        return float(((b @ tri) ** 2).sum(axis=1).mean())

    assert is_deterministic(sampled_keyed, CANON)
    DeltaCache(sampled_keyed, CANON)                          # accepted

    # ... and it is still a FUNCTION of its input: a different triangle gives a different answer
    moved = CANON.copy()
    moved[2, 1] += 1e-6
    assert sampled_keyed(CANON) != sampled_keyed(moved)


def test_the_unsoundness_is_a_disagreement_the_cache_gets_blamed_for():
    # This is what the failure LOOKS like, and why C4 exists. The cache serves its first draw forever.
    stream = np.random.default_rng(0)

    def sampled_global(tri):
        return float(stream.normal())

    a, b = sampled_global(CANON), sampled_global(CANON)
    assert a != b                                             # the uncached path keeps drawing
    cached = a
    assert cached != b                                        # ... and disagrees with the cache, which never moved


# ---------------------------------------------------------------------------------------------------------
# the delta id
# ---------------------------------------------------------------------------------------------------------

def test_delta_id_is_a_stable_content_hash():
    A, b = sample_transform("rotate", np.random.default_rng(0))
    A2, b2 = sample_transform("rotate", np.random.default_rng(1))
    assert delta_id(A, b) == delta_id(A, b)
    assert delta_id(A, b) != delta_id(A2, b2)
    assert len(delta_id(A, b)) == 16


def test_delta_id_ignores_differences_below_the_evaluators_precision():
    # Two deltas that differ at 1e-15 are the same delta. Refusing to say so makes every key unique and the cache
    # useless -- a correctness-shaped decision that is really a usefulness one, and it is stated.
    A, b = sample_transform("rotate", np.random.default_rng(0))
    assert delta_id(A, b) == delta_id(A + 1e-15, b)
    assert delta_id(A, b) != delta_id(A + 1e-6, b)


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    scene, deltas = _scene(n_shapes=16, n_mats=8, n=128)
    rep = m.delta_cache_report(area, CANON, scene, deltas, op_name="area", transform_family="rotate")
    assert rep["brute"]["computes"] == 128
    assert rep["read_set"]["computes"] == 16 and rep["read_set"]["bit_identical"] is True
    assert rep["equivariant"]["computes"] == 1

    c = m.delta_cache(_expensive, CANON, policy="read_set")
    c.evaluate(scene, deltas)
    assert c.stats()["hit_rate"] > 0.8

    assert m.is_deterministic(_expensive, CANON) is True

    assert "Dependency-keyed" in str(m.find_capability("cache a computation by its dependencies")[:3])


# ===========================================================================================
# PART C, END TO END -- the join a reachability audit found missing.
# C3 (recognise) -> C2 (equivariance table) -> C1 (cache key), on elements with NO shape ids.
# ===========================================================================================

def _raw_scene(n_bases=5, per_base=40, seed=0):
    """RAW triangles. The caller does not know which are the same thing modulo a delta -- that is the point."""
    rng = np.random.default_rng(seed)
    bases = [rng.normal(size=(3, 3)) for _ in range(n_bases)]
    els = [(rng.uniform(0.5, 2.0) * b) @ sample_transform("rotate", rng)[0].T + rng.normal(size=3)
           for b in bases for _ in range(per_base)]
    return bases, els


def test_a_composite_familys_verdict_is_the_weakest_of_its_parts():
    # `rigid` cannot see a scale, so `area` is invariant under it. The moment `uniform_scale` joins the family,
    # `area` becomes equivariant. Taking the STRONGEST part would reuse a value the delta actually changed.
    assert family_verdict("area", "rigid") == "invariant"
    assert family_verdict("area", "similarity") == "equivariant"
    assert family_verdict("centroid", "rigid") == "equivariant"
    assert family_verdict("max_x", "rigid") == "recompute"
    assert family_verdict("max_x", "similarity") == "recompute"

    with pytest.raises(ValueError, match="unknown family"):
        family_verdict("area", "projective")


def test_evaluate_elements_derives_the_classes_the_caller_never_knew():
    bases, els = _raw_scene()
    truth = np.array([area(t) for t in els])

    vals, st = evaluate_elements(els, area, "area", family="similarity")
    assert st["classes"] == len(bases)                       # recognition found the reuse
    assert st["computes"] == len(bases)                      # ... and the cache used it: 200 -> 5
    assert st["verdict"] == "equivariant"
    assert np.abs(np.array(vals) - truth).max() < 1e-12      # exact, via the registered law


def test_kept_negative_recognition_alone_is_not_enough():
    # Reusing the canonical's value directly under `similarity` is WRONG by 8.54: a uniform scale moves the area.
    # The FAMILY decides whether you may reuse the value, transform it, or must recompute.
    from holographic.mesh_and_geometry.holographic_canonmesh import recognize

    _bases, els = _raw_scene()
    truth = np.array([area(t) for t in els])
    rec = recognize(els, "similarity")
    naive = np.array([area(rec["canonicals"][k]) for k, _d in rec["instances"]])
    assert np.abs(naive - truth).max() > 1.0                 # measured 8.54

    vals, _st = evaluate_elements(els, area, "area", family="similarity")
    assert np.abs(np.array(vals) - truth).max() < 1e-12      # the law is exact where the reuse is not


def test_a_family_that_under_fits_finds_no_reuse_and_says_so():
    _bases, els = _raw_scene()
    _vals, st = evaluate_elements(els, area, "area", family="rigid")
    assert st["classes"] == len(els)                          # rigid cannot match a scaled copy
    assert st["computes"] == len(els)
    assert st["verdict"] == "invariant"                       # ... and the verdict is still honest


def test_recompute_finds_classes_and_still_pays_full_price():
    from holographic.mesh_and_geometry.holographic_equivariance import max_x

    _bases, els = _raw_scene()
    truth = np.array([max_x(t) for t in els])
    vals, st = evaluate_elements(els, max_x, "max_x", family="similarity")
    assert st["classes"] == 5                                 # the classes are real ...
    assert st["computes"] == len(els)                         # ... and buy nothing: there is no law
    assert st["verdict"] == "recompute"
    assert np.abs(np.array(vals) - truth).max() < 1e-12       # and the answer is still right


def test_the_affine_familys_rectangular_delta_is_refused_not_crashed_on():
    # canonmesh whitens within the element's own hull, so its `A` is a (3, rank) embedding -- rank 2 for a triangle,
    # because every triangle is planar. The equivariance laws take det(A) and inv(A). Feeding one the other raised
    # LinAlgError from deep inside `_law_area`: a crash where a refusal belongs.
    _bases, els = _raw_scene(n_bases=2, per_base=4)
    with pytest.raises(ValueError, match="square Jacobian"):
        evaluate_elements(els, area, "area", family="affine")


def test_evaluate_elements_refuses_a_non_deterministic_evaluator():
    _bases, els = _raw_scene(n_bases=2, per_base=2)
    stream = np.random.default_rng(0)
    with pytest.raises(ValueError, match="non-deterministic"):
        evaluate_elements(els, lambda t: float(stream.normal()), "area", family="similarity")


def test_the_join_is_wired_to_the_mind():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    _bases, els = _raw_scene(n_bases=4, per_base=10)
    vals, st = m.evaluate_elements(els, area, "area", family="similarity")
    assert st["classes"] == 4 and st["computes"] == 4
    assert np.abs(np.array(vals) - np.array([area(t) for t in els])).max() < 1e-12
    assert m.family_verdict("area", "rigid") == "invariant"

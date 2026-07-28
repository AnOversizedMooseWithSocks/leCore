"""Regression traps for the declare() resolution ladder (work plan item 2.1).

The primary metric is FALSE-ACTION RATE ON A NO-TOOL SET -- the number the reference system (NOOA) does
not publish and the number this architecture exists to drive to zero. It is pinned here as a committed,
seeded fixture so a later session re-runs the identical comparison rather than a similar one.

The no-tool arm is built the way the plan requires: from the catalog's OWN vocabulary at MATCHED TOKEN
COUNT, so the two arms differ in whether a tool exists, not in vocabulary, length or phrasing. See
test_the_no_tool_arm_is_not_an_easier_arm for the honest limit of that construction.
"""
import random

import numpy as np
import pytest

import lecore
from holographic.agents_and_reasoning.holographic_declare import (EXACT, INHERITS, NONE, Ladder,
                                                                  finite_score)
from holographic.caching_and_storage.holographic_catalog import _tokens, default_catalog


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


@pytest.fixture(scope="module")
def arms():
    """(has_tool, no_tool) -- the committed seeded fixture."""
    cat = default_catalog()
    pairs = [str(a) for c in cat.all() for a in (getattr(c, "aliases", ()) or [])
             if getattr(c, "method", None) and len(str(a).split()) >= 4]
    pairs = sorted(set(pairs))
    random.Random(0).shuffle(pairs)
    has = pairs[:60]
    vocab = sorted({t for c in cat.all() for t in _tokens(c.does)})
    rng = random.Random(1)
    no = [" ".join(rng.sample(vocab, len(_tokens(a)) or 4)) for a in has]
    return has, no


# --------------------------------------------------------------------------------------
# The NaN guard -- a MEASURED defect, not a hypothetical.
# --------------------------------------------------------------------------------------

def test_finite_score_rejects_everything_that_could_win_a_gate():
    assert finite_score(1.0) and finite_score(0) and finite_score(-3.5)
    for bad in (float("nan"), float("inf"), float("-inf"), None, "x", [1], {}):
        assert not finite_score(bad), "finite_score accepted %r" % (bad,)


def test_the_underlying_nan_defect_still_exists_so_the_guard_is_still_needed():
    # If this ever starts passing, argmax_tiebreak was fixed and the ladder's guard becomes belt-and-
    # braces rather than load-bearing -- worth knowing, and worth updating the docstrings for.
    from holographic.misc.holographic_determinism import argmax_tiebreak
    assert argmax_tiebreak([0.1, float("nan"), 0.9]) == 1, \
        "argmax_tiebreak no longer returns the NaN index; update holographic_declare's rationale"


# --------------------------------------------------------------------------------------
# THE PRIMARY METRIC.
# --------------------------------------------------------------------------------------

def test_false_action_rate_on_the_no_tool_set_is_zero(mind, arms):
    # THE HEADLINE. A request with no capability behind it must be REFUSED, not answered. A fluent filler
    # always returns something; refusing is the feature this whole ladder buys.
    _, no_tool = arms
    answered = sum(mind.declare_explain(q).ok for q in no_tool)
    assert answered == 0, "false-action rate %.1f%% (%d/%d)" % (
        100 * answered / len(no_tool), answered, len(no_tool))


def test_the_has_tool_arm_still_resolves(mind, arms):
    # The other half: an abstention rate of 100% is trivially achievable by refusing everything.
    has_tool, _ = arms
    answered = sum(mind.declare_explain(q).ok for q in has_tool)
    assert answered / len(has_tool) > 0.9, "only %d/%d resolved" % (answered, len(has_tool))


def test_the_no_tool_arm_is_not_an_easier_arm(arms):
    # HONESTY ON THE FIXTURE ITSELF. The arms must match on token count and draw from the same vocabulary,
    # or the ladder is discriminating length or word choice rather than tool presence.
    has_tool, no_tool = arms
    assert len(has_tool) == len(no_tool)
    for a, b in zip(has_tool, no_tool):
        assert abs(len(_tokens(a)) - len(_tokens(b))) <= 1
    # KEPT LIMIT: word salad is still SEMANTICALLY easier to refuse than a real description whose tool was
    # removed from the index. 0% here is a real result on the easier construction, not the hardest one.


# --------------------------------------------------------------------------------------
# Provenance, the descent log, and the cap.
# --------------------------------------------------------------------------------------

def test_every_result_carries_full_provenance(mind):
    for request in ("smooth a bumpy mesh", "purple monkey dishwasher"):
        d = mind.declare_explain(request).as_dict()
        for field in ("ok", "rung", "mechanism", "exactness", "reversibility",
                      "confidence", "why", "descent"):
            assert field in d, "%r is missing provenance field %r" % (request, field)


def test_a_refusal_records_why_every_rung_declined(mind):
    res = mind.declare_explain("purple monkey dishwasher")
    assert res.ok is False and res.exactness == NONE
    assert len(res.descent) == 4
    assert all(not r.answered and r.why for r in res.descent), \
        "a decline without a reason is not a decline"


def test_rung_zero_answers_a_real_request_and_names_the_faculty(mind):
    res = mind.declare_explain("smooth a bumpy mesh")
    assert res.ok and res.rung.index == 0
    assert res.exactness == INHERITS, "rung 0 cannot claim more than the faculty it dispatched to"
    assert "mesh_smooth" in res.why


def test_hits_tuples_are_unpacked_not_misread(mind):
    # A REGRESSION THAT ALREADY HAPPENED. route_or_abstain returns (Capability, score) tuples while
    # find_capability returns bare Capabilities; reading the tuple as a capability made rung 0 decline
    # with "import-only" on a perfectly callable faculty -- a wrong REASON, which is worse than a wrong
    # answer because it sends the next reader to the wrong place.
    res = mind.declare_explain("smooth a bumpy mesh")
    assert "import-only" not in res.why


def test_max_rung_is_a_hard_cap_and_skips_are_logged(mind):
    res = Ladder(mind, max_rung=0).resolve("purple monkey dishwasher")
    skipped = [r for r in res.descent if "above max_rung" in r.why]
    assert len(skipped) == 3, "rungs above the cap must be logged, not silently missing"


def test_rung_two_claims_exact_only_when_it_verified_by_execution(mind):
    from holographic.agents_and_reasoning.holographic_ai import permute, random_vector
    a = random_vector(256, np.random.default_rng(0))
    res = Ladder(mind).resolve("permute", args={"input_vec": a, "output_vec": permute(a, 1)})
    if res.ok and res.rung.index == 2:
        assert res.exactness == EXACT
    for r in res.descent:
        if r.index == 2 and not r.answered:
            assert r.why, "rung 2 declined without a reason"


def test_dry_run_executes_nothing_but_still_explains(mind):
    res = mind.declare_explain("smooth a bumpy mesh")
    assert res.ok and res.value is None, "a dry run must not carry a computed value"
    assert res.why.startswith("would "), "a dry run must say WOULD, not report a completed action"


def test_the_ladder_is_deterministic(mind):
    a = mind.declare_explain("smooth a bumpy mesh").as_dict()
    b = mind.declare_explain("smooth a bumpy mesh").as_dict()
    assert [r["why"] for r in a["descent"]] == [r["why"] for r in b["descent"]]
    assert a["confidence"] == b["confidence"]


def test_the_decorator_form_returns_a_resolution_not_a_bare_value(mind):
    @mind.declares
    def smooth_a_bumpy_mesh(mesh=None):
        """smooth a bumpy mesh"""
        ...

    assert smooth_a_bumpy_mesh.request == "smooth a bumpy mesh"
    assert hasattr(smooth_a_bumpy_mesh, "declared")
    res = smooth_a_bumpy_mesh()
    assert hasattr(res, "ok") and hasattr(res, "descent"), \
        "the wrapper must return provenance, or a caller can use an answer without knowing its class"


def test_the_ladder_is_discoverable(mind):
    for query in ("declare a method and let the engine fill it in",
                  "try cheap deterministic ways before calling a model",
                  "which rung answered my request"):
        assert "Declare a body" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the ladder" % query


def test_there_is_no_fusion_rung_and_the_reason_is_recorded():
    """KEPT NEGATIVE, PINNED (work plan item 3.1). A fusion rung was gated before building and failed:
    0/60 real resolutions produce a chain for it to act on, and only 3.2% of tagged io edges are
    image-kind. If someone adds a rung 4 for fusion, this fails and sends them to the measurement."""
    import inspect
    from holographic.agents_and_reasoning import holographic_declare as hd
    doc = hd.__doc__ or ""
    assert "THERE IS NO FUSION RUNG" in doc, "the fusion negative left the docstring"
    src = inspect.getsource(hd.Ladder.resolve)
    assert "_rung4" not in src, "a rung 4 appeared; re-run the fusion gate before adding one"


# --------------------------------------------------------------------------------------
# Resolution cache (work plan items 3.2 + 3.3, which interact and therefore land together).
# --------------------------------------------------------------------------------------

def test_the_cache_is_optional_and_defaults_off(mind):
    from holographic.agents_and_reasoning.holographic_declare import Ladder
    assert Ladder(mind).cache is None
    assert Ladder(mind).resolve("smooth a bumpy mesh", dry_run=True).ok


def test_an_identical_request_is_served_from_cache(mind):
    cache = {}
    a = mind.declare("smooth a bumpy mesh", cache=cache)
    b = mind.declare("smooth a bumpy mesh", cache=cache)
    assert len(cache) == 1 and a is b


def test_a_nondeterministic_resolution_is_never_cached(mind):
    """THE HARD RULE from the machine model's content-addressed tier, enforced in code because a docstring
    cannot refuse a write. Rungs 0-3 never produce a NONDETERMINISTIC answer, so this guard is written
    BEFORE rungs 6-7 exist -- which is the only time it is cheap to write."""
    from holographic.agents_and_reasoning.holographic_declare import Ladder, Resolution, Rung

    cache = {}
    lad = Ladder(mind, cache=cache)
    lad._store("k", Resolution(True, "v", Rung(0, "model", True, "guessed", "NONDETERMINISTIC"), []))
    assert cache == {}, "a nondeterministic resolution was cached"
    lad._store("k", Resolution(True, "v", Rung(0, "synth", True, "verified", "EXACT"), []))
    assert len(cache) == 1, "a deterministic resolution was refused"


def test_different_requests_get_different_keys(mind):
    cache = {}
    mind.declare("smooth a bumpy mesh", cache=cache)
    mind.declare("purple monkey dishwasher", cache=cache)
    assert len(cache) == 2


def test_config_is_part_of_the_key(mind):
    # A resolution computed at max_rung=0 is not the same answer as one at max_rung=5, so it must not be
    # served for it. Keying on the request alone would silently return the wrong descent.
    from holographic.agents_and_reasoning.holographic_declare import content_key
    assert content_key("q", None, 5, 0.8, 0) != content_key("q", None, 0, 0.8, 0)
    assert content_key("q", None, 5, 0.8, 0) != content_key("q", None, 5, 0.9, 0)


def test_array_args_are_hashed_by_content_not_identity():
    # Args are digested, never held: a live object in cached args once crashed a worker AFTER its job had
    # succeeded. Equal arrays must key the same; different ones must not.
    import numpy as np
    from holographic.agents_and_reasoning.holographic_declare import content_key
    a, b = np.arange(6.0), np.arange(6.0)
    assert content_key("q", {"v": a}, 5, 0.8, 0) == content_key("q", {"v": b}, 5, 0.8, 0)
    assert content_key("q", {"v": a}, 5, 0.8, 0) != content_key("q", {"v": a + 1}, 5, 0.8, 0)


def test_a_dry_run_is_not_cached(mind):
    # A dry run carries no value and says "would", so serving it for a real call would be wrong.
    cache = {}
    mind.declare_explain("smooth a bumpy mesh")
    mind.declare("smooth a bumpy mesh", dry_run=True, cache=cache)
    assert cache == {}


def test_a_broken_cache_cannot_take_down_a_resolution(mind):
    class Exploding(dict):
        def get(self, *a, **k):
            raise RuntimeError("cache offline")

        def __setitem__(self, *a):
            raise RuntimeError("cache offline")

    assert mind.declare("smooth a bumpy mesh", cache=Exploding()) is not None


def test_the_wall_verdict_is_not_wired_as_an_escalation_trigger():
    """KEPT NEGATIVE, PINNED (work plan item 4.4). Gated before building: 'wall' fired on 5/5 unserviceable
    AND 5/5 serviceable tasks -- discrimination zero. It is the CORRECT verdict for both, which is exactly
    why it cannot trigger anything: a signal that is always true partitions nothing."""
    import inspect
    from holographic.agents_and_reasoning import holographic_declare as hd
    assert "'wall' IS NOT AN ESCALATION TRIGGER" in (hd.__doc__ or ""), \
        "the wall negative left the docstring"
    src = inspect.getsource(hd.Ladder)
    assert "diagnose_scaling" not in src, \
        "diagnose_scaling was wired into the ladder; re-run the 4.4 gate before doing that"


def test_break_even_n_depends_on_the_baseline_so_the_pre_gate_cannot_exist(mind):
    """KEPT NEGATIVE, PINNED (work plan item 3.3, second half). A pre-gate was proposed that would skip
    placement WITHOUT measuring, on the claim that break_even_n is 'independent of the baseline'. Measured,
    it spans 1.13 to infinity across baselines -- so the gate would need the very number it exists to avoid
    measuring. If this ever becomes flat, the pre-gate becomes possible and the negative should be revisited."""
    sheet = mind.machine_spec_sheet()
    values = [mind.machine_place_unit("t2_baked_grid", b, 100, sheet=sheet)["break_even_n"]
              for b in (1000.0, 10000.0, 100000.0)]
    finite = [v for v in values if v == v and v != float("inf")]
    assert len(finite) >= 2
    assert max(finite) > 10 * min(finite), \
        "break_even_n became baseline-insensitive (%r); the 3.3 pre-gate may now be viable" % values

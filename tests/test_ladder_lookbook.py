"""Regression traps for the ladder's look-book (work plan item 2.4).

Two things are pinned. First the SCOPE, because the obvious reading of "a system that ranks N rungs and
reports the winner is a battery" does not describe this ladder and acting on it would have produced a
meaningless correction. Second the P-VALUE, because the ledger feeds an FDR correction and an
anti-conservative p there is worse than no ledger at all.
"""
import numpy as np
import pytest

import lecore
from holographic.agents_and_reasoning.holographic_declare import Ladder


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


# --------------------------------------------------------------------------------------
# The p-value, which is the part that has to be right.
# --------------------------------------------------------------------------------------

def test_route_or_abstain_reports_an_empirical_p(mind):
    for query in ("smooth a bumpy mesh", "purple monkey dishwasher"):
        v = mind.route_or_abstain(query)
        assert "p" in v and v["p"] is not None
        assert 0.0 < v["p"] <= 1.0, "p out of range: %r" % v["p"]


def test_the_p_is_counted_not_normal_approximated(mind):
    # THE CONTRACT THAT MATTERS. The null draws find_scored(fake, k=1), so it is a distribution of MAXIMA
    # -- right-skewed. A normal approximation 1-Phi(z) understates that tail and yields an
    # ANTI-CONSERVATIVE p, which is the wrong direction for anything feeding an FDR correction. An
    # empirical p is granular: with n_null draws it can only take values k/(n_null+1).
    v = mind.route_or_abstain("smooth a bumpy mesh", n_null=64)
    granularity = 1.0 / 65.0
    ratio = v["p"] / granularity
    assert abs(ratio - round(ratio)) < 1e-9, \
        "p=%r is not a multiple of 1/(n_null+1); it looks parametric" % v["p"]


def test_the_p_uses_the_plus_one_plug(mind):
    # p must never be exactly 0: n_null draws cannot support more evidence than that.
    for query in ("smooth a bumpy mesh", "nearest stored item", "render a mesh to an image"):
        assert mind.route_or_abstain(query)["p"] > 0.0


def test_a_nonsense_query_gets_a_high_p_and_abstains(mind):
    # AN OVER-FITTED ASSERTION, CORRECTED. This originally demanded p == 1.0 exactly, and it broke the
    # moment seven capabilities were added in the same session: the null draws from the CATALOG'S OWN
    # vocabulary, so growing the catalog legitimately moves every p. The contract is "nonsense does not
    # look significant and does not route", not a snapshot of one catalog state -- and a test that pins a
    # number the system is allowed to change is a test that will be edited rather than believed.
    verdict = mind.route_or_abstain("purple monkey dishwasher")
    assert verdict["abstain"] is True
    assert verdict["p"] > 0.05, "nonsense now looks significant at p=%.4f" % verdict["p"]


def test_the_z_floor_is_not_a_significance_test(mind):
    # AN HONEST CALIBRATION FACT, pinned because it is surprising and easy to forget: a query that CLEARS
    # the router's z_min=0.8 floor can still sit at p ~ 0.11. The floor is a practical routing threshold,
    # not a 0.05-level claim, and reading it as one would overstate what a successful route proves.
    v = mind.route_or_abstain("smooth a bumpy mesh")
    assert v["abstain"] is False
    assert v["p"] > 0.05, "the z floor now coincides with p<0.05; update the calibration note"


# --------------------------------------------------------------------------------------
# The scope of the correction -- the part the plan got wrong.
# --------------------------------------------------------------------------------------

def test_the_ladder_is_not_an_n_look_battery_over_its_rungs(mind):
    # It walks rungs IN ORDER and stops at the first pass; the declines are STRUCTURAL, not statistical.
    # Correcting for "4 looks" would be nonsense, so only rung 0 -- the only rung with a score at all --
    # is ever recorded.
    led = mind.selection_ledger()
    Ladder(mind, ledger=led).resolve("smooth a bumpy mesh", dry_run=True)
    rows = led.correct()["rows"]
    assert len(rows) == 1, "one resolution recorded %d looks; only rung 0 has a score" % len(rows)


def test_rung_zero_selection_is_already_multiplicity_corrected(mind):
    # The catalog-wide argmax is NOT an uncorrected look: route_or_abstain's null takes the TOP-1 score of
    # each scrambled query, so it is a distribution of maxima and the selection is priced in by
    # construction. If this ever changed to a mean-of-all-scores null, the ledger's scope note is wrong.
    import inspect
    from holographic.caching_and_storage import holographic_catalog as hc
    src = inspect.getsource(hc.Catalog.route_or_abstain)
    assert "find_scored(fake, k=1)" in src, \
        "the null no longer draws maxima; re-read the multiplicity reasoning in holographic_declare"


def test_failed_looks_are_recorded_too(mind):
    # A ledger that only sees survivors corrects for nothing.
    led = mind.selection_ledger()
    lad = Ladder(mind, ledger=led)
    for request in ("smooth a bumpy mesh", "purple monkey dishwasher", "flurb granp zzz qqq"):
        lad.resolve(request, dry_run=True)
    rows = led.correct()["rows"]
    assert len(rows) == 3
    assert any(r["p"] == 1.0 for r in rows), "the abstentions were not booked"


def test_the_ledger_is_optional_and_defaults_off(mind):
    assert Ladder(mind).ledger is None
    res = Ladder(mind).resolve("smooth a bumpy mesh", dry_run=True)
    assert res.ok, "adding the ledger parameter changed default behaviour"


def test_a_broken_ledger_cannot_take_down_a_resolution(mind):
    class Exploding:
        def record(self, *a, **k):
            raise RuntimeError("disk full")

    res = Ladder(mind, ledger=Exploding()).resolve("smooth a bumpy mesh", dry_run=True)
    assert res.ok, "a bookkeeping failure broke the answer"


def test_multiplicity_actually_bites(mind):
    # The point of the whole exercise: ask enough times and a nominally-good look stops surviving. Uses
    # the ledger directly so the assertion is about BH, not about routing scores.
    led = mind.selection_ledger()
    led.record("real", 0.01, family="declare")
    solo = led.correct()["rows"][0]["passed"]
    for i in range(40):
        led.record("probe%d" % i, 0.5, family="declare")
    crowded = [r for r in led.correct()["rows"] if r["name"] == "real"][0]["passed"]
    assert solo and not crowded, "the correction did not respond to added looks (%s -> %s)" % (solo, crowded)

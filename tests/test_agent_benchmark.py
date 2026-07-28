"""Regression traps for the agent-socket benchmark (work plan item 6.1).

The primary metric is PRE-REGISTERED as false-action rate on a no-tool set. The instrument that makes it
worth anything is `catalog_without`, so that gets tested hardest: if removal silently failed, every no-tool
measurement would be a has-tool measurement wearing a different label.
"""
import pytest

import lecore
from holographic.agents_and_reasoning.holographic_agentbench import (build_fixture, catalog_without,
                                                                     run_benchmark)
from holographic.caching_and_storage.holographic_catalog import _tokens, default_catalog


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


# --------------------------------------------------------------------------------------
# The instrument.
# --------------------------------------------------------------------------------------

def test_removal_actually_removes():
    full = default_catalog()
    victim = next(c.name for c in full.all() if getattr(c, "method", None))
    reduced = catalog_without([victim])
    assert len(reduced.all()) == len(full.all()) - 1
    assert all(c.name != victim for c in reduced.all())


def test_removal_does_not_mutate_the_original():
    # A benchmark that damages the system it measures is measuring something else by the second run.
    full = default_catalog()
    victim = next(c.name for c in full.all() if getattr(c, "method", None))
    before = len(full.all())
    catalog_without([victim])
    assert len(default_catalog().all()) == before
    assert default_catalog().get(victim) is not None


def test_a_removed_capabilitys_own_alias_no_longer_finds_it():
    # THE PROPERTY THE WHOLE NO-TOOL ARM RESTS ON. If this fails, the arm is not measuring absence.
    full = default_catalog()
    for cap in full.all():
        aliases = [str(a) for a in (cap.aliases or []) if len(_tokens(a)) >= 4]
        if getattr(cap, "method", None) and aliases:
            reduced = catalog_without([cap.name])
            hits = [getattr(h, "name", "") for h in reduced.find_capability(aliases[0])[:3]]
            assert cap.name not in hits, "%r still surfaced after removal" % cap.name
            return
    pytest.skip("no capability with a long alias to test")


def test_distractors_are_left_in_place():
    # Removing the answer must not remove its neighbours -- an empty index would refuse trivially and the
    # 0% would mean nothing.
    full = default_catalog()
    victim = next(c.name for c in full.all() if getattr(c, "method", None))
    assert len(catalog_without([victim]).all()) > 100


# --------------------------------------------------------------------------------------
# The fixture and the metric.
# --------------------------------------------------------------------------------------

def test_the_fixture_is_seeded_and_the_arms_are_disjoint():
    a1, b1 = build_fixture(n_has=20, n_no=10)
    a2, b2 = build_fixture(n_has=20, n_no=10)
    assert a1 == a2 and b1 == b2, "the fixture is not reproducible"
    assert not (set(t for t, _ in a1) & set(t for t, _ in b1)), "a task appears in both arms"


def test_both_arms_are_drawn_by_the_same_rule():
    # The arms must differ in ONE respect -- whether the capability is in the index. Same pool, same rule.
    has, no = build_fixture(n_has=20, n_no=10)
    for task, _ in list(has) + list(no):
        assert len(_tokens(task)) >= 4


@pytest.mark.slow                       # rebuilds the catalog once per no-tool task
def test_the_false_action_rate_is_zero_on_the_removal_set(mind):
    report = run_benchmark(mind, n_has=20, n_no=10, seed=0)
    assert report["false_action_rate"] == 0.0, \
        "false-action rate %.1f%% (%d/%d)" % (100 * report["false_action_rate"],
                                              report["false_actions"], report["n_no"])


@pytest.mark.slow
def test_the_has_tool_arm_resolves(mind):
    # Refusing everything would trivially satisfy the metric above.
    report = run_benchmark(mind, n_has=20, n_no=4, seed=0)
    assert report["resolution_rate"] > 0.9


def test_the_deterministic_arm_reaches_no_model(mind):
    assert run_benchmark(mind, n_has=4, n_no=2)["model_calls"] == 0


def test_run_to_run_variance_is_zero(mind):
    # The plan requires this to read exactly 0 at max_rung=5. It is the guarantee that makes every other
    # number here comparable across sessions.
    assert run_benchmark(mind, n_has=6, n_no=3, seed=0) == run_benchmark(mind, n_has=6, n_no=3, seed=0)


# --------------------------------------------------------------------------------------
# The negative the plan requires be published.
# --------------------------------------------------------------------------------------

def test_the_rung_distribution_negative_is_recorded(mind):
    """THE PLAN'S REQUIRED NEGATIVE. Its bar: if rungs 1-5 fire on under ~20% of realistic bodies, say so in
    bold. Measured: rungs 1-5 fired on 0/60. Rung 0 answered everything.

    What that does NOT mean: the plan frames the failure as "ceremony around an LLM call", and no model was
    reached at all -- the cheapest, fully deterministic rung sufficed. What it DOES mean: rungs 1-3 are
    unexercised, so their gates are unproven on real traffic. The fixture is free-text-to-faculty requests,
    which is exactly rung 0's job, so the FIXTURE is a plausible limiting factor rather than the ladder."""
    report = run_benchmark(mind, n_has=10, n_no=2, seed=0)
    assert set(report["rung_distribution"]) <= {0}, \
        "a rung above 0 fired -- update the recorded negative, this is news"


def test_the_negative_travels_with_the_faculty(mind):
    doc = mind.agent_benchmark.__doc__ or ""
    assert "RUNGS 1-5 FIRED ON 0/60" in doc
    assert "UNEXERCISED" in doc


def test_the_competitive_claim_is_scoped_honestly():
    """The comparison to NOOA was verified against arXiv:2607.20709 on 2026-07-26. Two of the plan's
    characterisations did not survive, and the corrected scope is pinned here so a future session does not
    re-inflate the claim from the plan's prose:

      * NOOA publishes no false-action or abstention metric -- CONFIRMED, and it is leCore's real edge.
      * NOOA "cannot express no tool fits" -- OVERSTATED. It has VALIDATED TERMINATION (a typed return
        carrying evidence and a verification command). That gates the EXIT; leCore's null gates the ENTRY.
      * NOOA "is not reproducible" -- WRONG. Open repo, pinned commits, public scorecards.

    leCore is NOT "everything NOOA does plus more": pass-by-reference previews, code-as-action, typed
    return validation and the memory subsystem are all absent here."""
    import pathlib
    doc = pathlib.Path("docs/COMPETITIVE_NOOA.md").read_text()
    assert "OVERSTATED" in doc and "VALIDATED TERMINATION" in doc.upper()
    assert "not" in doc and "everything NOOA does plus more" in doc

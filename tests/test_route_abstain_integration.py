"""Route-or-abstain (J1) at the seams: the acceptance battery from the backlog (misroutes abstain, real
matches keep routing), determinism, and the kept-negative demonstrated -- an in-vocabulary paraphrase of a
real capability routes while an out-of-vocabulary paraphrase of the SAME capability abstains, with the
abstention being correct behaviour and the alias being the fix."""
import lecore


def test_the_backlog_acceptance_battery():
    """The J1 acceptance criterion verbatim: 'the misroutes above return abstentions; real matches keep
    routing.'"""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    for q in ("counter traders", "purple monkey dishwasher zzz", "the the the the"):
        r = mind.route_or_abstain(q)
        assert r["abstain"], (q, r["z"])
        assert "no capability matches" in r["reason"] or "empty query" in r["reason"]
    for q, needle in (("where do the losses concentrate", "Loss space"),
                      ("render image to terminal", "ascii"),
                      ("veto committee", "Screen a battery"),
                      ("hour of day encoder", "Circular encoder")):
        r = mind.route_or_abstain(q)
        assert not r["abstain"], (q, r["z"])
        assert needle.lower() in r["hits"][0][0].name.lower(), (q, r["hits"][0][0].name)


def test_deterministic_given_seed():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    a = mind.route_or_abstain("counter traders", seed=3)
    b = mind.route_or_abstain("counter traders", seed=3)
    assert a["z"] == b["z"] and a["null_mean"] == b["null_mean"]


def test_the_kept_negative_out_of_vocabulary_abstention_and_its_fix():
    """The same underlying need phrased two ways: in catalog words it routes; in words the catalog never
    uses it abstains -- and the abstention is CORRECT (the catalog has no purchase on those tokens). The
    documented fix is an alias, not a lower z_min; this test pins the behaviour the docstring promises."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    r_in = mind.route_or_abstain("longest losing streak versus chance")
    assert not r_in["abstain"]
    r_out = mind.route_or_abstain("ouchies bunched up in the diary")   # same intent, alien vocabulary
    assert r_out["abstain"], r_out["z"]

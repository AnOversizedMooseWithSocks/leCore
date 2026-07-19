"""Pin the CATALOG-corpus retrieval baseline so it cannot rot silently.

This suite (tools/semantic/catalog_exam.py) is the prerequisite SEMANTIC_BACKLOG S4 demands EXIST before any
front-door hybrid is wired -- so that "better" is a measured claim against a real baseline rather than a strawman.
These tests do NOT assert a score is good. They assert:

  1. every gold NAMES a live capability (a gold that quietly stopped existing makes its ask unanswerable, so the
     ask drops out of the honest denominator and flatters every arm equally -- EIGHT of the first-draft golds were
     wrong, caught before a single score printed);
  2. the baseline has not COLLAPSED (a guard against a catalog change silently wrecking retrieval);
  3. the suite stays big enough to say anything at all.

Deliberately NOT asserted: that any particular ask passes. Six miss in every arm, and fixing them with aliases
would be teaching to the test -- see the module docstring.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "semantic"))

import pytest

import catalog_exam
from holographic.misc.holographic_unified import UnifiedMind


@pytest.fixture(scope="module")
def mind():
    return UnifiedMind(dim=64, seed=0)


def test_every_gold_names_a_live_capability(mind):
    missing = catalog_exam.verify_golds(mind)
    assert not missing, ("gold(s) no longer in the catalog -- the ask is unanswerable and silently inflates every "
                         "arm; retarget or drop it: %s" % missing)


def test_the_suite_is_big_enough_to_mean_something():
    """S4 requires >=30 asks. Below that a 1-ask delta is most of a standard error, which is how a noise result
    gets shipped as a win."""
    assert len(catalog_exam.ASKS) >= 30
    assert len({g for _a, g in catalog_exam.ASKS}) >= 30, "golds must be distinct or the exam grades one thing"


@pytest.mark.slow                            # 35 asks x 3 arms over 2,111 capabilities -- past the 15s budget
def test_the_token_baseline_has_not_collapsed(mind):
    """FIRST MEASUREMENT: token top1 14/35, top5 21/35, median 2.0. Pinned with slack -- this is a rot alarm, not
    a threshold to game. If a catalog edit drops it hard, that is a regression worth a conversation."""
    res = catalog_exam.run(mind)
    base = res["token (BASELINE)"]
    assert base["top1"] >= 11, ("front-door top-1 collapsed", base)
    assert base["top5"] >= 18, ("front-door top-5 collapsed", base)
    assert base["median"] <= 4.0, ("front-door median rank collapsed", base)


@pytest.mark.slow                            # same instrument, same cost; runs in CI's slow lane, not inline
def test_bm25_does_not_beat_the_baseline(mind):
    """KEPT NEGATIVE, pinned so nobody re-proposes it from memory: BM25 scored +1 ask over token and RRF added
    nothing on top. On 35 asks a 1-ask delta is ~0.5 SE -- noise, not a win. If a future change makes BM25
    genuinely dominant (say +5 top-1), THIS TEST FAILS and that is the signal to re-open S4 with real evidence."""
    res = catalog_exam.run(mind)
    base, bm = res["token (BASELINE)"], res["bm25"]
    assert bm["top1"] - base["top1"] < 5, (
        "BM25 now beats the token baseline decisively -- re-open S4 and consider shipping the hybrid", base, bm)

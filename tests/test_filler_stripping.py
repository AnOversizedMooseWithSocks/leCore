"""Regression traps for filler-prefix stripping in the capability router (work plan item 1.1).

THE FIXTURE IS COMMITTED AND SEEDED, per the plan's measurement rules: an analysis someone else cannot
re-run identically is not a result. The pairs are drawn from the live catalog's own author-written
aliases with a fixed shuffle seed, so a later session re-runs the IDENTICAL comparison.

The subtlety this file exists to protect is not the stripping itself -- it is WHERE the stripping has to
happen. The stop-word list already dropped 'how', 'do', 'i', so the TOKEN sets were always identical and
a token-level fix would have shipped green and moved nothing. What a prefix destroys is the whole-string
EXACT-ALIAS bonus (+5.0, the largest term in the score). Both facts are pinned below.
"""
import random

import pytest

import lecore
from holographic.caching_and_storage.holographic_catalog import (_strip_filler, _tokens,
                                                                 default_catalog)

FILLERS = ("how do i", "how do you", "can you help me", "i want to",
           "please", "i need to", "what's the best way to")


def _fixture(n=150, seed=0):
    """The committed fixture: author-written aliases of >=4 words, fixed shuffle, fixed size."""
    cat = default_catalog()
    pairs = []
    for cap in cat.all():
        for alias in getattr(cap, "aliases", ()) or ():
            if len(str(alias).split()) >= 4:
                pairs.append((str(alias), cap.name))
    pairs = sorted(set(pairs))
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def _top1(mind, query, want):
    hits = mind.find_capability(query)[:1]
    return bool(hits) and getattr(hits[0], "name", "") == want


# --------------------------------------------------------------------------------------
# The mechanism -- pinned so nobody "simplifies" the fix back into the tokenizer.
# --------------------------------------------------------------------------------------

def test_the_tokenizer_already_dropped_fillers_so_a_token_fix_would_be_ceremony():
    # This is WHY the fix lives at the phrase level. If these ever diverge, the reasoning in
    # _strip_filler's docstring is stale and the fix may belong somewhere else after all.
    assert _tokens("how do i smooth a bumpy mesh") == _tokens("smooth a bumpy mesh")
    assert _tokens("please smooth a bumpy mesh") != _tokens("smooth a bumpy mesh")   # 'please' leaks


def test_strip_filler_removes_prefixes_repeatedly():
    assert _strip_filler("how do i smooth a bumpy mesh") == "smooth a bumpy mesh"
    assert _strip_filler("please can you help me smooth a bumpy mesh") == "smooth a bumpy mesh"
    assert _strip_filler("  How Do I   Smooth A Bumpy Mesh ") == "smooth a bumpy mesh"


def test_strip_filler_never_returns_an_empty_query():
    # THE REAL INVARIANT, and the first draft of this test over-specified it to "unchanged", which
    # failed on 'can you help me' -> 'help me': the 'can you' prefix matches, the remainder is
    # non-empty so it strips, and 'help me' then has nothing after it to give up. That CHAINING is
    # correct behaviour -- the guard's job is to never hand find_capability an empty string, not to
    # freeze pure-filler input. Both queries are meaningless for retrieval either way; only emptiness
    # would change the result, by turning an honest best guess into [].
    for q in ("please", "how do i", "can you help me", "what's the best way to"):
        assert _strip_filler(q).strip(), "%r stripped to nothing" % q


def test_a_pure_filler_query_still_reaches_the_router(mind):
    # The consequence of the invariant above, checked through the real path rather than the helper.
    assert isinstance(mind.find_capability("can you help me"), list)


# --------------------------------------------------------------------------------------
# The measured gate.
# --------------------------------------------------------------------------------------

def test_filler_prefixed_queries_match_the_exact_alias_arm(mind):
    # THE GATE. Before the fix: exact 99.3%, every fully-stopped filler 91.3%, "what's the best way to"
    # 81.3%. After: all arms 99.3%. The bar is stated as "the filler arm is not materially worse than
    # the exact arm", which is the claim -- not a threshold on a point estimate.
    pairs = _fixture()
    exact = sum(_top1(mind, a, n) for a, n in pairs) / len(pairs)
    for filler in FILLERS:
        got = sum(_top1(mind, filler + " " + a, n) for a, n in pairs) / len(pairs)
        assert got >= exact - 0.02, "filler %r regressed: %.3f vs exact %.3f" % (filler, got, exact)


def test_the_exact_alias_arm_did_not_move(mind):
    # The other half of the gate: a fix that helps one arm by hurting another is not a fix.
    pairs = _fixture()
    assert sum(_top1(mind, a, n) for a, n in pairs) / len(pairs) >= 0.98


def test_aliases_that_themselves_begin_with_a_filler_still_resolve(mind):
    # THE REGRESSION THE NAIVE FIX WOULD HAVE CAUSED, pinned. Four shipped aliases start with a filler;
    # stripping the query before the exact-alias test would have destroyed the bonus for exactly these.
    # Both the raw and the stripped phrase are tested, so they keep it.
    for alias, want in (("how do I get from points to a mesh", "suggest_pipeline"),):
        assert _top1(mind, alias, want)
    cat = default_catalog()
    starts_with_filler = [(str(a), c.name) for c in cat.all()
                          for a in (getattr(c, "aliases", ()) or ())
                          if any(str(a).lower().startswith(f + " ") for f in FILLERS)]
    assert starts_with_filler, "fixture assumption broke: no alias starts with a filler any more"
    for alias, want in starts_with_filler:
        assert _top1(mind, alias, want), "%r no longer resolves to %r" % (alias, want)


def test_find_scored_got_the_same_fix_as_find_capability(mind):
    # find_scored's own docstring promises "same scoring". A fix applied to one path and not the other
    # would make them disagree on exactly the queries this change repairs -- the seam-defect shape the
    # work plan's 6.2 predicts.
    cat = default_catalog()
    pairs = _fixture(n=40)
    for alias, want in pairs:
        q = "how do i " + alias
        top_find = mind.find_capability(q)[:1]
        top_scored = cat.find_scored(q)[:1]
        if top_find and top_scored:
            assert getattr(top_find[0], "name", "") == getattr(top_scored[0][0], "name", "")


def test_stripping_is_deterministic():
    q = "what's the best way to smooth a bumpy mesh"
    assert _strip_filler(q) == _strip_filler(q)

"""The dictionary must know the words that carry logical structure.

The shipped dictionary is WordNet-derived, and WordNet covers only the OPEN word
classes (noun, verb, adjective, adverb) BY DESIGN. Every closed-class word is
therefore structurally absent -- and those are exactly the cause-and-effect and
ordering vocabulary a model is asked to reason with.

MEASURED against the shipped file before the supplement:
    causal/logical   7/12 present  -- missing because, since, unless, although, whereas
    conditional      3/ 8 present  -- missing if, else, when, whether, provided
    lookup("because") -> None

Source is Webster's Unabridged 1913, PUBLIC DOMAIN, and its definitions are
RELATIONAL rather than nominal: "because" is "by or for the cause that",
"unless" is "upon any less condition than". That is the part that teaches the
relation instead of naming a thing.
"""

import lecore


def _mind():
    return lecore.UnifiedMind(dim=64)


def test_the_reasoning_vocabulary_is_present():
    m = _mind()
    for word in ("because", "since", "unless", "although", "whereas",
                 "if", "else", "whether", "provided", "until"):
        assert m.lookup(word), (
            "%r is absent -- WordNet excludes closed-class words by design and "
            "the supplement did not load" % word)


def test_the_supplement_never_overwrites_a_wordnet_sense():
    """setdefault, not update: where WordNet has a sense it is better structured."""
    import holographic.misc.holographic_dictionary as D

    # entry() RESHAPES the record for callers, so provenance is asserted against
    # the raw store -- checking the reshaped view tests the view, not the merge.
    raw = D._load()
    assert raw["cause"].get("src") != "webster1913", (
        "a WordNet entry was clobbered by the supplement: %r" % raw["cause"])
    assert raw["cause"].get("d"), "the WordNet definition went missing"
    assert raw["because"].get("src") == "webster1913", raw["because"]


def test_a_missing_supplement_is_not_a_failure(tmp_path, monkeypatch):
    """The engine must still work if the file is absent -- it is an ADDITION."""
    import holographic.misc.holographic_dictionary as D

    n = D._merge_function_words({})          # real file present: adds entries
    assert n > 0

    monkeypatch.setattr(D, "_DATA_PATH", str(tmp_path / "nope.json.xz"))
    assert D._merge_function_words({}) == 0, "a missing supplement must be silent"

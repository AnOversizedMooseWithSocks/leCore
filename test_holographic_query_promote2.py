"""Query layer PROMOTE (P4-P6): fuzzy dedup, match explanation (exact/fuzzy fork), recommendation."""
from holographic_query import UserTable, near_duplicates, explain_match, recommend


def _t():
    t = UserTable("critters", ["legs", "sound"], dim=1024, seed=0)
    for r in ([{"legs": 4, "sound": "meow"}] * 3 + [{"legs": 2, "sound": "tweet"}] * 2 + [{"legs": 0, "sound": "hiss"}]):
        t.insert(r)
    return t


def test_p4_finds_duplicate_groups():
    groups = near_duplicates(_t(), threshold=0.9)
    sizes = sorted(g["size"] for g in groups)
    assert sizes == [2, 3]                                    # the meow-trio and the tweet-pair; the lone hiss excluded
    assert all(g["similarity"] > 0.99 for g in groups)


def test_p4_singletons_are_not_duplicates():
    t = UserTable("u", ["v"], dim=1024, seed=0)
    for r in [{"v": "a"}, {"v": "b"}, {"v": "c"}]:
        t.insert(r)
    assert near_duplicates(t, threshold=0.9) == []           # all distinct -> no duplicate groups


def test_p5_identical_rows_drive_on_all_columns():
    e = explain_match(_t(), {"legs": 4, "sound": "meow"}, {"legs": 4, "sound": "meow"})
    assert set(e["drives"]) == {"legs", "sound"} and e["against"] == []


def test_p5_respects_exact_fuzzy_fork():
    # share the NUMERIC legs=4 (exact side) but differ on the CATEGORICAL sound (vector side)
    e = explain_match(_t(), {"legs": 4, "sound": "meow"}, {"legs": 4, "sound": "tweet"})
    assert e["drives"] == ["legs"] and e["against"] == ["sound"]
    assert e["per_column"]["legs"] == 1.0                    # numeric compared EXACTLY (equal)


def test_p5_nothing_shared():
    e = explain_match(_t(), {"legs": 4, "sound": "meow"}, {"legs": 2, "sound": "tweet"})
    assert e["drives"] == [] and set(e["against"]) == {"legs", "sound"}


def test_p6_recommend_excludes_examples():
    recs = recommend(_t(), [{"legs": 4, "sound": "meow"}], k=3)
    assert all(r["sound"] != "meow" for r in recs)           # the cats (examples) are excluded
    assert all("_confidence" in r for r in recs)


def test_p6_recommend_can_keep_examples():
    recs = recommend(_t(), [{"legs": 2, "sound": "tweet"}], k=10, exclude=False)
    assert any(r["sound"] == "tweet" for r in recs)          # with exclude off, examples can appear

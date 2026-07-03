"""Query BUILD B9: SQL-surface fill-ins -- DISTINCT, OFFSET, HAVING, UNION / UNION ALL."""
from holographic_query import UserTable, run_sql


def _t():
    t = UserTable("critters", ["name", "legs", "habitat"], dim=1024, seed=0)
    for r in [{"name": "cat", "legs": 4, "habitat": "land"}, {"name": "dog", "legs": 4, "habitat": "land"},
              {"name": "duck", "legs": 2, "habitat": "water"}, {"name": "crab", "legs": 8, "habitat": "water"},
              {"name": "ant", "legs": 6, "habitat": "land"}, {"name": "newt", "legs": 4, "habitat": "water"}]:
        t.insert(r)
    return t


def test_distinct():
    assert sorted(r["habitat"] for r in run_sql("SELECT DISTINCT habitat FROM critters", _t())) == ["land", "water"]


def test_distinct_multi_column():
    rows = run_sql("SELECT DISTINCT legs, habitat FROM critters", _t())
    assert len(rows) == 5                                     # (4,land),(2,water),(8,water),(6,land),(4,water)


def test_offset():
    r = run_sql("SELECT name FROM critters ORDER BY legs ASC LIMIT 2 OFFSET 1", _t())
    assert len(r) == 2                                        # skip the smallest, take the next two


def test_offset_beyond_end_is_empty():
    assert run_sql("SELECT name FROM critters LIMIT 5 OFFSET 100", _t()) == []


def test_having_filters_groups():
    r = run_sql("SELECT habitat, COUNT(*) FROM critters GROUP BY habitat HAVING COUNT(*) >= 2", _t())
    assert {row["habitat"] for row in r} == {"land", "water"}   # both have 3


def test_having_excludes_small_groups():
    t = _t(); t.insert({"name": "yeti", "legs": 2, "habitat": "mountain"})   # a group of 1
    r = run_sql("SELECT habitat FROM critters GROUP BY habitat HAVING COUNT(*) > 1", t)
    assert "mountain" not in {row["habitat"] for row in r}


def test_union_dedupes():
    got = sorted(r["name"] for r in run_sql(
        "SELECT name FROM critters WHERE legs = 8 UNION SELECT name FROM critters WHERE habitat = 'land'", _t()))
    assert got == ["ant", "cat", "crab", "dog"]              # crab (legs=8) + land trio, no dupes


def test_union_all_keeps_duplicates():
    n = len(run_sql(
        "SELECT name FROM critters WHERE legs = 4 UNION ALL SELECT name FROM critters WHERE habitat = 'land'", _t()))
    assert n == 6                                            # 3 four-legged + 3 land, overlaps kept


def test_distinct_with_limit():
    # LIMIT applies to the DEDUPED rows, not the pre-dedupe scan
    assert len(run_sql("SELECT DISTINCT habitat FROM critters LIMIT 1", _t())) == 1

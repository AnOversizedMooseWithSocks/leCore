"""Query layer Tier FIX (F1-F3): measured bugs an evaluator hits first -- pinned so they can't regress."""
import pytest
from holographic_query import UserTable, run_sql, QueryError


def _t():
    t = UserTable("animals", ["name", "legs"], dim=1024, seed=0)
    for r in [{"name": "cat", "legs": 4}, {"name": "bird", "legs": 2}, {"name": "ant", "legs": 6}]:
        t.insert(r)
    return t


def test_f1_unknown_column_errors():
    with pytest.raises(QueryError):
        run_sql("SELECT nope FROM animals", _t())          # was: confident {'nope': None, '_confidence': 1.0}


def test_f1_unknown_where_column_errors():
    with pytest.raises(QueryError):
        run_sql("SELECT name FROM animals WHERE nope = 1", _t())


def test_f1_sparse_declared_column_is_allowed():
    t = _t(); t.insert({"name": "worm"})                   # 'legs' declared but absent on this row
    r = run_sql("SELECT name, legs FROM animals WHERE name = 'worm'", t)
    assert r[0]["name"] == "worm" and r[0]["legs"] is None  # sparse -> None, NOT an error


def test_f2_multi_predicate_now_works():
    # F2 (clean rejection) is SUPERSEDED by B3: AND/OR actually work now, no TypeError leak.
    r = run_sql("SELECT name FROM animals WHERE legs > 2 AND name = 'cat'", _t())
    assert [x["name"] for x in r] == ["cat"]


def test_f2_and_inside_quotes_not_misread():
    r = run_sql("SELECT name FROM animals WHERE name = 'black and white'", _t())
    assert r == []                                        # a legit value with 'and' is one predicate, not two


def test_f3_limit_zero_returns_no_rows():
    assert run_sql("SELECT name FROM animals LIMIT 0", _t()) == []   # was: returned all rows (0 is falsy bug)


def test_f3_limit_nonzero_still_works():
    assert len(run_sql("SELECT name FROM animals LIMIT 2", _t())) == 2

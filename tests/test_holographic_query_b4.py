"""Query BUILD B4: primary-key hash index -- O(1) `WHERE pk = X`, kept in sync through insert/update/delete."""
import time
import pytest
from holographic.agents_and_reasoning.holographic_query import UserTable, run_sql, update, delete, QueryError


def _indexed(n):
    t = UserTable("u", ["uid", "name"], dim=256, seed=0)
    for i in range(n):
        t.insert({"uid": "u%d" % i, "name": "n%d" % i})
    return t.set_primary_key("uid")


def test_pk_lookup_correct():
    t = _indexed(500)
    assert run_sql("SELECT name FROM u WHERE uid = 'u321'", t) == [{"name": "n321", "_confidence": 1.0}]
    assert run_sql("SELECT name FROM u WHERE uid = 'nope'", t) == []


def test_index_synced_on_insert():
    t = _indexed(10)
    t.insert({"uid": "u999", "name": "late"})
    assert run_sql("SELECT name FROM u WHERE uid = 'u999'", t)[0]["name"] == "late"


def test_index_synced_on_update():
    t = _indexed(10)
    update(t, "uid = 'u5'", {"name": "RENAMED"})
    assert run_sql("SELECT name FROM u WHERE uid = 'u5'", t)[0]["name"] == "RENAMED"   # sees the live version


def test_index_synced_on_delete():
    t = _indexed(10)
    delete(t, "uid = 'u5'")
    assert run_sql("SELECT name FROM u WHERE uid = 'u5'", t) == []                    # tombstoned -> absent


def test_bad_primary_key_errors():
    t = UserTable("u", ["uid"], dim=256, seed=0)
    with pytest.raises(QueryError):
        t.set_primary_key("nope")


def test_indexed_lookup_is_sublinear():
    # the indexed lookup should NOT grow with table size the way a scan does (loose bound to avoid flakiness)
    small, big = _indexed(500), _indexed(4000)
    key_s, key_b = "u499", "u3999"
    R = 200
    def timeit(t, key):
        s = time.time()
        for _ in range(R):
            run_sql("SELECT name FROM u WHERE uid = '%s'" % key, t)
        return (time.time() - s) / R
    ts, tb = timeit(small, key_s), timeit(big, key_b)
    assert tb < 3 * ts + 1e-4                                     # ~flat: 8x the rows is NOT ~8x the time

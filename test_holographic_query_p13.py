"""Query PROMOTE P13: add a column with no migration; sparse rows."""
import numpy as np
from holographic_query import UserTable, run_sql, QueryError
import pytest


def test_p13_sparse_rows_read_as_none():
    t = UserTable("t", ["name", "age"], dim=1024, seed=0)
    t.insert({"name": "alice"})                            # 'age' declared but absent
    assert run_sql("SELECT name, age FROM t", t)[0]["age"] is None


def test_p13_add_column_no_migration():
    t = UserTable("t", ["name"], dim=1024, seed=0)
    t.insert({"name": "alice"}); t.insert({"name": "bob"})
    before = t.records.copy()
    t.add_column("age")
    assert np.array_equal(before, t.records)              # existing records UNTOUCHED (no re-encode)
    t.insert({"name": "carol", "age": 30})
    rows = run_sql("SELECT name, age FROM t", t)
    ages = {r["name"]: r["age"] for r in rows}
    assert ages["alice"] is None and ages["carol"] == 30   # old rows sparse, new row has it


def test_p13_new_column_is_queryable():
    t = UserTable("t", ["name"], dim=1024, seed=0)
    t.insert({"name": "alice"})
    with pytest.raises(QueryError):
        run_sql("SELECT age FROM t", t)                   # before add_column: not a column
    t.add_column("age"); t.insert({"name": "bob", "age": 40})
    assert run_sql("SELECT name FROM t WHERE age > 25", t)[0]["name"] == "bob"   # queryable after

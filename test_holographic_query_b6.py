"""Query BUILD B6: single-writer transactions -- all-or-nothing, rollback on error or explicit Rollback."""
import pytest
from holographic_query import UserTable, run_sql, transaction, Rollback, ConstraintError


def _t():
    t = UserTable("accounts", ["id", "bal"], dim=512, seed=0).set_primary_key("id")
    t.insert({"id": 1, "bal": 100})
    return t


def test_rollback_on_error_is_atomic():
    t = _t()
    with pytest.raises(ConstraintError):
        with transaction(t):
            t.insert({"id": 2, "bal": 50})
            t.insert({"id": 3, "bal": 75})
            t.insert({"id": 1, "bal": 999})              # PK dup -> the WHOLE batch rolls back
    assert len(t.rows) == 1                              # none of the batch persisted


def test_explicit_rollback_is_swallowed():
    t = _t()
    with transaction(t):
        t.insert({"id": 4, "bal": 10})
        raise Rollback                                  # clean abort, no exception escapes
    assert len(t.rows) == 1


def test_clean_exit_commits():
    t = _t()
    with transaction(t):
        t.insert({"id": 5, "bal": 20})
        t.insert({"id": 6, "bal": 30})
    assert len(t.rows) == 3
    assert run_sql("SELECT bal FROM accounts WHERE id = 5", t)[0]["bal"] == 20


def test_pk_index_restored_on_rollback():
    t = _t()
    try:
        with transaction(t):
            t.insert({"id": 7, "bal": 1})
            raise Rollback
    except Rollback:
        pass
    # id=7 was rolled back; a fresh insert of id=7 must be allowed (index has no stale entry)
    t.insert({"id": 7, "bal": 2})
    assert run_sql("SELECT bal FROM accounts WHERE id = 7", t)[0]["bal"] == 2


def test_multi_table_transaction_rolls_back_both():
    a = _t()
    b = UserTable("b", ["id"], dim=512, seed=1).set_primary_key("id")
    b.insert({"id": 1})
    with pytest.raises(ConstraintError):
        with transaction(a, b):
            a.insert({"id": 2, "bal": 5})
            b.insert({"id": 1})                         # dup in b -> both a and b roll back
    assert len(a.rows) == 1 and len(b.rows) == 1        # a's insert undone too

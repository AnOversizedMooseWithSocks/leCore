"""Query BUILD B5: constraints -- NOT NULL, UNIQUE, PRIMARY KEY, FOREIGN KEY, CHECK, enforced on write."""
import pytest
from holographic_query import UserTable, ConstraintError


def test_pk_is_unique_and_not_null():
    t = UserTable("u", ["uid", "name"], dim=512, seed=0).set_primary_key("uid")
    t.insert({"uid": "u1", "name": "alice"})
    with pytest.raises(ConstraintError):
        t.insert({"uid": "u1", "name": "dup"})               # duplicate PK
    with pytest.raises(ConstraintError):
        t.insert({"name": "noid"})                           # missing PK (NOT NULL)


def test_not_null():
    t = UserTable("u", ["a", "b"], dim=512, seed=0).not_null("b")
    t.insert({"a": 1, "b": 2})
    with pytest.raises(ConstraintError):
        t.insert({"a": 3})                                   # b absent


def test_unique_non_pk():
    t = UserTable("u", ["id", "email"], dim=512, seed=0).unique("email")
    t.insert({"id": 1, "email": "x@y.com"})
    with pytest.raises(ConstraintError):
        t.insert({"id": 2, "email": "x@y.com"})


def test_foreign_key():
    users = UserTable("users", ["uid"], dim=512, seed=0).set_primary_key("uid")
    users.insert({"uid": "u1"})
    orders = UserTable("orders", ["oid", "uid"], dim=512, seed=1).set_primary_key("oid").foreign_key("uid", users, "uid")
    orders.insert({"oid": "o1", "uid": "u1"})                # references an existing user
    with pytest.raises(ConstraintError):
        orders.insert({"oid": "o2", "uid": "ghost"})         # dangling reference


def test_foreign_key_null_allowed():
    users = UserTable("users", ["uid"], dim=512, seed=0).set_primary_key("uid")
    orders = UserTable("orders", ["oid", "uid"], dim=512, seed=1).set_primary_key("oid").foreign_key("uid", users, "uid")
    orders.insert({"oid": "o1"})                             # uid absent -> allowed (resolve-or-null opt-out)


def test_check_predicate():
    t = UserTable("u", ["name", "age"], dim=512, seed=0).check(lambda r: r.get("age", 0) >= 0, "age>=0")
    t.insert({"name": "a", "age": 5})
    with pytest.raises(ConstraintError):
        t.insert({"name": "b", "age": -1})


def test_violating_insert_leaves_table_unchanged():
    t = UserTable("u", ["uid"], dim=512, seed=0).set_primary_key("uid")
    t.insert({"uid": "u1"})
    try:
        t.insert({"uid": "u1"})
    except ConstraintError:
        pass
    assert len(t.rows) == 1                                  # the bad insert did not persist


def test_no_constraints_means_no_enforcement():
    t = UserTable("u", ["a"], dim=512, seed=0)              # no constraints declared
    t.insert({"a": 1}); t.insert({"a": 1}); t.insert({})    # all fine
    assert len(t.rows) == 3

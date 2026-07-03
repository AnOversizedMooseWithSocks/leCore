"""Query BUILD B1: JOIN -- exact hash-join (inner/left) + fuzzy/semantic join."""
import pytest
from holographic_query import UserTable, join, fuzzy_join, scalar_key_encoder, QueryError


def _users_orders():
    u = UserTable("users", ["uid", "name"], dim=1024, seed=0)
    for r in [{"uid": "u1", "name": "alice"}, {"uid": "u2", "name": "bob"}, {"uid": "u3", "name": "carol"}]:
        u.insert(r)
    o = UserTable("orders", ["uid", "item"], dim=1024, seed=1)
    for r in [{"uid": "u1", "item": "book"}, {"uid": "u1", "item": "pen"}, {"uid": "u2", "item": "lamp"}]:
        o.insert(r)
    return u, o


def test_inner_join():
    u, o = _users_orders()
    res = join(u, o, on="uid")
    assert len(res) == 3                                       # alice x2, bob x1; carol dropped (no orders)
    assert {(r["name"], r["item"]) for r in res} == {("alice", "book"), ("alice", "pen"), ("bob", "lamp")}


def test_left_join_null_fills():
    u, o = _users_orders()
    res = join(u, o, on="uid", how="left")
    carol = [r for r in res if r["name"] == "carol"][0]
    assert carol["item"] is None                              # kept, right side null-filled


def test_join_key_appears_once_and_collisions_suffixed():
    a = UserTable("a", ["k", "v"], dim=512, seed=0)
    a.insert({"k": "x", "v": "left"})
    b = UserTable("b", ["k", "v"], dim=512, seed=1)
    b.insert({"k": "x", "v": "right"})
    row = join(a, b, on="k")[0]
    assert row["k"] == "x" and row["v_l"] == "left" and row["v_r"] == "right"   # shared non-key 'v' suffixed


def test_join_different_key_names():
    u = UserTable("u", ["uid", "name"], dim=512, seed=0); u.insert({"uid": "u1", "name": "alice"})
    o = UserTable("o", ["user", "item"], dim=512, seed=1); o.insert({"user": "u1", "item": "book"})
    row = join(u, o, on=("uid", "user"))[0]
    assert row["name"] == "alice" and row["item"] == "book"


def test_join_bad_key_errors():
    u, o = _users_orders()
    with pytest.raises(QueryError):
        join(u, o, on="nope")


def test_fuzzy_join_categorical_matches_identical_keys():
    u, o = _users_orders()
    res = fuzzy_join(u, o, on="uid", threshold=0.5)           # shared encoder -> identical keys match at ~1.0
    assert all(r["_confidence"] > 0.99 for r in res)
    assert {(r["name"], r["item"]) for r in res} == {("alice", "book"), ("alice", "pen"), ("bob", "lamp")}


def test_fuzzy_join_semantic_proximity_on_numbers():
    enc = scalar_key_encoder(0, 100, dim=1024, seed=0)
    a = UserTable("a", ["ts", "ev"], dim=1024, seed=2)
    for r in [{"ts": 10, "ev": "login"}, {"ts": 50, "ev": "buy"}]:
        a.insert(r)
    b = UserTable("b", ["ts", "ev"], dim=1024, seed=3)
    for r in [{"ts": 11, "ev": "click"}, {"ts": 80, "ev": "logout"}]:
        b.insert(r)
    res = fuzzy_join(a, b, on="ts", threshold=0.85, key_encoder=enc)
    assert len(res) == 1 and res[0]["ev_l"] == "login" and res[0]["ev_r"] == "click"   # only the CLOSE pair (10~11)

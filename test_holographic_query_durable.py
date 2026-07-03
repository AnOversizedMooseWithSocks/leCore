"""Query BUILD B7: durability & crash recovery -- snapshot + append-only redo journal."""
import os
import tempfile
import pytest
from holographic_query import Database, run_sql, update as _update
from holographic_query_durable import save_snapshot, load_snapshot, Journal, recover


def _db():
    db = Database()
    db.create_namespace("shop", tier="persistent")
    db.create_table("shop.items", ["name", "qty"], dim=256, seed=0)
    db.insert("shop.items", {"name": "apple", "qty": 3})
    db.insert("shop.items", {"name": "pear", "qty": 5})
    return db


def test_snapshot_roundtrips():
    d = tempfile.mkdtemp(); snap = os.path.join(d, "s.json")
    save_snapshot(_db(), snap)
    db2 = load_snapshot(snap)
    rows = {r["name"]: r["qty"] for r in run_sql("SELECT name, qty FROM items", db2.resolve("shop.items"))}
    assert rows == {"apple": 3, "pear": 5}


def test_journal_replay_recovers_writes():
    d = tempfile.mkdtemp(); snap = os.path.join(d, "s.json"); jrnl = os.path.join(d, "j")
    db = _db(); save_snapshot(db, snap)
    j = Journal(jrnl)
    db.insert("shop.items", {"name": "plum", "qty": 2}); j.log_insert("shop.items", {"name": "plum", "qty": 2})
    _update(db.resolve("shop.items"), "name = 'apple'", {"qty": 9}); j.log_update("shop.items", "name = 'apple'", {"qty": 9})
    recovered = recover(snap, jrnl)                                # simulate crash -> recover from disk
    rows = {r["name"]: r["qty"] for r in run_sql("SELECT name, qty FROM items", recovered.resolve("shop.items"))}
    assert rows == {"apple": 9, "pear": 5, "plum": 2}


def test_journal_delete_recovered():
    d = tempfile.mkdtemp(); snap = os.path.join(d, "s.json"); jrnl = os.path.join(d, "j")
    db = _db(); save_snapshot(db, snap)
    from holographic_query import delete as _del
    _del(db.resolve("shop.items"), "name = 'pear'"); Journal(jrnl).log_delete("shop.items", "name = 'pear'")
    recovered = recover(snap, jrnl)
    assert {r["name"] for r in run_sql("SELECT name FROM items", recovered.resolve("shop.items"))} == {"apple"}


def test_truncate_after_resnapshot():
    d = tempfile.mkdtemp(); snap = os.path.join(d, "s.json"); jrnl = os.path.join(d, "j")
    db = _db(); save_snapshot(db, snap)
    j = Journal(jrnl); j.log_insert("shop.items", {"name": "plum", "qty": 1})
    recovered = recover(snap, jrnl)
    save_snapshot(recovered, snap); j.truncate()                  # fold the log into the base
    assert j.entries() == []
    again = recover(snap, jrnl)
    assert {r["name"] for r in run_sql("SELECT name FROM items", again.resolve("shop.items"))} == {"apple", "pear", "plum"}


def test_atomic_write_leaves_no_tmp():
    d = tempfile.mkdtemp(); snap = os.path.join(d, "s.json")
    save_snapshot(_db(), snap)
    assert os.path.exists(snap) and not os.path.exists(snap + ".tmp")   # write-then-rename cleaned up

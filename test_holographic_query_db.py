"""Query Interface Phases 9-13: namespaces + the read-only system wall, user tables, bookmarks, views, persistence."""
import numpy as np
import pytest
from holographic_query import Database, UserTable, from_rows, run_db_sql, QueryError


def _db():
    mats = from_rows([
        {"id": 1, "name": "gold", "material": "gold"},
        {"id": 2, "name": "ring", "material": "gold"},
        {"id": 3, "name": "pipe", "material": "copper"},
    ], ["id", "name", "material"], dim=2048, seed=0)
    db = Database()
    db.register_system("scene", mats)
    return db, mats


def test_system_wall_reads_ok_writes_refused():
    db, _ = _db()
    assert [r["name"] for r in run_db_sql("SELECT name FROM system.scene", db)] == ["gold", "ring", "pipe"]
    with pytest.raises(QueryError):
        db.insert("system.scene", {"id": 9, "name": "hack"})
    with pytest.raises(QueryError):
        db.create_table("system.evil", ["x"])


def test_create_insert_select_roundtrip():
    db, _ = _db()
    db.create_database("curation")
    db.create_table("curation.favorites", ["obj_id", "note"], dim=2048, seed=0)
    db.insert("curation.favorites", {"obj_id": 1, "note": "shiny"})
    assert db.resolve("curation.favorites").rows == [{"obj_id": 1, "note": "shiny"}]


def test_reserved_system_name():
    db, _ = _db()
    with pytest.raises(QueryError):
        db.create_database("system")


def test_bookmark_snapshot_and_reference():
    db, mats = _db()
    db.create_database("curation")
    db.create_table("curation.gold", ["name", "material"], dim=2048, seed=1)
    db.insert_select("curation.gold", ["name", "material"], "system.scene",
                     where=("material", "=", "gold"), mode="snapshot")
    assert {r["name"] for r in db.resolve("curation.gold").rows} == {"gold", "ring"}
    # reference resolves live; and dangles to None if the source row is gone
    db.create_table("curation.refs", ["_ref_source", "_ref_id"], dim=2048, seed=2)
    db.insert_select("curation.refs", ["id"], "system.scene", where=("material", "=", "copper"), mode="reference")
    live = db.resolve_reference(db.resolve("curation.refs").rows[0])
    assert live["name"] == "pipe"
    assert db.resolve_reference({"_ref_source": "system.scene", "_ref_id": 999}) is None


def test_live_view_reflects_source():
    db, mats = _db()
    db.create_database("curation")
    db.create_view("curation.gold_view", ["name"], "system.scene", where=("material", "=", "gold"))
    assert {r["name"] for r in db.run_view("curation.gold_view")} == {"gold", "ring"}


def test_catalog_lists_user_objects():
    db, _ = _db()
    db.create_database("curation")
    db.create_table("curation.t", ["a"], dim=512)
    db.create_view("curation.v", ["a"], "curation.t")
    cat = [(r["namespace"], r["name"], r["kind"]) for r in db.catalog() if r["namespace"] == "curation"]
    assert ("curation", "t", "table") in cat and ("curation", "v", "view") in cat


def test_persistence_by_replay():
    db, mats = _db()
    db.create_database("curation")
    db.create_table("curation.favorites", ["obj_id", "note"], dim=2048, seed=0)
    db.insert("curation.favorites", {"obj_id": 1, "note": "shiny"})
    db.create_view("curation.gold_view", ["name"], "system.scene", where=("material", "=", "gold"))
    state = db.to_state()
    db2 = Database.from_state(state, system_tables={"scene": mats})
    assert db2.resolve("curation.favorites").rows == [{"obj_id": 1, "note": "shiny"}]
    assert {r["name"] for r in db2.run_view("curation.gold_view")} == {"gold", "ring"}
    # reloaded records are byte-identical (seed fixes every atom)
    assert np.array_equal(db.resolve("curation.favorites").records, db2.resolve("curation.favorites").records)


def test_sql_ddl_and_wall():
    db, _ = _db()
    run_db_sql("CREATE DATABASE curation", db)
    run_db_sql("CREATE TABLE curation.notes (obj, note)", db)
    run_db_sql("INSERT INTO curation.notes (obj, note) VALUES ('gold', 'shiny')", db)
    res = run_db_sql("SELECT obj, note FROM curation.notes", db)
    assert res[0]["obj"] == "gold" and res[0]["note"] == "shiny"
    with pytest.raises(QueryError):
        run_db_sql("INSERT INTO system.scene (id, name) VALUES (9, 'x')", db)

"""Regression traps for the paired query-layer defects (work plan items 5.1 + 5.2).

They land together because they COMPOUND: 5.2 manufactures a schema nobody can hit, and 5.1 then swallows
every INSERT that tries. Fixing either alone leaves a system that still cannot round-trip an ordinary
CREATE/INSERT pair, so the compound case is tested explicitly rather than only its halves.
"""
import pytest

import lecore
from holographic.agents_and_reasoning.holographic_query import QueryError, _column_name


@pytest.fixture
def db_mind():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    db = mind.database(dim=512)
    mind.db_query("CREATE DATABASE d", db)
    mind.db_query("CREATE TABLE d.t (a, b)", db)
    return mind, db


# --------------------------------------------------------------------------------------
# 5.1 -- the write path guarded the TIER but not the SCHEMA.
# --------------------------------------------------------------------------------------

def test_a_bad_column_insert_raises_instead_of_reporting_success(db_mind):
    mind, db = db_mind
    with pytest.raises(QueryError) as exc:
        mind.db_query("INSERT INTO d.t (zzz) VALUES (1)", db)
    assert "zzz" in str(exc.value) and "a, b" in str(exc.value)


def test_a_refused_insert_leaves_the_table_unchanged(db_mind):
    # The old behaviour appended a row of Nones AND reported _confidence 1.0 -- full confidence in an
    # empty row. Refusing is only half the fix; not writing is the other half.
    mind, db = db_mind
    with pytest.raises(QueryError):
        mind.db_query("INSERT INTO d.t (zzz) VALUES (1)", db)
    assert mind.db_query("SELECT * FROM d.t", db) == []


def test_a_partial_row_is_still_legal(db_mind):
    # Only UNKNOWN columns are refused. Requiring every column would have been a different, wrong fix.
    mind, db = db_mind
    assert mind.db_query("INSERT INTO d.t (a) VALUES (1)", db) == {"inserted": 1}
    row = mind.db_query("SELECT * FROM d.t", db)[0]
    assert row["a"] == 1 and row["b"] is None


def test_engine_internal_columns_are_exempt(db_mind):
    """Found by the standalone-database integration test, which failed on `_deleted` -- the tombstone the
    DELETE path writes. `_confidence` is the same kind of column. These are not user schema and are not
    declared in CREATE TABLE, so the first version of this validator refused the engine's own writes."""
    mind, db = db_mind
    mind.db_query("INSERT INTO d.t (a) VALUES (1)", db)
    assert mind.db_query("DELETE FROM d.t WHERE a = 1", db)["deleted"] == 1


def test_the_read_path_message_is_unchanged(db_mind):
    # The read path always had the right check and the right message; the fix shares it rather than
    # inventing a second wording that could drift.
    mind, db = db_mind
    with pytest.raises(QueryError) as read_exc:
        mind.db_query("SELECT zzz FROM d.t", db)
    with pytest.raises(QueryError) as write_exc:
        mind.db_query("INSERT INTO d.t (zzz) VALUES (1)", db)
    assert str(read_exc.value) == str(write_exc.value)


# --------------------------------------------------------------------------------------
# 5.2 -- CREATE TABLE kept the type token in the column NAME.
# --------------------------------------------------------------------------------------

def test_type_annotations_are_stripped_from_column_names(db_mind):
    mind, db = db_mind
    mind.db_query("CREATE DATABASE lab", db)
    mind.db_query("CREATE TABLE lab.runs (id INT, name TEXT, score FLOAT)", db)
    mind.db_query("INSERT INTO lab.runs (id, name) VALUES (1, 'x')", db)
    cols = [k for k in mind.db_query("SELECT * FROM lab.runs", db)[0] if not k.startswith("_")]
    assert cols == ["id", "name", "score"], cols


def test_only_recognised_type_tokens_are_stripped():
    # Better to keep a strange column name than to truncate a deliberate one -- and with 5.1 fixed, an
    # unrecognised name is now reported plainly the first time an INSERT misses it.
    assert _column_name("id INT") == "id"
    assert _column_name("score FLOAT") == "score"
    assert _column_name("name VARCHAR(32)") == "name"
    assert _column_name("plain") == "plain"
    assert _column_name("two words") == "two words"        # not a known type -> left alone


def test_case_is_ignored_on_the_type_token():
    assert _column_name("id int") == "id"
    assert _column_name("id Int") == "id"


# --------------------------------------------------------------------------------------
# The compounding -- the reason these land together.
# --------------------------------------------------------------------------------------

def test_the_compound_case_round_trips(db_mind):
    """THE POINT OF PAIRING THEM. Before: CREATE made a column called "id INT", so INSERT (id) missed it,
    and 5.1 reported success while writing Nones. Either fix alone leaves this broken -- with only 5.1 the
    insert raises on a schema the user cannot satisfy; with only 5.2 nothing catches the next mismatch."""
    mind, db = db_mind
    mind.db_query("CREATE DATABASE lab", db)
    mind.db_query("CREATE TABLE lab.runs (id INT, score FLOAT)", db)
    mind.db_query("INSERT INTO lab.runs (id) VALUES (7)", db)
    assert mind.db_query("SELECT id FROM lab.runs", db) == [{"id": 7, "_confidence": 1.0}]

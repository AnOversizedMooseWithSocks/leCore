"""Query BUILD B2: UPDATE / DELETE via append-only tombstone + compaction."""
from holographic_query import UserTable, run_sql, delete, update, compact


def _t():
    t = UserTable("animals", ["name", "legs"], dim=1024, seed=0)
    for r in [{"name": "cat", "legs": 4}, {"name": "bird", "legs": 2}, {"name": "ant", "legs": 6}, {"name": "crab", "legs": 8}]:
        t.insert(r)
    return t


def test_delete_tombstones_and_scans_skip():
    t = _t()
    assert delete(t, "legs > 6") == 1                         # crab
    assert {r["name"] for r in run_sql("SELECT name FROM animals", t)} == {"cat", "bird", "ant"}


def test_delete_multiple():
    t = _t()
    assert delete(t, "legs > 3") == 3                         # cat, ant, crab
    assert {r["name"] for r in run_sql("SELECT name FROM animals", t)} == {"bird"}


def test_update_retires_old_appends_new():
    t = _t()
    assert update(t, "name = 'bird'", {"legs": 3}) == 1
    assert run_sql("SELECT legs FROM animals WHERE name = 'bird'", t)[0]["legs"] == 3
    assert run_sql("SELECT name FROM animals WHERE legs = 2", t) == []   # old version gone from scans


def test_compact_reclaims_dead_rows():
    t = _t()
    delete(t, "legs > 6")
    update(t, "name = 'bird'", {"legs": 3})
    assert len(t.rows) == 5                                   # 4 original + 1 new bird (2 tombstoned)
    t2 = compact(t)
    assert len(t2.rows) == 3                                  # only the live rows survive
    assert {r["name"] for r in run_sql("SELECT name FROM animals", t2)} == {"cat", "ant", "bird"}


def test_deleted_row_not_returned_by_fuzzy_or_where():
    t = _t()
    delete(t, "name = 'cat'")
    assert run_sql("SELECT name FROM animals WHERE legs = 4", t) == []   # cat gone
    assert all(r["name"] != "cat" for r in run_sql("SELECT name FROM animals", t))


def test_update_then_delete_the_new_version():
    t = _t()
    update(t, "name = 'ant'", {"legs": 100})
    assert delete(t, "legs = 100") == 1                       # can delete the updated version
    assert run_sql("SELECT name FROM animals WHERE name = 'ant'", t) == []

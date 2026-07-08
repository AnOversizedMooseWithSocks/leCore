"""Query BUILD B8: concurrency -- single-writer lock + snapshot readers."""
import threading
import pytest
from holographic.agents_and_reasoning.holographic_query import UserTable, run_sql, QueryError
from holographic.agents_and_reasoning.holographic_query_concurrency import write_lock, snapshot_reader


def _t():
    t = UserTable("ledger", ["id", "amt"], dim=256, seed=0)
    t.insert({"id": "a", "amt": 1})
    return t


def test_write_lock_mutually_exclusive():
    t = _t()
    with write_lock(t):
        with pytest.raises(QueryError):
            with write_lock(t, blocking=False):
                pass                                            # second writer refused while held


def test_write_lock_free_after_release():
    t = _t()
    with write_lock(t):
        pass
    with write_lock(t, blocking=False):                         # released -> acquirable again
        pass


def test_snapshot_reader_isolated_from_later_writes():
    t = _t()
    snap = snapshot_reader(t)
    t.insert({"id": "b", "amt": 2})
    assert {r["id"] for r in run_sql("SELECT id FROM ledger", snap)} == {"a"}       # froze
    assert {r["id"] for r in run_sql("SELECT id FROM ledger", t)} == {"a", "b"}     # live grew


def test_snapshot_reader_carries_pk_index():
    t = UserTable("u", ["uid"], dim=256, seed=0).set_primary_key("uid")
    t.insert({"uid": "u1"})
    snap = snapshot_reader(t)
    assert run_sql("SELECT uid FROM u WHERE uid = 'u1'", snap)[0]["uid"] == "u1"     # pk fast path works on the snap


def test_threaded_writers_no_lost_update():
    t = _t()

    def worker(tag):
        for k in range(5):
            with write_lock(t):
                t.insert({"id": "%s%d" % (tag, k), "amt": k})

    threads = [threading.Thread(target=worker, args=(tag,)) for tag in ("x", "y")]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(run_sql("SELECT id FROM ledger", t)) == 11       # a + 5 + 5, none lost

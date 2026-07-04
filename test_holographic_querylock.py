"""Tests for holographic_querylock (B8 single-writer lock + snapshot readers)."""
from holographic_query import Database, update
from holographic_querylock import SingleWriterLock, ConcurrencyError


def _db():
    db = Database(); db.add_namespace("user")
    db.create_table("user.acct", ["id", "balance"], dim=256, seed=0)
    t = db.namespaces["user"]["tables"]["acct"]; t.set_primary_key("id")
    t.insert({"id": 1, "balance": 100})
    return t


def test_single_writer_serialises():
    t = _db(); lock = SingleWriterLock()
    with lock.write():
        assert lock.held()
        try:
            with lock.write(block=False): assert False        # second concurrent writer refused
        except ConcurrencyError:
            pass
    assert not lock.held()                                     # released on exit


def test_snapshot_is_consistent():
    t = _db(); lock = SingleWriterLock()
    snap = lock.snapshot(t)
    with lock.write():
        update(t, "id = 1", {"balance": 250})
    assert snap.rows()[0]["balance"] == 100                    # the reader's view didn't shift
    assert lock.snapshot(t).rows()[0]["balance"] == 250        # a fresh read sees the write


def test_blocking_acquire_after_release():
    t = _db(); lock = SingleWriterLock()
    with lock.write():
        update(t, "id = 1", {"balance": 200})
    with lock.write(block=True):                               # succeeds now that the first released
        update(t, "id = 1", {"balance": 300})
    assert lock.snapshot(t).rows()[0]["balance"] == 300


def test_snapshot_multiple_tables():
    t = _db(); lock = SingleWriterLock()
    snaps = lock.snapshot(t, t)
    assert isinstance(snaps, list) and len(snaps) == 2

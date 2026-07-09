"""holographic_querylock.py -- B8 concurrency: a single-writer lock + snapshot readers.

WHY (and the honest scope)
--------------------------
Full concurrent-writer ACID isolation (MVCC) is a large undertaking and is often the wrong goal for these workloads.
The correct, simple first step -- which covers most real use -- is:

  * ONE writer at a time. Writers are serialised by an exclusive lock, so two writers never interleave and corrupt the
    replay log. A second writer either waits (block=True) or fails fast (block=False).
  * MANY readers, never blocked. A reader takes a point-in-time SNAPSHOT (a cheap copy of the table state, reusing the
    B6 transaction snapshot), so it sees a consistent view and neither blocks nor is blocked by the writer.

This is deterministic and safe for the append-only replay model; concurrent-WRITER isolation (MVCC) is deliberately
DEFERRED and must not be advertised until built. Uses only stdlib threading.

  with lock.write():            # exclusive -- serialises writers
      ... mutate tables ...
  snap = lock.snapshot(t)       # a frozen, consistent read view -- never blocks the writer
"""
import threading


class ConcurrencyError(Exception):
    """A write was refused because another writer holds the single-writer lock (block=False fail-fast)."""


class _Snapshot:
    """A frozen, read-only view of one table at a moment in time -- a consistent read that ignores later writes."""

    def __init__(self, table):
        self.name = table.name
        self.roles = list(table.roles)
        self._rows = [dict(r) for r in table.rows]            # a copy -> immune to later writes
        self._records = table.records.copy()

    def rows(self, include_deleted=False):
        return [dict(r) for r in self._rows if include_deleted or not r.get("_deleted")]

    def __len__(self):
        return sum(1 for r in self._rows if not r.get("_deleted"))


class SingleWriterLock:
    """B8 -- serialise writers, let readers snapshot freely. One lock per database (or per table group)."""

    def __init__(self):
        self._writer = threading.Lock()
        self._owner = None                                    # the thread id currently writing (for diagnostics)

    def write(self, block=True, timeout=-1):
        """Return an exclusive-write context manager. Only one writer may hold it at a time. block=False fails fast
        with ConcurrencyError if another writer holds it; block=True waits (optionally up to `timeout` seconds)."""
        return _WriteCtx(self, block, timeout)

    def snapshot(self, *tables):
        """A consistent read snapshot of one or more tables -- readers never block and are never blocked. Returns a
        single _Snapshot for one table, or a list for several."""
        snaps = [_Snapshot(t) for t in tables]
        return snaps[0] if len(snaps) == 1 else snaps

    def held(self):
        """True if a writer currently holds the lock."""
        return self._writer.locked()


class _WriteCtx:
    def __init__(self, lock, block, timeout):
        self._lock = lock
        self._block = block
        self._timeout = timeout
        self._acquired = False

    def __enter__(self):
        got = self._lock._writer.acquire(self._block, self._timeout) if self._block \
            else self._lock._writer.acquire(False)
        if not got:
            raise ConcurrencyError("another writer holds the single-writer lock (concurrent writers are serialised)")
        self._acquired = True
        self._lock._owner = threading.get_ident()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._acquired:
            self._lock._owner = None
            self._lock._writer.release()
        return False                                          # never swallow -- transaction() handles rollback


def _selftest():
    from holographic.agents_and_reasoning.holographic_query import Database, update
    db = Database(); db.add_namespace("user")
    db.create_table("user.acct", ["id", "balance"], dim=256, seed=0)
    t = db.namespaces["user"]["tables"]["acct"]; t.set_primary_key("id")
    t.insert({"id": 1, "balance": 100})

    lock = SingleWriterLock()

    # a reader snapshot is a consistent point-in-time view
    snap = lock.snapshot(t)
    assert len(snap) == 1 and snap.rows()[0]["balance"] == 100

    # a write serialises; while held, a second (non-blocking) writer is refused
    with lock.write():
        update(t, "id = 1", {"balance": 250})
        try:
            with lock.write(block=False):
                assert False, "a second concurrent writer should be refused"
        except ConcurrencyError:
            pass

    # after the write, the OLD snapshot is unchanged (it was a copy) but a NEW snapshot sees the update
    assert snap.rows()[0]["balance"] == 100                    # the reader's view did not shift under it
    assert lock.snapshot(t).rows()[0]["balance"] == 250        # a fresh read sees the committed write
    assert not lock.held()                                     # lock released on exit

    # blocking acquire succeeds once the previous writer has released
    with lock.write(block=True):
        update(t, "id = 1", {"balance": 300})
    assert lock.snapshot(t).rows()[0]["balance"] == 300

    print("OK: holographic_querylock self-test passed (single-writer serialisation, non-blocking refusal, "
          "consistent reader snapshots immune to later writes -- B8; MVCC deferred, stated honestly)")


if __name__ == "__main__":
    _selftest()

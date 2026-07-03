"""holographic_query_concurrency.py -- CONCURRENCY for the query layer (query backlog B8).

The honest model for a NumPy-only, deterministic store is single-writer / multi-reader:

  * a WRITE LOCK -- at most one writer touches a table at a time, so two writers can't interleave and lose an update;
  * SNAPSHOT READERS -- a reader takes a cheap point-in-time COPY of the table and queries that, so it neither blocks
    the writer nor sees the writer's in-flight changes (readers never wait for writers, writers never wait for readers).

This is the classic MVCC-lite arrangement in its smallest form, built on the snapshot the transaction layer (B6)
already uses. It gives consistent reads and prevents lost updates without a heavyweight lock manager.

KEPT NEGATIVES (loud): this is SINGLE-writer (serialised writers), not multi-writer serialisable isolation -- two
writers run one after another, they do not merge. A snapshot reader is a COPY, so it costs memory proportional to the
table and does not see writes made after it was taken (that is the point -- a stable read). The lock is an in-process
threading.Lock (one process, many threads), not a cross-process file lock; durability across processes is B7's job.
"""
import threading

from holographic_query import UserTable, QueryError


_locks = {}                                                      # id(table) -> its write lock (created on demand)
_locks_guard = threading.Lock()                                  # protects the _locks dict itself


def _lock_for(table):
    key = id(table)
    with _locks_guard:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


class write_lock:
    """A single-writer lock over a table. Only one writer holds it at a time; a second writer blocks (or, with
    blocking=False, fails fast with a QueryError) until the first releases. Use it around a write section:

        with write_lock(table):
            table.insert(row)
    """

    def __init__(self, table, blocking=True, timeout=-1):
        self._lock = _lock_for(table)
        self._blocking = blocking
        self._timeout = timeout
        self._held = False

    def __enter__(self):
        self._held = self._lock.acquire(self._blocking, self._timeout)
        if not self._held:
            raise QueryError("could not acquire the write lock (another writer holds this table)")
        return self

    def __exit__(self, *exc):
        if self._held:
            self._lock.release()
            self._held = False
        return False


def snapshot_reader(table):
    """Return a point-in-time READ view of a table: a cheap copy of its rows and record vectors as they are NOW. The
    copy is a normal queryable table, so run_sql / Query work on it unchanged -- but later writes to the original do
    NOT appear in it (a stable, consistent read that never blocks the writer)."""
    snap = UserTable(table.name, list(table.roles), dim=table.dim, seed=getattr(table, "seed", 0))
    snap.role_vocab = table.role_vocab                          # share the (immutable, deterministic) codebooks
    snap.value_vocab = table.value_vocab
    snap.records = table.records.copy()                         # freeze the vectors at this instant
    snap.rows = [dict(r) for r in table.rows]                   # and the stored values
    if getattr(table, "_pk", None) is not None:                # carry the pk index so pk lookups still work on the snap
        snap._pk = table._pk
        snap._pk_index = {k: list(v) for k, v in table._pk_index.items()}
    return snap


def _selftest():
    """The write lock is mutually exclusive (a second non-blocking acquire fails while held); a snapshot reader is
    isolated from later writes; and two threaded writers under the lock both land their writes with no lost update."""
    from holographic_query import run_sql

    t = UserTable("ledger", ["id", "amt"], dim=256, seed=0)
    t.insert({"id": "a", "amt": 1})

    # mutual exclusion: while one writer holds the lock, a second non-blocking acquire fails
    with write_lock(t):
        try:
            with write_lock(t, blocking=False):
                raise AssertionError("second writer acquired the lock while held")
        except QueryError:
            pass
    # ...and after release, the lock is free again
    with write_lock(t, blocking=False):
        pass

    # snapshot isolation: a reader taken now does not see a later insert
    snap = snapshot_reader(t)
    t.insert({"id": "b", "amt": 2})
    snap_names = {r["id"] for r in run_sql("SELECT id FROM ledger", snap)}
    live_names = {r["id"] for r in run_sql("SELECT id FROM ledger", t)}
    assert snap_names == {"a"} and live_names == {"a", "b"}      # the snapshot froze at one row

    # two threaded writers, each appending under the lock -> both writes survive (no lost update)
    def worker(tag):
        for k in range(5):
            with write_lock(t):
                t.insert({"id": "%s%d" % (tag, k), "amt": k})

    threads = [threading.Thread(target=worker, args=(tag,)) for tag in ("x", "y")]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    n = len(run_sql("SELECT id FROM ledger", t))
    assert n == 2 + 10, n                                        # a, b, plus 5 from each writer -- none lost

    print("holographic_query_concurrency selftest OK: write lock is mutually exclusive (second non-blocking acquire "
          "refused while held, free after release); a snapshot reader froze at 1 row while the live table grew to 2; "
          "two threaded writers under the lock landed all 10 inserts with no lost update")


if __name__ == "__main__":
    _selftest()

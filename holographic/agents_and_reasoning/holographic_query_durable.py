"""holographic_query_durable.py -- DURABILITY & CRASH RECOVERY for the query layer (query backlog B7).

The engine already has the whole mechanism: to_state / from_state serialise a database by REPLAY (columns, dim, seed,
rows) and rebuild it byte-identically because the seed fixes every atom. Durability is just two more moves on top of
that spine:

  * a SNAPSHOT -- write to_state to disk as JSON (small, deterministic, human-readable);
  * a REDO JOURNAL -- an append-only log of the writes made SINCE the last snapshot, flushed on every append.

Recovery after a crash is: load the snapshot, then replay the journal. That gives point-in-time recovery to the last
durably-logged write without snapshotting the whole database on every insert. This is the classic snapshot + WAL
(write-ahead log) design, in the smallest honest form.

KEPT NEGATIVES (loud): this logs LOGICAL operations (insert/update/delete) and replays them, so a journal entry must
be appended (and flushed) BEFORE the crash to be recoverable -- an in-memory write not yet journalled is lost, exactly
as a real WAL bounds durability by fsync. We flush per append; we do not implement group commit or O_DIRECT. JSON keeps
the format readable and stdlib-only (no pickle -- deterministic and safe). Row values must be JSON-friendly (strings /
numbers), which the query layer already guarantees.
"""
import io
import json
import os

from holographic.agents_and_reasoning.holographic_query import Database, run_sql, delete as _delete, update as _update


def save_snapshot(db, path, tiers=("persistent",)):
    """Write a durable snapshot of the database to `path` (JSON). By default only the PERSISTENT tier -- the durable
    user data -- is snapshotted; the mind's system tables are live state, not user data, and are never persisted."""
    state = db.to_state(tiers=list(tiers)) if tiers is not None else db.to_state()
    tmp = path + ".tmp"                                          # write-then-rename: never leave a half-written file
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())                                     # force the bytes to disk before we commit the rename
    os.replace(tmp, path)                                        # atomic on POSIX -- the snapshot appears all-at-once
    return path


def load_snapshot(path, system_tables=None):
    """Rebuild a Database from a snapshot file (deterministic replay). Re-attach live system tables if given."""
    with io.open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return Database.from_state(state, system_tables=system_tables)


class Journal:
    """An append-only REDO log of writes since the last snapshot. Each entry is one logical operation; append flushes
    it to disk immediately (so a crash loses only writes never journalled). replay(db) re-applies the log in order."""

    def __init__(self, path):
        self.path = path

    def _append(self, entry):
        with io.open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")                    # one JSON object per line (a readable WAL)
            f.flush()
            os.fsync(f.fileno())                                 # durable before we return

    # -- the logged write verbs (mirror the query layer's own) --
    def log_insert(self, qualified, row):
        self._append({"op": "insert", "table": qualified, "row": row})

    def log_update(self, qualified, where, changes):
        self._append({"op": "update", "table": qualified, "where": where, "changes": changes})

    def log_delete(self, qualified, where):
        self._append({"op": "delete", "table": qualified, "where": where})

    def truncate(self):
        """Clear the journal -- called right after a fresh snapshot folds the log into the durable base."""
        if os.path.exists(self.path):
            os.remove(self.path)

    def entries(self):
        """The logged operations in order (empty if the journal does not exist)."""
        if not os.path.exists(self.path):
            return []
        out = []
        with io.open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def replay(self, db):
        """Re-apply every logged operation to `db`, in order -- the recovery step after loading a snapshot."""
        for e in self.entries():
            tbl = db.resolve(e["table"])
            if e["op"] == "insert":
                tbl.insert(e["row"])
            elif e["op"] == "update":
                _update(tbl, e["where"], e["changes"])
            elif e["op"] == "delete":
                _delete(tbl, e["where"])
        return db


def recover(snapshot_path, journal_path, system_tables=None):
    """Full crash recovery: load the last snapshot, then replay the redo journal on top of it."""
    db = load_snapshot(snapshot_path, system_tables=system_tables)
    Journal(journal_path).replay(db)
    return db


def _selftest():
    """Snapshot a database, journal a few more writes, then recover (load snapshot + replay journal) and confirm the
    recovered database has BOTH the snapshotted rows and the journalled ones -- point-in-time recovery."""
    import tempfile

    d = tempfile.mkdtemp()
    snap = os.path.join(d, "db.snapshot.json")
    jrnl = os.path.join(d, "db.journal")

    db = Database()
    db.create_namespace("shop", tier="persistent")
    db.create_table("shop.items", ["name", "qty"], dim=256, seed=0)
    db.insert("shop.items", {"name": "apple", "qty": 3})
    db.insert("shop.items", {"name": "pear", "qty": 5})

    # durable snapshot of what we have so far
    save_snapshot(db, snap)

    # further writes go through the journal (as they would in a live durable store) AND to the live db
    j = Journal(jrnl)
    db.insert("shop.items", {"name": "plum", "qty": 2}); j.log_insert("shop.items", {"name": "plum", "qty": 2})
    _update(db.resolve("shop.items"), "name = 'apple'", {"qty": 9}); j.log_update("shop.items", "name = 'apple'", {"qty": 9})

    # simulate a crash: throw the live db away and RECOVER from disk
    recovered = recover(snap, jrnl)
    rows = {r["name"]: r["qty"] for r in run_sql("SELECT name, qty FROM items", recovered.resolve("shop.items"))}
    assert rows == {"apple": 9, "pear": 5, "plum": 2}, rows       # snapshot rows + journalled insert + journalled update

    # a fresh snapshot folds the journal into the base; truncating the log is then safe
    save_snapshot(recovered, snap)
    j.truncate()
    assert Journal(jrnl).entries() == []
    again = recover(snap, jrnl)
    assert {r["name"] for r in run_sql("SELECT name FROM items", again.resolve("shop.items"))} == {"apple", "pear", "plum"}

    print("holographic_query_durable selftest OK: snapshot+journal recovers apple(qty 9 via journalled update), pear, "
          "plum(via journalled insert) after a simulated crash; re-snapshot folds the log and truncate is safe; "
          "atomic write-then-rename + fsync; deterministic replay")


if __name__ == "__main__":
    _selftest()

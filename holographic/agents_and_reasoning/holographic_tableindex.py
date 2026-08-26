"""Secondary indexes and tombstone reclamation for a UserTable.

EXTRACTED, NOT INVENTED. These five methods grew inside holographic_query.py over
a dozen storage sweeps -- create_index/index_lookup/index_range (a hash index on
ANY column, not just the primary key), and vacuum/dead_fraction (UPDATE here is
tombstone-and-reinsert, so a written table only ever GROWS). They pushed that file
past 2,000 lines and made it the SIXTH giant module against a budget of five, which
structure_audit fails CI over.
They come out cleanly because they were always a coherent unit: everything here
touches self.rows, self.roles and self._sec_index and nothing else. A MIXIN rather
than a helper module, so UserTable keeps them as METHODS -- the callers, the
catalog entries and the SELECT fast path all name them that way, and moving them
must not change a single call site.

MEASURED, and kept with the code that earns it:
    WHERE v = 42 at 16,000 rows   2.511 -> 0.073 ms with an index   (34x)
    v > 990 through sorted keys   3.654 -> 0.074 ms                 (49x)
    vacuum after 5 update rounds  1,260 -> 400 rows, 860 reclaimed
"""

import numpy as np

from holographic.agents_and_reasoning.holographic_query import QueryError, _encode_row


class TableIndexMixin:
    """create_index / index_lookup / index_range / vacuum / dead_fraction."""

    def vacuum(self, dry_run=False):
        """Drop tombstoned rows for good, and rebuild the indexes over what is left.

        UPDATE here is TOMBSTONE-AND-REINSERT, which is what keeps the indexes
        correct without an update hook -- the old row is marked `_deleted` and a new
        one appends, so the insert path maintains everything. The cost is that the
        table only ever GROWS: five successive updates over 400 rows took it to
        1,260 rows and 1,260 index entries, none of them reclaimable.
        MEASURED, and the shape is the doctrine-duplication bug again -- correct
        behaviour whose cost outlives the operation, with no repair path. Every
        SELECT then scans dead rows, every index bucket carries dead indices, and
        the only way back was to rebuild the table by hand.

        Returns {before, after, removed, dry_run}. `dry_run=True` measures without
        touching anything. Rebuilds `records` and every index from the surviving
        rows, so it is safe to call at any time -- the encoding is derived, not
        stored (see the cold-storage note for the same lever)."""
        live = [r for r in self.rows if not r.get("_deleted")]
        rep = {"before": len(self.rows), "after": len(live),
               "removed": len(self.rows) - len(live), "dry_run": bool(dry_run)}
        if dry_run or rep["removed"] == 0:
            return rep
        self.rows = [dict(r) for r in live]
        self.records = (np.vstack([
            _encode_row(r, self.roles, self.role_vocab, self.value_vocab,
                        self.dim)[None, :] for r in self.rows])
            if self.rows else np.zeros((0, self.dim)))
        self._rec_cap, self._rec_n = None, 0
        if self._pk is not None:                       # indices moved -- rebuild
            self._pk_index = {}
            for i, r in enumerate(self.rows):
                self._pk_index.setdefault(r.get(self._pk), []).append(i)
        for col in list(self._sec_index):
            self.create_index(col)
        return rep

    def dead_fraction(self):
        """What share of this table's rows are tombstones. 0.0 for a fresh table.

        The number a vacuum decision should be made on, exposed so a caller can
        make it themselves rather than guessing from row counts."""
        if not self.rows:
            return 0.0
        return sum(1 for r in self.rows if r.get("_deleted")) / float(len(self.rows))

    def create_index(self, col):
        """A hash index on ANY column, not just the primary key.

        MEASURED at 16,000 rows: `WHERE id = 9999` on a primary key is 0.064 ms and
        `WHERE v = 42` on an unindexed column is 2.623 ms -- A 41x GAP, and the
        second is a full scan that grows with the table while the first does not.
        The machinery for this ALREADY EXISTED and served exactly one column: the
        same dict-of-value-to-row-indices, built the same way, maintained on the
        same insert path. THIS ADDS NO NEW IDEA, it removes an arbitrary limit of
        one -- which is why it is a dozen lines rather than an index subsystem.

        Unlike a primary key this implies NO constraint: no NOT NULL, no UNIQUE.
        An index is a statement about lookup cost, not about the data.
        Rebuilt from `rows` here, so it is correct on an already-populated table.
        """
        if col not in self.roles:
            raise QueryError("cannot index %r: not a column" % col)
        # DO NOT SHADOW THE PRIMARY KEY. The pk already has an identical hash
        # index and the planner checks the pk fast path FIRST, so a secondary
        # index on the same column can never be read. MEASURED at 8,000 unique
        # values: the pk index is 71.5 KB and this built a BYTE-FOR-BYTE
        # DUPLICATE of it -- 100% overhead for zero benefit, silently.
        # Returning self rather than raising: asking to index a column that is
        # already indexed is not an error, it is already true.
        if col == self._pk:
            return self
        idx = {}
        for i, r in enumerate(self.rows):
            idx.setdefault(r.get(col), []).append(i)
        self._sec_index[col] = idx
        return self

    def index_range(self, col, op, value):
        """Live row indices for `col <op> value` using a secondary index, or None.

        A HASH INDEX ANSWERS RANGES TOO -- by sorting its KEYS, not the rows. The
        equality path was 0.023 ms at 16,000 rows while `v > 990` cost 3.435 ms
        returning 96 rows: A 150x GAP for a query that touched less data.
        Sorting the distinct keys is O(K log K) in the number of DISTINCT VALUES,
        which is what makes this worth doing -- a column with 997 distinct values
        over 16,000 rows sorts 997 things to avoid scanning 16,000.
        Returns None for an unindexed column, same contract as index_lookup: [] is
        "indexed, nothing matched" and None is "no index, go and scan".
        Mixed-type columns are refused (returns None -> scan) because Python 3
        will not order int against str and a TypeError here would be a crash where
        a slower correct answer was available."""
        idx = self._sec_index.get(col)
        if idx is None or op not in ("<", "<=", ">", ">="):
            return None
        keys = [k for k in idx if k is not None]
        try:
            keys.sort()
        except TypeError:
            return None                    # mixed types -> fall back to the scan
        import bisect
        if op in ("<", "<="):
            hi = bisect.bisect_right(keys, value) if op == "<=" else bisect.bisect_left(keys, value)
            hit = keys[:hi]
        else:
            lo = bisect.bisect_right(keys, value) if op == ">" else bisect.bisect_left(keys, value)
            hit = keys[lo:]
        out = []
        for k in hit:
            out.extend(i for i in idx[k] if not self.rows[i].get("_deleted"))
        out.sort()
        return out

    def index_lookup(self, col, value):
        """Live row indices for `col == value`, or None when the column is not indexed.

        RETURNS None RATHER THAN [] FOR AN UNINDEXED COLUMN, and the distinction is
        load-bearing: [] means "indexed, no matches" and None means "no index, go
        and scan". Collapsing them would silently turn every unindexed WHERE into
        a query that matches nothing."""
        idx = self._sec_index.get(col)
        if idx is None:
            return None
        return [i for i in idx.get(value, []) if not self.rows[i].get("_deleted")]


def _selftest():
    import lecore

    m = lecore.UnifiedMind(dim=64, seed=0)
    db = m.database(dim=128, seed=0)
    db.create_namespace("a")
    db.create_table("a.t", ["id", "v"])
    t = db.resolve("a.t")
    for i in range(300):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i % 7), db)

    # the exact numeric contract, not "no exception"
    scan = len(m.db_query("SELECT * FROM a.t WHERE v = 3", db))
    t.create_index("v")
    assert len(m.db_query("SELECT * FROM a.t WHERE v = 3", db)) == scan, "index changed the answer"
    assert t.index_lookup("id", 1) is None, "unindexed column must report None, not []"
    assert t.index_lookup("v", 3) is not None and len(t.index_lookup("v", 3)) == scan

    rng_scan = len(m.db_query("SELECT * FROM a.t WHERE v > 4", db))
    assert len(t.index_range("v", ">", 4)) == rng_scan, "indexed range != scan"

    m.db_query("UPDATE a.t SET v = 5 WHERE v = 3", db)
    assert t.dead_fraction() > 0, "tombstones not counted"
    before = len(m.db_query("SELECT * FROM a.t WHERE v = 5", db))
    rep = t.vacuum()
    assert rep["removed"] > 0 and t.dead_fraction() == 0.0, rep
    assert len(m.db_query("SELECT * FROM a.t WHERE v = 5", db)) == before, \
        "vacuum moved rows and an index was not rebuilt"
    print("holographic_tableindex selftest OK")


if __name__ == "__main__":
    _selftest()

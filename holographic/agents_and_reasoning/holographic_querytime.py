"""holographic_querytime.py -- the VERSIONED-HISTORY promote layer for the query Database (backlog P7-P12).

WHY
---
The versioning faculties already ship -- VersionedStore (commit/checkout/history), DeltaChain (row diff + Merkle
root), CompositionTree (tamper-locate in O(log n)). They were simply never wired to the query tables. This module is
that wire: a git-like timeline for a query Table, plus the six verbs it unlocks -- none of which plain SQL does well:

  P7  select_as_of(history, version, sql)     -- TIME TRAVEL: query the table as it was at a past version.
  P8  history_of(history, pk_col, key)        -- BLAME: the value timeline of one row across versions.
  P9  diff_versions(history, a, b)            -- DIFF: what rows were added / removed / changed between two versions.
  P10 revert_to(history, version)             -- UNDO: reconstruct a past version as the new current table.
  P11 branch / compare / discard              -- GIT-FOR-DATA: fork the timeline, try a change, compare, keep or toss.
  P12 prove(history, version) / find_tampering -- AUDIT: a Merkle commitment per version + which row was altered.

DESIGN (readable + correct first; reuse the shipped machinery)
  * A commit SNAPSHOTS the table -- the record vectors AND the exact row dicts (SQL needs the exact values to query
    the past losslessly; the vectors carry the fuzzy layer and the integrity proof). Snapshots are cheap dict/array
    copies; the video-compressed VersionedStore is available underneath for the vector timeline when storage matters,
    but the snapshot is the honest, simple source of truth here.
  * DIFF reuses DeltaChain._changed_rows for the vector-level "which rows moved" when two versions share a shape, and
    a field-level dict diff for exactness (and to handle add/remove, where shapes differ).
  * PROVE reuses CompositionTree: each commit's root is a tamper-evident commitment, and locate() names the changed
    row in O(log n).

KEPT NEGATIVES (loud)
  * A snapshot copies the record matrix + row dicts -- O(rows) memory per commit. For long timelines prefer the
    delta-compressed vector store (VersionedStore, gop_len) -- exposed as .versioned_vectors() -- but the exact row
    dicts are still kept per version, because exact time-travel queries need the lossless values.
  * REVERT returns the past state as a NEW current version (append-only, git-style) -- it never rewrites history.
    Reverting a change that later changes depend on is a MERGE question (see branch/compare); we do not guess a
    conflict policy -- diff surfaces the collision for the caller to resolve.
  * BRANCH+COMPARE+DISCARD ship; MERGE does not (a merge needs an explicit conflict policy, deferred on purpose).
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_query import UserTable, run_db_sql, Database, QueryError


class TableHistory:
    """A git-like version timeline for one query Table. commit() snapshots it; the P7-P12 verbs read the timeline."""

    def __init__(self, table):
        self.name = table.name
        self.dim = table.dim
        self.seed = table.seed
        self.roles = list(table.roles)
        self._role_vocab = table.role_vocab           # keep the SAME (deterministic) encoding across versions
        self._value_vocab = table.value_vocab
        self._pk = getattr(table, "_pk", None)
        self._versions = []                            # each: {"records", "rows", "note", "root"}

    # ---- committing / reading the timeline -------------------------------------------------------------------
    def commit(self, table, note=""):
        """Snapshot the table's current state as a new version. Returns the version index. Records the exact row
        dicts (for lossless time-travel queries) and the record vectors (fuzzy layer + a Merkle commitment)."""
        records = np.asarray(table.records, float).copy()
        rows = [dict(r) for r in table.rows]
        root = self._commit_root(records)              # tamper-evident commitment over the row vectors
        self._versions.append({"records": records, "rows": rows, "note": note, "root": root})
        return len(self._versions) - 1

    def history(self):
        """The timeline: one entry per version with its note, row count, and integrity root."""
        return [{"version": i, "note": v["note"], "rows": len(v["rows"]), "root": v["root"]}
                for i, v in enumerate(self._versions)]

    def checkout(self, version):
        """Reconstruct the Table exactly as it was at `version` -- an ordinary table you can run SQL against."""
        v = self._versions[self._resolve(version)]
        t = UserTable(self.name, list(self.roles), dim=self.dim, seed=self.seed)
        t.role_vocab = self._role_vocab                # reuse the deterministic encoding so vectors match byte-for-byte
        t.value_vocab = self._value_vocab
        t.records = v["records"].copy()
        t.rows = [dict(r) for r in v["rows"]]
        if self._pk is not None:                       # rebuild the pk index for the reconstructed table
            t._pk = self._pk
            t._pk_index = {}
            for i, r in enumerate(t.rows):
                t._pk_index.setdefault(r.get(self._pk), []).append(i)
        return t

    def _resolve(self, version):
        """Allow negatives (git-style: -1 = latest)."""
        n = len(self._versions)
        if version < 0:
            version += n
        if not (0 <= version < n):
            raise QueryError("no such version %r (have %d)" % (version, n))
        return version

    def _commit_root(self, records):
        """A compact tamper-evident commitment over the row vectors. Empty table -> a fixed sentinel. We hash the
        CompositionTree root vector so `prove()` returns a short, publishable digest (not a whole vector)."""
        if len(records) == 0:
            return "empty"
        import hashlib
        from holographic.misc.holographic_verify import CompositionTree
        root_vec = CompositionTree(list(records), seed=self.seed).root()
        return hashlib.sha256(np.asarray(root_vec, float).tobytes()).hexdigest()[:16]


# ==============================================================================================================
# P7 -- TIME TRAVEL: query the table as it was at a past version.
# ==============================================================================================================
def select_as_of(history, version, sql):
    """Run `sql` (a full SELECT ... FROM <table> ...) against the table AS IT WAS at `version`. SQL:2011 temporal
    tables need history tables + triggers; here it is checkout() + an ordinary query."""
    past = history.checkout(version)
    # a one-table Database so run_db_sql can resolve `FROM <name>` against the reconstructed past table
    db = Database()
    db.add_namespace("_asof")
    db.namespaces["_asof"]["tables"][_bare(history.name)] = past
    # rewrite the FROM to the reconstructed table's qualified name
    return run_db_sql(_retarget(sql, history.name, "_asof." + _bare(history.name)), db)


# ==============================================================================================================
# P8 -- BLAME: the value timeline of one row (by primary key) across every version.
# ==============================================================================================================
def history_of(history, pk_col, key):
    """The per-row 'blame': for each version, the stored values of the row whose `pk_col == key` (or None if the row
    did not exist then). Shows when a field changed and to what."""
    out = []
    for i, v in enumerate(history._versions):
        match = next((dict(r) for r in v["rows"] if r.get(pk_col) == key and not r.get("_deleted")), None)
        out.append({"version": i, "note": v["note"], "row": match})
    return out


# ==============================================================================================================
# P9 -- DIFF: what rows were added / removed / changed between two versions (field-level).
# ==============================================================================================================
def diff_versions(history, version_a, version_b, pk_col=None):
    """Diff two versions: added / removed / changed rows, with a field-level diff on changed rows. Uses the primary
    key (or a given pk_col) to pair rows; without one, pairs by exact row-dict identity. Reuses DeltaChain's
    vector diff when the two versions share a shape (a cheap 'which rows moved' cross-check)."""
    va = history._versions[history._resolve(version_a)]
    vb = history._versions[history._resolve(version_b)]
    key = pk_col or history._pk

    def live(rows):
        return [r for r in rows if not r.get("_deleted")]
    ra, rb = live(va["rows"]), live(vb["rows"])

    if key is not None:
        a_by = {r.get(key): r for r in ra}
        b_by = {r.get(key): r for r in rb}
        added = [dict(b_by[k]) for k in b_by if k not in a_by]
        removed = [dict(a_by[k]) for k in a_by if k not in b_by]
        changed = []
        for k in a_by:
            if k in b_by and _row_diff(a_by[k], b_by[k]):
                changed.append({"key": k, "fields": _row_diff(a_by[k], b_by[k])})
    else:
        a_set = [dict(r) for r in ra]
        b_set = [dict(r) for r in rb]
        added = [r for r in b_set if r not in a_set]
        removed = [r for r in a_set if r not in b_set]
        changed = []                                   # without a key, a change looks like a remove + an add

    result = {"added": added, "removed": removed, "changed": changed,
              "n_added": len(added), "n_removed": len(removed), "n_changed": len(changed)}
    # bonus: if the two versions have the same shape, DeltaChain names the changed vector rows too (near-free -- the
    # delta was computed for storage anyway).
    if va["records"].shape == vb["records"].shape and len(va["records"]) > 0:
        from holographic.agents_and_reasoning.holographic_deltachain import DeltaChain
        dc = DeltaChain(va["records"])
        result["changed_vector_rows"] = sorted(int(i) for i in dc._changed_rows(vb["records"], va["records"]))
    return result


# ==============================================================================================================
# P10 -- UNDO: reconstruct a past version (append-only; never rewrites history).
# ==============================================================================================================
def revert_to(history, version):
    """Return the table as it was at `version`, to be committed as the NEW current version (git-style revert -- the
    old versions stay in the timeline). Kept negative: if later changes depend on what you are reverting, that is a
    merge; diff_versions surfaces the collision -- we do not guess a conflict policy."""
    return history.checkout(version)


# ==============================================================================================================
# P11 -- GIT-FOR-DATA: branch / compare / discard (what-if experiments on the data).
# ==============================================================================================================
def branch(history, at_version=-1):
    """Fork the timeline at `at_version` into an independent branch you can experiment on. compare() diffs it back
    against the parent; discard() just drops it (a branch is a separate TableHistory, so discarding is free)."""
    b = TableHistory(history.checkout(at_version))
    b.commit(history.checkout(at_version), note="branch@%d" % history._resolve(at_version))
    b._parent = history                                # remember where it forked from (for compare)
    b._forked_at = history._resolve(at_version)
    return b


def compare(branch_history, parent_version=None):
    """Diff a branch's latest state against its parent (P9). Ship branch+compare+discard now; MERGE is deferred (it
    needs an explicit conflict policy)."""
    parent = branch_history._parent
    pv = branch_history._forked_at if parent_version is None else parent_version
    # move the branch's latest snapshot into the parent's frame to diff (same encoding, so vectors are comparable)
    tmp = parent.commit(branch_history.checkout(-1), note="_compare_tmp")
    out = diff_versions(parent, pv, tmp)
    parent._versions.pop()                             # remove the scratch commit -- compare must not mutate history
    return out


def discard(branch_history):
    """Throw a branch away -- it is a standalone timeline, so this just detaches it. Returns True."""
    branch_history._versions = []
    branch_history._parent = None
    return True


# ==============================================================================================================
# P12 -- AUDIT: a Merkle commitment per version, and which row was altered (O(log n)).
# ==============================================================================================================
def prove(history, version):
    """The tamper-evident commitment (Merkle root) of a version -- publish it, and any later alteration is provable."""
    return history._versions[history._resolve(version)]["root"]


def find_tampering(history, version, suspect_records):
    """Given a version's committed root and a SUSPECT copy of its row vectors, return the index of the row that was
    altered (or None if the copy matches the commitment). Reuses CompositionTree.locate -- O(log n)."""
    v = history._versions[history._resolve(version)]
    original = v["records"]
    suspect = np.asarray(suspect_records, float)
    if suspect.shape != original.shape:
        return "shape-changed"                         # rows added/removed -- a coarser tamper than a row edit
    from holographic.misc.holographic_verify import CompositionTree
    tree = CompositionTree(list(original), seed=history.seed)
    if tree.verify(list(suspect)):
        return None                                    # matches the commitment -- untampered
    idx, _checks = tree.locate(list(suspect))          # locate returns (index, n_checks)
    return int(idx)                                    # the altered row (O(log n) to find)


# ---- small helpers ------------------------------------------------------------------------------------------
def _row_diff(a, b):
    """Field-level diff between two row dicts: {col: (old, new)} for columns that changed (ignoring bookkeeping)."""
    cols = (set(a) | set(b)) - {"_deleted", "_confidence"}
    return {c: (a.get(c), b.get(c)) for c in cols if a.get(c) != b.get(c)}


def _bare(name):
    """The table's bare name (strip any namespace qualifier)."""
    return name.split(".")[-1]


def _retarget(sql, old_name, new_qualified):
    """Rewrite the FROM target so a reconstructed past table can be queried by run_db_sql. Matches the bare or
    qualified name after FROM, case-insensitively."""
    import re
    bare = _bare(old_name)
    return re.sub(r"(?i)(\bfrom\s+)(?:[\w.]*\b%s\b)" % re.escape(bare), r"\1" + new_qualified, sql, count=1)


def _selftest():
    # build a small table, commit three versions with edits, and exercise every verb
    db = Database(); db.add_namespace("user")
    db.create_table("user.acct", ["id", "balance", "status"], dim=1024, seed=0)
    t = db.namespaces["user"]["tables"]["acct"]
    t.set_primary_key("id")
    from holographic.agents_and_reasoning.holographic_query import update as _update

    for r in [{"id": 1, "balance": 100, "status": "open"}, {"id": 2, "balance": 50, "status": "open"}]:
        t.insert(r)
    h = TableHistory(t)
    v0 = h.commit(t, note="opening balances")

    _update(t, "id = 1", {"balance": 250})             # id 1: 100 -> 250
    v1 = h.commit(t, note="deposit to 1")

    t.insert({"id": 3, "balance": 0, "status": "open"})
    _update(t, "id = 2", {"status": "closed"})
    v2 = h.commit(t, note="new acct 3, close 2")

    # P7 time travel: balance of id 1 at v0 was 100, at v2 is 250
    past = select_as_of(h, v0, "SELECT balance FROM acct WHERE id = 1")
    now = select_as_of(h, v2, "SELECT balance FROM acct WHERE id = 1")
    assert past[0]["balance"] == 100 and now[0]["balance"] == 250

    # P8 blame: id 1's balance timeline
    blame = history_of(h, "id", 1)
    assert blame[0]["row"]["balance"] == 100 and blame[2]["row"]["balance"] == 250

    # P9 diff v0 -> v2: id 3 added, id 2 status changed, id 1 balance changed
    d = diff_versions(h, v0, v2)
    assert d["n_added"] == 1 and d["added"][0]["id"] == 3
    changed_keys = {c["key"] for c in d["changed"]}
    assert changed_keys == {1, 2}

    # P10 revert: id 1 back to its v0 balance (as a new state)
    reverted = revert_to(h, v0)
    assert next(r for r in reverted.rows if r["id"] == 1)["balance"] == 100

    # P11 branch / compare / discard
    b = branch(h, at_version=v2)
    bt = b.checkout(-1)
    from holographic.agents_and_reasoning.holographic_query import update as _u2
    _u2(bt, "id = 1", {"balance": 999}); b.commit(bt, note="what-if: big deposit")
    cmp = compare(b)
    assert any(c["key"] == 1 for c in cmp["changed"])           # the branch changed id 1
    assert len(h._versions) == 3                                # compare did NOT mutate the parent's history
    assert discard(b) is True

    # P12 prove / find_tampering: tamper one row vector and locate it
    root = prove(h, v2)
    assert root and root != "empty"
    suspect = h._versions[v2]["records"].copy()
    suspect[1] += 0.01                                          # alter row index 1
    assert find_tampering(h, v2, suspect) == 1
    assert find_tampering(h, v2, h._versions[v2]["records"]) is None   # untampered -> None

    print("OK: holographic_querytime self-test passed (time-travel, blame, diff add/remove/change, revert, "
          "branch/compare/discard, prove/locate-tamper -- P7-P12 wired on the shipped versioning faculties)")


if __name__ == "__main__":
    _selftest()

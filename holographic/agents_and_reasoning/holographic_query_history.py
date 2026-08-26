"""holographic_query_history.py -- the VERSIONED query table (PROMOTE P7-P12).

SUPERSEDED BY holographic_querytime -- the wired version (it has the curated catalog home). This earlier implementation of the same
backlog item is kept for its tests but is intentionally NOT wired into any pipeline; use holographic_querytime instead.

The differentiated "time-travel / diff / branch / tamper-proof" family. None of it is new machinery: the versioning
engine (holographic_history.VersionedStore), the delta/diff + Merkle root (holographic_deltachain), and the tamper
locator (holographic_verify) all SHIP. They simply were not wired to the query Database. This module is the enabling
wire the backlog calls for -- do it once, and P7-P12 are thin verbs on top.

A VersionedTable wraps a UserTable and, on each commit, snapshots BOTH halves of the exact/fuzzy fork: the record
VECTORS go into a VersionedStore (delta-compressed, exactly recoverable), and the exact stored ROW-DICTS are kept
per version (they are the source of truth for exact predicates). So a past version can be checked out exactly and
then queried, diffed, blamed, reverted, branched, and proven -- things a plain SQL store makes miserable or can't do.

KEPT NEGATIVES (loud): diff/blame identify rows by a KEY column when given one (the honest way -- positions shift as
rows are added/removed); without a key they fall back to a multiset content diff (added/removed only, no "changed").
Revert is recorded as a new commit (history is never erased). A branch is an independent copy of the timeline; MERGE
is deliberately not built here (it needs an explicit conflict policy -- P11's kept negative). NumPy + stdlib only;
deterministic (hashlib for the proofs, never Python's hash()).
"""
import hashlib
import json

import numpy as np

from holographic.caching_and_storage.holographic_history import VersionedStore
from holographic.agents_and_reasoning.holographic_deltachain import merkle_root
from holographic.agents_and_reasoning.holographic_query import UserTable, run_sql


def _row_hash(row):
    """A stable content hash of one stored row-dict (hashlib, sorted keys -> deterministic)."""
    blob = json.dumps(row, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).digest()


class VersionedTable:
    """A UserTable with a committed, recoverable history. commit() snapshots the current state; every P7-P12 verb
    reads or forks that history."""

    def __init__(self, table):
        self.table = table
        self.store = VersionedStore(table.dim)      # the shipped versioned vector store (delta-compressed)
        self._dict_snaps = []                       # exact stored row-dicts per version (the exact-side snapshot)
        self._notes = []

    # ---- the enabling wire: commit + checkout ----
    def commit(self, note=""):
        """Snapshot the table's current state as a new version. The vectors go into the VersionedStore (its job);
        the exact row-dicts are kept beside it. Returns the version index."""
        n = len(self.table)
        rows = {i: self.table.records[i] for i in range(n)}
        order = list(range(n))
        v = self.store.commit(rows, order, note=note)
        self._dict_snaps.append([dict(r) for r in self.table.rows])
        self._notes.append(note)
        return v

    def checkout(self, version):
        """Rebuild a UserTable exactly as it was at `version` -- records from the VersionedStore, exact values from
        the dict snapshot. Shares the live table's (deterministic, additive) vocabularies so decode stays consistent."""
        if not 0 <= version < len(self._dict_snaps):
            raise IndexError("no version %d" % version)
        recs, order = self.store.checkout(version)
        dicts = self._dict_snaps[version]
        t = UserTable(self.table.name, list(self.table.roles), dim=self.table.dim, seed=self.table.seed)
        t.role_vocab = self.table.role_vocab        # additive & seed-deterministic -> a superset is safe to share
        t.value_vocab = self.table.value_vocab
        t.records = np.array([recs[i] for i in order]) if order else np.zeros((0, self.table.dim))
        t.rows = [dict(d) for d in dicts]
        return t

    def head(self):
        return len(self._dict_snaps) - 1

    # ---- P7: time-travel ----
    def select_as_of(self, sql, version):
        """P7 -- run an ordinary query against the table AS IT WAS at `version`. SQL:2011 temporal tables are poorly
        supported and painful; here it is checkout-then-query."""
        return run_sql(sql, self.checkout(version))

    # ---- P9: diff ----
    def diff(self, va, vb, key=None):
        """P9 -- what changed between two versions. With a `key` column, rows are matched by key so added / removed /
        CHANGED are all exact; without one, a multiset content diff reports added / removed only (a changed row shows
        as one removed + one added). Exact on the stored props -- the honest, lossless side of the fork."""
        a, b = self._dict_snaps[va], self._dict_snaps[vb]
        if key is not None:
            amap = {r.get(key): r for r in a}
            bmap = {r.get(key): r for r in b}
            added = [bmap[k] for k in bmap if k not in amap]
            removed = [amap[k] for k in amap if k not in bmap]
            changed = [{"key": k, "from": amap[k], "to": bmap[k]}
                       for k in amap if k in bmap and amap[k] != bmap[k]]
            return {"added": added, "removed": removed, "changed": changed}
        # keyless: multiset content diff
        from collections import Counter
        ac = Counter(tuple(sorted(r.items())) for r in a)
        bc = Counter(tuple(sorted(r.items())) for r in b)
        added = [dict(t) for t, c in (bc - ac).items() for _ in range(c)]
        removed = [dict(t) for t, c in (ac - bc).items() for _ in range(c)]
        return {"added": added, "removed": removed}

    # ---- P8: blame (one row's timeline) ----
    def history_of(self, key_value, key):
        """P8 -- the timeline of ONE row, matched by `key`. Returns [(version, row_or_None, note)] so you can see
        when it appeared, how it changed, and when it went away."""
        timeline = []
        for v, snap in enumerate(self._dict_snaps):
            match = next((r for r in snap if r.get(key) == key_value), None)
            timeline.append((v, match, self._notes[v]))
        return timeline

    # ---- P10: revert ----
    def revert(self, version):
        """P10 -- revert the LIVE table to a past version, recorded as a new commit (history is never erased). Reuses
        VersionedStore.rollback for the vectors and restores the exact dicts. Returns the new version index."""
        past = self.checkout(version)
        self.table.records = past.records.copy()
        self.table.rows = [dict(r) for r in past.rows]
        self.store.rollback(version)                # records the vector rollback in the store's history
        self._dict_snaps.append([dict(r) for r in self.table.rows])
        self._notes.append("revert to v%d" % version)
        return self.head()

    # ---- P11: branch / compare / discard (git-for-data) ----
    def branch(self):
        """P11 -- fork the data: an independent VersionedTable sharing history UP TO NOW, which can then diverge.
        Experiment on the branch, diff it against main, keep or discard it. (MERGE is intentionally not built -- it
        needs an explicit conflict policy.)"""
        b = VersionedTable(UserTable(self.table.name, list(self.table.roles), dim=self.table.dim, seed=self.table.seed))
        b.table.role_vocab = self.table.role_vocab
        b.table.value_vocab = self.table.value_vocab
        b.table.records = self.table.records.copy()
        b.table.rows = [dict(r) for r in self.table.rows]
        # copy the committed history so the branch shares the past but diverges going forward
        b.store = VersionedStore(self.table.dim)
        for snap in self._dict_snaps:
            b._dict_snaps.append([dict(r) for r in snap])
        b._notes = list(self._notes)
        # rebuild the branch's vector store by re-committing each snapshot's records (exact, deterministic)
        for v in range(len(self._dict_snaps)):
            recs, order = self.store.checkout(v)
            b.store.commit({i: recs[i] for i in order}, list(order), note=self._notes[v])
        return b

    # ---- P12: provable audit ----
    def prove(self, version=None):
        """P12 -- a Merkle root over a version's rows (reuses deltachain.merkle_root over hashlib row hashes). Two
        stores agree iff their roots match; a single altered row changes the root."""
        v = self.head() if version is None else version
        leaves = [_row_hash(r) for r in self._dict_snaps[v]]
        return merkle_root(leaves) if leaves else b""

    def find_tampering(self, claimed_rows, version=None):
        """P12 -- given a claimed set of rows for a version, return the INDICES whose content differs from the
        committed snapshot (which row was altered), in O(n). Empty list => the claim matches (no tampering)."""
        v = self.head() if version is None else version
        truth = self._dict_snaps[v]
        bad = []
        for i in range(max(len(truth), len(claimed_rows))):
            t = truth[i] if i < len(truth) else None
            c = claimed_rows[i] if i < len(claimed_rows) else None
            if t is None or c is None or _row_hash(t) != _row_hash(c):
                bad.append(i)
        return bad


def _selftest():
    """Commit a few versions of a table, then time-travel, blame, diff, revert, branch, and prove -- all recovering
    exactly. Deterministic."""
    t = UserTable("accounts", ["id", "name", "balance"], dim=1024, seed=0)
    t.insert({"id": 1, "name": "alice", "balance": 100})
    t.insert({"id": 2, "name": "bob", "balance": 50})
    vt = VersionedTable(t)
    v0 = vt.commit("initial")

    # change bob's balance and add carol
    t.rows[1]["balance"] = 75
    t.insert({"id": 3, "name": "carol", "balance": 200})
    v1 = vt.commit("bob paid, carol joined")

    # P7 time-travel: v0 had 2 rows, v1 has 3
    assert len(vt.select_as_of("SELECT id FROM accounts", v0)) == 2
    assert len(vt.select_as_of("SELECT id FROM accounts", v1)) == 3

    # P9 diff by key: carol added, bob changed
    d = vt.diff(v0, v1, key="id")
    assert [r["name"] for r in d["added"]] == ["carol"]
    assert d["changed"] and d["changed"][0]["key"] == 2 and d["changed"][0]["to"]["balance"] == 75

    # P8 blame: bob's balance over time
    tl = vt.history_of(2, key="id")
    assert tl[0][1]["balance"] == 50 and tl[1][1]["balance"] == 75

    # P12 prove: a tampered row is located; the root changes
    root0 = vt.prove(v0)
    tampered = [dict(r) for r in vt._dict_snaps[v1]]
    tampered[0]["balance"] = 999999
    assert vt.find_tampering(tampered, v1) == [0]
    assert vt.prove(v1) != merkle_root([_row_hash(r) for r in tampered])

    # P11 branch: fork, diverge, diff against main; main unaffected
    br = vt.branch()
    br.table.insert({"id": 4, "name": "dave", "balance": 10})
    br.commit("dave on the branch")
    assert len(br.diff(vt.head(), br.head(), key="id")["added"]) == 1     # dave only on the branch
    assert len(vt.table) == 3                                             # main is untouched

    # P10 revert: back to v0's 2 rows, recorded as a new version
    vt.revert(v0)
    assert len(vt.table) == 2 and vt.head() == 2

    print("holographic_query_history selftest OK: committed versions time-travel (v0=2 rows, v1=3), diff-by-key finds "
          "carol added + bob changed, blame tracks bob's balance 50->75, a tampered row is located at index 0 and "
          "changes the Merkle root, a branch diverges without touching main, and revert restores v0; deterministic")


if __name__ == "__main__":
    _selftest()

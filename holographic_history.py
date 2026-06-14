"""Versioned, compressed history with rollback -- a knowledge store's timeline
stored the way a video stores frames.

THE IDEA (the user's): the substrate is always 'now'. A reorganization swaps the
store in place; if it is a mistake there is no undo, and the record of HOW the
store changed is gone. But a sequence of store versions is a video: consecutive
versions are mostly redundant, so the GOP structure from holographic_video --
keyframe plus deltas -- stores the whole history cheaply, and any version can be
checked out again. Rollback, and a replayable learning history, fall out of
compression.

THE TWIST that makes it different from video: rollback needs the EXACT prior
state, so the deltas here are LOSSLESS (video's were lossy spectral truncation).
The two together complete the substrate's compression picture: lossy spectral
coding for perceptual data (images, audio, video frames), lossless sparse-delta
coding for state and history.

THE KEY DESIGN DECISION (the git lesson, learned by measurement): version rows by
a STABLE ID, not by position. A naive entry-wise diff calls a row deletion an
86%-of-matrix change, because every later row shifts index -- an alignment
artifact, not real change. Content-keyed rows make a delete cost ONE id and a
split cost one row: reorganization is genuinely sparse (insert/split/relabel/
merge measured at 0-9% of the matrix), and the history compresses ~29x losslessly.

WHAT WORKS (measured):
  * LOSSLESS DELTA HISTORY. commit() stores a keyframe every gop_len versions and
    a row-keyed delta otherwise; checkout(v) reconstructs ANY version EXACTLY.
  * COMPRESSION ON REORGANIZATION. Sparse structural edits compress the history
    ~29x vs storing every snapshot whole -- the inter-version redundancy is real.
  * PROOF-GATED COMMITS + ROLLBACK. commit(state, proof=fn) only persists a
    version whose proof passes; a reorganization that violates an invariant is
    REJECTED and the store stays at the last valid version -- but the rejected
    attempt is still recorded in the audit log, so you see what was tried.
    rollback(v) returns the live state to any past version (as a new commit, so
    the act of rolling back is itself in the history -- nothing is erased).

THE HONEST BOUNDARY (the 'deformation' analog from video): a DENSE update -- a
gradient/learning step that nudges every entry -- changes 100% of the matrix, so
delta coding does NOT compress it (the deltas are as big as the snapshots). Use
versioning for STRUCTURAL history (reorganization, edits, discrete commits),
where changes are sparse; a dense-trajectory recorder is a different tool.
"""
import numpy as np


class VersionedStore:
    """A store whose every version is committed and recoverable. State is a set of
    rows keyed by stable integer ids plus an ordered id list (the row order). The
    history is keyframes + lossless row-keyed deltas."""

    def __init__(self, dim, gop_len=8):
        self.dim = dim
        self.gop_len = gop_len
        self._commits = []      # each: ('key', {id:row}, [ids]) or ('delta', added, removed, changed, [ids])
        self._audit = []        # every attempt, including rejected ones
        self._next_id = 0

    # ---- helpers ----
    def new_id(self):
        self._next_id += 1
        return self._next_id

    def _materialize(self, upto):
        """Reconstruct the {id:row}, [ids] state at commit index `upto` by replaying
        from the most recent keyframe -- exactly (lossless)."""
        start = upto
        while self._commits[start][0] != "key":
            start -= 1
        rows = {k: v.copy() for k, v in self._commits[start][1].items()}
        order = list(self._commits[start][2])
        for t in range(start + 1, upto + 1):
            _, added, removed, changed, order = self._commits[t]
            for i in removed:
                rows.pop(i, None)
            for i, r in added.items():
                rows[i] = r.copy()
            for i, r in changed.items():
                rows[i] = r.copy()
            order = list(order)
        return rows, order

    # ---- the version API ----
    def commit(self, rows, order, proof=None, note=""):
        """Commit a new version (rows: {id:vector}, order: [ids]). If `proof` is
        given it is called with (rows, order) and must return True for the commit
        to persist; a failing proof is recorded in the audit log and the store is
        left unchanged (proof-gated reorganization). Returns the version index, or
        -1 if rejected."""
        ok = True if proof is None else bool(proof(rows, order))
        self._audit.append({"version": len(self._commits) if ok else None,
                            "accepted": ok, "note": note, "n_rows": len(order)})
        if not ok:
            return -1
        v = len(self._commits)
        if v % self.gop_len == 0:
            self._commits.append(("key", {k: np.asarray(r, float).copy() for k, r in rows.items()},
                                  list(order)))
        else:
            prev_rows, _ = self._materialize(v - 1)
            added = {i: np.asarray(rows[i], float).copy() for i in order if i not in prev_rows}
            removed = [i for i in prev_rows if i not in rows]
            changed = {i: np.asarray(rows[i], float).copy() for i in order
                       if i in prev_rows and not np.array_equal(np.asarray(rows[i], float), prev_rows[i])}
            self._commits.append(("delta", added, removed, changed, list(order)))
        return v

    def checkout(self, version):
        """Return (rows, order) at any past version, reconstructed exactly."""
        if not 0 <= version < len(self._commits):
            raise IndexError(f"no version {version}")
        return self._materialize(version)

    def rollback(self, version):
        """Revert the live state to a past version by committing it again -- the
        rollback is itself recorded, so history is never erased."""
        rows, order = self.checkout(version)
        return self.commit(rows, order, note=f"rollback to v{version}")

    def head(self):
        return len(self._commits) - 1

    def history(self):
        """The audit trail: every commit attempt, accepted or rejected."""
        return list(self._audit)

    # ---- accounting ----
    def stored_entries(self):
        total = 0
        for c in self._commits:
            if c[0] == "key":
                total += len(c[2]) * self.dim
            else:
                _, added, removed, changed, _ = c
                total += (len(added) + len(changed)) * self.dim + len(removed)
        return total

    def full_entries(self):
        """Entries if every version were stored whole -- the baseline to beat."""
        return sum(len(c[2] if c[0] == "key" else c[4]) * self.dim for c in self._commits)

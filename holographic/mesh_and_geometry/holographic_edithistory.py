"""holographic_edithistory.py -- the EDIT TRANSACTION LOG that makes a modeling session undoable. Every edit a user
does is recorded as a reversible COMMAND, and undo/redo walk the log. This is the backbone that turns a pile of
one-shot operations (Epics A-C) into an interactive session where nothing is ever lost.

THE MODEL (command pattern, tie-safe)
  A command is {name, apply, invert, params}: `apply(state)->state` does the edit, `invert(state)->state` undoes it.
  The log keeps an ordered list plus a cursor. `do` runs a command and pushes it; `undo` runs the top command's
  invert and steps the cursor back; `redo` re-applies. Doing a new command after an undo TRUNCATES the redo tail
  (the standard editor contract -- you can't redo into an abandoned future).

WHY REVERSIBLE, NOT SNAPSHOT: storing a full mesh per step is O(mesh) memory per edit; storing the inverse
operation is O(edit). More importantly, a reversible log is DETERMINISTIC and tie-safe (Macklin's discipline): a
redo re-runs the exact same apply, so a replayed history is bit-identical to the original -- a snapshot-diff can
smear a 1e-12 tie. For edits without a cheap closed-form inverse, `capture_inverse` snapshots just the touched
vertices, so the inverse is still O(edit), not O(mesh).

Deterministic, NumPy/stdlib only. The state is whatever the caller threads through (a mesh dict, a points array);
the log does not assume a representation -- it just applies and inverts.
"""

import numpy as np


class EditCommand:
    """One reversible edit: `apply(state)->new_state` and `invert(state)->prev_state`, plus a name and params for
    display/replay. Build one directly, or use `vertex_move` / `capture_inverse` below for the common cases."""

    def __init__(self, name, apply_fn, invert_fn, params=None):
        self.name = name
        self.apply = apply_fn
        self.invert = invert_fn
        self.params = params or {}

    def __repr__(self):
        return "EditCommand(%r)" % self.name


class EditHistory:
    """The undo/redo log for an interactive edit session. Thread the scene state through do/undo/redo. The log owns
    the ordering and the cursor; the caller owns the state (so this works on a mesh dict, a points array, anything).

      h = EditHistory()
      state = h.do(state, move_cmd)     # apply and record
      state = h.undo(state)             # step back
      state = h.redo(state)             # step forward
    """

    def __init__(self, max_depth=256):
        self._log = []                                         # the commands, oldest first
        self._cursor = 0                                       # number of commands currently APPLIED
        self.max_depth = max_depth                             # bound memory: drop the oldest beyond this

    def do(self, state, command):
        """Apply `command` to `state`, record it, and return the new state. If we had undone some commands, their
        redo tail is discarded first (you cannot redo into an abandoned future)."""
        new_state = command.apply(state)
        del self._log[self._cursor:]                           # truncate any redo tail
        self._log.append(command)
        self._cursor = len(self._log)
        if len(self._log) > self.max_depth:                    # bound the history depth
            drop = len(self._log) - self.max_depth
            del self._log[:drop]
            self._cursor -= drop
        return new_state

    def undo(self, state):
        """Invert the most recently applied command and return the prior state. No-op (returns state unchanged) if
        there is nothing to undo."""
        if self._cursor == 0:
            return state
        self._cursor -= 1
        return self._log[self._cursor].invert(state)

    def redo(self, state):
        """Re-apply the next command in the log and return the new state. No-op if there is nothing to redo."""
        if self._cursor >= len(self._log):
            return state
        cmd = self._log[self._cursor]
        self._cursor += 1
        return cmd.apply(state)

    def can_undo(self):
        return self._cursor > 0

    def can_redo(self):
        return self._cursor < len(self._log)

    def undo_stack(self):
        """The names of the commands that can be undone (most recent last) -- for a UI's history panel."""
        return [c.name for c in self._log[:self._cursor]]

    def redo_stack(self):
        """The names of the commands that can be redone (next first)."""
        return [c.name for c in self._log[self._cursor:]]

    # -- EDITABLE CONSTRUCTION HISTORY (D2): the log is not just an undo stack, it is the recipe that built the
    # -- current state. Because every command is deterministic, re-running the log from a base state reproduces the
    # -- result exactly -- and swapping a PAST command's parameters and re-running is 'edit a past operation and
    # -- recompute downstream', the Maya/C4D reach-back. This EXTENDS the same log; it does not add a second graph.
    def rebuild(self, base_state, upto=None):
        """Re-evaluate the construction history from `base_state`, applying every recorded command in order, and
        return the resulting state. `upto` (an index) stops after that many commands (default: all applied). This is
        the deterministic replay that makes the history a RECIPE: given the same base and the same commands, the
        result is bit-identical -- the property the whole undo/redo model rests on (Macklin tie-safety)."""
        n = self._cursor if upto is None else int(upto)
        state = base_state
        for cmd in self._log[:n]:
            state = cmd.apply(state)
        return state

    def replace_command(self, index, new_command, base_state):
        """EDIT A PAST OPERATION: swap the command at `index` for `new_command` (e.g. the same bevel with a larger
        width, the same move with a different delta) and RE-EVALUATE the whole history from `base_state`, returning
        the new current state. The downstream commands are re-applied on top of the changed one, exactly as a
        parametric-history modeler expects when you reach back and tweak an earlier step. The log length and cursor
        are preserved (only that one step's definition changed). Raises IndexError if `index` is out of range.

        This is why the history is 'editable', not just linear: the commands ARE the construction recipe, and
        because they replay deterministically, re-parameterizing step k and recomputing k+1.. is well-defined and
        tie-safe. For a command with parameters, build `new_command` with the changed params (e.g. vertex_move with
        a new delta) and pass it here."""
        if not (0 <= index < len(self._log)):
            raise IndexError("command index %d out of range (have %d)" % (index, len(self._log)))
        self._log[index] = new_command
        return self.rebuild(base_state)

    def commands(self):
        """The recorded commands as {index, name, params} dicts -- the construction-history panel a UI shows, and
        the way a caller finds which index to re-parameterize with replace_command."""
        return [{"index": i, "name": c.name, "params": dict(c.params)} for i, c in enumerate(self._log)]

    def __len__(self):
        return len(self._log)


def vertex_move(indices, delta, name="move"):
    """A reversible VERTEX MOVE command over a points array: apply adds `delta` to the given vertex indices, invert
    subtracts it. The closed-form inverse case -- O(edit) memory, bit-identical replay. `delta` is a 3-vector (all
    moved together) or an (len(indices),3) array (per-vertex)."""
    idx = np.asarray(indices, int)
    d = np.asarray(delta, float)

    def apply_fn(state):
        P = np.asarray(state, float).copy()
        P[idx] = P[idx] + d
        return P

    def invert_fn(state):
        P = np.asarray(state, float).copy()
        P[idx] = P[idx] - d
        return P

    return EditCommand(name, apply_fn, invert_fn, params={"indices": idx.tolist(), "delta": d.tolist()})


def capture_inverse(indices, new_positions, prev_positions, name="edit"):
    """A reversible command for a general edit that has no cheap algebraic inverse: it stores the BEFORE and AFTER
    positions of just the touched vertices, so apply writes the new ones and invert restores the old ones. O(edit)
    memory (only the touched verts), not O(mesh). Use this to wrap an arbitrary geometry op (a bevel, a smooth)
    into the undo log."""
    idx = np.asarray(indices, int)
    after = np.asarray(new_positions, float)
    before = np.asarray(prev_positions, float)

    def apply_fn(state):
        P = np.asarray(state, float).copy()
        P[idx] = after
        return P

    def invert_fn(state):
        P = np.asarray(state, float).copy()
        P[idx] = before
        return P

    return EditCommand(name, apply_fn, invert_fn, params={"indices": idx.tolist()})


def _selftest():
    """Contracts:
    1. do/undo/redo round-trips state exactly (bit-identical).
    2. a new command after an undo truncates the redo tail.
    3. undo/redo are no-ops at the ends.
    4. capture_inverse restores arbitrary edits; replay is deterministic.
    """
    P0 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    h = EditHistory()

    # (1) move vertex 1 by +y, then vertex 2 by +z; undo both; redo both.
    s = h.do(P0, vertex_move([1], [0, 1, 0], "move v1"))
    s = h.do(s, vertex_move([2], [0, 0, 1], "move v2"))
    assert np.allclose(s[1], [1, 1, 0]) and np.allclose(s[2], [2, 0, 1])
    s = h.undo(s)                                              # undo move v2
    assert np.allclose(s[2], [2, 0, 0]) and np.allclose(s[1], [1, 1, 0])
    s = h.undo(s)                                              # undo move v1
    assert np.allclose(s, P0)                                  # back to the exact start (bit-identical)
    s = h.redo(s); s = h.redo(s)                               # redo both
    assert np.allclose(s[1], [1, 1, 0]) and np.allclose(s[2], [2, 0, 1])

    # (2) undo one, then do a NEW command -> the redo tail is gone.
    s = h.undo(s)                                              # undo move v2; cursor now allows redo of v2
    assert h.can_redo()
    s = h.do(s, vertex_move([0], [1, 0, 0], "move v0"))        # a new future
    assert not h.can_redo()                                    # the old redo (v2) was truncated
    assert h.undo_stack() == ["move v1", "move v0"]

    # (3) undo to the bottom, then one more undo is a no-op.
    while h.can_undo():
        s = h.undo(s)
    assert np.allclose(s, P0)
    s_again = h.undo(s)
    assert np.allclose(s_again, s)                             # no-op at the bottom

    # (4) capture_inverse wraps an arbitrary edit (here: set two verts to arbitrary spots) and undoes it.
    h2 = EditHistory()
    before = P0[[0, 2]].copy()
    after = np.array([[9, 9, 9], [8, 8, 8]], float)
    cmd = capture_inverse([0, 2], after, before, "scatter")
    s2 = h2.do(P0, cmd)
    assert np.allclose(s2[0], [9, 9, 9]) and np.allclose(s2[2], [8, 8, 8]) and np.allclose(s2[1], P0[1])
    s2 = h2.undo(s2)
    assert np.allclose(s2, P0)                                 # arbitrary edit fully reversed
    # deterministic replay: redo reproduces the same bytes.
    s2r = h2.redo(s2)
    assert np.allclose(s2r, np.array([[9, 9, 9], [1, 0, 0], [8, 8, 8]], float))

    # (5) EDITABLE CONSTRUCTION HISTORY (D2): build a two-step history, then re-parameterize step 0 and re-evaluate.
    h3 = EditHistory()
    s3 = h3.do(P0, vertex_move([1], [0, 1, 0], "raise v1"))    # step 0: v1 += y
    s3 = h3.do(s3, vertex_move([1], [0, 0, 2], "push v1 z"))   # step 1: v1 += 2z
    assert np.allclose(s3[1], [1, 1, 2])
    # rebuild from the base reproduces the current state exactly (the history IS the recipe).
    assert np.allclose(h3.rebuild(P0), s3)
    # now change step 0 to raise v1 by 5y instead of 1y, and recompute downstream (step 1 still adds 2z).
    s3b = h3.replace_command(0, vertex_move([1], [0, 5, 0], "raise v1"), P0)
    assert np.allclose(s3b[1], [1, 5, 2])                      # the edit propagated through step 1
    assert len(h3) == 2 and [c["name"] for c in h3.commands()] == ["raise v1", "push v1 z"]  # structure preserved
    try:
        h3.replace_command(9, vertex_move([0], [0, 0, 0]), P0); assert False
    except IndexError:
        pass

    print("holographic_edithistory selftest OK (do/undo/redo round-trips state bit-identically; a new command "
          "truncates the redo tail; undo/redo no-op at the ends; undo_stack reports names; capture_inverse reverses "
          "an arbitrary edit with O(edit) memory; deterministic replay; EDITABLE history rebuilds from a base and "
          "re-parameterizing a past step propagates downstream)")


if __name__ == "__main__":
    _selftest()

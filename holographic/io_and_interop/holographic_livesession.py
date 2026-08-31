"""LIVESESSION -- revisions and presence for concurrent editors, owned by NEITHER app.

leStudio's multiplayer is app-local: a monotonic `rev` bumped by a Flask
after_request hook on every mutating POST, an SSE feed carrying
{rev, src, editors}, presence defined as an open stream, plus host/kick/invite
bookkeeping -- all inside one app's web server. Poly Studio has none of it.

SO "SHARE A WORKSPACE" SPLITS IN TWO. File-level sharing works today: both apps
read and write one container, each preserving the other's sections. LIVE
CO-EDITING ACROSS TWO DIFFERENT APPS CANNOT EXIST while the sync layer lives
inside one app's HTTP server, because the second app would have to import the
first one's Flask app to join a session.

RULED OUT BEFORE BUILDING, and the backlog asks not to re-tread it:
WorkspaceManager is NOT this. Its methods are new_workspace / switch_workspace /
checkpoint / restore_checkpoint -- it checkpoints a live DB's scratch tables by
replay, and the container module's own docstring already says so ("KEPT
NEGATIVE: this is NOT the workspace_manager"). Checkpointing one editor's
history and coordinating several editors are different problems.

TRANSPORT-AGNOSTIC IS THE WHOLE POINT AND THE HARDEST PART TO HOLD. There is no
socket, no SSE, no thread and no Flask here. A session is a small piece of
STATE plus a monotonic counter, and each app drives it with whatever transport
it already has -- SSE in one, polling in another, a pipe in a test. The moment
this module imports a web framework it becomes leStudio's implementation with a
different filename, and the second app is locked out again.

WHAT A CHANGE FEED IS HERE: an append-only log of {rev, src, kind, meta} that a
participant reads FROM ITS LAST SEEN REVISION. That is enough to drive an SSE
stream, a long poll, or a diff-on-reconnect, and it is the smallest thing that
is. Presence is a heartbeat with a timeout rather than "an open stream",
because an open stream is a property of ONE transport.
"""

import time


class LiveSession:
    """A revision counter, a participant table, and a change feed. No transport.

    Every mutation a client makes is announced with `bump`, which returns the
    new revision. Every client polls `since` with the last revision it saw. That
    pair is the entire contract, and it is deliberately smaller than leStudio's
    -- host/kick/invite are POLICY and belong in whichever app is hosting."""

    def __init__(self, name="session", ttl=30.0, now=None):
        self.name = str(name)
        self.ttl = float(ttl)
        self._now = now or time.time      # injectable, so tests are not sleepy
        self.rev = 0
        self._log = []                    # [{rev, src, kind, meta, at}]
        self._seen = {}                   # participant -> last heartbeat

    # ---- mutation ----

    def bump(self, src, kind="edit", meta=None):
        """Record a mutation. Returns the new revision.

        MONOTONIC AND NEVER REUSED, because a client's whole resync strategy is
        "give me everything after N" -- a revision that goes backwards or
        repeats silently drops edits for every client that already passed it."""
        self.rev += 1
        self._log.append({"rev": self.rev, "src": str(src), "kind": str(kind),
                          "meta": dict(meta or {}), "at": float(self._now())})
        self.touch(src)
        return self.rev

    # ---- presence ----

    def touch(self, who):
        """Mark a participant alive. Called by any activity, not just edits."""
        self._seen[str(who)] = float(self._now())
        return self.rev

    def participants(self):
        """Who is currently present, oldest heartbeat first.

        PRESENCE IS A HEARTBEAT WITH A TIMEOUT, not "an open stream". A stream
        is a property of ONE transport; a client that polls or reconnects is
        just as present, and a client whose socket is open but whose process is
        wedged is not."""
        cut = float(self._now()) - self.ttl
        alive = [(t, w) for w, t in self._seen.items() if t >= cut]
        return [w for _t, w in sorted(alive)]

    def drop(self, who):
        """Remove a participant immediately (a clean disconnect)."""
        self._seen.pop(str(who), None)
        return self.participants()

    # ---- the change feed ----

    def since(self, rev, exclude=None):
        """Every change after `rev`, oldest first. The transport-agnostic feed.

        `exclude` skips a source's own edits, which is what a client wants when
        its local state already reflects them -- echoing an edit back is how a
        naive sync loop makes the cursor jump while someone is typing."""
        out = [e for e in self._log if e["rev"] > int(rev)]
        if exclude is not None:
            out = [e for e in out if e["src"] != str(exclude)]
        return out

    def state(self):
        """{rev, participants, name} -- what a status endpoint returns."""
        return {"name": self.name, "rev": self.rev,
                "participants": self.participants()}

    def compact(self, keep=1000):
        """Drop log entries older than the newest `keep`. Returns how many went.

        A SESSION THAT NEVER FORGETS IS A MEMORY LEAK WITH A REVISION NUMBER.
        Clients further behind than the retained window must do a full reload,
        which `oldest_rev` lets them detect rather than silently missing edits."""
        n = max(0, len(self._log) - int(keep))
        if n:
            self._log = self._log[n:]
        return n

    @property
    def oldest_rev(self):
        """The oldest revision still in the feed; below it, a client must reload."""
        return self._log[0]["rev"] - 1 if self._log else self.rev


def _selftest():
    clock = {"t": 1000.0}
    s = LiveSession("doc", ttl=10.0, now=lambda: clock["t"])

    # ---- REVISIONS ARE MONOTONIC AND OBSERVED BY THE OTHER PARTY ----
    assert s.bump("polystudio", "add_object") == 1
    assert s.bump("lestudio", "paint") == 2
    assert s.rev == 2
    # each app sees the OTHER's edits and not its own echo
    assert [e["src"] for e in s.since(0, exclude="lestudio")] == ["polystudio"]
    assert [e["src"] for e in s.since(0, exclude="polystudio")] == ["lestudio"]
    assert s.since(2) == []

    # ---- PRESENCE IS A HEARTBEAT, AND IT EXPIRES ----
    assert s.participants() == ["lestudio", "polystudio"]
    clock["t"] += 11.0                      # past the ttl, nobody heartbeats
    assert s.participants() == []
    s.touch("polystudio")
    assert s.participants() == ["polystudio"]
    s.drop("polystudio")
    assert s.participants() == []

    # ---- TWO DIFFERENT PROCESSES' WORTH OF CLIENTS, DRIVEN BY POLLING ----
    # neither "app" here imports the other; each holds only its last-seen rev.
    a_seen = b_seen = 0
    s.bump("appA", "edit", {"layer": 3})
    got_b = s.since(b_seen, exclude="appB")
    b_seen = got_b[-1]["rev"]
    s.bump("appB", "edit", {"object": "cube"})
    got_a = s.since(a_seen, exclude="appA")
    a_seen = got_a[-1]["rev"]
    assert [e["src"] for e in got_b][-1] == "appA"
    assert [e["src"] for e in got_a][-1] == "appB"
    assert a_seen == b_seen + 1

    # ---- COMPACTION BOUNDS THE LOG AND SAYS WHAT WAS LOST ----
    for i in range(50):
        s.bump("appA", "edit", {"i": i})
    dropped = s.compact(keep=10)
    assert dropped > 0 and len(s._log) == 10
    assert s.oldest_rev == s._log[0]["rev"] - 1
    assert s.since(s.oldest_rev)[0]["rev"] == s._log[0]["rev"]

    st = s.state()
    assert set(st) == {"name", "rev", "participants"}

    print("livesession selftest OK -- revisions and presence with NO transport: "
          "two clients that never import each other exchange edits by polling "
          "since(last_seen) and each is excluded from its own echo; presence is "
          "a HEARTBEAT WITH A TIMEOUT rather than an open stream, so a polling "
          "client is as present as a streaming one and a wedged process with an "
          "open socket is not; and compact() bounds the log while oldest_rev "
          "tells a client too far behind to reload instead of silently missing "
          "edits")


if __name__ == "__main__":
    _selftest()

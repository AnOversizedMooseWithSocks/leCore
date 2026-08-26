"""holographic_registry.py -- WHO IS ONLINE: a presence registry for principals and nodes.

The catalog answers "what can this instance DO"; this answers "who is HERE right now". A swarm or a farm needs it:
actors (agents, users, services, peer nodes) ANNOUNCE their presence with a heartbeat, and others LIST/discover them
to coordinate. It's deliberately tiny -- a dict of who was last seen when -- and it can ride the bus so presence is
visible across machines, not just in one process.

An actor that hasn't announced within `ttl` seconds is treated as OFFLINE (stale), so a crashed node drops out on its
own without anyone having to clean up.

    reg = Registry(ttl=30)
    reg.announce(alice)                 # "alice is online" (call again periodically = heartbeat)
    online = reg.list(kind="agent")     # discover live agents to coordinate with
    reg.is_online(alice)                # True until alice stops heartbeating for ttl seconds

Determinism: time is the only non-pure input, and it's injectable (`clock=`), so tests are fully deterministic.
numpy-free; stdlib only.
"""
import time


def _actor_id(actor):
    """Accept either a Principal (or anything with an .id) or a plain id string."""
    return getattr(actor, "id", actor)


class Registry:
    """A live-presence directory. Actors announce (heartbeat); others list the ones seen within `ttl`."""

    def __init__(self, ttl=30.0, clock=time.time, bus=None):
        self.ttl = ttl
        self._clock = clock                     # injectable so presence expiry is testable/deterministic
        self._bus = bus                         # optional: publish presence so remote nodes can see it too
        self._present = {}                      # id -> {id, kind, workspace, last_seen}

    def announce(self, actor, kind=None, workspace=None):
        """Record (or refresh) an actor's presence -- this IS the heartbeat; call it periodically to stay 'online'.
        `kind`/`workspace` default to the actor's own if it's a Principal. Returns the actor id."""
        aid = _actor_id(actor)
        info = {
            "id": aid,
            "kind": kind if kind is not None else getattr(actor, "kind", "actor"),
            "workspace": workspace if workspace is not None else getattr(actor, "workspace", None),
            "last_seen": self._clock(),
        }
        self._present[aid] = info
        if self._bus is not None:               # let other nodes learn about this actor
            self._bus.publish("presence.announce", dict(info), sender=aid)
        return aid

    # heartbeat reads more naturally in a loop; it's just another announce.
    heartbeat = announce

    def leave(self, actor):
        """Explicitly mark an actor offline now (e.g. a clean disconnect)."""
        aid = _actor_id(actor)
        self._present.pop(aid, None)
        if self._bus is not None:
            self._bus.publish("presence.leave", {"id": aid}, sender=aid)
        return aid

    def is_online(self, actor):
        """True iff the actor announced within the last `ttl` seconds."""
        info = self._present.get(_actor_id(actor))
        return info is not None and (self._clock() - info["last_seen"]) <= self.ttl

    def list(self, kind=None, workspace=None):
        """The actors currently online (seen within `ttl`), optionally filtered by kind and/or workspace. Sorted by id
        so the result is deterministic. Stale entries are skipped (and pruned)."""
        now = self._clock()
        out = []
        stale = []
        for aid, info in self._present.items():
            if now - info["last_seen"] > self.ttl:
                stale.append(aid)
                continue
            if kind is not None and info["kind"] != kind:
                continue
            if workspace is not None and info["workspace"] != workspace:
                continue
            out.append(dict(info))
        for aid in stale:                       # drop the ones that timed out
            self._present.pop(aid, None)
        return sorted(out, key=lambda d: str(d["id"]))

    def count(self, kind=None, workspace=None):
        """How many are online (with the same filters as list)."""
        return len(self.list(kind=kind, workspace=workspace))


def _selftest():
    # a controllable clock so expiry is deterministic
    t = {"now": 1000.0}
    reg = Registry(ttl=30.0, clock=lambda: t["now"])

    class _P:                                    # a stand-in principal
        def __init__(self, i, k, w="lab"):
            self.id, self.kind, self.workspace = i, k, w

    reg.announce(_P("alice", "user"))
    reg.announce(_P("bob", "user"))
    reg.announce(_P("agent7", "agent"))

    # everyone just announced -> all online; filter by kind/workspace
    assert reg.count() == 3
    assert [d["id"] for d in reg.list(kind="user")] == ["alice", "bob"]
    assert reg.count(kind="agent") == 1
    assert reg.is_online("alice")

    # advance time past ttl without a heartbeat -> everyone goes offline (stale, pruned)
    t["now"] += 31.0
    assert reg.count() == 0 and not reg.is_online("alice")

    # a heartbeat keeps you online across the window
    reg.announce(_P("alice", "user"))
    t["now"] += 20.0
    reg.heartbeat(_P("alice", "user"))           # refresh before ttl
    t["now"] += 20.0                             # 40s since first announce, but only 20s since heartbeat
    assert reg.is_online("alice")

    # explicit leave drops immediately
    reg.leave("alice")
    assert not reg.is_online("alice")

    # rides the bus: an announce publishes presence for remote nodes
    from holographic.misc.holographic_bus import MessageBus
    bus = MessageBus()
    bus.open_mailbox("watch", ["presence.*"])
    reg2 = Registry(ttl=30.0, clock=lambda: t["now"], bus=bus)
    reg2.announce(_P("carol", "peer"))
    seen = bus.poll("watch")
    assert any(m.topic == "presence.announce" and m.payload["id"] == "carol" for m in seen)

    print("OK: holographic_registry self-test passed (announce/list/filter by kind+workspace; stale actors go offline "
          "after ttl and are pruned; heartbeat keeps you online; leave drops immediately; announcing rides the bus so "
          "remote nodes see presence)")


if __name__ == "__main__":
    _selftest()

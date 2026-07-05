"""Tests for holographic_registry.py -- presence: who is online."""
from holographic_registry import Registry


class _P:
    def __init__(self, i, k, w="lab"):
        self.id, self.kind, self.workspace = i, k, w


def _reg(t):
    return Registry(ttl=30.0, clock=lambda: t["now"])


def test_announce_and_list_filtered():
    t = {"now": 100.0}; reg = _reg(t)
    reg.announce(_P("alice", "user")); reg.announce(_P("bob", "user")); reg.announce(_P("ag", "agent"))
    assert reg.count() == 3
    assert [d["id"] for d in reg.list(kind="user")] == ["alice", "bob"]
    assert reg.count(kind="agent") == 1
    assert reg.count(workspace="lab") == 3


def test_stale_actors_go_offline_after_ttl():
    t = {"now": 100.0}; reg = _reg(t)
    reg.announce(_P("alice", "user"))
    assert reg.is_online("alice")
    t["now"] += 31.0
    assert not reg.is_online("alice") and reg.count() == 0


def test_heartbeat_keeps_online():
    t = {"now": 100.0}; reg = _reg(t)
    reg.announce(_P("alice", "user"))
    t["now"] += 20.0; reg.heartbeat(_P("alice", "user"))
    t["now"] += 20.0
    assert reg.is_online("alice")                        # 40s since first, 20s since heartbeat


def test_leave_drops_immediately():
    t = {"now": 100.0}; reg = _reg(t)
    reg.announce(_P("alice", "user"))
    reg.leave("alice")
    assert not reg.is_online("alice")


def test_announce_rides_the_bus():
    from holographic_bus import MessageBus
    bus = MessageBus(); bus.open_mailbox("watch", ["presence.*"])
    reg = Registry(bus=bus)
    reg.announce(_P("carol", "peer"))
    seen = bus.poll("watch")
    assert any(m.topic == "presence.announce" and m.payload["id"] == "carol" for m in seen)

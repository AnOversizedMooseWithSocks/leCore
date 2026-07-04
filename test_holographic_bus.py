"""Tests for holographic_bus.py -- the message bus (topics, pub/sub, mailboxes, history)."""
from holographic_bus import MessageBus, topic_matches


def test_pubsub_by_topic_and_wildcard():
    bus = MessageBus()
    seen = []
    bus.subscribe("render.*", lambda m: seen.append(m.topic))
    bus.publish("render.start"); bus.publish("render.done"); bus.publish("sim.step")
    assert seen == ["render.start", "render.done"]


def test_message_ids_are_deterministic():
    a = MessageBus(); b = MessageBus()
    assert [a.publish("x").id, a.publish("y").id] == [b.publish("x").id, b.publish("y").id]


def test_unsubscribe():
    bus = MessageBus()
    n = []
    off = bus.subscribe("a", lambda m: n.append(1))
    bus.publish("a"); off(); bus.publish("a")
    assert len(n) == 1


def test_mailbox_pull():
    bus = MessageBus()
    bus.open_mailbox("agent", ["render.*"])
    bus.publish("render.done"); bus.publish("other")
    got = bus.poll("agent")
    assert [m.topic for m in got] == ["render.done"]
    assert bus.poll("agent") == []                 # drained


def test_directed_send():
    bus = MessageBus()
    got = []
    bus.subscribe("to:agent", lambda m: got.append(m.payload))
    bus.send("agent", {"hi": 1}, sender="user")
    assert got == [{"hi": 1}]


def test_broken_handler_is_logged_not_fatal():
    bus = MessageBus()
    bus.subscribe("boom", lambda m: 1 / 0)
    bus.publish("boom")                            # must not raise
    assert any(m.topic == "bus.handler_error" for m in bus.history())


def test_history_catch_up():
    bus = MessageBus()
    for i in range(3):
        bus.publish("render.step", {"i": i})
    bus.publish("sim.step")
    assert len(bus.history("render.*")) == 3
    assert len(bus.history()) == 4


def test_topic_matching_rules():
    assert topic_matches("*", "z")
    assert topic_matches("a.*", "a") and topic_matches("a.*", "a.b.c")
    assert not topic_matches("a.*", "ab")
    assert topic_matches("a.b", "a.b") and not topic_matches("a.b", "a.c")

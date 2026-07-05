"""Tests for holographic_distbus.py -- the message bus fanned out across nodes, + bounded-mailbox backpressure."""
import threading
import time
import pytest
from http.server import HTTPServer
from holographic_distbus import DistributedBus, _make_bus_handler
from holographic_bus import MessageBus


def _spin(bus, token, servers):
    httpd = HTTPServer(("127.0.0.1", 0), _make_bus_handler(bus, token))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    servers.append(httpd)
    return "127.0.0.1:%d" % port


@pytest.fixture
def two_nodes():
    servers = []
    a = DistributedBus(node_id="A", token="t")
    b = DistributedBus(node_id="B", token="t")
    a_addr = _spin(a, "t", servers); b_addr = _spin(b, "t", servers)
    a.add_peer(b_addr); b.add_peer(a_addr)
    time.sleep(0.15)
    yield a, b
    for h in servers:
        h.shutdown(); h.server_close()


def test_publish_crosses_to_peer(two_nodes):
    a, b = two_nodes
    b.open_mailbox("watch", ["plan.*"])
    a.publish("plan.step", {"n": 1}, sender="A")
    time.sleep(0.15)
    got = b.poll("watch")
    assert len(got) == 1 and got[0].payload == {"n": 1} and got[0].sender == "A"


def test_local_delivery_still_works(two_nodes):
    a, _ = two_nodes
    a.open_mailbox("local", ["plan.*"])
    a.publish("plan.step", {"n": 5}, sender="A")
    assert [m.payload["n"] for m in a.poll("local")] == [5]     # published locally too


def test_directed_send_crosses(two_nodes):
    a, b = two_nodes
    b.open_mailbox("agent7", ["to:agent7"])
    a.send("agent7", {"do": "x"}, sender="A")
    time.sleep(0.15)
    assert [m.payload for m in b.poll("agent7")] == [{"do": "x"}]


def test_duplicate_ids_deduped():
    b = DistributedBus(node_id="B")
    assert b.deliver_remote("t", {"n": 1}, "A", "A#1") is not None
    assert b.deliver_remote("t", {"n": 1}, "A", "A#1") is None   # echo dropped


def test_dead_peer_does_not_crash_publisher():
    a = DistributedBus(node_id="A", peers=["127.0.0.1:1"])       # nothing listening
    a.publish("plan.step", {"n": 2}, sender="A")                 # must not raise


def test_bounded_mailbox_backpressure():
    bus = MessageBus()
    bus.open_mailbox("slow", ["ev.*"], maxlen=3)
    for i in range(10):
        bus.publish("ev.tick", {"i": i}, sender="clock")
    st = bus.mailbox_stats("slow")
    assert st["pending"] == 3 and st["dropped"] == 7 and st["maxlen"] == 3
    assert [m.payload["i"] for m in bus.poll("slow")] == [7, 8, 9]   # newest kept, oldest dropped


def test_unbounded_mailbox_still_default():
    bus = MessageBus()
    bus.open_mailbox("all", ["ev.*"])                            # no maxlen
    for i in range(50):
        bus.publish("ev.tick", {"i": i}, sender="clock")
    assert bus.mailbox_stats("all")["pending"] == 50 and bus.mailbox_stats("all")["dropped"] == 0

"""holographic_distbus.py -- the message bus, spread across machines.

The in-process MessageBus already does topics, directed sends, `sender`, and a deterministic `seq`. This subclass
keeps ALL of that unchanged and adds one thing: when you publish, the message is also FORWARDED to peer nodes, which
deliver it to THEIR local subscribers. So agents on different machines share topics -- a swarm coordinates across the
farm exactly the way it does in one process, same publish/subscribe/send API.

    # on each node:
    bus = DistributedBus(peers=["hostB:9100", "hostC:9100"], token=SECRET, node_id="A")
    serve_bus(bus, port=9100, token=SECRET)         # (in a thread) receive peers' messages -> local delivery

    bus.publish("plan.step", {...}, sender="A")     # delivered locally AND forwarded to hostB + hostC
    bus.send("agent7", {...}, sender="A")           # a directed message reaches agent7 wherever it lives

How loops are avoided: a node only ever fans out its OWN publishes. A message ARRIVING from a peer is delivered
LOCAL-ONLY (never re-forwarded), and is deduplicated by a global id (node_id#seq), so even a fully-meshed set of
peers can't echo a message around forever. Delivery is best-effort/at-most-once: a down peer is skipped, not retried,
so one slow or dead node never blocks the publisher (pair with bounded mailboxes -- open_mailbox(maxlen=) -- for
backpressure at high fan-out).

KEPT LIMIT (loud): fan-out is DIRECT (originator -> its peers), which covers the common star and full-mesh topologies.
Multi-hop GOSSIP (B relays A's message on to C) is not done here -- the dedup id is already in place for it, but
forwarding-on is a deliberate follow-up. Topic SHARDING (hash topic -> node) for very high fan-out is likewise noted,
not built. Payloads must be JSON-serializable (this is the control plane -- big data goes through the farm, not here).

stdlib only (http.server + urllib + json); the local delivery path is the unchanged, deterministic MessageBus.
"""
import json as _json
import urllib.request as _urlreq
import urllib.error as _urlerr

from holographic.misc.holographic_bus import MessageBus


class DistributedBus(MessageBus):
    """A MessageBus whose publishes also reach subscribers on peer nodes. Local behaviour is identical to MessageBus;
    the only addition is the fan-out to `peers` and the receive path (deliver_remote)."""

    def __init__(self, peers=None, token=None, node_id="node", timeout=10.0):
        super().__init__()
        self.peers = list(peers or [])                 # OTHER nodes, "host:port" each running serve_bus
        self.token = token
        self.node_id = node_id                         # names this node so message ids are globally unique
        self.timeout = timeout
        self._seen = set()                             # global ids already delivered here (dedup)

    def add_peer(self, addr):
        """Add a peer node ('host:port') to fan out to. Returns self (chainable)."""
        if addr not in self.peers:
            self.peers.append(addr)
        return self

    def publish(self, topic, payload=None, sender="app", reply_to=None):
        """Publish locally (unchanged MessageBus behaviour, returns the Message) AND forward to every peer node."""
        msg = super().publish(topic, payload, sender=sender, reply_to=reply_to)   # local delivery + log + seq
        self._fanout(topic, payload, sender, "%s#%d" % (self.node_id, msg.seq))
        return msg

    # send() is inherited: it calls self.publish(), so a directed 'to:<n>' message fans out to peers automatically.

    def _fanout(self, topic, payload, sender, mid):
        """Forward one message to each peer, best-effort. A peer that's down or slow is skipped, never retried, so it
        can't block the publisher."""
        if not self.peers:
            return
        body = {"topic": topic, "payload": payload, "sender": sender, "mid": mid}
        for peer in self.peers:
            try:
                _post("http://%s/bus" % peer, body, self.token, self.timeout)
            except Exception:
                pass                                   # at-most-once; deliberately swallow a failed peer

    def deliver_remote(self, topic, payload, sender, mid):
        """Deliver a message FORWARDED FROM A PEER to this node's LOCAL subscribers only (no re-fan-out), deduplicated
        by its global id. Returns the delivered Message, or None if it was a duplicate."""
        if mid in self._seen:
            return None                                # already delivered here -> drop the echo
        self._seen.add(mid)
        return MessageBus.publish(self, topic, payload, sender=sender)   # BASE publish == local delivery, no fan-out


# ---- the receive side: a tiny HTTP daemon that hands forwarded messages to the bus ------------------------
def _make_bus_handler(bus, token):
    from http.server import BaseHTTPRequestHandler

    class _Handler(BaseHTTPRequestHandler):
        def _authed(self):
            return (not token) or self.headers.get("Authorization", "") == "Bearer %s" % token

        def _reply(self, code, obj):
            body = _json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._authed():
                return self._reply(401, {"ok": False, "error": "unauthorized"})
            if self.path == "/health":
                return self._reply(200, {"ok": True, "role": "bus", "node": bus.node_id, "peers": bus.peers})
            self._reply(404, {"ok": False, "error": "no such endpoint: %s" % self.path})

        def do_POST(self):
            if not self._authed():
                return self._reply(401, {"ok": False, "error": "unauthorized"})
            try:
                n = int(self.headers.get("Content-Length", 0) or 0)
                m = _json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
                if self.path == "/bus":
                    bus.deliver_remote(m.get("topic"), m.get("payload"), m.get("sender"), m.get("mid"))
                    return self._reply(200, {"ok": True})
                self._reply(404, {"ok": False, "error": "no such endpoint: %s" % self.path})
            except Exception as e:
                self._reply(500, {"ok": False, "error": "%s: %s" % (type(e).__name__, e)})

        def log_message(self, *a):
            pass

    return _Handler


def serve_bus(bus, host="0.0.0.0", port=9100, token=None):
    """Run the RECEIVE side of a DistributedBus (BLOCKING) on this node: accept POST /bus from peers and deliver each
    message to this node's local subscribers. GET /health. stdlib http.server + JSON; bearer-token auth. Ctrl-C stops.
    Run this in a thread alongside your app so the node both publishes (fan-out) and receives (local delivery)."""
    from http.server import HTTPServer
    httpd = HTTPServer((host, port), _make_bus_handler(bus, token))
    print("leCore distributed bus node %r on http://%s:%d -- peers: %s" % (bus.node_id, host, port, bus.peers))
    if host == "0.0.0.0":
        print("  NOTE: bound to ALL interfaces -- only behind auth/TLS on a trusted network.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping bus node.")
        httpd.server_close()


# ---- tiny stdlib HTTP POST (self-contained, bearer-token auth) -------------------------------------------
def _post(url, body, token=None, timeout=10.0):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer %s" % token
    req = _urlreq.Request(url, data=_json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except _urlerr.HTTPError as e:
        try:
            return _json.loads(e.read().decode("utf-8"))
        except Exception:
            raise


def _selftest():
    import threading
    import time
    from http.server import HTTPServer

    # two nodes, A and B, each running a receive server; A forwards to B and B forwards to A.
    servers = []

    def _spin(bus, token):
        httpd = HTTPServer(("127.0.0.1", 0), _make_bus_handler(bus, token))
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return "127.0.0.1:%d" % port

    a = DistributedBus(node_id="A", token="t")
    b = DistributedBus(node_id="B", token="t")
    a_addr = _spin(a, "t")
    b_addr = _spin(b, "t")
    a.add_peer(b_addr)
    b.add_peer(a_addr)
    time.sleep(0.2)

    # a subscriber on B collects; A publishes; the message crosses the wire to B's local mailbox.
    b.open_mailbox("watch", ["plan.*"])
    a.publish("plan.step", {"n": 1}, sender="A")
    time.sleep(0.2)
    got = b.poll("watch")
    assert len(got) == 1 and got[0].payload == {"n": 1} and got[0].sender == "A", got

    # a directed send from A reaches an inbox on B (send() fans out too)
    b.open_mailbox("agent7", ["to:agent7"])
    a.send("agent7", {"do": "x"}, sender="A")
    time.sleep(0.2)
    inbox = b.poll("agent7")
    assert len(inbox) == 1 and inbox[0].payload == {"do": "x"}

    # dedup: delivering the same global id twice is a no-op (no echo storms in a mesh)
    assert b.deliver_remote("plan.step", {"n": 9}, "A", "A#999") is not None
    assert b.deliver_remote("plan.step", {"n": 9}, "A", "A#999") is None

    # a down peer doesn't crash the publisher (best-effort fan-out)
    a.add_peer("127.0.0.1:1")          # nothing listening here
    a.publish("plan.step", {"n": 2}, sender="A")   # must not raise

    for h in servers:
        h.shutdown()
    print("OK: holographic_distbus self-test passed (publish on A reaches a subscriber on B; a directed send crosses "
          "too; duplicate global ids are dropped; a dead peer doesn't crash the publisher)")


if __name__ == "__main__":
    _selftest()

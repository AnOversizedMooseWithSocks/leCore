"""holographic_bus.py -- a small message bus so the app, the person, and an AGENT (an LLM) can all talk at once.

WHY THIS EXISTS
---------------
leCore is the core of an AI substrate harness (leOS). The idea: a person AND an agent are both connected to the
running tool at the same time. The person drives; the agent can be asked to do things, watch output, tweak settings
until a render looks right, and answer questions. For that to work without the agent constantly POLLING 'are we done
yet?', the app has to PUSH -- when a render (or any long job) finishes, the app itself reaches out and says so. This
module is that plumbing: a plain, readable, in-process message bus.

WHAT IT DOES
------------
  * TOPICS + pub/sub:  publish(topic, payload); subscribe(topic, handler) to be called the moment it happens.
  * MAILBOXES:         a party that would rather PULL than take a callback opens a mailbox and poll()s it (handy for a
                       remote agent behind HTTP, which reads its inbox on its own schedule).
  * a LOG:             every message is kept, so a late joiner can catch up and you can replay a session.
  * WATCHES:           watch('render.done', handler) -- plain-English sugar over subscribe for 'tell me when X happens'.
  * DIRECTED sends:    send('agent', payload) delivers to whoever is listening on the 'to:agent' topic (person<->agent).

The LLM is OPTIONAL and lives entirely behind a callback you supply (see holographic_agent_bridge). The bus itself
knows NOTHING about any LLM, so leCore runs perfectly with no agent attached -- publishing to a topic nobody watches
simply logs it. Deterministic (a monotonic sequence gives every message a stable id + order; wall-clock time is
metadata only) and stdlib only (threading for safety, hashlib for ids); no framework.
"""
import hashlib
import threading
import time
from collections import deque


class Message:
    """One message on the bus: a `topic` (like 'render.done'), a `payload` (any Python value), who `sender` it came
    from, a monotonic `seq` (the deterministic order), a stable `id` derived from seq+topic, and a wall-clock `ts`
    (informational). `reply_to` links a reply back to the message it answers."""

    __slots__ = ("seq", "topic", "payload", "sender", "id", "ts", "reply_to")

    def __init__(self, seq, topic, payload, sender, reply_to=None):
        self.seq = seq
        self.topic = topic
        self.payload = payload
        self.sender = sender
        self.reply_to = reply_to
        # a stable id from the ORDER + topic (not wall-clock), so a replayed session gets identical ids
        self.id = hashlib.sha256(("%d:%s" % (seq, topic)).encode()).hexdigest()[:16]
        self.ts = time.time()

    def as_dict(self):
        """A JSON-friendly view (for the HTTP service / logging). The payload is passed through as-is."""
        return {"id": self.id, "seq": self.seq, "topic": self.topic, "payload": self.payload,
                "sender": self.sender, "reply_to": self.reply_to, "ts": self.ts}

    def __repr__(self):
        return "Message(#%d %s from %s)" % (self.seq, self.topic, self.sender)


def topic_matches(pattern, topic):
    """Does `topic` match a subscription `pattern`? Three readable cases:
        '*'      matches everything,
        'a.*'    matches 'a', 'a.b', 'a.b.c' (a prefix and anything under it),
        'a.b'    matches exactly 'a.b'.
    Kept deliberately simple -- no regex to reason about."""
    if pattern == "*":
        return True
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return topic == prefix or topic.startswith(prefix + ".")
    return pattern == topic


class MessageBus:
    """The bus. Thread-safe (one lock); handlers are called OUTSIDE the lock so a handler may itself publish without
    deadlocking. Everything is ordered by a monotonic sequence, so the same run always produces the same message ids
    and order."""

    def __init__(self, keep=1000):
        self._lock = threading.RLock()
        self._seq = 0
        self._subs = []                                     # list of (pattern, handler)
        self._mailboxes = {}                                # name -> {"patterns": (...), "queue": deque}
        self.log = deque(maxlen=keep)                       # recent messages, newest last (for replay / late join)

    # ---- publish / subscribe ----------------------------------------------------------------------------
    def publish(self, topic, payload=None, sender="app", reply_to=None):
        """Post a message to `topic`. Delivers to every subscriber whose pattern matches, drops a copy into every
        matching mailbox, and appends to the log. Returns the Message. Publishing to a topic nobody listens on is
        fine -- it just gets logged (so the app works with no agent attached)."""
        with self._lock:
            self._seq += 1
            msg = Message(self._seq, topic, payload, sender, reply_to=reply_to)
            self.log.append(msg)
            handlers = [h for (pat, h) in self._subs if topic_matches(pat, topic)]     # snapshot under the lock
            for mb in self._mailboxes.values():
                if any(topic_matches(pat, topic) for pat in mb["patterns"]):
                    mb["queue"].append(msg)
        for h in handlers:                                  # call handlers OUTSIDE the lock (they may publish)
            try:
                h(msg)
            except Exception as e:                          # a broken handler must not take down the publisher
                self._log_handler_error(topic, e)
        return msg

    def subscribe(self, pattern, handler):
        """Call `handler(message)` whenever a message matches `pattern` (see topic_matches). Returns an unsubscribe
        function -- call it to stop listening."""
        entry = (pattern, handler)
        with self._lock:
            self._subs.append(entry)

        def _unsub():
            with self._lock:
                if entry in self._subs:
                    self._subs.remove(entry)
        return _unsub

    # a friendlier name for the 'tell me when X happens' case
    def watch(self, pattern, handler):
        """Sugar for subscribe: watch('render.done', handler). Reads the way you'd say it."""
        return self.subscribe(pattern, handler)

    def send(self, to, payload=None, sender="app", reply_to=None):
        """Send a DIRECTED message to a named recipient (delivered on the topic 'to:<name>'). The recipient listens
        with subscribe('to:<name>', ...) or open_mailbox(name, ['to:<name>']). This is how the person and the agent
        message each other point-to-point."""
        return self.publish("to:" + to, payload, sender=sender, reply_to=reply_to)

    # ---- mailboxes (pull side) --------------------------------------------------------------------------
    def open_mailbox(self, name, patterns=("*",)):
        """Open (or re-open) a named mailbox that collects messages matching any of `patterns`. A party that prefers
        to PULL -- e.g. a remote agent reading its inbox over HTTP -- opens one and calls poll(name). Returns the name."""
        with self._lock:
            self._mailboxes[name] = {"patterns": tuple(patterns), "queue": deque()}
        return name

    def poll(self, name, limit=None):
        """Pull and REMOVE the pending messages from mailbox `name` (oldest first). `limit` caps how many. Returns a
        list of Messages (empty if none waiting or the mailbox doesn't exist)."""
        with self._lock:
            mb = self._mailboxes.get(name)
            if mb is None:
                return []
            q = mb["queue"]
            n = len(q) if limit is None else min(limit, len(q))
            return [q.popleft() for _ in range(n)]

    def close_mailbox(self, name):
        with self._lock:
            self._mailboxes.pop(name, None)

    # ---- history ----------------------------------------------------------------------------------------
    def history(self, pattern="*", limit=None):
        """The recent messages matching `pattern`, oldest first -- for a late joiner to catch up or to replay. `limit`
        returns just the most recent that many."""
        with self._lock:
            msgs = [m for m in self.log if topic_matches(pattern, m.topic)]
        return msgs[-limit:] if limit else msgs

    def _log_handler_error(self, topic, err):
        # record a handler failure as a message so it's visible, but never raise out of publish()
        with self._lock:
            self._seq += 1
            self.log.append(Message(self._seq, "bus.handler_error",
                                    {"topic": topic, "error": repr(err)}, sender="bus"))


def _selftest():
    bus = MessageBus()

    # pub/sub with a callback
    seen = []
    unsub = bus.subscribe("render.*", lambda m: seen.append((m.topic, m.payload)))
    bus.publish("render.start", {"w": 320})
    bus.publish("render.done", {"shape": [240, 320, 3]})
    bus.publish("sim.step", {"t": 1})                       # not a render.* -> handler not called
    assert seen == [("render.start", {"w": 320}), ("render.done", {"shape": [240, 320, 3]})], seen

    # deterministic ids/order: same sequence of publishes -> same ids
    b2 = MessageBus()
    ids1 = [b2.publish("a").id, b2.publish("b").id]
    b3 = MessageBus()
    ids2 = [b3.publish("a").id, b3.publish("b").id]
    assert ids1 == ids2, (ids1, ids2)

    # unsubscribe stops delivery
    unsub()
    bus.publish("render.done", {"again": True})
    assert len(seen) == 2, "unsubscribed handler should not fire"

    # mailboxes: a PULL subscriber
    bus.open_mailbox("agent", ["render.*", "to:agent"])
    bus.publish("render.done", {"n": 1})
    bus.send("agent", {"hello": True})
    pulled = bus.poll("agent")
    assert [m.topic for m in pulled] == ["render.done", "to:agent"], [m.topic for m in pulled]
    assert bus.poll("agent") == [], "a second poll drains nothing new"

    # directed send + point-to-point
    got = []
    bus.subscribe("to:app", lambda m: got.append(m.payload))
    bus.send("app", {"ping": 1}, sender="agent")
    assert got == [{"ping": 1}]

    # a broken handler doesn't break publish; the failure is logged
    bus.subscribe("boom", lambda m: (_ for _ in ()).throw(RuntimeError("x")))
    bus.publish("boom")
    assert any(m.topic == "bus.handler_error" for m in bus.history()), "handler error should be logged"

    # topic matching rules
    assert topic_matches("*", "anything")
    assert topic_matches("a.*", "a") and topic_matches("a.*", "a.b.c")
    assert not topic_matches("a.*", "ab") and not topic_matches("a.b", "a.c")

    # history catch-up
    assert len(bus.history("render.*")) >= 3

    print("OK: holographic_bus self-test passed (pub/sub by topic + wildcard; deterministic message ids; "
          "unsubscribe; pull mailboxes; directed send; a broken handler is logged not fatal; history catch-up)")


if __name__ == "__main__":
    _selftest()

"""holographic_agent_bridge.py -- connect an OPTIONAL agent (an LLM) to the bus, and run tasks that announce when done.

This is the piece that lets the app REACH OUT to an agent instead of the agent polling. You give the bridge one thing
-- a callable `llm(text) -> reply` (your own function around any model; the bridge imports no LLM library, so leCore
stays dependency-free and works with no agent at all) -- and tell it which topics the agent should be told about. When
one fires, the bridge formats the message into a prompt, calls your llm(), and posts the reply back on the bus as
'agent.reply'. So: a render finishes -> the app publishes 'render.done' -> the bridge calls the LLM 'here's the render,
does it look right?' -> the LLM's answer lands on the bus for the person (or the app) to read.

  run_task(bus, 'render', fn, background=True, summarize=...) runs a job and publishes 'render.start' then
  'render.done' (or 'render.error'); the summary it carries is what the agent sees (an LLM can't read a NumPy image,
  so you hand it shape/stats/a saved path, not the array).

Everything is optional: with no bridge, the events still fire and get logged; with a bridge but llm=None, the bridge
wires up but simply notes that no agent is attached. Deterministic + stdlib only (threading for the background case).
"""
import threading


class AgentBridge:
    """A thin, LLM-agnostic connector between the bus and an agent. `llm` is any callable text->text (or None).
    Nothing here depends on a specific model or SDK -- you bring the function."""

    def __init__(self, bus, llm=None, name="agent"):
        self.bus = bus
        self.llm = llm
        self.name = name

    def notify_on(self, topic, prompt=None):
        """When `topic` fires, tell the agent. If an llm is attached, format the message (plus an optional `prompt`
        like 'does this look right?'), call the llm, and publish its reply as 'agent.reply'. If no llm is attached,
        publish an 'agent.unattached' note so it's visible that the event WOULD have gone to an agent. Chainable."""
        def handler(msg):
            if self.llm is None:
                self.bus.publish("agent.unattached", {"about": msg.topic, "message_id": msg.id}, sender=self.name)
                return
            text = self._format(msg, prompt)
            try:
                reply = self.llm(text)
            except Exception as e:                          # a failing agent must not break the app
                self.bus.publish("agent.error", {"about": msg.topic, "error": repr(e)}, sender=self.name)
                return
            self.bus.publish("agent.reply", {"about": msg.topic, "in_reply_to": msg.id, "reply": reply},
                             sender=self.name)
        self.bus.watch(topic, handler)
        return self

    def ask(self, text, wait=True):
        """The person or the app asks the agent something directly. Publishes 'agent.ask', calls the llm, publishes
        the answer as 'agent.reply', and returns the reply text (or None if no agent is attached)."""
        self.bus.publish("agent.ask", {"text": text}, sender="app")
        if self.llm is None:
            return None
        reply = self.llm(text)
        self.bus.publish("agent.reply", {"reply": reply}, sender=self.name)
        return reply

    def on_reply(self, handler):
        """Be called with each agent reply (so a UI or the app can show what the agent said). `handler(message)`."""
        return self.bus.subscribe("agent.reply", handler)

    def _format(self, msg, prompt):
        """Turn a bus Message into a readable prompt for the llm: the topic, a compact view of the payload, and any
        extra instruction. Kept plain so you can see exactly what the agent is told."""
        lines = ["[leCore event] %s" % msg.topic]
        if msg.payload is not None:
            lines.append("details: %s" % _compact(msg.payload))
        if prompt:
            lines.append(prompt)
        return "\n".join(lines)


def _compact(payload, limit=600):
    """A short, readable string for a payload (dict/list/scalar). Long values are truncated -- the agent gets a
    SUMMARY, never a giant array dumped in."""
    try:
        s = repr(payload)
    except Exception:
        s = "<unreprable payload>"
    return s if len(s) <= limit else s[:limit] + " ...(truncated)"


def run_task(bus, name, fn, *args, background=False, summarize=None, sender="app", **kwargs):
    """Run `fn(*args, **kwargs)` as a named task that ANNOUNCES itself on the bus: it publishes '<name>.start' before
    and '<name>.done' after (or '<name>.error' if it raises). The 'done' message carries summarize(result) if you give
    a summariser, else a light auto-summary -- that summary is what an agent watching '<name>.done' will see.

    background=False (default): runs now, returns the result.
    background=True: runs in a thread and returns the Thread immediately, so the caller isn't blocked -- the bus's
      '<name>.done' is how everyone (including the agent) finds out it finished. This is the whole point: nobody polls.
    """
    def _body():
        bus.publish(name + ".start", {"args": len(args)}, sender=sender)
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            bus.publish(name + ".error", {"error": repr(e)}, sender=sender)
            raise
        summary = summarize(result) if summarize else _auto_summary(result)
        bus.publish(name + ".done", summary, sender=sender)
        return result

    if not background:
        return _body()
    t = threading.Thread(target=_body, name="task:" + name, daemon=True)
    t.start()
    return t


def _auto_summary(result):
    """A best-effort small summary of a task result for the 'done' message: array-like -> shape/min/max/mean; a short
    dict/scalar -> itself; anything else -> its type. Never dumps a big object onto the bus."""
    try:
        import numpy as np
        if isinstance(result, np.ndarray):
            return {"kind": "array", "shape": list(result.shape),
                    "min": float(result.min()), "max": float(result.max()), "mean": float(result.mean())}
    except Exception:
        pass
    if isinstance(result, (int, float, bool, str)) or result is None:
        return {"kind": "value", "value": result}
    if isinstance(result, dict) and len(repr(result)) <= 600:
        return {"kind": "dict", "value": result}
    return {"kind": type(result).__name__}


def _selftest():
    from holographic.misc.holographic_bus import MessageBus

    # a fake 'llm' so the test needs no model: it just echoes what it was told
    calls = []

    def fake_llm(text):
        calls.append(text)
        return "looks good" if "render.done" in text else "ok"

    bus = MessageBus()
    bridge = AgentBridge(bus, llm=fake_llm)
    replies = []
    bridge.on_reply(lambda m: replies.append(m.payload["reply"]))

    # the app should REACH the agent when a render finishes -- no polling
    bridge.notify_on("render.done", "Here's the finished render -- does it look right?")

    import numpy as np
    img = np.zeros((4, 4, 3))
    result = run_task(bus, "render", lambda: img, summarize=lambda a: {"shape": list(a.shape)})
    assert result is img
    assert calls and "render.done" in calls[0] and "does it look right" in calls[0]     # the llm was invoked
    assert replies == ["looks good"], replies                                           # its reply is on the bus
    # the start+done events were published
    assert any(m.topic == "render.start" for m in bus.history())
    assert any(m.topic == "render.done" for m in bus.history())

    # ask() -- a direct question to the agent
    ans = bridge.ask("what can you do?")
    assert ans == "ok" and any(m.topic == "agent.ask" for m in bus.history())

    # OPTIONAL: with no agent attached, everything still runs; an 'unattached' note is posted
    bus2 = MessageBus()
    bridge2 = AgentBridge(bus2, llm=None)
    bridge2.notify_on("render.done")
    assert bridge2.ask("hello?") is None                       # no agent -> no answer, no crash
    run_task(bus2, "render", lambda: 42)
    assert any(m.topic == "agent.unattached" for m in bus2.history())   # the event still fired, just noted

    # a task that raises publishes '<name>.error'
    bus3 = MessageBus()
    try:
        run_task(bus3, "job", lambda: (_ for _ in ()).throw(ValueError("boom")))
    except ValueError:
        pass
    assert any(m.topic == "job.error" for m in bus3.history())

    # background: the task runs off-thread and the bus tells us when it's done
    bus4 = MessageBus()
    done = threading.Event()
    bus4.watch("bg.done", lambda m: done.set())
    t = run_task(bus4, "bg", lambda: sum(range(1000)), background=True)
    assert done.wait(timeout=5.0), "background task should publish bg.done"
    t.join(timeout=1.0)

    print("OK: holographic_agent_bridge self-test passed (a finished render REACHES the agent with a summary + gets a "
          "reply on the bus; ask() answers directly; with no agent attached everything still runs (unattached note); "
          "a failing task publishes .error; a background task announces .done off-thread -- no polling)")


if __name__ == "__main__":
    _selftest()

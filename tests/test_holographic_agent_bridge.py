"""Tests for holographic_agent_bridge.py -- optional LLM bridge + task runner that announces done."""
import threading
import numpy as np
from holographic.misc.holographic_bus import MessageBus
from holographic.agents_and_reasoning.holographic_agent_bridge import AgentBridge, run_task


def _echo_llm(calls):
    def llm(text):
        calls.append(text)
        return "reply:" + text.splitlines()[0]
    return llm


def test_render_done_reaches_the_agent():
    bus = MessageBus()
    calls = []
    bridge = AgentBridge(bus, llm=_echo_llm(calls))
    replies = []
    bridge.on_reply(lambda m: replies.append(m.payload["reply"]))
    bridge.notify_on("render.done", "does it look right?")
    run_task(bus, "render", lambda: np.zeros((4, 4, 3)), summarize=lambda a: {"shape": list(a.shape)})
    assert calls and "render.done" in calls[0] and "does it look right" in calls[0]
    assert replies and replies[0].startswith("reply:")


def test_task_publishes_start_and_done():
    bus = MessageBus()
    run_task(bus, "job", lambda: 7)
    topics = [m.topic for m in bus.history()]
    assert "job.start" in topics and "job.done" in topics


def test_task_error_is_published_and_reraised():
    bus = MessageBus()
    try:
        run_task(bus, "job", lambda: 1 / 0)
        assert False, "should reraise"
    except ZeroDivisionError:
        pass
    assert any(m.topic == "job.error" for m in bus.history())


def test_ask_directly():
    bus = MessageBus()
    bridge = AgentBridge(bus, llm=lambda t: "hi")
    assert bridge.ask("hello") == "hi"
    assert any(m.topic == "agent.ask" for m in bus.history())


def test_optional_no_agent_still_runs():
    bus = MessageBus()
    bridge = AgentBridge(bus, llm=None)
    bridge.notify_on("render.done")
    assert bridge.ask("hello") is None                     # no crash, no answer
    run_task(bus, "render", lambda: 1)
    assert any(m.topic == "agent.unattached" for m in bus.history())


def test_agent_error_does_not_break_the_app():
    bus = MessageBus()
    bridge = AgentBridge(bus, llm=lambda t: 1 / 0)         # a broken agent
    bridge.notify_on("render.done")
    run_task(bus, "render", lambda: 1)                     # must not raise
    assert any(m.topic == "agent.error" for m in bus.history())


def test_background_task_announces_off_thread():
    bus = MessageBus()
    done = threading.Event()
    bus.watch("bg.done", lambda m: done.set())
    t = run_task(bus, "bg", lambda: sum(range(100)), background=True)
    assert done.wait(timeout=5.0)
    t.join(timeout=1.0)


def test_auto_summary_shrinks_an_array():
    bus = MessageBus()
    run_task(bus, "r", lambda: np.ones((8, 8, 3)))
    done = [m for m in bus.history() if m.topic == "r.done"][0]
    assert done.payload["kind"] == "array" and done.payload["shape"] == [8, 8, 3]

"""Regression traps for llm_tool (work plan item 4.1).

The plumbing is small. The thing worth pinning is the CONSEQUENCE: once the model is a registered tool it
can be failed over away from. That demonstration is the competitive claim, so it is a test rather than a
docstring, and it is written so a regression in the breaker or the registration both break it.
"""
import pytest

import lecore
from holographic.scene_and_pipeline.holographic_orchestrator import CircuitBreaker, Tool


@pytest.fixture
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_attach_llm_alone_does_not_register_a_tool(mind):
    """THE GAP THIS ITEM CLOSES, pinned. attach_llm's docstring says the model is "now usable as a tool";
    that was aspirational. If a future change makes attach_llm register automatically, this fails and the
    llm_tool docstring's rationale needs rewriting."""
    before = len(mind.orchestrator.tools())
    mind.attach_llm(lambda t: t.upper())
    assert len(mind.orchestrator.tools()) == before, \
        "attach_llm now registers a tool by itself; llm_tool's rationale is stale"


def test_llm_tool_registers_the_attached_model(mind):
    mind.attach_llm(lambda t: t.upper())
    tool = mind.llm_tool(name="llm", description="rewrite text")
    assert isinstance(tool, Tool)
    assert tool.fn("hello") == "HELLO"
    # orchestrator.tools() returns NAMES, not Tool objects -- checked rather than assumed, because the
    # first version of this assertion read t.name off a string.
    assert "llm" in list(mind.orchestrator.tools())


def test_llm_tool_refuses_when_no_model_is_attached(mind):
    # Silently registering nothing would be worse than raising: the planner would look complete and never
    # reach a language tool, which is exactly the failure mode this item exists to fix.
    with pytest.raises(RuntimeError):
        mind.llm_tool()


def test_a_callable_can_be_registered_without_attaching(mind):
    tool = mind.llm_tool(name="direct", llm=lambda t: t[::-1])
    assert tool.fn("abc") == "cba"


def test_a_non_callable_is_refused(mind):
    with pytest.raises(TypeError):
        mind.orchestrator.register_llm(42)


def test_failures_reach_the_breaker_by_default(mind):
    # on_error='raise' is the default SO THAT the breaker can see failures. A swallowed error trips nothing,
    # which would leave a dead model in the plan forever.
    def boom(_):
        raise RuntimeError("model timeout")

    tool = mind.llm_tool(name="llm", llm=boom)
    with pytest.raises(RuntimeError):
        tool.fn("hi")


def test_on_error_empty_degrades_but_hides_the_failure(mind):
    # Documented and available, but not the default, precisely because it hides failures from the breaker.
    def boom(_):
        raise RuntimeError("model timeout")

    tool = mind.llm_tool(name="llm", llm=boom, on_error="empty")
    assert tool.fn("hi") == ""


def test_a_flaky_model_is_failed_over_away_from(mind):
    """THE DEMONSTRATION. A flaky model's breaker must open and the planner must then be offered only the
    deterministic tool. This is the thing a system whose only mechanism IS the model cannot do."""
    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        raise RuntimeError("model timeout")

    llm = mind.llm_tool(name="llm", description="rewrite text", llm=flaky)
    det = mind.orchestrator.register_llm(lambda t: t.upper(), name="deterministic_rewrite",
                                         description="rewrite text")
    breaker = CircuitBreaker(fail_max=3, cooldown=5)
    chosen = []
    for _ in range(6):
        breaker.new_cycle()
        available = [t for t in (llm, det) if breaker.available(t)]
        assert available, "the breaker starved the planner of every tool"
        pick = available[0]
        try:
            pick.fn("hello")
            ok = True
        except Exception:
            ok = False
        breaker.report(pick, ok)
        chosen.append(pick.name)

    assert calls["n"] == 3, "the flaky model was called %d times, not 3" % calls["n"]
    assert chosen[-1] == "deterministic_rewrite", "no reroute happened: %r" % chosen
    assert breaker.status(llm) == "open"


def test_the_registered_model_is_just_a_tool(mind):
    # No special-casing: it must carry the same keyword vector and success-rate machinery as anything else,
    # or the planner cannot reason about it uniformly.
    tool = mind.llm_tool(name="llm", description="rewrite text", llm=lambda t: t)
    assert tool.vec is not None
    assert 0.0 < tool.success_rate() < 1.0
    assert tool.in_type == "text" and tool.out_type == "text"


def test_llm_tool_is_discoverable(mind):
    for query in ("let the planner use the language model", "register an llm as a tool",
                  "fail over away from a flaky model"):
        assert "planner-visible" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces llm_tool" % query


# --------------------------------------------------------------------------------------
# Off-machine door (work plan item 4.5).
# --------------------------------------------------------------------------------------

def test_a_farm_worker_must_be_a_name_not_a_callable(mind):
    """The farm's security property is that ONLY DATA CROSSES THE WIRE. It used to hold BY ACCIDENT --
    handing a callable raised 'Object of type function is not JSON serializable' from inside the encoder,
    which enforces the rule without ever stating it. An accidental guarantee is one refactor away from not
    being a guarantee."""
    farm = mind.farm(["localhost:9999"])
    with pytest.raises(TypeError) as exc:
        farm.run([[1, 2]], lambda x: x, None, None)
    message = str(exc.value)
    assert "must be a NAME" in message
    assert "never code" in message


def test_the_off_machine_preference_is_recorded_before_it_is_needed():
    # No rung leaves the machine today, so this is a decision written down in advance -- and the two
    # mechanisms look interchangeable from a distance, which is exactly why it needs writing down.
    from holographic.agents_and_reasoning import holographic_declare as hd
    doc = hd.__doc__ or ""
    assert "THE DOOR IS `farm`, NOT `command_tool`" in doc

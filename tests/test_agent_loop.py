"""Regression traps for the in-process agent loop (work plan item 4.2).

The property under test is an ORDERING: the null-referenced gate runs BEFORE the model is consulted. A loop
that asks the model first and checks afterwards would pass most of these tests and fail the one that
counts (test_the_model_is_never_consulted_on_a_no_tool_task), so that one is written with a counting stub
rather than a checking one.
"""
import random

import pytest

import lecore
from holographic.agents_and_reasoning.holographic_agentloop import (AgentLoop, arg_fingerprint,
                                                                    parse_action)
from holographic.caching_and_storage.holographic_catalog import _tokens, default_catalog


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


@pytest.fixture(scope="module")
def arms():
    """Committed seeded fixture at MATCHED token count, so the arms differ in tool presence only."""
    cat = default_catalog()
    pairs = sorted({str(a) for c in cat.all() for a in (getattr(c, "aliases", ()) or [])
                    if getattr(c, "method", None) and len(_tokens(a)) == 4})
    random.Random(0).shuffle(pairs)
    has = pairs[:12]
    vocab = sorted({t for c in cat.all() for t in _tokens(c.does)})
    rng = random.Random(1)
    return has, [" ".join(rng.sample(vocab, 4)) for _ in has]


ALWAYS_DONE = lambda _: "DONE: finished"


# --------------------------------------------------------------------------------------
# The ordering, which is the whole architecture.
# --------------------------------------------------------------------------------------

def test_the_model_is_never_consulted_on_a_no_tool_task(mind):
    # THE TEST THAT MATTERS. A loop that consults the model and checks afterwards would still refuse, and
    # would still look correct -- until the model has a side effect. Counting is the only way to see it.
    calls = {"n": 0}

    def counting(_):
        calls["n"] += 1
        return "DONE: finished"

    out = AgentLoop(mind, counting).run("purple monkey dishwasher")
    assert out["refused"] and not out["done"]
    assert calls["n"] == 0, "the model was consulted %d times before the gate" % calls["n"]


@pytest.mark.slow                       # ~90 s: each distinct token count builds a cold router null
def test_false_action_rate_on_the_no_tool_arm_is_zero(mind, arms):
    # The stub ALWAYS claims completion, so it never abstains: any refusal here is the engine's.
    _, no_tool = arms
    acted = sum(AgentLoop(mind, ALWAYS_DONE, max_steps=2).run(t)["done"] for t in no_tool)
    assert acted == 0, "false-action rate %.0f%%" % (100 * acted / len(no_tool))


@pytest.mark.slow                       # ~90 s, same reason
def test_the_has_tool_arm_still_completes(mind, arms):
    # Refusing everything would trivially satisfy the test above.
    has_tool, _ = arms
    done = sum(AgentLoop(mind, ALWAYS_DONE, max_steps=2).run(t)["done"] for t in has_tool)
    assert done / len(has_tool) > 0.9, "only %d/%d completed" % (done, len(has_tool))


# --------------------------------------------------------------------------------------
# What the loop refuses to do.
# --------------------------------------------------------------------------------------

def test_a_tool_outside_the_manifest_is_refused(mind):
    out = AgentLoop(mind, lambda _: '{"tool": "os_system", "args": {}}', max_steps=1).run(
        "smooth a bumpy mesh")
    assert "not in the offered manifest" in out["steps"][0]["why"]


def test_non_finite_args_are_refused(mind):
    # json parses bare NaN by default, so a model CAN emit one and it arrives as a float.
    out = AgentLoop(mind, lambda _: '{"tool": "mesh_smooth", "args": {"iters": NaN}}',
                    max_steps=1).run("smooth a bumpy mesh")
    assert "non-finite" in out["steps"][0]["why"]


def test_an_unparsed_reply_is_never_turned_into_an_action(mind):
    out = AgentLoop(mind, lambda _: "hmm, perhaps we should try something", max_steps=2).run(
        "smooth a bumpy mesh")
    assert len(out["steps"]) == 2
    assert all("unparsed" in s["why"] and s["tool"] is None for s in out["steps"])
    assert not out["done"]


def test_only_callable_capabilities_are_advertised(mind):
    # Import-only capabilities cannot be invoked, so offering them would invite a refusal the model cannot
    # avoid -- and a manifest you cannot act on is worse than a shorter one.
    for row in AgentLoop(mind, ALWAYS_DONE).manifest("smooth a bumpy mesh"):
        assert row["tool"]
        assert mind.describe_skill(row["tool"]) is not None


# --------------------------------------------------------------------------------------
# Transcript hygiene and robustness.
# --------------------------------------------------------------------------------------

def test_args_are_recorded_as_a_digest_not_the_object():
    # A live object in a job's args once crashed a worker AFTER the job had succeeded. A transcript holding
    # caller objects also pins them and lets them be mutated after the fact.
    fp = arg_fingerprint({"a": [1, 2, 3]})
    assert set(fp) == {"digest", "repr"} and len(fp["digest"]) == 16
    assert arg_fingerprint({"a": 1}) == arg_fingerprint({"a": 1})
    assert arg_fingerprint({"a": 1}) != arg_fingerprint({"a": 2})


def test_parse_action_reads_json_and_done_and_refuses_the_rest():
    assert parse_action('{"tool": "x", "args": {"a": 1}}')["kind"] == "call"
    assert parse_action('sure! {"tool": "x"} hope that helps')["kind"] == "call"
    assert parse_action("DONE: all good")["answer"] == "all good"
    assert parse_action("no idea")["kind"] == "unparsed"
    assert parse_action("{not json}")["kind"] == "unparsed"


def test_a_model_that_raises_does_not_take_the_loop_down(mind):
    def boom(_):
        raise RuntimeError("model offline")

    out = AgentLoop(mind, boom).run("smooth a bumpy mesh")
    assert not out["done"] and "model raised" in out["why"]


def test_max_steps_is_honoured(mind):
    out = AgentLoop(mind, lambda _: "nonsense", max_steps=3).run("smooth a bumpy mesh")
    assert len(out["steps"]) == 3 and "ran out of steps" in out["why"]


def test_guards(mind):
    with pytest.raises(TypeError):
        AgentLoop(mind, 42)
    with pytest.raises(RuntimeError):
        lecore.UnifiedMind(dim=64, seed=0).tool_loop("smooth a bumpy mesh")


def test_the_loop_is_discoverable(mind):
    for query in ("let a model use my tools", "agent loop", "refuse a step when no tool fits"):
        assert "tool-use loop" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the agent loop" % query

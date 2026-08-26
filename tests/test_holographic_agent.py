"""Tests for the upgraded creature agent: affect, pain reflex, void-gap action synthesis (AGENT-1)."""

import numpy as np
from holographic.agents_and_reasoning.holographic_ai import random_vector
from holographic.agents_and_reasoning.holographic_agent import Agent


def _state(seed):
    return random_vector(512, np.random.default_rng(seed))


def test_reward_makes_the_agent_pick_an_action():
    ag = Agent(["N", "S", "E", "W"], dim=512, seed=0)
    s = _state(1)
    ag.reward(s, "E", 1.0)
    d = ag.decide(s)
    assert d["source"] == "value" and d["action"] == "E"


def test_pain_reflex_blocks_an_action_after_one_event():
    ag = Agent(["N", "S", "E", "W"], dim=512, seed=0)
    s = _state(2)
    ag.pain(s, "N", 1.0)                                      # a single painful experience
    d = ag.decide(s)
    assert "N" in d["avoided"]                                # blocked immediately, no value convergence needed


def test_reward_and_pain_compose():
    ag = Agent(["N", "S", "E", "W"], dim=512, seed=0)
    s = _state(3)
    ag.reward(s, "E", 1.0).pain(s, "N", 1.0)
    d = ag.decide(s)
    assert d["action"] == "E" and "N" in d["avoided"]


def test_void_gap_synthesizes_a_plan_for_a_reachable_goal():
    ag = Agent(["N", "S", "E", "W", "A", "B"], dim=512, seed=1)
    s = _state(10)                                            # a novel state, no learned values
    goal = ag.program_signature(["E", "A", "W"])             # reachable from the action library
    d = ag.decide(s, goal_vec=goal)
    assert d["source"] == "synthesized" and len(d["program"]) >= 1
    assert d["coherence"] >= ag.synth_threshold


def test_void_gap_abstains_on_an_unreachable_goal():
    ag = Agent(["N", "S", "E", "W"], dim=512, seed=1)
    s = _state(11)
    junk = random_vector(512, np.random.default_rng(999))    # not reachable from the library
    d = ag.decide(s, goal_vec=junk)
    assert d["source"] == "abstain" and "action" in d        # falls back to a safe default, does not execute junk


def test_plan_signature_is_embeddable_in_a_program():
    ag = Agent(["grab", "lift", "place"], dim=512, seed=5)
    sig = ag.program_signature(["grab", "lift", "place"])
    assert sig.shape == (512,)
    # a single-action "plan" recalls its own atom
    one = ag.program_signature(["grab"])
    from holographic.agents_and_reasoning.holographic_ai import cosine
    assert cosine(one, ag.action_vec["grab"]) > 0.99


def test_decide_explains_itself():
    ag = Agent(["N", "S", "E", "W"], dim=512, seed=0)
    d = ag.decide(_state(7))                                  # no learning, no goal
    assert "why" in d and d["source"] in ("explore", "value", "abstain")

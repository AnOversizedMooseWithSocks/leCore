"""An upgraded creature agent (AGENT-1): affect (reward AND pain), an action LIBRARY as VSA atoms, a pain-driven
avoidance REFLEX, and -- the headline -- VOID-GAP ACTION SYNTHESIS: when no learned single action fits the
situation, the agent SYNTHESISES a multi-step action program toward a goal and gates it (verify or abstain),
reusing the SYNTH-1 loop.

WHY A NEW LAYER (not an edit to HolographicMind)
------------------------------------------------
The existing creature RL engine (HolographicMind) is deterministic, tie-sensitive (a 1e-16 change flips a maze
trajectory, by its own kept-negative), and built for one job: greedy value-recall over per-action prototypes. This
adds capabilities AROUND it without disturbing that suite:
  * ACTIONS AS ATOMS. Each action is a near-orthogonal hypervector, so a sequence of actions has a composed
    signature (chain_signature) -- which means actions can be SYNTHESISED (voidsynth) and EMBEDDED in VSA programs
    (the agent can drive a program, the thing the user noted is now possible).
  * AFFECT: reward AND pain. Reward folds a positive value into the (state->action) memory; pain folds a negative
    value AND records the (state, action) as something to AVOID -- a separate, faster channel than slow value
    learning.
  * PAIN REFLEX. Before consulting values, the agent checks whether a candidate (state, action) strongly resembles
    a remembered painful one; if so it is excluded immediately -- a safety reflex, not a learned gradient.
  * VOID-GAP ACTION SYNTHESIS. When no allowed action has confident value here (support below a floor -- the
    agent's OWN void gap), and a goal is given, it synthesises an action PROGRAM toward the goal, verifies the
    coherence, and either commits the program or ABSTAINS to a safe default. This is the creature analogue of
    filling a registry void: don't flail randomly, compose a plan -- but only if it verifies.

Everything is deterministic and self-explaining (decide returns WHY). Honest scope: the value memory here is a
simple soft-kNN (the bespoke per-action prototype engine still lives in HolographicMind and beats a generic memory
-- see its docstring); this layer is about AFFECT + SYNTHESIS + the action ATOMS, not a better Q-estimator.
"""

import numpy as np
from holographic_ai import bind, bundle, cosine, random_vector
from holographic_orchestrator import chain_signature
from holographic_voidsynth import synthesize_for_goal


class Agent:
    """A creature agent with an action library (VSA atoms), reward/pain affect, a pain-avoidance reflex, and
    void-gap action-program synthesis. Deterministic and self-explaining."""

    def __init__(self, actions, dim=512, seed=0, value_floor=0.25, pain_reflex=0.6, synth_threshold=0.8):
        self.actions = list(actions)
        self.dim = int(dim)
        self.value_floor = float(value_floor)               # min support to trust a learned action here
        self.pain_reflex = float(pain_reflex)               # cosine above which a remembered pain blocks an action
        self.synth_threshold = float(synth_threshold)       # coherence a synthesised program must clear
        rng = np.random.default_rng(seed)
        self.action_vec = {a: random_vector(dim, rng) for a in self.actions}   # the atoms
        self.library = np.stack([self.action_vec[a] for a in self.actions])
        self.mem = {a: [] for a in self.actions}            # action -> list of (state_vec, value) prototypes
        self.pain_trace = []                                # list of (state_vec, action) that hurt -> avoid

    # ---- affect inputs --------------------------------------------------------------------------
    def reward(self, state_vec, action, r=1.0):
        """Positive reinforcement: fold (state, +r) into the action's value memory."""
        self._remember(state_vec, action, float(r))
        return self

    def pain(self, state_vec, action, p=1.0):
        """Negative reinforcement AND an avoidance memory: fold (state, -p) into value, and record the painful
        (state, action) so the reflex can block it fast next time -- a separate, quicker channel than value
        learning."""
        self._remember(state_vec, action, -float(p))
        self.pain_trace.append((np.asarray(state_vec, float), action))
        return self

    def _remember(self, state_vec, action, value):
        s = np.asarray(state_vec, float)
        proto = self.mem[action]
        for i, (ps, pv) in enumerate(proto):                # fold into a near-matching prototype (denoise the return)
            if cosine(s, ps) >= 0.92:
                n = self.mem[action][i]
                proto[i] = (ps + s, pv + (value - pv) * 0.3)   # running mean of value, bundle of states
                return
        proto.append((s.copy(), value))                     # else a new prototype

    # ---- value + support ------------------------------------------------------------------------
    def value(self, state_vec, action):
        """Similarity-weighted value of `action` in this state (soft-kNN over its prototypes). 0 if unknown."""
        s = np.asarray(state_vec, float)
        proto = self.mem[action]
        if not proto:
            return 0.0
        sims = np.array([max(0.0, cosine(s, ps)) for ps, _ in proto])
        vals = np.array([pv for _, pv in proto])
        w = sims ** 2
        return float((w @ vals) / (w.sum() + 1e-9))

    def support(self, state_vec, action):
        """How well this state is RECOGNISED for an action (max prototype cosine) -- the confidence that any value
        estimate here is grounded. Low support = the agent's own void gap."""
        s = np.asarray(state_vec, float)
        proto = self.mem[action]
        return 0.0 if not proto else float(max(cosine(s, ps) for ps, _ in proto))

    def _pain_blocked(self, state_vec, action):
        s = np.asarray(state_vec, float)
        return any(a == action and cosine(s, ps) >= self.pain_reflex for ps, a in self.pain_trace)

    # ---- decide (the upgraded policy) -----------------------------------------------------------
    def decide(self, state_vec, goal_vec=None, allowed=None):
        """Choose what to do, returning a dict {action|program, source, why, ...}. Order:
          1. PAIN REFLEX -- drop any allowed action that strongly resembles a remembered painful (state, action).
          2. VALUE -- if any surviving action is confidently recognised here (support >= value_floor), take the
             highest-value one (ties -> the first, deterministically).
          3. VOID-GAP SYNTHESIS -- if nothing is confidently known here and a `goal_vec` is given, synthesise an
             action PROGRAM toward the goal and, if it clears `synth_threshold`, commit it; else
          4. ABSTAIN -- fall back to the first non-blocked allowed action (a safe default), flagged honestly."""
        allowed = list(self.actions if allowed is None else allowed)
        safe = [a for a in allowed if not self._pain_blocked(state_vec, a)]
        blocked = [a for a in allowed if a not in safe]
        if not safe:
            safe = allowed                                  # everything hurts here: don't freeze, pick something
        known = [a for a in safe if self.support(state_vec, a) >= self.value_floor]
        if known:
            best = max(known, key=lambda a: self.value(state_vec, a))
            return {"action": best, "source": "value", "value": round(self.value(state_vec, best), 3),
                    "avoided": blocked, "why": f"highest-value recognised action ({len(known)} known here)"}
        if goal_vec is not None:                            # the void gap: synthesise a plan
            res = synthesize_for_goal(self.library, goal_vec, max_length=4, threshold=self.synth_threshold)
            if res["status"] == "synthesized":
                program = [self.actions[i] for i in res["chain"]]
                return {"program": program, "source": "synthesized", "coherence": round(res["coherence"], 3),
                        "avoided": blocked, "why": f"no learned action fit; synthesised a {len(program)}-step plan"}
            return {"action": safe[0], "source": "abstain", "coherence": round(res["coherence"], 3),
                    "avoided": blocked, "why": "void gap: no coherent plan found, taking a safe default"}
        return {"action": safe[0], "source": "explore", "avoided": blocked,
                "why": "no learned value and no goal given; safe default"}

    def program_signature(self, program):
        """The composed VSA signature of an action program -- so a synthesised plan can be EMBEDDED in / blended
        with other VSA programs (the agent drives a program)."""
        return chain_signature(np.stack([self.action_vec[a] for a in program]))


def _selftest():
    ag = Agent(["N", "S", "E", "W"], dim=512, seed=0)
    rng = np.random.default_rng(1)
    s = random_vector(512, rng)                              # a state
    ag.reward(s, "E", 1.0).pain(s, "N", 1.0)                # E is good here, N hurts
    d = ag.decide(s)
    assert d["source"] == "value" and d["action"] == "E", d  # picks the rewarded action
    assert "N" in d["avoided"]                               # and avoids the painful one (reflex)
    # void gap: a brand-new state, no learned values, but a reachable goal -> synthesise a plan
    s2 = random_vector(512, rng)
    goal = ag.program_signature(["E", "N", "E"])             # a reachable 3-step goal
    d2 = ag.decide(s2, goal_vec=goal)
    assert d2["source"] == "synthesized" and len(d2["program"]) >= 1, d2
    # unreachable goal -> abstain to a safe default, do NOT execute junk
    junk = random_vector(512, rng)
    d3 = ag.decide(s2, goal_vec=junk)
    assert d3["source"] == "abstain", d3
    print(f"agent selftest ok: reward->{d['action']} (avoided {d['avoided']}); void gap synthesised "
          f"{d2['program']} (coh {d2['coherence']}); unreachable goal -> {d3['source']}")


if __name__ == "__main__":
    _selftest()

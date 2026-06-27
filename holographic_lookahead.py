"""Re-anchored lookahead for the creature -- D4 (cross-cutting: RAY-1 re-anchoring -> model-based planning). KEPT NEGATIVE.

THE PROPOSAL (the research item; Baker's and Adamatzky's seats): the creature (HolographicMind) is purely
REACTIVE -- it picks the action with the best learned value in the current state, with no forward model. Give it
one: learn a per-action transition operator from its own experience, ROLL it out a few steps to imagine the
consequences of each action, and pick the action whose rollout looks best. The RAY-1 lesson says the rollout will
need RE-ANCHORING -- clean up each predicted state to the manifold of real states each step, or the prediction
error compounds multiplicatively and deep lookahead is useless (the same reason a bind-chain needs a cleanup at
every hop).

WHAT THE MEASUREMENT SAID -- two parts, both kept:

  1. THE MECHANISM WORKS (RAY-1 confirmed in a new domain). A per-action bind-displacement forward model
     (delta_a = mean unbind(next, state) over observed transitions; predict(s,a) = bind(s, delta_a)) rolled out
     naively DEGRADES with depth -- cosine(predicted, true) falls 0.65 -> 0.53 over four steps as the error
     compounds. RE-ANCHORING the predicted state to a codebook of seen states each step keeps it ON-MANIFOLD:
     cosine stays ~constant at 0.77 across depth. Re-anchoring is exactly as load-bearing here as it is for
     bind-chain traversal.

  2. BUT THE APPLICATION IS REDUNDANT (the kept negative). Re-anchored lookahead does not improve the creature's
     decisions: it ranks the four actions IDENTICALLY to the plain reactive value function -- 98-100% action-rank
     agreement -- so it can never choose differently, and on stars collected it ties the reactive policy (and
     loses the reactive policy's small epsilon, so it is marginally worse). The PRECISE root cause, diagnosed:
     the four predicted leaves sit at 0.974-0.994 PAIRWISE COSINE -- a single per-action bind displacement
     COLLAPSES all actions to nearly the same predicted next-state, because in the egocentric sense-space the
     AVERAGE sense-change is similar across directions (the directional specificity is lost in the averaging). So
     the lookahead bonus barely varies across actions (std 0.0075) while the reactive value varies a lot (std
     0.37) -- lookahead carries no differentiated signal. (A secondary, structural reason also holds: the
     creature's value IS the Monte-Carlo discounted return, already horizon-aware, so model-based planning is
     recomputing -- less accurately, through a noisy model -- what the model-free value already encodes.)

THE CROSS-CUTTING LESSON (the throughline with C4, D1, B1): the re-anchoring transfer is mechanically sound, but
the creature's state space does not admit a forward model good enough for the application. The structural
mismatch is egocentric-sense-space averaging: a per-action linear/bind operator cannot capture the
position-dependent, action-specific consequences a real lookahead would need, so it predicts the same future for
every action and the planner is blind. A right technique applied to an operation whose shape defeats it -- the
same failure as the splat sharpener (deconvolving a sum that is not a blur), low-discrepancy exploration
(independent points for a sequential walk), and MIS generation (a gate, not a second estimator).

No faculty, no tour line -- the finding is the negative.
"""

import numpy as np
from holographic_ai import bind, unbind, cosine


def learn_action_deltas(transitions, dim):
    """Learn a per-action bind-displacement forward model from observed transitions. `transitions[a]` is a list
    of (state_vec, next_state_vec) pairs for action a. Returns a list of unit delta vectors, one per action, with
    delta_a = normalize(mean unbind(next, state)) -- the HRR estimate of the transformation that bind applies to
    carry a state to its successor under action a. (Measured to be too COARSE for lookahead: it captures only the
    average sense-change, which is similar across directions -- see the module docstring.)"""
    deltas = []
    for a in range(len(transitions)):
        if transitions[a]:
            d = np.mean([unbind(s2, s) for (s, s2) in transitions[a]], axis=0)
            deltas.append(d / (np.linalg.norm(d) + 1e-12))
        else:
            deltas.append(np.zeros(dim))
    return deltas


def predict_next(state_vec, delta_a):
    """One forward-model step: predict the next state by binding the current state with the action's learned
    displacement. Unit-normalised."""
    p = bind(state_vec, delta_a)
    return p / (np.linalg.norm(p) + 1e-12)


def reanchor(pred_vec, codebook):
    """RE-ANCHOR a predicted state to the manifold of real states: snap it to the nearest state in `codebook` (a
    matrix of unit row vectors). This is what keeps a multi-step rollout from compounding error -- the RAY-1
    cleanup-every-hop lesson, applied to a learned forward model. (Confirmed to keep rollout cosine ~constant
    across depth where the naive rollout degrades -- see the module docstring.)"""
    p = pred_vec / (np.linalg.norm(pred_vec) + 1e-12)
    return codebook[int((codebook @ p).argmax())]


def _selftest():
    """D4 (kept negative): re-anchored lookahead is redundant with the creature's reactive value function. CI-fast
    (~1s): train a small creature, learn the per-action forward model, and measure the two facts that sink the
    application -- (a) the per-action forward model COLLAPSES the actions (the four predicted leaves are nearly
    identical), so (b) the model-based lookahead ranks the actions the SAME as the plain reactive value (it can
    never decide differently). The re-anchoring mechanism itself works; there is simply nothing for it to improve."""
    from holographic_creature import GridWorld, CreatureEncoder, HolographicMind, run_episode
    dim = 512
    rng = np.random.default_rng(0)
    world = GridWorld(width=5, height=5, n_poison=1, seed=0)
    enc = CreatureEncoder(dim, seed=1)
    mind = HolographicMind(dim, actions=["N", "S", "E", "W"], seed=2)
    for _ in range(50):
        run_episode(world, enc, mind, learn=True, explore=True, max_steps=40, mem=0)

    # collect transitions under the trained (still exploring) policy
    trans = [[] for _ in range(4)]
    seen = []
    senses = world.reset()
    s = enc.build_state(senses, [], 0)
    for _ in range(60):
        for _t in range(40):
            a = mind.decide(s, explore=True, epsilon=0.4) if np.linalg.norm(s) > 1e-9 else int(rng.integers(4))
            a = mind.actions.index(a) if isinstance(a, str) else a
            senses2, _r, _ate, alive = world.step(mind.actions[a])
            s2 = enc.build_state(senses2, [], 0)
            if np.linalg.norm(s) > 1e-9 and np.linalg.norm(s2) > 1e-9:
                trans[a].append((s, s2))
                seen.append(s2)
            s = s2
            if not alive:
                senses = world.reset()
                s = enc.build_state(senses, [], 0)
                break

    deltas = learn_action_deltas(trans, dim)
    SB = np.stack(seen)
    SB = SB / (np.linalg.norm(SB, axis=1, keepdims=True) + 1e-12)

    def lookahead_score(state, a0, gamma=0.9):
        v0, _ = mind.value(state, a0)
        leaf = reanchor(predict_next(state, deltas[a0]), SB)
        return v0 + gamma * max(mind.value(leaf, a1)[0] for a1 in range(4))

    agree = 0
    n = 0
    leaf_cos = []
    senses = world.reset()
    s = enc.build_state(senses, [], 0)
    for _ in range(40):
        if np.linalg.norm(s) > 1e-9:
            a_reactive = int(np.argmax([mind.value(s, a)[0] for a in range(4)]))
            a_look = int(np.argmax([lookahead_score(s, a0) for a0 in range(4)]))
            agree += int(a_reactive == a_look)
            n += 1
            leaves = [reanchor(predict_next(s, deltas[a0]), SB) for a0 in range(4)]
            leaf_cos.append(np.mean([cosine(leaves[i], leaves[j]) for i in range(4) for j in range(i + 1, 4)]))
        a = mind.decide(s, explore=True, epsilon=0.3)
        a = mind.actions.index(a) if isinstance(a, str) else a
        senses, _r, _ate, alive = world.step(mind.actions[a])
        s = enc.build_state(senses, [], 0)
        if not alive:
            senses = world.reset()
            s = enc.build_state(senses, [], 0)

    leaf_collapse = float(np.mean(leaf_cos))
    rank_agreement = agree / max(1, n)
    # the forward model collapses the actions: a per-action bind displacement predicts nearly the same next-state
    assert leaf_collapse > 0.9, leaf_collapse
    # so lookahead cannot decide differently from the reactive value -- it is redundant
    assert rank_agreement > 0.9, rank_agreement


if __name__ == "__main__":
    _selftest()
    print("holographic_lookahead D4 negative selftest passed "
          "(forward model collapses actions; lookahead == reactive value)")

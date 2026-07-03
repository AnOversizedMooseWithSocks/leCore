"""holographic_condprop.py -- the CONDITIONAL PROPAGATOR: one learned dynamics operator PER ACTION, so an agent
gets model-based planning from the very same "predict = bind a transform to a state" mechanism the dynamics
module already uses.

WHY THIS EXISTS (Above/Below Sweep 3 -- the unification the sweep surfaced)
--------------------------------------------------------------------------
The sweep noticed that four things are ONE operation -- bind a transform onto a state:
  * `dynamics.Propagator`  : state(t+1) = bind(U, state)                  (advance)
  * `lookahead`            : a per-ACTION forward model                   (predict, was a coarse mean-delta)
  * `video`                : motion encoded as a bind                     (motion-compensate)
  * `backwardwarp`         : reproject with unbind                        (the inverse bind)
So `lookahead`'s per-action model should just BE a Propagator, conditioned on the action. `lookahead`'s kept
negative was that a single averaged delta vector per action is too COARSE -- it captures only the mean
sense-change, similar across directions. A Propagator learns the full per-FREQUENCY transfer instead (a whole
operator, not one delta), which is the richer form, and it comes with a regularised INVERSE for free -- so
planning is content-addressable (you can ask "what state was I in before that action?").

This module is deliberately thin: a ConditionalPropagator is just a dict of `dynamics.Propagator`s, one per
action, learned from that action's observed (state -> next_state) transitions. It REUSES the dynamics module
rather than reimplementing the transfer -- the unification only counts if the code is actually one (§5.1).

HONEST SCOPE (kept loud): a linear-in-Fourier operator per action -- genuinely nonlinear, chaotic action
dynamics need a lift (the reservoir, `chaos`), the same boundary `dynamics` documents. Multi-step rollout
compounds error unless you cleanup-every-hop; `plan()` therefore re-anchors to a state codebook each step (the
RAY-1 lesson). Deterministic; NumPy + stdlib; delegates to holographic_dynamics.Propagator and the lookahead
re-anchor.
"""
import numpy as np

from holographic_dynamics import Propagator
from holographic_lookahead import reanchor


class ConditionalPropagator:
    """One `dynamics.Propagator` per action. `predict(state, a) = bind(U_a, state)`; `back(state, a)` runs the
    action's inverse operator (content-addressable planning); `plan(state, actions, codebook)` rolls a sequence
    out, re-anchoring each hop so error does not compound."""

    def __init__(self, propagators):
        self.props = dict(propagators)                       # action -> Propagator

    @classmethod
    def learn(cls, transitions, ridge=1e-3):
        """Learn one Propagator per action. `transitions` maps an action (a list index, or a dict key) to a list
        of (state_vec, next_state_vec) pairs observed under that action. An action with no data is skipped."""
        items = transitions.items() if isinstance(transitions, dict) else enumerate(transitions)
        props = {}
        for a, pairs in items:
            if not pairs:
                continue
            X0 = np.stack([np.asarray(s, float) for s, _ in pairs])
            X1 = np.stack([np.asarray(s2, float) for _, s2 in pairs])
            props[a] = Propagator.learn_pairs(X0, X1, ridge=ridge)
        return cls(props)

    def actions(self):
        return list(self.props.keys())

    def predict(self, state, a):
        """Predict the next state under action `a` -- a single bind with that action's learned operator."""
        return self.props[a].step(np.asarray(state, float))

    def back(self, state, a):
        """Recover the state BEFORE action `a` -- a single bind with the action's regularised inverse operator.
        This is what makes a plan content-addressable (query the state just before a chosen action)."""
        from holographic_ai import bind
        return bind(self.props[a].U_inv, np.asarray(state, float))

    def plan(self, state, actions, codebook=None):
        """Roll out a SEQUENCE of actions from `state`, re-anchoring to `codebook` (a matrix of unit state rows)
        after each hop so multi-step error does not compound (the cleanup-every-hop discipline). Returns the
        predicted end state. Without a codebook it is the raw (degrading) rollout, kept available for comparison."""
        s = np.asarray(state, float)
        for a in actions:
            s = self.predict(s, a)
            if codebook is not None:
                s = reanchor(s, codebook)
        return s


def _selftest():
    """One Propagator per action predicts that action's successor accurately (per-action, not a single averaged
    delta); on a state-graph whose places are codebook atoms, re-anchored planning composes actions EXACTLY across
    depth where the naive rollout degrades; the inverse operator recovers the prior state; deterministic."""
    from holographic_ai import bind, cosine
    rng = np.random.default_rng(0)
    D = 512

    # a state GRAPH: K "places" (codebook atoms), and each action is a permutation of places (place i -> place
    # perm_a[i]). This is a world where planning stays ON the codebook manifold, so re-anchor is exact.
    K = 8
    n_actions = 3
    places = rng.standard_normal((K, D)); places /= np.linalg.norm(places, axis=1, keepdims=True)
    perms = [np.roll(np.arange(K), a + 1) for a in range(n_actions)]     # action a shifts every place by a+1

    def place_index(v):
        return int((places @ (v / (np.linalg.norm(v) + 1e-12))).argmax())

    # observe every place's transition under every action, learn one Propagator per action
    transitions = [[(places[i], places[perms[a][i]]) for i in range(K)] for a in range(n_actions)]
    cp = ConditionalPropagator.learn(transitions)

    # (1) each action's predictor lands on the correct next PLACE (re-anchored), and is action-specific
    for a in range(n_actions):
        for i in range(K):
            assert place_index(cp.predict(places[i], a)) == perms[a][i], (a, i)
    # predicting place 0 with action 0 vs action 1 lands on different places (genuinely per-action)
    assert place_index(cp.predict(places[0], 0)) != place_index(cp.predict(places[0], 1))

    # (2) content-addressable: the inverse operator recovers the prior place
    nxt = cp.predict(places[3], 2)
    assert place_index(cp.back(nxt, 2)) == 3

    # (3) re-anchored planning composes actions EXACTLY across depth; the naive rollout degrades
    start = 0
    plan_actions = [0, 1, 2, 0, 1, 2, 0]                                 # 7 hops -- deep enough to compound error
    true_end = start
    for a in plan_actions:
        true_end = perms[a][true_end]
    end_anchored = cp.plan(places[start], plan_actions, codebook=places)
    end_naive = cp.plan(places[start], plan_actions, codebook=None)
    assert place_index(end_anchored) == true_end                        # exact after 7 hops with re-anchor
    anchored_ok = place_index(end_anchored) == true_end
    naive_ok = place_index(end_naive) == true_end
    cos_anchored = cosine(end_anchored, places[true_end])
    cos_naive = cosine(end_naive, places[true_end])
    assert cos_anchored > cos_naive                                     # re-anchor keeps the signal; naive erodes it

    # (4) per-action prediction cosine on the training places (report it)
    per_action = [float(np.mean([cosine(cp.predict(places[i], a), places[perms[a][i]]) for i in range(K)]))
                  for a in range(n_actions)]

    # (5) deterministic
    cp2 = ConditionalPropagator.learn(transitions)
    assert np.allclose(cp.predict(places[0], 0), cp2.predict(places[0], 0))
    print("holographic_condprop selftest OK: one Propagator per action lands every place transition exactly "
          "(per-action cos %s); re-anchored planning composes 7 hops exactly (naive cos %.2f vs anchored %.2f); "
          "the inverse recovers the prior place; deterministic" % (["%.2f" % a for a in per_action], cos_naive, cos_anchored))


if __name__ == "__main__":
    _selftest()

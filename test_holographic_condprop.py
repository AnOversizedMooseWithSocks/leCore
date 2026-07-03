"""Sweep 3 item 9: conditional Propagator -- one Propagator per action, planning via re-anchor."""
import numpy as np
from holographic_condprop import ConditionalPropagator
from holographic_ai import cosine


def _world(D=512, K=8, n_actions=3, seed=0):
    rng = np.random.default_rng(seed)
    places = rng.standard_normal((K, D)); places /= np.linalg.norm(places, axis=1, keepdims=True)
    perms = [np.roll(np.arange(K), a + 1) for a in range(n_actions)]
    transitions = [[(places[i], places[perms[a][i]]) for i in range(K)] for a in range(n_actions)]
    return places, perms, transitions


def _idx(places, v):
    return int((places @ (v / (np.linalg.norm(v) + 1e-12))).argmax())


def test_per_action_transitions_exact():
    places, perms, tr = _world()
    cp = ConditionalPropagator.learn(tr)
    for a in range(len(perms)):
        for i in range(len(places)):
            assert _idx(places, cp.predict(places[i], a)) == perms[a][i]
    assert _idx(places, cp.predict(places[0], 0)) != _idx(places, cp.predict(places[0], 1))


def test_inverse_recovers_prior():
    places, perms, tr = _world()
    cp = ConditionalPropagator.learn(tr)
    nxt = cp.predict(places[3], 2)
    assert _idx(places, cp.back(nxt, 2)) == 3


def test_reanchored_planning_composes_at_depth():
    places, perms, tr = _world()
    cp = ConditionalPropagator.learn(tr)
    plan = [0, 1, 2, 0, 1, 2, 0]
    true_end = 0
    for a in plan:
        true_end = perms[a][true_end]
    end_anchored = cp.plan(places[0], plan, codebook=places)
    end_naive = cp.plan(places[0], plan, codebook=None)
    assert _idx(places, end_anchored) == true_end
    assert cosine(end_anchored, places[true_end]) > cosine(end_naive, places[true_end])


def test_deterministic():
    _, _, tr = _world()
    a = ConditionalPropagator.learn(tr); b = ConditionalPropagator.learn(tr)
    _, _, _ = _world()
    import numpy as np
    assert np.allclose(a.predict(np.ones(512) / np.sqrt(512), 0), b.predict(np.ones(512) / np.sqrt(512), 0))

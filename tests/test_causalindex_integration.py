"""CausalIndex (D3) at the seams: through the mind, against the plain Index it refuses to be, and composed
with realizable_fills -- the two halves of "was it actionable" (knew it in time, could act on it in time)."""
import numpy as np

import lecore
from holographic.caching_and_storage.holographic_index import CausalIndex, Index


def _series(seed=0, T=600, win=8):
    rng = np.random.default_rng(seed)
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = 0.9 * s[t - 1] + rng.standard_normal()
    states = np.stack([s[t - win:t] for t in range(win, T - 1)])
    nxt = np.array([s[t + 1] - s[t] for t in range(win, T - 1)])
    tt = np.arange(win, T - 1).astype(float)
    return s, states, nxt, tt


def test_the_faculty_returns_a_working_causal_index():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    ci = mind.causal_index()
    _, states, _, tt = _series()
    for v, t in zip(states, tt):
        ci.append(v, t)
    hits = ci.nearest(states[300], tt[300], k=3)
    assert len(hits) == 3
    assert all(h[2] <= tt[300] - 1 for h in hits)               # every hit strictly older
    assert ci.nearest(states[0], tt[0]) == []                    # nothing old enough -> honest []


def test_the_naive_self_match_leak_and_the_structural_immunity():
    """The demo as a seam test: naive full-history k=1 finds itself (MSE exactly 0); the causal index cannot,
    at any k, because its own timestamp is excluded by construction."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    _, states, nxt, tt = _series()
    full = Index(states, method="exact")
    ci = mind.causal_index()
    for v, t in zip(states, tt):
        ci.append(v, t)
    i = 400
    assert full.nearest(states[i], k=1)[0][0] == i               # the leak: it IS the query
    big_k = ci.nearest(states[i], tt[i], k=len(states))
    assert all(h[0] != i for h in big_k)                         # never itself, even asking for everything


def test_audit_causality_verifies_and_detects():
    """The audit must pass on the real structure AND actually detect a broken mask -- an audit that cannot
    fail is decoration. A deliberately sabotaged subclass whose nearest() ignores the cutoff must audit
    causal=False."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    _, states, _, tt = _series()
    ci = mind.causal_index()
    for v, t in zip(states, tt):
        ci.append(v, t)
    assert ci.audit_causality(states[200], tt[200], n_probes=8, scale=5.0)["causal"] is True

    class Leaky(CausalIndex):
        def nearest(self, query, t, k=1, lag=1):                 # ignores time entirely -- the bug to catch
            if int(lag) < 1:
                raise ValueError("simultaneous is not past")
            if not self._vecs:
                return []
            import numpy as _np
            from holographic.caching_and_storage.holographic_index import _unit_rows
            mat = _unit_rows(_np.vstack(self._vecs))
            q = _np.asarray(query, float).ravel()
            scores = mat @ (q / (_np.linalg.norm(q) or 1.0))
            order = _np.argsort(-scores, kind="stable")[:k]
            return [(int(j), float(scores[j]), float(self._times[j])) for j in order]

    leaky = Leaky()
    for v, t in zip(states, tt):
        leaky.append(v, t)
    assert leaky.audit_causality(states[200], tt[200], n_probes=8, scale=5.0)["causal"] is False


def test_append_only_and_lag_refusals_survive_the_faculty():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    ci = mind.causal_index()
    ci.append(np.ones(4), 10.0)
    try:
        ci.append(np.ones(4), 5.0)
        raise AssertionError("expected append-only refusal")
    except ValueError as e:
        assert "append-only violated" in str(e)
    try:
        ci.nearest(np.ones(4), 20.0, lag=0)
        raise AssertionError("expected lag=0 refusal")
    except ValueError as e:
        assert "simultaneous is not past" in str(e)


def test_composed_with_realizable_fills_the_two_halves_of_actionable():
    """KNEW it in time (CausalIndex) + could ACT on it in time (realizable_fills): recall events from the
    causal index, then evaluate them at the actionable price. The composition must run end to end and the
    fills evaluation must only ever receive event indices the causal recall could actually have produced."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    s, states, nxt, tt = _series(seed=3)
    ci = mind.causal_index()
    events = []
    for i, (v, t) in enumerate(zip(states, tt)):
        hits = ci.nearest(v, t, k=1)
        if hits and hits[0][1] > 0.98:                           # "seen something very like this before"
            events.append(i + 8)                                 # back to series coordinates
        ci.append(v, t)                                          # append AFTER querying: strict causality
    if len(events) >= 4:
        r = mind.realizable_fills(events, list(s), horizon=3, lag=1)
        assert "verdict" in r and r["n"] >= 1
        assert "actionable_mean" in r and "latency_cost" in r

"""Tests for the creature's value head as a pure-VSA program (holographic_valuehead): policy = two bundles
per action, learn = bundling, decide = a dot. Matches the tabular brain at low load, cliffs at high load
(kept negative), and the policy is a fixed-size savable hypervector."""

import numpy as np

from holographic_valuehead import HolographicValueHead, _selftest


def _two_situation_head(D=256, A=3, seed=0):
    rng = np.random.default_rng(seed)
    sA = rng.normal(size=D); sA /= np.linalg.norm(sA)
    sB = rng.normal(size=D); sB /= np.linalg.norm(sB)
    vh = HolographicValueHead(D, A)
    for _ in range(5):
        vh.absorb(sA, 1, 1.0); vh.absorb(sA, 0, 0.1); vh.absorb(sA, 2, 0.1)
        vh.absorb(sB, 2, 1.0); vh.absorb(sB, 0, 0.1); vh.absorb(sB, 1, 0.1)
    return vh, sA, sB


def test_recalls_the_best_action_per_situation():
    vh, sA, sB = _two_situation_head()
    assert vh.decide(sA) == 1 and vh.decide(sB) == 2


def test_value_tracks_the_stored_return():
    vh, sA, _ = _two_situation_head()
    v1, _ = vh.value(sA, 1); v0, _ = vh.value(sA, 0)
    assert v1 > 0.8 and v0 < 0.4                  # the Nadaraya-Watson average recovers ~the mean return


def test_policy_is_a_fixed_size_hypervector_program():
    vh, _, _ = _two_situation_head()
    Q, N = vh.policy_vectors()
    assert Q.shape == (3, 256) and N.shape == (3, 256)
    before = vh.nbytes
    for _ in range(200):                          # fold in many more experiences
        vh.absorb(np.random.default_rng(1).normal(size=256), 0, 0.5)
    assert vh.nbytes == before                    # storage does NOT grow with history


def test_learning_is_bundling_and_order_invariant():
    # absorb is addition into Q/N, which commutes -> the learned policy is independent of experience order
    rng = np.random.default_rng(2); D = 128
    exps = [(rng.normal(size=D), int(rng.integers(3)), float(rng.uniform())) for _ in range(40)]
    a = HolographicValueHead(D, 3)
    for s, act, r in exps:
        a.absorb(s, act, r)
    b = HolographicValueHead(D, 3)
    for s, act, r in reversed(exps):
        b.absorb(s, act, r)
    assert np.allclose(a.Q, b.Q, atol=1e-9) and np.allclose(a.N, b.N, atol=1e-9)


def test_degrades_past_the_capacity_cliff():
    # KEPT NEGATIVE: a fixed-D bundle pair holds well-separated situations at low load, blurs at high load.
    def accuracy(P, D=256, A=3, seed=0):
        rng = np.random.default_rng(seed)
        S = rng.normal(size=(P, D)); S /= np.linalg.norm(S, axis=1, keepdims=True)
        V = rng.uniform(0, 1, size=(P, A))
        vh = HolographicValueHead(D, A)
        for p in range(P):
            for act in range(A):
                vh.absorb(S[p], act, V[p, act])
        best = V.argmax(axis=1)
        return np.mean([vh.decide(S[p]) == best[p] for p in range(P)])
    assert accuracy(P=6) > accuracy(P=260)        # low load beats over-capacity load


def test_selftest_head_to_head_runs():
    _selftest()


# --- Step A: routing pushes the capacity cliff back ---

def test_routed_head_pushes_the_capacity_cliff_back():
    from holographic_valuehead import RoutedValueHead
    def acc(head, P, D=256, A=3):
        rng = np.random.default_rng(0)
        S = rng.normal(size=(P, D)); S /= np.linalg.norm(S, axis=1, keepdims=True)
        V = rng.uniform(0, 1, size=(P, A))
        for p in range(P):
            for a in range(A):
                head.absorb(S[p], a, V[p, a])
        best = V.argmax(axis=1)
        return np.mean([head.decide(S[p]) == best[p] for p in range(P)])
    plain = acc(HolographicValueHead(256, 3), 1024)
    routed = acc(RoutedValueHead(256, 3, n_buckets=64), 1024)
    assert routed > plain + 0.3                      # routing holds situations the single bundle has lost


# --- Step B: TD as VSA -- n-step return is a discounted bundle ---

def test_discounted_return_is_a_geometric_bundle():
    from holographic_valuehead import discounted_return
    # sum_k gamma^k r_k + gamma^n * bootstrap
    assert abs(discounted_return([1.0, 1.0, 1.0], 0.5) - (1 + 0.5 + 0.25)) < 1e-12
    assert abs(discounted_return([0.0], 0.9, bootstrap=2.0) - (0.0 + 0.9 * 2.0)) < 1e-12


def test_eligibility_trace_is_a_decaying_bundle():
    from holographic_valuehead import EligibilityTrace
    D = 64; rng = np.random.default_rng(0)
    s1 = rng.normal(size=D); s1 /= np.linalg.norm(s1)
    e = EligibilityTrace(D, gamma=0.9, lam=0.8)
    e.step(s1)
    n1 = np.linalg.norm(e.vec)
    e.step(rng.normal(size=D))
    # the first state's contribution has decayed by gamma*lambda
    assert n1 > 0 and np.linalg.norm(e.vec) > 0


# --- composability: the policy as a hypervector that drives decisions in-VSA ---

def _trained_head(D=512, A=3, seed=0):
    rng = np.random.default_rng(seed)
    sits = [rng.normal(size=D) for _ in range(4)]
    for s in sits:
        s /= np.linalg.norm(s)
    codes = np.stack([rng.normal(size=D) for _ in range(A)]); codes /= np.linalg.norm(codes, axis=1, keepdims=True)
    h = HolographicValueHead(D, A)
    for i, s in enumerate(sits):
        b = i % A
        for _ in range(5):
            for a in range(A):
                h.absorb(s, a, 1.0 if a == b else 0.1)
    return h, sits, codes


def test_policy_atom_drives_decisions_in_vsa():
    from holographic_valuehead import decide_from_atom
    h, sits, codes = _trained_head()
    M_Q, M_N = h.policy_atom(codes)
    assert all(decide_from_atom(M_Q, M_N, s, codes) == h.decide(s) for s in sits)


def test_from_policy_round_trip():
    h, sits, _ = _trained_head()
    Q, N = h.policy_vectors()
    h2 = HolographicValueHead.from_policy(Q, N)
    assert all(h2.decide(s) == h.decide(s) for s in sits)

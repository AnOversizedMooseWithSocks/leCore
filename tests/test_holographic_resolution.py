"""Coarse-to-fine cleanup: exact when there's an answer, cheap when it's easy,
honest (degrades to full scan on ties, abstains on no-match), plus the
resolution-profile / stabilisation measurement."""
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import random_vector
from holographic.misc.holographic_resolution import coarse_to_fine, full_scan, resolution_profile, stabilisation_dim


def _store(n=300, D=4096, seed=0):
    rng = np.random.default_rng(seed)
    V = np.array([random_vector(D, rng) for _ in range(n)])
    return V, rng


def test_coarse_to_fine_matches_full_scan_when_answer_exists():
    # Over queries with a real nearest neighbour, coarse-to-fine returns the SAME
    # winner as a full-dimension scan -- the gate only stops when the ranking is
    # statistically settled.
    V, rng = _store()
    agree = 0
    for _ in range(100):
        t = rng.integers(len(V))
        q = V[t] + rng.uniform(0.4, 2.0) * random_vector(V.shape[1], rng)
        cf = coarse_to_fine(q, V)[0]
        fl = full_scan(q, V)[0]
        agree += (cf == fl)
    assert agree == 100


def test_easy_queries_use_far_fewer_dimensions():
    # A strong match resolves at low resolution: the dimension-work is a small
    # fraction of a full scan over the store.
    V, rng = _store()
    full_cost = V.shape[0] * V.shape[1]
    used = []
    for _ in range(50):
        t = rng.integers(len(V))
        q = V[t] + 0.4 * random_vector(V.shape[1], rng)        # strong match
        _, _, dims, _ = coarse_to_fine(q, V)
        used.append(dims / full_cost)
    assert np.mean(used) < 0.4                                  # measured ~0.05


def test_no_match_abstains():
    # With no real neighbour the top score is tiny; a min_score floor returns -1
    # (abstain) rather than promoting noise.
    V, rng = _store()
    q = random_vector(V.shape[1], rng)
    idx, score, _, _ = coarse_to_fine(q, V, min_score=0.2)
    assert idx == -1
    assert score < 0.2


def test_degrades_to_full_scan_on_ties_without_error():
    # THE HONEST BOUNDARY: when many candidates are near-ties, the gate cannot
    # resolve cheaply and escalates to full width -- still correct, just no saving.
    rng = np.random.default_rng(1)
    D = 1024
    base = random_vector(D, rng)
    # 40 nearly-identical vectors (all tiny perturbations of base): genuine ties
    V = np.array([base + 0.02 * random_vector(D, rng) for _ in range(40)])
    q = base + 0.02 * random_vector(D, rng)
    _, _, _, stopped_k = coarse_to_fine(q, V)
    assert stopped_k == D                                       # had to go full width
    assert coarse_to_fine(q, V)[0] == full_scan(q, V)[0]        # still exact


def test_resolution_profile_and_stabilisation():
    # The persistent-homology measurement: a clear match stabilises at low
    # resolution (robust to truncation); the profile reports the winner per scale.
    V, rng = _store(n=100)
    t = 7
    q = V[t] + 0.3 * random_vector(V.shape[1], rng)             # very strong
    prof = resolution_profile(q, V)
    assert prof[-1][1] == t                                     # full-dim winner is the target
    assert stabilisation_dim(q, V) <= V.shape[1] // 2           # settled before full width


def test_cleanup_coarse_equals_full():
    # The wired-in path: _cleanup with coarse-to-fine returns the same symbol as
    # the exhaustive scan over a large vocabulary.
    from holographic.agents_and_reasoning.holographic_ai import Vocabulary
    from holographic.misc.holographic_relations import _cleanup
    voc = Vocabulary(2048, seed=0)
    names = [f"s{i}" for i in range(150)]
    for n in names:
        voc.get(n)
    rng = np.random.default_rng(0)
    agree = 0
    for _ in range(60):
        t = names[rng.integers(len(names))]
        probe = voc.get(t) + 1.0 * random_vector(2048, rng)
        agree += (_cleanup(probe, names, voc, coarse=True)[0]
                  == _cleanup(probe, names, voc, coarse=False)[0])
    assert agree == 60

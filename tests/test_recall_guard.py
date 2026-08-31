"""Recall guard: the certificate as a checkable theorem, the head-preservation contract, and the
never-overclaim rule -- plus the mind round-trip."""
import numpy as np

from holographic.caching_and_storage.holographic_perfectrecall import PerfectRecallIndex
from holographic.semantic_router.holographic_recallguard import guard_candidates


def _mk(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vocab = ["w%02d" % i for i in range(50)]
    docs, idx = [], PerfectRecallIndex(tile=64)
    for _ in range(n):
        terms = list({vocab[int(z) % 50] for z in rng.zipf(1.4, size=int(rng.integers(4, 9)))})
        docs.append(set(terms)); idx.add({"token": terms})
    return docs, idx


def test_certificate_is_a_theorem():
    docs, idx = _mk()
    q = ["w03", "w07", "w11"]
    cand, cert = guard_candidates(list(range(100)), q, idx, budget=900)
    c = cert["complete_down_to"]
    cs = set(cand)
    for i, d in enumerate(docs):
        if sum(t in d for t in q) >= c:
            assert i in cs


def test_ranked_head_is_never_repainted():
    docs, idx = _mk(seed=1)
    ranked = [5, 3, 999, 42]
    cand, _ = guard_candidates(ranked, ["w01", "w02"], idx, budget=500)
    assert cand[:4] == ranked


def test_small_budget_admits_less_never_lies():
    docs, idx = _mk(seed=2)
    q = ["w01", "w04"]
    _, big = guard_candidates([], q, idx, budget=1500)
    _, small = guard_candidates([], q, idx, budget=40)
    assert small["complete_down_to"] >= big["complete_down_to"]
    # and whatever the small one DOES claim still holds
    cand_s, cert_s = guard_candidates([], q, idx, budget=40)
    cs = set(cand_s); c = cert_s["complete_down_to"]
    for i, d in enumerate(docs):
        if sum(t in d for t in q) >= c:
            assert i in cs


def test_deterministic():
    docs, idx = _mk(seed=3)
    a = guard_candidates([9, 1], ["w05", "w06"], idx, budget=300)
    b = guard_candidates([9, 1], ["w05", "w06"], idx, budget=300)
    assert a == b


def test_mind_faculty_round_trip():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    ix = m.perfect_recall_index(tile=8)
    ix.add({"token": ["cat", "sat"]}); ix.add({"token": ["cat"]}); ix.add({"token": ["dog"]})
    cand, cert = m.guard_candidates([2], ["cat", "sat"], ix, budget=10)
    assert cand[0] == 2 and 0 in cand and cert["complete_down_to"] <= 2

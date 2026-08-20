"""Perfect-recall index: the guarantee as exact set equality, on synthetic AND real data.

The real-data test is the one that matters for the 'no silly custom data' bar: on the SciFact corpus
(if the benchmark data is present locally) every containment query must return exactly the brute-force
truth. Skipped cleanly when the data folder is absent (CI does not fetch it).
"""
import os
import numpy as np
import pytest

from holographic.caching_and_storage.holographic_perfectrecall import PerfectRecallIndex


def _mk(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    vocab = ["w%03d" % i for i in range(400)]
    docs, idx = [], PerfectRecallIndex(tile=128)
    for _ in range(n):
        terms = list({vocab[min(399, int(z))] for z in rng.zipf(1.3, size=int(rng.integers(6, 18)))})
        docs.append(set(terms)); idx.add({"token": terms})
    return docs, idx


def test_exact_equality_against_brute_force():
    docs, idx = _mk()
    rng = np.random.default_rng(1)
    for _ in range(40):
        src = docs[int(rng.integers(0, len(docs)))]
        q = list(rng.choice(sorted(src), size=min(3, len(src)), replace=False))
        truth = sorted(i for i, d in enumerate(docs) if all(t in d for t in q))
        assert idx.query(q) == truth                     # ==, not F1


def test_zero_false_negatives_is_structural_zero_false_positives_is_verified():
    docs, idx = _mk(1000, seed=2)
    # every doc must find itself from any subset of its own terms (no false negatives)...
    for i in (0, 137, 999):
        sub = sorted(docs[i])[:2]
        assert i in idx.query(sub)
    # ...and a term set no doc contains returns exactly [] (no false positives)
    assert idx.query(["w000", "w001", "w002", "w003", "w399"]) == [] or all(
        {"w000", "w001", "w002", "w003", "w399"} <= docs[i]
        for i in idx.query(["w000", "w001", "w002", "w003", "w399"]))


def test_tiling_changes_cost_never_the_answer():
    docs, coarse = _mk(2000, seed=3)
    fine = PerfectRecallIndex(tile=32)
    for d in docs:
        fine.add({"token": sorted(d)})
    q = ["w010", "w030"]
    s1, s2 = {}, {}
    assert coarse.query(q, stats=s1) == fine.query(q, stats=s2)
    assert s2["docs_tested"] <= s1["docs_tested"]        # finer tiles never test more


def test_multi_channel_independence():
    ix = PerfectRecallIndex(tile=8)
    ix.add({"token": ["alpha"], "trigram": ["alp", "lph", "pha"]})
    ix.add({"token": ["beta"], "trigram": ["bet", "eta"]})
    assert ix.query(["alp"], channel="trigram") == [0]
    assert ix.query(["alpha"], channel="token") == [0]
    assert ix.query(["alp"], channel="token") == []      # channels do not leak


SCIFACT = "/home/claude/bench/scifact-retrieval-system-main/data/corpus.jsonl"


@pytest.mark.skipif(not os.path.exists(SCIFACT), reason="benchmark data not fetched")
def test_perfect_recall_on_real_scifact_corpus():
    import json
    from holographic.semantic_router.holographic_bm25 import tokenize
    docs = []
    for line in open(SCIFACT):
        d = json.loads(line)
        docs.append(set(tokenize(d.get("title", "") + " " + " ".join(d.get("abstract", [])))))
    idx = PerfectRecallIndex(tile=256)
    for d in docs:
        idx.add({"token": sorted(d)})
    rng = np.random.default_rng(0)
    for _ in range(25):
        src = docs[int(rng.integers(0, len(docs)))]
        q = list(rng.choice(sorted(src), size=min(3, len(src)), replace=False))
        truth = sorted(i for i, d in enumerate(docs) if all(t in d for t in q))
        assert idx.query(q) == truth                     # perfect recall on 5,183 REAL abstracts

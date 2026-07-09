"""Benchmark: holostuff's HoloForest (sublinear approximate nearest-neighbour) vs the exact brute-force
nearest-neighbour scan a practitioner reaches for when they just want the closest stored vector (BLD-2).

The honest question: does the forest's sublinearity actually pay off? Answer, measured below: the forest
matches exact recall@1 up to ~10k items and uses a small and shrinking FRACTION of the comparisons (down to
~3% at 20k) -- the structural win is real and always present. But on WALL-TIME the exact scan is `items @ q`,
a single BLAS matrix-vector product, which is so fast that the forest's pure-Python tree traversal only
overtakes it past ~20k items. So the standard tool wins on wall-clock at modest scale even though it loses on
comparison count -- kept on the record. (The right read: the forest buys you sublinear *work*, which matters
when each comparison is expensive or N is large; against a tight BLAS inner loop on small N, raw wall-time
favours the scan.)

Run:  python benchmarks/bench_recall.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import time
from holographic.misc.holographic_tree import HoloForest


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _unit_rows(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def compare_recall(D=512, noise=0.5, Q=60, Ns=(500, 2000, 8000, 20000), seed=0):
    """For each N: exact brute-force NN vs the forest -- recall@1, recall@8, comparisons/query, wall-time.
    Cues are the true item plus `noise` of a unit-norm random direction (cosine ~0.89 to truth at 0.5), so the
    exact scan recovers ~100% and the forest's number is its recall RELATIVE to exact."""
    rng = np.random.default_rng(seed)
    rows = []
    for N in Ns:
        items = _unit_rows(rng.standard_normal((N, D)))
        qs = [(int(i), _unit(items[int(i)] + noise * _unit(rng.standard_normal(D))))
              for i in rng.integers(N, size=Q)]
        forest = HoloForest(D, n_trees=4, leaf_size=64, seed=0)
        forest.build(items)
        t0 = time.perf_counter()
        brute1 = sum((items @ q).argmax() == i for i, q in qs)      # exact scan: N comparisons, one BLAS matvec
        brute_us = (time.perf_counter() - t0) / Q * 1e6
        t0 = time.perf_counter()
        f1 = fk = fc = 0
        for i, q in qs:
            f1 += (forest.recall(q, beam=4) == i)
            fk += (i in forest.recall_k(q, k=8, beam=4)[0])
            fc += forest.last_comparisons
        forest_us = (time.perf_counter() - t0) / Q * 1e6
        rows.append({"N": N, "brute_recall1": brute1 / Q, "forest_recall1": f1 / Q, "forest_recall8": fk / Q,
                     "brute_cmp": N, "forest_cmp": fc // Q, "brute_us": brute_us, "forest_us": forest_us,
                     "speedup": brute_us / forest_us})
    return rows


def _print(rows):
    print("Recall: HoloForest (sublinear approximate NN) vs exact brute-force scan")
    print("(recall@1 / recall@8, comparisons per query, microseconds per query)")
    for r in rows:
        sp = r["speedup"]
        print(f"  N={r['N']:6}: brute @1 {r['brute_recall1']:4.0%} ({r['brute_cmp']:6} cmp, {r['brute_us']:6.0f}us)  |  "
              f"forest @1 {r['forest_recall1']:4.0%} @8 {r['forest_recall8']:4.0%} "
              f"({r['forest_cmp']:5} cmp = {r['forest_cmp'] * 100 // r['N']:2}%, {r['forest_us']:6.0f}us)  ->  "
              f"{sp:4.1f}x {'faster' if sp > 1 else 'slower'}")


if __name__ == "__main__":
    _print(compare_recall())

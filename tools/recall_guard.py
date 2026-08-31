"""recall_guard.py -- HONEST RECALL SELECTION (cp107): pick the cleanup path by measurement,
not by which module sounds most sophisticated.

Two kept negatives and one kept positive from cp104-cp106 are compiled here into code.

B3 -- THE FOREST GUARD.
    cp105 found the occlusion forest losing to brute force on BOTH recall and speed.
    cp106 rebuilt it per Pharr's diagnosis and he was right about accuracy: with 16
    trees / beam 16 recall@1 returns to 1.000 at N=1,095 and 0.975 at N=20,000 with
    32/32. But buying that accuracy costs traversal, and the tree stayed 30-100x SLOWER
    than a single BLAS matmul at every setting measured on this box.
    This is not a leCore-specific embarrassment; it is the expected result. The FAISS
    paper states plainly that in high dimensions, branch-and-bound methods provide no
    speedup over brute force search (arXiv:2401.08281 S3.1), because batched brute force
    is a matrix multiplication and BLAS is extremely hard to beat.
    So: recommend brute force below a measured crossover, and make the caller pass an
    explicit override to use the tree anyway.

B4 -- PEEL RECALL, the best measured rung.
    Iterative peeling (choose best match, subtract its projection, renormalise, repeat)
    beat flat cleanup on real correlated text at every working load in cp104:
        K=32  flat 0.920 -> peel 1.000
        K=64  flat 0.697 -> peel 0.998
    It is exposed here as a plain function so callers stop re-implementing it per script.

    Budgets are SEPARATE (cp104/cp105): peeling extends the CAPACITY budget (how many
    items may be bundled) and does nothing for the NOISE budget (how degraded a cue may
    be). At D=1024 the wall sits where the cue shares only ~0.53 cosine with the clean
    bundle, and no rung tested crosses it.

Usage:
    python3 tools/recall_guard.py            # run the self-check / measurement
    from tools.recall_guard import peel_recall, recommend_recall
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Measured on this box (cp106, D=1024, float32). Brute force won at every N tested up to
# 20k; no crossover was observed, so the guard reports "no measured crossover" rather than
# inventing one.
MEASURED_BRUTE_WINS_TO = 20000


def peel_recall(cue, codebook, k, max_iter=None):
    """Rung 4: iterative peeling. Returns indices of the k best-explaining atoms.

    codebook rows should be unit-norm. Deterministic; no RNG.
    """
    A = np.asarray(codebook)
    res = np.asarray(cue, dtype=np.float64).copy()
    n = np.linalg.norm(res)
    if n < 1e-12:
        return []
    res /= n
    chosen = []
    limit = max_iter or k
    for _ in range(limit):
        sims = A @ res
        best = int(np.argmax(sims))
        if best in chosen:
            break
        chosen.append(best)
        if len(chosen) >= k:
            break
        res = res - A[best] * float(A[best] @ res)
        n = np.linalg.norm(res)
        if n < 1e-9:
            break
        res /= n
    return chosen


def flat_recall(cue, codebook, k):
    """Rung 1: plain cleanup, for comparison."""
    sims = np.asarray(codebook) @ np.asarray(cue, dtype=np.float64)
    return list(np.argsort(sims)[-k:][::-1])


def recommend_recall(n_items, dim, use_tree=False):
    """B3 guard. Returns a recommendation dict; callers should honour 'method'."""
    if use_tree:
        return {"method": "forest", "reason": "explicit override by caller"}
    if n_items <= MEASURED_BRUTE_WINS_TO:
        return {
            "method": "brute",
            "reason": ("brute force measured faster at every N tested up to %d on this box "
                       "(cp106: forest 30-100x slower even at recall parity); high-dimensional "
                       "branch-and-bound gives no speedup over brute force -- FAISS "
                       "arXiv:2401.08281 S3.1" % MEASURED_BRUTE_WINS_TO),
            "measured_to": MEASURED_BRUTE_WINS_TO,
        }
    return {
        "method": "brute",
        "reason": ("beyond the measured range (N > %d): no crossover has been observed, so "
                   "the honest default is still brute force. Measure before switching."
                   % MEASURED_BRUTE_WINS_TO),
        "measured_to": MEASURED_BRUTE_WINS_TO,
    }


def _selfcheck():
    rng = np.random.default_rng(0)
    dim = 1024
    n = 600
    A = rng.standard_normal((n, dim))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    print("recall_guard self-check (D=%d, N=%d)" % (dim, n))
    print("  K     flat    peel")
    ok = True
    for k in (8, 16, 32, 64):
        f = p = 0.0
        trials = 20
        for _ in range(trials):
            idx = rng.choice(n, k, replace=False)
            cue = A[idx].sum(0)
            cue /= np.linalg.norm(cue)
            gt = set(int(i) for i in idx)
            f += len(set(flat_recall(cue, A, k)) & gt) / k
            p += len(set(peel_recall(cue, A, k)) & gt) / k
        f /= trials
        p /= trials
        print("  %-4d  %.3f   %.3f" % (k, f, p))
        if k <= 32 and p < f - 1e-9:
            ok = False
    rec = recommend_recall(n_items=n, dim=dim)
    print("\n  guard at N=%d -> %s" % (n, rec["method"]))
    print("  reason: %s" % rec["reason"])
    t0 = time.time()
    for _ in range(20):
        cue = A[0]
        np.argmax(A @ cue)
    print("  brute query: %.3f ms" % ((time.time() - t0) / 20 * 1000))
    print("\nSELFCHECK %s" % ("ok" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())

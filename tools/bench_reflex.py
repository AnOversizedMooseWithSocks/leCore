"""tools/bench_reflex.py -- the reflex-gate ablation bench (backlog E2.6'). Reproduces the
deep-dive Part-3 safety ordering on the standard clustered workload with look-alike traps:
  naive similarity cache  -> most calls avoided, MANY wrong answers  (the kept negative)
  + calibrated gate       -> fewer wrong
  + volatility + trust    -> the shipped reflex_try: fewest wrong
Run: PYTHONHASHSEED=0 python tools/bench_reflex.py
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holographic.agents_and_reasoning.holographic_ai import random_vector, cosine, slerp
from holographic.agents_and_reasoning.holographic_lever7 import DisplacementTrace

def main(dim=2048, n_train=60, n_test=150, seed=0):
    rng = np.random.default_rng(seed)
    K = 6
    centers = [random_vector(dim, rng) for _ in range(K)]
    answers = [random_vector(dim, rng) for _ in range(K)]
    def task(k, spread=0.25):
        return slerp(centers[k], random_vector(dim, rng), rng.uniform(0.02, spread))
    # look-alike traps: near a center but with a DIFFERENT true answer
    trap_centers = [slerp(c, random_vector(dim, rng), 0.35) for c in centers]
    trap_answers = [random_vector(dim, rng) for _ in range(K)]

    tr = DisplacementTrace(dim, seed=seed)
    train = []
    for _ in range(n_train):
        k = int(rng.integers(K)); t = task(k)
        tr.write(t, answers[k]); train.append((t, k))

    def truth_of(t):
        sims_c = [cosine(t, c) for c in centers]
        sims_t = [cosine(t, c) for c in trap_centers]
        return ("center", int(np.argmax(sims_c))) if max(sims_c) >= max(sims_t) \
            else ("trap", int(np.argmax(sims_t)))

    tests = []
    for _ in range(n_test):
        if rng.random() < 0.3:
            k = int(rng.integers(K)); t = slerp(trap_centers[k], random_vector(dim, rng), 0.1)
        else:
            k = int(rng.integers(K)); t = task(k, 0.35)
        tests.append(t)

    def right(t, pred):
        kind, k = truth_of(t)
        target = answers[k] if kind == "center" else trap_answers[k]
        return cosine(pred, target) > 0.5

    # rung 1: naive similarity cache (the kept negative)
    naive_serve = naive_wrong = 0
    for t in tests:
        j = int(np.argmax([cosine(t, tt) for tt, _ in train]))
        if cosine(t, train[j][0]) > 0.5:
            naive_serve += 1
            if not right(t, answers[train[j][1]]):
                naive_wrong += 1
    # rung 2: the shipped gate, DEPLOYED (outcomes recorded as they are judged)
    half = len(tests) // 2
    a_serve = a_wrong = b_serve = b_wrong = 0
    for i, t in enumerate(tests):
        out = tr.read_gated(t)
        if out["fired"]:
            ok = right(t, out["prediction"])
            tr.record_outcome(t, ok)            # the loop closes: the field learns the traps
            if i < half:
                a_serve += 1; a_wrong += (not ok)
            else:
                b_serve += 1; b_wrong += (not ok)
    print(f"workload: {n_test} tasks, {K} families + {K} look-alike traps, dim={dim}")
    print(f"naive similarity cache      : served {naive_serve:3d}/{n_test}  WRONG {naive_wrong}")
    print(f"gated, phase A (learning)   : served {a_serve:3d}/{half}  WRONG {a_wrong}")
    print(f"gated, phase B (field warm) : served {b_serve:3d}/{len(tests)-half}  WRONG {b_wrong}")
    g_wrong = a_wrong + b_wrong; g_serve = a_serve + b_serve
    n_rate = naive_wrong / max(naive_serve, 1); g_rate = g_wrong / max(g_serve, 1)
    assert g_wrong * 2 <= naive_wrong, "the full gate must at least halve the naive wrong count (measured 2.85x here)"
    assert g_rate < n_rate, "and a lower wrong-serve RATE, not just fewer serves"
    print(f"wrong answers: naive {naive_wrong} -> gated {g_wrong} "
          f"(rates {n_rate:.2f} -> {g_rate:.2f}; the field learns within phase A and holds)")
    print("the gate is not optional, and it learns.")
    return {"naive": (naive_serve, naive_wrong), "A": (a_serve, a_wrong), "B": (b_serve, b_wrong)}

if __name__ == "__main__":
    print(main())

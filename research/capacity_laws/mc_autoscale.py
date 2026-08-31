"""Which knob actually escapes the Memory-Caching wall? Ask leCore, don't guess.

WHY: the paper fixes the architecture and tunes ONE knob (segment length). leCore's
`diagnose_scaling` scales EVERY declared knob in isolation from the current operating
point and ranks them, and `auto_scale` then doubles the most responsive one until the
target is met or a WALL is diagnosed. That is strictly more information than the paper's
segment-length sweep, and it is already shipped -- so this probe uses it rather than
re-deriving a sweep by hand (the Rule-0 lesson from the first pass).

Knobs declared: dim (D), n_seg (N), topk (k). Error = 1 - recall@1 on MQAR.
"""
import numpy as np
import lecore
from holographic.agents_and_reasoning.holographic_ai import bind, unbind

L_PAIRS = 192


def atoms(n, d, rng):
    v = rng.standard_normal((n, d)) / np.sqrt(d)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def eval_fn(dim=512, n_seg=8, topk=2, **_):
    """Deterministic error for the MC-on-HRR recall workload. Averaged over 4 seeds."""
    dim, n_seg, topk = int(round(dim)), max(1, int(round(n_seg))), max(1, int(round(topk)))
    topk = min(topk, n_seg)
    accs = []
    for s in range(4):
        rng = np.random.default_rng(s)
        K, V = atoms(L_PAIRS, dim, rng), atoms(L_PAIRS, dim, rng)
        seg = np.array_split(np.arange(L_PAIRS), n_seg)
        M = np.stack([np.sum([bind(K[t], V[t]) for t in idx], axis=0) for idx in seg])
        pool = np.stack([K[idx].mean(axis=0) for idx in seg])
        hit = 0
        for t in range(L_PAIRS):
            q = K[t]
            g = pool @ q
            keep = np.argsort(-g)[:topk]
            gk = np.exp(g[keep] - g[keep].max()); gk /= gk.sum()
            y = sum(gk[j] * unbind(M[i], q) for j, i in enumerate(keep))
            hit += int(np.argmax(V @ y) == t)
        accs.append(hit / L_PAIRS)
    # cost model: read work per query ~ topk FFT-binds at size dim; storage ~ n_seg*dim
    cost = topk * dim * np.log2(dim) + n_seg * dim
    return {"error": 1.0 - float(np.mean(accs)), "cost": float(cost)}


if __name__ == "__main__":
    m = lecore.UnifiedMind(dim=512, seed=0)
    start = {"dim": 512, "n_seg": 8, "topk": 2}
    print("operating point:", start, "-> error %.3f" % eval_fn(**start)["error"])

    print("\n--- mind.diagnose_scaling (which lever moves the error?) ---")
    d = m.diagnose_scaling(eval_fn, start, factor=2.0)
    print("verdict:", d.get("verdict"))
    for row in d.get("probes", d.get("table", [])):
        print("  ", row)
    print("ranked:", d.get("ranked", d.get("knobs")))

    print("\n--- mind.auto_scale (drive to 5% error) ---")
    a = m.auto_scale(eval_fn, start, target_error=0.05, max_rounds=6, factor=2.0)
    print("verdict:", a.get("verdict"), "| final:", a.get("knobs"),
          "| error:", a.get("error"))
    for step in a.get("trajectory", []):
        print("  ", step)

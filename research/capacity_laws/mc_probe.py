"""
Memory Caching (Behrouz et al. 2602.24281) evaluated on leCore's native HRR memory.

WHY this probe: the paper's Eq.13 proves that for a LINEAR matrix-valued memory the
Residual-Memory variant collapses back into a single pre-summed memory -- i.e. it is
mathematically a no-op. leCore's bundle-of-bindings IS a linear memory (circular
convolution is bilinear, superposition is a sum). So the paper's own algebra predicts
Residual MC buys leCore EXACTLY NOTHING, and only the query-dependent GATE (GRM) can
help. This measures that prediction instead of assuming it.

Task: MQAR-style multi-query associative recall -- write L (key,value) pairs, then
query each key and clean up against the value vocabulary.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, unbind

def atoms(n, d, rng):
    """n near-orthogonal Gaussian hypervectors, unit norm (leCore's Vocabulary default)."""
    v = rng.standard_normal((n, d)) / np.sqrt(d)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def run(d=512, L=192, n_seg=8, topk=2, seeds=range(12)):
    rows = {k: [] for k in ("single", "residual", "grm", "ssc", "oracle")}
    max_resid_gap = 0.0
    for s in seeds:
        rng = np.random.default_rng(s)
        K = atoms(L, d, rng)          # distinct key per slot (hardest case)
        V = atoms(L, d, rng)          # value vocabulary == the L values
        seg = np.array_split(np.arange(L), n_seg)

        # --- write: one bundle per segment. This is leCore's memory, verbatim.
        M = np.stack([np.sum([bind(K[t], V[t]) for t in idx], axis=0) for idx in seg])
        M_all = M.sum(axis=0)                       # the fixed-size RNN memory
        pool = np.stack([K[idx].mean(axis=0) for idx in seg])   # MeanPooling(S_i), Eq.10
        owner = np.concatenate([[i] * len(idx) for i, idx in enumerate(seg)])

        def score(y):
            return V @ (y / (np.linalg.norm(y) + 1e-12))

        hit = dict.fromkeys(rows, 0)
        for t in range(L):
            q = K[t]
            per = np.stack([unbind(M[i], q) for i in range(n_seg)])   # M_i(q), one per cache

            y_single = unbind(M_all, q)
            y_resid = per.sum(axis=0)                                  # Eq.7
            max_resid_gap = max(max_resid_gap,
                                float(np.max(np.abs(y_single - y_resid))))

            g = pool @ q                                              # gamma_i, Eq.10
            g = np.exp(g - g.max()); g /= g.sum()                     # softmax, as the paper does
            y_grm = (g[:, None] * per).sum(axis=0)                    # Eq.9

            keep = np.argsort(-g)[:topk]
            gk = g[keep] / g[keep].sum()
            y_ssc = (gk[:, None] * per[keep]).sum(axis=0)             # Eq.17

            for name, y in (("single", y_single), ("residual", y_resid),
                            ("grm", y_grm), ("ssc", y_ssc),
                            ("oracle", per[owner[t]])):
                hit[name] += int(np.argmax(score(y)) == t)
        for k in rows:
            rows[k].append(hit[k] / L)

    print(f"dim={d}  pairs={L}  segments={n_seg}  seeds={len(list(seeds))}  "
          f"(load L/d = {L/d:.2f}, well past the ~0.05-0.1*d cleanup cliff)")
    print(f"{'variant':<10} {'recall@1':>9} {'+/- sd':>8}   boot95%CI")
    for k in ("single", "residual", "grm", "ssc", "oracle"):
        a = np.array(rows[k])
        bs = np.array([np.mean(rng_.choice(a, a.size)) for rng_ in
                       [np.random.default_rng(1000 + i) for i in range(2000)]])
        print(f"{k:<10} {a.mean():9.3f} {a.std(ddof=1):8.3f}   "
              f"[{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
    print(f"\nKEPT NEGATIVE CHECK -- max |single - residual| over every query = "
          f"{max_resid_gap:.3e}")
    print("  (paper Eq.13: for linear memory, Residual MC collapses to the fixed memory.")
    print("   leCore's bundle is linear, so Residual MC is provably, measurably a NO-OP.)")

if __name__ == "__main__":
    run()

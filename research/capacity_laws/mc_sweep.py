"""Segment-count sweep + honest wall-clock overhead for MC-on-HRR.

WHY: the first probe showed SSC (hard top-k) >> GRM (soft gate) on a VSA bundle,
the OPPOSITE of the paper's ranking on trained deep memories. The hypothesis is that
on a linear memory the win is purely INTERFERENCE REDUCTION -- crosstalk in a bundle
grows with the number of superposed pairs, so reading k of N segments cuts effective
load by N/k. This sweeps N to test that, and times the cost so the trade is honest.
"""
import time
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, unbind


def atoms(n, d, rng):
    v = rng.standard_normal((n, d)) / np.sqrt(d)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def one(d, L, n_seg, topk, seed):
    rng = np.random.default_rng(seed)
    K, V = atoms(L, d, rng), atoms(L, d, rng)
    seg = np.array_split(np.arange(L), n_seg)
    M = np.stack([np.sum([bind(K[t], V[t]) for t in idx], axis=0) for idx in seg])
    M_all = M.sum(axis=0)
    pool = np.stack([K[idx].mean(axis=0) for idx in seg])

    t0 = time.perf_counter()
    base = sum(int(np.argmax(V @ unbind(M_all, K[t])) == t) for t in range(L))
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    ssc = 0
    for t in range(L):
        q = K[t]
        g = pool @ q
        keep = np.argsort(-g)[:topk]                  # router: pre-computable, cheap
        gk = np.exp(g[keep] - g[keep].max()); gk /= gk.sum()
        y = sum(gk[j] * unbind(M[i], q) for j, i in enumerate(keep))
        ssc += int(np.argmax(V @ y) == t)
    t_ssc = time.perf_counter() - t0
    return base / L, ssc / L, t_base / L * 1e6, t_ssc / L * 1e6


def main(d=512, L=192, topk=2, seeds=range(8)):
    print(f"dim={d} pairs={L} top-k={topk} seeds={len(list(seeds))}")
    print(f"{'segments':>8} {'eff.load':>9} {'baseline':>9} {'SSC':>7} "
          f"{'gain':>6} {'us/query base':>14} {'us/query SSC':>13} {'slowdown':>9}")
    for n_seg in (1, 2, 4, 8, 16, 32, 48):
        r = np.array([one(d, L, n_seg, min(topk, n_seg), s) for s in seeds])
        b, sc, tb, ts = r.mean(axis=0)
        eff = L / n_seg * min(topk, n_seg)
        print(f"{n_seg:>8} {eff:>9.0f} {b:>9.3f} {sc:>7.3f} "
              f"{sc/max(b,1e-9):>5.2f}x {tb:>14.1f} {ts:>13.1f} {ts/tb:>8.2f}x")
    print("\nStorage: N segment bundles = N*d floats vs 1*d for the fixed memory "
          f"(at d={d}, N=8 -> {8*d*8/1024:.0f} KB vs {d*8/1024:.0f} KB).")


if __name__ == "__main__":
    main()

"""OPEN MATH #3, ATTACKED: is the <0.1% utilization ceiling real, or a learning artifact?

THE TARGET. Zhou (arXiv:2605.05066) Experiment 5: on associative recall AR(n, V), every
tested architecture -- Mamba across five state sizes, Linear Transformer, GLA -- uses
LESS THAN 0.1% of the information-theoretic recall ceiling n* <= q_bits / [(1-e)log2(V)-1].
GLA is best at ~0.04%. The paper leaves the three-orders-of-magnitude gap unexplained
("models must also use their state for purposes beyond key-value storage").

THE HYPOTHESIS. The gap is an ENCODER artifact, not a state artifact. Shannon's random-
coding theorem says randomly-drawn codes achieve capacity; a VSA bundle of bind(key,val)
pairs with cleanup IS a random code (Clarkson-Ubaru-Yang formalize VSA <-> sketching).
The 52 architectures instead LEARN their encoder by SGD, and SGD on tiny models finds bad
codes. PREDICTION: a zero-training HRR bundle -- memory = sum_i bind(k_i, v_i), readout =
cleanup(unbind(memory, k_query)) -- reaches utilization ORDERS OF MAGNITUDE above 0.04%
at matched state bits. If it does not, the ceiling gap is deeper than encoding and the
hypothesis dies here.

THE BIT-ACCOUNTING, stated so the comparison is fair:
  * state bits = D x (bits per component). float64 = 64D, int8 = 8D, sign-binary = 1D.
    Quantization is applied to the MEMORY VECTOR (the state), which is what q(d) counts.
  * the atom codebooks are seed-derived (derived_atom(seed)): 64 bits, architectural,
    exactly as the 52 architectures' weights are not counted as state. leCore's
    determinism makes the dictionary literally regenerable -- this is lever 3 doing work.
  * V = 1024 (not Zhou's 32): their task caps n <= V by construction, so with V=32 the
    max REACHABLE utilization is 32/bound ~ 0.2% -- the ceiling could never be approached
    regardless of method. V=1024 makes the ceiling approachable and the denominator
    (1-e)log2(V)-1 = 8.0 at e=0.1. Same bound formula, honest change, declared.

THE KNOWN RISK, from the engine's own record: THEORY.md keeps the negative that BINARY
quantization distorts pairwise-similarity geometry enough to corrupt fine readback ("auto
never selects it"). AR recall is coarse argmax cleanup, not fine readback -- whether sign
survives HERE is exactly what gets measured, and if it collapses that negative extends.

WHAT A POSITIVE RESULT MEANS. Utilization is not capped at 0.1% by anything fundamental;
measure-then-allocate with a fixed algebraic code approaches the ceiling BY CONSTRUCTION,
zero training. Combined with the state-demand meter (TT ranks -> how much state the data
demands), the whole corner-choice architecture becomes: price the demand, allocate the
purse, hit the spec first try.
"""
import numpy as np

RFFT, IRFFT = np.fft.rfft, np.fft.irfft


def atoms(n, D, seed):
    """Seed-derived random code -- the whole 'dictionary' costs the seed (64 bits)."""
    v = np.random.default_rng(seed).standard_normal((n, D)) / np.sqrt(D)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def recall_accuracy(D, n, V, seed, precision, n_query=256):
    """Build memory = sum bind(k_i, v_i); quantize the MEMORY to `precision`; recall by
    cleanup(unbind(mem, k)) over the value codebook. Returns accuracy over queries."""
    rng = np.random.default_rng(seed)
    K = atoms(V, D, 1)                      # key alphabet   (architectural, seed=1)
    Vv = atoms(V, D, 2)                     # value alphabet (architectural, seed=2)
    keys = rng.choice(V, n, replace=False)
    vals = rng.integers(0, V, n)
    Kf = RFFT(K[keys], axis=1)
    Vf = RFFT(Vv[vals], axis=1)
    mem = IRFFT((Kf * Vf).sum(0), n=D)      # the state: ONE D-vector

    if precision == "int8":                 # quantize the state, count 8 bits/component
        s = np.max(np.abs(mem)) / 127.0
        mem = np.round(mem / s) * s
    elif precision == "bin":                # sign only: 1 bit/component
        mem = np.sign(mem)

    q = rng.choice(n, min(n_query, n), replace=False)
    Mf = RFFT(mem)
    # unbind all queried keys at once: correlate = conj(key) * memory in Fourier
    est = IRFFT(np.conj(RFFT(K[keys[q]], axis=1)) * Mf[None, :], n=D, axis=1)
    pred = np.argmax(est @ Vv.T, axis=1)
    return float(np.mean(pred == vals[q]))


def n_star(D, V, precision, alpha=0.90, seeds=(0, 1, 2)):
    """Largest n with mean accuracy >= alpha, by doubling then bisection."""
    lo, hi = 1, 2
    while hi < V:
        acc = np.mean([recall_accuracy(D, hi, V, s, precision) for s in seeds])
        if acc < alpha:
            break
        lo, hi = hi, min(2 * hi, V)
    else:
        return V                            # saturated the task itself
    while hi - lo > max(1, lo // 20):
        mid = (lo + hi) // 2
        acc = np.mean([recall_accuracy(D, mid, V, s, precision) for s in seeds])
        lo, hi = (mid, hi) if acc >= alpha else (lo, mid)
    return lo


def main(V=1024, eps=0.10):
    denom = (1 - eps) * np.log2(V) - 1      # = 8.0 at V=1024
    print(f"AR(n, V={V}), alpha={1-eps:.2f}, bound denominator = {denom:.1f}")
    print(f"reference: best trained architecture in Zhou Exp 5 (GLA) ~ 0.04% utilization\n")
    print(f"{'precision':>10} {'D':>6} {'state bits':>11} {'bound n*':>9} "
          f"{'measured n*':>12} {'utilization':>12}")
    results = {}
    for precision, bits in (("f64", 64), ("int8", 8), ("bin", 1)):
        for D in (512, 1024, 2048):
            q_bits = bits * D
            bound = q_bits / denom
            ns = n_star(D, V, precision)
            u = ns / bound
            results[(precision, D)] = (ns, u)
            print(f"{precision:>10} {D:>6} {q_bits:>11} {bound:>9.0f} "
                  f"{ns:>12} {u:>11.1%}")
    print("\nCAPACITY LAW (for the allocator): n* vs D per precision")
    for precision in ("f64", "int8", "bin"):
        ds = np.array([512, 1024, 2048])
        ns = np.array([results[(precision, D)][0] for D in ds])
        slope = float(np.sum(ds * ns) / np.sum(ds * ds))
        print(f"  {precision}: n* ~ {slope:.4f} x D")

    print("\nSINGLE-SHOT ALLOCATOR: demand n=300 at 90% -> allocate D from the law -> verify")
    for precision in ("f64", "int8"):
        ds = np.array([512, 1024, 2048])
        ns = np.array([results[(precision, D)][0] for D in ds])
        slope = float(np.sum(ds * ns) / np.sum(ds * ds))
        D_alloc = int(np.ceil(300 / slope * 1.10 / 64) * 64)   # 10% margin, round to 64
        acc = np.mean([recall_accuracy(D_alloc, 300, V, s, precision) for s in (5, 6, 7)])
        print(f"  {precision}: allocate D={D_alloc} -> measured accuracy {acc:.3f} "
              f"({'HIT' if acc >= 0.90 else 'MISS'} first try, no training)")


if __name__ == "__main__":
    main()

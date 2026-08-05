"""Testing the signature theory where it can actually FAIL.

WHY THE PREVIOUS COMPARISON PROVED NOTHING. Signature and orbit-trap both scored 1.000 on
the order-only control at every T. Two methods at the ceiling are not ranked; they are
saturated. Worse, that task had a discrete alphabet, which is the trap's home turf (its
traps were handed the alphabet) and is irrelevant to the signature. Three experiments
that CAN come out against the theory:

EXPERIMENT A -- CONTINUOUS PATHS, NO ALPHABET. The decisive one, because it is where the
  two methods genuinely differ. A path in R^3 with two smooth bumps in directions u and v;
  the class is WHICH BUMP CAME FIRST. Both classes have the IDENTICAL multiset of
  increments, so:
     level-1 signature (the total increment) is identical by construction -> must be chance
     Levy area (level-2, the antisymmetric part) flips sign with order -> must be perfect
  There is no codebook, so the trap has nothing to be matched to and must fall back on
  random traps, which inherit the sqrt(2 ln T / D) extreme-value floor measured earlier.
  PREDICTION: signature >> trap here. If the trap wins or ties, the "traps need an
  alphabet" story is wrong.

EXPERIMENT B -- LEVEL ABLATION (mechanism, not performance). Level-1 alone MUST equal the
  bag and MUST sit at chance on an order task. If level-1 alone scores above chance, my
  feature construction is leaking order somewhere it should not, and every other number
  here is suspect. This is the experiment designed to catch my own bug.

EXPERIMENT C -- DOES MY OWN READOUT OBEY THE IMPOSSIBILITY TRIANGLE? The signature readout
  is fixed-size and single-pass, so it satisfies E and C, so Theorem 10 says its recall
  must be O(poly(d)) and INDEPENDENT of T -- i.e. as the number of independent facts n
  grows, it must fail, and as T grows at fixed n it should NOT. A readout that appeared to
  recall n proportional to T would mean I have a bug, not a breakthrough. This is the
  self-check that the last two sessions of theory earn.

KEPT NEGATIVE CARRIED IN: signature term count grows exponentially in path dimension m and
truncation level (m + m^2 + m^3 ...), the same wall as VSA factorization. All of this uses
a random projection to small m first, which is the randomised-signature trick (Biagini,
Gonon, Walter) and is what keeps it NumPy-cheap.
"""
import numpy as np


def sig_features(x, level=2):
    """Discrete path signature to `level` of an (T, m) path. Level 1 is the total
    increment (== a bag); level 2 is the iterated sums, whose antisymmetric part is the
    Levy area and is what carries ORDER. No learning, no alphabet, one pass."""
    dx = np.diff(x, axis=0, prepend=x[:1])
    S1 = dx.sum(0)
    if level < 2:
        return S1
    path = np.cumsum(dx, 0) - dx                      # path strictly before each step
    S2 = (path[:, :, None] * dx[:, None, :]).sum(0)
    return np.concatenate([S1, S2.ravel()])


def ridge(Ftr, ytr, Fte, yte, k, lam=1e-3):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    A = np.hstack([(Ftr - mu) / sd, np.ones((len(Ftr), 1))])
    B = np.hstack([(Fte - mu) / sd, np.ones((len(Fte), 1))])
    Y = np.eye(k)[ytr]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    return float(np.mean(np.argmax(B @ W, axis=1) == yte))


def bumpy_path(T, order, rng, noise=0.15):
    """R^3 path: a bump along u then along v, or the reverse. IDENTICAL increment
    multiset either way -- only the ORDER differs, so level-1 cannot see it."""
    t = np.linspace(0, 1, T)
    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    g = lambda c: np.exp(-((t - c) ** 2) / (2 * 0.06 ** 2))
    a, b = (0.3, 0.7) if order == 0 else (0.7, 0.3)
    p = g(a)[:, None] * u + g(b)[:, None] * v
    p = p + noise * np.cumsum(rng.standard_normal((T, 3)), 0) / np.sqrt(T)
    return p


def experiment_A(T=200, n=400, seeds=range(4), D=256):
    print("EXP A -- CONTINUOUS paths, NO alphabet. Which bump came first?")
    print("  (identical increment multiset; only order differs)")
    print(f"{'noise':>7} {'signature L2':>14} {'trap(random)':>14} {'level-1 only':>14}")
    for noise in (0.0, 0.15, 0.4):
        r = {"s": [], "t": [], "b": []}
        for s in seeds:
            rng = np.random.default_rng(s)
            X = [bumpy_path(T, i % 2, rng, noise) for i in range(n)]
            y = np.array([i % 2 for i in range(n)])
            # random traps in the path's own space -- no codebook exists to match to
            traps = rng.standard_normal((16, 3))
            traps /= np.linalg.norm(traps, axis=1, keepdims=True)
            S = np.array([sig_features(p, 2) for p in X])
            L1 = np.array([sig_features(p, 1) for p in X])
            Tr = []
            for p in X:
                sims = p @ traps.T
                Tr.append(np.concatenate([sims.min(0), sims.max(0),
                                          sims.argmax(0) / T, sims.argmin(0) / T]))
            Tr = np.array(Tr)
            cut = n // 2
            for k, F in (("s", S), ("t", Tr), ("b", L1)):
                r[k].append(ridge(F[:cut], y[:cut], F[cut:], y[cut:], 2))
        print(f"{noise:>7.2f} {np.mean(r['s']):>14.3f} {np.mean(r['t']):>14.3f} "
              f"{np.mean(r['b']):>14.3f}")
    print("  chance=0.500.  PREDICT: L2 high, level-1 AT CHANCE (it is the bag).")


def experiment_B(T=200, n=400, seeds=range(4)):
    print("\nEXP B -- level ablation, the bug-catcher")
    print("  level-1 above chance on an order task would mean my features leak order.")
    for noise in (0.0, 0.15):
        l1, l2, anti = [], [], []
        for s in seeds:
            rng = np.random.default_rng(100 + s)
            X = [bumpy_path(T, i % 2, rng, noise) for i in range(n)]
            y = np.array([i % 2 for i in range(n)])
            cut = n // 2
            F1 = np.array([sig_features(p, 1) for p in X])
            F2 = np.array([sig_features(p, 2) for p in X])
            # the Levy area ALONE: the antisymmetric part of level 2
            AA = []
            for p in X:
                s2 = sig_features(p, 2)[3:].reshape(3, 3)
                AA.append(((s2 - s2.T) / 2)[np.triu_indices(3, 1)])
            AA = np.array(AA)
            l1.append(ridge(F1[:cut], y[:cut], F1[cut:], y[cut:], 2))
            l2.append(ridge(F2[:cut], y[:cut], F2[cut:], y[cut:], 2))
            anti.append(ridge(AA[:cut], y[:cut], AA[cut:], y[cut:], 2))
        print(f"  noise={noise:.2f}  level-1={np.mean(l1):.3f}  "
              f"level-1+2={np.mean(l2):.3f}  Levy-area-ONLY (3 numbers)={np.mean(anti):.3f}")


def experiment_C(seeds=range(3)):
    print("\nEXP C -- does the signature readout OBEY the Impossibility Triangle?")
    print("  Fixed-size single-pass features => E and C => recall must be capped in n,")
    print("  and must NOT decay in T at fixed n. Anything else means I have a bug.")
    m, lvl = 6, 2
    width = m + m * m
    print(f"  feature width = {width} numbers (independent of T and n)")
    print(f"\n{'n facts':>8} " + " ".join(f"T={T:<5}" for T in (100, 200, 400)))
    for nfacts in (1, 2, 4, 8, 16):
        row = []
        for T in (100, 200, 400):
            accs = []
            for s in seeds:
                rng = np.random.default_rng(s)
                P = np.random.default_rng(7).standard_normal((32, m)) / np.sqrt(32)
                alpha = rng.standard_normal((16, 32)) / np.sqrt(32)
                N = 300
                X, y = [], []
                for i in range(N):
                    idx = rng.integers(0, 16, T)
                    key_pos = rng.integers(0, T, nfacts)
                    tgt = int(rng.integers(0, 4))
                    idx[key_pos[0]] = tgt                # the queried fact
                    for kp in key_pos[1:]:
                        idx[kp] = int(rng.integers(0, 16))   # distractor facts
                    X.append(sig_features(alpha[idx] @ P, lvl))
                    y.append(tgt)
                X = np.array(X); y = np.array(y)
                accs.append(ridge(X[:150], y[:150], X[150:], y[150:], 4))
            row.append(np.mean(accs))
        print(f"{nfacts:>8} " + " ".join(f"{v:<7.3f}" for v in row))
    print("  chance=0.250. PREDICT: decays with n (bound), flat-ish in T (E^C is fine with T).")


if __name__ == "__main__":
    experiment_A()
    experiment_B()
    experiment_C()

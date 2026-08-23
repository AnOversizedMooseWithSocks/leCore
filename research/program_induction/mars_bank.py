"""MARS's move (parallel reservoir banks) applied to leCore's HolographicESN.

WHY: MARS (arXiv:2604.19343, Apr 2026) reports gradient-free reservoirs beating S5 /
Mamba / LRU on long-sequence classification, and attributes it to PARALLELIZED reservoirs
plus deeper composition -- not to a better single reservoir. leCore ships exactly ONE
reservoir (permute-recurrence, shift=1). This tests whether a bank of K reservoirs with
DIFFERENT hyperparameters, readout fit jointly on the concatenated states, beats the
single shipped one on NARMA10 -- the benchmark the shipped module already reports
(NRMSE ~0.367 +/- 0.001 over 5 seeds).

Baseline discipline: the comparison is against leCore's own tuned single reservoir at
EQUAL TOTAL STATE WIDTH, not against a crippled one. A bank of K reservoirs of width
D each is compared to one reservoir of width K*D. If the bank only wins because it has
more state, that is not a win.
"""
import numpy as np
import lecore


def narma10(n, seed):
    """The standard NARMA10 benchmark: 10th-order nonlinear autoregressive system."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n + 50)
    y = np.zeros(n + 50)
    for t in range(10, n + 50 - 1):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 9:t + 1])
                    + 1.5 * u[t - 9] * u[t] + 0.1)
    return u[50:].reshape(-1, 1), y[50:].reshape(-1, 1)


def nrmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.std(true) + 1e-12))


def states(mind, U, rho, leak, in_scale):
    """Run one reservoir and return its state trajectory (no readout)."""
    esn = mind.reservoir(n_in=1, rho=rho, leak=leak, in_scale=in_scale)
    return esn.run(U) if hasattr(esn, "run") else None


def ridge_fit(X, Y, lam=1e-4, washout=100):
    Xw, Yw = X[washout:], Y[washout:]
    A = np.hstack([Xw, np.ones((len(Xw), 1))])
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Yw)
    return W


def apply_w(X, W, washout=100):
    A = np.hstack([X[washout:], np.ones((len(X) - washout, 1))])
    return A @ W


def main(D=200, seeds=range(5)):
    Utr, Ytr = narma10(3000, 0)
    Ute, Yte = narma10(1500, 999)
    # a bank of K reservoirs, each a DIFFERENT operating point (MARS's diversity)
    bank = [(0.95, 1.0, 0.6), (0.80, 0.5, 0.6), (0.99, 1.0, 0.3), (0.90, 0.2, 0.9)]

    single_hi, bank_res = [], []
    for s in seeds:
        m = lecore.UnifiedMind(dim=D * len(bank), seed=s)     # EQUAL-WIDTH baseline
        Xtr = states(m, Utr, 0.95, 1.0, 0.6)
        Xte = states(m, Ute, 0.95, 1.0, 0.6)
        if Xtr is None:
            print("reservoir has no .run(); aborting"); return
        W = ridge_fit(Xtr, Ytr)
        single_hi.append(nrmse(apply_w(Xte, W), Yte[100:]))

        mb = lecore.UnifiedMind(dim=D, seed=s)                # K reservoirs of width D
        Btr = np.hstack([states(mb, Utr, *p) for p in bank])
        Bte = np.hstack([states(mb, Ute, *p) for p in bank])
        Wb = ridge_fit(Btr, Ytr)
        bank_res.append(nrmse(apply_w(Bte, Wb), Yte[100:]))

    a, b = np.array(single_hi), np.array(bank_res)
    print(f"NARMA10, {len(list(seeds))} seeds, EQUAL total state width "
          f"({D * len(bank)} vs {len(bank)}x{D})")
    print(f"  single wide reservoir : NRMSE {a.mean():.4f} +/- {a.std(ddof=1):.4f}")
    print(f"  bank of {len(bank)} reservoirs : NRMSE {b.mean():.4f} +/- {b.std(ddof=1):.4f}")
    d = a - b
    print(f"  paired diff (single - bank): {d.mean():+.4f} +/- {d.std(ddof=1):.4f}  "
          f"-> {'BANK WINS' if d.mean() > 2 * d.std(ddof=1) / np.sqrt(len(d)) else 'NOT SIGNIFICANT / SINGLE HOLDS'}")


if __name__ == "__main__":
    main()

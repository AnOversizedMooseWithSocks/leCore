"""ORBIT-TRAP READOUT -- Quilez's fractal colouring applied to a recurrent state.

THE DEMOSCENE SEAT'S ACTUAL METHOD. `orbit_trap_render` tracks each ray's CLOSEST APPROACH
to a trap set across the whole iterated orbit, then colours by that scalar. It is a
summary of an entire trajectory held in O(1) memory. Quilez has shaded fractals this way
for two decades; nothing about it is graphical.

WHY IT LANDS ON THE RNN PROBLEM. leCore's `ReservoirSequenceClassifier` reads the FINAL
state -- and its own kept negative on record is that this loses badly (Language ID on real
UDHR: bag-of-trigrams 0.97-0.99 vs reservoir final-state 0.33-0.36). A leaky recurrence
washes early information out of the final state. That is the SAME failure Memory Caching
exists to fix, and MC fixes it with N cached memory checkpoints -- storage that grows with
sequence length.

An orbit trap fixes it for free: keep `min_t d(s_t, trap_k)` over the trajectory for a few
fixed trap vectors. Cost is O(#traps), INDEPENDENT of sequence length. The trajectory
information the final state threw away is recovered without caching a single state.

THE TASK, chosen so the failure mode is unambiguous. A needle-in-a-haystack: a marker
symbol appears EARLY (position ~5) in a long sequence and determines the class; everything
after is identical noise drawn from a shared alphabet. The final state must forget it as T
grows. A bag-of-symbols baseline is included because on THIS task a bag can see the marker
trivially -- so if the trap only matches the bag, the trap is doing nothing interesting;
the trap has to beat the FINAL STATE, which is the readout actually shipped.

BASELINES, all three required:
  * final-state readout   -- what the engine ships today. The thing to beat.
  * orbit-trap readout    -- min/max/mean cosine to K fixed traps over the trajectory.
  * bag-of-symbols        -- the honest sanity ceiling for a marker task.

KEPT NEGATIVE TO WATCH FOR: a trap is a LOSSY summary. If two different trajectories have
the same closest approach to every trap, they are indistinguishable -- exactly the
collision risk the linear-bundle checksum negative describes. More traps buy separation at
linear cost, and where that curve saturates is the number that matters.
"""
import numpy as np


def permute_reservoir(seq_vecs, leak=1.0):
    """leCore's native recurrence: permute is the fixed recurrent operator (norm-preserving,
    hence the echo-state property), bind folds input in, tanh is the one nonlinearity."""
    d = seq_vecs.shape[1]
    s = np.zeros(d)
    traj = np.empty((len(seq_vecs), d))
    for t, x in enumerate(seq_vecs):
        s = (1 - leak) * s + leak * np.tanh(np.roll(s, 1) + x)
        n = np.linalg.norm(s)
        s = s / n if n > 0 else s
        traj[t] = s
    return traj


def make_task(n, T, dim, rng, n_class=4, marker_at=5):
    """Class is set by a marker symbol early in the sequence; the rest is shared noise."""
    alphabet = rng.standard_normal((12, dim)) / np.sqrt(dim)
    alphabet /= np.linalg.norm(alphabet, axis=1, keepdims=True)
    markers = alphabet[:n_class]
    filler = alphabet[n_class:]
    X, y = [], []
    for i in range(n):
        c = int(rng.integers(n_class))
        idx = rng.integers(0, len(filler), T)
        seq = filler[idx].copy()
        seq[marker_at] = markers[c]          # the ONLY class information, and it is early
        X.append(seq)
        y.append(c)
    return np.array(X), np.array(y), alphabet


def ridge_classify(F_tr, y_tr, F_te, y_te, n_class, lam=1e-3):
    """One closed-form ridge readout -- no gradients, the engine's own learning rule."""
    Y = np.eye(n_class)[y_tr]
    A = np.hstack([F_tr, np.ones((len(F_tr), 1))])
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    P = np.hstack([F_te, np.ones((len(F_te), 1))]) @ W
    return float(np.mean(np.argmax(P, axis=1) == y_te))


def features(X, traps, alphabet):
    """Three readouts from the SAME trajectory, so the comparison is honest."""
    fin, trap, bag = [], [], []
    for seq in X:
        traj = permute_reservoir(seq)
        fin.append(traj[-1])                                  # what the engine ships
        sims = traj @ traps.T                                 # (T, K) orbit vs trap set
        trap.append(np.concatenate([sims.min(0), sims.max(0), sims.mean(0)]))
        bag.append((seq @ alphabet.T).max(0))                 # sanity ceiling
    return np.array(fin), np.array(trap), np.array(bag)


def main(dim=256, n=400, n_class=4, K=16, seeds=range(3)):
    print("Needle-in-a-haystack: class-defining marker at position 5, rest shared noise.")
    print("Orbit-trap cost is O(K) and INDEPENDENT of T; Memory Caching's is O(T/segment).\n")
    print(f"{'T':>6} {'final-state':>12} {'orbit-trap':>12} {'bag':>8}   {'trap/final':>10}")
    for T in (20, 50, 100, 200, 400):
        accs = {"fin": [], "trap": [], "bag": []}
        for s in seeds:
            rng = np.random.default_rng(s)
            X, y, alpha = make_task(n, T, dim, rng, n_class=n_class)
            traps = rng.standard_normal((K, dim)) / np.sqrt(dim)
            traps /= np.linalg.norm(traps, axis=1, keepdims=True)
            F, Tr, B = features(X, traps, alpha)
            cut = n // 2
            accs["fin"].append(ridge_classify(F[:cut], y[:cut], F[cut:], y[cut:], n_class))
            accs["trap"].append(ridge_classify(Tr[:cut], y[:cut], Tr[cut:], y[cut:], n_class))
            accs["bag"].append(ridge_classify(B[:cut], y[:cut], B[cut:], y[cut:], n_class))
        f, t, b = (np.mean(accs[k]) for k in ("fin", "trap", "bag"))
        print(f"{T:>6} {f:>12.3f} {t:>12.3f} {b:>8.3f}   {t/max(f,1e-9):>9.2f}x")
    print(f"\n(chance = {1/n_class:.3f}; K = {K} traps, so the trap feature is 3K = {3*K} numbers")
    print(" regardless of T, while the final state is dim = {} numbers and forgets.)".format(dim))

    print("\nTRAP-COUNT SWEEP at T=200 -- where does adding traps stop paying?")
    print(f"{'K':>5} {'orbit-trap acc':>16} {'feature width':>14}")
    for K2 in (1, 2, 4, 8, 16, 32, 64):
        a = []
        for s in seeds:
            rng = np.random.default_rng(s)
            X, y, alpha = make_task(n, 200, dim, rng, n_class=n_class)
            tr = rng.standard_normal((K2, dim)) / np.sqrt(dim)
            tr /= np.linalg.norm(tr, axis=1, keepdims=True)
            _, Tr, _ = features(X, tr, alpha)
            cut = n // 2
            a.append(ridge_classify(Tr[:cut], y[:cut], Tr[cut:], y[cut:], n_class))
        print(f"{K2:>5} {np.mean(a):>16.3f} {3*K2:>14}")


if __name__ == "__main__":
    main()

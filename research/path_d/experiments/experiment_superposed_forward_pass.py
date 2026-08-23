"""
Path D, first data point: how wide a 'GPU thing' stays intact in the holographic space.

The claim under test: you can do neural-network-style computation (a layer's worth of
multiply-accumulate) *in superposition* -- one structure, many results at once -- instead
of with silicon parallelism. The honest question is not whether it works at all (it does)
but WHERE THE CAPACITY WALL IS: a D-dimensional vector can only hold so much superposed
content before crosstalk noise overwhelms the signal. This script measures that wall on
holostuff's OWN primitives, so it is the engine's real capacity, not a reimplementation.

Two measurements, building on each other:

  PART 1 -- the bedrock capacity curve. Store N (key,value) pairs in ONE bundled vector,
            recover each value, clean it up. Plot recall accuracy vs N for several D.
            This is the raw "how many things can you superpose and still read back."

  PART 2 -- a real forward pass. Take a one-layer linear classifier (nearest-prototype:
            logit_c = <weight_c, x>) and evaluate ITS ENTIRE FORWARD PASS out of a single
            superposed weight-memory: bundle all C weight rows (each bound to a class role
            key), then recover each row by unbinding and dot it with the input. Sweep the
            layer WIDTH C and the dimension D. Measure three things honestly:
              (a) logit fidelity  -- does the superposed readout reproduce the exact logits?
                                     (the pure capacity signal, independent of task difficulty)
              (b) accuracy        -- exact classifier vs superposed-readout classifier
                                     (the practical consequence)
              (c) wall-clock time -- the superposed forward pass vs a plain matmul on CPU
                                     (the honest "the FLOPs don't vanish" check)

Unitary keys are used throughout so unbinding is EXACT -- that way the ONLY source of error
is superposition crosstalk, which is exactly the wall we want to isolate.
"""
import sys, os, time
import numpy as np

# Build on the engine's real, frozen kernel -- not a private reimplementation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from holographic.misc.holographic_core import unitary_vector, random_vector, bind, unbind, involution, bundle, cosine
# vectorised binds for speed (bit-identical to the single-pair bind, per their docstrings)
from holographic.agents_and_reasoning.holographic_ai import bind_batch, bind_fixed

SEED = 7
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# small vectorised helpers (the loop versions are in the kernel; these just do
# the same circular-convolution maths over a whole stack at once for speed)
# ---------------------------------------------------------------------------
def involution_stack(A):
    """Row-wise involution: a[0] stays, the rest flips. Matches holographic_ai.involution."""
    return np.concatenate([A[:, :1], A[:, :0:-1]], axis=1)

def unbind_fixed(composite, KEYS):
    """Recover every filler from one composite at once: unbind(composite, KEYS[i]) for all i.
    unbind(M, k) == bind(M, involution(k)); bind_fixed convolves the fixed M against a stack."""
    return bind_fixed(composite, involution_stack(KEYS))

def mint_unitary(n, dim):
    """A stack of n fresh unitary atoms (unit-magnitude spectrum -> EXACT unbinding)."""
    return np.stack([unitary_vector(dim, rng) for _ in range(n)])

def mint_random(n, dim):
    """A stack of n fresh Gaussian atoms (the ordinary 'concept' vectors)."""
    return np.stack([random_vector(dim, rng) for _ in range(n)])


# ---------------------------------------------------------------------------
# PART 1 -- bedrock key/value superposition capacity
# ---------------------------------------------------------------------------
def kv_recall_accuracy(dim, n_pairs, trials=15):
    """Store n_pairs (key,value) in one bundle, recover & clean up each, return mean recall.

    M = bundle( bind(k_i, v_i) for i ).  recovered_i = unbind(M, k_i) ~= v_i + crosstalk.
    Cleanup = nearest of the n stored values. 'Correct' = the right value wins the argmax.
    Keys unitary (exact unbind) so the ONLY error is the crosstalk from piling n things into
    one fixed-width vector -- the capacity wall in its purest form.
    """
    accs = []
    for _ in range(trials):
        keys = mint_unitary(n_pairs, dim)            # (N, D) exact-unbind keys
        vals = mint_random(n_pairs, dim)             # (N, D) the things we store
        bound = bind_batch(keys, vals)               # (N, D) key_i (x) val_i
        M = bundle(bound)                            # one vector holds all N pairs
        recovered = unbind_fixed(M, keys)            # (N, D) noisy estimates of each val
        # cleanup: nearest stored value by cosine (values are unit norm -> dot == cosine*||rec||)
        rn = np.linalg.norm(recovered, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        sims = (recovered / rn) @ vals.T             # (N, N) each row: similarity to every value
        pred = sims.argmax(axis=1)
        accs.append(np.mean(pred == np.arange(n_pairs)))
    return float(np.mean(accs))


def run_part1():
    print("=" * 72)
    print("PART 1  -- key/value superposition capacity (the bedrock cliff)")
    print("=" * 72)
    dims = [256, 512, 1024]
    Ns = [2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    curves = {}
    for D in dims:
        accs = []
        for N in Ns:
            a = kv_recall_accuracy(D, N)
            accs.append(a)
        curves[D] = accs
        # report the cliff: largest N still above 90% and above 50%
        n90 = max([n for n, a in zip(Ns, accs) if a >= 0.90], default=0)
        n50 = max([n for n, a in zip(Ns, accs) if a >= 0.50], default=0)
        print(f"  D={D:5d}:  90%-recall holds to N={n90:5d}   "
              f"50%-recall holds to N={n50:5d}   (~{n50/D:.2f} x D)")
    return Ns, curves


# ---------------------------------------------------------------------------
# PART 2 -- a linear classifier's forward pass, evaluated in superposition
# ---------------------------------------------------------------------------
def make_task(n_classes, n_feat=20, per_class=80, sep=6.0, std=1.0, seed=0):
    """Cleanly-separable C-way blobs so the EXACT classifier is ~100% -- that way any drop
    in the SUPERPOSED classifier is purely the capacity wall, not the task getting harder."""
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    X, y = make_blobs(n_samples=per_class * n_classes, centers=n_classes,
                      n_features=n_feat, cluster_std=std, center_box=(-sep, sep),
                      random_state=seed)
    return train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)


def superposed_layer_eval(D, C, n_feat=20, seeds=(0, 1, 2)):
    """Build a one-layer linear classifier in a D-dim hypervector space, then evaluate its
    whole forward pass out of ONE superposed weight-memory. Return (logit_fidelity,
    acc_exact, acc_super) averaged over a few task seeds.

    The layer:   logit_c = <w_c, x>     (w_c = unit class prototype, the 'weights')
    Exact:       compute all C logits directly (a C x D matmul).
    Superposed:  W_mem = bundle( bind(role_c, w_c) for c );  one vector holds all C rows.
                 w_c_hat = unbind(W_mem, role_c)  (noisy);   logit_c_hat = <w_c_hat, x>.
                 -> the ENTIRE layer is read out of a single superposed object.
    """
    fids, ax, asu = [], [], []
    for s in seeds:
        Xtr, Xte, ytr, yte = make_task(C, n_feat=n_feat, seed=s)
        # fixed random projection into the D-dim holographic space, then unit-normalise
        R = rng.standard_normal((D, n_feat)) / np.sqrt(n_feat)
        def enc(F):
            H = F @ R.T
            return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        Htr, Hte = enc(Xtr), enc(Xte)
        # class prototypes = the layer weights (unit norm so logits are comparable)
        W = np.stack([Htr[ytr == c].mean(0) for c in range(C)])
        W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)         # (C, D)

        L_exact = Hte @ W.T                                               # (Nte, C) true logits

        roles = mint_unitary(C, D)                                        # (C, D) class role keys
        W_mem = bundle(bind_batch(roles, W))                              # one vector = all C rows
        W_hat = unbind_fixed(W_mem, roles)                               # (C, D) recovered rows
        L_super = Hte @ W_hat.T                                           # (Nte, C) superposed logits

        # (a) logit fidelity: per-test-point correlation between exact and superposed logits
        Le = L_exact - L_exact.mean(1, keepdims=True)
        Ls = L_super - L_super.mean(1, keepdims=True)
        num = (Le * Ls).sum(1)
        den = np.sqrt((Le ** 2).sum(1) * (Ls ** 2).sum(1)) + 1e-12
        fids.append(float(np.mean(num / den)))
        # (b) accuracy: exact vs superposed-readout
        ax.append(float(np.mean(L_exact.argmax(1) == yte)))
        asu.append(float(np.mean(L_super.argmax(1) == yte)))
    return float(np.mean(fids)), float(np.mean(ax)), float(np.mean(asu))


def run_part2():
    print()
    print("=" * 72)
    print("PART 2  -- a linear classifier's forward pass, run in superposition")
    print("=" * 72)
    dims = [256, 512, 1024]
    Cs = [4, 8, 16, 32, 48, 64, 96, 128, 160, 200, 250]
    fid_curves, accx_curves, accs_curves = {}, {}, {}
    for D in dims:
        fids, axs, asus = [], [], []
        for C in Cs:
            f, ae, asu_ = superposed_layer_eval(D, C)
            fids.append(f); axs.append(ae); asus.append(asu_)
        fid_curves[D], accx_curves[D], accs_curves[D] = fids, axs, asus
        # cliff: widest layer whose superposed readout still matches exact accuracy within 2 pts,
        # and where logit fidelity stays above 0.9
        c_acc = max([c for c, ae, asu_ in zip(Cs, axs, asus) if asu_ >= ae - 0.02], default=0)
        c_fid = max([c for c, f in zip(Cs, fids) if f >= 0.90], default=0)
        print(f"  D={D:5d}:  superposed acc tracks exact to width C={c_acc:4d}   "
              f"logit-fidelity>=0.90 to C={c_fid:4d}   (~{c_fid/D:.2f} x D)")
    return Cs, fid_curves, accx_curves, accs_curves


def run_timing():
    """The honest 'FLOPs don't vanish' check: the superposed forward pass vs a plain matmul."""
    print()
    print("=" * 72)
    print("TIMING -- superposed forward pass vs an exact matmul on CPU (D=512, C=128)")
    print("=" * 72)
    D, C, n_feat, Nte, reps = 512, 128, 20, 400, 50
    R = rng.standard_normal((D, n_feat)) / np.sqrt(n_feat)
    Hte = rng.standard_normal((Nte, D)); Hte /= np.linalg.norm(Hte, axis=1, keepdims=True)
    W = rng.standard_normal((C, D)); W /= np.linalg.norm(W, axis=1, keepdims=True)
    roles = mint_unitary(C, D)

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = Hte @ W.T                                    # the ordinary layer: one matmul
    t_exact = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        W_mem = bundle(bind_batch(roles, W))             # build the superposed weight-memory
        W_hat = unbind_fixed(W_mem, roles)               # recover all C rows (C FFT-unbinds)
        _ = Hte @ W_hat.T                                # then the same matmul
    t_super = (time.perf_counter() - t0) / reps

    print(f"  exact matmul      : {t_exact*1e6:8.1f} us / forward pass")
    print(f"  superposed readout: {t_super*1e6:8.1f} us / forward pass")
    print(f"  superposition is {t_super/t_exact:5.1f}x SLOWER on CPU "
          f"(the parallelism win is an energy/hardware story, not a CPU-speed one)")
    return t_exact, t_super


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
def make_plot(Ns, kv_curves, Cs, fid_curves, accx_curves, accs_curves, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {256: "#c0392b", 512: "#2c7fb8", 1024: "#239b56"}

    ax = axes[0]
    for D, accs in kv_curves.items():
        ax.plot(Ns, accs, "o-", color=colors[D], label=f"D={D}", markersize=4)
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("N items superposed in one vector")
    ax.set_ylabel("recall accuracy (after cleanup)")
    ax.set_title("(a) Bedrock: key/value capacity cliff")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-0.03, 1.03)

    ax = axes[1]
    for D, fids in fid_curves.items():
        ax.plot(Cs, fids, "o-", color=colors[D], label=f"D={D}", markersize=4)
    ax.axhline(0.9, color="0.6", ls=":", lw=1)
    ax.set_xlabel("layer width C (outputs read from one superposed vector)")
    ax.set_ylabel("logit fidelity vs exact (mean corr)")
    ax.set_title("(b) Forward pass: does the computation survive?")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-0.03, 1.03)

    ax = axes[2]
    for D in accx_curves:
        ax.plot(Cs, accx_curves[D], "--", color=colors[D], lw=1.2, alpha=0.7)
        ax.plot(Cs, accs_curves[D], "o-", color=colors[D], label=f"D={D}", markersize=4)
    ax.set_xlabel("layer width C")
    ax.set_ylabel("accuracy  (dashed = exact, solid = superposed)")
    ax.set_title("(c) Practical consequence: classifier accuracy")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-0.03, 1.03)

    fig.suptitle("Computing a neural-net layer in the holographic space: where the capacity wall is "
                 "(holostuff primitives, unitary keys -> crosstalk is the only error)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    print(f"\nplot -> {outpath}")


if __name__ == "__main__":
    Ns, kv_curves = run_part1()
    Cs, fid_curves, accx_curves, accs_curves = run_part2()
    run_timing()
    out = os.path.join(os.path.dirname(__file__), "capacity_cliff.png")
    make_plot(Ns, kv_curves, Cs, fid_curves, accx_curves, accs_curves, out)
    print("\ndone.")

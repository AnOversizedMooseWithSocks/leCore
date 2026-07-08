"""
A1 -- deep (multi-layer) forward pass in superposition. The win handled per-layer WIDTH by federating
the weight-memory; depth is the new question: a deep net feeds each layer's noisy output into the next,
so crosstalk can COMPOUND with depth. The hypothesis: a cleanup step between layers (snap the hidden
activation back onto the manifold of valid activations -- the dense-Hopfield move) stops the compounding.

Setup: a real trained MLP (separable blobs, so the exact net is ~100% and any drop is the substrate).
Each layer's weights are read from a FEDERATED weight-memory (K shards -> width handled). We measure
test accuracy vs depth, cleanup OFF vs ON, against the exact float MLP. Kept negative: without cleanup,
accuracy decays with depth.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from holographic.misc.holographic_core import unitary_vector, bundle
from holographic.agents_and_reasoning.holographic_ai import bind_batch, bind_fixed
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
rng = np.random.default_rng(7)
D = 1024
def inv_stack(A): return np.concatenate([A[:, :1], A[:, :0:-1]], axis=1)
def recover_federated(KEYS, VALS, Ksh):
    n = len(KEYS); Vhat = np.zeros_like(VALS)
    for k in range(Ksh):
        idx = np.arange(n)[np.arange(n) % Ksh == k]
        if len(idx) == 0: continue
        M = bundle(bind_batch(KEYS[idx], VALS[idx])); Vhat[idx] = bind_fixed(M, inv_stack(KEYS[idx]))
    return Vhat
def relu(z): return np.maximum(z, 0.0)

def softclean(A, book, beta=10.0):
    """Snap each row of A onto the manifold of valid activations (soft dense-Hopfield over a codebook)."""
    bn = book / (np.linalg.norm(book, axis=1, keepdims=True) + 1e-9)
    an = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    S = an @ bn.T
    W = np.exp(beta * (S - S.max(1, keepdims=True))); W /= W.sum(1, keepdims=True)
    out = W @ book
    sc = (np.linalg.norm(A, axis=1, keepdims=True) / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9))
    return out * sc                                  # keep the activation scale, denoise the direction

def superposed_forward(MLP, X, K, cleanup, books):
    """Run the trained MLP forward, but read every layer's weights from a federated weight-memory.
    Weights/inputs are JL-embedded into D dims (P with N(0,1/D) entries preserves inner products)."""
    Ws = MLP.coefs_; bs = MLP.intercepts_; L = len(Ws)
    a = X
    for l in range(L):
        n_in, n_out = Ws[l].shape
        P = rng.standard_normal((D, n_in)) / np.sqrt(D)        # JL embed: <Pw,Pa> ~ <w,a>
        Xe = a @ P.T                                           # (N, D) embedded inputs
        We = Ws[l].T @ P.T                                     # (n_out, D) embedded weight rows
        roles = np.stack([unitary_vector(D, rng) for _ in range(n_out)])
        Whe = recover_federated(roles, We, K)                  # noisy weight readout
        pre = Xe @ Whe.T + bs[l]                                # (N, n_out) pre-activations
        if l < L - 1:
            a = relu(pre)
            if cleanup and books[l] is not None:
                a = softclean(a, books[l])
        else:
            a = pre
    return a.argmax(1)

def run():
    C, F, Hwid, K = 12, 20, 64, 4
    print(f"A1 deep forward pass (C={C} classes, width {Hwid}, K={K} shards/layer, D={D})")
    print("=" * 74)
    depths = {1: (Hwid,), 2: (Hwid, Hwid), 3: (Hwid, Hwid, Hwid)}
    for d, hls in depths.items():
        accE, accNo, accCl = [], [], []
        for seed in (0, 1, 2):
            X, y = make_blobs(n_samples=80 * C, centers=C, n_features=F,
                              cluster_std=1.0, center_box=(-6, 6), random_state=seed)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
            mlp = MLPClassifier(hidden_layer_sizes=hls, activation="relu", max_iter=800,
                                random_state=seed).fit(Xtr, ytr)
            accE.append(mlp.score(Xte, yte))
            # cleanup codebooks: training hidden activations at each hidden layer
            books = []
            acts = Xtr
            for l in range(len(mlp.coefs_) - 1):
                acts = relu(acts @ mlp.coefs_[l] + mlp.intercepts_[l])
                books.append(acts.copy())
            accNo.append(np.mean(superposed_forward(mlp, Xte, K, False, books) == yte))
            accCl.append(np.mean(superposed_forward(mlp, Xte, K, True, books) == yte))
        print(f"  depth {d} ({len(depths[d])} hidden): exact={np.mean(accE):.3f}   "
              f"superposed no-cleanup={np.mean(accNo):.3f}   with-cleanup={np.mean(accCl):.3f}")
    print("=" * 74)
run()

"""
A1, re-opened with A2's lesson: was the depth collapse a genuine depth wall, or just the per-layer
ARITHMETIC crosstalk of the lossy weight-superposition readout, compounding layer over layer?

Test: rebuild each layer's matmul as EXACT RNS-phasor multiply-accumulate (fixed-point), the same
formula that flipped A2 -- weights/activations quantized to integers, matmul done channel-wise with
phasor-binding accumulation (exact), CRT-recomposed, dequantized, ReLU, next layer. If the depth decay
was arithmetic noise, it should vanish: an exact forward pass computes what the float net computes at
ANY depth. Honest residual to watch: fixed-point QUANTIZATION error, which could itself compound --
that is the real, separable depth question (a bit-depth issue, not a crosstalk wall).
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
rng = np.random.default_rng(0)

PRIMES = [101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,
          199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313]
def pick_moduli(need):
    mods, P = [], 1
    for p in PRIMES:
        mods.append(p); P *= p
        if P > need: return mods, P
    return mods, P

def rns_matmul_batch(Wq, Aq, moduli):
    """Exact integer matmul (Aq @ Wq^T): Aq (B,N) ints, Wq (M,N) ints -> (B,M) ints. Channel-wise,
    accumulation via exact FHRR phasor binding (prod of unit phasors = phase = sum mod m)."""
    B = Aq.shape[0]; M = Wq.shape[0]; res = []
    for m in moduli:
        Wm = Wq % m; Am = Aq % m
        prod = (Am[:, None, :] * Wm[None, :, :]) % m            # (B,M,N) residue of each product
        phase = np.prod(np.exp(2j * np.pi * prod / m), axis=2)  # binding => sum of phases
        res.append((np.round(np.angle(phase) / (2 * np.pi) * m).astype(np.int64) % m))
    P = 1
    for m in moduli: P *= m
    y = np.zeros((B, M), dtype=object)
    for r, m in zip(res, moduli):
        Mi = P // m; y = y + r.astype(object) * (Mi * pow(Mi, -1, m))
    y = y % P
    return np.where(y > P // 2, y - P, y).astype(np.int64)

def quant(x, bits=8):
    s = (2 ** (bits - 1) - 1) / (np.abs(x).max() + 1e-12)
    return np.round(x * s).astype(np.int64), s

def rns_forward(MLP, X, bits=8):
    """The trained MLP's forward pass, every layer's matmul done EXACTLY via RNS-phasor (fixed-point)."""
    Ws, bs = MLP.coefs_, MLP.intercepts_; L = len(Ws)
    a = X
    for l in range(L):
        Wq, sw = quant(Ws[l].T, bits)                          # (n_out, n_in) integer weights
        Aq, sa = quant(a, bits)                                # (B, n_in) integer activations
        need = 2 * int(np.abs(Aq @ Wq.T).max()) + 3            # dynamic range the result needs
        mods, _ = pick_moduli(need)
        pre = rns_matmul_batch(Wq, Aq, mods) / (sw * sa) + bs[l]
        a = np.maximum(pre, 0.0) if l < L - 1 else pre
    return a.argmax(1)

def run():
    C, F, Hwid = 12, 20, 64
    depths = {1: (Hwid,), 2: (Hwid, Hwid), 3: (Hwid, Hwid, Hwid)}
    old_lossy = {1: 0.976, 2: 0.418, 3: 0.149}                 # the original A1 (lossy readout)
    print(f"A1 re-opened with exact RNS-phasor per-layer matmul  (C={C}, width {Hwid}, 8-bit fixed-point)")
    print("=" * 86)
    for d, hls in depths.items():
        ex, rns_acc = [], []
        for seed in (0, 1, 2):
            X, y = make_blobs(n_samples=80 * C, centers=C, n_features=F,
                              cluster_std=1.0, center_box=(-6, 6), random_state=seed)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
            mlp = MLPClassifier(hidden_layer_sizes=hls, activation="relu", max_iter=800,
                                random_state=seed).fit(Xtr, ytr)
            ex.append(mlp.score(Xte, yte))
            rns_acc.append(np.mean(rns_forward(mlp, Xte) == yte))
        print(f"  depth {d}: exact float MLP={np.mean(ex):.3f}   "
              f"RNS-substrate forward pass={np.mean(rns_acc):.3f}   "
              f"(old lossy readout was {old_lossy[d]:.3f})")
    print("=" * 86)
    print("If RNS-substrate tracks exact at every depth, the depth collapse was per-layer ARITHMETIC")
    print("crosstalk -- an encoding artifact, the A2 disease -- not a genuine depth wall.")
run()

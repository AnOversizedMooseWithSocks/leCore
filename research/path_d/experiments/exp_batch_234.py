"""
Bucket A, items A2/A3/A4 -- three more single-vector walls, re-opened under the distributed premise.
All share one shape: a superposed readout capped by per-vector crosstalk, federated across K shards.

A2  general matmul W@x in superposition: fidelity vs #rows, single vs federated.
A3  hypothesis evaluation: pick/rank H candidates scored against a query; selection wall vs federation.
A4  sequence memory: recover a length-T symbol sequence; recoverable length vs federation.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from holographic.misc.holographic_core import unitary_vector, random_vector, bundle
from holographic.agents_and_reasoning.holographic_ai import bind_batch, bind_fixed
rng = np.random.default_rng(7)
D = 1024
def inv_stack(A): return np.concatenate([A[:, :1], A[:, :0:-1]], axis=1)
def recover_federated(KEYS, VALS, Ksh):
    """Bundle (key_i (x) val_i) into Ksh shards by (i mod Ksh); recover every row. Kept negative
    lives per shard: each shard's crosstalk is set by its ~n/Ksh rows."""
    n = len(KEYS); Vhat = np.zeros_like(VALS)
    for k in range(Ksh):
        idx = np.arange(n)[np.arange(n) % Ksh == k]
        if len(idx) == 0: continue
        M = bundle(bind_batch(KEYS[idx], VALS[idx]))
        Vhat[idx] = bind_fixed(M, inv_stack(KEYS[idx]))
    return Vhat

def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
# ---------- A2: superposed matmul fidelity (correlation of the product vs exact) ----------
def a2(Ms=(8,16,32,48,64,96,128,160,200,256), Ks=(1,4), trials=5):
    out = {K: [] for K in Ks}
    for K in Ks:
        for M in Ms:
            fs = []
            for t in range(trials):
                W = np.stack([random_vector(D, rng) for _ in range(M)]); W /= np.linalg.norm(W,axis=1,keepdims=True)
                x = random_vector(D, rng); x /= np.linalg.norm(x)
                exact = W @ x
                roles = np.stack([unitary_vector(D, rng) for _ in range(M)])
                What = recover_federated(roles, W, K)
                fs.append(corr(What @ x, exact))                             # fidelity of the product
            out[K].append(float(np.mean(fs)))
    return list(Ms), out

# ---------- A3: hypothesis selection + ranking ----------
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
def a3(Hs=(8,16,32,48,64,96,128,160,200,256), Ks=(1,8), trials=8):
    # real task: one planted hypothesis matches the query; pick it out of H bundled candidates.
    sel = {K: [] for K in Ks}; rnk = {K: [] for K in Ks}
    for K in Ks:
        for H in Hs:
            ss, rr = [], []
            for t in range(trials):
                Hyp = np.stack([random_vector(D, rng) for _ in range(H)]); Hyp /= np.linalg.norm(Hyp,axis=1,keepdims=True)
                j = int(rng.integers(0, H))                                  # the true best
                q = Hyp[j].copy()                                            # query = the planted match
                true = Hyp @ q
                roles = np.stack([unitary_vector(D, rng) for _ in range(H)])
                Hhat = recover_federated(roles, Hyp, K)
                sup = Hhat @ q
                ss.append(float(sup.argmax() == j))                          # picked the planted best?
                rr.append(spearman(sup, true))
            sel[K].append(float(np.mean(ss))); rnk[K].append(float(np.mean(rr)))
    return list(Hs), sel, rnk

# ---------- A4: sequence memory length ----------
def a4(Ts=(8,16,32,48,64,96,128,160,200,256), Ks=(1,8), V=64, trials=5):
    acc = {K: [] for K in Ks}
    for K in Ks:
        for T in Ts:
            aa = []
            for t in range(trials):
                cb = np.stack([random_vector(D, rng) for _ in range(V)])     # symbol codebook
                seq = rng.integers(0, V, size=T)                            # the sequence
                pos = np.stack([unitary_vector(D, rng) for _ in range(T)])   # position keys
                vals = cb[seq]
                Vhat = recover_federated(pos, vals, K)                       # recover each position
                pred = (Vhat @ cb.T).argmax(1)                              # cleanup to codebook
                aa.append(float(np.mean(pred == seq)))
            acc[K].append(float(np.mean(aa)))
    return list(Ts), acc

print("A2 superposed matmul -- fidelity of W@x vs #rows (D=%d)" % D)
Ms, a2o = a2()
for K in a2o:
    cliff = max([m for m, f in zip(Ms, a2o[K]) if f >= 0.90], default=0)
    print(f"   K={K}: cos>=0.95 to M={cliff:4d} rows")
print("A3 hypothesis eval -- selection accuracy & rank fidelity vs #hypotheses")
Hs, sel, rnk = a3()
for K in sel:
    c = max([h for h, s in zip(Hs, sel[K]) if s >= 0.95], default=0)
    print(f"   K={K}: pick-the-best>=0.95 to H={c:4d}   (rank-corr at H=256: {rnk[K][-1]:.2f})")
print("A4 sequence memory -- fraction recalled vs length T")
Ts, acc = a4()
for K in acc:
    c = max([t for t, a in zip(Ts, acc[K]) if a >= 0.90], default=0)
    print(f"   K={K}: 90%-recall to T={c:4d} symbols")
import json
json.dump({"A2": [Ms, a2o], "A3": [Hs, sel, rnk], "A4": [Ts, acc], "D": D},
          open("_batch234_cache.json", "w"))
print("cached.")

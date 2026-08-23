"""
Path D, the unblock: the superposed forward pass was capped at C ~ 0.02xD because the WHOLE
weight-memory was ONE vector. Now that we know how to distribute (federate vectors + a thin
layer), apply it to COMPUTE, not just storage -- and push the wall the single-model version hit.

Single-model (the blocked Path-D flagship): W_mem = bundle( bind(role_c, w_c) for ALL C classes ).
One vector holds every weight row; recovering row c carries crosstalk from the other C-1 rows, and
the continuous logit <w_hat_c, x> has no cleanup to absorb it -> fidelity dies at C ~ 0.02xD.

Distributed (the fix, identical to the array's move): split the C classes across K weight-memory
SHARDS, each holding C/K rows. Recovering a row now carries crosstalk from only its ~C/K shard-mates,
so each shard stays faithful while C/K <~ 0.02xD -- i.e. the wall moves to C ~ K x 0.02xD. The thin
layer is trivial: class c lives in shard (c mod K). Same federation, applied to the forward pass.

Honest note kept on the record: total unbinds are still C (grouped into K vectors), so this buys
FIDELITY/capacity, not fewer FLOPs -- exactly like the array (capacity comes from more vectors, and
on parallel/neuromorphic hardware the K shards run at once). The win measured here is the wall moving.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from holographic.misc.holographic_core import unitary_vector, bundle
from holographic.agents_and_reasoning.holographic_ai import bind_batch, bind_fixed
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(7)
D = 1024

def involution_stack(A):
    return np.concatenate([A[:, :1], A[:, :0:-1]], axis=1)
def unbind_fixed(M, KEYS):
    return bind_fixed(M, involution_stack(KEYS))      # recover every row from one shard at once
def mint_unitary(n, dim):
    return np.stack([unitary_vector(dim, rng) for _ in range(n)])

def make_task(C, n_feat=20, per_class=80, sep=6.0, std=1.0, seed=0):
    X, y = make_blobs(n_samples=per_class * C, centers=C, n_features=n_feat,
                      cluster_std=std, center_box=(-sep, sep), random_state=seed)
    return train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)

def evaluate(D, C, K, n_feat=20, seeds=(0, 1, 2)):
    """Forward pass read out of K weight-memory shards (K=1 is the original single-vector version)."""
    fids, ax, asu = [], [], []
    for s in seeds:
        Xtr, Xte, ytr, yte = make_task(C, n_feat=n_feat, seed=s)
        R = rng.standard_normal((D, n_feat)) / np.sqrt(n_feat)
        def enc(F):
            H = F @ R.T
            return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        Htr, Hte = enc(Xtr), enc(Xte)
        W = np.stack([Htr[ytr == c].mean(0) for c in range(C)])
        W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        L_exact = Hte @ W.T
        roles = mint_unitary(C, D)
        # ---- federate the weight rows across K shards by (class mod K) ----
        W_hat = np.zeros((C, D))
        for k in range(K):
            idx = np.arange(C)[np.arange(C) % K == k]
            if len(idx) == 0:
                continue
            shard = bundle(bind_batch(roles[idx], W[idx]))     # one vector holds this shard's rows
            W_hat[idx] = unbind_fixed(shard, roles[idx])       # recover only this shard's rows
        L_super = Hte @ W_hat.T
        Le = L_exact - L_exact.mean(1, keepdims=True)
        Ls = L_super - L_super.mean(1, keepdims=True)
        fids.append(float(np.mean((Le * Ls).sum(1) /
                    (np.sqrt((Le ** 2).sum(1) * (Ls ** 2).sum(1)) + 1e-12))))
        ax.append(float(np.mean(L_exact.argmax(1) == yte)))
        asu.append(float(np.mean(L_super.argmax(1) == yte)))
    return float(np.mean(fids)), float(np.mean(ax)), float(np.mean(asu))

Cs = [8, 16, 32, 48, 64, 96, 128, 160, 200, 250]
Ks = [1, 2, 4, 8]
print(f"Distributed superposed forward pass (D={D}). Does federating the weight-memory move the wall?")
print("=" * 78)
res = {K: {"fid": [], "ax": [], "asu": []} for K in Ks}
for K in Ks:
    for C in Cs:
        f, ae, asu_ = evaluate(D, C, K)
        res[K]["fid"].append(f); res[K]["ax"].append(ae); res[K]["asu"].append(asu_)
    cliff = max([c for c, f in zip(Cs, res[K]["fid"]) if f >= 0.90], default=0)
    print(f"  K={K} shard(s): logit-fidelity>=0.90 holds to C={cliff:4d} classes  "
          f"(~{cliff/D:.3f} x D)   [single-vector budget x{K}]", flush=True)
print("=" * 78)
base = max([c for c, f in zip(Cs, res[1]["fid"]) if f >= 0.90], default=0)
top = max([c for c, f in zip(Cs, res[8]["fid"]) if f >= 0.90], default=0)
print(f"  single vector held C={base}; 8 shards hold C={top}  -> ~{top/max(base,1):.1f}x more classes "
      f"faithful, from the same federation move that fixed storage")
import json
json.dump({"Cs": Cs, "Ks": Ks, "res": res, "D": D}, open("_fwd_cache.json", "w"))
print("cached.")

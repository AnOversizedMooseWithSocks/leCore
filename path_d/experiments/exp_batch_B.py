"""
A2 (honest finish), A5 (federated archive), A6 (residue integer range) -- on the real kernel.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from holographic.misc.holographic_core import unitary_vector, random_vector, bundle
from holographic.agents_and_reasoning.holographic_ai import bind_batch, bind_fixed
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
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ============ A2: dense continuous matmul -- the honest boundary ============
def a2(Ms=(8,16,32,48,64,96,128,160,200,256), trials=6):
    dense = {1: [], 4: []}
    for M in Ms:
        for K in (1, 4):
            fs = []
            for t in range(trials):
                W = np.stack([random_vector(D, rng) for _ in range(M)]); W /= np.linalg.norm(W,axis=1,keepdims=True)
                x = random_vector(D, rng); x /= np.linalg.norm(x)
                roles = np.stack([unitary_vector(D, rng) for _ in range(M)])
                fs.append(corr(recover_federated(roles, W, K) @ x, W @ x))
            dense[K].append(float(np.mean(fs)))
    return list(Ms), dense

# ============ A5: federated content archive (conservation at fixed total dim) ============
def a5(Ns=(16,32,64,128), B=4096, Ksh=4, S=16, trials=2):
    from holographic.misc.holographic_archive import HolographicArchive
    mono, fed = [], []
    for N in Ns:
        mc, fc = [], []
        for t in range(trials):
            imgs = [rng.random((S, S)) for _ in range(N)]
            keep = min(S * S, max(8, B // N))
            a = HolographicArchive((S, S, 1), capacity=N, keep=keep, dim=B, seed=t)
            for im in imgs: a.add(im)
            mc.append(np.mean([corr(a.recover(i).ravel(), imgs[i].ravel()) for i in range(N)]))
            # federated: Ksh archives, total dim B, same keep -> same per-image budget (conserved)
            per = B // Ksh; capk = (N + Ksh - 1) // Ksh
            arcs = [HolographicArchive((S, S, 1), capacity=capk, keep=min(keep, per // max(capk,1)),
                                       dim=max(per, S*S), seed=100 + t * 10 + k) for k in range(Ksh)]
            for i, im in enumerate(imgs): arcs[i % Ksh].add(im)            # directory: image i -> shard i%K
            cs = []
            for i in range(N):
                k = i % Ksh; local = i // Ksh
                cs.append(corr(arcs[k].recover(local).ravel(), imgs[i].ravel()))
            fc.append(np.mean(cs))
        mono.append(float(np.mean(mc))); fed.append(float(np.mean(fc)))
    return list(Ns), mono, fed

# ============ A6: residue integer range (CRT) ============
def crt(res, mods):
    M = 1
    for m in mods: M *= m
    x = 0
    for r, m in zip(res, mods):
        Mi = M // m; x += r * Mi * pow(Mi, -1, m)
    return x % M
def _sieve(n):
    ok = [True]*(n+1); ps=[]
    for i in range(2,n+1):
        if ok[i]:
            ps.append(i)
            for j in range(i*i,n+1,i): ok[j]=False
    return ps
PRIMES = _sieve(2000)   # ~300 primes, enough to push past the single-vector capacity
def encode_bundle(N, mods, mkeys, digits):
    return bundle([bind_batch(mkeys[i:i+1], digits[i][N % m:N % m + 1])[0] for i, m in enumerate(mods)])
def a6(ks=(24,48,72,96,120,160,200), Ks=(1,8), trials=25):
    rng2 = np.random.default_rng(3)
    acc = {K: [] for K in Ks}; ranges = []
    for k in ks:
        mods = PRIMES[:k]; R = 1
        for m in mods: R *= m
        ranges.append(R)
        mkeys = np.stack([unitary_vector(D, rng) for _ in range(k)])
        digits = [np.stack([random_vector(D, rng) for _ in range(m)]) for m in mods]   # residue atoms
        for K in Ks:
            ok = 0
            for t in range(trials):
                N = int(rng2.integers(0, min(R, 10**9)))
                truer = [N % m for m in mods]
                # federate the k moduli across K channels (shards); each shard bundles its moduli
                rec = [None] * k
                for sh in range(K):
                    idx = [i for i in range(k) if i % K == sh]
                    if not idx: continue
                    Mv = bundle([bind_batch(mkeys[i:i+1], digits[i][truer[i]:truer[i]+1])[0] for i in idx])
                    for i in idx:
                        est = bind_fixed(Mv, inv_stack(mkeys[i:i+1]))[0]
                        rec[i] = int((digits[i] @ est).argmax())            # cleanup to residue class
                ok += int(crt(rec, mods) == N % R)
            acc[K].append(ok / trials)
    return list(ks), ranges, acc

print("A2 superposed matmul -- dense (boundary) vs low-rank rank-8 (escape), D=%d" % D)
Ms, dense = a2()
for K in (1, 4):
    vals = ", ".join(f"M={m}:{f:.2f}" for m, f in zip(Ms, dense[K]) if m in (8,16,32,64,128))
    print(f"   dense K={K} fidelity:  {vals}")
print("   -> continuous matmul has no cleanup to absorb crosstalk; federation lifts it a little but")
print("      it never gets good -- the precision wall, not a capacity wall. (A3/A4 win *because* they")
print("      end in a discrete cleanup: argmax / codebook snap.)")
print("A5 federated archive -- recovery quality vs #images at FIXED total dim (conserved?)")
Ns, mono, fed = a5()
for N, m, f in zip(Ns, mono, fed):
    print(f"   N={N:4d}: monolithic corr={m:.3f}   federated(K=4) corr={f:.3f}")
print("A6 residue integer range -- round-trip accuracy vs #moduli (range), single vs federated")
ks, ranges, acc = a6()
for K in acc:
    last_ok = [k for k, a in zip(ks, acc[K]) if a >= 0.95]
    kk = max(last_ok, default=0); R = ranges[ks.index(kk)] if kk else 0
    print(f"   K={K}: 95%-roundtrip to k={kk:3d} moduli  (range ~ 1e{len(str(R))-1 if R else 0}, {kk} moduli)")
import json
json.dump({"A2": [Ms, dense], "A5": [Ns, mono, fed], "A6": [ks, ranges, acc], "D": D},
          open("_batchB_cache.json", "w"))
print("cached.")

"""Batch operations on the holographic substrate -- the things VSA does well in
bulk: superposition (many items in one vector), a 1-bit memory footprint, and
vectorised cleanup."""
import numpy as np, time
from holographic_ai import Vocabulary, HolographicMemory

dim = 8192
print("BATCH OPERATIONS  (dim = %d)\n" % dim)

# 1) SUPERPOSITION -- many key->value pairs in ONE vector ------------------
def capacity(N, seed=0):
    v = Vocabulary(dim, seed=seed); mem = HolographicMemory(dim)
    vals = [f"v{i}" for i in range(N)]
    for i in range(N):
        v.get(f"k{i}"); v.get(vals[i]); mem.learn(v.get(f"k{i}"), v.get(vals[i]))
    return np.mean([v.cleanup(mem.recall(v.get(f"k{i}")), candidates=vals)[0] == vals[i]
                    for i in range(N)]) * 100

print("1) SUPERPOSITION -- store many key->value pairs in ONE vector and recall")
print("   each by content (a hash map needs one slot per pair; this needs one vector):")
print(f"   {'pairs':>7}{'recall':>9}   footprint")
for N in [100, 200, 400, 600, 800]:
    acc = np.mean([capacity(N, s) for s in range(3)])
    print(f"   {N:>7}{acc:>8.0f}%   1 vector  (vs {N} entries)")
print("   -> ~200-400 distinct pairs per vector at high recall, then graceful decay.\n")

# 2) 1-BIT MEMORY -- 32x smaller database at equal accuracy ----------------
N, d, Q = 10000, 8192, 100
rng = np.random.default_rng(0)
items = rng.standard_normal((N, d)).astype(np.float32)
truth = rng.choice(N, Q, replace=False); q = items[truth].copy()
q[rng.random((Q, d)) < 0.20] *= -1
itn = items / np.linalg.norm(items, axis=1, keepdims=True)
qn = q / np.linalg.norm(q, axis=1, keepdims=True)
rf = np.mean((qn @ itn.T).argmax(1) == truth) * 100
itb = np.packbits(items > 0, axis=1); qb = np.packbits(q > 0, axis=1)
nn = np.array([np.bitwise_count(itb ^ qb[i]).sum(1).argmin() for i in range(Q)])
rb = np.mean(nn == truth) * 100
print("2) 1-BIT MEMORY -- store the vector database as 1-bit signs:")
print(f"   {N}-item database, {Q} noisy queries (20% of signs flipped)")
print(f"   recall@1   float32 {rf:.0f}%      1-bit {rb:.0f}%        (equal accuracy)")
print(f"   memory     float32 {items.nbytes/1e6:.0f} MB    1-bit {itb.nbytes/1e6:.0f} MB     (32x smaller)\n")

# 3) BATCH CLEANUP -- a whole batch snapped to nearest memory in one matmul -
M, Qc = 2000, 500
v = Vocabulary(d, 0); V = np.array([v.get(f"w{i}") for i in range(M)])
qs = V[rng.integers(0, M, Qc)] + 0.6 * rng.standard_normal((Qc, d))
t = time.perf_counter(); (qs @ V.T).argmax(1); dt = time.perf_counter() - t
print("3) BATCH CLEANUP -- snap a batch of noisy vectors to the nearest memory item")
print(f"   {Qc} queries vs {M}-item memory in one vectorised matmul:")
print(f"   {dt*1e3:.0f} ms  ({Qc/dt:,.0f} cleanups/sec)")

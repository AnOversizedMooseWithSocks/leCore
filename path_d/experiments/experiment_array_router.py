"""
Path D, pushing the limit: a VSA-native ROUTER over the shards to break the broadcast wall.

The stress test found the array's only real ceiling is routerless broadcast: it soft-erodes with
shard count and costs O(shards)/query. The fix is the same content-addressable trick one rung up
("as above, so below"): summarize each shard by a SKETCH = bundle of its keys (the holographic
'and' of what it holds), match a query against the sketches to pick a few candidate shards, then
unbind+cleanup only those. Routing by key-sketch is a CLEAN decision (a key sits ~1/sqrt(load)
inside its own shard's sketch, far above the 1/sqrt(D) noise from the others), so it stays accurate
where broadcast's value-cleanup vote drowns.

  PART 1 -- sketch-routed recall vs full broadcast vs the directory, pushed to ~100k items.
  PART 2 -- does a 2-level index (sketch-of-sketches) buy SUBLINEAR routing, or does the per-vector
            capacity wall cap the fan-out per level? Measured, negative kept if so.
"""
import sys, os, time, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
import holographic.agents_and_reasoning.holographic_ai as holographic_ai, holographic.misc.holographic_array as holographic_array
from holographic.misc.holographic_array import HoloArray
from holographic.misc.holographic_core import bind, unbind, bundle
D = 1024

_orig = holographic_ai.derived_atom
@functools.lru_cache(maxsize=None)
def _cached(seed, name, dim, unitary):
    return _orig(seed, name, dim, unitary)
_p = lambda s, n, d, unitary=False: _cached(s, n, d, unitary)
holographic_ai.derived_atom = _p
holographic_array.derived_atom = _p

# ---- build one big array; disable per-add sensing (probe huge), drive spin-up on a cadence ----
KMAX, CAP = 2048, 50
print(f"building array to {KMAX} shards x {CAP} = {KMAX*CAP} items ...", flush=True)
arr = HoloArray(D, seed=9, n_parity=0, add_threshold=0.0, probe=10**9)
members = [[]]
rng = np.random.default_rng(1)
t0 = time.perf_counter()
for g in range(KMAX * CAP):
    if g > 0 and g % CAP == 0:
        arr._spin_up(); members.append([])
    v = int(rng.integers(0, arr.n_vals)); arr.add(v); members[-1].append(g)
print(f"  built {len(arr.truth)} items / {len(arr.data)} shards in {time.perf_counter()-t0:.1f}s", flush=True)

# shard sketches: bundle of each shard's keys (the index). one extra vector per shard.
t0 = time.perf_counter()
sketches = np.stack([bundle(np.stack([arr._key(g) for g in members[s]])) for s in range(KMAX)])
print(f"  built {KMAX} shard-sketches in {time.perf_counter()-t0:.1f}s", flush=True)


def directory(g):
    return arr._recall_one(g)[0]

def full_broadcast(g, K):
    best_v, best_c = 0, -1.0
    for s in range(K):
        est = unbind(arr._norm(arr.data[s]), arr._key(g)); sims = arr.codebook @ est
        j = int(sims.argmax()); c = float(sims[j] / (np.linalg.norm(est) + 1e-12))
        if c > best_c: best_v, best_c = j, c
    return best_v

def routed(g, K, c=8):
    key = arr._key(g)
    scores = sketches[:K] @ key                          # ONE matmul: K cheap dot products
    cand = np.argpartition(-scores, min(c, K - 1))[:c]   # top-c candidate shards
    best_v, best_c = 0, -1.0
    for s in cand:                                       # unbind+cleanup only the candidates
        est = unbind(arr._norm(arr.data[s]), key); sims = arr.codebook @ est
        j = int(sims.argmax()); cf = float(sims[j] / (np.linalg.norm(est) + 1e-12))
        if cf > best_c: best_v, best_c = j, cf
    return best_v

def measure(fn, K, n, seed):
    r = np.random.default_rng(seed)
    pool = [g for g, (k, v) in arr.truth.items() if k < K]
    gs = [int(x) for x in r.choice(len(pool), size=min(n, len(pool)), replace=False)]
    gs = [pool[i] for i in gs]
    t0 = time.perf_counter()
    ok = sum(fn(g) == arr.truth[g][1] for g in gs)
    return ok / len(gs), (time.perf_counter() - t0) / len(gs)

print("\nPART 1 -- sketch-routed vs full broadcast vs directory, vs shard count", flush=True)
part1 = []
for K in [64, 256, 1024, 2048]:
    da, dt = measure(directory, K, 150, K)
    ra, rt = measure(lambda g: routed(g, K), K, 150, K + 1)
    if K <= 1024:
        ba, bt = measure(lambda g: full_broadcast(g, K), K, 80, K + 2)
    else:
        ba, bt = (None, None)                            # too expensive to bother -- that's the point
    part1.append([K, da, ra, ba, dt, rt, bt])
    bs = f"broadcast={ba:.3f}@{bt*1e6:.0f}us" if ba is not None else "broadcast=(skipped: O(K) too slow)"
    print(f"  K={K:5d} ({K*CAP:6d} items)  directory={da:.3f}  routed={ra:.3f}  {bs}  "
          f"| t: dir={dt*1e6:.0f} routed={rt*1e6:.0f}us", flush=True)

print("\nPART 2 -- does a 2-level index give sublinear routing? (routing-recall vs #comparisons)", flush=True)
def true_in_candidates_1level(g, K, c=8):
    scores = sketches[:K] @ arr._key(g)
    cand = set(int(x) for x in np.argpartition(-scores, min(c, K - 1))[:c])
    return arr.truth[g][0] in cand, K            # comparisons = K

def two_level(g, K, group, cg=4, c=8):
    G = K // group
    gsk = np.stack([bundle(sketches[grp*group:(grp+1)*group]) for grp in range(G)])  # group sketches
    key = arr._key(g)
    gs = gsk @ key
    top_groups = np.argpartition(-gs, min(cg, G - 1))[:cg]                            # top groups
    cand_shards = []
    for grp in top_groups:
        cand_shards += list(range(grp*group, (grp+1)*group))
    sub = np.array(cand_shards)
    ss = sketches[sub] @ key
    cand = set(int(sub[i]) for i in np.argpartition(-ss, min(c, len(sub)-1))[:c])     # top shards
    comparisons = G + len(cand_shards)
    return arr.truth[g][0] in cand, comparisons

K = KMAX
r = np.random.default_rng(99); pool = [g for g in arr.truth if arr.truth[g][0] < K]
sample = [int(x) for x in r.choice(len(pool), 200, replace=False)]; sample = [pool[i] for i in sample]
hit1 = np.mean([true_in_candidates_1level(g, K)[0] for g in sample]); comp1 = K
print(f"  1-level:           routing-recall={hit1:.3f}   comparisons={comp1}", flush=True)
part2 = [("1-level", hit1, comp1)]
for group in [16, 32, 64, 128]:
    res = [two_level(g, K, group) for g in sample]
    hit = np.mean([x[0] for x in res]); comp = int(np.mean([x[1] for x in res]))
    part2.append((f"2-level g={group}", hit, comp))
    print(f"  2-level g={group:3d} (G={K//group:3d}):  routing-recall={hit:.3f}   comparisons={comp}", flush=True)

# cache for plotting
import json
json.dump({"part1": part1, "part2": [[a, float(b), int(c)] for a, b, c in part2], "CAP": CAP, "KMAX": KMAX},
          open("_router_cache.json", "w"))
print("\ncached.", flush=True)

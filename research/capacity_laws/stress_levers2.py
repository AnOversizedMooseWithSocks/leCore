"""Phase 3 -- break the tie. A tie at 1.000 is a CEILING, not a comparison.

Phase 2 had lever 4 and lever 6 both at 1.000 for K=128 at 8704 floats. That says nothing
except that K=128 is easy at that budget. Push K until the ceiling breaks, then compare at
MATCHED TOTAL FLOATS. Also test lever 6's own recursion clause: "when the coordinator hits
its own limit, split again."
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, unitary_vector

def mk(dim, n, rng): return [unitary_vector(dim, rng) for _ in range(n)]

def snap(noisy, chunks):
    """The crosstalk reset: snap the noisy chunk to its exact stored pattern."""
    C = np.stack(chunks)
    s = C @ noisy / (np.linalg.norm(C, axis=1) * np.linalg.norm(noisy) + 1e-30)
    return chunks[int(np.argmax(s))]

def build_flat(keys, vals):
    out = np.zeros(len(keys[0]))
    for k, v in zip(keys, vals): out += bind(k, v)
    return out

def acc_flat(dim, K, seed, dt):
    rng = np.random.default_rng(seed)
    keys, vals = mk(dim, K, rng), mk(dim, K, rng)
    V = np.stack(vals); st = build_flat(keys, vals).astype(dt).astype(np.float64)
    return sum(int(np.argmax(V @ unbind(st, keys[i]))) == i for i in range(K)) / K, dim

def acc_grouped(dim, K, g, seed, dt, levels=1):
    rng = np.random.default_rng(seed)
    keys, vals = mk(dim, K, rng), mk(dim, K, rng)
    V = np.stack(vals)
    chunks = [build_flat(keys[i:i+g], vals[i:i+g]) for i in range(0, K, g)]
    chunks = [c.astype(dt).astype(np.float64) for c in chunks]
    gk = mk(dim, len(chunks), rng)
    floats = dim * len(chunks)
    if levels == 1:
        top = np.zeros(dim)
        for k, c in zip(gk, chunks): top += bind(k, c)
        top = top.astype(dt).astype(np.float64); floats += dim
        def leaf(i):
            return snap(unbind(top, gk[i // g]), chunks)
    else:                                   # LEVER 6 RECURSED: coordinator hit its own limit
        supers = [np.sum([bind(k, c) for k, c in zip(gk[i:i+g], chunks[i:i+g])], axis=0)
                  for i in range(0, len(chunks), g)]
        supers = [s.astype(dt).astype(np.float64) for s in supers]
        sk = mk(dim, len(supers), rng)
        root = np.zeros(dim)
        for k, s in zip(sk, supers): root += bind(k, s)
        root = root.astype(dt).astype(np.float64)
        floats += dim * (len(supers) + 1)
        def leaf(i):
            ci = i // g
            sup = snap(unbind(root, sk[ci // g]), supers)
            return snap(unbind(sup, gk[ci]), chunks)
    hits = sum(int(np.argmax(V @ unbind(leaf(i), keys[i]))) == i for i in range(K))
    return hits / K, floats

SEEDS = (0, 1)
for K in (512,):
    print("\n=== K=%d, base dim=512, MATCHED TOTAL FLOATS (3 seeds)" % K)
    g1 = [acc_grouped(512, K, 16, s, np.float32, 1) for s in SEEDS]
    budget = g1[0][1]
    g2 = [acc_grouped(512, K, 16, s, np.float32, 2) for s in SEEDS]
    l4 = [acc_flat(budget, K, s, np.float32) for s in SEEDS]
    base = [acc_flat(512, K, s, np.float32) for s in SEEDS]
    for tag, r in (("lever 6, 1 level (g=16)", g1), ("lever 6, 2 levels (g=16)", g2),
                   ("lever 4, flat dim=%d" % budget, l4), ("baseline flat dim=512", base)):
        print("   %-28s acc %.3f +- %.3f   floats %d"
              % (tag, np.mean([x[0] for x in r]), np.std([x[0] for x in r]), r[0][1]))

"""Phase 2 -- find the real cliff, then walk levers 4 and 6 against it at MATCHED BUDGET.

Phase 1 found no cliff: real-docstring recall was 98% and f32 matched f64 exactly. That means
the doc-recall regime is not where the wall is. The wall in a VSA is SUPERPOSITION CAPACITY --
K role->filler pairs bundled into ONE vector, recalled by unbinding. That is the regime the
browser tier actually lives in if the corpus is bundled rather than stored row-wise.

HONEST COMPARISON RULE: lever 4 (more dimensions) buys capacity WITH MEMORY, so comparing
dim=512 flat against dim=2048 flat is not a comparison, it is a purchase. Everything below is
compared at MATCHED TOTAL FLOATS.

Lever 6 says: a measured limit is a TILE SIZE -- group at the cliff, add a coordinator level,
and CLEAN UP BETWEEN LEVELS.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, unitary_vector

def flat_store(keys, vals):
    """One vector holding every pair. Capacity law applies."""
    out = np.zeros(len(keys[0]))
    for k, v in zip(keys, vals):
        out += bind(k, v)
    return out

def flat_recall(store, key, codebook):
    return int(np.argmax(codebook @ unbind(store, key)))

def grouped_store(keys, vals, gkeys, group):
    """LEVER 6: group at the cliff, bind each GROUP under a group key, bundle the groups.
    Recall unbinds the group key first, CLEANS UP to the exact chunk, then unbinds the leaf."""
    dim = len(keys[0]); chunks = []
    for g0 in range(0, len(keys), group):
        c = np.zeros(dim)
        for k, v in zip(keys[g0:g0+group], vals[g0:g0+group]):
            c += bind(k, v)
        chunks.append(c)
    top = np.zeros(dim)
    for gk, c in zip(gkeys, chunks):
        top += bind(gk, c)
    return top, chunks

def grouped_recall(top, chunks, gi, key, gkeys, codebook, clean=True):
    noisy = unbind(top, gkeys[gi])
    if clean:                                  # THE CROSSTALK RESET -- snap to the exact chunk
        sims = np.array([noisy @ c / (np.linalg.norm(noisy) * np.linalg.norm(c) + 1e-30)
                         for c in chunks])
        noisy = chunks[int(np.argmax(sims))]
    return int(np.argmax(codebook @ unbind(noisy, key)))

def run(dim, K, group, seed, dtype=np.float64):
    rng = np.random.default_rng(seed)
    keys = [unitary_vector(dim, rng) for _ in range(K)]
    vals = [unitary_vector(dim, rng) for _ in range(K)]
    V = np.stack(vals).astype(dtype).astype(np.float64)
    ng = (K + group - 1) // group
    gkeys = [unitary_vector(dim, rng) for _ in range(ng)]

    st = flat_store(keys, vals).astype(dtype).astype(np.float64)
    flat_hits = sum(flat_recall(st, keys[i], V) == i for i in range(K))

    top, chunks = grouped_store(keys, vals, gkeys, group)
    top = top.astype(dtype).astype(np.float64)
    chunks = [c.astype(dtype).astype(np.float64) for c in chunks]
    grp_hits = sum(grouped_recall(top, chunks, i // group, keys[i], gkeys, V) == i
                   for i in range(K))
    # floats actually stored: flat = one vector; grouped = top + every chunk
    return (flat_hits / K, grp_hits / K, dim, dim * (1 + len(chunks)))

if __name__ == "__main__":
    SEEDS = (0, 1, 2, 3, 4)

    print("A. WHERE IS THE CLIFF?  flat bundled store, dim=512, f64 (5 seeds, mean +- spread)")
    print("   K     flat_acc")
    for K in (8, 16, 32, 64, 128, 256):
        a = [run(512, K, 8, s)[0] for s in SEEDS]
        print("   %-5d %.3f +- %.3f" % (K, np.mean(a), np.std(a)))

    print("\nB. LEVER 4 vs LEVER 6 AT MATCHED TOTAL FLOATS (K=128, 5 seeds)")
    print("   arrangement                              acc      floats stored")
    a4 = [run(512, 128, 8, s) for s in SEEDS]
    print("   lever 6: dim=512, group=8  (16 chunks)   %.3f +- %.3f   %d"
          % (np.mean([x[1] for x in a4]), np.std([x[1] for x in a4]), a4[0][3]))
    # lever 4 at the SAME budget: one flat vector of the same total width
    budget = a4[0][3]
    aL4 = [run(budget, 128, 8, s)[0] for s in SEEDS]
    print("   lever 4: dim=%-5d flat, one vector      %.3f +- %.3f   %d"
          % (budget, np.mean(aL4), np.std(aL4), budget))
    aBase = [run(512, 128, 8, s)[0] for s in SEEDS]
    print("   baseline: dim=512 flat, one vector       %.3f +- %.3f   %d"
          % (np.mean(aBase), np.std(aBase), 512))

    print("\nC. DOES THE f32 SHADER PATH SURVIVE THE GROUPED FORM? (K=128, dim=512, group=8)")
    for tag, dt in (("f64", np.float64), ("f32", np.float32)):
        g = [run(512, 128, 8, s, dt)[1] for s in SEEDS]
        f = [run(512, 128, 8, s, dt)[0] for s in SEEDS]
        print("   %s  flat %.3f +- %.3f   grouped %.3f +- %.3f"
              % (tag, np.mean(f), np.std(f), np.mean(g), np.std(g)))

    print("\nD. GROUP SIZE SWEEP -- is the cliff number the right tile size? (K=128, dim=512)")
    print("   group   grouped_acc     floats")
    for g in (4, 8, 16, 32, 64):
        a = [run(512, 128, g, s) for s in SEEDS]
        print("   %-7d %.3f +- %.3f   %d" % (g, np.mean([x[1] for x in a]),
                                             np.std([x[1] for x in a]), a[0][3]))

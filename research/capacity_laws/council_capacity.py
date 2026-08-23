"""COUNCIL FIX #1 -- my capacity cliff was CONFOUNDED, and the literature says how.

Plate (1994, p.160ff) and the VSA-comparison literature both note that superposition-memory
accuracy depends on the ITEM MEMORY SIZE as well as the number of bundled pairs; the standard
experiment (e.g. the GHRR paper) fixes an item memory of ~1000 and bundles k of them.

My earlier sweep scored each query against a codebook that WAS the k stored values -- so as k
grew, crosstalk AND the number of distractors grew together. Two effects, one number.
Here the item memory N is FIXED and only k varies.

LITERATURE PREDICTION being tested (Frady/Kleyko/Sommer; per-slot SNR = 1/k with signal power
1/d and noise power k/d => capacity grows LINEARLY in d): the k at which accuracy crosses 50%
should scale ~1:2:4 across d = 256:512:1024.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, unitary_vector

N_ITEM = 512          # FIXED item memory, independent of k -- the confound removed

def capacity_curve(dim, ks, seeds=(0, 1, 2)):
    out = {}
    for k in ks:
        accs = []
        for s in seeds:
            rng = np.random.default_rng(1000 * s + dim)
            V = np.stack([unitary_vector(dim, rng) for _ in range(N_ITEM)])
            keys = [unitary_vector(dim, rng) for _ in range(k)]
            idx = rng.choice(N_ITEM, k, replace=False)
            st = np.zeros(dim)
            for kk, j in zip(keys, idx):
                st += bind(kk, V[j])
            hit = sum(int(np.argmax(V @ unbind(st, keys[i]))) == idx[i] for i in range(k))
            accs.append(hit / k)
        out[k] = (float(np.mean(accs)), float(np.std(accs)))
    return out

def crossing(curve, level=0.5):
    ks = sorted(curve)
    for a, b in zip(ks, ks[1:]):
        ya, yb = curve[a][0], curve[b][0]
        if ya >= level >= yb:
            return a + (ya - level) / (ya - yb) * (b - a)
    return float("nan")

if __name__ == "__main__":
    grids = {256: (4, 8, 16, 24, 32, 48, 64, 96),
             512: (8, 16, 32, 48, 64, 96, 128, 192),
             1024: (16, 32, 64, 96, 128, 192, 256, 384)}
    xs = {}
    for dim, ks in grids.items():
        c = capacity_curve(dim, ks)
        xs[dim] = crossing(c)
        print("dim=%-5d " % dim + "  ".join("k=%d:%.2f" % (k, c[k][0]) for k in ks))
        print("        50%% crossing at k = %.1f" % xs[dim])
    base = xs[256]
    print("\nLITERATURE CHECK -- capacity should be LINEAR in d (ratios ~1 : 2 : 4)")
    for dim in (256, 512, 1024):
        print("   d=%-5d k50=%6.1f   ratio vs d=256: %.2f  (linear predicts %.2f)"
              % (dim, xs[dim], xs[dim] / base, dim / 256))

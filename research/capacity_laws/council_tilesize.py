"""RE-DERIVE THE TILE SIZE, and settle WHICH law sets it.

Correction first: the group sweep itself was NOT distractor-confounded -- K was fixed at 128
across every arm, so the codebook was constant. What the corrected capacity curve breaks is the
INTERPRETATION: I celebrated "lever 6 named its own tile size" because g=16 matched a cliff I
had measured at ~16-32. The corrected d=512 curve puts the 50% point at k=59, so that
coincidence needs re-earning.

TWO COMPETING LAWS, and they make DIFFERENT predictions when K changes:
  (a) CAPACITY LAW  -- the tile is the per-vector capacity bound, so g* is FIXED in K.
  (b) BALANCE LAW   -- a 2-level tree loads the leaves with g and the coordinator with K/g,
                       so g* ~ sqrt(K) and MOVES with K.
K=128 -> sqrt=11.3, K=512 -> sqrt=22.6, K=2048 -> sqrt=45.3. If the optimum walks right as K
grows, it is the balance law; if it sits still, it is the capacity law.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, unitary_vector

def grouped_acc(dim, K, g, seed):
    rng = np.random.default_rng(seed)
    keys = np.stack([unitary_vector(dim, rng) for _ in range(K)])
    V    = np.stack([unitary_vector(dim, rng) for _ in range(K)])
    ch   = np.stack([np.sum([bind(keys[j], V[j]) for j in range(i, min(i + g, K))], axis=0)
                     for i in range(0, K, g)])
    gk   = np.stack([unitary_vector(dim, rng) for _ in range(len(ch))])
    top  = np.sum([bind(gk[i], ch[i]) for i in range(len(ch))], axis=0)
    cn   = np.linalg.norm(ch, axis=1) + 1e-30
    hits = 0
    for i in range(K):
        noisy = unbind(top, gk[i // g])
        c = ch[int(np.argmax(ch @ noisy / cn))]          # crosstalk reset
        hits += int(np.argmax(V @ unbind(c, keys[i]))) == i
    return hits / K, dim * (1 + len(ch))

if __name__ == "__main__":
    SEEDS = (40, 41, 42)
    for K in (128, 512):
        print("\nK=%d, dim=512, 3 held-out seeds   (sqrt(K)=%.1f)" % (K, np.sqrt(K)))
        print("   g     acc              floats   acc/float x1e4")
        best = None
        for g in (8, 12, 16, 23, 32, 45, 64):
            a = [grouped_acc(512, K, g, s) for s in SEEDS]
            m, sd, fl = np.mean([x[0] for x in a]), np.std([x[0] for x in a]), a[0][1]
            print("   %-5d %.4f +- %.4f   %-8d %.3f" % (g, m, sd, fl, 1e4 * m / fl))
            if best is None or m > best[1] + 1e-9:
                best = (g, m)
        print("   -> best accuracy at g=%d" % best[0])

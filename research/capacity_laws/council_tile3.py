"""The balance law makes a DERIVED prediction, so test it instead of sweeping again.

2-tier optimum moved 8 -> 16 when K went 128 -> 512, i.e. g* ~ sqrt(K) (shifted below it,
because a leaf recall faces K distractors while a coordinator recall faces only K/g).
The law's next claim is mechanical: with THREE tiers the loads are g, K/g^2 ... balanced at
g* ~ K^(1/3). For K=512 that is exactly 8 -- the value that was CATASTROPHIC at two tiers.
If 3-tier g=8 beats 2-tier g=16, the balance law predicted a number rather than fitting one.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, unitary_vector

def tri(dim, K, g, seed):
    rng = np.random.default_rng(seed)
    keys = np.stack([unitary_vector(dim, rng) for _ in range(K)])
    V    = np.stack([unitary_vector(dim, rng) for _ in range(K)])
    ch = np.stack([np.sum([bind(keys[j], V[j]) for j in range(i, min(i+g, K))], axis=0)
                   for i in range(0, K, g)])
    gk = np.stack([unitary_vector(dim, rng) for _ in range(len(ch))])
    su = np.stack([np.sum([bind(gk[j], ch[j]) for j in range(i, min(i+g, len(ch)))], axis=0)
                   for i in range(0, len(ch), g)])
    sk = np.stack([unitary_vector(dim, rng) for _ in range(len(su))])
    root = np.sum([bind(sk[i], su[i]) for i in range(len(su))], axis=0)
    cn, sn = np.linalg.norm(ch, axis=1)+1e-30, np.linalg.norm(su, axis=1)+1e-30
    hits = 0
    for i in range(K):
        ci = i // g
        s = su[int(np.argmax(su @ unbind(root, sk[ci // g]) / sn))]
        c = ch[int(np.argmax(ch @ unbind(s, gk[ci]) / cn))]
        hits += int(np.argmax(V @ unbind(c, keys[i]))) == i
    return hits / K, dim * (len(ch) + len(su) + 1)

SEEDS = (40, 41, 42)
print("K=512, dim=512, 3 tiers, held-out seeds.  K^(1/3) = %.1f" % 512 ** (1/3))
print("   g     acc              floats")
for g in (8, 12, 16):
    a = [tri(512, 512, g, s) for s in SEEDS]
    print("   %-5d %.4f +- %.4f   %d" % (g, np.mean([x[0] for x in a]),
                                         np.std([x[0] for x in a]), a[0][1]))
print("   (2-tier best was g=16: 0.9707 +- 0.0125 at 16896 floats)")

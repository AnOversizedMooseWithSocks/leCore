"""exp_a2_permroles.py -- SYNTHETIC. Follow-up to exp_a: a MAP product is commutative, so FROM*x*TO*y ==
FROM*y*TO*x and bound role atoms cannot carry direction inside ONE product. Test PERMUTATION roles
(roles_by_shift: FROM = rho^1, TO = rho^2) inside the product, and the phasor factor's swap rate."""
import sys, time, json
import numpy as np
sys.path.insert(0, "/home/claude/leCore")
from holographic.misc.holographic_resonator import ResonatorNetwork, map_codebook, map_bind
from holographic.agents_and_reasoning import holographic_phasor as PH

NV, NC = 20, 50


def perm_product(D, trials, restarts=20, iters=200, noise=0.0, seed=0):
    verbs = map_codebook(NV, D, seed + 1)
    vals = map_codebook(NC, D, seed + 2)
    Bf, Bt = np.roll(vals, 1, axis=1), np.roll(vals, 2, axis=1)      # role = a power of one shift
    net = ResonatorNetwork([verbs, Bf, Bt])
    rng = np.random.default_rng(1000 + seed)
    ok = sw = solved = 0; ms = []; rs = []; its = []
    for t in range(trials):
        v, x, y = int(rng.integers(NV)), int(rng.integers(NC)), int(rng.integers(NC))
        while y == x:
            y = int(rng.integers(NC))
        c = map_bind(verbs[v], Bf[x], Bt[y])
        if noise:
            c = np.where(rng.random(D) < noise, -c, c)
        t0 = time.perf_counter(); r = net.factor(c, restarts=restarts, iters=iters)
        ms.append(1000 * (time.perf_counter() - t0))
        f = tuple(int(i) for i in r["factors"])
        ok += f == (v, x, y); sw += f == (v, y, x); solved += bool(r["solved"])
        if r["solved"]:
            rs.append(r["restarts"]); its.append(r["iterations"])
    return {"D": D, "trials": trials, "noise_flip": noise, "exact_correct": ok / trials, "swapped_xy": sw / trials,
            "solved_flag": solved / trials, "median_restarts": float(np.median(rs)) if rs else None,
            "median_iters": float(np.median(its)) if its else None, "ms_mean": round(float(np.mean(ms)), 1)}


def phasor_swap(D, trials, seed=0):
    rng = np.random.default_rng(3000 + seed)
    verbs = np.stack([PH.atom("verb:%d" % i, D) for i in range(NV)])
    vals = np.stack([PH.atom("val:%d" % i, D) for i in range(NC)])
    FROM, TO = PH.atom("role:FROM", D), PH.atom("role:TO", D)
    Bf, Bt = vals * FROM, vals * TO
    ok = sw = verb_ok = 0
    for t in range(trials):
        v, x, y = int(rng.integers(NV)), int(rng.integers(NC)), int(rng.integers(NC))
        while y == x:
            y = int(rng.integers(NC))
        f = tuple(int(i) for i in PH.factor(verbs[v] * Bf[x] * Bt[y], [verbs, Bf, Bt], iters=100))
        ok += f == (v, x, y); sw += f == (v, y, x); verb_ok += f[0] == v
    return {"D": D, "trials": trials, "exact": ok / trials, "swapped": sw / trials, "verb_right": verb_ok / trials}


if __name__ == "__main__":
    T0 = time.perf_counter()
    res = {"perm_product": [perm_product(2048, 30), perm_product(1024, 30), perm_product(2048, 20, noise=0.05)],
           "phasor_product": [phasor_swap(2048, 20)]}
    res["cpu_s"] = round(time.perf_counter() - T0, 1)
    print(json.dumps(res, indent=1))
    json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)

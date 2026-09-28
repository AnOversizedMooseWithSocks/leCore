"""exp_a_resonator.py -- SYNTHETIC. Can the engine's ResonatorNetwork factor a method call
encoded as ONE bound product VERB x FROM-slot x TO-slot, and how does that compare with the
role-filler SUM encoding decoded by per-role unbind + cleanup (no resonator)?

Deterministic seeds; scratch only; imports the repo read-only.
Codebooks (synthetic, realistic sizes): 20 verbs, 50 values shared by both slots.
"""
import sys, time, json
import numpy as np
sys.path.insert(0, "/home/claude/leCore")
from holographic.misc.holographic_resonator import ResonatorNetwork, map_codebook, map_bind
from holographic.agents_and_reasoning import holographic_phasor as PH

OUT = {}
NV, NC = 20, 50


def books(D, seed=0):
    verbs = map_codebook(NV, D, seed + 1)
    vals = map_codebook(NC, D, seed + 2)
    FROM = map_codebook(1, D, seed + 3)[0]
    TO = map_codebook(1, D, seed + 4)[0]
    return verbs, vals, FROM, TO


def run_product(D, trials, restarts, iters, role_bound=True, noise=0.0, seed=0):
    """Product encoding c = verb * (FROM*x) * (TO*y). role_bound=False drops FROM/TO (x*y is commutative)."""
    verbs, vals, FROM, TO = books(D, seed)
    B_from = vals * FROM if role_bound else vals
    B_to = vals * TO if role_bound else vals
    net = ResonatorNetwork([verbs, B_from, B_to])
    rng = np.random.default_rng(1000 + seed)
    ok = swapped = solved = 0
    rs, its, ms = [], [], []
    for t in range(trials):
        v, x, y = int(rng.integers(NV)), int(rng.integers(NC)), int(rng.integers(NC))
        while y == x:
            y = int(rng.integers(NC))
        c = map_bind(verbs[v], B_from[x], B_to[y])
        if noise > 0:                                     # flip a fraction of components (a noisy composite)
            flip = rng.random(D) < noise
            c = np.where(flip, -c, c)
        t0 = time.perf_counter()
        r = net.factor(c, restarts=restarts, iters=iters)
        ms.append(1000 * (time.perf_counter() - t0))
        f = tuple(int(i) for i in r["factors"])
        solved += bool(r["solved"])
        ok += f == (v, x, y)
        swapped += f == (v, y, x)
        if r["solved"]:
            rs.append(r["restarts"]); its.append(r["iterations"])
    return {"D": D, "trials": trials, "restarts_cap": restarts, "iters_cap": iters, "role_bound": role_bound,
            "noise_flip": noise, "exact_correct": ok / trials, "swapped_xy": swapped / trials,
            "solved_flag": solved / trials,
            "median_restarts_when_solved": float(np.median(rs)) if rs else None,
            "median_iters_when_solved": float(np.median(its)) if its else None,
            "ms_mean": round(float(np.mean(ms)), 1), "ms_max": round(float(np.max(ms)), 1),
            "search_space": NV * NC * NC}


def run_sum(D, trials, noise_sigma=0.0, seed=0, superpose_two=False):
    """Role-filler SUM encoding s = VERB*v + FROM*x + TO*y (MAP). Decode each role by ONE unbind + cleanup."""
    verbs, vals, FROM, TO = books(D, seed)
    VERB = map_codebook(1, D, seed + 5)[0]
    rng = np.random.default_rng(2000 + seed)
    ok = chim = 0
    ms = []
    for t in range(trials):
        v, x, y = int(rng.integers(NV)), int(rng.integers(NC)), int(rng.integers(NC))
        while y == x:
            y = int(rng.integers(NC))
        s = VERB * verbs[v] + FROM * vals[x] + TO * vals[y]
        if superpose_two:                                  # a readout that blends TWO learned calls
            v2, x2, y2 = int(rng.integers(NV)), int(rng.integers(NC)), int(rng.integers(NC))
            while len({x, y, x2, y2}) < 4 or v2 == v:
                v2, x2, y2 = int(rng.integers(NV)), int(rng.integers(NC)), int(rng.integers(NC))
            s = s + VERB * verbs[v2] + FROM * vals[x2] + TO * vals[y2]
        if noise_sigma > 0:
            s = s + noise_sigma * np.sqrt(3.0) * rng.standard_normal(D)   # sigma relative to one term's norm
        t0 = time.perf_counter()
        gv = int(np.argmax(verbs @ (VERB * s)))
        gx = int(np.argmax(vals @ (FROM * s)))
        gy = int(np.argmax(vals @ (TO * s)))
        ms.append(1000 * (time.perf_counter() - t0))
        if superpose_two:
            call1, call2 = (v, x, y), (v2, x2, y2)
            got = (gv, gx, gy)
            ok += got in (call1, call2)
            chim += got not in (call1, call2) and all(g in (a, b) for g, a, b in zip(got, call1, call2))
        else:
            ok += (gv, gx, gy) == (v, x, y)
    r = {"D": D, "trials": trials, "noise_sigma_per_term": noise_sigma, "superpose_two": superpose_two,
         "exact_correct": ok / trials, "ms_mean": round(float(np.mean(ms)), 3)}
    if superpose_two:
        r["chimera_rate"] = chim / trials
    return r


def run_phasor(D, trials, seed=0):
    """The same product in FHRR (unit phasors) -- the algebra that also gives EXACT unbind for role-filler
    sums (unitary HRR is FHRR in the Fourier domain). holographic_phasor.factor (no restarts)."""
    rng = np.random.default_rng(3000 + seed)
    verbs = np.stack([PH.atom("verb:%d" % i, D) for i in range(NV)])
    vals = np.stack([PH.atom("val:%d" % i, D) for i in range(NC)])
    FROM, TO = PH.atom("role:FROM", D), PH.atom("role:TO", D)
    Bf, Bt = vals * FROM, vals * TO
    ok = 0; ms = []
    for t in range(trials):
        v, x, y = int(rng.integers(NV)), int(rng.integers(NC)), int(rng.integers(NC))
        while y == x:
            y = int(rng.integers(NC))
        c = verbs[v] * Bf[x] * Bt[y]
        t0 = time.perf_counter()
        f = PH.factor(c, [verbs, Bf, Bt], iters=100)
        ms.append(1000 * (time.perf_counter() - t0))
        ok += tuple(int(i) for i in f) == (v, x, y)
    return {"D": D, "trials": trials, "exact_correct": ok / trials, "ms_mean": round(float(np.mean(ms)), 1)}


if __name__ == "__main__":
    T0 = time.perf_counter()
    res = {"product": [], "sum": [], "phasor": []}
    for D in (2048, 1024):
        res["product"].append(run_product(D, 30, restarts=20, iters=200))
    res["product"].append(run_product(2048, 30, restarts=20, iters=200, role_bound=False))
    res["product"].append(run_product(2048, 20, restarts=20, iters=200, noise=0.05))
    # refusal cost: a composite that is NOT in the space (random bipolar vector)
    verbs, vals, FROM, TO = books(2048)
    net = ResonatorNetwork([verbs, vals * FROM, vals * TO])
    junk = map_codebook(1, 2048, 999)[0]
    t0 = time.perf_counter(); r = net.factor(junk, restarts=20, iters=200)
    res["product_refusal"] = {"D": 2048, "solved": r["solved"], "ms": round(1000 * (time.perf_counter() - t0), 1)}
    for D in (2048, 1024):
        res["sum"].append(run_sum(D, 500))
        res["sum"].append(run_sum(D, 500, noise_sigma=1.0))
        res["sum"].append(run_sum(D, 500, superpose_two=True))
    for D in (2048, 1024):
        res["phasor"].append(run_phasor(D, 30))
    res["cpu_s"] = round(time.perf_counter() - T0, 1)
    print(json.dumps(res, indent=1))
    json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)

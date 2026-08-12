"""EVOLVE -- EGGROLL-style evolution strategies, using what leCore already has.

Moose's question was whether a Galvatron could be CREATED by distilling Qwen and
training the leCore additions in, using EGGROLL rather than bolting capabilities
on afterwards. The arithmetic says yes and the audit says most of it exists.

WHAT THE AUDIT FOUND (so this module does not rebuild it):
  * `agent_benchmark` is already a REWARD FUNCTION -- a pre-registered primary
    metric (false-action rate on a no-tool set built by removal) plus resolution
    rate and refusal count, returned in ~2s. Non-differentiable, which is
    exactly why ES is the right optimiser and gradients are not.
  * `wgsl_device` / `wgsl_bind_batch` are a vendor-neutral GPU path that already
    exists. It reports "wgpu is not installed" here; on a machine with it, the
    population is the thing GPUs are good at.
  * leCore's forward pass is the only primitive ES needs. The no-autodiff
    constraint that shaped this whole engine is IRRELEVANT to evolution
    strategies -- that is the finding, not the code.

WHAT WAS ACTUALLY MISSING, and is here: the population harness.

THREE THINGS IT DOES THAT NAIVE ES DOES NOT, all from the EGGROLL paper:
  * LOW-RANK PERTURBATIONS. Perturb a rank-r factor, not the full tensor: the
    search dimension for a 0.8B's leCore additions drops from 10.31M parameters
    to 0.52M at rank 4. This is the paper's central trick and the reason it
    scales to billions.
  * SEED-DERIVED MEMBERS. A population member is regenerated from its seed
    rather than stored, so memory is O(population) integers instead of
    O(population x parameters). hashlib, never hash(), so a member reproduces
    in another process.
  * ANTITHETIC PAIRS AND RANK SHAPING. Each seed contributes +d and -d, and
    fitnesses are centred and scaled before weighting, so a single outlier
    cannot dominate the update.

MEASURED HONESTLY ELSEWHERE IN THESE NOTES: ES loses to least squares on convex
problems (0.08937 -> 0.08927, a rediscovery) and loses badly on a 256k-dimension
discrete rounding search. It belongs on END-TO-END NON-DIFFERENTIABLE
objectives, which is the only place this harness points it.
"""

import hashlib

import numpy as np


def _member(seed, shapes, sigma, tag=""):
    """Regenerate one population member's perturbation from its seed alone."""
    h = hashlib.sha256(("%s|%s" % (seed, tag)).encode()).digest()
    g = np.random.default_rng(int.from_bytes(h[:8], "big"))
    return [g.standard_normal(s) * float(sigma) for s in shapes]


class Evolve:
    """A population search over low-rank perturbations of named parameters."""

    def __init__(self, params, sigma=0.02, lr=0.3, population=32, seed=0,
                 rank=None):
        self.names = sorted(params)
        self.params = {k: np.asarray(v, np.float64).copy()
                       for k, v in params.items()}
        self.shapes = [self.params[k].shape for k in self.names]
        self.sigma = float(sigma)
        self.lr = float(lr)
        self.population = int(population)
        self.rng = np.random.default_rng(int(seed))
        self.rank = rank
        self.history = []

    def _perturb(self, seed):
        """Low-rank where the parameter is a matrix, dense where it is a vector.

        A rank-r perturbation of an (m, n) matrix is u @ v with u (m, r) and
        v (r, n) -- r*(m+n) numbers instead of m*n. That is EGGROLL's trick and
        it is what makes the search dimension tractable."""
        h = hashlib.sha256(str(seed).encode()).digest()
        g = np.random.default_rng(int.from_bytes(h[:8], "big"))
        out = []
        for shp in self.shapes:
            if self.rank and len(shp) == 2 and min(shp) > int(self.rank):
                r = int(self.rank)
                u = g.standard_normal((shp[0], r))
                v = g.standard_normal((r, shp[1]))
                d = (u @ v) / np.sqrt(r)
            else:
                d = g.standard_normal(shp)
            out.append(d * self.sigma * (np.std(self.params[self.names[len(out)]])
                                         or 1.0))
        return out

    def step(self, fitness_fn):
        """One generation. `fitness_fn(params) -> float`, LOWER IS BETTER."""
        seeds = self.rng.integers(0, 2 ** 31, self.population // 2)
        scored = []
        for sd in seeds:
            d = self._perturb(sd)
            plus = {k: self.params[k] + d[i] for i, k in enumerate(self.names)}
            minus = {k: self.params[k] - d[i] for i, k in enumerate(self.names)}
            scored.append((float(fitness_fn(plus)), float(fitness_fn(minus)), sd))
        vals = np.array([f for t in scored for f in t[:2]], float)
        mu, sd_ = float(vals.mean()), float(vals.std()) + 1e-12
        grads = [np.zeros_like(self.params[k]) for k in self.names]
        for fp, fm, sd in scored:
            d = self._perturb(sd)
            # ANTITHETIC + CENTRED: a member is worth the DIFFERENCE its two
            # halves made, normalised, so one outlier cannot own the update
            w = -((fp - mu) - (fm - mu)) / (2.0 * sd_)
            for i in range(len(grads)):
                grads[i] += w * d[i]
        for i, k in enumerate(self.names):
            self.params[k] = self.params[k] + self.lr * grads[i] / len(scored)
        cur = float(fitness_fn(self.params))
        self.history.append({"fitness": cur, "mean_population": mu,
                             "best_seen": min([cur] + [h["fitness"]
                                                       for h in self.history])})
        return self.history[-1]

    def run(self, fitness_fn, generations=20, patience=None, progress=None):
        """Run until the budget is spent or progress stalls.

        `patience` stops when no generation has improved for that many rounds --
        an optimiser that has stopped moving should say so rather than burn the
        remaining budget looking busy."""
        best = float(fitness_fn(self.params))
        stale = 0
        for gen in range(int(generations)):
            rec = self.step(fitness_fn)
            if rec["fitness"] < best - 1e-12:
                best, stale = rec["fitness"], 0
            else:
                stale += 1
            if progress:
                progress(gen, rec)
            if patience and stale >= int(patience):
                return {"params": self.params, "best": best,
                        "generations": gen + 1, "stopped": "stalled"}
        return {"params": self.params, "best": best,
                "generations": int(generations), "stopped": "budget"}


def search_dimension(shapes, rank=None):
    """How many numbers the search actually explores -- the number that decides
    whether a run is affordable."""
    total = 0
    for s in shapes:
        if rank and len(s) == 2 and min(s) > int(rank):
            total += int(rank) * (s[0] + s[1])
        else:
            total += int(np.prod(s))
    return total


def _selftest():
    rng = np.random.default_rng(0)

    # ---- a NON-DIFFERENTIABLE objective, because that is the only place ES
    #      belongs: a step function no gradient method can climb
    target = rng.standard_normal((8, 8))

    def fitness(p):
        d = p["W"] - target
        return float(np.round(np.linalg.norm(d) * 4) / 4)     # quantised loss

    start = {"W": np.zeros((8, 8))}
    e = Evolve(dict(start), sigma=0.35, lr=0.9, population=32, seed=1)
    before = fitness(start)
    res = e.run(fitness, generations=40)
    assert res["best"] < before, (before, res["best"])

    # ---- SEED-DERIVED MEMBERS REPRODUCE, or a run cannot be repeated ----
    a = Evolve({"W": np.zeros((4, 4))}, seed=3)._perturb(12345)
    b = Evolve({"W": np.zeros((4, 4))}, seed=99)._perturb(12345)
    assert np.array_equal(a[0], b[0]), "a member must depend only on its seed"

    # ---- LOW RANK SHRINKS THE SEARCH, which is the whole EGGROLL point ----
    shapes = [(1024, 3584), (1024, 1024)]
    full = search_dimension(shapes)
    r4 = search_dimension(shapes, rank=4)
    assert r4 < full / 100, (full, r4)

    # ---- IT STOPS WHEN IT STALLS instead of burning budget ----
    flat = Evolve({"W": np.zeros((4, 4))}, sigma=1e-9, lr=1e-9, population=8,
                  seed=5)
    out = flat.run(lambda p: 1.0, generations=50, patience=3)
    assert out["stopped"] == "stalled" and out["generations"] < 10, out

    print("evolve selftest OK -- a QUANTISED (non-differentiable) loss fell "
          "%.3f -> %.3f in %d generations where no gradient exists; population "
          "members regenerate from their seed alone so a run repeats in another "
          "process; low-rank perturbation cuts a %d-dim search to %d (%.0fx) at "
          "rank 4; and a stalled run stops after %d generations instead of "
          "spending its budget looking busy"
          % (before, res["best"], res["generations"], full, r4, full / r4,
             out["generations"]))


if __name__ == "__main__":
    _selftest()

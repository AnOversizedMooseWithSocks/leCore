"""Scaling diagnosis: detect WHICH limit a workload is hitting, and apply the right lever.

WHY THIS MODULE EXISTS
----------------------
The engine's standing observation (Moose's rule): whenever we hit a "limit," it is
almost always because SCALING is needed -- and the right response depends on which
kind of limit it is. The house diagnostic has lived only in prose until now:

    "double dim -- if error drops you are variance-limited; if not, raise the
     margin (the problem is structural, not statistical)."

This module generalises that rule from one knob (dim) to ANY set of knobs, and
makes it a measured procedure instead of folklore:

    A limit is DIAGNOSED by which knob's doubling reduces the error.
    A WALL is when no knob does.

Given an evaluation function err = eval_fn(**knobs) and a set of scaling knobs
(dimension, tile count, bits, grid resolution, samples, accumulators, ...),
`diagnose_scaling` doubles each knob in isolation, measures the error response,
and returns the levers RANKED by measured responsiveness -- with the honest
"structural wall: no knob helps, this is a bad approach not a scaling problem"
verdict when that is what the data says. `auto_scale` then closes the loop:
greedily double the most responsive knob until the target error is met, the
evaluation budget runs out, or a wall is declared.

The knob names deliberately mirror the engine's five levers, so a diagnosis maps
straight onto a known fix:
    dim / width      -> more dimensions (capacity, SNR ~ sqrt(D))
    tiles / buckets  -> tile the domain under an orchestrator (capacity per tile)
    bits / levels    -> precision (quantisation / accumulator width)
    res / samples    -> resolution / sampling density (bake grids, AA)
Anything can be a knob though -- the diagnosis is purely empirical, which is what
makes it GENERAL: it never needs to know what the knob means, only whether turning
it moves the measured error.

KEPT NEGATIVES (loud)
---------------------
  * The diagnosis is LOCAL and FIRST-ORDER: it doubles each knob once, from the
    current operating point. A knob that only pays after 4x, or two knobs that
    only pay together (interaction), will read unresponsive. auto_scale's repeated
    rounds recover the first case (each round re-probes from the new point); joint
    interactions are out of scope and stated, not silently mis-ranked.
  * Responsiveness is measured against the eval function the CALLER supplies. A
    noisy eval (nondeterministic, seed-varying) can fake or hide a response; the
    engine's rule applies -- make eval_fn deterministic (fixed seeds) or average
    it yourself before asking for a diagnosis. We measure; we cannot launder a
    noisy instrument.
  * A wall verdict means "none of THESE knobs helps at this point," not
    "mathematically impossible." Walls-are-bad-approaches remains the house
    posture: the verdict includes the probe table so the caller can see what was
    tried and go find a different approach (the five-levers walk), and files the
    two cases separately, exactly as the ledger discipline requires.
  * Cost is tracked but not optimised: auto_scale minimises error subject to a
    doubling budget, it does not solve the error-vs-cost Pareto. If eval_fn
    returns a cost we report it per step so the caller can stop when cost bites;
    choosing the tradeoff is the caller's call, not ours.

Only NumPy + stdlib. Deterministic given a deterministic eval_fn.
"""

import numpy as np

# A knob is "responsive" when doubling it cuts the error by at least this
# fraction. WHY 0.1: below ~10% the drop is routinely within eval noise/plateau
# wiggle for the engine's workloads; a genuine sqrt(D) variance limit gives ~29%
# per doubling (1 - 1/sqrt(2)), comfortably above.
RESPONSIVE_DROP = 0.10


def _eval(eval_fn, knobs):
    """Run one evaluation; normalise the return to (error, cost).

    eval_fn may return a bare float (error) or a dict {'error':..., 'cost':...}.
    WHY normalise here: every caller of the probe loop then handles one shape.
    """
    out = eval_fn(**knobs)
    if isinstance(out, dict):
        return float(out["error"]), float(out.get("cost", 0.0))
    return float(out), 0.0


def diagnose_scaling(eval_fn, knobs, factor=2.0):
    """Probe each knob by scaling it `factor`x in isolation; rank the levers.

    Parameters
    ----------
    eval_fn : callable(**knobs) -> error (float) or {'error':..,'cost':..}
        The workload under diagnosis. MUST be deterministic (fixed seeds) --
        see the module kept-negatives.
    knobs : dict{name: value}
        The current operating point. Every value must be a positive number the
        workload accepts scaled by `factor` (int knobs are rounded).

    Returns dict:
      base_error : float           -- error at the current operating point.
      probes     : list of dicts   -- per knob: scaled value, error, drop
                                      (fractional error reduction), cost.
      ranked     : list of names   -- knobs sorted by measured drop, best first.
      verdict    : 'scale:<knob>'  -- the best knob, when it is responsive;
                   'wall'          -- NO knob is responsive: this operating point
                                      is not scaling-limited by any probed knob.
      responsive : bool

    WHY doubling in isolation: it is the cheapest experiment that separates "this
    resource is the binding constraint" from "this resource is already sufficient"
    -- the same one-factor-at-a-time logic as the engine's dim-doubling rule, now
    applied uniformly to every declared resource.
    """
    base_error, base_cost = _eval(eval_fn, dict(knobs))

    probes = []
    for name, value in knobs.items():
        scaled = dict(knobs)
        newval = value * factor
        scaled[name] = int(round(newval)) if isinstance(value, (int, np.integer)) else newval
        err, cost = _eval(eval_fn, scaled)
        drop = (base_error - err) / base_error if base_error > 0 else 0.0
        probes.append({"knob": name, "scaled_value": scaled[name],
                       "error": err, "drop": float(drop), "cost": cost})

    ranked = [p["knob"] for p in sorted(probes, key=lambda p: -p["drop"])]
    best = max(probes, key=lambda p: p["drop"]) if probes else None
    responsive = bool(best and best["drop"] >= RESPONSIVE_DROP)

    return {"base_error": base_error, "base_cost": base_cost, "probes": probes,
            "ranked": ranked,
            "verdict": ("scale:%s" % best["knob"]) if responsive else "wall",
            "responsive": responsive}


def auto_scale(eval_fn, knobs, target_error, max_rounds=8, factor=2.0):
    """Close the loop: repeatedly double the most responsive knob until done.

    Each round runs a fresh diagnosis FROM THE CURRENT POINT (so a knob that only
    becomes binding later is caught when it does -- the octree/adaptive-record
    pattern, generalised), applies the winning doubling, and records the step.
    Stops when: target_error met (met=True); a WALL is diagnosed (wall=True --
    no knob helps here, scaling is the wrong tool); or max_rounds is spent.

    Returns dict: met, wall, final_error, final_knobs, trajectory (per round:
    the diagnosis verdict, the knob doubled, the error after). The trajectory IS
    the evidence -- every scaling decision arrives with the probe that justified
    it, per the no-win-without-a-baseline rule.
    """
    current = dict(knobs)
    trajectory = []
    err, _ = _eval(eval_fn, current)

    for _ in range(max_rounds):
        if err <= target_error:
            return {"met": True, "wall": False, "final_error": err,
                    "final_knobs": current, "trajectory": trajectory}
        diag = diagnose_scaling(eval_fn, current, factor=factor)
        if not diag["responsive"]:
            return {"met": False, "wall": True, "final_error": err,
                    "final_knobs": current, "trajectory": trajectory,
                    "wall_probes": diag["probes"]}
        best = diag["verdict"].split(":", 1)[1]
        value = current[best]
        newval = value * factor
        current[best] = int(round(newval)) if isinstance(value, (int, np.integer)) else newval
        err, cost = _eval(eval_fn, current)
        trajectory.append({"doubled": best, "knobs": dict(current),
                           "error": err, "cost": cost})

    return {"met": err <= target_error, "wall": False, "final_error": err,
            "final_knobs": current, "trajectory": trajectory}


def _selftest():
    """Assert the contracts on three synthetic limit types + one REAL engine workload.

    1. VARIANCE-LIMITED: err ~ 1/sqrt(dim). Diagnosis picks 'dim'; auto_scale
       reaches the target by doubling dim, with the trajectory as evidence.
    2. CAPACITY-LIMITED: err depends on items-per-tile. Doubling tiles helps,
       doubling dim does not; diagnosis picks 'tiles'.
    3. STRUCTURAL WALL: err constant in every knob. Diagnosis says 'wall';
       auto_scale stops early with wall=True and does NOT burn its budget.
    4. REAL: HRR bundle recall through the engine -- recall error at load falls
       when dim doubles (Plate's sqrt scaling); the diagnostician, fed the real
       workload, picks 'dim'. The prose rule, now executable.
    5. Determinism.
    """
    # (1) variance-limited.
    fn1 = lambda dim, tiles: 1.0 / np.sqrt(dim)
    d1 = diagnose_scaling(fn1, {"dim": 64, "tiles": 4})
    assert d1["verdict"] == "scale:dim", d1["verdict"]
    assert d1["probes"][0]["drop"] > 0.25 or d1["ranked"][0] == "dim"
    r1 = auto_scale(fn1, {"dim": 64, "tiles": 4}, target_error=0.05)
    assert r1["met"] and not r1["wall"], r1
    assert all(step["doubled"] == "dim" for step in r1["trajectory"]), r1
    assert r1["final_error"] <= 0.05

    # (2) capacity-limited: 100 items spread over tiles; error grows with load
    # per tile, independent of dim.
    fn2 = lambda dim, tiles: (100.0 / tiles) / (100.0 / tiles + 10.0)
    d2 = diagnose_scaling(fn2, {"dim": 256, "tiles": 2})
    assert d2["verdict"] == "scale:tiles", d2["verdict"]

    # (3) structural wall: nothing helps.
    fn3 = lambda dim, tiles, bits: 0.42
    d3 = diagnose_scaling(fn3, {"dim": 64, "tiles": 4, "bits": 8})
    assert d3["verdict"] == "wall" and not d3["responsive"], d3
    r3 = auto_scale(fn3, {"dim": 64, "tiles": 4, "bits": 8}, target_error=0.1,
                    max_rounds=8)
    assert r3["wall"] and not r3["met"], r3
    assert len(r3["trajectory"]) == 0          # stopped before wasting doublings
    assert "wall_probes" in r3                  # the evidence table travels

    # (4) REAL workload: HRR bundle recall error vs dim, through the engine.
    from holographic.agents_and_reasoning.holographic_ai import (
        random_vector, bundle)

    def bundle_recall_error(dim, n_items=40, n_distractors=40, seed=0):
        # WHY this workload: the engine's canonical variance-limited case. NOTE the
        # first draft measured mean member-cosine, which is ~1/sqrt(N) regardless
        # of dim -- and the diagnostician correctly called it a WALL. The quantity
        # that scales with dim is DISCRIMINABILITY: crosstalk variance falls as
        # sqrt(D), so members separate from distractors as dim grows. Error =
        # the overlap between the member-cosine and distractor-cosine populations
        # (fraction of distractors scoring above the weakest member).
        dim = int(dim)
        items = [random_vector(dim, np.random.default_rng(seed + 1 + i))
                 for i in range(n_items)]
        b = bundle(items)
        nb = np.linalg.norm(b)
        mcos = np.array([float(np.dot(b, it)) / (nb * np.linalg.norm(it))
                         for it in items])
        dis = [random_vector(dim, np.random.default_rng(seed + 1000 + i))
               for i in range(n_distractors)]
        dcos = np.array([float(np.dot(b, d)) / (nb * np.linalg.norm(d))
                         for d in dis])
        # overlap: how often a distractor beats the weakest genuine member --
        # the retrieval-failure driver, and the thing sqrt(D) actually improves.
        return float(np.mean(dcos > np.min(mcos)))

    d4 = diagnose_scaling(bundle_recall_error, {"dim": 128})
    assert d4["verdict"] == "scale:dim", d4
    assert d4["probes"][0]["drop"] > 0.1, d4["probes"]

    # (5) determinism.
    a = diagnose_scaling(fn2, {"dim": 256, "tiles": 2})
    b = diagnose_scaling(fn2, {"dim": 256, "tiles": 2})
    assert a == b

    print("holographic_scalinglaw selftest OK (real bundle workload: doubling dim "
          "drops recall error %.1f%% -- the prose rule, now executable)"
          % (100 * d4["probes"][0]["drop"]))


if __name__ == "__main__":
    _selftest()

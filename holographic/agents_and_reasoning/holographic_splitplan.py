"""SPLITPLAN -- dispatch work PROPORTIONALLY across contending paths instead of picking one winner.

WHY, and the idea is not mine. FreeToken (arXiv 2608.16157, MIT/UC Berkeley) serves an MoE expert miss
two ways -- copy the expert over PCIe and run on the GPU, or run it on the CPU where it already lives --
and its central observation is that **both read the same system memory, so they COMPETE for one pool
rather than adding to each other**. Existing engines pick one strategy at load time and freeze it;
FreeToken measures both bandwidths on the actual machine and splits each step's misses in proportion.

`compute_plan` PICKS: it returns one tier (memo / cpu-numpy / zig / gpu) and a reason. That is right when
one path dominates and wrong whenever two paths are comparable, because the loser's capacity sits idle.

THIS IS THE SAME REASONING leCore ALREADY REACHED FROM THE OTHER SIDE. `machine_spec_sheet`'s founding
negative is that a latency-ordered hierarchy is the wrong frame -- "none of these are scalar units; every
one is a BATCH unit whose per-access cost collapses with N". Units that contend for one resource do not
form a ladder. Proportional split is that finding applied to dispatch.

ADDITIVE BY CONSTRUCTION: `argmax` of the returned weights is exactly the tier a picker would choose, so
an existing caller reading a winner is unaffected."""


def split_plan(paths, contended=True):
    """Weights over `paths`, proportional to measured throughput, honest about contention.

    `paths` is [{"name", "throughput": items/sec, "shares_bus": bool}, ...] -- throughput MEASURED, never
    a spec-sheet number. FreeToken's finding is precisely that two machines with the same GPU can want
    opposite strategies and none of it is readable off a spec sheet.

    `contended=True` (default) treats paths marked `shares_bus` as drawing on ONE pool: their combined
    contribution is capped at the fastest of them rather than summed, because adding the bandwidths of
    two readers of the same memory is the error the whole idea exists to avoid.

    Returns {weights, winner, split_gain, contended}. `split_gain` is the fraction of extra throughput
    the split buys over the single best path -- and it is 0.0 when one path dominates, which is the
    honest answer and the case where a picker was right all along."""
    ps = [p for p in (paths or []) if float(p.get("throughput", 0.0)) > 0.0]
    if not ps:
        return {"weights": {}, "winner": None, "split_gain": 0.0, "contended": bool(contended),
                "why": "no path reports positive measured throughput"}
    if len(ps) == 1:
        return {"weights": {ps[0]["name"]: 1.0}, "winner": ps[0]["name"], "split_gain": 0.0,
                "contended": bool(contended), "why": "only one measured path"}

    best = max(ps, key=lambda p: float(p["throughput"]))
    solo = float(best["throughput"])

    # EFFECTIVE throughput under contention: bus-sharing paths cannot be summed. Capping their group at
    # its fastest member is deliberately CONSERVATIVE -- the true figure needs a measured interference
    # curve, and over-claiming a split's benefit is exactly how a "win" becomes a regression in production.
    shared = [p for p in ps if p.get("shares_bus")]
    solo_paths = [p for p in ps if not p.get("shares_bus")]
    if contended and shared:
        total = max(float(p["throughput"]) for p in shared) + sum(float(p["throughput"]) for p in solo_paths)
    else:
        total = sum(float(p["throughput"]) for p in ps)

    w = {p["name"]: float(p["throughput"]) / sum(float(q["throughput"]) for q in ps) for p in ps}
    gain = max(0.0, (total - solo) / solo) if solo > 0 else 0.0
    return {"weights": w, "winner": best["name"], "split_gain": round(gain, 4),
            "contended": bool(contended), "effective_throughput": total, "best_solo": solo,
            "why": ("split buys %.1f%% over %s alone" % (100 * gain, best["name"])) if gain > 0.01
                   else ("%s dominates -- a split buys nothing, pick it" % best["name"])}


def _selftest():
    # 1. ONE DOMINANT PATH -> the split must buy ~nothing and say so. A dispatcher that always splits is
    #    as wrong as one that always picks; the honest answer here is "you were right to pick".
    r = split_plan([{"name": "gpu", "throughput": 100.0}, {"name": "cpu", "throughput": 1.0}])
    assert r["winner"] == "gpu" and r["split_gain"] < 0.02, r
    assert "dominates" in r["why"]

    # 2. TWO COMPARABLE INDEPENDENT PATHS -> a real gain, and weights in proportion.
    r2 = split_plan([{"name": "a", "throughput": 50.0}, {"name": "b", "throughput": 50.0}])
    assert abs(r2["weights"]["a"] - 0.5) < 1e-9 and r2["split_gain"] > 0.9, r2

    # 3. THE FREETOKEN POINT, and the reason this module exists: two paths that SHARE A BUS do not add.
    #    Summing them would claim a 2x that the hardware cannot deliver.
    shared = split_plan([{"name": "pcie", "throughput": 50.0, "shares_bus": True},
                         {"name": "cpu", "throughput": 50.0, "shares_bus": True}])
    assert shared["split_gain"] < 0.02, ("bus-sharing paths must not be summed", shared)
    assert shared["effective_throughput"] == 50.0, shared
    #    ...and turning contention OFF must restore the naive sum, so the flag is doing the work and the
    #    difference between the two is visible rather than buried.
    naive = split_plan([{"name": "pcie", "throughput": 50.0, "shares_bus": True},
                        {"name": "cpu", "throughput": 50.0, "shares_bus": True}], contended=False)
    assert naive["split_gain"] > 0.9, naive

    # 4. ADDITIVE: argmax of the weights is the tier a PICKER would have chosen, so a caller reading a
    #    winner is unaffected by this existing.
    for case in ([{"name": "x", "throughput": 3.0}, {"name": "y", "throughput": 7.0}],
                 [{"name": "x", "throughput": 9.0}, {"name": "y", "throughput": 2.0}]):
        rr = split_plan(case)
        assert max(rr["weights"], key=rr["weights"].get) == rr["winner"]

    # 5. REFUSES rather than guesses when nothing is measured -- an unmeasured device is NAMED blocked,
    #    which is compute_plan's own standing rule.
    assert split_plan([])["winner"] is None
    assert split_plan([{"name": "z", "throughput": 0.0}])["winner"] is None
    print("splitplan selftest OK -- a dominant path says PICK ME; equal independent paths split 50/50 "
          "for +100%%; bus-SHARING paths cap at the fastest (the FreeToken point) instead of summing; "
          "argmax(weights) == the picker's winner; unmeasured refuses")


if __name__ == "__main__":
    _selftest()

"""holographic_merge.py -- reconcile forked worlds (multiplayer <-> single-player), conflict-free where they agree.

A shared world is a SEED + DELTAS: everyone regenerates the same base locally, and only the sparse changes travel. So
forking is cheap (take the shared seed, keep your own delta layer), and MERGING is reconciling the deltas. Two forks
edited the same slot? If they AGREE, merge is conflict-free. If they DISAGREE, we don't guess -- we detect it (the
opponent decomposition) and hand the real conflict to a policy or a human.

WHY PAIRWISE (matching leOS): the opponent channels are a two-sided (A vs B) decomposition, and leOS's own cross-user
merge (project/rendering/multi_user.merge_displacements) compares each PAIR of users. So this reconciles a slot by its
pairwise divergences: forks agree on a slot when every pair of them is within `tol`; otherwise it's a conflict.

    res = merge_forks([mine, theirs], policy="select")   # each fork: {slot: vector}
    mind.apply(res["merged"])                             # auto-merge the agreements
    hand_to_user(res["conflicts"])                        # resolve the real conflicts by choice

Reuses holographic_opponent (divergence + blend). numpy/stdlib only; deterministic.
"""
import numpy as np

from holographic.rendering.holographic_opponent import opponent_channels


def merge_forks(forks, policy="select", tol=0.2):
    """Reconcile several forks, each a {slot: vector} delta layer. For every slot, if only one fork touched it, keep
    it; if several did and they AGREE (every pair within `tol` radians of geodesic divergence), merge conflict-free
    into the consensus; if they DISAGREE, resolve by `policy`:

        'select'  (default) -- surface the conflict for a human: it goes into `conflicts`, not `merged`.
        'auto'              -- keep only the agreements: a disagreeing slot is left out of `merged` (and `conflicts`).
        'left' / 'right'    -- the first / last fork wins that slot.
        callable(slot, vals)-- your own per-slot resolver returns the merged value.

    Returns {"merged": {slot: vector}, "conflicts": [(slot, [vals...]), ...]}. `tol` is on divergence_score (radians):
    0.2 rad ~ 11 degrees ~ cosine 0.98, i.e. "essentially the same edit."
    """
    merged, conflicts = {}, []
    all_slots = {s for f in forks for s in f}
    for slot in sorted(all_slots, key=str):
        vals = [f[slot] for f in forks if slot in f]
        if len(vals) == 1:
            merged[slot] = vals[0]                              # only one fork changed it -> no conflict possible
            continue
        if _all_agree(vals, tol):
            merged[slot] = _consensus(vals)                    # every pair agrees -> conflict-free consensus
        elif policy == "left":
            merged[slot] = vals[0]
        elif policy == "right":
            merged[slot] = vals[-1]
        elif callable(policy):
            merged[slot] = policy(slot, vals)
        elif policy == "auto":
            pass                                               # keep only agreements: drop the disagreeing slot
        else:                                                  # 'select' -- surface it for a human
            conflicts.append((slot, vals))
    return {"merged": merged, "conflicts": conflicts}


def _all_agree(vals, tol):
    """True iff every PAIR of the values is within `tol` radians of divergence (the leOS pairwise convention)."""
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if opponent_channels(vals[i], vals[j])["divergence_score"] > tol:
                return False
    return True


def _consensus(vals):
    """The agreed value when forks agree: the normalized superposition (mean) of the values -- SYMMETRIC, so no fork
    is favored. (Deliberately not opponent.blend, which is an asymmetric 70/30 cross-model mix for a different job;
    a democratic merge of agreeing edits wants their average.)"""
    total = np.sum([np.asarray(v, float) for v in vals], axis=0)
    n = np.linalg.norm(total)
    return total / n if n > 0 else total


def _selftest():
    rng = np.random.default_rng(0)
    dim = 512
    base = rng.standard_normal(dim); base /= np.linalg.norm(base)

    # --- a slot only one fork touched: kept as-is ---
    only = merge_forks([{"a": base}, {}])
    assert "a" in only["merged"] and not only["conflicts"]

    # --- two forks AGREE on a slot (near-identical edits): conflict-free consensus, no conflict ---
    v1 = base + 0.003 * rng.standard_normal(dim)
    v2 = base + 0.003 * rng.standard_normal(dim)
    agree = merge_forks([{"pos": v1}, {"pos": v2}], policy="select")
    assert "pos" in agree["merged"] and not agree["conflicts"]
    # the merged consensus is close to the base both edited
    mc = agree["merged"]["pos"]
    assert float(np.dot(mc, base) / (np.linalg.norm(mc) * np.linalg.norm(base))) > 0.9

    # --- two forks DISAGREE (very different edits): surfaced as a conflict under 'select' ---
    w1 = rng.standard_normal(dim)
    w2 = rng.standard_normal(dim)
    sel = merge_forks([{"col": w1}, {"col": w2}], policy="select")
    assert not sel["merged"] and len(sel["conflicts"]) == 1 and sel["conflicts"][0][0] == "col"

    # --- policies on a disagreeing slot ---
    assert np.allclose(merge_forks([{"c": w1}, {"c": w2}], policy="left")["merged"]["c"], w1)
    assert np.allclose(merge_forks([{"c": w1}, {"c": w2}], policy="right")["merged"]["c"], w2)
    picked = merge_forks([{"c": w1}, {"c": w2}], policy=lambda slot, vals: vals[0] * 0.0)
    assert np.allclose(picked["merged"]["c"], 0.0)             # callable resolver ran
    auto = merge_forks([{"c": w1}, {"c": w2}], policy="auto")
    assert not auto["merged"] and not auto["conflicts"]        # auto keeps only agreements -> nothing here

    # --- three forks, two agree + one strays: they don't ALL agree, so it's a conflict (pairwise, per leOS) ---
    three = merge_forks([{"x": v1}, {"x": v2}, {"x": w1}], policy="select")
    assert three["conflicts"] and three["conflicts"][0][0] == "x"

    # --- three forks that ALL agree: conflict-free consensus ---
    v3 = base + 0.003 * rng.standard_normal(dim)
    allok = merge_forks([{"x": v1}, {"x": v2}, {"x": v3}], policy="select")
    assert "x" in allok["merged"] and not allok["conflicts"]

    print("OK: holographic_merge self-test passed (single-fork slot kept; agreeing forks -> conflict-free consensus; "
          "disagreeing forks surfaced under 'select'; left/right/callable/auto policies; N forks reconciled pairwise "
          "per the leOS convention -- all agree -> merge, one strays -> conflict)")


if __name__ == "__main__":
    _selftest()

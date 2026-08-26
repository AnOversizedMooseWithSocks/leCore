"""holographic_lossspace.py -- E1: WHERE the losses live. Not "how much did I lose" (any sum answers that)
but the SHAPE of the loss: how concentrated in the tail, how clustered in time, which named conditions carry
it -- each axis judged against the null that erases only the structure under test.

WHY THIS MODULE EXISTS
----------------------
The campaign's post-mortems kept finding the same thing late: the aggregate looked survivable while the
losses were secretly ONE thing wearing many events -- a single regime carrying most of the damage, a
dependence structure that turned individually-tolerable losses into streaks, a tail so heavy the mean was a
comfort blanket. Each of those has a cheap measurement and an obvious null; this module runs all of them at
once so the shape is on the record BEFORE the aggregate is quoted.

THE THREE AXES, each with its own null:
  TAIL          share of total loss carried by the worst 5% of events, judged against a GAUSSIAN of the same
                mean and std (the comparison a reader silently assumes). Heavier than Gaussian is the "your
                mean is a comfort blanket" flag.
  TIME          longest losing streak and loss-day clustering, judged against the PERMUTATION null (same
                losses, shuffled order): dependence in time is exactly what a shuffle erases and nothing
                else. A streak z >> 0 means losses arrive together -- sizing that assumed independence is
                wrong in the direction that ruins you.
  CONDITION     per named condition mask: the loss share vs the occupancy share, ratio and shift-null z (the
                event-study's circular shift, reused: it preserves the mask's own run structure). A condition
                holding 10% of the time and carrying 60% of the loss is the finding; the aggregate hid it.

This is the LOSS-side sibling of insurance_profile (which asks where the VALUE concentrates); the two
docstrings point at each other and the pair answers "should I gate?" from both directions.

NumPy + stdlib only, deterministic given seed.
"""

import numpy as np


def loss_space_report(values, conditions=None, tail_frac=0.05, n_null=400, seed=0):
    """The report. `values` are per-event signed outcomes (negative = loss). `conditions` is an optional
    dict name -> boolean mask aligned with values.

    Returns:
      totals        {sum, n, n_loss, loss_sum} (loss_sum <= 0)
      tail          {share, gaussian_share, ratio, verdict} -- worst tail_frac of events' share of total loss
      time          {longest_streak, null_mean, z, verdict} -- vs the permutation null
      conditions    per name: {occupancy, loss_share, ratio, z, verdict} -- vs the circular-shift null
      verdict       one line naming the dominant axis, or the honest "unremarkable shape"

    KEPT NEGATIVES: (1) the tail comparison is against a GAUSSIAN because that is the silent default
    assumption being tested, not because Gaussian is a plausible model -- a fat-tailed but HONEST process
    also flags, and the flag means "size for the tail you actually have", not "bug". (2) the shift null for
    conditions assumes the value series is exchangeable under rotation -- the same caveat as event_study;
    difference a trending series first. (3) with fewer than ~10 loss events every axis is starved and says
    so rather than reporting a z from nothing."""
    v = np.asarray(values, float).ravel()
    n = v.size
    if n < 40:
        raise ValueError("need at least 40 events to characterise a loss shape (got %d)" % n)
    losses = v < 0
    n_loss = int(losses.sum())
    loss_sum = float(v[losses].sum())
    out = {"totals": {"sum": float(v.sum()), "n": n, "n_loss": n_loss, "loss_sum": loss_sum}}
    rng = np.random.default_rng(seed)

    if n_loss < 10:
        out["tail"] = out["time"] = {"verdict": "only %d loss events -- too few to characterise a shape; "
                                                "that scarcity is the report" % n_loss}
        out["conditions"] = {}
        out["verdict"] = out["tail"]["verdict"]
        return out

    # TAIL: worst tail_frac of ALL events, their share of total loss.
    k = max(int(np.ceil(n * float(tail_frac))), 1)
    worst = np.sort(v)[:k]
    share = float(worst.sum() / loss_sum) if loss_sum < 0 else 0.0
    g_shares = np.empty(n_null)
    mu, sd = v.mean(), v.std()
    for i in range(n_null):
        g = rng.normal(mu, sd, n)
        gl = g[g < 0].sum()
        g_shares[i] = (np.sort(g)[:k].sum() / gl) if gl < 0 else 0.0
    g_share = float(g_shares.mean())
    ratio = share / g_share if g_share > 0 else float("inf")
    tail_sd = float(g_shares.std()) or 1.0
    tail_flag = share > g_share + 2 * tail_sd
    out["tail"] = {"share": share, "gaussian_share": g_share, "ratio": float(ratio),
                   "verdict": ("worst %.0f%% of events carry %.0f%% of the loss vs %.0f%% under a Gaussian "
                               "of the same mean/std (%.1fx) -- the mean is a comfort blanket; size for THIS "
                               "tail" % (100 * tail_frac, 100 * share, 100 * g_share, ratio)) if tail_flag
                   else ("tail share %.0f%% vs Gaussian %.0f%% -- unremarkable" % (100 * share, 100 * g_share))}

    # TIME: longest losing streak vs the permutation null (same multiset of outcomes, order erased).
    def longest_streak(mask):
        best = cur = 0
        for m in mask:
            cur = cur + 1 if m else 0
            best = max(best, cur)
        return best
    streak = longest_streak(losses)
    null_streaks = np.empty(n_null)
    for i in range(n_null):
        null_streaks[i] = longest_streak(rng.permutation(losses))
    smu, ssd = float(null_streaks.mean()), float(null_streaks.std()) or 1.0
    sz = (streak - smu) / ssd
    out["time"] = {"longest_streak": int(streak), "null_mean": smu, "z": float(sz),
                   "verdict": ("losses arrive in streaks (longest %d vs %.1f shuffled, z=%.1f) -- "
                               "independence-based sizing is wrong in the ruinous direction"
                               % (streak, smu, sz)) if sz > 2 else
                              ("streaks consistent with shuffled order (longest %d vs %.1f)" % (streak, smu))}

    # CONDITIONS: loss share vs occupancy, shift null preserving the mask's run structure.
    conds = {}
    for name, mask in (conditions or {}).items():
        m = np.asarray(mask, bool).ravel()
        if m.size != n:
            raise ValueError("condition %r has %d flags for %d events" % (name, m.size, n))
        occ = float(m.mean())
        ls = float(v[m & losses].sum() / loss_sum) if loss_sum < 0 else 0.0
        null_ls = np.empty(n_null)
        for i in range(n_null):
            mm = np.roll(m, int(rng.integers(1, n)))
            null_ls[i] = v[mm & losses].sum() / loss_sum if loss_sum < 0 else 0.0
        cmu, csd = float(null_ls.mean()), float(null_ls.std()) or 1.0
        cz = (ls - cmu) / csd
        conds[name] = {"occupancy": occ, "loss_share": ls, "ratio": float(ls / occ) if occ > 0 else float("inf"),
                       "z": float(cz),
                       "verdict": ("%r holds %.0f%% of the time and carries %.0f%% of the loss (%.1fx, "
                                   "z=%.1f) -- the gate candidate" % (name, 100 * occ, 100 * ls, ls / max(occ, 1e-12), cz))
                       if (cz > 2 and ls > occ) else
                       ("%r: loss share %.0f%% vs occupancy %.0f%% -- proportionate" % (name, 100 * ls, 100 * occ))}
    out["conditions"] = conds

    flags = []
    if tail_flag:
        flags.append("TAIL %.1fx Gaussian" % ratio)
    if sz > 2:
        flags.append("TIME streak z=%.1f" % sz)
    flags += ["CONDITION %s %.1fx" % (nm, c["ratio"]) for nm, c in conds.items()
              if c["z"] > 2 and c["loss_share"] > c["occupancy"]]
    out["verdict"] = ("loss shape flags: " + "; ".join(flags)) if flags else \
        "unremarkable loss shape on every measured axis -- the aggregate is an honest summary here"
    return out


def _selftest():
    """Contracts, one plant per axis plus the all-clear:
    1. TAIL: a mixture with rare huge losses flags vs Gaussian; a plain Gaussian does not.
    2. TIME: losses forced into runs flag the streak z; the shuffled same losses do not.
    3. CONDITION: a mask holding ~15% of the time carrying most of the loss flags with the gate-candidate
       verdict; an irrelevant mask reads proportionate.
    4. The all-clear fixture reads 'unremarkable' on every axis -- the report can say nothing is wrong.
    5. Too-few losses is a scarcity report, not a z; refusals name their reason.
    """
    rng = np.random.default_rng(0)
    n = 2000

    # (4) all-clear first: plain Gaussian, no conditions.
    v_plain = rng.normal(0.05, 1.0, n)
    r_plain = loss_space_report(v_plain, seed=1)
    assert "unremarkable loss shape" in r_plain["verdict"], r_plain["verdict"]

    # (1) tail: 2% of events drawn from a x8 loss scale.
    v_tail = rng.normal(0.05, 1.0, n)
    idx = rng.choice(n, n // 50, replace=False)
    v_tail[idx] = -np.abs(rng.normal(0, 8.0, idx.size))
    r_tail = loss_space_report(v_tail, seed=2)
    assert r_tail["tail"]["ratio"] > 1.3, r_tail["tail"]   # 1.45 measured: the Gaussian comparator inherits the planted tail via its matched std, damping the ratio -- the flag still fires on the 2-sigma line
    assert "comfort blanket" in r_tail["tail"]["verdict"]

    # (2) time: the SAME multiset of outcomes, losses gathered into blocks -- then its own shuffle clears.
    v_runs = np.abs(rng.normal(1.0, 0.3, n))
    block = np.zeros(n, bool)
    for s in range(100, n - 60, 400):
        block[s:s + 60] = True
    v_runs[block] *= -1.0
    r_runs = loss_space_report(v_runs, seed=3)
    assert r_runs["time"]["z"] > 2, r_runs["time"]
    r_shuf = loss_space_report(rng.permutation(v_runs), seed=3)
    assert r_shuf["time"]["z"] < 2, r_shuf["time"]

    # (3) conditions: storm mask carries the losses; moon-phase mask is noise.
    v_cond = np.abs(rng.normal(0.8, 0.2, n))
    storm = np.zeros(n, bool)
    for s in range(150, n - 80, 500):
        storm[s:s + 75] = True
    v_cond[storm] = rng.normal(-1.5, 1.0, int(storm.sum()))
    moon = rng.random(n) < 0.3
    r_cond = loss_space_report(v_cond, conditions={"storm": storm, "moon": moon}, seed=4)
    assert r_cond["conditions"]["storm"]["z"] > 2 and "gate candidate" in r_cond["conditions"]["storm"]["verdict"]
    assert "proportionate" in r_cond["conditions"]["moon"]["verdict"], r_cond["conditions"]["moon"]
    assert "CONDITION storm" in r_cond["verdict"]

    # (5) scarcity + refusals
    r_few = loss_space_report(np.abs(rng.normal(1, 0.1, 100)) * np.where(np.arange(100) < 3, -1, 1), seed=5)
    assert "too few" in r_few["verdict"]
    for bad, needle in ((lambda: loss_space_report(rng.normal(0, 1, 10)), "at least 40"),
                        (lambda: loss_space_report(v_plain, conditions={"bad": [True] * 5}), "flags for")):
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert needle in str(e)

    print("holographic_lossspace selftest OK (plain Gaussian reads 'unremarkable' on every axis; planted "
          "rare-huge-loss tail reads %.1fx the Gaussian share with the comfort-blanket verdict; blocked "
          "losses read streak z=%.1f and the SAME outcomes shuffled read z=%.1f; a 15%%-occupancy storm "
          "carrying the losses reads %.1fx z=%.1f as the gate candidate while the moon mask reads "
          "proportionate; 3 losses in 100 events is a scarcity report, not a z)"
          % (r_tail["tail"]["ratio"], r_runs["time"]["z"], r_shuf["time"]["z"],
             r_cond["conditions"]["storm"]["ratio"], r_cond["conditions"]["storm"]["z"]))


if __name__ == "__main__":
    _selftest()

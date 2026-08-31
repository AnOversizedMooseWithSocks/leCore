"""ACTR -- NOOA's memory ranking, computed by the ladder that is already installed.

Moose asked whether we install any of the NOOA machinery. The repo already holds
an honest competitive note (docs/COMPETITIVE_NOOA.md, checked against
arXiv:2607.20709) listing six NOOA capabilities. FIVE ARE HARNESS FEATURES --
pass-by-reference previews, code-as-action in a persistent REPL, typed return
validation, sandboxed execution, event history -- and none of those live in
weights. They are things a runner does.

THE SIXTH IS DIFFERENT AND IT IS THE ONE WITH A NUMBER: a long-term memory
subsystem with ACT-R ACTIVATION RANKING and DECAY-BASED FORGETTING, measured at
+11.8 RHAE POINTS over the same agent with markdown notes. leCore was marked
PARTIAL there -- `recall` exists, the curation and decay do not.

AND IT TURNS OUT WE HAD ALREADY INSTALLED THE HARD PART WITHOUT NAMING IT.
ACT-R's base-level activation is A = ln(sum_j t_j^-d) with d about 0.5 -- A
POWER LAW over how long ago each use was. The HRNN ladder is a sum of
EXPONENTIALS at GEOMETRIC half-lives. A geometric sum of exponentials
approximates a power law, which is a known result, and measured here against
t^-0.5 over five decades:
    2 rungs   max rel err 0.2236   R^2 0.85055
    4 rungs   max rel err 0.0515   R^2 0.99282
    6 rungs   max rel err 0.0401   R^2 0.99891
    8 rungs   max rel err 0.0423   R^2 0.99873
FOUR RUNGS ALREADY GIVE R^2 0.993, and four rungs is what install_lecore puts in
by default. So the ladder is ACT-R base-level activation IN THE WEIGHTS, rather
than in a SQLite file beside the agent.

WHAT THIS MODULE ADDS is the RANKING that reads it -- activation from recency
and frequency, a retrieval threshold that ABSTAINS rather than returning the
least-bad item, and decay-based forgetting that follows from the same numbers.

WHAT IT DOES NOT CLAIM: NOOA's +11.8 was measured on RHAE with a full agent
loop. Nothing here reproduces that benchmark, and leCore still has no result on
any external agentic benchmark -- which the competitive note already says
plainly. The claim here is that the MECHANISM is present and correct, not that
the outcome is reproduced.
"""

import numpy as np

#: ACT-R's decay exponent. 0.5 is the value the literature settles on and the
#: one the power-law fit above was measured against.
DECAY_D = 0.5


def base_level(use_times, now, d=DECAY_D, floor=1e-9):
    """ACT-R base-level activation: A = ln(sum_j (now - t_j)^-d).

    RECENCY AND FREQUENCY IN ONE NUMBER -- each past use contributes a decaying
    term, so an item used often and recently outranks one used once long ago,
    without either being tracked separately."""
    t = np.asarray(use_times, np.float64)
    age = np.maximum(float(now) - t, float(floor))
    return float(np.log(np.sum(age ** (-float(d)))))


def ladder_activation(use_times, now, half_lives, weights=None):
    """The same quantity, computed as the LADDER computes it.

    This is what an installed HRNN ladder already holds: a sum of exponential
    accumulators at geometric half-lives. Given the rung half-lives the model
    was installed with, the activation is a weighted read of those rungs -- no
    external log of use times required at inference, because the state IS the
    log."""
    t = np.asarray(use_times, np.float64)
    hl = np.asarray(half_lives, np.float64)
    age = np.maximum(float(now) - t, 1e-9)
    per_rung = np.array([np.sum(np.exp(-age / h)) for h in hl])
    w = np.ones(len(hl)) if weights is None else np.asarray(weights, np.float64)
    return float(np.sum(w * per_rung))


def fit_rung_weights(half_lives, d=DECAY_D, span=(1.0, 1e5), n=60):
    """Weights making the ladder match ACT-R's power law. Closed form, no tuning.

    Least squares over log-spaced ages -- the ladder's half-lives are fixed by
    the install, so the only free thing is how much each rung contributes."""
    t = np.logspace(np.log10(span[0]), np.log10(span[1]), int(n))
    B = np.stack([np.exp(-t / float(h)) for h in half_lives], 1)
    w, *_ = np.linalg.lstsq(B, t ** (-float(d)), rcond=None)
    approx = B @ w
    ref = t ** (-float(d))
    return w, {"max_rel_err": float(np.max(np.abs(approx - ref)) / np.max(ref)),
               "r2": float(1 - np.sum((approx - ref) ** 2)
                           / np.sum((ref - ref.mean()) ** 2))}


def rank(items, now, half_lives=None, threshold=None, d=DECAY_D,
         weights=None):
    """Rank memories by activation, ABSTAINING below a retrieval threshold.

    ACT-R has a retrieval threshold and so does this: an item whose activation
    falls below it is NOT RETRIEVED, rather than returned as the least-bad
    option. That is the same discipline as decide_or_abstain and the same reason
    -- a confident wrong memory costs more than a missing one."""
    # THE RUNG WEIGHTS ARE NOT OPTIONAL. Reading the ladder with UNIT weights
    # over-counts the long rungs, because every rung contributes ~1 for an item
    # of any age below its half-life. Measured: one recent use ranked BELOW two
    # old ones, which inverts the whole point of a recency-weighted memory.
    # fit_rung_weights exists for exactly this and I had computed the weights
    # and then not passed them.
    if half_lives is not None and weights is None:
        weights, _rep = fit_rung_weights(half_lives, d=d)
    scored = []
    for name, uses in dict(items).items():
        a = (base_level(uses, now, d=d) if half_lives is None
             else np.log(max(ladder_activation(uses, now, half_lives,
                                               weights=weights), 1e-300)))
        scored.append((name, float(a)))
    scored.sort(key=lambda kv: -kv[1])
    if threshold is None:
        return scored
    kept = [(n, a) for n, a in scored if a >= float(threshold)]
    return kept


def forget(items, now, threshold, d=DECAY_D):
    """Which items have decayed below the retrieval threshold and can be dropped."""
    return [n for n, uses in dict(items).items()
            if base_level(uses, now, d=d) < float(threshold)]


def _selftest():
    # ---- THE LADDER MUST APPROXIMATE THE POWER LAW, or the claim is empty ----
    hl4 = np.geomspace(2, 1e5, 4)
    _w4, r4 = fit_rung_weights(hl4)
    hl2 = np.geomspace(2, 1e5, 2)
    _w2, r2 = fit_rung_weights(hl2)
    assert r4["r2"] > 0.99, r4
    # and MORE rungs must be BETTER, or the geometric spacing is not the reason
    assert r4["r2"] > r2["r2"], (r4, r2)

    # ---- RECENCY AND FREQUENCY MUST BOTH RAISE ACTIVATION ----
    now = 1000.0
    once_old = base_level([10.0], now)
    once_recent = base_level([990.0], now)
    often_old = base_level([10.0, 20.0, 30.0, 40.0], now)
    assert once_recent > once_old, (once_recent, once_old)
    assert often_old > once_old, (often_old, once_old)

    # ---- THE THRESHOLD MUST ABSTAIN, not return the least-bad item ----
    items = {"fresh": [995.0, 998.0], "stale": [3.0]}
    all_ranked = rank(items, now)
    assert all_ranked[0][0] == "fresh", all_ranked
    kept = rank(items, now, threshold=all_ranked[0][1] - 0.5)
    assert [n for n, _ in kept] == ["fresh"], kept
    assert forget(items, now, threshold=all_ranked[0][1] - 0.5) == ["stale"]

    # ---- AND THE LADDER RANKING MUST AGREE WITH THE POWER-LAW RANKING ----
    many = {"a": [999.0], "b": [500.0, 600.0], "c": [5.0]}
    p_order = [n for n, _ in rank(many, now)]
    l_order = [n for n, _ in rank(many, now, half_lives=hl4)]
    assert p_order == l_order, (p_order, l_order)
    # ---- AND UNIT WEIGHTS MUST GET IT WRONG, or the fit is decoration ----
    bad_order = [n for n, _ in rank(many, now, half_lives=hl4,
                                    weights=np.ones(len(hl4)))]
    assert bad_order != p_order, ("unit weights should mis-rank -- if they do "
                                  "not, fit_rung_weights is doing nothing",
                                  bad_order)

    print("actr selftest OK -- NOOA's memory subsystem is the one of its six "
          "capabilities that is not a harness feature, and the HRNN ladder "
          "ALREADY COMPUTES IT: a geometric sum of exponentials matches ACT-R's "
          "t^-0.5 power law at R^2 %.5f with FOUR rungs (against %.5f with two), "
          "which is the default install. Activation rises with both recency and "
          "frequency, the retrieval threshold ABSTAINS rather than returning the "
          "least-bad item, and the ladder ranking agrees with the power-law "
          "ranking item for item -- but ONLY with the fitted rung weights; unit "
          "weights over-count the long rungs and rank one recent use BELOW two "
          "old ones, which the selftest now pins as a negative"
          % (r4["r2"], r2["r2"]))


if __name__ == "__main__":
    _selftest()

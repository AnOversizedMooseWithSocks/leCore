"""D1 probe (cross-cutting: SAMPLE-1 low-discrepancy -> creature exploration). KEPT NEGATIVE.

THE PROPOSAL (Togelius's seat, with the caveat already on record): SAMPLE-1's low-discrepancy sampling gives
tighter, more even coverage than i.i.d. random when you SAMPLE points in a space. So -- the reasoning went --
drive the creature's exploration from a low-discrepancy sequence instead of epsilon-greedy random, to cover the
state space faster.

THE MEASURED ANSWER: it does NOT work -- it actively HURTS, and the reason is structural. SAMPLE-1's win is for
placing each sample INDEPENDENTLY in the space. A creature does not sample the state space; it WALKS it, and a
walk ACCUMULATES displacement. A low-discrepancy sequence over the actions is BALANCED (north/south/east/west
spread evenly in time), so the steps CANCEL and the agent stays pinned near its start. A random walk, by
contrast, has runs and imbalances -- and that imbalance is exactly the diffusive DRIFT that carries it across
the space. Measured on an open grid (pure coverage, no walls), 400 steps: random walk visits ~162 distinct
cells; low-discrepancy action selection visits ~12 (it barely leaves home). An order of magnitude WORSE.

THE LESSON (and why it is worth keeping): this pins down precisely WHY a transfer that pays for direct sampling
fails for sequential exploration. Low discrepancy MINIMISES the imbalance of a point set; spatial exploration
NEEDS the imbalance, because displacement is the cumulative sum of the steps and a balanced sum is ~zero. The
two goals are opposed. The honest below-stack finding the sweep predicted (Togelius's caveat: over a handful of
discrete actions it "buys almost nothing") is, if anything, stronger than predicted -- it is not neutral, it is
harmful. The real coverage lever for a sequential agent is count-based / novelty-driven exploration (go where
you have been least), a piece of which the brain's existing novelty_bonus already provides.

No faculty, no tour line -- the finding is the negative.
"""

import numpy as np

_MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]                          # N, S, E, W


def coverage(strategy, T=400, seed=0, size=60):
    """Distinct cells visited in T steps on an open grid from the centre, under an exploration `strategy`:
    'random' (i.i.d. uniform actions) or 'ld' (a low-discrepancy sequence mapped to the four actions)."""
    from holographic.sampling_and_signal.holographic_lowdiscrepancy import low_discrepancy
    rng = np.random.default_rng(seed)
    pos = np.array([size // 2, size // 2])
    seen = {tuple(pos)}
    ld = low_discrepancy(T, 1, seed=seed).ravel() if strategy == "ld" else None
    for t in range(T):
        a = int(ld[t] * 4) % 4 if strategy == "ld" else int(rng.integers(4))
        dx, dy = _MOVES[a]
        pos = np.clip(pos + [dx, dy], 0, size - 1)
        seen.add(tuple(pos))
    return len(seen)


def _selftest():
    """CI-fast: records the D1 negative. Low-discrepancy action selection covers FAR FEWER distinct cells than
    epsilon-random exploration on an open grid -- because a walk accumulates displacement and a BALANCED
    (low-discrepancy) action sequence cancels it, while random's imbalance is the diffusive drift that explores.
    The transfer that pays for direct sampling (SAMPLE-1) is harmful for sequential exploration."""
    rand = np.mean([coverage("random", seed=s) for s in range(12)])
    ld = np.mean([coverage("ld", seed=s) for s in range(12)])
    assert ld < rand * 0.5, (ld, rand)                              # low-discrepancy covers FAR less -- the negative
    assert rand > 80, rand                                          # sanity: the random walk does drift and explore


if __name__ == "__main__":
    _selftest()
    print("holographic_ldexplore D1 negative selftest passed (low-discrepancy steps cancel; random drift explores)")

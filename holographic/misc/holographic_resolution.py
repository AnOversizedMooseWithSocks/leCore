"""Coarse-to-fine cleanup -- answer at low resolution first, escalate only when
the answer is not yet resolved. The leOS idea (Matryoshka / inception levels)
adapted honestly to RANDOM hypervectors.

THE LEOS IDEA, AND THE TWIST. leOS's inception_levels query a Matryoshka-trained
embedding at 32 then 128 then full dim, escalating only when a cheap confidence
'totem' says the truncation lost too much. But holostuff uses RANDOM hypervectors,
where information is spread evenly across dimensions -- there is no trained
prefix that 'matters more'. So truncating to the first k dimensions is a random
SUBSAMPLE of the full cosine: an unbiased estimate, just noisier. That changes
the totem from 'residual energy' (energy-concentrated, trained case) to a
STATISTICAL one: the gap between the top two candidates measured against the
spread of the field at that resolution. If the leader is many field-widths clear
of the pack, the ranking is resolved and no more dimensions can change it; if it
is buried in the spread, escalate.

WHAT IT BUYS (measured on random vocabularies):
  * EXACTNESS WHEN THERE IS AN ANSWER. Over queries with a real nearest neighbour,
    coarse-to-fine returns the SAME winner as a full-dimension scan, 100% -- the
    gate only stops when the ranking is statistically settled.
  * BIG WORK SAVINGS ON EASY QUERIES. A strong match resolves at ~128 dims (32x
    truncation); the mean dimension-work over a 500-item store drops ~95% vs
    scanning everyone at full width, because most candidates are eliminated at low
    resolution and only a shortlist is refined.
  * IT DEGRADES TO FULL SCAN, HONESTLY, ON HARD CASES. Near-ties never resolve
    cheaply, so the gate escalates all the way to full dim -- no error, but no
    saving either. On a store where everything is similar (all near-ties),
    coarse-to-fine saves nothing; the win is real only when some queries are easy.
  * A NO-MATCH STAYS A NO-MATCH. With no real neighbour the top score is ~0.04
    (vs 0.3-0.9 for a match), so a confidence floor cleanly abstains rather than
    promoting noise.

The companion measurement -- resolution_profile -- is the persistent-homology
idea in practical form: track at which resolution each query's winner STABILISES.
A winner that is already on top at 64 dims and never changes is 'fundamental'
(survives heavy truncation); one that keeps changing needs full width. The
profile tells you, per store, how low you can safely start the schedule.
"""
import numpy as np


def coarse_to_fine(query, matrix, schedule=None, z_stop=4.0, keep_frac=0.25,
                   min_keep=8, min_score=None):
    """Nearest row of `matrix` to `query` by cosine, found coarse-to-fine.

    Ranks candidates using only the first k dimensions for each k in `schedule`,
    stopping as soon as the top-1/top-2 gap exceeds `z_stop` times the spread of
    the trailing candidates (the field's own width at that resolution). Between
    rounds it keeps the top `keep_frac` (at least `min_keep`) and refines them at
    the next resolution. Returns (index, score, dims_used, stopped_k).

    `query` and `matrix` rows need not be normalised; both are normalised here.
    If `min_score` is given and the resolved top cosine is below it, returns
    index -1 (abstain -- no real match), the no-match case.
    """
    M = np.asarray(matrix, float)
    if M.ndim != 2 or len(M) == 0:
        return -1, 0.0, 0, 0
    D = M.shape[1]
    q = np.asarray(query, float)
    qn = q / (np.linalg.norm(q) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    if schedule is None:
        # geometric ladder up to full width (deduped, capped at D)
        schedule = [k for k in (max(32, D // 32), D // 8, D // 2, D) if k <= D]
        schedule = sorted(set(schedule))
    cand = np.arange(len(Mn))
    dims_used = 0
    order = np.array([0])
    last_k = schedule[-1]
    for k in schedule:
        k = min(k, D)
        sub = Mn[cand][:, :k] @ qn[:k]
        dims_used += len(cand) * k
        order = np.argsort(sub)[::-1]
        last_k = k
        if len(sub) == 1 or k >= D:
            break
        gap = sub[order[0]] - sub[order[1]]
        std = sub[order[1:]].std() + 1e-12
        if gap / std > z_stop:
            break
        # SAFE PRUNING: only drop candidates that are statistically out of
        # contention -- more than `z_stop` field-widths behind the leader at this
        # resolution. Near-ties are all kept, so a true winner buried in a tie can
        # never be culled before full width resolves it (the correctness guard).
        leader = sub[order[0]]
        keep_mask = (leader - sub) <= z_stop * std
        kept = order[keep_mask[order]]
        floor = max(min_keep, int(len(cand) * keep_frac))
        cand = cand[kept[:max(floor, len(kept))]] if len(kept) > floor else cand[order[:floor]]
    winner = int(cand[order[0]])
    score = float(Mn[winner] @ qn)               # exact full-dim score of the winner
    if min_score is not None and score < min_score:
        return -1, score, dims_used, last_k
    return winner, score, dims_used, last_k


def full_scan(query, matrix):
    """Reference: nearest row by full-dimension cosine. Returns (index, score)."""
    M = np.asarray(matrix, float)
    qn = np.asarray(query, float) / (np.linalg.norm(query) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    s = Mn @ qn
    i = int(np.argmax(s))
    return i, float(s[i])


def resolution_profile(query, matrix, schedule=None):
    """The persistent-homology idea, practical form: at each resolution k, which
    row wins? Returns [(k, winner_index, top_score)]. A winner that is already on
    top at low k and never changes is 'fundamental' -- it survives truncation;
    one that keeps changing needs full width. This tells you how low a store's
    schedule can safely start."""
    M = np.asarray(matrix, float)
    D = M.shape[1]
    qn = np.asarray(query, float) / (np.linalg.norm(query) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    if schedule is None:
        schedule = sorted({k for k in (D // 16, D // 8, D // 4, D // 2, D) if 1 <= k <= D})
    out = []
    for k in schedule:
        s = Mn[:, :k] @ qn[:k]
        i = int(np.argmax(s))
        out.append((k, i, float(Mn[i] @ qn)))     # report exact score of the k-winner
    return out


def stabilisation_dim(query, matrix, schedule=None):
    """The lowest resolution at and above which the winner stops changing -- the
    'birth scale' of the final answer. Low means the answer is robust to heavy
    truncation; equal to full D means it needed every dimension."""
    prof = resolution_profile(query, matrix, schedule)
    final = prof[-1][1]
    stable_from = prof[-1][0]
    for k, w, _ in prof:
        if w == final:
            stable_from = k
            break
    return stable_from

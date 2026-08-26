"""holographic_refine.py -- the pipeline middle: produce a result, have a CRITIC judge it, adjust, retry.

WHY THIS EXISTS
---------------
leCore often sits between a big compute and a checker: generate something, score it, nudge it, try again until it is
good enough or a budget runs out. That is the shape of an optimisation loop, an analysis-by-synthesis loop, a
"draft then revise" loop -- and of a human-in-the-loop review. This is that loop, with every piece a plain callable
so you can drop in whatever produces, whatever judges (a small model, a leCore metric, the `opponent` agreement
score, or a person), and whatever adjusts.

    result = produce()                    # make a first attempt
    score  = critique(result)             # 0..1 : how good is it?  (higher = better)
    while score < accept and tries left:
        result = adjust(result, score)    # nudge it, informed by the score
        score  = critique(result)

RELATION TO project_onto_constraints (honest, not a forced merge)
-----------------------------------------------------------------
The engine already has `project_onto_constraints` (holographic_denoise): iterate a fixed set of vector projections
until they converge by a numeric tolerance. That is the SAME "iterate until good" spirit, but its acceptance test is
a fixed convergence tolerance on a vector, and its step is a geometric projection. `refine` is the sibling with a
CALLABLE critic and a CALLABLE adjust -- the general case where "good" is whatever your critic says and "adjust" is
whatever you supply. They are two acceptance modes of one idea; kept as two readable functions rather than one
over-general engine, because the mechanisms (numeric projection vs. arbitrary produce/critique/adjust) really differ.

Pure stdlib/numpy-free control flow; whatever your callables do is your business.
"""


def refine(produce, critique, adjust, accept=0.9, budget=8, attempts=1, backoff=0.1):
    """Produce a result, critique it, and adjust-and-retry until the critic's score reaches `accept` or `budget`
    attempts are spent.

      produce()          -> a result (called once, to make the first attempt).
      critique(result)   -> a score, higher = better (0..1 is the natural range; `accept` is on the same scale).
      adjust(result, s)  -> a new, hopefully-better result, given the last result and its score `s`.
      accept             -> stop as soon as the score reaches this.
      budget             -> at most this many adjust/critique rounds after the first attempt.

    Returns {"result", "score", "accepted", "tries"} where `tries` is how many adjust rounds were used (0 if the
    first attempt already passed)."""
    # P9 -- `attempts>1` runs each production through `hardening.retrying` (bounded retry with backoff). A critic
    # loop that talks to a flaky node, a subprocess or a model endpoint should not die on the first transient
    # error; the retry belongs to the hardening home, not re-rolled here. attempts=1 (default) is a plain call,
    # so nothing existing changes.
    _produce, _adjust = produce, adjust
    if attempts > 1:
        from holographic.misc.holographic_hardening import retrying
        _produce = lambda: retrying(produce, attempts=attempts, backoff=backoff)
        _adjust = lambda r, s_: retrying(lambda: adjust(r, s_), attempts=attempts, backoff=backoff)

    result = _produce()
    score = float(critique(result))
    tries = 0
    while score < accept and tries < budget:
        result = _adjust(result, score)
        score = float(critique(result))
        tries += 1
    return {"result": result, "score": score, "accepted": score >= accept, "tries": tries}


def opponent_critic(reference):
    """A ready-made critic: score a candidate by how much it AGREES with a reference (or a set of references), using
    the opponent decomposition's cosine_similarity in [-1,1] -- so it plugs straight into `refine` as `critique`.
    Handy when 'good' means 'consistent with what we already trust'."""
    from holographic.rendering.holographic_opponent import opponent_channels
    import numpy as np

    refs = np.asarray(reference, float)
    refs = refs[None, :] if refs.ndim == 1 else refs

    def critique(candidate):
        cand = np.asarray(candidate, float)
        sims = [opponent_channels(r, cand)["cosine_similarity"] for r in refs]   # agreement with each reference
        return float(np.mean(sims))

    return critique


def _selftest():
    # --- converges to accept: adjust always halves the error, so a few rounds cross the threshold ---
    target = 100.0
    state = {"value": 0.0}
    log = refine(
        produce=lambda: state["value"],
        critique=lambda v: 1.0 - abs(target - v) / target,          # 1.0 when v == target
        adjust=lambda v, s: v + (target - v) * 0.5,                 # move halfway to the target each time
        accept=0.95, budget=10,
    )
    assert log["accepted"] and log["score"] >= 0.95, log
    assert 0 < log["tries"] <= 10

    # --- already good on the first try: zero adjust rounds ---
    log0 = refine(produce=lambda: target, critique=lambda v: 1.0, adjust=lambda v, s: v, accept=0.9, budget=5)
    assert log0["accepted"] and log0["tries"] == 0

    # --- budget runs out without accepting: honest 'not accepted', tries == budget ---
    stuck = refine(produce=lambda: 0.0, critique=lambda v: 0.1, adjust=lambda v, s: v, accept=0.9, budget=4)
    assert not stuck["accepted"] and stuck["tries"] == 4

    # --- the opponent_critic: refine a noisy vector toward a reference by blending it in ---
    import numpy as np
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(256); ref /= np.linalg.norm(ref)
    crit = opponent_critic(ref)
    start = ref + 1.5 * rng.standard_normal(256)                    # a noisy version of the reference
    out = refine(
        produce=lambda: start,
        critique=crit,
        adjust=lambda v, s: 0.5 * (v / np.linalg.norm(v)) + 0.5 * ref,   # blend toward the reference
        accept=0.9, budget=12,
    )
    assert out["accepted"], out["score"]

    print("OK: holographic_refine self-test passed (halving loop converges past accept in a few tries; an "
          "already-good result uses 0 tries; a stuck loop honestly reports not-accepted at budget; opponent_critic "
          "drives a noisy vector to agree with its reference)")


if __name__ == "__main__":
    _selftest()

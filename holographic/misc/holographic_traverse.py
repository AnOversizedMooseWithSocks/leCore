"""Throughput-gated traversal -- Russian roulette for holographic paths.

WHY THIS EXISTS
---------------
In the FFT/phasor domain a `bind` is elementwise complex MULTIPLICATION, so a chain of binds is a running
PRODUCT of per-step transfer functions -- exactly a ray's THROUGHPUT, accumulating multiplicatively. A
holographic traversal -- a multi-hop associative recall, the resonator's iterative peeling, a recursive
scene descent -- is therefore a ray bouncing through the space, and its recoverable signal ATTENUATES the
same way: once the throughput has decayed, every further step is noise.

Path tracers solve this with Russian roulette: terminate a path once its throughput is negligible. This
module ports that -- drive a traversal with a cheap running confidence (a cleanup cosine, a convergence
margin) and STOP when it falls below a floor. The payoff measured on the substrate: the cheap confidence
tracks the true recoverable signal almost exactly while it lasts, so gating on it stops EXACTLY when the ray
goes dark -- without ground truth, at lower average cost than a fixed-depth traversal.

MEASURED (see `_selftest`)
  * On a directed linked list stored in superposition, the gate recovers every valid hop and then abstains
    the moment the chain is exhausted (the recoverable signal gone) -- the correct prefix at a fraction of a
    fixed depth's steps.

SCOPE / KEPT NEGATIVE
  * The gate keys on LOW confidence (the ray dark). It does NOT catch a CONFIDENT-but-WRONG step -- the
    capacity-ambiguity regime where crosstalk hands back a wrong atom at moderate confidence. That is a
    CALIBRATION problem (the calibrated-null / MIS items), not a throughput one.
  * This is deterministic FLOOR termination, which is right for FOLLOWING a path. The unbiased STOCHASTIC
    Russian roulette (terminate with prob 1-T, boost survivors by 1/T) is for ACCUMULATING a sum and is a
    separate, not-yet-measured extension.
"""

from collections import namedtuple

import numpy as np

TraversalResult = namedtuple("TraversalResult", "payloads throughputs steps stopped final_throughput")


def gated_traverse(step, start, floor=0.15, max_steps=64, min_steps=1):
    """Drive an iterative holographic traversal with a throughput gate (a Russian-roulette stop).

    `step(state) -> (next_state, throughput, payload)` advances one bounce: `next_state` is what to continue
    from, `throughput` in [0, 1] is a CHEAP confidence in this step (a cleanup cosine, a convergence
    margin), and `payload` is whatever the step yields (the recalled atom, the decoded node). `step` may
    return None to signal a natural end.

    The traversal runs until the throughput falls below `floor` -- the ray has gone dark, so that step is
    ABSTAINED (its payload is NOT recorded) -- or `next_state` is None, or `max_steps` is reached. At least
    `min_steps` steps are always taken before the gate can fire (so a single noisy first step can't stop it).

    Returns TraversalResult(payloads, throughputs, steps, stopped, final_throughput); `stopped` is one of
    'floor' / 'natural_end' / 'max_steps'. The point: it stops EXACTLY when the recoverable signal is gone,
    without ground truth, at lower average cost than running a fixed depth. NOTE: `floor` is on whatever
    scale the step reports as throughput -- tune it to your confidence measure."""
    state = start
    payloads, throughputs = [], []
    stopped, final_t = "max_steps", 1.0
    for _ in range(max_steps):
        out = step(state)
        if out is None:                                   # the step itself signals a natural end
            stopped = "natural_end"
            break
        next_state, tput, payload = out
        tput = float(tput)
        final_t = tput
        if len(payloads) >= min_steps and tput < floor:   # the ray has gone dark -- abstain on this step
            stopped = "floor"
            break
        payloads.append(payload)
        throughputs.append(tput)
        if next_state is None:                            # produced a payload but no continuation
            stopped = "natural_end"
            break
        state = next_state
    return TraversalResult(payloads, throughputs, len(payloads), stopped, final_t)


def _selftest():
    """CI-fast: prove (1) the gate's LOGIC -- it stops at the floor, recovers the good prefix, reports why;
    and (2) it works on a REAL holographic traversal -- a directed linked list in superposition where the
    gate recovers every valid hop and abstains the instant the chain is exhausted, at a fraction of a fixed
    depth's steps."""
    # (1) gating logic on a KNOWN throughput profile (high for three steps, then below the floor)
    profile = [0.90, 0.85, 0.88, 0.12, 0.40, 0.30]
    def step(k):
        return None if k >= len(profile) else (k + 1, profile[k], f"item{k}")
    r = gated_traverse(step, 0, floor=0.20, max_steps=10, min_steps=1)
    assert r.stopped == "floor", r.stopped
    assert r.payloads == ["item0", "item1", "item2"], r.payloads   # the three above floor; abstains on 0.12
    assert r.steps == 3 and r.final_throughput < 0.20
    # a step returning None ends cleanly (no false floor)
    r2 = gated_traverse(lambda k: None if k >= 4 else (k + 1, 0.9, k), 0, floor=0.20, max_steps=10)
    assert r2.stopped == "natural_end" and r2.steps == 4

    # (2) a REAL holographic traversal: a directed linked list stored in superposition
    from holographic.agents_and_reasoning.holographic_ai import bind, involution
    rng = np.random.default_rng(0)
    D, L = 8192, 10
    def unit():
        v = rng.standard_normal(D)
        return v / np.linalg.norm(v)
    perm = rng.permutation(D)
    inv = np.argsort(perm)                                          # a fixed permutation = the DIRECTION role
    chain = [unit() for _ in range(L + 1)]                          # nodes 0..L
    cb = np.array(chain + [unit() for _ in range(10)])             # then distractors
    cbn = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    M = np.zeros(D)
    for i in range(L):
        M = M + bind(chain[i], chain[i + 1][perm])                 # directed link: the next node is permuted

    def rstep(cur):
        probe = bind(M, involution(cur))[inv]                      # unbind, then undo the direction permutation
        cs = cbn @ (probe / (np.linalg.norm(probe) + 1e-12))
        j = int(np.argmax(cs))
        return (cb[j], cs[j], j)                                   # next state, throughput (cleanup cos), payload

    g = gated_traverse(rstep, chain[0], floor=0.20, max_steps=30, min_steps=1)
    assert g.payloads == list(range(1, L + 1)), g.payloads          # recovered every chain node, in order
    assert g.stopped == "floor"                                    # stopped when the chain ran out (signal gone)
    assert g.steps == L and g.steps < 30                           # all L hops, far fewer than the fixed depth
    assert all(t >= 0.20 for t in g.throughputs) and g.final_throughput < 0.20  # valid hops lit, the stop dark


if __name__ == "__main__":
    _selftest()
    print("holographic_traverse selftest passed")

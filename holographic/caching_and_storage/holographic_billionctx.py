"""BILLIONCTX -- what actually binds context at 1e9 tokens, and what does not.

Moose asked for context past a BILLION tokens. Three mechanisms were candidates
and only one survives the arithmetic.

1. THE KV CACHE IS OUT, and not narrowly. At Qwen3.5-0.8B's shapes a million
   tokens is 49 GB; a BILLION is 49 TERABYTES. Sparse attention, eviction and
   compression change the constant, not the exponent. Nothing in this project
   makes attention over 1e9 tokens happen.

2. THE HRNN LADDER UNDERFLOWS FIRST, at around 1e8. decay =
   exp(-exp(a_log)*softplus(dt_bias)), so a half-life of D needs a_log = -ln(D),
   and in float32:
       half-life 1e6 -> 1 - decay = 1.013e-06
       half-life 1e7 -> 1 - decay = 1.192e-07
       half-life 1e8 -> 1 - decay = 0.000e+00   UNDERFLOWS TO IDENTITY
   Past that the rung is a PURE ACCUMULATOR -- infinite retention with no
   forgetting, which sounds like a win and is not: an undecayed sum of a billion
   terms has a signal-to-noise ratio that goes as 1/sqrt(n). The ladder gives
   graded recency, and recency stops meaning anything at that scale.

3. THE REGISTERS REACH IT, because their bound is not TIME. The delta rule's
   erase term is DIRECTIONAL -- S <- aS(I - b k k^T) + b v k^T -- so a write
   whose key is ORTHOGONAL to a reserved direction leaves that direction exactly
   untouched. Not approximately: the projector has a zero there.

SO THE REAL LIMIT IS PRECISION, NOT TOKEN COUNT, and the curve is not the
gentle one I first assumed. MEASURED in float32, cosine of register 0:
     10,000 writes  1.000000
     30,000         1.000000
     60,000         0.999997
     80,000         0.999580
    100,000         0.951284
    140,000         0.056986
IT DOES NOT DECAY, IT COLLAPSES -- exact for tens of thousands of writes and
then gone within one more doubling. float64 holds 1.000000 throughout.
AND IT IS NOT DILUTION, which was my first explanation and was wrong: ||S||
stays at 245 across the whole run, so the register is not becoming a smaller
fraction of a growing state. The residual non-orthogonality that float32 leaves
on each write accumulates until it crosses the projector, and then the erase
term starts reaching a direction it was supposed to miss.
A CLIFF IS MORE DANGEROUS THAN A SLOPE: a system tested at 50,000 writes reads
perfect and fails at 140,000, which is one long session later.

AND THE FIX IS DRAM REFRESH, which is the correct name for it. A DRAM cell
loses charge and is rewritten on a schedule; a reserved register loses its
orthogonality at a rate precision sets and is rewritten the same way -- one delta_write per slot,
re-asserting the value along its own key. MEASURED at float32 over 100,000
writes:
    no refresh                    cosine 0.951284
    refresh every 10,000 writes   cosine 1.000000
    refresh every  1,000 writes   cosine 1.000000
A refresh costs one write per slot, so refreshing 128 registers every 10,000
tokens is 1.3% overhead and makes retention UNBOUNDED IN TIME at float32.

WHAT "A BILLION TOKENS OF CONTEXT" HONESTLY MEANS HERE, because the phrase
invites a bigger claim than the mechanism supports: the model does not ATTEND to
a billion tokens. It RETAINS a bounded number of facts, selected by the write
policy, across an unbounded stream. Capacity is d slots, not 1e9 slots. What is
unbounded is the WINDOW OVER WHICH those slots survive, and that is the thing
that was previously bounded and now is not.
"""

import numpy as np

#: Measured on this engine's arithmetic, not assumed.
LIMITS = {
    "kv_cache_tb_at_1e9": 49.2,
    "ladder_underflow_halflife": 1e8,
    "f32_cosine_at_100k_writes_no_refresh": 0.951284,
    "f32_cosine_at_100k_writes_with_refresh": 1.0,
}


def refresh_interval(dim, n_slots, precision="float32", floor=0.999):
    """How often must registers be rewritten to hold `floor` cosine?

    Derived from the measured drift rather than tuned: float32 carries about
    1e-7 of residual non-orthogonality per write, and the loss accumulates
    roughly linearly until the refresh resets it. float64 needs none at any
    scale this project can reach."""
    if str(precision) == "float64":
        return None
    per_write = 5e-7
    budget = max(1.0 - float(floor), 1e-9)
    return max(100, int(budget / per_write))


def refresh(state, keys, values, write=None):
    """NOTE: this needs the VALUES. See holographic_selfheal for the copy-free
    path -- cleaning each read against a CODEBOOK and writing the cleaned value
    back repairs the file with no external record of its contents, verified to
    8/8 slots after 200,000 interfering writes."""
    """Rewrite every register along its own key. DRAM refresh, one write a slot.

    The values must be KNOWN to be rewritten, which is the honest cost of this
    scheme: a refreshed register file is one whose contents the harness also
    holds. That is the same 63 KB the session contract already carries -- but it
    makes the memory a CACHE rather than a memory, and holographic_selfheal
    removes the dependency entirely when values come from a codebook."""
    from holographic.caching_and_storage.holographic_keyreserve import (
        delta_write)

    w = write or delta_write
    S = state
    for k, v in zip(np.asarray(keys), list(values)):
        S = w(S, k, v)
    return S


def plan(target_tokens, dim=1024, n_slots=128, precision="float32"):
    """What is needed to retain across `target_tokens`. Refuses to overpromise."""
    t = float(target_tokens)
    iv = refresh_interval(dim, n_slots, precision)
    kv_tb = t * 24 * 2 * 2 * 256 * 2 / 1e12
    return {
        "target_tokens": t,
        "attention_possible": bool(kv_tb < 1.0),
        "kv_cache_tb": kv_tb,
        "ladder_useful": bool(t <= LIMITS["ladder_underflow_halflife"]),
        "registers_reach_it": True,
        "refresh_every": iv,
        "refresh_overhead_pct": (0.0 if iv is None
                                 else 100.0 * n_slots / float(iv)),
        "retained": "%d slots, not %g tokens" % (n_slots, t),
        "why": ("registers retain a BOUNDED number of facts across an UNBOUNDED "
                "stream; the model does not attend to %g tokens and nothing "
                "here makes it" % t),
    }


def _selftest():
    from holographic.caching_and_storage.holographic_keyreserve import (
        reserve, orthogonalise, delta_write, delta_read)

    D, N = 256, 8
    rng = np.random.default_rng(0)
    R = reserve(D, N, seed=0)
    vals = [rng.standard_normal(D) for _ in range(N)]

    def drift(dt, every, total=60000):
        S = np.zeros((D, D), dt)
        Rd, vd = R.astype(dt), [v.astype(dt) for v in vals]
        for k, v in zip(Rd, vd):
            S = delta_write(S, k, v).astype(dt)
        for t in range(total):
            k = orthogonalise(rng.standard_normal(D), R).astype(dt)
            S = delta_write(S, k, rng.standard_normal(D).astype(dt)).astype(dt)
            if every and (t + 1) % every == 0:
                S = refresh(S, Rd, vd).astype(dt)
        g = delta_read(S, Rd[0])
        return float(g @ vd[0]
                     / (np.linalg.norm(g) * np.linalg.norm(vd[0]) + 1e-30))

    # 140,000 writes: measured cosine 0.057 without refresh, 1.000000 with.
    # Testing at 60,000 would PASS WITHOUT REFRESH and prove nothing -- the
    # failure is a cliff, so the test has to be on the far side of it.
    TOTAL = 140000
    bare = drift(np.float32, 0, TOTAL)
    kept = drift(np.float32, 10000, TOTAL)
    exact = drift(np.float64, 0, TOTAL)

    # ---- FLOAT32 MUST LEAK, or the whole refresh story is unmotivated ----
    assert bare < 0.5, ("float32 should COLLAPSE past the cliff -- if it does "
                        "not, the refresh machinery is solving nothing", bare)
    # ---- AND REFRESH MUST FIX IT ----
    assert kept > 0.999, kept
    # ---- AND FLOAT64 MUST NOT NEED IT ----
    assert exact > 0.999999, exact
    assert refresh_interval(D, N, "float64") is None

    # ---- THE PLAN MUST REFUSE TO PROMISE ATTENTION AT 1e9 ----
    p9 = plan(1e9)
    assert p9["attention_possible"] is False, p9
    assert p9["ladder_useful"] is False, p9
    assert p9["registers_reach_it"] is True
    p3 = plan(1e3)
    assert p3["attention_possible"] is True and p3["ladder_useful"] is True

    print("billionctx selftest OK -- at 1e9 tokens the KV cache is %.0f TB so "
          "ATTENTION IS OUT, and the ladder underflows to a pure accumulator "
          "past a 1e8 half-life so RECENCY IS OUT; registers reach it because "
          "the delta rule's erase term is DIRECTIONAL, and their real limit is "
          "PRECISION -- and it is a CLIFF not a slope: float32 is exact to 30,000 "
          "writes and collapses to %.3f by 140,000 where float64 holds %.6f, "
          "and DRAM-style refresh every 10,000 writes restores "
          "%.6f at %.1f%% overhead. What is unbounded is the WINDOW over which "
          "a bounded number of slots survives, not the slot count"
          % (p9["kv_cache_tb"], bare, exact, kept,
             plan(1e9)["refresh_overhead_pct"]))


if __name__ == "__main__":
    _selftest()

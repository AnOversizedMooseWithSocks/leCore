"""holographic_tokensample.py -- temperature + nucleus (top-p) sampling over ANY symbol distribution.

WHY THIS MODULE EXISTS (promote + generalize, do NOT reinvent)
-------------------------------------------------------------
The character generator (holographic_text.NGram.generate) already shipped a correct, tested temperature
+ nucleus sampler -- but it was WELDED to a char dict inside one method. The recipe-grammar / transform
work needs the SAME sampler over (opcode, operand) tokens, and PredictiveMemory needs it over its stored
symbols. So the logic is lifted out here into one primitive and the callers delegate to it. Same
operation, different symbol alphabet -- the constitution's "generalize on contact".

(The name `holographic_sampler` was already taken by the SPATIAL scene read-probe -- a different thing
entirely -- so this token sampler lives under its own name to avoid a collision.)

THE MEASURED REASON IT IS NEEDED (kept negative, loud)
------------------------------------------------------
A DETERMINISTIC next-symbol predictor (argmax, or the engine's complete_instruction) LIMIT-CYCLES when
used as a generator. Measured on a transform-recipe corpus: greedy MMD2 0.599 vs a 0.011 sampled arm,
8-gram verbatim-copy 0.091 vs a real rate of 0.006, net rotation 30 vs a real 3.9 -- it locks onto the
single most-probable continuation and loops forever. Sampling the distribution fixes it. So:
GENERATION MUST SAMPLE; PREDICTION MAY ARGMAX. This module is the sampling half.

KEPT NEGATIVES FROM REAL-DATA TESTING (bouncing-body physics traces, loud)
--------------------------------------------------------------------------
  * HEAVY-TAILED ALPHABETS: nucleus and low temperature silently DELETE rare events. On a stream that
    is 98.5% one token, top_p=0.95 trimmed every bounce event out of existence and T=0.6 sharpened past
    them -- the generated physics had ZERO collisions. Rule: top_p must exceed 1 - (rare-event mass);
    on heavy-tailed real streams the safe setting is T=1.0, top_p=1.0. The knobs are for near-uniform
    alphabets.
  * WELL-FORMEDNESS IS THE CALLER'S JOB: this primitive samples one symbol; it cannot know the
    language's syntax. Measured on the recipe grammar: 18% of raw emissions violated run/event
    alternation, merging adjacent runs and destroying the residual autocorrelation (+0.29 -> +0.12).
    Enforcing the alternation constraint in the caller's decode loop recovered it (+0.15 vs a +0.02
    null). Sample the distribution here; enforce the grammar there.
"""
import numpy as np


def sample_from_distribution(dist, temperature=1.0, top_p=1.0, rng=None):
    """Sample one symbol from a {symbol: weight} distribution with temperature and optional nucleus.

    This is the char generator's sampler (holographic_text.NGram.generate) lifted to any alphabet.
    `dist` weights may be probabilities, similarities, or raw counts (non-negative); they are
    temperature-scaled then renormalised. Returns the chosen symbol, or None if the distribution is
    empty or has no positive mass (the caller decides whether to stop or fall back).

    temperature: weight ** (1/T). T -> 0 approaches argmax; T = 1 is the raw distribution; T > 1 flattens.
    top_p < 1.0: nucleus -- keep the smallest set of top symbols whose mass reaches top_p, renormalise,
    sample within it. top_p = 1.0 reproduces plain-temperature EXACTLY, so it is a strict generalisation.

    WHY temperature exponentiates the weight, not a logit: callers hand us frequency/similarity weights,
    not logits, so weight ** (1/T) is the correct sharpening in THIS representation, and it matches the
    character generator bit-for-bit where they overlap. A log/exp round-trip would be equivalent but adds
    two transcendental ops per step for nothing.
    """
    rng = rng or np.random.default_rng(0)
    if not dist:
        return None
    syms = list(dist)
    w = np.clip(np.array([dist[s] for s in syms], float), 0.0, None)
    if w.sum() <= 0:
        return None
    T = max(float(temperature), 1e-6)                # guard tiny T against overflow
    w_sharp = w ** (1.0 / T)
    # WHY this guard: as T -> 0, weight ** (1/T) underflows every entry to 0.0 and the sum vanishes,
    # which would divide to NaN. But T -> 0 MEANS argmax, so that is exactly the answer to fall back to:
    # put all mass on the largest ORIGINAL weight. (The text generator never hit this because it never
    # used near-zero T; the self-test does, and caught it.)
    if not np.isfinite(w_sharp).all() or w_sharp.sum() <= 0:
        p = np.zeros_like(w)
        p[int(np.argmax(w))] = 1.0
    else:
        p = w_sharp / w_sharp.sum()
    if top_p < 1.0:                                  # nucleus: keep only the top_p probability mass
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order])
        keep = order[:max(1, int(np.searchsorted(cum, top_p)) + 1)]
        masked = np.zeros_like(p)
        masked[keep] = p[keep]
        p = masked / masked.sum()
    return syms[int(rng.choice(len(syms), p=p))]


def _selftest():
    """Regression trap: the numeric contract, not just 'no exception'."""
    dist = {"a": 0.7, "b": 0.2, "c": 0.1}

    # (1) T -> 0 is argmax
    rng = np.random.default_rng(0)
    assert set(sample_from_distribution(dist, temperature=1e-4, rng=rng) for _ in range(50)) == {"a"}

    # (2) large T flattens 'a' well below its raw 0.70 monopoly, toward 1/3
    rng = np.random.default_rng(1)
    hot = [sample_from_distribution(dist, temperature=8.0, rng=rng) for _ in range(3000)]
    assert 0.34 < hot.count("a") / len(hot) < 0.50, hot.count("a") / len(hot)

    # (3) nucleus top_p 0.7: 'a' alone reaches 0.7, so 'b','c' are NEVER sampled
    rng = np.random.default_rng(2)
    assert set(sample_from_distribution(dist, 1.0, 0.7, rng) for _ in range(200)) == {"a"}

    # (4) top_p = 1.0 identical to the plain-temperature path under the same rng
    r1 = np.random.default_rng(7); r2 = np.random.default_rng(7)
    assert [sample_from_distribution(dist, 0.8, 1.0, r1) for _ in range(100)] == \
           [sample_from_distribution(dist, 0.8, rng=r2) for _ in range(100)]

    # (5) determinism
    r1 = np.random.default_rng(9); r2 = np.random.default_rng(9)
    assert [sample_from_distribution(dist, 0.6, 0.9, r1) for _ in range(30)] == \
           [sample_from_distribution(dist, 0.6, 0.9, r2) for _ in range(30)]

    # empty / dead distributions return None, never raise
    assert sample_from_distribution({}, rng=np.random.default_rng(0)) is None
    assert sample_from_distribution({"x": 0.0}, rng=np.random.default_rng(0)) is None

    print("OK: holographic_tokensample self-test passed (temperature sharpen/flatten; nucleus trims the "
          "tail and never samples it; top_p=1.0 equals plain-temperature exactly; deterministic; empty->None)")


if __name__ == "__main__":
    _selftest()

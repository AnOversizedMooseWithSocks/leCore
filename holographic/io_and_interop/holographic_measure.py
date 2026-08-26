"""MEASURE -- perplexity with error bars, and decisions that respect them.

Moose asked what assimilation is actually doing. The answer, from his own run:
265 tensors examined over 149 seconds, 18 CHANGED, of which repair reverted 12
as harmful, leaving SIX; original 76.83, assimilated 81.71 (6.4% WORSE),
repaired 75.06 -- reported as "beats the original: True".

Then I measured the measurement. On his real model, from the assessment
bundle's own per-token likelihoods:
    bootstrap 95% CI over 161 positions:  16.90 .. 36.61, i.e. +/-38.5%
    in 40-token chunks the spread is      +/-47.4%
THE 2.3% "WIN" SITS DEEP INSIDE THE NOISE. It is not a small effect, it is an
effect that was never measured. Every gate in this pipeline compared two point
estimates on a few dozen tokens and reported a verdict as if it were a fact.

This module makes that impossible. It returns a perplexity WITH a bootstrap
confidence interval, and `better_than` returns one of BETTER, WORSE or
INDISTINGUISHABLE -- because "indistinguishable" is the honest verdict for most
of what this pipeline has been deciding, and a comparison that cannot say so
will always find a winner.

THE PRACTICAL CONSEQUENCE, and it is uncomfortable: with a 42-token probe
nothing under about 40% is decidable. Either measure on far more tokens, or
stop claiming small wins. Both are fine; pretending is not.
"""

import numpy as np


def measure(runtime, token_ids, resamples=200, alpha=0.05, seed=0):
    """Perplexity AND its uncertainty, from the per-token likelihoods.

    The bootstrap resamples POSITIONS, which is the right unit: perplexity is a
    mean over per-token surprises, and the question is how much that mean would
    move if the probe had been different text of the same kind."""
    ids = list(token_ids)
    if len(ids) < 2:
        raise ValueError("measure needs at least 2 tokens, got %d" % len(ids))
    logits = np.asarray(runtime.forward(ids), np.float64)[:-1]
    targets = np.asarray(ids[1:], np.int64)
    m = logits.max(axis=-1, keepdims=True)
    lse = (np.log(np.exp(logits - m).sum(axis=-1)) + m.ravel())
    nll = lse - logits[np.arange(len(targets)), targets]
    # A BLOCK BOOTSTRAP, BECAUSE TOKENS ARE NOT INDEPENDENT. leCore's
    # `convergence_guard` states the trap outright: a variance interval is right
    # for i.i.d. increments and A LIE for correlated sampling. MEASURED on real
    # per-token surprise, autocorrelation at lags 1..8 is
    # 0.085 0.145 0.008 0.079 0.030 0.052 0.013 0.047, giving an integrated
    # autocorrelation time of 1.91 -- so 1,199 tokens carry the information of
    # 626. Resampling single positions therefore reported intervals about 45%
    # TOO NARROW (half-width 10.5% against 15.2% at block 32), and every
    # confidence interval this arc quoted was overconfident by that much.
    # The block length is derived from the measured tau rather than picked.
    rng = np.random.default_rng(int(seed))
    x = nll - nll.mean()
    denom = float(x @ x) or 1.0
    ac = [float((x[:-k] @ x[k:]) / denom) for k in range(1, min(16, len(x)))]
    tau = 1.0 + 2.0 * sum(a for a in ac if a > 0)
    block = max(1, int(round(2.0 * tau)))
    n = len(nll)
    if block <= 1 or n <= 2 * block:
        boots = np.array([np.exp(rng.choice(nll, n, replace=True).mean())
                          for _ in range(int(resamples))])
    else:
        k = max(1, n // block)
        boots = np.empty(int(resamples))
        for i in range(int(resamples)):
            starts = rng.integers(0, n - block, k)
            boots[i] = np.exp(np.concatenate(
                [nll[s:s + block] for s in starts]).mean())
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    ppl = float(np.exp(nll.mean()))
    return {"perplexity": ppl, "lo": lo, "hi": hi, "n_tokens": len(nll),
            "nll": nll, "autocorr_time": float(tau), "block": int(block),
            "effective_n": int(len(nll) / max(tau, 1.0)), "half_width_pct": 100.0 * (hi - lo) / 2.0 / max(ppl, 1e-9)}


def better_than(a, b, alpha=0.05, seed=0, resamples=400):
    """Is model A better than model B, or is the difference undecidable?

    PAIRED bootstrap over the same positions -- the two models saw the same
    tokens, so the difference per position is the statistic, and pairing removes
    the probe-choice variance that swamps everything otherwise. This is why a
    paired test can call a 2% difference while the unpaired intervals overlap by
    40%."""
    na, nb = np.asarray(a["nll"]), np.asarray(b["nll"])
    if len(na) != len(nb):
        raise ValueError("paired comparison needs the same probe: %d vs %d"
                         % (len(na), len(nb)))
    d = na - nb
    rng = np.random.default_rng(int(seed))
    boots = np.array([rng.choice(d, len(d), replace=True).mean()
                      for _ in range(int(resamples))])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    pct = 100.0 * (a["perplexity"] - b["perplexity"]) / max(b["perplexity"], 1e-9)
    # AN INTERVAL THAT TOUCHES ZERO IS INDISTINGUISHABLE. A model compared to
    # ITSELF gives a difference of exactly zero at every position, so the
    # interval is [0, 0] -- and a strict `lo < 0 < hi` called that WORSE. The
    # test that exists to stop the pipeline manufacturing winners was itself
    # manufacturing one, on the easiest case there is.
    if lo <= 0 <= hi:
        verdict = "INDISTINGUISHABLE"
    elif hi < 0:
        verdict = "BETTER"
    else:
        verdict = "WORSE"
    return {"verdict": verdict, "delta_pct": pct, "ci_lo_nats": lo,
            "ci_hi_nats": hi, "n_tokens": len(d)}


def tokens_needed(reference, effect_pct, alpha=0.05):
    """How many tokens would be needed to RESOLVE an effect of this size.

    A CLOSED FORM, AND leCORE HAS A BETTER ONE. `min_detectable_effect` turns
    "we found nothing" into "there is nothing here above X" by INJECTING
    synthetic effects of known size into surrogates of the real data and
    measuring which sizes the test actually catches -- so the noise it reports
    against is the noise you face, not a normal approximation to it. This
    function assumes normality and inverts a z-test, which is fast and adequate
    for sizing a probe, and WRONG when per-token surprise is heavy-tailed, which
    it usually is. Prefer min_detectable_effect for any claim that has to hold
    up; use this to decide how long a probe to build.

    Answers the question a point estimate hides: was this comparison capable of
    detecting the thing it claimed to detect? Scales as 1/n, so halving the
    detectable effect costs four times the probe."""
    nll = np.asarray(reference["nll"])
    n = len(nll)
    s = float(nll.std())
    target = abs(np.log1p(float(effect_pct) / 100.0))
    if target <= 0:
        return float("inf")
    z = 1.96
    need = (z * s / target) ** 2
    return {"tokens_needed": int(np.ceil(need)), "have": n,
            "sufficient": bool(need <= n),
            "detectable_pct_now": float(100.0 * (np.exp(z * s / np.sqrt(n)) - 1))}


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import load_runtime

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("measure selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, cfg = load_runtime(src)
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    ids = [int(b) for b in raw[3000:3600].encode()][:300]

    a = measure(rt, ids)
    assert a["lo"] < a["perplexity"] < a["hi"], a
    assert a["half_width_pct"] > 0

    # ---- A MODEL COMPARED TO ITSELF MUST BE INDISTINGUISHABLE, or the test
    #      manufactures winners, which is exactly the failure it exists to stop
    same = better_than(a, measure(rt, ids), resamples=400)
    assert same["verdict"] == "INDISTINGUISHABLE", same

    # ---- AND A GENUINELY DAMAGED MODEL MUST COME OUT WORSE ----
    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime
    from holographic.io_and_interop.holographic_unicron import load_safetensors
    w = load_safetensors(os.path.join(src, "model.safetensors"))
    rng = np.random.default_rng(0)
    dmg = {k: (np.asarray(v, np.float64)
               + 0.02 * rng.standard_normal(np.asarray(v).shape)
               ).astype(np.asarray(v).dtype)
           if np.asarray(v).ndim == 2 else v for k, v in w.items()}
    d = measure(GDNRuntime(dmg, dict(rt.cfg)), ids)
    verdict = better_than(d, a, resamples=400)
    assert verdict["verdict"] == "WORSE", verdict

    # ---- AND IT SAYS WHEN A PROBE IS TOO SHORT TO DECIDE ----
    short = measure(rt, ids[:40])
    need = tokens_needed(short, 2.0)
    assert not need["sufficient"], need
    assert need["detectable_pct_now"] > 2.0, need

    print("measure selftest OK -- perplexity %.2f with a 95%% CI of %.2f..%.2f "
          "(+/-%.1f%%); a model compared to ITSELF reads INDISTINGUISHABLE "
          "instead of finding a winner, a noised model reads WORSE, and a "
          "40-token probe reports that it can only resolve effects above "
          "%.0f%% -- so a 2%% claim would need %d tokens, not 40"
          % (a["perplexity"], a["lo"], a["hi"], a["half_width_pct"],
             need["detectable_pct_now"], need["tokens_needed"]))


if __name__ == "__main__":
    _selftest()

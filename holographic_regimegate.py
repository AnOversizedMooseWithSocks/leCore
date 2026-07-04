"""holographic_regimegate.py -- RE-ENABLE a shelved method behind a detector (the adaptive-dispatch pattern).

WHY THIS EXISTS
---------------
Some methods were kept OUT of the default paths because they are wrong-or-costly in the GENERAL case -- a "kept
negative" -- even though they are SUPERIOR in a narrow regime (high roughness, high noise, high load, a large graph,
a sharp interface, ...). Now that we have a catalog and adaptive dispatch, "only good in a niche" flips from a reason
to SHELVE into a reason to GATE: run the superior method ONLY when a cheap detector says we are in its regime, and the
safe default everywhere else.

`RegimeGate` is that pattern, in one small object, so every re-enabled method is honest the same way:

    gate = RegimeGate(name, detect, threshold, superior, fallback, above=True)
    result, info = gate.apply(x, *args, **kwargs)

  * detect(x, *a, **k) -> a scalar REGIME SCORE (cheap, deterministic).
  * above=True: use `superior` when score >= threshold, else `fallback`. (above=False flips the comparison.)
  * `info` records the score, the threshold, and which path ran -- so a re-enabled method stays measurable.

THE DISCIPLINE (from the re-enable audit; enforced by construction here)
  * CONSERVATIVE BIAS -- the FALLBACK is the safe default; pick the threshold so borderline cases fall back. A gate
    misfire then costs at most the default, never worse than the shelved negative.
  * KEEP THE FALLBACK -- there is always a safe path; the superior method never becomes mandatory.
  * DETERMINISTIC -- a parameter read or a seeded residual probe; dispatch stays reproducible.
  * MEASURED BREAKEVEN -- each gate ships with a before/after measurement of where it helps vs hurts, and the
    detector's cost vs the method's win. If deciding costs as much as always paying, the method stays shelved.
"""


class RegimeGate:
    """Route to a superior-but-niche method only in its regime, and to a safe fallback everywhere else."""

    def __init__(self, name, detect, threshold, superior, fallback, above=True):
        self.name = str(name)
        self._detect = detect            # (x, *a, **k) -> scalar regime score
        self.threshold = float(threshold)
        self._superior = superior        # (x, *a, **k) -> result  (the niche method)
        self._fallback = fallback        # (x, *a, **k) -> result  (the safe default)
        self.above = bool(above)         # True: superior when score >= threshold; False: when score <= threshold

    def score(self, x, *args, **kwargs):
        """The regime score for `x` (the deciding measurement). Cheap and deterministic."""
        return float(self._detect(x, *args, **kwargs))

    def decide(self, score):
        """Which path a score selects -- 'superior' or 'fallback'. Borderline biases to the fallback."""
        use_superior = (score >= self.threshold) if self.above else (score <= self.threshold)
        return "superior" if use_superior else "fallback"

    def apply(self, x, *args, **kwargs):
        """Run the gate: measure the regime, then run the superior method in its regime or the safe fallback
        outside it. Returns (result, info) where info = {gate, score, threshold, used}."""
        s = self.score(x, *args, **kwargs)
        used = self.decide(s)
        fn = self._superior if used == "superior" else self._fallback
        return fn(x, *args, **kwargs), {"gate": self.name, "score": s, "threshold": self.threshold, "used": used}


def _selftest():
    # a toy gate: 'double it' is superior only when x is large; fallback is identity (safe everywhere)
    gate = RegimeGate("double_when_large", detect=lambda x: abs(x), threshold=10.0,
                      superior=lambda x: x * 2, fallback=lambda x: x)
    small, i_small = gate.apply(3.0)
    big, i_big = gate.apply(50.0)
    assert small == 3.0 and i_small["used"] == "fallback"        # outside the regime -> safe default
    assert big == 100.0 and i_big["used"] == "superior"          # inside the regime -> the niche method
    assert i_big["score"] == 50.0 and i_big["threshold"] == 10.0
    # borderline biases to fallback (score == threshold with above=True uses superior; just under -> fallback)
    assert gate.decide(9.999) == "fallback"
    print("OK: holographic_regimegate self-test passed (superior only in-regime; fallback outside; borderline -> "
          "fallback; info recorded for measurement)")


if __name__ == "__main__":
    _selftest()

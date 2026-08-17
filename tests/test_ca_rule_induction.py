"""C3: recovering a cellular-automaton rule from observation -- with COVERAGE reported.

SOTA constraint this test exists to honour: CA rule inference is DEGENERATE. "Multiple
distinct rulesets can both perfectly map some ICs to matching GS2s... finding the 'correct'
RM is impossible, since there may be multiple equally plausible options", and the degeneracy
is governed by how many distinct neighbourhoods the trajectory actually exhibits (the
literature calls it coverage). Reporting a recovered rule WITHOUT its coverage would be
precisely the overclaim the field warns about, so both halves are pinned here: exact
recovery at full coverage, and honest non-uniqueness below it.
"""
import numpy as np


def eca(rule, x0, steps):
    x = np.asarray(x0, int).copy()
    hist = [x.copy()]
    for _ in range(steps):
        idx = (np.roll(x, 1) << 2) | (x << 1) | np.roll(x, -1)
        x = ((rule >> idx) & 1).astype(int)
        hist.append(x.copy())
    return np.array(hist)


def induce(hist):
    """Return (rule_bits, unobserved_neighbourhoods). Every unobserved neighbourhood is a
    FREE BIT: 2**k rules fit the data equally well."""
    n = hist.shape[1]
    seen = {}
    for t in range(len(hist) - 1):
        for i in range(n):
            nb = (hist[t][(i - 1) % n] << 2) | (hist[t][i] << 1) | hist[t][(i + 1) % n]
            assert seen.get(nb, hist[t + 1][i]) == hist[t + 1][i], "non-deterministic data"
            seen[nb] = hist[t + 1][i]
    rule = 0
    for nb, out in seen.items():
        rule |= int(out) << nb
    return rule, [b for b in range(8) if b not in seen]


def test_exact_recovery_at_full_coverage():
    """Six Wolfram rules spanning the behavioural classes, from random initial conditions."""
    rng = np.random.default_rng(0)
    x0 = (rng.random(40) < 0.5).astype(int)
    for rule in (110, 30, 90, 184, 254, 0):
        got, missing = induce(eca(rule, x0, 12))
        assert missing == [], (rule, missing)
        assert got == rule, (rule, got)


def test_low_coverage_is_reported_as_non_unique():
    """The degeneracy, reproduced and pinned. Rule 90 from a SINGLE active cell over 6 steps
    exhibits only 5 of 8 neighbourhoods, so 8 rules fit the data equally well and the
    recovered one need not be 90. The right answer is not a better search -- it is saying
    'one of 8' out loud. Twenty steps of the same initial condition reaches full coverage
    and recovers 90 exactly, which is the actionable advice: observe longer."""
    x0 = np.zeros(31, int)
    x0[15] = 1
    got, missing = induce(eca(90, x0, 6))
    assert len(missing) == 3 and 2 ** len(missing) == 8
    got20, missing20 = induce(eca(90, x0, 20))
    assert missing20 == [] and got20 == 90

"""H3 -- the numerics adoption audit, pinned as a test so its verdict cannot silently rot.

THE SWEEP'S MEASURED VERDICT (2026-07-22): the numerically DANGEROUS rolling pattern -- a cumulative sum of
SQUARES differenced to get a windowed second moment, the one holographic_rolling measured at an absolute
error of 8.75 vs 2e-9 exact on data offset by 1e8 -- exists NOWHERE in the tree outside the rolling kit
itself (which is exact by default). The surviving cumsum-window sites are all FIRST-moment box means:
holographic_envelope (mean of |diffs| -- differencing kills any offset before the sum ever sees it),
holographic_hazedepth and holographic_shapefromshading (image-space box filters over bounded pixel values).

WHY THOSE SITES ARE LEFT AS-IS, i.e. the sweep's kept negative: adopting the rolling kit there would change
the floating-point summation ORDER, which changes emitted bytes at the ULP, which the backward-compatibility
constraint forbids for zero numerical benefit (first moments of bounded/offset-free data are not where
cumsum bites). The QEM precedent applies: exactness-by-default for NEW code, opt-in never-flip for OLD.

WHAT THIS TEST DOES: greps the live tree for the second-moment-cumsum pattern outside the kit and FAILS if
one ever appears -- so the next person who writes `cumsum(x**2)` for a rolling variance gets sent to
mind.rolling_stats by a test failure instead of by a code review that may not happen. It also re-verifies
the kit's own exactness claim on the hostile fixture, so the number this audit leans on stays measured.
"""
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent / "holographic"

# cumsum of a squared quantity: np.cumsum(x**2), np.cumsum(x*x), np.cumsum(np.square(...))
# Three shapes of the same sin: cumsum(x**2), cumsum(x*x), cumsum(np.square(x)). The backreference makes
# cumsum(a*a) match while cumsum(a*b) (a legitimate cross-product) does not.
SIMPLE = re.compile(r"cumsum\s*\(\s*([A-Za-z_]\w*)\s*\*\s*\1\s*\)"
                    r"|cumsum\s*\(\s*[\w\.]+\s*\*\*\s*2"
                    r"|cumsum\s*\(\s*np\.square\s*\(")
# The idiom this audit must NOT flag, learned from the sweep's own first run: cumulative ENERGY FRACTIONS
# of a singular-value spectrum -- cumsum(s**2) normalised by sum(s**2) to pick a rank. Six sites
# (denoise, creature x2, ratedistortion, tucker x2) all match it. No window differencing happens, so the
# catastrophic cancellation that motivates this audit cannot occur: the sin is cumsum-of-squares
# DIFFERENCED ACROSS A WINDOW, and a spectrum energy fraction never subtracts shifted partial sums.
ENERGY_FRACTION = re.compile(r"cumsum\s*\([^)]*\*\*\s*2\s*\)\s*/")

ALLOWED = {"holographic_rolling.py"}                             # the kit may do it: it is the exact home


def test_no_second_moment_cumsum_outside_the_rolling_kit():
    offenders = []
    for p in ROOT.rglob("*.py"):
        if "__pycache__" in p.parts or p.name in ALLOWED:
            continue
        text = p.read_text(errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if SIMPLE.search(line) and not ENERGY_FRACTION.search(line):
                offenders.append("%s:%d: %s" % (p.name, i, line.strip()))
    assert not offenders, (
        "second-moment cumsum found outside the rolling kit -- this is the pattern measured at abs error "
        "8.75 on offset data; use mind.rolling_stats (exact by default) instead:\n" + "\n".join(offenders))


def test_the_number_this_audit_leans_on_stays_measured():
    """The 8.75-vs-exact claim, re-run live: windowed std of offset data via the cumsum trick vs the kit."""
    from holographic.sampling_and_signal.holographic_rolling import rolling_std
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000) + 1e8
    w = 50
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    t = np.arange(w, x.size + 1)
    mean = (cs[t] - cs[t - w]) / w
    var_trick = (cs2[t] - cs2[t - w]) / w - mean ** 2            # catastrophic cancellation at 1e8 offset
    trick = np.sqrt(np.maximum(var_trick, 0.0))
    exact = rolling_std(x, w)[w - 1:]
    err = float(np.max(np.abs(trick - exact)))
    assert err > 0.5, err                                        # the trick IS this bad here -- that is the point
    truth = np.array([x[i - w:i].std(ddof=0) for i in range(w, w + 200)])
    assert float(np.max(np.abs(exact[:200] - truth))) < 1e-6     # and the kit is not

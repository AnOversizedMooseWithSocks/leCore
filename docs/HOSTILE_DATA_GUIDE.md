# THE HOSTILE-DATA GUIDE

*How to find real structure in noisy sequential data using leCore's honesty layer -- and how to refuse to be
fooled. Every number quoted here was measured by the tools' own selftests; every snippet below is executable
and IS executed by `tests/test_hostile_data_guide.py`, so this guide cannot rot without a test failing.*

The one-sentence version: **your pipeline manufactures structure, your evaluation leaks the future, your
battery selects its winners, and your aggregate hides the shape of its losses -- there is a tool for each,
and the order below is the order to run them.**

---

## 0. The mindset

Hostile data is not adversarial data. It is ordinary data plus an ordinary analyst: smoothers that create
momentum, thresholds that contain the move they "predict", twenty tries and one report. The measured
precedents this engine carries:

* a trailing smoother + persistence count reads **79.4% direction persistence on pure white noise**
  (`pipeline_null`'s own selftest);
* a naive full-history nearest-neighbour "history match" reports **perfect skill, MSE exactly 0.0** -- it
  finds the query itself (`CausalIndex`'s selftest);
* under overlapping event windows, a naive across-events t-test **false-alarms at 28%** where nominal is 5%
  (`event_study`'s selftest);
* a p = 4e-4 finding that cleanly wins its 4-test family **dies on the same session's 64-look book**
  (`SelectionLedger`'s selftest);
* a forecast squashed to **38x worse calibration keeps 100% of its decision value**
  (`calibration_vs_value`'s selftest) -- so the number you optimised may not be the number that pays.

None of those required bad faith. All of them required a null that shares the machinery.

---

## 1. Before anything: is your pipeline causal, and is it inventing things?

Two different questions. Ask both, in this order.

**Is it causal?** `lookahead_lint` recomputes your signal on truncated prefixes and demands the shared range
be *identical* -- a causal pipeline cannot know whether data exists after t. Exact, not statistical.

```python
import numpy as np
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
x = np.cumsum(np.random.default_rng(0).standard_normal(400))

leaky = lambda s: (s - s.mean()) / (s.std() or 1.0)          # full-sample z-score: sees the future
honest = lambda s: np.concatenate([[0.0], np.diff(s)])
assert mind.lookahead_lint(leaky, x)["causal"] is False
assert mind.lookahead_lint(honest, x)["max_drift"] == 0.0
```

The classics it catches at machine precision: full-sample z-scores, centred smoothers, global min-max,
global detrend, and the EW variance seeded at the full sample's variance. Use the causal replacements in
`mind.rolling_stats` / `mind.streaming_stats` -- every stat there lints at 0.0 drift and is bit-identical to
the conditioning gates' lambdas.

**Is it inventing structure?** Causal is not honest: the 79.4% chain above never peeks and still
manufactures its result. `pipeline_null` runs your *whole pipeline* on surrogates of its own input, so the
statistic is judged against what the machinery makes from nothing.

```python
def persistence(s):
    y = np.empty(len(s)); y[0] = s[0]
    for i in range(1, len(s)):
        y[i] = 0.8 * y[i - 1] + 0.2 * s[i]
    sg = np.sign(y); sg = sg[sg != 0]
    return float(np.mean(sg[1:] == sg[:-1]))

noise = np.random.default_rng(0).standard_normal(1500)
assert persistence(noise) > 0.7                               # the seductive number...
r = mind.pipeline_null(persistence, noise, surrogate="iid_shuffle", n=100, seed=0)
assert abs(r["z"]) < 3.0                                      # ...is what the machinery makes from nothing
```

Choosing the surrogate is part of the honesty: a null that preserves what you are testing is not a null
(`sign_flip` is degenerate for magnitude statistics -- null std exactly 0.0; `iid_shuffle` preserves the
mean exactly). The surrogate family's docstrings carry each one's kept negative.

---

## 2. Measuring an effect: baselines, replication, floors

* **`split_half`** -- does it replicate? Contiguous mode kills decaying artifacts; interleaved vs contiguous
  *disagreeing* is itself a finding (the effect is regime-bound).
* **`min_detectable_effect`** -- what could you even see? Quote the floor **with its injection shape**; a
  floor for a level shift says nothing about a burst.
* **`mutual_information_vs_null`** -- effect size in **bits**, because z inflates with n: the same ~0.01-bit
  coupling reads z=2.5 / 11.7 / 31.5 at n = 3k / 12k / 48k.
* **`dpi_guard`** -- is a "new feature" new information or a transform of what you have? (tanh-of-quadratic
  reads TRANSFORM at holdout R^2 = 0.98.)

---

## 3. Many candidates: the battery, the committee, and the book

Screening is where analyses go wrong, in a specific order: one detector, then twenty, then report the best.

```python
rng = np.random.default_rng(0)
states = rng.standard_normal((1200, 12))
target = np.sign(states[:, 0]) * np.abs(rng.standard_normal(1200))

prog = mind.signal_program(dim=256, seed=0)
prog.add_check("real", lambda s: s[:, 0])
for j in range(1, 12):
    prog.add_check("noise_%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
rep = prog.screen(states, target)
assert rep["passed"] == ["real"]                              # gates INSIDE the loop: FDR + split-half,
assert rep["clusters"] == 1                                   # correlated checks collapse to one finding
```

An empty pass-list is a *result* with a reason. `build_committee(rep)` seats one representative per cluster,
votes with tie = abstain, and the committee must pass **its own** gates on fresh data -- members passing
individually does not transfer (the veto-cancellation negative is pinned in its selftest).

And the debt no battery can see: batteries you ran last week. `mind.selection_ledger()` is append-only --
`record` every test *when it is run*, `correct` over the whole book. Withdrawals keep their multiplicity
cost; re-runs are countable; a book with a deleted row refuses to load.

---

## 4. Time is not your friend: events, clocks, recall

* **`event_study`** -- what happens after the trigger, vs the **circular-shift null** (the pattern slides,
  spacings and overlap ride along, only alignment is tested). Read `pre_trend` first: a large pre-trend z
  means your event *definition* already contains the move. Never rebuild a CI from `mean_path` -- that is
  the 28% false-alarm path.
* **`reclock`** -- movement clocks are legitimate and they *manufacture direction as a mechanism property*
  (renko made +72% fake momentum; the total-variation clock makes ~25% fake reversion on the same noise).
  `null_persistence` carries the honest chain built in.
* **`mind.causal_index()`** -- history-matching that structurally cannot see the future. The naive
  full-history call finds itself (perfect fake skill); this one cannot self-match at any k.
* **`target_shift_probe`** -- is the signal *ahead* of its target or explaining it? Catches the
  contemporaneous leak; is blind to symmetric (centred-label) leaks, which belong to `lookahead_lint` run
  on the label constructor. The two blind spots are complementary and each docstring names the other's case.

---

## 5. Conditions, coverage, and where the losses live

* **`causal_gate` / `trailing_gate`** -- act only on what you knew; `audit_causality` *verifies* the mask by
  perturbing the future rather than asserting it in prose.
* **`conditional_coverage`** -- marginal coverage is an average and an average can hold while both sides are
  wrong: 89% overall was 97% calm / 65% storm on the measured fixture. The guarantee you quote is
  conditional on nothing unless you check.
* **`envelope_forecast` / `envelope_vs_constant`** -- forecast the half that is forecastable (scale, not
  direction), and read the baseline's `verdict`: CONSTANT-FAILED means the comparison collapsed, not that
  the envelope lost.
* **`loss_space_report`** -- the *shape* of the loss before the aggregate is quoted: tail vs matched
  Gaussian, streaks vs the permutation null, per-condition share vs occupancy. Its value-side sibling is
  `insurance_profile`; run both before gating.
* **`calibration_vs_value`** -- the statistician's verdict and the decision-maker's, kept separate.
  Calibration is a repair; resolution is the source.

---

## 6. The costs of acting

* **`net_of_costs`** -- the cost wall: +9.3bp gross dies at a 17bp wall (t = -7.8) and survives 5bp.
  State-dependent costs interact with *which* events they land on -- costs on the good events can RAISE the
  t (the covariance surprise, kept).
* **`realizable_fills`** -- emission price vs actionable price: a completion-detector's +2.01 idealized edge
  was **-0.89 actionable** -- 144% of it was latency. lag=0 is refused by name: simultaneous is not past.

---

## 7. The order to run them (the checklist)

1. `lookahead_lint` your signal fn; `target_shift_probe` your signal against its target.
2. `pipeline_null` the whole chain on its own surrogates. Pick the surrogate that does NOT preserve your
   statistic.
3. Effects: `split_half`, `min_detectable_effect` (with its injection named), bits not z.
4. Batteries through `signal_program`; every battery's rows onto the `selection_ledger`.
5. Events through `event_study`; read `pre_trend` before `forward`.
6. Conditions: gate candidates from `loss_space_report`, safety from `insurance_profile`, causality of the
   gate from `audit_causality`, coverage from `conditional_coverage`.
7. Only then the economics: `net_of_costs`, `realizable_fills`, `calibration_vs_value`.
8. Committee last, on fresh data, and its verdict is the verdict.

If a step refuses, **the refusal is the result.** Write it down; the ledger is there for exactly that.

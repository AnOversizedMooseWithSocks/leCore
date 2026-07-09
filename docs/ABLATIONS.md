# Ablations: where is VSA actually load-bearing?

`holostuff` honestly reports, again and again, that a simple baseline ties the
holographic version of a task. Individually each is healthy. Together they pose a question
this table answers system-wide: **which subsystems genuinely need superposition / binding
/ corruption-robustness, and which are a VSA showcase where VSA isn't the reason they
work?**

The method is the only honest one: for each subsystem, run the **dumbest honest
non-holographic baseline** on the *same task, data, and metric*, measure both **across
seeds** with the variance harness (`holographic_measure.py`), and let the 95% confidence
intervals decide:

- **VSA load-bearing** — the holographic lower CI sits above the baseline's upper CI. The
  superposition/binding/cleanup is the reason it works. Invest here.
- **uniformity** — the intervals overlap. The simple baseline ties it: the *idea* works,
  not the VSA *encoding* of it. Relabel honestly.
- **baseline wins** — the baseline's lower CI sits above the holographic upper CI. VSA is
  decorative for the stated metric (though it may buy something else, e.g. scale).

All rows run on real corpora (Reuters, UDHR, Brown) or at real scale. Reproduce with
`python holographic_ablate.py`.

## The table

| Subsystem | Holographic | Dumbest honest baseline | Verdict |
|---|---|---|---|
| topic classify (Reuters, 5-cat) | **0.83 ± 0.05** | bag-of-words centroid: 0.61 ± 0.06 | **VSA load-bearing** |
| key→value, **noisy** keys | **0.89 ± 0.07** | exact dict: 0.00 ± 0.00 | **VSA load-bearing** |
| language ID (UDHR, 6 lang) | 0.99 ± 0.01 | bag-of-trigrams centroid: 0.99 ± 0.00 | uniformity |
| segmentation (Brown, spaceless) | 0.60 ± 0.01 | exact count-based entropy: 0.61 ± 0.00 | not load-bearing\*\* |
| recall index (2000 items) | 0.82 ± 0.03 | exact brute-force scan: 1.00 ± 0.00 | baseline wins\* |

\* The forest *loses* on raw recall — an exact scan is trivially perfect — but reaches its
recall at **~41% of the comparisons**. Its win is **sublinear scale**, not accuracy; read
this row as "decorative for accuracy, load-bearing for cost."

\*\* The exact count-based entropy marginally *edges* the holographic estimate
(0.612 vs 0.604) — a razor-thin gap that, across seeds, lands between "uniformity" and a
hair's-breadth baseline win. Either way the conclusion is the same: VSA is **not
load-bearing** for segmentation, and the exact estimator is at least as good, exactly as
you'd expect of an exact method versus an approximate one.

## What this says about where the value lives

**The genuine, irreplaceable VSA wins are corruption/approximation, not classification
accuracy in general.** The two clean "load-bearing" rows are both cases where the baseline
is structurally unable to do the job:

- **Noisy-key key→value** is the sharpest. An exact dict scores a flat **0.00** the moment
  a key is perturbed — a hash either matches or it doesn't. VSA's cosine cleanup recovers
  the value from an *approximate* cue at ~0.89. This is binding + cleanup doing something
  no count-based structure can, and it is the mechanism the whole engine leans on.
- **Topic classification** is a real ~0.22 win over raw word counts. The holographic
  encoding folds co-occurrence structure into each document vector, which generalises where
  a sparse bag-of-words centroid does not. Here VSA earns its place on accuracy outright.

**Two subsystems are uniformity — and that's worth saying out loud.** For **language ID**
and **segmentation**, the dumb baseline ties (or marginally beats) the holographic version.
The honest reading is that the *idea* carries the result, not the VSA machinery:
character-trigram statistics tell languages apart whether you superpose them into a profile
vector or just count them; branching-entropy finds word boundaries whether you estimate the
entropy holographically or from exact counts. These are legitimate, working subsystems —
but they are demonstrations of an idea expressed in VSA, not evidence that VSA is *necessary*
for the task. The exact count-based entropy even edges ahead (0.612 vs 0.604), exactly as
you'd expect from an exact estimator versus an approximate one.

**One subsystem buys scale, not accuracy.** The recall forest is honestly behind exact scan
on recall, but it gets there sublinearly. That's its entire reason to exist, and the table
records it as such rather than dressing the recall number up as a win.

## The throughline

Pointed at itself, the engine's "measured, not promised" discipline says: **VSA is
load-bearing exactly where the problem is approximate or compositional — recovering a value
from a corrupted cue, folding structure into a representation — and decorative where an
exact, countable statistic already settles the task.** That is a sharper and more useful
self-description than "a holographic system that does text and memory," and it tells you
where the next unit of effort belongs: the damage-tolerant, approximate-cue corners, not
the ones a `Counter` already wins.

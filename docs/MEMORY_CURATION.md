# Memory curation, decay and reflection

`docs/COMPETITIVE_NOOA.md` §3 rates leCore **PARTIAL** on NOOA's long-term memory subsystem:
*"`recall` exists; the curation/decay/reflection subsystem does not"*. §6 calls it the strongest
published evidence for any single item on that list, at **+11.8 RHAE** for NOOA's memory
subsystem over the same agent with markdown notes. This is that subsystem, built from parts the
engine already had.

## The one door

```python
import lecore
m = lecore.UnifiedMind(dim=256, seed=0)
m.teach("who wrote dune", "frank herbert")
m.teach("who wrote dune", "f. herbert")          # a RE-TEACH is the same fact, not a second one

c = m.memory_curate()                            # a curator over this mind's taught partition
c.observe("who wrote dune")                      # record a USE at the next logical tick
c.plan(keep=200)                                 # pure: what curation WOULD do. Mutates nothing.
c.apply(keep=200)                                # demote to a restorable archive + a journal
c.reflect()                                      # what is hot, what was superseded, and why
c.restore()                                      # exact undo
```

`mind.memory_curate()` returns a `MemoryCurator`; everything else is a method on that object, the
same shape as `mind.tiered_memory()` and `mind.celled_memory()`.

## Three legs, one index

| leg | what it does | reused from |
|---|---|---|
| **decay** | ACT-R base-level activation per fact — recency and frequency in one number | `holographic_actr` (`base_level`, `rank`) |
| **curation** | a bounded hot set; everything else demoted | the promotion/demotion shape of `tiered_memory` |
| **reflection** | a re-taught question is ONE fact whose newest answer is current; older rows superseded with a pointer | new |

They are one object because they read one index. Ranking without consolidation double-counts a
fact taught twice; consolidation without ranking has no order to consolidate in.

## It cannot delete anything, and that is the design

`unicron_turn_memory` already paid for this lesson and wrote it down:

> ACT-R eviction made that survivable by overwriting the least active slot, **but eviction is a
> loss, and a flat file is the reason it was needed.**

That sweep found the eviction was an artefact of a flat layout and replaced it with nested bases —
four times the capacity at 100% recall, forgetting nothing. So there is **no delete path in this
module**. Curation *demotes* into a restorable archive, `apply()` journals every action with its
reason, `plan()` mutates nothing at all, and `restore()` returns the store byte for byte. A
curation pass that silently drops a fact a user taught is a data-loss bug, not a feature.

`apply(write_through=True)` is the only call that touches the mind's own taught log, it is off by
default, and `restore(write_through=True)` puts the log back exactly.

**Kept negative:** write-through rewrites a list the ladder also builds reflex keys from. The
archive makes the *rows* recoverable; it does not roll back a derived index. Restore, then
re-teach, if you need the reflex keys back too.

## Time is logical, never wall-clock

A decay that reads the system clock gives a different answer on Tuesday. Here a row's birth is its
**index in the taught log** and `observe()` takes the next integer tick, so the same partition
curates identically on any day. Fact identity is `sha256` of the normalised question — `hashlib`,
never `hash()` — so a journal from one process is comparable with one from another.

## Measured — and self-measured, which is the honest label

**There is no widely adopted public benchmark that scores forgetting, decay or consolidation
directly.** LoCoMo, LongMemEval and BEAM score retrieval and reasoning; only BEAM's "updating
information over time" comes close. So the numbers below are **not comparable** to Mem0's
published bests (LoCoMo 92.5, LongMemEval 94.4, BEAM-1M 64.1) and this is not a SOTA claim.

`curation_benchmark` asks one question: given a bounded hot set of 40 facts out of 200 chosen from
a history of uses, what fraction of **future** queries land in it? The access stream is a Zipf
popularity that shifts topic partway through the history; `drift` is the fraction of probability
mass that moves onto previously-cold facts.

20 seeds, paired bootstrap 95% CI (`drift_sweep()`):

| drift | keep-everything | random | recency window | frequency | **activation** | winner |
|---|---|---|---|---|---|---|
| 0.00 | 1.0000 | 0.2000 | 0.6915 | 0.7437 | 0.7397 | frequency |
| 0.25 | 1.0000 | 0.2000 | 0.6080 | 0.6584 | 0.6538 | frequency |
| 0.50 | 1.0000 | 0.2000 | 0.5939 | 0.6246 | **0.6340** | activation |
| 0.75 | 1.0000 | 0.2000 | 0.6020 | 0.6100 | **0.6399** | activation |
| 1.00 | 1.0000 | 0.2000 | **0.6915** | 0.6224 | 0.6610 | recency |

Paired against the recency window: **+0.048 / +0.046 / +0.040 / +0.038** at drift 0.00–0.75, every
interval clear of zero — and **−0.031 [−0.041, −0.020] at drift 1.00, a significant LOSS.**
Against frequency it is a genuine tie on a stationary process (CI straddles zero at drift 0.00 and
0.25) and wins from drift 0.50 up.

**The finding, stated plainly: activation is the robust middle, not a dominant policy.** It is the
only policy that is never significantly worse than both baselines, but under a complete topic
switch a plain recency window beats it, and on a stationary process plain frequency counting is
just as good. If all you have is a stationary workload, count hits and skip this.

`keep-everything` is 1.0000 by construction — it is the unbounded upper bound, and its distance
from every other row is the price of the bound, not a result.

## The bug this harness caught in itself

The first version of the stream made `drift` a fraction of **rank slots**. Under Zipf(1.1) the top
25 of 200 ranks carry 71% of the traffic, so "drift=0.25" silently moved 71% of future queries onto
the facts that had been *coldest* in the history — history was not uninformative, it was
**anti-predictive**, and all three policies scored 0.086–0.13 against a 0.20 random baseline.
Three policies losing to chance is a broken generator, not a result about the policies, and it read
as a finding for a full measurement cycle. The random baseline is now reported in every row and
the module selftest asserts every policy beats it.

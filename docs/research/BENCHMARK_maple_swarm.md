# The maple field-journal swarm (2026-09-27): what a real run did to the reflex arc

The task was to draw a realistic sugar maple on a nature-journal page in leStudio (a graphite sketch, then watercolour
on new layers). Five subagents did it in two phases (w1–w3 for the sketch, p1–p2 for the paint). They shared ONE
leCore service (port 8080) as their external memory, and every worker followed the ask-first contract through
`serve()`, answering as the model end with `meaning_resolve` (same / new / unclear). Raw numbers are in
`evidence/maple_swarm_reflex.json`; the logs and scripts stayed in the session scratchpad.

## What the arc did

| | asks | served from memory (T0) | escalated |
|---|---|---|---|
| sketch phase (w1–w3, cold on leStudio) | 22 | 2 (9%) — both taught by a sibling worker minutes earlier | 20 |
| paint phase (p1–p2, fresh workers) | 24 | 6 (25%) — lessons the sketch phase taught | 18 |

- **serve latency:** median 10.2 ms, p90 23.8 ms (n = 49). The first call of a fresh service took 568 ms.
- **Verbatim re-asks after a verdict:** a wording that has been answered once serves at T0 from then on. It also
  carries to near-rewordings: "how should a swarm of agents share memory?" served after "how should a swarm share
  memory" was linked to the swarm doctrine row.
- **Unseen rewordings, 16 per round, judged by a keyword the right answer must contain:**
  - 1/16 after the swarm;
  - 2/16 after the fixes below;
  - 2/16 on a third round of wordings.
  - Across all 48 there were 0 wrong serves. Across 21 must-not probes (off-domain, a near-miss species, a bare
    ticker) there were 0 wrong serves.
  - The gate is honest but data-limited. Paraphrase scores sit at s1 0.4–0.75, where 77 verdicts put the
    calibrated P(agree) at 0.42–0.85, below the 0.95 bar. Every verdict raises it.
- **A WRONG lesson got in.** w3 taught, as a "new" answer, that `/api/hatchfill` only fills circles. The route takes
  `poly` + `feather`, and w2 had found that. Memory cannot verify an agent's claim. The coordinator corrected it with a
  re-teach. The distiller keeps model- and agent-provenance rows OUT of the shipped core memory (86 excluded as
  provisional). This run is the case for that rule.

## Faults found, fixed, measured

1. **The escalation ledger was not a to-do list.** `meaning_resolve()` and `teach()` never cleared the question
   they answered, and a question that later served stayed "open". After the swarm, 26 of 55 open questions were
   already serving at T0. Now an applied verdict clears it, a verbatim teach clears it, and a served answer clears
   it. `resolve()` still reports `cleared`.
2. **Swarm races poisoned the calibrated gate.** Three workers answered "new" for a wording a sibling had taught
   seconds earlier. Re-ranked at resolve time, the top row was that very wording (score 1.0), so each verdict became
   a label saying "top row wrong at g ≈ 1.7". These were 3 of the 4 labels above g 1.1. The isotonic curve at
   g ≥ 1.2 fell to 0.71, and the gate would serve **0** of 32 fresh rewordings. Labels are now taken against the
   candidates the model was SHOWN (`meaning_packet` remembers them, bounded at 4096). A "new" verdict on an exact
   existing wording, with no shown list, is applied but not labelled. With the three race labels removed from the
   partition (a one-off repair, evidence in the log), p(1.2) is 0.994 and the gate serves 11 of the same 32.
3. **A restarted service saved into a file nobody reads.** After a restart, `learning_save` without a rollover wrote
   the legacy `learning/state.lecore`. The loader ignores that file while a dated generation exists, so 26
   calibration labels and 28 verdicts would have vanished at the next boot. A mind that LOADED a generation now
   saves back into it, and a generation under another root is never the write target.
4. **The core memory dropped learned phrasings of doctrine.** A wording linked to a seedpack doctrine row was
   excluded along with the row, which the seedpack ships. Every fresh install escalated it again. The link now ships
   and the doctrine text still does not. The bundle carries 2 wordings (it carried 0 before), and `--check` passes on
   both copies (127/127).

5. **A confirmed wording scored like a guess.** A row's score is 0.2 × its best phrasing plus 0.8 × its centroid,
   so the more varied wordings a row gathers, the LOWER an exact confirmed wording scores. A row with seven
   wordings scored 0.68–0.78. After a restart, 15 of the 48 wordings the swarm's `same` verdicts had linked escalated
   again, which meant a repeat model call each time. A wording that a verdict linked to a row, or the row's own
   canonical wording, is now served as that row, like the ladder's exact T0. Negatives still win, and method rows
   still go through argument extraction. After the final restart, **86/86** learned wordings serve (same, new and
   unclear verdicts). Trade-off: a wording linked by a WRONG verdict now serves until it is vetoed, which is the
   same trust as a taught T0 answer. The noisy-teacher benchmark below measures that.

Tests: `tests/test_swarm_learning_races.py` (10) and
`tests/test_distill_release.py::test_a_learned_wording_of_a_doctrine_row_reaches_the_core_memory`.

## leStudio findings (a separate repo; not changed here)

- **OOM kill at 5.6 GB.** At boot the studio restores every document in its workspace directory by replaying all
  their strokes. Nine old documents cost 2.35 GB idle. A run on its own `LESTUDIO_WS` drew the whole sketch at
  0.78 GB and finished the paint phase at 2.6 GB. Recommended fix: restore non-active documents lazily. Boot took
  15.1 s.
- **Overlapping paint batches from two users slow down 15–20×.** 93 water strokes took 30.9 s and 178 took 79.4 s,
  against 12–25 ms per stroke alone (measured by the workers).
- **`/api/advise` is a second, in-process leCore mind** that autoboots its own partition and saves to a relative
  `lecore_memory`. The swarm's lessons never reach it. It should delegate to a shared service when one is configured.
- **API friction:**
  - `/api/composite.png` has a median of 1.7 s (max 7.6 s), while `/api/layer/<id>.png` has a median of 0.12 s.
  - `/api/text` returns no extent.
  - A new layer's owner reads `""` until its first edit.
  - A paint_batch whose strokes all fall under 6 points is silently empty.

## The noisy-teacher benchmark (CLINC150, 10% wrong verdicts, seed 0)

This is the trade-off check for fault 5 and the labelling change for fault 2. `tools/bench_meaning.py online
--model noisy:0.1`. Each variant runs the current tree with one change switched off.

| variant | stream model calls | stream memory right / WRONG | detached in-scope right / wrong | out-of-scope wrong |
|---|---|---|---|---|
| both changes (shipped) | 9,593 | 5,221 / 136 | 1,989 (44.2%) / 71 (1.6%) | 24 (2.4%) |
| without the exact-wording serve | 9,593 | 5,221 / 136 | 1,989 (44.2%) / 71 (1.6%) | 24 (2.4%) |
| without shown-list / race labelling | 9,593 | 5,221 / 136 | 1,989 (44.2%) / 71 (1.6%) | 24 (2.4%) |
| recorded 2026-09-26 (evidence/bench_meaning_online.json) | 10,239 | 4,591 / 120 | 1,245 (27.7%) / 23 (0.5%) | 5 (0.5%) |

Neither change moves this benchmark. The stream never repeats a wording, and its model end is in-process, so it
passes the ranked list itself. **A LOUD NEGATIVE that predates this run:** the tree as a whole drifted from the
09-26 recording. Seed 0 now serves more (44.2% vs 27.7%) but wrongly more often: 1.6% vs 0.5% in scope and 2.4% vs
0.5% out of scope. The E2.3 acceptance bar is ≥ 30% correct at ≤ 2.0% wrong and ≤ 2.0% out-of-scope served. Seed 0
now passes coverage and in-scope wrong, and FAILS out-of-scope (2.4%). For comparison, the shipped seeds 1 and 2 read
2.0% and 1.8% out of scope on 09-26. Bisect the sweep-182 waves against this benchmark (all three seeds) before the
next release.

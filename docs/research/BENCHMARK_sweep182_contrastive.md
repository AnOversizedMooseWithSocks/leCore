# Sweep 182 — the CLM backlog pass: one learning loop under every door (2026-09-26)

Owner direction (Moose, 2026-09-26): gather the expert panel, spin up a swarm on the shared memory, and have JEV,
NOOA and CLM all help improve the reflex arc and the substrate — *adapt, don't copy; holographic where it makes
sense*. The panel measured before it argued (`docs/research/evidence/clm_panel_20260926/`); the backlog it wrote
(a local working note, `docs/BACKLOG_contrastive.md`, gitignored like every backlog) was then implemented in two
waves of workers. The owner's answers to the backlog's open questions: build BOTH trace-correction designs and let
the measurement choose; a GPU host is fine for the CLM plugin but it MUST work on CPU; learned prototypes MAY travel
through `contribute` / `commons_pool`, behind the guard.

This document is the whole pass, item by item: the number, its baseline, the bar it had to clear, PASS or FAIL,
the kept negatives, how to reproduce it, and whether it ships ON or OFF. The loop every item serves:

- **DECIDE** (JEV) — every door returns ONE typed record: `value, ranked, margin, p_correct, p_null, set, via, id,
  evidence`.
- **VERIFY** (NOOA) — only a verified or reported truth becomes a label.
- **OUTCOME** — `decision_outcome(id, truth)` is the only outcome path, and it survives a restart.
- **LEARN** (CLM, done holographically) — one contrastive rule (`holographic_protostore.ProtoStore`: InfoNCE on
  per-option prototypes, `A_k += lr (t_k − p_k) q`, `p = softmax(cos / τ)`, τ 0.05, lr 0.3; as τ → 0 it IS the old
  miss-only AdaptHD); per door: its own store, its own calibrator (`DoorCalibrator`), its own label stream.

The syntax, with live-generated JSON for every door, is `docs/TYPED_DECISIONS.md` sections 8d (Phase A) and 8e
(wave 2). The agent-facing workflow is `skills/lecore-boot/SKILL.md` Step 7c.

## How to read the numbers, and how to reproduce them

- **Data.** Real human wording, not templates: CLINC150 (`clinc150_full.json`, CC BY 3.0, sha256 `36923c37…`),
  Banking77 (`banking77_train.csv` sha256 `b06e26ac…`, `banking77_test.csv` `d12d6e3b…`, CC BY 4.0). Neither ships in
  the repo; every tool takes `--data DIR`. Hand-labelled sets that DO ship: `tests/data/exchange_direction.json`
  (direction), `tests/data/guard_heldout.json` (the guard), `docs/research/evidence/bench_verifier_runs.json`
  (recorded swarm runs). Every evidence JSON records its inputs' sha256 and the command that wrote it.
- **Environment.** `PYTHONHASHSEED=0 OMP_NUM_THREADS=1`, Python 3.11, NumPy only, a 2-core box. Latency numbers
  carry the machine's load average (other workers were running): read them as ratios.
- **The gate protocol (E0.4)**, used by every serve/abstain number from E1 on: calibrate on val in-scope + a sha256
  half of CLINC's `oos_test`, report on test + the other half; precision under the deployment prior 4,500 : 1,000;
  AURC, precision at 20–60% served, realised precision / coverage / out-of-scope at the calibrated threshold;
  paired bootstrap CI; 3 seeds; the number of configurations tried. Oracle test thresholds only as a symmetric
  diagnostic. WHY: coverage at exactly P 0.97 read **37.7% or 3.3%** on the same index depending on which ~500
  out-of-scope questions calibrated it (val holds 3.2% out-of-scope, test 18%) — reproduced to the digit.
- **The teacher** in every online number is a SCRIPTED stand-in that answers from the dataset's labels (`noisy:r`
  gives a wrong verdict with probability r). It measures the learning loop and its robustness to a fallible
  teacher, never a real model's judgement.
- "PASS" means the backlog's written bar was met on the stated protocol. "FAIL" results are kept and shipped OFF (or
  shipped ON only where the old path was measurably worse on the same denominators — said where it happens).

## Scoreboard

| item | what | result (baseline) | bar | verdict | ships |
|---|---|---|---|---|---|
| E0.5 | decisions + SystemOne tables persist | `decision_outcome(<old id>)` lands after save + load (KeyError) | works after a restart | PASS | ON |
| E0.6 | one meaning of `p` | `p_correct` (high = confident) + `p_null` (low = significant) on every door; bare `p` deprecated | a test per door | PASS | ON |
| E0.7 | calibration per door | ladder veto unchanged while the bridge writes (one shared curve) | ladder untouched | PASS | ON |
| E1.5 | the reflex trace can unlearn | same key 1.000 (0.453); neighbour c=0.8 1.000 (0.900); replay 1.000 | ≥ 0.95 / ≥ 0.85 / 1.0 | PASS (`lms_apa`); `provenance` FAILS 0.780 | ON (`lms_apa`) |
| E0.1 | one harness, pinned inputs | 49 of 49 panel rows reproduced | every baseline to the digit | PASS | tool |
| E0.4 | the gate protocol | implemented; the 37.7% / 3.3% trap reproduced | — | done | protocol |
| E0.2 | routing baseline | top-1 0.4016, answer tier 37.1% at 0.604 (panel 0.402 / 0.371 / 0.604) | recorded | done | — |
| E0.3 | negative-quality flag | teacher margin AUROC 0.936 / 0.41% false flags (draft 0.330) | ≥ 0.8 and ≤ 1% | PASS (margin); draft = kept negative | function |
| E1.1 | SystemOne `scorer="contrastive"` | 0.8263 / 0.8218 (AdaptHD 0.7253 / 0.7377) | ≥ 0.80 | PASS | opt-in |
| E4.1 | the router learns | held-out top-1 0.412 (lexical 0.397; panel 0.402), +1.52 [+0.45, +2.5] | > 0.402, answer acc > 0.604, refusal not lower, thresholds re-derived | PASS | OFF (cost) |
| E1.2 | confusion miner | CLINC precision@50 0.747 vs 0.640 centroid (CI > 0 in 3/3 seeds); Banking77 tie | beats centroid miner | PASS on CLINC, TIE on Banking | in ProtoStore |
| E3.4 | cache `families()` | route_tiered p50 32.7 → 1.43 ms, 0 records differ | ~4 ms, bit-identical | PASS | ON |
| E2.1 | one negative store | one view; 0 of 1,691 negatives same-intent (perfect teacher) | audit false negatives | done | ON |
| E2.3 | noisy teacher by self-agreement | 27.7 / 45.7 / 43.6% correct at ≤ 1.5% wrong, ≤ 2.0% oos (18.0 / 8.1 / 10.5%) | ≥ 30% at ≤ 2.0% / ≤ 2.0%, 3 seeds | FAIL on seed 0 (2 of 3 pass) | ON (better than before on every seed) |
| E1.1c | learned meaning rows | +2.6 to +3.7 points top-1, AURC lower (CI) | + out-of-scope ≤ base + 0.5 pt | FAIL (out-of-scope, every seed) | OFF |
| E2.2 | curriculum | clean: +0.05 / −0.29 points; 10% noise: +0.85 / +1.48 | beat seed spread | clean: no; noisy: yes | OFF (arm) |
| E1.3 | NULL option | `off_domain` verdict field recorded | — | deferred (as planned) | recorded only |
| E1.4 | centering readout | — | — | dropped earlier (kept negative) | — |
| E4.2 | typed learning guard | B/C credential 28/35, live 28/40; fresh held-out 71/91, 247/305; FP 3/705 (old: 47/705) | ≥ 30/35, ≥ 34/40, 0 static FP | FAIL (recall) | ON (fewer false positives on every line) |
| E4.3 | tool choice through ProtoStore | `bench_tool_learning` re-run 2026-09-27: judge calls and solved identical on all 9 judged rows, 0 experience-wrong; restart 214/214 right | not lower, no new wrong calls | PASS (judged arms); the UNJUDGED self-report arm learned more wrong on seed 2 (2 -> 14, see below) | ON (cold start bit-identical) |
| E4.4 | one tool learner | UsageTrace → read-compat shim | — | done | ON |
| E4.5 | `serve` asks the router | 19/20 served, 0 wrong (0/20) | a measured number, 0 wrong above the bar | PASS | ON |
| Q3 | prototypes through the commons | router + tool door shareable; guard-gated; 3 `contribute` gaps fixed | owner: yes, behind the guard | done | ON (per-door opt-in) |
| E3.1 | encoding spec + direction | direction 49/51 (positional 35/51) on ≥ 100 labelled items | beat positional | PASS | ON (meaning door) |
| E3.2 | compose a call nobody handed us | 96/102 never-seen-whole calls right, ECE 0.026; chimeras 0 served (71–74% naive) | ≥ 50% right, measured ECE, wrong never executes | PASS | available |
| E3.2r | resonator noise-tolerant exit | 5% flips: 20/20 (5/20), 0 wrong accepted | fix the exact-only exit | PASS | opt-in (`tolerant=True`) |
| E3.3 | `mind.rank` | static 0.697 (TF-IDF 0.696); learned 0.892 last third (0.690) | latency reported | done; static TIES lexical (loud) | available |
| E5.1 | trajectory encoding | swaps and insert-one-step above the noise floor (AUROC 0.97–1.00) | both distinguishable | PASS, except an insertion at the very END of a long trajectory | available |
| E5.2 | verifier prototypes | AUROC 0.951 (reflex 0.924); novel tasks 0.545 (0.603) | above the reflex on recorded runs | PASS on recorded runs; FAIL on novel tasks | ON (orders on strong evidence only) |
| E5.3 | meaning serves verifiable | `verify_decision(q, row, key="meaning")` vouches after the outcome | vouches | PASS | ON |
| — | phasor `factor(tolerant=True)` | 20/20 at 0.0 / 0.3 / 0.6 phase noise (0/20, 2/20, 0/20) | — | PASS | opt-in |
| — | ledger hooks combine per id | two doors on one id both learn (the second replaced the first) | — | done | ON |
| E6.1 | superposed encoders | n-grams alone TIE or WIN | beat the best single channel | FAIL (kept negative) | available, unused |
| E6.2/3 | CLM as a model-end tier / encoder | contract-tested with a stub (19 tests); no live number | stub + CPU path | done (contract only) | plugin |
| E7.1 | learning curves | ~16 verdicts / row for top-1 0.80 at d 2048 (positive-only rule: ~38) | fit with seed spread | done | — |

## Phase A — substrate truth (the reflex arc's defects first)

**E0.5 Decisions survive a restart — PASS, ON.** The ledger lived in process memory: `decision_outcome` on an id
issued before a restart raised KeyError, and every SystemOne's learned tables were lost. Now `learning_save` writes
`lecore.learning.decisions` (records as TEXT so ids re-derive and a changed id function migrates by replay; the
stored id is checked and kept as an alias on mismatch; newest 4,096; a record carrying a secret is never written),
`learning_load` restores it, `learning_rollover` carries it (`decisions_merged`). SystemOne's learned tables
(prototype matrices, NB counts, isotonic fit, conformal quantiles, the streams) travel in the same section. Hooks are
RE-DERIVED from the record (`meta["hook"]`), not closures. Also fixed: a hook that raised aborted the whole report; a
yes/no decision learned the STRING "no" as yes. Cost (2 cores, load ~7): 4,200 decisions → 4,096 kept, partition
file 246,019 bytes, save 3.2 s, cold load 0.46 s; the ledger's HRR encoding is now lazy (it cost ~14 s to reload
4,096 records at boot). Re-checked on a service over HTTP this sweep: a typed id issued, saved, the process killed
and restarted — KeyError before `boot`, landed after `boot`. KEPT LOUD, same check: a ROUTE outcome after the
restart feeds the route calibrator and the reflex but not the learned router (its live pair hook died,
`"forwarded": null`), and a COMPOSE outcome teaches nothing. FIXED by the learning-loop audit (2026-09-27, `tools/audit_learning_loop.py`, `docs/research/evidence/audit_learning_loop.json`): route and tool-door records now carry a restart-proof hook spec in their meta, and compose and the CLM plugin use the restart-proof `calibrate` spec -- after a restart, an outcome reported by id trains the router, the tool door, the compose calibrator and the CLM calibrator (all four were 0 of 1 before). The door x property matrix (learns live / after a restart / survives a reload / guard holds) is all PASS for 14 doors; `reflex_retile` keeps the trace's outcome fields (before, a retile dropped the failure field's norm from 2.23 to 0.00).
Reproduce: `python -m pytest tests/test_arc_substrate.py tests/test_reflex_bridge_persist.py -q`.

**E0.6 One meaning of `p` — PASS, ON.** `route_tiered`'s catalog path returned a null p-value (low = confident)
while its reflex path returned 1 − error (high = confident). Every door now carries `p_correct` (calibrated
P(correct), HIGH = confident, `None` until that door's own calibrator has labelled outcomes of both kinds) and
`p_null` (a significance p-value where the door has a null, LOW = significant). The bare `p` keeps its old per-door
value for one release and warns when read (`DecisionDict`: JSON and `dict(r)` never warn). Record ids did not
change. Also fixed: the typed reflex turned a calibrated 0.0 into `None`. Reproduce: `tests/test_arc_substrate.py`.

**E0.7 Calibration streams per door — PASS, ON.** One isotonic curve fed by the ladder AND the bridge drove both the
ladder's veto and the bridge's p. Now `door_calibrator("ladder")` and `("bridge")` are separate instances of one
class; pinned: twelve ladder labels, then bridge outcomes — the ladder's pairs, its p_correct at eleven confidences
and its veto are unchanged. The ladder veto is now wired ON LOAD (it waited for 8 new reports after a restart). A
v1 partition's untagged pairs migrate into both streams.

**E1.5 The reflex trace can unlearn — PASS with design (a), ON.** Both designs were built and measured on real
CLINC150 keys (hashed n-grams, dim 2048, 40 background rows per trace, 150 trials per cell;
`bench_trace_correction.json`):

| arm | same key, 1 correction | 3 corrections | flip-flop | neighbour kept, c = 0.5 / 0.8 / 0.9 | background | replay |
|---|---|---|---|---|---|---|
| before (BASELINE) | 0.453 | 0.660 | 0.453 | 1.000 / 0.900 / 0.813 | 1.000 | 1.000 |
| **(a) `lms_apa` (default)** | **1.000** | **1.000** | **1.000** | **1.000 / 1.000 / 0.993** | 0.995 worst | **1.000** |
| (b) `provenance` | 0.993 | 1.000 | 0.993 | 1.000 / **0.780** / 0.533 | 1.000 | 1.000 |

Bar: same key ≥ 0.95, neighbour at 0.8 ≥ 0.85, replay 1.000. The baseline reproduces the panel's 45.3% / 66% / 90%.
KEPT NEGATIVES: the LMS step WITHOUT the affine projection fixes the key (1.000) but the neighbour keeps its answer
0.000 at c = 0.8 — the projection is what protects it; a protection line AT 0.5 fell on the wrong side of float
rounding (moved to 0.4); design (b) without retracting an earlier rejection read flip-flops back 0.373; the cost of
(a) is 0.5% of background rows flipping when a neighbour at c = 0.9 is protected. Raw audit entries replay VERBATIM
(a signed delta pushed back through `write()` would be clamped away on every replay).
Reproduce: `PYTHONHASHSEED=0 python3 tools/bench_trace_correction.py --data DIR`.

**E0.1 One harness — PASS.** `tools/bench_contrastive.py` reproduces every panel row: 49 rows, 47 equal to the
panel's 3-digit figure and the other 2 equal to the panel's own 4-digit probes (the panel rounded 0.1065 and 0.0855
twice). AdaptHD arms at lr 0.3 and 1; Banking77 with a hash-carved val; the learned-index baseline is 0.8413.
Reproduce (`all`, or one section): `PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python3 tools/bench_contrastive.py --data DIR
[--index idx_clinc.json] repro|teacher|learned|prior-trap|gate|negq|miner|curriculum|systemone` (the learned
CLINC index: `python3 tools/bench_meaning.py --data DIR online --model oracle --save-index idx_clinc.json`). The
direction-labelled exchange set (≥ 100 items) is `tests/data/exchange_direction.json`: 179 directional items.

**E0.3 Which negatives to trust — PASS with the teacher margin; the draft is a kept negative.** A negative is the
verdict "q is not A, it is B". On CLINC150 with planted teacher errors (16 configurations tried):

| flag | AUROC at 10% noise (20%) | false flags, perfect teacher | planted caught at that threshold |
|---|---|---|---|
| draft: quality = harmonic mean of hardness and safety | **0.330** (0.324) — worse than chance | 1.2% | 0% |
| teacher margin `cos(q, B) − cos(q, A)`, meaning index | **0.936** (0.929) | **0.41%** | 44% |
| same, hashed n-gram store | 0.939 (0.928) | 0.43% | 38% |
| ECI wording margin (meaning index) | 0.924 (0.908) | 0.65% | 37% |

WHY the draft fails: hardness REWARDS exactly the teacher's errors — a wrong "not A" lands on a question A owns, so
A is very close. `negative_quality` returns the margin as the flag and keeps the quality number only for ranking.

## Phase B — the shared rule on its two strongest consumers

**E1.1 SystemOne `scorer="contrastive"` — PASS, opt-in.** Banking77, 77 intents, 5 examples each, then a 3,000-
verdict one-pass stream through `observe` (decide first, then learn), hashed n-grams d 2048:

| scorer | top-1 seed 0 / 1 | prequential | calibration feature AUROC | run time |
|---|---|---|---|---|
| `prototype` + AdaptHD lr 0.3 (BASELINE) | 0.7253 / 0.7377 | 0.676 / 0.668 | margin gap 0.839 / 0.822 | 2.0 s |
| **`contrastive`** | **0.8263 / 0.8218** | 0.758 / 0.762 | p_top(τ 0.02) 0.873 / 0.882 | 8.1 s |

Bar ≥ 0.80. Under the full gate protocol (seed 0) the same rule's AURC on CLINC150 is 0.077 vs 0.117 (AdaptHD lr
0.3) and 0.141 (lr 1). OFF by default because `prototype` must stay bit-identical for every existing decision, and the rule
costs ~2 ms more per observe (it moves every close rival, not two rows on a miss). With 10% teacher noise the rule
reads 0.783 against positive-only 0.740 and AdaptHD lr 1 0.568 (panel, reproduced).
Reproduce: `tools/bench_contrastive.py --data DIR systemone` (and `gate`).

**E4.1 The router learns — PASS, OFF by default.** `router_learn()` pretrains `ProtoStore("route")` on the catalog's
own alias → card pairs (free, noiseless labels; each alias's rivals = the lexical candidates it meets when it is NOT
in the index); every reported route outcome is one more pair; the learned score reranks the lexical top 30 as
`lexical + λ·cos + μ·prior` (λ 2.0, μ 0.45 chosen on val; 40 configurations tried).

| on 3,954 held-out aliases (hash split) | top-1 | top-5 |
|---|---|---|
| panel baseline (E0.2) | 0.402 | 0.638 |
| lexical on this split (BASELINE) | 0.3968 | 0.6353 |
| **learned router (val-chosen)** | **0.4120** (+1.52 points, CI [+0.45, +2.50]) | 0.6646 |
| usage prior only | 0.4082 | 0.6459 |
| KEPT NEGATIVE: prototypes alone | 0.3561 (−4.07) | 0.6254 |
| KEPT NEGATIVE: untrained store | 0.3483 (−4.86) | 0.6163 |

The tiers' thresholds were re-derived with E0.4 (they did not scale from 893 to 3,952 cards): at P ≥ 0.70 the
answer tier takes 12.5% of in-scope aliases at **0.822** accuracy (the old z tier: 36.9% at 0.596 — above the
backlog's 0.604 bar only at the new threshold), refusals stay 30.8% in-scope and 92.0% out-of-scope, out-of-scope
answered 0.0% (was 0.6%). The re-derived refuse floor (−0.43) refuses exactly what −0.5 refuses (z is discrete),
so −0.5 stands. WHY OFF: pretrain 43–71 s in the bench (45 s at load 0.9 on a service, 10,792 pairs → 3,990 rows),
and the saved partition grows 5.6 MB → 20.8 MB (measured on a service save). A fresh mind routes bit-identically.
Reproduce: `PYTHONHASHSEED=0 python3 tools/bench_router.py --data DIR` → `bench_router.json`.

**E1.2 Confusion miner — PASS on CLINC150, TIE on Banking77.** `ProtoStore.confusions` accumulates `M[y,k] += p_k` on
each verdict. Mean precision@K of the pairs it mines from TRAIN against the pairs actually confused on TEST (3
seeds): CLINC150 K=50 **0.747 vs 0.640** for a centroid-cosine miner (CI above 0 in 3 of 3 seeds; K=100 0.617 vs
0.557, 2 of 3); Banking77 K=50 0.947 vs 0.947 (a tie; CI above 0 in 1 seed at K=100 only). KEPT NEGATIVE (panel):
decision-time reverse normalisation cost −6.0 top-1 points. Reproduce: `tools/bench_contrastive.py --data DIR miner`.

**E3.4 Cache what is slow — PASS, ON.** `route_tiered` rebuilt `families()` on every call. Memoised on the catalog's
content version: over 300 queries at 3,995 cards (load 4.5), p50 **32.7 → 1.43 ms** (mean 34.1 → 3.26, p95 41.2 →
9.7); 0 records differ from the old code (pinned against a copy of it in `tests/test_router_learning.py`).
KEPT NEGATIVE of the CLM speed story at our scale (panel): a candidate-vector cache would save ~1 ms of a 1.55 ms
scorer. Reproduce: `tools/bench_router.py` (latency section).

## Phase C — the teacher: labels, negatives, noise

**E2.1 One negative store — done, ON.** `negatives_report()` shows every negative in one view (meaning vetoes,
prototype negatives, the ladder's bad payloads, the trace's outcome fields). With a perfect teacher 0 of 1,691
stored negatives point at a same-intent row; with 10% noise 305–370 do — the teacher's own errors, which E0.3's
margin flags. KEPT NEGATIVE, the merge rule (a `new` answer whose answer key matches a row joins that row): CLINC150
48.3% → 52.7% served correct, but **Banking77 37.2% → 25.5%** (83 rows instead of 437) — shipped OFF
(`MeaningIndex.MERGE_SAME_ANSWER = False`), the switch kept per mind.

**E2.3 A noisy teacher measured by itself — FAILS the bar on 1 of 3 seeds; ON.** About 3% of typed escalations are
re-asked with the candidates permuted (`MEANING_REASK_RATE`); eps_hat is the flip rate that explains the two answers'
agreement (0.0 until 20 re-asks). With eps_hat > 0 the gate serves on the noise-corrected P(correct) using a
one-sided LOWER bound on the flip rate (`TEACHER_EPS_Z = 1`), rows train on noise-corrected targets, and a lone
`same` picking the 5th candidate or lower is held back from learning (still a calibration label). The sweep-181
teacher-relative ceiling is retired (it read 0.941 / 0.988 for a PERFECT teacher). CLINC150 with 10% wrong verdicts,
model detached after the stream:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| pre-change snapshot (BASELINE), correct | 18.0% | 8.1% | 10.5% |
| **shipped: correct / wrong / out-of-scope served** | **27.7 / 0.5 / 0.5%** | **45.7 / 1.3 / 2.0%** | **43.6 / 1.5 / 1.8%** |
| eps_hat (re-asks) | 0.100 (296) | 0.102 (275) | 0.108 (286) |

Bar: ≥ 30% correct at ≤ 2.0% wrong and ≤ 2.0% out-of-scope, 3 seeds. Seed 0 is 2.3 points short, inside both safety
bars — FAIL, kept loud. Shipped ON because it beats the old code on every seed at every safety number. With a
perfect teacher eps_hat is 0 and nothing moves (CLINC150 48.3% / 1.6%, the pre-change numbers). 5 configurations
tried. KEPT NEGATIVES: the POINT estimate met the bar on 1 of 3 seeds (seed 2 read eps_hat 0.1355 and served bins
~91% right); z = 0.5 on 0 of 2; skipping held-back verdicts as calibration LABELS over-served (60.5% at 3.2% wrong,
4.3% out-of-scope). CAVEAT: the two asks are independent for the stand-in; a real model may repeat its own
mistake. Static probe for comparison: the ε̂-corrected gate serves 20.1% at 0.47% wrong where the uncorrected gate
served 0.33%. Reproduce: `PYTHONHASHSEED=0 python3 tools/bench_meaning.py --data DIR online --model noisy:0.1
--seed S --record` → `bench_meaning_online.json`; `tools/bench_contrastive.py teacher`.

**E1.1 on the meaning rows (`meaning_protos`) — FAILS the out-of-scope bar; OFF.** The rows' learned prototypes
(the live centroid plus a learned delta), one pass and four, 3 seeds, the learned CLINC150 index (0.8413) and a
learned Banking77 index:

| | CLINC150 one pass | Banking77 one pass |
|---|---|---|
| top-1 gain, points (paired CI low) | +2.71 / +3.13 / +2.71 (1.89 / 2.36 / 2.00) | +3.32 / +2.55 / +2.87 (2.04 / 1.15 / 1.59) |
| AURC | lower, CI < 0 on every seed | lower, CI < 0 on every seed |
| precision within 0.005 of target | yes | yes |
| **out-of-scope served at P 0.95** (base) | **4.1 / 3.7 / 4.1%** (2.2%) | **2.3 / 2.5 / 2.5%** (1.2%) |

Bar: CI low > +1.0 on BOTH indexes, AURC lower with CI, precision within 0.005, out-of-scope ≤ base + 0.5 points.
Three of four clauses pass on every seed; the out-of-scope clause fails on every seed (four passes: +3.1 to +3.7
points, the same failure). WHY, from the matched-coverage diagnostic: at the BASE coverage (45.5%) the learned rows
serve LESS out-of-scope (1.0% vs 2.2%) — the calibrated threshold moves and coverage rises to ~63%, and the
out-of-scope rate rises with it. The rows are better; the operating point is not held. OFF until a gate holds it.
Reproduce: `python3 tools/bench_meaning.py --data DIR protos --index idx_clinc.json --index-banking idx_banking.json`.

**E2.2 Curriculum — a test arm, OFF.** Per-row temperature τ(1 + 3/n_y), no pushing of rows with < 3 wordings.
Clean: +0.05 (K=1) and −0.29 points (K=5), inside the seed spread (1.0 / 0.6 points) — CLM's +6.8-point staging gain
did not reproduce. Under 10% noise it helps: +0.85 (K=1, CI excludes 0 on 1 seed) and **+1.48** (K=5, 3 of 3 seeds).
Kept as `ProtoStore.update(curriculum_k0=…)`, off by default until a noisy-teacher door wants it.
Reproduce: `tools/bench_contrastive.py --data DIR curriculum`.

**E1.3 NULL / "none of these" — deferred, the field recorded.** A verdict may now carry `"off_domain": true`; it is
recorded (newest 1,024), never trained on: the absolute floor already catches what a NULL centroid would (+0.8 / 0.0
points, panel), and `new` verdicts must not train it (91% were in-scope duplicates).

## Phase D — the guard and the tools, on the shared rule

**E4.2 The typed learning guard — FAILS its recall bar; ON.** `credential` / `live_value` / `unclear` / `ordinary`
as a typed decision (`unclear` → ask the person); real positives; hard negatives from Banking77 PIN talk and
CLINC150 static-number intents; a fresh hash-split held-out set built BEFORE any tuning (`tests/data/
guard_heldout.json`, sha256 `24ba7f5b…`). Margins and the floor were chosen on the train half only (5-fold CV,
safety +0.03, 15 configurations; `TYPED_MARGIN` credential 0.0456 / live 0.0516, `GUARD_FLOOR` 0.18).

| | old guard (sweep 181) | typed guard | bar |
|---|---|---|---|
| credential, old sets B/C | 24/35 | **28/35** | ≥ 30/35 — FAIL |
| live reading, old sets B/C | 32/40 | **28/40** | ≥ 34/40 — FAIL |
| fresh held-out: credential / live / unclear | 71/91 / 151/305 / 2/39 | **71/91 / 247/305 / 33/39** | 86% / 85% — FAIL |
| false positives, fresh ordinary questions | 47/705 | **3/705** | 0 — FAIL (3 time-word questions: "what time does the pharmacy open on sundays") |
| same, stress answers ("8080", a routing number, "about 45 minutes") | 168/2,115 | 15/2,115 | — |
| false positives the semantic layer ADDS on: the partition / catalog ×3 held-out / tuning / tests + guides facts / labelled static | 0/565 / +4 / +3 / 1/150 / 10/100 | **0/565 / +0 / +0 / 0/150 / 0/100** | 0 — MET |
| verdicts identical after 50 synthetic catalog aliases | — | identical | MET |

Shipped ON because it beats the old guard on every false-positive line and on live / unclear recall, losing 4 on
the old B/C live set. Where recall goes: hand-written live questions generalise poorly (26/66 held out vs CLINC150
221/239) — the guard learns the intents it was shown, not "liveness"; short slang ("eth rn", "sol funding") sits
under the floor. KEPT LOUD: the held-out set and B/C were judged THREE times, not once (run 1 had credential 31/35
before CI tests broke on it); run 3's one change was motivated by run 2's B/C drop, so B/C are not clean for it.
Cost: base build ~3 s and ~165 MB RSS per process (sweep 181: 6.7 s and 877 MB); `decide()` 0.48 ms.
Reproduce: `python3 tools/bench_learnguard.py --semantic` (no datasets), `--cv DATA`, `--build-heldout DATA`
→ `bench_learnguard.txt`.

**E4.3 / E4.4 One tool learner through ProtoStore — done, PASS on the judged arms (re-run 2026-09-27); ON.** `serve`'s tool reflex
picks among learned tools by prototype (`ProtoStore("tool")`, seeded by taught patterns; a verified call pulls; a
correction pulls the truth and pushes the wrong pick as a labelled negative; a failed verify is a negative; an
unjudged call teaches nothing). Cold start (no judged verdict yet) keeps the old two-word overlap pick bit for bit.
UsageTrace became a read-compat shim: an old partition's success / failure counts migrate into the door's audit
tallies, its trace vector does not (a different key space). The backlog's acceptance bench
(`tools/bench_tool_learning.py`: "not lower, no new wrong calls") was re-run on 2026-09-27 at the end of the pass:
judge calls and solved counts are identical on every judged row, 0 experience-wrong, and the restart probe answers
214 of 360 wordings from experience, all right. KEPT LOUD: the routing drift described under the scoreboard moved
many requests from menu to answer, and the UNJUDGED self-report arm learned more wrong answers on seed 2 (2 -> 14). The door is pinned by 16 tests (`tests/test_tool_call_learning.py`: a correction changes the next
pick for a paraphrase, a failed verify moves it, unjudged calls do not, it survives a restart and a legacy
UsageTrace still loads).

**E4.5 `serve` asks the router — PASS, ON.** After a memory miss, a route ANSWER whose gate (z + lead over the
runner-up) clears the threshold re-derived for precision 0.90 (lexical 7.1680; fused, with the learned router on,
7.3469; re-derived on the final 3,996-card catalog, first derived as 6.8685 / 7.0465) — and, once the route door is calibrated, `p_correct ≥ 0.90` — is served as "use capability X" (via
`route`, tier T1, with its id); a menu or refusal escalates and the escalation carries it. The panel's 20 alias
probes: **0/20 → 19/20 served, 19 right, 0 wrong** (1 escalated: "kernel"). KEPT LOUD: with those aliases removed
from the index, **0/20** — no paraphrase clears a 0.90 bar at a router top-1 of ~0.40, so `serve` answers near-exact
capability wording only. KEPT LOUD, a latency regression: the 20-probe serve round trip now reads mean 77 ms, p50
6.1 ms, p95 340 ms, against the panel's pre-change 6.1 ms mean (p50 3.6, p95 15.2) — the router consult runs on every
memory miss. Re-measured after the speed and memory fixes (2026-09-27, in-process, warm, load 0.5): `serve` taught hit p50 5.5 / p95 8.3 ms; `serve` router answer p50 3.0 / p95 6.6 ms; `serve` escalation p50 3.2 / p95 4.2 ms; `route_tiered` p50 1.3 / p95 2.0 ms. The 340 ms p95 was first-call cost, not the steady state: a process's first route builds the catalog (436 ms measured), and the first question of a NEW word count computes the router's null (64 scrambled queries, 231 ms measured for a 9-word question), cached on disk per catalog content after that. `tests/test_external_abstention.py` 33.9 s -> 6.9 s (base 965fdb1: 1.95 s; 62 fresh minds each route once). Reproduce: `tools/bench_router.py` (serve section).

**Q3 Learned prototypes through the commons — done, ON per door.** `contribute` / `commons_pool` carry learned
`ProtoStore`s. `protostore_share(door)` is the owner's per-door opt-in (router and tool door by default); a store
travels only if every question it learned from passed the learning guard and none was session-salted (a prototype
is a sum: one refused question bars it for good). Merging weights rows by verdict counts; prototypes for one label
pointing apart (cosine < 0.3) are FLAGGED, never averaged. Three `contribute` gaps fixed on the way: a vetoed row was
exported, a provisional `model-cached` row could travel, and a row the guard refused inside a bundle was counted as
kept. Pinned: `tests/test_commons_prototypes.py` (7 tests).

## Phase E — holographic candidates and verification (what CLM cannot do)

**E3.1 The encoding spec and direction — PASS, ON in the meaning door.** A method call is a role-filler SUM
(`VERB⊛v + Σ SLOT_k⊛x_k`, HRR at D = 2048, unitary roles); a product form needs permutation roles. Direction lives in
the QUESTION, not the encoding (a swapped call's cosine 0.024, panel), so the context-role reader learns per-slot
prototypes from `from_question` verdicts. On `tests/data/exchange_direction.json` (179 directional items + symmetric
ones; the sweep-181 exchange truth was itself positional and wrong on 29/74 train items):

| | context-role reader | positional (BASELINE) |
|---|---|---|
| held out (51) | **49/51** | 35/51 |
| 5-fold CV, all 179 | 172/179 | 111/179 |
| end to end through `ask()` (51 test): right / WRONG direction / escalated | **27 / 2** / 21 | 18 / **11** / 21 |
| bind level, every test sentence | 41 right / 2 wrong | 29 / 14 |
| symmetric sentences ("the rate between X and Y"), 14 | 13 left open | 12 filled by position |

KEPT NEGATIVES: below ~8 taught examples the reader is WORSE than positional (4 taught: 33/51 vs 35/51); in the
methods mode (29 exchange questions) the reader gives fewer right calls (11 vs 15) and fewer wrong ones (3 vs 5) with
more escalations (15 vs 9) — it buys safety, not coverage there. Found on the way: "20,000" was read as two numbers
and froze a learned method's amount (fixed). Reproduce: `PYTHONHASHSEED=0 python3 tools/bench_rolecall.py`
(`bench_rolecall.json`); `tools/bench_meaning.py --data DIR direction` and `methods --record`.

**E3.2 Compose a call nobody handed us — PASS; available (`call_compose`, `call_from_question`).** Per-role unbind +
cleanup + a CHIMERA check (recompose, explain away, re-read the residual). 500/500 per-role decodes at D 2048 and
1024, robust to noise equal to a term's norm; an equal blend of two calls built a chimera 71.2% (D 2048) / 74% (D
1024) of the time when read naively — checked, **0 chimeras served** (all flagged `ambiguous`). On the direction set
(cross-fitted): 172/179 right (0.961), ECE 0.0097; on the **102 calls never seen whole, 96 right (0.941), ECE 0.026**;
23/23 on new FROM/TO pairs. A proposal is a candidate, never an action (`executes: false`). The resonator's
noise-tolerant exit (agreement + a procedure-matched null, `ResonatorNetwork.factor(tolerant=True)`), on a bound
product at 20×50×50, D 2048, 20 trials:

| flips | before (exact exit only): right, ms | after (tolerant): right / accepted / wrong accepted, ms |
|---|---|---|
| 0% | 20, 62 ms | 20 / 20 / 0, 91 ms |
| 5% | **5**, 422 ms | **20 / 20 / 0**, 81 ms |
| 10% | 6, 579 ms | 19 / 19 / 0, 105 ms |
| random composite | 4,008 ms to "refuse" (no p) | refused in 266 ms, p 0.75 |

Cost: the null fit, 41 s once per codebook shape. Opt-in (`tolerant=True`); the default path is bit-identical.
The phasor resonator got the same fix (`holographic_phasor.factor(tolerant=True)`, now reachable as
`mind.phasor_factor(..., tolerant=True)`): 0/20 → 20/20 at phase noise 0.0 / 0.3 / 0.6 (the default path solved 0,
2 and 0 of 20 even EXACT products at 20×50×50), 0 wrong accepted, a random composite refused in 1.2 s (median p
0.45); its null costs 126 s once per shape under load 7. KEPT NEGATIVE: the old snap rule with 20 restarts and the
agreement exit still solved only 1 of 20 — the exits are not the fix on their own; the FHRR superposition update is.
Reproduce: `tools/bench_rolecall.py`; `python -m holographic.agents_and_reasoning.holographic_phasor --bench`.

**E3.3 `mind.rank(state, candidates)` — done; the static door only TIES lexical (loud).** CLINC150, 4,500 test
questions against the 150 intents, each candidate = its name + 5 training examples, 3 seeds:

| | top-1 | AURC (margin) | off-scope AUROC |
|---|---|---|---|
| word Jaccard | 0.477 | 0.305 | 0.697 |
| TF-IDF word cosine (BASELINE) | **0.696** | **0.097** | 0.809 |
| **rank door (bundle of name + examples)** | **0.697** | 0.106 | **0.859** |
| one concatenated string | 0.673 | 0.121 | 0.844 |
| KEPT NEGATIVE: rolecall `option_encode` via its TEXT role | 0.630 | 0.153 | 0.828 |
| **learning from outcomes, last third** (static on the same stream) | **0.892 ± 0.003** (0.690) | 0.0161 | — |

The floor 0.15 keeps 99.8% of right answers and refuses 6.3% of out-of-scope questions (0.20: 99.1% / 23.9%) — a
gross "nothing here" gate. The conformal-style set covers 0.902 at α 0.1, mean size 1.74. **`p_correct` is
calibrated on the MARGIN, not the absolute top score** — KEPT LOUD because it contradicts the backlog's rule for this
door: for free-form candidates the margin ranks correctness better at every set size (AURC k=2 0.0015 vs 0.0029;
k=5 0.0056 vs 0.0098; k=20 0.0208 vs 0.0314; k=150 0.106 vs 0.139; after learning 0.0161 vs 0.0322, ECE 0.012).
The ORDER, the floor and `p_null` stay absolute; no candidate-relative softmax is served as a probability.
Latency, 150 candidates, 2 cores at load 5–6: warm cache-on p50 10.4 / p95 22.1 ms (n 1,000); cache off 688 ms with
every n-gram atom resident, 5.2 s at the default atom cap; cold first call 6.9 s p50 (atom generation dominates);
the candidate cache is worth ~65× per call. Re-measured after the speed fixes (2026-09-27, load 0.5): 5 candidates warm p50 0.26 / p95 6.1 ms. The shared n-gram atom cache is now an LRU capped at 256 MB (`LECORE_NGRAM_CACHE_MB`): on all 23,700 CLINC150 texts peak RSS 1,513 MB -> 301 MB, output identical, at 1.9x the time on a whole-corpus stream (service-sized use never evicts). Reproduce: `PYTHONHASHSEED=0 python3 tools/bench_rank.py --data DIR` → `bench_rank.json`.

**E5.1 Trajectory encoding — PASS, with one kept negative.** A trajectory is `Σ_t ρ^t(step_t)` over role-filler step
records. Against the same-plan noise floor (200 trials, D 2048, text weight 1.0), detected at 1% false alarms / AUROC:
adjacent swap 0.970 / 0.986 (T=5), 0.865 / 0.980 (T=20); insert anywhere 0.945 / 0.997 (T=5), 0.965 / 0.998 (T=20);
delete one 0.940 / 0.995 (T=5). KEPT NEGATIVE: an insertion at the very END of a long trajectory (T=20: 0.145 / 0.783) and a
changed argument (0.16 / 0.749) hide under incidental text at weight 1.0; at text weight 0.5 both read 0.965 /
0.915 or better. The draft's permutation-only gate passed trivially; this one does not.

**E5.2 Verifier prototypes — PASS on recorded runs, FAIL on novel tasks; ON (orders on strong evidence only).**
`swarm_step` learns every verified step per tool (success AND, new, failure); `verify_precheck` orders on STRONG
evidence (|score| ≥ 0.5) and never skips a verify. On recorded swarm runs (`bench_verifier_runs.json`: 2,691 steps,
425 passing, 3 seeds):

| | verifier score | reflex best (BASELINE; picked post hoc from five) | router rank |
|---|---|---|---|
| AUROC, prequential stream | **0.951** (ECE 0.045) | 0.924 | 0.615 |
| held-out wordings, 5 salts | 0.706 ± 0.097 | 0.675 ± 0.056 | 0.584 |
| **novel tasks** (both wordings held out) | **0.545 ± 0.027** | **0.603 ± 0.058** | 0.604 |

Judge calls to solve the same 425 of 480 tasks: router order 1,469 with the reflex / 2,691 without; the pre-check
1,469 (no cost) / **1,755 (−34.8%)**. KEPT NEGATIVES: reordering by the RAW score cost +7.8% (1,583) — every tool
that ever failed reads negative on any English state; `min_p = 0.05` pre-filtering dropped the truth on 18 of 480
draws (off by default). Reproduce: `PYTHONHASHSEED=0 python3 tools/bench_verifier.py` (`--fresh` redraws the log).

**E5.3 Meaning serves become verifiable — PASS, ON.** A reported meaning decision joins the seen gate under its own
key kind, so `verify_decision(query, row_id, key="meaning")` vouches: re-run on a service this sweep, `valid` false
before `decision_outcome(id, row)`, true after (seen 1.00). The row id never becomes a trace label.

**Ledger hooks combine per id — done, ON.** Two doors can file one decision under one id (an escalated typed answer
and the CLM plugin's record of it did); the second `add()` replaced the first door's hook. Now closures are kept per
code site, specs merge, and `report()` runs every one, catching each. Pinned: `tests/test_ledger_hooks.py` (8).

## Phase F — plugins and learning curves

**E6.1 Superposed frozen encoders — FAIL, a kept negative.** n-gram, word and synonym channels under role keys with
learned GRLVQ-style relevance weights (3 values of the relevance rate tried on val), ProtoStore defaults, 3 seeds:

| | Banking77 top-1 | CLINC150 top-1 | verdict vs n-grams |
|---|---|---|---|
| **n-grams alone (best single)** | **0.8813** | 0.9029 | — |
| superposed | 0.8761 | 0.8999 | TIE (CI spans 0) |
| superposed + relevance | 0.8746 | 0.9021 | Banking LOSS (−0.67 [−1.29, −0.08]); CLINC TIE |
| exact score fusion (not holographic) | 0.8794 | **0.9072** | CLINC WIN +0.43 [+0.01, +0.95]; Banking TIE |
| words / synonyms alone | 0.8466 / 0.8169 | 0.8699 / 0.8592 | — |

AURC ties everywhere. The superposition adds nothing; only an exact (non-holographic) fusion of channel scores wins,
by less than half a point, on one dataset. The encoder stays available (`superposed_encoder`), unused by any door.
Reproduce: `python3 tools/bench_superposed.py --data DIR --cache DIR`.

**E6.2 / E6.3 CLM as a model-end tier and a state encoder — contract only.** The `clm` plugin speaks clm-serve's
`/v1/systemone` and `/v1/rank` over stdlib HTTP, or runs CLM in process (torch on CUDA, and — the owner's rule — on
CPU). Every reply is one more typed verdict: strictly validated (an answer that is not an option, probabilities off
the options or not summing to 1, a choice that is not its own argmax, a candidate list that drops or invents one —
each REFUSED with the reason), recorded, and calibrated by OUR door calibrator (`p_correct` is never CLM's own
probability). `clm_embedder` goes behind `set_embedder`, whose own verify gate decides. Contract-tested against a
stub (`tests/test_clm_plugin.py`, 19 tests); NO live number — no GPU, no torch and no reachable clm-serve here
(`clm_status` says so). The CPU cost figures in the plugin (Qwen3-8B float32 ~33 GB of RAM, ~2–4 s per new state on
a 16-core server) are ESTIMATES, labelled as such.

**E7.1 Learning curves — done.** CLINC150, rows cold-started from one wording, then 1–50 verdicts per row, 3 seeds, d
1,024–8,192, through the E0.4 gate. The shared rule against the positive-only rule (every confirmed wording joins
its row) at d 2,048:

| verdicts / row | 1 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| InfoNCE top-1 | 0.346 | 0.623 | 0.750 | 0.829 | **0.885** |
| positive-only top-1 | 0.346 | **0.663** | 0.733 | 0.777 | 0.807 |
| InfoNCE coverage at ≤ 1.5% wrong | 0.084 | 0.207 | 0.365 | 0.513 | 0.629 |
| InfoNCE model calls per 1,000 | 915 | 816 | 687 | 563 | 469 |

Fit `err(n) = a·n^(−b)`: b = 0.466 at d 2,048 (seed range 0.462–0.469; 0.451 at 1,024, 0.476 at 8,192); verdicts per
row to reach top-1 0.80: **~16** (17.6 / 15.9 / 14.7 / 14.5 across d) against ~38 for the positive-only rule, whose
fit saturates (asymptotic error 0.13–0.15 at d 2,048; the InfoNCE fit's asymptote is 0). KEPT NEGATIVE: at 2–5 verdicts per row the positive-only rule is AHEAD (0.663
vs 0.623 at 5) — the contrastive rule needs about 10 verdicts per row before it pays. Our analogue of CLM's
tokens-per-parameter. Reproduce: `PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python3 tools/bench_curves.py --data DIR --dims
1024,2048,4096,8192` → `bench_curves.json`.

## Kept negatives, in one list (do not re-litigate)

- Design (b) of trace correction, `provenance`: neighbour kept 0.780 < 0.85; the LMS step without the projection:
  0.000 at c = 0.8.
- The draft negative-quality score: AUROC 0.33 — hardness rewards the teacher's errors.
- The router's prototypes alone (0.356) and an untrained store (0.348) lose to lexical (0.397); only fusion wins.
- `serve`'s router consult answers near-exact capability wording only: 0/20 with the probes' aliases held out.
- The meaning rows' learned prototypes serve out-of-scope past the bar on every seed (OFF).
- The merge rule cut Banking77 served-correct from 37.2% to 25.5% (OFF).
- The point estimate of the teacher's noise (1 of 3 seeds) and skipping held-back verdicts as calibration labels.
- CLM's staged curriculum did not reproduce on clean data (+0.05 / −0.29 points).
- The typed guard misses every recall bar and 3/705 ordinary time-word questions are refused as live readings.
- The tool door's acceptance bench was not re-run.
- Static `rank` only ties TF-IDF; `option_encode` costs 6.7 points; the backlog's "absolute top score as the
  confidence" rule loses to the margin on this door.
- `verify_precheck` on novel tasks: 0.545 vs 0.603; raw-score reordering +7.8% judge calls; `min_p` dropped truths.
- The direction reader under 8 examples is worse than positional; in the methods mode it trades right calls for
  safety.
- Trajectories: an insertion at the end of a long trajectory, or a changed argument, hides under incidental text.
- Superposed encoders: n-grams alone tie or win.
- The resonator's old snap rule with restarts and exits: 1/20. The tolerant nulls cost 41 s (bound product) and
  126 s (phasor) once per codebook shape.
- The contrastive rule needs ~10 verdicts per row before it beats the positive-only rule.

## What ships ON, OFF, and why

ON: `p_correct` / `p_null` (bare `p` deprecated one release), persisted decisions + SystemOne tables, per-door
calibrators, `lms_apa` trace correction, meaning serves verifiable, the families memo, `serve` asking the router, the
tool door (cold start bit-identical), guard-gated prototype sharing (router + tool door), the typed learning guard,
teacher-noise correction (3% re-asks, lower-bound eps), the direction reader, verifier learning in `swarm_step`,
ledger hooks combining. New verbs available: `rank`, `verify_precheck`, `call_encode` / `call_compose` /
`call_from_question` / `call_factor_product`, `direction_learn` / `direction_read`, `trajectory_*`, `router_learn` /
`router_mode` / `router_report`, `protostore` / `door_calibrator` / `door_calibration_report`,
`reflex_correction_mode`, `meaning_protos` / `meaning_consolidate` / `meaning_teacher_report`, `negatives_report`,
`superposed_*`, `protostore_share`, `tool_key`, and the `clm_*` plugin verbs.

OFF (or opt-in), each with its reason above: the learned router (cost), the meaning rows' learned prototypes
(out-of-scope bar), the merge rule (Banking77), the curriculum (no clean gain), `SystemOne(scorer="contrastive")`
(the default must stay bit-identical; 4× the observe cost), the resonator / phasor tolerant exits (the default path
stays bit-identical; the null costs seconds once per shape), `verify_precheck(min_p=…)` pre-filtering (dropped
truths), the `off_domain` field (recorded, nothing trains on it).

**Routing drift, measured and kept loud (2026-09-27).** Re-running `tools/bench_tool_learning.py` on its FROZEN
workload moved requests from the menu tier to the answer tier on every seed (no_learning seed 1: answer 21 -> 136)
while judge calls, solved counts and the judged arms' 0 wrong stayed identical. Cause: the catalog's CONTENT grew (44
new faculty / module cards and the longer docstrings of the modules this pass rewrote), which enlarges the router's
null vocabulary and lifts every z by about +0.25 (frozen-workload mean z 0.024 -> 0.261; with the 44 new cards
removed, 0.156 -- the rest is docstring text). Measured on the workload's 360 wordings: answer tier 108 -> 290, answer
precision 0.95 -> 0.98; CLINC out-of-scope + word salad still refused 297/305 (300/305 before). The panel's held-out
alias protocol is unchanged (top-1 0.4011, answer 36.8% at 0.606). The one worse number is the UNJUDGED arm: agents
that believe themselves act on more answers and so learn more wrong ones (seed 2 experience-wrong 2 -> 14; seeds 0/1
1 -> 1) -- the bench's standing lesson that an unjudged outcome must never train a door.

## Landed last: the learning-loop sweep and the speed / memory fixes

- **Restart-proof learning (owner request, 2026-09-27: "make sure our self improvement and self learning process is
  working as best it can").** FIXED by the learning-loop audit (2026-09-27, `tools/audit_learning_loop.py`, `docs/research/evidence/audit_learning_loop.json`): route and tool-door records now carry a restart-proof hook spec in their meta, and compose and the CLM plugin use the restart-proof `calibrate` spec -- after a restart, an outcome reported by id trains the router, the tool door, the compose calibrator and the CLM calibrator (all four were 0 of 1 before). The door x property matrix (learns live / after a restart / survives a reload / guard holds) is all PASS for 14 doors; `reflex_retile` keeps the trace's outcome fields (before, a retile dropped the failure field's norm from 2.23 to 0.00). Also fixed by the same audit: a guard refusal wrote the refused SECRET into
  the saved guard section (now dropped on save and redacted before learning); 60 of 90 ladder payload keys went
  stale after one tile split and 3 corrections vetoed 2 OTHER questions after a restart (now 0 / 0, and fuzzy
  near-repeats served from the reflex rose 9 -> 26); open escalations were lost on restart (now saved, newest 2,048,
  secrets never written); unknown container sections were dropped by a rollover (now written back untouched);
  verify's profile and drift stream, the direction reader and other doors' stores are persisted. The pattern layer
  of the learning guard caught only 4 of 10 ordinary ways of pasting a password; it catches 10 of 10 now with 0
  false positives on every labelled set (secrets 170/170, public 90/90, readings 100/100, static 100/100, partition
  565/565, catalog 9,330/9,330). `learning_save` now returns `sections_changed`; `drift_vs_previous_save` counts
  changed SECTIONS (a save with nothing changed reads 1.0), not how much was learned. Reproduce:
  `PYTHONHASHSEED=0 python tools/audit_learning_loop.py --label after`.
- **Speed and memory.** Re-measured after the speed and memory fixes (2026-09-27, in-process, warm, load 0.5): `serve` taught hit p50 5.5 / p95 8.3 ms; `serve` router answer p50 3.0 / p95 6.6 ms; `serve` escalation p50 3.2 / p95 4.2 ms; `route_tiered` p50 1.3 / p95 2.0 ms. The 340 ms p95 was first-call cost, not the steady state: a process's first route builds the catalog (436 ms measured), and the first question of a NEW word count computes the router's null (64 scrambled queries, 231 ms measured for a 9-word question), cached on disk per catalog content after that. `tests/test_external_abstention.py` 33.9 s -> 6.9 s (base 965fdb1: 1.95 s; 62 fresh minds each route once). Re-measured after the speed fixes (2026-09-27, load 0.5): 5 candidates warm p50 0.26 / p95 6.1 ms. The shared n-gram atom cache is now an LRU capped at 256 MB (`LECORE_NGRAM_CACHE_MB`): on all 23,700 CLINC150 texts peak RSS 1,513 MB -> 301 MB, output identical, at 1.9x the time on a whole-corpus stream (service-sized use never evicts). The typed guard's base build is cached on disk by content
  (`lecore_guard_<sha256>.npz`, `LECORE_GUARD_CACHE=0` turns it off): 3.06 s -> 0.06-0.09 s once cached, 8 cold
  processes 23.6 s -> 3.76 s; a second mind's catalog 0.27-0.34 s -> 0.022-0.028 s. No decision changed: byte receipts
  for Banking77 seed 0, the guard's 500 decisions and the catalog's routes are identical before/after.

## Core memory: what is distilled

Owner direction (2026-09-27): *"The point of the seed memory, and the learning process in general, is to reduce LLM
calls and improve automatic tool calling capabilities ... If a user uses unknown phrasing ... it should be learned
permanently ... If the system learns how to use an api (substituting keys for env variables or something), or a new
method or technique, that's something that should make its way to the seed or core memory."* Two published memories
exist: the seedpack DOCTRINE (`holographic_seedpack.py`, in the wheel, taught at boot: 42 -> 44 rows) and
`release_bundle/` (the distilled partition autoboot falls back to on a fresh clone). Until this change
`tools/distill_release.py` shipped only taught Q&A rows.

### The audit: every learned artifact, where it lives, what it would leak, what it saves

| Artifact | Lives in / persists as | Generalises? | Leak risk | Saves | Distilled now |
|---|---|---|---|---|---|
| Taught Q&A rows | `lecore.learning.taught` (text, replayed) | engine rows yes; domain rows are one deployment's | session salts, paths, keys, history | a model call per repeat | YES, the row rule (unchanged) + a credential VALUE in an answer becomes a `${SERVICE_PARAM}` placeholder first |
| Meaning wordings (confirmed phrasings) | `lecore.learning.meaning` rows[].phrasings | yes: how people ask | a wording can carry a secret / a session salt | a model call per NEW wording | YES for shipped rows and carried method rows, each wording scanned (markers + guard) |
| Learned word associations | `lecore.learning.meaning` assoc | yes | pairs learned from rows that stay home | widens every future question | RE-LEARNED in the bundle from shipped wordings only (never copied) |
| METHOD / CLARIFY rows | `lecore.learning.meaning` rows[kind=method] | yes: verb + slots + words->value maps | a constant could be a key or a user's default city | a model call AND picks the call | YES when confirmed (see the promotion rule); constants as placeholders |
| Exact negatives ("this wording is not that row") | meaning `neg` | yes | a wording | a wrong serve | YES for carried rows |
| Direction reader (ContextRoles) | meaning `direction` / `lecore.learning.direction` | yes: from/to context | sums of context words (no text) | a clarification / a wrong direction | YES when every method wording that could have taught it passed the scan |
| Learned API specs | taught rows `api spec record: <svc>` | yes (public APIs) | keys in URLs / headers; a user's own LAN service | the whole call path | YES: newest per service, auth as `${ENV}` placeholders, never a private host |
| API discoverability cards | taught rows `use the learned api tool ...` | yes | base URL | a model call | YES with their API |
| Tool reflexes | taught rows `toolreflex: <pattern>` | yes | a key in fixed params | a model call AND picks the tool | YES for a carried API, params as placeholders |
| Tool door / router / opted-in ProtoStores | `lecore.learning.tooldoor` / `.router` / `.protostores` | yes when guard-clean | a store is a SUM of question vectors | tool-pick accuracy | YES for stores the commons rule calls shareable + eligible (p33 `_shareable_stores`); non-route stores MERGED by `_protostore_merge`; the router's store rides the commons carrier only (its merge needs the router built) |
| SOPs | KnowledgeStore notes `[sop:<name>]` | yes | keys in shell / python steps | a plan | YES: newest per name, parses, guard-clean, placeholders; not the ones the seedpack installs |
| Distilled workflows | taught rows `what workflow solved X` (the live `_workflow_lib` is in-memory only) | only repeatable tasks | the build's own episodes | a warm plan | YES through the row rule + a new EPISODE rule (dates, backlog, upstream, branch, named third parties); on the live partition 6 passed the row rule and 5 were episodes |
| Reflex bridge experience | `lecore.learning.reflexbridge` | would | question VECTORS without text: nothing to audit | a route per repeat | NO (unauditable) |
| SystemOne tables, decision ledger | `lecore.learning.decisions` | no | one user's decisions about their own states | -- | NO |
| Open escalations, query ledger, tool cache, goals | their sections | no | session facts | -- | NO |
| Guard's learned examples | `lecore.learning.guard` | partly | users' refused questions (redacted) | -- | NO (the shipped examples live in `holographic_learnguard_examples`) |
| Door calibration streams, meaning calibration labels | `.calibration`, meaning `calib` | population-bound | numbers only | the gate's serve rate | NO by default; `--carry-calibration` measured below (it is the difference between 24 and 308 first-contact serves on CLINC150, and a KEPT NEGATIVE on Banking77 online) |
| Teacher-noise estimate, semantic context vectors | meaning `teacher`, `.semantic` | no | -- | -- | NO |

Every carried item and every exclusion by reason is written to `distill_report.json` (`artifacts.<class>.carried /
excluded / items`, `not_distilled`, `promoted`, `env`); `--check` re-derives every class from a shipped bundle
(rows, wordings owned by a shipped row, method rows free of raw credentials, api specs placeholder-clean and public,
reflexes of a carried api, SOPs) and scans every string leaf of the container with the learning guard. A distillation
now BOOTS A TEMP COPY of the partition (a boot may roll a partition's files over; the session partition is never
modified by a distillation).

### Secrets become environment placeholders -- at learn time, on load, and at distill time

`holographic_apilearn` (the one home, reusing the learning guard's detectors): `placeholder_name(service, param)`
(`toyair` + `X-Api-Key` -> `TOYAIR_API_KEY`; `Authorization` -> `SERVICE_TOKEN`), `placeholderize` / `_url` / `_text`,
`resolve_placeholders` (os.environ at CALL time; an unset variable raises `MissingCredential` naming every one).
LEARN TIME: `api_learn` reads the spec's `securitySchemes` (and an optional `auth=`) into a placeholder template; a key
handed once to `api_use(headers=...)` is used for that call and, on success, learned as a placeholder
(`learned_auth`); `tool_reflex_teach` params and a method row's constants (`MeaningIndex.generalize_method`) are
placeholderised. CALL TIME: `api_use` and a method row's in-process tool (`p28 _meaning_run`) resolve from the
environment; missing -> `ok=False` / `via=method-failed` with `missing_env`, and NO request is made. ON LOAD: a spec
record, a tool-reflex row or a method row that still holds a raw key is rewritten to placeholders (the toolbox
rehydrate, the reflex rebuild in serve, `MeaningIndex.from_state`). DISTILL TIME: every artifact is placeholderised
again and the container scan is the second net. A side effect worth naming: a tool reflex taught WITH a key in its
params used to be refused by the guard on its durable row (it lived in process memory and died on restart); it now
persists as a placeholder. KEPT NEGATIVE: a credential under an innocuous name with an innocuous shape (a 10-letter
lower-case token in a param called `q`) is not detectable; the guard is the floor.

### The promotion rule: permanence of what a model taught

A wording linked to a row by a `same` verdict (or a correction by decision id) IS confirmed -- the model judged it
against the row's own evidence -- and ships with its row. A row the MODEL created (`new` verdict: provenance
`model:<by>` / `model-cached`) ships only when a SECOND event confirmed it (a later verdict linked another wording to
it, a positive `answer_feedback`, or a reported outcome on a meaning decision naming it), as provenance `confirmed`.
Method and clarify rows follow the same rule (a single-verdict method is excluded as "unconfirmed (one model
verdict)"). A held-back verdict (noisy-teacher rule) was never linked, so it never ships. Measured cost of the rule:
the CLINC150 learner's 96 model-created rows -> 30 promoted, 66 left home; Banking77 62 -> 10 / 52.

### What it buys a fresh mind (`tools/bench_core_memory.py` -> `docs/research/evidence/bench_core_memory.json`)

Protocol: a mind LEARNS (1 taught wording per intent, then a stream of real training wordings through `ask()` with the
scripted stand-in model end of `tools/bench_meaning.py`), is distilled three ways, and a FRESH mind boots each arm
(`boot(partition=<copy>, doctrine=True)`: the seedpack is in every arm) and meets HELD-OUT test wordings. First
contact = memory alone, model detached (every abstention is a model call a served mind would make); online = the
model attached over 2 held-out orders (a spot-checked serve counts as a model call). L arms run the DOMAIN profile
(anchor off: a CLINC answer is not engine knowledge, so the release rule would carry none of it). 350 s of CPU.
Reproduce: `PYTHONHASHSEED=0 python3 tools/bench_core_memory.py --data DATA --record` (deterministic: two runs
identical).

CLINC150, 40 intents, 800 held-out (600 in-scope + 200 out-of-scope):

| arm | first contact correct / WRONG / oos served | online model calls / 1,000 (WRONG, oos served, mean of 2) |
|---|---|---|
| a nothing | 0 / 0 / 0 | 735.6 (3.5, 1.5) |
| b old bundle (2026-08-26, 497 rows) | 0 / 0 / 0 | 761.9 (3.0, 4.0) |
| c rows-only bundle (before this change) | 0 / 0 / 0 | 725.0 (5.0, 2.0) |
| d extended bundle (after) | 0 / 0 / 0 | 728.8 (5.0, 2.0) |
| L rows-only | 19 / 0 / 0 | 673.7 (3.0, 3.5) |
| L extended | 24 / 0 / 0 | 631.2 (2.5, 1.5) |
| L extended + calibration | **308 / 7 / 6** | **565.6** (5.0, 4.5) |
| the learner itself, never restarted (ceiling) | 312 / 8 / 6 | 558.8 (1 order) |

Banking77, 20 intents, 800 held-out (600 + 200 CLINC out-of-scope):

| arm | first contact | online calls / 1,000 (WRONG, oos) |
|---|---|---|
| a nothing | 0 / 0 / 0 | 704.4 (16.5, 0.5) |
| b old bundle | 0 / 0 / 0 | 753.1 (12.5, 1.5) |
| c rows-only | 0 / 0 / 0 | 721.2 (14.0, 1.0) |
| d extended | 0 / 0 / 0 | 721.9 (14.0, 1.0) |
| L rows-only | 12 / 1 / 0 | 740.0 (8.0, 0.0) |
| L extended | 8 / 0 / 0 | **651.9** (13.0, 2.0) |
| L extended + calibration | **240 / 11 / 0** | 701.2 (7.5, 2.0) |
| the learner itself (ceiling) | 237 / 8 / 0 | 656.2 |

Methods (weather / exchange rate from CLINC150, crypto prices; 130 held-out, scored on the CALL, model attached):
nothing 830.8 calls/1,000 (12 right calls, 0 wrong), old bundle 876.9 (10 / 0), rows-only and extended release
bundles 884.6 (7 / 0), L rows-only 807.7 (14 / 0), L extended 861.5 (9 / 0), **L extended + calibration 561.5 (32
right / 0 wrong)**, the learner itself 576.9 (27 / 2); 7 method rows carried with 172 wordings and the direction
reader, 2 single-verdict methods excluded. Unclear bare tickers acted on: 0 of 6 in every arm.

Tools (a 127.0.0.1 API with an `X-Api-Key` header, 3 taught reflexes, 15 hand-written held-out requests through
`serve()`): every arm but L extended escalates all 15 (0 served); **L extended serves 13 of 15 with the right
argument, 11 the right tool** -- identical to the learner itself (13 / 11); with `TOYAIR_API_KEY` UNSET the same
bundle fails LOUDLY on 13 of 13 matched requests naming the variable, 0 unauthenticated calls. The FAKE key appears
nowhere in the learner's taught rows nor in the bundle's bytes or decompressed text. Carried: 1 api spec
(`env: [TOYAIR_API_KEY]`), 3 cards, 3 reflexes, the tool door's ProtoStore.

Read plainly:
- **Today's bundles carry nothing a fresh user's wording needs** (arms b-d serve 0 at first contact on both datasets;
  their online differences, 704-762, are calibration dynamics of rows unrelated to the questions, not knowledge).
- **Rows alone carry almost none of what a mind learned from real wording** (19 of 800 first-contact serves on
  CLINC150; 12 on Banking77): the rewordings are what was learned, and they were left behind.
- **The extended distillate reproduces the learner** once its calibration travels too (CLINC150 308 vs 312 correct at
  7 vs 8 wrong; Banking77 240 vs 237 at 11 vs 8; methods 561.5 vs 576.9 calls/1,000) and cuts online model calls
  against rows-only by 108 per 1,000 on CLINC150 (565.6 vs 673.7) and 39-88 on Banking77.
- **Tool calling** goes from 0 of 15 served to 13 of 15 for a fresh mind, keys never stored.

KEPT NEGATIVES (loud):
- **Carrying calibration is NOT free: on Banking77 online it costs 49 more model calls per 1,000 than the extended
  bundle without it (701.2 vs 651.9)** while halving wrong serves (7.5 vs 13.0); on CLINC150 it saves 66 (565.6 vs
  631.2) at 2.5 more wrong. First contact it is the whole difference (24 -> 308, 8 -> 240) at a wrong count close to
  the learner's (7 vs 8, 11 vs 8). The likely cause, not yet measured: the labels were earned with the 52 unconfirmed model rows present,
  which the promotion rule leaves home. So `carry_calibration` stays OFF in the release rule (the doctrine: re-derive
  thresholds when the population moves) and is a measured switch (`--carry-calibration`).
- **Without calibration, extended barely beats rows-only at first contact** (24 vs 19; 8 vs 12 on Banking77, i.e. 4
  fewer): the carried wordings sit behind the cold-start gate (score >= 0.80 and lead >= 0.15) until 40 labels of
  both kinds exist. Their value shows online (-42.5 and -88.1 calls/1,000).
- **Extended serves more out-of-scope questions online on CLINC150 with calibration** (4.5 vs 1.5 of 200).
- **On methods, extended WITHOUT calibration is worse than carrying nothing** (861.5 vs 830.8 calls/1,000; rows-only
  807.7): every serve of a carried method is spot-checked by the model until the gate has 40 labels of both kinds, so
  memory that knows the method still pays a call per question -- the carried rows only pay once the labels travel
  too (561.5).
- **Two tool picks are wrong in the learner AND the distillate** ("uv index for zone 11" -> air_quality): the cold
  word-overlap rule ties on "index / zone" and keeps insertion order; the tool door's prototype only overrides it
  after a judged call. The distillate reproduces the learner exactly -- not better.
- **The live partition carries almost none of this**: lecore_memory learned no methods, APIs, reflexes, SOPs or
  shareable stores, and 12 of its 13 confirmed wordings belong to rows the release rule excludes (a 1930s-cel
  technique taught for one picture). The rebuilt `release_bundle/`: 125 rows of 611 (same rows as before), + 1
  confirmed wording ("how should the reflex arc store negative evidence from a wrong answer", served at T0 by meaning
  on a cold boot), + 1 workflow ("simulate the ci inside lecore, find the failures, and fix them"); 5 workflow
  episodes, 4 negatives of rows that stay home and the seedpack's own SOP excluded by reason. `--check`: 126/126
  pass, container scan clean. Cold boot: generic engine questions and the new doctrine answer at T0; the session
  probes ("[s:full-stack] what is the calibration constant of sensor array 7 in bay 0", the kepler-442 probe, the
  cel painting row, "what did cp110 find") escalate.

### Should the bundle ride in the wheel?

Today it does not: `setup.py` ships `lecore_data` (`package_data`: knowledge, definitions, routing, capabilities.json)
and `lecore.autoboot` looks for `release_bundle/` next to `lecore.py` -- present in a clone, ABSENT in site-packages,
so a pip user boots with no core memory at all. Recommendation: yes, it should ride -- it is the distillate meant for
everyone, 0.56 MB, audited by `--check`. The change (not made here; packaging is not this role's):
1. build step: `python tools/distill_release.py --check release_bundle && cp -r release_bundle lecore_data/release_bundle`
   (or keep the one copy under `lecore_data/` and point the distiller's `--out` there);
2. `setup.py` `package_data["lecore_data"]` += `"release_bundle/*.json", "release_bundle/*.lecore",
   "release_bundle/learning/*.lecore"` (explicit globs, the rule the file itself documents; MANIFEST.in already has
   `recursive-include lecore_data *`);
3. `lecore.autoboot`: when no partition is found, fall back to `lecore_data.file("release_bundle")` if the repo-root
   copy is absent -- and SEED a user partition from it (copy to `$LECORE_PARTITION` or `~/.lecore/memory` on first
   boot) rather than mounting the package directory, so a writable site-packages is never rolled over or saved into;
4. CI: `python tools/distill_release.py --check lecore_data/release_bundle`.

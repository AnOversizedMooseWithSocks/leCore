# leCore vs NOOA — verified against the source, 2026-07-26

The Agent Architecture Work Plan framed leCore's agent socket as competing with NVIDIA NOOA
(arXiv:2607.20709, submitted 22 Jul 2026). Every claim the plan made about NOOA was taken on trust. This
note checks them against the paper, and states plainly where leCore stands.

**Headline: the plan's central competitive claim survives. Its characterisation of NOOA does not.**
leCore has one real, measured advantage. NOOA has at least four capabilities leCore does not have at all.
"Everything they do plus more" is **not** true today, and it is worth being precise about why.

---

## 1. What the plan claimed, and whether it holds

| Plan's claim | Verdict | Evidence |
|---|---|---|
| 4,309/4,400 capability records = 97.9% | **CORRECT** | Table 1, verbatim |
| Small models 96.0%, frontier 99.2% | **CORRECT** | §4.1 |
| Agent = Python object; methods are actions, fields state, docstrings prompts, annotations contracts; `...` body filled at runtime | **CORRECT** | Abstract, §1 |
| Publishes no false-action rate and no abstention metric | **CORRECT** | Every evaluation reports pass rate or solve rate: capability tests, SWE-bench, Terminal-Bench, CyberGym, ARC-AGI-3. No abstention metric appears anywhere. |
| "Cannot express *no tool fits*" | **OVERSTATED** | See §2 — NOOA has validated termination, which is a real guard, just not a calibrated one. |
| "Is not reproducible" | **WRONG** | Open-source repo, pinned commits and package versions for all 14 compared systems, public ARC-AGI-3 scorecards, full test suite in-repo. |

So the plan was right about the *metric gap* and wrong about NOOA being weak. It read a systems paper as
if it were a marketing claim.

## 2. The correction that matters: NOOA does guard termination

The plan asserted NOOA "cannot express *no tool fits*." What NOOA actually has is **validated
termination**: the model must return a typed result carrying evidence and a verification command, checked
by the harness before the call returns. The paper's trace analysis makes the point sharply — a comparison
harness "stops whenever the model responds without a tool call," and 77% of its failed Terminal-Bench
trials terminate within ten steps. NOOA's typed contract prevents that class of unsupported completion.

**That is not the same thing as abstention, and the difference is the whole of leCore's edge:**

- **Validated termination** answers *did you prove you finished?* — a gate on the **exit**.
- **Null-referenced abstention** answers *should you have started?* — a gate on the **entry**.

NOOA has the first. leCore has the second and measures it. A task with no tool behind it will still be
attempted by NOOA; it will simply fail its return contract afterwards, having spent the model calls. That
is a real difference and it is worth defending — but it is a narrower claim than the plan made.

## 3. What NOOA has that leCore does not

Audited live against the current tree. Four of six are **absent**, not partial:

| NOOA capability | leCore |
|---|---|
| **Pass-by-reference with bounded previews** — the model gets live Python objects; the prompt carries only type, true length, and a head/tail sample, so a multi-million-row input is processed by writing code against it | **PRESENT since sweep 130.** `mind.bounded_preview` / `mind.value_cost`, and an opt-in `budget` on `/invoke`: over budget you get type + true shape + dtype + head/tail + a `ref` handle to the live value; under it, the previous bytes exactly. Measured 20,269,744 B → 364 B on a 1e6-float array. The handle half (`ObjectRefs`) predates it. |
| **Code as action** — the model writes Python in a persistent REPL, calling methods inline, with variables surviving cell to cell | **ABSENT.** `agent_loop` uses one JSON tool call per turn, the modality the paper argues against directly. |
| **Typed return validation with retry** — invalid return goes back to the model as an error and the loop continues | **PRESENT since sweep 130.** `agent_loop(contracts=)` holds a step's return to a CALLER-declared contract and retries bounded, feeding the verdict back as a typed error. Measured on a 5-executor harness: false passes 2 → 0, at +60% executor calls. False-retry rate on a correct abstention: 0.000. |
| **Long-term memory subsystem** — seven model-callable tools, ACT-R activation ranking, decay-based forgetting, asynchronous consolidation, one inspectable SQLite file; **measured +11.8 RHAE points** over the same agent with markdown notes | **PARTIAL, materially narrowed in sweep 131.** `mind.memory_curate` adds a use journal on a LOGICAL clock over the real partition, ACT-R activation ranking (`holographic_actr` existed and had nothing to point at), plan/apply/reflect, and an archive — **no delete path anywhere**, with a pinned conservation law `rows_after + archived == rows_before`. Still missing NOOA's asynchronous consolidation and its model-callable tool surface. And the result is SELF-MEASURED and honestly mixed: +0.04 hit rate over a recency window at drift 0.00–0.75, and **−0.031 [−0.041, −0.020] — a significant LOSS — under a full topic switch**. |
| **Sandboxed execution** — Landlock filesystem default-deny, seccomp network block, per-cell timeouts, 18-pass red-team audit with zero leakage | **ABSENT** — though leCore also never executes model-written code, so the exposure differs. |
| **Model-queryable event history** | **PARTIAL** — `ask_traced` exists. |

And NOOA is measured on **real agentic benchmarks**: SWE-bench Verified 82.2%, Terminal-Bench 2.0 73.0%,
CyberGym L1 86.8% (top open-source, beating most closed systems), ARC-AGI-3 85.1% at under $20/game.
leCore's benchmark measures leCore's own catalog. Those are not comparable numbers, and leCore currently
has no result on any external agentic benchmark.

## 4. Where leCore genuinely leads

One thing, measured, and it is not nothing:

- **False-action rate 0.0%** on a no-tool set built by **removal** — each task is a real capability's own
  author-written alias, asked against an index rebuilt without that capability, with every near neighbour
  left in place to tempt a match. 60 has-tool tasks resolve at 100.0%. Run-to-run variance is exactly
  zero. Model calls: zero.
- **Calibrated, null-referenced abstention** — the floor is a distribution of maxima built from the
  catalog's own vocabulary at matched token count, so the catalog-wide argmax is priced in by
  construction, and the reported p is empirical rather than a normal approximation.
- **Determinism as a hard guarantee**, not a property of a fixed seed: `max_rung=5` means no model is
  reachable, and the resolution cache refuses to store any non-deterministic result.

Also worth stating: the published literature supports putting that gate *below* the model rather than
inside it. Abstention benchmarks find reasoning fine-tuning **degrades** abstention, which is the argument
for a deterministic floor the model cannot talk its way past.

## 5. Honest position

leCore is **not** "everything NOOA does plus more." It is a different bet:

> NOOA maximises what a capable model can do through a clean interface.
> leCore maximises what can be answered **without** a model, and refuses — measurably — when nothing fits.

Those are complementary, and the overlap is smaller than the plan implied. leCore's socket is genuinely
better at *not acting*. NOOA's is genuinely better at *acting*, and has the external benchmark results to
show it while leCore has none.

## 6. If the goal is parity plus the abstention edge

Ordered by value per unit of work, and the first two are the ones that actually matter:

1. **Bounded object previews + pass-by-reference** in `agent_loop`. The single biggest capability gap and
   the most self-contained: keep live objects in a session namespace, render type + length + head/tail
   into the prompt. Everything downstream (large inputs, fewer tokens, fewer turns) follows from it.
2. **Typed return validation with retry.** Small, and it closes the *exit* gate leCore currently lacks —
   pairing it with the entry gate leCore already has would be strictly ahead of NOOA on both.
3. **An external benchmark result.** Without one, every leCore number is self-referential. Terminal-Bench
   is the most tractable; SWE-bench needs repository tooling leCore does not have.
4. **Code-as-action** — large, and it collides directly with the no-`exec`-REPL decision already on
   record. Do not start it without revisiting that decision explicitly.
5. **Memory curation/decay/reflection** — NOOA measured +11.8 RHAE for it, which is the strongest
   published evidence for any single item on this list.

**Do not** claim parity in the meantime. The measured abstention result is strong and defensible on its
own; attaching it to an overstated comparison is the fastest way to lose the argument.

---

## 7. Status of this plan — updated 2026-08-31 (sweeps 130-131)

| item | state |
|---|---|
| 1. Bounded object previews + pass-by-reference | **DONE** (sweep 130) |
| 2. Typed return validation with retry | **DONE** (sweep 130) |
| 3. An external benchmark result | **HARNESS BUILT (sweep 135), result still open** — see §8 |
| 4. Code-as-action | **NOT STARTED, deliberately** — it collides with the no-`exec`-REPL decision, and that decision has not been revisited |
| 5. Memory curation/decay/reflection | **PARTIAL** (sweep 131) — built, self-measured, and it loses to a recency window under a full topic switch |

**Items 1, 2 and 5 were all measured on leCore's own harnesses.** That is exactly the criticism §3
made of leCore's benchmark, and closing three capability gaps does not answer it. Until item 3 lands,
every number in this document except the NOOA figures is self-referential, and the honest summary of
sweeps 130-131 is "the capability table is less lopsided", not "leCore has caught up".

### What item 3 should actually be, now that the field has moved

The agent-memory benchmarks that matter are **LoCoMo**, **LongMemEval** and **BEAM**. Mem0's published
bests: LoCoMo 92.5%, LongMemEval 94.4%, BEAM-1M 64.1%, BEAM-10M 48.6%.

**LongMemEval is the on-target one, and for a specific reason: abstention is one of its five measured
capabilities** — "correctly declining to answer about events that never occurred". That is leCore's
single genuine lead, defined by someone else, on someone else's data. Every abstention number in this
repo today comes from a no-tool set built by REMOVAL from leCore's own catalog; LongMemEval's does not.
It is the cheapest way to convert the one real advantage from a self-referential claim into a citable one.

Two honest obstacles, both worth stating before anyone starts:
- A full LongMemEval run needs a model rung. leCore's abstention arm is model-free by design
  (`max_rung=5`), so the comparison has to be scoped carefully or it measures the model, not the gate.
- The field's own survey notes there is **no widely adopted public benchmark that scores forgetting,
  decay or consolidation directly**. So item 5 cannot be validated externally at all, today. It must
  stay labelled self-measured — which it is, in the module docstring, the catalog card and every
  returned dict.

---

## 8. The correction sweep 135 had to make to §7, before it could build anything

§7 named LongMemEval as "the on-target external benchmark" because abstention is one of its five
measured abilities. That is right about the benchmark and **was wrong about which of leCore's gates
it measures.** leCore has *two* abstentions and they are not the same instrument:

| | what it refuses | the number this repo quotes |
|---|---|---|
| `route_or_abstain` | "no **capability** matches this request" | false-action **0.0%**, `agent_benchmark` |
| `serve` | "I hold no **fact** that answers this" | not previously benchmarked |

**LongMemEval's abstention ability is the second one.** Its questions ask about events that never
happened in a user's chat history — ordinary English about a life, not about a capability catalog.
Measured before building: `route_or_abstain("What did I say about my sister's wedding in March?")`
abstains at **z = −1.69**, and a catalog-vocabulary control abstains too at z = −1.15. Pointing the
routing gate at that benchmark would abstain on **100%** of it, scoring a meaningless perfect on the
abstention split and zero on the answerable one. That is a category error, not a result.

So the headline this document leads with — 0.0% false-action, calibrated null-referenced abstention —
**is not the property an external memory benchmark would validate.** It remains true and remains
self-referential; item 3 is not closed by measuring a different gate well.

### What sweep 135 actually built

`mind.external_abstention(records)` (`holographic_extbench`) drives the **memory** gate over a task
file in LongMemEval's published schema, honouring their `_abs` question-id convention, with a fresh
mind taught per record so no fact leaks between questions. The task file is a **parameter** — that is
the whole point, and it is the first leCore benchmark whose questions leCore did not write.

**The real 500-question set was NOT run here.** What ran is a corpus built to the published schema.
The harness is proven; the score is not claimed.

### And the caveat that decides whether the number would mean anything

On a fixture shaped like LongMemEval's *other four* abilities — knowledge-update, multi-session,
temporal — the T0 memory answered the single-session lookup and declined the rest:

```
recall 0.25 | abstention 1.00 | false-answer 0.00 | PAIRED 0.40
```

**A system that answers nothing scores 100% abstention.** Both halves of that line are true and only
the pair is honest, which is exactly what `paired_benchmark` already argues internally ("a PAIR counts
only if BOTH are right") — carried out to an external corpus, where it matters more, not less. Mem0's
published LongMemEval **94.4%** is dominated by recall, so the comparison leCore has to survive is the
paired one, and on this evidence **recall is the binding constraint, not abstention.**

That reframes item 3: the cheapest honest external result is no longer "show the abstention lead on
someone else's data" — it is "show the pair", and the pair needs retrieval work this engine has not
done. A test (`test_THE_CAVEAT_a_system_that_answers_nothing_scores_perfect_abstention`) pins the 0.25
so nobody can quote the abstention rate without meeting the recall rate.

### §8b — sweep 136 did the retrieval work, and measured what it costs

A retrieval rung was added after `serve` declines (`retrieve="semantic"` on a cosine floor, or
`"bm25"` on a raw Okapi floor). Default off, so the sweep-135 numbers reproduce exactly. On a
4-answerable / 4-abstention fixture — **wider than sweep 135's, so its baseline paired rate is 0.25,
not 0.40; a rate is only comparable within one corpus digest**:

| rung | floor | recall | abstention | false-answer | **paired** |
|---|---|---|---|---|---|
| none | — | 0.25 | 1.00 | 0.00 | **0.25** |
| semantic | 0.20 | 0.75 | 0.00 | 1.00 | 0.00 |
| semantic | 0.45 | 0.50 | 0.50 | 0.50 | **0.50** |
| **semantic** | **0.50** | **0.50** | **0.75** | **0.25** | **0.50** |
| semantic | 0.70 | 0.25 | 1.00 | 0.00 | 0.25 |
| bm25 | 3.0 | 0.50 | 0.75 | 0.25 | **0.50** |

**The rung doubles the paired rate, and it buys that with abstention.** leCore's 0.00 false-answer
rate was purchased at recall 0.25; spending a quarter of the abstention buys a doubling of the pair.
That is the exchange rate, and it is the first time this repo has priced its headline property.

Two things the curve says that a single number would not. **The floor is the whole design** — at 0.20
the engine answers every abstention question (it invents), at 0.70 nothing clears the bar (it
declines), and only the interior is useful. And **bm25 reaches the same peak at floor 3.0 rather than
0.50**, because its scores are raw and corpus-dependent: two rungs agreeing on the peak is evidence
the peak belongs to the corpus rather than to one scorer, and floors differing by 6× is the warning
that neither floor is a constant anyone should copy.

`paired_rate` is a **proxy**: LongMemEval ships no paired instances, so this pairs over the split —
how many complete (answered, declined) pairs the two halves can form. It is the honest reading
available from an unpaired corpus and it is not the same statistic as a benchmark that ships twins.

Item 3 remains open: the real 500-question set has still not been run, and 0.50 paired on an
eight-question fixture is a mechanism result, not a score.

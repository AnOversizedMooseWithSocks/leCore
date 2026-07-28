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
| **Pass-by-reference with bounded previews** — the model gets live Python objects; the prompt carries only type, true length, and a head/tail sample, so a multi-million-row input is processed by writing code against it | **ABSENT.** `agent_loop` passes JSON args. This is the property that lets NOOA "scale past the context window", and leCore has no analogue. |
| **Code as action** — the model writes Python in a persistent REPL, calling methods inline, with variables surviving cell to cell | **ABSENT.** `agent_loop` uses one JSON tool call per turn, the modality the paper argues against directly. |
| **Typed return validation with retry** — invalid return goes back to the model as an error and the loop continues | **ABSENT.** `agent_loop` accepts whatever `invoke` returns. |
| **Long-term memory subsystem** — seven model-callable tools, ACT-R activation ranking, decay-based forgetting, asynchronous consolidation, one inspectable SQLite file; **measured +11.8 RHAE points** over the same agent with markdown notes | **PARTIAL.** `recall` exists; the curation/decay/reflection subsystem does not. |
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

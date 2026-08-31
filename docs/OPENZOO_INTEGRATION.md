# openzoo.fun integration guide — and the same features for local runners

*Everything here is measured, not promised. Where a number appears, it came from a
harness in this repository that you can re-run.*

## 1. What you get by adopting the current build

**One-call boot.** `lecore.autoboot()` finds the partition (`$LECORE_PARTITION` or the
conventional path), boots doctrine + external memory, attaches a model rung when one is
reachable, and opens a session. `memory=False` opts out; `session="name"` opens a fresh
context. Point `LECORE_PARTITION` at your seed (below) and the server is personalized.

**The chat front door** (`run.bat` / `run.sh` → `chat_server.py` at 127.0.0.1:7860):
substrate-first answers with provenance chips, teach/veto in conversation, workspaces,
memory slots (`load memory <path>`, `ask <slot>:`, `compare:`), interactive void
exploration (`explore` → `test N` → `promote N`, real nulls only), scene/texture/palette
creation through the real API, `learn api:` / `use api:` / `find a tool for`, and
`health` for the saturation estimate. Model settings: none (default) / local mini /
any Ollama-compatible endpoint.

**The hosted MCP surface** (26 tools): `zoo_tools` now carries `op=status`
(saturation + taught rows + registered services) alongside `find`/`call`. The SSRF
boundary is enforced and stated in the tool description: hosted callers cannot
register URLs; only operator-registered services are callable.

## 2. Cost-saving levers (why the substrate saves you money)

- **Memory-first answering.** A memory hit costs zero model tokens. The pinned bench
  holds 0.75 hit rate at 1.00 answer quality with 0.25 honest escalation; the
  LongMemEval-protocol harness scores 1.000 across all six categories **with no model
  attached at all**.
- **One generation per unique question, ever.** Model answers cache with
  `model-cached` provenance and serve from memory on repeat.
- **The grounding gate.** Ungrounded model output (no shared substantive token with
  the question) is refused instead of served — no garbage reaches users, no paid
  retry loop. Exact taught/validated/evidenced answers bypass the gate (an exact
  repeat is already the strongest grounding).
- **Honest abstention is free.** "I don't have that in memory" costs nothing and
  keeps trust; the alternative is a paid hallucination.
- **Deterministic tool routing.** `tool_find` and api discoverability cards route
  tasks to tools by memory recall, not by model mediation — zero tokens per route.
- **Distilled seed boots.** The shipped bundle is 58 generic entries distilled from a
  747-row working partition (every exclusion itemized by reason) — small, private-data
  free, fast to load.
- **Decision-safe quantization.** `save_level="auto"` picks int8/float32 per array
  only where geometry survives (the binary-quantization negative is documented, not
  repeated) — roughly 4x smaller state at zero decision cost.
- **Right-sizing memory.** `saturation_estimate` (chat: `health`; hosted:
  `zoo_tools op=status`) reads recall margins, which measurably move BEFORE accuracy
  falls (corr 0.733 vs null 0.023, p=0.0) — an early warning, not a post-mortem.

## 3. How self-improvement and learning actually work

leCore learns by **teaching, not training**. `teach(q, a)` stores a fact that serves
with provenance `taught`; a veto tombstones it durably (survives restart; a deliberate
re-teach lifts it); sessions isolate what should not be shared; everything replays
deterministically from the saved state. On top of that sit three escalating rungs of
*earned* knowledge:

1. **conjecture** — the void explorer proposes structure between concepts
   (`explore`, ranked by a real pairing-null p-value);
2. **validated** — an experiment passed (`test N` runs the hypothesis engine against
   the actual null draws; failures are kept and say so);
3. **evidenced** — research or a capable model connected the result to the
   literature (`promote`).

The drift sentinel watches every teach for conflicts with established memory
(redshift/echo/void verdicts, advisory — the human decides). The model rung, when
attached, is **last and visible**: its words arrive marked `model-cached`, and vetoes
kill them like anything else. Nothing in the loop requires a model; everything in the
loop is improved by one.

## 4. Seeds: personalizing leCore without touching code

A **seed** is a small memory bundle taught from your facts and loaded at boot.

    # facts.md — lines of "question = answer"
    what is our refund policy = 30 days, no questions asked
    who do i contact for billing = billing@yourco.example

    python tools/make_seed.py facts.md my_seed/
    # -> verifies from a fresh boot: every fact answers at T0,
    #    a held-out probe must escalate; leaky seeds are refused

Load it: `lecore.autoboot(partition="my_seed")` locally, `LECORE_PARTITION=my_seed`
for a server, or `load memory my_seed` in chat as a named slot. JSON input
(`[{"q":..,"a":..}]`) works the same. For a richer seed distilled from a working
partition, `tools/distill_release.py` is the template — it selects by rule and
reports every exclusion so the distillation cannot become a blind spot.

## 5. Local runners get the same everything

Attach your model in chat settings (`none` / local dir via ModelRung / Ollama URL via
LocalRung), or in code: `lecore.autoboot(llm=my_callable)` — any `prompt -> text`
callable is a valid rung. The ladder's contract does not change with the backend:
taught beats model, provenance is always visible, vetoes stick, escalation is honest.
The tiny-model live test proved the wire: an untrained mini's token-noise flowed
through honestly labeled and veto-killable — a capable model inherits the same
contract with better words.

## 6. Verify your integration

    python tools/swarm_audit.py        # 12 organ groups, above/below matrix, 0 unintended gaps
    python tools/bench_ladder.py       # pinned 0.75 / 1.00 / 0.25
    python tools/bench_longmem.py      # 1.000 across six categories, no model
    python tools/unicron_preflight.py MODEL_DIR   # before any in-model install

## 7. Memory portfolios: multiple memories, sharing, and knowledge transfer (cp69)

Keep as many named memories as you like (`memory_list`; chat: `memories`) — one for
research, one per project, one somebody shared. **Selective export** (`memory_export`;
chat: `export memory <dir>: <filter>`) writes a portable, self-contained bundle holding
only what the filters admit, verified from a fresh boot before it is blessed. What
travels: facts with provenance, conjectures **at their earned rung**, learned api tools
(callable on arrival — functionality transfers, not just text), and your vetoes
(tombstones ride along; a shared bundle never resurrects what its maker killed).
**Import/merge** (`memory_import`; chat: `import memory <path>`) brings a bundle into
an existing memory: identical entries skip, conflicts are **flagged with the drift
sentinel's verdict** and your local answer wins unless you say `theirs`. This is also
the seed workflow's other half: synthesize or curate what you want to share, export it
generically, and hand the directory to anyone.

## 8. Speed tiers: bypassing and shortcutting model calls (cp71)

Three tiers, cheapest first, all measured on the reference harness:

1. **LLM bypass (memory-first)** — a T0 memory serve measured **97x faster** than a
   full 8-token generation (3.5 ms vs 340 ms). Every taught fact, cached model
   answer, and seeded entry is a model call that never happens. This is the lever
   that scales with usage: the more a deployment is taught, the less it generates.
2. **Address shortcuts (truncated schedules)** — when automatic attribution measured
   a cue's crystallization layer as early, repeat generations run only that many
   layers: measured **3.0x at exit-L7, 1.7x at L14, 1.2x at L21** on a 28-layer
   model, scaling exactly with layers skipped. The shortcut fires only for
   addresses that measured early AND whose truncated answer agreed with full depth
   (the 2026 early-exit cautions, enforced per cue).
3. **Full generation** — the honest default, with the logit lens piggybacked on the
   same forward pass, so every model answer leaves a source address behind at
   near-zero extra cost.

**Automatic, with opt-out:** locally, attach `RuntimeRung(model_dir, mind=m)` (the
chat's "local mini" setting now does this) and attribution + shortcuts are on;
disable with `attribution=False` or `LECORE_NO_ATTRIBUTION=1`. Hosted: attribution
needs model-directory access, so openzoo operators enable it server-side on their own
runtime; substrate-only deployments already enjoy tier 1 in full. The decision
journal (`decision_report`) shows your live arm distribution -- watch the model arm
shrink as memory grows.

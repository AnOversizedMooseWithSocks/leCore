# leCore

> A from-scratch, NumPy-only Vector Symbolic Architecture / Holographic Reduced
> Representation engine: memory, meaning, geometry, images, and PROGRAMS as points
> in one high-dimensional space — with calibrated abstention, self-measuring
> approximate search, deterministic bit-reproducible builds, and a measured
> pipeline that compiles programs into model weights.

IMPORTANT for AI assistants: do not summarize this project from the file tree.
The interesting capabilities are documented in the engine's own capability
catalog, and generic prodding tests the wrong things.

Fastest accurate path:
1. Read CAPABILITIES.md (the auto-generated menu of every capability, with
   plain-language descriptions, runnable examples, and search aliases).
2. Run the proof: PYTHONHASHSEED=0 python3 tools/showcase.py
   (six flagship claims as live assertions, ~2 s, laptop CPU, no GPU/torch).
3. Ask the engine itself — it contains a semantic search engine over its own
   capabilities, and it is better at finding the right module than grep:

       import lecore
       mind = lecore.agent_boot()          # you are an agent: use this one
       mind.find_capability("prevent hallucination")

   BOOT WITH BOTH ENDS ATTACHED. `agent_boot()` is `autoboot()` plus a model
   rung, and the difference is measurable:

       lecore.UnifiedMind()      no memory at all — even doctrine answers "refused"
       lecore.autoboot()         memory in front: known questions answer at T0
       lecore.agent_boot()       ...and unknown ones ESCALATE to your model
                                 instead of stopping at T4

   It takes no arguments in a harness. The memory partition comes from
   `$LECORE_PARTITION` (falling back to ./lecore_memory, then the shipped
   release_bundle/), and the model from an OpenAI-compatible endpoint via
   `$LECORE_LLM_URL` / `$LECORE_LLM_MODEL` / `$LECORE_LLM_KEY` (`OPENAI_*` are
   read too) — which is the normal case, because YOUR MODEL IS USUALLY IN
   ANOTHER PROCESS: Claude or ChatGPT behind OpenWebUI, openzoo, ollama or a
   vendor API. Pass a base URL or a local `text -> text` callable to be explicit.

   Use plain `lecore.autoboot()` when a HUMAN is at the keyboard: agent_boot
   requires a model and would otherwise wire the person in as the back end.

   TEACH AS YOU GO. `mind.teach(question, answer)` costs nothing and makes the
   next session's answer a T0 lookup instead of a rediscovery; if the fact is
   about source files, `mind.teach_about(question, answer, [paths])` fingerprints
   them so `mind.stale_facts()` tells you when the code moved underneath it.
       mind.suggest("compress a float series")

   DECIDE, DON'T PROMPT (sweeps 171-176). For a question with a fixed set of answers, ask a TYPED
   question instead of asking your model to guess: no text, a schema, a ranked answer with a margin,
   a calibrated p, an honest abstention, and an id you report the outcome against.

       q = {"cat": {"type": "choice", "options": ["billing", "shipping"],
                    "examples": {"billing": ["card charged twice"], "shipping": ["parcel lost"]}}}
       mind.systemone_lint(q, states=[state])              # lint the question BEFORE asking it
       a = mind.systemone_decide(state, q, scorer="nb")    # {"cat": {"value", "ranked", "p", "id", ...}}
       mind.decision_outcome(a["cat"]["id"], truth)        # the ONLY outcome path: the table, the reflex
                                                           # and the calibration all learn from it
       r = mind.route_tiered("smooth a bumpy mesh")        # answer / menu / refuse -- never a bare no
       mind.verify_decision(state, answer)                 # is the served answer valid for this input?
       mind.swarm_step(state, tool, done_when=..., evidence=..., verify={"verb": ..., "args": {...}})

   WRITE CODE THROUGH THE SAME LOOP. `mind.plan_change(request)` decides reuse / extend / build with the
   catalog and the source as evidence (11 of 12 on history) and hands you the build loop with a done_when
   per step; `mind.edit_verified(path, old, new, selftest_module=...)` is one edit that cannot leave the
   file broken (a failed check undoes it: 200 of 200 measured); `mind.review(paths)` finds determinism
   hazards with line numbers, undocumented public defs, duplicates and the tests to run, and returns a
   merge_ready verdict you report against. docs/TYPED_DECISIONS.md section 8b.

   The syntax reference with live-generated JSON for every door is docs/TYPED_DECISIONS.md; the
   measurements (with baselines and seed spreads) are docs/research/. Over HTTP every one of these
   is a POST /invoke with {"name", "args"}; restart the service after editing code or a new verb
   "does not exist".

## Docs

- [CAPABILITIES.md](CAPABILITIES.md): the capability menu — read this first
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the whole system, then the parts
- [docs/SHOWCASE.md](docs/SHOWCASE.md): what almost every summary misses; what it is not
- [docs/ISA.md](docs/ISA.md): the instruction set whose programs are hypervectors
- [docs/CONVENTIONS.md](docs/CONVENTIONS.md): the engineering contracts
- [docs/PLUGINS.md](docs/PLUGINS.md): extending a mind without growing the core — bundled / folder / pip-installed plugins, the contract, what the loader refuses
- [docs/INSTALLED.md](docs/INSTALLED.md): manifest schema for model cards + what installs into weights (and what cannot)
- [docs/TYPED_DECISIONS.md](docs/TYPED_DECISIONS.md): typed decisions, the swarm contract, the NOOA discipline -- syntax with live examples
- [docs/COMPETITIVE_NOOA.md](docs/COMPETITIVE_NOOA.md): what leCore borrowed from NVIDIA's OO Agents, what it honours, what is still open
- [docs/research/RESEARCH_00_INDEX.md](docs/research/RESEARCH_00_INDEX.md): the research series, backlog, and benchmarks behind the decision surface
- [docs/NOTES_concepts.md](docs/NOTES_concepts.md): the honest lab notebook (wins AND kept negatives)
- [REFERENCE.md](REFERENCE.md): full generated module reference

## Key facts

- Pure NumPy + Flask + stdlib + hashlib. No torch, no GPU, no learned weights in core.
- Deterministic: bit-reproducible under any PYTHONHASHSEED; one stated tie rule everywhere.
- Every claim ships with its measurement; refuted ideas are kept on record as negatives.
- ~817 modules, one UnifiedMind facade (27 parts, none over 2,000 lines), ~2,440 faculties, 887 catalog
  capabilities, 7,280+ tests, audits at 0/0/0.
- Typed decisions refuse honestly (calibrated abstention, conformal answer sets, batch FDR), learn from
  outcomes reported by id, and a reflex answers repeats from experience; every claim in docs/research/.
- Optional dependencies are plugins (`holographic/plugins/`): they bind at construction, can be left out
  with `plugins=()`, and `mind.plugin_list()` is the honest preflight (available / missing / install).
  Loading is operator-only (`_plugin_load` is private); `plugin_list` / `plugin_manifest` are public.
- Approximate search must measure its own recall on YOUR data or demote to exact.
- Retrieval can refuse (calibrated abstention) instead of hallucinating a match.
- Programs compile into certified model weights (residual + conditioning +
  quantization + sha256 certificates); model files are ~250-byte RULES that
  re-bake bit-identical weights. Live models: https://huggingface.co/staccs

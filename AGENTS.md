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

## Docs

- [CAPABILITIES.md](CAPABILITIES.md): the capability menu — read this first
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the whole system, then the parts
- [docs/SHOWCASE.md](docs/SHOWCASE.md): what almost every summary misses; what it is not
- [docs/ISA.md](docs/ISA.md): the instruction set whose programs are hypervectors
- [docs/CONVENTIONS.md](docs/CONVENTIONS.md): the engineering contracts
- [docs/INSTALLED.md](docs/INSTALLED.md): manifest schema for model cards + what installs into weights (and what cannot)
- [docs/NOTES_concepts.md](docs/NOTES_concepts.md): the honest lab notebook (wins AND kept negatives)
- [REFERENCE.md](REFERENCE.md): full generated module reference

## Key facts

- Pure NumPy + Flask + stdlib + hashlib. No torch, no GPU, no learned weights in core.
- Deterministic: bit-reproducible under any PYTHONHASHSEED; one stated tie rule everywhere.
- Every claim ships with its measurement; refuted ideas are kept on record as negatives.
- ~600 modules, one UnifiedMind facade, ~2,000 faculties, 6,300+ tests, audits at 0/0/0.
- Approximate search must measure its own recall on YOUR data or demote to exact.
- Retrieval can refuse (calibrated abstention) instead of hallucinating a match.
- Programs compile into certified model weights (residual + conditioning +
  quantization + sha256 certificates); model files are ~250-byte RULES that
  re-bake bit-identical weights. Live models: https://huggingface.co/staccs

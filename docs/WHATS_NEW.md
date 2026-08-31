# What's new, feature by feature (one example each)

Testers told us the recent changes were hard to grasp from the changelog. This page
is the other direction: each feature, what it's for, and one copy-paste example.
Everything below is deterministic and runs offline.

## One-call boot (external memory automatic, opt-out)
    import lecore
    m = lecore.autoboot()                      # partition or shipped bundle + model rung if present
    m = lecore.autoboot(memory=False)          # clean boot
    m = lecore.autoboot(session="my-topic")    # fresh isolated context
On a fresh machine with no partition, the shipped 62-entry bundle at
`release_bundle/` loads automatically.

## The chat (the front door)
`run.bat` / `./run.sh` → http://127.0.0.1:7860. Substrate-first: answers come from
memory with **provenance chips**; unknowns escalate honestly. Say **`commands`** for
the full cheatsheet. `teach: q = a` stores; `wrong` vetoes durably; `health` shows
memory saturation; workspaces isolate.

## Grounded answering (the engine floor)
    g = m.ask_grounded("your question")
    # {"answer", "provenance", "escalate"} -- taught beats model; a model's words
    # only ever arrive marked "model-cached"; ungrounded fuzzy output is refused.

## Memory portfolios: many memories, sharing, real transfer
    m.memory_export("research_seed", query="resonator")   # selective, verified, tombstones ride
    m2.memory_import("research_seed")                      # conflicts FLAGGED, local wins
    # what travels: facts + provenance, conjectures AT their earned rung,
    # learned api tools (callable on arrival), and vetoes.
Chat: `memories`, `export memory <dir>: <filter>`, `import memory <path> [theirs]`.

## The void loop (discover -> prove, as a conversation)
Chat: `explore` → numbered conjectures → `test 1` (a REAL 300-draw pairing null) →
`promote 1` if it passed. Failures are kept and say so.

## API learning (from leOS)
    m.api_learn(openapi_spec_or_url)     # no LLM in the parse; endpoints become tools
    m.api_use("svc", "endpoint", params={...})
    m.tool_find("get the temperature")   # contextual discovery over EVERYTHING
Every learned endpoint teaches a discoverability card — finding a tool is just recall.

## Local models with automatic attribution
    rung = m.attach_runtime("/path/to/model")   # NumPy runtime; logit-lens rides the
    # generating forward; every model answer leaves a source address
    # ("model:28L/L14/ab12cd34"); early+agreed addresses run TRUNCATED schedules.
    # Opt out: attribution=False or LECORE_NO_ATTRIBUTION=1.
Measured speed tiers: memory bypass **97×**; exit-L7 **3.0×**.

## Instruments the expert panel demanded
    m.signal_scan(x, subbands=4)         # windowed FDR scan; drift-tolerant; calibrated null
    m.solve_maze_grid(grid, start, goal) # Physarum on YOUR topology
    sm = m.splat_memory(); sm.add(...); sm.recall_region(...)   # abstains when empty
    m.market_signal_test(signal, future_returns)   # corrected null + fee-swept PnL
    m.pnp_restore(None)                  # the generative denoiser (Eno's door)

## Self-explanation
    m.explain("drift sentinel")          # docs-derived card
    m.tool_find("scan a signal")         # 2,200+ faculty docstrings are the index

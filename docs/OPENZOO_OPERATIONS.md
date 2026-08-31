# openzoo.fun operations manual (v0.7.80)

*Everything an operator needs, specific and current. The tool table below is
generated from `holographic_mcp.py` itself — if it disagrees with the code, the
generator broke, and `tools/regen_docs.py --check` fails CI.*

## 1. Boot

    LECORE_PARTITION=/srv/openzoo/partition \
    python3 holographic_mcp.py           # MCP server; tool pin = 26

- **`LECORE_PARTITION`** — your operator memory (a directory holding
  `learning/state.lecore`). Absent → the shipped `release_bundle/` loads (62 generic
  entries, leakage-audited 0/4).
- **`LECORE_MODEL`** — optional model directory for `autoboot(llm="auto")` /
  `attach_runtime`; attribution is automatic, **`LECORE_NO_ATTRIBUTION=1`** disables.
- **`LECORE_MEMORIES`** — root for named memory portfolios (`memory_list`).

Build your operator memory with `tools/make_seed.py facts.md seed_dir/` (verified:
every fact answers at T0 from a fresh boot; held-out probes must escalate; leaky
seeds are refused) or distill a working partition with `tools/distill_release.py`
(every exclusion itemized by reason).

## 2. The 26 hosted tools (generated from source)

| Tool | Description |
|---|---|
| `lecore_map` | THE TERRITORY IN ONE CALL: leCore's capability families, what you should  |
| `lecore_find` | BEFORE implementing any algorithm, math routine, data structure, or file  |
| `lecore_describe` | Full contract for one faculty: what it does, a runnable example, and  |
| `corpus_bind` | Bind a corpus once: pass documents (or one long text, auto-chunked) and  |
| `corpus_ask` | Ask a bound corpus anything: BM25-ranked chunks with scores, best  |
| `void_explore` | THE DISCOVERY TOOL: find what a bound corpus's own structure LICENSES  |
| `zoo_ask` | THE HOSTED ANSWER LADDER: the server walks its FREE rungs first --  |
| `zoo_panel` | DELIBERATION UNDER THE CONTRAST LAW: give a question and a map of  |
| `zoo_tools` | CONTEXTUAL TOOL DISCOVERY AND USE: op='find' ranks the whole  |
| `zoo_void` | LEAP ON PURPOSE, WITH RECEIPTS: explore the gaps between known  |
| `zoo_teach` | CLOSE THE LOOP: after you (the model rung) answer an escalated zoo_ask,  |
| `zoo_do` | THE HOSTED TASK PATH: pass a request; if the server has a LEARNED PLAN  |
| `zoo_synthesize` | SYNTHESIZE A TOOL ON THE HOSTED SERVICE: compose a typed chain of  |
| `zoo_query` | HOSTED DATA SUPERPOWERS, both dialects: dialect='sql' runs the  |
| `zoo_report` | THE FULL-ADVANTAGE DASHBOARD for this tenant: per-tier serves,  |
| `receipt_verify` | Re-run a prior call and check its receipt: pass the original tool name,  |
| `memory_write` | Write to YOUR external memory -- a persistent leCore partition managed  |
| `memory_search` | Search YOUR external memory partition (ranked, best first). Check here  |
| `zoo_model3d` | Model a 3D scene from a shape spec and render it through leCore's  |
| `zoo_research` | LOSSLESS research archive: give texts (+sources) to preserve them in  |
| `zoo_backtest` | Walk-forward market backtest (no lookahead): routed-forecaster d-grid  |
| `zoo_assimilate` | Assimilate API/framework documentation: archive the doc losslessly,  |
| `zoo_feedback` | CLOSE THE LEARNING LOOP over the wire: report whether an answer  |
| `zoo_boot` | BOOT this hosted substrate and receive your operating screen: POST  |
| `zoo_agent` | Run one round of a long-running agent loop: gather from the  |
| `lecore_invoke` | Run any public leCore faculty. args is a JSON object of keyword  |


## 3. Security boundaries (enforced in code, stated here)
- **SSRF**: hosted callers cannot register API URLs. Only operator-registered
  services are callable through `zoo_tools op=call`. Register server-side:
  `service.mind.api_learn(spec)` at boot.
- **Memory upload/import is local-runtime only** — an abuse surface hosted; the
  swarm-audit matrix records this as a deliberate gap with its reason.
- Per-user teachings are session-salted; `taught_only` callers are never served
  model output.

## 4. Cost levers (measured; see docs/OPENZOO_INTEGRATION.md §2/§8 for numbers)
- Memory-first answering: **97×** cheaper than generating; scales with teaching.
- One generation per unique question (model-cached provenance).
- The grounding gate: ungrounded model output refused — no paid retry loops.
- `zoo_tools op=status`: saturation + taught rows + registered services — watch the
  model arm shrink in `decision_report` as memory grows.
- Attribution shortcuts (operator-side, model-dir access): early-address cues run
  truncated schedules, measured 3.0× at exit-L7.

## 5. Adopting a new build
1. Unzip the release over your deployment (additive; generated docs regenerate).
2. `python3 tools/swarm_audit.py` — must be 0 unintended gaps.
3. `python3 -c "from holographic_mcp import _selftest; _selftest()"` — tool pin 26.
4. `python3 tools/bench_ladder.py && python3 tools/bench_longmem.py` — 0.75/1.00 and 1.000.
5. Point `LECORE_PARTITION` at your operator memory; restart.

## 6. Troubleshooting
- **A tool answered with token-noise marked model-cached** — an untrained/weak rung;
  memory still wins where taught, vetoes kill the noise durably, and the grounding
  gate blocks it from fuzzy serving. Attach a capable model or none.
- **`teach` returned `taught: False`** — the control-token guard (standalone
  `__word__` reserved); rephrase. The refusal reason is in the return.
- **A question that should answer, escalates** — check `zoo_tools op=status`
  saturation; `nearing-cliff` means margins are thinning (the cp59 early warning).
- **Slow first call** — the runtime rung loads weights lazily on first use.

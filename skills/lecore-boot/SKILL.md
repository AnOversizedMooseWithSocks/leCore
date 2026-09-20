---
name: lecore-boot
description: "Boot the leCore holographic engine (leos-core) as the working environment for a task: clone/refresh, prove it, start the service or MCP, mount the external memory, roll the learning generation, attach the model end, spin up a swarm on the shared memory. Use when the user says boot up leCore, clone the leCore repo, connect to both ends, spin up a swarm, use the external memory, or names leCore / leos-core as the tool for a job. For 3D rendering with leCore, pair with the lecore-render skill."
---

# leCore: boot it, connect both ends, work through it

leCore is a NumPy-only Vector Symbolic Architecture engine: memory, meaning, geometry,
images and programs as points in one high-dimensional space, with calibrated abstention
(it refuses rather than guesses), self-measuring search, bit-reproducible determinism,
and ~2,440 faculties behind one `UnifiedMind` facade (27 parts; the decision faculties live in part 27). Its own instruction to agents is
short and worth taking literally: **do not summarize this project from the file tree —
ask the engine.** It carries a semantic search over its own capabilities and is better
at finding the right module than grep.

The user's standing opener is a sentence like "clone the leCore repo if we don't have it
already, boot up leCore, connect to both ends of it, spin up a swarm that uses the
external memory", followed by the task. Every command below was run and measured on the
`silly` branch (Sep 2026); numbers in parentheses are baselines. Report each step's
number as you go — a step with no number did not happen.

Repo: https://github.com/AnOversizedMooseWithSocks/leCore (working branch `silly`).
PyPI: `leos-core` (a clone is preferred: the memory partition and tools live in the repo).
Read AGENTS.md once per session; CAPABILITIES.md is the generated menu of every
capability with runnable examples and search aliases.

## Step 1 — clone or refresh

    [ -d leCore ] || git clone --branch silly https://github.com/AnOversizedMooseWithSocks/leCore.git
    cd leCore && git pull --ff-only; git log -1 --format='%h %ad %s' --date=short

Report the commit and date. Never delete and re-clone an existing checkout — it may hold
uncommitted work; if `git pull` refuses, say so and keep it. `pip install -e .` needs
`--no-build-isolation` in a sandbox behind a proxy.

## Step 2 — preflight: proof, not vibes

    python3 -c "import numpy, flask" 2>/dev/null || pip install --break-system-packages numpy flask
    PYTHONHASHSEED=0 python3 tools/showcase.py | tail -1

The last line must read `ALL CLAIMS HELD` (1.4–2.8 s on a laptop CPU). The six claims
it asserts live are the project: calibrated abstention (false alarms 0.003 at α=0.01,
power 1.000 on real vectors), a measured recall budget that demotes approximate search
to exact, exact top-k at any scale in one tile of memory, a 258-byte model file that
re-bakes bit-identical weights, VM == installed weights (cosine 0.99999998), and one
stated tie rule everywhere. If it fails, stop and show the failing claim.

Then the accelerator picture, honestly: `python3 -c "import lecore; m=lecore.UnifiedMind(dim=512); print(m.backend_status()); print(m.gpu_report())"`.
See "Acceleration" below before promising a GPU anything.

## Step 3 — start a front door (service, MCP, or both)

**HTTP service** — the one every swarm agent shares:

    export PYTHONHASHSEED=0                                  # the engine is deterministic; keep it fixed
    export LECORE_PARTITION=${LECORE_PARTITION:-$PWD/lecore_memory}   # the external memory
    export LECORE_MEMORY_ROOT=$LECORE_PARTITION
    nohup setsid python3 holographic_service.py --port 8080 \
        --persist "$LECORE_PARTITION/service_store.json" > lecore_service.log 2>&1 < /dev/null &
    sleep 5; curl -s http://127.0.0.1:8080/health           # {"ok": true, "name": "leCore", "capabilities": 868, ...}

`setsid` + `< /dev/null`: a plain `nohup … &` was reaped mid-turn in a cloud sandbox, and
background processes do not survive between turns there. Start every turn with
`curl /health`; if silent, redo Steps 3–4 before touching the task. If 8080 already
answers `name: leCore`, reuse it. `./serve.sh --host 0.0.0.0 --token secret` adds a
Bearer token when the service must leave localhost. Every agent gets this helper:

    inv() { curl -s -X POST http://127.0.0.1:8080/invoke \
            -H 'Content-Type: application/json' -d "$1"; echo; }
    # inv '{"name":"<faculty>","args":{...}}'   — any public faculty; names starting with _ are refused
    # arg names: curl -s -X POST http://127.0.0.1:8080/skills/card -H 'Content-Type: application/json' -d '{"name":"<faculty>"}'
    #            -> card["primary"] is the real first-argument name
    # NO apostrophes or double quotes inside the JSON strings (they break the shell quoting): rephrase.

**MCP server** — for an MCP host (Claude Desktop, Claude Code, any client). It is stdio
JSON-RPC, stdlib only, and wraps the same service (`POST /invoke` underneath):

    python3 holographic_mcp.py --selftest        # initialize / tools/list / tools/call in-process
    lecore-mcp                                    # console entry point after pip install leos-core

    {"mcpServers": {"lecore": {"command": "python3", "args": ["/abs/path/leCore/holographic_mcp.py"],
        "cwd": "/abs/path/leCore",
        "env": {"PYTHONHASHSEED": "0", "LECORE_MEMORY_ROOT": "/abs/path/leCore/lecore_memory"}}}}

The MCP reads `LECORE_MEMORY_ROOT`, not `LECORE_PARTITION` (the ops doc names the wrong
one). Its 40 tools: the curated trio `lecore_map` (call ONCE: the territory, what never to
hand-roll, the phrases to ask), `lecore_find(query)`, `lecore_describe(name)`, plus
`lecore_invoke(name, args)` for any public faculty; memory (`memory_write`,
`memory_search`, `zoo_teach`, `zoo_ask`, `zoo_boot` — call `zoo_boot` FIRST when
attaching: the MCP's mind has no doctrine until it runs); corpora (`corpus_bind`,
`corpus_ask`, `study`, `study_ask`); analysis (`series_analyze`, `dataset_decompose`,
`fact_check`, `math_eval`, `chart_make`); scenes and images (`scene_create`,
`scene_adjust`, `scene_export`, `image_tool`, `zoo_model3d`); the zoo agent loop
(`zoo_agent`, `zoo_do`, `zoo_research`, `void_explore`, …). Every `tools/call` result
carries `_meta.lecore.receipt` (input/output sha256 — determinism is the proof system)
and `_meta.lecore.cost`. Errors come back in `content` with `isError: true` and a `hint`
naming the real parameters. Both doors, and the Python API, are the same faculties.

## Step 4 — connect end one: memory in front, then roll the generation

    inv '{"name":"boot","args":{"partition":"'"$LECORE_PARTITION"'","doctrine":true}}'
    inv '{"name":"learning_rollover","args":{"root":"'"$LECORE_PARTITION"'"}}'

The service starts with an EMPTY mind; `boot` mounts the partition (order: POST → mount →
doctrine → services → report). Require `mounted` == the partition path and every row of
`post_after_mount` `true` or `skipped`; report `inventory.taught` (385 on the shipped
partition). If `mounted` is null, STOP — never work on a virgin mind while believing
memory is loaded.

`learning_rollover` is NOT optional. Measured bug: `learning_save` without a prior
rollover writes the legacy `learning/state.lecore`, which the loader ranks OLDEST — a
session's teaching vanished on the next boot (402 rows taught, 385 came back). After the
rollover the partition holds exactly one generation, `learning/state-YYYYMMDD-HHMMSSZ.lecore`,
and saves go there. Report the row count and the file name, then prove recall with one
`ask` of something taught in an earlier session, in its taught wording — recall matches
wording closely, and a loose rewording is refused by design.

## Step 5 — connect end two: the model behind

"Both ends" is the engine's `agent_boot()` doctrine: memory in front, a model behind, so
an unknown question escalates instead of dying at T4. In Python: `lecore.UnifiedMind()`
has no memory (even doctrine answers `refused`); `lecore.autoboot()` mounts memory (known
questions answer at T0) and is for a human at the keyboard; `lecore.agent_boot()` adds
the model rung — it takes the model from `LECORE_LLM_URL` / `LECORE_LLM_MODEL` /
`LECORE_LLM_KEY` (`OPENAI_*` are read too) and refuses a non-callable on purpose, so a
person is never wired in as the back end. Over the service the same contract is manual:

- If `$LECORE_LLM_URL` is set, the engine escalates to that model itself.
- Otherwise the agents ARE the model end: `serve` returning `{"served": false, "via":
  "escalate", "reason": …}` means memory could not answer. Answer from your own reasoning
  or tools, then teach it back:

      inv '{"name":"resolve","args":{"question":"...","answer":"...","by":"<agent-name>"}}'

  The next `serve` is T0 with provenance `human:<agent-name>`. Before finishing,
  `inv '{"name":"escalations","args":{}}'` must be empty. Note `escalations()` is
  in-process only (it does not survive a restart) and only `serve()` populates it.

## The semantic system — what the tiers and words mean

`ask` / `serve` climb an answer ladder and every answer says which rung served it:
**T0** memory (reflex; wrong-answer-free by construction), **T1** substrate retrieval
with a freshness check (a stale hit is a miss), **T2** a bound tool call with declared
argument extraction (never guessed), **T3** a small model, **T4** the main model — the
exception, not the path — and **refused** (`via: null`, confidence 0.0), a first-class
result, not an error. Provenance travels with every answer: `taught`, `validated`,
`evidenced` are established; `model-cached` is PROVISIONAL (a hosted ladder caches what a
model said and would serve it forever — `taught_only=true` refuses those); `human:<by>`,
`wisdom:<author>`, `conjecture` (every `void_*` result), `tool-reflex`. Blank answers
never wear T0. Doctrine is the seedpack `boot` registers (≥ 14 rows, served at T0).

Discovery faculties and their real argument names — the same concept is spelled three
ways across the doors, so check `card["primary"]` when in doubt:

| want | Python | HTTP | MCP |
|---|---|---|---|
| a faculty for a problem | `find_capability(problem, k, accepts, produces)` | `POST /capabilities/search {"query"}` | `lecore_find(query)` |
| rank skills for a task | `suggest(task, k)`, `route(task)` → act / choose / unknown | `POST /skills/suggest`, `/skills/route` | — |
| one faculty's contract | `describe_skill(name)`, `complete_method(prefix)` | `POST /skills/card`, `/skills/complete` | `lecore_describe(name)` |
| ask / teach | `ask(query)`, `serve(query)`, `teach(query, answer)`, `teach_about(question, answer, paths)`, `resolve(question, answer, by)` | `/invoke` | `zoo_ask`, `zoo_teach` |
| what is stale | `stale_facts(root)`, `codebase_sync(root, only_stale=True)` | `/invoke` | — |
| plugins present | `plugin_list()` → available / missing / install | `/invoke` | `lecore_map` |

`teach()` can refuse (`{"taught": false, "reason"}`) — check the flag. `suggest_pipeline
(start_kind, goal_kind)` chains faculties by io kind. `route_semantic` is the embedding
router (7/12 top-1 vs 2/12 for token overlap, measured) and returns `None` rather than
fabricate an embedding — fall back to `find_capability`. Once oriented, the engine is
its own manual: `find_capability` → `describe_skill` → the example → `invoke`.

## Step 6 — spin up the swarm

One service process IS the shared memory. Agents share it by talking to the same URL —
never by booting their own in-process minds on the same partition. Measured: agent A
taught a fact and saved; agent B, told nothing, recalled it at T0 and refused a question
nobody had taught. Size the swarm to the task: one worker per independent piece, at most
4 unless the user asks for more. You orchestrate: keep the plan, hand out roles, merge
results, own Step 8. Each worker is a subagent whose prompt carries the service URL, the
partition, its one-line role, the `inv` helper, and this contract verbatim:

    - Before starting, ask the shared memory:
        inv '{"name":"ask","args":{"query":"<your task, phrased the way a stranger would ask it>"}}'
      The work may already be done.
    - Before hand-rolling any algorithm, codec, search, geometry, or file format:
        inv '{"name":"find_capability","args":{"problem":"<what you need, plain words>"}}'
      Hand-roll only when it returns nothing relevant.
    - Write as you go: inv '{"name":"teach","args":{"query":"<question>","answer":"<answer>"}}'
      after every measurement, bug located, decision settled, and approach RULED OUT.
    - A refused or escalated answer is a result. Never paraphrase it into a guess.
    - Report numbers with baselines. Keep negatives loud.
    - Finish by listing every teach/resolve you made, with the exact query strings.

Worker shapes that have paid off: a reference scout (web search, teaches facts with
URLs), a probe (one small script that measures one engine claim in a fixed region with a
metric), an auditor (re-runs a number the orchestrator reported). The engine's own swarm
patterns are runnable in docs/USE_CASES.md: roles on one bus (`mind.role(name, topic,
handler, emit=…)`, `bus.publish/history`, `distributed_bus`, `farm`), a development
swarm sharing one understanding of a codebase (`codebase_sync` + `stale_facts`), and the
HTTP bus (`POST /bus/publish|poll|history`) plus checkpointable jobs (`/jobs/*`, bucket +
monoid reducer, resume only the remaining buckets).

## Step 7 — the task: common and uncommon ways in

Prefer the engine's faculties over reimplementing; when the task is editing leCore
itself, doubly so. What people reach for, with the entry faculty:

Common: a persistent memory that outlives the process (`teach` / `ask`, a partition
directory is the whole API; LongMemEval 1.000 with no LLM attached); answer before
spending a model call and escalate honestly (`serve`: memory → tool reflex → escalate,
3.5 ms vs 340 ms generating); find the right capability from plain English
(`find_capability`, `suggest`, `route_or_abstain`); nearest-neighbour search with a
recall guarantee (`build_index(recall_budget=…, abstain=α)`; recall@10 1.000 at 9.7 ms
on 100k×768); exact top-k over a file too big for RAM (`tiled_topk`, memmap, one tile
of memory); grounded QA over a directory with citations or a refusal (`study(root)`;
MCP `corpus_bind`/`corpus_ask`); a 3D scene from words (`build_scene(text).render()`,
`scene.adjust`); a picture with no tuning (`render_auto(quality=…)`); shapes as math
(`sdf_*`, `crystal_*`, `creature_*`); lossless compression that amortises
(`compress_lossless`, `cold_store`); a content-addressable store that degrades
gracefully (100% recall@1 with 40% of storage destroyed).

Uncommon: your own logic as a program that is itself one hypervector (`HoloMachine`:
`vm.assemble`, `vm.run`, `IFMATCH` a cosine conditional in one instruction); compiling
that program into audited model weights (`compile_program_installed`, `native_model`,
`verify_conformance`) and knowing in advance what can install (every pure linear read
installs; every data-dependent branch is host-shape; I/O is substrate-impossible);
installing a capability into a real pretrained LLM and uninstalling exactly
(`unicron_install`, delta 3.7e-09); reading and writing a running model's memory with
zero forward passes (the linear-attention state IS an HRR trace; `memory_write` /
`memory_search` over MCP); emitting shaders from the engine's own math (`emit_kernel`
in wgsl/glsl/slang/c/js/zig, `sdf_shader`, `to_shadertoy`, `compose_shader`; the
emitter refuses rather than guesses); hostile-data analysis that catches its own
leakage (`lookahead_lint` → `pipeline_null` → `event_study` → `net_of_costs`; a
trailing smoother reads 79% persistence on white noise); scientific verdicts with a
matched null (`science_report(kind=light_curve|pulsar_panel|spectrum|decay|levels|chsh)`
— refusal names what would decide, `suspect-instrument` accuses the apparatus);
discovering what a corpus is missing (`void_explore`, `void_map`, three wrong plants
refused before one right leap); creatures and game worlds with a readability gate
(`creature`, `creature_readability_score`, `game_shard` with lockstep digests);
receipts that prove a computation happened (`receipt_verify`); `levers()` when you hit
a wall (seven levers in cost order); `explore_series(auto_demux=True)` on data with no
schema. Extend it from outside with a plugin: a `PLUGIN` dict + `register(mind, config)`
returning verbs; the loader refuses a name that already exists, a verb without `does`,
an unknown key — a verb that cannot be found does not exist.

Sandbox facts that shape a task in a cloud container: no GPU; outbound image downloads
may be blocked by egress policy (ask the user to upload assets; do not hand-roll
stand-ins silently); the memory cgroup may kill a process well below what `free -m`
reports (5.1 GB on an "8 GB" box; an OOM kill leaves no traceback — check `dmesg | grep
Killed`); a tool call has its own wall-clock ceiling, so long work runs detached with a
log, and A/B batches go in a shell script. Never trust `pgrep -f <script>` from inside a
`bash -c` — it matches the checking shell; use the log mtime and /proc/loadavg.

### Acceleration — say only what is measured

The engine's bit-exact guarantees are a CPU/NumPy property; GPU paths are opt-in, for
throughput, and match to a tolerance. Four separate paths, not interchangeable:

- **CuPy backend** (`pip install cupy-cuda12x`, matched to the driver; `tools/install_gpu.py`
  picks the wheel): `HOLOSTUFF_GPU=1` before the process starts (read once at import),
  or `mind.use_gpu(True)` (vetoed by `ResourcePolicy(gpu='off')`). Only three things
  compute on the device — the fluid solver, the FFT shader pipeline, the GDN LLM forward
  pass. `bind`/`bundle`/`cleanup` have no CuPy path. Its speedup is wired and
  code-reviewed but NOT measured anywhere in the repo; the only CuPy claim is parity.
  `mind.gpu_report()` says what is present and why not; `mind.should_offload(n_bytes,
  flops_per_byte)` gates on PCIe arithmetic (100 KB, 4 flops/byte — provisional); a
  sphere trace is 144 flops/byte, an elementwise post pass 0.8 and correctly refused.
- **wgpu / WGSL** (`pip install leos-core[wgsl]`, vendor-neutral, in the `all` extra):
  compute kernels emitted from the authoritative Python (`run_kernel`, `wgsl_matmul`,
  `wgsl_cleanup`, `sdf_depth_device`), `verify_against_numpy` differential-tests them on
  your data. Correctness measured bit-exact on a software adapter; **speed unmeasured**
  — `gpu_crossover` benchmarks THIS path, not CuPy, and says "MEANINGLESS ON THIS
  ADAPTER" on llvmpipe. Argmax decisions are always resolved host-side (ties flip at a
  1e-7 gap: 3/150).
- **Numba** (`pip install leos-core[jit]`, no flag): four fast-sweeping SDF functions,
  ~33× on a sequential recurrence (run `holographic/misc/holographic_jit.py` for the
  live figure; two files quote different constants), plus the `jit` plugin's
  `compiled_sdf_numba` / `render_sdf_fast` (~9–15× claimed). Never `parallel=` /
  `fastmath=` (one int32 GEMV is the documented exception).
- **Zig** (`pip install leos-core[zig]`): native batch kernels + raymarcher, measured
  2–5×, 3.8× on raymarch, bit-identical in safe mode. This is the one that reliably
  pays for renders.
- Browser **GLSL/WebGL2** kernels are source-only in core: measured on an RTX A4500,
  fields/physics win 51–307×, retrieval loses (plain JavaScript beats the GPU ~300× on
  browser retrieval; a published 106× figure is retracted).

There is no `[accel]` extra; `requirements-accel.txt` installs numba and sympy only.
`mind.plugin_list()` is the honest preflight (available / missing / install line).
When the user's machine has a CUDA card, turn the CuPy path on for simulations and
measure before and after — and report that the render stays on CPU unless zig/numba
carry it.

## Step 7b — typed decisions, outcomes by id, and the swarm contract (sweeps 171–176)

When a task has a fixed set of answers, do not ask the model end to guess — ask the engine a
TYPED question and report the outcome. Every door here is one `POST /invoke`; the syntax
reference with live-generated JSON is `docs/TYPED_DECISIONS.md`; measurements are in
`docs/research/`. Numbers in parentheses are baselines on real data, 3 seeds.

    # 0. the natural path: route() is tiered and learns; typed() is the one-line decision with the lint built in
    inv '{"name":"route","args":{"task":"smooth a bumpy mesh"}}'          # act / choose / abstain + tier, z, id
    inv '{"name":"typed","args":{"state":"courier lost the package","options":["billing","shipping"]}}'
    #    -> value, ranked, p, id, lint (heed the warns); then decision_outcome(id, truth) -- everything learns

    # 1. lint the question BEFORE asking (every failure the tool experiments hit is a finding)
    inv '{"name":"systemone_lint","args":{"questions":{"cat":{"type":"choice","options":["billing","shipping"],
         "examples":{"billing":["card charged twice","refund my invoice"],"shipping":["parcel lost","courier late"]}}},
         "states":["my card was charged twice. also the parcel is late"]}}'
    #    -> findings: budgets imbalanced >2x, <3 examples, recommended scorer, multi-clause state (split), contrastive

    # 2. decide; scorer nb from ~20 examples per option; labeled rows give p and (with conformal_alpha) a SET
    inv '{"name":"systemone_decide","args":{"state":"courier lost the package","questions":{...},
         "scorer":"nb","encoder":"ngram","labeled":[["parcel is lost",{"cat":"shipping"}],...],"conformal_alpha":0.1}}'
    #    -> {"cat":{"value","ranked","margin_gap","p","set","via","id"}}   (coverage 0.960 at nominal 0.95)

    # 3. report the outcome BY ID -- the only outcome path; the table, the reflex and the calibration learn
    inv '{"name":"decision_outcome","args":{"record_id":"<id>","outcome":"shipping"}}'
    #    (prequential 0.699 -> 0.764 with zero model calls; report "failed" only when NO truth is known)

    # 4. route without a bare no: answer / menu / refuse (old gate accepted 4.7% of paraphrases; 0/30 gibberish answered)
    inv '{"name":"route_tiered","args":{"problem":"smooth a bumpy mesh","reflex":true,"verify":true}}'
    #    reflex=true answers a REPEAT from experience (99.3% at 0.982); verify=true attaches the verdict

    # 5. a swarm step is refused without done_when + evidence; verify runs BEFORE the step is accepted
    inv '{"name":"swarm_step","args":{"state":"resolve families","tool":"catalog_families","done_when":"500 resolve",
         "evidence":{"resolved":501},"worker":"w1","verify":{"verb":"catalog_families","args":{}},"expect":null}}'
    inv '{"name":"swarm_evaluate","args":{}}'      # reachability, catalog_gaps, skill_lint as the exit -> all_ok

    # 6. a compound request -> a cold plan, no model (both steps recovered 0.860 vs 0.540 for the whole request)
    inv '{"name":"plan_from_request","args":{"request":"smooth a bumpy mesh and then grow crystals on a surface"}}'

    # 7. CODE through the same loop: plan (reuse/extend/build with evidence), edit under validated termination, review
    inv '{"name":"plan_change","args":{"request":"add a min_seg parameter to the drift report"}}'
    inv '{"name":"edit_verified","args":{"path":"holographic/x/holographic_y.py","old":"...","new":"...","selftest_module":"holographic.x.holographic_y"}}'
    #    -> ok False + why + the file restored byte-identical when any check fails; ok True + a step id otherwise
    inv '{"name":"review","args":{"paths":["holographic/x/holographic_y.py"]}}'     # merge_ready + findings with lines
    #    then swarm_evaluate for the audits, and decision_outcome on the plan and the review with what happened

Schema syntax, exactly: `choice` = `options` (≥2 strings) + `examples` per option (non-empty
lists, keys ⊆ options); `noul` = `examples` keyed `yes`/`no`; `score` = numeric `min` < `max`
+ `anchors` as **(text, value) pairs** — a value-keyed dict is refused.

The model end on typed decisions: in process, `systemone_decide(..., escalate=callable)` routes
an abstention through `decide_or_escalate` (schema-validated, one retry); the callable gets
`{question, spec, state, evidence, why, prompt}` — `prompt` is the four-part prompt generated
from the schema — and returns one option. Its answer is a record `via="model_end"`; report the
outcome and the reflex answers the next similar task without the model (leOS's self-extending
instruction, measured: one model call, then none).

After `boot` on a partition persisted before sweep 176, the reflex trace is tiled past its
capacity cliff and reads back under the floor; the first reflex read re-tiles itself, or run
`inv '{"name":"reflex_retile","args":{"advisory_load":0.03}}'` once (6 → 14 tiles measured).

Sandbox lessons that cost hours, kept: the service loads the code it was STARTED with — restart
it after any edit or a new verb "does not exist"; a `pkill`/`pgrep -f` from a shell whose own
command line contains the pattern kills that shell — launch and kill in separate calls; a script
run from another directory imports whatever `lecore` is first on `sys.path` (a PyPI `leos-core`
installed by a studio app shadows the checkout — `PYTHONPATH=.`); a 30-minute job dies at a
turn boundary with nothing written — checkpoint per unit (strips, instances) and resume; the
scorer bench was reaped by memory pressure with the studio, the service and the bench resident.

## Step 8 — land it

    inv '{"name":"learning_save","args":{"root":"'"$LECORE_PARTITION"'"}}'

Report `bytes`, `drift_vs_previous_save`, the file written (the dated generation, never
the legacy `state.lecore`), and the consolidated list of every teach/resolve. Then the
deliverable, per the user's standing rule: ONE zip of only the changed/new files, paths
relative to the repo root so it unzips in place over the checkout — no split archives,
no full-repo zip, no commits or pushes unless asked. Include
`lecore_memory/learning/state-<stamp>.lecore` (and `knowledge.lecore` if it changed) so
the memory travels with the work. Leave the service running unless told otherwise.
Report every number next to its baseline and say what did NOT work.

## Rules

- Ask the engine (`find_capability`, `/capabilities/search`, CAPABILITIES.md); never
  summarize it from the file tree.
- No probe facts into the real partition: the boot report IS the smoke test. Experiment
  on `partition=/tmp/lecore_scratch` and never ship it.
- Every step reports its number: commit, showcase time, taught count, rollover rows,
  save bytes and drift. "It booted" without the POST line is not a report.
- One service process per partition; N agents, one URL.
- `refused` and `escalate` are honest and final for that call; the model end answers
  and teaches back, or leaves it escalated and says so.
- Decide, don't prompt: a fixed-answer question is a typed decision (`systemone_decide`) with an
  outcome reported by id, not a model guess; a rejection is a menu (`route_tiered`), never a bare no.
- Decompose before you tune: when a result looks wrong, measure the terms that make it
  (a probe with a fixed region and a metric) before changing a parameter.
- Look at every image you produce (Read the PNG); a byte count is not a picture.
# Why a holographic virtual machine — and what a swarm does with one

leCore is a computer whose memory, programs and tools are all *vectors in one space*.
This page is the case for running one, what it can do, and how a swarm of models shares
it. Every code block that opens with `# guide-check` runs verbatim in CI
(`tests/test_guide_examples.py`), so the claims here are executable, not aspirational.

## 1. Why

A conventional agent stack keeps memory in a database, tools in a registry, skills in
prompt files, and reasoning in a model — four systems with four failure modes, and the
model paying tokens to bridge them on every turn. A holographic machine keeps all four in
**one algebra**: facts, roles, programs and tool descriptions are hypervectors; *bind*
composes, *unbind* queries, *superposition* stores many things in one vector, and
*cleanup* recovers a clean symbol from a noisy one. That buys five things a database and
a prompt cannot:

- **Memory answers are free.** A taught fact is recalled in ~2 ms with no model call, and
  recall latency stays flat from a thousand facts to twenty thousand (measured: 2–3 ms
  across that whole range, every planted fact exact, never-taught questions refused).
- **Determinism is a proof system.** No learned weights, `hashlib` not `hash()`, seeded
  RNG everywhere: two minds fed the same stream produce **byte-identical** partitions
  (measured at 5,000 rows). Anything a swarm shares can be verified by hash.
- **Programs are data.** A vector program runs on the machine, can be stored in the same
  memory as facts, and can call other programs by name — a skill is a vector, not a file.
- **Everything is discoverable.** ~3,700 catalog capabilities behind one `find_capability`,
  so a model asks for "make my picture less grainy" instead of reading a tool roster.
- **Walls fall by rule.** Seven levers, walked in order, before anything is called
  impossible — the README's "seven levers" section is the engine's operating manual.

The smallest program: define a function in the machine's library, then call it from a
program. The accumulator ends up holding `bind(a, b)`, recovered by cosine, and the trace
shows exactly what ran.

```python
# guide-check
from holographic.agents_and_reasoning.holographic_machine import HoloMachine
from holographic.agents_and_reasoning.holographic_ai import bind, cosine
m = HoloMachine(dim=4096, seed=7)
m.define("tag_b", [("BIND", "b"), ("HALT", "")])          # a stored function: ACC = bind(ACC, b)
acc, trace = m.run(m.assemble([("LOAD", "a"), ("CALL", "tag_b"), ("HALT", "")]))
assert ("CALL", "tag_b") in trace
assert cosine(acc, bind(m.data_atoms["a"], m.data_atoms["b"])) > 0.99
```

## 2. What can be done with it

The catalog is the map; `orient(topic=...)` is the compass. In broad strokes: a memory
ladder (teach / recall / sessions / partitions / rollover / regen), simulation and
physics (FEM, fluids, bodies, orbits, polarization), rendering and geometry (SDF scenes,
meshes, splats, shaders, animation), analysis (regimes, forecasts, formulas, drift,
fact checking with citations), code tools (grep / view / replace / syntax check, repo
maps, `study` with citations, `merge_trees`), the language layer (dictionary, semantic
word search, a default corpus), and the agent substrate (`serve`, tool reflexes,
`api_learn`, wisdom, the commons, the MCP server). FEATURE_GUIDE walks each family;
§10 is the agent tour.

```python
# guide-check
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
o = mind.orient()
assert o["counts"]["capabilities"] > 3000 and len(o["workflow"]) == 5
```

## 3. Swarm memory: shared, partitioned, and distilled

Three topologies, all on the same rails:

**One shared partition.** Every model on the platform boots against the same memory
root (`agent_boot`); what any of them teaches, all of them recall. Sessions inside it are
a privacy boundary (the reflex tier's session guard is pinned by a cross-session probe
battery).

**Partitioned memory with distillation.** Each user or agent keeps its own partition —
isolation is a directory. Knowledge flows *between* partitions only through the
commons: `contribute` screens a partition's shared rows (session-salted rows never
leave; path, email, phone and key shapes are rejected with reasons on a review sheet),
`commons_pool` merges bundles with conflicts flagged, and every partition draws back
with `memory_import`. Wisdom rides the same pipe with authorship intact: a lesson one
model bequeaths reaches every partition, attributed.

```python
# guide-check
import lecore, os, tempfile
t = tempfile.mkdtemp()
bundles = []
for i, (q, a) in enumerate([("what is 7 times 8", "56"), ("boiling point of water at sea level", "100 C"),
                            ("what is the derivative of x squared", "2x")]):
    p = lecore.UnifiedMind(dim=256, seed=0)                 # three partitions, three owners
    p.teach(q, a)
    p.teach("my email", "owner%d@example.com" % i)           # private: must not travel
    if i == 0:
        p.bequeath("measure before building", author="model-%d" % i, topic="discipline")
    bundles.append(os.path.join(t, "p%d" % i))
    assert p.contribute(bundles[-1], author="owner-%d" % i)["kept"] >= 1
pool = lecore.UnifiedMind(dim=256, seed=0).commons_pool(bundles, os.path.join(t, "commons"))
assert pool["rows"] >= 3
fourth = lecore.UnifiedMind(dim=256, seed=0)                # a partition that taught nothing
fourth.memory_import(os.path.join(t, "commons"))
assert fourth.ask("what is 7 times 8")["answer"] == "56"
assert fourth.ask("what is the derivative of x squared")["tier"] == "T0"
assert fourth.wisdom()["authors"] == ["model-0"]            # distilled, attributed
assert fourth.ask("my email")["tier"] != "T0"               # nothing private crossed
```

**Roles coordinating through slots, not chatter.** `shared_workspace()` gives swarm roles
named slots they read and write while deliberating; writes buffer within a round and
commit together, so a scout that only leaves a map *is* the round's progress.

```python
# guide-check
import lecore
ws = lecore.UnifiedMind(dim=256, seed=0).shared_workspace()
ws.write(0, "layout", [1, 2, 3]); ws.commit(1)
assert ws.read("layout") == [1, 2, 3]
```

**Across machines.** `mind.farm([...hosts], token)` round-robins partition-and-reduce work
across nodes and reassembles by the monoid reduce; `mind.distributed_bus(peers, token,
node_id)` spreads the publish/subscribe bus across nodes so agents on different machines
share topics. Same swarm, more hardware.

## 4. Group learning: the swarm improves as one

Learning is not per model here. A fact taught by any member is recalled by every member
that draws from the commons; a tool reflex taught once serves every later matching ask
from the tool with no model call; a lesson bequeathed by one model is inherited by
models that did not exist when it was written. The improvement loops are measured, not
hoped for: `agent_benchmark` scores the substrate against planted truths and kept
negatives, `alias_gaps` finds capabilities a stranger's phrasing cannot reach and repairs
their aliases, and `serve()` reports honestly when it escalates — so the swarm can see
where it is still paying for a model and teach that away.

```python
# guide-check
import lecore
a = lecore.UnifiedMind(dim=256, seed=0)
a.teach("what does the resonator do", "factors a bound composite into its parts")
assert a.serve("what does the resonator do")["via"] == "memory"       # a member learned it
r = a.serve("what does the peel decoder do")
assert not r["served"] and r["reason"]                                # and says what it cannot yet serve
```

## 5. Many models, one substrate

The memory end and the model end are separate. `agent_boot` attaches a memory root and
any OpenAI-compatible endpoint (`remote_llm(url, model)`); ten harnesses are wired under
`integrations/`; every connected model sees the same catalog, the same memory and the
same workflow contract through the MCP server's `initialize` banner. Models that are not
language models join through `api_learn`: an OpenAPI spec — a forecaster, a robot's
status endpoint — becomes callable, discoverable tools that survive save/load, and a
tool reflex lets the substrate answer from them directly (FEATURE_GUIDE §10 runs one).

## 6. Skills and tools: import, discover, synthesize

- **Import.** `api_learn(spec)` turns any OpenAPI spec into tools; MCP tools ride the
  server; `tool_reflex_teach` teaches how a tool answers a question shape.
- **Discover.** `find_capability` over the whole catalog — measured on 2,787 real
  endpoints (GitHub + Stripe + Kubernetes): the host model sees three candidate cards,
  never a 25 MB spec.
- **Synthesize.** Capabilities declare what they *consume* and *produce* (io kinds:
  mesh, image, points, sdf, timeseries ...). `suggest_pipeline(start, goal)` chains
  them into the shortest route for a requirement nobody wrote a tool for — a new skill
  assembled from the ones you already have. Inside the machine, `define` stores a vector
  program under a name and `CALL` composes it into larger programs: skills built from
  skills, as data.

```python
# guide-check
import lecore
mind = lecore.UnifiedMind(dim=256, seed=0)
route = mind.suggest_pipeline("image", "mesh")
assert route and "image" in route[0]["consumes"] and "mesh" in route[-1]["produces"]
back = mind.suggest_pipeline("mesh", "image")
assert back and "image" in back[-1]["produces"]           # and the way back, from the same primitives
```

## Where this is measured

Every number above has a pin: `tests/test_mcp_server.py` (one test per sweep),
`tests/test_guide_examples.py` (the blocks on this page), and the gauntlet records in
`docs/NOTES_concepts.md` (sweeps 105–108: the tool-library experiment, the limits
gauntlet, the determinism twins, the isolation bug that was found and fixed).

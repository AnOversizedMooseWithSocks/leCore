# leCore — Improvements & Integration Backlog
### make the engine better, and make it clean to build on top of

*A self-contained work list for leCore. Two parts: genuine **engine improvements** (new math/patterns worth
having), and light **integration** work so external programs and agent harnesses can drive leCore cleanly. All of
it is numpy/stdlib, deterministic, and reuse-heavy — grounded in the current codebase, so "reuse X" below means a
module that already exists. No outside context needed; the developer only needs to care about making leCore good.*

---

## Part 1 — Engine improvements (math / patterns / concepts)

### 1a. Opponent-channel faculty — decompose ensemble disagreement into a signal *(BUILD — the one clear addition)*

leCore already has a *scalar* version of "do these agree?" — `HoloForest` returns a cross-tree **agreement**
score, and `consolidate` extracts a shared subspace. This generalizes that: given **N estimates of the same
thing** (several recall paths, several encoders, several sensors, or the forest's own trees), decompose *how* they
agree and disagree — and treat the disagreement itself as information.

```python
# holographic_opponent.py -- ensemble disagreement, decomposed. Pure VSA algebra (projections, residuals, sums),
# deterministic. Reuses: moe (ensemble), tree (agreement), consolidate (shared subspace), honesty/RecallNull (gate).
import numpy as np
def opponent_channels(vecs):                    # vecs: N unit vectors estimating the SAME thing, comparable space
    V = np.array([v/(np.linalg.norm(v)+1e-9) for v in vecs])
    shared = V.mean(0); shared /= np.linalg.norm(shared)+1e-9    # the AGREEMENT direction (what all share)
    excl   = [v - (v@shared)*shared for v in V]                  # each source's EXCLUSIVE residual (what only it sees)
    purple = np.sum(excl, 0)                                     # the RESIDUAL-SUM ("purple") -- in NO single source alone
    return dict(shared=shared, exclusive=excl, purple=purple,
                disagreement=float(np.mean([np.linalg.norm(e) for e in excl])))

def opponent_relational(neighbours):            # ALIGNMENT-FREE: compare neighbour SETS -> works across DIFFERENT spaces
    sh = set.intersection(*map(set, neighbours)); ex = [set(n)-sh for n in neighbours]
    return dict(shared=sh, exclusive=ex, purple=set().union(*ex))
# classify(shared, excl) -> redundant / complementary / contradictory / hierarchical / novel, by residual geometry.
# a "divergence signal" fires when `disagreement` crosses a RecallNull-calibrated floor (real novelty, not noise).
```

Why it belongs in leCore regardless of anything downstream: the moment two of *anything* estimate the same
quantity, the residual-sum ("purple") channel tells you what neither captures alone, and the calibrated
disagreement tells you when to stop and look closer. The **relational** form compares neighbour sets, so it needs
no space alignment and works even across different encoders. `HoloForest.recall` can then optionally return the
full decomposition of its own trees, not just the scalar. *Effort: LOW–MED, reuse-heavy.*

### 1b. Proportional navigation — a guidance law leCore is missing *(CANDIDATE)*

leCore has beam-search `Navigator` and the `propagator` for prediction, but no *guidance law*. Proportional
navigation (the classic interception rule) steers toward where a target is **going**, not where it **is**, by
turning proportional to the line-of-sight rotation rate. With the propagator predicting a moving goal, this can
reach it in fewer steps than greedy "walk toward it," which could sharpen model-based navigation/planning.

```python
# Proportional navigation: steer proportional to the LINE-OF-SIGHT rotation rate -> intercept a MOVING target.
def pro_nav_step(pos, target, target_vel, N=3.0):
    los = target - pos; r = np.linalg.norm(los) + 1e-9
    los_rate = (target_vel - (target_vel @ los)/r**2 * los) / r   # target motion ACROSS the line of sight
    return N * los_rate                                           # gain N times that turn rate
```
*Honest: a candidate. It's genuine, self-contained math leCore lacks and it pairs with the propagator — but it
owes a measured baseline against greedy pursuit in leCore's navigator before it earns a place.* *Effort: LOW to
try; the value is the measurement.*

### 1c. Coverage-saturation stopping rule — "know when you've gathered enough" *(CANDIDATE, tiny)*

A small pattern leCore doesn't have as such: accumulate everything gathered into one **coverage vector** and stop
when it *spans* the goal — `cos(coverage, goal)` crosses a threshold. It complements `RecallNull` (which answers
*is this hit real?*) with a different, missing signal (*have I covered the goal region yet?*).

```python
# fold each hit into a coverage vector; you're done when coverage spans the goal (a saturation / stopping signal).
def coverage_step(coverage, hit, goal):
    coverage = coverage + unit(hit)                 # bundle the new hit in
    return coverage, float(cosine(coverage, goal))  # stop when this >= a threshold (e.g. 0.85)
```
*A helper, but a genuinely useful concept for any retrieval/search loop. Reuse `bundle`/`cosine`.* *Effort: LOW.*

---

## Part 2 — Integration: driving leCore from outside (agent harnesses, other programs)

leCore should be easy to embed in another program or drive from an agent harness (an autonomous LLM loop that
picks and calls tools). The good news from the codebase: **most of this already exists.** Confirm and reuse it;
the only real addition is a standard tool manifest.

### 2a. Already present — reuse, no work

| An external caller / harness wants… | leCore already has |
|---|---|
| to drive the engine over the network, from any language | **`service`** — a stdlib HTTP/JSON API (`serve(host, port, token, persist_path)`) |
| to discover capabilities and get the **exact call** to make | **`skills.suggest` / `route` / `complete`** — introspected off the code, always in sync; each result carries a confidence and the concrete call |
| to search capabilities in plain English | **`find_capability`** (+ the `catalog`) |
| to start long work and poll / cancel / resume it | **`jobs`** — start/pause/resume/cancel with checkpoints that survive a restart |
| to persist and restore engine state | **`UnifiedMind.save(path)` / `load(path)`** |
| named, tiered stores for its own data | **`query.Database.add_namespace(name, tier)`** + `insert` / recall / fuzzy WHERE |
| to inject its own LLM without leCore importing one | **`agent_bridge`** — pass a callable `llm(text)->reply`; optional |
| an event channel between app / person / agent | **`bus`** — `MessageBus`, topics, pub/sub |

### 2b. The one light addition — a standard tool manifest + invoke-by-name *(LIGHT)*

Agent harnesses expect tools in a standard **function-calling / JSON-schema** shape (name, description, parameter
schema) and a way to **call one by name** with JSON in/out. leCore already introspects the pieces (`skills` has
each method's name, signature, and summary); this just formats them and wires two endpoints on the existing
`service`.

```python
# expose leCore's capabilities as a STANDARD tool manifest a harness can consume, and let it call one BY NAME.
# Reuses: skills (already introspects name/signature/summary off the code) + service (already serves HTTP/JSON).
def tools_manifest(mind):
    return [{"name": s["name"], "description": s["summary"],
             "parameters": params_from_signature(s["signature"])}   # -> {type:"object", properties:{...}} JSON schema
            for s in mind.skills_index()]                           # REUSE: skills introspection

def invoke(mind, name, args):
    fn = getattr(mind, name)                                        # a capability IS a method
    return json_safe(fn(**args))                                    # serialize result (vectors -> lists) [REUSE: service._json_default]

# On the existing Service:  GET /tools -> tools_manifest(mind)   ;   POST /invoke {name,args} -> invoke(mind,name,args)
```

That's the whole harness story: one manifest formatter, one dispatcher, two endpoints — over machinery that's
already there. Any OpenAI-tools / MCP / LangChain-style harness can then list leCore's tools and call them.

### 2c. Two tiny embed-convenience helpers *(LIGHT)*

- **`from_external(vec, source)`** — let a vector computed *outside* leCore (by any embedder or sensor) enter the
  shared space tagged with where it came from, so multiple sources can be compared (feeds 1a):
  ```python
  def from_external(vec, source):                 # source e.g. "clip", "sensor_3"
      return bind(source_role(source), unit(vec)) # a bundle of these across sources = one multi-source vector [REUSE: bind]
  ```
- **A short `SUBSTRATE.md`** — one page documenting the stable surface an external program builds against
  (`service` endpoints, `skills`/`find_capability`, `jobs`, `save`/`load`, `add_namespace`, `agent_bridge`,
  `bus`), so callers never reach into internals. *(Docs, not code.)*

---

## Honest scope

- **Part 1a is a real new faculty; 1b/1c are candidates** — flagged because they're genuine math leCore lacks, but
  each owes a measured before/after against a baseline before it ships (1b vs greedy navigation; 1c vs a fixed
  budget). No win without a baseline.
- **Part 2 is mostly confirmation** — the HTTP service, capability discovery, jobs, and save/load already exist;
  the only new code is the tool manifest + invoke endpoints and the two tiny helpers. If a "harness" task grows
  beyond that, it's application glue that belongs in the caller, not in leCore.
- **No new dependencies.** Everything is numpy/stdlib. leCore never imports an LLM or embedding library — external
  models arrive as *callables* (`agent_bridge`) or *vectors* (`from_external`); leCore's own `UniversalEncoder` is
  the zero-dependency default.
- **Determinism holds** — all of Part 1 and the helpers are pure/seeded numpy.

## Anti-silo

Each new piece lands as a `UnifiedMind` faculty with a cross-faculty integration test that runs with no external
dependency: for `opponent`, two in-process mock estimators that agree on one item and disagree on another, asserting
the shared / exclusive / purple channels and the calibrated divergence; for the tool manifest, list `/tools` and
round-trip a `/invoke` call end-to-end over the in-process `service`. Candidates (1b/1c) wire into `navigator` / the
recall loop with their baseline tests when they graduate.

## Sequencing

1. **2b + 2c** — the tool manifest, invoke endpoints, `from_external`, and `SUBSTRATE.md`. Light, unblocks anyone
   driving leCore, and it's the fastest visible win.
2. **1a `opponent`** — the one clear engine improvement; small and reuse-heavy.
3. **1b / 1c** — try each behind a baseline; keep only if it beats greedy navigation / a fixed budget.

---

### Bottom line

Two buckets. **Engine improvements:** one clear new faculty — `opponent`, which decomposes how an ensemble of
estimates agrees and disagrees (shared / each-exclusive / residual-sum) and turns disagreement into a calibrated
signal, generalizing the forest's existing agreement score — plus two honest candidates, proportional navigation
(a guidance law leCore lacks) and a coverage-saturation stopping rule, each to be proven against a baseline.
**Integration:** leCore is already drivable from outside — HTTP/JSON `service`, introspected capability discovery
with the exact call, `jobs`, `save`/`load`, tiered stores, callable-LLM injection, an event bus — so making it
comfortable for an agent harness is light: format the already-introspected capabilities as a standard tool
manifest, add `/tools` + `/invoke` to the existing service, and ship two tiny helpers and a one-page substrate
doc. All numpy/stdlib, deterministic, readable, and reuse-heavy.

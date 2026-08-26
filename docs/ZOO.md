# leCore × openzoo — mounting the engine in stacc's zoo

openzoo.fun's own sentence — "bind a corpus once, ask it anything. Local x402 proxy +
MCP" — is a description of this engine's front door, so the integration is deliberately
thin. Three pieces, in order of leverage:

## 1. The MCP server (shipped: `holographic_mcp.py`)

leCore now speaks Model Context Protocol over stdio, stdlib-only, delegating to the
existing `/tools` + `/invoke` service (token gate, private-method refusals, and the
`{"__bytes_b64__": ...}` wire convention all inherited). An MCP host config is one block:

```json
{"mcpServers": {"lecore": {"command": "python3", "args": ["holographic_mcp.py"]}}}
```

Design honesty: 1,944 faculties would make an unusable `tools/list`, so the adapter
exposes a curated trio — `lecore_find` (the engine's own Rule-0 search),
`lecore_describe` (one faculty's full contract), `lecore_invoke` (run anything) — and
every faculty stays reachable through the third. Tool-level failures ride in `content`
with `isError` so the host's model sees the message and adapts, per MCP convention.

## 2. "Bind a corpus once, ask it anything" — the recipe

Shipped as first-class MCP tools: **`corpus_bind`** (documents in, content-addressed
handle out — rebinding the same corpus is idempotent) and **`corpus_ask`** (BM25-ranked
chunks with scores, best first; leCore retrieves, the host model reads and answers — the
MCP division of labor). Verified over real stdio on 400 WikiText chunks: "battle ship
armament guns" returns the Erzherzog Ferdinand Max's turret armament first at score 17.6.
Handles live for the server process; the zoo proxy owns persistence, and the error for an
unknown handle says exactly that. The deeper faculties remain one `lecore_invoke` away:
build an index or knowledge store over the corpus once (`Index` with a recall budget:
recall measured on *your* vectors, demote-to-exact when data defeats structure), then
query with calibrated abstention (promised false-alarm rate realized within binomial CI
— measured 0.013 @ α=0.01 on shuffled-real noise, power 1.000). A zoo answering machine
that can *decline to hallucinate* at a promised rate is a differentiator no other zoo
citizen offers.

## 3. The economics — rule-sized models for a 435-model zoo

Zoo hosting cost is storage × bandwidth × models. leCore's lane: a 175-byte model file
re-bakes 2,048 certified parameters bit-identically (weights-from-rule; Tracr-lane
systems store the weights); `ModelLibrary` packs many programs against shared certified
payloads in one <600-byte rule file; `holographic_recipe` compresses real LLM weights 3×
byte-exact; and `cold_store(codec='small')` holds cold float arrays at 1.19× where
general codecs get 1.08×. For x402 metering, the natural seam is `tools/call` on
`lecore_invoke` — one tool name, per-call pricing, no per-faculty price list needed.

## Verification

`python3 holographic_mcp.py --selftest` proves the protocol in-process: initialize,
curated trio, find/call round trips, private-faculty refusal through the inherited gate,
unknown-method -32601, silent notifications. The same file is the server; there is no
second implementation to drift.

## 4. What the heavy faculties cost — measured, and the pricing it implies

Cost census on the serving box (per call, warm):

| faculty family | compute | payload |
|---|---|---|
| bind (one HRR op, dim 512) | 0.025 ms | ~10 KB |
| corpus_ask (BM25, 400 chunks) | 3.2 ms | ~0.1 KB |
| image op, 24×24 certified | 0.11 ms | ~12 KB |
| physics, 100 steps stepped | 2.0 ms | ~1.2 KB |
| physics, 100 steps **collapsed** | **0.003 ms** | ~1.2 KB |

Three conclusions a zoo operator should price on. **Compute is nearly free** — everything
above runs 300–300,000 calls per CPU-second; at any cloud rate that's micro-cents. **The
wire often dominates**: bind's JSON payload outweighs its compute ~400:1, so metering
bytes matters more than metering ops for vector-returning faculties. **The installed lane
destroys marginal cost**: the same 100-step simulation is 680× cheaper collapsed than
stepped — leCore's compile-once architecture converts recurring compute into a one-time
compile, which in x402 terms means charge for the compile, serve queries at noise-level
cost. The *iterative* faculties are the heavy end, and even they measured cheaper than the
narrative said: two complete render-critique loops (installed renderer + reference eye +
two swarm roles) ran in **477 ms total** at reference scale; a shared workspace boots in
0.8 ms. Cost scales with rounds × resolution × eye — swap in a real vision tower at
production resolution and *that* is where seconds live. The round count stays a knob the
caller holds, and the original "seconds-class" guess is corrected here because a
measurement beat it.

Every `tools/call` response now carries `_meta["lecore.cost"] = {elapsed_ms,
payload_bytes}` — measured per call, reproducible because the engine is deterministic —
so the proxy bills reality instead of a price list.

## 5. Teaching the host model not to hand-roll

A zoo LLM hand-rolls what it doesn't know exists. Three mechanisms now make leCore's
capabilities impossible to miss, each aimed at a different moment in the model's life:

**At connect** — `initialize` returns an `instructions` block (MCP hosts inject this
into the model's context): Rule-0 translated for LLMs — *before implementing any
algorithm, call `lecore_map`, then `lecore_find`; hand-roll only after find returns
nothing relevant*. It also mentions that every call returns measured cost, so "cheaper
than your hand-rolled version" is a claim the model can verify.

**At task start** — `lecore_map`: the whole territory in one call. Twelve families, each
with a "never hand-roll" line (nearest-neighbor search, float compression, mesh ops,
certified image operators, physics stepping and fast-forward, forecasting, corpus QA,
bootstrap CIs, swarm deliberation, program-to-weights compilation, tiered memory,
composable generative models) and the exact phrases to hand to `lecore_find`.

**At every decision** — tool descriptions rewritten to be directive ("BEFORE
implementing any algorithm... hand-rolling what this returns is wasted tokens and
worse code").

The map is curated but **un-rottable by construction**: the selftest runs every listed
phrase through the live catalog and fails the build if any family's phrasing stops
resolving — the map is data, validated against the engine it describes.

## 6. Server-side leCore + installed models: what the closed loop buys

Running leCore on the serving box next to Unicron-modified models is not redundancy —
the outer engine and the installed weights share a *vocabulary of certified operators*,
and that shared vocabulary is where the magic lives. Five concrete dividends, each
marked by what's verified where:

**Certified inference — the model becomes auditable.** [engine side verified] Every
installed pathway carries a sha256 and a referee. A server-side leCore holds the same
certified matrices as ground truth, so it can *attest at serving time* that the model's
installed math still computes what the certificate says — spot-check a layer's output
against the exact operator, per request or per deployment. No other LLM stack can audit
its model's internals against a spec, because no other model *has* a spec.

**Two lanes to the same operator, referee live.** [needs the model — laptop lane] A
certified faculty exists twice: in-weights (fused into the forward pass, no tool-call
round-trip, model-native precision) and in-context (an MCP call to the exact f64
engine, micro-cent cost per the census). The router chooses per query; disagreements
between lanes are a *measurement*, not a mystery — the three-referee pattern extended
into production.

**Session state as a holographic object.** [engine machinery verified; model coupling
is laptop lane] The GDN head state *is* leCore's HRNN, and `SessionStore(carry=
"memory")` already writes constant-size session files. Server-side leCore can snapshot,
superpose, and transport session carries as first-class vectors — multi-tenant zoo
sessions at KB scale, x402-meterable, with the engine's cleanup/abstention machinery
available to read the model's own memory at calibrated confidence.

**Distribution by rule, not by bytes.** [distbus + coordinator selftests green] leCore
nodes all speak the same `/tools` + `/invoke` shape; the bus and coordinator ship
(pub/sub across machines, monoid reduce, shared-memory caches). Because the engine is
deterministic and models re-bake from recipes (3× byte-exact; 175-byte rule files),
*distributing a model is distributing kilobytes* — every node regenerates bit-identical
weights. And determinism makes results content-addressable: same query + seed = same
bytes, so the farm caches inference the way a CDN caches files — charge once, serve
the hash.

**The compile loop closes.** [engine side verified] Server-side leCore is a compiler
whose deployment target is the serving model: new faculties certify into matrices
(drift heads, gated targets, collapsed recurrences) and install between sessions
without retraining. The zoo's models grow capabilities as weight patches with
certificates attached — and the referee that checked the install stays resident to
keep checking it.

## 7. Ouroboros — the closed memory loop, named

The process in §7–8 has a name: **Ouroboros** — the engine consuming the memory produced
by the engine installed inside the model, and feeding it back. The serpent's mouth is
server-side leCore (read / write / delete / capacity-account / consolidate); the tail is
the installed model's memory in both its speeds — the GDN head state (fast, in-weights,
*our own HRR trace* in the host's clothes) and the durable partition (§8). Everything
below is Ouroboros.

### Managing the installed model's memory from outside

Because the GDN head state *is* leCore's holographic memory (S += b·k·vᵀ with decay — an
outer-product accumulator, our data structure in the host's clothes), a server-side
leCore can be the model's **external memory manager**, and every claim below was measured
on the exact GDN algebra (dₖ=128, 40-pair session, a=0.98):

**Read** the model's memory without running the model: the model's own readout Sᵀk
returns the stored value at cos 0.935 (newest) decaying to 0.678 (oldest) — age-graded
recall leCore can inspect per key. **Write** facts the model never saw, in the state's
native algebra: an externally injected binding reads back at **0.951** by the model's own
readout — zero forward passes. **Delete** using only the readout estimate (no ground
truth): 0.951 → −0.236. **Account** capacity: the crosstalk law predicted mean recall
0.932 vs 0.905 measured — the manager knows *when the state is saturating* before the
model starts confabulating, which is abstention applied to the model's own memory.

**Optimize — with a kept negative that shapes the design.** Rehearsing from the state's
*own reads* measured **negative** (0.767 → 0.730, and it damaged fresh memories too):
consolidation from your own noise is self-pollution. But the external manager holds the
*transcript* — ground truth it legitimately owns — and transcript consolidation lifted
oldest-10 recall **0.767 → 0.918** at a small, measured tax on the newest (0.905 →
0.872). Sleep-style memory consolidation, run externally, priced honestly.

**Economics:** the session state is low-rank by construction — exact factors store it
1.59× smaller at 2.9e-16 error; rank-20 truncation is 3.2× smaller at a measured recall
cost (0.905 → 0.870) — a compression/recall dial the x402 proxy can price. All of this
composes with the constant-size session carry and content-addressed caching from §6.

## 8. The external-memory partition — a directory the model remembers with

Taking the architecture literally: assign a partition (a directory; one per tenant via
`LECORE_MEMORY_ROOT` or `memory_root=`) as the model's external memory, and regard it as
an ordinary leCore data structure. Two MCP tools make it the model's own: `memory_write`
(ids, content hashes, dedupe, tags) and `memory_search` (ranked, best first) — and the
connect-time charter tells the model it *has* persistent memory and should check it
before claiming it doesn't remember. The partition is a `KnowledgeStore`, so it outlives
the server process (pinned: a fresh server over the same root finds the same memories),
and because it's a real store rather than a scratch string, the whole engine applies to
it — compression, cold tiering, audit, distribution over the bus, the works. A workspace,
a database, session notes: whatever the tenant's model accumulates, managed like any
other leCore structure. The closed loop from §7 composes: the GDN state is the model's
fast in-weights memory; the partition is its durable one; leCore manages both.

## 9. Void exploration — the discovery loop, aimed at the LLM

Vanilla leCore's Void Explorer becomes a killer app when the LLM is in the loop, because
each side does what only it can: **leCore finds *measured* voids** (never brainstorming
— every candidate carries a statistical warrant, and thin structure gets the epicycle
refusal: "the corpus's grammar has no right to vouch for unseen combinations"); **the
model elaborates** candidates into hypotheses; **the engine verifies** — corpus_ask for
evidence, the abstraction ladder (`ladder_summary`) for structural grounding, calibrated
abstention for the final honesty gate.

Shipped: the `void_explore` MCP tool runs `structured_voids` (the Mendeleev move —
combinations the corpus's slot structure licenses but the corpus lacks) over any bound
corpus, gate included. The cross-disciplinary warrant is one `lecore_invoke` away:
`transfer_voids` finds regions *present in corpus B, absent in corpus A* — "reality
already contains it, elsewhere," the strongest warrant short of execution.

**Measured on real data this session:** two Wikipedia topic slices (naval engineering vs
music, 300 chunks each), shared encoder space over a discriminative 3-axis frame,
**15 of 48 transfer candidates kept** — dense in music, A-density down to −0.002 —
each grounding to real chunks (singles, chart trajectories, release discourse: knowledge
structures naval articles lack). And the instrument-honesty arc that got there is the
point: the first run kept **zero** because the probed bandwidth was 90× wider than the
domain gap — a units mismatch, diagnosed from the z-distributions and fixed by
standardizing to the data's own scale — and the toy corpus was refused outright by the
gate. A discovery tool you can trust is one that mostly says no.

## 10. The Leap — answering the critics with a measurement

The standing criticism: LLMs cannot leave the shape of their training data — the latent
space is chaotic, so "outside the distribution" is indistinguishable from noise. The
void machinery converts that from a philosophical objection into an engineering claim,
because it makes *outside* an **addressable, warranted, gated** set: not "generate
something weird" but "here is the specific combination your knowledge's structure
licenses and your knowledge lacks, with a p-value on the structure's right to vouch."

**The closed leap, measured this session** (synthetic-exact, the referee for the
installed lane): a memory corpus with real slot structure and one licensed gap →
`structured_voids` passes its gate at **p=0.020** and returns the held-out combination
as the *sole* candidate → the target is encoded and written through the Ouroboros mouth
into the GDN state → the model's own readout recalls the leapt fact at **cos −0.000 →
0.759**, existing memories intact (min 0.739). Find the gap, leap, remember — every
step warranted.

And the road there is the trust argument: **three wrong plants were refused before one
right plant passed** — a near-complete factorial (independent slots: "every unseen
combination is equally valid" = noise wearing a grammar), a uniform-count corpus (no
concentration for the shuffle test to detect), and a thin corpus whose pairs fell below
the support floor. A leap engine that fires on anything is a hallucination engine with
better marketing; this one refused three times, then leapt once, correctly.

Ouroboros improves with it, structurally: the loop's consolidation step (§7) can now be
*targeted* — instead of rehearsing what memory already holds, the manager asks the void
instruments what memory is missing *that its own structure licenses*, acquires or
elaborates exactly that (the LLM's job), and writes it back through the mouth. The
snake doesn't just eat its tail; it grows where the growth is warranted.

## 11. The stacc lane — three things nobody else can offer

Only one operator has 435 models behind a *deterministic* engine with priced calls. That
combination unlocks products that don't exist elsewhere, and this section is their spec.

**1. Proof-of-inference receipts (shipped).** Every `tools/call` now carries
`_meta["lecore.receipt"]`: sha256 of the canonical input, sha256 of the output,
`deterministic: true`. Because the engine's outputs are functions of (tool, arguments)
alone, that pair is a complete, *re-verifiable* claim about what was computed — and the
`receipt_verify` tool settles any dispute by re-running and comparing 64 hex chars.
Billing audits, cache validation, third-party verification: no zero-knowledge machinery,
because **determinism is the proof system**. Wall-clock deliberately lives in the cost
block, not the receipt — time is the one thing an honest re-run won't reproduce. The
x402 corollary: "charge once, serve the hash" — identical requests hit the receipt
cache at marginal cost zero, and the receipt *proves* the cached answer is the answer.

**2. The federated leap (shipped).** `void_explore` takes a second corpus handle:
combinations corpus A's own grammar licenses but A lacks, checked for *instantiation in
corpus B* — the transfer warrant ("reality already contains it, elsewhere") in discrete
form, across tenants. Pinned end to end: a planted licensed-absent triple in A, present
in B, surfaces flagged with the warrant. This is a product category that has never
existed: **cross-corpus discovery with statistical warrants** — tenant A pays to learn
what the rest of the zoo's knowledge instantiates in A's own blind spots, gate-refused
when A's structure can't vouch. Novelty as a priced, warranted good.

**3. Memory as a commodity (laptop lane, referee shipped).** Ouroboros states are exact
snapshot/restore artifacts with low-rank factor compression (1.59× exact) and capacity
certificates. A session state — or a pre-loaded expert memory written externally at
0.951 readback with zero forward passes — is a content-addressed file stacc can host,
price, and let buyers *verify by receipt* before purchase. The synthetic-exact numbers
in §7 are the referee; the real-weight lane runs where the weights live.

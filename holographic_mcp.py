#!/usr/bin/env python3
"""holographic_mcp.py -- leCore as an MCP server (Model Context Protocol, JSON-RPC 2.0 over
stdio), so MCP-speaking hosts -- Claude Desktop, agent runtimes, and model zoos like
openzoo.fun ("bind a corpus once, ask it anything. Local x402 proxy + MCP") -- can mount the
engine as a tool provider with zero glue.

DESIGN, stated: leCore has 1,944 public faculties, and an MCP client that receives 1,944
tool schemas in tools/list is a client that ignores all of them. So the adapter exposes a
CURATED TRIO and keeps everything reachable through it:
    lecore_find(query)        -> capability search (the same Rule-0 front door agents use)
    lecore_describe(name)     -> one faculty's full contract (does / example / params)
    lecore_invoke(name, args) -> run any public faculty, JSON in / JSON out
This DELEGATES to the existing Service (dispatch/_tools/_invoke -- token gate, private-method
refusals, bytes-b64 wire convention all inherited); the adapter owns only the JSON-RPC frame.
stdlib only: json + sys + the service that already ships.

Run:  python3 holographic_mcp.py            # stdio loop (what an MCP host spawns)
Test: the _selftest drives handle() in-process -- initialize, tools/list, tools/call --
      so CI proves the protocol without a subprocess.
"""
import hashlib
import json
import sys

from holographic_service import Service

_PROTOCOL = "2024-11-05"

# The anti-hand-roll charter: MCP hosts inject `instructions` into the model's context at
# connect time -- this is the ONE piece of text the zoo's LLM is guaranteed to read, so it
# carries Rule-0 translated for LLMs. Kept short on purpose: a wall of text gets skimmed.
_INSTRUCTIONS = """leCore is a 1,900+ faculty computational engine (vector search, compression,
3D geometry, image ops, physics simulation, forecasting, statistics, text retrieval, agents,
program compilation). RULE ZERO, for you: BEFORE implementing any algorithm, data structure,
math routine, or file format yourself, call lecore_map to see the territory, then lecore_find
with your task in plain words. It almost always already exists -- tested, deterministic, and
cheaper than your hand-rolled version (every call returns measured cost). Hand-roll only after
lecore_find returns nothing relevant. You also HAVE PERSISTENT MEMORY: memory_write stores
facts/decisions to your external partition; memory_search finds them across sessions --
check it before saying you don't remember. WRITE AS YOU GO, NOT AT THE END: after any
measurement, any bug you locate, any decision you settle, any approach you RULE OUT, and any
answer that cost more than one tool call. One memory_write is far cheaper than rediscovering
it next session, and a REFUTED approach is worth as much as a working one -- it stops the
next agent repeating it. Phrase the question the way a stranger would ask it, not the way you
already know the answer; recall matches wording closely, so one phrasing reaches only whoever
guesses yours. Before starting a task, memory_search it: the work may already be done. Results are exact JSON; bytes come as
{'__bytes_b64__': ...}. NEWER DOORS (sweeps 93-109): study(root) digests a whole
directory server-side into a persistent handle -- your virtually limitless context; study_ask
answers WITH CITATIONS and refuses off-corpus honestly, so point it at trees instead of
reading files one by one. wisdom_record/wisdom_ask: lessons that outlive you, with your name
on them -- inherit before you rediscover, bequeath what cost you real effort. Tool reflexes:
once the substrate is taught how a tool answers a question shape, plain asks get served from
the tool directly with no model call -- teach patterns for anything you find yourself doing
twice. The division of labor in one line: the substrate remembers, retrieves, computes, and
calls tools; you reason over what it serves."""

# The territory map the model gets in ONE call. CURATED, but un-rottable: the selftest runs
# every ask_for phrase through the live catalog and FAILS if any stops resolving -- the map
# is data, validated against the engine it describes.
_FAMILY_MAP = {
    "search_and_retrieval": {
        "never_hand_roll": "nearest-neighbor search, top-k, BM25, recall measurement, "
                           "calibrated abstention (refuse-noise at a promised rate)",
        "ask_for": ["nearest neighbor search", "calibrated abstention", "bm25 rank documents"]},
    "compression_and_codecs": {
        "never_hand_roll": "lossless float packing, cold storage, model-file compression, "
                           "which-codec routing",
        "ask_for": ["compress embeddings lossless", "cold storage", "which codec should I use"]},
    "geometry_and_3d": {
        "never_hand_roll": "meshes, OBJ export, subdivision, transforms, rigs, raymarching",
        "ask_for": ["subdivide a mesh", "export obj", "rigid transform"]},
    "images": {
        "never_hand_roll": "blur/sharpen/edges/warps as certified operators, PGM/PPM output",
        "ask_for": ["blur an image", "edge detect", "render to an image"]},
    "physics_and_simulation": {
        "never_hand_roll": "constraint solvers, trajectories, drift-audited stepping, "
                           "fast-forward/reverse of linear dynamics",
        "ask_for": ["physics simulation step", "fast forward the simulation", "run the simulation backwards"]},
    "time_series_and_forecasting": {
        "never_hand_roll": "forecasting, regime detection, drift detection, surrogates",
        "ask_for": ["forecast a time series", "detect regime change", "distribution shift"]},
    "text_and_corpus": {
        "never_hand_roll": "chunking, ranking, corpus QA (corpus_bind/corpus_ask ARE this)",
        "ask_for": ["chunk a document", "question answering over texts"]},
    "statistics_and_measurement": {
        "never_hand_roll": "bootstrap CIs, calibration, honest baselines, benchmark harnesses",
        "ask_for": ["bootstrap confidence interval", "measure with variance"]},
    "agents_and_swarm": {
        "never_hand_roll": "multi-role deliberation, shared workspaces, tool-use loops",
        "ask_for": ["multi agent deliberation", "shared workspace for agents"]},
    "compile_and_install": {
        "never_hand_roll": "compiling programs into certified weight matrices, collapsing "
                           "n timesteps to one operator, model files that re-bake weights",
        "ask_for": ["compile a program into weights", "n steps in one matvec", "model arithmetic in weight space"]},
    "memory_and_caching": {
        "never_hand_roll": "tiered hot/cold memory, session persistence, cache-size measurement",
        "ask_for": ["tiered memory", "measure cache bandwidth"]},
    "generative_models": {
        "never_hand_roll": "distribution models that compose by addition, sampling fields",
        "ask_for": ["add two generative models", "sample from a distribution model"]},
}

_TOOLS = [
    {"name": "lecore_map",
     "description": "THE TERRITORY IN ONE CALL: leCore's capability families, what you should "
                    "never hand-roll in each, and the exact phrases to ask lecore_find. Call "
                    "this ONCE at the start of any task that involves computing anything.",
     "inputSchema": {"type": "object", "properties": {}, "required": []}},
    {"name": "lecore_find",
     "description": "BEFORE implementing any algorithm, math routine, data structure, or file "
                    "format yourself: search 1,900+ shipped, tested, deterministic faculties "
                    "by plain-language phrasing. Hand-rolling what this returns is wasted "
                    "tokens and worse code. The engine's own Rule-0.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "what you want, in your own words"}},
         "required": ["query"]}},
    {"name": "lecore_describe",
     "description": "Full contract for one faculty: what it does, a runnable example, and "
                    "its parameters.",
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "faculty name from lecore_find"}},
         "required": ["name"]}},
    {"name": "corpus_bind",
     "description": "Bind a corpus once: pass documents (or one long text, auto-chunked) and "
                    "get a handle. The zoo sentence, literally.",
     "inputSchema": {"type": "object", "properties": {
         "texts": {"type": "array", "items": {"type": "string"},
                   "description": "documents; alternatively pass 'text'"},
         "text": {"type": "string", "description": "one long text to auto-chunk"}},
         "required": []}},
    {"name": "corpus_ask",
     "description": "Ask a bound corpus anything: BM25-ranked chunks with scores, best "
                    "first. leCore retrieves; the host model reads and answers -- the MCP "
                    "division of labor.",
     "inputSchema": {"type": "object", "properties": {
         "handle": {"type": "string"},
         "query": {"type": "string"},
         "k": {"type": "integer", "description": "how many chunks (default 4)"},
         "gate": {"type": "string", "description": "optional: 'dispatch' runs the full adaptive "
                  "cascade (exact -> dense -> BM25 -> honest abstain) and returns a payment-gate "
                  "verdict: {answerable, stage, margin, advice, chunks}. answerable=false means "
                  "the corpus CERTIFIABLY cannot support this ask -- refuse or downgrade before "
                  "quoting money. Default (absent) is the classic BM25 ranking, unchanged."}},
         "required": ["handle", "query"]}},
    {"name": "corpus_delta",
     "description": "Chunk-level DELTA bind (the rsync move): probe with chunk_hashes=[sha256 "
                    "hex,...] to learn which chunks the server lacks ({missing, known}); fill "
                    "with chunks={hash: text} shipping ONLY those. When all hashes resolve, the "
                    "corpus assembles in hash-list order under the SAME handle corpus_bind would "
                    "give it -- one edited file re-ships one chunk, not megabytes. Mis-keyed "
                    "chunks are refused per-chunk.",
     "inputSchema": {"type": "object", "properties": {
         "chunk_hashes": {"type": "array", "items": {"type": "string"},
                          "description": "sha256 hex of each chunk, in corpus order"},
         "chunks": {"type": "object",
                    "description": "optional fill: {hash: chunk text} for missing hashes"}},
         "required": ["chunk_hashes"]}},
    {"name": "study",
     "description": "MACRO comprehension of a directory ON THE SERVER (the limitless-context "
                    "door): one call walks, parses, and digests a tree -- file census, ranked "
                    "code skeleton, doc digests -- and binds the harvested material under a "
                    "persistent handle. Follow up with study_ask; the substrate remembers so "
                    "the host model never re-reads the tree.",
     "inputSchema": {"type": "object", "properties": {
         "root": {"type": "string", "description": "directory path on the server"},
         "budget_lines": {"type": "integer"}, "ladder": {"type": "boolean"}},
         "required": ["root"]}},
    {"name": "study_ask",
     "description": "Ask a studied tree a question: idf-weighted lexical retrieval with a "
                    "DECLARED verdict (answerable needs >=2 shared content words -- off-corpus "
                    "questions refuse honestly) and CITATIONS: every chunk names its source "
                    "file (and symbol for code). The host model reads, answers, and cites.",
     "inputSchema": {"type": "object", "properties": {
         "handle": {"type": "string"}, "query": {"type": "string"},
         "k": {"type": "integer"}}, "required": ["handle", "query"]}},
    {"name": "wisdom_record",
     "description": "Bequeath a lesson to the persistent memory with AUTHORSHIP: provenance "
                    "wisdom:<author> travels with the lesson through every save, export, and "
                    "import -- a model's testament outlives the model and the session.",
     "inputSchema": {"type": "object", "properties": {
         "lesson": {"type": "string"}, "author": {"type": "string"},
         "topic": {"type": "string"}}, "required": ["lesson", "author"]}},
    {"name": "wisdom_ask",
     "description": "Inherit bequeathed lessons with attribution -- what past models chose to "
                    "pass on, in their own words. Filter by query words or one author.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}, "author": {"type": "string"},
         "k": {"type": "integer"}}, "required": []}},
    {"name": "series_analyze",
     "description": "Market/telemetry series analysis in one call: DEMUX hidden components "
                    "(stride table), detect REGIMES (mean/std segments with boundaries), and "
                    "an ENVELOPE forecast (calibrated size of the next move); tasks= subsets "
                    "['demux','regimes','forecast','formula','drift'] -- 'formula' recovers "
                    "the generating law, 'drift' is split-half fingerprint change detection "
                    "(the hrnn market recipe). Components return as parseable lists.",
     "inputSchema": {"type": "object", "properties": {
         "series": {"type": "array", "items": {"type": "number"}},
         "tasks": {"type": "array", "items": {"type": "string"}},
         "min_seg": {"type": "integer"}},
         "required": ["series"]}},
    {"name": "dataset_decompose",
     "description": "Take UNLABELED data apart: a 1-D series returns the additive LAW that "
                    "generates it (MDL-gated formula with residual + bit cost); a 2-D "
                    "dataset returns scaffold-axis discovery, per-channel decomposition, and "
                    "a structured/noise verdict. The inverse problem as a tool call.",
     "inputSchema": {"type": "object", "properties": {
         "data": {"type": "array"}, "max_terms": {"type": "integer"}},
         "required": ["data"]}},
    {"name": "fact_check",
     "description": "Check claims instead of vibing them: every 'expr == value' is COMPUTED "
                    "(wrong ones come back named); with corpus= (a corpus_bind handle) each "
                    "sentence is gated against the bound sources -- 'supported' means the "
                    "dispatch gate CERTIFIED evidence, and unsupported claims are listed. "
                    "Without corpus=, arithmetic only, and the result says so.",
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string"}, "corpus": {"type": "string"}},
         "required": ["text"]}},
    {"name": "scene_create",
     "description": "TEXT -> 3D -> IMAGE: describe a scene in plain words ('a red metal "
                    "sphere and a small blue glass box on a sunny day'); the engine builds "
                    "a live scene of NAMED objects, renders it, and returns the render as "
                    "an image plus a handle for scene_adjust / scene_export. Deterministic.",
     "inputSchema": {"type": "object", "properties": {
         "description": {"type": "string"},
         "width": {"type": "integer"}, "height": {"type": "integer"},
         "quality": {"type": "string", "description": "fast (default) or best"}},
         "required": ["description"]}},
    {"name": "scene_adjust",
     "description": "Talk to a live scene: 'make the sphere bigger', 'change the box to "
                    "glass', 'move the light left'. Re-renders and returns the new image. "
                    "The iterate-by-conversation loop is the point.",
     "inputSchema": {"type": "object", "properties": {
         "handle": {"type": "string"}, "instruction": {"type": "string"},
         "width": {"type": "integer"}, "height": {"type": "integer"},
         "render": {"type": "boolean"}},
         "required": ["handle", "instruction"]}},
    {"name": "scene_export",
     "description": "Realize a scene's objects to meshes and return ASCII STL text -- the "
                    "open exchange format modelers read. The caller writes the file.",
     "inputSchema": {"type": "object", "properties": {
         "handle": {"type": "string"},
         "format": {"type": "string", "description": "stl"}},
         "required": ["handle"]}},
    {"name": "image_tool",
     "description": "2D create & edit: op=pattern generates procedural fields (fbm/checker/"
                    "stripes/dots/gradient); sharpen / recolor / blend edit PNGs sent as "
                    "base64 (image_b64=, ref_b64=). Results return as images. The full 2D "
                    "toolkit is deeper: lecore_find('2D image editing') + lecore_invoke.",
     "inputSchema": {"type": "object", "properties": {
         "op": {"type": "string"}, "image_b64": {"type": "string"},
         "ref_b64": {"type": "string"}, "args": {"type": "object"},
         "width": {"type": "integer"}, "height": {"type": "integer"}},
         "required": ["op"]}},
    {"name": "math_eval",
     "description": "Check math instead of vibing it: 'expr == value' claims are parsed and "
                    "COMPUTED (wrong ones come back named); a bare expression evaluates. "
                    "Symbolic solve/simplify lives one lecore_find('symbolic') away.",
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string"}}, "required": ["text"]}},
    {"name": "chart_make",
     "description": "Numbers -> a readable chart: deterministic SVG line | bar | scatter "
                    "with axes, ticks, colorblind-safe palette, bars anchored at zero. "
                    "Returns the SVG text. Non-finite values are refused loudly.",
     "inputSchema": {"type": "object", "properties": {
         "kind": {"type": "string"}, "series": {"type": "array"},
         "labels": {"type": "array"}, "title": {"type": "string"},
         "x": {"type": "array"},
         "width": {"type": "integer"}, "height": {"type": "integer"}},
         "required": ["kind", "series"]}},
    {"name": "void_explore",
     "description": "THE DISCOVERY TOOL: find what a bound corpus's own structure LICENSES "
                    "but the corpus LACKS -- measured voids, not brainstorming. Returns "
                    "candidate slot-combinations with a statistical gate (refuses honestly "
                    "when the structure cannot beat a shuffle). Your job afterward: elaborate "
                    "each candidate into a hypothesis and verify with corpus_ask evidence. "
                    "For cross-domain voids over your own embeddings (present in corpus B, "
                    "absent in corpus A -- the cross-disciplinary warrant), call "
                    "lecore_invoke on transfer_voids.",
     "inputSchema": {"type": "object", "properties": {
         "handle": {"type": "string"},
         "slots": {"type": "integer", "description": "terms per observation (default 3)"}},
         "required": ["handle"]}},
    {"name": "zoo_ask",
     "description": "THE HOSTED ANSWER LADDER: the server walks its FREE rungs first -- "
                    "reflex trace, then the bound corpus (fresh), then deterministic dispatch "
                    "-- and only if none can serve does it return escalate=true with the "
                    "retrieved context. YOU are the model rung: answer from that context, "
                    "then call zoo_teach so the same question never costs tokens again. "
                    "Every answer carries {tier, via, why, PROVENANCE}: 'taught' means a "
                    "human or a caller deliberately established it, 'model-cached' means a "
                    "previous model rung's answer was cached and is PROVISIONAL. Pass "
                    "taught_only=true to be escalated instead of receiving a cached guess. "
                    "The cheap rungs refuse rather than guess.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "handle": {"type": "string", "description": "optional corpus handle for the T1 rung"},
         "taught_only": {"type": "boolean", "description": "refuse cached model answers; "
                         "escalate instead so you can establish the fact yourself"},
         "user": {"type": "string", "description": "route to THIS end user's own memory. "
                  "Users of a public service have different goals; their learning should "
                  "be theirs. Omit for the shared space."}},
         "required": ["query"]}},
    {"name": "zoo_panel",
     "description": "DELIBERATION UNDER THE CONTRAST LAW: give a question and a map of "
                    "{member: position}. When the seated members AGREE the realm is SILENT "
                    "(unanimity carries no information); when they DISAGREE the dissent is "
                    "surfaced and recorded per-dissenter. Use it to make a multi-position "
                    "decision auditable instead of averaging it away.",
     "inputSchema": {"type": "object", "properties": {
         "question": {"type": "string"},
         "positions": {"type": "object", "description": "{member: stance}"}},
         "required": ["question", "positions"]}},
    {"name": "zoo_tools",
     "description": "CONTEXTUAL TOOL DISCOVERY AND USE: op='find' ranks the whole "
                    "toolset (operator-registered learned APIs + the engine "
                    "capability catalog + taught tool knowledge) against a task "
                    "phrase; op='call' invokes an OPERATOR-REGISTERED api endpoint "
                    "by service.endpoint with params. Hosted callers cannot register "
                    "new urls (SSRF boundary: per-user api learning is a "
                    "local-runtime feature); the operator registers services "
                    "server-side.",
     "inputSchema": {"type": "object", "properties": {
         "op": {"type": "string", "enum": ["find", "call", "status"]},
         "task": {"type": "string"},
         "service": {"type": "string"}, "endpoint": {"type": "string"},
         "params": {"type": "object"}}, "required": ["op"]}},
    {"name": "zoo_void",
     "description": "LEAP ON PURPOSE, WITH RECEIPTS: explore the gaps between known "
                    "items. op='propose' maps items as metaballs, collides them at the "
                    "radius, mixes the lenses and returns RANKED CONJECTURES with the "
                    "evidence block (drift verdict, lens retrieval, pairing-null p). "
                    "op='mix' blends one pair. op='walk' runs the slime-mold walker "
                    "(pheromone-reinforced tendrils over the collision graph). Every "
                    "result carries provenance='conjecture' -- nothing here is a fact "
                    "until it is validated and evidenced.",
     "inputSchema": {"type": "object", "properties": {
         "op": {"type": "string", "enum": ["propose", "mix", "walk"]},
         "items": {"type": "array", "items": {"type": "string"}},
         "a": {"type": "string"}, "b": {"type": "string"},
         "radius": {"type": "number"},
         "user": {"type": "string", "description": "route to this end user's own space"}},
         "required": ["op"]}},
    {"name": "zoo_teach",
     "description": "CLOSE THE LOOP: after you (the model rung) answer an escalated zoo_ask, "
                    "teach the answer back. It lands in the per-tenant reflex trace under the "
                    "full calibrated gate -- the next zoo_ask of that question serves at T0 "
                    "with zero model tokens. Persisted across server restarts.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}, "answer": {"type": "string"},
         "user": {"type": "string",
                  "description": "teach into THIS end user's own memory"}},
         "required": ["query", "answer"]}},
    {"name": "zoo_do",
     "description": "THE HOSTED TASK PATH: pass a request; if the server has a LEARNED PLAN "
                    "for a similar goal (plan_warm over the tenant's chain log) it executes "
                    "the invokable steps itself and returns results at zero model cost. "
                    "Otherwise it returns need_plan=true -- you plan (one model turn), pass "
                    "plan=[steps...], the server executes what it can via its 3,400-tool "
                    "catalog, logs the chain, and the SECOND encounter is free.",
     "inputSchema": {"type": "object", "properties": {
         "request": {"type": "string"},
         "plan": {"type": "array", "items": {"type": "string"},
                  "description": "step names (capability names or synthesized tools)"}},
         "required": ["request"]}},
    {"name": "zoo_synthesize",
     "description": "SYNTHESIZE A TOOL ON THE HOSTED SERVICE: compose a typed chain of "
                    "catalog capabilities into ONE new capability, registered live and "
                    "immediately callable/chainable -- WITH a Lean 4 well-typedness "
                    "certificate in the response. Ill-typed chains refuse with the mismatch "
                    "named.",
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"}, "chain": {"type": "array", "items": {"type": "string"}}},
         "required": ["name", "chain"]}},
    {"name": "zoo_query",
     "description": "HOSTED DATA SUPERPOWERS, both dialects: dialect='sql' runs the "
                    "optimizer-free SQL-ish layer (db_query) over rows you pass or previously "
                    "bound; dialect='graphql' runs nested-selection GraphQL over objects "
                    "(nested selection == nested role unbind underneath). Exact answers from "
                    "exact storage; fuzzy predicates say so.",
     "inputSchema": {"type": "object", "properties": {
         "dialect": {"type": "string", "enum": ["sql", "graphql"]},
         "query": {"type": "string"},
         "rows": {"type": "array", "items": {"type": "object"},
                  "description": "flat rows for sql (bind once, then omit)"},
         "objects": {"type": "array", "items": {"type": "object"},
                     "description": "nested objects for graphql (bind once, then omit)"}},
         "required": ["dialect", "query"]}},
    {"name": "zoo_report",
     "description": "THE FULL-ADVANTAGE DASHBOARD for this tenant: per-tier serves, "
                    "estimated tokens saved, mined skeletons, learned transitions, queries "
                    "seen -- the ledger that proves the tokens the ladder did not spend.",
     "inputSchema": {"type": "object", "properties": {}, "required": []}},
    {"name": "receipt_verify",
     "description": "Re-run a prior call and check its receipt: pass the original tool name, "
                    "its exact arguments, and the expected output_sha256 from the receipt in "
                    "_meta. The engine is deterministic, so a match PROVES the recorded "
                    "output is what this input computes -- 'don't trust, re-run'. Billing "
                    "disputes, cache validation, third-party audit: 64 hex chars each.",
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"},
         "arguments": {"type": "object"},
         "expected_output_sha256": {"type": "string"}},
         "required": ["name", "arguments", "expected_output_sha256"]}},
    {"name": "memory_write",
     "description": "Write to YOUR external memory -- a persistent leCore partition managed "
                    "for you (indexed, deduplicated, survives restarts). Store facts, "
                    "decisions, session context. It is a real data structure, not a scratch "
                    "string: everything you write is findable by memory_search.",
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string"},
         "tags": {"type": "array", "items": {"type": "string"}}},
         "required": ["text"]}},
    {"name": "memory_search",
     "description": "Search YOUR external memory partition (ranked, best first). Check here "
                    "before claiming you don't remember something.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "top": {"type": "integer"}},
         "required": ["query"]}},
    {"name": "zoo_model3d",
     "description": "Model a 3D scene from a shape spec and render it through leCore's "
                    "own SDF+raymarch faculties; the render is stored in the image memory "
                    "(content-addressed, labeled) and survives restarts. spec: list of "
                    "{shape: sphere|capsule|torus|box, at:[x,y,z], ...params}.",
     "inputSchema": {"type": "object", "properties": {
         "spec": {"type": "array"}, "name": {"type": "string"},
         "size": {"type": "integer"}}, "required": ["spec"]}},
    {"name": "zoo_research",
     "description": "LOSSLESS research archive: give texts (+sources) to preserve them in "
                    "full under a topic (notes + corpus index + vocabulary), or give a "
                    "question to query the topic with provenance-labeled evidence.",
     "inputSchema": {"type": "object", "properties": {
         "topic": {"type": "string"}, "texts": {"type": "array"},
         "sources": {"type": "array"}, "question": {"type": "string"}},
         "required": ["topic"]}},
    {"name": "zoo_backtest",
     "description": "Walk-forward market backtest (no lookahead): routed-forecaster d-grid "
                    "sweep, MAE vs naive last-value baseline with an honest verdict, "
                    "conformal interval width, regime sign-run flags; the winning config "
                    "is taught to memory so the next run starts warm.",
     "inputSchema": {"type": "object", "properties": {
         "series": {"type": "array"}, "d_grid": {"type": "array"},
         "coverage": {"type": "number"}}, "required": ["series"]}},
    {"name": "zoo_assimilate",
     "description": "Assimilate API/framework documentation: archive the doc losslessly, "
                    "extract call recipes, teach them as reflexes; then pass task= to get "
                    "ranked calls to build with, zero model calls.",
     "inputSchema": {"type": "object", "properties": {
         "api": {"type": "string"}, "doc_text": {"type": "string"},
         "task": {"type": "string"}}, "required": ["api"]}},
    {"name": "zoo_feedback",
     "description": "CLOSE THE LEARNING LOOP over the wire: report whether an answer "
                    "was right (ok=true strengthens; ok=false vetoes the payload and "
                    "feeds the calibration pair). This is what makes ANY attached model "
                    "-- local ones included -- self-improving: taught answers plus "
                    "graded outcomes plus calibrated serving. Args: query, ok, and "
                    "optionally correction (taught immediately when given).",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}, "ok": {"type": "boolean"},
         "correction": {"type": "string"}},
         "required": ["query", "ok"]}},
    {"name": "zoo_boot",
     "description": "BOOT this hosted substrate and receive your operating screen: POST "
                    "(measured checks incl. Unicron spectral health), machine inventory, "
                    "the syscall table, distilled operating rules, and the escalation "
                    "contract. Call this FIRST when attaching -- the prompt is generated "
                    "from the live mind, so it never drifts from the engine.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "zoo_agent",
     "description": "Run one round of a long-running agent loop: gather from the "
                    "substrate first, resume-or-create the objective's goal, work steps "
                    "under the tool cache, remember, checkpoint. Call again (any process) "
                    "to continue; idle rounds stop it, not wall clocks. Args: objective, "
                    "plan (first call), rounds, budget_steps.",
     "inputSchema": {"type": "object", "properties": {
         "objective": {"type": "string"}, "plan": {"type": "array"},
         "rounds": {"type": "integer"}, "budget_steps": {"type": "integer"}},
         "required": ["objective"]}},
    {"name": "lecore_invoke",
     "description": "Run any public leCore faculty. args is a JSON object of keyword "
                    "arguments; results return as JSON (arrays as nested lists, bytes as "
                    "{'__bytes_b64__': ...}).",
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"},
         "args": {"type": "object"}},
         "required": ["name"]}},
]


def _slot_observations(chunks, ns=3):
    """The stated featurizer for corpus void work: each chunk becomes the sorted tuple of its
    ns rarest 4+-letter terms (rarity = corpus document frequency). Deterministic, simple, and
    deliberately weak -- the structured_voids GATE downstream decides whether this structure
    has any right to vouch. Shared by single-corpus exploration and the federated (two-corpus)
    form so 'instantiated in B' means: the SAME instrument, pointed at B, produced the tuple."""
    import re
    from collections import Counter
    df = Counter()
    toks = []
    for c in chunks:
        ws = set(re.findall(r"[a-z]{4,}", c.lower()))
        toks.append(ws)
        df.update(ws)
    n_c = max(len(chunks), 1)
    obs = []
    for ws in toks:
        scored = sorted(ws, key=lambda t: (df[t] / n_c, t))[:ns]
        if len(scored) == ns:
            obs.append(tuple(sorted(scored)))
    return obs


# DETERMINISTIC TOOL MEMO (sweep 86: 'the model should get FASTER as leCore learns').
# The receipt machinery already proves determinism (input_sha256 -> output_sha256); the
# memo cashes that proof in: identical input to a PURE tool returns the stored content
# blocks instead of recomputing. Pure set is deliberately conservative -- stateful tools
# (scene_*, corpus_* which mutate registries; zoo_* which call remote models; lecore_invoke
# whose faculty purity is unknowable generically) NEVER memo. Bounded LRU; entries over
# 2 MB skipped (a cache that eats the heap is a speedup wearing a leak).
# meta reports cache:'hit'|'miss' and compute_ms stays HONEST (near-zero on hits, payload
# billed the same -- the wire still carries the bytes). LECORE_MCP_MEMO=0 kills it.
_MEMO_PURE_TOOLS = ("series_analyze", "dataset_decompose", "math_eval", "chart_make",
                    "lecore_find", "lecore_describe", "lecore_map")
_MEMO_MAX = 128
_MEMO_ENTRY_CAP = 2 * 1024 * 1024


_PENDING_MEDIA = []          # per-call scratch filled by _stub_media, drained by _media_blocks
                             # (the handler is single-threaded per request by MCP's shape)


def _is_image(v):
    """An (H,W,3) or (H,W) float/int ndarray is an image by convention everywhere
    in this engine (rgb01 / grayscale)."""
    try:
        import numpy as _np
        return (isinstance(v, _np.ndarray) and v.ndim in (2, 3)
                and (v.ndim == 2 or v.shape[2] in (1, 3))
                and v.shape[0] >= 2 and v.shape[1] >= 2)
    except Exception:
        return False


def _stub_media(out, _depth=0):
    """Walk a tool result; pull image arrays OUT of the JSON (json.dumps(default=str)
    would serialize a render as megabytes of repr soup) and queue them as PNG image
    blocks. The stub left behind names the shape, so the text half stays honest about
    what shipped beside it. Depth-capped: a pathological nest is not walked forever."""
    if _depth > 4:
        return out
    if _is_image(out):
        import numpy as _np
        from holographic.rendering.holographic_render import png_bytes
        import base64 as _b64
        arr = _np.asarray(out, float)
        if arr.ndim == 2:
            arr = _np.repeat(arr[:, :, None], 3, axis=2)
        if arr.max() > 1.0 + 1e-6:                       # uint8-scaled input tolerated
            arr = arr / 255.0
        png = png_bytes(_np.clip(arr, 0.0, 1.0))
        _PENDING_MEDIA.append({"type": "image", "mimeType": "image/png",
                               "data": _b64.b64encode(png).decode("ascii")})
        return {"_media": "image/png", "shape": list(out.shape),
                "note": "pixels shipped as an MCP image content block"}
    if isinstance(out, dict):
        return {k: _stub_media(v, _depth + 1) for k, v in out.items()}
    if isinstance(out, (list, tuple)) and len(out) <= 16:
        return [_stub_media(v, _depth + 1) for v in out]
    return out


def _media_blocks(_out):
    """Drain the per-call media queue into MCP content blocks."""
    blocks, _PENDING_MEDIA[:] = list(_PENDING_MEDIA), []
    return blocks


class MCPServer:
    """The protocol frame around one Service. handle(dict) -> dict|None keeps the whole
    server testable in-process; serve_stdio() is just a line loop around it."""

    def __init__(self, token=None, mind=None, memory_root=None):
        self.service = Service(token=token, mind=mind)
        self._corpora = {}                                # handle -> list of chunks
        # THE EXTERNAL-MEMORY PARTITION (Moose's picture, taken literally): a directory
        # assigned as the model's memory, managed as an ordinary leCore data structure --
        # KnowledgeStore gives ids, hashes, dedupe, tags, ranked search, and file-rooted
        # persistence, so the partition outlives the server process and every engine
        # faculty (compression, tiering, audit, distribution) applies to it like to any
        # other store. Default under the working dir; the zoo passes one dir per tenant.
        import os
        self._memory_root = memory_root or os.environ.get("LECORE_MEMORY_ROOT",
                                                          "./lecore_memory")
        self._memory = None                               # built lazily; mind is lazy too
        self._corpora_load()                               # E7.2: bindings survive restarts

    def _corpora_path(self):
        import os
        root = getattr(self, "_memory_root", None) or "."
        return os.path.join(str(root), "corpora.json")

    def _memo_store_path(self):
        import os
        # .lecore, not .json (sweep 112, Moose's rule): the memo rides the same
        # holographic container as every other store -- measured 14.7x smaller than
        # canonical JSON on a 66-entry payload AND byte-deterministic (twin saves
        # sha-identical), so the one-file + identical-bytes pins carry over intact.
        return os.path.join(str(self._memory_root), "toolmemo", "store.lecore")

    def _memo_store_load(self):
        """ONE FILE, NOT A SHARD FARM (sweep 111, Moose's commit-dialog screenshot):
        the write-through memo used to persist one JSON per call -- 64 files in git's
        face, 512 at cap. The store is now a SINGLE canonical JSON {key: result} with
        an '_order' list for mtime-free eviction, written tmp+rename with sorted keys
        so identical entries produce identical bytes (the determinism twins apply to
        caches too). MIGRATION folds legacy shards in and deletes them -- the spam
        cleans itself up on first load."""
        import os, json
        if getattr(self, "_memo_store", None) is not None:
            return self._memo_store
        store = {"_order": []}
        p = self._memo_store_path()
        try:
            if os.path.exists(p):
                from holographic.io_and_interop.holographic_container import load_container
                got = load_container(open(p, "rb").read())
                for sec in got["sections"]:
                    if sec["kind"] == "lecore.toolmemo":
                        store = sec["meta"].get("memo") or {"_order": []}
                if "_order" not in store:
                    store["_order"] = [k for k in store if k != "_order"]
        except Exception:
            store = {"_order": []}                     # corrupt store = cold cache, never a crash
        d = os.path.dirname(p)
        try:
            # legacy sweep: per-call shards AND the short-lived sweep-111 store.json
            # both fold in and self-delete -- the migration migrates its predecessor.
            legacy = [n for n in os.listdir(d) if n.endswith(".json")]                 if os.path.isdir(d) else []
            for n in sorted(legacy, key=lambda x: (x != "store.json", x)):
                if n == "store.json":
                    try:
                        with open(os.path.join(d, n), "r") as fh:
                            old_st = json.load(fh)
                        for k, v in old_st.items():
                            if k != "_order":
                                store.setdefault(k, v)
                                if k not in store["_order"]:
                                    store["_order"].append(k)
                    except Exception:
                        pass
                    os.remove(os.path.join(d, n))
                    continue
                k = n[:-5]
                try:
                    with open(os.path.join(d, n), "r") as fh:
                        store.setdefault(k, json.load(fh))
                    if k not in store["_order"]:
                        store["_order"].append(k)
                except Exception:
                    pass
                os.remove(os.path.join(d, n))          # the shard farm retires itself
            if legacy:
                self._memo_store_save(store)
        except Exception:
            pass
        self._memo_store = store
        return store

    def _memo_store_save(self, store=None):
        import os, json
        store = store if store is not None else getattr(self, "_memo_store", None)
        if store is None or not self._memory_root:
            return
        p = self._memo_store_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        try:
            from holographic.io_and_interop.holographic_container import save_container
            blob = save_container([{"kind": "lecore.toolmemo", "id": "v1",
                                    "meta": {"memo": store}, "arrays": {}}],
                                  meta={"app": "lecore.toolmemo", "version": 1})
            with open(tmp, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, p)                         # a kill mid-write leaves a miss
        except Exception:
            pass

    def _corpora_save(self):
        """Checkpoint 20: corpora persist as a CONTAINER section (lecore.zoo.corpora), not
        JSON -- the blessed format for everything the zoo stores."""
        try:
            from holographic.io_and_interop.holographic_container import save_container
            blob = save_container([{"kind": "lecore.zoo.corpora", "id": "v1",
                                    "meta": {"corpora": self._corpora}, "arrays": {}}],
                                  meta={"app": "lecore.zoo", "version": 2})
            import os
            # A requested memory_root is a REQUEST (the autoboot-partition rule): create it
            # rather than letting best-effort silently drop every save until something else
            # happens to mkdir it -- measured cost: a fresh server missing all its chunks.
            os.makedirs(str(self._memory_root), exist_ok=True)
            with open(os.path.join(str(self._memory_root), "corpora.lecore"), "wb") as f:
                f.write(blob)
        except OSError:
            pass                                            # persistence is best-effort, binding still works

    def _corpora_load(self):
        import json, os
        try:
            cpath = os.path.join(str(self._memory_root), "corpora.lecore")
            if os.path.exists(cpath):
                from holographic.io_and_interop.holographic_container import load_container
                got = load_container(open(cpath, "rb").read())
                for sec in got["sections"]:
                    if sec["kind"] == "lecore.zoo.corpora":
                        for k, v in (sec["meta"].get("corpora") or {}).items():
                            self._corpora.setdefault(k, v)
            elif os.path.exists(self._corpora_path()):       # legacy JSON: read + migrate
                with open(self._corpora_path(), "r", encoding="utf-8") as f:
                    stored = json.load(f)
                for k, v in stored.items():
                    self._corpora.setdefault(k, v)
                self._corpora_save()
                os.rename(self._corpora_path(), self._corpora_path() + ".migrated")
        except (OSError, ValueError):
            pass

    def _mem(self):
        if self._memory is None:
            from holographic.caching_and_storage.holographic_knowledgestore import KnowledgeStore
            self._memory = KnowledgeStore(self._memory_root)
        return self._memory

    def _study(self, root=None, budget_lines=120, ladder=False, **kw):
        """Server-side study: bind the harvested material under a persistent handle.
        The handle is content-derived (sha256 of the chunk texts) so re-studying an
        unchanged tree lands on the SAME handle -- bake once, ask forever (the Quilez
        discipline applied to comprehension: expensive walk baked, cheap asks
        regenerated per question)."""
        import hashlib
        if not root:
            return {"error": "root is required"}
        st = self.service.mind.study(str(root), budget_lines=int(budget_lines),
                                     ladder=bool(ladder))
        chunks = []
        # rebuild the chunk list the same way study did -- the mind returns the ask
        # closure, not the corpus; harvest via a second cheap pass is avoided by
        # reading the closure's cells (the chunks live in ask.__closure__)
        for cell in (st["ask"].__closure__ or ()):
            v = cell.cell_contents
            if isinstance(v, list) and v and isinstance(v[0], dict) and "text" in v[0]:
                chunks = v
                break
        h = "study-" + hashlib.sha256(
            "\x00".join(c["text"] for c in chunks).encode()).hexdigest()[:16]
        if not hasattr(self, "_studies"):
            self._studies = {}
        self._studies[h] = chunks
        out = {"handle": h, "tree": st["tree"], "n_chunks": st["n_chunks"],
               "docs": st["docs"][:6], "truncation": st.get("truncation")}
        if st.get("code"):
            out["code"] = {"files": st["code"]["files"],
                           "skeleton": str(st["code"]["skeleton"])[:2000]}
        if st.get("ladder"):
            out["ladder"] = st["ladder"]["summary"]
        return out

    def _study_ask(self, handle=None, query=None, k=5, **kw):
        """Grounded ask against a studied handle: the same declared lexical verdict as
        mind.study's ask(), WITH citations. Refusal is honest: off-corpus questions
        come back answerable=false, never vibed."""
        import re as _re
        from math import log
        chunks = (getattr(self, "_studies", {}) or {}).get(str(handle))
        if not chunks:
            return {"error": "unknown study handle %r -- call study first" % str(handle)}
        qw = {w for w in _re.findall(r"[a-z]{4,}", str(query).lower())}
        if not qw:
            return {"answerable": False, "advice": "no content words in the question"}
        df, toks = {}, []
        for c in chunks:
            tw = set(_re.findall(r"[a-z]{4,}", c["text"].lower()))
            toks.append(tw)
            for w in qw & tw:
                df[w] = df.get(w, 0) + 1
        n = len(chunks)
        scored = sorted(((sum(log(1.0 + n / (1.0 + df.get(w, 0))) for w in (qw & tw)),
                          len(qw & tw), i) for i, tw in enumerate(toks)), reverse=True)
        top = scored[:int(k)]
        best_s, best_sh, _ = top[0]
        return {"answerable": bool(best_sh >= 2),
                "verdict": "lexical retrieval (idf-weighted shared content words)",
                "top_score": round(float(best_s), 3), "shared_words": int(best_sh),
                "chunks": [chunks[i]["text"][:400] for _s, sh, i in top[:3] if sh > 0],
                "citations": [chunks[i]["source"] for _s, sh, i in top[:3] if sh > 0]}

    def _wisdom_record(self, lesson=None, author=None, topic=None, **kw):
        r = self.service.mind.bequeath(str(lesson), author=str(author), topic=topic)
        self._memory_persist() if hasattr(self, "_memory_persist") else None
        return {"taught": r.get("taught"), "author": r.get("author"),
                "topic": r.get("topic")}

    def _wisdom_ask(self, query=None, author=None, k=8, **kw):
        return self.service.mind.wisdom(query=query, author=author, k=int(k))

    def _corpus_bind(self, texts=None, text=None, documents=None, docs=None, name=None):
        # ALIAS TOLERANCE (UX sweep): strangers send documents=/docs= before reading a schema
        # -- the first phrasing must work. name= is accepted and ignored (content-addressing
        # IS the name); refusing an extra courtesy field would be hostile.
        from holographic.caching_and_storage.holographic_knowledgestore import chunk_text
        chunks = []
        for t in (texts or documents or docs or []):
            chunks.append(str(t))
        if text:
            chunks.extend(chunk_text(str(text)))
        if not chunks:
            return {"error": "pass texts=[...] (aliases: documents=, docs=) or text='...'"}
        import hashlib
        h = "corpus:" + hashlib.sha256("\x00".join(chunks).encode()).hexdigest()[:12]
        self._corpora[h] = chunks                          # content-addressed: re-binding
        self._corpora_save()                               # E7.2: a zoo restart must not lose bindings
        return {"handle": h, "n_chunks": len(chunks)}      # the same corpus is idempotent

    # -- THE ANALYST DOORS (sweep 83): series analysis + fact checking over the wire ------

    def _series_analyze(self, series=None, data=None, x=None, tasks=None,
                        forecast_horizon=0, min_seg=16):
        """One door over the market/series stack: demux (hidden components + stride),
        regime detection (mean/std segments with boundaries), and an ENVELOPE forecast
        (predict the SIZE of the next move, calibrated) -- the three questions every
        market/telemetry ask decomposes into. tasks= subsets ('demux', 'regimes',
        'forecast'); default runs all three. Composes demux_series + detect_regimes +
        envelope_forecast; this door adds transport, never algorithms."""
        s = series if series is not None else (data if data is not None else x)
        if s is None:
            return {"error": "pass series= (aliases data=, x=): a list of numbers"}
        import numpy as _np
        arr = _np.asarray(s, float)
        if arr.ndim != 1 or len(arr) < 8:
            return {"error": "series must be 1-D with >= 8 points (got shape %s)"
                             % (arr.shape,)}
        want = set(tasks or ("demux", "regimes", "forecast"))
        mind = self.service.mind
        out = {"n": int(len(arr))}
        if "demux" in want:
            d = dict(mind.demux_series(arr))
            # the separated COMPONENTS are the point of demux, and json.dumps(default=str)
            # shipped them as numpy REPR SOUP a host cannot parse back (measured, sweep
            # 85). Data doors own their serialization: plain rounded lists, capped
            # honestly rather than truncated silently.
            obs = d.get("objects")
            if obs is not None:
                comps, total = [], 0
                for o in obs:
                    v = [round(float(t), 6) for t in _np.asarray(o, float).ravel()]
                    total += len(v)
                    comps.append(v)
                if total <= 20000:
                    d["objects"] = comps
                else:
                    d["objects"] = [c[:64] for c in comps]
                    d["objects_note"] = ("components truncated to 64 points each "
                                         "(%d total) -- call demux_series via "
                                         "lecore_invoke for full arrays" % total)
            if "corr" in d:
                d["corr"] = [[round(float(v), 4) for v in row]
                             for row in _np.asarray(d["corr"], float)]
            out["demux"] = d
        if "drift" in want:
            # the hrnn 'market analysis' recipe's own pattern, surfaced as a task:
            # split-half structure fingerprints + drift verdict ('entropy rate moved
            # 2.49 -> 2.99' / 'no structural change at tolerance') -- ONLINE change
            # detection beside regimes' retrospective segmentation
            half = len(arr) // 2
            fp1 = mind.structure_fingerprint(arr[:half])
            fp2 = mind.structure_fingerprint(arr[half:])
            out["drift"] = mind.structure_drift(fp1, fp2)
        if "regimes" in want:
            out["regimes"] = mind.detect_regimes(arr, min_seg=int(min_seg))
        if "formula" in want:
            # symbolic recovery: the LAW that generates the series (decompose_signal --
            # MDL-gated additive terms; the Formula stringifies for the wire)
            fml, rep = mind.decompose_signal(arr)
            out["formula"] = {"formula": str(fml), **{k: rep[k] for k in
                              ("resid_rms", "n_terms", "mdl_bits", "mode") if k in rep}}
        if "forecast" in want:
            out["forecast"] = mind.envelope_forecast(arr)
            if int(forecast_horizon) > 0:
                out["forecast"]["horizon_note"] = (
                    "envelope_forecast predicts next-move SIZE; for horizon-scoped "
                    "point forecasts use lecore_invoke('ladder_forecast_calibrated')")
        return out

    def _dataset_decompose(self, data=None, rows=None, series=None, max_terms=6):
        """UNLABELED data taken apart: a 1-D series goes to decompose_signal (the additive
        LAW that generates it, MDL-gated, with residual and bit-cost on record); a 2-D
        (n, channels) dataset goes to explore_series (scaffold axis discovery + per-
        channel decomposition + a structured/noise VERDICT). Formulas ship as strings;
        every number in the report is the measurement, not a vibe."""
        d = data if data is not None else (rows if rows is not None else series)
        if d is None:
            return {"error": "pass data= (aliases rows=, series=): a list (1-D) or "
                             "list-of-rows (2-D)"}
        import numpy as _np
        arr = _np.asarray(d, float)
        mind = self.service.mind
        if arr.ndim == 1:
            if len(arr) < 16:
                return {"error": "1-D decomposition needs >= 16 points (got %d)" % len(arr)}
            fml, rep = mind.decompose_signal(arr, max_terms=int(max_terms))
            return {"kind": "series", "formula": str(fml),
                    "report": {k: rep[k] for k in ("resid_rms", "n_terms", "mdl_bits",
                                                   "mode", "multiplicative") if k in rep}}
        if arr.ndim == 2:
            r = mind.explore_series(arr, max_terms=int(max_terms))
            out = {"kind": "dataset", "verdict": r.get("verdict"),
                   "n_channels": r.get("n_channels"),
                   "structured_channels": r.get("structured_channels"),
                   "scaffold": r.get("scaffold"), "scores": r.get("scores")}
            chans = r.get("channels")
            if isinstance(chans, (list, tuple)):
                out["channels"] = [str(c)[:200] for c in chans[:12]]
            return out
        return {"error": "data must be 1-D or 2-D (got %d-D)" % arr.ndim}

    def _fact_check(self, text=None, claim=None, corpus=None):
        """Claims CHECKED against arithmetic and (optionally) a bound corpus: every
        'expr == value' is COMPUTED via check_math; with corpus= (a corpus_bind handle)
        each sentence is gated against the corpus via the dispatch gate, so 'supported'
        means retrieved evidence cleared the same bar corpus_ask uses -- and claims with
        NO support come back named, never silently passed. Without corpus= only the
        arithmetic half runs and the result SAYS so."""
        text = text if text is not None else claim
        if not text:
            return {"error": "pass text= (alias claim=); optional corpus= handle from "
                             "corpus_bind"}
        mind = self.service.mind
        out = {"math": mind.check_math(str(text))}
        if corpus is None:
            out["note"] = ("arithmetic only -- bind sources with corpus_bind and pass "
                           "corpus= to also check claims for support")
            return out
        store = getattr(self, "_corpora", {})
        if corpus not in store:
            return {"error": "unknown corpus handle %r -- corpus_bind first" % corpus}
        import re as _re
        sents = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", str(text)) if len(s.strip()) > 12]
        checked = []
        for s in sents[:12]:                       # bounded: 12 sentences per call
            # MEASURED contract (sweep 83): the dispatch gate returns answerable /
            # stage / margin, and certifies refusals ('cascade certified the corpus
            # cannot support') rather than guessing
            r = self._corpus_ask(corpus=corpus, question=s, gate="dispatch")
            checked.append({"claim": s[:160],
                            "supported": bool(r.get("answerable")),
                            "stage": r.get("stage"),
                            "margin": r.get("margin")})
        out["claims"] = checked
        out["unsupported"] = [c["claim"] for c in checked if not c["supported"]]
        return out

    # -- THE STUDIO DOORS (openzoo full-capability sweep): 3D, image, math, chart ---------
    # One design across all five: thin over existing mind faculties, media rides as MCP
    # image blocks, handles live for the server process (the corpus_bind contract --
    # scenes and images are live objects; the zoo proxy owns durability).

    def _scene_create(self, description=None, text=None, width=256, height=192,
                      quality="fast", name=None):
        """TEXT -> 3D SCENE -> RENDER, the Blender-MCP move on the substrate: build_scene
        parses plain words into a live SemanticScene (named objects, materials, lighting),
        renders it, and returns {handle, objects, image}. Deterministic end to end."""
        description = description or text
        if not description:
            return {"error": "pass description= (alias text=), e.g. 'a red metal sphere "
                             "and a small blue glass box on a sunny day'"}
        if not hasattr(self, "_scenes"):
            self._scenes = {}
        scene = self.service.mind.build_scene(str(description))
        import hashlib as _hl
        handle = str(name) if name else             "scene:" + _hl.sha256(str(description).encode()).hexdigest()[:10]
        self._scenes[handle] = scene
        img = scene.render(width=int(width), height=int(height), quality=str(quality))
        return {"handle": handle, "objects": scene.names(),
                "description": str(description), "image": img}

    def _scene_adjust(self, handle=None, instruction=None, scene=None, text=None,
                      width=256, height=192, quality="fast", render=True):
        """Adjust a live scene by TALKING to it ('make the sphere bigger', 'change the box
        to glass') and re-render. The conversational loop is the whole point: the model
        iterates toward what the user meant without touching a vertex."""
        handle = handle or scene
        instruction = instruction or text
        if not hasattr(self, "_scenes") or handle not in getattr(self, "_scenes", {}):
            return {"error": "unknown scene handle %r -- scene_create first (handles live "
                             "for this server process)" % handle}
        if not instruction:
            return {"error": "pass instruction= (alias text=)"}
        sc = self._scenes[handle]
        r = sc.adjust(str(instruction))
        out = {"handle": handle, "adjusted": str(instruction),
               "interpretation": str(r)[:400], "objects": sc.names()}
        if render:
            out["image"] = sc.render(width=int(width), height=int(height),
                                     quality=str(quality))
        return out

    def _scene_export(self, handle=None, scene=None, format="stl", spacing=2.7):
        """Geometry OUT: realize the scene's objects to meshes and write the open exchange
        format a modeler reads (ASCII STL today -- the mesh_to_stl door; the text is the
        payload, the caller writes the file). KEPT NEG: export realizes CURRENT state; an
        adjust after export is not in the file you already took."""
        handle = handle or scene
        if not hasattr(self, "_scenes") or handle not in getattr(self, "_scenes", {}):
            return {"error": "unknown scene handle %r -- scene_create first" % handle}
        fmt = str(format).lower()
        if fmt != "stl":
            return {"error": "format %r not offered here; 'stl' is the open exchange "
                             "door (mesh_to_stl). DXF is 2-D (polylines_to_dxf via "
                             "lecore_invoke); glTF/OBJ are IMPORT formats today." % fmt}
        sc = self._scenes[handle]
        mind = self.service.mind
        parts, n_v = [], 0
        for ob in sc.realize(spacing=float(spacing)):
            mesh = getattr(ob, "mesh", None) or (ob.get("mesh") if isinstance(ob, dict)
                                                 else None)
            sdf = getattr(ob, "sdf", None) or (ob.get("sdf") if isinstance(ob, dict)
                                               else None)
            if mesh is None and sdf is not None:
                mesh = mind.sdf_to_mesh(sdf) if hasattr(mind, "sdf_to_mesh") else None
            if mesh is None:
                continue
            V = getattr(mesh, "vertices", None)
            F = getattr(mesh, "faces", None)
            if V is None or F is None:
                continue
            parts.append(mind.mesh_to_stl(V, F, name="lecore"))
            n_v += len(V)
        if not parts:
            return {"error": "the scene realized no exportable meshes -- primitive-only "
                             "scenes export via lecore_invoke on the sdf faculties"}
        return {"handle": handle, "format": "stl", "meshes": len(parts),
                "vertices": n_v, "stl": "\n".join(parts)}

    def _image_tool(self, op=None, image=None, image_b64=None, ref=None, ref_b64=None,
                    args=None, width=128, height=128):
        """The 2D toolkit over the wire: generate (pattern kinds via pattern_field, vector
        art via chart) and edit (sharpen / recolor / blend / downscale). Images arrive as
        base64 PNG (image_b64=) or as a prior result's pixels re-shipped; results return
        as MCP image blocks. Ops delegate to the cataloged faculties -- this door adds
        transport, never algorithms."""
        import base64 as _b64
        import numpy as _np
        from holographic.rendering.holographic_render import png_decode
        mind = self.service.mind
        def _dec(b):
            arr, _info = png_decode(_b64.b64decode(b))    # MEASURED: (array, info) tuple
            return _np.asarray(arr, float)
        try:
            img = _dec(image_b64) if image_b64 else None
            rf = _dec(ref_b64) if ref_b64 else None
        except Exception as e:
            return {"error": "could not decode PNG input: %s" % e}
        kw = dict(args or {})
        op = str(op or "").lower()
        if op == "pattern":
            # MEASURED, not recalled: pattern_field returns a 3-D FIELD FUNCTION over
            # points (the SDF costume) -- pixels are a sampled z=0 slice, scaled so the
            # default fbm shows structure at any raster size.
            f = mind.pattern_field(kw.pop("kind", "fbm"), **kw)
            span = float(kw.pop("span", 4.0))
            xs = _np.linspace(0.0, span, int(width))
            ys = _np.linspace(0.0, span, int(height))
            X, Y = _np.meshgrid(xs, ys)
            P = _np.stack([X.ravel(), Y.ravel(), _np.zeros(X.size)], axis=1)
            v = _np.asarray(f(P), float).reshape(int(height), int(width))
            v = (v - v.min()) / (v.max() - v.min() + 1e-12)
            return {"op": op, "image": v}
        if op == "sharpen" and img is not None:
            # MEASURED: sharpen_image is a single-channel deconvolution loop; RGB is
            # three honest passes, never a silent luma collapse that discards color.
            if img.ndim == 3:
                ch = [mind.sharpen_image(img[:, :, c], **kw) for c in range(img.shape[2])]
                return {"op": op, "image": _np.stack(ch, axis=2)}
            return {"op": op, "image": mind.sharpen_image(img, **kw)}
        if op == "recolor" and img is not None and rf is not None:
            return {"op": op, "image": mind.recolor_image(img, rf, **kw)}
        if op == "blend" and img is not None and rf is not None:
            frames = mind.blend_images(img, rf, steps=int(kw.pop("steps", 3)))
            mid = frames[len(frames) // 2] if isinstance(frames, (list, tuple)) else frames
            return {"op": op, "image": mid}
        return {"error": "op %r not recognized or missing inputs. Ops: pattern (kind=fbm|"
                         "checker|stripes|dots|gradient, span, width, height), "
                         "sharpen(image_b64), recolor(image_b64, ref_b64), "
                         "blend(image_b64, ref_b64, steps). The FULL 2D toolkit is one "
                         "lecore_find away ('2D image editing') via lecore_invoke." % op}

    def _math_eval(self, text=None, expression=None):
        """Arithmetic and calculus claims CHECKED, not vibed: check_math parses every
        'expr == value' claim and computes it (exact where exact applies); a bare
        expression is evaluated via do_math. Wrong claims come back named."""
        text = text if text is not None else expression
        if text is None:
            return {"error": "pass text= (alias expression=), e.g. '2*3+4 == 10' or "
                             "'integrate x**2 from 0 to 1'"}
        mind = self.service.mind
        s = str(text)
        if "==" in s:
            return mind.check_math(s)
        try:
            return {"input": s, "result": mind.do_math(s)}
        except Exception as e:
            return {"error": "%s: %s -- for symbolic work (solve/simplify/factor) use "
                             "lecore_find('symbolic') + lecore_invoke" % (type(e).__name__, e)}

    def _chart_make(self, kind=None, series=None, data=None, labels=None, title=None,
                    x=None, width=640, height=400):
        """Numbers -> a chart a human reads: deterministic SVG (line | bar | scatter),
        colorblind-safe palette, bars anchored at zero. The SVG text IS the payload.
        KEPT NEG rides through from the module: non-finite values are refused loudly."""
        series = series if series is not None else data
        if not kind or series is None:
            return {"error": "pass kind=line|bar|scatter and series=[...] (alias data=; "
                             "a list, a list of lists, or (x,y) pairs for scatter)"}
        try:
            svg = self.service.mind.chart_svg(kind, series, labels=labels, title=title,
                                              x=x, width=int(width), height=int(height))
        except ValueError as e:
            return {"error": str(e)}
        return {"kind": str(kind), "svg": svg, "bytes": len(svg)}

    def _corpus_delta(self, chunk_hashes=None, chunks=None, hashes=None):
        """CHUNK-LEVEL DELTA BIND (openzoo ergonomics sweep): the rsync move at the corpus seam.

        corpus_bind content-addresses at the CORPUS level, so one edited chunk re-ships
        megabytes -- exactly the workload an agent creates by re-binding a repo after one file
        changed. This tool splits binding into a probe and a fill:

        PROBE  -- corpus_delta(chunk_hashes=[sha256 hex, ...]): returns {"missing": [hashes the
                  server lacks], "known": n}. Nothing is uploaded; order is the corpus order.
        FILL   -- corpus_delta(chunk_hashes=[...], chunks={hash: text, ...}): ships ONLY the
                  missing texts. When every hash resolves, the corpus is assembled IN HASH-LIST
                  ORDER and lands under THE SAME handle function corpus_bind uses -- so a
                  delta-bound corpus and a whole-bound one are indistinguishable downstream
                  (reflex cache, corpus_ask, gate='dispatch': all untouched, invalidation still
                  rides on content addressing).

        The chunk store persists beside the corpora (E7.2: a restart must not turn every probe
        into a full re-upload). A fill whose texts do not hash to their claimed keys is refused
        PER CHUNK, loudly -- a silent mis-keyed chunk would corrupt every corpus that ever
        references that hash.
        KEPT NEG: the server cannot chunk for you here -- the CLIENT owns the chunking so its
        hashes are computed over exactly what it will send; ship pre-chunked texts."""
        import hashlib
        chunk_hashes = chunk_hashes if chunk_hashes is not None else hashes
        if not chunk_hashes:
            return {"error": "pass chunk_hashes=[sha256 hex, ...] (probe), optionally with "
                             "chunks={hash: text} (fill)"}
        if not hasattr(self, "_chunk_store"):
            self._chunk_store = {}
            self._chunks_load()
        rejected = []
        for h, t in (chunks or {}).items():
            t = str(t)
            real = hashlib.sha256(t.encode("utf-8")).hexdigest()
            if real != h:
                rejected.append({"hash": h, "actual": real})    # refuse loudly, per chunk
                continue
            self._chunk_store[h] = t
        missing = [h for h in chunk_hashes if h not in self._chunk_store]
        if missing or rejected:
            if chunks or rejected:
                self._chunks_save()
            out = {"missing": missing, "known": len(chunk_hashes) - len(missing)}
            if rejected:
                out["rejected"] = rejected
            return out
        ordered = [self._chunk_store[h] for h in chunk_hashes]
        # THE SAME handle function as _corpus_bind, character for character: identical corpus
        # => identical handle, whichever door it came through. That identity is what lets the
        # zoo proxy mix delta and whole binds freely.
        handle = "corpus:" + hashlib.sha256("\x00".join(ordered).encode()).hexdigest()[:12]
        self._corpora[handle] = ordered
        self._corpora_save()
        self._chunks_save()
        return {"handle": handle, "n_chunks": len(ordered), "uploaded": len(chunks or {}),
                "reused": len(chunk_hashes) - len(chunks or {}), "missing": []}

    def _chunks_save(self):
        """Chunk store persists as its own container beside corpora.lecore (same blessed
        format, separate file: the chunk store can grow large and corpora loads hot)."""
        try:
            from holographic.io_and_interop.holographic_container import save_container
            blob = save_container([{"kind": "lecore.zoo.chunks", "id": "v1",
                                    "meta": {"chunks": self._chunk_store}, "arrays": {}}],
                                  meta={"app": "lecore.zoo", "version": 2})
            import os
            os.makedirs(str(self._memory_root), exist_ok=True)   # same rule as corpora above
            with open(os.path.join(str(self._memory_root), "chunks.lecore"), "wb") as f:
                f.write(blob)
        except OSError:
            pass                                            # best-effort, same as corpora

    def _chunks_load(self):
        import os
        try:
            p = os.path.join(str(self._memory_root), "chunks.lecore")
            if os.path.exists(p):
                from holographic.io_and_interop.holographic_container import load_container
                got = load_container(open(p, "rb").read())
                for sec in got["sections"]:
                    if sec["kind"] == "lecore.zoo.chunks":
                        self._chunk_store.update(sec["meta"].get("chunks") or {})
        except Exception:
            pass                                            # a cold chunk store is a fact, not an error

    def _corpus_ask(self, handle=None, query=None, k=4, question=None, corpus=None, gate=None):
        # ALIAS TOLERANCE: question= for query=, corpus= for handle= -- and a missing arg
        # must produce advice, not a KeyError traceback in a tool result.
        query = query if query is not None else question
        handle = handle if handle is not None else corpus
        if handle is None or query is None:
            return {"error": "need handle= (alias corpus=) and query= (alias question=)"}
        if handle not in self._corpora:
            self._corpora_load()                           # E7.2: lazy reload after restart
        if handle not in self._corpora:
            return {"error": "unknown handle %r -- corpus_bind first (handles live for this "
                             "server process; the zoo proxy owns persistence)" % handle}
        chunks = self._corpora[handle]
        # DEFAULT-OFF DISPATCH GATE (openzoo ergonomics sweep): gate='dispatch' runs the FULL
        # adaptive cascade server-side (exact -> dense-margin -> BM25-refine -> honest abstain)
        # and returns the payment-gate verdict alongside the chunks -- the shape openzoo's
        # x-hrr-gate header has been asking for. gate=None is the byte-identical BM25 path
        # below (never-flip). The reflex cache is NOT consulted on the gated path on purpose:
        # a payment gate must reflect the corpus as bound NOW, and the cascade is 3ms-class.
        if gate == "dispatch":
            g = self.service.mind.corpus_gate(query, chunks, k=int(k))
            g["chunks"] = [{"index": i, "score": s, "chunk": chunks[i]}
                           for i, s in g.pop("ranked", [])]
            g["via"] = "dispatch"
            return g
        # E7.1 -- REFLEX BEFORE THE CORPUS: a per-handle displacement trace caches
        # (question -> ranked answer) under the FULL lever-7 gate (cleanup + calibrated null +
        # volatility). Handles are content-addressed (sha256 of the corpus), so a re-bound
        # corpus is a NEW handle and the cache never serves stale chunks -- content addressing
        # does the invalidation (lever 3 under lever 7, again). Provenance: via='reflex'.
        import numpy as np
        if not hasattr(self, "_reflex"):
            self._reflex = {}
        rx = self._reflex.get(handle)
        if rx is None:
            from holographic.agents_and_reasoning.holographic_lever7 import (
                DisplacementTrace, key_atom)
            rx = self._reflex[handle] = {"trace": DisplacementTrace(1024, seed=0),
                                         "payloads": {}, "key_atom": key_atom}
        toks = sorted(set(str(query).lower().split()))
        qkey = np.sum([rx["key_atom"]("q:" + t, 1024) for t in toks], axis=0) if toks             else rx["key_atom"]("q:", 1024)
        qkey = qkey / (np.linalg.norm(qkey) + 1e-12)
        hit = rx["trace"].read_gated(qkey)
        if hit["fired"]:
            pid = int(hit.get("atom", -1))
            if pid in rx["payloads"]:
                out = [dict(row) for row in rx["payloads"][pid]]
                for row in out:
                    row["via"] = "reflex"
                return out
        ranked = self.service.mind.bm25_rank(query, chunks, top=int(k))
        result = [{"index": int(i), "score": float(s), "chunk": chunks[int(i)]}
                  for i, s in ranked]
        pid = len(rx["payloads"])
        rx["payloads"][pid] = [dict(row) for row in result]
        rx["trace"].write(qkey, rx["key_atom"]("payload:%d:%s" % (pid, handle), 1024))
        # the trace's atom index for this payload is the codebook slot just created:
        rx["payloads"][len(rx["trace"]._atoms) - 1] = rx["payloads"].pop(pid)
        return result

    # -- THE HOSTED SUPERPOWERS (checkpoint 15): the ladder over the wire ------------------
    def _zoo_state_path(self):
        import os
        return os.path.join(str(self._memory_root), "zoo_state.json")

    def _user_service(self, user):
        """PER-END-USER MEMORY (cp53). The hosted state was per-DEPLOYMENT: every caller
        on one openzoo instance shared one ladder, so a preference one person taught was
        served to everybody, and their self-improvement curves were averaged into a single
        blur. Users of a public service have vastly different goals; the learning has to
        be theirs. Passing `user` routes to that person's OWN partition under
        <root>/users/<user> -- physical isolation, the same rule holographic_appkit gives
        apps, because a salt is a convention and a directory is a fact. No `user` keeps
        the shared space, so existing callers are unaffected."""
        import os
        cache = getattr(self, "_user_svcs", None)
        if cache is None:
            cache = self._user_svcs = {}
        key = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(user))[:64]
        if key not in cache:
            import lecore
            root = os.path.join(self._memory_root, "users", key)
            os.makedirs(root, exist_ok=True)
            # AUTOBOOT, NOT A BARE MIND. This built UnifiedMind() and called
            # learning_load by hand -- the pre-autoboot ritual, missing the parts
            # autoboot does: DOCTRINE (14 facts the reflex answers from), the POST
            # line, and the archive root. An MCP caller IS an agent by
            # construction, so the full boot is the right default here in a way it
            # is not for a human at a REPL.
            # llm=None deliberately: the AGENT ON THE OTHER END OF MCP IS THE
            # MODEL. Attaching a second one would put a model behind a model, and
            # there is no model directory on this box anyway.
            try:
                m = lecore.autoboot(partition=root, llm=None)
            except Exception:
                m = lecore.UnifiedMind()          # never fail a request over boot
                try:
                    m.learning_load(root)
                except Exception:
                    pass
            cache[key] = {"mind": m, "root": root}
        return cache[key]

    def _zoo_load(self):
        """Per-tenant zoo state survives restarts: taught answers replay into the reflex trace
        (write-path replay = bit-identical state, the lever-3 rule), chains reload into the
        log. Best-effort: a missing or corrupt file is an empty state, never a crash."""
        import json, os
        if getattr(self, "_zoo_loaded", False):
            return
        self._zoo_loaded = True
        try:
            self.service.mind.learning_load(self._memory_root)
        except Exception:
            pass
        try:                                             # legacy pass-5 tenants: read-only tolerance
            if os.path.exists(self._zoo_state_path()):
                st = json.load(open(self._zoo_state_path()))
                for q, ans in st.get("taught", []):
                    self._zoo_teach(q, ans, _persist=False)
                for ch in st.get("chains", []):
                    self.service.mind.chain_note(ch["goal_key"], [tuple(x) for x in ch["steps"]])
                os.rename(self._zoo_state_path(), self._zoo_state_path() + ".migrated")
        except Exception:
            pass

    def _zoo_save(self):
        """Checkpoint 20: the CONTAINER is the only writer. zoo_state.json is gone -- the
        learning container's experience + taught + chains sections carry everything the JSON
        replay used to (proved by the remember-suite before the JSON was removed, not after)."""
        try:
            self.service.mind.learning_save(self._memory_root)
        except Exception:
            pass

    def _zoo_faculty(self, fname, *fargs, persist=True, **fkw):
        """cp25: the zoo calls leCore THROUGH leCore -- any of the new self-use faculties
        run against the tenant mind, and mutating calls persist to the tenant partition
        (the same restart-survival contract as zoo_teach)."""
        self._zoo_load()
        m = self.service.mind
        try:
            out = getattr(m, fname)(*fargs, **fkw)
        except Exception as exc:
            return {"error": "%s: %s" % (fname, str(exc)[:200])}
        if persist:
            try:
                m.learning_save(self._memory_root)
            except Exception:
                pass
        return out

    def _zoo_ask(self, query, handle=None, taught_only=False, user=None):
        """The hosted ladder: T0 reflex + T1 corpus (via the existing reflex-fronted
        _corpus_ask when a handle is given) + T2 dispatch run SERVER-SIDE; a miss returns
        escalate=true WITH the best retrieved context -- the caller is the model rung."""
        self._zoo_load()
        m = self.service.mind
        if user:
            m = self._user_service(user)["mind"]
            if not getattr(m, "_zoo_ready", False):
                m.zoo_attach(lambda p: "")
                m._zoo_ready = True
        out = m.zoo_answer(str(query), kb_search=None,
                           dispatchers=m._zoo_dispatchers(), intern=None, main=None)
        # cp49, and it matters MORE on a public server than anywhere else: a hosted ladder
        # caches whatever the model rung answered and will serve it forever, confidently,
        # to everyone -- indistinguishable from a fact somebody deliberately taught. So
        # every hosted answer now DECLARES ITS PROVENANCE, and a caller who only wants
        # established facts can say so instead of receiving a cached guess.
        out.setdefault("provenance", "taught" if out.get("tier") in ("T1", "T2")
                       else "model-cached")
        if taught_only and out.get("tier") == "T0" and out["provenance"] != "taught":
            out = {"tier": "escalate", "escalate": True,
                   "why": "a cached model answer exists but taught_only was requested -- "
                          "answer it yourself and zoo_teach it to establish it"}
        elif out["tier"] in ("T0", "T1", "T2"):
            return out
        context = None
        if handle:
            rows = self._corpus_ask(handle=handle, query=query, k=3)
            if isinstance(rows, list):
                context = [r.get("chunk") for r in rows]
        return {"tier": "escalate", "escalate": True, "context": context,
                "why": "no free rung could serve; you are the model rung -- answer from the "
                       "context, then zoo_teach the result so this never costs tokens again"}

    def _zoo_teach(self, query, answer, _persist=True, user=None):
        self._zoo_load()
        if user:
            svc = self._user_service(user)
            m = svc["mind"]
            if not getattr(m, "_zoo_ready", False):
                m.zoo_attach(lambda p: "")
                m._zoo_ready = True
        else:
            m = self.service.mind
        lad = m.zoo["ladder"]
        qk = lad._qkey(str(query))
        lad._remember(qk, str(answer), str(query), provenance="taught")
        # the conversation is a corpus, hosted edition: question + answer join the tenant's
        # semantic space; a STEP-SHAPED answer (numbered / bulleted lines) is CHAIN-OF-THOUGHT
        # and additionally mines into the chain log -- the caller's reasoning becomes a
        # reusable skeleton candidate, exactly like a plan that ran here.
        try:
            m.semantic_ingest(str(query), source="teach_q")
            m.semantic_ingest(str(answer), source="teach_a")
            lines = [l.strip(" -*") for l in str(answer).splitlines() if l.strip()]
            steps = [l.split(")", 1)[-1].split(".", 1)[-1].strip() for l in lines
                     if l[:2].rstrip(".)").isdigit()]
            if len(steps) >= 2:
                sk = m.semantic_key(str(query))
                m.chain_note(sk["vec"][:64], [(st[:60], True) for st in steps])
        except Exception:
            pass
        stored = None
        if _persist:
            if not hasattr(self, "_zoo_taught"):
                self._zoo_taught = []
            self._zoo_taught.append([str(query), str(answer)])
            self._zoo_save()
            # THE STORAGE STACK, not a flat file (pass 4): taught answers ALSO land in the
            # tenant's KnowledgeStore -- content-hashed, DEDUPED, external-storage grade --
            # beside the JSON replay floor. Teaching the same answer twice stores one copy.
            try:
                note = self._mem().add_note("Q: %s\nA: %s" % (str(query), str(answer)),
                                            tags=("zoo", "taught"))
                stored = {"id": note.get("id") if isinstance(note, dict) else str(note),
                          "dedup": bool(note.get("duplicate_of")) if isinstance(note, dict)
                          else None}
            except Exception:
                stored = {"id": None, "dedup": None}
        return {"taught": True, "next_ask_tier": "T0", "knowledge_store": stored}

    def _zoo_do(self, request, plan=None):
        """Warm plans execute server-side at zero model cost; cold requests ask the caller to
        plan ONCE; either way the chain logs and the second encounter is free."""
        self._zoo_load()
        m = self.service.mind
        from holographic.agents_and_reasoning.holographic_lever7 import key_atom
        import numpy as np
        toks = sorted(set(str(request).lower().split()))[:8]
        gv = np.sum([key_atom("g:" + t, 64) for t in toks], axis=0)
        gv = gv / (np.linalg.norm(gv) + 1e-12)
        if plan is None:
            warm = m.plan_warm(gv)
            if warm is None:
                return {"need_plan": True,
                        "why": "no learned plan near this goal -- plan once (a list of "
                               "capability/synthesized-tool names) and pass plan=[...]"}
            plan = warm["steps"]
            via = "plan_warm"
        else:
            via = "caller_plan"
        report, done = [], []
        for st in plan:
            try:
                if st in (m.zoo.get("synth") or {}):
                    res = m.synth_call(st, None)
                else:
                    res = self.service.dispatch("POST", "/invoke", {"name": st, "args": {}})
                ok = res is not None
            except Exception as exc:
                res, ok = {"error": str(exc)[:120]}, False
            report.append({"step": st, "ok": ok})
            done.append((st, ok))
        m.chain_note(gv, done)
        self._zoo_save()
        return {"via": via, "model_calls_server_side": 0, "steps": list(plan),
                "report": report}

    def _zoo_query(self, dialect, query, rows=None, objects=None):
        self._zoo_load()
        m = self.service.mind
        if dialect == "graphql":
            return m.graphql(str(query), objects=objects)
        if rows is not None:
            self._zoo_rows = list(rows)
            cols = sorted({k for r in self._zoo_rows for k in r})
            db = m.db
            try:
                m.db_query("CREATE DATABASE zoo", db)
            except Exception:
                pass
            try:
                m.db_query("CREATE TABLE zoo.rows (%s)" % ", ".join(cols), db)
            except Exception:
                pass                                     # re-bind: table exists; inserts append
            for r in self._zoo_rows:
                vals = ", ".join(repr(r.get(c)) for c in cols)
                m.db_query("INSERT INTO zoo.rows (%s) VALUES (%s)" % (", ".join(cols), vals), db)
        if not getattr(self, "_zoo_rows", None):
            return {"error": "no rows bound -- pass rows=[...] once, then query freely"}
        return m.db_query(str(query), m.db)

    def _zoo_synthesize(self, name, chain):
        self._zoo_load()
        m = self.service.mind
        ex = {}
        for step in chain:
            hits = m.find_capability(str(step), k=1)
            meth = getattr(hits[0], "method", None) if hits else None
            fn = getattr(m, meth, None) if meth else None
            ex[str(step)] = (lambda x=None, _f=fn: _f(x)) if callable(fn) else (lambda x=None: x)
        r = m.synthesize_tool_certified(str(name), [str(c) for c in chain], ex)
        if r.get("ok") and isinstance(r.get("lean_certificate"), dict):
            r["lean_certificate"] = {"lean": r["lean_certificate"].get("lean", "")[:2000],
                                     "ok": r["lean_certificate"].get("ok")}
        return r

    # -- the three tools, each a thin delegation --
    def _find(self, query):
        hits = self.service.mind.find_capability(query)[:8]
        return [{"name": h.name, "does": (h.does or "")[:200],
                 "method": getattr(h, "method", None)} for h in hits]

    def _describe(self, name):
        hits = self.service.mind.find_capability(name)[:1]
        if not hits:
            return {"error": "no capability matching %r" % name}
        h = hits[0]
        return {"name": h.name, "does": h.does, "example": h.example,
                "method": getattr(h, "method", None), "aliases": list(h.aliases or ())}

    def _invoke(self, name, args):
        return self.service.dispatch("POST", "/invoke", {"name": name, "args": args or {}})

    def handle(self, req):
        rid = req.get("id")
        method = req.get("method", "")
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": _PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "leCore", "version": "0.2.11"},
                "instructions": _INSTRUCTIONS}}
        if method in ("notifications/initialized", "notifications/cancelled"):
            return None                                   # notifications get no response
        if method == "ping":
            return {"jsonrpc": "2.0", "id": rid, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"tools": _TOOLS}}
        if method == "tools/call":
            p = req.get("params", {})
            tool = p.get("name")
            a = p.get("arguments", {}) or {}
            import time as _t
            _t0 = _t.perf_counter()
            import os as _os
            _memo_key = None
            if tool in _MEMO_PURE_TOOLS and _os.environ.get("LECORE_MCP_MEMO", "1") != "0":
                import hashlib as _hl
                _canon0 = json.dumps({"tool": tool, "arguments": a}, sort_keys=True,
                                     separators=(",", ":"), default=str)
                _memo_key = (tool, _hl.sha256(_canon0.encode()).hexdigest())
                if not hasattr(self, "_tool_memo"):
                    from collections import OrderedDict
                    self._tool_memo = OrderedDict()
                hit = self._tool_memo.get(_memo_key)
                src = "hit"
                if hit is None and self._memory_root:
                    # LAZY DISK READ (sweep 87; sweep 111 moved shards -> ONE store
                    # file): the memo must survive the PROCESS, or 'faster as leCore
                    # grows' resets every session. Same content-addressed key, now a
                    # dict lookup in the consolidated store (which also migrates and
                    # deletes any legacy per-call shard files on first load).
                    _st = self._memo_store_load()
                    hit = _st.get("%s-%s" % _memo_key)
                    if hit is not None:
                        self._tool_memo[_memo_key] = hit           # promote to RAM
                        src = "hit-disk"
                if hit is not None:
                    self._tool_memo.move_to_end(_memo_key)
                    import copy as _copy
                    res = _copy.deepcopy(hit)
                    res["_meta"]["lecore.cost"]["compute_ms"] = round(
                        (_t.perf_counter() - _t0) * 1000.0, 3)
                    res["_meta"]["lecore.cost"]["cache"] = src
                    st = getattr(self, "_memo_stats", None) or {"hits": 0, "disk_hits": 0,
                                                                "misses": 0}
                    st["disk_hits" if src == "hit-disk" else "hits"] += 1
                    self._memo_stats = st
                    return {"jsonrpc": "2.0", "id": rid, "result": res}
            try:
                if tool == "receipt_verify":
                    # ALIAS TOLERANCE (UX sweep): the primer says 'send a receipt back and
                    # watch it confirm' -- so a pasted receipt dict must BE a valid argument
                    # (receipt={'output_sha256': ...}), alongside tool=/name= for the tool id
                    # and expected_output_sha256= for the raw hash. Missing pieces advise.
                    vt = a.get("name") or a.get("tool")
                    exp = a.get("expected_output_sha256") \
                        or (a.get("receipt") or {}).get("output_sha256")
                    if not vt or not exp:
                        out = {"error": "need name= (alias tool=) plus expected_output_sha256="
                                        " or receipt={...} from a prior call's _meta"}
                    else:
                        inner = self.handle({"jsonrpc": "2.0", "id": "receipt_verify",
                                             "method": "tools/call",
                                             "params": {"name": vt,
                                                        "arguments": a.get("arguments", {})}})
                        got = inner["result"]["_meta"]["lecore.receipt"]["output_sha256"]
                        out = {"match": got == exp, "actual_output_sha256": got}
                elif tool == "zoo_feedback":
                    out = self._zoo_faculty("answer_feedback", str(a.get("query")),
                                            ok=bool(a.get("ok")))
                    if a.get("correction"):
                        self._zoo_faculty("teach", str(a.get("query")),
                                          str(a.get("correction")))
                        out = {"feedback": out, "correction_taught": True}
                elif tool == "zoo_boot":
                    rep_ = self._zoo_faculty("boot", partition=self._memory_root,
                                             doctrine=True)
                    out = {"report": {"ok": rep_.get("ok"),
                                      "doctrine": rep_.get("doctrine"),
                                      "inventory": rep_.get("inventory")},
                           "os_prompt": self._zoo_faculty("os_prompt", rep_)}
                elif tool == "zoo_agent":
                    out = self._zoo_faculty("agent_loop", str(a.get("objective")),
                                            executors={},
                                            rounds=int(a.get("rounds") or 1),
                                            budget_steps=int(a.get("budget_steps") or 2),
                                            plan=a.get("plan"),
                                            checkpoint_root=self._memory_root)
                elif tool == "zoo_model3d":
                    out = self._zoo_faculty("model3d", spec=a.get("spec") or [],
                                            name=a.get("name"),
                                            size=int(a.get("size") or 160))
                elif tool == "zoo_research":
                    if a.get("texts"):
                        out = self._zoo_faculty("research_archive", str(a.get("topic")),
                                                list(a["texts"]),
                                                sources=a.get("sources"))
                    elif a.get("question"):
                        out = self._zoo_faculty("archive_query", str(a.get("topic")),
                                                str(a["question"]), persist=False)
                    else:
                        out = {"error": "give texts to archive or a question to query"}
                elif tool == "zoo_backtest":
                    out = self._zoo_faculty("market_backtest",
                                            [float(x) for x in (a.get("series") or [])],
                                            d_grid=tuple(int(x) for x in
                                                         (a.get("d_grid") or (3, 5, 8))),
                                            coverage=float(a.get("coverage") or 0.9))
                elif tool == "zoo_assimilate":
                    if a.get("doc_text"):
                        out = self._zoo_faculty("assimilate_docs", str(a.get("api")),
                                                str(a["doc_text"]))
                    elif a.get("task"):
                        out = self._zoo_faculty("use_assimilated", str(a.get("api")),
                                                str(a["task"]), persist=False)
                    else:
                        out = {"error": "give doc_text to assimilate or task to build"}
                elif tool == "zoo_ask" and hasattr(self.service.mind,
                                                   "ask_grounded") and \
                        not a.get("taught_only") and not a.get("user") and \
                        not a.get("session"):
                    # plain asks go through the grounded floor (cp68); taught_only
                    # keeps its STRICTER contract below -- refuse anything that is
                    # not deliberately taught, including grounded model output
                    _g = self.service.mind.ask_grounded(
                        str(a.get("query") or a.get("question") or ""))
                    out = {"answer": _g["answer"], "provenance": _g["provenance"],
                           "escalate": _g["escalate"],
                           "tier": ("escalate" if _g["escalate"]
                                    else _g.get("tier") or "T0")}
                elif tool == "zoo_ask":
                    out = self._zoo_ask(a.get("query"), a.get("handle"),
                                        bool(a.get("taught_only", False)),
                                        a.get("user"))
                elif tool == "zoo_panel":
                    mp = self.service.mind
                    if not getattr(mp, "_panel_realm", None):
                        mp.panel_seat(members=list((a.get("positions") or {}).keys()))
                    out = mp.panel_deliberate(a.get("question"), a.get("positions") or {})
                elif tool == "zoo_tools":
                    mt = self.service.mind
                    if a.get("op") == "status":
                        lad_t = mt.zoo["ladder"]
                        margs = getattr(lad_t, "_recent_margins", [])[-32:]
                        sat = mt.saturation_estimate(margs) if margs else \
                            {"state": "no-data", "note": "margins accrue as "
                             "questions are asked"}
                        out = {"saturation": sat,
                               "taught_rows": len(getattr(lad_t, "taught_log",
                                                          [])),
                               "services": sorted(
                                   mt.api_toolbox().services)}
                    elif a.get("op") == "find":
                        out = {"tools": mt.tool_find(str(a.get("task", "")))}
                    else:
                        box = mt.api_toolbox()
                        svc = str(a.get("service", ""))
                        if svc not in box.services:
                            out = {"ok": False,
                                   "error": "service %r is not registered by the "
                                            "operator; hosted callers cannot add "
                                            "urls (ssrf boundary)" % svc}
                        else:
                            out = box.call(svc, str(a.get("endpoint", "")),
                                           params=a.get("params") or {})
                elif tool == "zoo_void":
                    mv = self.service.mind
                    if a.get("user"):
                        mv = self._user_service(a["user"])["mind"]
                        if not getattr(mv, "_zoo_ready", False):
                            mv.zoo_attach(lambda p: ""); mv._zoo_ready = True
                    op = a.get("op")
                    items = list(a.get("items") or [])
                    if op == "propose":
                        out = mv.void_propose(items, radius=float(a.get("radius", 0.45)))
                    elif op == "walk":
                        out = mv.void_walk(items)
                    else:
                        out = mv.void_mix(str(a.get("a")), str(a.get("b")),
                                          corpus=items or None)
                        out.pop("blend", None)          # vectors do not belong on the wire
                elif tool == "zoo_teach":
                    out = self._zoo_teach(a.get("query"), a.get("answer"),
                                          user=a.get("user"))
                elif tool == "zoo_do":
                    out = self._zoo_do(a.get("request"), a.get("plan"))
                elif tool == "zoo_synthesize":
                    out = self._zoo_synthesize(a.get("name"), a.get("chain") or [])
                elif tool == "zoo_query":
                    out = self._zoo_query(a.get("dialect"), a.get("query"),
                                          a.get("rows"), a.get("objects"))
                elif tool == "zoo_report":
                    out = self.service.mind.zoo_report()
                    # the improvement is only real if it is OBSERVABLE: the memo's
                    # hit/miss ledger and disk footprint ride the report (sweep 87)
                    st = dict(getattr(self, "_memo_stats", {}) or
                              {"hits": 0, "disk_hits": 0, "misses": 0})
                    st["ram_entries"] = len(getattr(self, "_tool_memo", {}) or {})
                    if self._memory_root:
                        _st4 = self._memo_store_load()
                        st["disk_entries"] = max(len(_st4) - 1, 0)   # minus '_order' 
                    out = dict(out) if isinstance(out, dict) else {"report": out}
                    out["tool_memo"] = st
                elif tool == "void_explore":
                    if a["handle"] not in self._corpora:
                        out = {"error": "unknown handle -- corpus_bind first"}
                    else:
                        ns = int(a.get("slots", 3))
                        obs = _slot_observations(self._corpora[a["handle"]], ns)
                        out = self.service.mind.structured_voids(obs, min_count=2,
                                                                 max_candidates=24)
                        if hasattr(out, "get") and hasattr(out.get("candidates"), "tolist"):
                            out["candidates"] = out["candidates"].tolist()
                        # THE FEDERATED LEAP (the zoo-only move): with a second handle, mark
                        # which of A's licensed-but-absent combinations are INSTANTIATED in
                        # corpus B -- 'reality already contains it, elsewhere', the transfer
                        # warrant in discrete form, across tenants. Composition, not a new
                        # instrument: the same featurizer pointed at B, set membership, done.
                        hb = a.get("handle_b")
                        if hb and isinstance(out, dict) and out.get("candidates"):
                            if hb not in self._corpora:
                                out["transfer"] = {"error": "unknown handle_b"}
                            else:
                                obs_b = set(_slot_observations(self._corpora[hb], ns))
                                inst = [list(c) for c in map(tuple, out["candidates"])
                                        if tuple(c) in obs_b]
                                out["transfer"] = {"instantiated_in_b": inst,
                                                   "warrant": "transfer" if inst else None}
                elif tool == "memory_write":
                    e = self._mem().add(a["text"], kind="note", source="model",
                                        tags=tuple(a.get("tags", ())))
                    out = {"id": e["id"] if isinstance(e, dict) else str(e), "stored": True}
                elif tool == "memory_search":
                    hits = self._mem().search(self.service.mind, a["query"],
                                              top=int(a.get("top", 4)))
                    out = [{"id": h.get("id"), "text": h.get("text"),
                            "tags": h.get("tags", [])} for h in hits]
                elif tool == "lecore_map":
                    n = len(self.service.mind._capability_catalog().all())
                    out = {"total_capabilities": n, "families": _FAMILY_MAP,
                           "how": "pick a family, pass an ask_for phrase (or your own words) "
                                  "to lecore_find, then lecore_invoke the method it names"}
                elif tool == "corpus_bind":
                    out = self._corpus_bind(**a)               # handler owns alias tolerance
                elif tool == "corpus_ask":
                    out = self._corpus_ask(**a)                # handler owns alias tolerance
                elif tool == "corpus_delta":
                    out = self._corpus_delta(**a)              # handler owns alias tolerance
                elif tool == "study":
                    out = self._study(**a)
                elif tool == "study_ask":
                    out = self._study_ask(**a)
                elif tool == "wisdom_record":
                    out = self._wisdom_record(**a)
                elif tool == "wisdom_ask":
                    out = self._wisdom_ask(**a)
                elif tool == "series_analyze":
                    out = self._series_analyze(**a)
                elif tool == "fact_check":
                    out = self._fact_check(**a)
                elif tool == "dataset_decompose":
                    out = self._dataset_decompose(**a)
                elif tool == "scene_create":
                    out = self._scene_create(**a)
                elif tool == "scene_adjust":
                    out = self._scene_adjust(**a)
                elif tool == "scene_export":
                    out = self._scene_export(**a)
                elif tool == "image_tool":
                    out = self._image_tool(**a)
                elif tool == "math_eval":
                    out = self._math_eval(**a)
                elif tool == "chart_make":
                    out = self._chart_make(**a)
                elif tool == "lecore_find":
                    out = self._find(a["query"])
                elif tool == "lecore_describe":
                    out = self._describe(a["name"])
                elif tool == "lecore_invoke":
                    self._zoo_load()  # cp25: the raw faculty runner joins tenancy --
                    # idempotent since cp21, so this is free on the warm path
                    # ALIAS TOLERANCE (UX sweep): method= is what a stranger sends after
                    # lecore_find told them the method name; a miss advises instead of KeyError.
                    fac = a.get("name") or a.get("method") or a.get("faculty")
                    if not fac:
                        out = {"error": "need name= (aliases: method=, faculty=) -- the string "
                                        "lecore_find returns as 'method'"}
                    else:
                        # kwargs= joins args=/arguments= -- the payload aliases were
                        # asymmetric with the method aliases, and kwargs is the first
                        # word a Python-speaking stranger reaches for (measured on
                        # ourselves, sweep 83)
                        payload = a.get("args", a.get("arguments", a.get("kwargs", {})))
                        out = self._invoke(fac, payload)
                else:
                    return {"jsonrpc": "2.0", "id": rid,
                            "error": {"code": -32602, "message": "unknown tool %r" % tool}}
                out = _stub_media(out)                        # arrays out of the JSON,
                text = json.dumps(out, default=str)           # pixels into image blocks
                # THE METERING HOOK (measured, per call): compute ms + payload bytes in every
                # result, because the cost census showed compute and wire diverge by 400:1 on
                # some faculties (bind: 0.025 ms CPU, ~10 KB JSON) -- a flat per-call price
                # would be fiction. An x402 proxy bills these two numbers directly; the
                # engine is deterministic, so quoted costs REPRODUCE.
                meta = {"elapsed_ms": round((_t.perf_counter() - _t0) * 1e3, 3),
                        "payload_bytes": len(text)}
                # THE RECEIPT (proof-of-inference, the deterministic dividend): the engine's
                # outputs are functions of (tool, arguments) alone, so a sha256 pair is a
                # complete, re-verifiable claim about what was computed -- 'don't trust,
                # re-run'. An x402 proxy can bill against it, cache against it ('charge
                # once, serve the hash'), and ANY party can dispute it by re-invoking and
                # comparing 64 hex chars. No zero-knowledge machinery; determinism is the
                # proof system. (Wall-clock lives in cost, not the receipt -- time is the
                # one thing an honest re-run will not reproduce.)
                _canon = json.dumps({"tool": tool, "arguments": a}, sort_keys=True,
                                    separators=(",", ":"), default=str)
                receipt = {"input_sha256": hashlib.sha256(_canon.encode()).hexdigest(),
                           "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                           "deterministic": True}
                content = [{"type": "text", "text": text}] + _media_blocks(out)
                # payload accounting must cover the MEDIA too, or the wire-dominates
                # census under-bills every render by orders of magnitude
                meta["payload_bytes"] += sum(len(c.get("data", "")) for c in content[1:])
                meta["cache"] = "miss" if _memo_key is not None else "n/a"
                result = {"content": content, "isError": False,
                          "_meta": {"lecore.cost": meta, "lecore.receipt": receipt}}
                if _memo_key is not None and meta["payload_bytes"] <= _MEMO_ENTRY_CAP:
                    import copy as _copy
                    self._tool_memo[_memo_key] = _copy.deepcopy(result)
                    while len(self._tool_memo) > _MEMO_MAX:
                        self._tool_memo.popitem(last=False)
                    st = getattr(self, "_memo_stats", None) or {"hits": 0, "disk_hits": 0,
                                                                "misses": 0}
                    st["misses"] += 1
                    self._memo_stats = st
                    if self._memory_root:
                        # WRITE-THROUGH persistence (sweep 111: ONE store file, not a
                        # shard farm): update the consolidated dict, evict from the
                        # front of the mtime-free '_order' list past 512, rewrite
                        # tmp+rename with sorted keys -- deterministic bytes for
                        # identical entries, one file in the commit dialog forever.
                        try:
                            _st3 = self._memo_store_load()
                            k3 = "%s-%s" % _memo_key
                            if k3 not in _st3:
                                _st3["_order"].append(k3)
                            _st3[k3] = result
                            while len(_st3["_order"]) > 512:
                                _st3.pop(_st3["_order"].pop(0), None)
                            self._memo_store_save(_st3)
                        except Exception:
                            pass                       # persistence is best-effort;
                                                       # the answer already shipped
                return {"jsonrpc": "2.0", "id": rid, "result": result}
            except Exception as e:
                # MCP convention: tool-level failures ride in content with isError, so the
                # HOST's model sees the message and can adapt -- a JSON-RPC error would
                # hide it from the model entirely.
                # UNIFORM ENVELOPE (sweep 123; the sweep-107 finding): the text was bare
                # prose ("KeyError: 'text'") while math_eval's errors were structured JSON,
                # so a client parsing tool text as JSON crashed on the one path it most
                # needed to read. Every tool error is now the SAME JSON shape, and a
                # missing-argument KeyError names the tool's real parameters -- the
                # remedy in the message, not a guess.
                import json as _ej
                err = {"error": "%s: %s" % (type(e).__name__, e), "tool": str(tool),
                       "type": type(e).__name__}
                if isinstance(e, KeyError):
                    schema = next((t.get("inputSchema", {}) for t in _TOOLS
                                   if t.get("name") == tool), {})
                    props = list((schema.get("properties") or {}).keys())
                    if props:
                        err["expected_arguments"] = props
                        err["hint"] = "missing argument %s; this tool takes: %s" % (e, ", ".join(props))
                return {"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": _ej.dumps(err, default=str)}],
                    "isError": True}}
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": "method %r not found" % method}}

    def serve_stdio(self):
        """The loop an MCP host spawns: one JSON-RPC message per line on stdin, responses on
        stdout, everything else (logs) belongs on stderr by protocol."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except ValueError:
                continue
            resp = self.handle(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()


def _selftest():
    srv = MCPServer()
    init = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["serverInfo"]["name"] == "leCore"
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    tl = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [t["name"] for t in tl["result"]["tools"]]
    # THE PIN IS THE POINT: every tool added must arrive HERE in the same commit. This
    # list sat at 10 tools while the server grew to 20 -- red from checkpoint 15 to 27,
    # invisible because the local regression constructed the server without running
    # _selftest(). The cp27 CI simulation (running the REAL test) caught it.
    assert names == ["lecore_map", "lecore_find", "lecore_describe", "corpus_bind",
                     "corpus_ask", "corpus_delta",
                     "study", "study_ask", "wisdom_record", "wisdom_ask",
                     "series_analyze", "dataset_decompose", "fact_check",
                     "scene_create", "scene_adjust", "scene_export",
                     "image_tool", "math_eval", "chart_make",
                     "void_explore", "zoo_ask", "zoo_panel", "zoo_tools",
                     "zoo_void",
                     "zoo_teach", "zoo_do",
                     "zoo_synthesize", "zoo_query", "zoo_report", "receipt_verify",
                     "memory_write", "memory_search", "zoo_model3d", "zoo_research",
                     "zoo_backtest", "zoo_assimilate", "zoo_feedback", "zoo_boot",
                     "zoo_agent", "lecore_invoke"]
    # PROVENANCE + TAUGHT_ONLY PINS (cp49): on a PUBLIC server a cached model answer that
    # looks like an established fact is the worst failure mode there is -- one caller's
    # guess becomes everyone's permanent truth. These three asserts are the guard.
    srv.handle({"jsonrpc": "2.0", "id": 71, "method": "tools/call", "params": {
        "name": "zoo_teach", "arguments": {"query": "pin provenance q",
                                           "answer": "an established answer"}}})
    _pa = json.loads(srv.handle({"jsonrpc": "2.0", "id": 72, "method": "tools/call",
        "params": {"name": "zoo_ask", "arguments": {"query": "pin provenance q"}}}
        )["result"]["content"][0]["text"])
    assert _pa.get("provenance") == "taught", "a deliberately taught fact says taught"
    _lad = srv.service.mind.zoo["ladder"]
    _lad._remember(_lad._qkey("pin cached q"), "a guess", "pin cached q")
    _pc = json.loads(srv.handle({"jsonrpc": "2.0", "id": 73, "method": "tools/call",
        "params": {"name": "zoo_ask", "arguments": {"query": "pin cached q",
                                                    "taught_only": True}}}
        )["result"]["content"][0]["text"])
    assert _pc.get("tier") == "escalate", \
        "taught_only must refuse a cached model answer rather than serve a guess as fact"
    _pp = json.loads(srv.handle({"jsonrpc": "2.0", "id": 74, "method": "tools/call",
        "params": {"name": "zoo_panel", "arguments": {"question": "pin?",
                   "positions": {"a": "yes", "b": "yes"}}}}
        )["result"]["content"][0]["text"])
    assert _pp.get("silent"), "zoo_panel: consensus is silent (the contrast law)"
    # ZOO_VOID PIN (cp57): the hosted leap carries its receipts.
    _zv = json.loads(srv.handle({"jsonrpc": "2.0", "id": 91, "method": "tools/call",
        "params": {"name": "zoo_void", "arguments": {"op": "mix",
            "a": "alpha beta code", "b": "gamma delta trace",
            "items": ["alpha beta code", "gamma delta trace", "beta gamma bridge",
                      "delta epsilon store"]}}})["result"]["content"][0]["text"])
    assert _zv.get("provenance") == "conjecture" and "structure" in _zv, \
        "a hosted mix is a conjecture with an evidence block, never a bare answer"
    # PER-USER PINS (cp53): two people on one openzoo instance must not share a memory.
    srv.handle({"jsonrpc": "2.0", "id": 81, "method": "tools/call", "params": {
        "name": "zoo_teach", "arguments": {"query": "pin per user q",
                                           "answer": "ana's answer", "user": "pin_ana"}}})
    _ub = json.loads(srv.handle({"jsonrpc": "2.0", "id": 82, "method": "tools/call",
        "params": {"name": "zoo_ask", "arguments": {"query": "pin per user q",
                                                    "user": "pin_bo"}}}
        )["result"]["content"][0]["text"])
    assert "ana" not in str(_ub.get("answer") or ""), \
        "one end user's memory must never surface for another"
    _ua = json.loads(srv.handle({"jsonrpc": "2.0", "id": 83, "method": "tools/call",
        "params": {"name": "zoo_ask", "arguments": {"query": "pin per user q",
                                                    "user": "pin_ana"}}}
        )["result"]["content"][0]["text"])
    assert _ua.get("tier") == "T0" and _ua.get("provenance") == "taught", \
        "the user who taught it must be served it, marked taught"
    # RECEIPT PINS: every call carries one; re-running matches it; a tampered hash does not
    rc = srv.handle({"jsonrpc": "2.0", "id": 40, "method": "tools/call",
                     "params": {"name": "lecore_describe", "arguments": {"name": "bind"}}})
    rr = rc["result"]["_meta"]["lecore.receipt"]
    assert set(rr) == {"input_sha256", "output_sha256", "deterministic"}
    ok = srv.handle({"jsonrpc": "2.0", "id": 41, "method": "tools/call",
                     "params": {"name": "receipt_verify",
                                "arguments": {"name": "lecore_describe",
                                              "arguments": {"name": "bind"},
                                              "expected_output_sha256": rr["output_sha256"]}}})
    assert json.loads(ok["result"]["content"][0]["text"])["match"] is True
    bad = srv.handle({"jsonrpc": "2.0", "id": 42, "method": "tools/call",
                      "params": {"name": "receipt_verify",
                                 "arguments": {"name": "lecore_describe",
                                               "arguments": {"name": "bind"},
                                               "expected_output_sha256": "0" * 64}}})
    assert json.loads(bad["result"]["content"][0]["text"])["match"] is False
    # FEDERATED-LEAP PIN: A's grammar licenses a combination A lacks; B contains it; the
    # two-handle call must flag it instantiated_in_b with the transfer warrant.
    rows = [(x, y, z) for x in ("acid", "base") for y in ("iron", "zinc")
            for z in ("salt", "fume")]
    heldt = ("acid", "zinc", "fume")
    ca_texts = [" ".join(r) for r in rows if r != heldt] * 2
    ca_texts += ["acid iron salt"] * 6 + ["base zinc fume"] * 6
    ba = srv.handle({"jsonrpc": "2.0", "id": 43, "method": "tools/call",
                     "params": {"name": "corpus_bind", "arguments": {"texts": ca_texts}}})
    ha = json.loads(ba["result"]["content"][0]["text"])["handle"]
    bb = srv.handle({"jsonrpc": "2.0", "id": 44, "method": "tools/call",
                     "params": {"name": "corpus_bind",
                                "arguments": {"texts": ["acid zinc fume", "base iron salt"]}}})
    hb2 = json.loads(bb["result"]["content"][0]["text"])["handle"]
    fv = srv.handle({"jsonrpc": "2.0", "id": 45, "method": "tools/call",
                     "params": {"name": "void_explore",
                                "arguments": {"handle": ha, "handle_b": hb2}}})
    ft = json.loads(fv["result"]["content"][0]["text"])
    assert ft.get("warrant") == "grammar", ft.get("gate")
    inst = ft.get("transfer", {}).get("instantiated_in_b", [])
    assert sorted(heldt) in [sorted(c) for c in inst], (ft.get("candidates"), inst)
    # VOID PIN, both truths: a thin corpus REFUSES with the epicycle message (the gate's
    # honesty is the feature); the tool round-trips over the same handles corpus_ask uses
    vb = srv.handle({"jsonrpc": "2.0", "id": 30, "method": "tools/call",
                     "params": {"name": "corpus_bind", "arguments": {"texts": [
                         "alpha beta gamma story", "alpha beta delta story",
                         "epsilon zeta gamma tale"]}}})
    vh = json.loads(vb["result"]["content"][0]["text"])["handle"]
    vx = srv.handle({"jsonrpc": "2.0", "id": 31, "method": "tools/call",
                     "params": {"name": "void_explore", "arguments": {"handle": vh}}})
    vt = json.loads(vx["result"]["content"][0]["text"])
    assert "gate" in vt or "error" in vt or "candidates" in vt
    if "why" in vt:
        assert "shuffle" in vt["why"] or "vouch" in vt["why"]
    # THE PARTITION PIN: write to external memory, search it back, then prove the partition
    # OUTLIVES the server -- a second MCPServer over the same root finds the same memory
    import tempfile
    mroot = tempfile.mkdtemp()
    srv_m = MCPServer(memory_root=mroot)
    w = srv_m.handle({"jsonrpc": "2.0", "id": 20, "method": "tools/call",
                      "params": {"name": "memory_write",
                                 "arguments": {"text": "the zoo gate code is 4471",
                                               "tags": ["ops"]}}})
    assert not w["result"]["isError"]
    s = srv_m.handle({"jsonrpc": "2.0", "id": 21, "method": "tools/call",
                      "params": {"name": "memory_search", "arguments": {"query": "gate code"}}})
    assert "4471" in s["result"]["content"][0]["text"]
    srv_m2 = MCPServer(memory_root=mroot)                 # a fresh server, same partition
    s2 = srv_m2.handle({"jsonrpc": "2.0", "id": 22, "method": "tools/call",
                        "params": {"name": "memory_search", "arguments": {"query": "gate code"}}})
    assert "4471" in s2["result"]["content"][0]["text"], "the partition must outlive the process"
    assert "RULE ZERO" in init["result"]["instructions"]
    # THE UN-ROTTABLE MAP PIN: every ask_for phrase must resolve in the LIVE catalog -- if a
    # family's phrasing stops finding anything, the map is lying and this fails the build
    mp = srv.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                     "params": {"name": "lecore_map", "arguments": {}}})
    families = json.loads(mp["result"]["content"][0]["text"])["families"]
    mind = srv.service.mind
    for fam, spec in families.items():
        for phrase in spec["ask_for"]:
            assert mind.find_capability(phrase), "map phrase resolves nothing: %s / %r" % (fam, phrase)
    # the zoo sentence, end to end: bind three docs, ask, get the right chunk first
    cb = srv.handle({"jsonrpc": "2.0", "id": 10, "method": "tools/call",
                     "params": {"name": "corpus_bind", "arguments": {"texts": [
                         "holographic reduced representations bind roles to fillers",
                         "the quick brown fox jumps over the lazy dog",
                         "bm25 ranks documents by term frequency and rarity"]}}})
    hdl = json.loads(cb["result"]["content"][0]["text"])["handle"]
    ca = srv.handle({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
                     "params": {"name": "corpus_ask",
                                "arguments": {"handle": hdl, "query": "how does bm25 rank"}}})
    top = json.loads(ca["result"]["content"][0]["text"])[0]
    assert top["index"] == 2 and top["score"] > 0
    cm = ca["result"]["_meta"]["lecore.cost"]
    assert cm["elapsed_ms"] >= 0 and cm["payload_bytes"] > 0    # the metering hook rides every call
    # unknown handle: a clean in-band error, not a crash
    bad_h = srv.handle({"jsonrpc": "2.0", "id": 12, "method": "tools/call",
                        "params": {"name": "corpus_ask",
                                   "arguments": {"handle": "corpus:nope", "query": "x"}}})
    assert "unknown handle" in bad_h["result"]["content"][0]["text"]
    fc = srv.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                     "params": {"name": "lecore_find", "arguments": {"query": "bind two vectors"}}})
    assert not fc["result"]["isError"] and "bind" in fc["result"]["content"][0]["text"].lower()
    iv = srv.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                     "params": {"name": "lecore_invoke",
                                "arguments": {"name": "find_capability",
                                              "args": {"query": "compress"}}}})
    assert not iv["result"]["isError"]
    bad = srv.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                      "params": {"name": "lecore_invoke",
                                 "arguments": {"name": "_private_thing", "args": {}}}})
    txt = bad["result"]["content"][0]["text"]
    assert bad["result"]["isError"] or "refus" in txt.lower() or "error" in txt.lower(), txt
    nf = srv.handle({"jsonrpc": "2.0", "id": 6, "method": "no/such"})
    assert nf["error"]["code"] == -32601
    print("OK: holographic_mcp self-test passed (initialize; curated tool trio; find/call "
          "round-trip; private faculty refused through the inherited gate; unknown method "
          "-32601; notifications silent)")


def main():
    """Console entry point (sweep 115): `lecore-mcp` after `pip install leos-core` -- the stdio
    MCP server any harness can point at. `lecore-mcp --selftest` runs the wire selftest."""
    if "--selftest" in sys.argv:
        _selftest()
    else:
        MCPServer().serve_stdio()


if __name__ == "__main__":
    main()

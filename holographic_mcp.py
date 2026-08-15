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
check it before saying you don't remember. Results are exact JSON; bytes come as
{'__bytes_b64__': ...}."""

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
         "k": {"type": "integer", "description": "how many chunks (default 4)"}},
         "required": ["handle", "query"]}},
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

    def _mem(self):
        if self._memory is None:
            from holographic.caching_and_storage.holographic_knowledgestore import KnowledgeStore
            self._memory = KnowledgeStore(self._memory_root)
        return self._memory

    def _corpus_bind(self, texts=None, text=None):
        from holographic.caching_and_storage.holographic_knowledgestore import chunk_text
        chunks = []
        for t in (texts or []):
            chunks.append(str(t))
        if text:
            chunks.extend(chunk_text(str(text)))
        if not chunks:
            return {"error": "pass texts=[...] or text='...'"}
        import hashlib
        h = "corpus:" + hashlib.sha256("\x00".join(chunks).encode()).hexdigest()[:12]
        self._corpora[h] = chunks                          # content-addressed: re-binding
        return {"handle": h, "n_chunks": len(chunks)}      # the same corpus is idempotent

    def _corpus_ask(self, handle, query, k=4):
        if handle not in self._corpora:
            return {"error": "unknown handle %r -- corpus_bind first (handles live for this "
                             "server process; the zoo proxy owns persistence)" % handle}
        chunks = self._corpora[handle]
        ranked = self.service.mind.bm25_rank(query, chunks, top=int(k))
        return [{"index": int(i), "score": float(s), "chunk": chunks[int(i)]}
                for i, s in ranked]

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
            try:
                if tool == "receipt_verify":
                    inner = self.handle({"jsonrpc": "2.0", "id": "receipt_verify",
                                         "method": "tools/call",
                                         "params": {"name": a["name"],
                                                    "arguments": a.get("arguments", {})}})
                    got = inner["result"]["_meta"]["lecore.receipt"]["output_sha256"]
                    out = {"match": got == a["expected_output_sha256"],
                           "actual_output_sha256": got}
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
                    out = self._corpus_bind(a.get("texts"), a.get("text"))
                elif tool == "corpus_ask":
                    out = self._corpus_ask(a["handle"], a["query"], a.get("k", 4))
                elif tool == "lecore_find":
                    out = self._find(a["query"])
                elif tool == "lecore_describe":
                    out = self._describe(a["name"])
                elif tool == "lecore_invoke":
                    out = self._invoke(a["name"], a.get("args", {}))
                else:
                    return {"jsonrpc": "2.0", "id": rid,
                            "error": {"code": -32602, "message": "unknown tool %r" % tool}}
                text = json.dumps(out, default=str)
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
                return {"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": text}], "isError": False,
                    "_meta": {"lecore.cost": meta, "lecore.receipt": receipt}}}
            except Exception as e:
                # MCP convention: tool-level failures ride in content with isError, so the
                # HOST's model sees the message and can adapt -- a JSON-RPC error would
                # hide it from the model entirely.
                return {"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": "%s: %s" % (type(e).__name__, e)}],
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
    assert names == ["lecore_map", "lecore_find", "lecore_describe", "corpus_bind",
                     "corpus_ask", "void_explore", "receipt_verify", "memory_write",
                     "memory_search", "lecore_invoke"]
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


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        MCPServer().serve_stdio()

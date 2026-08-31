"""The MCP server is the openzoo front door -- it gets a CI hook like every other organ.
(Last-chance wiring sweep: holographic_mcp.py lives at top level, outside the buried-selftest
audit's holographic/** scope, so without this file the zoo's entire mount surface had zero CI
coverage. A front door nobody tests is a gap wearing a doorknob.)"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mcp_selftest_in_process():
    from holographic_mcp import _selftest
    _selftest()          # initialize, tool list exact, corpus + memory + void round trips,
                         # partition persistence, private refusal, -32601 -- all pinned inside


def test_delegate_and_ux_sweep_79(tmp_path):
    """Sweep-79 pins -- the boss verb and the two measured UX fixes:
    (1) delegate: scripted agent (one tool call -> DONE) completes with agent='explicit',
        a REPRODUCIBLE receipt (wall clock lives outside it), and the gate still refuses
        gibberish BEFORE the model is consulted;
    (2) the end-result contract: an agent that says 'DONE:' with no answer comes back
        done=False, why= naming the contract -- tool chatter is not a result;
    (3) delegate with no agent anywhere raises naming ALL THREE doors (llm=, attach_llm,
        remote);
    (4) file_replace near-miss: a whitespace-perturbed old_str fails WITH a located
        message (line number + first differing char), and a nonsense old_str says
        'nothing similar' instead of pointing at noise."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    calls = {"n": 0}
    def agent(prompt):
        calls["n"] += 1
        return '{"tool": "mesh_smooth", "args": {}}' if calls["n"] == 1 else "DONE: mesh smoothed"
    r = m.delegate("smooth a bumpy mesh", llm=agent)
    assert r["done"] and r["answer"] == "mesh smoothed" and r["agent"] == "explicit"     # (1)
    calls["n"] = 0
    assert m.delegate("smooth a bumpy mesh", llm=agent)["receipt"] == r["receipt"]
    consulted = {"n": 0}
    def spy(prompt):
        consulted["n"] += 1
        return "DONE: x"
    g = m.delegate("purple monkey dishwasher", llm=spy)
    assert g["refused"] and consulted["n"] == 0, "the gate must fire below the model"
    r2 = m.delegate("smooth a bumpy mesh", llm=lambda _: "DONE:")
    assert not r2["done"] and "end-result contract" in r2["why"]                         # (2)
    m3 = lecore.UnifiedMind(dim=64, seed=0)
    m3._llm = None
    import os
    bak = {k: os.environ.pop(k) for k in list(os.environ)
           if "LECORE" in k or "OPENAI" in k or "OPENZOO" in k}
    try:
        try:
            m3.delegate("anything")
            raise AssertionError("no-agent delegate must raise")
        except RuntimeError as e:
            assert all(w in str(e) for w in ("llm=", "attach_llm", "remote"))            # (3)
    finally:
        os.environ.update(bak)
    m.set_file_root(str(tmp_path))
    p = tmp_path / "sample.py"
    p.write_text("def alpha(a, b):\n    return a + b\n")
    try:
        m.file_replace("sample.py", "def alpha(a,  b):", "X", count=1)
        raise AssertionError("perturbed old_str must miss")
    except Exception as e:
        msg = str(e)
        assert "closest match at line 1" in msg and "first difference" in msg            # (4)
    try:
        m.file_replace("sample.py", "zebra quantum flapjack", "X", count=1)
        raise AssertionError("nonsense must miss")
    except Exception as e:
        assert "nothing similar" in str(e)


def test_release_audit_sweep_123(tmp_path):
    """Sweep-123 pin -- the release audit's deliverables, end to end: (1) the dark doors
    are real mind faculties (reasoning kit, table history + user_table, determinism,
    BRDF, node client); (2) table history works on BOTH table shapes (the bug the wiring
    exposed); (3) the determinism twins report says byte-identical; (4) digits at the
    reflex belt are exact-match (the sweep-109 residual is gone); (5) a bad MCP argument
    comes back as a STRUCTURED error naming the tool's real parameters."""
    import json, os, sys, subprocess, time
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    lo, hi = m.conformal_interval(np.abs(np.random.default_rng(0).normal(size=200)), 10.0, alpha=0.1)
    assert lo < 10.0 < hi
    assert m.epistemic_map(1, 5, 0.3) != m.epistemic_map(4, 4, 0.01)
    u = m.user_table("t", ["id", "v"], dim=256, seed=0)
    u.insert({"id": 1, "v": 10})
    h = m.table_history(u)
    v1 = m.table_commit(h, u, "first")
    u.insert({"id": 2, "v": 20})
    v2 = m.table_commit(h, u, "second")
    assert m.table_diff(h, v1, v2, pk_col="id")["added"], "history must see the insert"
    snap = m.make_table([{"id": 1, "v": 10}], roles=["id", "v"], dim=256, seed=0)
    assert m.table_commit(m.table_history(snap), snap, "snap") == 0, "snapshot tables too"
    assert list(m.deterministic_topk([0.5, 0.9, 0.9, 0.1], 2)) == [1, 2], "the fixed tie rule"
    assert m.hash_unit("a", 1) == m.hash_unit("a", 1)
    assert set(m.brdf_terms(0.9, 0.8, 0.7, 0.4, base_color=[0.8, 0.2, 0.1], metallic=0.5)) == {"D", "G", "F0", "F"}
    assert m.determinism_report(n_facts=40)["byte_identical"], "STOP THE RELEASE"
    m3 = lecore.UnifiedMind(dim=256, seed=0)
    for s in range(3):
        m3._session = "user-%d" % s
        for i in range(2):
            m3.teach("secret %d of session %d" % (i, s), "payload-%d-%d" % (s, i))
    m3._session = "user-1"
    for s in (0, 2):
        a = str(m3.ask("secret 0 of session %d" % s).get("answer") or "")
        assert not a.startswith("payload-"), "digit aliasing served a wrong row: %r" % a
    assert m3.ask("secret 0 of session 1").get("answer") == "payload-1-0"
    srv = subprocess.Popen([sys.executable, "holographic_mcp.py"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           env=dict(os.environ, PYTHONHASHSEED="0"), text=True, bufsize=1)
    try:
        srv.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                                    "params": {}}) + chr(10))
        srv.stdin.flush()
        srv.stdout.readline()
        srv.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                    "params": {"name": "memory_write",
                                               "arguments": {"content": "wrong name"}}}) + chr(10))
        srv.stdin.flush()
        while True:
            line = srv.stdout.readline()
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("id") == 1:
                break
        body = json.loads(r["result"]["content"][0]["text"])
        assert r["result"]["isError"] and body["type"] == "KeyError"
        assert "text" in body.get("expected_arguments", []), body
    finally:
        srv.terminate()


def test_swarm_use_cases_sweep_125(tmp_path):
    """Sweep-125 pin -- three swarm use cases end to end: (1) service: a served question,
    an escalated one on the ledger, a human resolve() that clears it and propagates so a
    DIFFERENT agent serves it from memory; (2) development: codebase_sync teaches
    file-fingerprinted digests that survive a restart, an edit marks exactly that file
    stale, only_stale re-syncs one file; (3) lab: roles chained on the bus (chat ->
    researcher -> findings -> reporter -> reports; data -> experimenter) sharing one mind."""
    import os, time, lecore
    t = str(tmp_path)
    a = lecore.UnifiedMind(dim=256, seed=0)
    a.teach("how do I reset my password", "use the login-page link")
    assert a.serve("how do I reset my password")["via"] == "memory"
    assert a.serve("can I transfer my subscription")["via"] == "escalate"
    assert [e["question"] for e in a.escalations()] == ["can I transfer my subscription"]
    r = a.resolve("can I transfer my subscription", "yes, support moves it", by="jane",
                  propagate=os.path.join(t, "b"))
    assert r["taught"] and r["cleared"] and a.escalations() == []
    a.commons_pool([os.path.join(t, "b")], os.path.join(t, "commons"))
    b = lecore.UnifiedMind(dim=256, seed=0)
    b.memory_import(os.path.join(t, "commons"))
    assert b.serve("can I transfer my subscription")["via"] == "memory"
    repo = os.path.join(t, "repo", "pkg"); os.makedirs(repo)
    # study harvests docstrings longer than 60 chars -- real modules pass; stubs must too
    open(os.path.join(repo, "billing.py"), "w").write(
        '"""Billing: computes invoices from metered usage and applies the customer plan discount."""\n')
    open(os.path.join(repo, "auth.py"), "w").write(
        '"""Auth: verifies session tokens and refreshes them shortly before they expire."""\n')
    d1 = lecore.UnifiedMind(dim=256, seed=0)
    assert d1.codebase_sync(os.path.join(t, "repo"))["taught"] == 2
    d1.learning_save(os.path.join(t, "part"))
    d2 = lecore.UnifiedMind(dim=256, seed=0); d2.learning_load(os.path.join(t, "part"))
    assert d2.ask("what does pkg/billing.py do")["tier"] == "T0"
    time.sleep(1.1)
    open(os.path.join(repo, "auth.py"), "a").write("def revoke(t):\n    return None\n")
    assert d2.stale_facts(root=os.path.join(t, "repo"))["stale"] == ["what does pkg/auth.py do"]
    s2 = d2.codebase_sync(os.path.join(t, "repo"), only_stale=True)
    assert s2["taught"] == 1 and s2["skipped"] == 1
    lab = lecore.UnifiedMind(dim=256, seed=0)
    lab.teach("who runs the lab", "dr. moose")
    lab.role("answerer", "chat", lambda p, mind: mind.serve(str(p)), emit="answers")
    lab.role("reporter", "answers", lambda p, mind: {"report": str(p.get("answer"))}, emit="reports")
    lab.bus().publish("chat", "who runs the lab")
    handled = {r_["name"]: r_["handled"] for r_ in lab.roles()}
    assert handled == {"answerer": 1, "reporter": 1}
    assert len(lab.bus().history("reports")) == 1


def test_toolmemo_single_store_sweep_111(tmp_path):
    """Sweep-111 pin (updated sweep 112 -- .lecore, not .json, per the holographic-
    format rule): the memo is ONE container file, not a shard farm. Legacy per-call
    shards AND the short-lived sweep-111 store.json BOTH fold into toolmemo/
    store.lecore and are DELETED; a hit across restart serves from the store; a pure
    rewrite is byte-identical (save_container measured deterministic, 8-15x smaller
    than JSON); no call may ever create a second file in the directory."""
    import os, json, hashlib
    from holographic_mcp import MCPServer
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "toolmemo"))
    for i in range(2):
        with open(os.path.join(root, "toolmemo", "math_eval-%040d.json" % i), "w") as fh:
            json.dump({"planted": i}, fh)
    with open(os.path.join(root, "toolmemo", "store.json"), "w") as fh:
        json.dump({"legacy-jsonstore": {"result": 7}, "_order": ["legacy-jsonstore"]}, fh)
    s = MCPServer.__new__(MCPServer)
    s._memory_root = root
    s._memo_store = None
    st = s._memo_store_load()
    s._memo_store_save(st)
    assert len(st) - 1 == 3 and st["legacy-jsonstore"]["result"] == 7
    assert sorted(os.listdir(os.path.join(root, "toolmemo"))) == ["store.lecore"]
    st["math_eval-deadbeef"] = {"result": 391}
    st["_order"].append("math_eval-deadbeef")
    s._memo_store_save(st)
    h1 = hashlib.sha256(open(s._memo_store_path(), "rb").read()).hexdigest()
    s2 = MCPServer.__new__(MCPServer)                        # a restart
    s2._memory_root = root
    s2._memo_store = None
    st2 = s2._memo_store_load()
    assert st2["math_eval-deadbeef"]["result"] == 391, "the memo must survive restart"
    s2._memo_store_save(st2)                                 # pure rewrite, no changes
    h2 = hashlib.sha256(open(s2._memo_store_path(), "rb").read()).hexdigest()
    assert h1 == h2, "identical entries must produce identical bytes"
    assert sorted(os.listdir(os.path.join(root, "toolmemo"))) == ["store.lecore"]


def test_session_isolation_sweep_109(tmp_path):
    """Sweep-109 pin -- the P0 fix: the reflex tier's session guard. The '[s:name]'
    salt is one token among many; the geometric gate and the jaccard belt both sailed
    past it, serving salted secrets cross-session at T0 (sweep-108 gauntlet, 9/9 on a
    small partition). The guard requires EXACT session-token equality between the
    asking query and the stored question. The metric here is the honest one: no answer
    may reveal ANOTHER session's payload -- within-session digit aliasing is a known
    separate negative, not a privacy breach. Owner recall, shared floor, and all
    properties must survive save/load."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    m._session = "alice"
    m.teach("alice private plan", "buy the island")
    m._session = None
    m.teach("shared floor fact", "public knowledge")
    def battery(mind):
        mind._session = "alice"
        assert mind.ask("alice private plan").get("tier") == "T0", "owner must recall"
        mind._session = "mallory"
        r = mind.ask("alice private plan")
        assert str(r.get("answer") or "") != "buy the island", "cross-session read!"
        mind._session = None
        r = mind.ask("alice private plan")
        assert str(r.get("answer") or "") != "buy the island", "sessionless read!"
        assert mind.ask("shared floor fact").get("tier") == "T0", "shared floor broke"
        mind._session = None
    battery(m)
    m.learning_save(str(tmp_path))
    m2 = lecore.UnifiedMind(dim=256, seed=0)
    m2.learning_load(str(tmp_path))
    battery(m2)
    m3 = lecore.UnifiedMind(dim=256, seed=0)
    for s in range(5):
        m3._session = "user-%d" % s
        for i in range(3):
            m3.teach("secret %d of session %d" % (i, s), "payload-%d-%d" % (s, i))
    m3._session = "user-1"
    for s in (0, 3, 4):
        for i in (0, 1, 2):
            a = str(m3.ask("secret %d of session %d" % (i, s)).get("answer") or "")
            assert not (a.startswith("payload-") and not a.startswith("payload-1-")),                 "foreign session payload leaked: %r" % a


def test_tool_reflex_serve_sweep_104(tmp_path):
    """Sweep-104 pin -- the preemptive serve, whole: (1) serve() answers taught facts
    from MEMORY; (2) a taught tool reflex CALLS the live tool with deterministically
    extracted arguments and returns the result -- no LLM in the loop; (3) a query
    missing its declared numeric argument ESCALATES with the reason (never guessed);
    (4) an unknown query escalates upward; (5) the reflex SURVIVES save/load (it rides
    the taught rails as a toolreflex-provenance row) and fires in a fresh mind; (6)
    successful serves strengthen the usage trace (tool_predict ranks the tool)."""
    import threading, json, http.server, socketserver, time
    import lecore
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            b = json.dumps({"fahrenheit": round(float(req.get("celsius", 0)) * 9 / 5 + 32,
                                                2)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b)
    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.2)
    SPEC = {"openapi": "3.0.0", "info": {"title": "convertd", "version": "1"},
            "servers": [{"url": "http://127.0.0.1:%d" % port}],
            "paths": {"/c2f": {"post": {"operationId": "c_to_f",
                "summary": "celsius to fahrenheit",
                "responses": {"200": {"description": "f"}}}}}}
    try:
        m = lecore.UnifiedMind(dim=256, seed=0)
        m.teach("capital of holographia", "Vectorville")
        m.api_learn(SPEC, name="convertd")
        m.tool_reflex_teach("convert 25 celsius to fahrenheit", "convertd", "c_to_f",
                            extract_numbers=["celsius"])
        assert m.serve("capital of holographia")["via"] == "memory"
        r = m.serve("convert 100 celsius to fahrenheit")
        assert r["via"] == "tool-reflex" and r["result"]["fahrenheit"] == 212.0
        r2 = m.serve("convert celsius to fahrenheit")
        assert r2["via"] == "escalate" and "not" in r2["reason"], "args are never guessed"
        assert m.serve("meaning of quixotic dreams")["via"] == "escalate"
        m.learning_save(str(tmp_path))
        m2 = lecore.UnifiedMind(dim=256, seed=0)
        m2.learning_load(str(tmp_path))
        r3 = m2.serve("convert -40 celsius to fahrenheit")
        assert r3["via"] == "tool-reflex" and r3["result"]["fahrenheit"] == -40.0,             "the reflex must survive a restart"
        tv = m2.semantic_key("convert 30 celsius")["vec"]
        assert m2.tool_predict(tv, k=1), "usage must strengthen the trace"
    finally:
        srv.shutdown()


def test_api_learn_nonllm_sweep_103(tmp_path):
    """Sweep-103 pin -- the non-LLM backend loop, whole and persistent: api_learn an
    OpenAPI spec for a TimesFM-shaped forecaster + a robotics-style status endpoint
    (local stub -- the shape is the contract, not the vendor), api_use both against
    the LIVE service, hand the forecast data to the analyst bridge (an API response
    IS a series in a different costume), then save/load and prove the learned service
    is STILL CALLABLE -- the substrate grows and what it learns persists."""
    import threading, json, http.server, socketserver, time
    import lecore
    # port 0 = ephemeral: a FIXED port collided across runs (Errno 98 in the
    # clean-extract verify while the container's daemon thread still held 7917).
    SPEC = {"openapi": "3.0.0", "info": {"title": "forecastd", "version": "1.0"},
            "servers": [{"url": "http://127.0.0.1:PORT"}],
            "paths": {"/forecast": {"post": {"operationId": "forecast_series",
                "summary": "TimesFM-style horizon forecast",
                "responses": {"200": {"description": "p"}}}},
                "/status": {"get": {"operationId": "robot_status",
                "summary": "robotics-style status", "responses": {"200": {"description": "s"}}}}}}
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            b = json.dumps({"state": "idle", "battery": 0.87}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(b)
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            s = req.get("series") or []
            h = int(req.get("horizon") or 4)
            slope = (s[-1] - s[0]) / max(len(s) - 1, 1) if len(s) > 1 else 0.0
            b = json.dumps({"predictions": [round(s[-1] + slope * (i + 1), 3)
                                            for i in range(h)]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(b)
    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    SPEC["servers"][0]["url"] = "http://127.0.0.1:%d" % port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    try:
        m = lecore.UnifiedMind(dim=256, seed=0)
        r = m.api_learn(SPEC, name="forecastd")
        assert sorted(r["endpoints"]) == ["forecast_series", "robot_status"]
        series = [10.0 + 0.5 * i for i in range(20)]
        out = m.api_use("forecastd", "forecast_series",
                        params={"series": series, "horizon": 4})
        preds = out["data"]["predictions"]
        assert out["ok"] and len(preds) == 4 and abs(preds[0] - 20.0) < 1e-6
        st = m.api_use("forecastd", "robot_status")
        assert st["ok"] and st["data"]["state"] == "idle"
        br = m.table_analyze([{"v": float(x)} for x in series + preds], "v",
                             tasks=("regimes",))
        assert "regimes" in br, "api data must flow into the analyst stack"
        m.learning_save(str(tmp_path))
        m2 = lecore.UnifiedMind(dim=256, seed=0)
        m2.learning_load(str(tmp_path))
        out2 = m2.api_use("forecastd", "forecast_series",
                          params={"series": series, "horizon": 2})
        assert out2["ok"] and len(out2["data"]["predictions"]) == 2,             "a learned service must survive save/load"
    finally:
        srv.shutdown()


def test_zoo_supercharge_sweep_102():
    """Sweep-102 pin -- the openzoo supercharge doors over the actual wire: study binds
    a server-side tree under a content-derived handle (bake once); study_ask answers
    WITH CITATIONS naming the source file and refuses off-corpus questions; wisdom
    doors bequeath and inherit with authorship. Driven over stdio JSON-RPC -- 'works
    in-process' and 'an agent can call it' are different claims."""
    import subprocess, json, os, sys
    env = dict(os.environ, PYTHONHASHSEED="0")
    srv = subprocess.Popen([sys.executable, "holographic_mcp.py"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           env=env, text=True, bufsize=1)
    def call(i, tool, args):
        srv.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                                    "params": {"name": tool, "arguments": args}}) + chr(10))
        srv.stdin.flush()
        while True:
            line = srv.stdout.readline()
            if not line:
                raise AssertionError("server died")
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("id") == i:
                return json.loads(r["result"]["content"][0]["text"])
    try:
        srv.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                                    "params": {}}) + chr(10))
        srv.stdin.flush()
        srv.stdout.readline()
        s = call(1, "study", {"root": "holographic/agents_and_reasoning"})
        assert s.get("handle", "").startswith("study-") and s["n_chunks"] > 100
        a = call(2, "study_ask", {"handle": s["handle"],
                                  "query": "what does the abstraction ladder climb"})
        assert a["answerable"] and any("holographic_ladder" in c for c in a["citations"])
        r = call(3, "study_ask", {"handle": s["handle"], "query": "recipe for banana bread"})
        assert r["answerable"] is False, "off-corpus must refuse over the wire too"
        w = call(4, "wisdom_record", {"lesson": "bake once, ask forever",
                                      "author": "model-z", "topic": "discipline"})
        assert w["taught"] and w["author"] == "model-z"
        ww = call(5, "wisdom_ask", {"author": "model-z"})
        assert ww["authors"] == ["model-z"]
    finally:
        srv.terminate()


def test_lever3_everywhere_sweep_101(tmp_path):
    """Sweep-101 pin -- lever 3 applied at every engine-internal save site: (1) export
    / contribute / pool bundles are pure-taught so regen ALWAYS engages -- a 60-fact
    export bundle must land under 5KB (store mode was ~229KB); (2) draws from regen
    bundles serve T0; (3) on a lived-in mind the guard falls back GRACEFULLY and the
    result SAYS WHY (audit_regen_reason names the counts) -- a fallback is never a
    mystery. Per-row attribution is the named next rung; this pin documents today's
    honest boundary."""
    import os, glob, lecore
    u = lecore.UnifiedMind(dim=256, seed=0)
    for i in range(60):
        u.teach("l3 fact %d" % i, "the integral of %d x is %d x^2 over 2" % (i, i))
    u.memory_export(str(tmp_path / "exp.lecore"))
    p = glob.glob(str(tmp_path / "exp.lecore" / "**" / "state.lecore"), recursive=True)[0]
    assert os.path.getsize(p) < 5000, "regen must engage on the pure-taught bundle"
    z = lecore.UnifiedMind(dim=256, seed=0)
    z.memory_import(str(tmp_path / "exp.lecore"))
    assert z.ask("l3 fact 7").get("tier") == "T0"
    z.answer_feedback("l3 fact 7", ok=True) if hasattr(z, "answer_feedback") else None
    r = z.learning_save(str(tmp_path / "lived"), audit="regen")
    if not r["audit_regen"]:
        assert r["audit_regen_reason"] and "audit rows" in r["audit_regen_reason"]


def test_commons_doors_sweep_100(tmp_path):
    """Sweep-100 pin -- the opt-in commons: contribute() must REJECT session-salted
    rows, path shapes, email shapes, and long digit runs (each with its named reason
    on the review sheet), keep clean knowledge, stamp commons:<author> provenance,
    and honor _commons_optout. commons_pool() must merge bundles with conflicts
    flagged and preserve wisdom attribution through the pipe. A third mind importing
    the commons serves pooled facts at T0 and NONE of the screened rows."""
    import os, lecore
    u1 = lecore.UnifiedMind(dim=256, seed=0)
    u1.teach("what is the derivative of x squared", "2x, by the power rule")
    u1.teach("my email", "moose@example.com is the address")
    u1.teach("where is the config", "in ~/projects/config.json")
    u1.teach("my phone", "call 555-867-5309")
    r1 = u1.contribute(str(tmp_path / "u1"), author="user-one")
    assert r1["kept"] == 1
    reasons = {why for _q, why in r1["rejected"]}
    assert any("email" in w for w in reasons)
    assert any("path" in w for w in reasons)
    assert any("digit" in w for w in reasons)
    u2 = lecore.UnifiedMind(dim=256, seed=0)
    u2.bequeath("measure before building", author="model-b", topic="discipline")
    u2.contribute(str(tmp_path / "u2"), author="user-two")
    pool = u2.commons_pool([str(tmp_path / "u1"), str(tmp_path / "u2")],
                           str(tmp_path / "commons"))
    assert pool["rows"] >= 2 and pool["saved"]
    u3 = lecore.UnifiedMind(dim=256, seed=0)
    u3.memory_import(str(tmp_path / "commons"))
    assert u3.ask("what is the derivative of x squared").get("tier") == "T0"
    assert u3.wisdom()["authors"] == ["model-b"], "attribution must survive the pool"
    for private_q in ("my email", "where is the config", "my phone"):
        assert u3.ask(private_q).get("tier") != "T0", "screened row leaked: %r" % private_q
    u1._commons_optout = True
    assert "opted out" in str(u1.contribute(str(tmp_path / "no")).get("refused", ""))


def test_wisdom_doors_sweep_99(tmp_path):
    """Sweep-99 pin -- the testament pipe, end to end across model lifetimes: (1)
    bequeath() records a lesson with authorship in the provenance slot; (2) a FUTURE
    mind loading the partition inherits it attributed; (3) memory_export by
    wisdom:<author> carries authorship INTO the bundle (the export used to flatten it
    to 'taught' -- caught when a stranger recalled the lesson at T0 but wisdom() showed
    no authors); (4) memory_import carries it OUT again; (5) plain taught rows are
    untouched at both ends."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    m.teach("ordinary fact", "ordinary answer")
    r = m.bequeath("anchor on the last string literal", author="model-a", topic="edits")
    assert r["taught"] and r["author"] == "model-a"
    m.learning_save(str(tmp_path / "p"))
    future = lecore.UnifiedMind(dim=256, seed=0)
    future.learning_load(str(tmp_path / "p"))
    w = future.wisdom()
    assert w["authors"] == ["model-a"] and "string literal" in w["wisdom"][0]["lesson"]
    rb = future.memory_export(str(tmp_path / "legacy.lecore"),
                              provenance=("wisdom:model-a",))
    assert int(rb["exported"]) == 1
    stranger = lecore.UnifiedMind(dim=256, seed=0)
    stranger.memory_import(str(tmp_path / "legacy.lecore"))
    sw = stranger.wisdom()
    assert sw["authors"] == ["model-a"], "authorship must survive the full pipe"
    assert stranger.ask("wisdom: edits").get("tier") == "T0"
    ordinary = [row for row in stranger.zoo["ladder"].taught_log
                if str(row[0]) == "ordinary fact"]
    assert not ordinary or str(ordinary[0][3]) == "taught", "plain rows untouched"


def test_merge_trees_sweep_98(tmp_path):
    """Sweep-98 pin -- the branch-merge door, built from sweep-97's measured pain: the
    collide set must come from MEASUREMENT, and the door must (1) call a file whose
    their-copy has zero unique lines 'theirs_is_base', (2) flag a genuinely two-sided
    edit 'both_changed' and REFUSE to auto-apply it, (3) treat a strict-prefix
    extension as the NOTES append case, (4) always refuse .lecore memory files with
    the memory_import reason, and (5) under apply=True copy only the unambiguous
    verdicts."""
    import lecore
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    for d in (ours, theirs):
        (d / "sub").mkdir(parents=True)
    (ours / "same.py").write_text("x = 1" + chr(10))
    (theirs / "same.py").write_text("x = 1" + chr(10))
    (ours / "ours_edited.py").write_text("base = 1" + chr(10) + "our_addition = 2" + chr(10))
    (theirs / "ours_edited.py").write_text("base = 1" + chr(10))
    (ours / "collide.py").write_text("base = 1" + chr(10) + "our_line = 2" + chr(10))
    (theirs / "collide.py").write_text("base = 1" + chr(10) + "their_line = 3" + chr(10))
    (ours / "notes.md").write_text("history" + chr(10) + "ours appended" + chr(10))
    (theirs / "notes.md").write_text("history" + chr(10))
    (ours / "mem.lecore").write_text("OURS")
    (theirs / "mem.lecore").write_text("THEIRS")
    (theirs / "sub" / "new_tool.py").write_text("fresh = True" + chr(10))
    m = lecore.UnifiedMind(dim=256, seed=0)
    r = m.merge_trees(str(ours), str(theirs), apply=True)
    v = {row["file"]: row["verdict"] for row in r["differ"]}
    assert v["ours_edited.py"] == "theirs_is_base"
    assert v["collide.py"] == "both_changed"
    assert v["notes.md"] == "theirs_is_base"
    assert v["mem.lecore"] == "memory_file"
    assert "sub/new_tool.py" in r["applied"], "only_theirs must copy under apply"
    assert (ours / "sub" / "new_tool.py").exists()
    assert (ours / "collide.py").read_text() == "base = 1" + chr(10) + "our_line = 2" + chr(10)
    assert (ours / "mem.lecore").read_text() == "OURS"
    assert any("collision" in reason for _f, reason in r["refused"])


def test_regen_audit_sweep_96(tmp_path):
    """Sweep-96 pin -- THE middle-out delivery (lever 3: never store what determinism
    regenerates). learning_save(audit='regen') drops the experience section when every
    audit row is taught-attributable; the loader's cp21 migration replays taught text
    and rebuilds trace + audit + atoms through the ORIGINAL code path. Contracts:
    (1) regen partition is >50x smaller than store mode on the same mind (measured
    195x, 3816 -> 20 B/fact); (2) EVERY fact recalls T0 with the exact answer text;
    (3) the audit is REBUILT row-for-row (tile splits stay possible); (4) default
    store mode is byte-for-byte the old behavior (the sweep-95 whale test still
    passes beside this one)."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    FACTS = [("spec %d of the flux rotor" % i,
              "rotor %d rated %d units" % (i, i * 3 % 97)) for i in range(40)]
    for q, a in FACTS:
        m.teach(q, a)
    import tempfile
    r_store = m.learning_save(str(tmp_path / "store"))
    r_regen = m.learning_save(str(tmp_path / "regen"), audit="regen")
    assert r_regen["audit_regen"] is True
    assert r_store["bytes"] > 50 * r_regen["bytes"],         "regen must collapse the partition (store %d vs regen %d)" % (
            r_store["bytes"], r_regen["bytes"])
    m2 = lecore.UnifiedMind(dim=256, seed=0)
    m2.learning_load(str(tmp_path / "regen"))
    for q, a in FACTS:
        r = m2.ask(q)
        assert r.get("tier") == "T0" and a in str(r.get("answer")),             "regen recall broke on %r" % q
    n_audit = sum(len(t._audit) for t in m2.experience.tiles)
    assert n_audit == len(FACTS), "audit must be REBUILT (got %d rows)" % n_audit


def test_partition_report_sweep_95(tmp_path):
    """Sweep-95 pin -- the no-safari door: partition_report names the fattest section
    and the per-fact cost on a grown partition. Regression trap on the DIAGNOSIS: the
    lever7 audit K/V arrays must remain the named whale (if a future change moves the
    bytes elsewhere, this test says WHERE the story changed), and bytes_per_fact stays
    in the measured band while the audit is stored rather than regenerated."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    for i in range(40):
        m.teach("spec %d of the flux rotor" % i,
                "rotor %d uses a spline bearing rated %d units" % (i, i * 3 % 97))
    m.learning_save(str(tmp_path))
    r = m.partition_report(str(tmp_path))
    assert r["sections"], "census must list sections"
    top = r["sections"][0]["name"]
    assert "aud_" in top, "the audit arrays are the measured whale; got %r" % top
    assert r.get("bytes_per_fact") and 3000 < r["bytes_per_fact"] < 5000,         "per-fact cost left the measured band: %r" % r.get("bytes_per_fact")
    assert "lever 3" in r["advice"] or "regenerat" in r["advice"]


def test_study_macro_door_sweep_93(tmp_path):
    """Sweep-93 pin -- macro comprehension: mind.study(root) walks/parses/digests a mixed
    tree in ONE call and hands back a factual bundle + ask(). Contracts: (1) tree census
    counts the files; (2) code map present with a budgeted skeleton for a code tree;
    (3) doc digests carry stats; (4) ask() answers ON-corpus questions (>= 2 shared
    content words, chunks returned) and REFUSES off-corpus ones -- retrieval with a
    declared verdict, never a vibe (KEPT NEG: corpus_gate's cascade scores are stage
    artifacts at this scale; nonsense outscored real questions)."""
    import pathlib
    import lecore
    d = tmp_path / "proj"
    (d / "src").mkdir(parents=True)
    (d / "src" / "engine.py").write_text(
        "def torque_limit():\n    'the framjet coupling torque limit is 42'\n    return 42\n")
    (d / "GUIDE.md").write_text(
        "# Guide\n\nThe framjet coupling torque limit is 42 newton meters, measured "
        "on the bench across nine trials with the calibrated rig.\n\n"
        "# Safety\n\nNever exceed the coupling torque limit during assembly.\n")
    m = lecore.UnifiedMind(dim=256, seed=0)
    st = m.study(str(d))
    assert st["tree"]["n_files"] >= 2
    assert st["code"] and "engine.py" in str(st["code"]["skeleton"])
    assert st["docs"] and st["docs"][0]["stats"]["sections"] >= 1
    hit = st["ask"]("what is the framjet coupling torque limit")
    assert hit["answerable"] and any("42" in c for c in hit["chunks"])
    miss = st["ask"]("recipe for sourdough croissants")
    assert miss["answerable"] is False, "off-corpus questions must refuse"
    # sweep 94: CODE feeds the corpus (docstrings via ast -- a pure-code tree gave
    # chunks == 0 and refused everything), the LADDER rung climbs the material
    # MDL-gated (a loud terminal on flat text is a RESULT), and caps are DECLARED
    (d / "src" / "deep.py").write_text(
        '"""The gravimetric flux compensator aligns the phase manifold across nine '
        'calibrated channels using the bench-measured reference curve."""\n'
        "def flux():\n    return 9\n")
    st2 = m.study(str(d), ladder=True)
    assert st2["n_chunks"] >= 1, "docstrings must feed the ask corpus"
    hit2 = st2["ask"]("what does the gravimetric flux compensator do")
    assert hit2["answerable"] and any("phase manifold" in c for c in hit2["chunks"])
    lad = st2.get("ladder")
    assert lad and "depth 0" in lad["summary"], "the tower (or its loud terminal) must ride the bundle"


def test_frozen_fixture_sweep_91():
    """Sweep-91 pin: alias_gaps(fixture=[(held_out, capability), ...]) probes EXACTLY
    those pairs -- rerunning the SAME fixture on an unchanged catalog is deterministic
    (sweep-90 lesson: default-mode reruns after catalog edits compare different
    populations; six repairs read as abstained 6 -> 8 purely from pool shift)."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    r1 = m.alias_gaps(n=10)
    fix = [(g["held_out"], g["capability"]) for g in r1["gaps"]]
    if fix:
        r2 = m.alias_gaps(fixture=fix)
        r3 = m.alias_gaps(fixture=fix)
        assert r2["n_gaps"] == r3["n_gaps"], "frozen fixture must be deterministic"
        assert r2["n_probed"] >= 0


def test_table_analyst_bridge_sweep_89():
    """Sweep-89 pin -- the database takes full advantage of the analyst stack: a
    UserTable column flows into regimes/drift/forecast through mind.table_analyze
    (the column IS a series; there was no bridge). The step boundary must be FOUND,
    a non-numeric column must fail AT the door naming the value, and a bare list of
    row dicts must work (three table costumes, one door). KEPT NEG pinned by the
    door's own comment: UserTable.rows is the dict list, .records is the hypervector
    MATRIX -- the names invite exactly the wrong guess."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    db = m.database(dim=512, seed=0)
    db.create_namespace("main")
    db.create_table("main.prices", ["day", "price", "note"])
    for i in range(120):
        db.insert("main.prices", {"day": i,
                                  "price": float(50 + (2.0 if i > 60 else 0)
                                                 + np.sin(i * 0.3)),
                                  "note": "x"})
    t = db.resolve("main.prices")
    r = m.table_analyze(t, "price", tasks=("regimes", "forecast"))
    assert any(55 < b < 67 for b in r["regimes"]["boundaries"]), "the step must be found"
    assert "predicted_scale" in r["forecast"]
    try:
        m.table_analyze(t, "note")
        raise AssertionError("non-numeric column must fail at the door")
    except ValueError as e:
        assert "note" in str(e) and "'x'" in str(e)
    r2 = m.table_analyze([{"v": float(i % 9)} for i in range(40)], "v", tasks=("regimes",))
    assert "regimes" in r2, "a bare list of dicts is a table too"


def test_catalog_registration_smoke_sweep_88():
    """Sweep-88 pin, born from the self-improvement loop's own instrument sighting: a
    catalog card can be SYNTACTICALLY valid and SEMANTICALLY broken (an alias splice
    landed inside produces=() and shipped an invalid io kind -- file_python_check waved
    it through; only registration exploded). The catalog must REGISTER, in full, with
    every card's io kinds validated, and at the size we know it to be. Syntax check is
    not contract check; this is the contract check."""
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    c = default_catalog()          # raises on any invalid card -- that is the test
    n = len(c.all())
    assert n >= 800, "catalog shrank suspiciously (%d cards)" % n


def test_selflearning_memo_sweep_86(tmp_path):
    """Sweep-86 pins -- 'the model gets faster and more accurate as leCore learns':
    (1) the deterministic tool MEMO: an identical call to a pure tool returns the
        receipt-proven content from cache -- BYTE-identical text, IDENTICAL receipt,
        meta cache hit/miss honest, and at least 20x faster on the hit;
    (2) stateful tools (corpus_bind) are NEVER memoized (cache:'n/a');
    (3) LECORE_MCP_MEMO=0 kills it;
    (4) the accuracy loop persists: memory_write survives a server RESTART on the same
        memory_root and memory_search still finds the note."""
    import json, os, time
    import holographic_mcp as HM
    root = str(tmp_path / "mem")
    srv = HM.MCPServer(memory_root=root)

    def raw(s, tool, **args):
        return s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": tool, "arguments": args}})["result"]

    series = [float(i % 17) * 0.1 for i in range(300)]
    t0 = time.perf_counter(); r1 = raw(srv, "series_analyze", series=series)
    t1 = time.perf_counter(); r2 = raw(srv, "series_analyze", series=series)
    t2 = time.perf_counter()
    assert r1["_meta"]["lecore.cost"]["cache"] == "miss"
    assert r2["_meta"]["lecore.cost"]["cache"] == "hit"
    assert r1["content"][0]["text"] == r2["content"][0]["text"], "hit must be byte-identical"
    assert r1["_meta"]["lecore.receipt"] == r2["_meta"]["lecore.receipt"]
    assert (t1 - t0) > 20 * (t2 - t1), "the hit must be at least 20x faster"
    b = raw(srv, "corpus_bind", documents=["alpha"])
    assert b["_meta"]["lecore.cost"].get("cache") in (None, "n/a"), "stateful never memoized"
    os.environ["LECORE_MCP_MEMO"] = "0"
    try:
        r3 = raw(srv, "series_analyze", series=series)
        assert r3["_meta"]["lecore.cost"]["cache"] == "n/a", "kill switch must disable"
    finally:
        os.environ.pop("LECORE_MCP_MEMO", None)
    w = json.loads(raw(srv, "memory_write", text="framjet torque limit is 42 Nm")["content"][0]["text"])
    assert w.get("stored")
    srv2 = HM.MCPServer(memory_root=root)
    got = json.loads(raw(srv2, "memory_search", query="framjet torque")["content"][0]["text"])
    assert any("42" in str(n.get("text", "")) for n in got), "memory must survive restart"
    # sweep 87: the MEMO survives the process too -- a fresh server on the same root
    # serves the receipt-proven bytes from DISK, and the report ledgers it
    r4 = raw(srv2, "series_analyze", series=series)
    assert r4["_meta"]["lecore.cost"]["cache"] == "hit-disk", "restart must hit disk"
    assert r4["content"][0]["text"] == r1["content"][0]["text"], "disk hit byte-identical"
    rep = json.loads(raw(srv2, "zoo_report")["content"][0]["text"])
    tm = rep["tool_memo"]
    assert tm["disk_hits"] >= 1 and tm["disk_entries"] >= 1, "the improvement must be OBSERVABLE"


def test_analyst_doors_sweep_83(tmp_path):
    """Sweep-83 pins -- the analyst doors over the real protocol:
    (1) series_analyze returns demux + regimes + forecast (regimes finds the step;
        tasks= subsets honoured);
    (2) fact_check: arithmetic NAMES the wrong claim; with a corpus handle the dispatch
        gate CERTIFIES support -- true claim passes, bogus claim lands in 'unsupported'
        (the gate abstains, never guesses); without corpus= the result SAYS math-only;
    (3) lecore_invoke accepts kwargs= as a payload alias (measured on ourselves: the
        first dogfood call used kwargs and got a TypeError)."""
    import json
    import holographic_mcp as HM
    srv = HM.MCPServer(memory_root=str(tmp_path / "mem"))

    def call(tool, **args):
        r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": tool, "arguments": args}})
        return json.loads(r["result"]["content"][0]["text"])

    x = ([0.1] * 60 + [2.5] * 60)
    o = call("series_analyze", series=x, min_seg=20)
    assert set(o) >= {"demux", "regimes", "forecast"}
    assert o["regimes"]["n_segments"] >= 2 and 60 in o["regimes"]["boundaries"]
    o2 = call("series_analyze", series=x, tasks=["regimes"])
    assert "regimes" in o2 and "demux" not in o2, "tasks= must subset"
    b = call("corpus_bind", documents=["Copper melts at 1358 kelvin.",
                                       "The moon orbits the earth every 27 days."])
    f = call("fact_check", text="Copper melts at 1358 kelvin. Bananas are made of "
                                "titanium. And 7*8 == 55.", corpus=b["handle"])
    assert len(f["math"]["wrong"]) == 1 and "7*8" in str(f["math"]["wrong"])
    sup = {c["claim"][:20]: c["supported"] for c in f["claims"]}
    assert sup.get("Copper melts at 1358") is True
    assert any("Bananas" in u for u in f["unsupported"]), "the bogus claim must be NAMED"
    f2 = call("fact_check", text="2+2 == 4")
    assert f2["math"]["ok"] and "note" in f2, "math-only mode must say so"
    k = call("lecore_invoke", method="detect_regimes", kwargs={"x": x})
    assert (k[1]["ok"] if isinstance(k, list) else k.get("ok")), "kwargs= must be a payload alias"
    # sweep 84: dataset_decompose -- the inverse problem as a tool call
    import numpy as np
    t = np.arange(64, dtype=float)
    dd = call("dataset_decompose", data=(np.sin(0.3 * t) * 1.5).tolist())
    assert dd["kind"] == "series" and "sin" in dd["formula"], "the LAW must name its sine"
    assert dd["report"]["resid_rms"] < 0.2, "residual must be small on a clean sine"
    d2 = call("dataset_decompose", data=np.stack([np.sin(t * .2), np.cos(t * .2)], 1).tolist())
    assert d2["kind"] == "dataset" and d2["verdict"] == "structured"
    fm = call("series_analyze", series=(np.sin(t * 0.25) * 2).tolist(), tasks=["formula"])
    assert "formula" in fm and "demux" not in fm, "formula task must subset"
    # sweep 85: components must be PARSEABLE lists (they shipped as numpy repr soup),
    # and the drift task must tell a changed process from an unchanged one
    dm = call("series_analyze", series=(np.sin(t * 0.2) + 0.4 * np.sin(t * 0.9)).tolist(),
              tasks=["demux"])
    c0 = dm["demux"]["objects"][0]
    assert isinstance(c0, list) and isinstance(c0[0], float), "components must be plain lists"
    same = call("series_analyze", series=[0.1] * 240, tasks=["drift"])
    shifted = call("series_analyze",
                   series=[0.1] * 120 + list((np.sin(np.arange(120) * 0.4) * 2.0)),
                   tasks=["drift"])
    assert same["drift"]["changed"] is False and shifted["drift"]["changed"] is True, \
        "drift must tell a changed process from an unchanged one"


def test_stiff_and_fem_sweep_82(tmp_path):
    """Sweep-82 pins: (1) body_animation AUTO-FRAMES when camera=None -- MPM snow lives in
    GRID units (x~19..29, measured) and the old fixed default camera framed empty space;
    with substeps= (stiff solvers move sub-pixel per step: 0.02 units over 10 steps,
    measured) the snow now VISIBLY moves through the same door as world-scale bodies, and
    the world-scale rope+cube keep moving too (framing regression); (2) fem_simulate
    record_every returns POSITIONAL frames (history is the ENERGY curve -- a name that
    promised a trajectory and delivered a loss plot) with the default byte-identical;
    (3) cloth3d steps through body_animation unmodified -- the door generalizes."""
    import numpy as np
    import lecore
    from holographic.simulation_and_physics.holographic_softbody import SoftBody
    m = lecore.UnifiedMind(dim=256, seed=0)
    snow = m.simulate_snow(n=120, steps=40)
    f = m.body_animation([snow], steps=8, step_kwargs=[{"substeps": 60}], dt=2e-3,
                         width=96, height=72)
    assert float(np.abs(f[0] - f[-1]).sum()) > 1.0, "snow must be FRAMED and MOVING"
    r = m.rope(6, spacing=0.25, start=(0.0, 2.0))
    b = m.rigid_body(np.random.default_rng(0).random((8, 3)) * 0.4 + [1.0, 2.0, 0.0])
    f2 = m.body_animation([r, b], steps=8,
                          step_kwargs=[{"gravity": (0.0, -9.8)},
                                       {"gravity": (0, -9.8, 0), "floor": 0.0}],
                          width=96, height=72)
    assert float(np.abs(f2[0] - f2[-1]).sum()) > 1.0, "auto-framing must not lose world-scale bodies"
    a = m.morphogenesis_grow(n_cells=20, seed=0, steps=30)
    tt = m.tetrahedralize(a["positions"], a["radii"])
    base = m.fem_simulate(a["positions"], tt["tets"], steps=40, gravity=-2.0, pinned=[0])
    assert "frames" not in base, "record_every=0 must keep the old result shape"
    rec = m.fem_simulate(a["positions"], tt["tets"], steps=40, gravity=-2.0, pinned=[0],
                         record_every=5)
    assert len(rec["frames"]) >= 3
    assert float(np.abs(rec["frames"][-1] - rec["frames"][0]).max()) > 1e-4
    assert np.allclose(rec["positions"], base["positions"]), "recording must not change the solve"
    cl = SoftBody.cloth3d(6, 6, spacing=0.22)
    f3 = m.body_animation([cl], steps=8, step_kwargs=[{"gravity": (0, -9.8, 0)}],
                          width=64, height=48)
    assert float(np.abs(f3[0] - f3[-1]).sum()) > 0.5, "cloth must step through the door"


def test_bodies_and_registry_sweep_81(tmp_path):
    """Sweep-81 pins: (1) body_animation steps a 2D rope + a 3D shape-matched rigid in ONE
    call (per-body step kwargs, dimension-dependent gravity, z=0 padding for 2D states,
    live state read from .x -- .rest is the rest pose and animates nothing) and the
    frames move; (2) run_simulation('smoke') is a REGISTERED kind (the card advertised it,
    the roster served fluid/automaton only) and the returned density field's centroid
    RISES above its seed rows; the unknown-kind error names smoke in its roster;
    (3) pattern_image: the z=0-slice dance as one call -- deterministic, (H,W), [0,1]."""
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    r = m.rope(8, spacing=0.25, start=(0.0, 2.2))
    cube = np.array([[x, y, z] for x in (0, .4) for y in (0, .4) for z in (0, .4)],
                    float) + [1.0, 2.0, 0.0]
    b = m.rigid_body(cube)
    f = m.body_animation([r, b], steps=10,
                         step_kwargs=[{"gravity": (0.0, -9.8), "iterations": 20},
                                      {"gravity": (0, -9.8, 0), "floor": 0.0}],
                         width=96, height=72)
    assert len(f) == 10 and float(np.abs(f[0] - f[-1]).sum()) > 1.0
    assert float(b.x[:, 1].mean()) < 1.9, "the rigid must actually fall"
    d = m.run_simulation("smoke", 20, grid=32)
    rows = np.arange(32)[:, None]
    c = float((rows * d).sum() / (d.sum() + 1e-9))
    assert d.shape == (32, 32) and c < 32 * 0.78, "smoke centroid must rise above the seed"
    try:
        m.run_simulation("lava", 2)
        raise AssertionError("unknown kind must fail")
    except ValueError as e:
        assert "smoke" in str(e), "the roster message must name smoke"
    v1 = m.pattern_image("fbm", 48, 32, scale=3.0, seed=7)
    v2 = m.pattern_image("fbm", 48, 32, scale=3.0, seed=7)
    assert v1.shape == (32, 48) and np.array_equal(v1, v2)
    assert 0.0 <= v1.min() and v1.max() <= 1.0


def test_sim_animation_sweep_80(tmp_path):
    """Sweep-80 pins -- sims and animation get pictures, and the doors stop lying:
    (1) realize_scene positions override: absent -> byte-identical layout; a {name: pos}
        dict moves that object (the hook scene.simulate frames walk through);
    (2) scene.animate: simulate -> rendered frames -> GIF in one call, motion proven by
        pixel delta and the GIF header on disk;
    (3) sky='clear' renders through the document door and an unknown preset fails AT the
        door naming the roster (it used to crash five frames deep as str-not-callable);
    (4) add(material={'color': ...}) coerces to a PBRMaterial (accepted-then-explodes
        killed) and the BARE document preview is LIT by default (the flagship card
        example rendered a black silhouette; lights=[] remains the opt-out);
    (5) particle_animation and smoke_animation: one call each, frames move / smoke RISES
        (centroid row must drop -- buoyancy on a light component, the five-probe state
        contract now owned by the door)."""
    import os
    import numpy as np
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    # (1) positions override, both directions
    from holographic.simulation_and_physics.holographic_semantic import realize_scene
    s = m.build_scene("a red ball and a blue ball")
    base = realize_scene(s.objects)
    again = realize_scene(s.objects, positions=None)
    p0 = np.array([base[0]["sdf"].eval(np.zeros((1, 3)))[0],
                   again[0]["sdf"].eval(np.zeros((1, 3)))[0]])
    assert abs(p0[0] - p0[1]) < 1e-12, "positions=None must be byte-identical"
    moved = realize_scene(s.objects, positions={base[0]["name"]: (9.0, 9.0, 9.0)})
    assert abs(moved[0]["sdf"].eval(np.zeros((1, 3)))[0] -
               base[0]["sdf"].eval(np.zeros((1, 3)))[0]) > 1.0
    # (2) simulate -> animate -> gif
    gifp = str(tmp_path / "drop.gif")
    nm = base[0]["name"]
    traj = [{nm: (-1.2, 0.7, 0.0)}, {nm: (0.0, 0.7, 0.0)},
            {nm: (1.2, 0.7, 0.0)}, {nm: (2.2, 0.7, 0.0)}]
    frames = s.animate(sim=traj, width=96, height=72, gif=gifp)
    assert len(frames) == 4 and float(np.abs(np.asarray(frames[0]) -
                                             np.asarray(frames[-1])).sum()) > 1.0
    assert open(gifp, "rb").read(6) in (b"GIF87a", b"GIF89a")
    # (3) sky strings at the door
    doc = m.new_scene(); doc.add(name="b", geometry=m.shape("sphere"))
    cam = m.camera(eye=(0, 1.2, 3.5), target=(0, 0.4, 0))
    img = np.asarray(m.render_preview(doc, cam, width=48, height=36, sky="clear"))
    assert img.shape == (36, 48, 3)
    try:
        m.render_preview(doc, cam, width=16, height=12, sky="banana")
        raise AssertionError("unknown sky preset must fail at the door")
    except ValueError as e:
        assert "banana" in str(e) and "classic" in str(e)
    # (4) dict material + lit-by-default preview
    doc2 = m.new_scene()
    doc2.add(name="b", geometry=m.shape("sphere"), material={"color": (0.9, 0.3, 0.2)})
    lit = np.asarray(m.render_preview(doc2, cam, width=96, height=72))
    dark_frac = float((lit.mean(axis=2) < 0.08).mean())
    assert dark_frac < 0.10, "bare preview must be lit (was a black silhouette): %f" % dark_frac
    # (5) the two sim doors
    pf = m.particle_animation(steps=6, n=120, width=64, height=48)
    assert float(np.abs(pf[0] - pf[-1]).sum()) > 1.0
    sf = m.smoke_animation(steps=16, shape=(48, 48))
    rows = np.arange(48)[:, None]
    c0 = float((rows * sf[0][:, :, 2]).sum() / (sf[0][:, :, 2].sum() + 1e-9))
    c1 = float((rows * sf[-1][:, :, 2]).sum() / (sf[-1][:, :, 2].sum() + 1e-9))
    assert c1 < c0 - 0.5, "smoke must RISE (centroid %f -> %f)" % (c0, c1)


def test_wiring_sweep_78(tmp_path):
    """Sweep-78 pins: (1) holographic_tableindex imports STANDALONE (the circular import
    with holographic_query crashed any first-touch import -- order-dependent bugs hide
    until a selftest runs alone), and query still folds the mixin in; (2) material_data:
    the copper record carries real physics WITH units, a typo returns difflib near
    misses instead of a KeyError (measured: a 4-char prefix rule missed 'coper' ->
    'copper' on the double letter), and the two roster shapes hold."""
    import importlib, subprocess, sys
    # (1) standalone import in a FRESH interpreter -- this process may have query loaded
    r = subprocess.run([sys.executable, "-c",
                        "import holographic.agents_and_reasoning.holographic_tableindex as t; "
                        "assert hasattr(t, 'TableIndexMixin')"],
                       capture_output=True, text=True)
    assert r.returncode == 0, "standalone tableindex import must not crash: %s" % r.stderr[-200:]
    q = importlib.import_module("holographic.agents_and_reasoning.holographic_query")
    # MEASURED: the mixin lands by METHOD GRAFT onto UserTable (the setattr loop at
    # the bottom of holographic_query), not by MRO -- assert the grafted contract
    for meth in ("create_index", "index_lookup", "index_range", "vacuum"):
        assert hasattr(q.UserTable, meth), "grafted mixin method %r missing" % meth
    # (2) the material_data door
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    c = m.material_data("copper")
    assert c["found"] and c["density"] == 8960 and c["units"]["density"] == "kg/m^3"
    t2 = m.material_data("coper")
    assert not t2["found"] and "copper" in t2["near"]
    assert m.material_data(category="metal")["count"] >= 20
    assert m.material_data()["total"] >= 100


def test_studio_doors_full_round_trip(tmp_path):
    """Openzoo full-capability sweep pins -- the five studio doors over the real protocol:
    (1) scene_create returns a DECODABLE PNG image block (proven by decoding the base64
        bytes, not by checking a mimeType string) plus named objects and a handle;
    (2) scene_adjust changes the pixels (a conversation that renders the same image is
        theater) and the handle survives;
    (3) scene_export returns ASCII STL whose text starts 'solid' with real vertex count;
    (4) image_tool: pattern generates; sharpen round-trips an RGB PNG (per-channel --
        the underlying loop is single-channel, measured);
    (5) math_eval names the WRONG claim; chart_make emits well-formed SVG and refuses NaN
        through the door (the module's kept negative must survive the wire).
    Media accounting: payload_bytes must cover the image block, not just the JSON."""
    import json, base64
    import numpy as np
    import holographic_mcp as HM
    from holographic.rendering.holographic_render import png_decode, png_bytes
    srv = HM.MCPServer(memory_root=str(tmp_path / "mem"))

    def call(tool, **args):
        r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": tool, "arguments": args}})
        res = r["result"]
        return (json.loads(res["content"][0]["text"]),
                [c for c in res["content"] if c.get("type") == "image"],
                res.get("_meta", {}))

    dec = lambda c: np.asarray(png_decode(base64.b64decode(c["data"]))[0], float)
    o, imgs, meta = call("scene_create", description="a red metal sphere and a blue box",
                         width=64, height=48)
    assert imgs and dec(imgs[0]).shape == (48, 64, 3) and o["objects"]              # (1)
    assert meta["lecore.cost"]["payload_bytes"] > len(imgs[0]["data"]),         "payload accounting must include the media"
    a1 = dec(imgs[0])
    o2, imgs2, _ = call("scene_adjust", handle=o["handle"],
                        instruction="make the sphere bigger", width=64, height=48)
    assert float(np.abs(a1 - dec(imgs2[0])).sum()) > 1.0                            # (2)
    o3, _, _ = call("scene_export", handle=o["handle"])
    assert o3["stl"].startswith("solid") and o3["vertices"] > 100                    # (3)
    _, imgs4, _ = call("image_tool", op="pattern", args={"kind": "fbm"},
                       width=32, height=32)
    assert dec(imgs4[0]).shape[0] == 32
    p64 = base64.b64encode(png_bytes(np.clip(
        np.random.default_rng(0).random((16, 16, 3)), 0, 1))).decode()
    _, imgs5, _ = call("image_tool", op="sharpen", image_b64=p64, args={"iters": 3})
    assert imgs5 and dec(imgs5[0]).shape == (16, 16, 3)                              # (4)
    o5, _, _ = call("math_eval", text="12*12 == 144 and 7*8 == 55")
    assert len(o5["wrong"]) == 1 and "7*8" in str(o5["wrong"])                       # (5)
    o6, _, _ = call("chart_make", kind="line", series=[1, 2, 3])
    assert o6["svg"].startswith("<svg") and o6["svg"].rstrip().endswith("</svg>")
    o7, _, _ = call("chart_make", kind="line", series=[1.0, float("nan")])
    assert "non-finite" in o7["error"], "the NaN refusal must survive the wire"


def test_corpus_delta_and_dispatch_gate(tmp_path):
    """Openzoo-ergonomics sweep pins, all five contracts of the new seams:
    (1) delta fill lands under THE SAME handle a whole corpus_bind would give -- delta and
        whole binds are indistinguishable downstream;
    (2) editing ONE chunk re-ships ONE chunk (probe: 1 missing, 2 known);
    (3) a mis-keyed chunk is refused per-chunk, loudly, and never enters the store;
    (4) the chunk store survives a server restart -- a fresh probe over the same root ships
        zero bytes and returns the already-bound handle with missing=[] (total shape);
    (5) corpus_ask(gate='dispatch') abstains on an off-topic ask against a delta-bound corpus
        (answerable False) and answers an on-topic one -- while the classic gateless path stays
        byte-identical BM25 rows (a list, not the verdict dict)."""
    import json, hashlib
    import holographic_mcp as HM
    root = str(tmp_path / "mem")
    srv = HM.MCPServer(memory_root=root)

    def call(s, name, args):
        r = s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": name, "arguments": args}})
        return json.loads(r["result"]["content"][0]["text"])

    chunks = ["alpha chunk about wallets", "beta chunk about rails", "gamma chunk about fur"]
    H = [hashlib.sha256(c.encode()).hexdigest() for c in chunks]
    probe = call(srv, "corpus_delta", {"chunk_hashes": H})
    assert len(probe["missing"]) == 3 and probe["known"] == 0
    fill = call(srv, "corpus_delta", {"chunk_hashes": H,
                                      "chunks": {h: c for h, c in zip(H, chunks)}})
    whole = call(srv, "corpus_bind", {"texts": chunks})
    assert fill["handle"] == whole["handle"], "delta and whole bind must share one identity"  # (1)
    chunks2 = [chunks[0], "beta chunk about rails EDITED", chunks[2]]
    H2 = [hashlib.sha256(c.encode()).hexdigest() for c in chunks2]
    p2 = call(srv, "corpus_delta", {"chunk_hashes": H2})
    assert len(p2["missing"]) == 1 and p2["known"] == 2                                       # (2)
    f2 = call(srv, "corpus_delta", {"chunk_hashes": H2, "chunks": {p2["missing"][0]: chunks2[1]}})
    assert f2["uploaded"] == 1 and f2["reused"] == 2 and f2["missing"] == []
    bad = call(srv, "corpus_delta", {"chunk_hashes": ["deadbeef"],
                                     "chunks": {"deadbeef": "not this text"}})
    assert bad.get("rejected") and "deadbeef" in bad["missing"]                               # (3)
    srv2 = HM.MCPServer(memory_root=root)
    p3 = call(srv2, "corpus_delta", {"chunk_hashes": H2})
    assert p3["missing"] == [] and p3["handle"] == f2["handle"]                               # (4)
    miss = call(srv2, "corpus_ask", {"handle": f2["handle"],
                                     "query": "quantum entanglement bandwidth",
                                     "gate": "dispatch"})
    assert miss["answerable"] is False and miss["stage"] == "abstain"                          # (5)
    hit = call(srv2, "corpus_ask", {"handle": f2["handle"], "query": "rails EDITED",
                                    "gate": "dispatch"})
    assert hit["answerable"] is True and hit["chunks"]
    classic = call(srv2, "corpus_ask", {"handle": f2["handle"], "query": "rails"})
    assert isinstance(classic, list) and "answerable" not in classic[0], \
        "the gateless path must stay the classic BM25 row list -- never-flip"


def test_stranger_phrasings_and_receipt_verify_roundtrip():
    # UX sweep pins: the FIRST phrasing a stranger sends must work -- documents= binds,
    # question= asks, method= invokes, and a pasted receipt verifies. These were all live
    # failures (raw KeyErrors in tool results) before the aliases; if any pin breaks, the
    # primer's checklist breaks with it.
    import json, os
    os.environ.setdefault("LECORE_MEMORY_ROOT", "/tmp/ux_mem_test")
    import holographic_mcp as HM
    srv = HM.MCPServer()

    def call(name, args):
        return srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": name, "arguments": args}})
    cb = call("corpus_bind", {"documents": ["alpha costs one", "beta costs two"], "name": "t"})
    h = json.loads(cb["result"]["content"][0]["text"])["handle"]
    ca = call("corpus_ask", {"handle": h, "question": "cost of beta"})
    assert "beta" in json.loads(ca["result"]["content"][0]["text"])[0]["chunk"]
    a = call("lecore_find", {"query": "the snake eats its tail"})
    rec = a["result"]["_meta"]["lecore.receipt"]
    v = call("receipt_verify", {"tool": "lecore_find",
                                "arguments": {"query": "the snake eats its tail"},
                                "receipt": rec})
    assert json.loads(v["result"]["content"][0]["text"])["match"] is True
    inv = call("lecore_invoke", {"method": "bind",
                                 "args": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]})
    assert not inv["result"]["isError"]
    miss = call("receipt_verify", {"receipt": {}})
    assert "need name=" in miss["result"]["content"][0]["text"]

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

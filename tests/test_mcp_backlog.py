"""tests/test_mcp_backlog.py -- the MCP backlog (docs/research/BACKLOG_mcp.md, M1-M16), each item pinned by
the measurement that accepted it. In-process through MCPServer.handle(), no subprocess, no network."""
import json
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import holographic_mcp as H  # noqa: E402


def _srv(proto="2025-06-18", caps=None):
    s = H.MCPServer()
    s.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": proto, "capabilities": caps or {}}})
    return s


def _call(s, name, args, rid=9):
    return s.handle({"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": {"name": name, "arguments": args}})["result"]


def test_m1_annotations_and_guidance_on_every_tool():
    tools = H._annotate_tools(list(H._TOOLS))
    assert all(set(t["annotations"]) == {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"} for t in tools)
    assert all(("instead" in t["description"].lower() or "use it only" in t["description"].lower()) for t in tools)


def test_m2_profiles_and_m13_banner():
    s = _srv()                                                      # one server; the profile is read per tools/list
    for prof, max_n, max_bytes in (("minimal", 8, 6000), ("standard", 32, 40000), ("full", 200, 80000)):
        os.environ["LECORE_MCP_PROFILE"] = prof
        try:
            tl = s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
        finally:
            os.environ.pop("LECORE_MCP_PROFILE", None)
        core = [t for t in tl if not t["name"].startswith(("studio", "asset_"))]
        assert len(core) <= max_n and len(json.dumps(core)) <= max_bytes, (prof, len(core))
    assert len(H._INSTRUCTIONS) <= 1300


def test_m3_prompts_render_the_four_part_shape():
    s = _srv()
    names = [p["name"] for p in s.handle({"jsonrpc": "2.0", "id": 1, "method": "prompts/list"})["result"]["prompts"]]
    assert set(names) >= {"decide", "review", "plan", "route"}
    g = s.handle({"jsonrpc": "2.0", "id": 2, "method": "prompts/get", "params": {"name": "decide", "arguments": {"state": "courier lost the package", "options": ["billing", "shipping"]}}})["result"]
    text = g["messages"][0]["content"]["text"]
    assert text.index("GOAL:") < text.index("RETURN FORMAT:") < text.index("CONSTRAINTS:") < text.index("VERIFICATION:")


def test_m4_resources_list_and_read():
    s = _srv()
    uris = [r["uri"] for r in s.handle({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})["result"]["resources"]]
    assert "lecore://decisions/recent" in uris and "lecore://map" in uris
    r = s.handle({"jsonrpc": "2.0", "id": 2, "method": "resources/read", "params": {"uri": "lecore://map"}})["result"]["contents"][0]
    assert json.loads(r["text"])
    assert "error" in s.handle({"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {"uri": "lecore://nope"}})


def test_m7_negotiation_and_structured_content():
    new = _srv("2025-06-18")
    r = _call(new, "lecore_decide", {"state": "courier lost the package", "options": ["billing", "shipping"]})
    assert r["structuredContent"]["value"] == "shipping" and r["content"][0]["type"] == "text"
    old = _srv("2024-11-05")
    r2 = _call(old, "lecore_decide", {"state": "courier lost the package", "options": ["billing", "shipping"]})
    assert "structuredContent" not in r2


def test_tiered_find_never_answers_off_catalog_and_learns():
    s = _srv()
    g = json.loads(_call(s, "lecore_find", {"query": "asdf qwer zxcv"})["content"][0]["text"])
    assert g["tier"] != "answer"
    r = json.loads(_call(s, "lecore_find", {"query": "smooth a bumpy mesh"})["content"][0]["text"])
    assert r["tier"] == "answer" and r["id"]
    json.loads(_call(s, "lecore_outcome", {"id": r["id"], "outcome": r["answer"]})["content"][0]["text"])
    assert json.loads(_call(s, "lecore_find", {"query": "smooth a bumpy mesh"})["content"][0]["text"])["via"] == "reflex"


def test_m9_m10_code_and_verify_tools(tmp_path):
    s = _srv()
    s.service.mind.set_file_root(str(tmp_path))
    (tmp_path / "m.py").write_text("def f():\n    \"\"\"d\"\"\"\n    return 1\n")
    bad = _call(s, "lecore_edit", {"path": "m.py", "old": "    return 1\n", "new": "    return (\n"})["structuredContent"]
    assert bad["ok"] is False and (tmp_path / "m.py").read_text().endswith("return 1\n")
    rv = _call(s, "lecore_review", {"paths": ["m.py"], "duplicates": False, "purity": False, "tests": False})["structuredContent"]
    assert rv["merge_ready"] is True and "files" in rv
    v = _call(s, "lecore_verify", {"state": "anything", "answer": "x"})["structuredContent"]
    assert v["valid"] is None                                    # no experience -> undecided, never a guess


def test_m11_image_block_argument_is_normalised():
    s = _srv()
    import base64, io
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (8, 8), (255, 128, 0)).save(buf, format="PNG")
    blk = {"type": "image", "mimeType": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}
    r = _call(s, "lecore_analyze", {"image_b64": blk, "regions": [{"name": "all", "box": [0, 0, 1, 1]}]})["structuredContent"]
    assert 0.4 < r["regions"][0]["brightness"] < 0.6 and 20 < r["regions"][0]["dominant_hue"] < 40


def test_m12_sampling_escalates_once_then_the_reflex_answers():
    s = _srv(caps={"sampling": {}})
    calls = []
    s._sampler = lambda req: (calls.append(req), {"jsonrpc": "2.0", "id": req["id"], "result": {"role": "assistant", "content": {"type": "text", "text": "\"shipping\""}}})[1]
    a = _call(s, "lecore_decide", {"state": "zzz qqq nothing here", "options": ["billing", "shipping"], "escalate": True})["structuredContent"]
    assert a["escalated"] and a["via"] == "model_end" and a["value"] == "shipping" and "RETURN FORMAT" in calls[0]["params"]["messages"][0]["content"]["text"]
    _call(s, "lecore_outcome", {"id": a["id"], "outcome": "shipping"})
    b = _call(s, "lecore_decide", {"state": "zzz qqq nothing here", "options": ["billing", "shipping"], "escalate": True})["structuredContent"]
    assert b["via"] == "reflex" and len(calls) == 1
    s2 = _srv()                                                     # a host that cannot sample
    c = _call(s2, "lecore_decide", {"state": "zzz qqq nothing here", "options": ["billing", "shipping"], "escalate": True})["structuredContent"]
    assert c["escalated"] is False


def test_m5_progress_notifications_queue_and_drain():
    s = _srv()
    s.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "lecore_decide", "arguments": {"state": "x", "options": ["a", "b"]}, "_meta": {"progressToken": "t1"}}})
    msgs = [n["params"]["message"] for n in s._drain_notifications()]
    assert msgs == ["lecore_decide started", "lecore_decide done"] and s._drain_notifications() == []


def test_m8_studio_tools_are_gated_on_reachability():
    os.environ["LESTUDIO3D_URL"] = "http://127.0.0.1:1"
    os.environ["LESTUDIO_URL"] = "http://127.0.0.1:1"
    try:
        s = _srv()
        assert not [t for t in s._studio_tools() if t["name"].startswith("studio")]
    finally:
        os.environ.pop("LESTUDIO3D_URL", None); os.environ.pop("LESTUDIO_URL", None)
    assert all(t["name"] in ("studio3d_new", "studio3d_op", "studio3d_assign", "studio3d_scene", "studio3d_render", "studio3d_analyze", "studio3d_export", "studio3d_ops") for t in H._STUDIO3D_TOOLS)


def test_cache_is_salted_with_the_code_hash():
    assert len(H._server_code_hash()) == 16


def test_m14_server_manifest_is_valid():
    d = json.load(open(os.path.join(_REPO, "server.json")))
    assert d["packages"][0]["identifier"] == "leos-core" and d["remotes"][0]["transport_type"] == "streamable-http"


@pytest.mark.slow
def test_m15_mcp_lint_passes():
    """The whole lint is a ~30 s run (15 off-catalog finds, the prompts, the resources); it is the MCP gate."""
    p = subprocess.run([sys.executable, os.path.join("tools", "mcp_lint.py")], cwd=_REPO, capture_output=True, text=True,
                       timeout=900, env=dict(os.environ, PYTHONHASHSEED="0"))
    assert p.returncode == 0, p.stdout[-600:] + p.stderr[-300:]

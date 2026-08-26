"""Tests for the tool interface both directions: the service's /tools + /invoke, and holographic_toolclient."""
import threading
import time
import numpy as np
import pytest
from http.server import HTTPServer
from holographic_service import Service, make_handler
from holographic.io_and_interop.holographic_toolclient import remote_tools, call, list_tools, RemoteTool


@pytest.fixture
def server():
    """A real leCore service on an OS-assigned free port (port 0 -> parallel-safe), token-gated, torn down after."""
    svc = Service(token="secret")
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(svc))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    yield "http://127.0.0.1:%d" % port, "secret"
    httpd.shutdown()
    httpd.server_close()


def test_tools_manifest_lists_faculties(server):
    base, tok = server
    tools = list_tools(base, token=tok)
    assert len(tools) > 100                                     # the mind exposes many faculties
    names = {t["name"] for t in tools}
    assert "opponent_channels" in names and "refine" in names
    assert all(not n.startswith("_") for n in names)           # only public faculties


def test_remote_invoke_runs_a_faculty(server):
    base, tok = server
    # orthogonal vectors -> divergence ~pi/2, purple = a+b (present)
    r = call(base, "opponent_channels", {"vec_a": [1, 0, 0, 0], "vec_b": [0, 1, 0, 0]}, token=tok)
    assert abs(r["divergence_score"] - np.pi / 2) < 1e-3
    assert r["channel_magnitudes"]["purple"] > 1.0             # numpy result serialized over HTTP


def test_remote_tools_yields_callables(server):
    base, tok = server
    tools = {t.name: t for t in remote_tools(base, token=tok)}
    assert isinstance(tools["opponent_channels"], RemoteTool)
    r = tools["opponent_channels"].run({"vec_a": [1, 0], "vec_b": [1, 0]})
    assert r["divergence_score"] < 1e-6                         # identical -> no divergence


def test_auth_required(server):
    base, _ = server
    with pytest.raises(Exception):
        call(base, "opponent_channels", {"vec_a": [1, 0], "vec_b": [0, 1]}, token="WRONG")


def test_private_and_unknown_tools_are_refused(server):
    base, tok = server
    r1 = call(base, "_secret", {}, token=tok) if False else None   # RemoteTool.run raises on ok=False; use raw check
    from holographic.io_and_interop.holographic_toolclient import _post
    resp = _post(base + "/invoke", {"name": "_private", "args": {}}, token=tok)
    assert not resp["ok"]
    resp2 = _post(base + "/invoke", {"name": "no_such_tool_xyz", "args": {}}, token=tok)
    assert not resp2["ok"]


def test_orchestrator_registers_local_remote_and_command(server):
    base, tok = server
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=128, seed=0)
    orch = m.orchestrator
    # a remote tool
    for t in remote_tools(base, token=tok):
        if t.name == "opponent_channels":
            orch.register(t)
            break
    # a shell command (echo is allowlisted here)
    orch.register_command("shout", ["echo", "LOUD:"], allow=True)
    names = orch.tools()
    assert "opponent_channels" in names and "shout" in names
    shout = [x for x in orch.registry.tools if x.name == "shout"][0]
    assert shout.fn("hi").strip() == "LOUD: hi"


def test_attach_llm_returns_bridge_on_the_mind_bus():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=128, seed=0)
    seen = []
    bridge = m.attach_llm(lambda text: "echo:" + text, name="tester")
    assert bridge.llm("hello") == "echo:hello"
    assert bridge.bus is m.bus()                                # wired to the mind's own bus

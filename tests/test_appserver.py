"""The standard agent surface (holographic_appserver): what leStudio and Poly Studio each built inside their own
Flask servers, mounted once. Sweep 163."""
import json
import tempfile

import pytest

import lecore
from holographic.io_and_interop import holographic_appserver as A
from holographic.io_and_interop.holographic_lews import Workspace

flask = pytest.importorskip("flask")


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def make_app():
    from flask import Flask, jsonify, Response
    app = Flask("demo"); state = {"n": 0}

    @app.route("/api/paint", methods=["POST"])
    def paint():
        """Paint one stroke."""
        state["n"] += 1; return jsonify(ok=True, strokes=state["n"])

    @app.route("/api/render")
    def render():
        """A PNG frame."""
        return Response(b"\x89PNG\r\n\x1a\nfake", mimetype="image/png", headers={"X-Converged": "1"})

    @app.route("/api/photo")
    def photo():
        return Response("{}\n", mimetype="application/x-ndjson")
    return app


def test_manifest_is_this_mount_only_and_types_returns(mind):
    app = make_app(); root = tempfile.mkdtemp()
    mind.agent_surface(app, base="/api", app_name="demo", workspace_root=root, image_routes=("render",), stream_routes=("photo",))
    man = app.test_client().get("/api/agent/tools").get_json()
    by = {t["name"]: t for t in man["tools"]}
    assert set(by) == {"paint", "render", "photo"}                       # standard doors and /agent/* are not tools
    assert by["render"]["returns"] == "image" and by["photo"]["returns"] == "stream" and by["paint"]["body"] == "json"
    assert by["paint"]["summary"] == "Paint one stroke." and man["identity"]["X-User"]


def test_invoke_json_and_base64_image_and_stream_refusal(mind):
    app = make_app(); mind.agent_surface(app, workspace_root=tempfile.mkdtemp(), image_routes=("render",), stream_routes=("photo",))
    c = app.test_client()
    r = c.post("/api/agent/invoke", json={"tool": "paint", "json": {}}).get_json(); assert r["result"]["strokes"] == 1
    img = c.post("/api/agent/invoke", json={"tool": "render"}).get_json()
    assert img["result"]["image"].startswith("data:image/png;base64,") and img["result"]["headers"]["X-Converged"] == "1"
    bad = c.post("/api/agent/invoke", json={"tool": "photo"}); assert bad.status_code == 400 and bad.get_json()["path"] == "/api/photo"


def test_mind_door_rejects_with_signatures_and_answers_allowlist(mind):
    app = make_app(); mind.agent_surface(app, workspace_root=tempfile.mkdtemp())
    c = app.test_client()
    bad = c.post("/api/mind", json={"name": "file_replace"})              # a mutating faculty is never exposed
    assert bad.status_code == 400 and "file_replace" not in bad.get_json()["allowed"] and bad.get_json()["allowed"]["version"].startswith("(")
    ok = c.post("/api/mind", json={"name": "find_capability", "args": {"problem": "shared workspace"}}).get_json()
    assert ok["ok"] and isinstance(ok["result"], list)


def test_engine_status_is_one_preflight(mind):
    es = mind.engine_status()
    assert es["engine"] and es["faculties"] > 1000 and set(es["extras"]) >= {"numba", "cupy", "flask"}
    assert es["policy"]["bit_exact"] in (True, False) and "any_available" in es["gpu"]
    json.dumps(es)


def test_mutations_become_workspace_notes_and_presence_is_by_person(mind):
    app = make_app(); root = tempfile.mkdtemp(); mind.agent_surface(app, workspace_root=root, app_name="lestudio")
    c = app.test_client()
    for tab in ("tab1", "tab2", "tab3"):                                  # one person, three tabs
        c.post("/api/paint", json={}, headers={"X-Client": tab, "X-User": "moose"})
    c.get("/api/presence")                                                  # a GET changes nothing
    other = Workspace(root, app="polystudio", create=False)
    assert [e["kind"] for e in other.since(0)] == ["POST paint"] * 3 and [e["app"] for e in other.since(0)] == ["tab1", "tab2", "tab3"]
    assert other.participants() == ["moose"]                             # not tab1/tab2/tab3: no ghosts
    assert other.since(0, exclude="tab2") and len(other.since(0, exclude="tab2")) == 2
    pres = c.get("/api/presence", headers={"X-User": "moose"}).get_json()
    assert pres["participants"][0]["who"] == "moose" and pres["participants"][0]["host"] and pres["rev"] == 3


def test_sse_generator_emits_pings_and_excludes_own_echo():
    from holographic.io_and_interop.holographic_livesession import LiveSession
    s = LiveSession("t"); s.bump("run1", "paint", touch=False); s.bump("run2", "extrude", touch=False)
    frames = list(A.sse_events(s, client="run1", user="moose", sleep=lambda t: None, ping_every=0.0, max_events=3))
    d = json.loads(frames[0][6:])
    assert d["rev"] == 2 and [c["src"] for c in d["changes"]] == ["run2"] and d["participants"] == ["moose"]
    assert frames[1].startswith(": ping") and frames[2].startswith(": ping")

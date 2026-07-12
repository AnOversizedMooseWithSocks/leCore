"""Tests for holographic_service (the standalone API service): route dispatch, SQL over the API, and a real HTTP
round-trip with the token gate."""
import json
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer

from holographic_service import Service, make_handler, __version__


def test_health_and_index():
    svc = Service()
    assert svc.dispatch("GET", "/health", {})[1]["ok"]
    idx = svc.dispatch("GET", "/", {})[1]
    assert idx["version"] == __version__ and "GET /health" in idx["endpoints"]


def test_capabilities_list_and_search():
    svc = Service()
    caps = svc.dispatch("GET", "/capabilities", {})[1]
    assert caps["ok"] and caps["count"] > 0
    hit = svc.dispatch("POST", "/capabilities/search", {"query": "network render farm another machine"})[1]
    assert any("farm" in m["name"].lower() or "network" in m["name"].lower() for m in hit["matches"])


def test_affected_tests_invokable_via_tool_protocol():
    """The affected-test selector (mind.affected_tests) must be a real /tools + /invoke citizen, not just an
    in-process method -- an agent driving leCore only ever sees it through this protocol."""
    svc = Service()
    tools = svc.dispatch("GET", "/tools", {})[1]["tools"]
    assert any(t["name"] == "affected_tests" for t in tools)
    status, body = svc.dispatch("POST", "/invoke",
                                {"name": "affected_tests", "args": {"changed_paths": ["README.md"]}})
    assert status == 200 and body["ok"] and body["result"] == []


def test_sql_crud_over_the_api():
    svc = Service()
    assert svc.dispatch("POST", "/sql", {"sql": "CREATE TABLE user.t (id, color)"})[1]["ok"]
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.t (id, color) VALUES (1, red)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.t (id, color) VALUES (2, blue)"})
    sel = svc.dispatch("POST", "/sql", {"sql": "SELECT id, color FROM user.t WHERE color = 'red'"})[1]
    assert sel["ok"] and len(sel["result"]) == 1 and sel["result"][0]["color"] == "red"


def test_error_statuses():
    svc = Service()
    assert svc.dispatch("POST", "/sql", {"sql": "SELECT nope FROM system.does_not_exist"})[0] == 400
    assert svc.dispatch("GET", "/nope", {})[0] == 404
    assert svc.dispatch("POST", "/sql", {})[0] == 400          # missing body


def _run_server(svc):
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(svc))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port


def _call(port, method, path, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_real_http_roundtrip():
    svc = Service()
    httpd, port = _run_server(svc)
    try:
        status, body = _call(port, "GET", "/health")
        assert status == 200 and body["ok"]
        _call(port, "POST", "/sql", {"sql": "CREATE TABLE user.h (id, name)"})
        _call(port, "POST", "/sql", {"sql": "INSERT INTO user.h (id, name) VALUES (1, alice)"})
        s, b = _call(port, "POST", "/sql", {"sql": "SELECT name FROM user.h WHERE id = 1"})
        assert s == 200 and b["result"][0]["name"] == "alice"
    finally:
        httpd.shutdown(); httpd.server_close()


def test_token_gate_over_http():
    svc = Service(token="secret")
    httpd, port = _run_server(svc)
    try:
        assert _call(port, "GET", "/health")[0] == 401                    # no token -> refused
        assert _call(port, "GET", "/health", token="secret")[0] == 200    # right token -> ok
        assert _call(port, "GET", "/health", token="wrong")[0] == 401     # wrong token -> refused
    finally:
        httpd.shutdown(); httpd.server_close()


# --- the full DB surface: UPDATE / DELETE / JOIN / DROP, GraphQL, and persistence -------------------------
def test_full_sql_surface():
    svc = Service()
    svc.dispatch("POST", "/sql", {"sql": "CREATE TABLE user.t (id, color)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.t (id, color) VALUES (1, red)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.t (id, color) VALUES (2, blue)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.t (id, color) VALUES (3, red)"})
    assert svc.dispatch("POST", "/sql", {"sql": "UPDATE user.t SET color = 'crimson' WHERE id = 1"})[1]["result"]["updated"] == 1
    assert svc.dispatch("POST", "/sql", {"sql": "DELETE FROM user.t WHERE color = 'blue'"})[1]["result"]["deleted"] == 1
    rows = svc.dispatch("POST", "/sql", {"sql": "SELECT id, color FROM user.t"})[1]["result"]
    assert {r["color"] for r in rows} == {"crimson", "red"}         # id1 crimson, id3 red, id2 (blue) deleted


def test_join_over_api():
    svc = Service()
    svc.dispatch("POST", "/sql", {"sql": "CREATE TABLE user.a (id, x)"})
    svc.dispatch("POST", "/sql", {"sql": "CREATE TABLE user.b (id, y)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.a (id, x) VALUES (1, A1)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.a (id, x) VALUES (2, A2)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.b (id, y) VALUES (1, B1)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.b (id, y) VALUES (2, B2)"})
    j = svc.dispatch("POST", "/sql", {"sql": "SELECT x, y FROM user.a JOIN user.b ON id WHERE id = 2"})[1]["result"]
    assert j == [{"x": "A2", "y": "B2"}]


def test_where_required_on_update_delete():
    svc = Service()
    svc.dispatch("POST", "/sql", {"sql": "CREATE TABLE user.t (id, c)"})
    assert svc.dispatch("POST", "/sql", {"sql": "UPDATE user.t SET c = 'x'"})[0] == 400
    assert svc.dispatch("POST", "/sql", {"sql": "DELETE FROM user.t"})[0] == 400


def test_graphql_over_documents():
    svc = Service()
    docs = [{"id": "o1", "name": "ring", "material": "gold", "transform": {"position": [1, 0, 0]}},
            {"id": "o2", "name": "coin", "material": "gold", "transform": {"position": [3, 0, 0]}},
            {"id": "o3", "name": "pipe", "material": "copper", "transform": {"position": [0, 2, 0]}}]
    assert svc.dispatch("POST", "/documents", {"objects": docs})[1]["count"] == 3
    gq = svc.dispatch("POST", "/graphql", {"query": '{ objects(where: {material: "gold"}) { name } }'})[1]
    assert [o["name"] for o in gq["data"]["objects"]] == ["ring", "coin"]
    # inline objects also work (stateless)
    gq2 = svc.dispatch("POST", "/graphql",
                       {"query": '{ objects(where: {material: "copper"}) { name } }', "objects": docs})[1]
    assert [o["name"] for o in gq2["data"]["objects"]] == ["pipe"]


def test_persistence_survives_restart(tmp_path):
    path = str(tmp_path / "store.json")
    svc = Service(persist_path=path)
    svc.dispatch("POST", "/sql", {"sql": "CREATE TABLE user.acct (id, bal)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.acct (id, bal) VALUES (1, 100)"})
    svc.dispatch("POST", "/sql", {"sql": "UPDATE user.acct SET bal = 250 WHERE id = 1"})
    svc.dispatch("POST", "/documents", {"objects": [{"id": "d1", "name": "keep"}]})
    # a fresh service (a restart) loads the same file
    reborn = Service(persist_path=path)
    rows = reborn.dispatch("POST", "/sql", {"sql": "SELECT id, bal FROM user.acct"})[1]["result"]
    assert rows[0]["bal"] == 250.0                                  # the UPDATE survived
    assert reborn.dispatch("GET", "/documents", {})[1]["count"] == 1


def test_explicit_save_load(tmp_path):
    path = str(tmp_path / "explicit.json")
    svc = Service()
    svc.dispatch("POST", "/sql", {"sql": "CREATE TABLE user.t (id, c)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.t (id, c) VALUES (7, saved)"})
    assert svc.dispatch("POST", "/save", {"path": path})[1]["ok"]
    fresh = Service()
    assert fresh.dispatch("POST", "/load", {"path": path})[1]["ok"]
    assert fresh.dispatch("POST", "/sql", {"sql": "SELECT c FROM user.t WHERE id = 7"})[1]["result"][0]["c"] == "saved"


# --- job lifecycle over the service ------------------------------------------------------------------------
def test_jobs_create_start_result():
    import time
    svc = Service()
    svc.dispatch("POST", "/jobs/create", {"id": "j", "buckets": [[i] for i in range(10)], "worker": "sum"})
    svc.dispatch("POST", "/jobs/start", {"id": "j"})
    svc._jobs.wait("j")
    res = svc.dispatch("POST", "/jobs/result", {"id": "j"})[1]
    assert res["status"] == "done" and res["result"] == 45.0


def test_jobs_pause_resume():
    import time
    svc = Service()
    svc.dispatch("POST", "/jobs/create", {"id": "r", "buckets": [[i] for i in range(20)], "worker": "sum_slow"})
    svc.dispatch("POST", "/jobs/start", {"id": "r", "batch": 1})
    time.sleep(0.05)
    p = svc.dispatch("POST", "/jobs/pause", {"id": "r"})[1]["job"]
    assert p["status"] == "paused" and 0 < p["done"] < 20
    svc.dispatch("POST", "/jobs/resume", {"id": "r"})
    svc._jobs.wait("r")
    assert svc.dispatch("POST", "/jobs/result", {"id": "r"})[1]["result"] == float(sum(range(20)))


def test_jobs_cancel_and_list():
    import time
    svc = Service()
    svc.dispatch("POST", "/jobs/create", {"id": "c", "buckets": [[i] for i in range(20)], "worker": "sum_slow"})
    svc.dispatch("POST", "/jobs/start", {"id": "c", "batch": 1})
    time.sleep(0.03)
    svc.dispatch("POST", "/jobs/cancel", {"id": "c"})
    assert svc.dispatch("POST", "/jobs/status", {"id": "c"})[1]["job"]["status"] == "cancelled"
    assert any(j["id"] == "c" for j in svc.dispatch("GET", "/jobs", {})[1]["jobs"])


def test_jobs_survive_restart(tmp_path):
    import time
    path = str(tmp_path / "store.json")
    svc = Service(persist_path=path)
    svc.dispatch("POST", "/jobs/create", {"id": "big", "buckets": [[i] for i in range(30)], "worker": "sum_slow"})
    svc.dispatch("POST", "/jobs/start", {"id": "big", "batch": 1})
    time.sleep(0.1)
    svc.dispatch("POST", "/jobs/pause", {"id": "big"})
    done_at_pause = svc.dispatch("POST", "/jobs/status", {"id": "big"})[1]["job"]["done"]
    assert 0 < done_at_pause < 30

    reborn = Service(persist_path=path)                             # a fresh app session loads checkpointed jobs
    jobs = reborn.dispatch("GET", "/jobs", {})[1]["jobs"]
    assert any(j["id"] == "big" and j["status"] == "paused" for j in jobs)
    reborn.dispatch("POST", "/jobs/resume", {"id": "big"})
    reborn._jobs.wait("big")
    assert reborn.dispatch("POST", "/jobs/result", {"id": "big"})[1]["result"] == float(sum(range(30)))


def test_jobs_bad_requests():
    svc = Service()
    assert svc.dispatch("POST", "/jobs/create", {"id": "x"})[0] == 400          # missing buckets/worker
    assert svc.dispatch("POST", "/jobs/status", {"id": "nope"})[0] == 400       # unknown job


def test_game_room_deterministic_replay():
    # /game is the game's HTTP face: the same POST sequence must replay to the same world digest
    # (the interaction layer inherits the engine's determinism constitution over the wire too).
    def play(svc):
        svc.dispatch("POST", "/game", {"world": "t", "create": {"cell": 4.0, "dt": 0.1},
                                        "cmds": [{"op": "spawn", "id": 1, "pos": [3.5, 1, 1], "vel": [2, 0, 0]},
                                                 {"op": "spawn", "id": 2, "pos": [1.0, 2.5, 1]}]})
        _, r = svc.dispatch("POST", "/game", {"world": "t", "ticks": 8,
                                               "aoi": {"center": [3, 1, 1], "radius": 8}})
        return r
    a, b = play(Service()), play(Service())
    assert a["digest"] == b["digest"] and a["migrated"] == b["migrated"] == [1]
    assert sorted(a["aoi"]["ids"]) == [1, 2]          # AOI spans the shard seam over the wire


def test_game_stream_deltas_first_full_then_changes():
    # The SSE generator's payload contract, tested at the WorldStreamer seam the handler uses:
    # first event = full AOI as 'added'; second = only what changed.
    svc = Service()
    svc.dispatch("POST", "/game", {"world": "s", "create": {"cell": 8.0, "dt": 0.1},
                                    "cmds": [{"op": "spawn", "id": 1, "pos": [1, 1, 1], "vel": [2, 0, 0]},
                                             {"op": "spawn", "id": 2, "pos": [1.5, 2.5, 1]}]})
    world, streamer = svc._game_room("s")
    world.tick()
    e1 = streamer.next_event("c1", center=(1.5, 1, 1), radius=6.0)
    assert sorted(x["id"] for x in e1["added"]) == [1, 2] and not e1["moved"]
    world.tick()
    e2 = streamer.next_event("c1", center=(1.5, 1, 1), radius=6.0)
    assert [x["id"] for x in e2["moved"]] == [1] and not e2["added"] and not e2["removed"]

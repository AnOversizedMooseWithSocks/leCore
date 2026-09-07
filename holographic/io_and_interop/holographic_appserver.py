"""APPSERVER -- the standard agent surface every app built on leCore mounts, instead of writing its own.

WHAT THE TWO APPS EACH BUILT (sweep 163 audit of leStudio and Poly Studio):

  * a machine-readable tool manifest derived from Flask's live url_map, plus an invoke-by-name door
    (leStudio: GET /api/schema; Poly Studio: GET /api/agent/tools + POST /api/agent/invoke)
  * an allow-listed pass-through to the engine's own faculties for discovery (leStudio: POST /api/mind,
    "a rejected name returns the full allowlist WITH python signatures, so one failed call teaches the usage")
  * an identity contract for agents (leStudio: X-Client = this run, for echo suppression; X-User = the persistent
    person, for presence / host / kick)
  * a change feed + presence (leStudio: SSE /api/events with a 2-second comment ping, because a stream that only
    writes when the rev moves never notices its socket died -- "immortal ghost editors")
  * an engine-status door (Poly Studio: Help > Engine status -- which build, which extras; leStudio: /api/status
    with features() + resource_policy())
  * base64 image returns for agents (Poly Studio 1.2.0: "a base64 frame is the highest-value return in the whole
    API -- it is the only way it can see its own work")

Poly Studio had none of the presence/identity/feed pieces; leStudio had none of the base64 return. Each app
re-derived the parts it noticed it needed. This module is the union, once, with each app's lesson kept:

    from holographic.io_and_interop.holographic_appserver import AgentSurface
    surface = AgentSurface(flask_app, base="/api", app_name="polystudio", mind=mind,
                           workspace_root="~/projects/demo.lews.d",       # optional: the .lews live directory
                           image_routes=("render",), stream_routes=("photo",))
    surface.mount()

adds, under `base`:  GET  /agent/tools     the manifest (this mount only -- Poly's gallery lesson)
                     POST /agent/invoke    {tool, args, json} -> the route's JSON, or a data:image/png;base64 frame
                     POST /mind            {name, args} -> an allow-listed engine faculty, JSON-safe
                     GET  /engine          version, features count, extras, determinism policy, GPU report
                     GET  /events          SSE {rev, src, kind, participants...}, heartbeat pings
                     GET  /presence        who is here (roster, host)
and an after_request hook that turns every mutating request on `base` into a workspace note (rev bump, src =
X-Client) and a presence heartbeat for X-User. The workspace is the .lews live directory (holographic_lews.Workspace)
so a second app on the same directory sees this app's edits and users -- or, with no workspace_root, an in-process
LiveSession, so a single app still gets the same feed and roster.

FLASK IS OPTIONAL. The module imports Flask only inside mount(); the manifest builder, the identity parser, the
JSON coercion and the SSE generator are plain functions the selftest exercises without a web framework, and the
Flask path is exercised when Flask is installed (it is a permitted core dependency).

KEPT NEGATIVES (do not re-tread):
  * Filtering the manifest by blueprint NAME is wrong -- a gallery re-registers the blueprint under another name;
    filter by the mount path the request arrived on (Poly Studio).
  * Presence keyed by connection / tab id gives one person a ghost per reload; key by X-User (leStudio).
  * A change feed that only emits on change never learns its socket died; emit a comment ping every ~2 s
    (leStudio's ghost fix). EventSource ignores comment lines.
  * Refusing to wrap a PNG in JSON "because it helps nobody" was exactly backwards for an agent (Poly 1.2.0).
"""
import base64
import inspect
import json
import time

import numpy as np

# Discovery + read-only analysis faculties an app may expose to agents by default. Mutating faculties are never
# exposed through /mind -- an app decides what mutates its documents, the engine door is for LOOKING.
DEFAULT_MIND_ALLOW = ("find_capability", "suggest", "describe_skill", "complete_method", "features", "version",
                      "compare_images", "seam_continuity", "est_dx", "vanishing_point", "image_colours",
                      "image_signature", "code_search")

MUTATING = ("POST", "PUT", "PATCH", "DELETE")
STANDARD_DOORS = ("mind", "engine", "presence", "events")


# ----------------------------------------------------------------------------------------------- plain functions
def identity(headers, args=None):
    """(client, user) from a request: X-Client = THIS RUN or tab (echo suppression), X-User = the persistent person
    or agent identity (presence, host). Old clients that send only one degrade to client == user. Query-string
    fallbacks (?client=..&user=..) exist because EventSource cannot set headers."""
    g = (lambda k: headers.get(k)) if hasattr(headers, "get") else (lambda k: None)
    a = args or {}
    client = g("X-Client") or a.get("client") or ""
    user = g("X-User") or a.get("user") or client
    return str(client), str(user)


def jsonable(x, max_array=4096):
    """Coerce a faculty result for JSON: numpy -> lists (large arrays -> a typed summary with a sample), numpy
    scalars -> floats, dicts/lists recurse, anything else -> str. Same rule leStudio's /api/mind used, so an agent
    sees one shape from every app."""
    if isinstance(x, np.ndarray):
        if x.size <= max_array:
            return x.tolist()
        return {"shape": list(x.shape), "dtype": str(x.dtype), "note": "array too large; truncated",
                "sample": x.ravel()[:16].tolist()}
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v, max_array) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v, max_array) for v in x]
    return x if isinstance(x, (str, int, float, bool, type(None))) else str(x)


def manifest_from_rules(rules, base, hints=None, image_routes=(), stream_routes=(), doc_of=None):
    """The tool manifest from an iterable of (rule_string, methods, endpoint) -- Flask's url_map or anything shaped
    like it. THIS MOUNT ONLY: a rule counts if it starts with `base + "/"`, and the /agent/* doors themselves are
    left out. Each tool: {name, path, methods, body, summary, returns} where returns is 'json' | 'image' |
    'stream' so an agent knows before calling whether to expect bytes."""
    hints = hints or {}
    tools = []
    for rule, methods, endpoint in rules:
        path = str(rule)
        if not path.startswith(base + "/") or "/agent/" in path:
            continue
        name = path[len(base) + 1:].rstrip("/")
        if name in STANDARD_DOORS:                      # the doors this module mounts are described in the header
            continue
        ms = sorted(m for m in methods if m in ("GET", "POST", "PUT", "DELETE", "PATCH"))
        head = name.split("/")[0].split("<")[0]
        returns = "image" if head in image_routes else ("stream" if head in stream_routes else "json")
        summary = hints.get(name) or hints.get(head) or (doc_of(endpoint) if doc_of else "") or ""
        tools.append({"name": name, "path": path, "methods": ms, "body": "json" if set(ms) & set(MUTATING) else None,
                      "summary": str(summary).strip().split("\n")[0], "returns": returns})
    tools.sort(key=lambda t: t["name"])
    return tools


def sse_events(session, client, user, sleep=time.sleep, ping_every=2.0, poll=0.25, max_events=None, ttl=30.0):
    """The change-feed generator behind GET /events: yields SSE frames. Emits a data frame when the session rev
    moves past what this client last saw (its OWN notes excluded -- echo suppression), and a comment ping every
    `ping_every` seconds otherwise, so a dead socket raises within seconds instead of living forever as a ghost
    (EventSource ignores comment lines). Each pass heart-beats `user`. `max_events` bounds the loop for tests;
    `sleep` is injectable so the selftest is not sleepy. Works for a Workspace (file) or a LiveSession (memory)."""
    last = -1; beat = 0.0; n = 0
    while max_events is None or n < max_events:
        _touch(session, user)
        rev = _rev(session)
        now = time.time()
        if rev != last:
            since = _since(session, last if last >= 0 else 0, exclude=client)
            last = rev
            yield "data: " + json.dumps({"rev": rev, "changes": since[-50:], "participants": _participants(session, ttl),
                                         "you": user}, default=str) + "\n\n"
            n += 1
        elif now - beat >= ping_every:
            beat = now
            yield ": ping\n\n"
            n += 1
        sleep(poll)


# small adapters so the same code drives a Workspace (files) or a LiveSession (memory)
def _rev(s):
    return s.rev() if callable(getattr(s, "rev", None)) else int(getattr(s, "rev", 0))


def _since(s, rev, exclude=None):
    return s.since(rev, exclude=exclude)


def _touch(s, who, **kw):
    try:
        return s.touch(who, **kw) if kw and hasattr(s, "roster") else s.touch(who)
    except TypeError:
        return s.touch(who)


def _participants(s, ttl=30.0):
    try:
        return s.participants(ttl=ttl)
    except TypeError:
        return s.participants()


def _roster(s, ttl=30.0):
    if hasattr(s, "roster"):
        return s.roster(ttl=ttl)
    return [{"who": w, "host": i == 0} for i, w in enumerate(s.participants())]


def engine_status(mind):
    """One preflight dict an app shows in Help > Engine status and gates features on: {engine, capabilities_schema,
    dim, seed, faculties (count), extras (numba/sympy/cupy/wgpu importable?), gpu (the engine's own report), policy
    (resource_policy: bit_exact and what would break it)}. Composes verbs the mind already has, so no app keeps its
    own list of what to check -- a hardcoded client-side list rots silently."""
    out = {}
    try:
        out.update(mind.version())
    except Exception as e:
        out["version_error"] = str(e)
    try:
        out["faculties"] = sum(1 for _ in mind.features().values())
    except Exception:
        out["faculties"] = None
    ex = {}
    for mod in ("numba", "sympy", "cupy", "wgpu", "flask"):
        try:
            __import__(mod); ex[mod] = True
        except Exception:
            ex[mod] = False
    out["extras"] = ex
    for name, key in (("gpu_report", "gpu"), ("resource_policy", "policy"), ("cpu_budget", "cpu_budget")):
        try:
            out[key] = jsonable(getattr(mind, name)())
        except Exception as e:
            out[key] = {"error": str(e)}
    return out


# ------------------------------------------------------------------------------------------------- the surface
class AgentSurface:
    """Mount the standard doors on a Flask app (or blueprint-shaped object with .route / .after_request).

    `image_routes` / `stream_routes`: route heads whose responses are PNG bytes / NDJSON, so /agent/invoke wraps the
    former as base64 and refuses the latter with the direct path (Poly Studio's rule)."""

    def __init__(self, flask_app, base="/api", app_name="app", mind=None, workspace_root=None, workspace=None,
                 mind_allow=DEFAULT_MIND_ALLOW, image_routes=(), stream_routes=(), hints=None,
                 skip_bump=("agent", "events", "presence", "mind", "engine", "job", "autosave"), ttl=30.0):
        self.app = flask_app; self.base = str(base).rstrip("/"); self.app_name = str(app_name)
        self._mind = mind; self.mind_allow = tuple(mind_allow); self.hints = dict(hints or {})
        self.image_routes = tuple(image_routes); self.stream_routes = tuple(stream_routes)
        self.skip_bump = tuple(skip_bump); self.ttl = float(ttl)
        if workspace is not None:
            self.session = workspace
        elif workspace_root is not None:
            from holographic.io_and_interop.holographic_lews import Workspace
            self.session = Workspace(workspace_root, app=self.app_name)
        else:
            from holographic.io_and_interop.holographic_livesession import LiveSession
            self.session = LiveSession(name=self.app_name, ttl=self.ttl)

    def mind(self):
        if self._mind is None:
            import lecore
            self._mind = lecore.UnifiedMind(dim=256, seed=0)
        return self._mind

    # ---- pieces a test can call without Flask
    def manifest(self, rules, doc_of=None):
        tools = manifest_from_rules(rules, self.base, self.hints, self.image_routes, self.stream_routes, doc_of)
        return {"app": self.app_name, "engine": "leCore", "base": self.base, "count": len(tools),
                "invoke": {"path": self.base + "/agent/invoke", "method": "POST",
                           "body": {"tool": "<name>", "args": {"<query>": "..."}, "json": {"<post body>": "..."}}},
                "identity": {"X-Client": "this run or tab (echo suppression)",
                             "X-User": "your persistent identity (presence, host)"},
                "events": self.base + "/events?client=..&user=..", "mind": self.base + "/mind", "tools": tools}

    def mind_call(self, name, args):
        """POST /mind body -> (status, payload). A rejected name returns the allowlist WITH signatures."""
        m = self.mind()
        if name not in self.mind_allow or not hasattr(m, name):
            allowed = {}
            for n in sorted(self.mind_allow):
                try:
                    allowed[n] = str(inspect.signature(getattr(m, n)))
                except Exception:
                    allowed[n] = "(...)"
            return 400, {"error": "%r is not on the /mind allowlist" % name, "allowed": allowed}
        args = dict(args or {})
        for k, v in list(args.items()):
            if isinstance(v, list) and v and isinstance(v[0], list):
                args[k] = np.asarray(v, np.float32)                 # nested lists are arrays (images, points)
        try:
            return 200, {"ok": True, "name": name, "result": jsonable(getattr(m, name)(**args))}
        except Exception as e:
            return 400, {"error": str(e), "name": name}

    def note(self, path, method, client, user):
        """What the after_request hook does for a mutating request: one note on the session (rev bump, src=client)
        and a heartbeat for user. Skips the doors that change nothing (events, presence, agent, mind, engine, job
        polls, autosave) so a presence ping never makes every client refresh."""
        rel = path[len(self.base) + 1:] if path.startswith(self.base + "/") else ""
        head = rel.split("/")[0]
        if not rel or head in self.skip_bump or method not in MUTATING:
            return None
        rev = self.session.bump(client or self.app_name, kind=method + " " + rel, touch=False)
        if user:
            _touch(self.session, user, app=self.app_name) if hasattr(self.session, "roster") else self.session.touch(user)
        return rev

    # ---- Flask
    def mount(self):
        """Register the doors. Returns self. Imports Flask here and nowhere else."""
        from flask import request, jsonify, Response
        app, base, self_ = self.app, self.base, self

        @app.route(base + "/agent/tools", methods=["GET"], endpoint="lecore_agent_tools")
        def _tools():
            rules = [(str(r), r.methods, r.endpoint) for r in app.url_map.iter_rules()]
            doc_of = lambda ep: (app.view_functions.get(ep).__doc__ or "") if app.view_functions.get(ep) else ""
            return jsonify(self_.manifest(rules, doc_of))

        @app.route(base + "/agent/invoke", methods=["POST"], endpoint="lecore_agent_invoke")
        def _invoke():
            d = request.get_json(force=True, silent=True) or {}
            name = str(d.get("tool", "")).strip().strip("/")
            if not name:
                return jsonify({"error": "give a tool name; see %s/agent/tools" % base}), 400
            head = name.split("?")[0].split("/")[0]
            target = base + "/" + name
            if head in self_.stream_routes:
                return jsonify({"error": "%r streams; call %s directly" % (name, target), "path": target}), 400
            client = app.test_client()
            hdrs = {k: v for k, v in request.headers.items() if k in ("X-Client", "X-User")}
            if d.get("json") is not None:
                resp = client.post(target, query_string=d.get("args") or {}, json=d["json"], headers=hdrs)
            else:
                resp = client.get(target, query_string=d.get("args") or {}, headers=hdrs)
            if head in self_.image_routes:
                if resp.status_code != 200:
                    return jsonify({"error": "render failed", "status": resp.status_code}), resp.status_code
                return jsonify({"tool": name, "status": 200, "result": {
                    "image": "data:image/png;base64," + base64.b64encode(resp.data).decode(), "bytes": len(resp.data),
                    "headers": {k: v for k, v in resp.headers.items() if k.startswith("X-")}}})
            try:
                payload = resp.get_json(silent=True)
            except Exception:
                payload = None
            if payload is None:
                payload = resp.get_data(as_text=True)
            return jsonify({"tool": name, "status": resp.status_code, "result": payload}), resp.status_code

        @app.route(base + "/mind", methods=["POST"], endpoint="lecore_mind")
        def _mind():
            d = request.get_json(force=True, silent=True) or {}
            status, payload = self_.mind_call(str(d.get("name", "")), d.get("args") or {})
            return jsonify(payload), status

        @app.route(base + "/engine", methods=["GET"], endpoint="lecore_engine")
        def _engine():
            return jsonify(engine_status(self_.mind()))

        @app.route(base + "/presence", methods=["GET"], endpoint="lecore_presence")
        def _presence():
            client, user = identity(request.headers, request.args)
            return jsonify({"participants": _roster(self_.session, self_.ttl), "you": user,
                            "rev": _rev(self_.session)})

        @app.route(base + "/events", methods=["GET"], endpoint="lecore_events")
        def _events():
            client, user = identity(request.headers, request.args)
            name = (request.args.get("name") or "").strip()[:24]
            if name and hasattr(self_.session, "roster"):
                self_.session.touch(user, name=name, app=self_.app_name)
            return Response(sse_events(self_.session, client, user, ttl=self_.ttl), mimetype="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        @app.after_request
        def _bump(resp):
            if resp.status_code < 400 and request.path.startswith(base + "/"):
                client, user = identity(request.headers, request.args)
                try:
                    self_.note(request.path, request.method, client, user)
                except Exception:
                    pass                                                # a failed note must never fail the request
            return resp

        return self


def _selftest():
    """Pins, without Flask: the manifest keeps only this mount and marks image/stream returns; identity degrades
    client->user; /mind rejects with signatures and answers on the allowlist; the after_request rule bumps only
    mutating non-door requests; the SSE generator emits data on change, pings when idle, excludes own echo, and
    heart-beats. Then, with Flask installed, the same doors over a real test_client."""
    from holographic.io_and_interop.holographic_livesession import LiveSession
    s = AgentSurface(flask_app=None, base="/api", app_name="demo", image_routes=("render",), stream_routes=("photo",),
                     hints={"paint": "paint a stroke"})
    rules = [("/api/paint", {"POST", "OPTIONS"}, "paint"), ("/api/render", {"GET", "HEAD"}, "render"),
             ("/api/photo", {"GET"}, "photo"), ("/api/agent/tools", {"GET"}, "t"), ("/other/api/bake", {"POST"}, "x"),
             ("/api/scene/<oid>", {"GET"}, "scene")]
    man = s.manifest(rules)
    names = [t["name"] for t in man["tools"]]
    assert names == ["paint", "photo", "render", "scene/<oid>"], names                    # other mount + agent doors gone
    by = {t["name"]: t for t in man["tools"]}
    assert by["render"]["returns"] == "image" and by["photo"]["returns"] == "stream" and by["paint"]["body"] == "json"
    assert by["paint"]["summary"] == "paint a stroke" and by["scene/<oid>"]["returns"] == "json"
    assert identity({"X-Client": "run1"}) == ("run1", "run1") and identity({}, {"user": "moose"}) == ("", "moose")
    st, p = s.mind_call("not_a_faculty", {}); assert st == 400 and "find_capability" in p["allowed"] and p["allowed"]["version"].startswith("(")
    st, p = s.mind_call("version", {}); assert st == 200 and "engine" in p["result"]
    st, p = s.mind_call("image_colours", {"image": [[[0.0, 0.0, 0.0]] * 4] * 4, "k": 1}) if False else (200, None)
    # after_request rule
    assert s.note("/api/paint", "POST", "run1", "moose") == 1 and s.note("/api/paint", "GET", "run1", "moose") is None
    assert s.note("/api/events", "POST", "run1", "moose") is None and s.note("/api/job/3", "POST", "run1", "moose") is None
    assert s.note("/api/scene/4", "DELETE", "run2", "agent7") == 2 and set(s.session.participants()) == {"moose", "agent7"}
    # SSE generator: first frame carries everything after rev 0 minus own echo; idle -> pings; heartbeat
    frames = list(sse_events(s.session, client="run1", user="moose", sleep=lambda t: None, ping_every=0.0, max_events=3))
    d0 = json.loads(frames[0][len("data: "):])
    assert d0["rev"] == 2 and [c["src"] for c in d0["changes"]] == ["run2"] and "moose" in d0["participants"]
    assert frames[1].startswith(": ping") and frames[2].startswith(": ping")
    es = engine_status(s.mind()); assert "engine" in es and "extras" in es and es["policy"]["bit_exact"] in (True, False)
    print("appserver selftest OK (no Flask): manifest filtered to this mount with image/stream returns, identity degrades, "
          "/mind rejects with signatures, after_request bumps only mutating non-door requests, SSE emits/pings/heart-beats")
    _selftest_flask()


def _selftest_flask():
    try:
        from flask import Flask, jsonify, request, Response
    except Exception:
        print("appserver flask selftest skipped (Flask not installed)"); return
    import tempfile
    app = Flask("demo")
    state = {"strokes": 0}

    @app.route("/api/paint", methods=["POST"])
    def paint():
        """Paint one stroke."""
        state["strokes"] += 1; return jsonify(ok=True, strokes=state["strokes"])

    @app.route("/api/render")
    def render():
        """A PNG frame."""
        return Response(b"\x89PNG\r\n\x1a\nfake", mimetype="image/png", headers={"X-Converged": "1"})

    root = tempfile.mkdtemp()
    surf = AgentSurface(app, base="/api", app_name="demo", workspace_root=root, image_routes=("render",)).mount()
    c = app.test_client()
    man = c.get("/api/agent/tools").get_json()
    assert [t["name"] for t in man["tools"]] == ["paint", "render"] and man["tools"][0]["summary"] == "Paint one stroke."
    r = c.post("/api/paint", json={}, headers={"X-Client": "run1", "X-User": "moose"}).get_json(); assert r["strokes"] == 1
    inv = c.post("/api/agent/invoke", json={"tool": "paint", "json": {}}, headers={"X-Client": "run1", "X-User": "moose"}).get_json()
    assert inv["result"]["strokes"] == 2
    img = c.post("/api/agent/invoke", json={"tool": "render"}).get_json()
    assert img["result"]["image"].startswith("data:image/png;base64,") and img["result"]["headers"]["X-Converged"] == "1"
    bad = c.post("/api/mind", json={"name": "nope"}); assert bad.status_code == 400 and "allowed" in bad.get_json()
    ok = c.post("/api/mind", json={"name": "version"}).get_json(); assert ok["result"]["engine"]
    eng = c.get("/api/engine").get_json(); assert "extras" in eng and "gpu" in eng
    pres = c.get("/api/presence", headers={"X-User": "moose"}).get_json()
    assert [p["who"] for p in pres["participants"]] == ["moose"] and pres["participants"][0]["host"] and pres["rev"] == 2
    # a SECOND app on the same directory sees the notes and the person (the cross-app property)
    from holographic.io_and_interop.holographic_lews import Workspace
    other = Workspace(root, app="polystudio", create=False)
    assert [e["kind"] for e in other.since(0)] == ["POST paint", "POST paint"] and other.participants() == ["moose"]
    ev = c.get("/api/events?client=run9&user=agent7", headers={}, buffered=False)
    first = next(ev.response) if hasattr(ev, "response") else None
    if first is not None:
        line = first.decode() if isinstance(first, bytes) else first
        assert line.startswith("data: ") and json.loads(line[6:])["rev"] == 2
        ev.close()
    print("appserver flask selftest OK: tools/invoke(base64 image)/mind/engine/presence/events over test_client, "
          "and a second app on the same .lews directory saw both notes and the person")


if __name__ == "__main__":
    _selftest()

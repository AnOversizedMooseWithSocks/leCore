"""holographic_service.py -- leCore as a STANDALONE API service. Start it on any OS; talk to it over HTTP/JSON.

WHY
---
"Run the app standalone and talk to it via an API" -- so any client (a browser, curl, another program, in any
language) can drive the engine over the network. This is deliberately STDLIB ONLY (http.server + json); the only real
dependency is numpy, which the engine needs anyway. That means it runs on any Python 3 with essentially zero setup --
the launcher scripts (serve.sh / serve.bat) just find Python and start this.

THE API (JSON in, JSON out)
  GET  /                      -- a self-describing index of the endpoints.
  GET  /health               -- {ok, name, version, python, capabilities} -- a liveness + version probe.
  GET  /capabilities         -- every capability the running instance advertises (name + description).
  POST /capabilities/search  -- {"query": "..."} -> the capability homes that match, plain-English search.
  POST /sql                  -- {"sql": "..."} -> run a SQL statement against the service's VSA Database
                                (CREATE TABLE / INSERT / SELECT ... -- the whole query layer, over HTTP).

Design: a tiny ROUTE REGISTRY (method, path) -> handler, so adding an endpoint is one line and the whole surface reads
top to bottom. Extend `Service._register` to expose more faculties.

SECURITY (kept honest, same spirit as the farm)
  * Binds to 127.0.0.1 by DEFAULT (local only). Pass --host 0.0.0.0 to expose on the network ONLY behind auth/TLS on
    a trusted network -- exposing a compute+SQL endpoint openly is a real risk.
  * Optional --token: if set, every request must carry `Authorization: Bearer <token>` or it is refused (401). A
    minimal shared-secret gate -- not a substitute for TLS across the internet.
  * The SQL surface is the hand-rolled subset (no string-concatenated SQL), so classic injection is N/A; but a caller
    can still create/insert/select freely -- treat the endpoint as trusted-client only unless you add per-route auth.
"""
import json
import platform
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from holographic.agents_and_reasoning.holographic_query import Database, run_db_sql, QueryError

__version__ = "1.0"


class Service:
    """Holds the standalone app's state (a VSA Database + the capability catalog) and the route table. One instance
    per running server."""

    def __init__(self, token=None, persist_path=None, mind=None):
        self.token = token
        self.persist_path = persist_path
        self._mind = mind                                   # the tool-interface mind (lazily built if left None)                    # if set: auto-load on start, auto-save after writes
        self.db = Database()
        self.db.add_namespace("user")                       # a ready-to-use writable namespace for SQL clients
        self.documents = []                                 # nested-object store for the GraphQL front door
        self._routes = {}                                   # (method, path) -> handler(payload) -> dict
        self._jobs = self._make_job_manager()               # long-running job control (start/pause/resume/cancel)
        from holographic.misc.holographic_bus import MessageBus
        self._bus = MessageBus()                            # the message bus: person + agent both connected (push, no poll)
        self._frames = None                                 # lazily-built FrameServer (per-session quality control)
        self._games = {}                                    # world_id -> (ShardWorld, WorldStreamer): the live game rooms
        import threading as _threading
        self._game_lock = _threading.Lock()                 # one clock writer at a time (stream tick vs POSTed commands)
        self._register()
        if persist_path:
            self._load_from_disk()                          # restore a previous session's data if the file exists

    def _make_job_manager(self):
        """A JobManager over a local process pool, checkpointing beside the persist file (so paused jobs survive a
        restart). Registers a demo 'sum' worker; a real app registers its own trusted workers here by name."""
        from holographic.scene_and_pipeline.holographic_jobs import JobManager, _sum_bucket, _slow_sum
        from holographic.scene_and_pipeline.holographic_coordinator import InProcessBackend
        jobs_dir = (self.persist_path + ".jobs") if self.persist_path else None
        mgr = JobManager(InProcessBackend(), store_dir=jobs_dir)
        mgr.register_worker("sum", _sum_bucket)             # a built-in demo worker; apps add their own
        mgr.register_worker("sum_slow", _slow_sum)          # a slower demo worker (each bucket ~20ms) for pause demos
        if jobs_dir:
            mgr.load_all()                                  # bring back any paused/checkpointed jobs on startup
        return mgr

    # ---- the route table (extend here to expose more faculties) --------------------------------------------
    def _register(self):
        self._routes[("GET", "/")] = self._index
        self._routes[("GET", "/health")] = self._health
        self._routes[("GET", "/capabilities")] = self._capabilities
        self._routes[("POST", "/capabilities/search")] = self._capabilities_search
        self._routes[("GET", "/tools")] = self._tools           # the standard tool manifest (name, description, params)
        self._routes[("POST", "/invoke")] = self._invoke         # run one faculty: {name, args} -> its result
        self._routes[("POST", "/frame")] = self._frame           # real-time frame serving: adaptive quality per session
        self._routes[("POST", "/game")] = self._game             # game rooms: create / submit commands / tick / snapshot
        self._routes[("GET", "/game/stream")] = self._game_stream_doc  # SSE push: per-client world deltas
        self._routes[("POST", "/pick")] = self._pick             # viewport picking: screen coord -> vert/edge/face
        self._routes[("GET", "/frame/stream")] = self._frame_stream_doc  # SSE push: stream frames at a target rate
        self._routes[("POST", "/sql")] = self._sql
        self._routes[("POST", "/graphql")] = self._graphql       # GraphQL over nested documents
        self._routes[("POST", "/documents")] = self._set_documents
        self._routes[("GET", "/documents")] = self._get_documents
        self._routes[("POST", "/save")] = self._save             # persist to disk
        self._routes[("POST", "/load")] = self._load             # restore from disk
        self._routes[("GET", "/jobs")] = self._jobs_list         # long-running job control
        self._routes[("POST", "/jobs/create")] = self._jobs_create
        self._routes[("POST", "/jobs/start")] = self._jobs_start
        self._routes[("POST", "/jobs/pause")] = self._jobs_pause
        self._routes[("POST", "/jobs/resume")] = self._jobs_resume
        self._routes[("POST", "/jobs/cancel")] = self._jobs_cancel
        self._routes[("POST", "/jobs/status")] = self._jobs_status
        self._routes[("POST", "/jobs/result")] = self._jobs_result
        # message bus: a remote person/agent publishes, polls its inbox, and reads history (see holographic_bus)
        self._routes[("POST", "/bus/publish")] = self._bus_publish
        self._routes[("POST", "/bus/poll")] = self._bus_poll
        self._routes[("POST", "/bus/history")] = self._bus_history
        self._routes[("GET", "/skills")] = self._skills_manifest       # agent-friendly discovery + invocation
        self._routes[("POST", "/skills/suggest")] = self._skills_suggest
        self._routes[("POST", "/skills/route")] = self._skills_route
        self._routes[("POST", "/skills/complete")] = self._skills_complete
        self._routes[("POST", "/skills/card")] = self._skills_card

    def dispatch(self, method, path, payload):
        """Route a request to its handler; a QueryError becomes a clean 400-style error, anything else a 500-style."""
        handler = self._routes.get((method, path))
        if handler is None:
            return 404, {"ok": False, "error": "no such endpoint: %s %s" % (method, path)}
        try:
            return 200, handler(payload)
        except QueryError as e:                             # expected, user-facing (bad SQL, unknown column, ...)
            return 400, {"ok": False, "error": str(e)}
        except Exception as e:                              # unexpected -- report the type, don't leak a traceback
            return 500, {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}

    # ---- the handlers --------------------------------------------------------------------------------------
    def _index(self, _payload):
        """The self-describing index of the API. Body: none. Returns: {ok, name, version, endpoints} (every registered METHOD /path)."""
        return {"ok": True, "name": "leCore", "version": __version__,
                "endpoints": {"%s %s" % (m, p): "" for (m, p) in sorted(self._routes)}}

    def _health(self, _payload):
        """Liveness + version probe. Body: none. Returns: {ok, name, version, python, platform, capabilities}."""
        from holographic.caching_and_storage.holographic_catalog import default_catalog
        return {"ok": True, "name": "leCore", "version": __version__,
                "python": platform.python_version(), "platform": platform.system(),
                "capabilities": len(default_catalog())}

    def _capabilities(self, _payload):
        """Every capability the running instance advertises. Body: none. Returns: {ok, count, capabilities:[{name, description}]}."""
        from holographic.caching_and_storage.holographic_catalog import default_catalog
        caps = default_catalog().all()
        return {"ok": True, "count": len(caps),
                "capabilities": [{"name": c.name, "description": c.does} for c in caps]}

    def _capabilities_search(self, payload):
        """The capability homes matching a plain-English query. Body: {query}. Returns: {ok, query, matches}."""
        from holographic.caching_and_storage.holographic_catalog import default_catalog
        query = (payload or {}).get("query", "")
        hits = default_catalog().find_capability(query)
        return {"ok": True, "query": query,
                "matches": [{"name": c.name, "description": c.does} for c in hits]}

    # ---- the tool interface: this node AS a tool (GET /tools + POST /invoke) --------------------------------
    @property
    def mind(self):
        """The UnifiedMind whose faculties /tools advertises and /invoke runs. Built lazily on first use, so a service
        that only does SQL/docs/jobs never pays for it. Pass your own configured mind to serve(mind=...) to override."""
        if getattr(self, "_mind", None) is None:
            from holographic.misc.holographic_unified import UnifiedMind
            self._mind = UnifiedMind()
        return self._mind

    def _tools(self, _payload):
        """The standard tool manifest: every public faculty an /invoke can run, as {name, description, params}. Body:
        none. Returns: {ok, tools:[...]}. This is the shape a harness, an LLM, or another leCore reads to drive us."""
        from holographic.misc.holographic_skills import manifest
        tools = []
        for m in manifest(include_methods=True).get("methods", []):
            name = m["name"]
            # params = the argument names from the introspected signature, minus self (best-effort, for display)
            call = m.get("call", "")
            params = call[call.find("(") + 1:call.rfind(")")] if "(" in call else ""
            param_list = [p.strip().split("=")[0].split(":")[0].strip()
                          for p in params.split(",") if p.strip() and p.strip() != "self"]
            tools.append({"name": name, "description": m.get("summary", ""),
                          "params": param_list, "call": call})
        return {"ok": True, "count": len(tools), "tools": tools}

    def _invoke(self, payload):
        """Run ONE faculty on this node's mind. Body: {name, args:{...}}. Returns: {ok, name, result}. Only PUBLIC
        faculties are callable (no leading underscore); the result is coerced to a JSON-safe form. This is the single
        call a tool client (remote_tools) or a harness makes to use us."""
        payload = payload or {}
        name = payload.get("name", "")
        args = payload.get("args", {}) or {}
        if not name or name.startswith("_"):
            return {"ok": False, "error": "invalid or private tool name: %r" % name}
        fn = getattr(self.mind, name, None)
        if not callable(fn):
            return {"ok": False, "error": "no such tool: %r" % name}
        result = fn(**args) if isinstance(args, dict) else fn(*args)
        return {"ok": True, "name": name, "result": _jsonable(result)}

    def _frame_stream_doc(self, _payload):
        """SSE PUSH channel (Server-Sent Events): GET /frame/stream?session=&target_fps=&frames= keeps the
        connection open and PUSHES a frame at the target rate as 'data: {json}\\n\\n' events, so a browser
        EventSource receives real-time frames WITHOUT polling. Each event is the same shape as POST /frame inline
        mode: {session, preset, level, frame_ms, payload, stats}. This handler is only the doc stub for the index;
        the actual streaming is done in the HTTP handler's do_GET (it must hold the socket open, which the normal
        request/response route table can't). frames=0 streams until the client disconnects."""
        return {"ok": True, "note": "GET /frame/stream is a streaming SSE endpoint; connect with an EventSource, "
                "not a plain fetch. Query: session, target_fps, frames (0=unbounded)."}

    def _frame(self, payload):
        """REAL-TIME FRAME SERVING: adaptive quality per client, the request/response form of a frame stream (this
        stdlib service has no websocket). Body: {session, target_fps?, last_frame_ms?, include_payload?, t?,
        kinds?}. The service keeps one frame-budget controller PER SESSION. Two modes: WITHOUT include_payload it
        reports the client's last frame time and returns the QUALITY PRESET to render with. WITH
        include_payload=true it renders a real frame at the chosen quality and returns it inline -- one round-trip
        per frame. `kinds` selects the OUTPUT PROJECTION(S): any subset of pixels/mesh/splats/shader/lod, MULTIPLE
        at once (e.g. ["pixels","mesh"]) -- every projection is of the same scene, so they can't drift. Returns {ok,
        session, preset, level, budget_ms, target_fps, stats, and (inline) frame_ms + payload}. {drop:true} forgets
        a session. GET /frame/stream is the SSE push variant (?kinds=pixels&kinds=mesh for multiple)."""
        payload = payload or {}
        session = payload.get("session")
        if not session:
            return {"ok": False, "error": "POST /frame needs a JSON body {\"session\": \"...\"}"}
        if self._frames is None:
            from holographic.scene_and_pipeline.holographic_framebudget import FrameServer
            self._frames = FrameServer()
        if payload.get("drop"):
            return {"ok": True, "dropped": self._frames.drop_session(session)}
        # INLINE PAYLOAD MODE: render a real frame at the chosen quality and return the content in THIS response,
        # so the client gets a displayable frame in one round-trip instead of three (decide quality, render, fetch).
        # Uses the built-in demo renderer (a raymarched animated SDF) so it works with no client-supplied scene;
        # a client with its own scene uses the faculty (frame_server.serve_frame) with its own render callback.
        if payload.get("include_payload"):
            from holographic.scene_and_pipeline.holographic_framebudget import demo_frame_payload, PROJECTION_KINDS
            t = float(payload.get("t", 0.0))
            kinds = payload.get("kinds") or ["pixels"]        # the client picks one or MORE output projections
            if isinstance(kinds, str):
                kinds = [kinds]
            bad = [k for k in kinds if k not in PROJECTION_KINDS]
            if bad:
                return {"ok": False, "error": "unknown projection kind(s) %r; known: %s"
                        % (bad, list(PROJECTION_KINDS))}
            # DISTRIBUTED mode: render the frame's pixels tiled across workers (the render farm / distribute_bricks),
            # lifting the single-node resolution cap while the same budget controller holds the rate. Pixels only
            # (the tiled path renders the image); other projections stay single-node.
            tiles = payload.get("tiles")
            if tiles and kinds == ["pixels"]:
                info = self._frames.serve_frame_distributed(session, self.mind.distribute_bricks,
                                                            target_fps=payload.get("target_fps", 60),
                                                            tiles=tuple(tiles), t=t)
                info["ok"] = True
                return info
            info = self._frames.serve_frame(session, lambda preset: demo_frame_payload(preset, t=t, kinds=kinds),
                                            target_fps=payload.get("target_fps", 60))
            info["ok"] = True
            return info
        info = self._frames.next_frame(session, target_fps=payload.get("target_fps", 60),
                                       last_frame_ms=payload.get("last_frame_ms"))
        info["ok"] = True
        return info

    def _game_room(self, world_id, params=None):
        """Get-or-create a game room: (ShardWorld, WorldStreamer). The streamer is built
        NON-advancing -- the clock is owned by whoever drives it (a POST {'ticks':N} or the SSE
        loop's explicit advance), never implicitly by a read, so two clients can't double-step."""
        room = self._games.get(world_id)
        if room is None:
            from holographic.simulation_and_physics.holographic_gameshard import ShardWorld, WorldStreamer
            p = params or {}
            w = ShardWorld(cell=float(p.get("cell", 64.0)), dt=float(p.get("dt", 1.0 / 60.0)),
                           seed=int(p.get("seed", 0)), gravity=tuple(p.get("gravity", (0.0, 0.0, 0.0))),
                           restitution=float(p.get("restitution", 0.2)))
            room = (w, WorldStreamer(w, advance_per_event=False))
            self._games[world_id] = room
        return room

    def _game(self, payload):
        """GAME ROOMS -- the interaction layer's HTTP face. Body: {world, create?, cmds?, ticks?,
        aoi?, drop?}. `create` passes world params (cell, dt, seed, gravity, restitution) on first
        contact; `cmds` is a list of shard commands routed by world.submit (spawns route by
        position); `ticks` advances the authoritative clock N ticks; `aoi` {center, radius}
        returns a cross-shard snapshot; {drop:true} deletes the room. Deterministic on purpose:
        the same POST sequence replays to the same world digest. GET /game/stream is the SSE push
        of per-client deltas over the same room."""
        payload = payload or {}
        world_id = payload.get("world", "w0")
        if payload.get("drop"):
            self._games.pop(world_id, None)
            return {"ok": True, "dropped": world_id}
        w, _st = self._game_room(world_id, payload.get("create"))
        # WHY a lock and not "the GIL is enough": submit/tick interleaving from a concurrent SSE
        # thread could apply a command mid-collision-pass -- the digest would still be *a* valid
        # state, just not a REPLAYABLE one, silently breaking the determinism contract.
        with self._game_lock:
            for c in payload.get("cmds", []) or []:
                w.submit(dict(c))
            migrated = []
            for _ in range(int(payload.get("ticks", 0))):
                migrated.extend(w.tick()["migrated"])
        out = {"ok": True, "world": world_id, "tick": w.tick_count, "shards": len(w.shards),
               "n": sum(len(s.ids) for s in w.shards.values()), "migrated": migrated,
               "digest": w.world_digest()}
        aoi = payload.get("aoi")
        if aoi:
            out["aoi"] = w.snapshot(center=aoi["center"], radius=float(aoi["radius"]))
        return out

    def _game_stream_doc(self, _payload):
        """SSE PUSH channel for game rooms: GET /game/stream?world=&session=&target_fps=&frames=
        &cx=&cy=&cz=&r=&advance= keeps the connection open and pushes one DELTA event per frame
        ('data: {tick, added, removed, moved, digest, n}'): first event is the client's full
        area-of-interest as 'added', later events only what changed -- the wire format a three.js
        client feeds straight into its scene graph. advance=1 (default) makes THIS stream drive
        the world clock at target_fps; pass advance=0 for extra viewers of a world something else
        is ticking. This handler is only the doc stub; streaming happens in do_GET (it must hold
        the socket open)."""
        return {"ok": True, "note": "GET /game/stream is a streaming SSE endpoint; connect with an "
                "EventSource. Query: world, session, target_fps, frames (0=unbounded), cx, cy, cz, r, advance."}

    def _pick(self, payload):
        """VIEWPORT PICKING for a 3D-modeling client: which vert/edge/face is under the cursor. Body: {wireframe,
        u, v, want?}. `wireframe` is a cage {vertices, edges, faces} (from a /frame include_payload kinds=wireframe
        response); `u`,`v` are the normalized screen coordinate (-1..1) under the cursor; `want` is 'vertex' (default),
        'edge', or 'face'. Returns {ok, pick:{kind, index, distance, position/vertices}}. This is the select step
        before an edit; the client sends the cage it already has, so no scene state is needed server-side."""
        payload = payload or {}
        wf = payload.get("wireframe")
        if not wf or "vertices" not in wf:
            return {"ok": False, "error": "POST /pick needs {wireframe:{vertices,edges,faces}, u, v, want?}"}
        from holographic.scene_and_pipeline.holographic_framebudget import pick_element
        pick = pick_element(wf, float(payload.get("u", 0.0)), float(payload.get("v", 0.0)),
                            want=payload.get("want", "vertex"))
        return {"ok": True, "pick": pick}

    def _sql(self, payload):
        """Run SQL against the store: CREATE/INSERT/SELECT/UPDATE/DELETE/JOIN/DROP (UPDATE and DELETE require a WHERE, as a safety guard). Body: {sql}. Returns: {ok, ...} (rows for SELECT, rowcount for writes)."""
        sql = (payload or {}).get("sql")
        if not sql:
            raise QueryError("POST /sql needs a JSON body {\"sql\": \"...\"}")
        result = run_db_sql(sql, self.db)
        if self.persist_path and _is_write(result):         # a write -> persist so it survives a restart
            self._save_to_disk()
        return {"ok": True, "sql": sql, "result": result}

    # ---- GraphQL front door (nested documents) -------------------------------------------------------------
    def _graphql(self, payload):
        """Resolve a GraphQL query. Runs against the objects in the body if given, otherwise the service's stored
        document set. GraphQL is the natural fit for NESTED data, where SQL is the fit for flat rows."""
        query = (payload or {}).get("query")
        if not query:
            raise QueryError("POST /graphql needs a JSON body {\"query\": \"{ ... }\"}")
        objects = payload.get("objects", self.documents)
        from holographic.io_and_interop.holographic_graphql import Scene, resolve
        scene = Scene(objects, dim=2048, seed=0)
        return {"ok": True, "data": resolve(scene, query)}

    def _set_documents(self, payload):
        """Replace the stored nested-document set (the data GraphQL queries run against)."""
        objects = (payload or {}).get("objects")
        if objects is None:
            raise QueryError("POST /documents needs a JSON body {\"objects\": [ ... ]}")
        self.documents = list(objects)
        if self.persist_path:
            self._save_to_disk()
        return {"ok": True, "count": len(self.documents)}

    def _get_documents(self, _payload):
        """The stored nested-document set that GraphQL queries run against. Body: none. Returns: {ok, count, objects}."""
        return {"ok": True, "count": len(self.documents), "objects": self.documents}

    # ---- persistence (be a real database: data survives a restart) -----------------------------------------
    def _save(self, payload):
        """Persist the whole store (SQL tables + documents) to a JSON file. Body: {path} (or the server's --persist path). Returns: {ok, path}."""
        path = (payload or {}).get("path", self.persist_path)
        if not path:
            raise QueryError("POST /save needs a {\"path\": \"...\"} (or start the server with --persist FILE)")
        self._save_to_disk(path)
        return {"ok": True, "path": path}

    def _load(self, payload):
        """Restore the whole store from a JSON file. Body: {path} (or the server's --persist path). Returns: {ok, path}."""
        path = (payload or {}).get("path", self.persist_path)
        if not path:
            raise QueryError("POST /load needs a {\"path\": \"...\"} (or start the server with --persist FILE)")
        self._load_from_disk(path)
        return {"ok": True, "path": path, "documents": len(self.documents)}

    def _save_to_disk(self, path=None):
        """Serialise the whole store -- the SQL database (by deterministic replay) + the document set -- to one JSON
        file. to_state saves each table's (columns, dim, seed, rows), which re-encodes byte-identically on load."""
        import json as _json
        path = path or self.persist_path
        state = {"db": self.db.to_state(), "documents": self.documents}
        with open(path, "w") as f:
            _json.dump(state, f)

    def _load_from_disk(self, path=None):
        """Restore a saved store. Missing file -> a fresh start (so --persist works on first run). Rebuilds the SQL
        database from its replay state and restores the documents."""
        import json as _json
        import os
        path = path or self.persist_path
        if not path or not os.path.exists(path):
            return
        with open(path) as f:
            state = _json.load(f)
        self.db = Database.from_state(state.get("db", {"namespaces": {}}))
        if "user" not in self.db.namespaces:                # always keep a ready writable namespace
            self.db.add_namespace("user")
        self.documents = state.get("documents", [])

    # ---- long-running job control (start/pause/resume/cancel; survives a restart) ---------------------------
    def _jobs_list(self, _payload):
        """List every job with its status + progress. Body: none. Returns: {ok, jobs}."""
        return {"ok": True, "jobs": self._jobs.list()}

    def _jobs_create(self, payload):
        """Define a job: {id, buckets, worker, reduce?, cache?, meta?}. `worker` is a name registered server-side;
        `buckets` and `reduce` (sum/min/max/bundle) are the client's. Does not start it."""
        p = payload or {}
        for field in ("id", "buckets", "worker"):
            if field not in p:
                raise QueryError("POST /jobs/create needs {id, buckets, worker[, reduce, cache, meta]}")
        import numpy as np
        cache = np.asarray(p["cache"], float) if p.get("cache") is not None else None
        self._jobs.create(p["id"], p["buckets"], p["worker"], reduce=p.get("reduce", "sum"),
                          cache=cache, meta=p.get("meta"))
        return {"ok": True, "job": self._jobs.status(p["id"])}

    def _jobs_start(self, payload):
        """Start (or resume) a job in the BACKGROUND so the API stays responsive. {id, batch?}."""
        jid = self._need_job_id(payload)
        self._jobs.start(jid, background=True, batch=int((payload or {}).get("batch", 1)))
        return {"ok": True, "job": self._jobs.status(jid)}

    def _jobs_pause(self, payload):
        """Pause a job at the next bucket boundary and checkpoint it. Body: {id}. Returns: {ok, job}."""
        jid = self._need_job_id(payload)
        self._jobs.pause(jid)                               # stops at the next bucket boundary + checkpoints
        return {"ok": True, "job": self._jobs.status(jid)}

    def _jobs_resume(self, payload):
        """Resume a paused or restored job (remaining buckets only), in the background. Body: {id, batch?}. Returns: {ok, job}."""
        jid = self._need_job_id(payload)
        self._jobs.resume(jid, background=True, batch=int((payload or {}).get("batch", 1)))
        return {"ok": True, "job": self._jobs.status(jid)}

    def _jobs_cancel(self, payload):
        """Cancel a job. Body: {id}. Returns: {ok, job}."""
        jid = self._need_job_id(payload)
        self._jobs.cancel(jid)
        return {"ok": True, "job": self._jobs.status(jid)}

    def _jobs_status(self, payload):
        """One job's status + progress. Body: {id}. Returns: {ok, job}."""
        return {"ok": True, "job": self._jobs.status(self._need_job_id(payload))}

    def _jobs_result(self, payload):
        """The reduced result of a job -- valid once its status is 'done'."""
        jid = self._need_job_id(payload)
        job = self._jobs.jobs.get(jid)
        if job is None:
            raise QueryError("no such job %r" % jid)
        import numpy as np
        res = job.result()
        return {"ok": True, "id": jid, "status": job.status,
                "result": res.tolist() if isinstance(res, np.ndarray) else res}

    # ---- message bus: connect a remote person/agent ---------------------------------------------------------
    def _bus_publish(self, payload):
        """Publish a message onto the bus. Body: {topic, payload?, sender?, reply_to?}. Returns the stored message.
        This is how a remote party (a person's UI or an agent) sends a command/event/reply into the running app."""
        p = payload or {}
        if "topic" not in p:
            raise QueryError("POST /bus/publish needs {topic[, payload, sender, reply_to]}")
        msg = self._bus.publish(p["topic"], p.get("payload"), sender=p.get("sender", "client"),
                                reply_to=p.get("reply_to"))
        return {"ok": True, "message": msg.as_dict()}

    def _bus_poll(self, payload):
        """Drain a mailbox (a remote party's INBOX). Body: {mailbox, patterns?, limit?}. On first call for a mailbox
        the patterns register what it collects (default everything); later calls just pull what has arrived since. The
        messages are PUSHED into the inbox the instant they happen, so a poll returns news immediately -- you are not
        busy-polling a status flag. Returns {messages}."""
        p = payload or {}
        name = p.get("mailbox")
        if not name:
            raise QueryError("POST /bus/poll needs {mailbox[, patterns, limit]}")
        if name not in self._bus._mailboxes:                # open it on first sight with the requested patterns
            self._bus.open_mailbox(name, tuple(p.get("patterns", ("*",))))
        msgs = self._bus.poll(name, limit=p.get("limit"))
        return {"ok": True, "mailbox": name, "messages": [m.as_dict() for m in msgs]}

    def _bus_history(self, payload):
        """Recent messages for catch-up/replay. Body: {pattern?, limit?}. Returns {messages} oldest-first."""
        p = payload or {}
        msgs = self._bus.history(p.get("pattern", "*"), limit=p.get("limit"))
        return {"ok": True, "messages": [m.as_dict() for m in msgs]}

    def _need_job_id(self, payload):
        jid = (payload or {}).get("id")
        if not jid:
            raise QueryError("this endpoint needs {\"id\": \"...\"}")
        if jid not in self._jobs.jobs:
            raise QueryError("no such job %r" % jid)
        return jid

    # ---- agent-friendly skills layer (discover / suggest / route / autocomplete) ---------------------------
    def _skills_manifest(self, _payload):
        """The full machine-readable skill list -- every capability + method with how to call it. An agent loads this
        once to know the whole surface it can drive."""
        import holographic.misc.holographic_skills as _sk
        return {"ok": True, "skills": _sk.manifest()}

    def _skills_suggest(self, payload):
        """A plain-English task -> ranked skills with a confidence and the concrete call. {"task": "...", "k"?: N}."""
        import holographic.misc.holographic_skills as _sk
        task = (payload or {}).get("task")
        if not task:
            raise QueryError("POST /skills/suggest needs {\"task\": \"...\"}")
        return {"ok": True, "task": task, "suggestions": _sk.suggest(task, k=int((payload or {}).get("k", 5)))}

    def _skills_route(self, payload):
        """A task -> a decision: 'act' (with the call) when confident, else 'choose' (the options). {"task": "..."}."""
        import holographic.misc.holographic_skills as _sk
        task = (payload or {}).get("task")
        if not task:
            raise QueryError("POST /skills/route needs {\"task\": \"...\"}")
        return {"ok": True, "task": task, **_sk.route(task)}

    def _skills_complete(self, payload):
        """Method-name autocomplete: {"prefix": "learn_"} -> matching mind methods with their signatures."""
        import holographic.misc.holographic_skills as _sk
        prefix = (payload or {}).get("prefix", "")
        return {"ok": True, "prefix": prefix, "completions": _sk.complete(prefix, k=int((payload or {}).get("k", 15)))}

    def _skills_card(self, payload):
        """A skill card for one capability or method by name: {"name": "..."}."""
        import holographic.misc.holographic_skills as _sk
        name = (payload or {}).get("name")
        if not name:
            raise QueryError("POST /skills/card needs {\"name\": \"...\"}")
        card = _sk.skill_card(name)
        if card is None:
            raise QueryError("no skill named %r (try GET /skills or POST /skills/suggest)" % name)
        return {"ok": True, "card": card}


# ============================================================================================================
# The HTTP server (stdlib) -- wraps a Service.
# ============================================================================================================
def make_handler(service):
    """Build a request handler bound to a Service. Kept a closure so the handler class can see the service."""

    class _Handler(BaseHTTPRequestHandler):
        server_version = "leCore/" + __version__

        def _authorized(self):
            if service.token is None:
                return True
            return self.headers.get("Authorization", "") == "Bearer " + service.token

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return None                                 # signals a bad body

        def _reply(self, status, obj):
            body = json.dumps(obj, default=_json_default).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve(self, method):
            if not self._authorized():
                return self._reply(401, {"ok": False, "error": "missing or bad Authorization bearer token"})
            payload = {} if method == "GET" else self._read_json()
            if payload is None:
                return self._reply(400, {"ok": False, "error": "request body was not valid JSON"})
            status, obj = service.dispatch(method, self.path, payload)
            self._reply(status, obj)

        def do_GET(self):
            # SSE PUSH: /frame/stream keeps the connection open and PUSHES frames at the target rate, so the client
            # (an EventSource) receives frames without polling -- a true push channel over stdlib http.server. Query
            # ?session=&target_fps=&frames= (frames caps the count so the server doesn't stream forever in a test).
            if self.path.split("?")[0] == "/frame/stream":
                if not self._authorized():
                    return self._reply(401, {"ok": False, "error": "missing or bad Authorization bearer token"})
                return self._stream_frames()
            if self.path.split("?")[0] == "/game/stream":
                if not self._authorized():
                    return self._reply(401, {"ok": False, "error": "missing or bad Authorization bearer token"})
                return self._stream_game()
            self._serve("GET")

        def _stream_frames(self):
            import time as _time
            import urllib.parse as _up
            q = _up.parse_qs(_up.urlparse(self.path).query)
            session = (q.get("session", ["stream"]) or ["stream"])[0]
            target_fps = float((q.get("target_fps", ["30"]) or ["30"])[0])
            max_frames = int((q.get("frames", ["0"]) or ["0"])[0])        # 0 = unbounded (until disconnect)
            kinds = q.get("kinds", ["pixels"]) or ["pixels"]              # ?kinds=pixels&kinds=mesh -> multiple outputs
            if service._frames is None:
                from holographic.scene_and_pipeline.holographic_framebudget import FrameServer
                service._frames = FrameServer()
            from holographic.scene_and_pipeline.holographic_framebudget import (demo_frame_payload,
                                                                                FrameBudgetController)
            # start a fresh stream session at the CHEAPEST level so the first pushed frame is already real-time
            # (a stream should never open with a stall); the controller climbs from there if there's headroom.
            if session not in service._frames._sessions:
                service._frames._sessions[session] = FrameBudgetController(
                    target_fps=target_fps, ladder=service._frames._ladder,
                    headroom=service._frames._headroom, start_level=0)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")         # the SSE content type
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            period = 1.0 / target_fps
            n = 0
            t_anim = 0.0
            try:
                while max_frames == 0 or n < max_frames:
                    start = _time.perf_counter()
                    info = service._frames.serve_frame(session, lambda p: demo_frame_payload(p, t=t_anim, kinds=kinds),
                                                        target_fps=target_fps)
                    line = "data: " + json.dumps(info, default=_json_default) + "\n\n"
                    self.wfile.write(line.encode())            # push this frame; a broken pipe ends the stream
                    self.wfile.flush()
                    n += 1
                    t_anim += period
                    # pace to the target rate: sleep off whatever time is left in the frame interval.
                    elapsed = _time.perf_counter() - start
                    if elapsed < period:
                        _time.sleep(period - elapsed)
            except (BrokenPipeError, ConnectionResetError):
                pass                                           # the client closed the EventSource -- stop cleanly

        def _stream_game(self):
            # SSE for game rooms: same push mechanics as _stream_frames, different payload -- one
            # WorldStreamer DELTA per event. advance=1 means this stream owns the world clock.
            import time as _time
            import urllib.parse as _up
            q = _up.parse_qs(_up.urlparse(self.path).query)
            world_id = (q.get("world", ["w0"]) or ["w0"])[0]
            session = (q.get("session", ["stream"]) or ["stream"])[0]
            target_fps = float((q.get("target_fps", ["30"]) or ["30"])[0])
            max_frames = int((q.get("frames", ["0"]) or ["0"])[0])
            advance = (q.get("advance", ["1"]) or ["1"])[0] not in ("0", "false", "no")
            center = None
            radius = None
            if "r" in q:
                radius = float(q["r"][0])
                center = (float((q.get("cx", ["0"]))[0]), float((q.get("cy", ["0"]))[0]),
                          float((q.get("cz", ["0"]))[0]))
            world, streamer = service._game_room(world_id)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            period = 1.0 / target_fps
            n = 0
            try:
                while max_frames == 0 or n < max_frames:
                    start = _time.perf_counter()
                    with service._game_lock:
                        if advance:
                            world.tick()      # this stream is the designated clock
                        ev = streamer.next_event(session, center=center, radius=radius)
                    self.wfile.write(("data: " + json.dumps(ev, default=_json_default) + "\n\n").encode())
                    self.wfile.flush()
                    n += 1
                    elapsed = _time.perf_counter() - start
                    if elapsed < period:
                        _time.sleep(period - elapsed)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                streamer.drop(session)        # forget the baseline; sessions are per-connection

        def do_POST(self):
            self._serve("POST")

        def log_message(self, *a):                          # keep the console clean (no default access log)
            pass

    return _Handler


def _is_write(sql_result):
    """True if a run_db_sql result represents a write (so persistence knows to save). SELECT returns a list of rows;
    every write returns a dict with a telltale key."""
    return isinstance(sql_result, dict) and any(
        k in sql_result for k in ("created_table", "created_database", "inserted", "updated", "deleted", "dropped_table"))


def _json_default(o):
    """Make numpy scalars / arrays JSON-safe if a handler ever returns one."""
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not JSON serializable: %r" % type(o))


def _jsonable(o):
    """Coerce a faculty result into something JSON can carry. Basic types and numpy pass straight through; dicts and
    lists recurse; anything else (a Mesh, a LoadedMesh, ...) becomes a typed summary so /invoke never crashes on an
    un-serializable return value."""
    import numpy as np
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return {"type": type(o).__name__, "repr": repr(o)[:500]}    # object -> a typed summary, not a crash


def serve(host="127.0.0.1", port=8080, token=None, persist_path=None, mind=None, threads=False):
    """Start the standalone API server (blocking). Returns nothing; Ctrl-C to stop. Pass `mind` to expose a specific
    configured UnifiedMind at /tools + /invoke; otherwise a default one is built on first use."""
    service = Service(token=token, persist_path=persist_path, mind=mind)
    if threads:
        # ThreadingHTTPServer: required for game streaming (an open SSE stream must not block
        # command POSTs). Default OFF -- single-threaded is the pinned old behaviour, and the
        # database routes have not been audited for concurrent writers; game routes carry their
        # own lock.
        from http.server import ThreadingHTTPServer
        httpd = ThreadingHTTPServer((host, port), make_handler(service))
    else:
        httpd = HTTPServer((host, port), make_handler(service))
    where = "%s:%d" % (host, port)
    print("leCore API service v%s serving on http://%s" % (__version__, where))
    print("  try:  curl http://%s/health" % where)
    if persist_path:
        print("  data persists to: %s (auto-loaded on start, auto-saved after writes)" % persist_path)
    if token:
        print("  auth: send  Authorization: Bearer <token>  on every request")
    if host == "0.0.0.0":
        print("  NOTE: bound to ALL interfaces -- only do this behind auth/TLS on a trusted network.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
        httpd.server_close()


def _selftest():
    """Drive the Service directly (no socket) so the routes and the SQL/GraphQL/persistence paths are proven fast +
    deterministically."""
    import os
    import tempfile
    svc = Service()
    # health + index
    assert svc.dispatch("GET", "/health", {})[1]["ok"]
    assert svc.dispatch("GET", "/", {})[1]["version"] == __version__
    # capabilities list + search
    caps = svc.dispatch("GET", "/capabilities", {})[1]
    assert caps["ok"] and caps["count"] > 0
    hit = svc.dispatch("POST", "/capabilities/search", {"query": "time travel version history"})[1]
    assert hit["ok"] and any("time-travel" in m["name"].lower() for m in hit["matches"])

    # SQL: the FULL surface over the API -- create, insert, select, update, delete, join, drop
    assert svc.dispatch("POST", "/sql", {"sql": "CREATE TABLE user.t (id, color)"})[1]["ok"]
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.t (id, color) VALUES (1, red)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.t (id, color) VALUES (2, blue)"})
    sel = svc.dispatch("POST", "/sql", {"sql": "SELECT id, color FROM user.t WHERE color = 'red'"})[1]
    assert sel["ok"] and sel["result"][0]["color"] == "red"
    assert svc.dispatch("POST", "/sql", {"sql": "UPDATE user.t SET color = 'crimson' WHERE id = 1"})[1]["result"]["updated"] == 1
    assert svc.dispatch("POST", "/sql", {"sql": "DELETE FROM user.t WHERE color = 'blue'"})[1]["result"]["deleted"] == 1
    # join
    svc.dispatch("POST", "/sql", {"sql": "CREATE TABLE user.a (id, x)"})
    svc.dispatch("POST", "/sql", {"sql": "CREATE TABLE user.b (id, y)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.a (id, x) VALUES (1, A1)"})
    svc.dispatch("POST", "/sql", {"sql": "INSERT INTO user.b (id, y) VALUES (1, B1)"})
    j = svc.dispatch("POST", "/sql", {"sql": "SELECT x, y FROM user.a JOIN user.b ON id"})[1]
    assert j["result"][0] == {"x": "A1", "y": "B1"}
    assert svc.dispatch("POST", "/sql", {"sql": "DROP TABLE user.a"})[1]["result"]["dropped_table"] == "user.a"

    # GraphQL over nested documents
    docs = [{"id": "o1", "name": "ring", "material": "gold", "transform": {"position": [1.0, 0.0, 0.0]}},
            {"id": "o2", "name": "coin", "material": "gold", "transform": {"position": [3.0, 0.0, 0.0]}},
            {"id": "o3", "name": "pipe", "material": "copper", "transform": {"position": [0.0, 2.0, 0.0]}}]
    assert svc.dispatch("POST", "/documents", {"objects": docs})[1]["count"] == 3
    gq = svc.dispatch("POST", "/graphql",
                      {"query": '{ objects(where: {material: "gold"}) { name } }'})[1]
    names = [o["name"] for o in gq["data"]["objects"]]
    assert names == ["ring", "coin"] and gq["ok"]

    # errors: bad SQL 400, unknown route 404, missing body 400, no-WHERE update refused
    assert svc.dispatch("POST", "/sql", {"sql": "SELECT nope FROM user.t"})[0] == 400
    assert svc.dispatch("GET", "/does-not-exist", {})[0] == 404
    assert svc.dispatch("POST", "/sql", {})[0] == 400
    assert svc.dispatch("POST", "/sql", {"sql": "UPDATE user.t SET color = 'x'"})[0] == 400   # WHERE required

    # PERSISTENCE: save, make a fresh service that loads it, confirm the data (SQL + documents) survived a "restart"
    path = os.path.join(tempfile.gettempdir(), "_lecore_svc_test.json")
    try:
        svc.dispatch("POST", "/save", {"path": path})
        reborn = Service(persist_path=path)                 # a fresh process would do exactly this on start
        rows = reborn.dispatch("POST", "/sql", {"sql": "SELECT id, color FROM user.t"})[1]["result"]
        assert any(r["color"] == "crimson" for r in rows)   # the UPDATE survived
        assert reborn.dispatch("GET", "/documents", {})[1]["count"] == 3   # the documents survived
    finally:
        if os.path.exists(path):
            os.remove(path)

    print("OK: holographic_service self-test passed (full SQL: create/insert/select/update/delete/join/drop; GraphQL "
          "over documents; save+load persistence survives a restart; clean 400/404; token field -- standalone DB)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="leCore standalone API service (talk to the engine over HTTP/JSON).")
    p.add_argument("--host", default="127.0.0.1", help="bind address (127.0.0.1 = local only; 0.0.0.0 = all NICs)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--token", default=None, help="optional shared-secret bearer token required on every request")
    p.add_argument("--persist", default=None, help="a JSON file the store is saved to/loaded from (be a real DB: "
                                                    "data survives a restart)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        _selftest()
    else:
        serve(host=args.host, port=args.port, token=args.token, persist_path=args.persist)

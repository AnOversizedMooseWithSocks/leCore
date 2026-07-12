# leCore — Standalone API Service

Run the engine as a standalone server on any OS and talk to it over **HTTP/JSON**. Stdlib-only (numpy aside), so
there's almost nothing to install.

## Launch

| OS | Command |
|---|---|
| Linux / macOS | `./serve.sh` |
| Windows | `serve.bat` |
| any (direct) | `python holographic_service.py` |

The launchers find Python 3, set `PYTHONHASHSEED=0` (the engine is deterministic), make sure `numpy` is installed, and
start the server. Extra arguments pass straight through:

```
./serve.sh --port 9000                     # a different port (default 8080)
./serve.sh --host 0.0.0.0 --token secret   # expose on the network -- ONLY behind auth/TLS on a trusted network
```

By default it binds to **127.0.0.1** (local only). `--token X` requires `Authorization: Bearer X` on every request.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | — | a self-describing index of the endpoints |
| GET | `/health` | — | `{ok, name, version, python, platform, capabilities}` |
| GET | `/capabilities` | — | every capability the instance advertises (name + description) |
| POST | `/capabilities/search` | `{"query":"..."}` | the capability homes that match, plain-English |
| GET | `/tools` | — | the tool manifest: every public faculty as `{name, description, params}` (drive it with `/invoke`) |
| POST | `/invoke` | `{"name":"...","args":{...}}` | run one faculty on this node's mind and return its result as JSON (public faculties only) |
| POST | `/sql` | `{"sql":"..."}` | run SQL: `CREATE TABLE` / `INSERT` / `SELECT` / `UPDATE` / `DELETE` / `JOIN` / `DROP TABLE` |
| POST | `/graphql` | `{"query":"{...}"[, "objects":[...]]}` | resolve a GraphQL query over nested documents |
| POST | `/documents` | `{"objects":[...]}` | set the stored document set GraphQL queries run against |
| GET | `/documents` | — | the stored documents |
| POST | `/save` | `{"path":"..."}` | persist the whole store (SQL + documents) to a JSON file |
| POST | `/load` | `{"path":"..."}` | restore the store from a JSON file |
| GET | `/jobs` | — | list all jobs with status + progress |
| POST | `/jobs/create` | `{"id","buckets","worker"[,"reduce","cache","meta"]}` | define a long-running job (worker is a name registered server-side) |
| POST | `/jobs/start` | `{"id"[,"batch"]}` | start (or resume) a job in the background |
| POST | `/jobs/pause` | `{"id"}` | pause at the next bucket boundary + checkpoint |
| POST | `/jobs/resume` | `{"id"[,"batch"]}` | resume a paused/restored job (remaining buckets only) |
| POST | `/jobs/cancel` | `{"id"}` | cancel a job |
| POST | `/jobs/status` | `{"id"}` | one job's status + progress |
| POST | `/jobs/result` | `{"id"}` | the reduced result (once `done`) |
| POST | `/bus/publish` | `{"topic", "payload"?, "sender"?, "reply_to"?}` | publish a message onto the bus (person/agent send) |
| POST | `/bus/poll` | `{"mailbox", "patterns"?, "limit"?}` | drain a mailbox (a remote party's inbox; events are pushed into it) |
| POST | `/bus/history` | `{"pattern"?, "limit"?}` | recent messages for catch-up / replay |
| GET | `/skills` | — | machine-readable manifest: every capability + method with how to call it |
| POST | `/skills/suggest` | `{"task"[,"k"]}` | rank skills for a plain-English task, with a confidence + the call |
| POST | `/skills/route` | `{"task"}` | a decision: `act` (with the call) when confident, else `choose` (options) |
| POST | `/skills/complete` | `{"prefix"[,"k"]}` | method-name autocomplete with signatures |
| POST | `/skills/card` | `{"name"}` | a skill card for one capability or method |
| POST | `/pick` | `{"wireframe", ...}` | viewport picking for a 3D-modeling client: which vert/edge/face is under the cursor |
| POST | `/frame` | `{...}` | real-time frame serving: adaptive quality per client (the request/response form of a frame stream) |
| GET | `/frame/stream` | `?session=&target_fps=&frames=` | SSE push channel (Server-Sent Events) that keeps streaming frames to a client |
| POST | `/game` | `{world, create?, cmds?, ticks?, aoi?, drop?}` | game rooms: create a sharded world, route player commands, advance the authoritative clock, cross-shard AOI snapshot |
| GET | `/game/stream` | query: `world, session, target_fps, frames, cx, cy, cz, r, advance` | SSE push of per-client world DELTAS (first event = full AOI as 'added'); `advance=1` makes this stream the clock; needs `serve(threads=True)` |

Every response is JSON with an `ok` flag; a bad request returns HTTP 400, an unknown route 404, an unexpected error 500.

*(This endpoint table — and the CLI flags in the Launch section above — are checked in CI by `servicedoc.py` against
the service's actual routes and argparse, so they can't quietly fall out of date. If you add or rename a route or flag,
CI names it to fix; `python servicedoc.py --print` prints a fresh endpoint table to paste here.)*

### Long-running jobs (start / pause / resume / cancel, survive a restart)

A job is a set of **buckets** processed by a **worker** (a name registered server-side) and combined by a monoid
**reducer** (`sum`/`min`/`max`/`bundle`). Because the reduce is order-independent, a job can pause at a bucket boundary,
**checkpoint to disk**, survive the app closing, and resume only the *remaining* buckets. With `--persist FILE`, job
checkpoints live in `FILE.jobs/` and are reloaded on startup — so a paused render reappears in `GET /jobs` after a
restart, ready to resume. The same job code runs on a local pool or the network farm; only *where* the buckets run
changes.

### SQL surface (a drop-in database)

`CREATE TABLE ns.t (cols)` · `INSERT INTO ns.t (cols) VALUES (...)` · `SELECT cols FROM ns.t [WHERE ...] [ORDER BY]
[LIMIT] [OFFSET]` · `UPDATE ns.t SET c=v,... WHERE ...` · `DELETE FROM ns.t WHERE ...` · `SELECT cols FROM a [LEFT]
JOIN b ON key [WHERE c op v]` · `DROP TABLE ns.t`.

> **Safety guard:** `UPDATE` and `DELETE` **require a `WHERE`** — a networked SQL endpoint should not let a typo rewrite
> a whole table. Use the object API for a deliberate no-WHERE bulk write.

### Persistence (data survives a restart)

Start with `--persist FILE` and the store is auto-loaded on startup and auto-saved after every write — so the service
behaves like a real database across restarts. Or call `/save` and `/load` explicitly with a `{"path":...}`.

```
./serve.sh --persist mydb.json      # a durable store in mydb.json
```

## Examples

```sh
# liveness / version
curl http://127.0.0.1:8080/health

# discover what the running instance can do
curl -X POST http://127.0.0.1:8080/capabilities/search \
     -H 'Content-Type: application/json' \
     -d '{"query":"time travel version history"}'

# the query layer, over HTTP
curl -X POST http://127.0.0.1:8080/sql -H 'Content-Type: application/json' \
     -d '{"sql":"CREATE TABLE user.items (id, name, color)"}'
curl -X POST http://127.0.0.1:8080/sql -H 'Content-Type: application/json' \
     -d '{"sql":"INSERT INTO user.items (id, name, color) VALUES (1, widget, red)"}'
curl -X POST http://127.0.0.1:8080/sql -H 'Content-Type: application/json' \
     -d '{"sql":"SELECT name, color FROM user.items WHERE color = '\''red'\''"}'
```

GraphQL over nested documents:

```sh
curl -X POST http://127.0.0.1:8080/documents -H 'Content-Type: application/json' \
     -d '{"objects":[{"id":"o1","name":"ring","material":"gold"},{"id":"o2","name":"pipe","material":"copper"}]}'
curl -X POST http://127.0.0.1:8080/graphql -H 'Content-Type: application/json' \
     -d '{"query":"{ objects(where: {material: \"gold\"}) { name } }"}'
```

Persistence:

```sh
curl -X POST http://127.0.0.1:8080/save -H 'Content-Type: application/json' -d '{"path":"mydb.json"}'
curl -X POST http://127.0.0.1:8080/load -H 'Content-Type: application/json' -d '{"path":"mydb.json"}'
```

With a token:

```sh
curl http://127.0.0.1:8080/health -H 'Authorization: Bearer secret'
```

## Extending the API

The endpoint table is a small route registry in `holographic_service.py` — `Service._register` maps `(method, path)`
to a handler. Add a faculty by registering one more handler; the whole surface reads top to bottom.

## Security notes (kept loud)

- Binds **local** by default. Exposing a compute + SQL endpoint on `0.0.0.0` is a real risk — do it only behind
  **auth + TLS** on a trusted network.
- The optional bearer token is a minimal shared-secret gate, **not** a substitute for TLS across the internet.
- The SQL surface is a hand-rolled subset (no string-concatenated SQL), so classic injection is N/A — but a caller can
  still create/insert/select freely, so treat the endpoint as **trusted-client** unless you add per-route auth.

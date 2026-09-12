# Building an app on leCore — the foundation every app stands on

This is the contract between the engine and the apps built on it (leStudio, the 2-D image editor; Poly Studio, the
3-D modeller; whatever comes next). It exists because sweep 163 audited both apps with the engine's own tools and
found the same layer built twice, differently, with the same mistakes in mirror image. Everything below is provided
by the engine, tested, and discoverable through `find_capability`. An app that does these things its own way is not
wrong — it is alone, and the next app cannot talk to it.

Every snippet in this document is executed by `tests/test_app_foundation_doc.py`; an example that stops running
fails the build.

The lint that checks an app against this document: `python3 tools/app_lint.py <app-root>` (or `m.app_lint(root)`).
Measured before adoption: leStudio R54 **5/16**, Poly Studio 1.2.0 **4/16** (`docs/app_lint/`).

---

## 0. The one rule

**Ask the engine first.** Before writing a helper, a format, a protocol or a sync layer:

```python
import lecore
m = lecore.UnifiedMind(dim=256, seed=0)
hits = m.find_capability("who else is editing this workspace", k=3)
assert hits and "workspace" in str(hits[0].name).lower()
```

Both apps hand-rolled UV unwrap, midpoint subdivision, banded grid bakes, Gaussian blur (leStudio wrote three),
render caches and a fake `.lews` reader — every one an engine faculty they did not find. `app_lint` now names the
nearest card for each hand-rolled helper. If `find_capability` returns nothing relevant, that is the licence to
build — and the thing built belongs in the engine, not the app, if a second app could want it.

## 1. One mind, one preflight

An app holds **one** `UnifiedMind` (a singleton, built lazily) and gates features on the build in front of it, never
on a version pin — a missing faculty and a renamed one both look like an absent attribute at call time.

```python
import lecore
m = lecore.UnifiedMind(dim=256, seed=0)
have = m.features(["lews_open", "agent_surface", "render_quality_gate", "not_a_thing"])
assert have["lews_open"] and not have["not_a_thing"]
status = m.engine_status()          # what Help > Engine status shows
assert status["engine"] and "extras" in status and status["policy"]["bit_exact"] in (True, False)
```

`engine_status()` is the whole panel: version, faculty count, which optional extras import (`numba`, `sympy`,
`cupy`, `wgpu`, `flask`), the GPU report with the reason a path is unavailable, the determinism policy (what would
break bit-exactness) and the CPU budget. Do not keep a client-side list of what to check; it rots silently.

## 2. The workspace: `.lews` is the document, the bus and the session

A `.lews` **file** is a `lecore.container`: a ZIP of `manifest.json` + `.npy` payloads holding typed sections
`{kind, id, meta, arrays}`. Unknown kinds round-trip untouched, so a modeller carries a painter's layers it cannot
open. A `.lews` **directory** (`Workspace`) is the same file kept live: locked, atomic, journalled writes; a change
feed; presence; ids; assets. Full contract: `docs/LEWS_SPEC.md`.

```python
import numpy as np, tempfile, lecore
m = lecore.UnifiedMind(dim=64, seed=0)
root = tempfile.mkdtemp()
painter = m.lews_open(root, app="lestudio")
modeller = m.lews_open(root, app="polystudio")

# canonical kinds: any app means the same thing by them
from holographic.io_and_interop.holographic_container import image_section
tex = image_section(np.ones((8, 8, 4), np.float32), name="skin"); tex["id"] = "tex1"
painter.put(tex)
modeller.put(m.lews_mesh_section([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], sid="m1"))
modeller.put(m.lews_material_section("amethyst point", sid="mat1", library="amethyst"))
modeller.put(m.lews_scene_section([{"id": "o1", "mesh": "m1", "material": "mat1", "texture": "tex1"}]))

# the feed: the modeller learns what the painter did, by revision, minus its own echo
seen = modeller.since(0, exclude="polystudio")
assert [e["kind"] for e in seen] == ["lecore.image"]
assert m.lews_describe(root)["rev"] == 4
```

What the canonical kinds are for:

| kind | one line |
|---|---|
| `lecore.image` | RGBA float 0..1, straight alpha — the texture both apps mean |
| `lecore.mesh` / `lecore.sdf` | geometry as arrays / as the engine's dialect text |
| `lecore.material` | a **name into the physical material library** + overrides — a painter and a modeller mean the same thing by "amethyst" because both ask `glass_optics("amethyst")` |
| `lecore.camera` / `lecore.scene` | a view; bindings of objects → mesh/material/texture **section ids** |
| `lecore.asset` | a blob stored **once**, addressed by sha256 (`put_asset`); GC'd when nothing references it |
| `lecore.journal` | JSON ops with explicit seeds and asset keys that render a target deterministically (inline arrays refused) |
| `lecore.preset` | a named JSON recipe for a brush / material / render / shader that any app can list and apply |

Private kinds keep the app prefix (`lestudio.document`, `polystudio.object`). Every section carries
`meta["schema"]`; `register_kind_schema(kind, version, migrate={old: fn})` lifts old sections and marks newer-than-
known ones read-only. **Rule:** write with `Workspace.put`, never bare `save_container` — the bare save has no
revisions, no lock, no journal, and two apps clobber each other.

### 2a. Ids

```python
import tempfile, lecore
m = lecore.UnifiedMind(dim=64, seed=0); root = tempfile.mkdtemp(); m.lews_open(root, app="lestudio")
assert [m.lews_mint(root, "L"), m.lews_mint(root, "L", app="polystudio"), m.lews_mint(root, "O")] == ["L1", "L2", "O1"]
```

One persisted counter per prefix, advanced under the workspace lock, recorded in the journal. leStudio's process-
global `Layer._next` minted different ids in a fresh process and broke every reference on replay (its P0.3); Poly
Studio reassigns ids on load, so agents must re-read the scene after every open. Neither is acceptable in a shared
workspace.

### 2b. Journal-first documents

The document is an **op journal**; pixels and vertices are a deterministic render of it. Undo, timelapse, autosave
and collaboration all ride the journal. Pixel data survives only as imported **assets** (stored once) and optional
baked caches. leStudio measured one stroke at ~21 MB as a snapshot, 0.13 MB windowed, ~2.7 KB as a path record.

```python
import numpy as np, tempfile, lecore
m = lecore.UnifiedMind(dim=64, seed=0); root = tempfile.mkdtemp(); w = m.lews_open(root, app="lestudio")
tip = m.lews_put_asset(root, np.ones((3, 3), np.float32), "round tip")
assert m.lews_put_asset(root, np.ones((3, 3), np.float32), "again") == tip        # stored once
w.put(m.lews_journal_section("img", [{"op": "stamp", "asset": tip, "x": 1, "y": 1, "seed": 7}]))
orphan = m.lews_put_asset(root, np.zeros((2, 2), np.float32), "unused")
assert m.lews_gc_assets(root) == [orphan]
```

Rules that fall out: seed every random choice explicitly and store the seed in the op; never `hash()` (salted per
process — leStudio's grain textures rendered differently per launch until `zlib.crc32`); never wall clock in a
render or replay path (frames are indexed, not timed); `np.random.default_rng(seed)`, never the legacy global RNG.

### 2c. Presence and the live session

```python
import tempfile, lecore
m = lecore.UnifiedMind(dim=64, seed=0); root = tempfile.mkdtemp(); m.lews_open(root, app="lestudio")
m.lews_touch(root, "moose", activity={"tool": "brush", "section": "tex1"}, name="Moose", app="lestudio")
m.lews_touch(root, "agent-7", activity={"tool": "extrude", "section": "m1"}, app="polystudio")
rev = m.lews_note(root, "moose", "selection", {"layer": 3})     # a non-section change: one journal line, no rewrite
who = m.lews_presence(root)
assert [(p["who"], p["app"], p["host"]) for p in who] == [("moose", "lestudio", True), ("agent-7", "polystudio", False)]
assert m.lews_wait(root, rev - 1, timeout=0.2)[-1]["kind"] == "selection"
```

Presence is a **heartbeat with a timeout**, keyed by the **person or agent** (`who`), never by a connection or tab
— leStudio's "six editors, close tabs, refresh, seven editors" ghost. The host is the earliest-joined participant
still alive, so the role survives its holder's reloads. `lews_wait` is the long-poll an SSE endpoint is one loop
around. The same contract exists in-process as `m.live_session()` for a single app; the directory is what makes it
cross-app.

## 3. Agents: mount the standard surface, do not write one

Both apps derived a tool manifest from Flask's `url_map` by hand (leStudio `/api/schema`, Poly Studio
`/api/agent/tools`) and each missed pieces the other had. The engine mounts the union:

```python
import tempfile, lecore
from flask import Flask, jsonify
app = Flask("demo")

@app.route("/api/paint", methods=["POST"])
def paint():
    """Paint one stroke."""
    return jsonify(ok=True)

m = lecore.UnifiedMind(dim=64, seed=0)
m.agent_surface(app, base="/api", app_name="demo", workspace_root=tempfile.mkdtemp(), image_routes=("render",))
c = app.test_client()
man = c.get("/api/agent/tools").get_json()
assert [t["name"] for t in man["tools"]] == ["paint"] and man["identity"]["X-User"]
assert c.post("/api/agent/invoke", json={"tool": "paint", "json": {}}).get_json()["result"]["ok"]
assert c.post("/api/mind", json={"name": "nope"}).status_code == 400          # answer carries the allowlist + signatures
assert c.post("/api/mind", json={"name": "version"}).get_json()["result"]["engine"]
assert "extras" in c.get("/api/engine").get_json()
```

| door | what an agent gets |
|---|---|
| `GET base/agent/tools` | every route of **this mount** (a gallery mounting several apps must not leak the others' routes — Poly's lesson), with `returns: json / image / stream` |
| `POST base/agent/invoke` | call by name; image routes come back as `data:image/png;base64,…` — "the only way an agent sees its own work" |
| `POST base/mind` | allow-listed **read-only** engine faculties (discovery + analysis); a rejected name returns the allowlist with signatures, so one failed call teaches the usage |
| `GET base/engine` | `engine_status()` |
| `GET base/events` | SSE `{rev, changes, participants}` with a comment ping every 2 s (a stream that only writes on change never learns its socket died) |
| `GET base/presence` | the roster with host |

**Identity contract:** every agent POST carries `X-Client` (this run — echo suppression) and `X-User` (the
persistent identity — presence, host). The after-request hook turns each mutating request into a workspace note,
so an agent driving the painter is visible to the modeller.

For things an agent should read but not call through the app, the engine's own service (`serve.sh`, `/tools` +
`/invoke`) already exposes every faculty; mount it beside the app rather than proxying it.

## 4. Swarms: the engine already staffs and coordinates them

Do not write a swarm protocol. leStudio's swarm painting ("~8000 `/api/paint` calls for a portrait") produced
`paint_batch`, an honest fast path; the coordination layer above it already exists:

- **Roles from task phrases:** `m.dispatch_roles(tasks, spec)` routes "leave a map of the target", "adjust the
  texture gains" to registry roles via the engine's own BM25 — nobody hand-builds member stacks; ambiguity raises.
- **Coordinate through slots, not chatter:** `m.shared_workspace()` — named slots roles read and write while
  deliberating; writes buffer within a round and commit together.
- **Presence across nodes:** `m.registry.announce(principal)` / `registry.list(kind=)` on the bus; in a `.lews`
  directory, `lews_touch` / `lews_presence` (§2c) are the app-level equivalent.
- **Development swarms:** `m.codebase_sync(root, only_stale=True)` teaches file-fingerprinted digests of the app's
  own repo so every agent in the swarm shares one understanding of it; `m.serve(q)` records what could not be
  answered and `m.escalations()` / `m.resolve(q, a, by=)` close the loop with a human's answer, propagated.
- **Batching:** many strokes / objects in one request is the app's business (`paint_batch`), but every generated
  op must still be a **journaled** op with its seed (§2b), or replay dies.

## 5. Long work: jobs, not threads

`m.job_submit(name, args)` runs any faculty in the background with `job_status` (progress), `job_result`,
`job_cancel`, `job_pause`, `job_resume`. Poly Studio's `JobManager` and leStudio's `JOBS = {}` table are the same
thing without pause/resume and invisible to agents. Determinism rule for offloading: an offloaded result must be
**bit-identical** to the in-process one, or saved documents stop reproducing — test it that way.

## 6. Memory per user

`m.app_substrate(app_name, user=...)` (holographic_appkit) gives each `(app, user)` a physically separate partition:
`remember` / `recall` with provenance (taught vs model-cached), `observe` / `suggest` / `habits` (procedures mined
from what the user actually does), `forget`, and a capability preflight. leStudio's "studio sage" (748 taught
lessons) runs on it. Never key one user's memory on a prefix in another's store.

## 7. Presets, quality gates, tests

- **Presets** are `lecore.preset` sections (§2): a brush, a material, a render setup, a shader — JSON recipes, so a
  render preset saved in the modeller is at least listable in the painter.
- **Render regressions** are gated against **absolute** thresholds: `m.render_quality_gate(frame)` — terracing,
  jagged silhouettes, colour fringe. Comparing against the last render cannot see a defect both frames share (Poly
  shipped one that way). Keep **golden `.lews` files** as fixtures: the standard is only a standard if the file an
  app already writes is a valid instance (`tests/fixtures/lews/`).
- **Tests without the engine:** Poly Studio ships a `fake_engine` — a plumbing-only stand-in with the real call
  signatures, because "a dead POST is indistinguishable from a slider you have not touched". That pattern is right
  and stays app-side for now; what the engine guarantees is that `features()` and `engine_status()` are cheap
  enough to call in every test.

```python
import numpy as np, lecore
from holographic.rendering.holographic_qualitygate import _disc
m = lecore.UnifiedMind(dim=64, seed=0)
assert m.render_quality_gate(_disc(120, 160, 80, 60, 30, ss=4, contrast=True))["ok"]
assert m.render_quality_gate(_disc(120, 160, 80, 60, 30, ss=1, contrast=True))["failed"] == ["edge_tones"]
```

## 8. What stays in the app

The document model (layers, strokes, brushes; objects, modifiers), the UI, the node graph's op catalogue, decoders
(PIL, cv2, ffmpeg — never in core), hosting policy (kick/allow/invite are policy), and the choice of what mutates.
The engine owns representation (kinds), coordination (workspace, session, jobs, roles), discovery (catalog, mind
door), measurement (gates, determinism policy) and memory.

## 9. Checklist (what `app_lint` checks)

- [ ] one mind; `features()` / `engine_status()` gate every optional path — no version pins as capability tests
- [ ] documents live in a `.lews` `Workspace` (`put`, `since`, `mint`, `put_asset`); private kinds carry the app prefix; canonical kinds where one exists
- [ ] ids from `lews_mint`; seeds explicit; no `hash()`, no legacy `np.random`, no wall clock in render/replay paths
- [ ] `agent_surface` mounted; `X-Client` / `X-User` honoured; presence keyed by person
- [ ] long work through `job_submit`; offloaded results bit-identical
- [ ] per-user memory through `app_substrate`
- [ ] render changes pass `render_quality_gate`; golden `.lews` fixtures kept
- [ ] every hand-rolled helper checked against `find_capability` / `app_lint` before it ships

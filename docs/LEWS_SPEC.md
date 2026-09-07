# LEWS 1.0 — the leCore shared workspace standard

`.lews` is the one file (and the one live directory) every app built on leCore reads and writes, so an image editor,
a 3-D modeller, a video editor and an agent all work on the same project. This document is the contract; the
reference implementation is `holographic/io_and_interop/holographic_lews.py` (on top of `holographic_container.py`).

## 1. The file

A `.lews` file is a `lecore.container`: a ZIP holding `manifest.json` and `sections/<i>/<name>.npy` array payloads.
Arrays are `.npy` with `allow_pickle=False` — a workspace can never carry executable code. Bytes are deterministic
(fixed ZIP epoch, sorted manifest, fixed entry order): save → load → save is byte-identical.

Top-level manifest `meta` under LEWS:

| key | meaning |
|---|---|
| `lews` | the LEWS spec version this file follows (`"1.0"`) |
| `app`, `app_version` | the app that last wrote the file |
| `engine` | the leCore version that wrote it |
| `rev` | the workspace revision after the last write (monotonic integer) |

Readers that predate LEWS ignore these keys; nothing in the container layout changed.

## 2. Sections and versioning

A section is `{kind, id, meta, arrays}`. Two rules make the format survive other apps and future versions:

1. **Unknown kinds round-trip untouched.** A reader keeps every section it does not understand and writes it back
   verbatim. `describe()` names them so a UI can say "3 `polystudio.object` sections (not editable here)".
2. **Every section carries `meta["schema"]`** — the version of *its kind's* schema. A build declares what it knows
   with `register_kind_schema(kind, version, describe, migrate={old: fn})`. On read, `upgrade_section` applies the
   migration chain to lift an older section; a section **newer** than the build knows comes back with
   `meta["_read_only"] = True` — display it, carry it, never edit it (and `Workspace.put` refuses to write it).
   A section with no `schema` key is schema 1 (everything written before LEWS).

Kinds are namespaced: `lecore.*` are canonical and every app should prefer them; `<app>.*` are private to an app.

## 3. Canonical kinds (schema 1)

| kind | arrays | meta |
|---|---|---|
| `lecore.image` | `image` (H,W,4) float32 0..1, straight alpha | `colour_space` ("srgb"), `dpi`, `name` |
| `lecore.mesh` | `verts` (N,3) f32, `faces` (M,3) i32 triangles, optional `uv` (N,2), `normals` (N,3) | `name`, `n_verts`, `n_faces` |
| `lecore.material` | — | `name`, `library` (a name from the engine's physical material library, or null), `overrides` {channel: value} |
| `lecore.sdf` | — | `name`, `dsl` (the engine's SDF dialect text — any app re-evaluates it with `parse_dsl`) |
| `lecore.camera` | — | `eye`, `target`, `fov_deg`, `aspect` |
| `lecore.scene` | — | `objects: [{id, mesh, material, texture, transform}]` — mesh/material/texture are **section ids** in the same workspace; transform is 16 floats row-major |

`lecore.material` is deliberately a *name into the physical library* plus overrides: a painter and a modeller mean
the same thing by "amethyst" because both ask `glass_optics("amethyst")` for n_d, Abbe and absorption. Builders:
`mesh_section`, `material_section` (validates the library name), `sdf_section`, `camera_section`, `scene_section`,
and `image_section` from the container module; the mind exposes them as `lews_*_section`.

## 4. The live workspace

`Workspace(root, app)` opens a directory:

```
<root>/workspace.lews     the container -- ALWAYS a complete, valid file
<root>/journal.ndjson     one JSON line per change: {rev, op, id, kind, app, sha, t}
<root>/lock               O_EXCL lock file (holder pid/app/time); stale after lock_timeout (default 30 s)
```

Writes (`put`, `delete`) take the lock, **re-read** the file, apply the change, stamp the section's `meta["rev"]`
and `meta["app"]`, write a temp file and `os.replace` it (atomic), append the journal line, release. Two apps writing
different sections interleave instead of clobbering. Readers never lock. `put(section, expected_rev=r)` raises
`ConflictError` if the section changed since `r` (optimistic concurrency); by default the last writer wins, because a
texture and a mesh are different sections and rarely collide.

Catching up: `changes_since(rev)` returns every journal line after `rev` (a restart is just a big gap);
`wait_for_change(rev, timeout)` polls it. `sha` is `section_hash` — sha256 over kind, id, meta (minus the writer
stamps) and every array's bytes — so an app can tell a real change from a re-save.

Agents: `m.lews_describe(root)` and `m.lews_changes(root, since)` are JSON-safe and callable over `/invoke`.

### 4a. The live session (LEWS 1.0, sweep 162)

The directory is also the multiplayer bus leStudio built inside its Flask server (`SYNC`: rev counter, SSE feed,
presence, host) — lifted out so a second app can join without importing the first. Same contract as the in-process
`holographic_livesession.LiveSession`, realised on files:

| call | file effect | meaning |
|---|---|---|
| `bump(src, kind, meta)` / `lews_note` | one journal line `{op:"note", kind, meta, app:src}`; **no container rewrite** | a non-section change ("selection moved", "render done") |
| `since(rev, exclude)` / `lews_wait` | reads the journal | the feed minus your own echo; `wait_for_change` long-polls it |
| `touch(who, activity, name)` / `lews_touch` | `presence/<sha(who)>.json` `{who, app, name, activity, joined, t}` | a heartbeat; `who` is a PERSON or agent id, never a connection |
| `roster(ttl)` / `lews_presence` | reads `presence/`, reaps stale files | who is here across apps; `host` = earliest `joined` still alive |
| `drop(who)` / `lews_leave` | removes the file | a clean exit |

`rev` is the max of the container's meta rev and the last journal rev; both advance under the one lock, so a note
and a `put` share one monotonic counter. Lessons carried from leStudio's R-series: presence keyed by connection gave
one person a ghost chip per reload (so `who` is the person); the host role flapped on reload until it became
"earliest joined still present"; a stream that only wrote on change never noticed dead sockets (a heartbeat with a
ttl has no such failure).

`Workspace.from_file(path, root)` / `lews_import` opens any app's single-file `.lews` as a live directory; leStudio's
golden fixtures (`tests/fixtures/lews/*.lews`) are pinned to open and round-trip byte-identically beside another
app's sections.

### 4b. Journal-first documents: `lecore.asset` and `lecore.journal`

leStudio's DETERMINISM_BACKLOG doctrine, made portable: a document is an op journal, pixels are its render. One stroke
measured ~21 MB as a snapshot, 0.13 MB windowed, ~2.7 KB as a path record.

| kind | id | content |
|---|---|---|
| `lecore.asset` | `asset:<sha256(shape, dtype, bytes)>` | `arrays.data`; meta `{name, sha256, nbytes, shape, dtype}`. `Workspace.put_asset` stores identical bytes ONCE (no write, no rev the second time). |
| `lecore.journal` | `journal:<target>` | meta `{target, ops:[{op, seed?, asset?, ...}], n}` — plain JSON; an inline array is **refused** (`journal_section`), randomness is an explicit `seed`, pixels are `asset` keys. |

`journal_asset_refs(ops)` finds every `asset` / `*_asset` value at any depth; `Workspace.gc_assets()` deletes the
assets nothing references (the natural GC on save). Replaying a journal is the app's job (leCore's `edit_history`
does it for command lists); the standard only guarantees another app can read the recipe and fetch its assets.

### 4c. Ids and presets (sweep 163)

| | |
|---|---|
| `Workspace.mint(prefix)` / `lews_mint` | `'<prefix><n>'` from one persisted counter per prefix (`<root>/ids.json`), advanced under the workspace lock, recorded as a `{op:"mint"}` journal line. Deterministic on replay, collision-free across apps and processes. |
| `lecore.preset` | id `preset:<target>:<name>`; meta `{name, target, params (JSON), tags, author}`. A brush, material, render or shader recipe any app can list and apply. Arrays are refused — a recipe is not a texture. |

The standard for apps as a whole — one mind, the workspace, agents, swarms, jobs, memory, gates — is
`docs/APP_FOUNDATION.md`; `tools/app_lint.py` checks an app tree against it.

## 5. Kept negatives (measure before removing)

- No automatic merge of two edits to one section: the journal shows the collision; the app resolves it.
- Polling, not push. The file is the bus. The engine's `distributed_bus` exists for push when an app wants it.
- A `put` rewrites the whole container. Fine at the sizes leStudio and Poly Studio produce; when a workspace crosses
  the point where that hurts (measure it), shard by section into `sections/<id>.lews` — lever 5, tile under an
  orchestrator — without changing the section contract.

## 6. For Poly Studio and leStudio

Both already read and write `lecore.container` files with `app` in the top-level meta; those files are LEWS with
`schema` absent → schema 1, and open unchanged. To join the live workspace: build sections with `make_section` (or
the canonical builders) so they carry `schema`, write with `Workspace.put` instead of `save_container`, and poll
`changes_since` from the last rev seen. Poly Studio's `polystudio.object` → prefer `lecore.mesh` + `lecore.scene`;
leStudio's `lestudio.document` stays private (a layer stack), and its composite goes out as `lecore.image`.

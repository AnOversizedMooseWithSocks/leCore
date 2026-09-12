"""holographic_lews.py -- the SHARED WORKSPACE standard for apps built on leCore: versioned section kinds, the
canonical kinds every app can read, and a LIVE workspace on disk that several apps edit at once.

WHY THIS EXISTS
---------------
`holographic_container` is the file format: a ZIP of typed sections {kind, id, meta, arrays} where a section whose
kind a reader does not understand round-trips untouched. Poly Studio and leStudio already trade `.lews` files that
way. What they could NOT do, and what Moose asked for, is three things:

  1. VERSIONING. A kind's schema changes. A reader must know which version it is looking at, upgrade an older
     section it understands, and carry a NEWER one through read-only instead of misreading it. The container had one
     number for the whole file; the kinds had none.
  2. CANONICAL KINDS. `lecore.image` existed; a mesh, a material, an SDF, a camera and a scene did not -- so each app
     pair needed its own adapter (N^2). One documented kind per thing makes it N.
  3. A LIVE WORKSPACE. Not just a file you export and import, but a place on disk two apps hold open together: the
     painter saves a texture, the modeller sees the change on its next poll and re-textures the object; the modeller
     moves a mesh, the painter's 3-D preview follows. That needs atomic writes, a lock, per-section revisions, and a
     JOURNAL of changes any app can read from the revision it last saw.

THE STANDARD (LEWS_SPEC "1.0")
------------------------------
* The file is a `lecore.container` (format tag, container version 1). LEWS adds top-level meta:
      {"lews": "1.0", "app": <writer app>, "app_version": <str>, "engine": <leCore version>, "rev": <int>}
  Older readers ignore the extra keys; nothing about the container layout changes.
* Every section's meta carries `"schema": <int>` -- the version of ITS KIND's schema -- stamped by `make_section`.
  A section without it is schema 1 (everything written before this module).
* `register_kind_schema(kind, version, describe, migrate={old_version: fn})` declares what THIS build understands.
* Journal-first documents: `lecore.asset` (a blob stored once, addressed by sha256; `Workspace.put_asset/get_asset/gc_assets`)
  and `lecore.journal` (ops with explicit seeds and asset keys that render a target section deterministically).
* The live session: `Workspace.touch/roster/participants/drop` (presence as heartbeat files, host = earliest joined),
  `bump/since` (notes on the journal, no container rewrite), `Workspace.from_file` opens any app's single-file .lews.
  `upgrade_section(sec)` applies the migration chain up to the known version; a section whose schema is NEWER than
  the build knows is returned unchanged with `meta["_read_only"] = True` -- the reader may display it and must
  write it back verbatim, never edit it.
* Canonical kinds (all schema 1):
      lecore.image     (H,W,4) float 0..1 straight alpha             meta {colour_space, dpi, name}      (container)
      lecore.mesh      verts (N,3) f32, faces (M,3) i32, [uv (N,2), normals (N,3)]   meta {name}
      lecore.material  meta {name, library: <matlib name or null>, overrides: {...}}  -- the PHYSICAL material library
                       is the vocabulary; an app names a library material and may override channels, so a modeller
                       and a painter agree on what "amethyst" means because both ask the engine.
      lecore.sdf       meta {name, dsl}  -- the engine's own SDF dialect text; any app can re-evaluate it
      lecore.camera    meta {eye, target, fov_deg, aspect}
      lecore.scene     meta {objects: [{id, mesh, material, texture, transform (16 floats, row-major)}]} -- BINDINGS by
                       section id; this is how "the painter's texture is on the modeller's mesh" is expressed
  App-private kinds keep their app prefix (`polystudio.object`, `lestudio.document`) and are carried by everyone.

THE LIVE WORKSPACE (`Workspace`)
--------------------------------
A directory: `<root>/workspace.lews` (the container, always a complete valid file), `<root>/journal.ndjson` (one JSON
line per change: rev, op, id, kind, app, sha256 of the section's arrays), `<root>/lock` (an O_EXCL lock file with the
holder's pid and time; stale after `lock_timeout`). Every `put`/`delete` takes the lock, RE-READS the file (so two
apps' writes interleave instead of clobbering), applies the change, bumps the section's `meta["rev"]`, writes to a
temp file and `os.replace`s it (atomic on POSIX and NT), appends the journal line, releases the lock. Readers never
lock: they read a complete file or the previous complete file. `changes_since(rev)` is how an app catches up after
any gap -- restart included -- and `wait_for_change(rev, timeout)` polls it. Optimistic concurrency: `put(...,
expected_rev=r)` raises `ConflictError` if someone else changed that section first -- last-writer-wins is the default
because a texture and a mesh are different sections and rarely collide; when they do, the app decides.

NEGATIVES, kept: no merge of two edits to the SAME section (the journal tells you it happened; the app resolves);
polling, not push (a file is the bus -- the engine's distributed_bus exists for push when apps want it); no
partial-file updates (a 200 MB workspace rewrites 200 MB per put -- lever 5 says shard by section into
`sections/<id>.lews` when that bites; measured threshold not yet reached by either app).

Deterministic: the container bytes are the container module's (fixed zip epoch, sorted manifest); revisions are
integers, hashes are sha256 of the array bytes -- no wall clock in the file. The journal carries a wall-clock `t`
for humans only. NumPy + stdlib.
"""
import hashlib
import json
import os
import time

import numpy as np

from holographic.io_and_interop.holographic_container import (save_container, load_container, register_kind,
                                                               known_kinds, IMAGE_KIND)

LEWS_SPEC = "1.0"

#: kind -> {"version": int, "migrate": {from_version: fn(section) -> section}}
_SCHEMAS = {}


class ConflictError(RuntimeError):
    """put(expected_rev=...) found the section changed by someone else first."""


# ------------------------------------------------------------------------------------------ kind schemas
def register_kind_schema(kind, version=1, describe="", migrate=None):
    """Declare a section kind with the SCHEMA VERSION this build reads and writes, and the migrations that lift older
    sections to it. `migrate` is {old_version: fn(section) -> section}; the chain is applied in order until the
    current version. Also registers the kind's description with the container (known_kinds)."""
    register_kind(kind, describe)
    _SCHEMAS[str(kind)] = {"version": int(version), "migrate": dict(migrate or {})}
    return kind


def kind_schema_version(kind):
    """The schema version this build knows for `kind`, or None for an unregistered kind."""
    s = _SCHEMAS.get(str(kind))
    return None if s is None else s["version"]


def make_section(kind, sid="", meta=None, arrays=None):
    """A section stamped with its kind's current schema version. Use this rather than a bare dict so every file this
    build writes says which version of the kind it carries."""
    m = dict(meta or {})
    m["schema"] = int(kind_schema_version(kind) or m.get("schema", 1))
    return {"kind": str(kind), "id": str(sid), "meta": m, "arrays": dict(arrays or {})}


def upgrade_section(section):
    """Lift `section` to the schema version this build knows: apply migrations for an OLDER section, pass an unknown
    kind through untouched, and mark a NEWER-than-known section `meta["_read_only"] = True` (carry it, never edit it).
    Returns a new section dict; the input is not modified."""
    sec = {"kind": section.get("kind"), "id": section.get("id", ""), "meta": dict(section.get("meta") or {}),
           "arrays": dict(section.get("arrays") or {})}
    s = _SCHEMAS.get(str(sec["kind"]))
    if s is None:
        return sec                                                    # foreign kind: opaque, carried verbatim
    have = int(sec["meta"].get("schema", 1))
    want = s["version"]
    while have < want:
        fn = s["migrate"].get(have)
        if fn is None:
            raise ValueError("no migration for %s schema %d -> %d" % (sec["kind"], have, have + 1))
        sec = fn(sec)
        sec["meta"] = dict(sec.get("meta") or {}); sec["meta"]["schema"] = have + 1
        have += 1
    if have > want:
        sec["meta"]["_read_only"] = True                              # written by a newer build: display, don't touch
    return sec


# ------------------------------------------------------------------------------------------ canonical kinds
MESH_KIND = register_kind_schema("lecore.mesh", 1, "a triangle mesh: verts (N,3) f32, faces (M,3) i32, optional uv/normals")
MATERIAL_KIND = register_kind_schema("lecore.material", 1, "a physical material: a material-library name plus channel overrides")
SDF_KIND = register_kind_schema("lecore.sdf", 1, "an SDF in the engine's own dialect text (meta.dsl)")
CAMERA_KIND = register_kind_schema("lecore.camera", 1, "a camera: eye, target, fov_deg, aspect")
SCENE_KIND = register_kind_schema("lecore.scene", 1, "bindings: objects -> mesh / material / texture section ids + transforms")
register_kind_schema(IMAGE_KIND, 1, "an RGBA float 0..1 image any app can use as a texture; meta: {colour_space, dpi}")
PRESET_KIND = register_kind_schema("lecore.preset", 1, "a named parameter set for some target kind (brush, material, render, shader): meta {name, target, params, tags}")
ASSET_KIND = register_kind_schema("lecore.asset", 1, "a content-addressed blob stored ONCE: id 'asset:<sha256>', arrays.data, meta {name, sha256, nbytes}")
JOURNAL_KIND = register_kind_schema("lecore.journal", 1, "an op journal: meta.ops (each op explicit seeds + asset refs) that renders meta.target deterministically")
register_kind_schema("lestudio.document", 1, "leStudio layer document")
register_kind_schema("polystudio.object", 1, "Poly Studio scene object")


def mesh_section(verts, faces, sid="", name="mesh", uv=None, normals=None):
    """A canonical `lecore.mesh`. Faces must be triangles (M,3); triangulate first (meshverbs2.triangulate_ngons)."""
    V = np.asarray(verts, np.float32).reshape(-1, 3); F = np.asarray(faces, np.int32).reshape(-1, 3)
    if F.size and (F.min() < 0 or F.max() >= len(V)):
        raise ValueError("faces index outside verts")
    arrays = {"verts": V, "faces": F}
    if uv is not None:
        arrays["uv"] = np.asarray(uv, np.float32).reshape(len(V), 2)
    if normals is not None:
        arrays["normals"] = np.asarray(normals, np.float32).reshape(len(V), 3)
    return make_section(MESH_KIND, sid, {"name": str(name), "n_verts": int(len(V)), "n_faces": int(len(F))}, arrays)


def material_section(name, sid="", library=None, overrides=None):
    """A canonical `lecore.material`: `library` names a material in the engine's physical library (glass_optics /
    matlib) so every app resolves the same physics; `overrides` are channel overrides (json scalars/lists)."""
    if library is not None:
        from holographic.materials_and_texture.holographic_matlib import _IOR, _ABSORB
        if library not in _IOR and library not in _ABSORB:
            raise ValueError("unknown library material %r" % (library,))
    return make_section(MATERIAL_KIND, sid, {"name": str(name), "library": library, "overrides": dict(overrides or {})})


def sdf_section(dsl, sid="", name="sdf"):
    """A canonical `lecore.sdf` carrying the engine's SDF dialect text."""
    return make_section(SDF_KIND, sid, {"name": str(name), "dsl": str(dsl)})


def camera_section(eye, target, fov_deg=40.0, aspect=1.0, sid=""):
    return make_section(CAMERA_KIND, sid, {"eye": [float(x) for x in eye], "target": [float(x) for x in target],
                                           "fov_deg": float(fov_deg), "aspect": float(aspect)})


def scene_section(objects, sid="scene", name="scene"):
    """A canonical `lecore.scene`: objects = [{id, mesh, material, texture, transform}] where mesh/material/texture
    are SECTION IDS in the same workspace (any may be null) and transform is 16 floats row-major (default identity)."""
    objs = []
    for o in objects:
        t = o.get("transform")
        t = [float(x) for x in (np.eye(4).ravel() if t is None else np.asarray(t, float).ravel())]
        if len(t) != 16:
            raise ValueError("transform must be 16 floats")
        objs.append({"id": str(o.get("id", "")), "mesh": o.get("mesh"), "material": o.get("material"),
                     "texture": o.get("texture"), "transform": t})
    return make_section(SCENE_KIND, sid, {"name": str(name), "objects": objs})


# ------------------------------------------------------------------------------------------ journal-first documents
# THE DOCTRINE (leStudio's DETERMINISM_BACKLOG, measured there): a document is an OP JOURNAL -- paths, parameters,
# explicit seeds and references to imported assets -- and pixels/vertices are a deterministic RENDER of it. Undo,
# timelapse, saves, autosave and collaboration all ride the journal. Pixel data survives only as (a) imported assets,
# stored once and content-addressed, and (b) optional baked caches that can be thrown away. Evidence: one stroke's
# undo cost ~21 MB as a snapshot, 0.13 MB windowed, ~2.7 KB as a path record. Two canonical kinds carry the doctrine
# across apps so a modeller can replay a painter's texture (or just read its assets) without either importing the other.

def asset_key(data):
    """sha256 over shape, dtype and bytes -> the asset's content address. Two identical pastes share one key."""
    a = np.ascontiguousarray(np.asarray(data))
    h = hashlib.sha256(repr((a.shape, str(a.dtype))).encode()); h.update(a.tobytes())
    return h.hexdigest()


def asset_section(data, name=""):
    """A `lecore.asset` section: id 'asset:<sha256>', the array under arrays.data. The id IS the content, so putting
    the same bytes twice is a no-op (see Workspace.put_asset) and a journal op references it by key."""
    a = np.ascontiguousarray(np.asarray(data)); key = asset_key(a)
    return make_section(ASSET_KIND, "asset:" + key, {"name": str(name), "sha256": key, "nbytes": int(a.nbytes),
                                                      "shape": [int(s) for s in a.shape], "dtype": str(a.dtype)},
                        {"data": a})


def journal_section(target, ops, sid="", name="journal"):
    """A `lecore.journal` section: `ops` is a JSON list, each op a dict with at least {"op": name}; randomness must be
    an explicit "seed" and pixel data an "asset" key (never inline arrays -- that is the snapshot disease). `target` is
    the section id the journal renders (an image, a mesh). Validates the two rules that make replay honest."""
    out = []
    for i, o in enumerate(ops):
        if not isinstance(o, dict) or "op" not in o:
            raise ValueError("journal op %d must be a dict with an 'op' key" % i)
        for k, v in o.items():
            if isinstance(v, np.ndarray):
                raise ValueError("journal op %d carries an inline array %r -- store it as an asset and reference its key" % (i, k))
        out.append(json.loads(json.dumps(o, default=float)))          # plain JSON only, so any app can read it
    return make_section(JOURNAL_KIND, sid or ("journal:" + str(target)),
                        {"name": str(name), "target": str(target), "ops": out, "n": len(out)})


def journal_asset_refs(ops):
    """Every asset key an op list references: any string value under a key named 'asset' or ending in '_asset', at
    any depth. This is what makes GC safe -- a reference is a value, not a naming convention an app might skip."""
    refs = set()
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, str) and (k == "asset" or k.endswith("_asset")):
                    refs.add(v[6:] if v.startswith("asset:") else v)
                else:
                    walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(ops)
    return refs


def preset_section(name, target, params, sid="", tags=(), author=""):
    """A `lecore.preset` section: a NAMED PARAMETER SET for `target` -- 'lecore.material', 'lestudio.brush',
    'polystudio.render', a shader -- as plain JSON. Both apps keep preset libraries in their own shapes; one kind means
    a render preset saved in the modeller is readable (and at least listable) in the painter. `params` must be JSON
    (no arrays: a preset is a recipe, not a texture -- that is an asset)."""
    def _no_arrays(x):
        if isinstance(x, np.ndarray):
            raise ValueError("a preset carries no arrays (that is an asset); got one of shape %s" % (x.shape,))
        if isinstance(x, dict):
            for v in x.values(): _no_arrays(v)
        elif isinstance(x, (list, tuple)):
            for v in x: _no_arrays(v)
    _no_arrays(params)
    js = json.loads(json.dumps(params, default=float))
    return make_section(PRESET_KIND, sid or ("preset:%s:%s" % (target, name)),
                        {"name": str(name), "target": str(target), "params": js, "tags": [str(t) for t in tags],
                         "author": str(author)})


def section_hash(section):
    """sha256 over the kind, id, meta (sorted json) and every array's bytes -- the identity a journal line carries."""
    h = hashlib.sha256()
    h.update(str(section.get("kind")).encode()); h.update(str(section.get("id", "")).encode())
    meta = {k: v for k, v in (section.get("meta") or {}).items() if k not in ("rev", "app")}   # writer stamps are not content
    h.update(json.dumps(meta, sort_keys=True, default=str).encode())
    for name in sorted(section.get("arrays") or {}):
        a = np.ascontiguousarray(section["arrays"][name])
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ the live workspace
class Workspace:
    """A directory two or more apps hold open together. See the module docstring for the contract."""

    FILE = "workspace.lews"; JOURNAL = "journal.ndjson"; LOCK = "lock"

    def __init__(self, root, app="app", app_version="", lock_timeout=30.0, create=True):
        self.root = str(root); self.app = str(app); self.app_version = str(app_version)
        self.lock_timeout = float(lock_timeout)
        if create:
            os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, self.FILE)
        self.journal_path = os.path.join(self.root, self.JOURNAL)
        self.lock_path = os.path.join(self.root, self.LOCK)
        if create and not os.path.exists(self.path):
            self._write({"meta": self._file_meta(0), "sections": []})

    # ---- meta / io
    def _engine_version(self):
        try:
            import lecore
            return str(getattr(lecore, "__version__", ""))
        except Exception:
            return ""

    def _file_meta(self, rev, base=None):
        m = dict(base or {})
        m.update({"lews": LEWS_SPEC, "app": self.app, "app_version": self.app_version,
                  "engine": self._engine_version(), "rev": int(rev)})
        return m

    def _read(self):
        with open(self.path, "rb") as f:
            return load_container(f.read())

    def _write(self, cont):
        """Atomic: write a temp file beside the target, fsync, os.replace. A reader sees either the old complete file
        or the new one -- never a half-written ZIP."""
        blob = save_container(cont["sections"], meta=cont["meta"])
        tmp = self.path + ".tmp.%d" % os.getpid()
        with open(tmp, "wb") as f:
            f.write(blob); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.path)

    # ---- lock
    def _acquire(self):
        deadline = time.time() + self.lock_timeout
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"pid": os.getpid(), "app": self.app, "t": time.time()}).encode()); os.close(fd)
                return
            except FileExistsError:
                try:                                                  # a holder that died leaves a stale lock: break it
                    age = time.time() - os.path.getmtime(self.lock_path)
                    if age > self.lock_timeout:
                        os.remove(self.lock_path); continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError("workspace lock held too long: %s" % self.lock_path)
                time.sleep(0.02)

    def _release(self):
        try:
            os.remove(self.lock_path)
        except OSError:
            pass

    # ---- reads (never lock)
    def _journal_rev(self):
        """The rev of the last journal line (0 with no journal). Cheap: reads the tail of the file only."""
        try:
            with open(self.journal_path, "rb") as f:
                f.seek(0, os.SEEK_END); n = f.tell()
                f.seek(max(0, n - 4096)); tail = f.read().splitlines()
            for line in reversed(tail):
                if line.strip():
                    return int(json.loads(line).get("rev", 0))
        except (OSError, ValueError):
            pass
        return 0

    def rev(self):
        """The workspace's current revision. WHY the max of two places: section writes stamp the container's meta,
        while presence notes (`bump`) only append to the journal -- rewriting a whole ZIP to say "cursor moved" would
        make the cheap signal cost as much as the expensive one. Both counters advance under the same lock, so the
        larger is the truth."""
        return max(int((self._read()["meta"] or {}).get("rev", 0)), self._journal_rev())

    def sections(self, kind=None, upgrade=True):
        """Every section (optionally of one kind), upgraded to this build's schema versions."""
        out = [s for s in self._read()["sections"] if kind is None or s.get("kind") == kind]
        return [upgrade_section(s) for s in out] if upgrade else out

    def get(self, sid, upgrade=True):
        """One section by id, or None."""
        for s in self._read()["sections"]:
            if str(s.get("id", "")) == str(sid):
                return upgrade_section(s) if upgrade else s
        return None

    def describe(self):
        """{app, rev, lews, sections: [{id, kind, schema, known, rev}]} -- what a UI shows before opening anything."""
        cont = self._read(); kk = known_kinds()
        return {"app": (cont["meta"] or {}).get("app"), "rev": int((cont["meta"] or {}).get("rev", 0)),
                "lews": (cont["meta"] or {}).get("lews"),
                "sections": [{"id": s.get("id", ""), "kind": s.get("kind"),
                              "schema": int((s.get("meta") or {}).get("schema", 1)),
                              "known": s.get("kind") in kk, "rev": int((s.get("meta") or {}).get("rev", 0))}
                             for s in cont["sections"]]}

    # ---- writes (lock, re-read, apply, atomic write, journal)
    def _journal(self, entry):
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    def put(self, section, expected_rev=None):
        """Insert or replace the section with this id. Returns the new workspace rev. `expected_rev`: raise
        ConflictError unless the existing section's rev equals it (optimistic concurrency)."""
        sec = upgrade_section(section)
        if sec["meta"].get("_read_only"):
            raise ValueError("refusing to write a section from a newer schema than this build knows")
        self._acquire()
        try:
            cont = self._read()
            rev = max(int((cont["meta"] or {}).get("rev", 0)), self._journal_rev()) + 1
            idx = next((i for i, s in enumerate(cont["sections"]) if str(s.get("id", "")) == sec["id"]), None)
            if expected_rev is not None:
                have = int((cont["sections"][idx].get("meta") or {}).get("rev", 0)) if idx is not None else 0
                if have != int(expected_rev):
                    raise ConflictError("section %r is at rev %d, expected %d" % (sec["id"], have, expected_rev))
            sec["meta"]["rev"] = rev; sec["meta"]["app"] = self.app
            if idx is None:
                cont["sections"].append(sec)
            else:
                cont["sections"][idx] = sec
            cont["meta"] = self._file_meta(rev, cont["meta"])
            self._write(cont)
            self._journal({"rev": rev, "op": "put", "id": sec["id"], "kind": sec["kind"], "app": self.app,
                           "sha": section_hash(sec), "t": time.time()})
            return rev
        finally:
            self._release()

    def delete(self, sid):
        """Remove a section by id. Returns the new rev (unchanged if the id was absent)."""
        self._acquire()
        try:
            cont = self._read()
            before = len(cont["sections"])
            kept = [s for s in cont["sections"] if str(s.get("id", "")) != str(sid)]
            if len(kept) == before:
                return int((cont["meta"] or {}).get("rev", 0))
            rev = max(int((cont["meta"] or {}).get("rev", 0)), self._journal_rev()) + 1
            cont["sections"] = kept; cont["meta"] = self._file_meta(rev, cont["meta"])
            self._write(cont)
            self._journal({"rev": rev, "op": "delete", "id": str(sid), "kind": None, "app": self.app, "sha": None,
                           "t": time.time()})
            return rev
        finally:
            self._release()

    # ---- change feed
    def changes_since(self, rev):
        """Journal entries with rev > `rev`, oldest first -- how an app catches up after any gap."""
        out = []
        if not os.path.exists(self.journal_path):
            return out
        with open(self.journal_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if int(e.get("rev", 0)) > int(rev):
                    out.append(e)
        return out

    def wait_for_change(self, rev, timeout=5.0, poll=0.1):
        """Block until the workspace rev exceeds `rev` (or timeout); returns the new changes (possibly empty)."""
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            ch = self.changes_since(rev)
            if ch:
                return ch
            time.sleep(poll)
        return self.changes_since(rev)

    def export_bytes(self):
        """The current workspace as one `.lews` file's bytes (for download / hand-off)."""
        with open(self.path, "rb") as f:
            return f.read()

    # ---- ids: one counter per prefix, persisted, minted under the lock
    IDS = "ids.json"

    def mint(self, prefix="S"):
        """A fresh id '<prefix><n>' that no app on this workspace has issued before. WHY it lives here: leStudio's
        P0.3 lesson -- ids from a process-global counter depend on every other document opened in the process, so a
        journal replayed in a fresh process minted DIFFERENT ids and every id reference broke; Poly Studio's ids are
        reassigned on load, so agents must re-read the scene after every load. One persisted counter per prefix,
        advanced under the workspace lock and recorded in the journal, gives deterministic, replayable, collision-free
        ids across apps. Two processes minting 'L' concurrently get L7 and L8, never L7 twice."""
        path = os.path.join(self.root, self.IDS)
        self._acquire()
        try:
            try:
                with open(path) as f:
                    ids = json.load(f)
            except (OSError, ValueError):
                ids = {}
            n = int(ids.get(str(prefix), 0)) + 1
            ids[str(prefix)] = n
            tmp = path + ".tmp.%d" % os.getpid()
            with open(tmp, "w") as f:
                json.dump(ids, f, sort_keys=True)
            os.replace(tmp, path)
            sid = "%s%d" % (prefix, n)
            self._journal({"rev": self.rev() + 1, "op": "mint", "id": sid, "kind": str(prefix), "app": self.app,
                           "sha": None, "t": time.time()})
            return sid
        finally:
            self._release()

    def id_counters(self):
        """{prefix: last issued n} -- what mint has handed out so far."""
        try:
            with open(os.path.join(self.root, self.IDS)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    # ---- assets and journals (journal-first documents across apps)
    def put_asset(self, data, name=""):
        """Store a blob ONCE and return its key. If an asset with this content is already in the workspace nothing is
        written (no lock, no rev) -- the dedup that keeps a hundred stamps of one brush tip at one copy."""
        sec = asset_section(data, name)
        if self.get(sec["id"], upgrade=False) is not None:
            return sec["meta"]["sha256"]
        self.put(sec)
        return sec["meta"]["sha256"]

    def get_asset(self, key):
        """The array behind an asset key (with or without the 'asset:' prefix), or None."""
        key = str(key); sec = self.get(key if key.startswith("asset:") else "asset:" + key, upgrade=False)
        return None if sec is None else sec["arrays"]["data"]

    def asset_refs(self):
        """asset key -> set of section ids that reference it: every lecore.journal's ops plus any other section whose
        meta carries 'asset' / '*_asset' values (an app's private kind can reference assets the same way)."""
        refs = {}
        for s in self.sections(upgrade=False):
            if s.get("kind") == ASSET_KIND:
                continue
            for k in journal_asset_refs(s.get("meta") or {}):
                refs.setdefault(k, set()).add(str(s.get("id", "")))
        return refs

    def gc_assets(self, dry_run=False):
        """Delete lecore.asset sections nothing references any more (the natural GC on save leStudio does per
        document, here for the whole workspace). Returns the keys removed (or that would be, with dry_run)."""
        live = set(self.asset_refs()); gone = []
        for s in self.sections(ASSET_KIND, upgrade=False):
            key = (s.get("meta") or {}).get("sha256") or str(s.get("id", ""))[6:]
            if key not in live:
                gone.append(key)
                if not dry_run:
                    self.delete(s["id"])
        return gone

    @classmethod
    def from_file(cls, lews_file, root, app="app", app_version=""):
        """Open a single-file `.lews` (a plain container saved by ANY app -- leStudio's `workspace.lews` download,
        Poly Studio's save) as a live workspace directory. Every section is carried verbatim (foreign kinds included)
        and one journal line records the import, so `changes_since(0)` tells a late joiner where the content came
        from. WHY a classmethod and not a merge: a file is a snapshot with no journal; giving it a directory is what
        turns it into something two apps can hold open."""
        with open(lews_file, "rb") as f:
            cont = load_container(f.read())
        ws = cls(root, app=app, app_version=app_version, create=True)
        ws._acquire()
        try:
            rev = ws.rev() + 1
            ws._write({"meta": ws._file_meta(rev, cont.get("meta") or {}), "sections": list(cont["sections"])})
            ws._journal({"rev": rev, "op": "import", "id": os.path.basename(str(lews_file)), "kind": None, "app": ws.app,
                         "sha": hashlib.sha256(open(lews_file, "rb").read()).hexdigest(), "t": time.time(),
                         "meta": {"sections": len(cont["sections"]), "source_app": (cont.get("meta") or {}).get("app")}})
            return ws
        finally:
            ws._release()

    # ---- the live session: notes, presence, host (LiveSession's contract, realised on the directory)
    PRESENCE = "presence"

    def bump(self, src=None, kind="note", meta=None, touch=True):
        """Announce a change that is NOT a section write -- a selection moved, a render finished, "I am painting layer 3".
        Appends one journal line (no container rewrite) and returns the new rev. Same contract as
        holographic_livesession.LiveSession.bump, so an app written against the in-process session drives this one
        unchanged. Also heart-beats `src`."""
        who = str(src or self.app)
        self._acquire()
        try:
            rev = self.rev() + 1
            self._journal({"rev": rev, "op": "note", "id": "", "kind": str(kind), "app": who, "sha": None,
                           "t": time.time(), "meta": dict(meta or {})})
        finally:
            self._release()
        if touch:
            self.touch(who)                  # touch=False: src is a per-run client id, not a person (see LiveSession.bump)
        return rev

    def since(self, rev, exclude=None):
        """Every change after `rev`, minus `exclude`'s own echo -- LiveSession.since on the journal. WHY exclude: an
        app whose local state already reflects its own writes must not refresh on them (that is how a naive sync
        loop makes the canvas flicker while its own user paints)."""
        out = self.changes_since(rev)
        return [e for e in out if e.get("app") != str(exclude)] if exclude is not None else out

    def _presence_dir(self):
        d = os.path.join(self.root, self.PRESENCE); os.makedirs(d, exist_ok=True); return d

    def _presence_path(self, who):
        safe = hashlib.sha256(str(who).encode()).hexdigest()[:24]       # ids are user-chosen; never trust them as filenames
        return os.path.join(self._presence_dir(), safe + ".json")

    def touch(self, who=None, activity=None, name=None, app=None):
        """Heart-beat a participant. `who` is a PERSON or agent (a persistent id), never a tab or connection -- the
        leStudio lesson is that keying presence by connection gives one person five ghost chips after five reloads.
        `activity` is any JSON dict the app wants peers to see ({tool, section, doc}); `name` a display name.
        The first heartbeat fixes `joined`, which decides the host (see roster). Returns the current rev."""
        who = str(who or self.app); path = self._presence_path(who)
        rec = {"who": who, "app": str(app or self.app), "joined": time.time(), "name": "", "activity": None}
        try:
            with open(path) as f:
                old = json.load(f)
            rec["joined"] = float(old.get("joined", rec["joined"])); rec["name"] = old.get("name", ""); rec["activity"] = old.get("activity")
            if app is None:
                rec["app"] = old.get("app", rec["app"])          # a bare heartbeat keeps the app the participant announced
        except (OSError, ValueError):
            pass
        if name is not None:
            rec["name"] = str(name)[:64]
        if activity is not None:
            rec["activity"] = dict(activity)
        rec["t"] = time.time()
        tmp = path + ".tmp.%d" % os.getpid()
        with open(tmp, "w") as f:
            json.dump(rec, f, sort_keys=True)
        os.replace(tmp, path)
        return self.rev()

    def drop(self, who=None):
        """Remove a participant now (a clean leave). Returns the remaining participants."""
        try:
            os.remove(self._presence_path(str(who or self.app)))
        except OSError:
            pass
        return self.participants()

    def roster(self, ttl=30.0):
        """Who is here, oldest-joined first, each {who, name, app, activity, joined, age, host}. Presence is a
        HEARTBEAT WITH A TIMEOUT (`ttl` seconds), not an open connection -- a polling agent counts, a wedged process
        with an open socket does not. `host` is the earliest-joined participant still alive: a role that survives its
        holder's reloads and passes on only when they are really gone (the leStudio host-flap fix)."""
        now = time.time(); rows = []
        d = self._presence_dir()
        for fn in os.listdir(d):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, fn)) as f:
                    r = json.load(f)
            except (OSError, ValueError):
                continue
            age = now - float(r.get("t", 0))
            if age <= float(ttl):
                rows.append({"who": r.get("who"), "name": r.get("name", ""), "app": r.get("app", ""),
                             "activity": r.get("activity"), "joined": float(r.get("joined", now)), "age": round(age, 2)})
            else:
                try:
                    os.remove(os.path.join(d, fn))        # reap the ghost so the directory cannot grow without bound
                except OSError:
                    pass
        rows.sort(key=lambda r: (r["joined"], r["who"]))
        for i, r in enumerate(rows):
            r["host"] = (i == 0)
        return rows

    def participants(self, ttl=30.0):
        """Just the ids of who is here, oldest-joined first (LiveSession.participants)."""
        return [r["who"] for r in self.roster(ttl)]

    def state(self, ttl=30.0):
        """{name, rev, participants} -- what a status endpoint returns (LiveSession.state)."""
        return {"name": self.root, "rev": self.rev(), "participants": self.participants(ttl)}


def _selftest():
    """Pins: schema stamping and migration; newer-than-known is read-only; two 'apps' on one workspace see each
    other's changes through the journal; expected_rev conflicts; foreign kinds round-trip byte-identical."""
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        painter = Workspace(d, app="lestudio", app_version="t"); modeller = Workspace(d, app="polystudio", app_version="t")
        img = np.zeros((8, 8, 4), np.float32); img[..., 0] = 1.0; img[..., 3] = 1.0
        from holographic.io_and_interop.holographic_container import image_section
        isec = image_section(img, name="red"); isec["id"] = "tex1"
        r1 = painter.put(isec)
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32); F = np.array([[0, 1, 2]], np.int32)
        r2 = modeller.put(mesh_section(V, F, sid="m1", name="tri", uv=np.zeros((3, 2))))
        r3 = modeller.put(scene_section([{"id": "o1", "mesh": "m1", "texture": "tex1", "material": "mat1"}]))
        r4 = modeller.put(material_section("amethyst point", sid="mat1", library="amethyst"))
        assert (r1, r2, r3, r4) == (1, 2, 3, 4)
        seen = painter.changes_since(r1)                                   # the painter catches up on the modeller
        assert [e["kind"] for e in seen] == [MESH_KIND, SCENE_KIND, MATERIAL_KIND]
        sc = painter.get("scene"); assert sc["meta"]["objects"][0]["texture"] == "tex1"
        # the painter repaints; the modeller learns of it by rev
        img[..., 1] = 1.0; isec2 = image_section(img, name="yellow"); isec2["id"] = "tex1"
        r5 = painter.put(isec2)
        ch = modeller.changes_since(r4); assert len(ch) == 1 and ch[0]["id"] == "tex1" and ch[0]["rev"] == r5
        assert float(modeller.get("tex1")["arrays"]["image"][0, 0, 1]) == 1.0
        # optimistic concurrency
        try:
            painter.put(isec2, expected_rev=r1); raise AssertionError("expected a conflict")
        except ConflictError:
            pass
        # schema versioning: register a v2 kind with a migration, read a v1 section through it
        register_kind_schema("test.thing", 2, "test", migrate={1: lambda s: {**s, "meta": {**s["meta"], "b": s["meta"].get("a", 0) * 2}}})
        old = {"kind": "test.thing", "id": "t", "meta": {"a": 3, "schema": 1}, "arrays": {}}
        up = upgrade_section(old); assert up["meta"]["schema"] == 2 and up["meta"]["b"] == 6 and "_read_only" not in up["meta"]
        newer = {"kind": "test.thing", "id": "t", "meta": {"schema": 9}, "arrays": {}}
        assert upgrade_section(newer)["meta"].get("_read_only") is True
        # foreign kind round-trips byte-identical through a put by another app
        foreign = {"kind": "someone.else", "id": "f", "meta": {"x": [1, 2]}, "arrays": {"a": np.arange(5, dtype=np.int16)}}
        modeller.put(foreign); got = painter.get("f", upgrade=False)
        assert got["meta"]["x"] == [1, 2] and np.array_equal(got["arrays"]["a"], foreign["arrays"]["a"]) and got["arrays"]["a"].dtype == np.int16
        d2 = painter.describe(); assert d2["lews"] == LEWS_SPEC and d2["rev"] == 6
        # ---- the live session on the directory: notes advance the rev WITHOUT rewriting the container
        size_before = os.path.getsize(painter.path); mtime_before = os.path.getmtime(painter.path)
        r7 = painter.bump("moose", "selection", {"layer": 3}); assert r7 == 7 and painter.rev() == 7 and modeller.rev() == 7
        assert os.path.getsize(painter.path) == size_before and os.path.getmtime(painter.path) == mtime_before
        assert modeller.since(6) and modeller.since(6)[0]["op"] == "note" and modeller.since(6, exclude="moose") == []
        r8 = modeller.put(material_section("m2", sid="mat2", library="quartz")); assert r8 == 8      # a put continues the SAME counter
        # ---- presence: heartbeat with timeout, host = earliest joined, activity visible across apps
        painter.touch("moose", name="Moose", activity={"tool": "brush", "section": "tex1"}, app="lestudio")
        time.sleep(0.01)
        modeller.touch("agent-7", activity={"tool": "extrude", "section": "m1"}, app="polystudio")
        ro = painter.roster(ttl=30)
        assert [r["who"] for r in ro] == ["moose", "agent-7"] and ro[0]["host"] and not ro[1]["host"]
        assert ro[1]["activity"]["tool"] == "extrude" and ro[0]["name"] == "Moose"
        painter.touch("moose")                                             # a re-heartbeat keeps joined -> host does not flap
        assert painter.roster(ttl=30)[0]["who"] == "moose"
        assert painter.participants(ttl=0.0) == []                         # everyone is stale at ttl 0 (and reaped)
        modeller.touch("agent-7"); assert modeller.participants() == ["agent-7"]
        modeller.drop("agent-7"); assert modeller.participants() == []
        print("lews selftest OK: %d sections, rev %d, journal %d entries, schema v1->v2 migrated, newer read-only, conflict raised, "
              "notes + presence + host on the directory" % (len(d2["sections"]), painter.rev(), len(painter.changes_since(0))))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    _selftest_journal_first()
    _selftest_lestudio_golden()


def _selftest_journal_first():
    """Pins the journal-first doctrine: assets are stored once (a second identical put costs no rev), a journal op may
    not carry an inline array, GC removes exactly the unreferenced assets, and a journal + assets render the same
    pixels twice (the property everything else -- undo, autosave, collaboration -- is built on)."""
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        ws = Workspace(d, app="lestudio")
        tip = np.ones((3, 3), np.float32); k1 = ws.put_asset(tip, "round tip"); r = ws.rev()
        assert ws.put_asset(tip.copy(), "same bytes, other name") == k1 and ws.rev() == r     # stored ONCE, no rev burnt
        k2 = ws.put_asset(np.eye(3, dtype=np.float32), "diag"); assert k2 != k1
        k3 = ws.put_asset(np.zeros((2, 2), np.float32), "orphan")
        ops = [{"op": "stamp", "asset": k1, "x": 1, "y": 1}, {"op": "stamp", "asset": k1, "x": 4, "y": 2},
               {"op": "stamp", "asset": k2, "x": 6, "y": 5, "seed": 7}]
        try:
            journal_section("img", ops + [{"op": "bad", "pixels": np.zeros(2)}]); raise AssertionError("inline array accepted")
        except ValueError:
            pass
        ws.put(journal_section("img", ops))
        assert journal_asset_refs(ops) == {k1, k2}
        assert ws.gc_assets(dry_run=True) == [k3] and ws.gc_assets() == [k3] and ws.get_asset(k3) is None and ws.get_asset(k1) is not None
        # a toy renderer: the journal + assets ARE the picture; two renders agree to the bit
        def render(w):
            J = w.get("journal:img"); img = np.zeros((10, 10), np.float32)
            for o in J["meta"]["ops"]:
                a = w.get_asset(o["asset"]); h, wd = a.shape
                img[o["y"]:o["y"] + h, o["x"]:o["x"] + wd] += a
            return img
        a1 = render(ws); a2 = render(Workspace(d, app="polystudio", create=False))
        assert a1.tobytes() == a2.tobytes() and a1.sum() == 21.0
        # ids: one persisted counter per prefix, shared by both apps, replayable
        p2 = Workspace(d, app="polystudio", create=False)
        assert [ws.mint("L"), p2.mint("L"), ws.mint("O")] == ["L1", "L2", "O1"] and p2.id_counters() == {"L": 2, "O": 1}
        assert [e["op"] for e in ws.changes_since(0)][-3:] == ["mint"] * 3 and ws.rev() == p2.rev()
        # presets: JSON recipes, readable by the other app
        ws.put(preset_section("soft round", "lestudio.brush", {"radius": 12, "flow": 0.12}, tags=["skin"]))
        pr = p2.sections(PRESET_KIND)[0]; assert pr["meta"]["params"]["flow"] == 0.12 and pr["id"] == "preset:lestudio.brush:soft round"
        print("lews journal-first selftest OK: assets deduplicated by content, inline arrays refused, GC removed exactly the orphan, "
              "journal+assets rendered bit-identically from a second app")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _selftest_lestudio_golden():
    """A `.lews` written by leStudio (its golden fixture, kind lestudio.document) opens as a live workspace: every
    section carried, nothing understood and nothing damaged -- exporting gives back a container whose sections
    hash identically. WHY this pin: the standard is only a standard if the file an app ALREADY writes is a valid
    instance of it."""
    import tempfile, shutil
    here = os.path.dirname(os.path.abspath(__file__))
    fx = os.path.join(here, "..", "..", "tests", "fixtures", "lews", "golden_r47.lews")
    if not os.path.exists(fx):
        print("lews golden selftest skipped (fixture not present)"); return
    d = tempfile.mkdtemp()
    try:
        ws = Workspace.from_file(fx, os.path.join(d, "ws"), app="polystudio")
        desc = ws.describe()
        kinds = {s["kind"] for s in desc["sections"]}
        assert "lestudio.document" in kinds
        doc = ws.sections("lestudio.document")[0]; assert not doc["meta"].get("_read_only") and doc["meta"].get("schema", 1) == 1
        ch = ws.changes_since(0); assert ch[0]["op"] == "import" and ch[0]["meta"]["source_app"] == "lestudio"
        before = {s["id"]: section_hash(s) for s in load_container(open(fx, "rb").read())["sections"]}
        ws.put(camera_section([0, 0, 4], [0, 0, 0], sid="cam"))            # another app adds its own section beside it
        after = {s["id"]: section_hash(s) for s in load_container(ws.export_bytes())["sections"]}
        assert all(after[k] == v for k, v in before.items()) and "cam" in after
        print("lews golden selftest OK: leStudio's %d-section file opened, %d foreign kinds carried untouched beside a new camera"
              % (len(before), len(before)))
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    _selftest()

"""holographic_scene_doc.py -- the canonical Scene document (modeling-app backlog, item 0: A + B + E).

The single source of truth a modeling app is built around. Today the "scene" is fragmented: RenderSession holds an
SDF + materials + camera, scenegraph holds a transform hierarchy, anim holds keyframes, the solvers hold sim
state -- so an app builder has to assemble and keep those in sync by hand. This is ONE authoritative, mutable
document that every tool edits and every output (preview, render, sim, query) reads.

It is a VSA TABLE of object records: an object is a role-bound record (name, transform, geometry, material, tags),
and the scene is the set of them plus a hierarchy. Three keystone properties the whole app layer hangs off:

  A. ONE mutable document that OWNS the cross-cutting state (objects, hierarchy, selection, undo history), so
     tools don't each keep their own copy that drifts out of sync.

  B. STABLE handles. Object ids elsewhere are CONTENT hashes (great for storage dedup) -- but a content hash
     CHANGES every time you edit the object, so it breaks as a handle: a selection, a material assignment, or an
     animation target pointing at it would dangle the moment you move a vertex. The fix separates identity from
     content: mint a PERMANENT random hypervector atom as the handle at creation (a globally-unique, stable id
     that survives every edit), and keep the content hash SEPARATELY, used only for dedup.

  E. Change NOTIFICATION. A viewport and its property panels must react when the scene changes. Every mutation
     goes through add/edit/remove/select, which fire registered callbacks -- so the UI stays in sync for free.

Because every mutation goes through those methods, UNDO is automatic too: each records a cheap before/after
snapshot of the one affected record, and undo/redo swap between them -- a thin, readable, O(one-record) stack (the
reversible-delta idea at the record level; it composes with holographic_scenedelta for geometry-level deltas and
holographic_history.VersionedStore for VSA-row versioning when an app wants those). Deterministic (seeded atoms,
hashlib content hashes); NumPy + stdlib only.
"""
import copy
import hashlib
from contextlib import contextmanager

import numpy as np


class _UndoStep:
    """One undoable STEP: a human-readable label plus a batch of (handle, before, after) record snapshots that are
    applied together. A single mutation is a one-change step; a transaction (begin_group/end_group) coalesces many
    mutations into ONE step -- so a drag that fires a hundred edits is a single undo. Undo restores the `before`
    snapshots (in reverse), redo restores the `after` snapshots."""
    __slots__ = ("label", "changes")

    def __init__(self, label, changes):
        self.label = label
        self.changes = changes


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _content_hash(geometry):
    """A deterministic content hash for storage DEDUP -- explicitly NOT identity. It changes whenever the geometry
    changes, which is exactly why it must not be used as a handle (identity has to survive edits). hashlib, never
    Python's hash() (the determinism rule)."""
    h = hashlib.sha256()
    if geometry is None:
        h.update(b"none")
    elif isinstance(geometry, np.ndarray):
        h.update(np.ascontiguousarray(geometry).tobytes())
    else:
        h.update(repr(geometry).encode())
    return h.hexdigest()[:16]


class SceneObject:
    """One object in the scene, as a role-bound RECORD: a stable handle plus its properties. Tools mutate these
    through Scene.edit (never directly) so undo and notifications are automatic. `tags` are bound roles
    (role -> value); `params` are named parameters."""

    def __init__(self, handle, name, transform, geometry=None, material=None, tags=None, params=None,
                 overrides=None, parent=None):
        self.handle = handle          # STABLE identity (survives every edit) -- what selections/materials refer to
        self.name = name
        self.transform = np.eye(4) if transform is None else np.asarray(transform, float)
        self.geometry = geometry
        self.material = material
        self.tags = dict(tags) if tags else {}
        self.params = dict(params) if params else {}
        self.overrides = dict(overrides) if overrides else {}   # per-object RENDER overrides (a bound role w/ fallback)
        self.parent = parent          # the parent handle in the hierarchy (None = top level) -- ON the record, so undoable

    def __repr__(self):
        return "SceneObject(%s, name=%r, tags=%r)" % (self.handle, self.name, list(self.tags))


class Scene:
    """The single source of truth: a table of object records + a hierarchy, owning selection and undo history and
    firing change events. Edit ONLY through add/edit/remove/select -- that is what makes undo and notifications
    free and keeps every tool reading one coherent document."""

    _EDITABLE = ("name", "transform", "geometry", "material", "tags", "params", "overrides")

    def __init__(self, dim=1024, seed=0):
        self.dim = int(dim)
        self.objects = {}         # handle -> SceneObject   (the authoritative store)
        self.cameras = {}         # name -> camera record
        self.lights = {}          # name -> light record
        self.selection = set()    # the current selection (a set of handles)
        self._identity = {}       # handle -> permanent random atom (B: holographic identity, content-independent)
        self._content_key = {}    # handle -> content hash (dedup ONLY; changes on edit)
        self._callbacks = []      # change-notification callbacks (E)
        self._undo = []           # history: a list of _UndoStep (each = a label + a batch of record snapshots)
        self._redo = []
        self._group = None        # while a transaction is open, mutations accumulate here (a drag -> one undo)
        self._group_label = None
        self._group_depth = 0     # nesting count, so nested groups commit as ONE step
        self._max_undo = 200      # cap the history depth (drop the oldest step past this)
        self._rng = np.random.default_rng(seed)
        self._n = 0

    # -- change notification (E) -----------------------------------------------------------------------------
    def on_change(self, callback):
        """Register callback(kind, handle), fired on every change. kind is one of
        add / edit / remove / select / undo / redo. This is how a viewport or property panel stays in sync."""
        self._callbacks.append(callback)
        return callback

    def _notify(self, kind, handle):
        for cb in self._callbacks:
            cb(kind, handle)

    # -- stable handles (B) ----------------------------------------------------------------------------------
    def _mint_handle(self):
        """A permanent, content-independent identity: a stable string id PLUS a permanent random hypervector atom.
        The app refers to the string; the atom is the holographic identity (for labelled bundles, the query layer,
        the Sampler's overlap handling). Neither changes when the object is edited."""
        h = "obj_%08d" % self._n
        self._n += 1
        self._identity[h] = _unit(self._rng.standard_normal(self.dim))
        return h

    def handle_vector(self, handle):
        """The object's permanent identity atom -- its holographic handle (used for labelled bundles / query)."""
        return self._identity[handle]

    # -- mutations (all go through here, so undo + notify are automatic) --------------------------------------
    def add(self, name=None, transform=None, geometry=None, material=None, tags=None, params=None, parent=None,
            overrides=None, _record=True):
        """Add an object; return its STABLE handle. Records an undo entry and fires an 'add' event."""
        h = self._mint_handle()
        obj = SceneObject(h, name or h, transform, geometry, material, tags, params, overrides, parent)
        self.objects[h] = obj
        self._content_key[h] = _content_hash(geometry)
        if _record:
            self._push(h, None, self._snapshot(obj), "Add " + (name or h))   # inverse of add is remove (before = None)
        self._notify("add", h)
        return h

    def edit(self, handle, _record=True, **changes):
        """Mutate an object's fields (name/transform/geometry/material/tags/params). The HANDLE IS UNCHANGED --
        identity survives the edit (B) -- while the content hash is refreshed for dedup. Records undo + notifies.
        tags/params MERGE into the existing dicts; the rest replace."""
        obj = self.objects[handle]
        before = self._snapshot(obj)
        for k, v in changes.items():
            if k not in self._EDITABLE:
                raise KeyError("not an editable field: %r" % k)
            if k in ("tags", "params", "overrides"):
                getattr(obj, k).update(v)
            elif k == "transform":
                obj.transform = np.asarray(v, float)
            else:
                setattr(obj, k, v)
        self._content_key[handle] = _content_hash(obj.geometry)   # content hash tracks edits; the HANDLE does not
        if _record:
            self._push(handle, before, self._snapshot(obj), "Edit " + obj.name)
        self._notify("edit", handle)

    def remove(self, handle, _record=True):
        """Remove an object (and drop it from the selection/hierarchy). Records undo + notifies."""
        before = self._snapshot(self.objects[handle])
        del self.objects[handle]
        self.selection.discard(handle)
        if _record:
            self._push(handle, before, None, "Remove " + before.name)   # inverse of remove is re-add (after = None)
        self._notify("remove", handle)

    def remove_tag(self, handle, key, _record=True):
        """Remove one tag from an object. (edit() MERGES tags, so it can add or change a tag but not delete one --
        this is the delete. Records undo + fires a change event, like any mutation.)"""
        obj = self.objects[handle]
        if key not in obj.tags:
            return
        before = self._snapshot(obj)
        del obj.tags[key]
        if _record:
            self._push(handle, before, self._snapshot(obj), "Untag " + obj.name)
        self._notify("edit", handle)

    def clear_override(self, handle, prop, _record=True):
        """Remove one render override from an object, so it FALLS BACK to the scene default. (edit MERGES
        overrides; this is the delete -- same undo + notify as any mutation.)"""
        obj = self.objects[handle]
        if prop not in obj.overrides:
            return
        before = self._snapshot(obj)
        del obj.overrides[prop]
        if _record:
            self._push(handle, before, self._snapshot(obj), "Clear override " + obj.name)
        self._notify("edit", handle)

    def select(self, handles):
        """Set the current selection (a set of handles) and fire a 'select' event."""
        self.selection = set(handles)
        self._notify("select", None)

    def get(self, handle):
        return self.objects[handle]

    def __len__(self):
        return len(self.objects)

    # -- the hierarchy ---------------------------------------------------------------------------------------
    def set_parent(self, child, parent, _record=True):
        """Parent `child` under `parent` (both handles; parent=None for top level). The parent lives ON the child
        record, so re-parenting is an ordinary undoable edit. World transforms compose up the chain
        (holographic_scenegraph builds the SceneNode tree when a renderer needs flattened world matrices).

        REFUSES a parent that would create a CYCLE -- i.e. `parent` is `child` itself or a descendant of `child`.
        WHY this is not optional: flatten_scene walks parent -> child recursively, so a cycle (A under B under A)
        is an infinite recursion / stack overflow the first time anything asks for a world matrix. A modeling app
        creates this constantly by mis-dragging a group into its own member, so the guard belongs in the one place
        every re-parent goes through, not in each caller. (Regression-trapped: parenting a group under its own
        descendant used to succeed and cycle.)"""
        if parent is not None:
            if parent == child:
                raise ValueError("cannot parent an object under itself")
            # walk UP from the proposed parent; if we reach `child`, `child` is an ancestor of `parent`, so
            # making `child`'s parent = `parent` would close a loop.
            p = parent
            while p is not None:
                if p == child:
                    raise ValueError("cannot parent %r under its own descendant (would create a cycle)"
                                     % self.objects[child].name)
                p = self.objects[p].parent if p in self.objects else None
        obj = self.objects[child]
        before = self._snapshot(obj)
        obj.parent = parent
        if _record:
            self._push(child, before, self._snapshot(obj), "Parent " + obj.name)
        self._notify("edit", child)

    def parent_of(self, handle):
        """The parent handle of an object (None if top level)."""
        return self.objects[handle].parent

    def children_of(self, handle):
        return [h for h, o in self.objects.items() if o.parent == handle]

    # -- undo / redo (a thin snapshot-swap stack: O(one record), not the whole scene) ------------------------
    def _snapshot(self, obj):
        """A cheap snapshot of a record's editable state for undo: copy the mutable fields (the transform array and
        the small dicts) but keep geometry/material BY REFERENCE -- they are replaced wholesale on edit, so a
        reference is enough and we avoid copying big meshes."""
        snap = copy.copy(obj)
        snap.transform = obj.transform.copy()
        snap.tags = dict(obj.tags)
        snap.params = dict(obj.params)
        snap.overrides = dict(obj.overrides)
        return snap

    def _push(self, handle, before, after, label="Edit"):
        """Record one record change. If a transaction is open (begin_group), accumulate it into that group so the
        whole batch becomes ONE undo step; otherwise commit it as a single-change step immediately."""
        change = (handle, before, after)
        if self._group is not None:
            self._group.append(change)                        # inside a transaction -> coalesce into one step
        else:
            self._commit_step(_UndoStep(label, [change]))

    def _commit_step(self, step):
        self._undo.append(step)
        self._redo.clear()                                    # a fresh edit invalidates the redo branch
        if self._max_undo and len(self._undo) > self._max_undo:
            self._undo.pop(0)                                 # depth cap: drop the oldest step

    def _restore(self, handle, snapshot):
        if snapshot is None:
            self.objects.pop(handle, None)                    # object did not exist in this state
            self.selection.discard(handle)
        else:
            self.objects[handle] = self._snapshot(snapshot)   # install a FRESH copy (later edits won't touch the stored snap)
            self._content_key[handle] = _content_hash(snapshot.geometry)

    # -- transactions: coalesce many mutations into one undo step (a drag = one undo) ------------------------
    def begin_group(self, label="Edit"):
        """Open a transaction: every mutation until end_group() coalesces into ONE undo step labelled `label`.
        Nesting is allowed -- only the OUTERMOST group commits (so composing grouped operations still yields one
        step). Prefer the `group` context manager below."""
        if self._group is None:
            self._group = []
            self._group_label = label
        self._group_depth += 1

    def end_group(self):
        """Close the current transaction; commit the accumulated changes as a single step (nothing if empty)."""
        if self._group_depth > 0:
            self._group_depth -= 1
        if self._group_depth == 0 and self._group is not None:
            changes = self._group
            label = self._group_label or "Edit"
            self._group = None
            self._group_label = None
            if changes:
                self._commit_step(_UndoStep(label, changes))

    @contextmanager
    def group(self, label="Edit"):
        """`with scene.group("Move wheels"): ...` -- everything inside becomes one undo step. The readable way to
        make a multi-edit operation (a drag, a tool that edits several objects) a single undo."""
        self.begin_group(label)
        try:
            yield
        finally:
            self.end_group()

    # -- undo / redo (each step is applied as a whole) -------------------------------------------------------
    def undo(self):
        """Undo the last STEP by restoring its BEFORE snapshots (in reverse order within the step). Identity atoms
        are never touched, so handles stay valid across undo/redo. Returns False if there is nothing to undo."""
        if not self._undo:
            return False
        step = self._undo.pop()
        for handle, before, after in reversed(step.changes):
            self._restore(handle, before)
        self._redo.append(step)
        self._notify("undo", None)
        return True

    def redo(self):
        """Redo the last undone step by restoring its AFTER snapshots."""
        if not self._redo:
            return False
        step = self._redo.pop()
        for handle, before, after in step.changes:
            self._restore(handle, after)
        self._undo.append(step)
        self._notify("redo", None)
        return True

    # -- history view (what an Edit menu / history panel shows) ----------------------------------------------
    def history(self):
        """The undo stack's step labels, oldest first."""
        return [s.label for s in self._undo]

    def redo_history(self):
        """The redo stack's step labels (most-recently-undone last)."""
        return [s.label for s in self._redo]

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)


def _selftest():
    """A handle is stable across edits (identity != content -- the key B guarantee); every mutation fires a change
    event (E); undo/redo restore state and preserve identity; a selection survives an edit of the selected object;
    deterministic."""
    scene = Scene(dim=256, seed=0)

    # capture the change stream (E)
    events = []
    scene.on_change(lambda kind, h: events.append((kind, h)))

    # (A) add objects through the one document; each gets a STABLE handle
    a = scene.add(name="wheel", geometry=np.zeros((4, 3)), tags={"material": "metal"})
    b = scene.add(name="body", geometry=np.ones((4, 3)), tags={"material": "paint"})
    assert len(scene) == 2 and ("add", a) in events and ("add", b) in events

    # (B) THE keystone: editing the geometry changes the CONTENT hash but NOT the handle or its identity atom, so
    # a selection / material assignment pointing at the handle still resolves. This is why content hashes can't be
    # handles and why identity is a separate permanent atom.
    key_before = scene._content_key[a]
    id_before = scene.handle_vector(a).copy()
    scene.select([a])                                          # select by handle
    scene.edit(a, geometry=np.full((4, 3), 7.0), name="front-wheel")
    assert scene._content_key[a] != key_before                # content hash changed with the edit
    assert np.array_equal(scene.handle_vector(a), id_before)   # ...but identity is UNCHANGED
    assert a in scene.selection                                # ...so the selection still points at it
    assert scene.get(a).name == "front-wheel"

    # (E) the edit fired a change event
    assert ("edit", a) in events and ("select", None) in events

    # undo/redo: the edit reverts and re-applies; the handle stays valid throughout
    scene.undo()
    assert scene.get(a).name == "wheel" and scene._content_key[a] == key_before
    assert np.array_equal(scene.handle_vector(a), id_before)   # identity preserved across undo
    scene.redo()
    assert scene.get(a).name == "front-wheel"

    # undo of an ADD removes the object; redo brings it back with the SAME handle (identity intact)
    c = scene.add(name="mirror", geometry=np.zeros((2, 3)))
    assert c in scene.objects
    scene.undo()
    assert c not in scene.objects
    id_c = scene.handle_vector(c).copy()
    scene.redo()
    assert c in scene.objects and np.array_equal(scene.handle_vector(c), id_c)

    # remove + undo restores the record
    scene.remove(b)
    assert b not in scene.objects
    scene.undo()
    assert b in scene.objects and scene.get(b).name == "body"

    # hierarchy: parent one object under another
    scene.set_parent(a, b)
    assert a in scene.children_of(b)

    # CYCLE GUARD (regression): b is now an ancestor of a, so parenting b under a must be refused (else flatten
    # recurses forever). Self-parenting must be refused too. A legal re-parent still works.
    try:
        scene.set_parent(b, a); raise AssertionError("cycle should have been refused")
    except ValueError:
        pass
    try:
        scene.set_parent(a, a); raise AssertionError("self-parent should have been refused")
    except ValueError:
        pass
    scene.set_parent(a, None); assert scene.parent_of(a) is None   # legal re-parent to top level still allowed

    # determinism: two scenes with the same seed mint the same identity atoms in order
    s2 = Scene(dim=256, seed=0); a2 = s2.add(name="wheel")
    s3 = Scene(dim=256, seed=0); a3 = s3.add(name="something-else")
    assert np.array_equal(s2.handle_vector(a2), s3.handle_vector(a3))

    print("holographic_scene_doc selftest OK: one document owns objects/selection/history; handles are STABLE "
          "across edits (content hash changed, identity atom and handle did not, so the selection survived); every "
          "mutation fires a change event; undo/redo restore state and preserve identity; hierarchy parenting works")


if __name__ == "__main__":
    _selftest()

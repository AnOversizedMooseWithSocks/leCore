"""The creature EDITOR session -- the API a Spore-like app drives: edit, undo, save, validate, build.

WHY THIS MODULE EXISTS
----------------------
Everything underneath was already there: spine editing, thickness profiles, metaball skin, symmetry
groups, anatomy-space sockets, skin weights, paint, materials. What was missing is the thing that
turns a pile of faculties into an APPLICATION -- a session object that holds the document, records
every change so it can be taken back, writes itself to disk and reads back identical, says whether
what you have built is valid, and hands you geometry when you ask.

An app cannot be built on a set of pure functions alone. It needs a document with a history.

DESIGN: THE DOCUMENT IS A PLAIN DICT, AND EVERY EDIT RETURNS A NEW ONE
    The spine edits already worked this way, and it pays for itself here three times over:
      undo      is a snapshot swap -- correct by construction, not by carefully inverting each edit
      save      is json.dumps, because the document is already JSON-shaped
      preview   cannot corrupt the asset, because nothing mutates
    A spec is small (a few hundred bytes), so snapshotting per edit is cheap. If it ever stops being
    cheap the fix is a diff, NOT making the editor stateful.

REUSE
    The shipped `EditHistory` takes any command with .apply(state)/.invert(state), which is a fully
    general protocol -- so undo/redo is its machinery, not a second implementation. Geometry comes
    from creature/creatureskin/creaturesocket/creatureparts; nothing is re-derived here.

KEPT NEGATIVES (loud)
  * VALIDATION IS STRUCTURAL, NOT AESTHETIC. It catches a spine with no nodes, a limb anchored off
    the body, a socket whose ray misses the skin, a part the library does not define. It has no
    opinion on whether the creature can walk, balance, or survive -- those need the gait arc, which
    is not built.
  * THE COMPLEXITY BUDGET IS A COUNT, NOT A COST MODEL. It counts spine nodes, limb segments and
    parts against a cap so an app can have Spore's "you have used N of your DNA". It does not
    predict render time or memory, and pretending it did would be a fabricated number.
  * JSON ROUND-TRIP IS EXACT FOR THE DOCUMENT, not for derived geometry. Reloading gives back the
    same spec and therefore the same creature -- but floats go through repr, so a spec is portable,
    not bit-preserved across arbitrary precision changes. The selftest asserts the rebuild is
    identical, which is the property that matters.
  * NO MULTI-USER / CONCURRENT EDITING. One document, one history.
"""

import copy
import json

import numpy as np


def _foot_scale(library, part, spec, width_ratio):
    """A foot sized to the LIMB it sits on, measured rather than assumed: the part's own default
    extent against the mean limb radius, so a thin-legged creature does not get boots."""
    base_w = 1.0
    if library is not None:
        geo = getattr(library, "parts", {}).get(str(part), {}).get("geometry")
        if geo is not None:
            V = np.asarray(geo.vertices, float)
            base_w = float(np.max(V.max(0) - V.min(0))) or 1.0
    rs = [float(l.get("radius", 0.05)) for l in (spec.get("limbs") or [])]
    return float(width_ratio) * (float(np.mean(rs)) if rs else 0.05) / base_w


class SpecCommand:
    """One reversible edit to the creature document.

    Deliberately a SNAPSHOT command rather than a hand-written inverse per operation. Inverting
    "extend the spine by 2" sounds easy and is not -- it has to restore the radius profile that was
    resampled, the limb fractions that were renormalised, and anything else the edit touched. A
    snapshot cannot get that wrong, and for a document this small the memory is irrelevant. Matches
    the shipped EditHistory protocol (.apply / .invert), so the undo machinery is reused entirely.
    """

    __slots__ = ("name", "after", "before")

    def __init__(self, name, before, after):
        self.name = str(name)
        self.before = copy.deepcopy(before)
        self.after = copy.deepcopy(after)

    def apply(self, state):
        """Return the document as it is after this edit."""
        return copy.deepcopy(self.after)

    def invert(self, state):
        """Return the document as it was before this edit."""
        return copy.deepcopy(self.before)


class CreatureEditor:
    """A live creature editing session: document + history + geometry, with everything an app needs.

    Every mutating method records a command and returns self, so a UI can chain calls and still undo
    each one separately. `spec` is always a plain dict you may read, serialise, or hand to any of the
    underlying faculties directly -- the editor owns the history, not the data.
    """

    def __init__(self, spec=None, max_depth=256, part_library=None):
        from holographic.mesh_and_geometry.holographic_creature import quadruped_spec
        from holographic.mesh_and_geometry.holographic_edithistory import EditHistory
        self.spec = normalise(copy.deepcopy(spec) if spec is not None else quadruped_spec())
        self.spec.setdefault("sockets", [])
        self.history = EditHistory(max_depth=int(max_depth))
        self.library = part_library

    # ------------------------------------------------------------------ history --
    def _commit(self, name, new_spec):
        """Record an edit and adopt its result. One funnel, so no operation can accidentally skip the
        history and leave a change the user cannot take back."""
        new_spec = normalise(new_spec)
        new_spec.setdefault("sockets", list(self.spec.get("sockets", [])))
        self.spec = self.history.do(self.spec, SpecCommand(name, self.spec, new_spec))
        return self

    def undo(self):
        """Take back the last edit. A no-op when there is nothing to undo."""
        self.spec = self.history.undo(self.spec)
        return self

    def redo(self):
        """Re-apply the last undone edit."""
        self.spec = self.history.redo(self.spec)
        return self

    def can_undo(self):
        """Whether there is anything to take back -- what a UI greys the button on."""
        return bool(self.history.can_undo())

    def can_redo(self):
        """Whether there is an undone edit to re-apply."""
        return bool(self.history.can_redo())

    # ------------------------------------------------------------------- body --
    def extend_spine(self, n=1, keep_segment_length=True):
        """Add `n` segments to the tail -- the 'drag the tail out' edit."""
        import holographic.mesh_and_geometry.holographic_creatureskin as cs
        return self._commit("extend_spine",
                            cs.extend_spine(self.spec, n=n, keep_segment_length=keep_segment_length))

    def insert_spine_node(self, at=0.5):
        """Subdivide the spine: more resolution, same shape."""
        import holographic.mesh_and_geometry.holographic_creatureskin as cs
        return self._commit("insert_spine_node", cs.insert_node(self.spec, at=at))

    def set_thickness(self, at, radius, falloff=0.0):
        """Thicken or thin the body at a fraction along the spine; `falloff` blends into neighbours."""
        import holographic.mesh_and_geometry.holographic_creatureskin as cs
        return self._commit("set_thickness", cs.set_radius(self.spec, at, radius, falloff=falloff))

    def reshape_spine(self, curve=None, length=None, axis=None):
        """Reshape the body as a whole: its arch, length or axis."""
        import holographic.mesh_and_geometry.holographic_creatureskin as cs
        return self._commit("reshape_spine",
                            cs.move_node(self.spec, curve=curve, length=length, axis=axis))

    def set_profile(self, radii):
        """Replace the whole thickness profile at once (one radius per spine node, or a callable)."""
        import holographic.mesh_and_geometry.holographic_creatureskin as cs
        return self._commit("set_profile", cs.spine_profile(self.spec, radii))

    def add_limb(self, at=0.5, direction=(1.0, -1.0, 0.0), segments=3, length=0.6, radius=0.05,
                 mirror=True):
        """Attach a limb chain at fraction `at` along the spine."""
        s = copy.deepcopy(self.spec)
        s.setdefault("limbs", []).append({"at": float(at), "dir": list(direction),
                                          "segments": int(segments), "length": float(length),
                                          "radius": float(radius), "mirror": bool(mirror)})
        return self._commit("add_limb", s)

    def remove_limb(self, index):
        """Remove a limb by index."""
        s = copy.deepcopy(self.spec)
        limbs = s.get("limbs", [])
        if not 0 <= int(index) < len(limbs):
            raise IndexError("no limb %d (have %d)" % (index, len(limbs)))
        limbs.pop(int(index))
        return self._commit("remove_limb", s)

    # ------------------------------------------------------------------ parts --
    def add_part(self, part, t, theta, symmetry="bilateral", n=2, scale=1.0, handles=None):
        """Socket a part onto the body at ANATOMY coordinates (t along the spine, theta around it).

        Stored as (t, theta) rather than a world position, which is what lets the part ride the skin
        through every later spine and thickness edit.
        """
        s = copy.deepcopy(self.spec)
        s.setdefault("sockets", []).append({"part": str(part), "t": float(t), "theta": float(theta),
                                            "symmetry": str(symmetry), "n": int(n),
                                            "scale": float(scale), "handles": dict(handles or {})})
        return self._commit("add_part:%s" % part, s)

    def add_part_at_point(self, part, point, creature=None, **kw):
        """Socket a part at a WORLD point -- the click-to-place path. Converts the point to (t, theta)
        first, so what gets stored is still anatomy-space."""
        import holographic.mesh_and_geometry.holographic_creaturesocket as sk
        cr = creature if creature is not None else self.creature()
        s = sk.socket_at_point(cr, point)
        return self.add_part(part, s["t"], s["theta"], **kw)

    def remove_part(self, index):
        """Remove a socketed part by index."""
        s = copy.deepcopy(self.spec)
        socks = s.get("sockets", [])
        if not 0 <= int(index) < len(socks):
            raise IndexError("no socket %d (have %d)" % (index, len(socks)))
        socks.pop(int(index))
        return self._commit("remove_part", s)

    def move_part(self, index, t=None, theta=None):
        """Slide a socketed part along or around the body."""
        s = copy.deepcopy(self.spec)
        sock = s["sockets"][int(index)]
        if t is not None:
            sock["t"] = float(np.clip(t, 0.0, 1.0))
        if theta is not None:
            sock["theta"] = float(theta)
        return self._commit("move_part", s)

    def add_feet(self, part="foot", scale=None, digits=3, only_legs=True, ground_frac=0.35,
                 width_ratio=2.2):
        """Put a FOOT at the tip of every LEG -- what makes a creature stand rather than taper away.

        Uses the gait analyzer to decide which limbs are legs (by asking which reach the ground), so
        a body with arms up gets feet only where they belong and a hexapod gets six without anything
        authored. Sockets are derived from each tip via socket_at_point, so feet are stored in ANATOMY
        coordinates like every other part and ride the body through later edits.

        `scale=None` SIZES THE FOOT FROM THE LEG rather than from a magic constant. Measured on a real
        body, the old scale=1.0 default gave a foot 3.2x the limb RADIUS and 0.4x the whole leg's
        reach -- feet visibly larger than the torso, which is what dogfooding actually rendered. The
        foot's width is now `width_ratio` times the limb radius, so a chunky leg gets a chunky foot
        and a spindly one does not. Same class of error as a texture frequency in world units: an
        ABSOLUTE size where a RELATIVE one was needed.

        ONE undoable step even though it may add six parts: "add feet" is one action to a user, so it
        is one entry in the history.
        """
        import copy as _copy
        import holographic.mesh_and_geometry.holographic_gait as _g
        import holographic.mesh_and_geometry.holographic_creaturesocket as _sk
        cr = self.creature()
        rig = _g.analyze_rig(cr, ground_frac=ground_frac)
        chains = rig["legs"] if only_legs else list(cr.chains)
        s = _copy.deepcopy(self.spec)
        socks = s.setdefault("sockets", [])
        # the part's own default extent, so the ratio is measured rather than assumed
        base_w = 1.0
        if self.library is not None:
            geo = getattr(self.library, "parts", {}).get(str(part), {}).get("geometry")
            if geo is not None:
                V = np.asarray(geo.vertices, float)
                base_w = float(np.max(V.max(0) - V.min(0))) or 1.0
        # DELEGATE to auto_feet, which emits LIMB sockets. This method used to build SPINE-relative
        # sockets via socket_at_point -- the approximation already on record as a kept negative ("a
        # socket on a limb tip uses the nearest spine station and is approximate"). A foot belongs to
        # its LIMB's axis, not the spine's: resolve_limb_socket casts along_axis=True, which is what a
        # foot actually needs, and place_parts already dispatches on the socket kind. So this is not
        # deduplication for tidiness -- the spine-relative version was the WORSE of the two.
        sc = float(scale) if scale is not None else _foot_scale(self.library, part, s, width_ratio)
        for sock in _sk.auto_feet(cr, self.field(), part=part, scale=sc,
                                  ground_frac=ground_frac,
                                  handles={"digits": float(digits)}):
            if only_legs or sock.get("limb") in cr.chains:
                socks.append(dict(sock))
        chains = [k["limb"] for k in socks if k.get("part") == str(part)]
        self._added_feet = len(chains)
        return self._commit("add_feet:%d" % len(chains), s)

    # ------------------------------------------------------------------ build --
    def creature(self):
        """The rig for the current document."""
        from holographic.mesh_and_geometry.holographic_creature import Creature
        return Creature(self.spec)

    def field(self, spacing=0.9):
        """The current skin as a distance field -- what picking, sockets and the renderer all take."""
        from holographic.mesh_and_geometry.holographic_creatureskin import creature_field
        return creature_field(self.creature(), self.spec, spacing=spacing)

    def build(self, resolution=48, spacing=0.9, with_parts=True, mode="merge"):
        """Everything the document describes, as geometry: {"skin", "parts", "placements", "missed"}.

        `missed` names any socket whose outward ray never found the skin, so an app can flag the part
        in the UI instead of leaving the user wondering where it went.
        """
        import holographic.mesh_and_geometry.holographic_creatureskin as cs
        import holographic.mesh_and_geometry.holographic_creaturesocket as sk
        from holographic.mesh_and_geometry.holographic_meshbridge import (
            sample_field as _sample_field, marching_tetrahedra as _march)
        cr = self.creature()
        # MESH THE SAME SURFACE THE SOCKETS RESOLVE ON. `field()` returns the DISTANCE form and
        # `creature_metaball_mesh` meshes the DENSITY form -- and their zero level sets are NOT the
        # same surface. Measured on a real body: density crosses zero at r=0.2651, distance at
        # r=0.4195 -- a 0.154 gap. Parts therefore reported "placed, 0 missed" (they sat exactly on
        # the distance surface, field-dist 0.00000) while rendering 0.15 OFF the skin, visibly
        # floating. Two doors to "the creature's surface" that disagree is worse than one door.
        fld = cs.creature_field(cr, self.spec, spacing=spacing)
        lo, hi = fld.bounds()
        vals, axes = _sample_field(fld, (tuple(lo), tuple(hi)), int(resolution))
        skin = _march(vals, axes, level=0.0)
        out = {"skin": skin, "parts": None, "placements": [], "missed": []}
        if with_parts and self.spec.get("sockets"):
            r = sk.place_parts(cr, fld, self.spec["sockets"], self.library, mode=mode)
            out.update({"parts": r["geometry"], "placements": r["placements"], "missed": r["missed"]})
        return out

    def render(self, width=640, height=640, resolution=96, direction=(1.0, -1.1, 0.35),
               fov_deg=40.0, background=(0.08, 0.09, 0.12), paint=True, seed=0, **kw):
        """BUILD AND RENDER IN ONE CALL -- skin and parts merged, camera framed, body painted.

        Found by dogfooding: rendering a creature took six manual steps that every caller had to
        rediscover, and two of them were traps.
          * `build()` returns skin and parts SEPARATELY, so every caller hand-merged them with an
            index offset -- easy to get wrong and pure boilerplate.
          * `fit_camera()` returns a DICT while `render_mesh` wants a Camera OBJECT, so every caller
            wrote the same four-field conversion.
          * an unpainted creature renders pure WHITE, which reads as a bug rather than a default.
        Returns (image, mesh) so the mesh is still available for anything else.
        """
        from holographic.mesh_and_geometry.holographic_mesh import Mesh
        from holographic.rendering.holographic_render import Camera, Light
        import holographic.mesh_and_geometry.holographic_creatureskin as cs
        out = self.build(resolution=resolution)
        SV = np.asarray(out["skin"].vertices, float)
        SF = np.asarray(out["skin"].faces, int)
        if out["parts"] is not None:
            PV = np.asarray(out["parts"].vertices, float)
            PF = np.asarray(out["parts"].faces, int)
            mesh = Mesh(np.vstack([SV, PV]), np.vstack([SF, PF + len(SV)]))
        else:
            PV = np.zeros((0, 3)); mesh = out["skin"]
        cols = None
        if paint:
            cr = self.creature()
            C, R, bones = cs.creature_metaballs(cr, self.spec, spacing=0.9)
            from holographic.mesh_and_geometry.holographic_creatureparts import skin_weights
            from holographic.mesh_and_geometry.holographic_paintlod import paint_creature
            idx, w, names, _bk = skin_weights(SV, C, R, bones, dim=256, seed=seed)
            body = paint_creature(SV, idx, w, names, seed=seed)
            cols = np.vstack([body, np.tile(np.array([0.86, 0.82, 0.70]), (len(PV), 1))]) \
                if len(PV) else body
        cam = self._mind_fit_camera(mesh, direction, fov_deg, width, height)
        from holographic.rendering.holographic_render import rasterize_mesh as _render_mesh
        img = _render_mesh(mesh, cam, width=width, height=height, vertex_colors=cols,
                           lights=[Light("directional", direction=(-0.5, 0.6, -0.5), intensity=1.2),
                                   Light("directional", direction=(0.7, 0.3, -0.2), intensity=0.4)],
                           ambient=0.45, background=background, smooth=True, **kw)
        return np.asarray(img), mesh

    def _mind_fit_camera(self, mesh, direction, fov_deg, width, height):
        """fit_camera returns a dict; render_mesh wants a Camera. One place to convert, not every
        call site -- this conversion was the second friction point dogfooding surfaced."""
        from holographic.rendering.holographic_render import fit_camera as _fit
        from holographic.io_and_interop.holographic_coerce import as_camera
        c = _fit(mesh, direction=direction, up=(0.0, 0.0, 1.0), fov_deg=fov_deg,
                 aspect=float(width) / max(float(height), 1.0), margin=1.10)
        # `as_camera` already coerces fit_camera's DICT into a Camera. Dogfooding logged this
        # conversion as friction before finding it -- the capability existed, the discoverability did
        # not, which is the failure mode this engine names most often.
        return as_camera(c)

    # --------------------------------------------------------- validate & budget --
    def validate(self):
        """Is this document buildable? Returns {"ok", "errors", "warnings"}.

        ERRORS are things that cannot be built; WARNINGS are buildable but probably not what was
        meant (a part whose ray misses the body, a limb longer than the whole spine). Structural
        only -- see the module's kept negative before expecting it to judge whether a creature works.
        """
        errors, warnings = [], []
        sp = self.spec.get("spine") or {}
        if int(sp.get("segments", 0)) < 1:
            errors.append("spine has no segments")
        if float(sp.get("length", 0.0)) <= 0.0:
            errors.append("spine length must be positive")
        prof = sp.get("profile")
        if prof is not None:
            if len(prof) != int(sp.get("segments", 0)) + 1:
                errors.append("thickness profile has %d entries, spine needs %d"
                              % (len(prof), int(sp.get("segments", 0)) + 1))
            elif min(prof) <= 0:
                errors.append("thickness profile has a non-positive radius")
        for i, limb in enumerate(self.spec.get("limbs", []) or []):
            if not 0.0 <= float(limb.get("at", -1)) <= 1.0:
                errors.append("limb %d is anchored at t=%s, outside the spine" % (i, limb.get("at")))
            if int(limb.get("segments", 0)) < 1:
                errors.append("limb %d has no segments" % i)
            if float(limb.get("length", 0)) > 3.0 * float(sp.get("length", 1.0)):
                warnings.append("limb %d is much longer than the body" % i)
        for i, s in enumerate(self.spec.get("sockets", []) or []):
            if not 0.0 <= float(s.get("t", -1)) <= 1.0:
                errors.append("socket %d has t=%s, outside the spine" % (i, s.get("t")))
            if self.library is not None and s.get("part") not in getattr(self.library, "parts", {}):
                errors.append("socket %d wants part %r, which the library does not define"
                              % (i, s.get("part")))
        if not errors:
            # Only worth resolving rays once the document is structurally sound.
            try:
                import holographic.mesh_and_geometry.holographic_creaturesocket as sk
                cr, fld = self.creature(), self.field()
                for i, s in enumerate(self.spec.get("sockets", []) or []):
                    if not sk.resolve_socket(cr, fld, s["t"], s.get("theta", 0.0))["hit"]:
                        warnings.append("socket %d (%s) does not reach the skin" % (i, s.get("part")))
            except Exception as e:                            # a build failure IS an error
                errors.append("the document does not build: %s" % type(e).__name__)
        return {"ok": not errors, "errors": errors, "warnings": warnings}

    def complexity(self, cap=100):
        """Spore's 'DNA points': what this creature spends, and whether it is within `cap`.

        A COUNT, not a cost model -- spine nodes, limb segments and parts, weighted so parts cost more
        than segments because they are what an editor meters. It does not predict render cost.
        """
        sp = self.spec.get("spine") or {}
        nodes = int(sp.get("segments", 0)) + 1
        limb_seg = sum(int(l.get("segments", 0)) for l in (self.spec.get("limbs") or []))
        n_parts = sum(max(int(s.get("n", 2)) if s.get("symmetry") == "radial" else
                          (2 if s.get("symmetry") == "bilateral" else 1), 1)
                      for s in (self.spec.get("sockets") or []))
        spent = nodes * 1 + limb_seg * 2 + n_parts * 5
        return {"spent": int(spent), "cap": int(cap), "within": bool(spent <= int(cap)),
                "spine_nodes": nodes, "limb_segments": limb_seg, "parts": int(n_parts)}

    # ------------------------------------------------------------ serialisation --
    def to_json(self, indent=2):
        """The whole document as JSON -- save-to-disk. The history is deliberately NOT serialised: a
        saved creature is a creature, not an editing session, and reloading someone else's undo stack
        is a feature nobody asked for."""
        return json.dumps({"format": "lecore.creature/1", "spec": self.spec}, indent=indent,
                          sort_keys=True, default=_jsonable)

    @classmethod
    def from_json(cls, text, part_library=None):
        """Load a saved creature. Round-trips exactly: the rebuilt document equals the saved one."""
        d = json.loads(text)
        if d.get("format") != "lecore.creature/1":
            raise ValueError("not a lecore creature document (format=%r)" % d.get("format"))
        return cls(d["spec"], part_library=part_library)


def normalise(doc):
    """Put a document into its CANONICAL JSON shape: numpy scalars become floats, tuples become lists.

    WHY NORMALISE ON EVERY EDIT rather than only on save. Edits compute things -- a profile from a
    callable arrives as np.float64, an axis from a normalised vector arrives as a tuple -- and both
    survive json.dumps happily but come back as plain floats and lists. That made save/load a
    near-identity rather than an identity, and "near" is not a property you can assert or rely on.
    Keeping the document canonical at all times makes the round trip EXACT, so the invariant is real
    instead of approximately true. Cheap: a spec is a few hundred bytes.
    """
    if isinstance(doc, dict):
        return {str(k): normalise(v) for k, v in doc.items()}
    if isinstance(doc, (list, tuple)):
        return [normalise(v) for v in doc]
    if isinstance(doc, np.ndarray):
        return [normalise(v) for v in doc.tolist()]
    if isinstance(doc, (np.floating, np.integer)):
        return doc.item()
    return doc


def _jsonable(o):
    """NumPy scalars and arrays leak into specs from computed edits (a profile from a callable, an
    axis from a normalised vector). Convert rather than fail, so saving never depends on how the
    document happened to be built."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError("not JSON-serialisable: %r" % type(o))


def _selftest():
    """The session contract: every edit is undoable, save/load round-trips exactly, validation
    catches real breakage, parts survive body edits, and the budget counts what it claims."""
    from holographic.mesh_and_geometry.holographic_creature import quadruped_spec
    from holographic.mesh_and_geometry.holographic_creatureparts import PartLibrary
    from holographic.mesh_and_geometry.holographic_creaturesocket import _unit_cone, resolve_socket

    lib = PartLibrary(dim=256, seed=0)
    lib.define("horn", handles={"length": (0.5, 2.0)}, geometry=_unit_cone())

    ed = CreatureEditor(quadruped_spec(), part_library=lib)
    base_segments = ed.spec["spine"]["segments"]

    # 1) EVERY EDIT IS UNDOABLE, one at a time, in order.
    assert not ed.can_undo()
    ed.extend_spine(2).set_thickness(0.5, 0.22, falloff=0.3).add_part("horn", 0.5, 0.0)
    assert ed.can_undo() and len(ed.spec["sockets"]) == 1
    assert ed.spec["spine"]["segments"] == base_segments + 2
    ed.undo()
    assert len(ed.spec["sockets"]) == 0, "undo must take back the part"
    ed.undo(); ed.undo()
    assert ed.spec["spine"]["segments"] == base_segments, "undo must walk all the way back"
    assert not ed.can_undo() and ed.can_redo()
    ed.redo()
    assert ed.spec["spine"]["segments"] == base_segments + 2, "redo must re-apply"

    # 2) A NEW EDIT DISCARDS THE REDO TAIL -- you cannot redo into an abandoned future.
    ed.add_limb(at=0.4)
    assert not ed.can_redo()

    # 3) SAVE / LOAD ROUND-TRIPS EXACTLY.
    ed2 = CreatureEditor(quadruped_spec(), part_library=lib)
    ed2.extend_spine(1).set_thickness(0.4, 0.18).add_part("horn", 0.6, 1.1, symmetry="radial", n=3)
    text = ed2.to_json()
    back = CreatureEditor.from_json(text, part_library=lib)
    assert back.spec == ed2.spec, "a reloaded document must equal the saved one"
    assert normalise(ed2.spec) == ed2.spec, "the live document must already be canonical JSON"
    assert back.to_json() == text, "and re-saving must be byte-identical"
    try:
        CreatureEditor.from_json('{"format": "nope"}'); raise AssertionError("must reject foreign JSON")
    except ValueError:
        pass

    # 4) VALIDATION catches real breakage and passes a sound document.
    v = ed2.validate()
    assert v["ok"] and not v["errors"], v
    bad = CreatureEditor(quadruped_spec(), part_library=lib)
    bad.spec["limbs"][0]["at"] = 5.0
    assert not bad.validate()["ok"], "a limb off the end of the spine must be an ERROR"
    missing = CreatureEditor(quadruped_spec(), part_library=lib)
    missing.add_part("nonexistent_part", 0.5, 0.0)
    assert not missing.validate()["ok"], "a part the library lacks must be an ERROR"

    # 5) PARTS SURVIVE BODY EDITS -- the whole reason sockets are (t, theta). Resolve a part, then
    #    fatten the body, and the part must still land ON the new skin rather than inside or above it.
    ed3 = CreatureEditor(quadruped_spec(), part_library=lib)
    ed3.add_part("horn", 0.5, 0.0, symmetry="none")
    before = resolve_socket(ed3.creature(), ed3.field(), 0.5, 0.0)
    ed3.set_thickness(0.5, 0.26, falloff=0.4)
    after = resolve_socket(ed3.creature(), ed3.field(), 0.5, 0.0)
    assert before["hit"] and after["hit"]
    assert after["depth"] > before["depth"] + 1e-3, \
        "after fattening the body the part must sit further out (%.3f -> %.3f)" % (before["depth"], after["depth"])
    assert abs(float(ed3.field()(after["point"][None, :])[0])) < 1e-6, "still exactly on the skin"

    # 6) BUILD produces geometry, and reports any socket that missed rather than dropping it silently.
    out = ed3.build(resolution=26)
    assert len(np.asarray(out["skin"].vertices)) > 100
    assert out["parts"] is not None and len(out["placements"]) == 1 and not out["missed"]

    # 7) THE BUDGET counts what it says it counts.
    c = CreatureEditor(quadruped_spec()).complexity(cap=100)
    assert c["spine_nodes"] == 5 and c["within"]
    loaded = CreatureEditor(quadruped_spec())
    for i in range(12):
        loaded.add_part("horn", 0.1 + 0.05 * i, 0.0, symmetry="radial", n=5)
    assert loaded.complexity(cap=100)["parts"] == 60
    assert not loaded.complexity(cap=100)["within"], "60 parts must blow a 100-point cap"

    # 8) THE DOCUMENT IS NEVER MUTATED IN PLACE -- a held reference stays as it was.
    ed4 = CreatureEditor(quadruped_spec())
    held = ed4.spec
    ed4.extend_spine(3)
    assert held["spine"]["segments"] == base_segments, "an edit must not mutate the prior document"

    print("creatureeditor selftest OK: undo/redo walks the full stack, JSON round-trips byte-identical, "
          "validation catches off-spine limbs and unknown parts, a part rides the skin through a "
          "thickness edit (%.3f -> %.3f), budget 60 parts blows a 100 cap"
          % (before["depth"], after["depth"]))


if __name__ == "__main__":
    _selftest()

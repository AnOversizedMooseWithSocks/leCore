"""Part 11 of UnifiedMind's faculty surface -- 124 methods, encyclopedia_reset .. quick_material.

NOT A STANDALONE MODULE. This is one slice of the single `UnifiedMind` class, which grew to 17.4k lines
in one file and went past the 1 MB cap an agent can read in a single pass -- so the engine could no
longer read its own central nervous system. The class is assembled from these parts by
holographic/misc/holographic_unified.py, which is still the only import path anyone uses.

Every method here is a real attribute of UnifiedMind at runtime (mixin, not delegation), so `mind.x()`,
`dir(mind)`, the doc generators and the service's tool introspection all behave exactly as before. The
bodies were moved by line range, not regenerated, so they are byte-identical to the originals.

KEPT NEGATIVE, so nobody "tidies" it: these part classes are NOT a public API and must never be
imported or subclassed directly. They carry no `__init__` and assume the state UnifiedMind.__init__
builds; instantiated alone they would fail on the first attribute access. The leading underscore on
the class name says so, and the reachability audit reads them as referenced-by-unified, not as
standalone capabilities.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder, _Index
from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
from holographic.misc.holographic_creature import HolographicMind
from holographic.unified import check_part


class _UnifiedPart11:

    def encyclopedia_reset(self, dim=None, seed=None):
        """Start a fresh encyclopedia (dropping everything taught so far). Returns the number of concepts cleared."""
        from holographic.agents_and_reasoning.holographic_encyclopedia import Encyclopedia
        n = len(getattr(self._encyclopedia_faculty, "parent", {})) if getattr(self, "_encyclopedia_obj", None) else 0
        self._encyclopedia_obj = Encyclopedia(dim=int(dim or self.dim), seed=int(self.seed if seed is None else seed))
        return n

    def encyclopedia_add(self, concept, is_a=None, has=None):
        """Teach one concept: its is_a PARENT (taxonomy) and/or its HAS parts. Key concepts by a sense id like
        'dog.n.01' so different senses of a word do not collapse into one. Returns the concept.

        This is the third rung of the dictionary -> grammar -> encyclopedia curriculum: a dictionary tells you what
        a word MEANS; an encyclopedia places it in a web of relations. See holographic_encyclopedia.Encyclopedia."""
        self._encyclopedia_faculty.add(concept, is_a=is_a, has=has)
        return concept

    def encyclopedia_is_a(self, concept):
        """One hop up the taxonomy: (parent, cleanup_confidence). See holographic_encyclopedia.Encyclopedia.is_a."""
        parent, conf = self._encyclopedia_faculty.is_a(concept)
        return {"parent": parent, "confidence": float(conf)}

    def encyclopedia_climb(self, concept, hops=99, min_throughput=0.0, hop_discount=0.9):
        """Walk the is_a chain upward as a relation ray: {chain, throughput}. `throughput` is the product of per-hop
        confidences times an explicit `hop_discount` per step -- a longer chain of deductions is deliberately less
        certain than a short one, because with exact unitary-atom unbinding each hop is near-lossless and the depth
        penalty would otherwise vanish. It ABSTAINS rather than emit noise once throughput would fall below
        `min_throughput`. See holographic_encyclopedia.Encyclopedia.climb."""
        chain, tp = self._encyclopedia_faculty.climb(concept, hops=hops, min_throughput=min_throughput,
                                                     hop_discount=hop_discount)
        return {"chain": list(chain), "throughput": float(tp)}

    def encyclopedia_is_a_transitive(self, concept, ancestor):
        """Taxonomic membership: does `concept` reach `ancestor` by is_a? {reached, hops, throughput}.
        See holographic_encyclopedia.Encyclopedia.is_a_transitive."""
        reached, hops, tp = self._encyclopedia_faculty.is_a_transitive(concept, ancestor)
        return {"reached": bool(reached), "hops": int(hops), "throughput": float(tp)}

    def encyclopedia_siblings(self, concept):
        """Concepts sharing this one's is_a parent -- relatedness from STRUCTURE, not word overlap (which is the
        capability a dictionary lacks). See holographic_encyclopedia.Encyclopedia.siblings."""
        return list(self._encyclopedia_faculty.siblings(concept))

    def encyclopedia_relatedness(self, a, b):
        """A structural relatedness score in [0, 1]: 1 / (1 + depth_a + depth_b) to the nearest common ancestor.
        Measured -- identical 1.000, parent 0.500, siblings 0.333, cousins 0.200, unrelated 0.000. It ORDERS
        taxonomic distance; it is not a probability and it does not saturate at 1 for "closely related". (The
        underlying docstring used to claim 1.0 for siblings. It is 0.333; only a concept against itself scores 1.0.)

        This is what the encyclopedia layer ADDS over a dictionary: `dog` and `wolf` share no letters, yet sit
        beside each other in the taxonomy. See holographic_encyclopedia.Encyclopedia.relatedness."""
        return float(self._encyclopedia_faculty.relatedness(a, b))

    # -- external COMMANDS as tools (R4): run an ALLOWLISTED program, wired into the same VSA fabric -----------
    # SECURITY, and it is load-bearing: this mind is exposed over HTTP, where /invoke calls any public method by
    # name. So a general command runner reachable here is arbitrary-process execution reachable by any agent. The
    # gate is the ALLOWLIST, and the one rule that makes it a gate is that the allowlist is owned by the OPERATOR
    # and never fed from agent input: `register_command` is a configuration call (a human/deployer names the exe and
    # a FIXED argv template up front), and `run_command` can only ever run a name that is already on it. Values fill
    # "{key}" placeholders one-token-in-one-token-out, and there is NO shell -- so "a; rm -rf /" passed as a value
    # is echoed literally, never interpreted (verified in holographic_command's selftest). Do not add a faculty that
    # registers a command from a run_command argument; that would hand the allowlist to the caller and void the gate.
    @property
    def _commands(self):
        if getattr(self, "_command_runner", None) is None:
            from holographic.scene_and_pipeline.holographic_command import CommandRunner
            self._command_runner = CommandRunner(timeout=getattr(self, "_command_timeout", 30))
        return self._command_runner

    def _register_command(self, name, argv, doc=""):
        """OPERATOR configuration -- add an external program to the allowlist so run_command may later run it.

        UNDERSCORE-PREFIXED ON PURPOSE, and this is the whole security design: the HTTP service exposes every PUBLIC
        method to /invoke by name, blocking only names that start with '_'. If registration were public, an agent
        could POST /invoke {"name":"register_command","args":{"name":"sh","argv":["sh","-c","{input}"]}} and the
        allowlist would no longer be a boundary -- MEASURED: it did exactly that before this was made private. So
        registration is an IN-PROCESS operator call (configure the mind, THEN serve it), and only `run_command` --
        which can run nothing that is not already on the list -- is reachable over the wire.

        `argv` is a FIXED token list; "{path}" placeholders are filled from run_command's args one-token-in-one-
        token-out, and there is no shell. The executable must exist on PATH. Returns the name. See
        holographic_command.CommandRunner.register."""
        return self._commands.register(name, argv, doc=doc)

    def registered_commands(self):
        """The allowlist: {name: doc}. The ONLY external programs run_command can run. See
        holographic_command.CommandRunner.registered."""
        return self._commands.registered()

    def run_command(self, name, args=None):
        """Run an allowlisted external program: {stdout, stderr, returncode, ok}. Raises if `name` is not on the
        allowlist (the security gate -- commands never come from the caller, only their argument VALUES do) or the
        run times out. No shell, so an injection attempt in a value is a literal value. See
        holographic_command.CommandRunner.run."""
        return self._commands.run(name, args)

    def command_tool(self, name, in_type, out_type, keywords, args_from=None):
        """Wrap an allowlisted command as an orchestrator Tool, so the Planner can select and chain it and the
        CircuitBreaker trips on a flaky one -- an external program joining the same VSA fabric as an internal
        faculty. `name` must already be registered. See holographic_command.command_as_tool."""
        from holographic.scene_and_pipeline.holographic_command import command_as_tool
        return command_as_tool(self._commands, name, in_type, out_type, keywords, self.vocab, args_from=args_from)

    # -- agentic FILE / CODE editing (read/write/replace/insert/delete/archive/grep/list) ------------------
    # A lazily-built Editor scoped to a project root, so an agent (or an /invoke caller) can work the codebase
    # safely: every path is confined to `file_root`. Set the root once with set_file_root(); default is cwd.
    @property
    def _editor(self):
        if getattr(self, "_editor_obj", None) is None:
            from holographic.io_and_interop.holographic_codeedit import Editor
            self._editor_obj = Editor(getattr(self, "_file_root", "."))
        return self._editor_obj

    def set_file_root(self, root):
        """Scope all file_* operations to `root` (a directory). Every subsequent path is resolved inside it and
        cannot escape. Returns the absolute root."""
        from holographic.io_and_interop.holographic_codeedit import Editor
        self._editor_obj = Editor(root)
        self._file_root = root
        return self._editor_obj.root

    def file_read(self, path, max_bytes=1_000_000):
        """Read a file's text (utf-8), scoped to the file root. See holographic_codeedit.Editor.read."""
        return self._editor.read(path, max_bytes=max_bytes)

    def file_read_lines(self, path, start=1, end=None):
        """Read lines [start, end] (1-based inclusive) of a file as a list of strings -- look at a region before
        editing it. See holographic_codeedit.Editor.read_lines."""
        return self._editor.read_lines(path, start=start, end=end)

    def file_view(self, path, start=1, end=None):
        """Read lines [start, end] as one string WITH LINE NUMBERS -- the located form to look at before targeting
        an edit (pairs with file_replace_lines / file_insert / file_delete_lines). See Editor.view."""
        return self._editor.view(path, start=start, end=end)

    def file_read_many(self, paths, max_bytes=1_000_000):
        """Read several files at once -> {path: text} (a bad path maps to an "<error: ...>" string). Gathers
        context in one call. See holographic_codeedit.Editor.read_many."""
        return self._editor.read_many(paths, max_bytes=max_bytes)

    def file_count(self, path, text):
        """Count occurrences of `text` in a file -- check uniqueness BEFORE file_replace. See Editor.count_occurrences."""
        return self._editor.count_occurrences(path, text)

    def file_replace_lines(self, path, start, end, text):
        """Replace lines [start, end] (1-based inclusive) with `text` -- the range edit for when old text isn't
        unique enough for file_replace. Returns {path, replaced, new_lines}. See Editor.replace_lines."""
        return self._editor.replace_lines(path, start, end, text)

    def file_python_check(self, path):
        """Syntax-check a .py file (ast.parse only) right after editing it -> {ok, error}. Catch a broken edit
        immediately instead of at import time. See holographic_codeedit.Editor.python_check."""
        return self._editor.python_check(path)

    def file_write(self, path, text, overwrite=True):
        """Create or atomically replace a file with `text`. overwrite=False refuses to clobber. Returns
        {path, bytes, created}. See holographic_codeedit.Editor.write."""
        return self._editor.write(path, text, overwrite=overwrite)

    def file_replace(self, path, old, new, count=1):
        """Replace EXACT text `old` with `new`; `old` must occur exactly `count` times (count=1 = must be unique,
        count=0 = replace all). Returns {path, replacements, first_line}. The workhorse code edit. See
        holographic_codeedit.Editor.replace."""
        return self._editor.replace(path, old, new, count=count)

    def file_insert(self, path, after_line, text):
        """Insert `text` after 1-based line `after_line` (0 = top). Returns {path, inserted_at}. See
        holographic_codeedit.Editor.insert."""
        return self._editor.insert(path, after_line, text)

    def file_delete_lines(self, path, start, end):
        """Delete lines [start, end] (1-based inclusive). Returns {path, deleted}. See
        holographic_codeedit.Editor.delete_lines."""
        return self._editor.delete_lines(path, start, end)

    def file_grep(self, pattern, path=".", suffix=".py", max_hits=200, regex=False):
        """Search across files under `path` (filtered by `suffix`). Returns [{file, line, text}] -- the 'where is X
        used' an agent needs constantly. `regex=False` (default) is a plain SUBSTRING match, so `(` and `*` mean
        themselves; `regex=True` compiles the pattern with `re`. Additive and default-off.
        See holographic_codeedit.Editor.grep."""
        return self._editor.grep(pattern, relpath=path, suffix=suffix, max_hits=max_hits, regex=regex)

    def file_list(self, path=".", recursive=False, suffix=None):
        """List files under a directory (relative paths); skips __pycache__/hidden. See
        holographic_codeedit.Editor.list_dir."""
        return self._editor.list_dir(path, recursive=recursive, suffix=suffix)

    def file_archive(self, path, archive_dir=".lecore_archive"):
        """Move a file into a timestamped archive dir instead of deleting it (a reversible 'delete'). Returns
        {archived_from, archived_to}. See holographic_codeedit.Editor.archive."""
        return self._editor.archive(path, archive_dir=archive_dir)

    def file_delete(self, path):
        """Permanently remove a file (use file_archive to keep a copy). Returns {deleted}. See
        holographic_codeedit.Editor.delete."""
        return self._editor.delete(path)

    def file_move(self, src, dst, overwrite=False):
        """Move/rename a file within the root. Returns {moved_from, moved_to}. See holographic_codeedit.Editor.move."""
        return self._editor.move(src, dst, overwrite=overwrite)

    def file_undo(self, steps=1):
        """Reverse the last `steps` mutating file operations (write/replace/insert/delete_lines/replace_lines),
        restoring prior contents; a created file is removed. Returns {undone, files}. See Editor.undo."""
        return self._editor.undo(steps=steps)

    def file_find_definition(self, name, path=".", suffix=".py"):
        """Find where a Python function/class `name` is DEFINED under `path` -> [{file, line, kind, text}]. The
        'jump to definition' for the codebase. See holographic_codeedit.Editor.find_definition."""
        return self._editor.find_definition(name, relpath=path, suffix=suffix)

    def file_replace_across(self, old, new, path=".", suffix=".py", dry_run=False):
        """Replace exact `old` with `new` in every file under `path` that contains it (dry_run=True previews).
        Returns [{file, replacements}]. Codebase-wide rename. See Editor.replace_across."""
        return self._editor.replace_across(old, new, relpath=path, suffix=suffix, dry_run=dry_run)

    def file_tree(self, path=".", max_depth=3, suffix=None):
        """An indented directory tree under `path` (skips __pycache__/hidden). Returns a string. See Editor.tree."""
        return self._editor.tree(path, max_depth=max_depth, suffix=suffix)

    def file_import_check(self, path):
        """Import the module (dotted path under the file root) in a fresh subprocess and report success or the
        real ImportError tail -> {ok, error}. Catches load-time breakage a syntax check misses. See
        holographic_codeedit.Editor.import_check."""
        return self._editor.import_check(path)

    def affected_tests(self, changed_paths=None, since=None):
        """Which test files actually need to run for a change -- the fix for "why do thousands of tests run on
        every small commit?". Delegates entirely to tools/select_tests.py's static import graph (pure `ast`, no
        execution, no coverage tracing): a test is picked if it transitively imports a changed module, or IS the
        changed file. This is the SAME logic CI itself already runs on every push/PR (see NOTES_concepts.md, 'CI
        SPEEDUP') -- previously reachable only from the command line (tools/test_changed.py), now callable through
        the mind like any other faculty.

        Pass `changed_paths` explicitly (a list of file paths relative to the repo root: what YOU know changed),
        or leave it None to auto-detect from git via tools/test_changed.changed_files -- the working-tree diff vs
        HEAD (staged + unstaged + new untracked files), or the diff since `since` (a branch/tag, e.g.
        since='main') for the PR view.

        Returns a sorted list of test file paths to hand to pytest, the string "ALL" if a change can't be scoped
        safely (an unknown binary/data file, or a .py outside the module map -- fails toward running MORE tests,
        never fewer), or [] if nothing is affected (e.g. a docs-only change). See tools.select_tests.affected_tests
        / tools.test_changed.changed_files."""
        from tools.select_tests import affected_tests as _affected_tests
        from tools.test_changed import changed_files as _changed_files, REPO as _REPO
        root = getattr(self, "_file_root", None) or _REPO
        if changed_paths is None:
            changed_paths = _changed_files(since=since)
            if not changed_paths:
                return []
        return _affected_tests(changed_paths, root=root)

    def render_frame_delta(self, prev, curr, tile=32, thresh=1e-3):
        """The pixel-streaming primitive: return only the `tile`x`tile` image blocks that CHANGED between two
        frames, as (row, col, pixels), plus the fraction changed -- so a viewport pushes just the dirty tiles
        after a local edit / small camera move, the rendering analogue of the engine's O(change) delta protocol.
        See holographic_render.frame_delta_tiles."""
        from holographic.rendering.holographic_render import frame_delta_tiles
        return frame_delta_tiles(prev, curr, tile=tile, thresh=thresh)

    def deform(self, geometry, kind="bend", angle=0.0, factor=0.0, axis=2, up=None, center=None):
        """Apply a classic vectorised deformer to ANY point set -- a Mesh (returns a deformed Mesh, faces kept)
        OR an (N,3) array (a particle cloud / point set, returns the deformed array). kind: 'bend' (arc in the
        (axis,up) plane by `angle` rad), 'twist' (screw by `angle` rad along `axis`), 'taper' (cone by `factor`
        along `axis`). One array op per deformer -- no Python per-point loop, so a mesh and a particle cloud run
        the same path. See holographic_deform."""
        from holographic.mesh_and_geometry.holographic_deform import bend, twist, taper
        from holographic.mesh_and_geometry.holographic_mesh import Mesh
        is_mesh = hasattr(geometry, "vertices")
        P = geometry.vertices if is_mesh else np.asarray(geometry, float)
        if kind == "bend":
            Q = bend(P, angle, axis=(0 if axis == 2 else axis), up=(2 if up is None else up), center=center)
        elif kind == "twist":
            Q = twist(P, angle, axis=axis)
        elif kind == "taper":
            Q = taper(P, factor, axis=axis)
        else:
            raise ValueError("kind must be 'bend', 'twist' or 'taper'")
        return Mesh(Q, [tuple(f) for f in geometry.faces]) if is_mesh else Q

    def lattice_deform(self, geometry, bounds, control_offsets):
        """Free-form (FFD) deformation through a control lattice: each point moves by the TRILINEAR interpolation
        of the lattice's per-control displacements `control_offsets` (nx,ny,nz,3) over `bounds`. Works on a Mesh
        (returns a Mesh) or an (N,3) array. The sculpt-by-cage deformer, vectorised (8 gathers, no point loop).
        See holographic_deform."""
        from holographic.mesh_and_geometry.holographic_deform import lattice_deform as _ld
        from holographic.mesh_and_geometry.holographic_mesh import Mesh
        is_mesh = hasattr(geometry, "vertices")
        P = geometry.vertices if is_mesh else np.asarray(geometry, float)
        Q = _ld(P, bounds, control_offsets)
        return Mesh(Q, [tuple(f) for f in geometry.faces]) if is_mesh else Q

    def blend_shapes(self, base, targets, weights):
        """Morph-target / blendshape mix as a WEIGHTED BUNDLE: base + sum_i w_i (target_i - base). `base` and each
        target are (N,3) (mesh vertices or particles); `weights` length K. Vectorised (`weights @ deltas`) -- this
        is the engine's superposition primitive on geometry, so animating the weights over time IS the blendshape
        animation. Pass a base Mesh to get a Mesh back. See holographic_deform."""
        from holographic.mesh_and_geometry.holographic_deform import blendshapes
        from holographic.mesh_and_geometry.holographic_mesh import Mesh
        is_mesh = hasattr(base, "vertices")
        B = base.vertices if is_mesh else np.asarray(base, float)
        T = [t.vertices if hasattr(t, "vertices") else np.asarray(t, float) for t in targets]
        Q = blendshapes(B, T, weights)
        return Mesh(Q, [tuple(f) for f in base.faces]) if is_mesh else Q

    def timeline(self):
        """A keyframe Timeline (holographic_anim): `.key(channel, t, value, interp='linear')` then
        `.sample(channel, t)` for the interpolated value at time t (t may be an array -- vectorised). EASING per key:
        interp='linear' (default), 'step' (hold), 'smooth' (ease in-out), 'ease_in', or 'ease_out' -- the ease in
        ease out of an animation curve. Key blendshape weights, deform params, or transforms and drive the animation
        from it. See holographic_anim.Timeline."""
        from holographic.misc.holographic_anim import Timeline
        return Timeline()

    def transport(self, frame_fn, n_frames, fps=24.0, base=None):
        """An animation TRANSPORT / playhead (holographic_anim) -- start/pause/step/seek/scrub/rewind/fast-forward
        that the keyframe timeline and frame cache did not provide. `frame_fn(frame) -> state` computes any frame on
        demand (a deformed mesh's vertices, a sim field, packed params); the Transport holds a current frame + play
        state and caches computed frames so rewind / scrub-back / replay is O(1). Methods: play(speed) (1.0 forward,
        -1.0 rewind, 2.0 fast-forward, 0.5 slow-mo), pause(), stop() (pause+rewind to 0), tick(n) (advance the play
        loop), step(n) (frame-by-frame regardless of play state), seek(frame)/seek_time(sec) (scrub), at(frame)
        (cached state), .frame/.time. Deterministic: seeking to a frame gives the same state however you arrived (no
        scrub drift), because frame_fn is a pure function of the frame index. A stateful sim that can't be evaluated
        at an arbitrary frame should be baked first (bake_deformation), then driven through a Transport over the cache.
        Returns a Transport. See holographic_anim.Transport."""
        from holographic.misc.holographic_anim import Transport
        return Transport(frame_fn, n_frames, fps=fps, base=base)

    def frame_cache(self, base, hot=8, tol=1e-9):
        """A tiered delta FrameCache for playback: `.put(frame, state)` stores each frame as a sparse DELTA vs
        `base` (O(change) memory -- the engine's patch idea on the time axis), `.get(frame)` reconstructs it
        exactly, and the `hot` most-recent frames stay full in RAM for instant scrubbing. `.full_bytes()` /
        `.memory_bytes()` report the saving (big when frames change locally, ~full when a deformation is global).
        See holographic_anim.FrameCache."""
        from holographic.misc.holographic_anim import FrameCache
        return FrameCache(base, hot=hot, tol=tol)

    def bake_deformation(self, base, n_frames, frame_fn):
        """Evaluate an animation into a FrameCache: for each frame f, cache frame_fn(base, f) as a delta. Returns
        the FrameCache, ready to scrub. `frame_fn` is any vectorised deformer with a time-varying parameter; the
        bake never loops in Python over vertices, only over frames. See holographic_anim.bake_deformation."""
        from holographic.misc.holographic_anim import bake_deformation
        return bake_deformation(base, n_frames, frame_fn)

    def mirror_mesh(self, mesh, axis=0, plane=0.0, weld=True, tol=1e-5):
        """Mirror a mesh across the `axis`=const `plane`: append a reflected copy with reversed winding (normals
        stay consistent) and optionally WELD the seam -- the standard way to model a symmetric object from one
        half. Vectorised. See holographic_meshtools.mirror."""
        from holographic.mesh_and_geometry.holographic_meshtools import mirror
        return mirror(mesh, axis=axis, plane=plane, weld=weld, tol=tol)

    def mesh_symmetrize(self, mesh, axis=0, plane=0.0, side=1, tol=1e-5):
        """SYMMETRIZE a mesh across the `axis`=const `plane` (holographic_meshtools): KEEP the half on `side` (+1 =
        positive side, -1 = negative), MIRROR it back, and weld the seam -- a bilaterally-symmetric result. Unlike
        mirror_mesh (which doubles the WHOLE mesh), this first discards the far side, so it FIXES an off-axis sculpt
        instead of preserving the asymmetry. A face is kept iff its centroid is on `side`; near-plane vertices snap
        onto the plane so the seam welds. Composes mirror + weld. Returns a new Mesh."""
        from holographic.mesh_and_geometry.holographic_meshtools import symmetrize
        return symmetrize(mesh, axis=axis, plane=plane, side=side, tol=tol)

    def weld_mesh(self, mesh, tol=1e-5):
        """Merge-by-distance: weld vertices within `tol` into one (mean position), remap faces, drop the faces
        that collapse -- the cleanup after a mirror / import / boolean. Vectorised for triangle meshes. See
        holographic_meshtools.merge_by_distance."""
        from holographic.mesh_and_geometry.holographic_meshtools import merge_by_distance
        return merge_by_distance(mesh, tol=tol)

    def solidify_mesh(self, mesh, thickness, flip=False):
        """Give a surface thickness (the shell / solidify modifier): offset an inner copy along the vertex
        normals by `thickness`, reverse its winding, and BRIDGE the boundary so an open sheet becomes a
        watertight solid (a closed mesh becomes a hollow double wall). Vectorised offset; the bridge loops over
        boundary edges only. See holographic_meshtools.solidify."""
        from holographic.mesh_and_geometry.holographic_meshtools import solidify
        return solidify(mesh, thickness, flip=flip)

    def render_sdf(self, sdf, camera, width=256, height=256, light_dir=(-0.4, 0.7, -0.3),
                   base_color=(0.85, 0.5, 0.35), sky=None, ao=True, shadows=True, reflect=0.25,
                   refract=0.0, ior=1.5, sss=0.0, sss_color=(1.0, 0.4, 0.3), ambient=0.25):
        """Field-native SDF renderer: sphere-trace primary rays, then shade hits with Lambert direct light gated
        by a SOFT SHADOW, ambient gated by AMBIENT OCCLUSION, an HDRI-sky environment REFLECTION (Schlick
        fresnel), optional REFRACTION (the sky bent through the surface) and SUBSURFACE glow; misses show the sky
        dome. `sky` may be an equirectangular HDRI (H,W,3) array. These are light-transport effects that fall out
        cheaply because the engine is SDF-native (the field answers nearest-surface / occlusion / normal). All
        vectorised over pixels. See holographic_raymarch.render_sdf."""
        from holographic.rendering.holographic_raymarch import render_sdf
        if hasattr(base_color, "base_color"):
            base_color = base_color.base_color[:3]
        return render_sdf(sdf, camera, width=width, height=height, light_dir=light_dir, base_color=base_color,
                          sky=sky, ao=ao, shadows=shadows, reflect=reflect, refract=refract, ior=ior, sss=sss,
                          sss_color=sss_color, ambient=ambient)

    def depth_fog(self, color, depth, density=0.15, fog_color=(0.55, 0.65, 0.82), start=0.0):
        """Apply exponential DEPTH FOG (Beer-Lambert) to a rendered image (W16): fade each pixel toward
        `fog_color` by 1 - exp(-density * depth). `color` (H,W,3) LINEAR, `depth` (H,W) per-pixel distance (the
        raymarch t). Apply before gamma. The atmosphere of a scene in one pass. See
        holographic_atmosphere.depth_fog."""
        from holographic.rendering.holographic_atmosphere import depth_fog
        return depth_fog(color, depth, density=density, fog_color=fog_color, start=start)

    def light_shafts(self, color, light_uv=(0.5, 0.2), threshold=0.7, density=0.9, decay=0.92,
                     weight=0.5, exposure=0.35, samples=48):
        """Volumetric LIGHT SHAFTS / god rays by radial blur (W16, Mitchell GPU Gems 3): streak the bright pixels
        (sky/light at `light_uv` screen coords) outward from the source. Returns the shaft glow (H,W,3) to ADD to
        the scene. Screen-space -- only shafts a source on/near screen. See holographic_atmosphere.light_shafts."""
        from holographic.rendering.holographic_atmosphere import light_shafts
        return light_shafts(color, light_uv=light_uv, threshold=threshold, density=density, decay=decay,
                            weight=weight, exposure=exposure, samples=samples)

    def sphere_trace_trapped(self, sdf, O, D, trap=(0.0, 0.0, 0.0), trap_kind="origin",
                             max_steps=96, max_dist=20.0, surf_eps=1e-3):
        """Sphere-trace rays AND return each ray's ORBIT TRAP -- the closest approach of its march to a trap set
        (Quilez's fractal-colouring scalar). Returns (hit, t, pos, trap_val); the hit/t/pos are identical to
        sphere_trace (same march), and trap_val is the per-ray minimum distance to the trap. `trap_kind` in
        {'point','origin','axis','plane'}; `trap` is the point / axis / plane-normal. Feed trap_val through
        cosine_palette to colour a surface by how near its orbit came. See orbit_trap_render for the whole
        render in one call, and holographic_raymarch.sphere_trace_trapped."""
        from holographic.rendering.holographic_raymarch import sphere_trace_trapped
        return sphere_trace_trapped(sdf, O, D, trap=trap, trap_kind=trap_kind,
                                    max_steps=max_steps, max_dist=max_dist, surf_eps=surf_eps)

    def orbit_trap_render(self, sdf, camera, width=256, height=256, trap=(0.0, 1.0, 0.0),
                          trap_kind="axis", palette=None, trap_scale=1.4, light_dir=(0.4, 0.8, 0.3),
                          ambient=0.25, background=(0.05, 0.06, 0.11), max_steps=110, max_dist=20.0):
        """Render an SDF scene coloured by ORBIT TRAP -- the signature Quilez fractal look, in one call. Sphere-
        traces every pixel, tracks each ray's closest approach to the trap set, and maps that scalar through a
        cosine palette, lit by a simple Lambert term. `trap_kind` in {'point','origin','axis','plane'}; `palette`
        is a (a,b,c,d) cosine_palette tuple (default: a harmonious random_palette). `trap_scale` stretches the
        trap value into the palette's [0,1]. Returns (H,W,3) in [0,1] (sRGB). This is W3 -- orbit traps + cosine
        palettes, the two halves finally meeting. Composes with any domain-warped SDF (fold/repeat/twist).
        See holographic_raymarch.sphere_trace_trapped and holographic_domain.cosine_palette."""
        import numpy as _np
        from holographic.rendering.holographic_raymarch import sphere_trace_trapped, sdf_normal
        from holographic.mesh_and_geometry.holographic_domain import cosine_palette, random_palette
        eye, dirs = camera.ray_dirs(width, height)
        O = _np.broadcast_to(eye, (width * height, 3)).astype(float)
        D = dirs.reshape(-1, 3)
        hit, t, pos, trap_val = sphere_trace_trapped(sdf, O, D, trap=trap, trap_kind=trap_kind,
                                                     max_steps=max_steps, max_dist=max_dist)
        img = _np.zeros((width * height, 3))
        img[~hit] = _np.asarray(background)
        if hit.any():
            N = sdf_normal(sdf, pos[hit])
            L = _np.asarray(light_dir, float); L = L / _np.linalg.norm(L)
            lam = _np.clip(N @ L, 0, 1)[:, None]
            pal = palette if palette is not None else random_palette(seed=7, contrast=0.6)
            key = _np.clip(trap_val[hit] * trap_scale, 0, 1)
            col = cosine_palette(key, *pal)
            img[hit] = col * (ambient + (1.0 - ambient) * lam)
        return _np.clip(img.reshape(height, width, 3), 0, 1) ** (1 / 2.2)

    def ambient_occlusion(self, sdf, points, normals, samples=6, step=0.06, k=1.6):
        """SDF ambient occlusion at `points` with `normals`: march the normal and read the field -- a near
        surface darkens the point. Field-native, no hemisphere rays. See holographic_raymarch.ambient_occlusion."""
        from holographic.rendering.holographic_raymarch import ambient_occlusion
        return ambient_occlusion(sdf, points, normals, samples=samples, step=step, k=k)

    def sdf_extrude(self, sd2d, height=1.0):
        """EXTRUDE a 2-D SDF into a 3-D prism along Z (W10, iq's opExtrusion) -- a logo becomes a badge, a gear
        cross-section a gear. `sd2d` is a 2-D SDF callable f(Q:(n,2))->(n,) (from sdf2d.circle2d / box2d /
        polygon2d / ...). Returns a 3-D SDF f(P)->dist that plugs into sphere_trace / the mesher / the voxelizer.
        EXACT. See holographic_sdf2d.extrude."""
        from holographic.mesh_and_geometry.holographic_sdf2d import extrude
        return extrude(sd2d, height=height)

    def sdf_revolve(self, sd2d, offset=0.0):
        """REVOLVE a 2-D SDF around the Y axis into a solid of revolution (W10, a lathe) -- a vase, a bottle, a
        turned leg; an offset circle becomes a torus. `sd2d` is a 2-D SDF callable; `offset` shifts the profile
        off the axis. Returns a 3-D SDF f(P)->dist. See holographic_sdf2d.revolve."""
        from holographic.mesh_and_geometry.holographic_sdf2d import revolve
        return revolve(sd2d, offset=offset)

    def sdf2d(self, name, **params):
        """Build a 2-D SDF primitive by name (W10): 'circle' (r), 'box' (bx,by), 'rounded_box' (bx,by,r), 'ngon'
        (sides,r), 'polygon' (vertices). Returns f(Q:(n,2))->(n,) -- draw a cross-section, then sdf_extrude or
        sdf_revolve it into 3-D. See holographic_sdf2d."""
        from holographic.mesh_and_geometry import holographic_sdf2d as s2
        builders = {"circle": s2.circle2d, "box": s2.box2d, "rounded_box": s2.rounded_box2d,
                    "ngon": s2.ngon2d, "polygon": s2.polygon2d}
        if name not in builders:
            raise ValueError("unknown 2-D SDF %r; try %s" % (name, sorted(builders)))
        return builders[name](**params)

    def sdf_curvature(self, sdf, points, eps=2e-3):
        """MEAN CURVATURE of an SDF surface at `points` (W13) -- the field Laplacian (div of the unit gradient).
        POSITIVE on convex edges/ridges, NEGATIVE in concave creases/cavities, ~0 on flat regions (a sphere of
        radius r reads 2/r). Drives cavity darkening, edge highlighting, and curvature-aware LOD. See
        holographic_raymarch.sdf_curvature."""
        from holographic.rendering.holographic_raymarch import sdf_curvature
        return sdf_curvature(sdf, points, eps=eps)

    def soft_shadow(self, sdf, points, light_dir, k=12.0):
        """SDF soft shadow: march each point toward the light; the closest approach to any surface is the
        penumbra (0 blocked .. 1 clear). Field-native. See holographic_raymarch.soft_shadow."""
        from holographic.rendering.holographic_raymarch import soft_shadow
        return soft_shadow(sdf, points, light_dir, k=k)

    def sky_dome(self, directions, sun_dir=(-0.4, 0.7, -0.3), env=None):
        """HDRI sky dome: the environment radiance from `directions`. With `env` (an equirectangular (H,W,3)
        image) it samples a real HDRI by lon/lat; otherwise a procedural sky+sun+ground. The incoming light is a
        superposition (bundle) of directional radiance. See holographic_raymarch.sky_dome."""
        from holographic.rendering.holographic_raymarch import sky_dome
        return sky_dome(directions, sun_dir=sun_dir, env=env)

    def refract(self, directions, normals, ior=1.5):
        """Snell's-law refraction of rays at a surface (total-internal-reflection falls back to reflection).
        This is optics -- plain vector math -- exposed as a composable helper. See holographic_raymarch.refract_dir."""
        from holographic.rendering.holographic_raymarch import refract_dir
        return refract_dir(directions, normals, ior=ior)

    def subsurface(self, sdf, points, normals, light_dir, depth=0.6, sigma=4.0):
        """Field-native subsurface translucency: measure how much SOLID the light crosses inside the object to
        reach each point (the SDF interior); thin regions transmit more and glow. See holographic_raymarch.subsurface."""
        from holographic.rendering.holographic_raymarch import subsurface
        return subsurface(sdf, points, normals, light_dir, depth=depth, sigma=sigma)

    def irradiance_cache(self, sdf, points, normals, light_dir, base_color=(0.8, 0.6, 0.5),
                         n_cache=64, n_dirs=16, seed=0):
        """Global illumination via a sparse IRRADIANCE CACHE (Ward = the engine's adaptive-anchor idea): compute
        one-bounce indirect light at `n_cache` surface points (the slow integral, paid sparsely), returning a
        cache to read with `read_irradiance`. Indirect light is smooth, so a sparse cache reconstructs it cheaply.
        See holographic_globalillum.irradiance_cache."""
        from holographic.rendering.holographic_globalillum import irradiance_cache
        return irradiance_cache(sdf, points, normals, light_dir, base_color=base_color,
                                n_cache=n_cache, n_dirs=n_dirs, seed=seed)

    def read_irradiance(self, cache, query_points, k=4):
        """Read an irradiance cache at query points by inverse-distance interpolation of the k nearest cached
        samples (the GI cache read). See holographic_globalillum.read_cache."""
        from holographic.rendering.holographic_globalillum import read_cache
        return read_cache(cache, query_points, k=k)

    def caustics(self, sdf, light_dir=(0, -1, 0), receiver_y=-0.9, extent=2.0, res=128, ior=1.5, n_side=200):
        """Caustics by forward light tracing: shoot parallel light rays, refract them through the object, and
        SPLAT where they land on the receiver plane with np.add.at -- the scatter that is the engine's bundle.
        Where refracted rays converge the bundle piles up: the caustic. Returns a (res,res) intensity map.
        See holographic_globalillum.caustics."""
        from holographic.rendering.holographic_globalillum import caustics
        return caustics(sdf, light_dir=light_dir, receiver_y=receiver_y, extent=extent, res=res, ior=ior, n_side=n_side)

    def morph_scene(self, img_a, img_b, steps=9, method="dct", post=None):
        """Morph between two images. method='dct' (default) blends in the DCT-coefficient domain (structure
        slerp, not a ghosting crossfade). method='phase' (C2) blends in the 2-D FFT domain, interpolating each
        bin's magnitude and PHASE separately -- the phase-vocoder move (holographic_phasemorph.morph_image_phase):
        by the Fourier shift theorem a translation is a phase ramp, so the phase morph SLIDES a translated feature
        to its intermediate position (a compact moving blob) where the DCT slerp interpolates its SHAPE and smears
        it. Measured win for SMALL displacements; the kept BOUND is that a large translation wraps the phase ramp
        (bin phase differences exceed pi) and falls back to a crossfade -- so 'phase' is for small-motion morphs,
        'dct' for arbitrary structure change. With `post` (a holographic_postfx.PostChain) each output frame is run
        through the post-processing pipeline -- generate-and-polish in one call (the same polish post_process applies
        to any image; here it is wired at the generation site). Part of this mind's generative repertoire."""
        if method == "phase":
            from holographic.simulation_and_physics.holographic_phasemorph import morph_image_phase
            frames = morph_image_phase(np.asarray(img_a, float), np.asarray(img_b, float), steps=steps)
        else:
            from holographic.misc.holographic_archive import HolographicArchive
            from holographic.misc.holographic_generate import morph_images
            S = img_a.shape[0]
            arch = HolographicArchive(shape=img_a.shape, capacity=2,
                                      keep=min(900, (S * S) // 2), dim=32768, seed=self.seed)
            frames = morph_images(arch.M, img_a, img_b, steps=steps)
        if post is not None:                                  # polish each generated frame (post-fx pipeline)
            frames = [post.apply(np.clip(np.asarray(f, float), 0.0, 1.0)) for f in frames]
        return frames

    def discover_units(self, stream, order=4, percentile=70):
        """Self-discovery of structure: find the units in a raw symbol stream with
        no labels, by branching entropy on the substrate (holographic_segment) --
        prediction is tight inside a unit and uncertain at its end, so boundaries
        are the entropy peaks. Returns {'chunks', 'boundaries', 'chunk_bits',
        'symbol_bits'} -- including the MDL payoff (discovered chunks compress better
        than single symbols). Pass a string or a list of symbols."""
        from holographic.misc.holographic_segment import Segmenter, chunk_compression
        s = list(stream)
        seg = Segmenter(dim=self.dim, order=order, seed=0).fit(s)
        bounds = seg.boundaries(s, percentile)
        chunks = seg.segment(s, percentile)
        cb, sb = chunk_compression(s, chunks)
        return {"chunks": ["".join(map(str, c)) for c in chunks],
                "boundaries": sorted(bounds),
                "chunk_bits": float(cb), "symbol_bits": float(sb)}

    def compress_cost(self, tokens):
        """Better structure means better compression, measured: encode a sequence by
        the rank of each symbol under the meaning predictor and report the bits, the
        uniform baseline, and the compression ratio (below 1 means structure was
        exploited). A predictor IS a compressor. Needs build_meaning_predictor."""
        if not hasattr(self, "_meaning_pred"):
            return {"ratio": 1.0, "bits_per_symbol": 0.0, "n": 0}
        if not hasattr(self, "_compressor"):
            from holographic.misc.holographic_compress import PredictiveCompressor
            self._compressor = PredictiveCompressor(self._meaning_pred)
        toks = tokens.split() if isinstance(tokens, str) else list(tokens)
        return self._compressor.encode_cost(toks)

    def structure_compresses(self, windows):
        """The link itself: correlation between window structure scores and their
        compression ratios (negative = more structure -> better compression)."""
        if not hasattr(self, "_meaning_pred") or not hasattr(self, "_verifier"):
            return 0.0
        if not hasattr(self, "_compressor"):
            from holographic.misc.holographic_compress import PredictiveCompressor
            self._compressor = PredictiveCompressor(self._meaning_pred)
        from holographic.misc.holographic_compress import structure_compression_correlation
        return structure_compression_correlation(self._verifier, self._compressor, windows)

    def respond(self, query, length=30, query_weight=4.0):
        """Query-and-generate: answer a query with a continuation steered toward
        what the query is about, held coherent by the structure guard. Returns the
        generated token list. Needs build_meaning_predictor first."""
        if not hasattr(self, "_meaning_pred") or not hasattr(self, "_verifier"):
            return []
        from holographic.agents_and_reasoning.holographic_respond import respond as _respond
        return _respond(query, self._meaning_pred, self._verifier,
                        length=length, query_weight=query_weight)

    def respond_report(self, query, length=30, query_weight=4.0):
        """Answer a query AND measure the answer: returns the response with its
        relevance to the query (is it on-topic) and its structure score (is it
        coherent) -- both reported, so the answer is never trusted blindly."""
        if not hasattr(self, "_meaning_pred") or not hasattr(self, "_verifier"):
            return {"response": [], "relevance": 0.0, "structure": 0.0}
        from holographic.agents_and_reasoning.holographic_respond import respond_report as _rr
        return _rr(query, self._meaning_pred, self._verifier,
                   length=length, query_weight=query_weight)

    def deliberate(self, query, max_iters=8, target_quality=0.45, length=26,
                   query_weight=5.0, seed=0):
        """Think before answering: draft a response, judge it, and refine -- keeping
        the best -- stopping early once it is good enough. The number of iterations
        is the 'thinking time' and adapts to how hard the query is (easy ones settle
        fast, hard ones take longer). Returns the response with its quality, the
        iterations used, and the full trace of drafts (the inner deliberation made
        visible). Needs build_meaning_predictor first."""
        if not hasattr(self, "_meaning_pred") or not hasattr(self, "_verifier"):
            return {"response": [], "quality": 0.0, "iterations": 0, "trace": []}
        if not hasattr(self, "_deliberator"):
            from holographic.agents_and_reasoning.holographic_deliberate import Deliberator
            self._deliberator = Deliberator(self._meaning_pred, self._verifier)
        return self._deliberator.deliberate(
            query, max_iters=max_iters, target_quality=target_quality,
            length=length, query_weight=query_weight, seed=seed)

    def negotiate(self, query, max_iters=8, target_quality=0.55, length=26,
                  query_weight=5.0, seed=0):
        """Deliberate under competing judges (coherence vs novelty vs relevance):
        each draft is scored by all three and the kept draft is the most BALANCED
        (its weakest pressure least bad), not the one that wins a single axis. The
        per-judge trace shows the pressures resolving. Needs build_meaning_predictor."""
        if not hasattr(self, "_meaning_pred") or not hasattr(self, "_verifier"):
            return {"response": [], "scores": {}, "negotiated": 0.0, "iterations": 0, "trace": []}
        if not hasattr(self, "_deliberator"):
            from holographic.agents_and_reasoning.holographic_deliberate import Deliberator
            self._deliberator = Deliberator(self._meaning_pred, self._verifier)
        return self._deliberator.negotiate(
            query, max_iters=max_iters, target_quality=target_quality,
            length=length, query_weight=query_weight, seed=seed)

    def anticipate_meaning(self, recent):
        """Compose and settle a next-MEANING prediction; return (word, confidence).
        Even when the exact word is missed, the prediction lands near semantically
        appropriate words -- graded, compositional anticipation."""
        if not hasattr(self, "_meaning_pred"):
            return None, 0.0
        word, _vec, conf = self._meaning_pred.predict_meaning(list(recent))
        return word, conf

    def meaning_prediction_report(self, tokens):
        """Exact-symbol accuracy and semantic RANK (did the composed prediction
        land in the right neighbourhood; 0.5 = chance, 1 = always nearest)."""
        if not hasattr(self, "_meaning_pred"):
            return {"exact": 0.0, "semantic_rank": 0.5, "n": 0}
        toks = [w for s in [tokens] for w in (s if isinstance(s, list) else s.split())] \
            if isinstance(tokens, str) else list(tokens)
        return self._meaning_pred.evaluate(toks)

    def learn_word_generator(self, sentences, order=1, window=2):
        """Train a WORD-level context generator (holographic_generation): a word
        n-gram for local fluency plus learned meaning vectors for an optional
        topic pull. Kept separate from the char-level generate() above. Returns
        self."""
        from holographic.misc.holographic_generation import ContextGenerator
        self._wordgen = ContextGenerator(dim=self.dim, order=order, window=window,
                                         seed=0).fit(sentences)
        return self

    def generate_words(self, seed, length=40, topic_weight=0.0, temperature=0.7,
                       seed_rng=0):
        """Generate at the word level. topic_weight blends a topic-alignment pull
        into the n-gram choice (0 = bare n-gram). NOTE, measured honestly: the
        topic pull does NOT buy real coherence on this substrate -- it is flat
        when n-gram candidates are sparse and collapses into degenerate repetition
        when pushed hard. Use topic_pull_tradeoff() to see the curve. The missing
        piece for LLM-like behaviour is a high-capacity learned P(next|context),
        not this re-ranking lever."""
        if not hasattr(self, "_wordgen"):
            raise RuntimeError("call learn_word_generator(sentences) first")
        return self._wordgen.generate(seed, length=length, topic_weight=topic_weight,
                                      temperature=temperature, seed_rng=seed_rng)

    def topic_pull_tradeoff(self, seeds, weights=(0.0, 2.0, 8.0, 16.0), length=40):
        """The honest experiment surfaced on the brain: for each topic_weight,
        mean coherence, transition validity, and lexical diversity. Coherence that
        'rises' only as diversity collapses is the metric being gamed by repetition,
        not real on-topic language -- the kept negative that explains why deeper
        conditioning alone does not make the brain an LLM."""
        if not hasattr(self, "_wordgen"):
            raise RuntimeError("call learn_word_generator(sentences) first")
        return self._wordgen.sweep(seeds, weights=weights, length=length)

    def _seq_mem(self):
        if self._sequences is None:
            from holographic.misc.holographic_sequence import SequenceMemory
            # SHARE the encoder's symbol atoms, so sequence steps are the very
            # same vectors the rest of the mind uses -- a plan's steps can be
            # the labels it classifies, the records it relates, all one space
            self._sequences = SequenceMemory(dim=self.dim, vocab=self.encoder._symbols)
        return self._sequences

    def learn_hierarchical(self, name, observations):
        """Absorb observations of a possibly-NESTED plan. Each observation is a
        plan whose steps may themselves be (sub_name, [sub_steps]) pairs OR bare
        atomic step names. Stored so discover_hierarchy can test, recursively
        and by the SAME permutation test, which steps expand into ordered
        sub-plans -- structure unfolding fractally, one layer at a time."""
        if not hasattr(self, "_hier_obs"):
            self._hier_obs = {}
        self._hier_obs[name] = list(observations)
        return self

    def discover_hierarchy(self, name=None, z_threshold=2.0, _depth=0, _max_depth=6):
        """RECURSIVE self-discovery of order. Test the top-level observations for
        sequential structure (the permutation test); if they pass, self-assemble
        the canonical order, then for each step gather its OWN sub-observations
        (where the data provides them) and recurse -- the same test one layer
        down. The recursion STOPS honestly: at a step with no sub-observations,
        or whose sub-observations show no sequential signal (z below the same
        bar), or at a depth guard. Returns a nested tree:
            {step: subtree or None}  (None = atomic / order not found)
        so the discovered hierarchy is the data's own, measured at every layer,
        with no depth or shape declared in advance."""
        from holographic.misc.holographic_sequence import sequentiality_z
        obs = (self._hier_obs.get(name) if name is not None
               else getattr(self, "_hier_obs", {}).get(None))
        if not obs or _depth >= _max_depth:
            return None
        # an observation is a list of steps; a step is either a bare name or a
        # (sub_name, [sub_steps]) pair. Split into the top-level order view and
        # a per-step bag of sub-observations.
        top_members, sub_obs = [], {}
        for ob in obs:
            order = []
            for step in ob:
                if isinstance(step, (list, tuple)) and len(step) == 2 \
                        and isinstance(step[1], (list, tuple)):
                    sub_name, sub_steps = step
                    order.append(sub_name)
                    sub_obs.setdefault(sub_name, []).append(list(sub_steps))
                else:
                    order.append(step)
            top_members.append(order)
        z = sequentiality_z(top_members, self.encoder._symbols)
        if z < z_threshold:
            return None                                  # no order here: stop
        canonical = self._canonical_order(top_members)
        tree = {}
        for step in canonical:
            children = sub_obs.get(step)
            if not children:
                tree[step] = None                        # atomic: honest stop
                continue
            # recurse: does THIS step's sub-observations carry order?
            sub_z = sequentiality_z(children, self.encoder._symbols)
            if sub_z < z_threshold:
                tree[step] = None                        # sub-steps are a bag: stop
            else:
                # stash and recurse one layer down
                key = (name, step, _depth)
                self._hier_obs[key] = children
                tree[step] = self.discover_hierarchy(key, z_threshold,
                                                     _depth + 1, _max_depth)
                if tree[step] is None:                   # recursion bottomed out
                    tree[step] = self._canonical_order(children)
        return tree

    def learn_sequences(self, labeled_sequences):
        """Absorb (sequence, label) pairs, KEEPING the raw ordered members per
        label so the mind can later DISCOVER which classes are genuinely
        sequential. Each sequence is also encoded order-free into the normal
        memory (a bag of its elements) for classification -- the two views
        coexist; discovery decides which matters for each class."""
        if not hasattr(self, "_seq_members"):
            self._seq_members = {}
        for seq, label in labeled_sequences:
            self._seq_members.setdefault(label, []).append(list(seq))
            # order-free view into the standard memory (classification still works)
            self.learn(list(seq), label, modality="text")
        return self

    def discover_sequential(self, z_threshold=2.0):
        """SELF-DISCOVERY of order. For every absorbed class, run the permutation
        test (real order vs the class's own shuffled null) and report which
        classes carry genuine sequential structure -- no magic constant, the
        class is measured against itself, and z>2 is the standard significance
        bar (signal exceeds two sigma of the null), not a tuned threshold.

        Classes that pass get an order-aware prototype in the sequence memory
        (their canonical order recovered), so precedes()/validate_plan() work on
        the discovered structure. Returns {label: z_score} for all tested
        classes, so the continuous evidence is visible, not just the verdict."""
        from holographic.misc.holographic_sequence import sequentiality_z
        if not getattr(self, "_seq_members", None):
            return {}
        verdicts = {}
        for label, members in self._seq_members.items():
            z = sequentiality_z(members, self.encoder._symbols)
            verdicts[label] = round(z, 2)
            if z >= z_threshold:
                # the class scored sequential AND must PROVE executable (no
                # precedence cycle) before its order is trusted -- statistical
                # signal is necessary but not sufficient; the structure has to
                # be consistent enough to actually walk
                ok, _ = self.prove_executable(members)
                if ok:
                    canonical = self._canonical_order(members)
                    self._seq_mem().add(label, canonical)
                    verdicts[label] = (z, "executable")
                else:
                    verdicts[label] = (round(z, 2), "inconsistent")
        return verdicts

    def execute_plan(self, name, context=None, attempt_order=None, templates=None):
        """RUN a discovered, proven plan -- the loop from discovering structure to
        ACTING on it. The contract is honest: a step fires only when (a) every
        step that must PRECEDE it (by the discovered canonical order) has already
        fired, and (b) its context SLOTS can be bound from `context`. Otherwise it
        BLOCKS, and the block is reported with its reason -- an unmet precondition
        or an unbound slot -- rather than silently assumed away.

        `context` is a dict binding slot names to values (the scenario: the
        physics law is generic, context supplies m and a; 'open the book' needs
        'book' bound). `templates` optionally maps a step to (template, slot_keys)
        from extract_template, so a step's slots are filled from context as it
        fires. `attempt_order` defaults to the canonical order (the natural run);
        pass a different order to test what blocks.

        Returns a log: [(step, status, detail)] where status is 'fired' (with the
        bound form in detail) or 'blocked' (with the reason). A plan that wasn't
        registered (failed discovery/proof) raises -- you cannot run what was
        never proven."""
        if name not in self._seq_mem().seqs:
            raise ValueError(f"'{name}' is not a proven sequential plan -- "
                             "discover_sequential must register it first")
        order = self._seq_mem().seqs[name][1]
        context = context or {}
        templates = templates or {}
        attempt = attempt_order or order
        done, log = set(), []
        for step in attempt:
            idx = order.index(step)
            required = set(order[:idx])
            missing = required - done
            if missing:
                log.append((step, "blocked",
                            f"preconditions unmet: needs {sorted(missing)}"))
                continue
            # bind slots from context if this step is a template
            if step in templates:
                template, slot_keys = templates[step]
                unbound = [k for k in slot_keys if k not in context]
                if unbound:
                    log.append((step, "blocked",
                                f"context missing bindings: {unbound}"))
                    continue
                # fill each <_> in order from the slot_keys' context values
                vals = [context[k] for k in slot_keys]
                parts, vi = [], 0
                for t in template:
                    if t == "<_>":
                        parts.append(str(vals[vi])); vi += 1
                    else:
                        parts.append(t)
                done.add(step)
                log.append((step, "fired", " ".join(parts)))
            else:
                done.add(step)
                log.append((step, "fired", ""))
        return log

    def extract_template(self, observations):
        """DISCOVER the generic schema and its context-bound slots in a repeated
        step. A step like 'the material has density X' is a SCHEMA (fixed words:
        'the material has density') plus a SLOT filled from context (X = 5g, 3g,
        ...). Across observations the schema positions are STABLE and the slot
        positions VARY -- so token-entropy per position separates them, and the
        split is placed at the natural largest GAP in the entropy distribution
        (the data's own scale, no magic cutoff). This is the same insight as a
        physical law: 'F = m*a' is generic until a scenario BINDS m and a; the
        schema is the law, the slots are where context enters.

        Returns (template, slots): template is the step with slots marked as
        '<_>', slots is {position: [observed values]} -- the variable parts and
        what context has filled them with. A step with no varying position is
        fully explicit (empty slots)."""
        import numpy as np
        from collections import Counter
        obs = [list(o) for o in observations]
        if len(obs) < 2:
            return (list(obs[0]) if obs else []), {}
        L = min(len(o) for o in obs)

        def entropy(toks):
            c = Counter(toks); n = len(toks)
            return -sum((v / n) * np.log2(v / n) for v in c.values())

        ents = np.array([entropy([o[i] for o in obs]) for i in range(L)])
        order = np.sort(ents)
        if order[-1] - order[0] < 1e-6:
            return [obs[0][i] for i in range(L)], {}   # all positions stable: explicit
        gaps = np.diff(order)
        split = order[int(np.argmax(gaps))] + gaps.max() / 2  # cut at the biggest gap
        slots = {}
        template = []
        for i in range(L):
            if ents[i] > split:
                slots[i] = [o[i] for o in obs]
                template.append("<_>")
            else:
                template.append(obs[0][i])
        return template, slots

    def prove_executable(self, members):
        """SELF-PROOF that a discovered order is VALID, not merely predictable.
        A class can score z>2 (its order carries signal) and still be
        INCONSISTENT -- if its members' pairwise precedences form a cycle
        (A before B, B before C, C before A), no single ordering satisfies them
        all and the 'plan' cannot be executed. The proof: build the majority
        precedence edges, take the canonical (vote-sorted) order, and check that
        EVERY majority edge is respected by it. A cycle shows up as an edge the
        sorted order must violate. Returns (ok, violations): ok means the
        structure proved itself executable; violations name the contradictory
        precedences. Structure earns trust by passing this, not by z alone."""
        from collections import Counter
        before = Counter()
        elems = set()
        for m in members:
            for i, a in enumerate(m):
                elems.add(a)
                for b in m[i + 1:]:
                    before[(str(a), str(b))] += 1
        order = self._canonical_order(members)
        pos = {e: i for i, e in enumerate(order)}
        # a ROBUST majority edge a->b that the canonical order reverses signals a
        # real cycle. "Robust" = the majority is stronger than the sampling noise
        # of sparse partial observations: a single contradictory vote from a rare
        # pair is not a contradiction, a consistent reversal is. The bar is the
        # data's own: an edge counts only if its margin exceeds the median edge
        # margin (so typical-strength evidence, not a fluke), and the canonical
        # order must still reverse it. No fixed constant -- the observation set
        # sets the scale.
        margins = [abs(before[(a, b)] - before[(b, a)])
                   for a in elems for b in elems if a < b]
        import numpy as np
        med = np.median([mg for mg in margins if mg > 0]) if any(margins) else 0
        violations = []
        for a in elems:
            for b in elems:
                if a != b and pos[a] > pos[b]:
                    margin = before[(a, b)] - before[(b, a)]
                    if margin > 0 and margin >= med:        # robust reversed edge
                        violations.append((a, b))
        return (not violations), violations

    def _canonical_order(self, members):
        """Recover a class's canonical step order from its member sequences by a
        true TOPOLOGICAL SORT over the majority-precedence edges -- so the
        discovered order RESPECTS what the data agrees on (if cut beats plate
        4-0, cut comes first), not a score heuristic that can misplace rare
        elements. Edges are the net majority (a before b more often than after);
        ties and the occasional cycle-inducing weak edge are broken by net
        margin, so a consistent dataset yields its exact order and a contradictory
        one yields the least-bad order (whose remaining violations prove_executable
        then surfaces)."""
        from collections import Counter
        before = Counter()
        elems = set()
        for m in members:
            for i, a in enumerate(m):
                elems.add(a)
                for b in m[i + 1:]:
                    before[(str(a), str(b))] += 1
        elems = [str(e) for e in elems]
        # net majority edge weight a->b (positive => a should precede b)
        def net(a, b):
            return before[(a, b)] - before[(b, a)]
        # Kahn-style topological sort, picking among available nodes the one with
        # the strongest outgoing majority (greedy by net precedence) so stronger
        # evidence is honoured first; this respects every consistent edge exactly.
        remaining = set(elems)
        order = []
        while remaining:
            # a node is "available" if no remaining node has a majority edge INTO it
            avail = [e for e in remaining
                     if not any(net(o, e) > 0 for o in remaining if o != e)]
            if not avail:                                  # a cycle: break it by
                # picking the node with the best net outgoing balance
                avail = [max(remaining,
                             key=lambda e: sum(net(e, o) for o in remaining if o != e))]
            pick = max(avail, key=lambda e: sum(net(e, o) for o in remaining if o != e))
            order.append(pick)
            remaining.discard(pick)
        return order

    def learn_plan(self, name, steps, chunk=0):
        """Store an ORDERED plan/recipe/protocol by name. Unlike absorb (which
        files things order-free for classification and recall), this keeps the
        SEQUENCE queryable: meaning that lives in the order is preserved.

        For a LONG plan (a scientist's many-step protocol, a long itinerary), pass
        `chunk` (e.g. 14): the plan is stored as positional blocks so step_at /
        precedes / validate_plan stay EXACT past the single-bundle cap (the positional
        encoding alone caps with length -- ~100% to length ~50-100, decaying to ~15%
        by 800 at dim 2048). chunk=0 (default) is the original storage, ideal for short
        plans where chunking is a no-op."""
        self._seq_mem().add(name, steps, chunk=chunk)
        return self

    def step_at(self, name, i):
        """What is the i-th step of a stored plan?"""
        return self._seq_mem().step(name, i)

    def precedes(self, name, a, b):
        """In the stored plan, does step a come before step b? The order
        relation no bag store can answer (measured exact to ~40 steps at dim
        2048, ~93% at 120 -- graceful, not a hard cliff)."""
        return self._seq_mem().precedes(name, a, b)

    def validate_plan(self, name_or_steps, constraints):
        """Check a plan against ordering rules -- the PB&J test: does every
        'a must come before b' hold? Returns (ok, violations); a violation
        names exactly which step is out of order. Works on a stored plan name
        or a fresh step list."""
        return self._seq_mem().validate(name_or_steps, constraints)

    # -- executable procedures: HoloMachine as a faculty of the mind ---------------------------
    # A learn_plan stores an ordered list of opaque step LABELS ("beat the eggs"); a PROCEDURE stores
    # an executable recipe whose steps are actual VSA operations (LOAD/BIND/BUNDLE/PERMUTE/CALL), so
    # the recipe DOES something. The VM shares the mind's dim and seed, so a procedure's accumulator is
    # a vector in the mind's OWN space (seed it with a mind vector to transform it) and the format is
    # deterministic. This de-silos the stored-program machine: it is now a faculty, not an island.
    def _machine(self):
        """The HoloMachine VM, lazily built at the mind's dim & seed so procedures share the substrate.
        `mind.vm_fast_cleanup = True` BEFORE first use opts the VM's decode into the cached-codebook SIMD cleanup
        (measured 2x end-to-end with the atom cache; result-identical, pinned by test) -- default off per the
        never-flip rule."""
        if getattr(self, "_machine_vm", None) is None:
            from holographic.agents_and_reasoning.holographic_machine import HoloMachine
            self._machine_vm = HoloMachine(dim=self.dim, seed=self.seed,
                                           fast_cleanup=bool(getattr(self, "vm_fast_cleanup", False)),
                                           decode_plan=bool(getattr(self, "_vm_decode_plan", False)))
        return self._machine_vm

    def vm_decode_plan(self, on=True):
        """Turn the VM's DECODED-INSTRUCTION CACHE on (default) or off -- measured 6.7x-14x end-to-end on the
        procedure interpreter, with bit-identical accumulators and identical traces (126/126 programs across
        three dimensions, three seeds and both cleanup settings).

        WHY IT PAYS: decoding an instruction is a pure function of (program vector, address) -- it never reads
        the accumulator -- so the unplanned interpreter re-derives the same eight transforms every time the
        program counter revisits an address. A 64-iteration ITERATE over a 2-instruction body was measured
        doing 131 address decodes over 5 distinct addresses (26x redundancy). With the plan on, a whole block
        of addresses is decoded in ONE batched spectral sweep and every later visit is a dict lookup.

        Safe to leave on: the spectral half is bit-identical (batched FFT returns the same bytes as per-row,
        and the address keys are cached as rfft(involution(pos)) rather than the merely-5e-16-equal conjugate
        shortcut), and the codebook half re-arbitrates near-ties through the exact loop, so no regrouped float
        can flip a symbol on a different BLAS. Off by default anyway, per the never-flip rule.
        See holographic_vmplan.DecodePlan."""
        M = self._machine()
        self._vm_decode_plan = bool(on)
        M._plan_on = bool(on)
        if not on:
            M._plan = None                     # drop the cache; the library and defined procedures survive
        return self

    def vm_plan_stats(self):
        """Decoded-instruction cache telemetry: {hits, misses, sweeps, programs, hit_rate}, or None when the
        plan is off. `sweeps` is the number that matters -- it counts the times real spectral decode work
        actually happened, and on a loop-heavy procedure it should stay in single digits however many
        iterations run. A hit_rate that will not climb means the programs are not being reused (each CALL
        rebuilds its body, so those hit by CONTENT, not identity). See holographic_vmplan.DecodePlan.stats."""
        p = self._machine().plan()
        return None if p is None else p.stats()

    def learn_procedure(self, name, program):
        """Store a named PROCEDURE -- an executable ACC->ACC recipe of VSA operations, assembled into
        ONE hypervector and held in the machine's library, callable by name and composable (a procedure
        may CALL procedures defined earlier). `program` is a list of (opcode, operand). Define a
        procedure before any program that CALLs it."""
        self._machine().define(name, program)
        return self

    def gradient_cache_symbolic(self, expr, anchors, variables=("x", "y", "z")):
        """Build an irradiance/GI-style GradientCache with EXACT Jacobians from a symbolic field (SymPy) instead of
        finite differences -- no truncation error in the cached gradients, so first-order interpolation is more
        accurate at the same anchors. Needs sympy. See holographic_cache.gradient_cache_symbolic."""
        from holographic.caching_and_storage.holographic_cache import gradient_cache_symbolic
        return gradient_cache_symbolic(expr, anchors, variables)

    def render_sdf_fast(self, expr, camera, width=256, height=256, light_dir=(-0.4, 0.7, -0.3),
                        base_color=(0.85, 0.5, 0.35), ao=True, shadows=True, ambient=0.25, sky=None):
        """Render an analytic SDF (given as a symbolic expression) with the fully-JIT'd renderer: the whole march --
        primary ray, exact normal, AO, soft shadow -- compiles into one njit kernel (the closure barrier is gone),
        ~9-15x the numpy renderer for the field-native shading. Compiled renderer cached per SDF. Needs sympy+numba;
        falls back is the caller's (use render_sdf without jit_expr). See holographic_sdf_render.render_analytic."""
        from holographic.rendering.holographic_sdf_render import render_analytic
        return render_analytic(expr, camera, width=width, height=height, light_dir=light_dir,
                               base_color=base_color, ao=ao, shadows=shadows, ambient=ambient, sky=sky)

    def compiled_sdf_numba(self, expr, variables=("x", "y", "z")):
        """SymPy -> Numba, cached: compile a symbolic 3-D SDF to njit scalar+grid value/normal kernels ONCE and
        reuse them. The scalar njit SDF composes into other njit loops (a sphere-trace march) -- the closure barrier
        that blocked Numba from the raymarch is gone. Needs sympy + numba. See holographic_compile.compiled_sdf_numba."""
        from holographic.scene_and_pipeline.holographic_compile import compiled_sdf_numba
        return compiled_sdf_numba(expr, variables)

    def compile_program(self, program):
        """Assemble a HoloMachine program (list of (opcode, operand)) into its program vector ONCE via the compile
        cache and reuse it -- re-running the SAME program skips the ~L-bind assembly (measured ~15 ms / 60 instr).
        Returns the cached program vector. See holographic_compile.compiled_program."""
        from holographic.scene_and_pipeline.holographic_compile import compiled_program
        return compiled_program(self._machine(), program)

    def compiled_sdf_normal(self, expr, variables=("x", "y", "z")):
        """Compile a symbolic SDF's exact normal ONCE and reuse it via the content-addressed compile cache: the
        same expr returns the cached (value_fn, normal_fn) instantly instead of re-running the ~140-390 ms sympy
        lambdify, and recompiles only when the expr changes. The runtime use of the codegen pipeline -- compile a
        spec, cache the compiled version, hand it out everywhere. See holographic_compile.compiled_sdf_normal."""
        from holographic.scene_and_pipeline.holographic_compile import compiled_sdf_normal
        return compiled_sdf_normal(expr, variables)

    def compile_cache_stats(self):
        """Stats for the process-wide compile cache: hits/misses/compiles/evictions, size, hit_rate. See
        holographic_compile.DEFAULT_CACHE."""
        from holographic.scene_and_pipeline.holographic_compile import DEFAULT_CACHE
        return dict(DEFAULT_CACHE.stats, size=len(DEFAULT_CACHE), hit_rate=round(DEFAULT_CACHE.hit_rate(), 3))

    def exact_sdf_normal(self, expr, variables=("x", "y", "z")):
        """Derive an EXACT SDF surface normal from a symbolic SDF expression (SymPy, design-time) and return
        (value_fn, normal_fn) of pure NumPy -- no finite-difference step-size error, no autodiff. The Quilez-seat
        path: e.g. exact_sdf_normal('sqrt(x**2+y**2+z**2)-1.0'). Needs sympy (requirements-accel.txt); the returned
        functions are pure NumPy. See holographic_codegen.sdf_normal_fn."""
        from holographic.misc.holographic_codegen import sdf_normal_fn
        return sdf_normal_fn(expr, variables)

    def symbolic_gradient(self, expr, variables):
        """Exact gradient of a symbolic scalar field as a pure-NumPy function (force = -symbolic_gradient(energy)).
        The Baker-seat analytic-force path, autodiff-free. See holographic_codegen.gradient_fn."""
        from holographic.misc.holographic_codegen import gradient_fn
        return gradient_fn(expr, variables)

    def fft_backend(self, use_pyfftw=None):
        """Report or switch the FFT backend behind bind/bundle. Default 'numpy' is bit-exact and deterministic;
        pass use_pyfftw=True to opt into pyFFTW (MEASURED to regress at typical dims -- see fft_benchmark; off by
        default for good reason). Returns the active backend name."""
        from holographic.sampling_and_signal.holographic_fft import use_pyfftw as _u, fft_backend as _b
        if use_pyfftw is not None:
            return _u(bool(use_pyfftw))
        return _b()

    def fft_benchmark(self):
        """Reproduce the numpy-vs-pyFFTW comparison (ratios <1 mean pyFFTW is slower). Documents why numpy stays
        the default. See holographic_fft.benchmark."""
        from holographic.sampling_and_signal.holographic_fft import benchmark
        return benchmark()

    def signed_distance_field(self, inside_mask, h=1.0):
        """Occupancy/inside mask -> signed distance field (negative inside, positive outside) via the fast-sweeping
        eikonal solver, Numba-accelerated when numba is installed (measured ~270x on a 256^2 grid, bit-identical to
        the pure path) and pure-Python otherwise. The occupancy->SDF step the modelling/raymarch pipeline wants.
        See holographic_jit.signed_distance_2d."""
        from holographic.misc.holographic_jit import signed_distance_2d
        return signed_distance_2d(inside_mask, h=h)

    def parse_scene_description(self, text):
        """Parse a controlled 3-D scene DESCRIPTION into objects + environment (holographic_semantic). Each object is
        {shape, color, material, size, relation}. Controlled vocabulary + keyword grammar, deterministic -- NOT a
        general language model (see the module SCOPE note)."""
        from holographic.simulation_and_physics.holographic_semantic import parse_description
        return parse_description(text)

    def build_scene(self, text):
        """DESCRIBE a scene in plain words and get back a LIVE, adjustable scene the system built for you -- a
        SemanticScene of NAMED objects plus the environment. Then talk to it: scene.adjust('make the sphere bigger'),
        scene.render(), scene.simulate(), scene.describe(). The one-call 'describe it and let the engine create it'
        entry point; controlled-vocabulary + deterministic (see holographic_scene_semantic)."""
        from holographic.simulation_and_physics.holographic_scene_semantic import scene_from_description
        return scene_from_description(text, mind=self)

    def scene_from_image(self, image, k=5, seed=0, max_objects=3, background=False):
        """BUILD A SCENE FROM A PHOTO (machine-initialised, not hand-authored): segment the image into regions,
        keep the most object-like foreground ones, map each region's silhouette+colour to a primitive, and assemble a
        live SemanticScene you can adjust()/render()/refine_to_target()/to_node_graph(). Returns {scene, regions,
        roles, objects}. Deterministic. HONEST: shape comes from the 2-D silhouette and colour from the region mean;
        DEPTH is not reconstructed (shape-from-shading is a brightness relief, measured too degenerate to fit
        primitives to), so z=0 and front/back ordering is not recovered -- a machine STARTING POINT the critic and
        node drill-down then refine. See holographic_scene_semantic.scene_from_image."""
        from holographic.simulation_and_physics.holographic_scene_semantic import scene_from_image
        return scene_from_image(image, k=k, seed=seed, max_objects=max_objects, background=background, mind=self)


    def semantic_scene(self, objects, environment=None):
        """Wrap an EXISTING list of scene objects ({shape,color,material,size,...}) as a SemanticScene so you can
        reference its named objects and adjust them in words -- scene.set('the red sphere', material='glass'),
        scene.adjust('make everything matte'). For starting from a scene you already have rather than a description.
        See holographic_scene_semantic.SemanticScene."""
        from holographic.simulation_and_physics.holographic_scene_semantic import SemanticScene
        return SemanticScene(objects, environment=environment, mind=self)

    # ---- composable texture map graph (CMP1) --------------------------------------------------------------
    def texture_leaf(self, source=None, value=None, **kw):
        """Make a LEAF for a texture graph. Give `source=` a Texture name ('fbm','voronoi','synth',...) to wrap that
        field, e.g. mind.texture_leaf('fbm', n_dims=2); or give `value=` a number or a length-3/4 colour for a
        constant, e.g. mind.texture_leaf(value=[1,0,0]). Leaves are the inputs you feed to texture_op. See
        holographic_texturegraph."""
        from holographic.materials_and_texture.holographic_texturegraph import field_leaf, Const
        if source is not None:
            return field_leaf(source, **kw)
        if value is not None:
            return Const(value)
        raise ValueError("texture_leaf needs source= (a Texture name) or value= (a number/colour)")

    def texture_op(self, op, **inputs):
        """Compose a texture-map NODE: an op ('mix','multiply','over','scale','add','remap','min','max','clamp',
        'saturate') over TYPED inputs, each of which may be a leaf OR another texture_op -- so graphs nest to any
        depth. The input types are checked HERE, at compose time, so a bad graph (a colour used as a weight, a missing
        input, an unknown op) is refused up front with a clear message. Sample it with mind.sample_texture(node, uv);
        wrap it in 'saturate' to keep colours in [0,1]. See holographic_texturegraph."""
        from holographic.materials_and_texture.holographic_texturegraph import Map
        return Map(op, **inputs)

    def sample_texture(self, node, uv):
        """Sample a texture graph at a UV/point -> a value (a scalar, or an rgb colour if the graph produces one).
        Walks the tree: evaluate each child, then apply the op."""
        return node.sample(uv)

    def encode_texture(self, node):
        """The texture graph as ONE hypervector (this mind's dim/seed), for CACHING a baked result by its graph
        identity or SEARCHING a library of graphs -- structurally identical graphs encode identically. The readable
        object tree stays the source of truth; this is the derived vector form."""
        from holographic.materials_and_texture.holographic_texturegraph import encode
        return encode(node, self.dim, self.seed)

    # ---- multi-material blended/selected by masks (CMP3) --------------------------------------------------
    def multi_material(self, materials, weights, mode="blend", normalize=True):
        """Combine N Materials by per-point MASKS -- generalises Material.blend (2-way, one scalar) to N materials
        each weighted by a mask that varies over the surface. Each weight is a CMP1 texture graph (mind.texture_op),
        a field, or a constant; at a point a channel reads sum_i w_i(uv) * material_i.sample(channel, uv). mode='blend'
        is a soft weighted sum (weights normalised to a partition of unity so brightness stays put -- the kept
        negative); mode='select' hard-picks the dominant material (a material-ID / splat map). Returns a MultiMaterial
        you sample with .sample(channel, uv) / .sample_all(uv). See holographic_multimaterial."""
        from holographic.materials_and_texture.holographic_multimaterial import MultiMaterial
        return MultiMaterial(materials, weights, mode=mode, normalize=normalize)

    # ---- layered materials with an order schema (CMP2) ---------------------------------------------------
    def material_layer(self, kind, material, alpha=1.0):
        """Make one LAYER of a layered material: a `kind` ('base','diffuse','specular','reflection','coat','clearcoat')
        that fixes its place in the order, the Material carrying its channels, and a coverage `alpha` (how much shows
        through what's below -- a number, a field, or a CMP1 texture graph, so coverage can vary over the surface).
        Feed these to mind.layered_material. See holographic_layeredmaterial."""
        from holographic.materials_and_texture.holographic_layeredmaterial import Layer
        return Layer(kind, material, alpha=alpha)

    def layered_material(self, layers):
        """Stack material LAYERS bottom-to-top with the ORDER enforced: base < diffuse < specular/reflection <
        coat/clearcoat, so you cannot put a reflection under a diffuse -- an out-of-order stack is refused at COMPOSE
        time with a clear message, not rendered wrong. Each layer composites OVER the one below by its coverage alpha.
        Returns a LayeredMaterial you sample with .sample(channel, uv). Honest boundary: this fixes the STACKING, not
        the radiometry -- a physically energy-conserving layered BRDF is a separate, harder thing. See
        holographic_layeredmaterial."""
        from holographic.materials_and_texture.holographic_layeredmaterial import LayeredMaterial
        return LayeredMaterial(layers)

    # ---- type-correct binding + shared-definition instancing (CMP4) --------------------------------------
    def shared_definition(self, name, geometry, material, geometry_kind=None):
        """A shared, editable scene DEFINITION -- geometry bound to a material, with the binding TYPE-CHECKED at
        compose time: a surface material (paint/metal/glass) only binds to a mesh, a volumetric material (fog/smoke/
        fire) only to a volume; a bad binding is refused with a clear message. Place it many times in an
        instanced_scene and edit it ONCE (def.set_material / set_geometry) to update every instance. Returns a
        Definition. See holographic_instancing."""
        from holographic.misc.holographic_instancing import Definition
        return Definition(name, geometry, material, geometry_kind=geometry_kind)

    def instanced_scene(self):
        """An empty INSTANCED scene: place shared Definitions through transforms (scene.place(defn, transform)) so many
        instances share one definition -- editing the definition updates them all (edit-once). flatten_surface()
        materialises the surface instances into one mesh. Honest boundary: sharing is edit-once at the graph level;
        flatten is where instances become concrete geometry. See holographic_instancing."""
        from holographic.misc.holographic_instancing import InstancedScene
        return InstancedScene()

    # ---- pipeline composes the graphs: bake vs live (CMP5) -----------------------------------------------
    def render_graph(self, res=64):
        """An ORCHESTRATOR that prepares the CMP1-CMP4 graphs into a render-ready scene as pipeline stages. Register
        texture graphs (rg.add_texture(name, graph, static=True/False)) and a CMP4 instanced scene (rg.set_scene(...));
        rg.plan() shows what it will do and WHY (bake each STATIC texture to a grid for O(1) lookup, keep dynamic ones
        live, then bind+flatten the scene); rg.prepare() runs it and returns a PreparedScene. This is where 'adaptive'
        reaches down to the maps. See holographic_rendergraph."""
        from holographic.rendering.holographic_rendergraph import RenderGraph
        return RenderGraph(res=res)

    def bake_texture(self, graph, res=64, lo=0.0, hi=1.0):
        """Bake a CMP1 texture graph to a res x res grid -> a BakedTexture you sample in O(1) (bilinear lookup),
        instead of walking the graph every hit. Do this for a STATIC map sampled many times; the trade is memory +
        interpolation error (blurs detail finer than a cell -- raise res, or keep sharp maps live). See
        holographic_rendergraph."""
        from holographic.rendering.holographic_rendergraph import bake_texture
        return bake_texture(graph, res=res, lo=lo, hi=hi)

    # ---- SEE what you composed: previews for the composability stack -------------------------------------
    def preview_texture(self, graph, res=256):
        """Render a CMP1 texture graph as a flat RGB SWATCH -- a (res,res,3) float image in [0,1] you can save/view.
        Colour graph -> its rgb; scalar graph -> greyscale; out-of-range values are clamped for display. The missing
        step between 'I composed a texture' and 'let me look at it'. See holographic_preview."""
        from holographic.misc.holographic_preview import texture_image
        return texture_image(graph, res=res)

    def preview_material(self, material, res=192, base_color=(0.82, 0.80, 0.78)):
        """Render a material on a preview SPHERE -- the classic MATERIAL BALL. Works on a plain Material or a CMP2/CMP3
        layered/multi material: uses its roughness/metallic channels (else defaults), modulates base_color by an albedo
        channel if present, and shades with the same Cook-Torrance BRDF the real renderer uses. Returns a (res,res,3)
        float image in [0,1]. See holographic_preview."""
        from holographic.misc.holographic_preview import material_ball
        return material_ball(material, res=res, base_color=base_color)

    def preview_thumbnail_batch(self, materials, res=96, quality="draft", seed=None, fmt="png", out_res=None,
                                size=None, upsample=False):
        """MANY material thumbnails, fast: the camera and geometry are fixed, so the neutral reference frame
        and the active-pixel mask are rendered ONCE per (res, quality, seed) and cached for the process
        lifetime; each material then re-renders only the ~48% of pixels that can see the ball and composites
        the rest from the reference in linear light. Returns a list (PNG bytes by default, fmt='array' for
        floats) aligned with `materials` (matlib names / material objects / PBR dicts). Measured at res=96
        draft: 36 s full -> 26 s per material with the cache warm; first call pays one extra reference render.
        Composite-vs-full difference sits below the draft sampler's own seed-to-seed noise. out_res=N returns
        N-px images at res-px lighting cost (demodulated upscale with 2x-carrier coverage AA -- the
        anti-aliasing happens AFTER the upscale; transmissive outers
        auto-route to a native render -- refraction detail mushes under demod, measured). For a single
        never-composited frame use preview_thumbnail."""
        import holographic.misc.holographic_preview as _hp
        return _hp.preview_thumbnail_batch(materials, res=res, quality=quality,
                                           seed=self.seed if seed is None else seed, fmt=fmt, out_res=out_res,
                                           size=size, upsample=upsample)

    def preview_thumbnail(self, material=None, res=96, quality="draft", seed=None, fmt="png",
                          core=None, trim=None, trim_top=None, trim_bottom=None, base=None, out_res=None,
                          size=None, upsample=False):
        """ONE call: feed a material (matlib name, material object, or plain PBR dict {'base_color':...,
        'roughness':..., 'metallic':..., 'emissive':...}), get a THUMBNAIL of it on the shader ball back.
        Every fixture slot stays the neutral grey diffuse unless overridden, so the thumbnail is about the
        material. fmt='png' (default) returns PNG bytes -- over HTTP /invoke they travel as
        {'__bytes_b64__': ...}, ready to write to disk or hand to a UI; fmt='array' returns the raw float
        image. size=N asks for ANY delivery size: upsample=False (default) renders NATIVE at N (exact door);
        upsample=True takes the fast cached path, routing each material by where its detail lives --
        diffuse/rough get the demod upscale, transmissive and smooth-metal outers get a masked NATIVE
        render at N (measured: sampling metal reflections back on the upscale path costs more than native).
        Warm at 160: wax ~21 s, chrome ~37 s. res/out_res remain for direct control."""
        import holographic.misc.holographic_preview as _hp
        return _hp.preview_thumbnail(material, res=res, quality=quality,
                                     seed=self.seed if seed is None else seed, fmt=fmt, core=core, trim=trim,
                                     trim_top=trim_top, trim_bottom=trim_bottom, base=base, out_res=out_res,
                                     size=size, upsample=upsample)

    def preview_scene(self, material=None, core=None, trim=None, base=None, floor="matte_white",
                      res=192, quality="fast", seed=None, view="display", lighting="studio", floor_grid=True,
                      aa="fxaa", trim_top=None, trim_bottom=None):
        """Render the shader-ball PREVIEW SCENE: `material` on the classic COMPLEX preview object -- a hollow
        outer shell with a camera-facing cutaway window, a THIN LENS dish (translucency/SSS test region; see the core with almost no
        refraction), a CORE flush against the shell interior (its mesh light toggles ON automatically when the
        outer is translucent/SSS -- wax/skin/jade -- and stays OFF for glass/refractive/transparent and
        opaque outers), TWO FLUSH INLAY BELTS (trim_top above the window
        and lens, trim_bottom below -- cut into the ball, no outward bumps) and a wide thin puck base --
        on a graph-paper floor (floor_grid=False for plain), PATH-TRACED under a STUDIO RIG -- key/fill/rim
        softboxes, gradient backdrop with fluorescent ceiling panels in the reflections, off-axis
        window (lighting='plain' keeps the bare-renderer look; preview_material
        is the fast flat-lit thumbnail). The core slot is for the interacting cases:
        an emissive core glows THROUGH a glass shell (transmission) and through TRANSLUCENT outers (wax/skin/
        jade -- the interior-emission subsurface term, brightest at the thin lens), and the cutaway keeps the
        core visible under opaque outers. SLOT RULE: `material` dresses the OUTER; core and base default to the
        dark grey "90s mouse ball" diffuse, both belts to dark silicone; trim= dresses BOTH belts,
        trim_top=/trim_bottom= override each belt individually (e.g. glass top + chrome bottom); material=None -> neutral default diffuse on the outer;
        floor= styles the environment. Materials: matlib names, material objects, or plain PBR dicts.
        aa='fxaa' (default) cleans edge stair-stepping for milliseconds; 'ssaa2' true-supersamples (~4x
        time); 'off' is raw. Returns (res,res,3) float in [0,1]. Cost: ~75 s at res=160 opaque with the distance proxy, more with glass belts (the
        soft-light cache is OFF -- it paints false shadows on curved mirrors, measured); drop res to iterate.
        See holographic_preview.preview_scene / preview_scene_document (geometry + camera, no pixels)."""
        from holographic.misc.holographic_preview import preview_scene as _ps
        return _ps(material=material, core=core, trim=trim, base=base, floor=floor, res=res,
                   trim_top=trim_top, trim_bottom=trim_bottom,
                   quality=quality, seed=self.seed if seed is None else seed, view=view, lighting=lighting,
                   floor_grid=floor_grid, aa=aa)

    def quick_material(self, color=(0.8, 0.8, 0.8), roughness=0.5, metallic=0.0, res=192):
        """The material-editor SHORTCUT: plain numbers in, MATERIAL BALL image out -- no encoders, no channel
        fields. quick_material(color=(1,0.2,0.1), roughness=0.15, metallic=1.0) renders a polished red metal ball.
        This is the one-slider entry to the material system: it shades base_color with the SAME Cook-Torrance BRDF
        the real renderer uses at the given roughness/metallic, so what the ball shows is what a render does. For
        textured/layered materials build a real Material (channel fields) and use preview_material -- this shortcut
        deliberately carries no textures. Returns (res,res,3) float in [0,1]."""
        from holographic.misc.holographic_preview import material_ball

        class _Const:                          # the minimal duck: constant channels, no encoder needed
            channels = {"roughness": None, "metallic": None}

            def __init__(self, rough, metal):
                self._r, self._m = float(rough), float(metal)

            def sample(self, channel, uv):
                import numpy as np
                n = len(uv)
                if channel == "roughness":
                    return np.full(n, self._r)
                if channel == "metallic":
                    return np.full(n, self._m)
                return np.zeros(n)

        return material_ball(_Const(roughness, metallic), res=res, base_color=color)


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p11_encyclopedia_reset", "_UnifiedPart11")
    print("holographic_unified_p11_encyclopedia_reset selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

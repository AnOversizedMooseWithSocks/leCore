# lecore.py -- the friendly front door.
#
# The engine is ~436 `holographic_*.py` modules organized into family packages. A newcomer shouldn't need to know which
# one holds `Scene` versus `RenderSession` versus `look_at`. This module gathers the handful of things
# most callers actually want into five plain-English areas, so that after `pip install lecore` you can:
#
#     import lecore
#     doc = lecore.scene.Scene(dim=1024, seed=0)      # build a scene
#     img = lecore.render.path_trace(sdf, camera)     # render it
#     M   = lecore.transform.look_at(eye, target)     # aim a camera
#
# This is CURATION, not new engine code: every name below is re-exported from its real module, whose
# docstring stays authoritative. The full engine is still available directly (e.g.
# `from holographic_render import ...`) for anything not surfaced here.

import types

# ---------------------------------------------------------------------------------------------------
# The main entry point -- always present, imported eagerly so `lecore.UnifiedMind` works with no fuss.
# ---------------------------------------------------------------------------------------------------
from holographic.misc.holographic_unified import UnifiedMind

# Convenience: re-export the raw VSA algebra if it's present. Guarded so that a future rename can never
# break `import lecore` itself -- the mind is what matters, the loose ops are a nicety.
try:
    from holographic.agents_and_reasoning.holographic_ai import random_vector, unitary_vector, bind, unbind, involution
except Exception:                        # pragma: no cover
    pass


# ---------------------------------------------------------------------------------------------------
# The five curated areas.
#
# Each area is imported here and packed into a SimpleNamespace below. We keep the imports grouped by
# area (not alphabetised) so it reads as "here is everything the `scene` builder needs", etc. If any
# ONE of these underlying modules were ever renamed, that area's import would fail loudly at
# `import lecore` -- which is what we want for the curated surface (a silent hole is worse).
# ---------------------------------------------------------------------------------------------------

# scene -- author and store a scene document (objects, handles, transforms, undo snapshots).
from holographic.scene_and_pipeline.holographic_scene_doc import Scene, SceneObject

# model -- edit geometry: the modifier stack, object description, SDF primitives, key mesh verbs.
from holographic.misc.holographic_modifier import ModifierStack, describe_object
from holographic.mesh_and_geometry.holographic_sdf import sphere, box                       # SDF primitives  more live in holographic_sdf

# The TRANSFORM TOWER, at the top level, because it is not a geometry detail -- it is the sentence that says which
# floor every transform in the engine is standing on, and whether a delta can be pushed through it. Aff(n) = GL(n)
# semidirect R^n: translations are the abelian ideal, rotation and shear the non-commuting peers, scale the centre
# of the linear part. `classify_transform(fn)` answers "which floor" for anything; `TOWER` is the declared ladder;
# `affine_normality` is where the mechanism stops, at the projective ceiling.
from holographic.mesh_and_geometry.holographic_grouptower import (   # noqa: E402  -- the transform tower
    TOWER, classify_transform, commutator_table, hypervector_layer, semidirect_law)
from holographic.mesh_and_geometry.holographic_projectivetower import (   # noqa: E402  -- and its ceiling
    affine_normality, is_affine, texture_projection_error)
from holographic.mesh_and_geometry.holographic_meshverbs import extrude_face, inset_face, dissolve_vertex

# render -- turn a scene into pixels, with a config and a cooperative cancel handle.
from holographic.scene_and_pipeline.holographic_session import RenderSession
from holographic.rendering.holographic_pathtrace import path_trace
from holographic.misc.holographic_cancel import CancelToken
from holographic.scene_and_pipeline.holographic_pipeline import PipelineConfig

# sim -- physical simulation: shallow-water wave planner/solver, free surface, snow (MPM), stable fluid,
# and the particle<->grid transfer pair every sim needs.
from holographic.simulation_and_physics.holographic_waveadaptive import plan_waves, solve_waves
from holographic.mesh_and_geometry.holographic_freesurface import FreeSurface
from holographic.simulation_and_physics.holographic_mpm import MPMSnow
from holographic.simulation_and_physics.holographic_fluid import StableFluid
from holographic.misc.holographic_transfer import scatter, gather

# transform -- the gizmo / property-panel math kit: TRS decompose/compose, a full quaternion kit, look_at.
from holographic.misc.holographic_transform import decompose, compose_trs, look_at, quat_normalize, quat_mul, quat_from_axis_angle, quat_to_axis_angle, quat_from_matrix, quat_to_matrix, quat_from_euler, quat_to_euler, quat_slerp, quat_rotate


def _area(**members):
    """Pack a set of re-exported names into a lightweight namespace so `lecore.scene.Scene` works.
    We use SimpleNamespace rather than a real submodule on purpose: it's one readable object with no
    sys.modules bookkeeping, and the only access pattern callers need is attribute lookup."""
    return types.SimpleNamespace(**members)


# The five areas. These are the ONLY place a member is listed; areas() reads its map straight off these
# namespaces (see below) so the docs and the objects can never drift out of sync.
scene = _area(Scene=Scene, SceneObject=SceneObject)

model = _area(
    ModifierStack=ModifierStack, describe_object=describe_object,
    sphere=sphere, box=box,
    extrude_face=extrude_face, inset_face=inset_face, dissolve_vertex=dissolve_vertex,
)

render = _area(
    RenderSession=RenderSession, path_trace=path_trace,
    CancelToken=CancelToken, PipelineConfig=PipelineConfig,
)

sim = _area(
    plan_waves=plan_waves, solve_waves=solve_waves,
    FreeSurface=FreeSurface, MPMSnow=MPMSnow, StableFluid=StableFluid,
    scatter=scatter, gather=gather,
)

transform = _area(
    decompose=decompose, compose_trs=compose_trs, look_at=look_at,
    quat_normalize=quat_normalize, quat_mul=quat_mul,
    quat_from_axis_angle=quat_from_axis_angle, quat_to_axis_angle=quat_to_axis_angle,
    quat_from_matrix=quat_from_matrix, quat_to_matrix=quat_to_matrix,
    quat_from_euler=quat_from_euler, quat_to_euler=quat_to_euler,
    quat_slerp=quat_slerp, quat_rotate=quat_rotate,
)


# The names of the five areas, in the order a builder meets them (author -> model -> render -> sim ->
# aim). Kept as a tuple so `areas()` and any future __all__ share one source of truth.
_AREA_NAMES = ("scene", "model", "render", "sim", "transform")


def areas():
    """Map the curated surface: area name -> sorted list of the names it exposes.

    A one-call answer to "what's in the front door and where do I look?" -- e.g.
    `areas()["render"]` lists RenderSession, path_trace, CancelToken, PipelineConfig. Each name is a
    real object re-exported from its home module; consult that module's docstring for the details."""
    out = {}
    for name in _AREA_NAMES:
        ns = globals()[name]
        # vars() on a SimpleNamespace is its member dict; sorted for a stable, scannable listing.
        out[name] = sorted(vars(ns).keys())
    return out


def _read_version():
    """The single source of truth for the version is the VERSION file (CI bumps its patch digit on each merge).
    When leCore is installed as a wheel, that file is not shipped, but setuptools recorded the version into the
    package metadata AT BUILD TIME from the same VERSION file -- so read metadata first (the installed truth),
    then fall back to the VERSION file (running from a clone), then a sentinel. This means __version__ can never
    drift from setup.py the way a hardcoded literal did (it was stuck at 0.1.0 while setup.py said 0.2.0)."""
    try:
        from importlib.metadata import version as _v          # installed case: the wheel's recorded version
        return _v("leos-core")
    except Exception:
        pass
    try:
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "VERSION"), encoding="utf-8") as fh:   # clone case: read the source of truth
            return fh.read().strip()
    except Exception:
        return "0.0.0"


__version__ = _read_version()


def autoboot(partition=None, session=None, llm="auto", memory=True):
    """ONE CALL, BOTH ENDS, MEMORY IN (cp62): the standing boot ritual made standard so
    it never has to be asked for again. Finds the partition (arg, or $LECORE_PARTITION,
    or the conventional path), boots doctrine + external memory, attaches the model rung
    when one is reachable (llm="auto": ModelRung if importable and a model dir exists;
    pass a callable to override; llm=None for memory-end only), sets the archive root,
    and opens a session. Returns the mind, ready.

        import lecore
        m = lecore.autoboot()          # attached on both ends, memory loaded

    The POST line is available as m._autoboot_report."""
    import os
    # THE DEFAULT WAS ONE MACHINE'S ABSOLUTE PATH -- "/home/claude/claude_partition"
    # -- which exists on nobody else's disk, so every outside user fell through to
    # the shipped bundle without being told why. It still works (the fallthrough is
    # correct), but a default nobody can hit is a default that teaches nothing.
    # Order now: explicit argument, $LECORE_PARTITION, ./lecore_memory (the
    # conventional per-repo partition, and the one that actually has content),
    # then the shipped release_bundle/. The legacy absolute path stays LAST so an
    # existing setup that relies on it keeps working -- additive, not a flip.
    root = partition or os.environ.get("LECORE_PARTITION")
    if not root:
        for _cand in ("lecore_memory", "release_bundle",
                      "/home/claude/claude_partition"):
            if os.path.isdir(_cand):
                root = _cand
                break
        root = root or "lecore_memory"
    if not memory:
        root = "\0no-memory"          # nothing on disk matches; boots clean
    rung = None
    if llm == "auto":
        # cp78 polish: the auto arm now uses the ENGINE'S OWN RuntimeRung (ships
        # with the repo, automatic source attribution, opt-outs honored) instead
        # of a session-local /tmp tool that never shipped -- anyone else's
        # llm="auto" was silently getting no rung at all. Candidates:
        # $LECORE_MODEL first, then the conventional local paths.
        cands = [os.environ.get("LECORE_MODEL", "")] + \
            ["/tmp/mini_installed_full", "/tmp/mini_baked"]
        for cand in cands:
            if cand and os.path.isdir(cand):
                try:
                    from holographic.io_and_interop.holographic_runtimerung \
                        import RuntimeRung
                    rung = RuntimeRung(cand)
                    break
                except Exception:
                    rung = None
    elif callable(llm):
        rung = llm
    # THE SHIPPED-BUNDLE FALLBACK IS FOR AN ABSENT DEFAULT, NOT AN EXPLICIT ASK.
    # cp79 added it so `lecore.autoboot()` works out of the box on a fresh
    # machine, which is right -- but it fired unconditionally, so
    # `autoboot(partition="/my/new/memory")` SILENTLY MOUNTED release_bundle
    # instead. _autoboot_report then named release_bundle while the caller
    # believed they were on their own partition, and the first learning_save
    # went somewhere the next boot would not read.
    # An explicit partition= or $LECORE_PARTITION is a REQUEST: create it and
    # use it. Only the conventional-path search may fall back.
    _explicit = bool(partition or os.environ.get("LECORE_PARTITION"))
    if _explicit and memory and not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)
    if not _explicit and memory and not os.path.isdir(root):
        _shipped = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "release_bundle")
        if os.path.isdir(_shipped):
            root = _shipped
    m = UnifiedMind()
    # CREATE AN EXPLICITLY-REQUESTED PARTITION INSTEAD OF SILENTLY DROPPING IT.
    # `partition=root if isdir(root) else None` meant that asking for a NEW
    # directory -- the normal way to start your own memory -- fell through to
    # the shipped bundle, and _autoboot_report then said "release_bundle" while
    # the caller believed they were on their own partition. The first save would
    # go to the right place and every boot before it to the wrong one.
    # Only for an EXPLICIT partition= or $LECORE_PARTITION: the search-order
    # fallbacks must still fall back, because a missing ./lecore_memory is a
    # conventional absence, not a request.
    rep = m.boot(partition=root if os.path.isdir(root) else None,
                 doctrine=True, llm=rung or (lambda p: ""))
    m._archive_root = root
    if rung is not None:
        try:
            m.zoo_attach(rung)
            m._zoo_llm = rung
            if hasattr(rung, "mind"):
                rung.mind = m
        except Exception:
            pass
    if session:
        m.session_open(str(session))
    # REPORT THE POST TAKEN *AFTER* THE MOUNT WHEN THERE IS ONE. bios.boot runs
    # POST twice on purpose -- once before mounting a partition and once after,
    # as "post_after_mount", precisely because the spectral check needs state to
    # read. This surfaced the PRE-mount one, so booting a partition with 116
    # logged queries reported "virgin mind", identical to booting an empty one.
    # THE DESIGN WAS ALREADY RIGHT AND THE WRAPPER READ THE WRONG FIELD.
    m._autoboot_report = {"partition": root, "rung": type(rung).__name__
                          if rung is not None else None,
                          "post": rep.get("post_after_mount") or rep.get("post"),
                          "mounted": rep.get("mounted")}
    return m

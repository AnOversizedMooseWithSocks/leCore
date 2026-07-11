# lecore.py -- the friendly front door.
#
# The engine is a flat set of ~280 `holographic_*.py` modules. A newcomer shouldn't need to know which
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


__version__ = "0.1.0"

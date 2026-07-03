"""One render call that ADAPTS -- it looks at the scene and the workload and picks the methods itself, instead of the
caller choosing bake vs analytic, collapse vs trace, exact vs relaxed by hand.

The separate options built earlier (render_scene(bake=, relax=), render_dispatch, radiance_transfer) are the manual
override. This is the top of the composability stack: DERIVE the dispatch decisions from the scene and the usage, routed
by heuristics that are grounded in the break-evens already MEASURED, not guessed -- and hand back a plan that says WHY
each method was chosen, so the automation stays legible.

The decisions and their measured basis:
  * BAKE the SDF to a grid (O(1) sampling) only when it amortises: it LOSES on few-object single frames (measured 5
    objects: 0.3-0.7x) and WINS on complex scenes (6.4x at 64 primitives) or animation (1.78x at 24 objects x 8 frames).
    So bake when primitives >= ~16 OR frames >= ~4.
  * OVER-RELAX only helps grazing scenes and costs quality (~27 dB) -- measurement set its default OFF, so the adaptive
    path leaves the exact, bit-identical active-only marcher on unless the caller explicitly asks for relax.
  * COLLAPSE (PRT dot product) vs TRACE is decided per surface AND per workload: diffuse surfaces COLLAPSE only when
    RELIGHTING (PRT precompute amortises over many lights); for a single still frame diffuse shades directly (cheaper
    than precomputing transfer), which is what render_scene already does per material. Reflective surfaces always TRACE.

`plan_render` is the pure, testable decision layer; `render_adaptive` executes it and returns (frame, relight, plan).
Deterministic; NumPy only.
"""
import numpy as np

# measured thresholds (see the module docstring / NOTES) -- the break-evens, named so they are easy to audit and tune
_BAKE_MIN_PRIMS = 16
_BAKE_MIN_FRAMES = 4
_REFLECT_TRACE = 0.3                                             # reflectivity above which a surface must TRACE, not collapse
# (matte 0.0 / plastic 0.12 / ceramic 0.18 / default 0.2 collapse as diffuse; glossy 0.35 / metal 0.45 / mirror 0.85 trace)


def _reflect_of(obj):
    """Reflectivity of an object's material (0 = matte/diffuse) -- the signal that decides collapse vs trace."""
    from holographic_semantic import MATERIAL_RENDER
    mat = obj.get("material", "matte") if isinstance(obj, dict) else "matte"
    return float(MATERIAL_RENDER.get(mat, {}).get("reflect", 0.0))


def plan_render(objects, frames=1, relight=False, bake_res=96):
    """Decide -- purely from the scene and the workload -- which methods the adaptive pipeline should use. Returns a plan
    dict: bake (grid resolution or None), relax factor, path ('trace' single-frame renderer, or 'dispatch' relightable
    collapse/trace renderer), per-object methods (for the dispatch path), and a `reasons` map explaining each choice."""
    n = len(objects)
    reasons = {}
    # BAKE decision
    if n >= _BAKE_MIN_PRIMS or frames >= _BAKE_MIN_FRAMES:
        bake = bake_res
        reasons["bake"] = ("%d primitives / %d frames >= break-even (%d prims or %d frames): bake amortises"
                           % (n, frames, _BAKE_MIN_PRIMS, _BAKE_MIN_FRAMES))
    else:
        bake = None
        reasons["bake"] = ("%d primitives, %d frame(s): below break-even -- analytic SDF is cheaper than a bake" % (n, frames))
    # RELAX decision (measurement set the default off; adaptive never turns it on by itself)
    relax = 1.0
    reasons["relax"] = "exact active-only marcher (over-relaxation is a grazing-only, quality-costing manual opt-in)"
    # COLLAPSE vs TRACE decision
    if relight:
        methods = {}
        n_collapse = 0
        for i, o in enumerate(objects):
            if _reflect_of(o) > _REFLECT_TRACE:
                methods[i] = "trace"
            else:
                methods[i] = "collapse"; n_collapse += 1
        path = "dispatch"
        reasons["shade"] = ("relighting: %d diffuse surface(s) COLLAPSE (PRT, free relight), %d reflective TRACE"
                            % (n_collapse, n - n_collapse))
    else:
        methods = None
        path = "trace"
        reasons["shade"] = "single frame: direct material-dispatched shading (render_scene picks matte/mirror/glossy per hit)"
    return {"bake": bake, "relax": relax, "path": path, "methods": methods, "frames": frames, "reasons": reasons}


def render_adaptive(objects, camera, width=256, height=256, frames=1, relight=False, light=None,
                    sun="bright", sky="clear", post=None, bake_res=96, **render_kw):
    """ONE render call that adapts. Plans the methods from the scene + workload (`plan_render`), then executes:
      * single frame -> the full trace renderer `render_scene`, with the SDF auto-baked when the plan says it pays;
      * relighting  -> the collapse/trace dispatch renderer, with methods auto-derived from each surface's material,
                       returning a relight handle so further lights are ~free.
    Returns (frame, relight, plan). `relight` is a callable(new_light_env)->frame when relighting was planned, else None.
    `plan['reasons']` explains every choice, so the automation stays legible. The manual options (render_scene(bake=,
    relax=), render_dispatch, radiance_transfer) remain available for hand control."""
    plan = plan_render(objects, frames=frames, relight=relight, bake_res=bake_res)
    if plan["path"] == "dispatch":
        from holographic_semantic import realize_scene, _scene_setup
        from holographic_dispatch import render_dispatch
        rs = realize_scene(objects)
        ctx = _scene_setup(None, False, sky, sun, (0.75, 0.9, 0.85), rs=rs)   # object-only union (ids aligned to objects)
        union = ctx["union"]; colors = ctx["colors"]
        if light is None:
            light = lambda w: np.clip(w @ np.array([0.4, 0.7, 0.3]), 0, 1)[:, None] ** 4 * np.array([1.2, 1.1, 0.95]) + 0.06
        frame, relight_fn, info = render_dispatch(union, camera, width, height, plan["methods"], colors, light,
                                                  order=3, n=render_kw.get("prt_samples", 400))
        plan["dispatch_info"] = info
        return frame, relight_fn, plan
    # single-frame trace path -- render_scene already dispatches matte/mirror/glossy per hit; we add the auto-bake + relax
    from holographic_semantic import render_scene
    frame = render_scene(objects, camera, width=width, height=height, post=post, sun=sun, sky=sky,
                         bake=plan["bake"], relax=plan["relax"], **{k: v for k, v in render_kw.items() if k != "prt_samples"})
    return frame, None, plan


def _selftest():
    """The plan adapts to scene size and workload: bakes complex/animated scenes, not tiny stills; collapses diffuse
    surfaces only when relighting; always keeps the exact marcher."""
    small = [{"material": "matte"}] * 3
    big = [{"material": "matte"}] * 30
    mixed = [{"material": "matte"}, {"material": "mirror"}, {"material": "plastic"}]

    assert plan_render(small)["bake"] is None                    # 3 objects, one frame -> no bake
    assert plan_render(big)["bake"] is not None                  # 30 objects -> bake amortises
    assert plan_render(small, frames=8)["bake"] is not None      # animation -> bake amortises even for few objects
    assert plan_render(small)["relax"] == 1.0                    # exact marcher, always, unless asked

    single = plan_render(mixed)                                  # single frame -> trace path, no per-object methods
    assert single["path"] == "trace" and single["methods"] is None
    relit = plan_render(mixed, relight=True)                     # relight -> dispatch path, methods derived from material
    assert relit["path"] == "dispatch"
    assert relit["methods"][0] == "collapse" and relit["methods"][1] == "trace"   # matte collapses, mirror traces
    print("adaptive selftest ok: bakes complex/animated (not tiny stills), collapses diffuse only when relighting, "
          "keeps the exact marcher; reasons:", relit["reasons"]["shade"])


if __name__ == "__main__":
    _selftest()

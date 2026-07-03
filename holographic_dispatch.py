"""Composability of CALCULATION METHODS -- apply a different operator to different elements of one structure, chosen
per-element by a field, and switch method on the fly.

THE IDEA ("as above, so below", one more turn of the screw)
----------------------------------------------------------
holostuff already treats DATA as one substrate projected to whatever view is needed: the same field is a mesh, a fluid
body, a point cloud, a static collider -- and part of a surface can be one and part another, decided by a map (the
region field). This module carries that composability up one level, to the COMPUTATION: which *method* evaluates a value
should itself be a field. Trace to the first hit because tracing is the right tool for "find the first surface"; then at
the bounce, dispatch to whatever is best THERE -- collapse (a PRT dot product) on a diffuse patch, trace a reflection on
a mirror, a glossy bundle on a rough patch -- and if a traced reflection then lands on diffuse, switch to collapse for
the rest of the trip. The method adapts to what the ray meets, exactly as the material already does.

This is the same shape the substrate ALREADY uses at the whole-signal level (`denoise(method='auto')` picks
codebook/manifold/NLM by structure; `decompose_signal` picks a basis by topology). The upgrade here is PER-ELEMENT
selection by a field, with on-the-fly switching -- so a single computation is a composition of methods, not one method.

`dispatch_field` is the general primitive; `resolve_methods` turns a per-object method table plus an optional
per-region override into the per-hit tags the renderer dispatches on. Deterministic; NumPy only.
"""
import numpy as np


def dispatch_field(x, tags, ops, default=None):
    """Apply a DIFFERENT operator to different elements of `x`, selected per-element by `tags`. `x` is (N, ...);
    `tags` is (N,) method labels; `ops` is {label: fn(sub_x) -> sub_y}. Each label's elements are gathered, its op is
    applied to that whole group at once (staying vectorised), and the results are scattered back into place. This is the
    "part fluid, part static, by a field" idea applied to WHICH CALCULATION RUNS WHERE -- one structure, many methods.

    Returns an array whose rows are each element's result under its own method. All ops must return the same trailing
    shape and a compatible dtype (they are different ways of computing the SAME kind of value)."""
    x = np.asarray(x)
    tags = np.asarray(tags)
    if len(tags) != len(x):
        raise ValueError("tags and x must have the same length")
    out = None
    for label in _stable_unique(tags):                          # deterministic label order
        mask = tags == label
        op = ops.get(label, default)
        if op is None:
            raise KeyError("no operator provided for method %r" % (label,))
        res = np.asarray(op(x[mask]))
        if out is None:
            out = np.zeros((len(x),) + res.shape[1:], dtype=res.dtype)
        out[mask] = res
    if out is None:                                             # empty input
        out = np.zeros((0,))
    return out


def _stable_unique(tags):
    """Unique labels in first-appearance order -- deterministic dispatch order regardless of label type."""
    seen = []
    for t in tags:
        if t not in seen:
            seen.append(t)
    return seen


def resolve_methods(ids, method_table, points=None, region_field=None, default="trace"):
    """Build the per-hit method tags a renderer dispatches on. Base method comes from `method_table` (a dict/list mapping
    object id -> method label); an optional `region_field` overrides it per surface point (so PART of one surface can be
    a mirror and part diffuse, by a map -- method composability at sub-object resolution). Returns (N,) tags.

    `region_field` may expose `method_at(points, default)` returning per-point labels ('' or None = no override)."""
    ids = np.asarray(ids)
    if isinstance(method_table, dict):
        tags = np.array([method_table.get(int(i), default) for i in ids], dtype=object)
    else:
        mt = list(method_table)
        tags = np.array([mt[int(i)] if 0 <= int(i) < len(mt) else default for i in ids], dtype=object)
    if region_field is not None and points is not None and hasattr(region_field, "method_at"):
        over = region_field.method_at(points, default=None)
        for k in range(len(tags)):
            if over[k]:
                tags[k] = over[k]
    return tags


class BakedScene:
    """Everything a scene needs precomputed so that EVERY render -- including the FIRST -- is a relight (a dot product),
    not a cold first-time trace. Built once by `bake_scene`; consumed by `render_baked(scene, light)`. A builder on top
    of holostuff calls the bake at scene-load, then interactive relighting is free from frame one. Holds the primary
    visibility (hit mask, per-pixel method tags), the PRT transfer for every diffuse (collapse) hit, and the one-bounce
    reflection data for mirror hits (the bounce's own transfer, so a reflected diffuse surface also relights)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def bake_scene(sdf, camera, width, height, methods, colors, order=3, n=400, background=(0.55, 0.65, 0.82)):
    """PRECOMPUTE / BAKE a scene before any render. Traces primary visibility ONCE, dispatches each hit to its method,
    and precomputes the PRT transfer for every diffuse hit and for the diffuse surfaces behind each mirror bounce. The
    geometry work is done here, so `render_baked(scene, light)` afterwards is a pure dot-product relight -- the first
    frame is already a relight, not a first-time calculation. Returns a BakedScene. See render_baked / render_dispatch."""
    import numpy as np
    from holographic_raymarch import sphere_trace, sdf_normal
    from holographic_prt import precompute_transfer

    eye, dirs = camera.ray_dirs(width, height)
    O = np.broadcast_to(eye, (width * height, 3)).astype(float); D = dirs.reshape(-1, 3)
    hit, t, P = sphere_trace(sdf, O, D)
    Ph = P[hit]; ids = np.asarray(sdf.ids(Ph)); N = sdf_normal(sdf, Ph)
    tags = resolve_methods(ids, methods, points=Ph)
    colors = np.asarray(colors, float)

    collapse = tags == "collapse"; trace = ~collapse
    T_coll = precompute_transfer(sdf, Ph[collapse], N[collapse], order=order, n=n) if collapse.any() else np.zeros((0, order * order))
    alb_coll = colors[ids[collapse]] if collapse.any() else np.zeros((0, 3))

    refl_dir = D[hit][trace] - 2.0 * (D[hit][trace] * N[trace]).sum(1)[:, None] * N[trace]
    if trace.any():
        rhit, rt, rP = sphere_trace(sdf, Ph[trace] + N[trace] * 3e-3, refl_dir)
        rPh = rP[rhit]; rN = sdf_normal(sdf, rPh)
        T_refl = precompute_transfer(sdf, rPh, rN, order=order, n=n) if rhit.any() else np.zeros((0, order * order))
        alb_refl = colors[np.asarray(sdf.ids(rPh))] if rhit.any() else np.zeros((0, 3))
        switched = int(rhit.sum())
    else:
        rhit = np.zeros(0, bool); T_refl = np.zeros((0, order * order)); alb_refl = np.zeros((0, 3)); switched = 0

    return BakedScene(width=width, height=height, order=order, background=np.asarray(background, float),
                      hit=hit, D=D, tags=tags, Nhit=int(hit.sum()),
                      pos_coll=np.cumsum(collapse) - 1, pos_trace=np.cumsum(trace) - 1,
                      T_coll=T_coll, alb_coll=alb_coll, refl_dir=refl_dir, rhit=rhit, T_refl=T_refl, alb_refl=alb_refl,
                      info={"collapse": int(collapse.sum()), "trace": int(trace.sum()), "switched_to_collapse": switched})


def render_baked(scene, light):
    """Relight a BakedScene: shade every pixel from its precomputed transfer (a dot product) -- NO tracing. This is what
    runs per frame in an interactive relight loop, and it is the same work the first frame does, because the geometry
    was already baked. `light` is an environment function dirs(M,3)->rgb(M,3). Returns a (H,W,3) frame."""
    import numpy as np
    from holographic_prt import project_env_to_sh, shade_prt
    s = scene
    L = project_env_to_sh(light, order=s.order, n=1200)

    def op_collapse(idx):
        rows = s.pos_coll[idx]
        return shade_prt(s.T_coll[rows], L, albedo=s.alb_coll[rows])

    def op_trace(idx):
        rows = s.pos_trace[idx]
        out = np.zeros((len(idx), 3))
        rmask = s.rhit[rows]
        rrows = np.cumsum(s.rhit) - 1
        if rmask.any():
            rr = rrows[rows[rmask]]
            out[rmask] = shade_prt(s.T_refl[rr], L, albedo=s.alb_refl[rr])
        if (~rmask).any():
            out[~rmask] = np.clip(light(s.refl_dir[rows[~rmask]]), 0, 1)
        return out

    radiance = dispatch_field(np.arange(s.Nhit), s.tags, {"collapse": op_collapse, "trace": op_trace})
    frame = np.zeros((s.width * s.height, 3))
    frame[s.hit] = radiance
    frame[~s.hit] = np.clip(light(s.D[~s.hit]), 0, 1) * 0.5 + s.background * 0.5
    return np.clip(frame.reshape(s.height, s.width, 3), 0, 1)


def render_dispatch(sdf, camera, width, height, methods, colors, light,
                    order=3, n=400, sun=(0.4, 0.7, 0.3), background=(0.55, 0.65, 0.82)):
    """Render a scene by DISPATCHING each hit to the method best for its surface, and return a RELIGHT handle -- the real
    pipeline form of "collapse on diffuse, trace on a mirror, switch on the fly". Convenience wrapper over the explicit
    two-step form: it BAKES the scene (bake_scene) then RELIGHTS it (render_baked), and hands back a relight closure so
    further lights are free. If you want the first frame to already be a relight, call bake_scene once yourself and then
    render_baked per light. Returns (frame, relight, info)."""
    baked = bake_scene(sdf, camera, width, height, methods, colors, order=order, n=n, background=background)
    return render_baked(baked, light), (lambda L: render_baked(baked, L)), baked.info


def _selftest():
    """Dispatch applies each method to its own elements and reproduces per-group application; an on-the-fly two-stage
    dispatch (a 'trace' method that itself dispatches its result) composes correctly."""
    x = np.arange(10.0)
    tags = np.array(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])
    ops = {"A": lambda v: v * 10.0, "B": lambda v: v + 100.0}
    out = dispatch_field(x, tags, ops)
    # A-elements got *10, B-elements got +100 -- each computed by its own method
    assert np.allclose(out[tags == "A"], x[tags == "A"] * 10.0)
    assert np.allclose(out[tags == "B"], x[tags == "B"] + 100.0)

    # on-the-fly switch: a 'mirror' method that, for its elements, dispatches AGAIN to a second method
    tags2 = np.array(["mirror", "collapse", "mirror", "collapse"])
    def mirror_then_collapse(sub):
        # a mirror "reflects" (negate) then the rest of the trip is a collapse (halve) -- method switch mid-op
        reflected = -sub
        return dispatch_field(reflected, np.array(["collapse"] * len(reflected)), {"collapse": lambda v: v * 0.5})
    ops2 = {"mirror": mirror_then_collapse, "collapse": lambda v: v * 0.5}
    y = dispatch_field(np.array([2.0, 4.0, 6.0, 8.0]), tags2, ops2)
    assert np.allclose(y, [-1.0, 2.0, -3.0, 4.0])               # mirrors: -x*0.5 ; collapses: x*0.5

    # method resolution from an object table
    ids = np.array([0, 1, 2, 1, 0])
    tags3 = resolve_methods(ids, {0: "collapse", 1: "mirror", 2: "glossy"})
    assert list(tags3) == ["collapse", "mirror", "glossy", "mirror", "collapse"]
    print("dispatch selftest ok: per-element methods compose, on-the-fly switch works, methods resolve from a table")


if __name__ == "__main__":
    _selftest()

"""Coherent secondary rays (RAY COHERENCE). Two ideas, both Moose's:

1. A secondary (bounce) ray is not a new thing to trace from scratch -- it is a TRANSFORM of its parent ray with the
   bounce counter incremented: origin moves to the hit point, direction reflects about the surface normal, depth += 1.
   `reflect_transform` is that transform, written once; N bounces are N applications of it.

2. Neighbouring reflection rays off a SMOOTH surface are COHERENT -- the reflected image varies smoothly across the
   reflector, because the normal (and so the reflected direction) varies smoothly. So we do not have to trace every
   reflection ray. Trace a SPARSE set (a stride grid over the reflective pixels) and reconstruct the perpendicular
   neighbours by a Gaussian/bilinear interpolation, GATED by surface continuity (same object, aligned normal) so we
   never blend across a reflection edge, and FALL BACK to an exact trace only where the reconstruction is uncertain
   (few coherent samples, or the samples disagree -- a silhouette in the reflection). This is the irradiance-cache /
   reflection-interpolation move every production renderer makes, here on the engine's own ray G-buffer.

WHY IT MATTERS. Secondary rays were the flagged weakness: a move (or a camera turn) re-shades a large fraction of a
mirror/glass surface because the reflected solid angle is wide. Reconstructing them from a sparse trace turns "trace
every reflection ray" into "trace a few and interpolate" -- fewer rays for the same smooth reflection, with the cost
concentrated exactly on the reflection edges where it belongs.

HONEST SCOPE. This reconstructs the REFLECTION component (one bounce) from screen-space coherence. It is exact on the
traced + fallback pixels and interpolated (approximate) on the smooth interior -- the same class of approximation as
the frame reprojection. It shines on smooth/curved reflectors and degrades on high-frequency (bumpy) ones, where the
variance fallback correctly traces more. Deterministic, NumPy/stdlib only.
"""
import numpy as np


def reflect_transform(O, D, P_hit, N, bounce=None, eps=3e-3):
    """The secondary ray as a transform of the parent: origin -> hit point (nudged off the surface), direction ->
    reflected about the normal, bounce counter -> +1. Returns (O2, D2[, bounce2]) -- the same rays, transformed."""
    D2 = D - 2.0 * (D * N).sum(1)[:, None] * N                  # mirror reflection: D - 2 (D.N) N
    O2 = P_hit + N * eps                                        # start just off the surface to avoid self-hit
    if bounce is None:
        return O2, D2
    return O2, D2, np.asarray(bounce) + 1                       # the ONLY new information a bounce adds: depth += 1


def trace_reflection_color(ctx, O2, D2):
    """Trace + shade one bounce of reflected rays: where they hit an object, Lambert + AO + soft shadow; else the sky
    dome. Returns rgb per ray. (This is the per-ray cost we want to pay for only a sparse subset.)"""
    from holographic_raymarch import sphere_trace, sdf_normal, sky_dome
    from holographic_shadowhome import Shadow      # visibility via the Shadow home (R8)
    union = ctx["union"]; colors = ctx["colors"]; sun_dir = ctx["sun_dir"]; amb = ctx["amb"]; sun_i = ctx["sun_i"]
    rc = sky_dome(D2, sun_dir=tuple(sun_dir))
    hm, tm, Pmh = sphere_trace(union, O2, D2)
    if np.any(hm):
        Nm = sdf_normal(union, Pmh[hm]); idm = union.ids(Pmh[hm])
        lamm = np.clip((Nm * sun_dir).sum(1), 0, 1)
        shm = Shadow.soft(union, Pmh[hm] + Nm * 3e-3, sun_dir); aom = Shadow.ambient_occlusion(union, Pmh[hm], Nm)
        rc[hm] = np.clip(colors[idm] * (amb * aom + lamm * shm * sun_i)[:, None], 0, 1)
    return rc


def coherent_reflection(ctx, P, N, D, ids, mirror, W, H, stride=3, cont_tol=0.9, var_tol=0.02):
    """Reconstruct the reflection over the reflective pixels from a SPARSE trace + gated bilinear interpolation, with an
    exact-trace fallback where the reconstruction is uncertain.

    P, N, D, ids: per-pixel primary-hit world point, normal, incoming dir, object id (each shape (H*W, ...), only the
    `mirror` pixels are used). `mirror`: bool mask (H*W,) of reflective pixels. Returns:
      reflected (H*W, 3)  -- reflection colour (0 where not mirror),
      n_traced            -- reflection rays actually traced (samples + fallback),
      n_mirror            -- reflective pixels total (the count a per-pixel trace would pay).
    """
    reflected = np.zeros((H * W, 3))
    midx = np.where(mirror)[0]
    n_mirror = len(midx)
    if n_mirror == 0:
        return reflected, 0, 0

    # --- sparse sample set: reflective pixels on a stride grid; trace each one's reflection ray exactly ---
    ys, xs = midx // W, midx % W
    on_grid = (ys % stride == 0) & (xs % stride == 0)
    sidx = midx[on_grid]
    O2, D2 = reflect_transform(None, D[sidx], P[sidx], N[sidx])
    scol = trace_reflection_color(ctx, O2, D2)                  # the sparse traced reflection samples

    # scatter samples into full-frame lookup buffers (colour / valid / id / normal) for corner gather
    samp_col = np.zeros((H * W, 3)); samp_ok = np.zeros(H * W, bool)
    samp_col[sidx] = scol; samp_ok[sidx] = True

    # --- reconstruct every reflective pixel from its 4 surrounding grid corners (bilinear), gated by continuity ---
    y = ys.astype(float); x = xs.astype(float)
    y0 = (ys // stride) * stride; x0 = (xs // stride) * stride
    y1 = np.minimum(y0 + stride, H - 1); x1 = np.minimum(x0 + stride, W - 1)
    fy = (y - y0) / max(stride, 1); fx = (x - x0) / max(stride, 1)
    corners = [(y0, x0, (1 - fy) * (1 - fx)), (y0, x1, (1 - fy) * fx),
               (y1, x0, fy * (1 - fx)), (y1, x1, fy * fx)]
    acc = np.zeros((n_mirror, 3)); wsum = np.zeros(n_mirror)
    csum = np.zeros((n_mirror, 3)); c2sum = np.zeros((n_mirror, 3)); nvalid = np.zeros(n_mirror)
    my, mx = ids[midx], N[midx]
    for cy, cx, w in corners:
        ci = (cy * W + cx).astype(int)
        ok = samp_ok[ci]                                        # the corner was actually traced (is a reflective pixel)
        same_obj = ids[ci] == my                                # same object -> same reflector, safe to blend
        nalign = (N[ci] * mx).sum(1) > cont_tol                 # aligned normal -> reflection direction is coherent
        good = ok & same_obj & nalign
        w = np.where(good, w, 0.0)
        cc = samp_col[ci]
        acc += w[:, None] * cc; wsum += w
        csum += good[:, None] * cc; c2sum += good[:, None] * cc * cc; nvalid += good
    have = wsum > 1e-6
    reflected[midx[have]] = acc[have] / wsum[have, None]
    reflected[sidx] = scol                                      # traced sample pixels keep their EXACT colour

    # variance among the valid corners (a reflection EDGE makes them disagree) -> low confidence
    mean = np.zeros((n_mirror, 3)); mean[nvalid > 0] = csum[nvalid > 0] / nvalid[nvalid > 0, None]
    var = np.zeros(n_mirror)
    vv = nvalid > 0
    var[vv] = np.maximum(c2sum[vv] / nvalid[vv, None] - mean[vv] ** 2, 0.0).sum(1)

    # --- fallback: trace exactly where reconstruction is uncertain -- but NOT the sample pixels (already exact) ---
    fallback = ((~have) | (var > var_tol)) & (~on_grid)
    if np.any(fallback):
        fidx = midx[fallback]
        O2f, D2f = reflect_transform(None, D[fidx], P[fidx], N[fidx])
        reflected[fidx] = trace_reflection_color(ctx, O2f, D2f)

    n_traced = len(sidx) + int(fallback.sum())
    return reflected, n_traced, n_mirror


def _selftest():
    """The reflection transform is its own inverse direction-wise; coherent reconstruction on a smooth mirror sphere
    traces far fewer rays than per-pixel and stays close to the exact reflection."""
    from holographic_semantic import _scene_setup, parse_description, realize_scene
    from holographic_render import Camera
    from holographic_raymarch import sphere_trace, sdf_normal
    # reflect_transform sanity: reflecting a downward ray off an up-normal flips it upward
    O2, D2 = reflect_transform(None, np.array([[0, -1.0, 0]]), np.array([[0, 0, 0.0]]), np.array([[0, 1.0, 0]]))
    assert D2[0, 1] > 0.9

    objs = parse_description("a huge mirror ball")["objects"]
    rs = realize_scene(objs); ctx = _scene_setup(None, True, "clear", "bright", (0.75, 0.9, 0.85), rs=rs)
    cam = Camera(eye=(0, 0.4, 3.4), target=(0, 0.1, 0), fov_deg=48); W = H = 110
    eye, dirs = cam.ray_dirs(W, H); O = np.broadcast_to(eye, (W * H, 3)).astype(float); Dd = dirs.reshape(-1, 3)
    union = ctx["union"]
    hit, t, Pp = sphere_trace(union, O, Dd)
    P = np.zeros((W * H, 3)); Nn = np.zeros((W * H, 3)); ids = -np.ones(W * H, int)
    P[hit] = Pp[hit]; Nn[hit] = sdf_normal(union, Pp[hit]); ids[hit] = union.ids(Pp[hit])
    refl = ctx["refl"]; mirror = np.zeros(W * H, bool)
    mirror[hit] = refl[ids[hit]] > 0.05
    full = np.zeros((W * H, 3))
    O2, D2 = reflect_transform(None, Dd[mirror], P[mirror], Nn[mirror])
    full[mirror] = trace_reflection_color(ctx, O2, D2)
    approx, n_traced, n_mirror = coherent_reflection(ctx, P, Nn, Dd, ids, mirror, W, H, stride=4, var_tol=0.03)
    mse = float(np.mean((full[mirror] - approx[mirror]) ** 2))
    assert n_traced < 0.6 * n_mirror                           # traced far fewer rays than per-pixel
    assert mse < 5e-3                                           # and stayed close to the exact reflection
    print("raycoherence selftest ok: %d/%d reflection rays traced (%.0f%%), reflection MSE %.2e"
          % (n_traced, n_mirror, 100 * n_traced / n_mirror, mse))


if __name__ == "__main__":
    _selftest()

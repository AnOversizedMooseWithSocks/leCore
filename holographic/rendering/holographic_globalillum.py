"""Global illumination and caustics (LIGHT-2): the two expensive light-transport terms, built on the engine's
GENUINE contributions rather than a pretend "VSA renderer."

THE HONEST CONNECTIONS
----------------------
* GLOBAL ILLUMINATION is an irradiance integral over the hemisphere at every surface point -- ruinously
  expensive per pixel. Ward's classic acceleration is an IRRADIANCE CACHE: compute the slow integral at a SPARSE
  set of points and interpolate between them, because indirect light varies smoothly. holostuff already owns that
  idea (holographic_adaptive_cache.adaptive_anchors / holographic_cache -- "place samples where the field bends,
  interpolate the rest"). So GI here is: gather one-bounce indirect at a sparse cache of surface points, then
  inverse-distance interpolate -- the engine's sparse-cache contribution, measured against the dense ground truth.
* CAUSTICS are the light a refractive/reflective object focuses onto a receiver. The honest method is FORWARD
  light tracing: shoot rays from the light, bend them through the object, and SPLAT where they land. That splat --
  accumulating many rays into a grid with np.add.at -- IS the engine's scatter = bundle (the adjoint of sampling).
  Where rays converge, the bundle piles up: the caustic.

Neither is "hypervector magic." The contribution is real and named: sparse caching + interpolation for GI, a
scatter/bundle accumulation for caustics. Both vectorised; both measured with their negatives.
"""

import numpy as np
from holographic.rendering.holographic_raymarch import sphere_trace, sdf_normal, refract_dir


def _cosine_hemisphere(N, n, seed=0):
    """n cosine-weighted sample directions around each unit normal in N:(M,3). Returns (M, n, 3). Vectorised.
    Delegates to the Sampling home (consolidation R4) -- one shared implementation, bit-identical."""
    from holographic.sampling_and_signal.holographic_samplinghome import Sampling
    return Sampling.cosine_hemisphere(N, n, seed=seed)


def gather_indirect(sdf, P, N, light_dir, base_color=(0.8, 0.6, 0.5), n_dirs=16, seed=0):
    """One-bounce indirect irradiance at points P:(M,3): sample `n_dirs` cosine-weighted directions over the
    hemisphere, sphere-trace each, and at the secondary hits gather the DIRECT light (the surface re-radiates the
    sun it receives). Average over directions. Returns (M,3) indirect colour. Vectorised over ALL M*n_dirs rays
    at once."""
    P = np.asarray(P, float); L = np.asarray(light_dir, float); L = L / (np.linalg.norm(L) + 1e-12)
    base = np.asarray(base_color, float)
    M = len(P)
    dirs = _cosine_hemisphere(N, n_dirs, seed)                # (M, n_dirs, 3)
    O = np.repeat(P + N * 3e-3, n_dirs, axis=0)               # offset off the surface to avoid self-hit
    D = dirs.reshape(-1, 3)
    hit, t, Q = sphere_trace(sdf, O, D, max_steps=48, max_dist=8.0)
    irr = np.zeros((len(D), 3))
    if hit.any():
        Nq = sdf_normal(sdf, Q[hit])
        from holographic.rendering.holographic_brdf import lambert                  # the Shading home's diffuse term  consolidation R3
        irr[hit] = lambert(Nq, L, base)                       # = clip(Nq.L,0)*base, the bounce re-radiating direct light
    return irr.reshape(M, n_dirs, 3).mean(axis=1)            # average -> indirect irradiance


def irradiance_cache(sdf, P, N, light_dir, base_color=(0.8, 0.6, 0.5), n_cache=64, n_dirs=16, seed=0):
    """Build a sparse irradiance cache: subsample `n_cache` of the surface points, compute one-bounce indirect
    there (the slow part, paid only n_cache times), and return (cache_positions, cache_irradiance). Read it with
    `read_cache`. This is Ward's irradiance caching = the engine's adaptive-anchor sparse-cache idea, applied to
    indirect light."""
    P = np.asarray(P, float)
    idx = np.linspace(0, len(P) - 1, min(n_cache, len(P))).astype(int)
    cP = P[idx]; cN = N[idx]
    cIrr = gather_indirect(sdf, cP, cN, light_dir, base_color, n_dirs, seed)
    return cP, cIrr


def read_cache(cache, query_P, k=4, power=2.0):
    """Read the irradiance cache at query points by inverse-distance interpolation of the k nearest cache points
    (the cache read: indirect light is smooth, so a few nearby samples reconstruct it). Vectorised."""
    cP, cIrr = cache
    Q = np.asarray(query_P, float)
    d2 = ((Q[:, None, :] - cP[None, :, :]) ** 2).sum(axis=2) + 1e-9    # (Q, n_cache)
    nn = np.argsort(d2, axis=1)[:, :k]                       # k nearest cache points
    w = 1.0 / d2[np.arange(len(Q))[:, None], nn] ** (power / 2)
    w /= w.sum(axis=1, keepdims=True)
    return np.einsum("qk,qkc->qc", w, cIrr[nn])             # weighted blend of the nearest cached irradiances


def caustic_hits(sdf, light_dir=(0, -1, 0), receiver_y=-0.9, extent=2.0, ior=1.5, n_side=200,
                 refracted_only=False, emitter_center=None, aim=None, lam=None, occluder=None, through=False,
                 max_internal=4):
    """The NONLINEAR half of a caustic, on its own: emit a parallel grid down `light_dir`, sphere-trace to the
    object, refract at the hit, continue to the receiver plane y=`receiver_y`. Returns (land (N,3), valid (N,),
    hit (N,)) -- the landing points and which rays count. NO accumulation happens here.

    `occluder` (default None): an OPAQUE SDF that blocks light -- rays it stops before the glass, and refracted rays
    it stops before the receiver, are marked invalid. WHY: a geode's crystals sit in a rock bowl; without this the
    lining threw a caustic onto the floor through the rind, as a bright ring under an object that should cast a
    shadow. Rock is not a thin lens.

    `through=True` (default off, byte-identical otherwise): the FULL transit -- after the entry refraction the ray
    is marched through the interior (_march_through) to its EXIT face and refracted OUT (TIR reflects and continues,
    up to `max_internal` bounces), and the interior PATH LENGTH per ray is returned as a 4th value (plen (N,)).
    WHY: the historical single-refraction "thin lens" bends light once and drops it on the floor with no colour --
    a prism's exit refraction is what sets where the caustic lands, and the path length is what lets the caller
    tint it by Beer-Lambert: light through an amethyst is purple on the floor (Moose: "light passed through a
    coloured gemstone would have a colour tint"). Rays still trapped after the cap are marked invalid.

    WHY IT IS SEPARATE. A caustic is two operations of different character glued together: the refraction, which
    is Snell's law and does not superpose, and the ACCUMULATION of landings into an image, which is a pure sum --
    a bundle. `caustics()` does the sum as a histogram (np.add.at into a res*res grid). holographic_holocaustic
    does the same sum as an FPE bundle: one hypervector, readable at any resolution, with wavelength as an axis.
    Both need exactly these landings, so the landings are computed once, here. Refactored out of caustics() with
    its output byte-identical.

    `ior` may be a scalar or PER-RAY (n_side*n_side,) -- with `lam` (per-ray wavelengths, same shape) that is how
    a single ray set carries a continuous spectrum: each ray is refracted by the index of ITS OWN wavelength, in one
    pass, instead of re-tracing once per wavelength band. `lam` is passed through untouched; it is the caller's
    bookkeeping for which ray was which colour."""
    L = np.asarray(light_dir, float); L = L / (np.linalg.norm(L) + 1e-12)
    g = np.linspace(-extent, extent, n_side)
    GX, GZ = np.meshgrid(g, g)
    ox, oz = (0.0, 0.0) if emitter_center is None else (float(emitter_center[0]), float(emitter_center[1]))
    if aim is not None:
        # Back-project the launch plane along the light so the grid lands ON the target rather than
        # wherever a straight-down assumption would have put it. See the worked example in caustics().
        a = np.asarray(aim, float)
        t = (3.0 - float(a[1])) / max(abs(float(L[1])), 1e-6)
        ox, oz = float(a[0]) - float(L[0]) * t, float(a[2]) - float(L[2]) * t
    O = np.stack([GX.ravel() + ox, np.full(GX.size, 3.0), GZ.ravel() + oz], axis=1)
    D = np.broadcast_to(L, O.shape).copy()
    hit, t, P = sphere_trace(sdf, O, D, max_steps=80, max_dist=10.0)
    blocked = np.zeros(len(O), bool)
    if occluder is not None:
        ho, to, _ = sphere_trace(occluder, O, D, max_steps=80, max_dist=10.0)
        blocked = ho & (~hit | (to < t))                       # the rock is in front of the glass (or there is no glass)
    out = D.copy()
    if hit.any():                                            # refract the rays that struck the object
        Nh = sdf_normal(sdf, P[hit])
        eta = np.asarray(ior, float)
        eta_h = eta[hit] if eta.ndim else eta                # per-ray index rides with its ray
        out[hit] = refract_dir(D[hit], Nh, eta_h)
    start = np.where(hit[:, None], P, O)                     # continue from the hit (or the origin if it missed)
    plen = np.zeros(len(O))
    if through and hit.any():
        from holographic.rendering.holographic_pathtrace import _march_through
        eta = np.asarray(ior, float)
        h = np.flatnonzero(hit)
        cur_P = P[h] + out[h] * 3e-3; cur_D = out[h].copy()
        eta_h = (eta[h] if eta.ndim else np.full(len(h), float(eta)))
        done = np.zeros(len(h), bool); out_pos = np.zeros((len(h), 3)); out_dir = np.zeros((len(h), 3))
        for _ in range(int(max_internal) + 1):
            live = ~done
            if not live.any():
                break
            exitP = _march_through(sdf, cur_P[live], cur_D[live], require_inside=True)
            Nx = sdf_normal(sdf, exitP)
            r_out = refract_dir(cur_D[live], Nx, eta_h[live])
            leaving = np.sum(r_out * Nx, axis=1) > 0.0                  # transmitted (else TIR: reflect inside)
            plen[h[live]] += np.linalg.norm(exitP - cur_P[live], axis=1)
            idx = np.flatnonzero(live)
            out_pos[idx[leaving]] = exitP[leaving]; out_dir[idx[leaving]] = r_out[leaving]; done[idx[leaving]] = True
            cur_P[idx[~leaving]] = exitP[~leaving] + r_out[~leaving] * 3e-3; cur_D[idx[~leaving]] = r_out[~leaving]
        start[h[done]] = out_pos[done]; out[h[done]] = out_dir[done]
        trapped = h[~done]                                              # still inside after the cap: honest loss
    dy = out[:, 1]
    tplane = (receiver_y - start[:, 1]) / np.where(np.abs(dy) < 1e-6, -1e-6, dy)   # reach the receiver plane
    land = start + out * tplane[:, None]
    valid = (tplane > 0) & ~blocked
    if through and hit.any() and len(trapped):
        valid[trapped] = False
    if occluder is not None and valid.any():
        # after the refraction: does the rock stand between the exit and the floor?
        v = np.flatnonzero(valid)
        ho, to, _ = sphere_trace(occluder, start[v] + out[v] * 2e-3, out[v], max_steps=80, max_dist=10.0)
        valid[v[ho & (to < tplane[v])]] = False
    if refracted_only:
        valid = valid & hit                    # drop the aperture's own shadow; see caustics()
    if through:
        return land, valid, hit, plen
    return land, valid, hit


def caustics(sdf, light_dir=(0, -1, 0), receiver_y=-0.9, extent=2.0, res=128, ior=1.5, n_side=200, seed=0,
             center=(0.0, 0.0), window=None, refracted_only=False, emitter_center=None, aim=None):
    """Forward-traced caustics: shoot a grid of parallel light rays down the `light_dir`, refract those that hit
    the object, continue to the receiver plane at y=`receiver_y`, and SPLAT each landing point into a res*res
    grid with np.add.at -- the scatter that is the engine's bundle. Where refracted rays converge the bundle
    piles up: the bright caustic. Returns the (res,res) intensity map (normalised). Vectorised.

    `aim=(x, y, z)` POINTS THE EMITTER AT A TARGET, and with an oblique light it is the difference
    between a caustic and an empty image. The ray grid is a square launched from y=3.0, and it was
    always centred on the ORIGIN -- fine for a light travelling straight down, and silently wrong for
    any other. Worked example, measured: with light travelling (0.551, -0.752, -0.361) toward an object
    at y=0.46, a ray needs t=(3.0-0.46)/0.752=3.38 to fall to the object's height, and in that distance
    drifts x by +1.86. So a grid spanning x in [-0.66, 0.66] arrives at x in [1.2, 2.5] -- entirely past
    an object spanning [-0.61, 0.61]. NOT ONE RAY HITS IT, refracted_only returns an empty image, and
    nothing in the output says why. `aim` back-projects the grid along the light so it lands on the
    target; `emitter_center=(x, z)` sets the offset by hand. Defaults reproduce the original exactly.

    `refracted_only=True` splats ONLY rays that actually struck the object. Default False keeps every
    ray, including the ones that sailed past -- which is right for a standalone caustic image, because
    those rays are the surrounding lit floor the caustic sits in.

    IT IS WRONG FOR A COMPOSITE, though, and the failure is loud once seen. The emitter is a SQUARE
    grid of parallel rays; the ones that miss the object carry straight on and splat as a hard-edged
    bright QUADRILATERAL -- the aperture's own shadow, not a caustic. Composited onto a render that
    already has direct lighting it both double-counts that light and paints a rectangle across the
    floor, which reads unmistakably as a broken render. Baseline subtraction does not remove it,
    because it is a shape rather than an offset. For holographic_causticpass, pass refracted_only=True.

    `center=(x, z)` and `window` FRAME THE RECEIVER independently of the light. Defaults (0,0) and None
    reproduce the original exactly.

    WHY THEY EXIST -- `extent` was doing two unrelated jobs. It sized the emitter grid AND the receiver
    window, both centred on the origin, so a caustic thrown off-centre by an OBLIQUE light could only be
    brought into view by widening the window -- which widened the light grid to match, spreading the same
    rays over a larger area and washing the caustic out. Chromatic separation is a fixed distance in world
    units (measured: 0.049 world units between 400nm and 700nm through SF11 at receiver_y=-2.5), so it is
    only VISIBLE at high pixels-per-unit; zooming was exactly what could not be done. `window` sets the
    receiver half-width alone, `center` sets where it looks. An axial light needs neither."""
    land, valid, hit = caustic_hits(sdf, light_dir, receiver_y, extent, ior, n_side,
                                    refracted_only=refracted_only, emitter_center=emitter_center, aim=aim)
    img = np.zeros((res, res))
    # The receiver window is framed separately from the emitter grid: half-width `window` (default the
    # emitter's own extent, which is the original behaviour) about `center`.
    half = float(extent) if window is None else float(window)
    cx, cz = float(center[0]), float(center[1])
    xi = ((land[:, 0] - cx + half) / (2 * half) * (res - 1)).astype(int)
    zi = ((land[:, 2] - cz + half) / (2 * half) * (res - 1)).astype(int)
    inb = valid & (xi >= 0) & (xi < res) & (zi >= 0) & (zi < res)
    np.add.at(img, (zi[inb], xi[inb]), 1.0)                  # the splat = the bundle (accumulate landings)
    return img / (img.mean() + 1e-9)                         # normalise to mean 1 (so peaks read as focusing)


def _selftest():
    from holographic.mesh_and_geometry.holographic_sdf import sphere, plane
    scene = sphere(0.7).union(plane(-0.85))
    # GI: a sparse cache reconstructs the dense indirect within a tolerance, at a fraction of the rays
    cam_pts = np.array([[x, -0.85, z] for x in np.linspace(-1, 1, 12) for z in np.linspace(-1, 1, 12)])
    N = np.broadcast_to(np.array([0., 1, 0]), cam_pts.shape).copy()
    dense = gather_indirect(scene, cam_pts, N, (-0.4, 0.7, -0.3), n_dirs=12, seed=1)
    cache = irradiance_cache(scene, cam_pts, N, (-0.4, 0.7, -0.3), n_cache=24, n_dirs=12, seed=1)
    approx = read_cache(cache, cam_pts)
    err = np.abs(approx - dense).mean()
    assert err < 0.15, err                                   # sparse cache ~ dense GI
    # caustics: a refractive sphere focuses light -> the splat map has a peak well above uniform
    c = caustics(scene, ior=1.5, n_side=160)
    assert c.max() > 3.0                                     # focusing concentrates the bundle
    # The framing arguments must be EXACTLY inert at their defaults -- this is the backward-compatibility
    # contract, and "close enough" would not be additive.
    assert np.array_equal(c, caustics(scene, ior=1.5, n_side=160, center=(0.0, 0.0), window=None))
    # window=None must mean "the emitter's own extent" EXACTLY, not approximately.
    assert np.array_equal(c, caustics(scene, ior=1.5, n_side=160, window=2.0))
    # And a tighter window must actually RESOLVE MORE, which is peak-to-mean rising rather than more lit
    # pixels -- the first version of this assertion demanded a larger lit fraction and failed, because
    # splatting is point-based: a tighter window admits FEWER rays even as it spreads them over more
    # pixels. The lit fraction goes DOWN (0.913 -> 0.256); the detail goes UP (peak 138 -> 215).
    wide = caustics(scene, ior=1.5, n_side=160, window=2.0)
    tight = caustics(scene, ior=1.5, n_side=160, window=1.0)
    assert tight.max() > wide.max(), (wide.max(), tight.max())
    assert (tight > 0).mean() < (wide > 0).mean()
    # aim= must be inert by default and must actually move the emitter. Its absence made an oblique
    # light render an EMPTY caustic with nothing to say why -- the bug this argument exists for.
    assert np.array_equal(c, caustics(scene, ior=1.5, n_side=160, emitter_center=None, aim=None))
    oblique = (0.551, -0.752, -0.361)
    blind = caustics(scene, oblique, -0.85, 0.8, 96, ior=1.5, n_side=200, window=6.0, refracted_only=True)
    aimed = caustics(scene, oblique, -0.85, 0.8, 96, ior=1.5, n_side=200, window=6.0, refracted_only=True,
                     aim=(0.0, 0.0, 0.0))
    assert aimed.max() > blind.max(), "aim= did not bring the object into the beam (%.3f vs %.3f)" % (blind.max(), aimed.max())
    # center= must move the picture, or the framing argument is decorative.
    assert not np.array_equal(tight, caustics(scene, ior=1.5, n_side=160, window=1.0, center=(0.5, 0.0)))
    # refracted_only must be INERT by default and must actually drop the misses when asked. The scene
    # here is a small sphere in a wide aperture, so most rays miss and the difference is large.
    full = caustics(scene, ior=1.5, n_side=160, window=1.5)
    assert np.array_equal(full, caustics(scene, ior=1.5, n_side=160, window=1.5, refracted_only=False))
    only = caustics(scene, ior=1.5, n_side=160, window=1.5, refracted_only=True)
    assert (only > 0).mean() < (full > 0).mean(), "refracted_only did not drop the unrefracted rays"
    print(f"globalillum selftest ok: GI sparse-cache err {err:.3f} (24 vs 144 gather points); "
          f"caustic peak {c.max():.1f}x mean (light focused by refraction)")


if __name__ == "__main__":
    _selftest()

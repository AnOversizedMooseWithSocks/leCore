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
from holographic_raymarch import sphere_trace, sdf_normal, refract_dir


def _cosine_hemisphere(N, n, seed=0):
    """n cosine-weighted sample directions around each unit normal in N:(M,3). Returns (M, n, 3). Vectorised.
    Delegates to the Sampling home (consolidation R4) -- one shared implementation, bit-identical."""
    from holographic_samplinghome import Sampling
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
        from holographic_brdf import lambert                  # the Shading home's diffuse term (consolidation R3)
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


def caustics(sdf, light_dir=(0, -1, 0), receiver_y=-0.9, extent=2.0, res=128, ior=1.5, n_side=200, seed=0):
    """Forward-traced caustics: shoot a grid of parallel light rays down the `light_dir`, refract those that hit
    the object, continue to the receiver plane at y=`receiver_y`, and SPLAT each landing point into a res*res
    grid with np.add.at -- the scatter that is the engine's bundle. Where refracted rays converge the bundle
    piles up: the bright caustic. Returns the (res,res) intensity map (normalised). Vectorised."""
    L = np.asarray(light_dir, float); L = L / (np.linalg.norm(L) + 1e-12)
    g = np.linspace(-extent, extent, n_side)
    GX, GZ = np.meshgrid(g, g)
    O = np.stack([GX.ravel(), np.full(GX.size, 3.0), GZ.ravel()], axis=1)   # start above the scene
    D = np.broadcast_to(L, O.shape).copy()
    hit, t, P = sphere_trace(sdf, O, D, max_steps=80, max_dist=10.0)
    out = D.copy()
    if hit.any():                                            # refract the rays that struck the object
        Nh = sdf_normal(sdf, P[hit])
        out[hit] = refract_dir(D[hit], Nh, ior)
    start = np.where(hit[:, None], P, O)                     # continue from the hit (or the origin if it missed)
    dy = out[:, 1]
    tplane = (receiver_y - start[:, 1]) / np.where(np.abs(dy) < 1e-6, -1e-6, dy)   # reach the receiver plane
    land = start + out * tplane[:, None]
    valid = tplane > 0
    img = np.zeros((res, res))
    xi = ((land[:, 0] + extent) / (2 * extent) * (res - 1)).astype(int)
    zi = ((land[:, 2] + extent) / (2 * extent) * (res - 1)).astype(int)
    inb = valid & (xi >= 0) & (xi < res) & (zi >= 0) & (zi < res)
    np.add.at(img, (zi[inb], xi[inb]), 1.0)                  # the splat = the bundle (accumulate landings)
    return img / (img.mean() + 1e-9)                         # normalise to mean 1 (so peaks read as focusing)


def _selftest():
    from holographic_sdf import sphere, plane
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
    print(f"globalillum selftest ok: GI sparse-cache err {err:.3f} (24 vs 144 gather points); "
          f"caustic peak {c.max():.1f}x mean (light focused by refraction)")


if __name__ == "__main__":
    _selftest()

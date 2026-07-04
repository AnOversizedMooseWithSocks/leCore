"""Monte-Carlo path tracer -- true multi-bounce global illumination, the core of V-Ray, Redshift, and Arnold.
Where the engine's existing GI is a single-bounce irradiance cache (holographic_globalillum), this solves the full
rendering equation by following light along many random paths and averaging: at each surface hit it samples a
bounce direction from the material's BRDF, multiplies the carried throughput by f_r*cos/pdf, and continues until a
ray escapes to the (emissive) environment or Russian roulette ends it. Indirect light -- color bleeding, soft
ambient in concavities, light that reaches a point only after several bounces -- falls out for free, because the
estimator integrates over ALL paths, not just the direct one.

Three standard pieces make it correct and not-too-slow:
  * IMPORTANCE SAMPLING (holographic_brdf.sample_brdf): bounce directions are drawn from the BRDF itself (a one-
    sample MIS mix of a cosine-diffuse and a GGX-specular lobe), so the f_r/pdf weight is low-variance.
  * RUSSIAN ROULETTE: after a couple of bounces, terminate a path with probability tied to its throughput and
    divide survivors by the survival probability -- unbiased, and it stops wasting work on near-black paths.
  * VECTORISATION over rays: all H*W rays of a sample march together each bounce (the NumPy way), so the Python
    loop is only over the few bounces, never over pixels.

HONEST (kept loud):
  * Pure NumPy: this is the OFFLINE renderer. A small frame is seconds-to-minutes, not the interactive GPU path
    tracing of Redshift RT. Method-parity, not speed-parity -- the whole point of the brain/muscle split.
  * No NEXT-EVENT ESTIMATION / no explicit light sampling: light is gathered only when a bounce ray happens to
    hit the emissive ENVIRONMENT. That converges well for a big sky (the demo) but would be very noisy for a
    small bright emitter -- NEE/MIS-with-lights is the honest next step. Measured: noise falls as 1/sqrt(spp).
  * Energy uses the single-scatter GGX BRDF, so it inherits that model's high-roughness energy loss.
"""

import numpy as np
from holographic_raymarch import sphere_trace, sdf_normal, sky_dome, refract_dir
from holographic_brdf import sample_brdf, fresnel_dielectric


def constant_material(albedo=(0.7, 0.7, 0.7), metallic=0.0, roughness=0.5, emission=(0.0, 0.0, 0.0)):
    """A material callback returning the same (albedo, metallic, roughness, emission) at every hit point."""
    alb = np.asarray(albedo, float); em = np.asarray(emission, float)

    def fn(P):
        n = len(P)
        return (np.broadcast_to(alb, (n, 3)).copy(), np.full(n, metallic),
                np.full(n, roughness), np.broadcast_to(em, (n, 3)).copy())
    return fn


def _unpack_mat(out, n):
    """Material callbacks may return a 4-tuple (albedo, metallic, roughness, emission), a 5-tuple with a trailing
    per-point IOR (>1 = dielectric/glass, 0 = opaque), a 6-tuple that ALSO carries a per-point SUBSURFACE strength
    (0 = opaque surface; >0 = translucent), or a 7-tuple that ALSO carries a per-point IRIDESCENCE film thickness
    in nanometres (0 = not iridescent; >0 = thin-film sheen, soap/oil). Normalise to 7 arrays. Old 4/5/6-tuple
    callbacks keep working unchanged -- the trailing channels default to 0 (off)."""
    if len(out) == 7:
        alb, met, rough, emis, ior, sss, irid = out
        return (alb, met, rough, emis, np.asarray(ior, float) * np.ones(n),
                np.asarray(sss, float) * np.ones(n), np.asarray(irid, float) * np.ones(n))
    if len(out) == 6:
        alb, met, rough, emis, ior, sss = out
        return (alb, met, rough, emis, np.asarray(ior, float) * np.ones(n),
                np.asarray(sss, float) * np.ones(n), np.zeros(n))
    if len(out) == 5:
        alb, met, rough, emis, ior = out
        return alb, met, rough, emis, np.asarray(ior, float) * np.ones(n), np.zeros(n), np.zeros(n)
    alb, met, rough, emis = out
    return alb, met, rough, emis, np.zeros(n), np.zeros(n), np.zeros(n)


def _march_through(sdf, O, D, max_steps=32, surf_eps=1e-3):
    """March rays that START INSIDE a solid (negative SDF) along D by |SDF| until they reach the EXIT surface
    (SDF crosses back to >0). sphere_trace can't do this -- it treats the interior as an immediate hit -- so this is
    the dedicated interior traversal a refracted ray needs to pass THROUGH glass to its far face. Returns exit points."""
    P = np.asarray(O, float).copy()
    for _ in range(max_steps):
        d = sdf.eval(P)
        outside = d > surf_eps
        if outside.all():
            break
        P = P + np.where(outside, 0.0, np.abs(d) + 1e-3)[:, None] * D
    return P


def path_trace(sdf, camera, width=96, height=96, spp=16, max_bounce=4, rr_start=2,
               material=None, sky=None, seed=0, return_variance=False, active=None,
               on_progress=None, progress_every=0, should_stop=None, antialias=False,
               sss_dir=None, sss_depth=0.6, sss_sigma=4.0, lights=None):
    """Render an SDF scene by path tracing. `material(P)` -> (albedo(n,3), metallic(n,), roughness(n,),
    emission(n,3)[, ior(n,)]); `sky(D)` -> (n,3) environment radiance for escaped rays. Returns an (H,W,3) HDR image
    (un-tonemapped). spp = samples per pixel; max_bounce = path length; rr_start = bounce after which Russian
    roulette kicks in.

    PROGRESSIVE PREVIEW: pass `on_progress(running_image, samples_done, spp)` and `progress_every=k` to get the
    running mean image handed back every k samples -- the refine stream a RenderSession streams to a viewport while
    the final render accumulates. Defaults (progress_every=0) mean NO callback and byte-identical behaviour to before.

    ANTI-ALIASING (antialias=True, opt-in): by default every sample shoots the ray through the pixel CENTRE, so
    more samples reduce noise but never smooth the jaggies on an edge. With antialias=True each sample jitters the
    ray to a different sub-pixel position, drawn from a LOW-DISCREPANCY sequence (holographic_lowdiscrepancy --
    Roberts' R-sequence, which spreads the offsets evenly instead of clumping the way plain random does), so the
    edges anti-alias as the samples accumulate. Needs a camera whose ray_dirs accepts a `jitter=(dx,dy)` argument;
    if it doesn't, we fall back to centre rays (still correct, just no AA). Default OFF = byte-identical to before.

    GLASS: if the material returns a 5th value IOR>1 at a hit, that surface is a smooth DIELECTRIC -- per ray, it
    REFLECTS with the Fresnel probability and otherwise REFRACTS (Snell via refract_dir, total-internal-reflection
    handled), so light bends through and behind glass shows through. Transmission is tinted by the albedo (a simple
    Beer-Lambert-ish coloured-glass approximation). This is the PBRT smooth-dielectric BSDF (Fresnel-importance-
    sampled reflect/refract), not microfacet-rough glass.

    `return_variance=True` also returns a per-pixel variance-of-the-mean map (for adaptive sampling: where it is high,
    the estimate is noisy and wants more samples). `active` is an optional (H*W,) boolean mask -- only those pixels
    are traced (the rest stay 0), so a second adaptive pass can spend samples ONLY on the noisy pixels."""
    rng = np.random.default_rng(seed)
    material = material or constant_material()
    skyfn = sky if sky is not None else (lambda D: sky_dome(D))
    if sss_dir is not None:
        sss_dir = np.asarray(sss_dir, float); sss_dir = sss_dir / (np.linalg.norm(sss_dir) + 1e-12)
    eye, dirs = camera.ray_dirs(width, height)
    npix = height * width
    base_D = dirs.reshape(-1, 3)
    base_active = np.ones(npix, bool) if active is None else np.asarray(active, bool).reshape(-1)
    accum = np.zeros((npix, 3))
    accsq = np.zeros(npix)                                       # sum of per-sample luminance^2 -> variance

    # anti-alias sub-pixel offsets: one well-distributed (dx,dy) in [-0.5,0.5) per sample (backlog H1). Computed
    # once up front; a camera that can't jitter just ignores them. seed-rotated so different passes don't align.
    aa_offsets = None
    if antialias:
        try:
            from holographic_samplinghome import Sampling
            aa_offsets = Sampling.low_discrepancy(spp, d=2, seed=seed) - 0.5      # (spp,2) in [-0.5, 0.5)
        except Exception:
            aa_offsets = None

    for s in range(spp):
        if aa_offsets is not None:
            try:
                eye, dirs = camera.ray_dirs(width, height, jitter=(aa_offsets[s, 0], aa_offsets[s, 1]))
                base_D = dirs.reshape(-1, 3)                     # this sample's jittered ray directions
            except TypeError:
                aa_offsets = None                                # camera doesn't support jitter -> centre rays
        O = np.broadcast_to(eye, (npix, 3)).astype(float).copy()
        D = base_D.copy()
        throughput = np.ones((npix, 3))
        radiance = np.zeros((npix, 3))
        active_s = base_active.copy()
        for b in range(max_bounce):
            idx = np.where(active_s)[0]
            if idx.size == 0:
                break
            Oi = O[idx]; Di = D[idx]
            hit, t, P = sphere_trace(sdf, Oi, Di)
            gmiss = idx[~hit]                                    # rays that escaped -> gather the environment
            if gmiss.size:
                radiance[gmiss] += throughput[gmiss] * skyfn(Di[~hit])
                active_s[gmiss] = False
            ghit = idx[hit]
            if ghit.size:
                Ph = P[hit]; Dh = Di[hit]
                Ng = sdf_normal(sdf, Ph)                         # geometric normal (out of the solid)
                Nf = Ng.copy()
                flip = np.sum(Nf * Dh, axis=-1) > 0             # faced normal: against the incoming ray
                Nf[flip] = -Nf[flip]
                alb, met, rough, emis, ior, sss, irid = _unpack_mat(material(Ph), len(Ph))
                radiance[ghit] += throughput[ghit] * emis       # emissive surfaces add light
                if (irid > 0).any():
                    # IRIDESCENCE (thin-film): a soap/oil film on the surface reflects a rainbow tint that depends
                    # on the film thickness AND the view angle (holographic_thinfilm). We compute the tint from the
                    # angle between the faced normal and the view direction (-Dh) and MULTIPLY it into the albedo,
                    # so the surface's own reflection takes on the shifting sheen. Only where the material flagged
                    # a film thickness (irid > 0, in nanometres); irid == 0 leaves the albedo untouched.
                    from holographic_thinfilm import interference_reflectance, spectrum_to_rgb
                    it = np.where(irid > 0)[0]
                    cos_v = np.abs(np.sum(Nf[it] * (-Dh[it]), axis=-1))     # view angle from the normal (k,)
                    spec = interference_reflectance(irid[it], cos_v)        # (k, n_lambda), per-point thickness+angle
                    alb[it] = alb[it] * spectrum_to_rgb(spec)              # the reflection takes the sheen
                if sss_dir is not None and (sss > 0).any():
                    # SUBSURFACE glow: a translucent surface lets light leak THROUGH thin regions. We measure how
                    # much solid the light crosses inside the object to reach this point (holographic_raymarch.
                    # subsurface -- Beer-Lambert on the SDF interior toward the sun) and add that as coloured glow,
                    # like emission but modulated by thinness. Only for hits the material flagged (sss>0).
                    # ANTI-BANDING: the march quantizes thickness into depth/steps levels, which reads as contour
                    # bands on smooth objects. Use enough steps that the quantum is small, and DITHER each point's
                    # sample offsets by a deterministic hash of its position (no RNG state -- reproducible), so the
                    # residual quantization becomes fine noise the SVGF pass smooths away.
                    from holographic_raymarch import subsurface as _sss_transmit
                    st = np.where(sss > 0)[0]
                    n_steps = max(10, int(sss_depth / 0.035))            # keep the step quantum ~0.035 world units
                    jit = np.modf(np.sin(Ph[st] @ np.array([12.9898, 78.233, 37.719])) * 43758.5453)[0] % 1.0
                    glow = _sss_transmit(sdf, Ph[st], Nf[st], sss_dir, depth=sss_depth, sigma=sss_sigma,
                                         steps=n_steps, jitter=np.abs(jit))   # (k,)
                    radiance[ghit[st]] += throughput[ghit[st]] * (sss[st] * glow)[:, None] * alb[st]
                glass = ior > 1.0
                surf = ~glass
                if surf.any():                                  # opaque: GGX/diffuse BRDF bounce
                    si = np.where(surf)[0]
                    if lights:
                        # NEXT-EVENT ESTIMATION: besides the random bounce below (which carries INDIRECT light),
                        # look straight at each light with a shadow ray and add its DIRECT contribution. This is
                        # what makes small bright lamps converge and gives real, correctly-shaped shadows. The
                        # bounce still happens, so indirect light (colour bleeding, ambient) is not lost.
                        from holographic_lights import direct_lighting
                        direct = direct_lighting(sdf, Ph[si], Nf[si], -Dh[si], alb[si], met[si], rough[si],
                                                 lights, rng)                # (k,3) direct radiance
                        radiance[ghit[si]] += throughput[ghit[si]] * direct
                    L, weight = sample_brdf(Nf[si], -Dh[si], alb[si], met[si], rough[si], rng)
                    throughput[ghit[si]] = throughput[ghit[si]] * weight
                    O[ghit[si]] = Ph[si] + Nf[si] * 2e-3
                    D[ghit[si]] = L
                if glass.any():                                 # dielectric: Fresnel reflect-or-refract THROUGH
                    gi = np.where(glass)[0]
                    cosg = np.abs(np.sum(Nf[gi] * (-Dh[gi]), axis=-1))
                    R = fresnel_dielectric(cosg, ior[gi])
                    do_refl = rng.random(gi.size) < R
                    refl = Dh[gi] - 2.0 * np.sum(Dh[gi] * Nf[gi], axis=-1)[:, None] * Nf[gi]
                    refr_in = refract_dir(Dh[gi], Ng[gi], ior[gi])           # bend entering the glass
                    exitP = _march_through(sdf, Ph[gi] + refr_in * 3e-3, refr_in)   # traverse to the far face
                    Nx = sdf_normal(sdf, exitP)
                    refr_out = refract_dir(refr_in, Nx, ior[gi])             # bend again exiting -> into the scene
                    Lg = np.where(do_refl[:, None], refl, refr_out)
                    newO = np.where(do_refl[:, None], Ph[gi] + Nf[gi] * 3e-3, exitP + refr_out * 3e-3)
                    tint = np.where(do_refl[:, None], 1.0, alb[gi])          # colour only the transmitted light
                    throughput[ghit[gi]] = throughput[ghit[gi]] * tint
                    O[ghit[gi]] = newO
                    D[ghit[gi]] = Lg
                if b >= rr_start:                               # Russian roulette: kill weak paths, unbiased
                    q = np.clip(throughput[ghit].max(axis=-1), 0.05, 1.0)
                    survive = rng.random(ghit.size) < q
                    throughput[ghit] = throughput[ghit] / q[:, None]
                    active_s[ghit[~survive]] = False
        accum += radiance
        accsq += radiance.mean(axis=-1) ** 2
        if progress_every and on_progress is not None and (s + 1) % progress_every == 0 and (s + 1) < spp:
            # hand back the running mean so a viewport can show the image refining (the progressive preview)
            on_progress(np.clip((accum / (s + 1)).reshape(height, width, 3), 0.0, None), s + 1, spp)
        if should_stop is not None and should_stop():
            # CANCELLED (item F): stop early and return the partial image accumulated so far -- a partial render
            # beats a frozen UI. Checked only between passes, so it costs nothing per sample.
            done = s + 1
            mean = accum / done
            img = np.clip(mean.reshape(height, width, 3), 0.0, None)
            if return_variance:
                lum = mean.mean(axis=-1)
                var = np.maximum(accsq / done - lum ** 2, 0.0) / max(done, 1)
                return img, var
            return img
    mean = accum / spp
    img = np.clip(mean.reshape(height, width, 3), 0.0, None)
    if return_variance:
        lum = mean.mean(axis=-1)
        var = np.maximum(accsq / spp - lum ** 2, 0.0) / max(spp, 1)   # variance of the MEAN estimator
        return img, var.reshape(height, width)
    return img


def _selftest():
    # a convex sphere SDF + a tiny camera
    class _SDF:
        def eval(self, P):
            return np.linalg.norm(P, axis=-1) - 1.0

    class _Cam:
        eye = np.array([0.0, 0.0, 3.0])
        def ray_dirs(self, w, h):
            ys, xs = np.mgrid[0:h, 0:w]
            u = (xs / (w - 1) - 0.5) * 1.4; v = -(ys / (h - 1) - 0.5) * 1.4
            d = np.stack([u, v, -np.ones_like(u)], axis=-1)
            return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)

    sdf, cam = _SDF(), _Cam()
    white_env = lambda D: np.ones((len(D), 3))
    # 1. WHITE FURNACE: a diffuse convex sphere in a unit environment reflects ~ its albedo (unbiased estimator)
    mat = constant_material(albedo=(0.6, 0.6, 0.6), metallic=0.0, roughness=1.0)
    img = path_trace(sdf, cam, width=48, height=48, spp=48, max_bounce=3, material=mat, sky=white_env, seed=0)
    lit = img.reshape(-1, 3)
    lit = lit[lit.sum(1) > 0.1]                                 # the sphere pixels (background is also 1, exclude via mask below)
    # the sphere disk: pixels whose value is clearly below the env (reflectance < 1)
    sphere_px = img.reshape(-1, 3)[(img.reshape(-1, 3).mean(1) < 0.95)]
    assert 0.5 < sphere_px.mean() < 0.62, sphere_px.mean()      # converges to ~albedo (single-scatter slack)
    # 2. noise falls with more samples (~1/sqrt(spp)): variance at 64 spp < variance at 8 spp
    a = path_trace(sdf, cam, width=40, height=40, spp=8, max_bounce=3, material=mat, sky=white_env, seed=1)
    b = path_trace(sdf, cam, width=40, height=40, spp=64, max_bounce=3, material=mat, sky=white_env, seed=1)
    ref = path_trace(sdf, cam, width=40, height=40, spp=256, max_bounce=3, material=mat, sky=white_env, seed=2)
    mask = ref.mean(-1) < 0.95
    err_a = np.sqrt(((a - ref)[mask] ** 2).mean()); err_b = np.sqrt(((b - ref)[mask] ** 2).mean())
    assert err_b < err_a, (err_a, err_b)                       # more samples -> less noise
    print(f"pathtrace selftest ok: white-furnace sphere reflectance {sphere_px.mean():.3f} (~albedo 0.6, unbiased); "
          f"noise {err_a:.3f}(8spp) -> {err_b:.3f}(64spp), falls with sqrt(spp)")


if __name__ == "__main__":
    _selftest()

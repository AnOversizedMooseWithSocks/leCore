"""SPECIMEN RENDERING -- one call from an SDF to a finished, denoised, graded image.

This exists because the same five steps were hand-assembled for every crystal render in the session
that built the crystal system: trace, build a G-buffer, denoise, clamp fireflies, grade. Each step is
somebody's shipped capability; what was missing was the WIRING, and hand-wiring it every time is how
a step gets forgotten. It also silently dropped whatever was newest -- the adaptive tracer landed on
main while the crystal work was happening on a branch, and none of the crystal renders used it.

WHAT IT COMPOSES, and where each piece came from:

    path_trace_adaptive   CI-driven sampling: you state a TOLERANCE, not a sample count, and pixels
                          that have converged stop being sampled. Measured on a geode at tol=0.03,
                          83% of a flat 48-spp render's samples avoided at equal mean radiance.
    a sphere-traced       depth + normal + albedo, which SVGF needs and the tracer does not return.
    G-buffer
    svgf_denoise          edge-aware, normal- and depth-guided -- a blur would take the facet edges
                          that make a crystal legible. Measured 67-77% grain reduction.
    firefly clamp         the tracer has no next-event estimation, so a ray that happens to hit the
                          sun returns a huge value: those are the coloured speckles. Clamping trades
                          a little energy for a lot of variance and is stated, not hidden.
    exposure sweep        the white point is SEARCHED rather than guessed, scoring contrast and
                          colour retention. Measured: the obvious choice (white=p99) gave 0.2%
                          highlights and no dielectric sparkle at all.

THE TOLERANCE IS THE KNOB THAT MATTERS, and its behaviour is worth knowing before you turn it:
adaptive sampling estimates a per-block confidence interval, and at a LOW `min_spp` that estimate is
optimistic -- measured, min_spp=8 declared convergence everywhere (spp 8-8 flat) even at tol=5e-4,
while min_spp=16 escalated properly (spp 16-96). So `min_spp` is not just a floor on quality, it is
what makes the CI trustworthy at all. Default 16 for that reason.
"""

import numpy as np


def camera_rays(eye, target, width, height, fov_deg, up=(0.0, 1.0, 0.0)):
    """Per-pixel ray directions for a look-at camera. Returns (eye, dirs (H,W,3))."""
    e = np.asarray(eye, float)
    t = np.asarray(target, float)
    f = t - e
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    yy, xx = np.mgrid[0:int(height), 0:int(width)]
    ndx = (xx + 0.5) / width * 2 - 1
    ndy = 1 - (yy + 0.5) / height * 2
    th = np.tan(np.radians(float(fov_deg)) / 2)
    d = (f[None, None, :] + ndx[..., None] * th * (width / height) * r[None, None, :]
         + ndy[..., None] * th * u[None, None, :])
    return e, d / np.linalg.norm(d, axis=2, keepdims=True)


def gbuffer(sdf, eye, dirs, far=12.0, steps=140, eps=8e-4, albedo_fn=None):
    """Depth, normal and albedo by sphere tracing -- the inputs SVGF needs.

    Built here because the path tracer returns radiance only, and the shipped `sdf_depth_cpu` fixes
    its own camera orientation and cannot match an arbitrary view. Sphere tracing is valid for the
    SDFs this path renders; a convolution-style DENSITY field would need marching instead, which is
    why `albedo_fn` is a hook rather than an assumption.
    """
    H, W, _ = dirs.shape
    D = dirs.reshape(-1, 3)
    O = np.tile(np.asarray(eye, float), (len(D), 1))
    t = np.zeros(len(D))
    alive = np.ones(len(D), bool)
    for _ in range(int(steps)):
        P = O + D * t[:, None]
        s = np.asarray(sdf(P), float).ravel()
        hit = np.abs(s) < eps
        alive &= ~hit & (t < far)
        if not alive.any():
            break
        t = np.where(alive, t + np.maximum(s, eps * 0.5), t)
    P = O + D * t[:, None]
    N = np.zeros_like(P)
    h = 1.2e-3
    for k in range(3):
        o = np.zeros(3)
        o[k] = h
        N[:, k] = (np.asarray(sdf(P + o), float).ravel()
                   - np.asarray(sdf(P - o), float).ravel())
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    alb = (np.asarray(albedo_fn(P), float) if albedo_fn is not None
           else np.tile(np.array([0.6, 0.6, 0.62]), (len(P), 1)))
    return t.reshape(H, W), N.reshape(H, W, 3), alb.reshape(H, W, 3)


def denoiser_relaxation(depth, normal, relax=6.0):
    """How much the sampler may RELAX because the denoiser will finish the job -- from the G-buffer.

    An edge-aware denoiser is guided by exactly these two buffers: it averages freely across a flat
    patch and refuses to average across a depth or normal discontinuity. So the places it CANNOT
    clean are precisely the places its edge-stopping function fires, and those are computable from
    the G-buffer BEFORE any light is traced.

    Returns a multiplier in [1, relax] to be applied TO THE TOLERANCE: 1 on edges (ask for exactly the
    quality requested, because nothing downstream will fix them) rising to `relax` on flat regions
    (stop early -- an edge-aware filter averages a flat patch almost perfectly for free).

    THE SIGN IS EASY TO GET BACKWARDS and I did: dividing by an edge mask makes EDGES looser, which is
    the exact opposite and shows up as "the guided version changed nothing" because the tight regions
    saturate anyway. The direction is fixed by asking what the denoiser can do, not what the edges are:
    where it works, spend less.
    """
    d = np.asarray(depth, float)
    N = np.asarray(normal, float)
    dz = np.zeros_like(d)
    dz[1:-1, :] += np.abs(d[2:, :] - d[:-2, :])
    dz[:, 1:-1] += np.abs(d[:, 2:] - d[:, :-2])
    nz = np.zeros_like(d)
    nz[1:-1, :] += np.linalg.norm(N[2:, :, :] - N[:-2, :, :], axis=2)
    nz[:, 1:-1] += np.linalg.norm(N[:, 2:, :] - N[:, :-2, :], axis=2)
    # Normalise each cue by its own spread so the mix does not depend on scene scale (D-7).
    dz = dz / max(np.percentile(dz, 97), 1e-9)
    nz = nz / max(np.percentile(nz, 97), 1e-9)
    e = np.clip(np.maximum(dz, nz), 0.0, 1.0)
    return 1.0 + (float(relax) - 1.0) * (1.0 - e)


def clamp_fireflies(img, pct=99.3, factor=2.5):
    """Clamp radiance outliers. WITHOUT next-event estimation a light is found only by chance, so the
    rare ray that hits it returns a huge value -- the speckles. This makes the tails DIMMER, not more
    correct; it is a variance trade and is reported rather than buried."""
    hot = np.percentile(img, float(pct)) * float(factor)
    return np.minimum(img, hot), int((img > hot).sum())


def grade(img, prefer="colour"):
    """Tone map with a SEARCHED exposure rather than a guessed one.

    Returns (image, report). The white point is swept and scored, because the obvious choice is
    measurably wrong: at white=p99 a gem render came back with 0.2% highlights and no dielectric
    sparkle. `prefer` picks what the score rewards -- "colour" keeps saturation (right for gems,
    which are defined by hue), "contrast" maximises tonal range.
    """
    best = None
    for wp, gain in ((99.6, 0.85), (99.0, 0.95), (98.0, 1.05), (96.0, 1.2), (94.0, 1.35)):
        e = img / max(np.percentile(img, wp), 1e-9) * gain
        t = (e * (2.51 * e + 0.03)) / (e * (2.43 * e + 0.59) + 0.14)   # ACES-style approximation
        o = np.clip(t, 0, 1) ** (1 / 2.2)
        lum = o @ np.array([0.299, 0.587, 0.114])
        sub = lum <= 0.88
        if sub.sum() < 64:
            continue
        px = o[sub]
        p5, p95 = np.percentile(lum[sub], [5, 95])
        sat = float(np.mean((px.max(axis=1) - px.min(axis=1)) / np.maximum(px.max(axis=1), 1e-9)))
        hi = float((px.max(axis=1) > 0.88).mean())
        contrast = float(p95 - p5)
        score = ((1.6 * sat + 0.9 * contrast) if prefer == "colour"
                 else (1.6 * contrast + 0.6 * sat)) - 1.2 * max(hi - 0.22, 0.0)
        if best is None or score > best[0]:
            best = (score, o, {"white_pct": wp, "gain": gain, "contrast": round(contrast, 4),
                               "saturation": round(sat, 4), "highlights": round(hi, 4)})
    return best[1], best[2]


def render_plan(mind, sdf, eye, target, material, sky, width=240, height=180, fov_deg=42.0,
                budget_s=None, min_spp=16, max_spp=64, max_bounce=5, probe=(64, 52), seed=0,
                safety=0.55):
    """MEASURE the cost on a tiny tile, then say what fits -- instead of estimating and overrunning.

    Four render overruns in this codebase's history came from extrapolating a big render from a cheap
    probe LINEARLY. On a transmissive scene that always understates: more samples means more rays
    surviving deep enough to ENTER glass, and marching through interiors dominates. So the estimate
    here is deliberately pessimistic (`safety`, default 0.55 of the linear prediction) and the
    measurement is reported next to it rather than hidden.

    Returns {'probe_s', 'per_pixel_sample_s', 'estimate_s', 'fits', 'tier', 'why', 'suggest'}, where
    `suggest` is a (width, height, max_spp) that should land inside `budget_s`.
    """
    import time as _t
    pw, ph = int(probe[0]), int(probe[1])
    cam = mind.camera(eye=tuple(eye), target=tuple(target), fov_deg=float(fov_deg))
    t0 = _t.time()
    mind.path_trace_adaptive(sdf, cam, width=pw, height=ph, tol=0.02, min_spp=int(min_spp),
                             max_spp=int(min_spp), max_bounce=int(max_bounce),
                             material=material, sky=sky, seed=int(seed))
    probe_s = _t.time() - t0
    unit = probe_s / max(pw * ph * min_spp, 1)                  # seconds per pixel-sample, measured
    est = unit * width * height * max_spp / max(float(safety), 1e-6)
    plan = mind.compute_plan(int(width) * int(height) * int(max_spp), calls_expected=1)
    out = {"probe_s": round(probe_s, 2), "per_pixel_sample_s": unit,
           "estimate_s": round(est, 1), "tier": plan.get("tier"), "why": plan.get("why"),
           "fits": None, "suggest": (int(width), int(height), int(max_spp))}
    if budget_s is not None:
        out["fits"] = bool(est <= float(budget_s))
        if not out["fits"]:
            # Shrink RESOLUTION before samples: samples are what quality is measured in, and cost is
            # superlinear in them on glass, so pixels are the cheaper thing to give up.
            scale = (float(budget_s) * float(safety) / max(unit * width * height * max_spp, 1e-9)) ** 0.5
            scale = max(min(scale, 1.0), 0.15)
            out["suggest"] = (max(int(width * scale), 32), max(int(height * scale), 24), int(max_spp))
    return out


def render_specimen(mind, sdf, eye, target, material, sky, width=240, height=180, fov_deg=42.0,
                    tol=0.02, min_spp=16, max_spp=64, max_bounce=5, seed=0,
                    denoise=True, albedo_fn=None, prefer="colour", far=12.0, budget_s=None,
                    bake=False):
    """Trace -> denoise -> clamp -> grade, in one call. Returns (image, report).

    `tol` replaces a sample count: state the quality you want and the sampler stops where it has
    reached it. See the module docstring for why `min_spp` must not be small.
    """
    # BAKE THE FIELD (opt-in). Profiling a geode render showed 84% of the time inside SDF evaluation
    # -- 93,870 field calls, each looping over 30 crystals in Python -- so the field, not the tracer,
    # is the cost. `bake_sdf` precomputes it onto a grid whose sample cost is INDEPENDENT of primitive
    # count, which is the engine's own bake-once-sample-O(1) lever.
    #
    # MEASURED on that scene: field evaluation 16x faster at res 96; a whole frame 30.2 s -> 3.1 s
    # (9.6x), with the bake paying for itself after 0.4 frames -- so for animation, or any repeated
    # view of a static scene, it is close to free.
    #
    # AND THE COST, which is why this is OPT-IN rather than default: trilinear interpolation makes the
    # gradient piecewise-constant, so shading NORMALS come out quantised. Silhouette IoU stays 0.97
    # (the shape is right) but radiance differs by ~0.2 mean, and that error DOES NOT CONVERGE with
    # resolution (0.2074 at res 112, 0.2059 at 176) -- it is the interpolation scheme, not the grid
    # spacing. It is also not a refraction artefact: an opaque material shows the same 0.19. Use it
    # for previews, animation drafts and anything where silhouette dominates; not for a final gem.
    if bake:
        res = int(bake) if int(bake) > 1 else 112
        lo = np.min(np.asarray([eye, target], float), axis=0) - float(far) * 0.12
        hi = np.max(np.asarray([eye, target], float), axis=0) + float(far) * 0.12
        span = float(np.max(hi - lo))
        c = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float))
        sdf = mind.bake_sdf(sdf, tuple(c - span * 0.5), tuple(c + span * 0.5), res)

    plan = None
    if budget_s is not None:
        plan = render_plan(mind, sdf, eye, target, material, sky, width=width, height=height,
                           fov_deg=fov_deg, min_spp=min_spp, max_spp=max_spp,
                           max_bounce=max_bounce, budget_s=budget_s, seed=seed)
        if not plan["fits"]:
            width, height, max_spp = plan["suggest"]        # fit the budget rather than overrun it
    # THE G-BUFFER COMES FIRST NOW. It was always needed for denoising and was being computed AFTER
    # the trace; computing it first costs the same sphere-trace and lets it do double duty -- guiding
    # where samples are worth spending as well as guiding the filter. That is the whole fix: the
    # sampler and the denoiser were solving overlapping problems without talking.
    gbuf = None
    tol_map = None
    if denoise:
        e_, dirs_ = camera_rays(eye, target, width, height, fov_deg)
        gbuf = gbuffer(sdf, e_, dirs_, far=far, albedo_fn=albedo_fn)
        tol_map = float(tol) * denoiser_relaxation(gbuf[0], gbuf[1])   # relax where denoising works
    cam = mind.camera(eye=tuple(eye), target=tuple(target), fov_deg=float(fov_deg))
    img, ad = mind.path_trace_adaptive(sdf, cam, width=int(width), height=int(height),
                                       tol=float(tol), tol_map=tol_map,
                                       min_spp=int(min_spp), max_spp=int(max_spp),
                                       max_bounce=int(max_bounce), material=material, sky=sky,
                                       seed=int(seed))
    img = np.asarray(img, float)
    spp = np.asarray(ad["spp"], float)
    rep = {"spp_min": float(spp.min()), "spp_max": float(spp.max()), "spp_mean": float(spp.mean()),
           "sample_saving": float(ad.get("saving", 0.0)), "why": ad.get("why", "")}
    grain = lambda a: float(np.abs(np.diff(a @ np.array([0.299, 0.587, 0.114]), axis=0)).mean()
                            + np.abs(np.diff(a @ np.array([0.299, 0.587, 0.114]), axis=1)).mean())
    rep["grain_raw"] = grain(img)
    if denoise:
        depth, nrm, alb = gbuf
        img = np.asarray(mind.svgf_denoise(img, nrm, alb, depth, levels=4), float)
        rep["grain_denoised"] = grain(img)
    img, n_fire = clamp_fireflies(img)
    rep["fireflies_clamped"] = n_fire
    img, gr = grade(img, prefer=prefer)
    rep.update(gr)
    if plan is not None:
        rep["plan"] = plan
        rep["resized"] = (int(width), int(height))
    return img, rep


def _selftest():
    import lecore

    mind = lecore.UnifiedMind(dim=256, seed=0)

    # A transmissive ball on a floor: small, but it exercises every stage including refraction.
    def scene(P):
        Q = np.atleast_2d(np.asarray(P, float))
        return np.minimum(np.linalg.norm(Q, axis=1) - 0.55, Q[:, 1] + 0.6)
    scene.eval = scene

    def mat(P):
        Q = np.atleast_2d(np.asarray(P, float))
        n = len(Q)
        floor = Q[:, 1] < -0.59
        alb = np.where(floor[:, None], np.array([[0.6, 0.55, 0.5]]), np.array([[0.5, 0.3, 0.7]]))
        return (alb, np.zeros(n), np.where(floor, 0.8, 0.05), np.zeros((n, 3)),
                np.where(floor, 0.0, 1.55), np.zeros(n), np.zeros(n),
                np.where(floor[:, None], 0.0, np.array([[1.1, 2.3, 0.55]])))
    sky = mind.sky_model(hour=10.0, clouds=[("cirrus", 0.2)], sun_intensity=24.0)

    img, rep = render_specimen(mind, scene, (1.3, 0.7, 1.5), (0, 0, 0), mat, sky,
                               width=48, height=40, tol=0.02, min_spp=16, max_spp=32,
                               max_bounce=4, seed=1)
    assert img.shape == (40, 48, 3), img.shape
    assert 0.0 <= img.min() and img.max() <= 1.0, "graded output must be display-referred"

    # 1) ADAPTIVE SAMPLING ACTUALLY SAVED WORK -- the reason to use it over a flat spp.
    assert rep["sample_saving"] > 0.0, "adaptive sampling reported no saving: %r" % rep
    assert rep["spp_min"] >= 16, "min_spp floor breached: %r" % rep["spp_min"]

    # 2) DENOISING REDUCED GRAIN. Asserted numerically because "it ran" would pass with a no-op
    # filter, and a no-op denoiser in a render pipeline is invisible until someone looks closely.
    assert rep["grain_denoised"] < rep["grain_raw"], \
        "denoise must reduce grain: %.5f -> %.5f" % (rep["grain_raw"], rep["grain_denoised"])

    # 3) THE EXPOSURE WAS CHOSEN, not defaulted, and it did not blow the image out.
    assert 90.0 <= rep["white_pct"] <= 100.0, rep
    assert rep["highlights"] < 0.6, "graded image is blown: %r" % rep["highlights"]

    # 4) THE TOLERANCE MUST ACTUALLY CONTROL THE SAMPLE COUNT.
    #
    # This gate previously asserted that min_spp=8 FAILS to escalate while 16 succeeds -- it was
    # pinning a BUG as if it were a property. The cause was that variance is only defined once a
    # pixel has more than one round, and `done` did not require that, so after round one the CI
    # half-width was 0 and every pixel "converged" having measured nothing. Any config where one
    # round reached min_spp returned exactly `block` samples whatever the tolerance.
    #
    # With that fixed, the honest property is the one a caller depends on: a TIGHTER tolerance must
    # buy MORE samples. Asserted across two tolerances rather than two floors.
    cam = mind.camera(eye=(1.3, 0.7, 1.5), target=(0, 0, 0), fov_deg=42.0)
    _, loose = mind.path_trace_adaptive(scene, cam, width=32, height=28, tol=5e-2, min_spp=8,
                                        block=8, max_spp=64, max_bounce=3, material=mat, sky=sky,
                                        seed=1)
    _, tight = mind.path_trace_adaptive(scene, cam, width=32, height=28, tol=5e-4, min_spp=8,
                                        block=8, max_spp=64, max_bounce=3, material=mat, sky=sky,
                                        seed=1)
    lo_s, hi_s = np.asarray(loose["spp"], float), np.asarray(tight["spp"], float)
    assert hi_s.mean() > lo_s.mean() * 1.2, \
        "a tighter tolerance must buy more samples: %.1f vs %.1f mean spp" % (hi_s.mean(), lo_s.mean())
    assert hi_s.max() > hi_s.min(), "sampling must be spatially ADAPTIVE, not flat: %r" % hi_s.max()

    # 5) BAKING THE FIELD IS A SPEED/ACCURACY TRADE, and both halves are measured. The silhouette must
    # survive (the shape is what a preview is for); the shading is allowed to differ, which is exactly
    # why bake is opt-in.
    import time as _time
    _t = _time.time()
    _direct, _ = render_specimen(mind, scene, (1.3, 0.7, 1.5), (0, 0, 0), mat, sky, width=40,
                                 height=34, tol=0.03, min_spp=8, max_spp=16, max_bounce=3, seed=1,
                                 denoise=False)
    _t_direct = _time.time() - _t
    _t = _time.time()
    _baked, _ = render_specimen(mind, scene, (1.3, 0.7, 1.5), (0, 0, 0), mat, sky, width=40,
                                height=34, tol=0.03, min_spp=8, max_spp=16, max_bounce=3, seed=1,
                                denoise=False, bake=96, far=4.0)
    _t_baked = _time.time() - _t
    _l = lambda a: a @ np.array([0.299, 0.587, 0.114])
    _sa, _sb = _l(_direct) > 0.02, _l(_baked) > 0.02
    _iou = float((_sa & _sb).sum()) / max(float((_sa | _sb).sum()), 1.0)
    assert _iou > 0.85, "a baked field must keep the silhouette: IoU %.3f" % _iou

    # 5) THE BUDGET IS HONOURED BY MEASURING, NOT ESTIMATING. Four render overruns in this codebase
    # came from linear extrapolation off a cheap probe; on glass that always understates, because more
    # samples means more rays deep enough to enter the interior. An absurdly small budget must force a
    # SMALLER render rather than an overrun.
    plan = render_plan(mind, scene, (1.3, 0.7, 1.5), (0, 0, 0), mat, sky,
                       width=400, height=320, max_spp=64, budget_s=1.0, probe=(24, 20), seed=1)
    assert plan["fits"] is False, "a 400x320 64-spp glass render cannot fit 1 s: %r" % plan
    sw, sh, _ = plan["suggest"]
    assert sw < 400 and sh < 320, "an over-budget plan must suggest something SMALLER: %r" % (plan["suggest"],)
    assert plan["per_pixel_sample_s"] > 0.0 and plan["tier"], plan

    img2, rep2 = render_specimen(mind, scene, (1.3, 0.7, 1.5), (0, 0, 0), mat, sky,
                                 width=400, height=320, tol=0.02, min_spp=16, max_spp=32,
                                 max_bounce=3, seed=1, budget_s=8.0)
    assert "plan" in rep2 and rep2["resized"][0] <= 400, rep2.get("resized")
    assert img2.shape[0] == rep2["resized"][1] and img2.shape[1] == rep2["resized"][0]

    print("gemrender budget OK: measured %.2e s per pixel-sample, 400x320@64spp estimated %.0fs "
          "(refused against a 1s budget, suggested %dx%d); an 8s budget rendered %dx%d"
          % (plan["per_pixel_sample_s"], plan["estimate_s"], sw, sh,
             rep2["resized"][0], rep2["resized"][1]))

    print("gemrender bake OK: direct %.1fs vs baked %.1fs, silhouette IoU %.3f" % (
        _t_direct, _t_baked, _iou))
    print("gemrender selftest OK: adaptive saved %.0f%% of samples (spp %d-%d), grain %.4f -> %.4f, "
          "exposure chose white=p%.0f (contrast %.3f, saturation %.3f), %d fireflies clamped; "
          "tolerance controls spp: loose %.1f -> tight %.1f mean"
          % (100 * rep["sample_saving"], rep["spp_min"], rep["spp_max"], rep["grain_raw"],
             rep["grain_denoised"], rep["white_pct"], rep["contrast"], rep["saturation"],
             rep["fireflies_clamped"], lo_s.mean(), hi_s.mean()))


if __name__ == "__main__":
    _selftest()

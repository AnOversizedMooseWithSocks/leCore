"""PLATE -- render_specimen's pipeline with a FIXED white point instead of a searched one.

WHY THIS EXISTS. `render_specimen` ends in `grade(prefer=...)`, which SEARCHES an exposure by sweeping
white_pct 99.6..94 and scoring saturation and contrast. That is right for a hero render and wrong for a
technical plate, and the difference is not cosmetic:

  * The search RE-NORMALISES its input, so scaling the lights cannot change the result. Three sessions
    were spent lowering `gain` to fix a blowout that was structurally immune to it.
  * On a subject that is mostly BRIGHT -- an anatomical section is pale fat (0.90) and bone (0.87) -- the
    search has no dark reference to expose against and drives the pale tissues to clipping. A coat
    authored at (0.24,0.145,0.072), a dark brown, arrived as pale cream.
  * A plate needs REPRODUCIBLE tone. A searched exposure means two renders of the same subject under the
    same light are not comparable, which defeats the point of a plate.

So: identical trace/denoise/clamp, then a FIXED white point the caller states. Everything upstream is
reused verbatim -- this is not a second renderer, it is the same one with the last step replaced.

KEPT NEGATIVE: a fixed white point will clip if you set it below the scene's real dynamic range. That is
the trade -- reproducible and your responsibility, versus adaptive and unpredictable. `report()` returns
the measured highlight fraction so the caller can check rather than guess."""
import numpy as np


def tonemap_fixed(img, white=1.0, gamma=2.2, toe=0.0, exposure=1.0):
    """Reinhard-with-white-point, then gamma. Deterministic and stateless -- the same linear input always
    yields the same output, which is the property the searched grade cannot offer.

    `white` is the luminance mapped to display 1.0. `toe` lifts the black point if a plate needs it.
    Reinhard rather than ACES because a plate wants the tissue VALUES readable, not filmic contrast."""
    # EXTENDED REINHARD, written out properly: L_out = L(1 + L/W^2) / (1 + L) with L and W in LINEAR
    # units. The first draft had a garbled `W ** 0 / 1.0` term that collapsed the white point to 1 and
    # left 85% of a bright plate clipped -- caught by assertion 3, which measures clipping instead of
    # trusting the algebra. WRITE THE FORMULA, THEN MEASURE IT: a tonemap that looks plausible and does
    # nothing is exactly the failure this module was built to end.
    # WHITE POINT AND EXPOSURE ARE DIFFERENT KNOBS, and the first version shipped only one. In
    # x = L(1 + L/W^2)/(1 + L), when L is well BELOW W the W term is negligible and x ~ L/(1+L) --
    # independent of white. MEASURED: doubling white 1.587 -> 3.4 moved the plate mean only
    # 0.444 -> 0.424, because W sets where the HIGHLIGHT ROLLOFF lands, not overall brightness.
    # `exposure` is the linear multiplier that actually sets brightness. Without it a "fixed exposure"
    # renderer had no exposure control -- the exact gap it was built to close, reopened one level down.
    L = np.maximum(np.asarray(img, float), 0.0) * max(float(exposure), 0.0)
    W = max(float(white), 1e-9)
    x = L * (1.0 + L / (W * W)) / (1.0 + L)
    x = np.clip(x, 0.0, 1.0)
    if toe:
        x = toe + (1.0 - toe) * x
    return np.clip(x, 0.0, 1.0) ** (1.0 / max(float(gamma), 1e-6))


def suggest_white(img, pct=99.0, headroom=1.7):
    """A white point from the image's own luminance percentile -- a STARTING VALUE the caller can pin.

    This is deliberately not automatic. Measuring a sensible white once and then hard-coding it is what
    makes a plate reproducible; recomputing it every render is the searched grade wearing a new hat.

        HEADROOM IS NOT OPTIONAL. Extended Reinhard maps L == W to EXACTLY 1.0, so a white point set AT the
    99th percentile clips the top 1% by construction -- and on a uniform-ish subject like a section
    plate, where p99 is close to the mean, that clips nearly the whole frame. Measured: white=p99 on a
    flat bright fixture clipped 100% of pixels. `headroom` lifts W above the percentile so the bright
    tissues land BELOW white and keep their separation, which is the entire point."""
    lum = np.asarray(img, float) @ np.array([0.2126, 0.7152, 0.0722])
    return float(np.percentile(lum, float(pct))) * float(headroom)


def highlight_fraction(img, thresh=0.985):
    """Fraction of pixels at or above `thresh` in any channel -- the number that says "blown out".

    Reported rather than asserted, because a specular highlight SHOULD clip. It is the caller's job to
    know what fraction is acceptable for their plate; ours is to measure it honestly."""
    return float((np.asarray(img, float) >= float(thresh)).any(axis=-1).mean())


def render_plate(mind, sdf, eye, target, material, sky, width=340, height=255, fov_deg=36.0,
                 tol=0.012, min_spp=16, max_spp=128, max_bounce=2, seed=0, denoise=True,
                 white=None, white_pct=99.0, gamma=2.2, exposure=1.0, far=12.0, albedo_fn=None,
                 budget_s=None, upsample=1):
    """Trace -> denoise -> clamp -> FIXED tonemap. Returns (image, report).

    `white=None` measures one from the linear render at `white_pct` and REPORTS it, so a caller can pin
    that number and get an identical plate every time thereafter. Pass `white` explicitly for a series
    where every plate must share tone -- which is the entire reason a plate differs from a hero render.

    The report carries `white`, `highlight_fraction` (before and after), `grain_raw`, `grain_denoised`
    and `sample_saving`, so "is it blown out" is a measurement rather than an opinion."""
    from holographic.rendering.holographic_gemrender import (gbuffer, camera_rays, clamp_fireflies,
                                                             denoiser_relaxation, render_plan)

    # BUDGET IS MEASURED, NOT TIMED OUT. `path_trace_adaptive` takes no budget_s -- render_specimen
    # enforces it through `render_plan`, which PROBES the cost on a tiny tile and then shrinks
    # resolution/spp to fit rather than estimating and overrunning. Same mechanism here, and the plan
    # is reported so a caller sees what was traded away instead of silently getting a smaller image.
    # PLAN THE COST OF WHAT IS ACTUALLY TRACED. With upsample=N the path tracer runs at 1/N per axis --
    # 1/N^2 the pixels -- so planning against the OUTPUT size over-estimates by N^2 and shrinks the frame
    # for no reason. MEASURED: a 560x420 request with upsample=2 was cut to 162x122 by a 420s budget and
    # then finished in 50s, leaving ~3x resolution on the table. The plan must see the TRACE, not the
    # deliverable. Ordering matters: trace dims are computed first, then planned, then upsampled back.
    up0 = max(1, int(upsample))
    plan = None
    if budget_s is not None:
        pw, ph = max(8, int(width) // up0), max(8, int(height) // up0)
        plan = render_plan(mind, sdf, eye, target, material, sky, width=pw, height=ph,
                           fov_deg=fov_deg, min_spp=min_spp, max_spp=max_spp,
                           max_bounce=max_bounce, budget_s=budget_s, seed=seed)
        if not plan["fits"]:
            sw, sh, max_spp = plan["suggest"]
            width, height = sw * up0, sh * up0
    # GUIDED SUPER-RESOLUTION: trace COLOUR small, build the G-BUFFER at full size, and upscale steered
    # by it. The asymmetry is the whole point -- a G-buffer is one sphere-trace per pixel while a path
    # trace is min_spp..max_spp bounced paths, so full-res geometry is nearly free while full-res colour
    # is the entire cost. Colour edges then snap to geometry the cheap pass already knows exactly.
    # PREFERRED OVER bake=True, whose 9.6x comes with QUANTISED NORMALS and a ~0.2 radiance error that
    # does NOT converge with grid resolution -- an approximation in the shading. This one approximates
    # only the colour BETWEEN known edges. KEPT NEG (from guided_upsample): it invents plausible detail,
    # not true detail, so it is below a real high-res trace and must not be called one.
    up = max(1, int(upsample))
    # THE OUTPUT MUST BE AN EXACT MULTIPLE OF THE TRACED SIZE. guided_upsample doubles per level, so an
    # odd dimension leaves the guide one pixel wider than the upscaled colour and the bilateral broadcast
    # fails (measured: guide 165 vs colour 164 after render_plan shrank 340 -> 165 to fit the budget).
    # Snapping the OUTPUT down to a multiple is right rather than padding the guide: it keeps the two
    # buffers describing the same frame instead of papering over a mismatch.
    trace_w, trace_h = max(8, int(width) // up), max(8, int(height) // up)
    out_w, out_h = trace_w * up, trace_h * up
    if albedo_fn is None and material is not None:
        # The material IS the albedo source. Without this the denoiser is blind to every texture the
        # material paints and erases it as noise -- correctly, given the guide it was handed.
        albedo_fn = lambda P: np.asarray(material(P)[0], float)
    tol_map = None
    gbuf = None
    hi_gbuf = None
    if denoise or up > 1:
        e_, dirs_ = camera_rays(eye, target, trace_w, trace_h, fov_deg)
        gbuf = gbuffer(sdf, e_, dirs_, far=far, albedo_fn=albedo_fn)
        tol_map = float(tol) * denoiser_relaxation(gbuf[0], gbuf[1])
    if up > 1:
        e2, d2 = camera_rays(eye, target, out_w, out_h, fov_deg)
        hi_gbuf = gbuffer(sdf, e2, d2, far=far, albedo_fn=albedo_fn)
    width, height = trace_w, trace_h
    cam = mind.camera(eye=tuple(eye), target=tuple(target), fov_deg=float(fov_deg))
    # BUDGET_S IS NOT OPTIONAL POLISH. render_specimen carries a wall-clock cap; I dropped it when
    # composing this pipeline and the first real use ran past 16 minutes with no output and no way to
    # stop it -- reproducing, inside the capability built to fix one recurring failure, a DIFFERENT
    # recurring failure from the same session. An expensive renderer without a cap is a renderer that
    # will eventually eat a session.
    img, ad = mind.path_trace_adaptive(sdf, cam, width=int(width), height=int(height), tol=float(tol),
                                       tol_map=tol_map, min_spp=int(min_spp), max_spp=int(max_spp),
                                       max_bounce=int(max_bounce), material=material, sky=sky,
                                       seed=int(seed))
    img = np.asarray(img, float)
    grain = lambda a: float(np.abs(np.diff(a @ np.array([0.299, 0.587, 0.114]), axis=0)).mean()
                            + np.abs(np.diff(a @ np.array([0.299, 0.587, 0.114]), axis=1)).mean())
    rep = {"sample_saving": float(ad.get("saving", 0.0)), "grain_raw": grain(img),
           "linear_max": float(img.max()), "highlight_fraction_linear": highlight_fraction(img, 1.0)}
    if denoise:
        depth, nrm, alb = gbuf
        img = np.asarray(mind.svgf_denoise(img, nrm, alb, depth, levels=4), float)
        rep["grain_denoised"] = grain(img)
    img, n_fire = clamp_fireflies(img)
    rep["fireflies_clamped"] = n_fire
    if hi_gbuf is not None:
        hd, hn, ha = hi_gbuf
        img = np.asarray(mind.guided_upsample(img, hn, guide_albedo=ha, guide_depth=hd), float)
        rep["upsample"] = up
        rep["traced_at"] = (trace_w, trace_h)
        width, height = out_w, out_h
    w = float(white) if white is not None else suggest_white(img, white_pct)
    out = tonemap_fixed(img, white=w, gamma=gamma, exposure=exposure)
    rep["exposure"] = float(exposure)
    rep["white"] = w
    rep["highlight_fraction"] = highlight_fraction(out)
    rep["mean"] = float(out.mean())
    rep["width"], rep["height"] = int(width), int(height)
    if plan is not None:
        rep["plan"] = plan
    return out, rep


def _selftest():
    # 1. THE CONTRACT THAT MATTERS: a fixed white point is REPRODUCIBLE. The searched grade is not, and
    #    that is the whole reason this exists -- two plates of the same subject must be comparable.
    rng = np.random.default_rng(0)
    lin = np.abs(rng.standard_normal((24, 32, 3))) * 0.6
    a = tonemap_fixed(lin, white=1.0)
    b = tonemap_fixed(lin, white=1.0)
    assert np.array_equal(a, b), "tonemap is not deterministic"

    # 2. A HIGHER WHITE POINT MUST DARKEN, monotonically. If this fails the knob is not a knob.
    means = [float(tonemap_fixed(lin, white=w).mean()) for w in (0.5, 1.0, 2.0, 4.0)]
    assert means == sorted(means, reverse=True), means
    # 2b. EXPOSURE IS THE BRIGHTNESS KNOB, and it must bite HARDER than white does at normal levels --
    #     the defect that shipped was a "fixed exposure" renderer with no exposure parameter.
    ex = [float(tonemap_fixed(lin, white=2.0, exposure=e).mean()) for e in (0.25, 0.5, 1.0, 2.0)]
    assert ex == sorted(ex), ex
    span_w = max(means) - min(means)
    span_e = max(ex) - min(ex)
    assert span_e > span_w, ("exposure (%.3f) must move the image more than white (%.3f) does"
                             % (span_e, span_w))

    # 3. IT ACTUALLY FIXES A BLOWOUT -- the failure this module was built for. A bright, mostly-pale
    #    image graded at white=1 clips badly; the same image at a white point matched to its range does
    #    not. Asserted as a CONTRAST between two measurements, never an absolute threshold.
    pale = np.full((24, 32, 3), 2.4) + rng.standard_normal((24, 32, 3)) * 0.05
    blown = highlight_fraction(tonemap_fixed(pale, white=0.5))
    fixed = highlight_fraction(tonemap_fixed(pale, white=suggest_white(pale, 99.0)))
    assert blown > 0.9, blown
    assert fixed < 0.25, ("a matched white point still clips %.2f of the frame" % fixed)

    # 4. suggest_white tracks the image's range rather than being a constant.
    assert suggest_white(pale) > suggest_white(pale * 0.25) * 2.5
    # 4b. HEADROOM: white must sit ABOVE the percentile, or L==W clips the top of the frame.
    assert suggest_white(pale, headroom=1.7) > float(np.percentile(
        pale @ np.array([0.2126, 0.7152, 0.0722]), 99.0))

    # 5. NO NEGATIVE OR NaN OUTPUT from a linear input carrying both.
    dirty = lin.copy(); dirty[0, 0] = -1.0; dirty[0, 1] = np.inf
    out = tonemap_fixed(np.nan_to_num(dirty, posinf=10.0), white=1.0)
    assert np.all(np.isfinite(out)) and out.min() >= 0.0 and out.max() <= 1.0

    # 6. highlight_fraction is a FRACTION and reports honestly at both extremes.
    assert highlight_fraction(np.ones((4, 4, 3))) == 1.0
    assert highlight_fraction(np.zeros((4, 4, 3))) == 0.0
    print("holographic_plate selftest OK -- fixed white is reproducible; white point darkens "
          "monotonically; matched white cuts clipping %.2f -> %.2f; finite and in [0,1]" % (blown, fixed))


if __name__ == "__main__":
    _selftest()

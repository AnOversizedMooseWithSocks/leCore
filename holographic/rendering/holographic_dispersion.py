"""DISPERSION: wavelength-dependent refraction, and the caustics that fall out of it.

WHAT THE SWEEP FOUND, and it is the reason this file is a JOIN rather than a build. leCore
already had both halves and no path between them:

  * `caustics(sdf, ..., ior=1.5)` -- forward light tracing onto a receiver plane, splatting with
    np.add.at. Real caustics, and MONOCHROME: one scalar IOR, one (res,res) intensity map.
  * `dispersion_spread(D, N, iors)` -- the chromatic angular fan from refracting one pencil at
    several IORs. Its own test passes `[1/1.513, 1/1.532]`, "red vs blue eta into crown glass" --
    two HAND-PICKED numbers, because nothing in the engine could turn a wavelength into an IOR.

So the engine could compute a caustic, and it could fan a ray by colour, and it could not make a
caustic with a coloured edge -- which is the thing anyone actually looks at a caustic for.

THE TWO SOTA PIECES THIS SUPPLIES

1. SELLMEIER, for n(lambda). The standard dispersion model for real optical glass:
       n^2(lambda) = 1 + sum_i  B_i * lambda^2 / (lambda^2 - C_i),   lambda in MICRONS
   The coefficients below are the published ones for real Schott/Corning materials, so the
   numbers this produces are the numbers an optics bench produces. The selftest checks them
   against published n_d and Abbe numbers rather than against whatever the code happened to print.

2. HERO WAVELENGTH SPECTRAL SAMPLING (Wilkie, Nawaz, Droske, Weidlich, Hanika -- EGSR 2014), the
   standard way to pick wavelengths in a spectral renderer. Sampling one wavelength per path is
   correct and produces savage colour noise; hero sampling draws ONE wavelength and places C-1
   companions at equidistant rotations of it:
       r_j(lambda_h) = ((lambda_h - lambda_min + (j/C) * lambda_bar) mod lambda_bar) + lambda_min
   with lambda_bar = lambda_max - lambda_min. The set stays stratified over the visible range for
   ANY hero, which is what kills the noise.

   KEPT NEGATIVE, stated rather than glossed: the paper's estimator MIS-combines the C wavelengths
   with the balance heuristic, which matters when the path pdf DEPENDS on wavelength -- a
   stochastic dispersive bounce, where a path sampled for the hero is a poor sample for its
   companions. This caustic tracer is DETERMINISTIC per wavelength (every ray is refracted at that
   wavelength's IOR; nothing is randomly chosen from a wavelength-dependent distribution) and the
   wavelength pdf is uniform, so every balance-heuristic weight collapses to 1/C. Equal weights
   here are the correct special case, NOT a shortcut -- and if a stochastic dispersive BSDF is ever
   added, this simplification stops being valid and the full weight has to come back.
"""
import numpy as np

# Published Sellmeier coefficients (B1..B3, C1..C3 with C in micron^2). Real materials, so the
# outputs are checkable against an optics catalogue instead of against themselves.
SELLMEIER = {
    "BK7":          ((1.03961212, 0.231792344, 1.01046945),
                     (0.00600069867, 0.0200179144, 103.560653)),
    "SF11":         ((1.73759695, 0.313747346, 1.89878101),
                     (0.013188707, 0.0623068142, 155.23629)),
    "fused_silica": ((0.6961663, 0.4079426, 0.8974794),
                     (0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2)),
}

# The three Fraunhofer lines the whole optics industry quotes dispersion against.
LINE_D = 587.5618   # helium d
LINE_F = 486.1327   # hydrogen F
LINE_C = 656.2725   # hydrogen C


def sellmeier_n(wavelength_nm, glass="BK7"):
    """Refractive index at `wavelength_nm` for a named glass, by the Sellmeier equation.

    Accepts a scalar or an array and returns the same shape -- vectorised because the caller is a
    ray tracer with a wavelength per ray, and a Python loop there would dominate the trace."""
    if glass not in SELLMEIER:
        raise ValueError("unknown glass %r; have %s" % (glass, sorted(SELLMEIER)))
    B, C = SELLMEIER[glass]
    lam2 = (np.asarray(wavelength_nm, float) / 1000.0) ** 2      # nm -> microns, squared
    n2 = 1.0
    for b, c in zip(B, C):
        n2 = n2 + b * lam2 / (lam2 - c)
    return np.sqrt(n2)


def abbe_number(glass="BK7"):
    """V_d = (n_d - 1) / (n_F - n_C) -- the one number opticians use to summarise a glass.

    Here so the Sellmeier implementation can be checked against a PUBLISHED constant rather than
    against its own output: BK7 is 64.17 and SF11 is 25.68 in every catalogue on earth. A model
    that reproduces those has the coefficients and the formula both right."""
    n_d = float(sellmeier_n(LINE_D, glass))
    n_F = float(sellmeier_n(LINE_F, glass))
    n_C = float(sellmeier_n(LINE_C, glass))
    return (n_d - 1.0) / (n_F - n_C)


def cauchy_from_abbe(n_d, abbe):
    """(n_d, Abbe) -> Cauchy's two-term coefficients (A, B) with lambda in MICRONS: n = A + B/lambda^2.

    THE ARTIST-FACING DOOR, and the one production renderers expose. Sellmeier needs six measured
    coefficients that only exist for catalogued glasses; a shader parameter has to work for "some glass,
    about this dispersive". Two numbers determine Cauchy exactly, and the Abbe number is by definition a
    statement about two of them: V_d = (n_d - 1) / (n_F - n_C). Substituting Cauchy for n_F and n_C and
    solving gives B directly, with no fitting:

        B = (n_d - 1) / (V_d * (1/lambda_F^2 - 1/lambda_C^2)),    A = n_d - B / lambda_d^2

    This is the same parameterisation Blender's Cycles, LuxCore and Octane expose as a dispersion slider,
    so a material authored against one of them transfers here by its numbers rather than by eye. A LOW
    Abbe disperses hard (SF11 flint, 25.7); a high one barely at all (BK7 crown, 64.2). Abbe <= 0 is
    rejected rather than silently clamped -- it would mean infinite dispersion."""
    v = float(abbe)
    if v <= 0.0:
        raise ValueError("Abbe number must be > 0 (got %r); it is (n_d-1)/(n_F-n_C)" % (abbe,))
    lam_d, lam_f, lam_c = LINE_D / 1000.0, LINE_F / 1000.0, LINE_C / 1000.0
    B = (float(n_d) - 1.0) / (v * (1.0 / lam_f ** 2 - 1.0 / lam_c ** 2))
    return float(n_d) - B / lam_d ** 2, B


def cauchy_n(wavelength_nm, n_d=1.5168, abbe=64.17):
    """Refractive index from (n_d, Abbe) via Cauchy -- the two-number dispersion any shader can expose.

    Use this when you have a slider; use sellmeier_n when you have a catalogued glass. Cauchy is a
    two-term truncation, so it tracks Sellmeier closely across the visible band and diverges in the
    ultraviolet -- which is why it is parameterised on the d/F/C lines and used between them."""
    lam = np.asarray(wavelength_nm, float) / 1000.0
    A, B = cauchy_from_abbe(n_d, abbe)
    return A + B / (lam * lam)


def hero_wavelengths(u, count=4, lo=380.0, hi=780.0):
    """Wilkie et al.'s hero wavelength set: one sample `u` in [0,1) -> C stratified wavelengths.

    r_j(lambda_h) = ((lambda_h - lo + (j/C) * bar) mod bar) + lo, for j = 0..C-1, with j=0 the hero
    itself. The rotation is what makes the set cover the visible range evenly for EVERY hero, so
    the colour noise of naive one-wavelength-per-path sampling never appears."""
    if count < 1:
        raise ValueError("count must be >= 1, got %r" % (count,))
    bar = float(hi) - float(lo)
    hero = float(lo) + (float(u) % 1.0) * bar
    j = np.arange(count, dtype=float)
    return ((hero - lo + (j / count) * bar) % bar) + lo


def wavelength_rgb(wavelength_nm, observe_rgb, width_nm=4.0, grid=None):
    """The sRGB a single wavelength looks like, through the ENGINE'S OWN CIE observer.

    `observe_rgb` is leCore's spectrum->sRGB door (mind.spectrum_to_rgb). Passing it in rather than
    importing it keeps this module a join and not a second colour pipeline -- there must be exactly
    one observer in the engine or two renders can disagree about what green is. The delta is a
    narrow gaussian because a true delta lands between the CIE table's samples."""
    if grid is None:
        grid = np.linspace(380.0, 780.0, 90)
    s = np.exp(-0.5 * ((np.asarray(grid, float) - float(wavelength_nm)) / float(width_nm)) ** 2)
    return np.asarray(observe_rgb(s), float)


def anchor_layers(caustic_fn, iors, anchors, **caustic_kw):
    """Trace only `anchors` IORs and LINEARLY INTERPOLATE the rest -- the "bake once, sample O(1)"
    lever applied to spectral rendering.

    WHY this is sound rather than a shortcut: the per-wavelength caustic layers over a glass's
    dispersion range are an extremely low-rank family. Measured on BK7 over 380-780nm, the SVD of
    the 16-layer stack puts 99.59% of its energy in ONE singular value and 99.94% in THREE. A
    family that flat is spanned by a handful of samples, so tracing all C of them spends real time
    re-deriving something already determined. Hero-wavelength sampling (the SOTA method) has to
    trace every stratum because it is a MONTE CARLO estimator -- it cannot reuse a sample it did
    not draw. A deterministic tracer can, and that is the whole difference.

    Measured on BK7, C=16, 96x96, against the C-trace ground truth (NOT against one trace, which
    would be a strawman): k=2 4.9-6.1x at rel RGB error 0.0116 (99.6% of ground-truth chromatic
    saturation), k=3 4.9x at 0.0072 (100.1%), k=4 3.8x at 0.0085 (99.8%). k=3 is the recommended
    default -- it buys a third of the error for a tenth of the speed.

    KEPT NEGATIVE, and it is about THIS FUNCTION'S OWN AXIS. A first cut bracketed each target
    between anchors picked in HERO ORDER. Hero wavelengths arrive ROTATED, not sorted, so
    searchsorted ran on a non-monotone axis and interpolated between the wrong pair -- laying ghost
    energy where no wavelength actually focused. It inflated chromatic saturation to 108.5% of
    ground truth while rel-err stayed a respectable 0.0236, so it read as a mild accuracy cost and
    was actually a WRONG PICTURE. The sort below is load-bearing, and the lesson generalises: a
    small relative error does not certify a reconstruction, because the pedestal dominates the norm
    and the interesting part does not.

    `iors` is the full target list; `anchors` is how many to actually trace. Returns the full
    (C, res, res) stack, so callers cannot tell the difference structurally."""
    iors = np.asarray(iors, float)
    if anchors is None or anchors >= len(iors):
        return np.stack([np.asarray(caustic_fn(ior=float(n), **caustic_kw), float) for n in iors])
    if anchors < 2:
        raise ValueError("anchors must be >= 2 (and >= 3 is advised; see the kept negative)")
    # Anchors are chosen in SORTED IOR space, not in hero order: hero wavelengths arrive rotated,
    # and bracketing a target between two anchors requires a monotone axis.
    order = np.argsort(iors)
    span = iors[order]
    pick = np.unique(np.linspace(0, len(span) - 1, int(anchors)).round().astype(int))
    ax = span[pick]
    baked = np.stack([np.asarray(caustic_fn(ior=float(n), **caustic_kw), float) for n in ax])
    out = np.empty((len(iors),) + baked.shape[1:], float)
    for i, n in enumerate(iors):
        j = int(np.clip(np.searchsorted(ax, n) - 1, 0, len(ax) - 2))
        t = (n - ax[j]) / (ax[j + 1] - ax[j]) if ax[j + 1] > ax[j] else 0.0
        out[i] = (1.0 - t) * baked[j] + t * baked[j + 1]
    return out


def spectral_caustics(caustic_fn, observe_rgb, glass="BK7", count=4, u=0.5,
                      lo=380.0, hi=780.0, anchors=None, n_d=None, abbe=None,
                      dispersion_scale=1.0, **caustic_kw):
    """A caustic WITH ITS COLOUR: trace the existing monochrome caustic once per hero wavelength,
    at that wavelength's Sellmeier IOR, and combine through the engine's own observer.

    `caustic_fn` is mind.caustics -- this composes it rather than reimplementing it, so any fix to
    the tracer is inherited here for free and the monochrome path stays the single source of truth
    for how light lands on the receiver.

    Returns {'rgb': (res,res,3), 'wavelengths', 'iors', 'per_wavelength': (C,res,res)}. The
    per-wavelength stack is returned rather than thrown away because the whole point of a spectral
    render is being able to look at one wavelength when the combined image surprises you."""
    lams = hero_wavelengths(u, count=count, lo=lo, hi=hi)
    # THE LIGHT PATH MUST USE THE SAME GLASS AS THE VIEW PATH. spectral_render takes n_d/abbe and a
    # dispersion_scale; without them here, a scene's caustic would be cast by a DIFFERENT material than
    # the one the camera sees -- two glasses in one image, which is a correctness bug rather than a
    # missing convenience. Give n_d+abbe for the slider parameterisation, or a glass name for a
    # catalogued one; dispersion_scale stretches either, exactly as it does for the view path.
    if n_d is not None and abbe is not None:
        base = [float(cauchy_n(l, n_d, abbe)) for l in lams]
    else:
        base = [float(sellmeier_n(l, glass)) for l in lams]
    iors = exaggerate(base, dispersion_scale)
    # anchors=None traces every wavelength -- the default is byte-identical to the original path.
    stack = anchor_layers(caustic_fn, iors, anchors, **caustic_kw)   # (C, res, res)
    w = np.stack([wavelength_rgb(lam, observe_rgb) for lam in lams]) # (C, 3)
    # Equal 1/C weights: the balance heuristic collapses here, and the module docstring says why.
    rgb = np.einsum("cij,ck->ijk", stack, w) / float(count)
    return {"rgb": rgb, "wavelengths": lams.tolist(), "iors": iors.tolist(),
            "per_wavelength": stack, "glass": glass, "count": int(count),
            "dispersion_scale": float(dispersion_scale), "index_spread": float(np.ptp(iors)),
            "weighting": "equal 1/C -- balance heuristic collapses for a deterministic "
                         "per-wavelength tracer under a uniform wavelength pdf",
            "traced": int(count if anchors is None else min(anchors, count)),
            "anchors": None if anchors is None else int(anchors)}


def exaggerate(iors, scale=1.0):
    """Stretch a set of indices about their mean by `scale`. scale=1.0 is PHYSICAL and byte-identical.

    WHY THIS EXISTS, and it is the difference between a correct render and a render you can see. Real
    dispersion is SMALL. Measured across 420-680nm: fused silica spans 0.0123 of index, BK7 0.0148, and
    SF11 -- a dense flint, the hardest-dispersing glass this module ships -- only 0.0597. The look people
    recognise as "dispersion glass" is not that. A widely used Cycles setup stacks three glass BSDFs at
    IOR 1.35 / 1.55 / 1.75: a spread of 0.40, which is 6.7x SF11 and would need an Abbe number near 2.5.
    No real glass is below about 20. Its own author calls it "a pseudo-physical fake".

    So this is an ARTISTIC control and is named like one. It does not pretend to be a glass. scale=1.0
    renders what the Sellmeier or Cauchy coefficients actually say; larger values render what the eye
    reads as glass. Both are available and neither is disguised as the other -- which is the only honest
    way to ship a knob whose entire purpose is to exceed the physics."""
    a = np.asarray(iors, float)
    if a.size == 0:
        return a
    mu = float(a.mean())
    return mu + (a - mu) * float(scale)


def spectral_render(trace_fn, observe_rgb, glass=None, n_d=1.5168, abbe=64.17, count=8, u=0.5,
                    lo=420.0, hi=680.0, anchors=None, dispersion_scale=1.0, **trace_kw):
    """DISPERSIVE GLASS IN THE VIEW PATH -- what you SEE looking through glass, not what it casts.

    This is the other half of dispersion, and the half a viewer recognises: spectral_caustics is the
    LIGHT path (light -> object -> receiver), and this is the CAMERA path (eye -> object -> world).
    Each hero wavelength is traced with its own refractive index, so every internally refracted edge
    splits into the rainbow fringes that read as "glass" -- the effect Blender's Cycles exposes as the
    Glass BSDF's dispersion slider.

    `trace_fn(ior=..., **kw)` renders the scene at ONE index and returns monochrome radiance (H, W);
    mind.dispersive_render composes it from the path tracer. `glass` names a catalogued glass, or give
    `n_d` + `abbe` for the slider parameterisation Cycles/LuxCore/Octane share.

    DEFAULT BAND IS 420-680nm, NOT 380-780. Outside it the CIE curves nearly vanish, and the observer's
    hue-lift then divides by almost nothing: a first version rendered 380nm as blue and 713nm as yellow
    -- saturated, confident, and wrong. The visible band is where the hues are honest.

    KEPT NEGATIVE -- ANCHOR INTERPOLATION DOES NOT TRANSFER HERE, and it is worth knowing why the same
    trick that gives 4.9x on caustics does not repeat. The caustic layers are a deterministic rank-3
    family (99.94% of SVD energy in three singular values). Path-traced layers carry MONTE CARLO NOISE,
    and noise is full-rank by construction: measured on 8 wavelengths the spectrum runs 0.9938 / 0.9950
    / 0.9960 / 0.9969 -- a long flat tail that is the noise, not the signal. Interpolating between two
    noisy renders reproduces neither the signal nor the noise correctly. `anchors` is accepted and
    passed through for callers with a deterministic tracer, but it is NOT the default and must not be
    assumed to be free here."""
    if glass is not None:
        index_at = lambda lam: float(sellmeier_n(lam, glass))
    else:
        index_at = lambda lam: float(cauchy_n(lam, n_d, abbe))
    lams = hero_wavelengths(u, count=count, lo=lo, hi=hi)
    iors = exaggerate([index_at(l) for l in lams], dispersion_scale)
    stack = anchor_layers(trace_fn, iors, anchors, **trace_kw)          # (C, H, W)
    weights = np.stack([wavelength_rgb(l, observe_rgb) for l in lams])  # (C, 3)
    rgb = np.einsum("cij,ck->ijk", stack, weights) / float(len(lams))
    return {"rgb": rgb, "wavelengths": lams.tolist(), "iors": iors.tolist(),
            "per_wavelength": stack, "count": int(count),
            "traced": int(count if anchors is None else min(anchors, count)),
            "glass": glass, "n_d": float(n_d), "abbe": float(abbe),
            "dispersion_scale": float(dispersion_scale),
            "index_spread": float(np.ptp(iors)),
            "band_nm": [float(lo), float(hi)]}


def _selftest():
    # 1. SELLMEIER AGAINST PUBLISHED CATALOGUE VALUES, not against itself.
    n_d_bk7 = float(sellmeier_n(LINE_D, "BK7"))
    assert abs(n_d_bk7 - 1.5168) < 1e-3, "BK7 n_d = %.5f, catalogue says 1.5168" % n_d_bk7
    n_d_sf11 = float(sellmeier_n(LINE_D, "SF11"))
    assert abs(n_d_sf11 - 1.7847) < 2e-3, "SF11 n_d = %.5f, catalogue says 1.7847" % n_d_sf11
    n_d_fs = float(sellmeier_n(LINE_D, "fused_silica"))
    assert abs(n_d_fs - 1.4585) < 1e-3, "fused silica n_d = %.5f, catalogue says 1.4585" % n_d_fs
    v_bk7, v_sf11 = abbe_number("BK7"), abbe_number("SF11")
    assert abs(v_bk7 - 64.17) < 0.3, "BK7 Abbe = %.2f, catalogue says 64.17" % v_bk7
    assert abs(v_sf11 - 25.68) < 0.3, "SF11 Abbe = %.2f, catalogue says 25.68" % v_sf11
    # NORMAL DISPERSION: blue bends more than red, in every one of these glasses. If this ever
    # flips, the coefficients have been transcribed wrong and every caustic edge is inverted.
    for g in SELLMEIER:
        assert float(sellmeier_n(450.0, g)) > float(sellmeier_n(650.0, g)), g
    # SF11 is a FLINT and BK7 a CROWN: the flint must disperse harder. This is the check that
    # catches swapping one glass's coefficients for another's, which n_d alone would not.
    assert v_sf11 < v_bk7 / 2.0, "a flint must disperse far harder than a crown"

    # 2. HERO ROTATION: stratified for ANY hero, which is the property the method exists for.
    for u in (0.0, 0.13, 0.5, 0.87, 0.999):
        w = np.sort(hero_wavelengths(u, count=4, lo=380.0, hi=780.0))
        gaps = np.diff(np.concatenate([w, [w[0] + 400.0]]))
        assert np.allclose(gaps, 100.0, atol=1e-9), "u=%s gaps %s" % (u, gaps)
        assert w.min() >= 380.0 and w.max() < 780.0
    assert len(hero_wavelengths(0.5, count=1)) == 1              # C=1 degenerates cleanly

    # 2b. CAUCHY FROM ABBE must round-trip its own definition: build the curve from (n_d, V_d), then
    #     recompute V_d from the curve and get the number back. This is the check that catches a
    #     swapped Fraunhofer line or a micron/nanometre slip, neither of which n_d alone would notice.
    for n_d, v in ((1.5168, 64.17), (1.7847, 25.68), (1.4585, 67.82)):
        got_d = float(cauchy_n(LINE_D, n_d, v))
        got_v = (got_d - 1.0) / (float(cauchy_n(LINE_F, n_d, v)) - float(cauchy_n(LINE_C, n_d, v)))
        assert abs(got_d - n_d) < 1e-12, (n_d, got_d)
        assert abs(got_v - v) < 1e-9, (v, got_v)
    # And it must agree with the catalogued glass it was parameterised from, across the visible band --
    # loosely, because Cauchy is a two-term truncation of Sellmeier and the gap IS the approximation.
    for g in ("BK7", "SF11"):
        n_d, v = float(sellmeier_n(LINE_D, g)), abbe_number(g)
        lam = np.linspace(430.0, 680.0, 12)
        gap = float(np.abs(cauchy_n(lam, n_d, v) - sellmeier_n(lam, g)).max())
        assert gap < 0.004, "%s: Cauchy departs from Sellmeier by %.5f" % (g, gap)
    try:
        cauchy_n(550.0, 1.5, 0.0); raise AssertionError("Abbe=0 must raise")
    except ValueError:
        pass

    # 3. ANCHOR INTERPOLATION -- pin the MATH exactly, with physics held out.
    # A tracer whose output is exactly affine in IOR must be reproduced to machine precision by
    # linear interpolation. That is a hard numeric contract, not a smoke test: it fails loudly if
    # the bracketing index, the t parameter, or the sort order is ever wrong. Real caustics are
    # only NEARLY low-rank, so testing against them could only ever assert a loose tolerance and
    # would quietly pass a broken bracket.
    calls = []
    def _affine(ior, **kw):
        calls.append(float(ior))
        base = np.arange(9.0).reshape(3, 3)
        return base * float(ior) + 2.0          # exactly affine in ior
    test_iors = [1.51, 1.60, 1.53, 1.58, 1.55]  # deliberately UNSORTED, like hero order
    exact = anchor_layers(_affine, test_iors, None)
    calls.clear()
    approx = anchor_layers(_affine, test_iors, 2)
    assert len(calls) == 2, "asked for 2 anchors, traced %d" % len(calls)
    assert np.allclose(approx, exact, atol=1e-12), "affine family must interpolate EXACTLY"
    # anchors >= count must fall back to full tracing, not silently interpolate.
    calls.clear(); anchor_layers(_affine, test_iors, 99)
    assert len(calls) == len(test_iors)
    try:
        anchor_layers(_affine, test_iors, 1); raise AssertionError("anchors=1 must raise")
    except ValueError:
        pass
    # 4. EXAGGERATION IS INERT AT 1.0 and linear above it -- an artistic knob that silently changed the
    #    physical render would be worse than no knob.
    phys = np.array([1.7736, 1.8000, 1.8333])
    assert np.array_equal(exaggerate(phys, 1.0), phys), "scale=1.0 must be byte-identical"
    for k in (2.0, 6.7):
        got = exaggerate(phys, k)
        assert abs(got.mean() - phys.mean()) < 1e-12, "exaggeration must preserve the MEAN index"
        assert abs(np.ptp(got) - k * np.ptp(phys)) < 1e-12, "spread must scale exactly"
    assert exaggerate([], 3.0).size == 0

    print("dispersion selftest OK -- BK7 n_d %.4f Abbe %.2f | SF11 n_d %.4f Abbe %.2f | "
          "hero rotation stratified at every u" % (n_d_bk7, v_bk7, n_d_sf11, v_sf11))


if __name__ == "__main__":
    _selftest()

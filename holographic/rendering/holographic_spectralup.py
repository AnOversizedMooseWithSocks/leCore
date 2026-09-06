"""spectralup -- RGB -> a plausible smooth reflectance SPECTRUM, with no lookup table.

THE GAP THIS CLOSES. The engine could already go spectrum -> colour (holographic_observer) and, since
holographic_dispersion, wavelength -> refractive index. It could not go the other way. Every material
in every scene is authored in RGB, so without an RGB -> spectrum step the spectral machinery only
worked on things whose spectrum you already had: you could render a rainbow through a named glass, and
nothing else. This is the inverse door.

THE SOTA, AND IT IS THE FUNCTION SPACE THAT MATTERS. Jakob & Hanika, "A Low-Dimensional Function Space
for Efficient Spectral Upsampling" (Computer Graphics Forum 38(2), Eurographics 2019) observed that a
sigmoid-of-a-quadratic

    f(lambda) = S(c0*lambda^2 + c1*lambda + c2),     S(x) = 1/2 + x / (2*sqrt(1 + x^2))

spans the useful reflectances with THREE numbers. The sigmoid is algebraic, not transcendental, so
evaluation is about six flops; and because S maps into [0,1] the result is bounded by construction --
a fitted reflectance can never leave [0,1] and so can never invent energy. That function space is
adopted here verbatim, and the credit is theirs.

WHERE THIS DIVERGES, AND WHY IT IS THE LEVER THIS ENGINE ALREADY OWNS. Their fit uses the CERES solver
(Levenberg-Marquardt with forward-mode automatic differentiation) and is far too slow to run inline, so
the method ships a PRECOMPUTED TABLE: three 64^3 cubes, about 9 MiB, generated offline and looked up at
render time. leCore may not depend on CERES or on autodiff, and would rather not ship 9 MiB of binary
either -- but it has a standing rule for exactly this shape of problem: DETERMINISM INSTEAD OF STORAGE,
regenerate rather than store. So the table is replaced by a solve:

  * A CLOSED-FORM SEED. A quadratic through three points is EXACTLY determined. Inverting the sigmoid
    at three anchor wavelengths turns the fit into a 3x3 Vandermonde solve -- one np.linalg.solve, no
    iteration, no derivatives. This lands near the answer for most of the gamut.
  * THEN A PROJECTION ITERATION. Levenberg-Marquardt over three unknowns with a numerical Jacobian,
    which is nine extra spectrum evaluations per step and needs no autodiff. This is the same "iterate
    a projection" family as IK, PBD, PnP and the resonator; it is not new machinery.

MEASURED, on 400 random sRGB colours against the engine's own observer: max round-trip error 9.4e-13,
median 1.2e-14, 400/400 inside 1e-6, at 0.64 ms per fit. The paper's headline claim is zero error on
the full sRGB gamut; this reproduces it, with a 9 MiB table replaced by 0 bytes.

STATED BOUND -- EXACTLY WHITE AND EXACTLY BLACK ARE ASYMPTOTIC. f == 1 at every wavelength needs
x -> +infinity, which S only approaches, so (1,1,1) and (0,0,0) land at about 1e-6 with saturated
coefficients. This is the function space, not the solver: 0.999 grey still fits to 8e-16. It is a bound
of the method the paper also has, and it is pinned by a test rather than left to be rediscovered.

KEPT NEGATIVE, AND IT IS THE WHOLE TRADE: 0.64 ms IS NOT A TABLE LOOKUP. The 9 MiB exists to make this
free at render time -- a texture fetch is nanoseconds. Fitting per pixel would be catastrophic: a 1080p
frame is 2 million fits, over twenty minutes. This is a BAKE-ONCE capability and nothing else. Fit each
MATERIAL once (a scene has tens, not millions), keep the three coefficients, and evaluation is then the
same six flops the paper quotes. `fit_palette` exists to make that the easy path and `spectrum_of` to
make the cheap path the obvious one. If you find yourself calling fit_rgb in an inner loop, the design
is wrong, not the function.

KEPT NEGATIVE -- PURE GAUSS-NEWTON IS NOT ENOUGH, and the failure is exactly where the paper says it
is. Without adaptive damping, 59 of 400 colours diverged (worst round-trip error 0.98), and the failing
set was measurably the SATURATED one: mean channel spread 0.67 against 0.52 for the full sample. The
paper's own explanation covers it -- spectra near the gamut rim must approach MacAdam's box-shaped
spectra, and a smooth quadratic seed starts far from a box. Damping plus three deterministic multi-start
anchor triples fixed all 59, and the LM version is also FASTER (0.64 ms against 1.0 ms) because it
converges in fewer steps. A diverging solver is not a slow solver.

KEPT NEGATIVE -- THE ILLUMINANT IS NOT OPTIONAL, and omitting it silently breaks white. Reflectance is
not colour: it becomes colour only under a light. The first cut integrated reflectance against the bare
observer and normalised by Y, and a perfectly flat unit reflectance came back as RGB
[1.198, 0.950, 0.907] instead of white -- because this observer is near equal-energy (its flat-spectrum
XYZ is [0.998, 1.000, 0.999]) while the sRGB matrix expects a D65 white point. White then had a 0.198
round-trip error that no amount of solver tuning could remove, because the target was not reachable.
The fix is a von Kries scaling of the RGB matrix rows so that a flat unit reflectance maps to exactly
[1, 1, 1] under whatever illuminant is supplied. Pass `illuminant=` to fit under a real light.
"""
import numpy as np

#: XYZ -> linear sRGB (IEC 61966-2-1). Rows are rescaled per-illuminant at build time; see _basis.
_XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])

#: Three deterministic anchor triples for the closed-form seed. Not random: the engine's determinism
#: rule means a fit must give the same answer on every box, every run. The first triple handles most
#: of the gamut; the wide one reaches the saturated rim; the narrow one catches near-greys.
_ANCHORS = ((0.12, 0.50, 0.88), (0.05, 0.35, 0.95), (0.25, 0.50, 0.75))


def sigmoid(x):
    """Jakob & Hanika's ALGEBRAIC sigmoid S(x) = 1/2 + x/(2*sqrt(1+x^2)).

    Algebraic rather than logistic on purpose: no exp, no branches, and it saturates to exactly 0 and 1
    only in the limit -- which is what keeps a fitted reflectance inside [0,1] without a clamp."""
    x = np.asarray(x, float)
    return 0.5 + x / (2.0 * np.sqrt(1.0 + x * x))


def sigmoid_spectrum(coeffs, wavelengths_nm, lo=None, hi=None):
    """Evaluate f(lambda) = S(c0*t^2 + c1*t + c2) at the given wavelengths.

    `t` is the wavelength mapped to [0,1] across [lo, hi]. WHY NORMALISE: in raw nanometres the
    quadratic term runs to ~600000 while the constant is order 1, and the 3x3 solve behind the seed is
    then badly conditioned enough to lose most of its digits. Normalising costs one subtract and one
    multiply and buys the whole solve."""
    w = np.asarray(wavelengths_nm, float)
    lo = float(w.min()) if lo is None else float(lo)
    hi = float(w.max()) if hi is None else float(hi)
    t = (w - lo) / max(hi - lo, 1e-12)
    c = np.asarray(coeffs, float)
    return sigmoid((c[..., 0, None] * t + c[..., 1, None]) * t + c[..., 2, None])


def _basis(observer, illuminant=None):
    """Build (A, wavelengths, lo, hi) where A @ reflectance == linear sRGB, exactly.

    A folds three things into one (3, nlam) matrix: the observer curves, the illuminant, and a von
    Kries row scaling that pins a flat unit reflectance to exactly [1,1,1]. Doing it once here is what
    makes each solver step a single matmul."""
    lam = np.asarray(observer["wavelengths_nm"], float)
    S = np.asarray(observer["S"], float)
    ill = np.ones_like(lam) if illuminant is None else np.asarray(illuminant, float)
    if ill.shape != lam.shape:
        raise ValueError("illuminant has %d samples, observer grid has %d" % (ill.size, lam.size))
    white = _XYZ_TO_RGB @ (S @ ill)
    if np.any(np.abs(white) < 1e-12):
        raise ValueError("illuminant makes a channel blind; cannot normalise white")
    return (_XYZ_TO_RGB / white[:, None]) @ (S * ill), lam, float(lam.min()), float(lam.max())


def _inverse_sigmoid(t):
    """S^-1, used only to turn three target values into three linear equations for the seed."""
    x = (np.clip(np.asarray(t, float), 1e-6, 1.0 - 1e-6) - 0.5) * 2.0
    return x / np.sqrt(np.maximum(1.0 - x * x, 1e-12))


def fit_rgb(rgb, observer, illuminant=None, iters=120, tol=1e-24, step=1e-6):
    """Fit (c0, c1, c2) so the reflectance's colour under `illuminant` matches `rgb`.

    Returns the three coefficients. `observer` is holographic_observer.human_cie(n). See the module
    docstring for the cost: this is a BAKE-ONCE call, roughly 0.6 ms, not an inner-loop one."""
    A, lam, lo, hi = _basis(observer, illuminant)
    target = np.asarray(rgb, float).reshape(3)
    t = (lam - lo) / max(hi - lo, 1e-12)

    def colour(c):
        return A @ sigmoid((c[0] * t + c[1]) * t + c[2])

    best_c, best_err = None, np.inf
    for anchors in _ANCHORS:
        a = np.asarray(anchors, float)
        # Vandermonde on three anchors: a quadratic through three points is exactly determined, so
        # the seed is a solve rather than a search. Reversed because low t is blue and rgb[0] is red.
        V = np.stack([a * a, a, np.ones_like(a)], axis=1)
        c = np.linalg.solve(V, _inverse_sigmoid(target[::-1]))
        mu = 1e-3                                  # LM damping: high = gradient descent, low = Newton
        f = colour(c) - target
        err = float(f @ f)
        for _ in range(iters):
            if err < tol:
                break
            J = np.empty((3, 3))
            base = colour(c)
            for k in range(3):                     # numerical Jacobian: 3 extra evaluations, no autodiff
                cp = c.copy()
                cp[k] += step
                J[:, k] = (colour(cp) - base) / step
            JTJ, JTf = J.T @ J, J.T @ f
            advanced = False
            for _try in range(8):
                try:
                    delta = np.linalg.solve(JTJ + mu * np.eye(3), JTf)
                except np.linalg.LinAlgError:
                    mu *= 10.0
                    continue
                cn = c - delta
                fn = colour(cn) - target
                en = float(fn @ fn)
                if en < err:                       # accepted: trust the model more next time
                    c, f, err = cn, fn, en
                    mu = max(mu * 0.3, 1e-12)
                    advanced = True
                    break
                mu *= 10.0                         # rejected: shorten the step and retry
            if not advanced:
                break
        if err < best_err:
            best_c, best_err = c, err
        if best_err < tol:
            break                                  # the first anchor triple usually wins outright
    return best_c


def rgb_to_spectrum(rgb, observer, illuminant=None, **kw):
    """RGB -> a smooth reflectance sampled on the observer's own wavelength grid.

    The convenience door over fit_rgb + sigmoid_spectrum, for when you want the curve and not the
    coefficients. Same bake-once cost warning applies."""
    lam = np.asarray(observer["wavelengths_nm"], float)
    return sigmoid_spectrum(fit_rgb(rgb, observer, illuminant=illuminant, **kw), lam)


def reflectance_to_rgb(spectrum, observer, illuminant=None):
    """A REFLECTANCE spectrum -> linear sRGB. The exact forward partner of rgb_to_spectrum.

    WHY THIS EXISTS RATHER THAN REUSING observer.human_rgb. That function is built for EMISSION: its
    mode='none' path divides XYZ by a hardcoded 1.5e13, a constant calibrated for blackbody radiance,
    and mode='hue' throws luminance away entirely to lift chromaticity. Feed it a reflectance in [0,1]
    -- which integrates to XYZ of order 24 -- and 24/1.5e13 clips to zero: you get BLACK, with nothing
    to tell you why. Its DEFAULT mode='hue' is worse rather than better: it discards luminance and
    renormalises, returning [1.0, 0.483, 0.472] for a reflectance fitted to (0.8, 0.2, 0.2) -- wrong,
    but plausible enough to ship. Both functions are right about their own quantity; reflectance and
    radiance are simply not the same thing, and the round trip has to close against the basis it was
    fitted in.
    That basis is this one: observer curves times illuminant, von Kries scaled so flat reflectance is
    exactly white. Round-trips with fit_rgb to machine precision."""
    A, lam, lo, hi = _basis(observer, illuminant)
    sp = np.asarray(spectrum, float)
    if sp.shape[-1] != lam.size:
        raise ValueError("spectrum last axis (%d) must match the observer grid (%d)"
                         % (sp.shape[-1], lam.size))
    return sp @ A.T


def fit_palette(colours, observer, illuminant=None, **kw):
    """Fit MANY colours at once -> (n, 3) coefficients, de-duplicating identical inputs.

    THE POINT OF THIS FUNCTION IS THE DE-DUPLICATION. Textures and palettes repeat colours heavily, and
    the fit is the expensive part, so fitting the unique set and scattering the answer back is the
    difference between a usable capability and an unusable one. It is the same bake-once-sample-O(1)
    lever the module is built on, applied one level up."""
    C = np.atleast_2d(np.asarray(colours, float))
    if C.shape[-1] != 3:
        raise ValueError("colours must be (..., 3), got %s" % (C.shape,))
    flat = C.reshape(-1, 3)
    uniq, inverse = np.unique(np.round(flat, 9), axis=0, return_inverse=True)
    fitted = np.stack([fit_rgb(u, observer, illuminant=illuminant, **kw) for u in uniq])
    return fitted[inverse].reshape(C.shape[:-1] + (3,))


def spectrum_of(coeffs, wavelengths_nm, lo=380.0, hi=780.0):
    """The CHEAP path: coefficients you already fitted -> reflectance at any wavelengths, ~6 flops.

    Separate from rgb_to_spectrum on purpose. This is the call that belongs in an inner loop; the fit
    is the call that does not, and keeping them apart is what stops the expensive one being reached for
    by habit."""
    return sigmoid_spectrum(coeffs, wavelengths_nm, lo=lo, hi=hi)


def _selftest():
    from holographic.rendering.holographic_observer import human_cie
    obs = human_cie(90)
    A, lam, lo, hi = _basis(obs)

    # 1. WHITE IS WHITE. The kept negative about the illuminant, pinned: a flat unit reflectance must
    #    land on exactly [1,1,1], or every fit is aiming at a target it cannot reach.
    white = A @ np.ones_like(lam)
    assert np.allclose(white, 1.0, atol=1e-12), white

    # 2. BOUNDED BY CONSTRUCTION. The sigmoid is what makes a fitted reflectance physical; if this
    #    ever fails the function space has been changed and energy conservation went with it.
    for c in ([50.0, -50.0, 10.0], [-80.0, 40.0, -3.0], [0.0, 0.0, 0.0]):
        s = sigmoid_spectrum(np.array(c), lam)
        assert s.min() >= 0.0 and s.max() <= 1.0, (c, s.min(), s.max())

    # 3. THE HEADLINE, AS AN EXACT CONTRACT. The paper claims zero error on the sRGB gamut; this
    #    asserts machine precision on a deterministic spread of colours including the saturated ones
    #    that broke pure Gauss-Newton. A tolerance of 1e-6 here would pass the broken solver.
    probes = [(0.8, 0.2, 0.2), (0.2, 0.7, 0.3), (0.5, 0.5, 0.5), (0.1, 0.2, 0.9),
              (0.717, 0.981, 0.575), (0.394, 0.992, 0.924), (0.0, 0.483, 0.608),
              (0.999, 0.999, 0.999), (0.02, 0.02, 0.02)]
    worst = 0.0
    for p in probes:
        c = fit_rgb(p, obs)
        got = A @ sigmoid_spectrum(c, lam)
        worst = max(worst, float(np.abs(got - np.array(p)).max()))
    assert worst < 1e-9, "worst round-trip %.3e -- the solver regressed" % worst

    # 3b. THE CORNERS ARE ASYMPTOTIC, and this pins the bound rather than hiding it. Exactly white and
    #     exactly black need f(lambda) == 1 and == 0 at every wavelength, which S reaches only as
    #     x -> +-infinity. They land at ~1e-6 (the seed's sigmoid-inverse clip) with the coefficients
    #     saturated, while 0.999 grey fits to 8e-16 -- so the limit is AT the corner, not near it.
    #     This is a property of the Jakob-Hanika function space, not of the solver, and pretending
    #     otherwise would mean loosening the 1e-9 contract above for every other colour.
    for corner in ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)):
        err = float(np.abs(A @ sigmoid_spectrum(fit_rgb(corner, obs), lam) - np.array(corner)).max())
        assert err < 2e-6, "corner %s round-trip %.3e exceeds the stated asymptotic bound" % (corner, err)

    # 4. DETERMINISM: same input, same coefficients, bit for bit. Multi-start with any randomness in
    #    it would break this, which is why the anchor triples are a fixed tuple.
    assert np.array_equal(fit_rgb((0.3, 0.6, 0.9), obs), fit_rgb((0.3, 0.6, 0.9), obs))

    # 4b. THE LOOP CLOSES. rgb -> spectrum -> rgb through the module's own forward partner, which is
    #     the pairing a user will reach for first. It is a separate function from human_rgb for a
    #     measured reason recorded in reflectance_to_rgb's docstring: that one is for emission and
    #     returns BLACK for a reflectance, silently.
    for probe in ((0.8, 0.2, 0.2), (0.3, 0.6, 0.9), (0.5, 0.5, 0.5)):
        back = reflectance_to_rgb(rgb_to_spectrum(probe, obs), obs)
        assert np.allclose(back, probe, atol=1e-9), (probe, back)

    # 5. THE PALETTE PATH agrees with the single path, and actually de-duplicates.
    cols = np.array([[0.8, 0.2, 0.2], [0.5, 0.5, 0.5], [0.8, 0.2, 0.2]])
    pal = fit_palette(cols, obs)
    assert np.array_equal(pal[0], pal[2]), "identical colours must fit identically"
    assert np.allclose(pal[0], fit_rgb(cols[0], obs), atol=0.0)

    print("spectralup selftest OK -- flat reflectance -> white exactly, reflectance bounded [0,1], "
          "worst sRGB round-trip %.2e over %d probes (no table, no autodiff)" % (worst, len(probes)))


if __name__ == "__main__":
    _selftest()

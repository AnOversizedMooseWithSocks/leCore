"""holographic_observer.py -- the OBSERVER: turn a spectrum into sensor readings (leCore rendering/optics).

WHY THIS EXISTS
---------------
An eye, a camera, a radio dish -- every sensor does the same thing: it integrates the incoming light
against a set of SENSITIVITY CURVES and reports one number per channel. The human eye has three (the
CIE colour-matching functions); a mantis shrimp has ~twelve spanning UV to far-red; a telescope has a
bandpass. Until now the engine turned a blackbody temperature straight to RGB (holographic_blackbody)
with the CIE curves BAKED IN. This module lifts that hidden step into a first-class, reusable OBSERVER:

  * the same code reads a human eye, a bug eye, or an instrument -- ONE core, many sensors. This is the
    unifier that says a bug eye and a radio dish are the same object (they differ only in their channels);
  * blackbody_rgb becomes a SPECIAL CASE -- the human CIE observer applied to a Planck spectrum -- and we
    prove that BYTE-IDENTICALLY rather than forking the colour path.

An observer is (wavelengths_nm, S, names): S is (nchan, nlam), one sensitivity row per channel. `observe`
integrates a spectrum sampled on the same wavelengths against S -> readings (..., nchan). That integral is
a weighted sum (a matmul), so it is FIELD-NATIVE: hand it a hyperspectral IMAGE (..., nlam) and get a
per-pixel reading image (..., nchan) in one call -- the "observer over a field" convergence.

BACKWARD COMPATIBILITY (non-negotiable): this module only ADDS; it does not touch holographic_blackbody.
The human observer REUSES blackbody's own CIE curves, XYZ->sRGB matrix and gamma (imported, not copied),
so human_rgb(planck(T)) equals blackbody_rgb(T) to the last bit -- asserted in _selftest.

DIRECTIONS (up/down/sideways -- a missed direction is a missed faculty)
  DOWN  -- observe a single channel (S with one row) -- native.
  UP    -- a hyperspectral image is a FIELD of spectra: `observe` broadcasts, (...,nlam)->(...,nchan). One
           implementation, pixel to cube. (For spectra at CONTINUOUS wavelengths rather than a grid,
           resample via scatter/gather first; the gridded matmul is the common case.)
  SIDEWAYS
    field    -- per-pixel readings image (feeds false-colour, O3).
    structure-- the observer IS a role-bound record: each channel bound to its curve.
    sequence -- a scanning sensor / sweeping spectrometer: observe over a time axis, same broadcast.
    program  -- the integral is a matmul; emit_kernel could project it. DECLARED not-yet-wired (thin).

Determinism: analytic CIE curves (from holographic_blackbody), pure numpy, no RNG. Exact.
"""

import numpy as np
# Import blackbody's OWN colour machinery (not a second copy) precisely so the human observer reproduces
# blackbody_rgb bit-for-bit -- one colour path, proven identical, never forked. This is the PROMOTE move:
# the old temperature->RGB call is just this observer applied to a hot body.
from holographic.misc import holographic_blackbody as _bb
from holographic.rendering import holographic_mueller as _mu  # polarization channels: the mantis reads e-vector and handedness via Mueller analyzers


def receptor_bank(wavelengths_nm, centers_nm, widths_nm, gains=None):
    """A bank of Gaussian receptor sensitivities on a wavelength grid -- the generic shape of a biological
    eye's cones. `centers`/`widths` in nm, one per channel; optional per-channel `gains`. Returns (nchan, nlam).
    The mantis observer (O2, later) is just this with ~12 centres from UV to far-red; the human observer uses
    the sharper CIE curves instead. Generalising the eye to 'a bank of tuned receptors' is why one core serves
    both."""
    wl = np.asarray(wavelengths_nm, float)
    c = np.asarray(centers_nm, float)[:, None]
    w = np.asarray(widths_nm, float)[:, None]
    g = np.ones(c.shape[0]) if gains is None else np.asarray(gains, float)
    # A Gaussian per channel, centred and width-scaled; broadcast over the shared wavelength axis.
    return np.exp(-0.5 * ((wl[None, :] - c) / w) ** 2) * g[:, None]


def make_observer(wavelengths_nm, sensitivities, names=None):
    """Assemble an observer from a wavelength grid and per-channel sensitivity curves.

    sensitivities: (nchan, nlam) array (or a single (nlam,) row). Returns {wavelengths_nm, S, names} -- a
    plain dict so it stays numpy-friendly and easy to introspect. The observer is data, not behaviour;
    `observe` supplies the behaviour."""
    wl = np.asarray(wavelengths_nm, float)
    S = np.asarray(sensitivities, float)
    if S.ndim == 1:
        S = S[None, :]
    if S.shape[-1] != wl.shape[0]:
        raise ValueError("sensitivity last axis (%d) must match wavelengths (%d)" % (S.shape[-1], wl.shape[0]))
    if names is None:
        names = ["ch%d" % i for i in range(S.shape[0])]
    return {"wavelengths_nm": wl, "S": S, "names": list(names)}


def human_cie(samples=90):
    """The human eye as an observer: the CIE 1931 colour-matching functions (xbar, ybar, zbar) on the SAME
    380-780 nm / `samples` grid holographic_blackbody uses. Feeding it a Planck spectrum and calling to_srgb
    reproduces blackbody_rgb exactly -- blackbody is this observer applied to a hot body."""
    lam = np.linspace(380.0, 780.0, int(samples))
    xb, yb, zb = _bb._cie_xyz_bar(lam)
    return make_observer(lam, np.stack([xb, yb, zb]), names=["X", "Y", "Z"])


def observe(spectrum, observer):
    """Integrate a spectrum (sampled on observer['wavelengths_nm']) against each channel -> readings (...,nchan).

    Field-native: `spectrum` may be (nlam,) for one sample or (...,nlam) for a whole hyperspectral image; the
    readings come back with the matching leading shape. It is a plain weighted sum (rectangle rule) -- the
    uniform dlambda is a constant the display/normalisation absorbs, and dropping it is exactly what keeps the
    human path bit-identical to blackbody. readings[...,c] = sum_lam spectrum[...,lam] * S[c,lam]."""
    S = observer["S"]
    spec = np.asarray(spectrum, float)
    if spec.shape[-1] != S.shape[-1]:
        raise ValueError("spectrum last axis (%d) must match observer wavelengths (%d)" % (spec.shape[-1], S.shape[-1]))
    # Reduce with np.sum (not einsum) over the wavelength axis: np.sum's pairwise accumulation order is the
    # SAME one holographic_blackbody uses (sum(B*xbar)), which is what makes the human path byte-identical.
    # spec[...,None,:] broadcasts each channel row of S over the spectrum; sum over the last (wavelength) axis.
    return np.sum(spec[..., None, :] * S, axis=-1)


def to_srgb(xyz, mode="hue"):
    """CIE XYZ readings (from the human observer) -> sRGB, reproducing holographic_blackbody's exact conversion
    (its matrix + gamma, hue- or luminance-normalised). Only meaningful for the CIE observer; other observers
    use false-colour mappings (O3) instead. `xyz` is (...,3).

    The single-vector path is written to MATCH blackbody's expression operation-for-operation so the byte-
    identical gate holds; the batched path vectorises the same arithmetic for image cubes."""
    xyz = np.asarray(xyz, float)
    if xyz.ndim == 1:
        # Exactly blackbody's scalar path: M @ [X,Y,Z], then normalise/clip/gamma. Same ops, same order.
        X, Y, Z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        if mode == "hue":
            s = max(X + Y + Z, 1e-30)
            X, Y, Z = X / s, Y / s, Z / s
            rgb = _bb._XYZ_TO_RGB @ np.array([X, Y, Z])
            rgb = rgb / max(rgb.max(), 1e-9)
        else:
            rgb = _bb._XYZ_TO_RGB @ np.array([X, Y, Z])
            rgb = rgb / 1.5e13
        return _bb._gamma_encode(np.clip(rgb, 0.0, 1.0))
    # Batched (image cube) path: same arithmetic, broadcast over leading axes.
    if mode == "hue":
        s = np.maximum(xyz.sum(axis=-1, keepdims=True), 1e-30)
        rgb = (xyz / s) @ _bb._XYZ_TO_RGB.T
        rgb = rgb / np.maximum(rgb.max(axis=-1, keepdims=True), 1e-9)
    else:
        rgb = (xyz @ _bb._XYZ_TO_RGB.T) / 1.5e13
    return _bb._gamma_encode(np.clip(rgb, 0.0, 1.0))


def human_rgb(spectrum, mode="hue", samples=90):
    """What the human eye sees from a spectrum, as sRGB = to_srgb(observe(spectrum, human_cie)). The convenience
    door; for a Planck spectrum it equals blackbody_rgb (the correctness gate)."""
    return to_srgb(observe(spectrum, human_cie(samples)), mode=mode)


# Representative stomatopod receptor peak wavelengths (nm), deep-UV through far-red. Real lambda-max values
# vary by species and are debated (Marshall 1988; Cronin & Marshall 1989; Bok et al. 2014 on the UV filters);
# we model the SHAPE that matters -- ~12 narrow bands tiling an unusually wide range INCLUDING deep UV -- not
# exact peaks. DECLARED NEGATIVE: these are illustrative centres, not a species-specific fit.
_MANTIS_CENTERS_NM = (315.0, 340.0, 365.0, 390.0, 425.0, 455.0, 490.0, 520.0, 550.0, 580.0, 620.0, 660.0)
# Coloured crystalline-cone filters NARROW the receptors (more so at long wavelengths), which is how the mantis
# gets 12 well-separated channels from broad opsins (Cronin & Marshall). Smaller width -> sharper band.
_MANTIS_WIDTHS_NM = (18.0, 18.0, 16.0, 16.0, 15.0, 14.0, 13.0, 12.0, 12.0, 11.0, 11.0, 10.0)


def mantis_receptors(wavelengths_nm):
    """The mantis shrimp's ~12 spectral receptors as an observer, evaluated on the given wavelength grid.
    Twelve narrow channels from deep UV (~315 nm) to far red (~660 nm) -- an unusually wide, finely tiled range
    (Marshall 1988; Bok et al. 2014). Pass a grid that reaches into the UV (e.g. 300-720 nm) or the UV channels
    see nothing. See holographic_observer.receptor_bank -- this is that bank with the mantis' centres/filters."""
    S = receptor_bank(wavelengths_nm, _MANTIS_CENTERS_NM, _MANTIS_WIDTHS_NM)
    names = ["R%02d_%dnm" % (i + 1, int(c)) for i, c in enumerate(_MANTIS_CENTERS_NM)]
    return make_observer(wavelengths_nm, S, names=names)


def polarization_readout(stokes):
    """Read POLARIZATION from a (broadband) Stokes field the way the mantis midband does: orthogonal LINEAR
    detectors and -- via a quarter-wave retarder (the R8 rhabdomere, Chiou et al. 2008) -- CIRCULAR detectors.

    stokes: (...,4). Returns a dict of intensity channels (all field-native):
      linear_0 / linear_45 / linear_90 / linear_135 -- behind linear polarizers at those angles;
      circular_R / circular_L -- behind a QWP(0) then a polarizer at +/-45 deg (the standard circular analyzer:
                                 RCP passes +45 fully, LCP passes -45 fully -- verified in _selftest).
    Plus derived evector_angle (recovered purely from the linear channels) and handedness_sign (from R-L).
    This REUSES holographic_mueller entirely -- the mantis' unique trick is just a retarder before ordinary
    linear detectors, so circular-polarization vision costs us almost nothing once the Mueller core exists."""
    s = np.asarray(stokes, float)

    def chan(element):
        return _mu.apply(element, s)[..., 0]  # S0 after the analyzer -- what a detector integrates

    lin0 = chan(_mu.linear_polarizer(0.0))
    lin45 = chan(_mu.linear_polarizer(np.pi / 4))
    lin90 = chan(_mu.linear_polarizer(np.pi / 2))
    lin135 = chan(_mu.linear_polarizer(3 * np.pi / 4))
    qwp = _mu.quarter_wave(0.0)
    circR = _mu.apply(_mu.linear_polarizer(np.pi / 4), _mu.apply(qwp, s))[..., 0]
    circL = _mu.apply(_mu.linear_polarizer(-np.pi / 4), _mu.apply(qwp, s))[..., 0]
    # e-vector angle from orthogonal detector PAIRS: (lin0-lin90) ~ Q, (lin45-lin135) ~ U; angle = 0.5*atan2.
    # Recovered from the channels themselves, not peeked from the Stokes vector -- this is what the eye computes.
    evec = 0.5 * np.arctan2(lin45 - lin135, lin0 - lin90)
    hand = np.sign(circR - circL)
    return {"linear_0": lin0, "linear_45": lin45, "linear_90": lin90, "linear_135": lin135,
            "circular_R": circR, "circular_L": circL, "evector_angle": evec, "handedness_sign": hand}


def mantis_view(spectral_stokes, wavelengths_nm):
    """The full mantis-shrimp reading of a spectral-Stokes signal: 12 spectral channels + linear + circular
    polarization, in one call. `spectral_stokes` is (...,nlam,4) sampled on `wavelengths_nm`. Returns a dict:
      spectral -- (...,12) the DIRECT per-receptor intensities (from S0), NOT colour-opponent processed;
      linear_*/circular_*, evector_angle, handedness_sign -- from the wavelength-integrated Stokes.

    KEPT NEGATIVE (Thoen et al. 2014, pinned): mantis colour DISCRIMINATION is coarse, with no evidence of a
    colour-opponent system; the eye appears to do a fast DIRECT readout of each receptor, not the fine opponent
    comparison a human does. So we return the raw 12 channels and deliberately DO NOT compute opponent
    differences -- the popular '12 channels = superb colour vision' story is exactly what the measurement
    overturned. Modelling the eye honestly means modelling the coarse direct readout."""
    ss = np.asarray(spectral_stokes, float)
    if ss.shape[-1] != 4:
        raise ValueError("spectral_stokes last axis must be 4 (Stokes); got %d" % ss.shape[-1])
    S0 = ss[..., 0]                                          # intensity spectrum per sample -> (...,nlam)
    spec = observe(S0, mantis_receptors(wavelengths_nm))     # (...,12) direct readout
    broadband = np.sum(ss, axis=-2)                          # integrate the Stokes state over wavelength -> (...,4)
    out = {"spectral": spec}
    out.update(polarization_readout(broadband))
    return out


def _selftest():
    """Regression trap: the human observer must REPRODUCE blackbody byte-identically (the promote gate), and
    the machinery must be field-native (UP) and channel-general (DOWN)."""
    samples = 90
    lam = np.linspace(380.0, 780.0, samples)
    lam_m = lam * 1e-9

    # --- THE GATE: human_rgb(planck(T)) == blackbody_rgb(T), to the last bit, for several temperatures ---
    for T in (1000.0, 2800.0, 6500.0, 12000.0):
        B = _bb.planck_radiance(lam_m, T)
        got = human_rgb(B, mode="hue", samples=samples)
        want = _bb.blackbody_rgb(T, normalize="hue", samples=samples)
        assert np.array_equal(got, want), "human observer != blackbody at %gK: %r vs %r" % (T, got, want)
    # also the luminance-normalised mode, byte-identical
    for T in (1500.0, 9000.0):
        B = _bb.planck_radiance(lam_m, T)
        assert np.array_equal(human_rgb(B, mode="none", samples=samples),
                              _bb.blackbody_rgb(T, normalize="none", samples=samples)), "mode=none mismatch at %gK" % T

    # --- UP: a hyperspectral image (2x3 pixels) observed in one call ---
    obs = human_cie(samples)
    img = np.stack([_bb.planck_radiance(lam_m, t) for t in (3000, 4000, 5000, 6000, 7000, 8000)]).reshape(2, 3, samples)
    readings = observe(img, obs)                       # (2,3,3)
    assert readings.shape == (2, 3, 3), readings.shape
    rgb_img = to_srgb(readings, mode="hue")            # (2,3,3)
    assert rgb_img.shape == (2, 3, 3)
    # the per-pixel field result must match the per-pixel scalar path (no drift across the up-direction)
    # T=5000 is the 3rd temperature -> flat index 2 -> pixel [0,2] (reshape is row-major); the field path
    # must match the scalar path to machine precision (measured ~2e-16), proving no up-direction drift.
    single = human_rgb(_bb.planck_radiance(lam_m, 5000.0), mode="hue", samples=samples)
    assert np.allclose(rgb_img[0, 2], single, atol=1e-12), "field path drifted from scalar path"

    # --- DOWN + generality: a non-human observer (a 4-band UV..red receptor bank) produces sane readings ---
    S = receptor_bank(lam, centers_nm=[360, 450, 540, 650], widths_nm=[30, 30, 30, 30])
    bug = make_observer(lam, S, names=["UV", "B", "G", "R"])
    r_hot = observe(_bb.planck_radiance(lam_m, 9000.0), bug)   # blue-hot -> more short-wavelength response
    r_cool = observe(_bb.planck_radiance(lam_m, 1200.0), bug)  # ember -> more long-wavelength response
    # a hot source leans bluer, a cool source leans redder, in the bug's own channels
    assert r_hot[1] / r_hot[3] > r_cool[1] / r_cool[3], "receptor bank did not separate hot vs cool"
    assert observe(_bb.planck_radiance(lam_m, 5000.0), {"wavelengths_nm": lam, "S": S[:1], "names": ["UV"]}).shape[-1] == 1

    # --- determinism ---
    B = _bb.planck_radiance(lam_m, 3000.0)
    assert np.array_equal(human_rgb(B), human_rgb(B)), "observer not deterministic"

    # ================= MANTIS (O2): circular/linear polarization + 12-band UV..red, direct readout =================
    lamM = np.linspace(300.0, 720.0, 140)
    bump = np.exp(-0.5 * ((lamM - 500.0) / 60.0) ** 2)     # a broadband visible spectrum to carry the polarization
    def spectral_stokes(S1f, S2f, S3f):
        S = np.zeros(lamM.shape + (4,))
        S[..., 0] = bump; S[..., 1] = bump * S1f; S[..., 2] = bump * S2f; S[..., 3] = bump * S3f
        return S
    # THE PAYOFF: right- vs left-circular are DISTINGUISHED; unpolarized is not (handedness is the mantis' gift).
    vR = mantis_view(spectral_stokes(0, 0, +1.0), lamM)
    vL = mantis_view(spectral_stokes(0, 0, -1.0), lamM)
    vU = mantis_view(spectral_stokes(0, 0, 0.0), lamM)
    assert vR["circular_R"] > vR["circular_L"] and vR["handedness_sign"] == 1, "RCP not detected"
    assert vL["circular_L"] > vL["circular_R"] and vL["handedness_sign"] == -1, "LCP not detected"
    assert abs(vU["circular_R"] - vU["circular_L"]) < 1e-9 and vU["handedness_sign"] == 0, "unpolarized should read no handedness"
    # linear e-vector angle recovered from the channels (input linear at 30 deg -> S1=cos60,S2=sin60)
    th = np.deg2rad(30.0)
    vLin = mantis_view(spectral_stokes(np.cos(2 * th), np.sin(2 * th), 0.0), lamM)
    dth = (vLin["evector_angle"] - th + np.pi / 2) % np.pi - np.pi / 2
    assert abs(dth) < np.deg2rad(0.5), "e-vector angle off by %.3f deg" % np.rad2deg(dth)
    # UV sensitivity: a UV-peaked spectrum lights the first (UV) receptors more than a red-peaked one does
    def spec_only(mu):
        S = np.zeros(lamM.shape + (4,)); S[..., 0] = np.exp(-0.5 * ((lamM - mu) / 20.0) ** 2); return S
    uv = mantis_view(spec_only(340.0), lamM)["spectral"]
    red = mantis_view(spec_only(650.0), lamM)["spectral"]
    assert uv[:4].sum() > uv[8:].sum() and red[8:].sum() > red[:4].sum(), "UV/red receptors did not separate"
    # DIRECT READOUT (Thoen negative): exactly 12 raw channels, equal to observe() -- no opponent processing added
    S0only = spec_only(500.0)[..., 0]
    assert vR["spectral"].shape[-1] == 12
    assert np.array_equal(mantis_view(spec_only(500.0), lamM)["spectral"], observe(S0only, mantis_receptors(lamM))), "mantis added hidden processing"

    print("holographic_observer selftest OK  |  human observer == blackbody byte-identical (4 temps x2 modes); "
          "field-native (UP) drift-free; receptor bank separates hot/cool; MANTIS reads RCP/LCP + e-vector + 12 UV-red bands (direct readout, no opponency per Thoen 2014)  |  KEPT NEGATIVE: 'program' (shader) "
          "costume not yet emitted -- declared, not an oversight")


if __name__ == "__main__":
    _selftest()

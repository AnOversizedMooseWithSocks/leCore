"""holographic_rmsynth.py -- ROTATION-MEASURE SYNTHESIS: the Faraday depth of polarized light (leCore rendering/optics).

WHY THIS EXISTS
---------------
When linearly polarized light travels through a magnetized plasma, its plane of polarization ROTATES,
and the rotation angle grows with the SQUARE of the wavelength:  chi(lambda^2) = chi0 + phi * lambda^2.
The constant `phi` is the FARADAY DEPTH (rad/m^2) -- it is a direct probe of the line-of-sight magnetic
field, and it is how a radio telescope weighs a galaxy's magnetism from the polarization of its glow.

Brentjens & de Bruyn (2005) showed that if you write the complex polarization

        P(lambda^2) = Q(lambda^2) + i*U(lambda^2)          (this is holographic_stokes.complex_linear)

then the distribution of emission over Faraday depth is recovered by a FOURIER-LIKE transform:

        F(phi) = (1/K) * sum_j  w_j * P_j * exp(-2i * phi * (lambda^2_j - lambda^2_0))

This module is the CONVERGENCE the Stokes module was built for. It is not new mathematics: it is the
engine's own complex-phasor transform, wearing an astronomy hat. The same operator (a phasor summed
over a linear axis) is what binding, FHRR memory, and the phase vocoder already are (the "U1" unifier).
We get the telescope's magnetic-field probe by REUSING the polarization state we already carry.

It is also a clean example of "lift to the rung where the problem is linear": Faraday rotation is
nonlinear in wavelength but LINEAR in lambda^2, and P(lambda^2) -> F(phi) is then a linear transform.

METHOD (honest scope)
  * The transform is a DIRECT SUM, not an FFT, because real observations sample lambda^2 UNEVENLY
    (radio bands have gaps). A direct sum is exact for any sampling; an FFT would force a regular grid
    we do not have. For N channels x M phi-values this is O(N*M) -- fine for real spectra, and it
    vectorizes over a whole image (the UP direction) with one matmul.
  * `rmtf` returns the ROTATION-MEASURE TRANSFER FUNCTION -- the "dirty beam" in Faraday space. Its
    width is the resolution; its sidelobes are why a raw F(phi) is a DIRTY spectrum. Deconvolving it
    (RM-CLEAN) is the same plug-and-play inverse loop as CLEAN/RML elsewhere in the arc; it is a
    DECLARED FUTURE extension here, not silently implied.

DIRECTIONS (up/down/sideways -- a missed direction is a missed faculty)
  DOWN  -- runs on any SUBSET of channels (a sub-band); it is just a sum over the channels you pass.
  UP    -- a per-pixel spectrum is a component of a polarization IMAGE CUBE. Every function broadcasts
           over leading spatial axes: P of shape (..., nchan) -> F of shape (..., nphi). One pixel is
           (nchan,); an image is (H, W, nchan). One implementation, all scales.
  SIDEWAYS
    sequence  -- this IS the sequence costume of Stokes: P(lambda^2) is a sampled complex wave.
    field     -- the F(phi) cube over an image; `peak_rm` reduces it to an RM map.
    structure -- `peak_rm` returns a {rm, polarized_intensity, angle0} record per source.
    program   -- DECLARED NEGATIVE: the reduction is over an irregular axis, awkward for a shader;
                 not emitted. (The forward chi(lambda^2) rotation is elementwise and could be.)

Determinism: pure closed-form numpy, no RNG, no hashing. Exact to floating point.
"""

import numpy as np

# 2 appears because polarization angle is defined modulo pi (a 180-degree flip of the e-vector is the
# same state), so the Faraday phase advances as 2*phi*lambda^2, not phi*lambda^2. Naming it once here
# keeps every formula below honest about that factor.
_TWO = 2.0


def _as_complex_P(P=None, Q=None, U=None):
    """Return the complex polarization P = Q + iU as a numpy array.

    Accept EITHER a ready-made complex `P` (e.g. from holographic_stokes.complex_linear) OR real
    `Q` and `U` arrays. This is the one place the two calling conventions meet, so callers upstream
    never have to care which they hold. Shape is preserved; the last axis is the channel axis.
    """
    if P is not None:
        return np.asarray(P, dtype=np.complex128)
    if Q is None or U is None:
        raise ValueError("provide either P (complex) or both Q and U (real)")
    return np.asarray(Q, dtype=np.float64) + 1j * np.asarray(U, dtype=np.float64)


def reference_lambda2(lambda2, weights=None):
    """Pick the reference lambda^2_0 the transform rotates about.

    Brentjens & de Bruyn recommend the WEIGHTED MEAN of lambda^2: it minimizes the wavelength-
    dependent smearing of the transfer function, so the RMTF is as sharp and symmetric as the
    sampling allows. Choosing it here (rather than 0) is why recovered intrinsic angles are stable.
    """
    lambda2 = np.asarray(lambda2, dtype=np.float64)
    if weights is None:
        return float(np.mean(lambda2))
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(weights * lambda2) / np.sum(weights))


def rmsynth(lambda2, phi, P=None, Q=None, U=None, weights=None, lambda2_0=None):
    """Faraday dispersion function F(phi) -- rotation-measure synthesis (Brentjens & de Bruyn 2005).

    lambda2   : (nchan,) the lambda^2 sample of each channel (may be unevenly spaced; gaps are fine).
    phi       : (nphi,)  the Faraday depths (rad/m^2) to evaluate. See `phi_grid` for a sensible one.
    P / Q,U   : the complex polarization per channel, shape (..., nchan). Pass P=complex, or Q and U.
    weights   : (nchan,) optional per-channel weights (e.g. inverse variance); default uniform.
    lambda2_0 : reference lambda^2; default = weighted mean (see reference_lambda2).

    Returns F of shape (..., nphi): the DIRTY Faraday spectrum. Field-native -- give it an image cube
    (H, W, nchan) and you get (H, W, nphi) back. `peak_rm` turns that into an RM map.
    """
    lambda2 = np.asarray(lambda2, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    Pc = _as_complex_P(P, Q, U)
    nchan = lambda2.shape[0]
    if Pc.shape[-1] != nchan:
        raise ValueError("last axis of P/Q/U (%d) must match lambda2 (%d)" % (Pc.shape[-1], nchan))
    if weights is None:
        w = np.ones(nchan, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    if lambda2_0 is None:
        lambda2_0 = reference_lambda2(lambda2, weights)
    K = np.sum(w)
    # The transform kernel: one complex phasor per (channel, phi) pair. Shape (nchan, nphi).
    # exp(-2i * phi * (lambda^2 - lambda^2_0)) -- de-rotate each channel back to the reference and
    # test every candidate Faraday depth. This is the whole method in one line.
    ph = np.exp(-1j * _TWO * np.outer(lambda2 - lambda2_0, phi))  # (nchan, nphi)
    wP = (Pc * w)                                                 # (..., nchan), weights applied
    # Sum over the channel axis against the kernel; einsum keeps it field-native for any leading dims.
    F = np.einsum("...c,cp->...p", wP, ph) / K
    return F


def rmtf(lambda2, phi, weights=None, lambda2_0=None):
    """Rotation-measure transfer function R(phi) -- the "dirty beam" in Faraday space.

    It is `rmsynth` run on a source of unit polarization at zero depth: the instrument's response to a
    single Faraday-thin point. Its main-lobe width is the resolution; its sidelobes are the artefacts a
    real F(phi) is convolved with. R(0) == 1 exactly by construction (a useful selftest anchor).
    """
    lambda2 = np.asarray(lambda2, dtype=np.float64)
    n = lambda2.shape[0]
    unitP = np.ones(n, dtype=np.complex128)
    return rmsynth(lambda2, phi, P=unitP, weights=weights, lambda2_0=lambda2_0)


def resolution_fwhm(lambda2):
    """Faraday-depth resolution (FWHM of the RMTF main lobe), closed form.

    delta_phi ~= 2*sqrt(3) / (lambda^2_max - lambda^2_min)  (Brentjens & de Bruyn 2005, their eq. 61).
    The wider your lambda^2 coverage, the finer the Faraday depths you can separate -- the exact analogue
    of aperture setting angular resolution. Returned so `phi_grid` and callers can size things honestly.
    """
    lambda2 = np.asarray(lambda2, dtype=np.float64)
    span = float(np.max(lambda2) - np.min(lambda2))
    if span <= 0.0:
        raise ValueError("lambda2 must span a nonzero range to resolve Faraday depth")
    return _TWO * np.sqrt(3.0) / span


def phi_grid(lambda2, oversample=5.0, extent=None):
    """Build a sensible grid of Faraday depths to evaluate, from the lambda^2 sampling itself.

    Sampled at `oversample` points across each RMTF resolution element (so the main lobe is not
    aliased), out to +/- `extent`. If `extent` is None we use a modest multiple of the max Faraday
    depth the CHANNEL SPACING can represent unambiguously, so the default grid is neither starved
    nor absurdly wide. Everything here is derived from the data -- no magic numbers the caller must guess.
    """
    lambda2 = np.asarray(lambda2, dtype=np.float64)
    dphi = resolution_fwhm(lambda2) / float(oversample)
    if extent is None:
        # Max |phi| set by the smallest lambda^2 step: phi_max ~= sqrt(3)/min(delta lambda^2).
        dl = np.diff(np.sort(lambda2))
        dl = dl[dl > 0]
        step = float(np.min(dl)) if dl.size else float(np.ptp(lambda2))
        extent = np.sqrt(3.0) / step if step > 0 else 10.0 * resolution_fwhm(lambda2)
    n = int(np.ceil(2.0 * extent / dphi))
    n = max(n, 3)
    return np.linspace(-extent, extent, n)


def peak_rm(F, phi, lambda2_0=None):
    """Reduce a Faraday spectrum to its brightest source: {rm, polarized_intensity, angle0}.

    Field-native: F may be (..., nphi); the reduction is over the last axis, so an image cube collapses
    to per-pixel RM / polarized-intensity / intrinsic-angle maps. The peak is refined by a 3-point
    PARABOLIC interpolation about the maximum |F| bin, so the recovered RM is not quantized to the grid
    (sub-bin accuracy -- the standard trick, and why the selftest can demand tight tolerances).

    angle0 = 0.5 * arg(F(phi_peak)) is the polarization angle, and by construction F de-rotates about
    the reference lambda^2_0 -- so this angle is referenced to lambda^2_0 (Brentjens & de Bruyn's
    convention, which decorrelates angle and RM errors). Pass `lambda2_0` and we additionally de-rotate
    to lambda^2 = 0 to give the true INTRINSIC angle chi0 = angle0_ref - rm*lambda2_0. The 0.5 is the
    same modulo-pi factor as everywhere else. Returned as a dict of arrays (or scalars for one spectrum).
    """
    F = np.asarray(F)
    phi = np.asarray(phi, dtype=np.float64)
    amp = np.abs(F)
    k = np.argmax(amp, axis=-1)
    nphi = phi.shape[0]
    kc = np.clip(k, 1, nphi - 2)  # keep the 3-point stencil in range at the edges

    # Gather the three amplitudes around each peak for the parabolic vertex estimate.
    def _take(idx):
        return np.take_along_axis(amp, idx[..., None], axis=-1)[..., 0]
    a0 = _take(kc - 1)
    a1 = _take(kc)
    a2 = _take(kc + 1)
    denom = (a0 - 2.0 * a1 + a2)
    # delta in [-0.5, 0.5]; guard the flat/degenerate case (denom==0 -> peak exactly on the bin).
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 0.5 * (a0 - a2) / denom
    delta = np.where(np.abs(denom) < 1e-300, 0.0, delta)
    delta = np.clip(delta, -0.5, 0.5)

    dphi = float(phi[1] - phi[0]) if nphi > 1 else 0.0
    rm = phi[kc] + delta * dphi
    # Complex value at the (integer) peak bin gives amplitude and intrinsic angle.
    Fpk = np.take_along_axis(F, kc[..., None], axis=-1)[..., 0]
    pol = np.abs(Fpk)
    angle0 = 0.5 * np.angle(Fpk)
    if lambda2_0 is not None:
        # De-rotate the reference-frame angle back to lambda^2 = 0 for the true intrinsic angle,
        # then wrap into (-pi/2, pi/2] since polarization angle is defined modulo pi.
        angle0 = ((angle0 - rm * float(lambda2_0)) + np.pi / 2) % np.pi - np.pi / 2
    out = {"rm": rm, "polarized_intensity": pol, "angle0": angle0}
    # For a single 1-D spectrum, hand back plain floats -- friendlier at the REPL and in the selftest.
    if F.ndim == 1:
        return {kk: float(vv) for kk, vv in out.items()}
    return out


def faraday_rotate(stokes0, lambda2, rm):
    """FORWARD Faraday model: rotate an intrinsic polarized signal as a magnetized plasma would -- the sky a
    telescope actually receives. `stokes0` is the intrinsic Stokes (...,4) at zero wavelength; `lambda2` is
    (nchan,); `rm` is the Faraday depth per pixel (...,). Returns (...,nchan,4).

    Faraday rotation turns the plane of LINEAR polarization by rm*lambda^2, i.e. it rotates (Q,U) by
    2*rm*lambda^2 while leaving intensity S0 and circular S3 untouched (a simple, standard model). This is the
    per-wavelength Mueller ROTATOR (holographic_mueller.rotator) applied across the band -- done directly here so
    it stays field-native over a whole sky. Its output is exactly what rmsynth inverts, so the two round-trip."""
    s0 = np.asarray(stokes0, float)
    lam2 = np.asarray(lambda2, float)
    rm = np.asarray(rm, float)
    if s0.shape[-1] != 4:
        raise ValueError("stokes0 last axis must be 4 (Stokes); got %d" % s0.shape[-1])
    Q0 = s0[..., 1]; U0 = s0[..., 2]
    # rotation angle in the (Q,U) plane, per pixel per channel: theta = 2 * rm * lambda^2. Broadcast rm(...) and
    # lam2(nchan) to (..., nchan) by adding a trailing channel axis to rm.
    theta = _TWO * rm[..., None] * lam2                       # (..., nchan)
    c = np.cos(theta); sn = np.sin(theta)
    Q = Q0[..., None] * c - U0[..., None] * sn
    U = Q0[..., None] * sn + U0[..., None] * c
    out = np.empty(s0.shape[:-1] + (lam2.shape[0], 4), dtype=float)
    out[..., 0] = s0[..., 0][..., None]                       # S0 (intensity) unchanged across the band
    out[..., 1] = Q
    out[..., 2] = U
    out[..., 3] = s0[..., 3][..., None]                       # S3 (circular) unchanged by Faraday rotation
    return out


def faraday_rm_map(lambda2, stokes_cube, phi=None, weights=None):
    """INVERSE: recover a per-pixel Faraday-depth map from a sky Stokes cube -- the telescope-as-observer result.
    `stokes_cube` is (...,nchan,4) (a polarization image over wavelength^2); forms P=Q+iU per pixel and runs
    rm synthesis over the whole field, then peaks it. Returns {rm, polarized_intensity, angle0, phi, F} where rm
    is the (...) RM map. This is the one-call 'sky cube -> line-of-sight magnetism' door; it just composes
    rmsynth + peak_rm, which are already field-native, so it works on any image shape."""
    cube = np.asarray(stokes_cube, float)
    if cube.shape[-1] != 4:
        raise ValueError("stokes_cube last axis must be 4 (Stokes); got %d" % cube.shape[-1])
    lam2 = np.asarray(lambda2, float)
    P = cube[..., 1] + 1j * cube[..., 2]                      # Q + iU per pixel -> (..., nchan)
    if phi is None:
        phi = phi_grid(lam2)
    lam2_0 = reference_lambda2(lam2, weights)
    F = rmsynth(lam2, phi, P=P, weights=weights, lambda2_0=lam2_0)
    pk = peak_rm(F, phi, lambda2_0=lam2_0)
    pk["phi"] = phi
    pk["F"] = F
    return pk


def _selftest():
    """Regression trap: inject known Faraday-thin sources and demand they come back, on a spectrum
    AND on a field (the UP direction), with the transfer function anchored exactly."""
    rng = np.random.default_rng(0)

    # An uneven lambda^2 sampling with a gap, like a real two-band radio observation.
    band_a = np.linspace(0.03, 0.09, 120)
    band_b = np.linspace(0.14, 0.24, 160)
    lam2 = np.concatenate([band_a, band_b])

    # --- anchor: the RMTF is exactly 1 at zero Faraday depth ---
    R0 = rmtf(lam2, np.array([0.0]))[0]
    assert abs(R0 - 1.0) < 1e-12, "RMTF(0) must be exactly 1, got %r" % (R0,)

    # --- single Faraday-thin source: P_j = p0 * exp(2i(chi0 + phi0*lambda^2_j)) ---
    phi0, chi0, p0 = 42.0, 0.6, 2.5
    P = p0 * np.exp(1j * _TWO * (chi0 + phi0 * lam2))
    phis = phi_grid(lam2, oversample=6.0)
    F = rmsynth(lam2, phis, P=P)
    lam2_0 = reference_lambda2(lam2)
    pk = peak_rm(F, phis, lambda2_0=lam2_0)
    res = resolution_fwhm(lam2)
    # RM recovered to well within one resolution element; parabolic refine makes this tight.
    assert abs(pk["rm"] - phi0) < 0.15 * res, "RM off: got %.3f want %.3f (res=%.3f)" % (pk["rm"], phi0, res)
    # Amplitude recovered to a few percent (sidelobe leakage of the dirty spectrum).
    assert abs(pk["polarized_intensity"] - p0) / p0 < 0.03, "polarized intensity off: %.4f vs %.4f" % (pk["polarized_intensity"], p0)
    # Intrinsic angle recovered to ~1 degree (angle is modulo pi).
    dchi = (pk["angle0"] - chi0 + np.pi / 2) % np.pi - np.pi / 2
    assert abs(dchi) < np.deg2rad(1.5), "intrinsic angle off by %.3f deg" % np.rad2deg(dchi)

    # --- UP direction: a tiny image of two pixels with different RMs, one call ---
    phiA, phiB = -30.0, 75.0
    Pa = 1.0 * np.exp(1j * _TWO * (0.2 + phiA * lam2))
    Pb = 1.7 * np.exp(1j * _TWO * (-0.4 + phiB * lam2))
    cube = np.stack([Pa, Pb], axis=0)            # (2, nchan) -- a 1-D "image"
    Fc = rmsynth(lam2, phis, P=cube)             # (2, nphi)  -- field-native
    assert Fc.shape == (2, phis.shape[0])
    pkc = peak_rm(Fc, phis)
    assert abs(pkc["rm"][0] - phiA) < 0.2 * res and abs(pkc["rm"][1] - phiB) < 0.2 * res, \
        "field RMs off: %r" % (pkc["rm"],)

    # --- determinism: identical inputs -> byte-identical outputs ---
    F2 = rmsynth(lam2, phis, P=P)
    assert np.array_equal(F, F2), "rmsynth is not deterministic"

    # ================= X1: telescope-as-observer -- forward Faraday rotate a SKY, recover the RM map =================
    # A 3x3 sky: random intrinsic linear polarization per pixel, a spatially-varying RM field across the image.
    rng2 = np.random.default_rng(7)
    ang0 = rng2.uniform(0, np.pi, size=(3, 3))                 # intrinsic e-vector angle per pixel
    p0 = 1.0
    s0 = np.zeros((3, 3, 4)); s0[..., 0] = 1.5
    s0[..., 1] = p0 * np.cos(2 * ang0); s0[..., 2] = p0 * np.sin(2 * ang0)
    rm_true = np.linspace(-60.0, 60.0, 9).reshape(3, 3)        # an RM gradient across the sky
    cube = faraday_rotate(s0, lam2, rm_true)                   # (3,3,nchan,4)
    assert cube.shape == (3, 3, lam2.shape[0], 4), cube.shape
    # S0 and S3 must be untouched by Faraday rotation (only the linear plane turns)
    assert np.allclose(cube[..., 0], 1.5) and np.allclose(cube[..., 3], 0.0), "Faraday rotation altered S0/S3"
    got = faraday_rm_map(lam2, cube)
    res2 = resolution_fwhm(lam2)
    assert np.max(np.abs(got["rm"] - rm_true)) < 0.25 * res2, "RM map off: %r vs %r" % (got["rm"], rm_true)
    # the recovered intrinsic angle map matches too (mod pi)
    dang = (got["angle0"] - ang0 + np.pi / 2) % np.pi - np.pi / 2
    assert np.max(np.abs(dang)) < np.deg2rad(3.0), "intrinsic angle map off by %.2f deg" % np.rad2deg(np.max(np.abs(dang)))
    # determinism of the forward model
    assert np.array_equal(faraday_rotate(s0, lam2, rm_true), faraday_rotate(s0, lam2, rm_true))

    # KEPT NEGATIVE (loud): the recovered amplitude is biased LOW by RMTF sidelobes on a dirty
    # spectrum; the ~3% tolerance above is that bias, not slop. Removing it needs RM-CLEAN
    # (deconvolve the RMTF) -- a declared future extension, not silently assumed here.
    print("holographic_rmsynth selftest OK  |  res=%.2f rad/m^2  peak_rm=%.3f (true %.1f)  |  KEPT NEGATIVE: dirty-spectrum amplitude biased low, needs RM-CLEAN" % (res, pk["rm"], phi0))


if __name__ == "__main__":
    _selftest()

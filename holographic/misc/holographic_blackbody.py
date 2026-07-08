"""holographic_blackbody.py -- T3: a BLACKBODY RADIATOR. Temperature (kelvin) -> the colour it glows.

WHY THIS EXISTS (thermodynamics foundation, item T3)
----------------------------------------------------
Hot things glow, and the colour tells you the temperature: a dull red ember (~900 K), orange flame (~1300 K),
a yellow-white filament (~2800 K), daylight (~6500 K), a blue-white star (~10000 K). This module turns a
temperature into that colour, from first principles, so the LATER process items can paint what they compute:
M6 (combustion) colours embers and flame tips by their temperature, and M7 (burning objects) makes glowing
char fade as it cools -- both just call blackbody_rgb(temperature).

HOW IT WORKS (real physics, not a lookup table)
-----------------------------------------------
  1. PLANCK'S LAW gives the spectral radiance of a blackbody at temperature T and wavelength lambda:
        B(lambda, T) = (2 h c^2 / lambda^5) / (exp(h c / (lambda k T)) - 1)
     -- the amount of light emitted at each colour. Cooler bodies peak in the red/infrared, hotter in the blue
     (Wien's displacement law falls straight out of this).
  2. The eye's response is the CIE 1931 colour-matching functions xbar/ybar/zbar. We use Wyman, Sloan &
     Shirley's (2013) multi-lobe GAUSSIAN approximations to those curves -- fully analytic, no data tables, so
     the whole module stays readable and dependency-free. Integrating Planck against them over the visible band
     gives the CIE tristimulus (X, Y, Z).
  3. XYZ -> linear sRGB (the standard matrix) -> gamma-encode -> clip to [0,1]. Normalising by luminance Y keeps
     the HUE (what we want) while letting the caller pick brightness.

HONEST SCOPE (kept negative): an ideal blackbody (emissivity 1) rendered to sRGB. Real flames add spectral line
emission (sodium orange, blue CH/C2 bands) that a pure blackbody misses; a real material's emissivity < 1 shifts
things slightly. This is the thermal-continuum colour -- correct and sufficient for embers/filament/star glow,
not a spectroscopic flame model. NumPy + stdlib only; deterministic.
"""
import numpy as np

# physical constants (SI) -- CODATA, the values astropy would carry
_H = 6.62607015e-34      # Planck constant, J*s
_C = 2.99792458e8        # speed of light, m/s
_KB = 1.380649e-23       # Boltzmann constant, J/K


def planck_radiance(wavelength_m, temp_K):
    """Planck's law: spectral radiance (W*sr^-1*m^-3) of a blackbody at `temp_K`, per wavelength. Vectorised over
    wavelength. This is THE emission curve -- everything else integrates against it."""
    lam = np.asarray(wavelength_m, float)
    # (2hc^2/lam^5) / (exp(hc/(lam k T)) - 1). Guard the exponent so very small lam/T don't overflow.
    x = _H * _C / (lam * _KB * float(temp_K))
    return (2.0 * _H * _C ** 2 / lam ** 5) / np.expm1(np.clip(x, 1e-9, 700.0))


def _gaussian(x, mu, s1, s2):
    """A piecewise (asymmetric) Gaussian: different width below and above the peak -- the shape Wyman et al. use to
    fit each lobe of the CIE curves."""
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) * s) ** 2)


def _cie_xyz_bar(lam_nm):
    """Analytic CIE 1931 colour-matching functions (Wyman, Sloan & Shirley 2013, multi-lobe Gaussian fit). Input
    wavelength in nm; returns (xbar, ybar, zbar). Readable and table-free -- the point of using this fit."""
    x = (_gaussian(lam_nm, 442.0, 0.0624, 0.0374) * 0.362
         + _gaussian(lam_nm, 599.8, 0.0264, 0.0323) * 1.056
         + _gaussian(lam_nm, 501.1, 0.0490, 0.0382) * -0.065)
    y = (_gaussian(lam_nm, 568.8, 0.0213, 0.0247) * 0.821
         + _gaussian(lam_nm, 530.9, 0.0613, 0.0322) * 0.286)
    z = (_gaussian(lam_nm, 437.0, 0.0845, 0.0278) * 1.217
         + _gaussian(lam_nm, 459.0, 0.0385, 0.0725) * 0.681)
    return x, y, z


# linear-sRGB <- CIE XYZ (D65), the standard IEC 61966-2-1 matrix
_XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])


def _gamma_encode(linear):
    """Linear light -> sRGB display values (the standard piecewise gamma). This is why a computed colour looks
    right on screen instead of too dark."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def blackbody_rgb(temp_K, normalize="hue", samples=90):
    """The sRGB colour a blackbody at `temp_K` glows, in [0,1]. Integrates Planck against the analytic CIE curves
    over the visible band (380-780 nm), converts XYZ->sRGB, and gamma-encodes.

    normalize='hue' (default) scales by luminance so you get the pure HUE at full value (good for tinting an
    ember/flame by temperature); normalize='none' keeps the raw luminance ratio (dim red glow stays dim). Below
    ~800 K a blackbody emits almost no visible light -> the colour goes to near-black under 'none'."""
    lam_nm = np.linspace(380.0, 780.0, int(samples))
    lam_m = lam_nm * 1e-9
    B = planck_radiance(lam_m, temp_K)
    xb, yb, zb = _cie_xyz_bar(lam_nm)
    # integrate (Riemann sum over the sampled band) -> CIE tristimulus
    X = float(np.sum(B * xb)); Y = float(np.sum(B * yb)); Z = float(np.sum(B * zb))
    if normalize == "hue":
        s = max(X + Y + Z, 1e-30)                                   # chromaticity: divide out overall brightness
        X, Y, Z = X / s, Y / s, Z / s
        rgb = _XYZ_TO_RGB @ np.array([X, Y, Z])
        rgb = rgb / max(rgb.max(), 1e-9)                            # lift the hue to full value
    else:
        rgb = _XYZ_TO_RGB @ np.array([X, Y, Z])
        rgb = rgb / 1.5e13                                          # a fixed scale so hotter really reads brighter
    rgb = np.clip(rgb, 0.0, 1.0)                                    # gamut-clip negative (out-of-sRGB) components
    return _gamma_encode(rgb)


def peak_wavelength_nm(temp_K):
    """Wien's displacement law: the wavelength (nm) where the Planck curve peaks. b = 2.897771955e-3 m*K.
    A quick sanity handle -- 5800 K (the Sun) peaks ~500 nm (green), an ember ~2900 K peaks in the near-IR."""
    return 2.897771955e-3 / float(temp_K) * 1e9


def _selftest():
    """The colour marches red -> orange -> white -> blue with temperature, matching what hot things actually look
    like, and Wien's law places the peak correctly."""
    cool = blackbody_rgb(1000.0)     # ember
    warm = blackbody_rgb(2800.0)     # incandescent filament
    day = blackbody_rgb(6500.0)      # daylight
    hot = blackbody_rgb(12000.0)     # blue-white star

    # an ember is red-dominant: R clearly the largest channel
    assert cool[0] > cool[1] > cool[2], cool
    # daylight is near-neutral: channels close together
    assert (day.max() - day.min()) < 0.35, day
    # the blue channel RISES from ember to hot star; the red/blue ratio FALLS as it heats
    assert hot[2] > warm[2] > cool[2], (cool[2], warm[2], hot[2])
    assert (cool[0] / (cool[2] + 1e-6)) > (hot[0] / (hot[2] + 1e-6))
    # Wien: the Sun (5772 K) peaks in the visible ~500 nm; an ember peaks in the IR (> 780 nm)
    assert 480 < peak_wavelength_nm(5772) < 520
    assert peak_wavelength_nm(1000) > 780
    # deterministic
    assert np.array_equal(blackbody_rgb(3000.0), blackbody_rgb(3000.0))
    print("holographic_blackbody selftest OK: ember=%s daylight=%s star=%s; red->white->blue with T, Wien peak "
          "correct" % (tuple(cool.round(2)), tuple(day.round(2)), tuple(hot.round(2))))


if __name__ == "__main__":
    _selftest()

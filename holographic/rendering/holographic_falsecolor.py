"""holographic_falsecolor.py -- FALSE COLOUR: show a human what a non-human sensor sees (leCore rendering).

WHY THIS EXISTS
---------------
An observer (holographic_observer) can read channels a human eye cannot: ultraviolet, the ANGLE of linear
polarization, the HANDEDNESS of circular polarization. To let a person actually SEE those, we must map the
invisible channels onto the three a human has (R, G, B). That mapping is the deliverable of the mantis-shrimp
arc -- "see what the mantis sees" -- and it is also the honest one: every choice here is a CHOICE, not neutral
truth. As Eno's dissent on the panel puts it, what counts as signal is a choice of manifold; a false-colour
image is an aesthetic decision, so each function states plainly what its output channels are a FUNCTION OF and
which knob is arbitrary. This module never pretends its pictures are "the real colour" of anything.

It is general: any multi-channel sensor readings (a mantis eye, a telescope's bands) can be false-coloured. It
REUSES the engine's own colour path -- blackbody's CIE curves and the observer's XYZ->sRGB -- so a spectral
band's display colour is consistent with everything else the engine draws.

DIRECTIONS (up/down/sideways)
  DOWN  -- maps a single reading vector (one pixel) -- native.
  UP    -- field-native: an image of readings (...,nchan) -> an RGB image (...,3), all vectorised.
  SIDEWAYS
    field    -- the RGB image itself.
    structure-- the band->colour assignment is a small legend record (band_display_colors).
    program  -- the maps are elementwise; emit_kernel could project them. DECLARED not-yet-wired.

Determinism: pure numpy, no RNG. Exact.
"""

import numpy as np
from holographic.misc import holographic_blackbody as _bb          # CIE curves -> a wavelength's display colour
from holographic.rendering import holographic_observer as _ob       # its XYZ->sRGB, and the mantis band centres


def hsv_to_rgb(h, s, v):
    """Vectorised HSV -> RGB, all args in [0,1] and broadcastable. The standard hexcone conversion; hue wraps.
    Used because hue is the natural home for a CYCLIC quantity like a polarization ANGLE (0 and pi are the same
    e-vector, and hue is a circle) -- that is the whole reason polarization false-colour uses HSV."""
    h = np.asarray(h, float) % 1.0
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    v = np.asarray(v, float)
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i % 6).astype(int)
    # np.choose picks, per element, the right sector formula -- the six faces of the HSV hexcone.
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def wavelength_to_rgb(nm):
    """The approximate sRGB a human sees for a MONOCHROMATIC light at wavelength `nm`. A single wavelength has
    XYZ equal to the CIE colour-matching values there, so we reuse blackbody's exact CIE curves and the
    observer's XYZ->sRGB (mode 'hue', full-value chromaticity). Outside the eye's ~380-780 nm range the curves
    are ~0, so UV/IR correctly map to BLACK -- they are invisible; that is the true-colour fact the false-colour
    maps below deliberately override. Field-native over an array of wavelengths."""
    nm = np.asarray(nm, float)
    xb, yb, zb = _bb._cie_xyz_bar(nm)
    xyz = np.stack([xb, yb, zb], axis=-1)
    rgb = _ob.to_srgb(xyz, mode="hue")
    # Force black outside the visible band (the CMF fit has small tails there that are not real sensitivity).
    visible = (nm >= 380.0) & (nm <= 780.0)
    return rgb * visible[..., None]


def band_display_colors(centers_nm, uv_hue=0.80, ir_hue=0.0):
    """Assign each spectral band a DISPLAY colour: visible bands get their true-ish wavelength colour; UV bands
    (< 380 nm) get a CHOSEN false hue so ultraviolet becomes visible at all (default violet, uv_hue in [0,1]);
    IR bands (> 780 nm) get a chosen deep-red hue. Returns (nchan, 3).

    THE CHOICE, stated plainly: mapping UV -> violet is arbitrary -- UV has no human colour. We pick a hue that
    reads as 'short wavelength / high energy'; change uv_hue to pick another. This is what makes a UV-bright
    object glow instead of vanishing."""
    c = np.asarray(centers_nm, float)
    rgb = wavelength_to_rgb(c)
    uv = c < 380.0
    ir = c > 780.0
    if np.any(uv):
        rgb[uv] = hsv_to_rgb(uv_hue, 1.0, 1.0)
    if np.any(ir):
        rgb[ir] = hsv_to_rgb(ir_hue, 1.0, 0.6)
    return rgb


def spectral_falsecolor(readings, centers_nm, uv_hue=0.80):
    """Collapse N spectral-band readings (..., nchan) into an RGB image (..., 3) by summing each band's DISPLAY
    colour weighted by its reading, then lifting each pixel to full value. UV bands glow in the chosen uv_hue,
    so 'what the mantis sees' includes light a human eye is blind to.

    CHOICE: the output colour is a FUNCTION OF the band readings and the (arbitrary) UV/IR hue assignment; it is
    a legible summary, not a measurement of 'the real colour'. Per-pixel value-lift means relative brightness
    between pixels is not preserved (hue-forward, like blackbody's 'hue' mode)."""
    readings = np.asarray(readings, float)
    band_rgb = band_display_colors(centers_nm, uv_hue=uv_hue)      # (nchan, 3)
    img = readings @ band_rgb                                      # (..., 3) weighted colour sum
    peak = np.max(img, axis=-1, keepdims=True)
    return np.clip(img / np.maximum(peak, 1e-9), 0.0, 1.0)


def polarization_falsecolor(evector_angle, dolp, value=1.0):
    """The standard polarization false-colour: HUE = e-vector angle, SATURATION = degree of linear polarization,
    VALUE = intensity. Unpolarised light (dolp 0) is therefore grey; strongly polarised light is vivid, its hue
    telling you the orientation. Field-native.

    CHOICE: which angle maps to which hue is arbitrary (we use angle/pi around the wheel); the image shows the
    STRUCTURE of the polarization field, not a real colour. Output is a FUNCTION OF (angle -> hue, dolp -> sat,
    intensity -> value)."""
    hue = (np.asarray(evector_angle, float) / np.pi) % 1.0         # angle in [0,pi) -> full hue circle
    return hsv_to_rgb(hue, dolp, value)


def handedness_falsecolor(circular_R, circular_L):
    """Diverging false-colour for CIRCULAR polarization sense -- the channel the mantis uniquely reads. Right-
    handed light -> red, left-handed -> blue, unpolarised -> white; saturation grows with the imbalance. Field-
    native.

    CHOICE: red=right / blue=left is a convention (like a coolwarm map), not physics. Output is a FUNCTION OF the
    normalised imbalance (R-L)/(R+L)."""
    R = np.asarray(circular_R, float)
    L = np.asarray(circular_L, float)
    net = (R - L) / np.maximum(R + L, 1e-12)                       # in [-1, 1]: +1 fully RCP, -1 fully LCP
    hue = np.where(net >= 0.0, 0.0, 2.0 / 3.0)                     # red for right, blue for left
    return hsv_to_rgb(hue, np.abs(net), 1.0)                       # |net| -> saturation; net 0 -> white


def mantis_falsecolor(view, centers_nm=None):
    """Turn a mantis_view() reading into three images a human can look at, each showing one thing the mantis
    perceives and we do not: `color` (12 bands incl. UV made visible), `polarization` (e-vector angle+strength),
    and `handedness` (circular sense). The headline "see what the mantis sees" deliverable.

    CHOICE (loud): all three are false-colour DECISIONS, not the mantis' qualia -- we cannot show a human a UV or
    a handedness percept, only re-map it into channels a human has. Returns a dict of RGB arrays. If `centers_nm`
    is omitted, the observer's own mantis receptor centres are used so the colour matches the readings."""
    if centers_nm is None:
        centers_nm = _ob._MANTIS_CENTERS_NM
    color = spectral_falsecolor(view["spectral"], centers_nm)
    # Degree of linear polarization from the orthogonal linear detector pairs (Q=l0-l90, U=l45-l135).
    l0 = np.asarray(view["linear_0"], float); l90 = np.asarray(view["linear_90"], float)
    l45 = np.asarray(view["linear_45"], float); l135 = np.asarray(view["linear_135"], float)
    Q = l0 - l90; U = l45 - l135; I = l0 + l90
    dolp = np.sqrt(Q * Q + U * U) / np.maximum(I, 1e-12)
    pol = polarization_falsecolor(view["evector_angle"], dolp)
    hand = handedness_falsecolor(view["circular_R"], view["circular_L"])
    return {"color": color, "polarization": pol, "handedness": hand}


def _selftest():
    """Regression trap: the maps must make the invisible visible in the agreed directions, be field-native, and
    stay deterministic. Every assertion checks a CHOICE behaves as documented, not that it is 'true colour'."""
    # --- hsv corners ---
    assert np.allclose(hsv_to_rgb(0.0, 1.0, 1.0), [1, 0, 0]), "hue 0 should be red"
    assert np.allclose(hsv_to_rgb(1.0 / 3.0, 1.0, 1.0), [0, 1, 0]), "hue 1/3 should be green"
    assert np.allclose(hsv_to_rgb(2.0 / 3.0, 1.0, 1.0), [0, 0, 1]), "hue 2/3 should be blue"
    assert np.allclose(hsv_to_rgb(0.0, 0.0, 1.0), [1, 1, 1]), "sat 0 should be white"

    # --- wavelength colours: long=red-dominant, short=blue-dominant, UV=black ---
    assert wavelength_to_rgb(660.0)[0] == wavelength_to_rgb(660.0).max(), "660nm should be red-dominant"
    assert wavelength_to_rgb(460.0)[2] == wavelength_to_rgb(460.0).max(), "460nm should be blue-dominant"
    assert np.allclose(wavelength_to_rgb(320.0), 0.0), "UV must be black in true colour"

    # --- spectral false-colour: UV-only reading becomes VISIBLE (the whole point) ---
    centers = _ob._MANTIS_CENTERS_NM
    uv_only = np.zeros(len(centers)); uv_only[0] = 1.0            # band 0 is ~315 nm (UV)
    rgb_uv = spectral_falsecolor(uv_only, centers)
    assert rgb_uv.max() > 0.1, "UV band did not become visible under false colour"
    red_only = np.zeros(len(centers)); red_only[-1] = 1.0         # band 11 is ~660 nm
    rgb_red = spectral_falsecolor(red_only, centers)
    assert rgb_red[0] >= rgb_red[2], "red band should read red-dominant"

    # --- polarization false-colour: orientation -> hue; unpolarised -> grey ---
    c0 = polarization_falsecolor(0.0, 1.0); c90 = polarization_falsecolor(np.pi / 2, 1.0)
    assert not np.allclose(c0, c90), "different e-vector angles must give different colours"
    grey = polarization_falsecolor(0.7, 0.0)
    assert abs(grey[0] - grey[1]) < 1e-9 and abs(grey[1] - grey[2]) < 1e-9, "unpolarised must be grey"

    # --- handedness: right->red, left->blue, none->white ---
    assert handedness_falsecolor(1.0, 0.0)[0] > handedness_falsecolor(1.0, 0.0)[2], "RCP should be red-dominant"
    assert handedness_falsecolor(0.0, 1.0)[2] > handedness_falsecolor(0.0, 1.0)[0], "LCP should be blue-dominant"
    assert np.allclose(handedness_falsecolor(0.5, 0.5), [1, 1, 1]), "no net handedness should be white"

    # --- UP: field-native over an image of readings ---
    img_readings = np.stack([uv_only, red_only, np.ones(len(centers))]).reshape(1, 3, len(centers))
    rgb_img = spectral_falsecolor(img_readings, centers)
    assert rgb_img.shape == (1, 3, 3), rgb_img.shape

    # --- end to end through a real mantis_view (RCP, UV-bright) ---
    lam = np.linspace(300.0, 720.0, 140)
    S = np.zeros(lam.shape + (4,)); S[..., 0] = np.exp(-0.5 * ((lam - 330.0) / 20.0) ** 2); S[..., 3] = S[..., 0]
    view = _ob.mantis_view(S, lam)
    fc = mantis_falsecolor(view, None)
    assert fc["color"].shape[-1] == 3 and fc["color"].max() > 0.1, "UV-bright mantis view should glow"
    assert fc["handedness"][0] > fc["handedness"][2], "RCP mantis view should read red in the handedness map"

    # --- determinism ---
    assert np.array_equal(spectral_falsecolor(uv_only, centers), spectral_falsecolor(uv_only, centers))

    print("holographic_falsecolor selftest OK  |  UV made visible, e-vector->hue, handedness->diverging; field-"
          "native; mantis_view end-to-end  |  NOTE: every map is a CHOICE (Eno), not the mantis' true percept")


if __name__ == "__main__":
    _selftest()

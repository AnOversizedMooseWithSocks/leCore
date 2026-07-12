"""Cross-faculty integration for the merged polarization/optics/sensor stack (backlog items 4/7): the same Stokes core
reads a mantis eye AND a radio telescope. Proves the chain stokes -> mueller -> faraday -> rm_synthesis -> falsecolor,
plus observer (spectrum->readings) and skydata (cube + WCS -> faraday_rm_map), through the mind."""
import os
import tempfile

import numpy as np
import lecore


def test_quarter_wave_plate_linear_to_circular():
    """A Mueller quarter-wave plate turns linear polarization into circular (the mantis R8 mechanism). stokes ->
    mueller -> apply is the optics pipeline."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    s_lin = np.array(m.stokes_linear(1.0, angle=np.pi / 4))   # linear at 45 deg (S2 arm)
    qwp = m.mueller_matrix("retarder", delta=np.pi / 2, angle=0.0)
    s_out = np.asarray(m.apply_mueller(qwp, s_lin))
    assert abs(abs(s_out[3]) - 1.0) < 1e-6, "QWP should produce fully circular light (|V|=1), got %.3f" % s_out[3]


def test_faraday_forward_inverse_recovers_rm():
    """faraday rotation forward model + rm_synthesis inverse recover the rotation measure -- the telescope-as-observer
    loop. A measured round-trip, not a painted-on answer."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    lam2 = np.linspace(0.03, 0.24, 160)
    rm_true = 42.0
    P = 2.0 * np.exp(2j * (0.3 + rm_true * lam2))
    phi = m.rm_phi_grid(lam2)
    F = m.rm_synthesis(lam2, phi, P=P)
    assert abs(m.rm_peak(F, phi)["rm"] - rm_true) < 2.0, "RM synthesis must recover the injected RM"


def test_skydata_cube_to_rm_map():
    """skydata (cube + world axes) feeds faraday_rm_map: a per-pixel line-of-sight magnetism map recovered in one
    call from a Stokes cube -- the field costume of the same core."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    freq = m.make_sky_axis("freq", 12, "Hz", crval=1e9, cdelt=5e7)
    data = np.zeros((2, 2, 12, 4)); data[..., 0] = 1.0; data[..., 1] = 1.0
    sky = m.make_skydata(data, [m.make_sky_axis("y", 2, "deg"), m.make_sky_axis("x", 2, "deg"),
                                freq, m.make_sky_axis("stokes", 4, "")])
    lam2 = m.sky_lambda2(sky)
    rm_true = np.array([[10.0, -30.0], [55.0, -5.0]])
    s_base = np.zeros((2, 2, 4)); s_base[..., 0] = 1.0; s_base[..., 1] = 1.0
    cube = m.faraday_rotate(s_base, lam2, rm_true)
    rec = m.faraday_rm_map(lam2, cube)
    assert np.allclose(np.round(rec["rm"]), rm_true, atol=1.0), "per-pixel RM map should match the injected field"


def test_skydata_save_load_roundtrip():
    """Deterministic save/load (json header + npy, no pickle) round-trips a sky observation byte-for-byte."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    data = np.arange(2 * 2 * 5, dtype=float).reshape(2, 2, 5)
    sky = m.make_skydata(data, [m.make_sky_axis("y", 2, "deg"), m.make_sky_axis("x", 2, "deg"),
                                m.make_sky_axis("freq", 5, "Hz", crval=1e9, cdelt=2e8)])
    d = tempfile.mkdtemp()
    m.save_skydata(sky, os.path.join(d, "sky"))
    sky2 = m.load_skydata(os.path.join(d, "sky"))
    a = sky["data"] if isinstance(sky, dict) else sky.data
    b = sky2["data"] if isinstance(sky2, dict) else sky2.data
    assert np.array_equal(a, b), "skydata data must round-trip exactly"


def test_observer_reproduces_blackbody():
    """The human observer on a Planck spectrum reproduces blackbody_rgb byte-identically -- blackbody IS this observer
    on a blackbody spectrum (the sensor unifier: one core, many sensors)."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    from holographic.misc import holographic_blackbody as bb
    lam = np.linspace(380, 780, 90)
    got = m.spectrum_to_rgb(bb.planck_radiance(lam * 1e-9, 5000.0))
    assert np.array_equal(np.asarray(got), np.asarray(bb.blackbody_rgb(5000.0))), \
        "observer on a Planck spectrum must equal blackbody_rgb"


def test_mantis_sees_circular_polarization():
    """The mantis view reads a spectral+polarized field into 12 bands plus handedness -- the sense the mantis uniquely
    has. observer + mueller composed, field-native. falsecolor then makes it human-visible."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    L = np.linspace(300, 720, 140)
    b = np.exp(-0.5 * ((L - 500) / 60) ** 2)
    S = np.zeros(L.shape + (4,)); S[..., 0] = b; S[..., 3] = b   # circularly polarized band
    view = m.mantis_view(S, L)
    assert "handedness_sign" in view
    fc = m.mantis_falsecolor(view)
    assert float(np.asarray(fc["color"]).max()) > 0.0, "false-colour image should be non-blank"

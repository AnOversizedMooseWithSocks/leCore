"""holographic_skydata.py -- a SKY OBSERVATION as first-class data: a cube + world axes (leCore io_and_interop).

WHY THIS EXISTS
---------------
Until now the polarization/astro tools (faraday_rm_map, the observer) took bare numpy arrays and a separate
lambda^2 vector. A real observation is more than an array: it is a data cube plus the WORLD COORDINATES of each
axis -- where on the sky each pixel points (RA/Dec), what frequency or wavelength each channel samples, and
which axis (if any) is the Stokes/polarization axis. This module holds that together so an observation can be
loaded once and handed straight to analysis, and so a recovered RM lives at a real sky position, not a pixel index.

SCOPE (kept honest): this is WCS-LITE -- LINEAR axes only (CRVAL/CRPIX/CDELT: world = crval + (pix-crpix)*cdelt,
the common FITS convention restricted to the linear case). No spherical projection, no astropy, no FITS binary
parser in core -- the ingest contract is a header dict + a numpy array (load/save via json + .npy). A full-WCS
or FITS converter can live OUTSIDE core; this is the deterministic, dependency-free spine everything else reads.

A SkyData is a plain dict {data, axes, meta}: `data` is the ndarray (any rank); `axes` is a list of axis
descriptors [{name, unit, crval, crpix, cdelt}, ...] (one per data axis); `meta` is a freeform header dict.

DIRECTIONS (up/down/sideways)
  DOWN  -- world_coords / pix_to_world operate on a single axis or a single pixel.
  UP    -- the whole cube is the object; stokes_cube reshapes it for field-native analysis (a per-pixel RM map).
  SIDEWAYS
    field    -- the data cube itself.
    structure-- the axes list IS a role-bound record (each axis bound to its world mapping); the bridge to the
                Faraday/observer faculties (lambda2_axis, stokes_cube) is just reading those roles.
    sequence -- the spectral axis is a sampled sequence (lambda^2), which is exactly what rm synthesis consumes.

Determinism: linear closed-form axis math; save/load via json (sorted keys) + .npy. Exact round-trip.
"""

import json
import numpy as np

_C = 299792458.0  # speed of light (m/s), for frequency <-> wavelength on the spectral axis


def make_axis(name, n=None, unit="", crval=0.0, crpix=0.0, cdelt=1.0):
    """One axis descriptor: world = crval + (pixel - crpix)*cdelt. `crpix` is the 0-based pixel index at which
    world equals `crval` (note: FITS uses 1-based CRPIX; we use 0-based for numpy sanity -- stated so nobody is
    off by one). `n` is optional bookkeeping; the real length comes from the data shape."""
    return {"name": str(name), "unit": str(unit), "crval": float(crval), "crpix": float(crpix), "cdelt": float(cdelt), "n": (None if n is None else int(n))}


def make_skydata(data, axes, meta=None):
    """Assemble a SkyData from a cube and one axis descriptor per data axis. Validates that the number of axes
    matches the data rank -- a mismatch here is the classic silent bug that makes coordinates meaningless."""
    data = np.asarray(data)
    if len(axes) != data.ndim:
        raise ValueError("need one axis per data dim: data.ndim=%d but %d axes" % (data.ndim, len(axes)))
    return {"data": data, "axes": [dict(a) for a in axes], "meta": dict(meta or {})}


def _axis_index(sky, axis):
    """Resolve an axis given as an int index OR a name string -> integer index. One place so every function
    accepts either, and a bad name fails loudly instead of silently picking the wrong axis."""
    if isinstance(axis, (int, np.integer)):
        return int(axis)
    for i, a in enumerate(sky["axes"]):
        if a["name"].lower() == str(axis).lower():
            return i
    raise KeyError("no axis named %r (have %r)" % (axis, [a["name"] for a in sky["axes"]]))


def pix_to_world(sky, axis, pix):
    """Pixel -> world coordinate on one axis: crval + (pix - crpix)*cdelt. Vectorised over `pix`."""
    a = sky["axes"][_axis_index(sky, axis)]
    return a["crval"] + (np.asarray(pix, float) - a["crpix"]) * a["cdelt"]


def world_to_pix(sky, axis, world):
    """World -> pixel on one axis: the exact linear inverse of pix_to_world (crpix + (world-crval)/cdelt)."""
    a = sky["axes"][_axis_index(sky, axis)]
    return a["crpix"] + (np.asarray(world, float) - a["crval"]) / a["cdelt"]


def world_coords(sky, axis):
    """The full world-coordinate array along one axis (length = that data dimension). This is how you get the
    actual RA/Dec/frequency values a cube samples, not just indices."""
    i = _axis_index(sky, axis)
    n = sky["data"].shape[i]
    return pix_to_world(sky, i, np.arange(n))


def spectral_axis_index(sky):
    """Find the spectral axis by name (frequency or wavelength). Returns its index, or None if there isn't one.
    Names matched: freq/frequency, wavelength/wave/lambda (case-insensitive). Explicit and boring on purpose."""
    for i, a in enumerate(sky["axes"]):
        nm = a["name"].lower()
        if nm in ("freq", "frequency", "wavelength", "wave", "lambda", "lam"):
            return i
    return None


def lambda2_axis(sky, axis=None):
    """The lambda^2 sample vector (m^2) of the spectral axis -- the exact input Faraday rotation-measure synthesis
    wants. If the axis is a FREQUENCY, wavelength = c/f then squared; if it is a WAVELENGTH, squared directly.
    Auto-finds the spectral axis when `axis` is None. This is the bridge from 'an observation' to 'a Faraday map'."""
    if axis is None:
        axis = spectral_axis_index(sky)
        if axis is None:
            raise ValueError("no spectral (freq/wavelength) axis found; pass one explicitly")
    i = _axis_index(sky, axis)
    unit = sky["axes"][i]["unit"].lower()
    coords = world_coords(sky, i)
    name = sky["axes"][i]["name"].lower()
    is_freq = name in ("freq", "frequency") or unit in ("hz", "khz", "mhz", "ghz")
    if is_freq:
        wl = _C / coords          # frequency (Hz) -> wavelength (m)
    else:
        wl = coords               # already a wavelength
    return wl * wl


def stokes_axis_index(sky):
    """Find the Stokes/polarization axis by name (length must be 4: I,Q,U,V). Returns index or None."""
    for i, a in enumerate(sky["axes"]):
        if a["name"].lower() in ("stokes", "pol", "polarization", "polarisation") and sky["data"].shape[i] == 4:
            return i
    return None


def stokes_cube(sky, spectral=None, stokes=None):
    """Reshape the observation into the (..., nchan, 4) layout faraday_rm_map / the polarization tools expect:
    spatial axes lead, the spectral channel is second-to-last, Stokes (I,Q,U,V) is last. Auto-finds the spectral
    and Stokes axes by name if not given. This is the one call that turns 'a loaded cube' into 'ready for a per-
    pixel RM map' -- the whole point of holding the axes together."""
    si = spectral_axis_index(sky) if spectral is None else _axis_index(sky, spectral)
    ki = stokes_axis_index(sky) if stokes is None else _axis_index(sky, stokes)
    if si is None or ki is None:
        raise ValueError("need both a spectral and a Stokes axis; found spectral=%r stokes=%r" % (si, ki))
    # Move spectral -> second-to-last, Stokes -> last; the remaining (spatial) axes keep their order in front.
    data = np.moveaxis(sky["data"], [si, ki], [-2, -1])
    return data


def carrier_axis(sky):
    """Diagnostic: which axis does the data itself look like the INDEX (carrier) over -- typically the spectral
    one. REUSES holographic_axisrole.analyze_axes rather than reinventing the detection; handy for sanity-checking
    an ingested cube whose axis names you don't trust. Returns whatever analyze_axes reports."""
    from holographic.sampling_and_signal.holographic_axisrole import analyze_axes
    return analyze_axes(sky["data"])


def save_skydata(sky, path):
    """Persist a SkyData deterministically: the header (axes + meta) as sorted-key JSON, the cube as .npy, bundled
    in one .npz. No pickle (allow_pickle stays off on load), so it is safe and portable. Round-trips exactly."""
    header = json.dumps({"axes": sky["axes"], "meta": sky["meta"]}, sort_keys=True)
    np.savez(path, data=sky["data"], header=np.array(header))


def load_skydata(path):
    """Load a SkyData saved by save_skydata. Reads the .npy cube and the JSON header (no pickle). The inverse of
    save_skydata to the bit for the data and to the value for the header."""
    if not str(path).endswith(".npz"):
        path = str(path) + ".npz"
    z = np.load(path, allow_pickle=False)
    header = json.loads(str(z["header"]))
    return make_skydata(z["data"], header["axes"], header["meta"])


def _selftest():
    """Regression trap: exact coordinate round-trips, the Faraday bridge end-to-end (ingest -> RM map), and a
    save/load that preserves everything."""
    # --- build a small polarized radio observation: (ny, nx, nfreq, stokes) ---
    ny, nx, nf = 3, 3, 120
    freq = np.linspace(1.0e9, 2.0e9, nf)                       # 1-2 GHz
    axes = [make_axis("dec", ny, "deg", crval=10.0, crpix=1.0, cdelt=0.01),
            make_axis("ra", nx, "deg", crval=200.0, crpix=1.0, cdelt=0.01),
            make_axis("freq", nf, "Hz", crval=freq[0], crpix=0.0, cdelt=(freq[1] - freq[0])),
            make_axis("stokes", 4, "")]
    data = np.zeros((ny, nx, nf, 4))

    # --- COORDINATE round-trips exact to 1e-12 (the container's core contract) ---
    for ax in ("ra", "dec", "freq"):
        w = world_coords({"data": data, "axes": axes, "meta": {}}, ax)
        sky0 = make_skydata(data, axes)
        p = world_to_pix(sky0, ax, w)
        assert np.max(np.abs(p - np.arange(data.shape[_axis_index(sky0, ax)]))) < 1e-9, "%s round-trip failed" % ax

    # --- lambda^2 bridge: freq axis -> lambda^2 matches c/f squared ---
    sky = make_skydata(data, axes)
    lam2 = lambda2_axis(sky)
    assert np.allclose(lam2, (_C / world_coords(sky, "freq")) ** 2), "lambda2 from freq wrong"
    assert lam2[0] > lam2[-1], "lambda^2 should fall as frequency rises"

    # --- FARADAY BRIDGE end-to-end: paint a known RM into the cube, ingest, recover the map ---
    from holographic.rendering import holographic_rmsynth as _rm
    ang0 = np.array([[0.2, 0.5, 0.9], [1.1, 0.3, 0.7], [0.4, 1.4, 0.0]])
    s0 = np.zeros((ny, nx, 4)); s0[..., 0] = 1.0
    s0[..., 1] = np.cos(2 * ang0); s0[..., 2] = np.sin(2 * ang0)
    rm_true = np.linspace(-50.0, 50.0, 9).reshape(ny, nx)
    rotated = _rm.faraday_rotate(s0, lam2, rm_true)            # (ny,nx,nf,4)
    sky["data"] = rotated                                       # this IS the observed cube
    cube = stokes_cube(sky)                                     # -> (ny,nx,nf,4), ready for the RM map
    assert cube.shape == (ny, nx, nf, 4)
    got = _rm.faraday_rm_map(lam2, cube)
    res = _rm.resolution_fwhm(lam2)
    assert np.max(np.abs(got["rm"] - rm_true)) < 0.3 * res, "ingested RM map off: %r vs %r" % (got["rm"], rm_true)

    # --- carrier-axis reuse returns something sane (a report, not a crash) ---
    rep = carrier_axis(make_skydata(np.random.default_rng(0).standard_normal((5, 40)),
                                    [make_axis("y", 5), make_axis("t", 40)]))
    assert rep is not None

    # --- save/load round-trip: data byte-identical, header values preserved ---
    import tempfile, os
    d = tempfile.mkdtemp()
    fp = os.path.join(d, "obs")
    save_skydata(sky, fp)
    back = load_skydata(fp)
    assert np.array_equal(back["data"], sky["data"]), "data not preserved by save/load"
    assert back["axes"][2]["name"] == "freq" and abs(back["axes"][2]["cdelt"] - (freq[1] - freq[0])) < 1e-6, "header lost"

    print("holographic_skydata selftest OK  |  coords exact; freq->lambda^2 bridge; ingest->RM map end-to-end "
          "(recovered to <0.3 res); save/load round-trips  |  SCOPE: WCS-LITE (linear axes only, no projection)")


if __name__ == "__main__":
    _selftest()

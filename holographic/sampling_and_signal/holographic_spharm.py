"""holographic_spharm.py -- ONE spherical-harmonic primitive for directional SOUND *and* LIGHT.

WHY THIS EXISTS (Above/Below Sweep 3 -- the unification the sweep surfaced)
--------------------------------------------------------------------------
The deferred `harmonic` extension ("directional SH not implemented") looked like an audio-only gap. But real
spherical harmonics are ALREADY in the tree for light -- `holographic_prt.sh_eval` computes the l=0..3 bands that
precomputed radiance transfer (and splat view-dependent colour) run on. And directional SOUND uses the very same
basis: first-order ambisonics (B-format) IS the l=0,1 spherical-harmonic coefficients of a sound field, higher
orders sharpen it. So we build the projection/reconstruction ONCE, on prt's existing basis, and it serves BOTH:
  * LIGHT -- project a radiance/irradiance function over the sphere to SH coefficients, reconstruct at any
    direction (the PRT transfer, and the splat view-dependent lobe);
  * SOUND -- encode a source's directional gain to the same coefficients (ambisonic encoding), decode toward a
    listening direction (ambisonic decoding).
Same coefficients, same code path -- the unification only pays if the code is actually one, so this module REUSES
`prt.sh_eval` rather than forking a second SH implementation. That is the §5.1 "which module is this in a
different costume?" discipline: light transfer and ambisonic sound are the same directional-function projection.

HONEST SCOPE (kept loud): band-limited to prt's order (up to l=3 / 16 coefficients) -- a sharp directional
feature past that band is smoothed (the standard SH truncation; raise `order` to sharpen at a coefficient cost).
Least-squares projection wants reasonably-spread sample directions (use `sphere_dirs`). Deterministic; NumPy +
stdlib; the SH basis and the sampling lattice come straight from prt.
"""
import numpy as np

from holographic.misc.holographic_prt import sh_eval, _sphere_dirs

# re-export prt's deterministic near-uniform sphere sampling under a public name (both domains want it)
def sphere_dirs(n, seed=0):
    """`n` deterministic near-uniform unit directions (a spherical Fibonacci lattice) -- good sample points for
    projecting a directional function to SH. Straight from prt (shared, not forked)."""
    return _sphere_dirs(n, seed=seed)


def sh_project(dirs, values, order=3):
    """Project a directional function onto real spherical-harmonic coefficients by least squares: find `coeffs`
    so that sh_eval(dirs, order) @ coeffs best matches `values` sampled at `dirs`. `values` may be (M,) scalar
    (a mono radiance or gain) or (M, C) vector (RGB radiance, or a multi-channel signal). Returns coeffs of
    shape (order^2,) or (order^2, C). This is the SAME projection PRT uses for light and ambisonics uses for
    sound -- the shared primitive."""
    basis = sh_eval(np.asarray(dirs, float), order)          # (M, order^2) -- prt's real SH basis
    vals = np.asarray(values, float)
    coeffs, *_ = np.linalg.lstsq(basis, vals, rcond=None)     # least-squares fit (works for scalar or vector vals)
    return coeffs


def sh_reconstruct(coeffs, dirs, order=3):
    """Evaluate the SH expansion at `dirs`: sh_eval(dirs, order) @ coeffs -- the reconstructed directional value
    (radiance toward a view direction, or ambisonic gain toward a listening direction). Inverse of sh_project."""
    basis = sh_eval(np.asarray(dirs, float), order)
    return basis @ np.asarray(coeffs, float)


def sh_rotate_dc(coeffs):
    """The band-0 (DC) term -- the direction-independent average of the function (ambient light / omni sound
    level). A convenience readout; the constant SH normalization means coeffs[0] scaled by the band-0 basis."""
    c = np.asarray(coeffs, float)
    return c[0] if c.ndim == 1 else c[0]


def _selftest():
    """The SAME project/reconstruct serves light and sound: a directional LIGHT lobe and a directional SOUND gain
    both reconstruct with low error from the shared primitive; higher order sharpens; scalar and RGB both work;
    deterministic."""
    dirs = sphere_dirs(400)

    # (1) LIGHT: a cosine-lobe radiance around a bright direction (the PRT use). Project -> reconstruct.
    light_dir = np.array([0.3, 0.5, 0.8]); light_dir /= np.linalg.norm(light_dir)
    radiance = np.clip(dirs @ light_dir, 0, None) ** 2                     # a soft directional highlight
    coeffs_l = sh_project(dirs, radiance, order=4)
    recon_l = sh_reconstruct(coeffs_l, dirs, order=4)
    err_l = float(np.sqrt(np.mean((recon_l - radiance) ** 2)) / (radiance.std() + 1e-9))
    assert err_l < 0.25, err_l                                            # SH captures the lobe well

    # (2) SOUND: an ambisonic directional gain for a source (the audio use) -- the SAME primitive, no fork.
    src_dir = np.array([-0.6, 0.2, 0.7]); src_dir /= np.linalg.norm(src_dir)
    gain = np.clip(dirs @ src_dir, 0, None)                               # cardioid-ish directional gain
    coeffs_s = sh_project(dirs, gain, order=4)
    recon_s = sh_reconstruct(coeffs_s, dirs, order=4)
    err_s = float(np.sqrt(np.mean((recon_s - gain) ** 2)) / (gain.std() + 1e-9))
    assert err_s < 0.25, err_s                                           # same code reconstructs the sound field

    # (3) higher order sharpens (lower error) for a tight feature
    tight = np.clip(dirs @ light_dir, 0, None) ** 8
    e2 = np.sqrt(np.mean((sh_reconstruct(sh_project(dirs, tight, 2), dirs, 2) - tight) ** 2))
    e4 = np.sqrt(np.mean((sh_reconstruct(sh_project(dirs, tight, 4), dirs, 4) - tight) ** 2))
    assert e4 < e2                                                       # more bands = sharper

    # (4) RGB (vector-valued) radiance also works from the same path
    rgb = np.stack([radiance, gain, 0.5 * radiance], axis=1)             # (M, 3)
    cc = sh_project(dirs, rgb, order=3)
    assert cc.shape == (9, 3)
    rr = sh_reconstruct(cc, dirs, order=3)
    assert rr.shape == (len(dirs), 3)

    # (5) deterministic
    assert np.allclose(sh_project(dirs, radiance, 3), sh_project(dirs, radiance, 3))
    print("holographic_spharm selftest OK: one shared SH primitive projects+reconstructs a directional LIGHT lobe "
          "(err %.2f) and a directional SOUND gain (err %.2f) from the SAME prt basis; higher order sharpens; "
          "scalar and RGB both work; deterministic" % (err_l, err_s))


if __name__ == "__main__":
    _selftest()

"""holographic_surfaceint.py -- SURFACE-FROM-GRADIENT by FFT (Frankot-Chellappa) -- inverse-rendering IR7.

IR1 goes height -> normal (take a gradient); IR7 is the INVERSE -- recover a single-valued, CONSISTENT height field
from a normal / gradient field. A measured or hand-authored gradient field is generally NOT integrable (its mixed
partials disagree, so no exact height exists); Frankot & Chellappa (1988) find the height whose gradient is the
closest integrable field, and the whole solve is PURE FFT -- the engine's native operator (a `bind` IS an FFT
convolution; the fluid solver projects on the same periodic Fourier domain).

The math, in three moves. Because the Fourier transform turns d/dx into a multiply by j*xi_x, the least-squares
height is one forward transform of the gradients, a per-frequency divide, and one inverse:

    Z_hat(xi) = ( -j*xi_x * P_hat  -  j*xi_y * Q_hat ) / (xi_x^2 + xi_y^2) ,   Z_hat(0,0) = 0

where P = dz/dx, Q = dz/dy, and the DC term is 0 because height is only defined up to an additive constant.

Two payoffs for auto-bump: the height<->normal round-trip becomes INTEGRABLE and DRIFT-FREE (no accumulated tilt),
and -- the useful accident -- the periodic boundary makes the result SEAMLESSLY TILEABLE, which is exactly what a
material texture wants.

KEPT NEGATIVE (loud): the periodic boundary is a SYSTEMATIC BIAS on a NON-periodic surface -- the textbook artifact,
opposite borders forced to agree (a face's cheek and nose pulled equal at the seam). For a tileable MATERIAL that
periodicity is a FEATURE; for a bounded scene surface, prefer a DCT/DST variant (Simchony et al. 1990) or a Poisson
solve with the real boundary. State which regime you're in. NumPy + stdlib only; deterministic.
"""
import numpy as np


def gradient_from_normals(nmap):
    """Recover the height gradient (p, q) = (dz/dx, dz/dy) from a tangent-space normal map. A tangent-space normal
    is normalize(-p, -q, 1), so p = -nx/nz and q = -ny/nz. nz is floored to avoid a divide-by-zero at a normal
    that lies flat against the surface (a grazing gradient)."""
    n = np.asarray(nmap, float)
    nz = np.clip(n[..., 2], 1e-6, None)
    return -n[..., 0] / nz, -n[..., 1] / nz


def height_from_gradient(p, q):
    """Frankot-Chellappa: the least-squares INTEGRABLE height whose gradient best matches (p, q). Pure FFT on a
    periodic domain, so the result is SEAMLESSLY TILEABLE; the height is returned with a zero mean (the arbitrary
    additive constant removed)."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    H, W = p.shape
    # angular wavenumbers for a periodic grid. rfft2 keeps only the non-redundant half of the x axis.
    wy = 2.0 * np.pi * np.fft.fftfreq(H)[:, None]        # xi_y, one per row
    wx = 2.0 * np.pi * np.fft.rfftfreq(W)[None, :]       # xi_x, one per (kept) column
    P = np.fft.rfft2(p)
    Q = np.fft.rfft2(q)
    denom = wx * wx + wy * wy
    denom[0, 0] = 1.0                                    # avoid 0/0 at DC; that mode is set to 0 on the next line
    Z = (-1j * wx * P - 1j * wy * Q) / denom
    Z[0, 0] = 0.0                                        # the height offset is arbitrary -> zero mean
    return np.fft.irfft2(Z, s=(H, W))


def height_from_normals(nmap):
    """Integrate a NORMAL MAP to a consistent, tileable height field (gradient_from_normals -> height_from_gradient).
    This is the inverse of IR1's normal_from_height, and it 'repairs' a non-integrable normal map into the nearest
    integrable one."""
    p, q = gradient_from_normals(nmap)
    return height_from_gradient(p, q)


def consistent_normals(nmap, strength=1.0):
    """Project a (possibly non-integrable) normal map onto the nearest INTEGRABLE one: integrate it to a height,
    then re-derive normals from that height. The result is guaranteed to be the gradient of a real surface."""
    from holographic.mesh_and_geometry.holographic_autobump import normal_from_height
    return normal_from_height(height_from_normals(nmap), strength=strength)


def _selftest():
    """The FFT solver recovers a known PERIODIC height from its analytic gradient (up to a constant); it recovers
    it from a normal map too (finite-difference tolerance); the result is periodic (seamlessly tileable); and
    re-deriving normals from an integrated height reproduces the same height (integrability). Deterministic."""
    H, W = 48, 64
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    # an exactly-periodic height (integer wavenumbers) so Frankot-Chellappa can recover it exactly
    h = np.sin(2 * np.pi * 2 * x / W) * np.cos(2 * np.pi * 3 * y / H)
    h = h - h.mean()

    # (1) analytic gradient -> integrate -> recover h (up to a constant)
    p = (2 * np.pi * 2 / W) * np.cos(2 * np.pi * 2 * x / W) * np.cos(2 * np.pi * 3 * y / H)   # dh/dx
    q = -(2 * np.pi * 3 / H) * np.sin(2 * np.pi * 2 * x / W) * np.sin(2 * np.pi * 3 * y / H)  # dh/dy
    z = height_from_gradient(p * np.ones_like(h), q * np.ones_like(h))
    z = z - z.mean()
    assert np.corrcoef(z.ravel(), h.ravel())[0, 1] > 0.999
    assert np.sqrt(np.mean((z - h) ** 2)) / (np.std(h) + 1e-12) < 0.02       # <2% relative RMS

    # (2) from a NORMAL map (normal_from_height with strength=1 -> gradient = dh/dx): round-trips h
    from holographic.mesh_and_geometry.holographic_autobump import normal_from_height
    nmap = normal_from_height(h, strength=1.0)
    z2 = height_from_normals(nmap)
    z2 = z2 - z2.mean()
    assert np.corrcoef(z2.ravel(), h.ravel())[0, 1] > 0.99                    # finite-diff derivative -> looser

    # (3) SEAMLESSLY TILEABLE: the recovered height is periodic, so the wrap-around difference is tiny relative to
    #     the in-plane variation (a plain first-difference across the seam matches the interior scale)
    seam = np.abs(z[:, 0] - z[:, -1]).mean()
    interior = np.abs(np.diff(z, axis=1)).mean()
    assert seam < 3.0 * interior                                             # no big discontinuity at the tile seam

    # (4) integrability: re-deriving normals from the integrated height gives a consistent field (round-trips h)
    cn = consistent_normals(nmap, strength=1.0)
    z3 = height_from_normals(cn); z3 = z3 - z3.mean()
    assert np.corrcoef(z3.ravel(), h.ravel())[0, 1] > 0.99

    # (5) deterministic
    assert np.array_equal(height_from_gradient(p * np.ones_like(h), q * np.ones_like(h)),
                          height_from_gradient(p * np.ones_like(h), q * np.ones_like(h)))

    print("holographic_surfaceint selftest OK: Frankot-Chellappa FFT recovers a known periodic height from its "
          "analytic gradient (corr %.4f, <2%% RMS) and from a normal map (corr %.3f); the result is periodic so it "
          "tiles seamlessly (seam %.3f vs interior %.3f); re-deriving normals from the integrated height reproduces "
          "the height (integrable); deterministic"
          % (np.corrcoef(z.ravel(), h.ravel())[0, 1], np.corrcoef(z2.ravel(), h.ravel())[0, 1],
             float(np.abs(z[:, 0] - z[:, -1]).mean()), float(np.abs(np.diff(z, axis=1)).mean())))


if __name__ == "__main__":
    _selftest()

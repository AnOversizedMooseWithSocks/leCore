"""Vectorized deformers and blendshapes (ANIM-1): deform any point set -- mesh vertices OR a particle cloud --
with no Python per-point loop.

WHY THIS MODULE EXISTS
----------------------
An animation / simulation tool deforms geometry over time. The classic deformers (bend, twist, taper, a
free-form lattice) and blendshapes (morph targets) are the workhorses. Every operation here is a single
VECTORISED array op over an (N,3) point array, so it runs the same on a mesh's vertices and on a particle
cloud, and it never loops in Python over points.

VSA-NATIVE WHERE IT IS TRUE (not forced)
  * A blendshape mix is a WEIGHTED BUNDLE: result = base + sum_i w_i (target_i - base) -- the engine's
    superposition primitive applied to geometry. `blendshapes` is literally `weights @ deltas`.
  * A rigid transform is a single BIND in the engine's HolographicField (translate = one bind); the affine /
    nonlinear deformers here are the extensions that a bind can't express, kept as explicit vectorized maps.
  The honest line: the SHAPE math (sin/cos of a bend) is plain NumPy, not a hypervector trick -- claiming
  otherwise would be dishonest. What is genuinely VSA-shaped is the blend (a bundle) and the rigid case (a bind).
"""

import numpy as np


def _axis_span(P, axis):
    """The min and (max-min) extent of `P` along `axis`, for normalising a deformer's parameter to [0,1]-ish."""
    lo = float(P[:, axis].min())
    span = float(P[:, axis].max() - lo)
    return lo, (span if span > 1e-12 else 1.0)


def taper(P, factor, axis=2):
    """Scale the cross-section (the two non-`axis` coords) in proportion to position along `axis` -- a cone /
    taper. factor>0 widens toward the high end, <0 pinches. Vectorised over all points."""
    P = np.asarray(P, float).copy()
    lo, span = _axis_span(P, axis)
    s = 1.0 + factor * (P[:, axis] - lo) / span               # per-point scale, by position along axis
    others = [k for k in range(3) if k != axis]
    P[:, others] *= s[:, None]
    return P


def twist(P, angle, axis=2):
    """Rotate the cross-section by an angle that grows along `axis` -- a twist / screw. `angle` (radians) is the
    total twist over the axis extent. Vectorised."""
    P = np.asarray(P, float).copy()
    lo, span = _axis_span(P, axis)
    th = angle * (P[:, axis] - lo) / span                     # per-point twist angle
    others = [k for k in range(3) if k != axis]
    a, b = others
    ca, sa = np.cos(th), np.sin(th)
    x, y = P[:, a].copy(), P[:, b].copy()
    P[:, a] = ca * x - sa * y
    P[:, b] = sa * x + ca * y
    return P


def bend(P, angle, axis=0, up=2, center=None):
    """Bend the geometry: a straight extent along `axis` is mapped to a circular arc in the (axis, up) plane,
    subtending `angle` radians total (Barr's bend). The third axis is unchanged. Vectorised, no per-point loop."""
    P = np.asarray(P, float).copy()
    lo, span = _axis_span(P, axis)
    if abs(angle) < 1e-9:
        return P
    k = angle / span                                          # curvature
    R = 1.0 / k                                               # arc radius
    cen = (P[:, axis].mean() if center is None else center)
    z_ref = float(P[:, up].min())                            # the bend baseline
    s = P[:, axis] - cen                                      # arc length along the bend axis
    h = P[:, up] - z_ref                                      # height above the baseline (rides the radial dir)
    th = k * s
    P[:, axis] = cen + (R - h) * np.sin(th)
    P[:, up] = z_ref + R - (R - h) * np.cos(th)
    return P


def lattice_deform(P, bounds, control_offsets):
    """Free-form (lattice / FFD) deformation: a regular control lattice spans `bounds`=(min,max); each control
    point is displaced by `control_offsets` (shape (nx,ny,nz,3)); every input point is moved by the TRILINEAR
    interpolation of the surrounding control displacements. Vectorised over all points. The general
    sculpt-by-cage deformer."""
    P = np.asarray(P, float)
    off = np.asarray(control_offsets, float)
    nx, ny, nz = off.shape[:3]
    lo = np.asarray(bounds[0], float); hi = np.asarray(bounds[1], float)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    # normalised lattice coords in [0, n-1]
    gx = np.clip((P[:, 0] - lo[0]) / span[0] * (nx - 1), 0, nx - 1 - 1e-9)
    gy = np.clip((P[:, 1] - lo[1]) / span[1] * (ny - 1), 0, ny - 1 - 1e-9)
    gz = np.clip((P[:, 2] - lo[2]) / span[2] * (nz - 1), 0, nz - 1 - 1e-9)
    ix = gx.astype(int); iy = gy.astype(int); iz = gz.astype(int)
    fx = (gx - ix)[:, None]; fy = (gy - iy)[:, None]; fz = (gz - iz)[:, None]
    disp = np.zeros_like(P)
    for dx in (0, 1):                                         # 8 corners -- 8 vectorised gathers (NOT a point loop)
        for dy in (0, 1):
            for dz in (0, 1):
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                disp += w * off[np.clip(ix + dx, 0, nx - 1), np.clip(iy + dy, 0, ny - 1), np.clip(iz + dz, 0, nz - 1)]
    return P + disp


def blendshapes(base, targets, weights):
    """A morph-target / blendshape mix -- a WEIGHTED BUNDLE of pose deltas: result = base + sum_i w_i (target_i
    - base). `base` is (N,3), `targets` a stack (K,N,3) or list of K (N,3), `weights` length K. Fully vectorised
    (`weights @ deltas`): this is the engine's superposition primitive applied to geometry, so animating the
    weights over time IS the blendshape animation. Weights need not sum to 1 (over/under-shoot allowed)."""
    base = np.asarray(base, float)
    T = np.asarray(targets, float)
    if T.ndim == 2:
        T = T[None]
    w = np.asarray(weights, float)
    deltas = T - base[None]                                   # (K, N, 3)
    return base + np.tensordot(w, deltas, axes=(0, 0))        # base + sum_k w_k * delta_k -- one einsum-class op


def _selftest():
    P = np.array([[x, 0.0, z] for x in np.linspace(-1, 1, 5) for z in np.linspace(0, 1, 5)])
    assert taper(P, 0.5, axis=0).shape == P.shape
    tw = twist(P, np.pi / 2, axis=0)
    assert not np.allclose(tw, P)                             # twist changed the shape
    # lattice identity: zero offsets -> no movement
    off = np.zeros((3, 3, 3, 3))
    assert np.allclose(lattice_deform(P, (np.array([-1., -1, -1]), np.array([1., 1, 1])), off), P)
    # lattice translate: a uniform offset shifts everything by it
    off[:] = np.array([0.1, 0.0, 0.0])
    assert np.allclose(lattice_deform(P, (np.array([-1., -1, -1]), np.array([1., 1, 1])), off) - P,
                       np.array([0.1, 0, 0]), atol=1e-6)
    # blendshape: weight 0 -> base, weight 1 -> target
    tgt = P + np.array([0.0, 0.5, 0.0])
    assert np.allclose(blendshapes(P, [tgt], [0.0]), P)
    assert np.allclose(blendshapes(P, [tgt], [1.0]), tgt)
    assert np.allclose(blendshapes(P, [tgt], [0.5]), P + np.array([0, 0.25, 0]))
    print("deform selftest ok: taper/twist/lattice/blendshapes vectorised, identity + endpoints exact")


if __name__ == "__main__":
    _selftest()

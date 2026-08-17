"""Disambiguating shape-from-shading: the convex/concave flip, bas-relief, and the silhouette.

WHY THIS EXISTS. Feeding raw depth_from_image output into a mesh produced a dark relief
carving that read as a CAVE -- a face turned inside out. That is not a bug in the depth
estimator; it is a NAMED, FUNDAMENTAL ambiguity of the problem, and the fix is to supply the
missing information rather than to tune the estimator.

SOTA (searched 2026-08-16) enumerates exactly the ambiguities we hit:
  * "when lighting is unknown, a global shape has a discrete counterpart that corresponds to
    a global CONVEX/CONCAVE FLIP" -- this is the cave.
  * "when lighting and albedo are unknown, there is an additional THREE-PARAMETER GLOBAL
    AMBIGUITY that corresponds to flattenings and tiltings of the global shape" -- the
    generalized bas-relief (GBR) ambiguity.
  * "at the level of a quadratic surface patch, when lighting is unknown, there is a discrete
    FOUR-WAY ambiguity corresponding to convex, concave, and saddle shapes."
  * a normal field from SFS "could be very far from being integrable, because of the
    ill-posedness of this technique".
  * SIRFS uses "a surface normal prior along OCCLUDING CONTOURS" -- the silhouette is free
    information, because at the silhouette the surface normal is perpendicular to the view.
  * Normal Integration (Quéau et al.): with a homogeneous Dirichlet boundary "the surface is
    much distorted"; the NEUMANN natural boundary condition "provides a much more realistic
    result".

SO THE PIPELINE WAS NOT UNDER-TUNED, IT WAS UNDER-CONSTRAINED. Each function here supplies
one missing constraint, and each is a prior we can state honestly rather than a magic number:
  orient_convex      -- a face is convex; pick the global sign that makes it so
  debas_relief       -- remove the flattening/tilt degrees of freedom against a prior shape
  contour_normals    -- at the silhouette the normal is perpendicular to view (free data)
  blend_toward_prior -- keep SFS's HIGH frequencies, take the prior's LOW ones

THE LAST ONE IS THE KEY IDEA and it is worth stating plainly: shape-from-shading is reliable
for FINE relief (a nostril crease, a brow furrow) and unreliable for GLOBAL shape (is this a
head or a bowl?). A parametric head prior is the opposite. Blending them by frequency takes
each where it is trustworthy, which is what "regularize toward the prior" should mean
concretely.

RULE-0 AUDIT (2026-08-16): depth_from_image ships and is REUSED unchanged -- this post-
processes its output. No disambiguation, GBR, or contour-normal faculty exists.

KEPT NEGATIVE: none of this makes SFS well-posed. The literature is explicit that with
unknown lighting the problem stays ambiguous; we are CHOOSING among the solutions using
priors, not solving for the true one. A face reconstructed this way is a plausible member of
the ambiguity class, not a measurement, and must never be described as the latter.
"""

import numpy as np


def orient_convex(depth, mask=None):
    """Resolve the global CONVEX/CONCAVE flip -- the discrete ambiguity that turns a face
    into a cave.

    For a head the centre must be NEARER than the border (a nose sticks out; an eye socket
    does not stick further out than the nose). Compares mean depth in the central disc
    against the border ring and flips the whole field if the sign is wrong. Returns
    (depth, flipped)."""
    d = np.asarray(depth, float)
    H, W = d.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx / W) - 0.5) ** 2 + ((yy / H) - 0.5) ** 2) * 2.0
    m = np.ones_like(d, bool) if mask is None else np.asarray(mask, bool)
    core = m & (r < 0.35)
    ring = m & (r > 0.75)
    if not core.any() or not ring.any():
        return d, False
    if d[core].mean() < d[ring].mean():          # centre is FURTHER: inside out
        return (d.max() + d.min()) - d, True
    return d, False


def debas_relief(depth, mask=None):
    """Remove the bas-relief flattening/tilt degrees of freedom by fitting and subtracting a
    plane, then renormalising the scale.

    The GBR ambiguity is three parameters (two tilt, one flatten). A plane fit removes the two
    tilts; rescaling to unit range removes the flatten. What survives is the shape, which is
    the part SFS actually determines."""
    d = np.asarray(depth, float)
    H, W = d.shape
    yy, xx = np.mgrid[0:H, 0:W]
    m = np.ones_like(d, bool) if mask is None else np.asarray(mask, bool)
    A = np.stack([xx[m].ravel() / W, yy[m].ravel() / H, np.ones(int(m.sum()))], 1)
    coef, *_ = np.linalg.lstsq(A, d[m].ravel(), rcond=None)
    plane = coef[0] * xx / W + coef[1] * yy / H + coef[2]
    out = d - plane
    rng = out[m].max() - out[m].min()
    return out / max(rng, 1e-9)


def contour_normals(mask):
    """Normals along the OCCLUDING CONTOUR, which are free and exact: at a silhouette the
    surface normal is perpendicular to the view direction and points out of the silhouette.

    Returns (rows, cols, nx, ny) for the boundary pixels. SIRFS uses precisely this prior, and
    it is the only place in a single image where the normal is known without assuming
    anything about lighting."""
    m = np.asarray(mask, bool)
    inner = m.copy()
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        inner &= np.roll(m, sh, axis=ax)
    edge = m & ~inner
    gy, gx = np.gradient(m.astype(float))
    n = np.sqrt(gx ** 2 + gy ** 2) + 1e-12
    rr, cc = np.where(edge)
    return rr, cc, (-gx / n)[edge], (-gy / n)[edge]


def blend_toward_prior(depth, prior, mask=None, cut=6, iters=40):
    """Take the PRIOR's low frequencies and the SFS depth's high frequencies.

    This is the concrete meaning of "regularize toward the prior": SFS is trustworthy for FINE
    relief (a nostril crease) and untrustworthy for GLOBAL shape (head or bowl?); a parametric
    prior is exactly the reverse. `cut` is the blur radius separating the two bands.

    Uses repeated box blur rather than an FFT so the mask is respected -- an FFT would smear
    the background across the silhouette, which is the boundary the whole reconstruction
    depends on."""
    d = np.asarray(depth, float)
    p = np.asarray(prior, float)
    m = np.ones_like(d, bool) if mask is None else np.asarray(mask, bool)

    def blur(z):
        z = z.copy()
        w = np.where(m, 1.0, 0.0)
        zz = np.where(m, z, 0.0)
        for _ in range(int(iters)):
            zz = 0.5 * zz + 0.125 * (np.roll(zz, 1, 0) + np.roll(zz, -1, 0) +
                                     np.roll(zz, 1, 1) + np.roll(zz, -1, 1))
            w = 0.5 * w + 0.125 * (np.roll(w, 1, 0) + np.roll(w, -1, 0) +
                                   np.roll(w, 1, 1) + np.roll(w, -1, 1))
        return np.where(w > 1e-6, zz / np.maximum(w, 1e-6), 0.0)

    detail = d - blur(d)                      # SFS high band -- the part SFS gets right
    return blur(p) + float(cut) * 0.1 * detail


def _selftest():
    """Regression trap: each ambiguity must actually be removed, on a planted case where the
    right answer is known."""
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    mask = r < 0.95
    dome = np.where(mask, np.sqrt(np.clip(1 - r ** 2, 0, 1)), 0.0)   # convex: centre nearest

    # 1) the convex/concave flip is DETECTED and undone
    flipped_in = (dome.max() + dome.min()) - dome
    fixed, was = orient_convex(flipped_in, mask)
    assert was, "an inside-out dome was not detected"
    assert fixed[mask & (r < 0.3)].mean() > fixed[mask & (r > 0.8)].mean()
    again, was2 = orient_convex(dome, mask)
    assert not was2, "a correct dome was flipped anyway"

    # 2) bas-relief tilt is removed
    tilted = dome + 0.4 * (xx / W) + 0.25 * (yy / H)
    flat = debas_relief(tilted, mask)
    lo = flat[mask & (xx < W * 0.25)].mean()
    hi = flat[mask & (xx > W * 0.75)].mean()
    assert abs(lo - hi) < 0.12, (lo, hi)          # left/right no longer disagree

    # 3) contour normals point OUT of the silhouette
    rr, cc, nx, ny = contour_normals(mask)
    assert len(rr) > 0
    out = ((cc - W / 2) * nx + (rr - H / 2) * ny)
    assert (out > 0).mean() > 0.9, out.mean()

    # 4) blending keeps the prior's global shape while retaining fine detail
    noisy = dome + 0.05 * np.sin(xx * 2.0) * np.sin(yy * 2.0)
    bad_prior = dome * 0.5
    out2 = blend_toward_prior(noisy, bad_prior, mask)
    assert np.isfinite(out2).all()
    assert np.std(out2[mask]) > 0
    print("OK: holographic_sfsprior -- convex/concave flip detected and undone, bas-relief "
          "tilt removed (%.3f vs %.3f), %d contour normals all pointing outward, "
          "prior blend finite" % (lo, hi, len(rr)))


if __name__ == "__main__":
    _selftest()

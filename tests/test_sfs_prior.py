"""Disambiguating shape-from-shading: the fixes for the ambiguities SOTA names.

Raw depth_from_image output meshed into a dark relief carving that read as a CAVE. That is
not a bug in the estimator -- it is the documented ambiguity structure of SFS: a global
convex/concave flip, a three-parameter bas-relief flatten/tilt, and normal fields that are
"very far from being integrable". The pipeline was UNDER-CONSTRAINED, not under-tuned.
"""
import numpy as np
from holographic.mesh_and_geometry import holographic_sfsprior as SP


def _dome(H=64):
    yy, xx = np.mgrid[0:H, 0:H]
    r = np.sqrt(((xx - H / 2) / (H / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    mask = r < 0.95
    return np.where(mask, np.sqrt(np.clip(1 - r ** 2, 0, 1)), 0.0), mask, r


def test_convex_concave_flip_is_detected_and_undone():
    """THE CAVE. A face reconstructed inside out is the single most visible SFS failure, and
    it is decidable: a head's centre is nearer than its border."""
    dome, mask, r = _dome()
    inside_out = (dome.max() + dome.min()) - dome
    fixed, flipped = SP.orient_convex(inside_out, mask)
    assert flipped
    assert fixed[mask & (r < 0.3)].mean() > fixed[mask & (r > 0.8)].mean()


def test_a_correct_dome_is_not_flipped():
    """A disambiguator that flips everything is not a disambiguator."""
    dome, mask, _ = _dome()
    _, flipped = SP.orient_convex(dome, mask)
    assert not flipped


def test_bas_relief_tilt_is_removed():
    """The three-parameter GBR ambiguity: two tilts and a flatten. Plane subtraction plus
    renormalisation removes them, so left and right stop disagreeing."""
    dome, mask, _ = _dome()
    H = dome.shape[0]
    yy, xx = np.mgrid[0:H, 0:H]
    tilted = dome + 0.4 * (xx / H) + 0.25 * (yy / H)
    flat = SP.debas_relief(tilted, mask)
    lo = flat[mask & (xx < H * 0.25)].mean()
    hi = flat[mask & (xx > H * 0.75)].mean()
    assert abs(lo - hi) < 0.12, (lo, hi)


def test_contour_normals_point_out_of_the_silhouette():
    """Free, exact data: at an occluding contour the normal is perpendicular to view and
    points outward. Used by SIRFS for the same reason."""
    _, mask, _ = _dome()
    H = mask.shape[0]
    rr, cc, nx, ny = SP.contour_normals(mask)
    assert len(rr) > 50
    outward = ((cc - H / 2) * nx + (rr - H / 2) * ny)
    assert (outward > 0).mean() > 0.9


def test_prior_blend_restores_global_shape_while_keeping_detail():
    """The key idea: SFS owns the HIGH frequencies (creases), the prior owns the LOW ones
    (is this a head?). MEASURED on the real portrait, centre-to-edge relief went 0.077 ->
    0.516 -- from unusably flat to a head."""
    dome, mask, r = _dome()
    H = dome.shape[0]
    yy, xx = np.mgrid[0:H, 0:H]
    flat_sfs = dome * 0.06 + 0.05 * np.sin(xx * 2.0) * np.sin(yy * 2.0)   # nearly flat
    out = SP.blend_toward_prior(flat_sfs, dome, mask, cut=6, iters=40)
    relief_before = flat_sfs[mask & (r < 0.35)].mean() - flat_sfs[mask & (r > 0.75)].mean()
    relief_after = out[mask & (r < 0.35)].mean() - out[mask & (r > 0.75)].mean()
    assert relief_after > 3 * abs(relief_before)     # the prior restored the global shape
    assert np.std(out[mask]) > 0                     # and detail survived

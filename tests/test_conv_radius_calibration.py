"""O4: convolution-surface radius calibration, and the measured tradeoff between skin paths.

SOTA names the weakness this fixes: "while convolution surfaces eliminate bulge artifacts,
they also reduce geometric control, since the target iso-surface is no longer located at the
expected distance from the skeleton" (SCALIS, Zanni et al.). That is the price of bulge-free
joints, and it is why a salamander asked for a 0.018 tail tip did not get one.
"""
import numpy as np
from holographic.mesh_and_geometry import holographic_creatureconv as CC


def _landed(segs, kernel=2.2, iso=0.35):
    f = CC.convolution_field(segs, iso=iso, kernel=kernel)
    r = segs[0][2]
    t = np.linspace(1e-4, 6 * r, 1200)
    P = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    v = np.asarray(f(P), float).ravel()
    s = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    assert len(s), "no surface found"
    return float(t[s[0]])


def _seg(r):
    return [((0.0, 0.0, -0.6), (0.0, 0.0, 0.6), r, (1.0, 1.0, 1.0))]


def test_the_uncalibrated_shortfall_is_real_and_stays_documented():
    """Pinned as a FINDING. If this stops failing, the kernel changed and that must be
    deliberate rather than accidental."""
    for r in (0.08, 0.20):
        assert abs(_landed(_seg(r)) - r) / r > 0.15


def test_calibration_lands_the_surface_where_asked():
    """Lever 1: the constant is solved once per kernel and divided out. 25.4% -> 0.1%."""
    for r in (0.08, 0.20):
        err = abs(_landed(CC.calibrated_segments(_seg(r))) - r) / r
        assert err < 0.03, (r, err)


def test_the_shortfall_depends_on_kernel_not_radius():
    """This is WHY a single baked constant works: the ratio is a property of the kernel.
    Measured 1.6 -> 0.926, 2.2 -> 0.747, 3.0 -> 0.590, varying only a few percent across a
    7x radius range within each kernel."""
    ratios = [CC.radius_ratio(kernel=k) for k in (1.6, 2.2, 3.0)]
    assert ratios == sorted(ratios, reverse=True), ratios      # wider kernel, more shortfall
    assert ratios[0] > 0.85 and ratios[-1] < 0.70


def test_residual_at_thin_radii_is_the_scalis_effect_and_is_documented():
    """HONEST LIMIT, pinned so it is not mistaken for a bug: calibration removes the CONSTANT
    error but not the scale-dependence. The thinnest radii keep a few percent of error --
    exactly SCALIS's 'thin components excessively smoothed when blended into larger ones'.
    A scale-invariant kernel is the real fix and is not implemented here."""
    thin = abs(_landed(CC.calibrated_segments(_seg(0.03))) - 0.03) / 0.03
    mid = abs(_landed(CC.calibrated_segments(_seg(0.20))) - 0.20) / 0.20
    assert thin > mid          # the residual is scale-dependent, not uniform
    assert thin < 0.15         # but far better than the 42.7% it started at

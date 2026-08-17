"""SCALIS: scale-invariant integral surfaces, so thin features survive next to thick ones.

Zanni et al. 2013. Plain convolution "failed to reconstruct prescribed radii and [was] unable
to model large shapes with fine details"; SCALIS fixes it by changing the NORMALIZATION
FACTOR -- integrating over the homothetic measure ds/tau rather than absolute arc length.
This is the fix O4 justified: our radius calibration removed the CONSTANT error but left a
scale-dependent residual, and that residual is exactly what scale-invariance addresses.
"""
import numpy as np
from holographic.mesh_and_geometry.holographic_creatureconv import (_seg_convolution,
                                                                    convolution_field)


def test_scalis_field_is_exactly_scale_invariant():
    """The defining property. Under (r, L, d) -> lam*(r, L, d) the exponent d^2/r^2 and the
    weight L/(n*r) are both unchanged, so the field is. Plain scales by lam."""
    vals_s, vals_p = [], []
    for lam in (0.25, 1.0, 4.0):
        r, L = 0.12 * lam, 1.2 * lam
        P = np.array([[r, 0.0, 0.0]])
        a, b = (0, 0, -L / 2), (0, 0, L / 2)
        vals_p.append(float(_seg_convolution(P, a, b, r, samples=48)[0]))
        vals_s.append(float(_seg_convolution(P, a, b, r, samples=48, scalis=True)[0]))
    assert max(vals_s) - min(vals_s) < 1e-9, vals_s          # invariant
    assert max(vals_p) / min(vals_p) > 10.0, vals_p          # plain is not


def _radius_at(f, z):
    t = np.linspace(1e-4, 0.5, 1200)
    P = np.stack([t, np.zeros_like(t), np.full_like(t, z)], 1)
    v = np.asarray(f(P), float).ravel()
    s = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    return float(t[s[0]]) if len(s) else float("nan")


def test_thin_feature_survives_beside_a_thick_one():
    """The case SCALIS exists for, and the salamander's vanishing tail tip. A spike 5.7x
    thinner than its trunk: plain renders it at ~9% of the requested radius (swallowed),
    SCALIS keeps it."""
    thick = ((0, 0, -0.6), (0, 0, 0.6), 0.20, (1., 1., 1.))
    thin = ((0, 0, 0.6), (0, 0, 1.4), 0.035, (1., 1., 1.))
    plain = _radius_at(convolution_field([thick, thin], iso=0.35, samples=32), 1.1)
    scal = _radius_at(convolution_field([thick, thin], iso=0.35, samples=32, scalis=True), 1.1)
    assert plain / 0.035 < 0.3, plain            # the thin feature is lost
    assert 0.7 < scal / 0.035 < 1.6, scal        # and SCALIS keeps it near the request
    assert scal > 5 * plain


def test_default_is_unchanged():
    """House rule: existing decisions never flip. scalis defaults off and must be bitwise
    identical to the shipped path."""
    segs = [((0, 0, -0.5), (0, 0, 0.5), 0.15, (1., 1., 1.))]
    P = np.random.default_rng(0).normal(size=(50, 3)) * 0.3
    a = np.asarray(convolution_field(segs, iso=0.35)(P), float)
    b = np.asarray(convolution_field(segs, iso=0.35, scalis=False)(P), float)
    assert np.array_equal(a, b)


def test_faculty_is_wired():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    f = m.convolution_field_scalis([((0, 0, -0.5), (0, 0, 0.5), 0.15, (1., 1., 1.))])
    assert np.isfinite(np.asarray(f(np.array([[0.1, 0.0, 0.0]])), float)).all()

"""Dispersion: the wavelength->IOR map and the rank-3 shortcut past hero-wavelength sampling.

These pin CONTRACTS, not the size of any backlog. The repo has lost five tests to the opposite habit
(a test whose fixture is a real bug dies the day somebody fixes it), so every assertion here is
either a published physical constant or a measured trade-off with a stated tolerance.
"""
import numpy as np
import pytest

import lecore
from holographic.rendering.holographic_dispersion import (
    SELLMEIER, abbe_number, anchor_layers, hero_wavelengths, sellmeier_n)


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


# --------------------------------------------------------------------------- the glass


@pytest.mark.parametrize("glass, n_d, v_d", [
    ("BK7", 1.5168, 64.17),            # Schott catalogue
    ("SF11", 1.7847, 25.68),
    ("fused_silica", 1.4585, 67.82),   # Malitson 1965
])
def test_catalogue_constants(glass, n_d, v_d):
    """The published numbers. A transcription error in the Sellmeier coefficients shows up as a
    subtly wrong colour that nobody can name, so it gets caught against the catalogue instead."""
    assert abs(float(sellmeier_n(587.5618, glass)) - n_d) < 1e-3
    assert abs(abbe_number(glass) - v_d) < 0.3


@pytest.mark.parametrize("glass", sorted(SELLMEIER))
def test_normal_dispersion(glass):
    """Blue bends more than red in every one of these glasses. If this flips, every caustic edge in
    the engine is inverted and no other test would say so."""
    assert float(sellmeier_n(450.0, glass)) > float(sellmeier_n(650.0, glass))


def test_flint_disperses_harder_than_crown():
    """SF11 is a flint, BK7 a crown. This is the check that catches swapping one glass's
    coefficients for another's -- which n_d alone would happily accept."""
    assert abbe_number("SF11") < abbe_number("BK7") / 2.0


# --------------------------------------------------------------------------- hero sampling


@pytest.mark.parametrize("u", [0.0, 0.13, 0.5, 0.87, 0.999])
def test_hero_rotation_is_stratified(u):
    """Wilkie et al. 2014's whole point: whatever hero wavelength you draw, the companions stay
    evenly spread over the visible band. Uneven spacing is a silently biased spectral estimator."""
    w = np.sort(hero_wavelengths(u, count=4, lo=380.0, hi=780.0))
    gaps = np.diff(np.concatenate([w, [w[0] + 400.0]]))
    assert np.allclose(gaps, 100.0, atol=1e-9)
    assert w.min() >= 380.0 and w.max() < 780.0


# --------------------------------------------------------------------------- anchor interpolation


def test_affine_family_interpolates_exactly():
    """A tracer whose output is exactly affine in IOR must be reproduced to machine precision. This
    pins the interpolation MATH with the physics held out -- real caustics are only NEARLY low-rank,
    so testing against them could assert only a loose tolerance and would pass a broken bracket."""
    def affine(ior, **kw):
        return np.arange(9.0).reshape(3, 3) * float(ior) + 2.0

    iors = [1.51, 1.60, 1.53, 1.58, 1.55]        # deliberately unsorted, like hero order
    assert np.allclose(anchor_layers(affine, iors, 2), anchor_layers(affine, iors, None), atol=1e-12)


def test_anchors_none_is_the_unaccelerated_path(mind):
    """Additive and backward-compatible: the default must be byte-identical to tracing every
    wavelength, not merely close to it."""
    calls = []

    def counting(ior, **kw):
        calls.append(float(ior))
        return np.full((2, 2), float(ior))

    iors = [1.50, 1.52, 1.54, 1.56]
    out = anchor_layers(counting, iors, None)
    assert calls == iors
    assert np.array_equal(out, np.stack([np.full((2, 2), n) for n in iors]))


def test_anchors_below_two_raises():
    """One anchor cannot bracket anything. Failing loudly beats returning a constant image."""
    with pytest.raises(ValueError):
        anchor_layers(lambda ior, **kw: np.zeros((2, 2)), [1.5, 1.6], 1)


# --------------------------------------------------------------------------- through the mind


def test_spectral_caustic_has_colour_and_monochrome_has_none(mind):
    """The result, against a TRUE zero: one wavelength through the same observer is exactly grey,
    because one wavelength cannot be two colours. That exact 0.0 is what makes the spectral number a
    measurement rather than a look."""
    sphere = lambda p: np.linalg.norm(p, axis=-1) - 1.0
    scene = dict(light_dir=(0.0, -1.0, 0.0), receiver_y=-1.6, extent=2.5, res=48, n_side=100, seed=0)

    def saturation(rgb):
        rgb = np.asarray(rgb, float)
        mx, mn = rgb.max(-1), rgb.min(-1)
        lit = mx > 0.02 * mx.max()
        return float(((mx - mn)[lit] / np.maximum(mx[lit], 1e-12)).mean()) if lit.any() else 0.0

    mono = np.asarray(mind.caustics(sphere, ior=1.5168, **scene), float)
    assert saturation(np.stack([mono] * 3, -1)) == 0.0
    assert saturation(mind.spectral_caustics(sphere, glass="BK7", count=8, **scene)["rgb"]) > 0.05


def test_anchor_shortcut_keeps_the_colour(mind):
    """The claim that actually matters. A shortcut that smears the wavelengths together would score a
    LOW rel-err -- the caustic is mostly a bright achromatic pedestal -- while destroying the one
    thing dispersion is for. So the gate is saturation retention; rel-err rides along as a check."""
    sphere = lambda p: np.linalg.norm(p, axis=-1) - 1.0
    scene = dict(light_dir=(0.0, -1.0, 0.0), receiver_y=-1.6, extent=2.5, res=48, n_side=100, seed=0)

    def saturation(rgb):
        rgb = np.asarray(rgb, float)
        mx, mn = rgb.max(-1), rgb.min(-1)
        lit = mx > 0.02 * mx.max()
        return float(((mx - mn)[lit] / np.maximum(mx[lit], 1e-12)).mean()) if lit.any() else 0.0

    full = mind.spectral_caustics(sphere, glass="BK7", count=16, anchors=None, **scene)
    fast = mind.spectral_caustics(sphere, glass="BK7", count=16, anchors=3, **scene)
    assert full["traced"] == 16 and fast["traced"] == 3
    a, b = np.asarray(full["rgb"], float), np.asarray(fast["rgb"], float)
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 0.05
    assert 0.85 < saturation(b) / saturation(a) < 1.15


# --------------------------------------------------------------------------- the slider (Blender parity)


@pytest.mark.parametrize("n_d, v", [(1.5168, 64.17), (1.7847, 25.68), (1.4585, 67.82)])
def test_cauchy_round_trips_the_abbe_definition(n_d, v):
    """Build the curve from (n_d, V_d), then recompute V_d FROM the curve and get the number back. This
    is what catches a swapped Fraunhofer line or a micron/nanometre slip -- neither of which n_d alone
    would notice, because n_d is the one point the construction pins by definition."""
    from holographic.rendering.holographic_dispersion import LINE_C, LINE_F, cauchy_n
    got_d = float(cauchy_n(587.5618, n_d, v))
    got_v = (got_d - 1.0) / (float(cauchy_n(LINE_F, n_d, v)) - float(cauchy_n(LINE_C, n_d, v)))
    assert abs(got_d - n_d) < 1e-12
    assert abs(got_v - v) < 1e-9


@pytest.mark.parametrize("glass", ["BK7", "SF11"])
def test_cauchy_tracks_sellmeier_across_the_visible_band(glass):
    """The two-number slider must agree with the six-coefficient catalogue where it is used. Loosely:
    Cauchy is a two-term TRUNCATION of Sellmeier and the residual IS the approximation, so this pins
    the size of that approximation rather than pretending there is none."""
    from holographic.rendering.holographic_dispersion import cauchy_n
    n_d, v = float(sellmeier_n(587.5618, glass)), abbe_number(glass)
    lam = np.linspace(430.0, 680.0, 12)
    assert float(np.abs(cauchy_n(lam, n_d, v) - sellmeier_n(lam, glass)).max()) < 0.004


def test_abbe_zero_raises_rather_than_clamping():
    """V_d = 0 means infinite dispersion. Failing loudly beats returning a silently absurd curve."""
    from holographic.rendering.holographic_dispersion import cauchy_n
    with pytest.raises(ValueError):
        cauchy_n(550.0, 1.5, 0.0)


def test_spectral_render_uses_one_index_per_wavelength():
    """The view path's contract: each wavelength is traced at ITS OWN index, and the indices are
    ordered by normal dispersion (blue higher than red). A tracer called with one index for all of
    them would still produce a coloured image -- an achromatic render times a tint -- so the INDICES
    are what gets asserted, not the picture."""
    from holographic.rendering.holographic_dispersion import spectral_render
    seen = []

    def fake_trace(ior, **kw):
        seen.append(float(ior))
        return np.full((4, 4), float(ior))

    out = spectral_render(fake_trace, lambda lam: np.array([1.0, 1.0, 1.0]),
                          glass="SF11", count=6, lo=420.0, hi=680.0)
    assert len(seen) == 6 and out["traced"] == 6
    order = np.argsort(out["wavelengths"])
    iors = np.asarray(out["iors"], float)[order]
    assert np.all(np.diff(iors) < 0.0), iors        # normal dispersion: shorter wavelength, higher index
    assert out["band_nm"] == [420.0, 680.0]


def test_spectral_render_accepts_the_slider_parameterisation():
    """glass=None + (n_d, abbe) must drive the same path, so a Cycles/LuxCore material transfers by
    its numbers. A low Abbe must spread the indices harder than a high one."""
    from holographic.rendering.holographic_dispersion import spectral_render
    trace = lambda ior, **kw: np.full((3, 3), 1.0)
    white = lambda lam: np.array([1.0, 1.0, 1.0])
    wide = spectral_render(trace, white, n_d=1.5, abbe=20.0, count=5)
    narrow = spectral_render(trace, white, n_d=1.5, abbe=80.0, count=5)
    spread = lambda r: max(r["iors"]) - min(r["iors"])
    assert spread(wide) > 3.0 * spread(narrow), (spread(wide), spread(narrow))


# --------------------------------------------------------------------------- the exaggeration slider


def test_exaggerate_is_inert_at_one():
    """An artistic knob that silently changed the physical render would be worse than no knob."""
    from holographic.rendering.holographic_dispersion import exaggerate
    phys = np.array([1.7736, 1.8000, 1.8333])
    assert np.array_equal(exaggerate(phys, 1.0), phys)
    assert exaggerate([], 3.0).size == 0


@pytest.mark.parametrize("k", [2.0, 6.0, 6.7])
def test_exaggerate_preserves_the_mean_and_scales_the_spread(k):
    """The mean index sets refraction's overall strength and the SPREAD sets the rainbow. Stretching
    about the mean changes only the second -- if the mean moved, turning up dispersion would also
    silently change how much the glass bends light at all."""
    from holographic.rendering.holographic_dispersion import exaggerate
    phys = np.array([1.7736, 1.8000, 1.8333])
    got = exaggerate(phys, k)
    assert abs(got.mean() - phys.mean()) < 1e-12
    assert abs(np.ptp(got) - k * np.ptp(phys)) < 1e-12


def test_real_dispersion_really_is_small(mind):
    """THE MEASUREMENT THAT EXPLAINS WHY THE SLIDER EXISTS, pinned so it cannot drift into folklore.
    Across the visible band no shipped glass spans even 0.06 of refractive index -- while the widely
    used Cycles 'dispersion glass' setup stacks IOR 1.35/1.55/1.75, a spread of 0.40. A physically
    correct render of real glass is nearly achromatic, and that is the glass's fault, not the
    renderer's."""
    spreads = {g: float(sellmeier_n(420.0, g)) - float(sellmeier_n(680.0, g))
               for g in ("fused_silica", "BK7", "SF11")}
    assert all(0.0 < v < 0.06 for v in spreads.values()), spreads
    assert spreads["SF11"] > spreads["BK7"] > spreads["fused_silica"]     # flint disperses hardest
    assert 0.40 / spreads["SF11"] > 5.0, "the reference look is no longer far outside the physics"


def test_both_light_paths_take_the_same_glass(mind):
    """A scene must not have two glasses. spectral_caustics (light path) and dispersive_render (view
    path) both accept n_d + abbe + dispersion_scale, so the caustic an object casts is cast by the
    material the camera sees. Before this the caustic could only be given a catalogued glass NAME."""
    from holographic.rendering.holographic_dispersion import spectral_render
    sph = lambda p: np.linalg.norm(p, axis=-1) - 0.5
    caus = mind.spectral_caustics(sph, n_d=1.55, abbe=25.68, dispersion_scale=6.0, count=4,
                                  light_dir=(0, -1, 0), receiver_y=-1.2, extent=0.55, res=32,
                                  n_side=200, window=1.2, refracted_only=True)
    view = spectral_render(lambda ior, **kw: np.zeros((2, 2)), lambda l: np.ones(3),
                           n_d=1.55, abbe=25.68, dispersion_scale=6.0, count=4, lo=380.0, hi=780.0)
    assert abs(caus["index_spread"] - view["index_spread"]) < 1e-9, (caus["index_spread"], view["index_spread"])
    assert caus["dispersion_scale"] == view["dispersion_scale"] == 6.0


# --------------------------------------------------------------------------- specular connection (MNEE)


def test_specular_connection_solves_a_plane_exactly():
    """A flat refractor has a solvable connection, so the residual must reach zero. Validating a solver
    only on the shape it was tuned for proves nothing; a plane is the case whose answer can be checked
    independently, by walking the path FORWARD rather than trusting the residual it minimised."""
    from holographic.rendering.holographic_specularconnect import connect, refract
    nrm = np.array([0.0, 1.0, 0.0])
    normal_fn = lambda X: np.repeat(nrm[None, :], np.atleast_2d(X).shape[0], axis=0)

    def project_fn(X):
        X = np.atleast_2d(np.asarray(X, float)).copy()
        X[:, 1] = 0.0
        return X

    L, P = np.array([[-1.0, 2.0, 0.3]]), np.array([[0.9, -1.5, -0.2]])
    out = connect(normal_fn, project_fn, np.zeros((1, 3)), L, P, 1.0 / 1.5, iters=60)
    assert out["converged"][0], out["residual"][0]
    assert abs(out["X"][0, 1]) < 1e-12, "the vertex left the surface"
    X = out["X"]
    d_in = X - L; d_in /= np.linalg.norm(d_in, axis=-1, keepdims=True)
    d_ref, ok = refract(d_in, normal_fn(X), np.array([1.0 / 1.5]))
    to_P = P - X; to_P /= np.linalg.norm(to_P, axis=-1, keepdims=True)
    assert ok[0] and float(np.sum(d_ref * to_P)) > 1.0 - 1e-6


def test_total_internal_reflection_is_reported_not_invented():
    """Under TIR no refracted ray exists, so no connection exists. Returning one would draw light along
    a path physics forbids."""
    from holographic.rendering.holographic_specularconnect import refract
    d = np.array([[0.999, -0.045, 0.0]]); d /= np.linalg.norm(d)
    _, valid = refract(d, np.array([[0.0, 1.0, 0.0]]), np.array([1.5]))
    assert not bool(valid[0])


def test_specular_connect_is_wired_and_works_on_a_curved_sdf(mind):
    """The governing rule: reachable through the faculty surface. And a plane is not enough evidence --
    the tangent-plane Newton step has to survive curvature, where the normal changes under the step."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    out = mind.specular_connect(sphere(1.0), np.array([[0.0, 1.0, 0.0]]), np.array([[-1.0, 3.0, 0.4]]),
                                np.array([[0.6, -2.2, -0.3]]), 1.0 / 1.5)
    assert out["valid"][0] and out["converged"][0], out["residual"][0]
    # The solved vertex must lie ON the sphere, not merely near the answer.
    assert abs(float(np.linalg.norm(out["X"][0])) - 1.0) < 1e-3

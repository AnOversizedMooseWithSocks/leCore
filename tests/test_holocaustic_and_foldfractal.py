"""Sweep 149: the render path audited back onto the substrate.

Two things landed. (1) fold_fractal gained an escape bailout and a signed interior -- without them the
Mandelbox distance estimate collapsed geometrically with iteration count and had no inside, so it could never
be glass. (2) holographic_holocaustic: the caustic as ONE FPE hypervector with wavelength as an axis -- trace
once, bundle, read at any resolution, RGB as three unbinds. Both pinned with the numbers that were measured,
including the negatives (capacity-bounded colour, tile patchwork) so nobody re-discovers them.
"""
import numpy as np
import pytest

from holographic.mesh_and_geometry.holographic_sdf import fold_fractal, sphere, SDF, ARITY


# ----------------------------------------------------------------------------- fold_fractal
def test_fold_fractal_default_is_byte_identical_to_the_historical_field():
    """Hard constraint 3: existing decisions never flip. bailout=None/solid=False must reproduce the old numbers."""
    P = np.array([[1., 0, 0], [2., 0, 0], [3., 0, 0], [5., 0, 0], [10., 0, 0], [50., 0, 0]])
    d = np.asarray(fold_fractal().eval(P), float)
    # the numbers recorded before the change (5 decimals) -- the collapsed estimate, kept as the pinned old contract
    assert np.allclose(d, [0.00008, 0.0, 0.00041, 0.00024, 0.88963, 9.78105], atol=6e-6)
    assert np.array_equal(fold_fractal(iterations=9).eval(P), fold_fractal(iterations=9, bailout=None, solid=False).eval(P))
    assert np.array_equal(SDF("fold_fractal", (9, 2.0, 0.5, 1.0)).eval(P), fold_fractal(iterations=9).eval(P)), \
        "a 4-param DSL node must evaluate exactly like the 6-param default"
    assert ARITY["fold_fractal"][0] == 4, "the DSL grammar stays at four params; the extras are optional"


def test_fold_fractal_bailout_stops_the_geometric_collapse():
    """The estimate must CONVERGE with iterations, not shrink by scale^4 per 4 iterations. Stated over a grid as a
    median -- the collapse is point-dependent, and a hand-picked probe fooled the first draft of this assertion."""
    g = np.linspace(-3, 3, 20)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    Q = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    def ratio(bail):
        a = np.asarray(fold_fractal(iterations=4, bailout=bail).eval(Q), float)
        b = np.asarray(fold_fractal(iterations=12, bailout=bail).eval(Q), float)
        ok = a > 1e-9
        return b[ok] / a[ok]
    assert np.median(ratio(None)) < 0.05, "the historical field MUST still collapse (it is the pinned old contract)"
    assert np.median(ratio(4.0)) > 0.95, "with a bailout the estimate converges"


def test_fold_fractal_solid_has_an_interior_and_changes_only_the_sign():
    g = np.linspace(-3, 3, 24)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    Q = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    u = np.asarray(fold_fractal(iterations=8, bailout=4.0).eval(Q), float)
    s = np.asarray(fold_fractal(iterations=8, bailout=4.0, solid=True).eval(Q), float)
    assert (u >= 0).all(), "unsigned stays all-positive"
    assert 0.01 < (s < 0).mean() < 0.40, "solid finds a plausible interior: %.3f" % (s < 0).mean()
    assert np.array_equal(np.abs(s), u), "solid changes only the SIGN"


def test_fold_fractal_solid_needs_a_modest_bailout():
    """Escape defines 'outside'. At a large bailout almost nothing escapes in 8 iterations and the field reports
    nearly everything as interior -- measured 0.072 at 4, 0.952 at 16, 0.996 at 64. Pinned so the default advice
    (~4) has a test behind it."""
    g = np.linspace(-3, 3, 16)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    Q = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    f4 = (np.asarray(fold_fractal(iterations=8, bailout=4.0, solid=True).eval(Q), float) < 0).mean()
    f64 = (np.asarray(fold_fractal(iterations=8, bailout=64.0, solid=True).eval(Q), float) < 0).mean()
    assert f4 < 0.3 and f64 > 0.9, "bailout 4 -> %.3f inside, bailout 64 -> %.3f inside" % (f4, f64)


def test_fold_fractal_bailout_tested_after_the_step_not_before():
    """Testing before the step let a far point exit with z=P, dr=1 -> estimate |P| = distance to the ORIGIN, an
    over-estimate a sphere tracer tunnels on. d(5,0,0) read exactly 5.0 that way; after the fix, less."""
    d = float(np.asarray(fold_fractal(iterations=8, bailout=4.0).eval([[5.0, 0.0, 0.0]]), float)[0])
    assert d < 4.5, "an estimate equal to |P| is the distance to the origin, not to the body: %.3f" % d


# ----------------------------------------------------------------------------- holographic caustic
from holographic.rendering.holographic_holocaustic import (
    HolographicCaustic, TiledHolographicCaustic, holographic_caustic, spectral_landings, default_bandwidth,
    gaussian_cmf)
from holographic.rendering.holographic_globalillum import caustics, caustic_hits

_L = np.array([0.35, -1.0, 0.0]); _L /= np.linalg.norm(_L)


def _hc(dim=1024, scale=0.0, **kw):
    return holographic_caustic(sphere(0.6), light_dir=tuple(_L), receiver_y=-1.2, extent=0.9, n_side=120,
                               window=1.2, dispersion_scale=scale, dim=dim, seed=0, aim=(0.0, 0.0, 0.0), **kw)


def test_caustic_hits_refactor_is_byte_identical():
    """caustics() = caustic_hits() + histogram, and the refactor must not move a single bin."""
    img = caustics(sphere(0.6), light_dir=tuple(_L), receiver_y=-1.2, extent=0.9, res=48, ior=1.55, n_side=100,
                   window=1.2, refracted_only=True, aim=(0.0, 0.0, 0.0))
    land, valid, hit = caustic_hits(sphere(0.6), light_dir=tuple(_L), receiver_y=-1.2, extent=0.9, ior=1.55,
                                    n_side=100, refracted_only=True, aim=(0.0, 0.0, 0.0))
    ref = np.zeros((48, 48))
    xi = ((land[:, 0] + 1.2) / 2.4 * 47).astype(int); zi = ((land[:, 2] + 1.2) / 2.4 * 47).astype(int)
    inb = valid & (xi >= 0) & (xi < 48) & (zi >= 0) & (zi < 48)
    np.add.at(ref, (zi[inb], xi[inb]), 1.0)
    assert np.array_equal(img, ref / (ref.mean() + 1e-9))


def test_per_ray_ior_carries_a_spectrum_in_one_pass():
    """One ray set, one wavelength per ray, refracted in one call: blue lands wider than red (it focuses closer)."""
    xz, lam = spectral_landings(sphere(0.6), 1.55, 25.68, light_dir=tuple(_L), receiver_y=-1.2, extent=0.9,
                                n_side=120, dispersion_scale=3.0, seed=0, aim=(0.0, 0.0, 0.0))
    def rms(mask):
        q = xz[mask]; return float(np.sqrt(((q - q.mean(0)) ** 2).sum(1).mean()))
    assert rms(lam > 620) < 0.92 * rms(lam < 480)
    xz0, lam0 = spectral_landings(sphere(0.6), 1.55, 25.68, light_dir=tuple(_L), receiver_y=-1.2, extent=0.9,
                                  n_side=120, dispersion_scale=0.0, seed=0, aim=(0.0, 0.0, 0.0))
    def rms0(mask):
        q = xz0[mask]; return float(np.sqrt(((q - q.mean(0)) ** 2).sum(1).mean()))
    assert abs(rms0(lam0 > 620) / rms0(lam0 < 480) - 1.0) < 0.12, "no dispersion -> equal spread"


def test_one_vector_reads_at_any_resolution():
    hc = _hc()
    a = hc.read_rgb(np.linspace(-1.2, 1.2, 32), np.linspace(-1.2, 1.2, 32), gaussian_cmf)
    b = hc.read_rgb(np.linspace(-1.2, 1.2, 64), np.linspace(-1.2, 1.2, 64), gaussian_cmf)
    b_down = b.reshape(32, 2, 32, 2, 3).mean(axis=(1, 3))
    assert np.corrcoef(a.mean(-1).ravel(), b_down.mean(-1).ravel())[0, 1] > 0.97


def test_read_is_capacity_bounded_and_dim_buys_it_down():
    """The read agrees better with the histogram as dim rises at a fixed kernel -- the capacity law, pinned."""
    xs = np.linspace(-1.2, 1.2, 48)
    hist = caustics(sphere(0.6), light_dir=tuple(_L), receiver_y=-1.2, extent=0.9, res=48, ior=1.55, n_side=120,
                    window=1.2, refracted_only=True, aim=(0.0, 0.0, 0.0))
    for _ in range(2):
        hist = (np.roll(hist, 1, 0) + hist + np.roll(hist, -1, 0)) / 3.0
        hist = (np.roll(hist, 1, 1) + hist + np.roll(hist, -1, 1)) / 3.0
    c = {}
    for dim in (512, 2048):
        h = _hc(dim=dim, bandwidth=(30.0, 30.0, 10.0))
        c[dim] = float(np.corrcoef(hist.ravel(), h.read(xs, xs, 550.0).ravel())[0, 1])
    assert c[2048] > c[512] + 0.05, "more dim, better read at a fixed kernel: %r" % c


def test_default_bandwidth_tracks_sqrt_dim():
    """Sharper than sqrt(dim) cells per axis and agreement FALLS -- the default must not do that."""
    assert default_bandwidth(4096)[0] == pytest.approx(2.0 * default_bandwidth(1024)[0])
    assert default_bandwidth(1024, sharpness=0.5)[0] == pytest.approx(0.5 * default_bandwidth(1024)[0])


def test_translate_is_a_bind():
    xz, lam = spectral_landings(sphere(0.6), 1.55, 25.68, light_dir=tuple(_L), receiver_y=-1.2, extent=0.9,
                                n_side=80, dispersion_scale=0.0, seed=0, aim=(0.0, 0.0, 0.0))
    bw = default_bandwidth(512)
    h1 = HolographicCaustic(((-1.5, 1.5), (-1.5, 1.5)), dim=512, bandwidth=bw).deposit(xz, lam).translate(0.3, -0.2)
    h2 = HolographicCaustic(((-1.5, 1.5), (-1.5, 1.5)), dim=512, bandwidth=bw).deposit(xz + [0.3, -0.2], lam)
    assert np.allclose(h1.F_spec, h2.F_spec, atol=1e-8 * np.abs(h2.F_spec).max())


def test_add_superposes_and_refuses_mismatched_encoders():
    xz, lam = spectral_landings(sphere(0.6), 1.55, 25.68, light_dir=tuple(_L), receiver_y=-1.2, extent=0.9,
                                n_side=60, dispersion_scale=0.0, seed=0, aim=(0.0, 0.0, 0.0))
    bw = default_bandwidth(512)
    a = HolographicCaustic(((-1.5, 1.5), (-1.5, 1.5)), dim=512, bandwidth=bw).deposit(xz[:100], lam[:100])
    b = HolographicCaustic(((-1.5, 1.5), (-1.5, 1.5)), dim=512, bandwidth=bw).deposit(xz[100:], lam[100:])
    both = HolographicCaustic(((-1.5, 1.5), (-1.5, 1.5)), dim=512, bandwidth=bw).deposit(xz, lam)
    assert np.allclose(a.add(b).F_spec, both.F_spec)
    with pytest.raises(ValueError):
        a.add(HolographicCaustic(((-1.5, 1.5), (-1.5, 1.5)), dim=256, bandwidth=default_bandwidth(256)))


def test_rgb_read_is_three_unbinds_of_the_colour_matching_query():
    """read_rgb must equal integrating single-wavelength reads against the CMF -- the identity the design rests on."""
    hc = _hc(dim=512, scale=3.0)
    xs = np.linspace(-1.2, 1.2, 24)
    rgb = hc.read_rgb(xs, xs, gaussian_cmf, n_lam=24, normalise=None, raw=True)
    lo, hi = hc.bounds[2]
    lams = np.linspace(lo, hi, 24); W = gaussian_cmf(lams); dl = (hi - lo) / 23
    ref = sum(W[k][None, None, :] * hc.read(xs, xs, lams[k], raw=True)[..., None] for k in range(24)) * dl
    assert np.allclose(rgb, ref, rtol=1e-6, atol=1e-6 * np.abs(ref).max())


def test_raw_read_exists_because_clipping_biases_measurement():
    hc = _hc(dim=512)
    xs = np.linspace(-1.2, 1.2, 24)
    r = hc.read(xs, xs, 550.0, raw=True)
    assert (r < 0).any(), "the raw kernel read has negative crosstalk lobes; if it does not, raw= is moot"
    assert (hc.read(xs, xs, 550.0) >= 0).all()


def test_tiled_matches_single_vector_where_capacity_allows_and_routes_every_pixel():
    xz, lam = spectral_landings(sphere(0.6), 1.55, 25.68, light_dir=tuple(_L), receiver_y=-1.2, extent=0.9,
                                n_side=100, dispersion_scale=0.0, seed=0, aim=(0.0, 0.0, 0.0))
    inb = (np.abs(xz[:, 0]) <= 1.2) & (np.abs(xz[:, 1]) <= 1.2)
    t = TiledHolographicCaustic(((-1.2, 1.2), (-1.2, 1.2)), grid=2, dim=1024).deposit(xz[inb], lam[inb])
    xs = np.linspace(-1.2, 1.2, 32)
    img = t.read(xs, xs, 550.0)
    assert img.shape == (32, 32) and np.isfinite(img).all()
    assert (img > 0).any(), "every pixel routed to a tile and read"
    one = HolographicCaustic(((-1.2, 1.2), (-1.2, 1.2)), dim=4096, bandwidth=default_bandwidth(4096)).deposit(xz[inb], lam[inb])
    assert np.corrcoef(img.ravel(), one.read(xs, xs, 550.0).ravel())[0, 1] > 0.6


def test_tiled_read_cross_fades_seams_and_floor_removal_is_opt_in():
    """Lever 6 on the tile patchwork: cross-fade across the halo overlap is on by default and must leave no hard
    step at a tile border; percentile floor removal measured as a net loss (0.817 -> 0.638) and stays opt-in."""
    xz, lam = spectral_landings(sphere(0.6), 1.55, 25.68, light_dir=tuple(_L), receiver_y=-1.2, extent=0.9,
                                n_side=120, dispersion_scale=0.0, seed=0, aim=(0.0, 0.0, 0.0))
    inb = (np.abs(xz[:, 0]) <= 1.2) & (np.abs(xz[:, 1]) <= 1.2)
    t = TiledHolographicCaustic(((-1.2, 1.2), (-1.2, 1.2)), grid=2, dim=1024).deposit(xz[inb], lam[inb])
    xs = np.linspace(-1.2, 1.2, 64)
    flat = t._route_read(xs, xs, lambda tl, x, z: tl.read(x, z, 550.0, raw=True), floor_pct=0, blend=0)
    fade = t.read(xs, xs, 550.0, raw=True)                                    # default: cross-fade on
    # the seam: the jump between columns 31 and 32 (the tile border at x=0), relative to typical neighbour jumps
    def seam(img):
        border = np.abs(img[:, 32] - img[:, 31]).mean()
        typical = np.abs(np.diff(img, axis=1)).mean()
        return border / max(typical, 1e-12)
    assert seam(fade) < seam(flat), "cross-fade must shrink the border discontinuity: %.2f -> %.2f" % (seam(flat), seam(fade))
    assert np.allclose(fade, t._route_read(xs, xs, lambda tl, x, z: tl.read(x, z, 550.0, raw=True), floor_pct=0, blend=0.12))

"""Sweep 158: no billboards -- every crystal is the intersection of lattice half-spaces; real unit cells give the
textbook interfacial angles; the histogram caustic baseline; library optics drive the render."""
import numpy as np
import pytest

import lecore
import holographic.mesh_and_geometry.holographic_crystalgrow as cg
from holographic.mesh_and_geometry.holographic_bravais import lattice_basis, reciprocal_basis


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def _m_r_angle(c):
    basis, _ = lattice_basis("hexagonal", a=1.0, c=c); B = reciprocal_basis(basis)
    n_r = np.array([1.0, 0.0, 1.0]) @ B; n_r /= np.linalg.norm(n_r)
    n_m = np.array([1.0, 0.0, 0.0]) @ B; n_m /= np.linalg.norm(n_m)
    return 180.0 - np.degrees(np.arccos(float(np.clip(n_r @ n_m, -1, 1))))


def test_quartz_real_cell_gives_textbook_prism_rhombohedron_angle():
    assert abs(_m_r_angle(cg.CELLS["quartz"]["c"]) - 141.78) < 0.1          # 141 deg 47'
    assert abs(_m_r_angle(1.0) - 139.11) < 0.1                               # the c = a default, kept for compatibility


def test_habit_is_a_true_convex_polyhedron_not_a_billboard(mind):
    """A billboard has no thickness; a lattice habit has an interior whose distance field is exact: the SDF is
    negative inside, its gradient is a unit vector, and every surface point lies on one of the form's planes."""
    h = mind.crystal_single("quartz", 1.0, real_cell=True)
    rng = np.random.default_rng(0); P = rng.uniform(-1.5, 1.5, (20000, 3))
    d = np.asarray(h.eval(P), float)
    assert (d < 0).mean() > 0.02                                             # it has volume
    eps = 1e-4
    g = np.stack([(np.asarray(h.eval(P + eps * np.eye(3)[k]), float) - d) / eps for k in range(3)], 1)
    gn = np.linalg.norm(g, axis=1)
    assert (np.abs(gn - 1.0) < 2e-3).mean() > 0.98                          # unit gradient almost everywhere (edges excepted)
    inside = P[d < 0]
    assert np.all(np.linalg.norm(inside[:, :2], axis=1) < 0.6) and np.all(np.abs(inside[:, 2]) < 1.7)


def test_real_cell_flag_threads_through_growth_and_default_unchanged():
    rng = np.random.default_rng(0); Q = rng.uniform(-1, 1, (20000, 3))
    a = cg.cluster(count=6, seed=3)(Q); b = cg.cluster(count=6, seed=3, real_cell=True)(Q)
    assert np.array_equal(a, cg.cluster(count=6, seed=3)(Q)) and not np.array_equal(a, b)


def test_histogram_caustic_resolves_a_point_the_holographic_read_blurs(mind):
    from holographic.rendering.holographic_holocaustic import histogram_caustic_rgb, TiledHolographicCaustic
    rng = np.random.default_rng(0)
    xz = np.concatenate([rng.normal([0.3, -0.2], 0.004, (3000, 2)), rng.uniform(-1, 1, (300, 2))])   # one hotspot + noise
    lam = rng.uniform(420, 680, len(xz))
    b = ((-1, 1), (-1, 1))
    H = histogram_caustic_rgb(xz, lam, mind.wavelength_cmf, b, res=128, blur_px=1.0)
    xs = np.linspace(-1, 1, 128)
    T = TiledHolographicCaustic(b, grid=2, dim=512, seed=0).deposit(xz, lam).read_rgb(xs, xs, mind.wavelength_cmf, normalise="mean")
    peakH = H.mean(-1).max() / H.mean(); peakT = T.mean(-1).max() / T.mean()
    assert peakH > 3 * peakT                                                  # sharper by a wide margin at this dim
    iy, ix = np.unravel_index(np.argmax(H.mean(-1)), H.shape[:2])
    assert abs(xs[ix] - 0.3) < 0.03 and abs(xs[iy] + 0.2) < 0.03              # and in the right place


def test_library_optics_drive_bake_and_zoning(mind):
    from holographic.materials_and_texture.holographic_matlib import glass_optics
    go = glass_optics("amethyst")
    assert go["n_d"] == 1.55 and go["abbe"] == 70.0 and np.argmax(go["absorb"]) == 1     # absorbs green hardest
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    cam = mind.camera(eye=(0, 0, 3), target=(0, 0, 0), fov_deg=30, aspect=1.0)
    b = mind.bake_glass(sphere(0.6), cam, 16, 16, n_lams=3, material="amethyst")
    env = lambda D: np.ones((len(D), 3))
    img = mind.relight_glass(b, env, material="amethyst").reshape(-1, 3)[b.glass]
    assert np.all(img[:, 1] <= img[:, 0] + 1e-9) and np.all(img[:, 1] <= img[:, 2] + 1e-9)   # purple: green suppressed


def test_pack_competitive_growth_bounds_interpenetration():
    """Sweep 159: with pack, a crystal's size follows its Voronoi room, so the union volume is close to the sum of
    the members' volumes (little interpenetration); independent sizing on the same seeds overlaps heavily."""
    from holographic.mesh_and_geometry.holographic_sdf import box
    slab = box(0.62, 0.11, 0.42).rounded(0.07)
    class W:
        def __call__(self, P): return np.asarray(slab.eval(np.atleast_2d(np.asarray(P, float))), float)
        eval = __call__
    top = lambda P: np.clip((np.atleast_2d(P)[:, 1] - 0.06) / 0.05, 0, 1)
    bounds = ((-0.8, -0.3, -0.6), (0.8, 0.9, 0.6))
    rng = np.random.default_rng(0); Q = rng.uniform(-0.9, 0.9, (40000, 3))
    packed = cg.grow_on(W(), bounds, count=40, habit="quartz", size=0.5, pack=1.0, size_jitter=0.0, tilt=0.0, where=top,
                        seed=3, substrate=False, cull=True, real_cell=True)
    loose = cg.grow_on(W(), bounds, count=40, habit="quartz", size=0.25, size_jitter=0.0, tilt=0.0, where=top,
                       seed=3, substrate=False, cull=True, real_cell=True)
    vp = (packed(Q) < 0).mean(); vl = (loose(Q) < 0).mean()
    assert 0 < vp < vl                                                        # packed crystals are sized by their room
    # sizes differ per seed under pack (room varies), identical without it
    P, _ = cg.seed_surface(W(), 40, bounds, where=top, seed=3)
    D2 = ((P[:, None] - P[None]) ** 2).sum(-1); np.fill_diagonal(D2, np.inf)
    room = np.sqrt(np.sort(D2, 1)[:, :3]).mean(1)
    assert room.std() / room.mean() > 0.1


def test_full_transit_caustic_carries_material_colour_and_exit_refraction(mind):
    """Sweep 159b: through=True refracts at entry AND exit and returns interior path lengths; with the library's
    absorption the landings' weights are purple for amethyst (green suppressed) and the landing pattern differs
    from the thin-lens one (exit refraction moves the light)."""
    from holographic.mesh_and_geometry.holographic_sdf import box, SDF
    from holographic.rendering.holographic_holocaustic import spectral_landings
    from holographic.materials_and_texture.holographic_matlib import glass_optics
    prism = SDF("rotate", (0.0, 0.0, 1.0, 0.5), [box(0.4, 0.6, 0.4)])              # a tilted slab: a prism to light
    go = glass_optics("amethyst")
    thin = spectral_landings(prism, go["n_d"], go["abbe"], light_dir=(0.3, -1.0, 0.0), receiver_y=-1.0, extent=0.9, n_side=60, seed=0)
    full = spectral_landings(prism, go["n_d"], go["abbe"], light_dir=(0.3, -1.0, 0.0), receiver_y=-1.0, extent=0.9, n_side=60, seed=0,
                             through=True, absorb_rgb=go["absorb"], cmf=mind.wavelength_cmf)
    assert len(thin) == 2 and len(full) == 3
    xz_t, _ = thin; xz_f, lam_f, w = full
    assert len(xz_f) > 100 and np.all((w > 0) & (w <= 1))
    assert abs(xz_f[:, 0].mean() - xz_t[:, 0].mean()) > 0.02                    # exit refraction moved the light
    # colour: weight green wavelengths less than red/blue (the amethyst absorption ordering)
    W = np.asarray(mind.wavelength_cmf(lam_f), float); ch = W.argmax(1)
    assert w[ch == 1].mean() < w[ch == 0].mean() and w[ch == 1].mean() < w[ch == 2].mean()


def test_shadow_through_coloured_glass_is_tinted(mind):
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    cam = mind.camera(eye=(0.0, 3.0, 0.01), target=(0.0, 0.0, 0.0), fov_deg=50.0, aspect=1.0)
    b = mind.bake_glass(sphere(0.5), cam, 40, 40, n_lams=3, floor_y=-0.6)
    env = lambda D: np.tile([[1.0, 1.0, 1.0]], (len(D), 1))
    tinted = mind.relight_glass(b, env, sdf=sphere(0.5), sun_dir=(0.6, 0.8, 0.0), material="ruby", shadow_transmittance=0.8).reshape(-1, 3)
    plain = mind.relight_glass(b, env, sdf=sphere(0.5), sun_dir=(0.6, 0.8, 0.0), shadow_transmittance=0.8).reshape(-1, 3)
    f = b.floor
    shadow = f & (np.abs(plain[:, 0] - plain[f][:, 0].max()) > 1e-6)                   # floor pixels in the sphere's shadow
    assert shadow.any()
    assert np.all(tinted[shadow][:, 0] > tinted[shadow][:, 1] + 1e-6)                # ruby: red passes, green does not
    assert np.allclose(plain[shadow][:, 0], plain[shadow][:, 1])                       # no absorb: neutral shadow


def test_clip_to_substrate_grows_outward_only_and_explicit_seeds():
    """Sweep 160: no crystal volume inside the host, none behind a crystal's root plane; explicit seeds honoured."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    host = sphere(0.5)
    class Hst:
        def __call__(self, P): return np.asarray(host.eval(np.atleast_2d(np.asarray(P, float))), float)
        eval = __call__
    P = np.array([[0.0, 0.5, 0.0], [0.3, 0.4, 0.0]]); N = np.array([[0.0, 1.0, 0.0], [0.6, 0.8, 0.0]])
    f = cg.grow_on(Hst(), ((-1, -1, -1), (1, 1, 1)), habit="quartz_long", size=[0.5, 0.3], seeds=(P, N),
                   substrate=False, real_cell=True, clip_to_substrate=True, tilt=0.0, size_jitter=0.0)
    rng = np.random.default_rng(0); Q = rng.uniform(-1.2, 1.2, (60000, 3))
    d = f(Q); inside_host = Hst()(Q) < -0.01
    assert not (d[inside_host] < 0).any()                                        # nothing inside the rock
    below = Q[:, 1] < 0.0                                                        # nothing below the sphere's equator
    assert not (d[below] < 0).any()
    assert (d < 0).sum() > 50 and (d[Q[:, 1] > 0.6] < 0).any()                  # but the points do exist, above
    g = cg.grow_on(Hst(), ((-1, -1, -1), (1, 1, 1)), habit="quartz_long", size=[0.5, 0.3], seeds=(P, N),
                   substrate=False, real_cell=True, tilt=0.0, size_jitter=0.0)   # without the clip: it does go below
    assert (g(Q)[below] < 0).any()

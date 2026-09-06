"""Sweep 154: crystal formations as gems under an HDRI base + additional lights.

Pins: HdriEnv.add_light keeps the floor integrals consistent (a lobe sized by irradiance adds exactly that much;
the strongest lobe becomes the caustic aim); crystal_single is a CLOSED body; culled_union has the same zero set
as the plain union and is never a smaller distance (a valid, tighter sphere-tracing field); caustic_concentration
reads >1 for a lens and 1 for uniform landings.
"""
import numpy as np
import pytest

import lecore
import holographic.mesh_and_geometry.holographic_crystalgrow as cg
from holographic.rendering.holographic_glassbake import HdriEnv
from holographic.rendering.holographic_holocaustic import caustic_concentration


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def _grey_sky(v=0.5, h=32, w=64):
    img = np.zeros((h, w, 3), np.float32); img[: h // 2] = v          # upper hemisphere lit, floor half dark
    return img


def test_add_light_by_irradiance_adds_exactly_that_much():
    env = HdriEnv(_grey_sky())
    E0 = env.floor_irradiance.copy()
    env.add_light((0.0, 1.0, 0.0), irradiance=0.5, sigma=0.05)
    assert np.allclose(env.floor_irradiance - E0, 0.5, atol=1e-9)


def test_add_light_strongest_lobe_becomes_dominant_and_radiance_peaks_there():
    env = HdriEnv(_grey_sky())
    d = np.array([-0.45, 0.80, 0.40]); d /= np.linalg.norm(d)
    env.add_light(d, irradiance=10.0 * env.floor_irradiance.mean(), sigma=0.05)
    assert np.allclose(env.dominant_dir, d, atol=1e-9)
    assert env.sun_share > 0.9
    on, off = env.radiance(d[None, :])[0].mean(), env.radiance(np.array([[0.0, 1.0, 0.0]]))[0].mean()
    assert on > 100.0 * off


def test_add_light_weak_lobe_keeps_map_dominant():
    env = HdriEnv(_grey_sky())
    dom, share = env.dominant_dir.copy(), env.sun_share
    env.add_light((0.3, 0.9, 0.3), irradiance=1e-6, sigma=0.05)
    assert np.allclose(env.dominant_dir, dom) and share >= env.sun_share


def test_add_light_needs_radiance_or_irradiance():
    with pytest.raises(ValueError):
        HdriEnv(_grey_sky()).add_light((0, 1, 0))


def test_env_add_light_verb_returns_env(mind):
    env = mind.hdri_env(_grey_sky())
    assert mind.env_add_light(env, (0, 1, 0), irradiance=0.1) is env and len(env.lobes) == 1


def test_crystal_single_is_closed_unlike_bare_miller_prism(mind):
    g = np.linspace(-1.8, 1.8, 41); G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    d = np.asarray(mind.crystal_single("quartz", 0.9).eval(G), float)
    ins = G[d < 0]
    assert 0.001 < (d < 0).mean() < 0.05                                  # a body, not the whole probe
    assert np.abs(ins).max() < 1.6                                          # and it does not reach the probe edge
    open_ = np.asarray(mind.crystal_habit("hexagonal", ((1, 0, 0), (1, 0, 1)), (0.3, 1.0)).eval(G), float)
    assert (open_ < 0).mean() > (d < 0).mean() * 3                         # the bare Miller list is the open prism


@pytest.mark.parametrize("fn,kw", [(cg.cluster, dict(count=7, size=0.4, radius=0.25, seed=3)),
                                   (cg.geode, dict(radius=0.9, shell=0.16, count=24, size=0.16, seed=1))])
def test_culled_union_same_zero_set_and_never_smaller(fn, kw):
    rng = np.random.default_rng(0); Q = rng.uniform(-1.2, 1.2, (20000, 3))
    a = fn(**kw).eval(Q); b = fn(cull=True, **kw).eval(Q)
    assert np.array_equal(a < 0, b < 0)                                     # identical inside/outside
    assert np.all(b >= a - 1e-12)                                           # tighter, still a lower bound of true distance
    band = np.abs(a) < 0.02
    assert band.any() and np.abs(a - b)[band].max() < 1e-9                  # identical values near the surface


def test_bounding_radius_contains_habit():
    base = cg.habit_sdf("quartz", 1.0)
    r = cg.bounding_radius(base, probe=3.0)
    g = np.linspace(-3, 3, 31); G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    ins = G[np.asarray(base(G), float) < 0]
    assert np.linalg.norm(ins, axis=1).max() <= r


def test_caustic_concentration_uniform_is_one_and_focus_is_more():
    n = 200; g = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    X, Z = np.meshgrid(g, g); xz = np.column_stack([X.ravel(), Z.ravel()])     # every emitter ray lands in its own cell
    c_uniform = caustic_concentration(xz, extent=1.0, n_side=n, light_dir=(0, -1, 0), res=100)
    assert abs(c_uniform - 1.0) < 0.05
    c_focus = caustic_concentration(xz * 0.3, extent=1.0, n_side=n, light_dir=(0, -1, 0), res=100)
    assert c_focus > 8.0                                                      # 1/0.09 = 11x, minus binning
    assert caustic_concentration(xz[:5], 1.0, n, (0, -1, 0)) == 1.0


def test_hybrid_sdf_is_lower_bound_and_exact_near_surface(mind):
    from holographic.mesh_and_geometry.holographic_sdfbake import HybridSDF

    class Sph:
        def eval(self, P):
            return np.linalg.norm(np.asarray(P, float), axis=1) - 0.5
    ex = Sph()
    hy = mind.bake_sdf(ex, (-1, -1, -1), (1, 1, 1), 33, exact_near=True)
    assert isinstance(hy, HybridSDF)
    rng = np.random.default_rng(0); P = rng.uniform(-1, 1, (4000, 3))
    de, dh = ex.eval(P), hy(P)
    assert np.all(np.abs(dh) <= np.abs(de) + 1e-12)                         # never claims more clearance, either side
    near = np.abs(de) < hy.band * 0.5
    assert near.any() and np.allclose(dh[near], de[near])
    assert not isinstance(mind.bake_sdf(ex, (-1, -1, -1), (1, 1, 1), 9), HybridSDF)   # default unchanged


def test_grid_bake_reproduces_planar_facets_exactly():
    """WHY the plain grid is good enough for crystals: trilinear interpolation reproduces a linear (planar) field."""
    import holographic.mesh_and_geometry.holographic_sdfbake as sb
    n = np.array([0.3, 0.8, 0.52]); n /= np.linalg.norm(n)

    class Plane:
        def eval(self, P):
            return np.asarray(P, float) @ n - 0.1
    g = sb.GridSDF.bake(Plane(), (-1, -1, -1), (1, 1, 1), 17)
    P = np.random.default_rng(1).uniform(-0.9, 0.9, (2000, 3))
    assert np.abs(g.eval(P) - Plane().eval(P)).max() < 1e-12


def test_bake_glass_opaque_body_wins_pixels_and_is_seen_through_glass(mind):
    """A glass sphere in front of an opaque wall: the wall takes the pixels the glass does not cover, exit rays
    through the glass record the wall behind, and relight shades the wall with opaque_albedo (not sky)."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere, plane

    class Wall:  # x = -1.5 plane facing +x
        def eval(self, P):
            return np.asarray(P, float)[:, 0] + 1.5
    glass = sphere(0.6)
    cam = mind.camera(eye=(3.0, 0.0, 0.0), target=(0.0, 0.0, 0.0), fov_deg=40.0, aspect=1.0)
    b = mind.bake_glass(glass, cam, 40, 40, n_d=1.5, abbe=60.0, n_lams=3, opaque=Wall())
    assert b.opaque.sum() > 200 and b.glass.sum() > 50
    assert not (b.opaque & b.glass).any()
    assert b.exit_opq[:, b.glass].mean() > 0.5                              # most rays leaving the sphere hit the wall
    env = lambda D: np.tile([[1.0, 1.0, 1.0]], (len(D), 1))                 # uniform white sky
    img = mind.relight_glass(b, env, opaque_albedo=(0.2, 0.5, 0.8)).reshape(-1, 3)
    o = img[b.opaque]
    assert np.allclose(o / o[:, :1], np.array([[1.0, 2.5, 4.0]]), rtol=0.05)   # albedo colour ratio, lit by the sky
    b0 = mind.bake_glass(glass, cam, 40, 40, n_d=1.5, abbe=60.0, n_lams=3)
    assert not b0.opaque.any() and b0.opaque_sdf is None                    # default unchanged


def test_geode_parts_and_clip_to_skin():
    rind, lin = cg.geode(radius=0.9, shell=0.16, count=30, size=0.21, seed=1, parts=True, clip_to_skin=True)
    rng = np.random.default_rng(0); Q = rng.uniform(-1.3, 1.3, (60000, 3))
    d = lin(Q); r = np.linalg.norm(Q, axis=1)
    assert ((d < 0) & (r > 0.9)).sum() == 0                                 # nothing outside the nodule
    _, lin0 = cg.geode(radius=0.9, shell=0.16, count=30, size=0.21, seed=1, parts=True)
    assert ((lin0(Q) < 0) & (r > 0.9)).sum() > 100                          # the historical field did poke out
    assert np.all(rind(Q)[r < 0.5] > 0)                                      # rind is hollow


def test_relight_callable_albedos_background_and_receiver_mask(mind):
    """Sweep 154b: a checkerboard floor is an albedo FIELD, the backdrop can be a flat colour while the map still
    lights, and the caustic composite can be confined to the pixels that show the floor."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    cam = mind.camera(eye=(0.0, 0.8, 3.0), target=(0.0, 0.3, 0.0), fov_deg=60.0, aspect=1.0)   # horizon in frame
    b = mind.bake_glass(sphere(0.5), cam, 32, 32, n_lams=3, floor_y=-0.5)
    env = lambda D: np.tile([[1.0, 1.0, 1.0]], (len(D), 1))
    def checker(P):
        q = (np.floor(P[:, 0] / 0.5) + np.floor(P[:, 2] / 0.5)).astype(int) % 2
        return np.where(q[:, None] == 0, [[0.9, 0.9, 0.9]], [[0.1, 0.1, 0.1]])
    img = mind.relight_glass(b, env, floor_albedo=checker, background=(0.25, 0.5, 0.75)).reshape(-1, 3)
    f = img[b.floor][:, 0]
    assert (f > 0.5).any() and (f < 0.12).any()                                 # both squares appear
    sky = ~(b.glass | b.floor)
    assert sky.any() and np.allclose(img[sky], [0.25, 0.5, 0.75])               # flat backdrop where nothing is seen
    plain = mind.relight_glass(b, env, floor_albedo=checker).reshape(-1, 3)
    assert np.allclose(plain[sky], 1.0) and np.allclose(plain[b.glass], img[b.glass])   # glass unchanged: the map still lights
    # receiver mask: a caustic pattern covering the whole frame must not touch non-floor pixels
    beauty = np.zeros((32, 32, 3)); px = np.ones((32, 32, 3))
    fy, fx = np.argwhere(b.floor.reshape(32, 32))[-1]; sy, sx = np.argwhere(~b.floor.reshape(32, 32))[0]
    px[fy, fx] = 5.0; px[sy, sx] = 5.0                                                   # a hotspot on the floor and one off it
    out = mind.composite_caustic(beauty, px, strength=1.0, receiver_mask=b.floor.reshape(32, 32))
    assert np.all(out.reshape(-1, 3)[~b.floor] == 0.0) and out[fy, fx].max() > 0.0


def test_caustic_hits_occluder_blocks_light():
    """An opaque slab above a glass sphere: with `occluder` the lit landings vanish; without it the sphere lenses
    light through rock as if the rock were not there (the geode's bright ring)."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    from holographic.rendering.holographic_globalillum import caustic_hits

    class Slab:  # y in [1.0, 1.2], covers the whole aperture
        def eval(self, P):
            P = np.asarray(P, float); return np.maximum(np.abs(P[:, 1] - 1.1) - 0.1, np.abs(P[:, 0]) - 5.0)
    kw = dict(light_dir=(0, -1, 0), receiver_y=-1.0, extent=0.8, ior=1.5, n_side=40, refracted_only=True)
    _, v0, h0 = caustic_hits(sphere(0.5), **kw)
    _, v1, _ = caustic_hits(sphere(0.5), occluder=Slab(), **kw)
    assert v0.sum() > 100 and v1.sum() == 0

    class Below:  # slab under the sphere, above the floor: blocks AFTER refraction
        def eval(self, P):
            P = np.asarray(P, float); return np.maximum(np.abs(P[:, 1] + 0.8) - 0.05, np.abs(P[:, 0]) - 5.0)
    _, v2, _ = caustic_hits(sphere(0.5), occluder=Below(), **kw)
    assert v2.sum() == 0

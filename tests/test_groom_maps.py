"""Groom attribute maps (density + length) and mammal-skin SSS.

The production workflow, per SideFX/Houdini: paint a DENSITY attribute on the skin to say
where hair grows, and a SEPARATE LENGTH attribute to say how long -- "hairs around the nose
and snout shorter... at the base of the neck longer". A beard and a scalp are one groom with
two maps. The axis-aligned bounds box groom_hair uses cannot express that, which is why
strands kept growing on foreheads and necks.
"""
import numpy as np
from holographic.mesh_and_geometry import holographic_groommap as GM


def _sphere(m, res=20):
    sph = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    return m.mesh_from_sdf(sph, ((-1.3,) * 3, (1.3,) * 3), res=res, vectorized=True)


def test_length_map_separates_two_regions():
    """The beard/scalp case: one groom, two lengths. If these collapse to one value the map
    is not being consulted."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    V = np.asarray(_sphere(m).vertices, float)
    ln = GM.region_map(V, [{"lo": (-2, 0.3, -2), "hi": (2, 2, 2), "value": 1.0},
                           {"lo": (-2, -2, -2), "hi": (2, -0.3, 2), "value": 0.0}])
    assert ln[V[:, 1] > 0.5].mean() > 0.9
    assert ln[V[:, 1] < -0.5].mean() < 0.1


def test_density_map_culls_strands_outside_the_region():
    """Density is what stops hair growing on a forehead. Zero density must grow nothing."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    mesh = _sphere(m)
    V = np.asarray(mesh.vertices, float)
    sph = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    raw = m.groom_hair(sph, n_strands=400, bounds=((-1.2,) * 3, (1.2,) * 3), length=0.2,
                       n_pts=6, seed=1)
    dens = GM.region_map(V, [{"lo": (-2, 0.2, -2), "hi": (2, 2, 2), "value": 1.0}])
    ln = np.ones(len(V))
    kept = GM.groom_with_maps(list(raw), V, dens, ln, base_length=1.0, seed=0)
    assert 0 < len(kept) < len(raw)
    for s in kept:                                  # every survivor is in the region
        i = int(np.argmin(np.linalg.norm(V - np.asarray(s.root, float), axis=1)))
        assert dens[i] > 0.0
    none = GM.groom_with_maps(list(raw), V, np.zeros(len(V)), ln,
                              base_length=1.0, seed=0)
    assert none == []


def test_smoothing_softens_the_region_edge():
    """A hard density edge reads as a shaved line; real hairlines fade."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    mesh = _sphere(m)
    V = np.asarray(mesh.vertices, float)
    hard = GM.region_map(V, [{"lo": (-2, 0.0, -2), "hi": (2, 2, 2), "value": 1.0}])
    soft = GM.smooth_map(V, mesh.faces, hard, m, iters=5)
    assert np.std(soft) < np.std(hard)
    assert soft.min() >= 0.0 and soft.max() <= 1.0


def test_sss_wraps_past_the_terminator_and_reddest():
    """Mammal skin is not Lambertian. At N.L = 0 a Lambert surface is BLACK; skin is not,
    because light entered elsewhere and came back out -- and what comes back is RED, since
    red scatters furthest. The wrap widths are tissue_pbr('skin')'s MEASURED radii."""
    dark = GM.sss_shade((0.6, 0.44, 0.36), np.array([0.0]), np.array([1.0]))
    assert dark[0, 0] > 0.0                       # not black at the terminator
    assert dark[0, 0] > dark[0, 1] > dark[0, 2]   # red wraps furthest, blue least
    lit = GM.sss_shade((0.6, 0.44, 0.36), np.array([1.0]), np.array([1.0]))
    assert lit[0, 0] >= dark[0, 0]                # still monotone in N.L
    thin = GM.sss_shade((0.6, 0.44, 0.36), np.array([0.0]), np.array([1.0]))
    thick = GM.sss_shade((0.6, 0.44, 0.36), np.array([0.0]), np.array([0.0]))
    assert thin[0, 0] > thick[0, 0]               # thin regions (ears) glow more

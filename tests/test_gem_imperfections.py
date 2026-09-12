"""Sweep 155: imperfections inside the gem, holographically -- one FPE hypervector per flaw density, closed-form
integral along each pixel's recorded interior path; colour zoning the same way; mixed habits in growth."""
import numpy as np
import pytest

import lecore
import holographic.mesh_and_geometry.holographic_crystalgrow as cg
from holographic.rendering.holographic_gemvolume import FlawVolume, flaw_shading


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_flaw_volume_closed_form_matches_true_density_march():
    rng = np.random.default_rng(0); C = rng.uniform(-0.5, 0.5, (12, 3))
    fv = FlawVolume.from_inclusions(C, radius=0.08, bounds=[(-1, 1)] * 3, dim=2048)
    O = np.array([[-0.9, C[3, 1], C[3, 2]]]); D = np.array([[1.0, 0.0, 0.0]]); L = 1.8
    tau = fv.segments_optical_depth([(O, O + D * L, np.array([0]))], 1)[0]
    ts = (np.arange(400) + 0.5) / 400 * L
    march = float(np.clip(fv.reference(O + ts[:, None] * D), 0, None).sum() * (L / 400))
    assert abs(tau - march) / march < 0.3                                  # one fitted scale; shape exact
    far = fv.optical_depth(np.array([[-0.9, 0.95, 0.95]]), D, L)[0]
    assert far < 0.3 * tau                                                  # crosstalk floor, not signal


def test_segments_guard_escaped_and_out_of_bounds():
    fv = FlawVolume.from_inclusions(np.zeros((1, 3)), radius=0.1, bounds=[(-1, 1)] * 3, dim=1024)
    huge = [(np.array([[0.0, 0.0, 0.0]]), np.array([[50.0, 0.0, 0.0]]), np.array([0]))]      # a march that escaped
    outside = [(np.array([[5.0, 0.0, 0.0]]), np.array([[5.5, 0.0, 0.0]]), np.array([0]))]
    assert fv.segments_optical_depth(huge, 1)[0] == 0.0 and fv.segments_optical_depth(outside, 1)[0] == 0.0
    real = [(np.array([[-0.5, 0.0, 0.0]]), np.array([[0.5, 0.0, 0.0]]), np.array([0]))]
    assert fv.segments_optical_depth(real, 1)[0] > 0.0


def test_flaw_shading_limits():
    tr = np.full((4, 3), 0.4)
    assert np.allclose(flaw_shading(tr, np.zeros(4)), tr)
    assert np.allclose(flaw_shading(tr, np.full(4, 1e6), flaw_albedo=(1, 1, 1), E_amb=(np.pi,) * 3), 1.0)
    mid = flaw_shading(tr, np.full(4, 1.0), sigma_t=1.0, flaw_albedo=(1, 1, 1), E_amb=(np.pi,) * 3)
    assert np.allclose(mid, 0.4 * np.exp(-1) + (1 - np.exp(-1)))


def test_bake_records_interior_segments_and_relight_uses_them(mind):
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    cam = mind.camera(eye=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), fov_deg=30.0, aspect=1.0)
    b = mind.bake_glass(sphere(0.6), cam, 24, 24, n_lams=3, max_internal=2)
    assert len(b.segs) >= 1 and all(len(s[2]) == len(s[0]) for s in b.segs)
    start, end, idx = b.segs[0]
    assert np.all(np.linalg.norm(start, axis=1) < 0.6 + 1e-2) and np.all(idx < 24 * 24)   # inside the body, valid pixels
    env = lambda D: np.tile([[1.0, 1.0, 1.0]], (len(D), 1))
    # a milky field that fills the sphere: the glass must get whiter (glow) and less transmissive than pure
    fv = mind.gem_flaw_volume(field=lambda P: np.ones(len(P)), bounds=[(-0.7, 0.7)] * 3, res=8, dim=1024)
    pure = mind.relight_glass(b, env).reshape(-1, 3)[b.glass]
    milky = mind.relight_glass(b, env, flaw_volume=fv, flaw_sigma=5.0, flaw_albedo=(0.5, 0.5, 0.5)).reshape(-1, 3)[b.glass]
    assert not np.allclose(pure, milky)
    # zoning: a chromophore volume tints per channel
    zoned = mind.relight_glass(b, env, absorb_volume=fv, absorb_volume_sigma=(0.0, 5.0, 0.0)).reshape(-1, 3)[b.glass]
    assert np.allclose(zoned[:, 0], pure[:, 0]) and np.all(zoned[:, 1] <= pure[:, 1] + 1e-12) and (zoned[:, 1] < pure[:, 1] - 1e-6).any()


def test_mixed_habits_keep_single_path_identical_and_grow():
    rng = np.random.default_rng(0); Q = rng.uniform(-1, 1, (20000, 3))
    a = cg.cluster(count=7, seed=3)(Q); b = cg.cluster(count=7, seed=3)(Q)
    assert np.array_equal(a, b)
    mx = cg.cluster(count=10, habit=("quartz", "cube", "dodecahedron"), size={"quartz": 0.35, "cube": 0.22, "dodecahedron": 0.25},
                    seed=3, cull=True)(Q)
    assert 0.005 < (mx < 0).mean() < 0.3
    fn = cg.cluster(count=10, habit=lambda p: "cube" if p[1] > 0 else "quartz", size=0.3, seed=3)(Q)
    assert (fn < 0).any()
    with pytest.raises(ValueError):
        cg.cluster(count=4, habit=("quartz", "nosuchhabit"), seed=3)(Q)


def test_opaque_bounce_lifts_cavity_only_where_asked(mind):
    from holographic.mesh_and_geometry.holographic_sdf import sphere, plane
    cam = mind.camera(eye=(0.0, 3.0, 0.01), target=(0.0, 0.0, 0.0), fov_deg=40.0, aspect=1.0)
    b = mind.bake_glass(sphere(0.4), cam, 24, 24, n_lams=3, opaque=plane(-1.0))
    env = lambda D: np.tile([[1.0, 1.0, 1.0]], (len(D), 1))
    base = mind.relight_glass(b, env, opaque_albedo=(0.5, 0.5, 0.5)).reshape(-1, 3)[b.opaque]
    lifted = mind.relight_glass(b, env, opaque_albedo=(0.5, 0.5, 0.5), opaque_bounce=1.0).reshape(-1, 3)[b.opaque]
    assert np.all(lifted > base) and np.allclose(lifted, base + base.mean(axis=0) / 0.5 * 0.5 / np.pi * np.pi, rtol=0.2)
    none = mind.relight_glass(b, env, opaque_albedo=(0.5, 0.5, 0.5), opaque_bounce=1.0,
                              opaque_bounce_where=lambda P: np.zeros(len(P))).reshape(-1, 3)[b.opaque]
    assert np.allclose(none, base)

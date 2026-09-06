"""The caustic pass: exact plane projection, occlusion, and a linear composite.

Contracts only. The geometry here is hand-checkable on purpose -- a camera looking straight down at a
plane -- because an off-by-one in the texel mapping slides the whole pattern and would otherwise only
show up as "the render looks a bit wrong".
"""
import numpy as np
import pytest

import lecore
from holographic.rendering.holographic_causticpass import composite, project_to_plane


def _down_camera(n=16):
    eye = np.array([0.0, 2.0, 0.0])
    g = np.linspace(-0.5, 0.5, n)
    ZZ, XX = np.meshgrid(g, g, indexing="ij")
    dirs = np.stack([XX, -np.ones_like(XX), ZZ], axis=-1)
    return eye, dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


def test_centre_ray_hits_centre_texel():
    """A ray straight down from (0,2,0) lands on the origin, which is the middle texel of a window
    centred there. Off by one and the entire pattern slides across the floor."""
    eye, dirs = _down_camera()
    caustic = np.zeros((32, 32, 3))
    caustic[16, 16] = 1.0
    out = project_to_plane(caustic, eye, dirs, plane_y=0.0, window=1.0)
    assert out.shape == (16, 16, 3)
    assert out[8, 8].max() > 0.0
    assert np.count_nonzero(out.sum(-1)) <= 4


def test_rays_facing_away_contribute_nothing():
    """The plane is behind the camera for these rays. They must be black, not wrapped around -- a sign
    error here would paint a mirror-image caustic on the sky."""
    eye, dirs = _down_camera()
    caustic = np.ones((32, 32, 3))
    assert project_to_plane(caustic, eye, -dirs, plane_y=0.0, window=1.0).sum() == 0.0


def test_occluder_blanks_only_the_covered_floor():
    """Load-bearing: without occlusion the caustic paints over the very object casting it, which reads
    as a rendering bug rather than as missing occlusion.

    The assertion is SCOPED, and the first version of it was not -- it demanded the whole frame go
    black under an occluder of radius 0.5, while the camera's floor hits span [-1, 1]. Most pixels are
    outside the occluder and SHOULD keep their caustic; the test was wrong, not the projection. What
    must hold is: covered floor loses it, uncovered floor keeps it."""
    eye, dirs = _down_camera()
    caustic = np.ones((32, 32, 3))
    plain = project_to_plane(caustic, eye, dirs, plane_y=0.0, window=1.0)
    blocked = project_to_plane(caustic, eye, dirs, plane_y=0.0, window=1.0,
                               occluder_sdf=lambda P: np.linalg.norm(P, axis=-1) - 0.5)
    # The mask must use the SAME eps the projection does (occluder_eps, default 0.03): the code
    # blanks sdf < eps, so the ring between r=0.50 and r=0.53 is deliberately blanked too. A first
    # version compared against a bare r < 0.5 and failed on exactly that ring -- the eps doing its job.
    hit = np.asarray(eye) + dirs * ((0.0 - eye[1]) / dirs[..., 1])[..., None]
    covered = np.linalg.norm(hit, axis=-1) - 0.5 < 0.03
    assert covered.any() and (~covered).any(), "the probe scene must have both regions"
    assert blocked[covered].sum() == 0.0, "covered floor still received the caustic"
    assert np.array_equal(blocked[~covered], plain[~covered]), "uncovered floor was wrongly blanked"


def test_window_frames_the_projection():
    """window= must mean the same thing here as it does in mind.caustics, or the pass samples the wrong
    part of the map. A tighter window over a full-bright map still fills the frame; the check is that
    the two calls disagree, i.e. the parameter is actually read."""
    eye, dirs = _down_camera()
    caustic = np.zeros((64, 64, 3))
    caustic[30:34, 30:34] = 1.0
    wide = project_to_plane(caustic, eye, dirs, plane_y=0.0, window=4.0)
    tight = project_to_plane(caustic, eye, dirs, plane_y=0.0, window=0.5)
    assert not np.array_equal(wide, tight)


def test_dirs_must_be_an_image():
    """A flat (N,3) ray list cannot be reshaped unambiguously into a frame, so it is refused rather
    than guessed at."""
    eye, dirs = _down_camera()
    with pytest.raises(ValueError):
        project_to_plane(np.zeros((8, 8, 3)), eye, dirs.reshape(-1, 3), plane_y=0.0, window=1.0)


def test_composite_is_linear_and_leaves_unlit_pixels_alone():
    """strength must scale only the added pattern. If it touched the beauty the parameter would be an
    exposure control wearing a caustic's name."""
    beauty = np.full((8, 8, 3), 0.1)
    caus = np.zeros((8, 8, 3))
    caus[4, 4] = 2.0
    c1 = composite(beauty, caus, strength=1.0, subtract_baseline=False)
    c2 = composite(beauty, caus, strength=3.0, subtract_baseline=False)
    assert np.allclose(c1[0, 0], 0.1) and np.allclose(c2[0, 0], 0.1)
    lit = caus.sum(-1) > 0
    assert np.allclose((c2 - beauty)[lit], 3.0 * (c1 - beauty)[lit])


def test_composite_normalises_by_percentile_not_max():
    """A caustic's peak is a few colliding splat cells -- an outlier. Normalising by it would make
    `strength` mean something different in every render, so the 99th percentile is used instead. Two
    patterns with the same body and different single-pixel spikes must composite almost identically."""
    beauty = np.zeros((64, 64, 3))
    body = np.zeros((64, 64, 3))
    body[8:56, 8:56] = 1.0                       # 2304 lit pixels -- a realistic caustic footprint
    spiked = body.copy()
    spiked[0, 0] = 500.0
    a = composite(beauty, body, strength=1.0, subtract_baseline=False)[8:56, 8:56]
    b = composite(beauty, spiked, strength=1.0, subtract_baseline=False)[8:56, 8:56]
    assert np.allclose(a, b), "a single hot pixel changed the normalisation of the whole pattern"


def test_the_percentile_needs_enough_lit_pixels_and_says_so():
    """The measured BOUND on the statistic above, pinned rather than left to be discovered in a render.
    With too few lit pixels the 99th percentile IS the outlier: one 500x spike moves the normalised
    body by 99.8% at 65 lit pixels and by exactly 0.0% at 200. Real caustics light thousands."""
    def shift(n):
        side = int(np.ceil(np.sqrt(n * 2)))
        flat = np.zeros((side * side, 3))
        flat[:n] = 1.0
        body = flat.reshape(side, side, 3)
        spiked = body.copy()
        spiked.reshape(-1, 3)[n - 1] = 500.0
        z = np.zeros_like(body)
        mask = np.zeros(body.shape[:2], bool)
        mask.reshape(-1)[: n - 1] = True         # the body, excluding the spike itself
        # subtract_baseline=False on purpose: this test is about the PERCENTILE statistic in
        # isolation, and a flat synthetic body is exactly what baseline subtraction removes.
        a = composite(z, body, 1.0, subtract_baseline=False)
        c = composite(z, spiked, 1.0, subtract_baseline=False)
        return float(np.abs(a[mask] - c[mask]).max() / max(a[mask].max(), 1e-9))

    assert shift(65) > 0.5, "the degenerate case stopped being degenerate -- re-read the bound"
    assert shift(400) == 0.0, "a realistic footprint must be immune to one outlier"


def test_baseline_is_subtracted_so_the_window_edge_does_not_show():
    """THE ARTIFACT THIS PREVENTS, pinned. mind.caustics normalises its splat map to MEAN 1.0, so the
    unfocused receiver reads ~1.0, not 0. Compositing that whole map adds a flat sheet of light the
    beauty render already has -- and only inside the caustic window, which paints a hard-edged bright
    QUADRILATERAL across the floor. It looks like a broken render; it is the window boundary, lit.
    Only the focused EXCESS belongs, which is what a caustic physically is."""
    beauty = np.zeros((32, 32, 3))
    flat = np.ones((32, 32, 3))              # an unfocused receiver: no caustic anywhere
    out = composite(beauty, flat, strength=1.0)
    assert np.allclose(out, beauty), "a flat map added light where there is no caustic"

    spot = np.ones((32, 32, 3))
    spot[14:18, 14:18] = 9.0                 # a focus on top of the same flat baseline
    out2 = composite(beauty, spot, strength=1.0)
    assert out2[16, 16].max() > 0.0, "the focus was subtracted away with the baseline"
    assert np.allclose(out2[0, 0], 0.0), "the unfocused corner still received light"
    # And the opt-out still does the old thing, for a map that is already baseline-free.
    assert composite(beauty, flat, strength=1.0, subtract_baseline=False).max() > 0.0


def test_mismatched_shapes_raise(mind):
    with pytest.raises(ValueError):
        composite(np.zeros((8, 8, 3)), np.zeros((4, 4, 3)))


def test_the_mind_verbs_are_wired(mind):
    """The governing rule: reachable through the faculty surface, not only by import."""
    assert hasattr(mind, "caustic_pass") and hasattr(mind, "composite_caustic")
    out = mind.composite_caustic(np.full((4, 4, 3), 0.1), np.zeros((4, 4, 3)))
    assert np.asarray(out).shape == (4, 4, 3)

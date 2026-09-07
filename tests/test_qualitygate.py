"""Render quality gate (holographic_qualitygate, upstreamed from Poly Studio): absolute defect thresholds. Sweep 163."""
import numpy as np
import pytest

import lecore
from holographic.rendering import holographic_qualitygate as Q


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_terracing_sees_plateaus_where_a_first_difference_cannot():
    good = Q._disc(120, 160, 80, 60, 30, ss=4)
    terraced = good.copy(); q = np.round(good[..., 0] * 12) / 12.0
    inside = (np.arange(120)[:, None] - 60) ** 2 + (np.arange(160)[None, :] - 80) ** 2 < 900
    terraced[..., :] = np.where(inside[..., None], terraced, q[..., None])
    t0, t1 = Q.terracing(good), Q.terracing(terraced)
    assert t0 < Q.DEFAULT_LIMITS["terracing"] < t1 and t1 > 3 * t0
    # a STEEP smooth gradient is not terracing (the first-difference test Poly rejected would flag it)
    steep = np.repeat(np.linspace(0, 1, 120)[:, None], 160, 1); steep = np.stack([steep] * 3, -1)
    assert Q.terracing(steep) == 0.0


@pytest.mark.parametrize("contrast", [True, False])
@pytest.mark.parametrize("size", [(120, 160, 30), (480, 640, 120)])
def test_edge_tones_is_size_and_tone_independent(contrast, size):
    h, w, r = size
    aa, jag = Q._disc(h, w, w // 2, h // 2, r, ss=4, contrast=contrast), Q._disc(h, w, w // 2, h // 2, r, ss=1, contrast=contrast)
    assert Q.edge_tones(aa) > 0.9 and Q.edge_tones(jag) < 0.05


def test_fringe_ratio_catches_added_edge_colour_not_honest_averaging():
    good = Q._disc(120, 160, 80, 60, 30, ss=4, contrast=True)
    single = good + np.random.default_rng(0).normal(0, 0.02, good.shape)
    fringed = good.copy(); e = Q._edge_mask(good.mean(-1)); fringed[..., 0][e] += 0.15; fringed[..., 2][e] -= 0.15
    honest = np.mean([good + np.random.default_rng(k).normal(0, 0.02, good.shape) for k in range(8)], 0)
    assert Q.fringe_ratio(np.clip(fringed, 0, 1), single) > Q.DEFAULT_LIMITS["fringe_ratio"] > Q.fringe_ratio(honest, single)


def test_gate_names_the_failing_metric(mind):
    jag = Q._disc(120, 160, 80, 60, 30, ss=1, contrast=True)
    r = mind.render_quality_gate(jag)
    assert r["failed"] == ["edge_tones"] and not r["ok"]
    assert mind.render_quality_gate(Q._disc(120, 160, 80, 60, 30, ss=4, contrast=True))["ok"]


def test_gate_on_a_real_engine_frame_records_the_rasterisers_jagged_edges(mind):
    """The gate's first catch on the engine itself: render_mesh (the mesh rasteriser) does not antialias silhouettes.
    Recorded as a KNOWN gap, not hidden -- the point of an absolute gate is that this number is visible."""
    box = mind.mesh_box(); cam = mind.fit_camera(box, width=160, height=120)
    img = np.asarray(mind.render_mesh(box, cam, width=160, height=120))
    r = mind.render_quality_gate(img)
    assert r["metrics"]["terracing"] < Q.DEFAULT_LIMITS["terracing"]              # flat faces do not terrace
    assert r["metrics"]["edge_tones"] < 0.3 and "edge_tones" in r["failed"]       # the known gap, pinned

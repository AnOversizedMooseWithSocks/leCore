"""Generalization sweep #2 (GS-A / GS-C): the merged branch's new functionality, re-costumed.

Both faculties here are LIFTS, not new algorithms -- a primitive written for one job, proven on another. The tests
pin the MEASURED contract that justified each lift, and (as loudly) the KEPT NEGATIVE that bounds it: a lift sold
past its regime is how the tree grows a plausible-looking wrong default.
"""
import numpy as np

from holographic.misc.holographic_unified import UnifiedMind


def _edge_guide(h=96, w=96, seed=0):
    """A guide image with one hard vertical edge plus faint texture -- the thing a guided filter must respect."""
    rng = np.random.default_rng(seed)
    g = np.zeros((h, w))
    g[:, w // 2:] = 1.0
    return np.clip(g + 0.03 * rng.standard_normal((h, w)), 0.0, 1.0)


def test_guided_filter_refines_a_map_it_was_never_designed_for():
    """GS-A: guided_filter lives in hazedepth (built for the transmission map) but is a GENERAL edge-aware map
    refiner. Proven on ambient occlusion -- a map with no depth semantics at all -- against the honest baseline
    (a same-support box blur): markedly closer to truth AND the guide's edge survives, which the box destroys."""
    from holographic.rendering.holographic_hazedepth import _box_filter

    m = UnifiedMind(dim=64, seed=0)
    h = w = 96
    rng = np.random.default_rng(1)
    guide = _edge_guide(h, w)

    ao = np.zeros((h, w))
    ao[:, :w // 2] = 0.35
    ao[:, w // 2:] = 0.9
    noisy = np.clip(ao + 0.15 * rng.standard_normal((h, w)), 0.0, 1.0)

    gf = m.guided_filter(guide, noisy, radius=6, eps=1e-3)
    box = _box_filter(noisy, 6)

    rmse = lambda x: float(np.sqrt(np.mean((x - ao) ** 2)))
    step = lambda x: float(np.mean(np.abs(x[:, w // 2] - x[:, w // 2 - 1])))

    assert gf.shape == noisy.shape
    assert rmse(gf) < 0.5 * rmse(box), ("guided must beat a box blur on a guide-aligned map", rmse(gf), rmse(box))
    assert step(gf) > 0.7 * step(ao), ("the guide's edge must survive", step(gf), step(ao))
    assert step(box) < 0.2 * step(ao), ("baseline sanity: the box blur is expected to destroy it", step(box))


def test_guided_filter_kept_negative_guide_must_explain_the_map():
    """GS-A KEPT NEGATIVE: it is NOT a universal denoiser. When the map's structure IGNORES the guide, the guided
    filter is no better than a box blur -- and injects a spurious edge borrowed from the guide. Pinned so nobody
    promotes it to a default smoother."""
    from holographic.rendering.holographic_hazedepth import _box_filter

    m = UnifiedMind(dim=64, seed=0)
    h = w = 96
    rng = np.random.default_rng(2)
    guide = _edge_guide(h, w)

    ramp = np.tile(np.linspace(0, 1, w), (h, 1))          # structure runs ACROSS the guide's edge, not along it
    noisy = np.clip(ramp + 0.15 * rng.standard_normal((h, w)), 0.0, 1.0)

    rmse = lambda x: float(np.sqrt(np.mean((x - ramp) ** 2)))
    assert rmse(m.guided_filter(guide, noisy, radius=6)) >= 0.95 * rmse(_box_filter(noisy, 6)), \
        "KEPT NEGATIVE broken: guided should NOT beat a box blur when the map ignores the guide"


def test_guided_filter_is_discoverable_and_deterministic():
    """A lift nobody can find is not a lift. Plus: no RNG anywhere in the filter."""
    m = UnifiedMind(dim=256, seed=0)
    for phrasing in ("refine a map so its edges follow the image", "smooth but keep edges", "guided filter"):
        top3 = [c.name for c in m.find_capability(phrasing)[:3]]
        assert any(n.startswith("Edge-aware map refiner") for n in top3), (phrasing, top3)

    guide = _edge_guide(48, 48)
    src = np.clip(guide + 0.1 * np.random.default_rng(3).standard_normal((48, 48)), 0.0, 1.0)
    a = m.guided_filter(guide, src, radius=4)
    b = m.guided_filter(guide, src, radius=4)
    assert np.array_equal(a, b), "guided_filter must be deterministic"


def test_workflow_propagate_accepts_an_arbitrary_graph():
    """GS-C: the propagate kernel was always general; only the faculty was pinned to the module graph. `graph=`
    lets any directed weighted graph use it, and defaults to the module bones so existing callers are unchanged."""
    m = UnifiedMind(dim=64, seed=0)
    # a scene-selection adjacency -- nothing to do with modules. Shape is workflowgraph's own: pair-lists.
    g = {"out": {"chair": [("table", 1.0), ("lamp", 0.2)], "table": [("rug", 0.8)]},
         "in": {"table": [("chair", 1.0)], "lamp": [("chair", 0.2)], "rug": [("table", 0.8)]}}

    ranked = dict(m.workflow_propagate({"chair": 1.0}, alpha=0.5, graph=g))
    assert ranked["table"] > 0.0 and ranked["lamp"] > 0.0, ("a seeded node's neighbours must be lifted", ranked)

    # alpha=0 is the kernel's own identity contract -- it must hold on a custom graph too
    zero = dict(m.workflow_propagate({"chair": 1.0}, alpha=0.0, graph=g))
    assert abs(zero["chair"] - 1.0) < 1e-12 and abs(zero.get("table", 0.0)) < 1e-12, zero

    # default path still routes over the module graph (backward compatible)
    default = m.workflow_propagate({"holographic_denoise": 1.0}, alpha=0.5, top=3)
    assert default and default[0][0] == "holographic_denoise"

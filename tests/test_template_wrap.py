"""O1: fixed-topology template wrapping -- the keystone of the creature/humanoid overhaul.

Vertex i must be the SAME anatomical point on every body, or there are no blendshapes, no
shared textures, no cross-species morphing and no correspondence. These tests pin that the
face array survives a wrap onto a different body, and pin the MEASURED quality tradeoff
between template sources so the next session inherits the numbers instead of re-deriving them.
"""
import numpy as np
from holographic.mesh_and_geometry import holographic_templatewrap as TW


def _fields():
    sphere = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    ax = np.array([1.35, 0.75, 1.0])
    ell = lambda P: (np.linalg.norm(np.asarray(P, float) / ax, axis=1) - 1.0) * ax.min()
    return sphere, ell


def test_wrap_preserves_topology_and_lands_on_the_target():
    """THE POINT OF O1. Same faces, new body -- which is what makes vertex i mean something
    across two creatures."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    sphere, ell = _fields()
    t = m.mesh_from_sdf(sphere, ((-1.4,) * 3, (1.4,) * 3), res=32, vectorized=True)
    V0, F0 = np.asarray(t.vertices, float), np.asarray(t.faces, int)
    V1 = TW.wrap_to_field(V0, F0, ell, rounds=6, mind=m)
    q = TW.wrap_quality(V1, F0, ell)
    assert len(V1) == len(V0)              # same vertex count => index correspondence
    assert q["surface_error"] < 0.02       # it actually landed on the new body
    assert q["flipped_faces"] == 0
    assert not np.allclose(V1, V0)         # and it genuinely moved


def test_wrapping_improves_triangle_quality_rather_than_degrading_it():
    """MEASURED, and it corrected my own first reading. I initially reported the wrap as
    causing bunching on a max/min edge ratio of 386 -- but the UNWRAPPED template already
    read 328, and a single degenerate edge dominates that statistic. On the robust p95/p5
    ratio the wrap IMPROVES quality (66.6 -> 38.3) because the Taubin relaxation between
    projection rounds evens the triangles out. A bad number needs a baseline before it
    means anything."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    sphere, ell = _fields()
    t = m.mesh_from_sdf(sphere, ((-1.4,) * 3, (1.4,) * 3), res=32, vectorized=True)
    V0, F0 = np.asarray(t.vertices, float), np.asarray(t.faces, int)
    before = TW.wrap_quality(V0, F0, sphere)
    after = TW.wrap_quality(TW.wrap_to_field(V0, F0, ell, rounds=6, mind=m), F0, ell)
    assert after["edge_ratio"] < before["edge_ratio"]
    assert after["degenerate_edges"] < before["degenerate_edges"]


def test_robust_metric_is_not_fooled_by_one_sliver():
    """The instrument itself, pinned. max/min read 59,000,000 on a mesh whose bulk triangles
    were fine; p95/p5 plus a separate degenerate count keeps 'mostly even with 3 slivers'
    distinguishable from 'uniformly terrible'."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1e-9, 0, 0]], float)
    F = np.array([[0, 1, 2], [0, 3, 2]], int)
    q = TW.wrap_quality(V, F, lambda P: np.zeros(len(P)))
    assert q["degenerate_edges"] >= 1          # the sliver is COUNTED
    assert q["edge_ratio"] < 1e4               # but does not blow up the bulk statistic

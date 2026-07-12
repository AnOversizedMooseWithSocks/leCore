"""holographic_snap.py (mesh_and_geometry) -- the MODELING-GIZMO snap adapter: it answers 'where does this dragged
point / transform delta actually go?' in the shapes the interactive edit spine wants (dict hit records, a corrected
transform delta), DELEGATING all the actual snap math to the canonical snap primitives in
caching_and_storage/holographic_snap ("snapping IS cleanup").

WHY THIS EXISTS ALONGSIDE caching_and_storage/holographic_snap (NOT a duplicate): the canonical module owns the
primitives -- snap_to_grid (now scalar OR per-axis, with a zero-axis passthrough that was generalized INTO it from
here), snap_to_points, snap_to_segment, snap_value, snap_angle, the Snapper class. This module adds only the two
things the gizmo needs that those don't provide: (a) thin adapters that reshape snap_to_points / snap_to_segment
into the {index, position, distance, t} hit records the picking/selection layer uses; and (b) snap_transform_delta,
the genuinely-new piece that corrects a TRANSFORM DELTA so a dragged point lands on a target -- keeping the transform
layer and the snap layer separate. Every line of distance/grid math is the canonical module's; this is the
input/output adapter for the modeler, not a second snap engine. (The grid snap itself is NOT re-exported here -- use
the canonical snap_to_grid, which now covers per-axis.)

Deterministic, NumPy/stdlib only.
"""

import numpy as np

# the canonical snap primitives -- this module is an adapter over them, never a reimplementation.
from holographic.caching_and_storage.holographic_snap import (
    snap_to_grid as _canon_grid,
    snap_to_points as _canon_points,
    snap_to_segment as _canon_segment,
)


def snap_to_vertices(point, vertices, max_dist=None):
    """Snap a point to the NEAREST vertex, returned as the {index, position, distance} hit record the picking layer
    uses, or None if beyond `max_dist`. DELEGATES the nearest-point search to the canonical snap_to_points (which
    returns a (point, index, distance) tuple); this only reshapes it into the dict form and applies the max_dist
    gate. The vertex-snap that makes two verts coincide exactly."""
    V = np.asarray(vertices, float)
    if len(V) == 0:
        return None
    pos, idx, dist = _canon_points(point, V, tol=max_dist)     # canonical cleanup-to-nearest-point
    if idx < 0:                                                # tol gate refused -> nothing close enough
        return None
    return {"index": int(idx), "position": np.asarray(pos, float).tolist(), "distance": float(dist)}


def snap_to_midpoints(point, vertices, edges, max_dist=None):
    """Snap a point to the nearest EDGE MIDPOINT, returned as {edge, position, distance}, or None if beyond max_dist.
    The classic 'midpoint' object-snap. Midpoints are derived from the edge endpoints (no new state); the nearest is
    delegated to the same canonical nearest-point search the vertex snap uses."""
    V = np.asarray(vertices, float)
    mids = np.array([0.5 * (V[a] + V[b]) for (a, b) in edges])
    if len(mids) == 0:
        return None
    pos, idx, dist = _canon_points(np.asarray(point, float), mids, tol=max_dist)
    if idx < 0:
        return None
    return {"edge": int(idx), "position": np.asarray(pos, float).tolist(), "distance": float(dist)}


def snap_to_intersections(point, polylines, max_dist=None, tol=None):
    """Snap a point to the nearest INTERSECTION of a set of 2-D polylines -- the 'intersection' object-snap. Computes
    the crossings with the robust curve-curve intersector (holographic_curveint, exact-sign straddle) and returns the
    nearest as {position, distance}, or None. Polylines are (n,2) arrays; each pair is intersected."""
    from holographic.mesh_and_geometry.holographic_curveint import intersect_polylines
    p = np.asarray(point, float)[:2]
    pts = []
    for i in range(len(polylines)):
        for j in range(i + 1, len(polylines)):
            for h in intersect_polylines(polylines[i], polylines[j], tol=tol):
                pts.append(np.asarray(h["point"], float))
    if not pts:
        return None
    pts = np.array(pts)
    d = np.linalg.norm(pts - p, axis=1)
    k = int(np.argmin(d))
    if max_dist is not None and d[k] > max_dist:
        return None
    return {"position": pts[k].tolist(), "distance": float(d[k])}


def snap_to_edge(point, vertices, edges, max_dist=None):
    """Snap a point to the nearest point ON any edge, returned as {edge, position, distance, t}, or None if beyond
    max_dist. DELEGATES the per-segment perpendicular-foot to the canonical snap_to_segment; this only loops the
    edges to find the nearest and reshapes into the hit record (with the edge index and the t parameter the modeler
    wants). The edge-snap for aligning to a line, not just a vertex."""
    V = np.asarray(vertices, float)
    p = np.asarray(point, float)
    best = None
    for ei, (a, b) in enumerate(edges):
        pa, pb = V[a], V[b]
        foot = np.asarray(_canon_segment(p, pa, pb), float)    # canonical nearest-point-on-segment
        # recover t (where along the edge the foot landed) for the caller.
        ab = pb - pa
        L2 = float(ab @ ab) + 1e-12
        t = float(np.clip((foot - pa) @ ab / L2, 0.0, 1.0))
        dist = float(np.linalg.norm(foot - p))
        if best is None or dist < best["distance"]:
            best = {"edge": int(ei), "position": foot.tolist(), "distance": dist, "t": t}
    if best is None or (max_dist is not None and best["distance"] > max_dist):
        return None
    return best


def snap_transform_delta(delta, target="grid", increment=1.0, moved_point=None, vertices=None, edges=None,
                         origin=(0.0, 0.0, 0.0), max_dist=None):
    """Snap a TRANSFORM DELTA so the moved point lands on a snap target, and return the corrected delta. This is the
    form the gizmo uses: it has a current delta and a reference point being dragged (`moved_point`, the point AFTER
    the raw delta), and wants the delta adjusted so that point snaps. `target` is 'grid' | 'vertex' | 'edge'.
    Returns {delta, snapped_to} where delta is the corrected 3-vector, or the original delta with snapped_to=None on
    a miss. Keeps the transform layer and the snap layer separate -- transform_selection just adds this delta."""
    d = np.asarray(delta, float)
    if moved_point is None:
        # snap the delta itself to the grid (translate in increments).
        return {"delta": np.asarray(_canon_grid(d, increment, (0, 0, 0)), float).tolist(),
                "snapped_to": "grid-delta"}
    mp = np.asarray(moved_point, float)
    if target == "grid":
        snapped = np.asarray(_canon_grid(mp, increment, origin), float)
        return {"delta": (d + (snapped - mp)).tolist(), "snapped_to": "grid"}
    if target == "vertex":
        hit = snap_to_vertices(mp, vertices, max_dist=max_dist)
        if hit is None:
            return {"delta": d.tolist(), "snapped_to": None}
        snapped = np.asarray(hit["position"], float)
        return {"delta": (d + (snapped - mp)).tolist(), "snapped_to": {"vertex": hit["index"]}}
    if target == "edge":
        hit = snap_to_edge(mp, vertices, edges, max_dist=max_dist)
        if hit is None:
            return {"delta": d.tolist(), "snapped_to": None}
        snapped = np.asarray(hit["position"], float)
        return {"delta": (d + (snapped - mp)).tolist(), "snapped_to": {"edge": hit["edge"], "t": hit["t"]}}
    raise ValueError("target must be grid/vertex/edge; got %r" % target)


def _selftest():
    """Contracts:
    1. grid snap rounds to the nearest node; per-axis increment works; a zero increment leaves that axis alone.
    2. vertex snap finds the nearest vertex and respects max_dist.
    3. edge snap lands the perpendicular foot on the closest segment.
    4. snap_transform_delta corrects a delta so the moved point lands on the target.
    """
    # (1) grid: delegated to the canonical snap_to_grid (now per-axis + zero-axis capable).
    assert np.allclose(_canon_grid([0.4, 0.6, -0.3], 1.0), [0.0, 1.0, 0.0])
    assert np.allclose(_canon_grid([0.4, 0.6, 0.9], [1.0, 0.5, 0.0]), [0.0, 0.5, 0.9])  # z spacing 0 -> unchanged

    # (2) vertices
    V = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0]], float)
    hit = snap_to_vertices([4.6, 0.1, 0.0], V)
    assert hit["index"] == 1 and hit["distance"] < 0.5
    assert snap_to_vertices([100, 100, 100], V, max_dist=1.0) is None

    # (3) edge: point near the middle of the edge from v0 to v1 snaps to (2.5,0,0)-ish.
    edges = [[0, 1], [0, 2]]
    e = snap_to_edge([2.5, 0.4, 0.0], V, edges)
    assert e["edge"] == 0 and abs(e["position"][0] - 2.5) < 1e-6 and abs(e["t"] - 0.5) < 1e-6

    # (4) delta correction: dragging a point to (0.4,0,0) with target grid snaps it to (0,0,0), so the corrected
    #     delta cancels the 0.4.
    res = snap_transform_delta([0.4, 0, 0], target="grid", increment=1.0, moved_point=[0.4, 0, 0])
    assert abs(res["delta"][0]) < 1e-9 and res["snapped_to"] == "grid"
    # vertex target: moved point near vertex 1 -> delta lands it exactly on vertex 1.
    res2 = snap_transform_delta([0, 0, 0], target="vertex", moved_point=[4.9, 0.05, 0], vertices=V)
    final = np.array([4.9, 0.05, 0]) + np.asarray(res2["delta"])
    assert np.allclose(final, V[1]) and res2["snapped_to"] == {"vertex": 1}

    # (5) midpoint osnap: nearest edge midpoint. Edge 0 (v0->v1) has midpoint (2.5,0,0).
    mp = snap_to_midpoints([2.4, 0.2, 0.0], V, edges)
    assert mp["edge"] == 0 and abs(mp["position"][0] - 2.5) < 1e-6 and abs(mp["position"][1]) < 1e-6
    assert snap_to_midpoints([100, 100, 100], V, edges, max_dist=1.0) is None

    # (6) intersection osnap: two crossing 2-D polylines meet at the origin; a query near it snaps there.
    A = np.array([[-1.0, 0.0], [1.0, 0.0]]); B = np.array([[0.0, -1.0], [0.0, 1.0]])
    xi = snap_to_intersections([0.1, 0.1], [A, B])
    assert xi is not None and abs(xi["position"][0]) < 1e-9 and abs(xi["position"][1]) < 1e-9
    # parallel lines -> no intersection to snap to
    assert snap_to_intersections([0.0, 0.0], [np.array([[0, 0], [1, 0]]), np.array([[0, 1], [1, 1]])]) is None

    print("holographic_snap selftest OK (grid snap rounds to the nearest node, per-axis increment and a zero-axis "
          "no-op; vertex snap finds the nearest and respects max_dist; edge snap lands the perpendicular foot at "
          "t=0.5; MIDPOINT osnap lands on the edge midpoint; INTERSECTION osnap lands on a polyline crossing (robust "
          "curve intersector) and returns None when parallel; snap_transform_delta corrects a delta so the dragged "
          "point lands on grid/vertex; deterministic)")


if __name__ == "__main__":
    _selftest()

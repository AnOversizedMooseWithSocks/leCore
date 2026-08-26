"""Inverse-rendering IR11: 3D object archive -- recall the whole object from a partial front view, or abstain."""
import numpy as np
from holographic.misc.holographic_objectarchive import ObjectArchive, front_points, object_fingerprint, _chamfer, _sphere, _box, _cylinder, _cone

VIEW = (0, 0, 1)


def _archive():
    a = ObjectArchive(view_dir=VIEW, grid=8, seed=0)
    return a.add(_sphere(500, 1), "sphere").add(_box(500, 2), "box").add(_cylinder(500, 3), "cylinder").build()


def test_recall_whole_from_front_by_shape():
    arch = _archive()
    q = front_points(_sphere(500, 5), VIEW)                   # a NEW sphere instance's front
    res = arch.complete_from_front(q, match_floor=0.85)
    assert not res["abstained"] and res["label"] == "sphere" and res["similarity"] > 0.85


def test_back_recovered_better_than_front_only():
    arch = _archive()
    true = _sphere(500, 5); q = front_points(true, VIEW)
    res = arch.complete_from_front(q, match_floor=0.85)
    back = lambda p: p[p[:, 2] < -0.1]
    d_recall = _chamfer(back(res["whole"]), back(true))
    d_front = _chamfer(q, back(true))
    assert d_recall < 0.3 * d_front                           # retrieval recovers the back; front-only can't


def test_abstains_on_unseen_shape():
    arch = _archive()
    res = arch.complete_from_front(front_points(_cone(500, 9), VIEW), match_floor=0.85)
    assert res["abstained"] and res["whole"] is None          # not in the library -> honest fallback


def test_fingerprint_unit_norm():
    fp = object_fingerprint(front_points(_sphere(500, 1), VIEW), VIEW, 8)
    assert abs(np.linalg.norm(fp) - 1.0) < 1e-9


def test_front_points_are_camera_facing():
    pts = _sphere(500, 1)
    f = front_points(pts, VIEW, keep=0.5)
    assert f[:, 2].mean() > pts[:, 2].mean()                 # the front half is nearer the +z camera
    assert len(f) < len(pts)


def test_box_and_cylinder_recall_their_own_shape():
    arch = _archive()
    assert arch.complete_from_front(front_points(_box(500, 22), VIEW), match_floor=0.8)["label"] == "box"
    assert arch.complete_from_front(front_points(_cylinder(500, 33), VIEW), match_floor=0.8)["label"] == "cylinder"


def test_deterministic():
    arch = _archive()
    q = front_points(_sphere(500, 5), VIEW)
    assert arch.complete_from_front(q)["index"] == arch.complete_from_front(q)["index"]

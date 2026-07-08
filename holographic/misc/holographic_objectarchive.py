"""holographic_objectarchive.py -- 3D OBJECT ARCHIVE: recall the whole from a partial view (inverse-rendering IR11).

Store a library of COMPLETE 3D objects; given only a partial FRONT view (what a photo/photo3d actually sees), recall
the nearest stored complete object by a view fingerprint and return the WHOLE thing -- including the unobserved back
-- or ABSTAIN when nothing in the library matches. Retrieval, not hallucination.

The holographic reading: "recover the clean whole from a corrupted part" is the engine's OLDEST move. It is cleanup
(snap a noisy vector to a stored one), consolidation (project onto the subspace real states occupy), the resonator
(factor a bundle into its parts), and analog recall (find the stored thing whose visible part matches, return the
whole). The 2-D plate does it for images; this does the SAME move for geometry, because a scene is a bundle of
splats exactly as a memory is a bundle of role-bound vectors. Nothing new is invented; a shipped primitive (analog
recall over a HoloForest) is pointed at a new field -- which is the whole reason it is cheap.

The unobserved back is "damage"; completing it is recovery from a partial measurement (Ozcan's seat). Here that
recovery is retrieval: recall the stored complete object whose front matches the query's front.

KEPT NEGATIVES (loud):
  * COVERAGE-LIMITED: it completes only objects it has STORED. Outside the library it ABSTAINS -- the honest output
    on an unseen shape is "front only", never an invented back. The win scales with library coverage. This is the
    point: retrieval, not hallucination.
  * REGISTRATION: matching a front to a stored object is by a coarse view fingerprint; a wrong match must surface as
    a LOW similarity and abstain, never a confident-wrong completion. Fine pose/scale alignment is left to the IR4
    loop; here the fingerprint gives the coarse match + the abstain gate.
  * Capacity is a DIAL not a wall (the plate/forest holds a bounded number before crosstalk; escapes are disjoint
    slots, tiling, importance-ordering -- bucket-size and spill, don't cap). Inherits the splat archive's lossy/
    isotropic trade.
NumPy + stdlib only; deterministic.
"""
import numpy as np

from holographic.misc.holographic_tree import HoloForest


def _chamfer(A, B):
    """Symmetric Chamfer distance between two point sets: the mean nearest-neighbour distance, both directions.
    O(N*M) -- fine for the modest point counts an object archive holds."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return 0.5 * (np.sqrt(d2.min(axis=1)).mean() + np.sqrt(d2.min(axis=0)).mean())


def _view_basis(view_dir):
    """A right/up/forward orthonormal basis with forward = view_dir (the axis the camera looks along)."""
    v = np.asarray(view_dir, float)
    v = v / (np.linalg.norm(v) + 1e-12)
    up = np.array([0.0, 1.0, 0.0]) if abs(v[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    r = np.cross(up, v)
    r = r / (np.linalg.norm(r) + 1e-12)
    u = np.cross(v, r)
    return r, u, v


def front_points(points, view_dir=(0, 0, 1), keep=0.5):
    """The camera-facing FRONT of an object: the fraction `keep` of points nearest to the camera along the view
    direction (largest projection onto view_dir). This is what a single photo actually observes."""
    p = np.asarray(points, float)
    _, _, v = _view_basis(view_dir)
    proj = p @ v
    thresh = np.quantile(proj, 1.0 - keep)
    return p[proj >= thresh]


def object_fingerprint(points, view_dir=(0, 0, 1), grid=8):
    """A coarse, unit-normalized VIEW FINGERPRINT: project the points onto the plane perpendicular to the view
    direction, bin into a grid x grid silhouette-occupancy map (+ a coarse depth map), flatten, normalize. Similar
    front silhouettes -> similar fingerprints, so a HoloForest can recall the nearest stored object sub-linearly."""
    p = np.asarray(points, float)
    r, u, v = _view_basis(view_dir)
    x, y, z = p @ r, p @ u, p @ v

    def n01(a):
        return (a - a.min()) / (np.ptp(a) + 1e-9)

    xi = np.clip((n01(x) * (grid - 1)).astype(int), 0, grid - 1)
    yi = np.clip((n01(y) * (grid - 1)).astype(int), 0, grid - 1)
    occ = np.zeros((grid, grid))
    dep = np.zeros((grid, grid))
    cnt = np.zeros((grid, grid))
    zn = n01(z)
    for k in range(len(p)):                                   # scatter into the silhouette + depth bins
        occ[yi[k], xi[k]] += 1.0
        dep[yi[k], xi[k]] += zn[k]
        cnt[yi[k], xi[k]] += 1.0
    dep = dep / np.clip(cnt, 1.0, None)
    fp = np.concatenate([occ.ravel() / (occ.sum() + 1e-9), dep.ravel()])
    return fp / (np.linalg.norm(fp) + 1e-12)


class ObjectArchive:
    """A content-addressable library of COMPLETE 3D objects, recalled by a front-view fingerprint. complete_from_
    front(front) returns the nearest stored WHOLE object (back included) or abstains when nothing matches."""

    def __init__(self, view_dir=(0, 0, 1), grid=8, n_trees=4, seed=0):
        self.view_dir = view_dir
        self.grid = grid
        self.n_trees = n_trees
        self.seed = seed
        self.objects = []          # list of (points (N,3), label)
        self.fingerprints = []     # each object's FRONT fingerprint
        self.forest = None

    def add(self, points, label=None):
        """Store a complete object; index it by the fingerprint of its own front view."""
        pts = np.asarray(points, float)
        self.objects.append((pts, label))
        self.fingerprints.append(object_fingerprint(front_points(pts, self.view_dir), self.view_dir, self.grid))
        return self

    def build(self):
        items = np.array(self.fingerprints, float)
        self.forest = HoloForest(items.shape[1], n_trees=self.n_trees, seed=self.seed).build(items)
        return self

    def complete_from_front(self, front, match_floor=0.6):
        """Recall the stored complete object whose front matches `front`'s, and return the WHOLE object (including
        the unobserved back). If the best fingerprint similarity is below `match_floor`, ABSTAIN (return whole=None)
        -- the caller then keeps photo3d's honest front-only reconstruction. Returns a dict."""
        fp = object_fingerprint(front, self.view_dir, self.grid)
        idx, agree = self.forest.recall(fp, with_agreement=True)
        sim = float(fp @ self.fingerprints[idx])              # cosine (both unit-normalized)
        if sim < match_floor:
            return {"whole": None, "index": int(idx), "similarity": sim, "agreement": float(agree),
                    "abstained": True}
        pts, label = self.objects[idx]
        return {"whole": pts, "index": int(idx), "similarity": sim, "agreement": float(agree),
                "abstained": False, "label": label}


# ---- shape samplers for the selftest (deterministic point clouds, front and back) ----
def _sphere(n=220, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _box(n=220, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-1, 1, (n, 3))
    face = rng.integers(0, 3, n)                              # snap one coordinate to +/-1 (a cube face)
    sign = rng.choice([-1.0, 1.0], n)
    p[np.arange(n), face] = sign
    return p


def _cylinder(n=220, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-1, 1, n)
    return np.stack([np.cos(th), z, np.sin(th)], axis=1)      # axis along y


def _cone(n=220, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    h = rng.uniform(0, 1, n)
    rad = 1.0 - h
    return np.stack([rad * np.cos(th), 2 * h - 1, rad * np.sin(th)], axis=1)


def _selftest():
    """Build a library (sphere, box, cylinder); complete a sphere from a DIFFERENT-instance sphere's FRONT half ->
    recall the whole sphere by shape (not memorized points), recovering the back FAR closer to ground truth than the
    front-only baseline (which has no back); abstain on a cone that is not in the library. Deterministic."""
    view = (0, 0, 1)
    arch = ObjectArchive(view_dir=view, grid=8, seed=0)
    arch.add(_sphere(500, 1), "sphere").add(_box(500, 2), "box").add(_cylinder(500, 3), "cylinder").build()

    # a "photo" of a DIFFERENT sphere instance (seed=5): its front half plus a little noise -> shape-level recall
    rng = np.random.default_rng(7)
    sphere_true = _sphere(500, 5)
    f = front_points(sphere_true, view)
    front = f + 0.01 * rng.standard_normal(f.shape)

    res = arch.complete_from_front(front, match_floor=0.85)
    assert not res["abstained"] and res["label"] == "sphere"   # recalled the right SHAPE from a new instance

    # BACK recovery: the recalled whole's back vs the true back, versus front-only (which has NO back)
    def back(pts):
        return pts[pts[:, 2] < -0.1]                           # the unobserved hemisphere (z<0)
    true_back = back(sphere_true)
    recalled_back = back(res["whole"])
    d_recall = _chamfer(recalled_back, true_back)              # recalled sphere's back ~ the true sphere's back
    d_frontonly = _chamfer(front, true_back)                  # front-only: nearest observed points are far from the back
    assert d_recall < 0.3 * d_frontonly                       # retrieval recovers the back; front-only cannot

    # ABSTAIN on an object not in the library (a cone)
    cone_front = front_points(_cone(500, 9), view)
    odd = arch.complete_from_front(cone_front, match_floor=0.85)
    assert odd["abstained"] and odd["whole"] is None          # not in the library -> honest front-only fallback

    # deterministic
    assert arch.complete_from_front(front)["index"] == arch.complete_from_front(front)["index"]

    print("holographic_objectarchive selftest OK: from a NEW sphere instance's FRONT half the archive recalled the "
          "whole sphere by shape (similarity %.2f, label '%s'), recovering the unobserved back to Chamfer %.3f vs "
          "the front-only baseline's %.3f (%.0fx closer); abstained on a cone not in the library (similarity %.2f "
          "< floor). Retrieval, not hallucination; deterministic"
          % (res["similarity"], res["label"], d_recall, d_frontonly, d_frontonly / max(d_recall, 1e-9),
             odd["similarity"]))


if __name__ == "__main__":
    _selftest()

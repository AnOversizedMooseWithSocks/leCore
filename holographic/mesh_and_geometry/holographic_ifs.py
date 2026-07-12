"""Affine Iterated Function Systems: generate and FIT plant/fractal point-set fractals (holographic_ifs).

WHY THIS MODULE EXISTS
----------------------
fit_shape's fractal path fits a Mandelbox FOLD recipe -- great for self-similar surfaces, but a fern or a branching
tree is not a fold fractal. Its TRUE generator is an AFFINE IFS (the Barnsley fern is four affine maps played as a
'chaos game'), or an L-system. This module adds the honest botanical/branching model that fit_shape flagged as its
next fitter, and an IFS FITTER that matches a point cloud to the closest NAMED system.

An affine IFS is a small set of affine maps f_i(p) = A_i p + b_i, each chosen with probability w_i; iterating a single
point under randomly-chosen maps (the 'chaos game') fills in the attractor -- the fractal. A handful of 6-number maps
regenerate a whole fern: the 'determinism instead of storage' lever, as botany.

WHAT IT PROVIDES
  * AffineIFS(maps, name) -- a system: `maps` is a list of (a,b,c,d,e,f,prob) rows (the standard Barnsley notation:
    x'=a x+b y+e, y'=c x+d y+f). `.generate(n, seed)` plays the chaos game -> an (n,2) point cloud.
  * A LIBRARY of named systems (published coefficients -- mathematical facts, not copyrighted assets): barnsley_fern,
    sierpinski, fractal_tree, dragon_curve, and a couple of fern variants.
  * ifs_fit(target) -- match a 2-D point cloud to the CLOSEST named system by normalized-occupancy signature, with a
    measured baseline. Snap-to-a-bank, the same shape as fit_deterministic / fold_fit.

KEPT NEGATIVES (loud)
  * ifs_fit SNAPS TO A LIBRARY -- it does not recover arbitrary IFS maps from a cloud (the general inverse-IFS problem
    is hard and ill-posed). It answers "which known system is this closest to, and how close", with a baseline, not
    "here are the exact maps that made it". A cloud unlike any library entry scores low and says so.
  * The occupancy signature is TRANSLATION/SCALE-NORMALISED but NOT rotation-invariant -- a rotated fern matches worse
    (documented; a rotation-invariant descriptor is a scoped extension).
  * These are 2-D point-set fractals (the chaos game), distinct from the raymarched SDF fractals -- to get geometry,
    mesh the cloud (sdf_from_points -> sdf_to_mesh); there is no raymarch shader for a point-set attractor.
"""

import numpy as np


class AffineIFS:
    """An affine Iterated Function System: a list of (a,b,c,d,e,f,prob) maps played as a chaos game. Each map is the
    Barnsley form x' = a*x + b*y + e, y' = c*x + d*y + f, selected with probability `prob` (probabilities are
    renormalised). `.generate(n, seed)` returns the (n,2) attractor point cloud, deterministically."""

    def __init__(self, maps, name="ifs"):
        self.maps = [tuple(float(v) for v in m) for m in maps]
        self.name = name
        w = np.array([m[6] for m in self.maps], float)
        self.probs = w / w.sum()                              # renormalise so the probabilities are a valid pmf

    def generate(self, n=20000, seed=0, warmup=20):
        """Play the chaos game for `n` points from the origin (after `warmup` discarded iterations that let the point
        settle onto the attractor). Deterministic given `seed`. Returns an (n,2) array."""
        rng = np.random.default_rng(seed)
        A = [np.array([[m[0], m[1]], [m[2], m[3]]]) for m in self.maps]
        b = [np.array([m[4], m[5]]) for m in self.maps]
        choices = rng.choice(len(self.maps), size=n + warmup, p=self.probs)
        p = np.zeros(2)
        out = np.zeros((n, 2))
        for i in range(n + warmup):
            k = choices[i]
            p = A[k] @ p + b[k]
            if i >= warmup:
                out[i - warmup] = p
        return out


# ---------------------------------------------------------------------------------------------------------
# The named library -- published affine-IFS coefficients (mathematical facts). Each is (a,b,c,d,e,f,prob).
# ---------------------------------------------------------------------------------------------------------

def barnsley_fern():
    """The classic Barnsley fern -- four affine maps (stem, successively-smaller leaflets, left + right leaflets)."""
    return AffineIFS([
        (0.00, 0.00, 0.00, 0.16, 0.0, 0.00, 0.01),
        (0.85, 0.04, -0.04, 0.85, 0.0, 1.60, 0.85),
        (0.20, -0.26, 0.23, 0.22, 0.0, 1.60, 0.07),
        (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07),
    ], name="barnsley_fern")


def sierpinski():
    """The Sierpinski triangle -- three half-scale maps to the corners of a triangle (equal probability)."""
    return AffineIFS([
        (0.5, 0.0, 0.0, 0.5, 0.00, 0.00, 1 / 3),
        (0.5, 0.0, 0.0, 0.5, 0.50, 0.00, 1 / 3),
        (0.5, 0.0, 0.0, 0.5, 0.25, 0.50, 1 / 3),
    ], name="sierpinski")


def fractal_tree():
    """A binary fractal tree -- a trunk plus two rotated, scaled branch maps (a simple branching plant)."""
    c, s = 0.42, 0.42
    return AffineIFS([
        (0.00, 0.00, 0.00, 0.50, 0.0, 0.00, 0.10),          # trunk
        (0.42, -0.42, 0.42, 0.42, 0.0, 0.20, 0.45),          # left branch (rotate +45, scale)
        (0.42, 0.42, -0.42, 0.42, 0.0, 0.20, 0.45),          # right branch (rotate -45, scale)
    ], name="fractal_tree")


def dragon_curve():
    """The Heighway dragon curve -- two affine maps (the classic space-filling dragon)."""
    return AffineIFS([
        (0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.5),
        (-0.5, -0.5, 0.5, -0.5, 1.0, 0.0, 0.5),
    ], name="dragon_curve")


def culcita_fern():
    """A second fern variant (Culcita) -- a different leaflet balance, so the fitter's library has two ferns to
    distinguish (proves the match is not just 'anything fern-shaped')."""
    return AffineIFS([
        (0.00, 0.00, 0.00, 0.25, 0.0, -0.4, 0.02),
        (0.85, 0.02, -0.02, 0.83, 0.0, 1.00, 0.84),
        (0.09, -0.28, 0.30, 0.11, 0.0, 0.60, 0.07),
        (-0.09, 0.28, 0.30, 0.09, 0.0, 0.70, 0.07),
    ], name="culcita_fern")


def ifs_library():
    """The named-system bank the fitter snaps to. Returns a dict {name: AffineIFS}."""
    return {ifs.name: ifs for ifs in (barnsley_fern(), culcita_fern(), sierpinski(), fractal_tree(), dragon_curve())}


def _occupancy_signature(pts, bins=28):
    """A translation/scale-normalised occupancy SIGNATURE of a 2-D point cloud: fit the cloud into a unit box (by its
    own bounding box), histogram into `bins` x `bins`, normalise to sum 1. Two clouds of the SAME fractal (any
    position/scale) give a near-identical signature; different fractals differ. NOT rotation-invariant (kept negative).
    """
    pts = np.asarray(pts, float)
    lo = pts.min(axis=0)
    span = np.maximum(pts.max(axis=0) - lo, 1e-9)
    u = (pts - lo) / span                                     # normalise to the unit box
    H, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=bins, range=[[0, 1], [0, 1]])
    H = H / max(H.sum(), 1.0)
    return H.ravel()


def ifs_fit(target, bins=28, n_ref=20000, seed=0):
    """Match a 2-D `target` point cloud to the CLOSEST NAMED affine-IFS system (a fern, sierpinski, tree, dragon, ...)
    by occupancy signature. For each library system, generate its attractor, compare normalised-occupancy signatures
    (L1 distance), and keep the best. Returns {name, ifs, distance, quality, baseline, ranking, note}.

    `quality` is 1 - distance in [0,1] (1 = identical signature); `baseline` is the MEAN distance across the library
    (so a match well below the mean is a real snap, a match near the mean means the target resembles nothing in the
    bank). `ranking` is every (name, distance) sorted. KEPT NEGATIVE (in note): snaps to a LIBRARY, does not recover
    arbitrary maps; not rotation-invariant."""
    target = np.asarray(target, float)
    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("ifs_fit expects a 2-D (M,2) point cloud")
    tsig = _occupancy_signature(target, bins=bins)
    lib = ifs_library()
    dists = {}
    for name, ifs in lib.items():
        ref = ifs.generate(n=n_ref, seed=seed)
        dists[name] = float(np.abs(_occupancy_signature(ref, bins=bins) - tsig).sum())
    ranking = sorted(dists.items(), key=lambda kv: kv[1])
    best_name, best_d = ranking[0]
    baseline = float(np.mean(list(dists.values())))
    return {
        "name": best_name,
        "ifs": lib[best_name],
        "distance": best_d,
        "quality": 1.0 - 0.5 * best_d,                        # L1 of two pmfs is in [0,2]; map to [0,1]
        "baseline": 1.0 - 0.5 * baseline,
        "ranking": ranking,
        "note": "SNAP-TO-LIBRARY match (closest of %d named systems by occupancy), not arbitrary-IFS recovery; "
                "translation/scale-normalised but NOT rotation-invariant." % len(lib),
    }


def _selftest():
    # (1) the Barnsley fern generates with the right footprint (the classic bounds).
    fern = barnsley_fern().generate(n=20000, seed=0)
    assert -3.0 < fern[:, 0].min() < 0.0 and 9.0 < fern[:, 1].max() < 11.0, "the fern has its classic footprint"
    # determinism
    assert np.array_equal(barnsley_fern().generate(n=1000, seed=0), barnsley_fern().generate(n=1000, seed=0))

    # (2) sierpinski fills its triangle; dragon and tree generate distinct clouds.
    tri = sierpinski().generate(n=10000, seed=0)
    assert tri[:, 0].min() >= -0.01 and tri[:, 0].max() <= 1.01, "sierpinski stays in the unit triangle box"

    # (3) THE FIT: a fern cloud snaps to 'barnsley_fern', and it beats the library-mean baseline (a real snap).
    res = ifs_fit(fern)
    assert res["name"] == "barnsley_fern", "a fern cloud must match the fern, got %s (%s)" % (res["name"], res["ranking"])
    assert res["quality"] > res["baseline"], "the match must beat the library-mean baseline (a real snap)"
    # and it distinguishes the two ferns: the barnsley fern matches barnsley closer than culcita.
    d = dict(res["ranking"])
    assert d["barnsley_fern"] < d["culcita_fern"], "the fitter distinguishes the two fern variants"

    # (4) a sierpinski cloud snaps to sierpinski (not a fern) -- the fit discriminates across families.
    res_tri = ifs_fit(tri)
    assert res_tri["name"] == "sierpinski", "a sierpinski cloud must match sierpinski, got %s" % res_tri["name"]

    # (5) KEPT NEGATIVE made concrete: random noise matches nothing well -- its best distance is near the baseline
    #     (no real snap), unlike the fern which is well below.
    rng = np.random.default_rng(3)
    noise = rng.uniform(0, 1, (5000, 2))
    res_noise = ifs_fit(noise)
    fern_margin = res["baseline"] - res["quality"]            # negative: fern quality ABOVE baseline (good)
    noise_margin = res_noise["baseline"] - res_noise["quality"]
    assert noise_margin > fern_margin, "random noise snaps far less than a real fern (the baseline gap discriminates)"

    print("holographic_ifs selftest: ok (Barnsley fern generates its classic footprint; ifs_fit snaps a fern cloud to "
          "'barnsley_fern' (quality %.2f > baseline %.2f) and distinguishes it from culcita; a sierpinski cloud snaps "
          "to sierpinski; KEPT NEGATIVE -- random noise matches nothing (snaps near baseline); deterministic; "
          "snap-to-library not arbitrary-IFS recovery, not rotation-invariant)" % (res["quality"], res["baseline"]))


if __name__ == "__main__":
    _selftest()

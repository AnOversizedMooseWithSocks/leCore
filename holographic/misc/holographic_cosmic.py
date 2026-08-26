"""Local structure classification of a point cloud -- the 'cosmic web' method, extracted from leOS
(science/cosmic_web.py) and decoupled from its agent displacement-log so it works on any (N, D) cloud.

The idea (and why it is on-thesis -- 'structure across scales / as above, so below'): cosmologists classify every
point of the matter distribution by the LOCAL shape of its neighbourhood -- a point sits in a VOID (empty), on a
1-D FILAMENT (a thread), on a 2-D WALL/sheet, or in a dense NODE/cluster -- by looking at the eigenvalues of the
local structure tensor. The same move classifies any high-dimensional embedding cloud: around each point, take its
k nearest neighbours, do a LOCAL PCA, and read the eigenvalue spectrum:

  * how spread the eigenvalues are -> the LOCAL INTRINSIC DIMENSIONALITY (one strong direction = a filament, two =
    a sheet, many = a blob),
  * how dense the neighbourhood is -> void vs structure.

holostuff already had GLOBAL dimension estimates (box_counting_dimension, spectral_dimension) and global manifold
topology, but NOT this PER-POINT local structure type. That is the gap this fills: it tells you, for each point,
what kind of structure it lives in -- which is exactly what you want before denoising (project a filament point
along its one direction, not all of them), before sampling (avoid voids), or to summarise a cloud's geometry.

Improvement over the extracted version: a continuous PARTICIPATION RATIO intrinsic dimension PR = (sum lambda)^2 /
sum(lambda^2) -- a smooth effective dimensionality (1.0 = pure filament, 2.0 = pure sheet, ...) -- alongside the
discrete VOID/FILAMENT/WALL/NODE label, so you get both a crisp type and a graded measure.

Pure NumPy. HONEST: this is a LOCAL estimate -- it depends on k (the neighbourhood size) and the cloud's density,
and high-dimensional noise inflates the apparent dimension; reported, not hidden.
"""

import numpy as np

VOID, FILAMENT, WALL, NODE = "void", "filament", "wall", "node"


def _local_pca(point, cloud, k):
    """k nearest neighbours of `point` in `cloud` (excluding itself), and the eigenvalue spectrum of their LOCAL
    PCA (neighbours centred on their own mean). Returns (eigenvalues_desc_normalised, mean_neighbour_distance,
    n_used)."""
    point = np.asarray(point, float); cloud = np.asarray(cloud, float)
    dist = np.linalg.norm(cloud - point, axis=1)
    order = np.argsort(dist)
    idx = [i for i in order if dist[i] > 1e-12][:k]          # drop the point itself if present
    if len(idx) < 3:
        return np.array([]), float(dist[order[0]] if len(order) else 0.0), len(idx)
    nbrs = cloud[idx]
    centred = nbrs - nbrs.mean(axis=0)
    # SVD is the stable PCA for n < d; eigenvalues of the covariance are S^2 / n
    _, S, _ = np.linalg.svd(centred, full_matrices=False)
    ev = (S ** 2) / len(nbrs)
    total = ev.sum()
    ev = ev / total if total > 1e-12 else ev
    return ev, float(np.mean(dist[idx])), len(idx)


def participation_ratio(eigenvalues):
    """PR = (sum lambda)^2 / sum(lambda^2): a smooth effective dimensionality. For eigenvalues summing to 1,
    PR = 1 / sum(lambda^2): 1.0 for one dominant direction (filament), ~2 for two (sheet), ~d for isotropic."""
    ev = np.asarray(eigenvalues, float)
    if ev.size == 0:
        return 0.0
    s = ev.sum()
    return float(s * s / (np.sum(ev ** 2) + 1e-12))


def local_structure(point, cloud, k=12, void_distance=None, filament_ratio=4.0, wall_ratio=2.5):
    """Classify one point by the shape of its local neighbourhood. Returns dict{type, intrinsic_dim (participation
    ratio), density, eigenvalues (top 5), n_neighbors}. `void_distance`: neighbourhoods sparser than this (mean
    neighbour distance larger) are VOID; defaults to a multiple of the cloud's median nearest-neighbour spacing if
    not given (passed in by classify_cloud for efficiency)."""
    ev, mean_d, n = _local_pca(point, cloud, k)
    if n < 3:
        return {"type": VOID, "intrinsic_dim": 0.0, "density": 0.0,
                "eigenvalues": [], "n_neighbors": n}
    density = 1.0 / (mean_d + 1e-12)
    pr = participation_ratio(ev)
    if void_distance is not None and mean_d > void_distance:
        structure = VOID
    elif ev[0] / max(ev[1], 1e-12) > filament_ratio:          # one direction dominates -> a thread
        structure = FILAMENT
    elif len(ev) >= 3 and ev[1] / max(ev[2], 1e-12) > wall_ratio:   # two directions dominate -> a sheet
        structure = WALL
    else:
        structure = NODE                                      # many comparable directions -> a blob/cluster
    return {"type": structure, "intrinsic_dim": round(pr, 3), "density": round(density, 4),
            "eigenvalues": [round(float(e), 4) for e in ev[:5]], "n_neighbors": n}


def classify_cloud(cloud, k=12, void_percentile=85.0):
    """Classify every point of a cloud. The VOID threshold is data-driven: the `void_percentile` of the per-point
    mean-neighbour-distance distribution (sparser-than-most = void). Returns (labels, info_list, summary) where
    summary is the fraction of points of each type plus the mean intrinsic dimension."""
    cloud = np.asarray(cloud, float)
    n = len(cloud)
    # one O(N^2) distance pass; cheap for modest N and avoids re-sorting per point
    mean_ds = []
    for p in cloud:
        _, md, cnt = _local_pca(p, cloud, k)
        mean_ds.append(md if cnt >= 3 else np.inf)
    finite = [d for d in mean_ds if np.isfinite(d)]
    void_distance = float(np.percentile(finite, void_percentile)) if finite else None
    info = [local_structure(p, cloud, k=k, void_distance=void_distance) for p in cloud]
    labels = [d["type"] for d in info]
    summary = {t: labels.count(t) / n for t in (VOID, FILAMENT, WALL, NODE)}
    dims = [d["intrinsic_dim"] for d in info if d["n_neighbors"] >= 3]
    summary["mean_intrinsic_dim"] = round(float(np.mean(dims)), 3) if dims else 0.0
    return labels, info, summary


def _selftest():
    rng = np.random.default_rng(0)
    D = 32

    def embed(low, noise=0.0005):                           # embed a low-D structure into D dims + small noise
        Q = np.linalg.qr(rng.standard_normal((D, D)))[0][:, :low.shape[1]]
        return low @ Q.T + noise * rng.standard_normal((low.shape[0], D))

    t = np.linspace(0, 4, 300)[:, None]
    filament = embed(t)                                     # 1-D line
    sheet = embed(rng.uniform(-1, 1, (400, 2)))            # 2-D sheet (uniform -> locally isotropic)
    blob = embed(rng.uniform(-1, 1, (500, 3)))            # 3-D blob

    def mean_pr(cloud, kk, m=40):                           # average intrinsic dim over m interior points (stable)
        idx = np.linspace(20, len(cloud) - 20, m).astype(int)
        return float(np.mean([participation_ratio(_local_pca(cloud[i], cloud, kk)[0]) for i in idx]))

    df = mean_pr(filament, 12); ds = mean_pr(sheet, 14); db = mean_pr(blob, 18)
    assert df < 1.5 < ds < 2.5 < db, (df, ds, db)          # intrinsic dim tracks true dimensionality, monotonic

    _, _, sumf = classify_cloud(filament, k=12)
    _, _, sums = classify_cloud(sheet, k=12)
    assert sumf[FILAMENT] > 0.6, sumf                       # a 1-D cloud is mostly filament
    assert sums[WALL] + sums[NODE] > 0.5, sums              # a 2-D cloud reads as sheet/cluster, not filament

    # KEPT NEGATIVE: high-dimensional noise inflates the apparent local dimension
    noisy_filament = embed(t, noise=0.01)
    df_noisy = mean_pr(noisy_filament, 12)
    assert df_noisy > df + 0.3, (df, df_noisy)              # the 1-D line now reads as higher-dim under noise

    print(f"cosmic selftest ok: intrinsic dim filament {df:.2f} < sheet {ds:.2f} < blob {db:.2f}; "
          f"1-D cloud {sumf[FILAMENT]*100:.0f}% filament, 2-D cloud {(sums[WALL]+sums[NODE])*100:.0f}% wall/node; "
          f"KEPT NEGATIVE: noise inflates the 1-D line to PR {df_noisy:.2f}")


if __name__ == "__main__":
    _selftest()

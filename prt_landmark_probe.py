"""PROBE (measure, do not ship yet): can we bake the diffuse PRT transfer on a SUBSET of representative points
(landmark 'bricks'), then PROJECT it out to the rest of the scene, cheaper than baking every point -- and where
the projection is untrustworthy (shadow edges), re-bake only those 'blind spots'? This is Moose's compress->pass->
project-out->connect idea, applied to the ONE part of light transport that is smooth and (mostly) linear: the
shadowed-sky transfer field. We keep the honest negative if it doesn't beat the full bake on a fair scene.

Method, three ways, on the same visible G-buffer:
  FULL      : precompute_transfer for every visible point            (ground truth + the baseline we must beat)
  LANDMARK  : bake only m farthest-point landmarks, then project the transfer to all points by an affinity-weighted
              blend of nearby landmarks (irradiance-cache / Nystrom-style interpolation in position+normal space)
  +REFINE   : detect the points where the contributing landmarks DISAGREE (a proxy for 'near a shadow edge',
              computed WITHOUT the ground truth) and re-bake only those exactly -- the blind-spot pass

We report, per m: the bake+project time vs the full time, and the shading error vs full. The claim under test is
that on a scene with real smooth regions the landmark path is cheaper at acceptable error, and that the disagreement
signal actually finds the high-error points so the refine pass spends its budget where it matters.
"""
import time
import numpy as np

from holographic_sdf import box, sphere
from holographic_render import Camera
from holographic_raymarch import sphere_trace, sdf_normal
from holographic_prt import precompute_transfer, project_env_to_sh, shade_prt
from holographic_lights import DomeLight
from holographic_nystrom import farthest_point_landmarks, gaussian_affinity

ORDER = 3          # SH bands for the diffuse transfer (9 coeffs -- the irradiance standard)
NDIRS = 96         # bake directions per point (earlier probe: n=96 is within 0.005 of n=512, far cheaper)


def build_scene():
    # the showcase scene: a big flat floor + backdrop (very smooth transfer) and pillars + a sphere (shadow edges).
    # A fair mix -- NOT cherry-picked to be all-smooth.
    return (box(5.0, 0.1, 3.0).translate((0, -0.7, 0))
            .smooth_union(box(5.0, 3.0, 0.15).translate((0, 0.8, -1.4)), k=0.001)
            .smooth_union(box(0.3, 1.0, 0.3).rounded(0.04).translate((-1.5, -0.15, 0.3)), k=0.001)
            .smooth_union(sphere(0.5).translate((0.0, -0.15, 0.3)), k=0.001)
            .smooth_union(box(0.3, 1.0, 0.3).rounded(0.04).translate((1.5, -0.15, 0.3)), k=0.001))


def gbuffer(scene, W, H):
    # primary visibility: the visible surface points + normals (what we must shade)
    cam = Camera(eye=(0.0, 1.0, 4.2), target=(0.0, -0.2, -0.3), fov_deg=46, aspect=W / H)
    eye, D = cam.ray_dirs(W, H)
    D = D.reshape(-1, 3)
    O = np.broadcast_to(np.array(eye, float), (W * H, 3)).copy()
    hit, _, P = sphere_trace(scene, O, D)
    Pv = P[hit]
    Nv = sdf_normal(scene, Pv)
    return Pv, Nv, hit


def features(P, N):
    # the space in which "nearby" means "similar transfer": position (normalised to the bbox) plus the normal.
    # transfer depends on BOTH where you are (occlusion) and which way you face (the cosine hemisphere), so a point
    # facing a different direction is a different landmark even if it sits nearby. normal_weight tunes their balance.
    lo = P.min(0); span = np.maximum(P.max(0) - lo, 1e-6)
    Pn = (P - lo) / span                                        # position in [0,1]^3
    normal_weight = 0.6
    return np.concatenate([Pn, normal_weight * N], axis=1)


def project_transfer(feat_all, feat_land, T_land, sigma):
    # affinity-weighted blend: each point's transfer = weighted average of the landmark transfers, weight falling off
    # with distance in (position+normal) space -- distant landmarks (behind an occluder, facing away) get ~0 weight.
    # Blocked over rows to bound memory (the N x m affinity is never materialised whole).
    N = len(feat_all); m = len(feat_land)
    T_proj = np.zeros((N, T_land.shape[1]))
    disagree = np.zeros(N)                                       # how much the contributing landmarks disagree
    blk = 4096
    for s in range(0, N, blk):
        A = gaussian_affinity(feat_all[s:s + blk], feat_land, sigma)   # (b, m)
        rowsum = A.sum(1, keepdims=True)
        near = (rowsum[:, 0] < 1e-9)                             # points with no nearby landmark -> use the nearest
        W = np.where(rowsum < 1e-9, 0.0, A / np.maximum(rowsum, 1e-30))
        if near.any():
            nn = np.argmax(A[near], axis=1)
            W[near] = 0.0
            W[np.where(near)[0], nn] = 1.0
        Tp = W @ T_land                                         # (b, ncoeff) projected transfer
        T_proj[s:s + blk] = Tp
        # disagreement = weighted spread of the landmark transfers around the blended value (a blind-spot detector
        # that needs NO ground truth): high where landmarks pull in different directions == near a shadow edge.
        # E[||T_land - Tp||^2] under the weights, summed over coeffs.
        d = (W * (T_land[None, :, :] - Tp[:, None, :]).__pow__(2).sum(2)).sum(1)
        disagree[s:s + blk] = np.sqrt(np.maximum(d, 0.0))
    return T_proj, disagree


def main():
    scene = build_scene()
    W, H = 200, 133
    Pv, Nv, hit = gbuffer(scene, W, H)
    n_pts = len(Pv)
    print(f"visible points (the shading workload): {n_pts}")

    # the dome we are relighting under (a constant SH vector -- cheap, shared by every method)
    dome = DomeLight(color=(0.30, 0.38, 0.55), ground_color=(0.14, 0.12, 0.10), intensity=1.0)
    light_sh = project_env_to_sh(lambda d: dome.radiance(d), order=ORDER, n=1024)
    albedo = np.array([0.8, 0.8, 0.8])

    # ---- FULL bake: the baseline we must beat, and the ground truth for error ----
    t = time.time()
    T_full = precompute_transfer(scene, Pv, Nv, order=ORDER, n=NDIRS)
    t_full = time.time() - t
    shade_full = shade_prt(T_full, light_sh, albedo[None, :]).mean(1)   # per-point luminance under the dome
    denom = max(shade_full.mean(), 1e-6)
    print(f"FULL bake: {t_full:6.1f}s   (baseline)   mean shade={shade_full.mean():.3f}\n")

    feat_all = features(Pv, Nv)

    print(f"{'m':>6} {'%pts':>5} | {'land_bake':>9} {'project':>8} {'total':>7} {'speedup':>7} | "
          f"{'meanErr%':>8} {'p95Err%':>7} | refine {'(+n)':>7} {'total*':>7} {'meanErr%*':>9}")
    for frac in (0.01, 0.02, 0.05, 0.10):
        m = max(8, int(n_pts * frac))
        # pick landmark 'bricks' that cover the position+normal variety
        land = farthest_point_landmarks(feat_all, m, seed=0)
        feat_land = feat_all[land]
        # bandwidth: a fraction of the median nearest-landmark spacing, so blending stays LOCAL
        dd = gaussian_affinity(feat_land[:min(m, 400)], feat_land, sigma=1.0)  # reuse as a distance proxy
        # median nearest-neighbour distance among landmarks (exclude self)
        Dll = np.sqrt(np.maximum(((feat_land[:min(m, 400), None, :] - feat_land[None, :, :]) ** 2).sum(2), 0.0))
        Dll[Dll == 0] = np.inf
        sigma = float(np.median(Dll.min(1))) * 0.9

        t = time.time()
        T_land = precompute_transfer(scene, Pv[land], Nv[land], order=ORDER, n=NDIRS)   # bake ONLY landmarks
        t_land = time.time() - t

        t = time.time()
        T_proj, disagree = project_transfer(feat_all, feat_land, T_land, sigma)
        t_proj = time.time() - t

        shade_proj = shade_prt(T_proj, light_sh, albedo[None, :]).mean(1)
        err = np.abs(shade_proj - shade_full) / denom * 100.0
        total = t_land + t_proj
        speed = t_full / max(total, 1e-6)

        # ---- REFINE: re-bake the worst 'blind spots' the disagreement flagged, measure error drop + added cost ----
        refine_frac = 0.10                                      # re-bake the 10% most-disagreeing points exactly
        k = int(n_pts * refine_frac)
        worst = np.argpartition(disagree, -k)[-k:]
        t = time.time()
        T_ref = precompute_transfer(scene, Pv[worst], Nv[worst], order=ORDER, n=NDIRS)
        t_ref = time.time() - t
        T_fixed = T_proj.copy(); T_fixed[worst] = T_ref
        shade_fixed = shade_prt(T_fixed, light_sh, albedo[None, :]).mean(1)
        err2 = np.abs(shade_fixed - shade_full) / denom * 100.0
        total2 = total + t_ref
        # does the disagreement signal actually find the high-error points? (correlation with true error)
        corr = float(np.corrcoef(disagree, np.abs(shade_proj - shade_full))[0, 1])

        print(f"{m:>6} {frac*100:>4.0f}% | {t_land:>8.1f}s {t_proj:>7.2f}s {total:>6.1f}s {speed:>6.1f}x | "
              f"{err.mean():>7.2f} {np.percentile(err,95):>6.2f} | {k:>7} {total2:>6.1f}s {err2.mean():>8.2f}   "
              f"[disagree~err corr={corr:+.2f}]")

    print("\nread: speedup > 1 means landmark+project is faster than the full bake; meanErr%/p95Err% is the shading"
          "\nerror vs the full bake; the *columns add the blind-spot re-bake. corr>0 means the (ground-truth-free)"
          "\ndisagreement signal really does point at the high-error pixels.")


if __name__ == "__main__":
    main()

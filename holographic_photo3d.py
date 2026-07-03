"""holographic_photo3d.py -- ABSTAINING photo-to-3D: turn a single depth map + colour image into per-pixel 3D
Gaussians (splats), but ONLY where the reconstruction is actually observed. A single view sees the FRONT of a
surface and nothing else, so an honest reconstruction emits the front and ABSTAINS on the rest -- it does not
invent the back of the object or stretch a wall of fake geometry across an occlusion edge.

WHY THIS EXISTS (Forecasting sweep, sec.5 -- the depth/photo-to-3D delegation; and the photo-to-3D backlog)
-----------------------------------------------------------------------------------------------------------
The sweep's principle -- every estimate should carry a confidence and abstain when it does not know -- applied to
geometry. Lifting a photo to 3D is an estimate, and it has three honest boundaries where the estimate is NOT
supported by what was seen:
  1. INVALID depth (a hole, a missing/zero reading) -- nothing was observed there,
  2. an OCCLUSION EDGE (a big depth step) -- naively unprojecting connects the near and far surface with a
     stretched sheet of points that exists in no scene; the region behind the near edge is occluded, unknown,
  3. a GRAZING surface (its normal nearly perpendicular to the view) -- the depth there is unreliable and thin.
And the deepest boundary, kept loudest: the BACK of every object is unobserved from one view. This pipeline emits
the visible front surface with a per-pixel confidence and abstains everywhere else -- the "we don't know the back
of the object" boundary made mechanical, not a closed watertight mesh guessed from one picture.

THE HONEST TOOL (kept): confidence here is a GEOMETRIC support score in [0,1] (valid AND continuous AND
front-facing), not a calibrated probability -- a single depth map gives no held-out calibration set per pixel, so
this is the estimator that fits (support, with abstention), the same correction made for the renderer's stop.

Real basis: the pinhole unprojection is standard; the per-pixel-Gaussian output is the Splatter Image / Flash3D
shape (one 3D Gaussian per pixel), here gated by observability. Deterministic; NumPy + stdlib only.
"""
import numpy as np


def unproject(depth, fx, fy, cx, cy):
    """Turn a depth map into 3D points in camera space via the pinhole model. depth[v, u] is the distance along
    the view; the 3D point is (x, y, z) = ((u - cx) * z / fx, (v - cy) * z / fy, z). Returns (H, W, 3)."""
    depth = np.asarray(depth, float)
    H, W = depth.shape
    u = np.arange(W)[None, :].astype(float)
    v = np.arange(H)[:, None].astype(float)
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def _relative_depth_step(depth):
    """Per-pixel depth-step size RELATIVE to the depth: max jump to a 4-neighbour, divided by the depth. A big
    value is an occlusion edge (a 1 cm step at 1 m matters far more than the same step at 10 m)."""
    d = np.asarray(depth, float)
    step = np.zeros_like(d)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.roll(np.roll(d, dy, axis=0), dx, axis=1)
        step = np.maximum(step, np.abs(d - shifted))
    return step / (np.abs(d) + 1e-6)


def _facing(points):
    """How front-facing each pixel's surface is, in [0,1]: |n_z| of the estimated normal (the camera looks down
    +z, so a surface facing the camera has n_z near +/-1, a grazing surface near 0). Normals come from the cross
    product of the unprojected point's du/dv differences."""
    p = np.asarray(points, float)
    dpdu = np.gradient(p, axis=1)                                  # change across columns (u)
    dpdv = np.gradient(p, axis=0)                                  # change across rows (v)
    n = np.cross(dpdu, dpdv)
    norm = np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
    n = n / norm
    return np.abs(n[:, :, 2])                                      # |n_z| = how much it faces the camera


def depth_confidence(depth, fx, fy, cx, cy, step_tol=0.05, facing_floor=0.2):
    """Per-pixel geometric confidence in [0,1] and an abstain mask. Confidence is the product of three supports:
      valid  -- depth is finite and > 0 (something was observed),
      edge   -- the relative depth step is small (NOT an occlusion boundary that would stretch fake geometry),
      facing -- the surface faces the camera (not a grazing, unreliable, thin surface).
    A pixel abstains when confidence is ~0 (any support failed). Returns (confidence (H,W), abstain (H,W) bool)."""
    d = np.asarray(depth, float)
    valid = np.isfinite(d) & (d > 0)
    points = unproject(np.where(valid, d, 1.0), fx, fy, cx, cy)    # unproject with a placeholder where invalid
    step = _relative_depth_step(np.where(valid, d, d[valid].mean() if valid.any() else 1.0))
    # smooth, continuous surface -> edge support near 1; a sharp step -> support falls to 0
    edge_support = np.clip(1.0 - step / (step_tol + 1e-9), 0.0, 1.0)
    facing = _facing(points)
    facing_support = np.clip((facing - facing_floor) / (1.0 - facing_floor + 1e-9), 0.0, 1.0)
    conf = valid.astype(float) * edge_support * facing_support
    abstain = conf <= 1e-6
    return conf, abstain


def photo_to_gaussians(depth, colour, fx, fy, cx, cy, confidence_floor=0.3):
    """Photo-to-3D with abstention: unproject the CONFIDENT pixels into per-pixel 3D Gaussians and abstain on the
    rest. Each emitted Gaussian carries a position (from unproject), a colour (from the image), a radius (from the
    local point spacing, so nearer/denser points get smaller splats), and its confidence weight. Returns a dict:
      positions (M,3), colours (M,3), radii (M,), confidences (M,),
      abstain_mask (H,W) bool, n_observed, n_abstained, coverage (observed / total).
    The honest boundary: M covers ONLY the visible, confident FRONT surface -- the back and the occlusion edges
    are abstained, never invented."""
    depth = np.asarray(depth, float)
    colour = np.asarray(colour, float)
    H, W = depth.shape
    points = unproject(np.where(np.isfinite(depth) & (depth > 0), depth, 1.0), fx, fy, cx, cy)
    conf, abstain = depth_confidence(depth, fx, fy, cx, cy)
    keep = (conf >= confidence_floor) & (~abstain)

    # a per-pixel splat radius from the local spacing between neighbouring unprojected points (in world units)
    dpdu = np.linalg.norm(np.gradient(points, axis=1), axis=2)     # spacing across columns (u)
    dpdv = np.linalg.norm(np.gradient(points, axis=0), axis=2)     # spacing across rows (v)
    radius = 0.5 * (dpdu + dpdv)

    idx = np.where(keep)
    out_positions = points[idx]
    out_colours = colour[idx] if colour.ndim == 3 else np.zeros((len(out_positions), 3))
    out_radii = radius[idx]
    out_conf = conf[idx]
    total = H * W
    return {
        "positions": out_positions,
        "colours": out_colours,
        "radii": out_radii,
        "confidences": out_conf,
        "abstain_mask": abstain,
        "n_observed": int(keep.sum()),
        "n_abstained": int(total - keep.sum()),
        "coverage": float(keep.sum()) / float(total),
    }


def _selftest():
    """A two-plane depth map (a near plane and a far plane meeting at an occlusion EDGE) unprojects to two flat
    surfaces; the abstention fires exactly at the edge (stretched geometry avoided) and on invalid/grazing pixels;
    the confident splats sit on the observed front planes at the right depths; the back is never emitted."""
    H = W = 40
    fx = fy = 40.0
    cx = cy = 20.0

    # left half is a near plane at z=1, right half a far plane at z=3 -> a hard occlusion edge down the middle
    depth = np.empty((H, W))
    depth[:, :W // 2] = 1.0
    depth[:, W // 2:] = 3.0
    colour = np.zeros((H, W, 3))
    colour[:, :W // 2] = [0.8, 0.2, 0.2]
    colour[:, W // 2:] = [0.2, 0.3, 0.8]
    depth[0, 0] = 0.0                                             # one invalid pixel (a hole)

    conf, abstain = depth_confidence(depth, fx, fy, cx, cy)

    # (1) the occlusion edge abstains: the column at the near/far boundary is a big relative step
    edge_col = W // 2
    assert abstain[H // 2, edge_col] or abstain[H // 2, edge_col - 1]
    # (2) the flat interior of each plane is confident (not near the edge)
    assert not abstain[H // 2, 5] and not abstain[H // 2, W - 5]
    assert conf[H // 2, 5] > 0.5
    # (3) the invalid pixel abstains
    assert abstain[0, 0]

    g = photo_to_gaussians(depth, colour, fx, fy, cx, cy, confidence_floor=0.3)
    # (4) the confident splats reconstruct the two front planes at z=1 and z=3
    z = g["positions"][:, 2]
    assert abs(z[np.isclose(z, 1.0, atol=0.1)].mean() - 1.0) < 1e-6
    assert np.isclose(z, 3.0, atol=0.1).any()
    # (5) abstention removed a real fraction (the edge band + the hole), and coverage is high but < 1 (honest)
    assert 0.0 < g["coverage"] < 1.0
    assert g["n_abstained"] >= 1

    # (6) deterministic
    g2 = photo_to_gaussians(depth, colour, fx, fy, cx, cy, confidence_floor=0.3)
    assert np.array_equal(g["positions"], g2["positions"])

    print("holographic_photo3d selftest OK: two-plane depth unprojects to front planes at z=1 and z=3; abstention "
          "fires at the occlusion edge and the invalid hole; %d confident splats cover %.0f%% of pixels, the rest "
          "abstained (the back and the edge are NOT invented); deterministic"
          % (g["n_observed"], 100.0 * g["coverage"]))


if __name__ == "__main__":
    _selftest()

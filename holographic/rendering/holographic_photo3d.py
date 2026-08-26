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


def depth_to_mesh(depth, colour=None, fx=None, fy=None, cx=None, cy=None,
                  depth_scale=1.0, discontinuity=0.08, smooth_iters=0):
    """DEPTH MAP -> a CLEAN triangulated HEIGHT-FIELD MESH (the mesh-cleanup path for single-view photo-to-3D). Every
    pixel becomes a vertex at its unprojected 3-D position; each 2x2 pixel block becomes two triangles -- EXCEPT
    where the depth jumps more than `discontinuity` across the quad, where the triangle is DROPPED so the mesh does
    not stretch a rubber sheet between the near foreground and the far background (the artifact that makes naive
    depth meshes look melted). Optionally per-vertex `colour` from the source photo is carried through.

    This is a REGULAR-GRID surface, so unlike the dual-contour path (points_to_mesh) it is watertight-by-construction
    as a 2-manifold-with-boundary and has NO non-manifold edges -- it is directly smoothable / decimatable / textured.
    `depth` is (H,W) in [0,1] with 1=NEAREST (the convention fuse_depth/haze_depth/shape_from_shading return); it is
    converted to camera Z internally (near = small Z). `fx,fy,cx,cy` default to a reasonable pinhole for the image
    size. `smooth_iters` runs that many Laplacian passes on the Z to reduce depth noise before meshing.

    Returns (mesh, vertex_colours) where mesh is a holographic_mesh.Mesh of triangles and vertex_colours is (V,3) or
    None. Deterministic, pure NumPy. KEPT NEGATIVES: single-view, so this is the VISIBLE FRONT as a relief, not a
    solid (the back and everything behind a depth discontinuity are unobserved -- the dropped triangles leave honest
    holes, they do not invent geometry); relative depth (not metric); a wrong `discontinuity` either melts the scene
    (too high) or shreds a smooth slope into confetti (too low)."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    depth = np.asarray(depth, float)
    H, W = depth.shape
    if fx is None: fx = 0.9 * W
    if fy is None: fy = 0.9 * W
    if cx is None: cx = W / 2.0
    if cy is None: cy = H / 2.0

    # depth is 1=near; camera Z should be SMALL for near, LARGE for far -> Z = (1 - depth). Scale for aspect.
    z = (1.0 - depth) * float(depth_scale) + 0.3        # +0.3 keeps everything in front of the camera (Z>0)

    # optional Laplacian smoothing of the depth to knock down per-pixel noise before it becomes geometry
    for _ in range(int(smooth_iters)):
        z = 0.5 * z + 0.5 * 0.25 * (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1))

    pts = unproject(z, fx, fy, cx, cy)                  # (H,W,3)
    V = pts.reshape(-1, 3)
    idx = np.arange(H * W).reshape(H, W)

    # build the two triangles of every 2x2 block, but CULL any triangle whose vertices span a depth jump bigger than
    # `discontinuity` -- this is what stops the near foreground from being welded to the far background.
    z00 = z[:-1, :-1]; z10 = z[1:, :-1]; z01 = z[:-1, 1:]; z11 = z[1:, 1:]
    # max pairwise |dz| within each quad (in the ORIGINAL 1=near depth units, scale-independent of depth_scale)
    d = depth
    d00 = d[:-1, :-1]; d10 = d[1:, :-1]; d01 = d[:-1, 1:]; d11 = d[1:, 1:]
    span = np.maximum.reduce([np.abs(d00 - d10), np.abs(d00 - d01), np.abs(d11 - d10),
                              np.abs(d11 - d01), np.abs(d00 - d11), np.abs(d10 - d01)])
    ok = span <= float(discontinuity)                   # quads flat enough to mesh

    i00 = idx[:-1, :-1][ok]; i10 = idx[1:, :-1][ok]; i01 = idx[:-1, 1:][ok]; i11 = idx[1:, 1:][ok]
    # two triangles per kept quad, consistent winding
    tris = np.concatenate([np.stack([i00, i10, i11], 1),
                           np.stack([i00, i11, i01], 1)], axis=0)
    faces = [tuple(int(a) for a in t) for t in tris]

    # drop unreferenced vertices (pixels whose every incident quad was culled) so the mesh is compact
    used = np.unique(tris.ravel())
    remap = -np.ones(H * W, dtype=int); remap[used] = np.arange(len(used))
    Vc = V[used]
    faces = [(int(remap[a]), int(remap[b]), int(remap[c])) for (a, b, c) in faces]
    mesh = Mesh(Vc, faces)

    vcol = None
    if colour is not None:
        col = np.asarray(colour, float)
        if col.max() > 1.5:
            col = col / 255.0
        vcol = col.reshape(-1, 3)[used]
    return mesh, vcol


def _selftest_depth_to_mesh():
    # a synthetic depth with a sharp step: left half near, right half far. depth_to_mesh should (a) mesh each flat
    # side, and (b) CULL the triangles bridging the step (discontinuity), leaving two panels not one rubber sheet.
    H = W = 24
    d = np.zeros((H, W)); d[:, :W // 2] = 0.9; d[:, W // 2:] = 0.1   # near|far step, jump = 0.8 at the seam
    mesh_all, _ = depth_to_mesh(d, discontinuity=0.9)               # thresh 0.9 > 0.8 -> bridge the step (one sheet)
    mesh_cut, _ = depth_to_mesh(d, discontinuity=0.2)               # thresh 0.2 < 0.8 -> cull the step (two panels)
    assert mesh_cut.n_faces < mesh_all.n_faces, "discontinuity culling should DROP the bridging triangles"
    # a flat depth meshes to a full grid (no culling) with the expected face count ~ 2*(H-1)*(W-1)
    flat, _ = depth_to_mesh(np.full((H, W), 0.5), discontinuity=0.1)
    assert flat.n_faces == 2 * (H - 1) * (W - 1), f"flat depth should fully triangulate, got {flat.n_faces}"
    print(f"depth_to_mesh ok: flat->{flat.n_faces} faces (full grid); step culls "
          f"{mesh_all.n_faces - mesh_cut.n_faces} bridging faces.")


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
    _selftest_depth_to_mesh()


if __name__ == "__main__":
    _selftest()

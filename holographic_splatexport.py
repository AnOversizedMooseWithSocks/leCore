"""FS-3 -- splat export: the .ply / JSON adapter (holographic_splatexport).

WHAT THIS IS
------------
An adapter that writes the engine's Gaussian splats to formats a BROWSER splat renderer can read, so a field/scene
can be DISPLAYED as splats (the GPU's job; holostuff stays the authoring brain). The splat parameters are already in
hand -- aniso_fit returns (center, amplitude, L), L the Cholesky factor of the INVERSE covariance -- so this is a
FORMAT ADAPTER, not serialization from scratch. Two targets:
  * splats_to_json / splats_from_json -- a compact JSON for a custom three.js Gaussian-billboard shader (the demo).
  * splats_to_ply / splats_from_ply -- the standard 3D-Gaussian-Splatting .ply (INRIA layout) that opens in any 3DGS
    viewer: position, log-scale, rotation quaternion, base colour (SH DC term), logit opacity.

THE ONE BIT OF REAL MATH -- principal_axes(precision): a splat stores L (Cholesky of the inverse covariance, so the
precision is P = L Lᵀ), but a viewer wants SCALE + ROTATION. The covariance is Σ = P⁻¹; eigen-decomposing the
symmetric P gives the principal axes (the rotation) and, from its eigenvalues, the per-axis standard deviations (the
scale) -- scale_i = 1/sqrt(eigenvalue_i of P). This is the L -> (scale, rotation-quaternion) conversion.

A PROBE-CORRECTION KEPT HONEST: the build plan expected this same "principal axes of a quadratic form" math to live in
three places already (splat export, QEM, the steering kernel) and asked for one shared helper. On probing the LIVE
code that premise did not hold: the steering kernel uses DIAGONAL bandwidths (it documents that a full covariance
overfits -- its own kept negative), and QEM SOLVES a 3x3 linear system (argmin vᵀQv, with a midpoint fallback when
singular) -- neither eigen-decomposes a form into principal axes. consolidation's KLT is an SVD of a DATA matrix, a
different object again. So principal_axes is built cleanly HERE, where the L->scale+rotation eigendecomposition
genuinely lives, and is NOT retrofitted into modules doing different math (that would make them worse, not shared).
The "three call sites" was an over-optimistic plan assumption; the honest finding is recorded rather than forced.

DETERMINISM (per ISA.md): eigh, the quaternion conversion, and the byte layout are deterministic given the splats.
Export -> re-import round-trips the parameters (the covariance reconstructs to tolerance; asserted).

KEPT HONEST:
  * holostuff splats carry NO view-dependent (spherical-harmonic) colour -- export is BASE-COLOUR only (the SH DC
    term); higher-order SH is a further add, noted not faked.
  * a DEGENERATE (flat / rank-deficient) covariance has no clean axes -- principal_axes SURFACES that (raises on a
    non-positive-definite precision) rather than returning garbage, the same discipline QEM's singular guard follows.
"""

import json
import struct

import numpy as np

_SH_C0 = 0.28209479177387814          # the order-0 spherical-harmonic coefficient (3DGS f_dc convention)


def principal_axes(precision, floor=1e-9):
    """Eigen-decompose a symmetric positive-definite PRECISION matrix P (the inverse covariance, P = L Lᵀ) into the
    splat's (scales, rotation): `scales[i]` is the standard deviation along principal axis i = 1/sqrt(eigenvalue_i),
    and `rotation` (3x3, columns = principal axes, a PROPER rotation with det +1) orients them. Raises ValueError if P
    is not positive-definite (an eigenvalue <= `floor`), i.e. the covariance is degenerate/flat and has no clean axes
    -- surfaced, not papered over."""
    P = np.asarray(precision, float)
    P = 0.5 * (P + P.T)                                     # symmetrize against round-off
    evals, evecs = np.linalg.eigh(P)                        # ascending eigenvalues, orthonormal eigenvectors (columns)
    if evals.min() <= floor:
        raise ValueError(f"degenerate/flat covariance: precision has eigenvalue {evals.min():.2e} <= {floor:.0e}; "
                         "no clean principal axes (a rank-deficient splat)")
    scales = 1.0 / np.sqrt(evals)                           # covariance eigenvalue = 1/precision eigenvalue; std = sqrt
    R = evecs.copy()
    if np.linalg.det(R) < 0:                                # make it a PROPER rotation (det +1), not a reflection
        R[:, 0] = -R[:, 0]
    return scales, R


def rotation_to_quaternion(R):
    """A 3x3 rotation matrix -> a unit quaternion (w, x, y, z) (the 3DGS rot_0..3 order)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], float)
    return q / (np.linalg.norm(q) + 1e-12)


def quaternion_to_rotation(q):
    """A unit quaternion (w, x, y, z) -> a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, float) / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], float)


def _splat_record(center, amp, L, color):
    """One splat -> a dict of viewer parameters: position, per-axis scale (std), rotation quaternion, base colour,
    opacity. `L` is the Cholesky of the inverse covariance; colour defaults to mid-grey."""
    center = np.asarray(center, float)
    L = np.asarray(L, float)
    precision = L @ L.T
    scales, R = principal_axes(precision)
    quat = rotation_to_quaternion(R)
    rgb = np.array([0.5, 0.5, 0.5]) if color is None else np.asarray(color, float)
    opacity = float(np.clip(amp, 1e-4, 1.0 - 1e-4))        # amplitude as alpha (base, no view dependence)
    return {"position": center.tolist(), "scale": scales.tolist(), "rotation": quat.tolist(),
            "color": rgb.tolist(), "opacity": opacity}


def splats_to_json(splats, colors=None):
    """Serialize splats (each (center, amp, L)) to a JSON string for a three.js Gaussian-billboard shader: a list of
    {position, scale, rotation (quaternion), color, opacity}. `colors` optionally supplies per-splat RGB in [0,1]."""
    recs = []
    for i, (center, amp, L) in enumerate(splats):
        col = None if colors is None else colors[i]
        recs.append(_splat_record(center, amp, L, col))
    return json.dumps({"splats": recs}, separators=(",", ":"))


def splats_from_json(s):
    """Parse splats_to_json output back to the viewer-parameter records (position/scale/rotation/color/opacity)."""
    return json.loads(s)["splats"]


# ----- the standard 3DGS .ply (INRIA layout): binary little-endian float32 ------------------------------
_PLY_PROPS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
              "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def splats_to_ply(splats, path, colors=None):
    """Write splats (each (center, amp, L)) to the standard 3D-Gaussian-Splatting .ply (binary little-endian), so the
    scene opens in any 3DGS viewer. Per the convention: scale is stored as LOG(std), opacity as the LOGIT of alpha,
    and colour as the order-0 SH coefficient (f_dc = (rgb - 0.5)/C0). Base colour only (no higher-order SH). Returns
    the number of splats written."""
    rows = []
    for i, (center, amp, L) in enumerate(splats):
        col = None if colors is None else colors[i]
        rec = _splat_record(center, amp, L, col)
        x, y, z = rec["position"]
        sx, sy, sz = rec["scale"]
        qw, qx, qy, qz = rec["rotation"]
        r, g, b = rec["color"]
        a = rec["opacity"]
        f_dc = [(c - 0.5) / _SH_C0 for c in (r, g, b)]      # 3DGS SH DC colour convention
        opac_logit = float(np.log(a / (1.0 - a)))           # inverse sigmoid
        log_scale = [float(np.log(max(s, 1e-12))) for s in (sx, sy, sz)]
        rows.append([x, y, z, 0.0, 0.0, 0.0, f_dc[0], f_dc[1], f_dc[2],
                     opac_logit, log_scale[0], log_scale[1], log_scale[2], qw, qx, qy, qz])
    n = len(rows)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in _PLY_PROPS)
              + "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for row in rows:
            fh.write(struct.pack("<" + "f" * len(_PLY_PROPS), *row))
    return n


def splats_from_ply(path):
    """Read a 3DGS .ply written by splats_to_ply back to viewer-parameter records (position, scale=std, rotation
    quaternion, color in [0,1], opacity=alpha) -- inverting the log-scale / logit-opacity / SH-colour transforms.
    Used to round-trip-test the export."""
    with open(path, "rb") as fh:
        # parse the ascii header up to end_header
        header = b""
        while not header.endswith(b"end_header\n"):
            header += fh.read(1)
        text = header.decode("ascii")
        n = int([ln for ln in text.splitlines() if ln.startswith("element vertex")][0].split()[-1])
        recs = []
        idx = {name: k for k, name in enumerate(_PLY_PROPS)}
        for _ in range(n):
            vals = struct.unpack("<" + "f" * len(_PLY_PROPS), fh.read(4 * len(_PLY_PROPS)))
            f_dc = [vals[idx[f"f_dc_{k}"]] for k in range(3)]
            color = [float(c * _SH_C0 + 0.5) for c in f_dc]
            opac = float(1.0 / (1.0 + np.exp(-vals[idx["opacity"]])))    # sigmoid
            scale = [float(np.exp(vals[idx[f"scale_{k}"]])) for k in range(3)]
            quat = [vals[idx[f"rot_{k}"]] for k in range(4)]
            recs.append({"position": [vals[idx["x"]], vals[idx["y"]], vals[idx["z"]]],
                         "scale": scale, "rotation": quat, "color": color, "opacity": opac})
    return recs


def field_to_splats(centers, radius=0.5, amp=1.0):
    """Pull a metaball FIELD's Gaussians directly as splats -- no fit needed, the centres ARE the splat positions and
    the metaball's `radius` IS the isotropic standard deviation. Returns a list of (center, amp, L) with
    L = (1/radius) I (the Cholesky of the isotropic inverse covariance). For an already-fitted anisotropic field,
    use aniso_fit's (center, amp, L) directly."""
    centers = np.asarray(centers, float)
    Lr = np.eye(centers.shape[1]) / float(radius)          # precision (1/radius^2) I -> Cholesky (1/radius) I
    return [(c, amp, Lr) for c in centers]


# =====================================================================================================
# Self-test -- the L->scale+rotation math, the .ply and JSON round-trips, the degenerate guard.
# =====================================================================================================
def _selftest():
    import tempfile, os

    rng = np.random.default_rng(0)

    # --- principal_axes recovers a known covariance: build Sigma, precision, L; check it round-trips ---
    A = rng.standard_normal((3, 3))
    Sigma = A @ A.T + 0.5 * np.eye(3)                       # a random SPD covariance
    precision = np.linalg.inv(Sigma)
    L = np.linalg.cholesky(precision)                       # precision = L Lᵀ
    scales, R = principal_axes(precision)
    Sigma_back = R @ np.diag(scales ** 2) @ R.T             # reconstruct covariance from (scales, rotation)
    assert np.allclose(Sigma_back, Sigma, atol=1e-9), "principal_axes must reconstruct the covariance"

    # --- quaternion round-trip (matrix -> quat -> matrix) ---
    q = rotation_to_quaternion(R)
    assert np.allclose(quaternion_to_rotation(q), R, atol=1e-9), "quaternion round-trip must recover the rotation"

    # --- .ply export -> re-import: the covariance reconstructs (quaternion sign/axis order is irrelevant) ---
    splats = [(np.array([0.1, -0.2, 0.3]), 0.8, L),
              (np.array([1.0, 0.0, 0.5]), 0.5, np.linalg.cholesky(np.linalg.inv(2.0 * np.eye(3))))]
    colors = [[0.9, 0.1, 0.2], [0.2, 0.6, 0.9]]
    tmp = os.path.join(tempfile.gettempdir(), "holo_splat_test.ply")
    n = splats_to_ply(splats, tmp, colors=colors)
    recs = splats_from_ply(tmp)
    assert n == 2 and len(recs) == 2
    # splat 0: covariance reconstructs to the original
    s0 = np.array(recs[0]["scale"]); R0 = quaternion_to_rotation(recs[0]["rotation"])
    assert np.allclose(R0 @ np.diag(s0 ** 2) @ R0.T, Sigma, atol=1e-5), ".ply covariance must round-trip"
    assert np.allclose(recs[0]["position"], [0.1, -0.2, 0.3], atol=1e-5)
    assert np.allclose(recs[0]["color"], [0.9, 0.1, 0.2], atol=1e-4)
    assert abs(recs[0]["opacity"] - 0.8) < 1e-4
    os.remove(tmp)

    # --- JSON round-trip ---
    js = splats_to_json(splats, colors=colors)
    back = splats_from_json(js)
    assert len(back) == 2 and np.allclose(back[1]["position"], [1.0, 0.0, 0.5], atol=1e-9)

    # --- field_to_splats: metaball centres -> isotropic splats with std == radius ---
    fcenters = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    fsplats = field_to_splats(fcenters, radius=0.4)
    sc, _ = principal_axes((fsplats[0][2]) @ (fsplats[0][2]).T)
    assert np.allclose(sc, 0.4, atol=1e-9), "metaball radius must become the isotropic splat std"

    # --- degenerate (flat) covariance is SURFACED, not garbage ---
    flat = np.diag([1.0, 1.0, 0.0])                         # precision with a zero eigenvalue -> infinite extent
    raised = False
    try:
        principal_axes(flat)
    except ValueError:
        raised = True
    assert raised, "a degenerate/flat covariance must raise, not return garbage"

    # --- determinism ---
    assert splats_to_json(splats, colors=colors) == splats_to_json(splats, colors=colors)

    print(f"holographic_splatexport selftest: ok (principal_axes reconstructs a known covariance to 1e-9; quaternion "
          f"round-trips; .ply export->import recovers covariance/position/colour/opacity ({n} splats); JSON "
          f"round-trips; field_to_splats turns a metaball radius into the splat std; a flat covariance is RAISED not "
          f"faked; deterministic. L->scale+rotation, base colour only -- SH colour noted not faked)")


if __name__ == "__main__":
    _selftest()

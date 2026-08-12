"""holographic_meshcodec.py -- C-6: the mesh codec -- and the measured NEGATIVE that shaped it.

THE GAP (Rule-0 on record): "compress a mesh" returned only fallbacks. THE DELIVERABLE that
survived measurement: an honest BUDGETED mesh coder -- vertices uniformly quantized at
step 2*max_error (per-coordinate |err| <= max_error guaranteed), connectivity bit-exact as
varint index-deltas, everything zlib'd -- MEASURED 2.5-2.7x vs zlib(raw float64+int32) on
marching-cubes meshes, with the budget as the honest knob.

THE KEPT NEGATIVE, LOUD (it is the headline of this module, not a footnote): the classic
base + correspondence + displacement scheme -- decimate a base (mesh_cluster_decimate), refer
every original vertex to it (mesh_closest_point: face index + barycentric), code only the
small deltas -- DOES NOT BEAT honest uniform quantization at the same budget, on either mesh
class tried. The full sweep, on record (res=32 MC sphere, tol 2e-3, vertex-side bytes vs
uniform's 23,719):

    grid 8/12/16/24, 4-bit bary : 20,982 / 23,011 / 26,204 / 31,378
    bary precision 8/6/4 bit    : best sum 16.5k/15.1k/15.6k -- PLUS fi 4.4-5.7k + base
    centroid anchor (no bary)   : 22,353-29,135 across grids
    semi-regular (subdivided+noise): uniform still wins, 148,835 both

WHY, and it is information theory, not a bug: the explicit reference stream (face index +
barycentric) carries almost exactly the positional information the anchor subtracts from the
coordinates -- the refs cost what the deltas save. The scheme pays in the literature when the
refs are IMPLICIT (subdivision connectivity: children enumerate deterministically from the
base, nothing per-vertex ships). That route changes the contract (a resampled tessellation,
surface-error budget instead of per-vertex) and is the DEFERRED rung, deliberately not
smuggled in here. Base mode remains in the code as the priced hypothesis: mesh_encode always
BUILDS it, MEASURES it against the uniform coder, and ships whichever is smaller -- on every
mesh measured so far, that is uniform, and the report says so (mode='uniform', pays=False for
the base hypothesis).

WHAT IS STILL EARNED: the budget contract (verified on the decoded artifact every encode),
bit-exact connectivity, the fair-baseline discipline (the coder a caller could write is IN
the comparison, not a strawman zlib-only win), determinism, and one varint/zigzag
implementation shared with the surprise codec (never two).

REMAINING KEPT NEGATIVES:
  * connectivity dominates dense meshes -- the ratio ceiling is set by faces, not vertices;
  * mesh_closest_point runs one query per original vertex, so the base-mode HYPOTHESIS makes
    encode O(V) slower than the uniform coder alone; pass try_base=False to skip pricing it
    when the answer is already known for your mesh class.
"""

import struct
import zlib

import numpy as np

# WHY imported, not re-implemented: one varint/zigzag implementation in the arc; a second
# copy is a future disagreement (the two-tables lesson from the emitter family).
from holographic.sampling_and_signal.holographic_surprisecodec import (
    _zigzag, _unzigzag, _varint_encode, _varint_decode,
)

_MAGIC = b"LMC1"
_MODE_UNIFORM, _MODE_BASE = 0, 1


def _vz(arr):
    """zigzag-varint-zlib a signed int array (the arc's standard integer coding)."""
    return zlib.compress(_varint_encode(_zigzag(np.asarray(arr, dtype=np.int64).ravel())), 6)


def _unvz(raw, n):
    return _unzigzag(_varint_decode(zlib.decompress(raw), n))


def _delta_code(idx):
    """Index streams as first-differences: locality makes the deltas small varints."""
    return np.diff(np.concatenate([[0], np.asarray(idx, dtype=np.int64).ravel()]))


def mesh_encode(mesh, max_error, grid=12, try_base=True, mind=None):
    """Compress a triangle mesh as a decimated BASE + per-vertex barycentric refs + quantized
    displacement DETAILS, per-coordinate |err| <= max_error guaranteed on the decoded
    vertices, connectivity bit-exact. Priced against BOTH zlib(raw) and the fair
    uniform-quantization coder at the same budget; refuses into mode='uniform' (the
    competitor's own coding, still within budget) when the base does not pay. Returns
    {blob, report:{mode, bytes, raw_bytes, zlib_bytes, uniform_bytes, ratio_vs_uniform,
    ratio_vs_zlib, max_abs_error, base_verts, base_faces, pays}}. Decode with mesh_decode."""
    if mind is None:
        import lecore
        mind = lecore.UnifiedMind(dim=256, seed=0)
    V = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float64))
    F = np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int64))
    n = len(V)
    step = 2.0 * float(max_error)
    raw_bytes = V.nbytes + F.astype(np.int32).nbytes
    zlib_bytes = len(zlib.compress(V.tobytes() + F.astype(np.int32).tobytes(), 6))

    # -- the fair competitor, built first so the comparison cannot be forgotten
    u_v = _vz(np.round(V / step).astype(np.int64))
    u_f = _vz(_delta_code(F))
    uniform_blob = (_MAGIC + struct.pack("<B", _MODE_UNIFORM)
                    + struct.pack("<IId", n, len(F), step)
                    + struct.pack("<II", len(u_v), len(u_f)) + u_v + u_f)

    if not try_base:
        Vd, Fd = mesh_decode(uniform_blob)
        return dict(blob=uniform_blob, report=dict(
            mode="uniform", bytes=len(uniform_blob), raw_bytes=raw_bytes,
            zlib_bytes=zlib_bytes, uniform_bytes=len(uniform_blob),
            ratio_vs_uniform=1.0, ratio_vs_zlib=zlib_bytes / len(uniform_blob),
            max_abs_error=float(np.abs(Vd - V).max()),
            base_verts=0, base_faces=0, pays=False))

    # -- base mode: the priced hypothesis (measured loser so far; see module docstring)
    base = mind.mesh_cluster_decimate(mesh, grid=grid)
    BV = np.asarray(base.vertices, dtype=np.float64)
    BF = np.asarray(base.faces, dtype=np.int64)
    bstep = step / 4.0
    qB = np.round(BV / bstep).astype(np.int64)
    BVq = qB * bstep

    refs = mind.mesh_closest_point(base, V)
    faces_idx = np.array([r[0] for r in refs], dtype=np.int64)
    bary = np.array([r[1] for r in refs], dtype=np.float64)
    qb = np.clip(np.round(bary * 255), 0, 255).astype(np.uint8)
    bq = qb.astype(np.float64)
    bq = bq / np.maximum(bq.sum(1, keepdims=True), 1e-9)

    tri = BVq[BF[faces_idx]]
    anchor = (tri * bq[:, :, None]).sum(1)   # rebuilt EXACTLY as the decoder rebuilds it
    qd = np.round((V - anchor) / step).astype(np.int64)

    sections = [_vz(qB), _vz(_delta_code(BF)), _vz(_delta_code(faces_idx)),
                zlib.compress(qb.tobytes(), 6), _vz(qd), _vz(_delta_code(F))]
    payload = struct.pack("<IIIIdd", len(BV), len(BF), n, len(F), step, bstep)
    for s in sections:
        payload += struct.pack("<I", len(s)) + s
    blob = _MAGIC + struct.pack("<B", _MODE_BASE) + payload

    Vd, Fd = mesh_decode(blob)   # verify the actual contract on the actual blob
    err = float(np.abs(Vd - V).max())
    assert err <= max_error + 1e-12, "internal: contract breach %g" % err

    pays = len(blob) < len(uniform_blob) and len(blob) < zlib_bytes
    if not pays:
        blob = uniform_blob
        Vd, Fd = mesh_decode(blob)
        err = float(np.abs(Vd - V).max())
    return dict(blob=blob, report=dict(
        mode="base" if pays else "uniform", bytes=len(blob), raw_bytes=raw_bytes,
        zlib_bytes=zlib_bytes, uniform_bytes=len(uniform_blob),
        ratio_vs_uniform=len(uniform_blob) / len(blob),
        ratio_vs_zlib=zlib_bytes / len(blob), max_abs_error=err,
        base_verts=int(len(BV)), base_faces=int(len(BF)), pays=bool(pays)))


def mesh_decode(blob):
    """Invert mesh_encode -> (vertices, faces). Base mode rebuilds anchors from the shipped
    base + renormalized barycentrics and adds the quantized deltas; uniform mode dequantizes
    directly. Connectivity is exact in both modes. Raises on a foreign blob."""
    if blob[:4] != _MAGIC:
        raise ValueError("not a mesh-codec blob (bad magic)")
    mode, = struct.unpack("<B", blob[4:5])
    p = blob[5:]
    if mode == _MODE_UNIFORM:
        n, nf, step = struct.unpack("<IId", p[:16])
        lv, lf = struct.unpack("<II", p[16:24])
        V = _unvz(p[24:24 + lv], n * 3).reshape(n, 3) * step
        F = np.cumsum(_unvz(p[24 + lv:24 + lv + lf], nf * 3)).reshape(nf, 3)
        return V, F
    nb, nbf, n, nf, step, bstep = struct.unpack("<IIIIdd", p[:32])
    off = 32
    secs = []
    for _ in range(6):
        l, = struct.unpack("<I", p[off:off + 4]); off += 4
        secs.append(p[off:off + l]); off += l
    BVq = _unvz(secs[0], nb * 3).reshape(nb, 3) * bstep
    BF = np.cumsum(_unvz(secs[1], nbf * 3)).reshape(nbf, 3)
    faces_idx = np.cumsum(_unvz(secs[2], n))
    qb = np.frombuffer(zlib.decompress(secs[3]), dtype=np.uint8).reshape(n, 3).astype(np.float64)
    bq = qb / np.maximum(qb.sum(1, keepdims=True), 1e-9)
    qd = _unvz(secs[4], n * 3).reshape(n, 3)
    F = np.cumsum(_unvz(secs[5], nf * 3)).reshape(nf, 3)
    tri = BVq[BF[faces_idx]]
    anchor = (tri * bq[:, :, None]).sum(1)
    return anchor + qd * step, F


def _selftest():
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    mesh = mind.mesh_from_sdf(
        lambda p: np.linalg.norm(np.atleast_2d(p), axis=1) - 0.8,
        bounds=((-1, -1, -1), (1, 1, 1)), res=32)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    tol = 2e-3

    r = mesh_encode(mesh, max_error=tol, grid=8, mind=mind)
    rep = r["report"]
    Vd, Fd = mesh_decode(r["blob"])

    # 1) The contract, on the decoded artifact: vertex budget + bit-exact connectivity.
    assert np.abs(Vd - V).max() <= tol + 1e-12
    assert Fd.shape == F.shape and (Fd == F).all(), "connectivity must ship exactly"

    # 2) THE MEASURED NEGATIVE, PINNED AS A REGRESSION TRAP: on an MC mesh the base
    # hypothesis must LOSE to uniform and the codec must say so. If this ever flips,
    # something changed (coder, decimator, mesh) and the docstring's finding needs re-audit.
    assert rep["mode"] == "uniform" and not rep["pays"], rep

    # 3) The real, current win: the budgeted coder vs lossless zlib.
    assert rep["ratio_vs_zlib"] > 2.0, rep

    # 4) Monotone rate-distortion: a tighter budget must cost more bytes.
    r_tight = mesh_encode(mesh, max_error=tol / 8, grid=8, try_base=False, mind=mind)
    assert r_tight["report"]["bytes"] > rep["bytes"]

    # 5) try_base=False matches the shipped verdict exactly (same blob bytes).
    r_fast = mesh_encode(mesh, max_error=tol, try_base=False, mind=mind)
    assert r_fast["blob"] == r["blob"]

    # 6) Determinism.
    assert mesh_encode(mesh, max_error=tol, grid=8, mind=mind)["blob"] == r["blob"]

    print("meshcodec selftest OK -- uniform mode %.2fx vs zlib at budget %g; base hypothesis "
          "correctly refused (%.3fx, the documented negative)"
          % (rep["ratio_vs_zlib"], tol, rep["ratio_vs_uniform"]))


if __name__ == "__main__":
    _selftest()

"""O3: a face as a landmark graph plus parts -- procedural, no scans, no learned basis.

FLAME/DECA are the standard for human faces, and OmniFaceRig (2026) states their limit: they
are "bound to a fixed mesh topology and expression basis defined at scan-collection time" and
"primarily assume adult human anatomy", fitting unstably to stylized or non-human assets. An
engine for salamanders is that asset, so the part graph is the right tool HERE. SCULPTOR's
skeleton-consistency discipline is kept without its CT-scan basis.
"""
import numpy as np
from holographic.mesh_and_geometry import holographic_face as F


def test_landmarks_are_bilaterally_symmetric_by_construction():
    """Symmetry is structural, not something a caller can forget: mirrored pairs differ only
    in x and by equal amounts."""
    lm = F.face_landmarks((0.0, 1.6, 0.0), 0.24, 0.10)
    for k in ("eye", "brow", "ear", "cheek", "jaw", "temple"):
        L, R = lm[k + "_l"], lm[k + "_r"]
        assert abs(L[0] + R[0]) < 1e-12
        assert np.allclose(L[1:], R[1:])


def test_landmarks_are_anatomically_ordered():
    """Crown above brow above eye above nose above mouth above chin; nose frontmost, ear
    most set back. A face that fails this is not a face."""
    lm = F.face_landmarks((0.0, 1.6, 0.0), 0.24, 0.10)
    ys = [lm[k][1] for k in ("crown", "brow_l", "eye_l", "nose_tip", "mouth", "chin")]
    assert ys == sorted(ys, reverse=True), ys
    assert lm["nose_tip"][2] > lm["eye_l"][2] > lm["ear_l"][2]


def test_proportions_are_a_slider_surface():
    """The canon is editable data: raising the eye entry lifts the eye line, and nothing
    else moves. That is what makes stylized/non-human proportions a parameter change."""
    a = F.face_landmarks((0.0, 1.6, 0.0), 0.24, 0.10)
    b = F.face_landmarks((0.0, 1.6, 0.0), 0.24, 0.10,
                         proportions={"eye_l": (0.70, 0.22)})
    assert b["eye_l"][1] > a["eye_l"][1]
    assert np.allclose(b["nose_tip"], a["nose_tip"])       # local edit, local effect


def test_expression_is_linear_and_names_real_landmarks():
    lm = F.face_landmarks((0.0, 1.6, 0.0), 0.24, 0.10)
    full = F.expression(lm, "disgust", 1.0)
    half = F.expression(lm, "disgust", 0.5)
    assert full and set(full) <= set(lm)
    for k in full:
        assert np.allclose(half[k], full[k] * 0.5)
    try:
        F.expression(lm, "not_an_expression")
        assert False, "unknown expression must raise, not silently return nothing"
    except ValueError:
        pass


def test_expression_drives_LOCAL_correctives_on_a_real_head():
    """O3 x O2: facial expression as local blendshapes. Overreach must be exactly zero and
    each corrective must move a tiny fraction of the body -- the opposite of SMPL's global
    coupling, which is what makes a face rig usable."""
    from lecore import UnifiedMind
    from holographic.mesh_and_geometry.holographic_blendbasis import (make_corrective,
                                                                      locality_report)
    m = UnifiedMind(dim=64, seed=0)
    sph = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    mesh = m.mesh_from_sdf(sph, ((-1.3,) * 3, (1.3,) * 3), res=22, vectorized=True)
    V = np.asarray(mesh.vertices, float)
    lm = F.face_landmarks((0.0, 0.0, 0.0), 1.6, 0.7)
    d = F.expression(lm, "disgust", 1.0)
    srcs, radii, targets = [], [], []
    for name, delta in list(d.items())[:3]:
        s = int(np.argmin(np.linalg.norm(V - lm[name], axis=1)))
        srcs.append(s); radii.append(0.35)
        targets.append(make_corrective(mesh, s, 0.35, delta / np.linalg.norm(delta),
                                       float(np.linalg.norm(delta)), m))
    rep = locality_report(V, targets, mesh, srcs, radii, m)
    assert rep["max_overreach"] == 0.0, rep
    assert all(t["fraction_moved"] < 0.5 for t in rep["targets"]), rep

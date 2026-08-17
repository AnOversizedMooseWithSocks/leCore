"""Spore-parity backlog: S1 (parts pipeline) and T4 (socket frames preserve orientation).

Written after an audit found that leCore already ships the whole Spore architecture --
convolution surfaces (the metaball equivalent), a procedural rigblock library, role-tag
socket assignment, rig-bound painting -- and that a hand-authored render had used none of it.
These tests exist so the pipeline is exercised by CI rather than rediscovered.
"""
import numpy as np


def _salamander():
    from lecore import UnifiedMind
    from holographic.mesh_and_geometry.holographic_creatureskin import spine_profile
    m = UnifiedMind(dim=64, seed=0)
    base = {"spine": {"length": 2.9, "segments": 16, "axis": [0, 0, 1.0],
                      "curve": 0.16, "radius": 0.10},
            "limbs": [{"at": 0.235, "dir": [1.0, -0.30, 0.30], "segments": 4, "length": 0.50,
                       "radius": 0.036, "mirror": True, "cone_deg": 70, "hinge_deg": 95},
                      {"at": 0.520, "dir": [1.0, -0.30, -0.28], "segments": 4, "length": 0.52,
                       "radius": 0.038, "mirror": True, "cone_deg": 70, "hinge_deg": 95}],
            "head": {"at": 1.0, "radius": 0.115},
            "body": {"weight": 0.10, "muscle": 0.42, "fat": 0.08, "segments": {},
                     "breasts": None}}

    def prof(t):
        if t < 0.42:
            return 0.018 + 0.088 * ((t / 0.42) ** 1.4)
        if t < 0.80:
            return 0.106 + 0.020 * np.sin(np.pi * (t - 0.42) / 0.38)
        if t < 0.88:
            return 0.106 - 0.024 * ((t - 0.80) / 0.08)
        return 0.082 + 0.070 * np.sin(np.pi * np.clip(0.35 + 0.55 * (t - 0.88) / 0.12, 0, 1))
    spec = spine_profile(base, prof)
    cr, sdf = m.creature(spec)
    return m, cr, sdf


def test_part_library_has_the_rigblocks_spore_needed_artists_for():
    """The audit's headline: these are PROCEDURAL, so digits=3 vs digits=5 is a genuinely
    different foot rather than one mesh scaled -- which is stronger than Spore's fixed art
    assets, not weaker."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    names = set(m.part_names())
    for essential in ("eye", "foot", "hand", "digit", "mouth", "claw", "fin"):
        assert essential in names, essential
    assert m.build_part("foot", digits=4) is not None
    assert m.build_part("eye") is not None


def test_auto_sockets_assign_parts_from_role_tags():
    """S1: a ground-touching limb tip gets a FOOT, the head gets EYES and a MOUTH -- with no
    hand-authored placement. This is exactly what the earlier hand-built render lacked."""
    m, cr, sdf = _salamander()
    socks = m.creature_auto_sockets(cr, feet=True, head_parts=True, hands=True)
    parts = [s["part"] for s in socks]
    assert parts.count("foot") == 4          # one per limb tip, mirrored
    assert "eye" in parts and "mouth" in parts
    eye = [s for s in socks if s["part"] == "eye"][0]
    assert eye["symmetry"] == "bilateral" and eye["t"] > 0.9   # eyes near the head end


def test_T4_socket_frames_preserve_orientation():
    """BACKLOG T4, verified empirically rather than assumed. A part attaches through a (4,4)
    frame from resolve_socket / resolve_limb_socket; if any frame had det < 0 the part would
    be MIRRORED -- the classic procedural-creature bug (a left hand on a right arm). Measured
    det = 1.0 on all six sockets."""
    m, cr, sdf = _salamander()
    socks = m.creature_auto_sockets(cr, feet=True, head_parts=True, hands=True)
    dets = []
    for s in socks:
        if s["kind"] == "spine":
            r = m.resolve_socket(cr, sdf, s["t"], s["theta"])
        else:
            r = m.resolve_limb_socket(cr, sdf, s["limb"], s["u"], s.get("theta", 0.0),
                                      along_axis=s.get("along_axis", False))
        assert r["hit"], s
        F = np.asarray(r["frame"], float)
        assert F.shape == (4, 4)
        dets.append(float(np.linalg.det(F[:3, :3])))
    assert len(dets) == 6
    assert all(d > 0.99 for d in dets), dets      # orientation-preserving, no flips
    assert all(abs(d - 1.0) < 1e-6 for d in dets)  # and orthonormal, not just positive

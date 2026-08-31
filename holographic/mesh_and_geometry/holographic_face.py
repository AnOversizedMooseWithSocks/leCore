"""A face as a LANDMARK GRAPH plus parts -- procedural, no scans, no learned basis.

BACKLOG O3 of the creature/humanoid overhaul, and the item the avatar attempt actually
needed: `humanoid`'s head is a smooth blob with no eye sockets, nose, mouth, jaw or brow, so
the only "face" available was two spheres stuck on a bump.

SOTA CHECK (searched 2026-08-16), and it validates this approach by the literature's OWN
admission rather than by our preference:
  * FLAME / DECA and the 3DMM line are the standard, and OmniFaceRig (2026) states their
    limit plainly: they are "bound to a FIXED MESH TOPOLOGY and expression basis defined at
    SCAN-COLLECTION TIME, and they primarily assume ADULT HUMAN ANATOMY: applying them to a
    novel asset with arbitrary topology, STYLIZED PROPORTIONS, or NON-HUMAN FEATURES often
    requires re-fitting a new mesh into the parametric basis (which can lose
    character-specific identity) or leads to unstable fits." An engine whose job is
    salamanders and centaurs is exactly that novel asset. FLAME is the wrong tool HERE --
    not a worse tool generally.
  * SCULPTOR (TOG 2022) contributes the structural idea worth stealing: SKELETON CONSISTENCY.
    Inner skeletal structure (mandible, maxilla) correlates with outer appearance, so a face
    built bone-first is anatomically coherent by construction. SCULPTOR learns that
    correlation from CT scans (the LUCY dataset); we get the same DISCIPLINE for free by
    placing landmarks on a skull proportion model and growing outward -- which is what
    tissue_fields already does ("grown OUTWARD from bone").
  * FaceMaker (procedural parametric face generator, no scans) is prior art for the
    slider-driven direction.

WHAT THIS IS AND IS NOT: a stylised, characterful, ANATOMICALLY-ORGANISED face driven by
proportion sliders. It is NOT a likeness of any individual and NOT a reconstruction from a
photograph -- there is no fitting step, because there is no scan basis to fit into. Anyone
wanting identity capture wants a 3DMM and should be told so.

RULE-0 AUDIT (2026-08-16): `skull`, `jaw` and a face-landmark schema all returned nothing --
genuine gap. REUSED and not rebuilt: part_library / build_part (eye, mouth, ear, horn already
ship), resolve_socket (marches the field outward and returns a surface point + frame, which
is exactly landmark placement), and holographic_blendbasis (O2) for expression as LOCAL
correctives rather than a learned expression basis.

KEPT NEGATIVE: proportions here follow classical artistic canon (eye line at head mid-height,
five eye-widths across, etc.), which is a DRAWING convention, not a measured anthropometric
distribution. It produces plausible faces; it does not produce a population.
"""

import numpy as np

# Classical head canon, as fractions of head height (t, measured from chin=0 to crown=1) and
# of head width (u, 0 = midline). These are DRAWING conventions -- see the module's kept
# negative -- chosen because they are legible and adjustable, not because they are measured.
FACE_CANON = {
    "chin":        (0.00, 0.00),
    "jaw_l":       (0.16, 0.38),
    "mouth":       (0.22, 0.00),
    "nose_tip":    (0.42, 0.00),
    "nose_l":      (0.40, 0.10),
    "cheek_l":     (0.46, 0.42),
    "eye_l":       (0.55, 0.22),
    "brow_l":      (0.63, 0.24),
    "ear_l":       (0.52, 0.50),
    "temple_l":    (0.68, 0.44),
    "crown":       (1.00, 0.00),
}

MIRRORED = tuple(k for k in FACE_CANON if k.endswith("_l"))


def face_landmarks(head_centre, head_height, head_width, depth=None, proportions=None):
    """Skull-canon landmark positions for a head, as {name: (3,) position}.

    `proportions` overrides any canon entry, which is the slider surface: raising `eye_l`'s
    first component lifts the eye line, widening its second sets the interocular distance.
    Left-suffixed landmarks are MIRRORED to `_r` automatically, so bilateral symmetry is
    structural rather than something a caller can forget.

    Depth (how far forward a feature sits) defaults to a fraction of width and is applied
    along +Z, so the face looks down +Z with +Y up."""
    c = np.asarray(head_centre, float)
    H = float(head_height)
    W = float(head_width)
    D = float(depth) if depth is not None else 0.62 * W
    canon = dict(FACE_CANON)
    if proportions:
        canon.update(proportions)
    # how far forward each feature protrudes, as a fraction of D -- a nose is the front of
    # the face, an ear is at the side and set BACK, a temple is behind the eye line
    forward = {"nose_tip": 1.00, "nose_l": 0.86, "mouth": 0.80, "chin": 0.74,
               "eye_l": 0.62, "brow_l": 0.66, "cheek_l": 0.52, "jaw_l": 0.44,
               "temple_l": 0.30, "ear_l": 0.10, "crown": 0.34}
    out = {}
    for name, (t, u) in canon.items():
        z = forward.get(name, 0.5) * D
        y = c[1] + (float(t) - 0.5) * H
        x = c[0] + float(u) * W
        out[name] = np.array([x, y, c[2] + z], float)
        if name in MIRRORED:
            out[name[:-2] + "_r"] = np.array([c[0] - float(u) * W, y, c[2] + z], float)
    return out


def face_part_graph(landmarks, scale=1.0):
    """Which PART goes at which landmark, with its size -- the rigblock assignment for a face.

    Returns a list of {landmark, part, size, mirror} that a caller feeds to build_part and
    place at the landmark. Kept as DATA rather than code so a non-human face (four eyes, no
    nose) is an edit to a list, not a new code path -- which is the whole reason this is a
    part graph and not a fixed template."""
    plan = [("eye_l", "eye", 0.115), ("eye_r", "eye", 0.115),
            ("mouth", "mouth", 0.26), ("ear_l", "ear", 0.20), ("ear_r", "ear", 0.20)]
    out = []
    for lm, part, size in plan:
        if lm in landmarks:
            out.append({"landmark": lm, "part": part, "size": float(size) * float(scale),
                        "position": landmarks[lm]})
    return out


def expression(landmarks, name, amount=1.0):
    """An EXPRESSION as per-landmark displacements -- the input to O2's local correctives.

    Not a learned basis: each expression names which landmarks move and where, so a new one
    is a dict entry. Returns {landmark: (3,) delta}. Amount scales linearly and may be
    extrapolated past 1, which is what animators do."""
    a = float(amount)
    table = {
        # the reference photo's expression: nose wrinkled, one eye squeezed, lip raised
        "disgust": {"nose_tip": (0.0, 0.05, -0.02), "nose_l": (0.0, 0.06, 0.0),
                    "nose_r": (0.0, 0.06, 0.0), "brow_l": (0.0, -0.06, 0.0),
                    "brow_r": (0.0, -0.02, 0.0), "eye_l": (0.0, -0.03, 0.0),
                    "mouth": (0.0, 0.05, 0.0), "cheek_l": (0.0, 0.05, 0.0)},
        "smile":   {"mouth": (0.0, 0.03, 0.0), "cheek_l": (0.02, 0.05, 0.0),
                    "cheek_r": (-0.02, 0.05, 0.0), "eye_l": (0.0, -0.015, 0.0),
                    "eye_r": (0.0, -0.015, 0.0)},
        "surprise": {"brow_l": (0.0, 0.07, 0.0), "brow_r": (0.0, 0.07, 0.0),
                     "mouth": (0.0, -0.06, 0.0), "chin": (0.0, -0.05, 0.0)},
    }
    if name not in table:
        raise ValueError("unknown expression %r; have %s" % (name, sorted(table)))
    return {k: np.asarray(v, float) * a for k, v in table[name].items()
            if k in landmarks}


def _selftest():
    """Regression trap: the canon must be bilaterally symmetric, ordered head-to-chin, and
    expressions must actually move the landmarks they name."""
    lm = face_landmarks((0.0, 1.6, 0.0), 0.24, 0.10)
    # bilateral symmetry is STRUCTURAL: mirrored pairs differ only in x, and by equal amounts
    for k in ("eye", "brow", "ear", "cheek", "jaw", "temple"):
        L, R = lm[k + "_l"], lm[k + "_r"]
        assert abs(L[0] + R[0]) < 1e-12, (k, L, R)          # x mirrored about the midline
        assert np.allclose(L[1:], R[1:]), (k, L, R)          # same height and depth
    # anatomical ordering: crown above brow above eye above nose above mouth above chin
    order = ["crown", "brow_l", "eye_l", "nose_tip", "mouth", "chin"]
    ys = [lm[k][1] for k in order]
    assert ys == sorted(ys, reverse=True), list(zip(order, ys))
    # the nose is the frontmost feature; the ear is the most set-back
    zs = {k: lm[k][2] for k in ("nose_tip", "eye_l", "ear_l")}
    assert zs["nose_tip"] > zs["eye_l"] > zs["ear_l"], zs

    parts = face_part_graph(lm)
    assert {p["part"] for p in parts} == {"eye", "mouth", "ear"}
    assert sum(p["part"] == "eye" for p in parts) == 2       # bilateral, not one cyclops eye

    d = expression(lm, "disgust", 1.0)
    assert d and all(np.linalg.norm(v) > 0 for v in d.values())
    half = expression(lm, "disgust", 0.5)
    assert np.allclose(half["nose_tip"], d["nose_tip"] * 0.5)   # linear and extrapolable
    print("OK: holographic_face -- %d landmarks (bilateral, anatomically ordered), %d parts "
          "placed, expression 'disgust' moves %d landmarks linearly"
          % (len(lm), len(parts), len(d)))


if __name__ == "__main__":
    _selftest()

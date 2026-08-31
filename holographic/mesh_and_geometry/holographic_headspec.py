"""head_spec: a skull skeleton FROM PARAMETERS, whose every parameter vector is a head.

TWO LESSONS FROM THE SALAMANDER, applied.

(1) USE A SPEC GENERATOR. The salamander only worked once it stopped being hand-authored --
spine_profile and quadruped_spec turned "type coordinates until it looks right" into "state
proportions". Every head in this session was 26 hand-tuned segments with magic numbers,
re-typed from scratch each attempt, which is why each attempt regressed in a different place.
quadruped_spec exists; head_spec did not. This is it.

(2) THE ANATOMY IS IN THE SKELETON, NOT THE RENDER. The salamander read as a salamander when
the SKELETON had a tapering tail and sprawling limbs -- no amount of material or lighting work
fixed it before that, and lighting work on the head has likewise been the wrong lever.

AND THE FIX FOR THE FAILURE THAT KEEPS RECURRING. Three separate fitting formulations
converged nicely and produced meaningless geometry -- 9 capsules at 3.34x baseline that looked
like blobs, and a 44%-better fit that was a PANCAKE. The diagnosis each time was
identifiability: the objective had a null space and the optimiser found it.

Proving an objective identifiable is hard. CONSTRAINING THE PARAMETERISATION SO THAT EVERY
POINT IN IT IS A HEAD IS TRACTABLE, and it is strictly stronger: a pancake stops being a
reachable solution at all, so no objective -- however badly designed -- can return one. That
is what this module is for, and lean/LeCoreHeadSpec.lean proves it holds for EVERY parameter
vector in range rather than for the ones that happen to get tested.

THE INVARIANTS, which are what "is a head" means operationally:
    crown > brow > eye > nose_tip > mouth > chin        (vertical ordering)
    nose_tip is the frontmost point                     (a face has a nose)
    every left landmark mirrors its right               (bilateral symmetry)
    height/width stays in the human range               (no pancakes, no needles)
They hold for all params in PARAM_RANGE by construction: each is built as a POSITIVE OFFSET
from the one below it, so the ordering cannot invert no matter what the optimiser does.

RULE-0 AUDIT (2026-08-16): head_spec returned nothing; quadruped_spec, spine_profile and
face_landmarks all ship and the first two are the pattern this follows. face_landmarks is
REUSED for the canon; this module turns that canon into the SKELETON SEGMENTS a convolution
field consumes, which is the step that was missing.

KEPT NEGATIVE: valid does not mean flattering, and it certainly does not mean anyone's
likeness. The guarantee is that every parameter vector produces something anatomically
well-formed -- not that any of them is the person in the photograph.
"""

import numpy as np

# (lo, hi) for every parameter. The proof is quantified over this box, so widening it
# without re-checking the proof is a real change, not a tweak.
PARAM_RANGE = {
    "skull_w":    (0.045, 0.075),   # half-width of the cranium
    # ALSO FRACTIONS OF skull_w, and for the same reason: independent heights and widths let
    # 7/400 vectors through as a wide-short head, i.e. a pancake -- the exact failure that
    # made an unconstrained fit converge to a lozenge. Tying vertical extents to the width
    # bounds the aspect ratio BY CONSTRUCTION, so the pancake is not merely rejected, it is
    # UNREACHABLE.
    "skull_h_f":  (0.95,  1.55),    # crown height above eye line, x skull_w
    "skull_d":    (0.045, 0.080),   # occiput depth behind the eye line
    # FRACTIONS OF nose_proj, not free lengths. The first version made these independent
    # distances and 292/400 random vectors then put the CHIN or the BROW in front of the
    # NOSE -- an anatomically impossible head that the parameterisation happily expressed.
    # Coupling them to nose_proj makes "the nose is frontmost" structural instead of a rule
    # to be checked, which is the whole design principle of this module.
    "brow_frac":  (0.05,  0.30),    # brow overhang as a fraction of nose projection
    "face_h_f":   (1.45,  2.15),    # eye line to chin, x skull_w
    "jaw_w":      (0.035, 0.065),   # half-width at the gonial angle
    "cheek":      (0.018, 0.036),   # zygomatic prominence
    "nose_len":   (0.030, 0.060),   # nasion to tip
    "nose_proj":  (0.030, 0.065),   # how far the tip sits ahead of the eye line
    "nose_w":     (0.010, 0.024),
    "lip_h":      (0.010, 0.026),
    "chin_frac":  (0.35,  0.88),    # chin projection as a fraction of nose projection
    "neck_r":     (0.035, 0.060),
}

DEFAULT = {k: 0.5 * (a + b) for k, (a, b) in PARAM_RANGE.items()}


def _derive(p):
    """Expand the fraction parameters into the absolute lengths the builder uses. Doing this
    in ONE place is what keeps the invariants structural rather than scattered."""
    p = dict(p)
    p["skull_h"] = p["skull_h_f"] * p["skull_w"]
    p["face_h"] = p["face_h_f"] * p["skull_w"]
    return p


def clamp_params(params=None):
    """Clamp to PARAM_RANGE. This is the gate that makes the invariants unconditional: an
    optimiser handed a clamped parameter vector CANNOT leave the manifold of heads, which is
    why a fit can no longer converge to a pancake."""
    p = dict(DEFAULT)
    if params:
        p.update({k: float(v) for k, v in params.items() if k in PARAM_RANGE})
    for k, (a, b) in PARAM_RANGE.items():
        p[k] = float(np.clip(p[k], a, b))
    return _derive(p)


def head_landmarks(params=None):
    """Anatomical landmarks from parameters, built so the ORDERING CANNOT INVERT.

    Every vertical position is the one below it PLUS A POSITIVE quantity, and every parameter
    is positive by PARAM_RANGE. That is the whole trick: the ordering invariant is structural,
    not checked afterwards. Origin is the eye line, +Y up, +Z forward."""
    p = clamp_params(params)
    eye_y = 0.0
    chin_y = eye_y - p["face_h"]
    mouth_y = chin_y + 0.34 * p["face_h"]                 # strictly above the chin
    nose_y = mouth_y + 0.30 * p["face_h"]                 # strictly above the mouth
    brow_y = eye_y + 0.26 * p["skull_h"]                  # strictly above the eyes
    crown_y = brow_y + 0.74 * p["skull_h"]                # strictly above the brow
    nose_z = p["nose_proj"]
    return {
        "crown":    np.array([0.0, crown_y, 0.30 * p["skull_d"]]),
        "brow_l":   np.array([0.62 * p["skull_w"], brow_y,
                              nose_z * (0.55 + 0.40 * p["brow_frac"])]),
        "eye_l":    np.array([0.46 * p["skull_w"], eye_y, 0.52 * nose_z]),
        "cheek_l":  np.array([0.95 * p["skull_w"], eye_y - 0.22 * p["face_h"], 0.42 * nose_z]),
        "nose_tip": np.array([0.0, nose_y, nose_z]),
        "mouth":    np.array([0.0, mouth_y, 0.72 * nose_z]),
        "chin":     np.array([0.0, chin_y, nose_z * p["chin_frac"]]),
        "jaw_l":    np.array([p["jaw_w"], chin_y + 0.40 * p["face_h"], 0.30 * nose_z]),
        "ear_l":    np.array([1.02 * p["skull_w"], eye_y + 0.05 * p["skull_h"],
                              -0.30 * p["skull_d"]]),
        "occiput":  np.array([0.0, eye_y + 0.30 * p["skull_h"], -p["skull_d"]]),
    }


def head_spec(params=None):
    """Parameters -> the segment list a convolution field consumes.

    Returns (segments, landmarks). Segments are (a, b, radius, aniso), exactly the shape
    convolution_field wants, so a head becomes `convolution_field(head_spec()[0], scalis=True)`
    instead of forty lines of hand-typed coordinates."""
    p = clamp_params(params)
    lm = head_landmarks(p)
    S = []

    def seg(a, b, r, an=(1., 1., 1.)):
        S.append((tuple(np.asarray(a, float)), tuple(np.asarray(b, float)), float(r), an))

    W, Hh, Dd = p["skull_w"], p["skull_h"], p["skull_d"]
    crown, occ = lm["crown"], lm["occiput"]
    # cranium: parietal, occiput, frontal -- three masses, not one sphere (the salamander
    # lesson about anatomy living in the skeleton)
    seg([0, crown[1] - 0.28 * Hh, -0.30 * Dd], [0, crown[1] - 0.55 * Hh, 0.10 * Dd],
        0.92 * W, (1., 0.98, 0.90))
    seg([0, occ[1], -0.72 * Dd], [0, occ[1] - 0.55 * Hh, -0.62 * Dd], 0.88 * W, (1., 1.02, 0.88))
    seg([0, lm["brow_l"][1] + 0.28 * Hh, 0.34 * lm["nose_tip"][2]],
        [0, lm["brow_l"][1] + 0.04 * Hh, 0.52 * lm["nose_tip"][2]], 0.78 * W, (1., 0.94, 0.84))
    for sx in (1, -1):
        seg([sx * 0.78 * W, lm["brow_l"][1], -0.16 * Dd],
            [sx * 0.86 * W, lm["eye_l"][1], 0.08 * Dd], 0.50 * W, (1., 1., 0.76))
        seg([sx * 0.12 * W, lm["brow_l"][1], lm["brow_l"][2]],
            [sx * lm["brow_l"][0], lm["brow_l"][1], lm["brow_l"][2] - 0.18 * p["nose_proj"]],
            0.30 * p["brow_frac"] * p["nose_proj"] + 0.010, (1., 0.55, 1.0))
        seg([sx * lm["cheek_l"][0], lm["cheek_l"][1] + 0.10 * p["face_h"], 0.14 * lm["nose_tip"][2]],
            [sx * lm["cheek_l"][0] * 0.82, lm["cheek_l"][1], lm["cheek_l"][2]],
            p["cheek"], (1., 0.82, 1.0))
        seg([sx * lm["cheek_l"][0] * 0.82, lm["cheek_l"][1], lm["cheek_l"][2]],
            [sx * 0.42 * W, lm["nose_tip"][1] - 0.10 * p["face_h"], 0.70 * lm["nose_tip"][2]],
            0.80 * p["cheek"], (1., 0.82, 1.0))
        seg([sx * 0.96 * W, lm["eye_l"][1] - 0.04 * p["face_h"], -0.12 * Dd],
            [sx * lm["jaw_l"][0], lm["jaw_l"][1], lm["jaw_l"][2]], 0.42 * p["cheek"] + 0.008)
        seg([sx * lm["jaw_l"][0], lm["jaw_l"][1], lm["jaw_l"][2]],
            [sx * 0.42 * p["jaw_w"], lm["chin"][1], 0.88 * lm["chin"][2]],
            0.40 * p["cheek"] + 0.008)
        seg([sx * lm["ear_l"][0], lm["ear_l"][1], lm["ear_l"][2]],
            [sx * (lm["ear_l"][0] + 0.10 * W), lm["ear_l"][1] - 0.16 * Hh, lm["ear_l"][2] - 0.06 * Dd],
            0.30 * p["cheek"], (1., 1.35, 0.32))
    # maxilla, chin bar, mentum
    seg([0, lm["mouth"][1] + 0.14 * p["face_h"], 0.86 * lm["nose_tip"][2]],
        [0, lm["mouth"][1] - 0.04 * p["face_h"], 0.80 * lm["nose_tip"][2]],
        0.52 * W, (1., 0.76, 0.95))
    seg([0.42 * p["jaw_w"], lm["chin"][1], 0.88 * lm["chin"][2]],
        [-0.42 * p["jaw_w"], lm["chin"][1], 0.88 * lm["chin"][2]], 0.36 * p["cheek"] + 0.008)
    # nose: root -> bridge -> tip, then alae
    seg([0, lm["brow_l"][1] - 0.10 * Hh, 0.60 * lm["nose_tip"][2]],
        [0, lm["nose_tip"][1] + 0.30 * p["nose_len"], 0.88 * lm["nose_tip"][2]],
        0.72 * p["nose_w"], (1., 1., 0.70))
    seg([0, lm["nose_tip"][1] + 0.30 * p["nose_len"], 0.88 * lm["nose_tip"][2]],
        [0, lm["nose_tip"][1], lm["nose_tip"][2]], 0.80 * p["nose_w"], (1., 1., 0.78))
    for sx in (1, -1):
        seg([sx * 0.25 * p["nose_w"], lm["nose_tip"][1] - 0.10 * p["nose_len"],
             0.92 * lm["nose_tip"][2]],
            [sx * 1.05 * p["nose_w"], lm["nose_tip"][1] - 0.14 * p["nose_len"],
             0.84 * lm["nose_tip"][2]], 0.58 * p["nose_w"])
    # lips
    seg([0.9 * p["lip_h"], lm["mouth"][1], 0.80 * lm["nose_tip"][2]],
        [-0.9 * p["lip_h"], lm["mouth"][1], 0.80 * lm["nose_tip"][2]],
        0.55 * p["lip_h"], (1., 0.58, 1.0))
    # neck + shoulders
    seg([0, lm["chin"][1] - 0.16 * p["face_h"], 0.05 * Dd],
        [0, lm["chin"][1] - 0.80 * p["face_h"], -0.05 * Dd], p["neck_r"])
    for sx in (1, -1):
        seg([0, lm["chin"][1] - 0.84 * p["face_h"], 0.0],
            [sx * 2.6 * W, lm["chin"][1] - 1.05 * p["face_h"], 0.0], 1.35 * p["neck_r"])
    return S, lm


def check_invariants(params=None):
    """Do the anatomical invariants hold? Returns {name: bool}.

    This is the RUNTIME mirror of lean/LeCoreHeadSpec.lean, which proves the same statements
    for EVERY parameter vector in PARAM_RANGE rather than for the ones a test happens to try."""
    lm = head_landmarks(params)
    order = ["crown", "brow_l", "eye_l", "nose_tip", "mouth", "chin"]
    ys = [float(lm[k][1]) for k in order]
    zs = {k: float(lm[k][2]) for k in lm}
    return {
        "vertical_order": all(ys[i] > ys[i + 1] for i in range(len(ys) - 1)),
        "nose_is_frontmost": all(zs["nose_tip"] >= v for v in zs.values()),
        "ear_behind_eye": zs["ear_l"] < zs["eye_l"],
        "bilateral": True,                       # only left landmarks are stored; -x mirrors
        "proportion_sane": 1.00 < (float(lm["crown"][1]) - float(lm["chin"][1])) / (
            2 * clamp_params(params)["skull_w"]) < 2.4,
    }


def _selftest():
    """Regression trap, and it is the point of the module: the invariants must hold for
    RANDOM parameter vectors across the whole range, not just the default."""
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(400):
        p = {k: rng.uniform(a, b) for k, (a, b) in PARAM_RANGE.items()}
        inv = check_invariants(p)
        if not all(inv.values()):
            bad += 1
            if bad == 1:
                print("FAILING:", {k: v for k, v in inv.items() if not v}, p)
    assert bad == 0, "%d/400 parameter vectors violated the invariants" % bad
    # clamping must rescue even absurd input -- this is what makes a pancake unreachable
    inv = check_invariants({"face_h": -99.0, "skull_w": 1e6, "skull_h": 0.0})
    assert all(inv.values()), inv
    S, lm = head_spec()
    assert len(S) > 20 and all(len(s) == 4 for s in S)
    print("OK: holographic_headspec -- 400/400 random parameter vectors satisfy every "
          "anatomical invariant, absurd input is clamped back into the manifold, and "
          "head_spec() yields %d segments" % len(S))


if __name__ == "__main__":
    _selftest()

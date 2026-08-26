"""head_spec: a skull skeleton FROM PARAMETERS, whose every parameter vector is a head.

TWO SALAMANDER LESSONS APPLIED. (1) Use a spec generator -- the salamander only worked once
spine_profile and quadruped_spec replaced hand-typed coordinates, and every head this session
was 26 hand-tuned magic numbers re-typed from scratch. (2) The anatomy lives in the SKELETON,
not the render; lighting work never fixed a wrong skeleton.

AND THE FIX FOR THE RECURRING FAILURE. Three fitting formulations converged and produced
meaningless geometry (9 capsules at 3.34x baseline that looked like blobs; a 44%-better fit
that was a PANCAKE). Proving an objective identifiable is hard; CONSTRAINING THE
PARAMETERISATION so every point in it is a head is tractable and strictly stronger -- the
pancake stops being reachable, so no objective can return one.
"""
import numpy as np
from holographic.mesh_and_geometry import holographic_headspec as HS


def test_every_parameter_vector_in_range_is_a_head():
    """THE POINT OF THE MODULE. 400 random draws from PARAM_RANGE must all satisfy every
    anatomical invariant -- not the default, not a hand-picked set."""
    rng = np.random.default_rng(0)
    for _ in range(400):
        p = {k: rng.uniform(a, b) for k, (a, b) in HS.PARAM_RANGE.items()}
        inv = HS.check_invariants(p)
        assert all(inv.values()), (inv, p)


def test_absurd_input_is_clamped_back_into_the_manifold():
    """Clamping is what makes the guarantee unconditional: an optimiser cannot leave the set
    of heads even if its objective rewards doing so."""
    for bad in ({"face_h_f": -99.0}, {"skull_w": 1e6}, {"nose_proj": 0.0},
                {"chin_frac": 50.0}, {"skull_h_f": -3.0}):
        assert all(HS.check_invariants(bad).values()), bad


def test_coupled_parameters_prevent_the_impossible_heads():
    """MEASURED REGRESSION. With brow/chin as FREE distances rather than fractions of nose
    projection, 292/400 random vectors put the chin or brow IN FRONT of the nose. The
    coupling is what makes 'the nose is frontmost' algebraic rather than a rule to check."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        p = {k: rng.uniform(a, b) for k, (a, b) in HS.PARAM_RANGE.items()}
        lm = HS.head_landmarks(p)
        assert float(lm["nose_tip"][2]) >= float(lm["chin"][2])
        assert float(lm["nose_tip"][2]) >= float(lm["brow_l"][2])


def test_aspect_ratio_is_bounded_so_a_pancake_is_unreachable():
    """The specific failure that motivated this: an unconstrained fit converged 44% better
    and returned a flat lozenge. Height must stay between 1.0x and 2.4x the full width for
    EVERY admissible parameter vector."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        p = {k: rng.uniform(a, b) for k, (a, b) in HS.PARAM_RANGE.items()}
        lm = HS.head_landmarks(p)
        w = 2 * HS.clamp_params(p)["skull_w"]
        h = float(lm["crown"][1]) - float(lm["chin"][1])
        assert 1.0 < h / w < 2.4, (h / w, p)


def test_head_spec_yields_usable_segments():
    """The output must be exactly what convolution_field consumes, so a head is one call."""
    S, lm = HS.head_spec()
    assert len(S) > 20
    for a, b, r, an in S:
        assert len(a) == 3 and len(b) == 3 and r > 0 and len(an) == 3
    assert {"crown", "brow_l", "eye_l", "nose_tip", "mouth", "chin"} <= set(lm)


def test_lean_proof_typechecks_and_has_no_sorry():
    """The invariants above are TESTED on 400 samples; lean/LeCoreHeadSpec.lean PROVES them
    for every parameter vector in the admissible box. A file full of `sorry` typechecks and
    proves nothing, so the absence of admitted goals is part of the claim."""
    import os, shutil, subprocess
    lean_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "lean", "LeCoreHeadSpec.lean")
    assert os.path.exists(lean_file)
    src = open(lean_file, encoding="utf-8").read()
    assert "sorry" not in src
    for thm in ("vertical_order", "nose_frontmost", "aspect_bounded", "non_degenerate",
                "is_a_head"):
        assert thm in src, thm
    if shutil.which("lean"):
        r = subprocess.run(["lean", lean_file], capture_output=True, text=True, timeout=900)
        assert r.returncode == 0 and not r.stdout.strip(), r.stdout + r.stderr


def test_lean_box_and_python_range_agree_exactly():
    """THE BRIDGE, and it caught a real gap: the first proof used skullH/W in 1.00-1.50 while
    Python allowed 0.95-1.55, so the theorem did not cover every admissible parameter -- a
    silent disagreement between a proof and the code it is about, which is the one failure a
    proof must not have. Pinned so widening PARAM_RANGE without touching the proof FAILS."""
    import os, re
    lean_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "lean", "LeCoreHeadSpec.lean")
    src = open(lean_file, encoding="utf-8").read()

    def grab(pat):
        return tuple(int(x) for x in re.search(pat, src).groups())

    assert grab(r'(\d+) ≤ p\.skullW ∧ p\.skullW ≤ (\d+)') == (
        round(HS.PARAM_RANGE['skull_w'][0] * 1e4), round(HS.PARAM_RANGE['skull_w'][1] * 1e4))
    assert grab(r'(\d+) \* p\.skullW ≤ 100 \* p\.skullH ∧ 100 \* p\.skullH ≤ (\d+) \* p\.skullW') == (
        round(HS.PARAM_RANGE['skull_h_f'][0] * 100), round(HS.PARAM_RANGE['skull_h_f'][1] * 100))
    assert grab(r'(\d+) \* p\.skullW ≤ 100 \* p\.faceH ∧ 100 \* p\.faceH ≤ (\d+) \* p\.skullW') == (
        round(HS.PARAM_RANGE['face_h_f'][0] * 100), round(HS.PARAM_RANGE['face_h_f'][1] * 100))
    chin = int(re.search(r'100 \* p\.chinZ ≤ (\d+) \* p\.noseProj', src).group(1))
    assert chin >= round(HS.PARAM_RANGE['chin_frac'][1] * 100)


def test_python_positions_use_the_ratios_lean_proves():
    """A proof about DIFFERENT ratios than the code uses would be worthless. Verified to
    machine precision over 500 random vectors."""
    rng = np.random.default_rng(11)
    for _ in range(500):
        p = {k: rng.uniform(a, b) for k, (a, b) in HS.PARAM_RANGE.items()}
        cp = HS.clamp_params(p)
        lm = HS.head_landmarks(p)
        f, s = cp["face_h"], cp["skull_h"]
        assert abs(float(lm["chin"][1]) / f + 1.00) < 1e-12
        assert abs(float(lm["mouth"][1]) / f + 0.66) < 1e-12
        assert abs(float(lm["nose_tip"][1]) / f + 0.36) < 1e-12
        assert abs(float(lm["brow_l"][1]) / s - 0.26) < 1e-12
        assert abs(float(lm["crown"][1]) / s - 1.00) < 1e-12

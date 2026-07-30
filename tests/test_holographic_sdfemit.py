

def test_the_two_emitters_are_executed_and_agree():
    """W4: sdfemit's header warns that two tables for one concept will disagree -- and the GLSL half was
    never executed, so 'they agree' was narrative. Both now RUN (GLSL via a g++ vec3 shim, C via cc) and are
    compared to the Python tree across the node zoo, including a rotation (mat3) and a compound."""
    import numpy as np
    import lecore
    import holographic.mesh_and_geometry.holographic_sdf as S
    from holographic.mesh_and_geometry.holographic_sdfemit import (emitters_agree, validate_glsl,
                                                                   GLSL_AGREEMENT_TOL, SdfEmitError)
    m = lecore.UnifiedMind(dim=64, seed=0)
    P = np.random.default_rng(0).uniform(-2.0, 2.0, (120, 3))

    trees = {
        "sphere": S.sphere(1.0),
        "box": S.box(0.8, 0.5, 0.6),
        "smooth_union": S.sphere(0.7).smooth_union(S.box(0.5, 0.3, 0.6), 0.25),
        "rotated": S.box(0.5, 0.3, 0.6).rotate((0.0, 1.0, 0.0), 0.7),
        "compound": S.sphere(0.7).translate((0.4, 0.0, -0.2)).smooth_union(
            S.box(0.5, 0.3, 0.6).rotate((0.0, 1.0, 0.0), 0.7), 0.25).scale(1.3),
    }
    for name, tree in trees.items():
        r = emitters_agree(tree, P)
        assert r["agree"], (name, r["why"], r["worst"])
        # the C dialect is held to EXACTNESS; only the 32-bit shader gets a tolerance
        assert r["c_f64"]["max_abs_diff"] <= 1e-12, (name, r["c_f64"])
        assert r["glsl"]["max_abs_diff"] <= GLSL_AGREEMENT_TOL, (name, r["glsl"])

    # the measured envelope: a rotation is the WORST case because to_glsl writes 6-significant-digit
    # literals (cos(0.7) -> 0.764842, itself 1.9e-7 off) on top of GLSL's 32-bit float.
    worst = emitters_agree(trees["compound"], P)["worst"]
    assert 1e-8 < worst < 1e-5, worst

    # the shim REFUSES what it cannot model rather than comparing wrongly
    try:
        validate_glsl("(this is not a tree)", P)
        raised = False
    except Exception:
        raised = True
    assert raised

    assert m.sdf_emitters_agree(trees["sphere"])["agree"]

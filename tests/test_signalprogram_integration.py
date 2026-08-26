"""SignalProgram (I1) driven through the mind, and composed with the rest of the null layer.

The module's own selftest pins its internal contracts. These tests pin the things that only break at the
SEAM: that the faculty hands back a working program, that the gates it applies are the same ones
holographic_honesty exposes separately (so a hand-rolled screen and a SignalProgram screen agree), and that
the refusal path survives the trip through UnifiedMind instead of being smoothed into a best-effort answer.
"""
import numpy as np

import lecore
from holographic.agents_and_reasoning.holographic_honesty import bh_fdr, split_half


def _fixture(n=1200, k=12, seed=0):
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((n, k))
    real_target = np.sign(states[:, 0]) * np.abs(rng.standard_normal(n))
    noise_target = rng.standard_normal(n)
    return states, real_target, noise_target


def _battery(mind, seed=0, dim=256):
    prog = mind.signal_program(dim=dim, seed=seed)
    prog.add_check("real", lambda s: s[:, 0])
    for j in range(1, 12):
        prog.add_check("noise_%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
    return prog


def test_the_faculty_returns_a_working_program_and_finds_the_real_check_alone():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    states, real_target, _ = _fixture()
    rep = _battery(mind).screen(states, real_target)
    assert rep["passed"] == ["real"], rep["reason"]
    assert not rep["refused"]
    assert rep["family_size"] == 12
    assert rep["clusters"] == 1


def test_a_battery_of_nulls_refuses_and_the_refusal_survives_the_faculty():
    """The refusal must arrive as a populated, quotable result -- not an exception, and not a quiet fallback to
    whichever check happened to look best. This is the property most likely to be 'helpfully' smoothed away by
    a later change, so it is pinned through the mind rather than only in the module."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    states, _, noise_target = _fixture()
    rep = _battery(mind).screen(states, noise_target)
    assert rep["refused"] is True
    assert rep["passed"] == []
    assert rep["clusters"] == 0
    assert "not conditionable" in rep["reason"]
    # the raw numbers ARE present -- refusal is a verdict on them, not a refusal to compute them.
    assert len(rep["checks"]) == 12
    assert all("p" in row and "effect" in row for row in rep["checks"])


def test_the_gates_are_the_same_ones_the_honesty_module_exposes_separately():
    """A hand-rolled screen using bh_fdr + split_half directly must reach the IDENTICAL verdict. If these ever
    disagree, one of the two paths has drifted and the 'gates inside the loop' claim is hollow."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    states, real_target, _ = _fixture()
    prog = _battery(mind)
    rep = prog.screen(states, real_target)

    # reproduce the screen by hand from the same definitions
    names = ["real"] + ["noise_%d" % j for j in range(1, 12)]
    contribs = [np.sign(states[:, 0]) * real_target] + \
               [np.sign(states[:, j]) * real_target for j in range(1, 12)]
    pvals = [row["p"] for row in rep["checks"]]
    rejected, _ = bh_fdr(pvals, alpha=0.1, dependent=True)
    manual = [n for n, c, r in zip(names, contribs, rejected)
              if bool(r) and split_half(c)["passed"]]
    assert manual == rep["passed"], (manual, rep["passed"])


def test_a_decaying_edge_clears_fdr_and_is_still_rejected_for_failing_replication():
    """The structural claim, end to end: passing one gate is not enough. An edge that is genuine in the first
    half and absent in the second produces a tiny p-value and clears FDR, and must still not pass."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    states, real_target, _ = _fixture()
    rng = np.random.default_rng(99)
    half = real_target.copy()
    half[len(half) // 2:] = rng.standard_normal(len(half) - len(half) // 2)
    prog = mind.signal_program(dim=256, seed=0)
    prog.add_check("decays", lambda s: s[:, 0])
    row = prog.screen(states, half)["checks"][0]
    assert row["p"] < 0.05
    assert row["fdr_rejected"]
    assert not row["split_half_passed"]
    assert not row["passed"]


def test_correlated_checks_collapse_to_one_finding():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    states, real_target, _ = _fixture()
    prog = mind.signal_program(dim=256, seed=0)
    prog.add_check("a", lambda s: s[:, 0])
    prog.add_check("b", lambda s: s[:, 0] * 3.0)          # same decisions, different scale
    prog.add_check("c", lambda s: -s[:, 0])               # anti-correlated: still the SAME finding
    rep = prog.screen(states, real_target)
    assert rep["clusters"] == 1, rep["cluster_members"]


def test_program_vector_fingerprints_the_battery_and_recovers_a_signature():
    mind = lecore.UnifiedMind(dim=1024, seed=0)
    states, _, _ = _fixture()
    prog = _battery(mind, dim=1024)
    pv, sigs = prog.program_vector(states)
    assert pv.shape == (1024,)
    from holographic.agents_and_reasoning.holographic_ai import unbind
    rec = unbind(pv, prog._roles()[0])
    cos = float(np.dot(rec, sigs["real"]) / (np.linalg.norm(rec) * np.linalg.norm(sigs["real"]) + 1e-12))
    floor = max(abs(float(np.dot(rec, sigs["noise_%d" % j]) /
                          (np.linalg.norm(rec) * np.linalg.norm(sigs["noise_%d" % j]) + 1e-12)))
                for j in range(1, 12))
    assert cos > 2 * floor, (cos, floor)
    # the same battery on the same states fingerprints identically -- the release-comparison use needs this.
    pv2, _ = _battery(lecore.UnifiedMind(dim=1024, seed=0), dim=1024).program_vector(states)
    assert np.allclose(pv, pv2), "the same battery on the same states must fingerprint identically"


def test_timing_evidence_is_reported_but_never_asserted():
    """KEPT NEGATIVE, and a lesson about pinning the wrong thing. The backlog predicted 'one bundled sweep, not
    N loops' would be a speed win; measurement refutes it at large K (0.02-0.47x over 5 repeats at K=200) and
    calls it a wash at small K (0.93-1.64x). An earlier version of this test PINNED `speedup < 1.0` -- and it
    flaked on the very next clean extract, where the identical code won at 1.19x. Wall-clock is not
    deterministic and cannot be a contract in a determinism-constrained suite. So the contract is only that
    the evidence is present and well-formed; the finding lives in the docstring and NOTES where it belongs."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    states, real_target, _ = _fixture()
    timing = _battery(mind).screen(states, real_target)["timing"]
    assert timing["batched_s"] > 0 and timing["loop_s"] > 0
    assert timing["speedup"] == timing["speedup"]                # reported and finite, not NaN


def test_duplicate_check_names_are_refused_so_the_family_size_stays_honest():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    prog = mind.signal_program(dim=256, seed=0)
    prog.add_check("x", lambda s: s[:, 0])
    try:
        prog.add_check("x", lambda s: s[:, 1])
        raise AssertionError("expected ValueError on a duplicate check name")
    except ValueError as e:
        assert "duplicate check name" in str(e)


def test_committee_seated_from_the_screen_holds_its_own_gates_on_fresh_data():
    """E2 end to end through the mind: screen on one draw, seat the committee, evaluate on a fresh draw. The
    committee must pass ITS OWN gates out-of-sample, and its membership is cluster representatives only."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(10)
    tr = rng.standard_normal((1500, 8))
    tgt = np.sign(tr[:, 0] + tr[:, 1] + tr[:, 2]) * np.abs(rng.standard_normal(1500))
    prog = mind.signal_program(dim=256, seed=0)
    for j in range(3):
        prog.add_check("real_%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
    prog.add_check("dup_of_0", lambda s: 2.0 * s[:, 0])          # correlated: must NOT get a second seat
    for j in range(3, 8):
        prog.add_check("noise_%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
    rep = prog.screen(tr, tgt)
    com = prog.build_committee(rep)
    assert len(com) == rep["clusters"]                           # one seat per independent finding
    te = rng.standard_normal((1500, 8))
    tgt_te = np.sign(te[:, 0] + te[:, 1] + te[:, 2]) * np.abs(rng.standard_normal(1500))
    ev = com.evaluate(te, tgt_te)
    assert ev["passed"], ev["verdict"]
    assert ev["n_votes"] + ev["n_abstain"] == 1500


def test_empty_committee_refusal_propagates_and_never_falls_back():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(11)
    states = rng.standard_normal((1000, 6))
    noise_t = rng.standard_normal(1000)
    prog = mind.signal_program(dim=256, seed=0)
    for j in range(6):
        prog.add_check("n%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
    rep = prog.screen(states, noise_t)
    assert rep["refused"]
    com = prog.build_committee(rep)
    assert len(com) == 0
    try:
        com.decide(states)
        raise AssertionError("expected the refusal to propagate")
    except ValueError as e:
        assert "refusal was the result" in str(e)

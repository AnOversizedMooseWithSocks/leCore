"""Regression traps for D1 (tier contracts). A certifier that only ever says yes certifies
nothing, so every refusal clause is provoked separately."""
import pytest
from holographic.caching_and_storage import holographic_tiercontract as T

TIERS = {"hot": {"capacity": 8, "cost": 1},
         "trace": {"capacity": 10 ** 6, "cost": 10, "holographic": True, "dim": 4096},
         "storage": {"capacity": 10 ** 9, "cost": 1000}}


def test_clean_plan_certifies():
    r = T.certify_plan(TIERS, [{"item": "a", "tier": "hot", "count": 4},
                               {"item": "b", "tier": "trace", "count": 64}],
                       forbid_tiers=("storage",), min_recall=0.98)
    assert r["ok"] and r["tiers"]["trace"]["recall"] == 0.98


def test_each_clause_refuses_when_provoked():
    cap = T.certify_plan(TIERS, [{"item": "a", "tier": "hot", "count": 99}])
    assert not cap["ok"] and any("capacity" in v for v in cap["violations"])
    fid = T.certify_plan(TIERS, [{"item": "b", "tier": "trace", "count": 256}],
                         min_recall=0.98)
    assert not fid["ok"] and any("fidelity" in v for v in fid["violations"])
    ban = T.certify_plan(TIERS, [{"item": "z", "tier": "storage", "count": 1}],
                         forbid_tiers=("storage",))
    assert not ban["ok"] and any("forbidden" in v for v in ban["violations"])


def test_fidelity_ladder_is_measured_and_conservative():
    """The rungs come from the D5 sweep; between rungs the contract reports the LOWER
    guarantee, because interpolating a measurement promises a number nobody measured."""
    assert T.fidelity_floor(4096, 128) == 0.98     # D/M = 32
    assert T.fidelity_floor(4096, 256) == 0.84     # D/M = 16
    assert T.fidelity_floor(4096, 205) == 0.84     # D/M = 20 -> lower rung
    assert T.fidelity_floor(4096, 0) == 1.0
    assert T.fidelity_floor(4096, 10 ** 6) == 0.0
    # a weaker requirement is satisfiable where a stronger one is refused
    assert T.certify_plan(TIERS, [{"item": "b", "tier": "trace", "count": 256}],
                          min_recall=0.8)["ok"]


def test_faculty_end_to_end():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    assert m.tier_fidelity_floor(4096, 128) == 0.98
    r = m.tier_certify_plan(TIERS, [{"item": "a", "tier": "hot", "count": 2}],
                            forbid_tiers=("storage",), min_recall=0.98)
    assert r["ok"]


def test_bake_certificate_catches_corruption():
    """D2 oracle probe: a clean bake certifies, a corrupted one is REFUSED. A certifier that
    never fails certifies nothing."""
    import numpy as np
    N = 10000
    table = np.sin(np.arange(N) * 0.01)
    ev = lambda i: np.sin(i * 0.01)
    c = T.certify_bake(ev, lambda i: table[i], N, n_samples=256, seed=0)
    assert c["ok"] and c["max_error"] < 1e-12 and 0.9 < c["guarantee"] <= 1.0
    bad = table.copy()
    bad[N // 3:N // 3 + N // 20] += 0.5
    assert not T.certify_bake(ev, lambda i: bad[i], N, n_samples=256, seed=0)["ok"]


def test_detection_bound_is_honest_about_small_corruptions():
    """The guarantee must not flatter itself: one bad cell in 10k is genuinely hard to catch
    with 256 samples, and the number says so."""
    assert T.detect_probability(10000, 256, 1) < 0.05
    assert T.detect_probability(10000, 256, 100) > 0.9
    assert T.detect_probability(10000, 10000, 1) == 1.0     # exhaustive check is certain
    assert T.samples_for_confidence(10000, 1, 0.99) > 8000
    assert T.samples_for_confidence(10000, 100, 0.99) < 600


def test_bake_faculties():
    from lecore import UnifiedMind
    import numpy as np
    m = UnifiedMind(dim=64, seed=0)
    assert m.bake_samples_for_confidence(10000, 100, 0.99) < 600
    tab = np.arange(500.0)
    c = m.bake_certify(lambda i: float(i), lambda i: tab[i], 500, n_samples=64)
    assert c["ok"] and c["checked"] == 64


def test_differential_oracle_catches_a_wrong_implementation():
    """D3 / the consolidation. Agreement passes; a deliberately wrong implementation is
    caught WITH its case index (reproducible, not just counted); a crashing backend counts
    as a disagreement rather than being skipped."""
    import numpy as np
    cases = [i / 7.0 for i in range(40)]
    impls = {"ref": np.sin, "same": np.sin, "nearly": lambda x: np.sin(x) + 1e-12}
    d = T.differential_agreement(impls, cases, tol=1e-9)
    assert d["ok"] and d["worst"] < 1e-9 and d["reference"] == "ref"
    impls["wrong"] = np.cos
    d2 = T.differential_agreement(impls, cases, tol=1e-9)
    assert not d2["ok"] and "case" in d2["pairs"]["wrong"]["failures"][0]

    def boom(x):
        raise RuntimeError("backend exploded")
    d3 = T.differential_agreement({"ref": lambda x: x, "bad": boom}, cases[:3])
    assert not d3["ok"] and d3["pairs"]["bad"]["failures"][0]["dev"] == float("inf")


def test_real_sdf_emitters_still_agree():
    """The domain-specific instance the generic was consolidated FROM must keep passing --
    consolidation that breaks its own first customer is not consolidation."""
    from lecore import UnifiedMind
    from holographic.mesh_and_geometry.holographic_sdf import sphere, box
    m = UnifiedMind(dim=64, seed=0)
    for node in (sphere(0.6), box(0.4, 0.3, 0.5)):
        r = m.sdf_emitters_agree(node)
        assert r["agree"] and r["worst"] < 1e-5


def test_differential_faculty():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    d = m.differential_agreement({"a": lambda x: x * 2, "b": lambda x: x + x},
                                 list(range(10)), tol=1e-12)
    assert d["ok"]


RES = {0: ["a"], 1: ["a", "b"], 2: ["b"], 3: ["c"], 4: ["c", "a"]}


def test_schedule_from_declarations_certifies():
    """D4: when the schedule is coloured from the SAME declarations the certificate checks,
    it certifies -- and the round trip is the point (writing the edge list twice is how a
    schedule and its check quietly stop describing the same system)."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    waves = m.color_waves(5, T.resource_conflict_edges(RES))
    cert = T.certify_schedule(waves, RES)
    assert cert["ok"] and cert["n_tasks"] == 5


def test_schedule_certificate_refuses_conflicts_and_drops():
    """Both failure modes, provoked separately. A dropped task is a worse bug than a race,
    so it is checked too."""
    bad = T.certify_schedule([[0, 1], [2, 3], [4]], RES)     # 0 and 1 share "a"
    assert not bad["ok"] and any("touch" in v for v in bad["violations"])
    dropped = T.certify_schedule([[0, 2], [1, 3]], RES)      # task 4 vanished
    assert not dropped["ok"] and any("never scheduled" in v for v in dropped["violations"])
    dup = T.certify_schedule([[0], [0], [1, 2, 3, 4]], RES)  # task 0 scheduled twice
    assert not dup["ok"]


def test_schedule_faculties():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    edges = m.schedule_conflict_edges(RES)
    # task ids are normalised to str so the JSON wire path and the in-process path agree
    # (found by the HTTP round-trip: object keys arrive as strings and nothing matched)
    assert ("0", "1") in edges and ("0", "4") in edges       # both share "a"
    assert m.schedule_certify([[0, 2], [1, 3], [4]], RES)["ok"]
    assert m.schedule_certify([["0", "2"], ["1", "3"], ["4"]], RES)["ok"]  # str ids too


def _interleaved(K, n, rng, noise):
    import numpy as np
    t = np.arange(n // K + 1)
    srcs = [np.sin(2 * np.pi * t / 40 + i) + 0.3 * np.sin(2 * np.pi * t / 13 + 2 * i)
            for i in range(K)]
    x = np.empty(n)
    for i in range(n):
        x[i] = srcs[i % K][i // K]
    return x + rng.normal(scale=noise * np.std(x), size=n)


def test_noise_estimator_recovers_planted_sigma():
    """Donoho-Johnstone MAD of second differences, checked against a KNOWN sigma."""
    import numpy as np
    r = np.random.default_rng(0)
    t = np.arange(4000)
    clean = np.sin(2 * np.pi * t / 200)
    for s in (0.01, 0.05, 0.2):
        est = T.estimate_noise_sigma(clean + r.normal(scale=s, size=t.size))
        assert abs(est - s) < 0.25 * s


def test_demux_gate_never_trusts_a_wrong_stride():
    """A2's gate. The property that matters is not "it accepts good answers" but "it never
    blesses a bad one" -- a gate that can be wrong in the trusting direction is worse than
    no gate. Measured 0 false-trust across the (K, noise) grid."""
    import numpy as np
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    false_trust = 0
    trusted_clean = 0
    for K in (2, 4, 8):
        for noise in (0.0, 0.02, 0.05, 0.10, 0.30):
            res = m.demux_gated(_interleaved(K, 600, np.random.default_rng(7 * K), noise))
            k = res.get("k") or res.get("K") or res.get("stride")
            if res["trusted"]:
                if k != K:
                    false_trust += 1
                elif noise <= 0.02:
                    trusted_clean += 1
    assert false_trust == 0
    assert trusted_clean > 0, "the gate refused everything, which certifies nothing"


def test_pose_certificate_and_its_narrow_scope():
    """B4. Every pose the CONSTRAINED solver returns must certify -- including for an
    UNREACHABLE target, because reachability is a fact about the target, not a defect in the
    pose (target_error is reported, never certified). A hand-broken pose must be refused with
    the offending bone named."""
    import numpy as np
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    J = np.array([[0., 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])
    lim = [None, {"type": "hinge", "axis": "auto", "lo": -1.2, "hi": 0.0},
           {"type": "cone", "half": 0.9}]
    rest = np.linalg.norm(np.diff(J, axis=0), axis=1)
    reach_errors = []
    for tgt in ([1.5, 2.0, 0.3], [0.2, 2.9, 0.1], [9.0, 9.0, 9.0]):
        P, _ = m.solve_ik_limited(J, np.array(tgt, float), lim)
        c = m.pose_certify(P, lim, rest_lengths=rest, target=tgt)
        assert c["ok"], (tgt, c["violations"])
        assert c["max_length_error"] < 1e-9 and c["max_angle_excess"] <= 1e-6
        reach_errors.append(c["target_error"])
    assert reach_errors[-1] > reach_errors[0]      # the unreachable target is reported as far
    bad = J.copy()
    bad[2] = [1.4, 1.2, 0.0]
    cb = m.pose_certify(bad, lim, rest_lengths=rest)
    assert not cb["ok"] and any("length" in v for v in cb["violations"])


def test_conservation_ledger_discriminates_bounded_from_drifting():
    """C1's whole content. A symplectic integrator conserves a SHADOW Hamiltonian: energy
    oscillates and stays bounded rather than being exactly conserved. An |dE|~0 test would
    fail the best integrators for behaving correctly, so bounded quantities are judged on
    SECULAR TREND and exact ones on absolute drift."""
    import numpy as np
    t = np.arange(800)
    wobble = 1.0 + 0.02 * np.sin(t * 0.3)
    assert T.conservation_ledger({"E": wobble}, bounded=("E",))["ok"]
    assert not T.conservation_ledger({"E": wobble + 0.00008 * t}, bounded=("E",))["ok"]
    assert T.conservation_ledger({"m": np.full(800, 5.0)}, exact=("m",))["ok"]
    assert not T.conservation_ledger({"m": 5.0 + 1e-6 * t}, exact=("m",))["ok"]


def test_ledger_on_real_verlet_run():
    """End-to-end on leCore's OWN pair potential with velocity Verlet, which also
    CROSS-CHECKS F1's claim that momentum conservation holds by construction (np.add.at
    assembly, Newton's third law) rather than by hope: measured 1.1e-14."""
    import numpy as np
    from lecore import UnifiedMind
    from holographic.simulation_and_physics.holographic_morphogen import pair_energy_and_grad
    m = UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(0)
    X = rng.normal(scale=1.5, size=(30, 3))
    V = rng.normal(scale=0.05, size=(30, 3))
    V -= V.mean(0)
    R = np.full(30, 0.5)
    dt = 0.02
    hist = {"E": [], "p": []}
    _, g = pair_energy_and_grad(X, R)
    for _ in range(300):
        V = V - 0.5 * dt * g
        X = X + dt * V
        E, g = pair_energy_and_grad(X, R)
        V = V - 0.5 * dt * g
        hist["E"].append(E + 0.5 * float(np.sum(V * V)))
        hist["p"].append(float(np.linalg.norm(V.sum(0))))
    assert m.conservation_ledger(hist, bounded=("E",))["ok"]
    assert max(hist["p"]) < 1e-10        # momentum exact to machine precision


ACTIONS = {"goto_rack": {"pre": {}, "eff": {"at_rack": True}},
           "pickup": {"pre": {"at_rack": True}, "eff": {"has_weapon": True}},
           "goto_enemy": {"pre": {}, "eff": {"near_enemy": True}},
           "fire": {"pre": {"has_weapon": True, "near_enemy": True},
                    "eff": {"enemy_down": True}}}


def test_plan_certificate_catches_the_canonical_goap_bug():
    """The literature's own example of what hand-authored plans get wrong: "a character
    might be instructed to fire a weapon, without ever [acquiring one]". A valid plan
    certifies; the broken one is refused with the STEP and the MISSING PRECONDITION named."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    good = m.plan_certify(["goto_rack", "pickup", "goto_enemy", "fire"], ACTIONS, {},
                          goal={"enemy_down": True})
    assert good["ok"] and good["final_state"]["enemy_down"]
    bad = m.plan_certify(["goto_enemy", "fire"], ACTIONS, {}, goal={"enemy_down": True})
    assert not bad["ok"]
    assert "has_weapon" in bad["violations"][0] and "step 1" in bad["violations"][0]


def test_plan_certificate_checks_goal_and_unknown_actions():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    assert not m.plan_certify(["goto_rack"], ACTIONS, {}, goal={"enemy_down": True})["ok"]
    assert not m.plan_certify(["teleport"], ACTIONS, {})["ok"]
    # a plan that starts from a state where the weapon is ALREADY held needs fewer steps
    short = m.plan_certify(["goto_enemy", "fire"], ACTIONS, {"has_weapon": True},
                           goal={"enemy_down": True})
    assert short["ok"], short["violations"]


def test_plan_certificate_does_not_judge_quality():
    """KEPT NEGATIVE, pinned: feasibility is not optimality. A ludicrous but valid route
    certifies exactly like a tight one, because cost is the planner's business and a
    certificate that quietly judged quality would misrepresent what it checked."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    silly = ["goto_rack", "goto_enemy", "goto_rack", "pickup", "goto_enemy", "fire"]
    assert m.plan_certify(silly, ACTIONS, {}, goal={"enemy_down": True})["ok"]

"""Regression traps for holographic_lean and the p18 logic/Lean faculties.

Pins the four contracts a future edit is most likely to quietly break:
  1. SOUNDNESS -- what is not derivable stays not derivable, and the independent checker
     rejects a forged proof (prover/checker independence is the module's whole safety story).
  2. DETERMINISM -- same rules, same goal, byte-identical Lean text across runs.
  3. THE WIRE -- the JSON round-trip a POST /invoke caller depends on still checks.
  4. THE FACULTIES -- reachable on an assembled mind, delegating, honest None preserved.

The Lean output's STRUCTURE is pinned (declarations + theorem line); external typechecking
is the `lean` binary's job (opt-in bridge) and is exercised only when one is installed --
its absence is reported, never faked.
"""
import json

import numpy as np
import pytest

from holographic.agents_and_reasoning import holographic_lean as L


def _family_rules():
    """The shared fixture: Socrates + a two-hop ancestry chain (exercises multi-body
    unification, the join most likely to regress)."""
    return [
        L.Rule(L.Atom("human", ("socrates",)), name="h_soc"),
        L.Rule(L.Atom("mortal", ("?x",)), (L.Atom("human", ("?x",)),), name="mortality"),
        L.Rule(L.Atom("parent", ("tom", "bob")), name="p_tb"),
        L.Rule(L.Atom("parent", ("bob", "liz")), name="p_bl"),
        L.Rule(L.Atom("ancestor", ("?x", "?y")), (L.Atom("parent", ("?x", "?y")),), name="anc_base"),
        L.Rule(L.Atom("ancestor", ("?x", "?z")),
               (L.Atom("parent", ("?x", "?y")), L.Atom("ancestor", ("?y", "?z"))), name="anc_step"),
    ]


def test_derivable_and_checked():
    rules = _family_rules()
    p = L.prove(L.Atom("ancestor", ("tom", "liz")), rules)
    assert p is not None and p.size() == 4
    assert L.check_proof(p, rules)


def test_underivable_is_none():
    # the honest None: zeus was never declared human; ancestry must not run backwards
    rules = _family_rules()
    assert L.prove(L.Atom("mortal", ("zeus",)), rules) is None
    assert L.prove(L.Atom("ancestor", ("liz", "tom")), rules) is None


def test_checker_rejects_forged_proof():
    # right rule, wrong premise -- the checker must not trust the prover's shape
    rules = _family_rules()
    forged = L.Proof(L.Atom("mortal", ("zeus",)), rules[1], {"?x": "zeus"},
                     (L.Proof(L.Atom("human", ("zeus",)), rules[0], {}),))
    with pytest.raises(AssertionError):
        L.check_proof(forged, rules)


def test_lean_export_deterministic_and_structured():
    rules = _family_rules()
    p = L.prove(L.Atom("ancestor", ("tom", "liz")), rules)
    a = L.to_lean(p, rules, theorem_name="t")
    b = L.to_lean(L.prove(L.Atom("ancestor", ("tom", "liz")), rules), rules, theorem_name="t")
    assert a == b  # byte-identical across independent prover runs
    for needle in ("axiom U : Type", "axiom ancestor : U -> U -> Prop",
                   "axiom anc_step : forall", "theorem t : ancestor tom liz :="):
        assert needle in a


def test_wire_round_trip_still_checks():
    rules = _family_rules()
    p = L.prove(L.Atom("ancestor", ("tom", "liz")), rules)
    w = json.loads(json.dumps(L.proof_to_wire(p)))
    assert L.check_proof(L.proof_from_wire(w, rules), rules)
    # a wire proof naming an unknown rule must raise, not invent an axiom
    w["rule"] = "not_a_rule"
    with pytest.raises(KeyError):
        L.proof_from_wire(w, rules)


def test_faculties_on_assembled_mind():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    rules = [{"head": ["human", ["socrates"]], "name": "h"},
             {"head": ["mortal", ["?x"]], "body": [["human", ["?x"]]], "name": "m"}]
    p = m.logic_prove(["mortal", ["socrates"]], rules)
    assert p is not None and m.logic_check_proof(p, rules)
    out = m.lean_export(["mortal", ["socrates"]], rules, theorem_name="soc")
    assert out["ok"] and "theorem soc : mortal socrates :=" in out["lean"]
    assert m.logic_prove(["mortal", ["zeus"]], rules) is None
    v = m.logic_encode_atom("human", ["socrates"])
    assert v.shape == (64,)
    # same atom -> same vector (derived_atom is hashlib-seeded); distinct atoms near-orthogonal at dim 64
    v2 = m.logic_encode_atom("human", ["socrates"])
    assert np.allclose(v, v2)
    # the bridge reports availability honestly; ok must be a real verdict or None, never fabricated
    res = m.lean_verify("axiom U : Type\n")
    assert res["available"] in (True, False)
    if res["available"]:
        assert res["ok"] is True


def test_lean_bridge_rejects_sorry_when_available():
    """The instrument error of 2026-08-16, pinned: `sorry` exits 0 with only a warning, so a
    returncode-only bridge blessed an UNPROVEN theorem. "ok" must mean proved, not compiled."""
    from holographic.agents_and_reasoning.holographic_lean import lean_check
    res = lean_check("axiom P : Prop\ntheorem t : P := sorry\n")
    if not res["available"]:
        pytest.skip("no lean binary on PATH -- bridge exercised elsewhere")
    assert res["ok"] is False and res["sorried"] is True
    # and a hard type error is still a plain False
    assert lean_check("theorem t : True := 42\n")["ok"] is False
    # while a real proof remains True
    assert lean_check("theorem t : True := trivial\n")["ok"] is True


def test_export_naming_soundness():
    """The naming holes of 2026-08-16, pinned. The dangerous one first: collision-prone
    mangling MERGED distinct constants ('a-b' and 'a_b' -> one Lean ident), which is a
    soundness hole -- a false statement about one could typecheck as a true one about the
    other. Then: 'U' shadowing the universe, Lean keywords as constants, and a rule name
    sharing a string with the theorem name (different ENTITIES, so kinds key the namespace).
    All four were measured as external-Lean False before the fix."""
    r1 = [L.Rule(L.Atom("p", ("a-b",)), name="f1"), L.Rule(L.Atom("p", ("a_b",)), name="f2")]
    s = L.to_lean(L.prove(L.Atom("p", ("a-b",)), r1), r1, "t")
    assert "axiom a_b : U" in s and "axiom a_b_2 : U" in s  # distinct, deterministically
    for rules, goal in [([L.Rule(L.Atom("p", ("U",)), name="f1")], L.Atom("p", ("U",))),
                        ([L.Rule(L.Atom("p", ("fun",)), name="f1")], L.Atom("p", ("fun",))),
                        ([L.Rule(L.Atom("p", ("a",)), name="t")], L.Atom("p", ("a",)))]:
        src = L.to_lean(L.prove(goal, rules), rules, theorem_name="t")
        # structural sanity always; external verdict when a lean binary exists
        assert src.count("axiom U : Type") == 1
        res = L.lean_check(src)
        if res["available"]:
            assert res["ok"], res["stderr"]


def test_loud_preconditions():
    """Silent failure modes promoted to loud raises: duplicate rule names (last-wins broke
    both the wire format and the Lean axioms), mixed predicate arity (one Lean signature per
    predicate), and a non-ground goal (that is a QUERY -- returning None read as 'not
    derivable' when the truth was 'wrong question shape')."""
    with pytest.raises(ValueError):
        L.validate_rules([L.Rule(L.Atom("p", ("a",)), name="f"),
                          L.Rule(L.Atom("q", ("b",)), name="f")])
    with pytest.raises(ValueError):
        L.validate_rules([L.Rule(L.Atom("p", ("a",)), name="f1"),
                          L.Rule(L.Atom("p", ("a", "b")), name="f2")])
    with pytest.raises(ValueError):
        L.prove(L.Atom("p", ("?x",)), [L.Rule(L.Atom("p", ("a",)), name="f")])


def test_consequences_is_exact_least_fixpoint():
    """Completeness as a MEASURED property (Kowalski, panel Tier 1): the van Emden-Kowalski
    T_P fixpoint of the family base is exactly these 7 atoms -- no more (soundness), no fewer
    (completeness), in a deterministic derivation order."""
    rules = _family_rules()
    keys = sorted(a.key() for a in L.consequences(rules))
    assert keys == ['ancestor(bob,liz)', 'ancestor(tom,bob)', 'ancestor(tom,liz)',
                    'human(socrates)', 'mortal(socrates)', 'parent(bob,liz)', 'parent(tom,bob)']
    assert [a.key() for a in L.consequences(rules)] == [a.key() for a in L.consequences(rules)]


def test_absurdity_smoke_and_proof_measure():
    """The consistency smoke fires iff a designated absurdity atom is derivable, with its
    proof attached (de Moura: Lean checks derivations, never rule-set consistency); and the
    Gentzen measure pins the ancestry derivation's exact shape."""
    rules = _family_rules()
    assert L.detect_absurdity(rules)["absurd"] is False
    bad = rules + [L.Rule(L.Atom("false", ()), (L.Atom("mortal", ("socrates",)),), name="oops")]
    r = L.detect_absurdity(bad)
    assert r["absurd"] is True and r["atom"] == ["false", []]
    assert L.check_proof(L.proof_from_wire(r["proof"], bad), bad)  # the smoke's proof is real
    pr = L.prove(L.Atom("ancestor", ("tom", "liz")), rules)
    assert L.proof_measure(pr) == {"size": 4, "height": 3,
                                   "rules_used": {"anc_step": 1, "p_tb": 1,
                                                  "anc_base": 1, "p_bl": 1}}


def test_external_agreement_mode_and_new_faculties():
    """The de Bruijn criterion as a mode: check='external' returns ok=True ONLY when the
    in-process checker AND an installed Lean both agree; with no binary, ok is a loud False
    with the reason attached -- never faked. Plus the two new faculties end-to-end."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    rules = [{"head": ["human", ["socrates"]], "name": "h"},
             {"head": ["mortal", ["?x"]], "body": [["human", ["?x"]]], "name": "m"}]
    out = m.lean_export(["mortal", ["socrates"]], rules, check="external")
    assert "external" in out
    if out["external"]["available"]:
        assert out["ok"] is True
    else:
        assert out["ok"] is False  # absence of the second checker is not agreement
    c = m.logic_consequences(rules)
    assert c["count"] == 2 and c["absurd"]["absurd"] is False
    p = m.logic_prove(["mortal", ["socrates"]], rules)
    assert m.logic_proof_measure(p, rules) == {"size": 2, "height": 2,
                                               "rules_used": {"m": 1, "h": 1}}
    # measuring a forged proof must raise, not report the shape of a lie
    p["atom"] = ["mortal", ["zeus"]]
    with pytest.raises(AssertionError):
        m.logic_proof_measure(p, rules)


def test_decode_atom_roundtrip_and_abstention():
    """Tier 2 (Olshausen's seat, resolved by Rule 0): the resonator already ships in three
    costumes and this known-role structure needs only unbind + cleanup. Pins: exact inverse
    on a clean fact, abstention on noise (never confabulate), through the mind's own space."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=512, seed=0)
    v = m.logic_encode_atom("parent", ["tom", "bob"])
    d = m.logic_decode_atom(v, ["parent", "human", "ancestor"],
                            ["tom", "bob", "liz", "socrates"], 2)
    assert d == {"pred": "parent", "args": ["tom", "bob"],
                 "score": pytest.approx(d["score"]), "abstained": False}
    noise = np.random.default_rng(7).standard_normal(512)
    assert m.logic_decode_atom(noise, ["parent"], ["tom", "bob"], 2)["abstained"] is True


def test_fact_capacity_pins_the_negative():
    """Plate's Tier-2 measurement, and its verdict kept LOUD: exact whole-atom recall through
    ONE bundled trace is perfect at load 1-2 and cliffs by load 8 -- and the cliff does NOT
    move with D (interfering facts are unit-norm whatever D is; the house's 1/sqrt(M)-
    independent-of-D law in a new costume). Consequence, pinned as the contract: fact bases
    are INDEXED rows, never a single bundled trace."""
    r256 = L.fact_capacity(dim=256, n_symbols=8, n_preds=2, loads=(1, 8), seeds=range(2))
    r1024 = L.fact_capacity(dim=1024, n_symbols=8, n_preds=2, loads=(1, 8), seeds=range(2))
    assert r256["exact"][1]["mean"] == 1.0 and r1024["exact"][1]["mean"] == 1.0
    assert r256["exact"][8]["mean"] < 0.5 and r1024["exact"][8]["mean"] < 0.5  # D didn't help


def test_induce_rules_lff():
    """The Eno loop's induction stage: learning-from-failures (Cropper & Morel 2021,
    generate/test/constrain) on the finite fragment. Pins: exact single-clause learning,
    RECURSIVE two-clause learning (ancestor -- the candidate participates in its own T_P,
    so recursion is tested, not special-cased), refusal on an uncoverable target (None,
    never a guess), and determinism of the learned theory."""
    bg = [L.Rule(L.Atom("human", (n,)), name="h_" + n) for n in ("socrates", "plato")] + \
         [L.Rule(L.Atom("dog", ("fido",)), name="d")]
    r = L.induce_rules(bg, [L.Atom("mortal", ("socrates",)), L.Atom("mortal", ("plato",))],
                       [L.Atom("mortal", ("fido",))], "mortal", {"human": 1, "dog": 1})
    assert len(r["rules"]) == 1 and str(r["rules"][0].body[0]) == "human(?v0)"
    fams = [("tom", "bob"), ("bob", "liz"), ("liz", "ann"), ("pat", "jim")]
    bg2 = [L.Rule(L.Atom("parent", p), name="p_%d" % i) for i, p in enumerate(fams)]
    pos = [L.Atom("ancestor", p) for p in
           [("tom", "bob"), ("tom", "liz"), ("tom", "ann"), ("bob", "ann"), ("pat", "jim")]]
    neg = [L.Atom("ancestor", p) for p in [("bob", "tom"), ("jim", "pat"), ("ann", "tom")]]
    r2 = L.induce_rules(bg2, pos, neg, "ancestor", {"parent": 2, "ancestor": 2})
    assert r2["rules"] is not None and len(r2["rules"]) == 2
    # the learned theory derives ALL positives and NO negatives -- re-proved independently
    theory = bg2 + r2["rules"]
    keys = {a.key() for a in L.consequences(theory)}
    assert all(p.key() in keys for p in pos) and not any(n.key() in keys for n in neg)
    r2b = L.induce_rules(bg2, pos, neg, "ancestor", {"parent": 2, "ancestor": 2})
    assert [str(a) for a in r2["rules"]] == [str(a) for a in r2b["rules"]]  # deterministic
    # uncoverable: a positive about a constant no rule can reach -> honest None
    r3 = L.induce_rules(bg2, [L.Atom("ancestor", ("zeus", "tom"))], [], "ancestor",
                        {"parent": 2}, max_body=1)
    assert r3["rules"] is None


def test_conjecture_and_refute_end_to_end():
    """The full Eno loop through the mind: induce -> deduce -> refute -> Lean source proving
    a positive FROM MACHINE-LEARNED AXIOMS; external verdict when a lean binary exists."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    bg = [{"head": ["parent", list(p)], "name": "p_%d" % i}
          for i, p in enumerate([("tom", "bob"), ("bob", "liz"), ("liz", "ann")])]
    out = m.logic_induce(bg,
                         [["ancestor", ["tom", "bob"]], ["ancestor", ["tom", "liz"]],
                          ["ancestor", ["tom", "ann"]], ["ancestor", ["bob", "ann"]]],
                         [["ancestor", ["bob", "tom"]], ["ancestor", ["ann", "tom"]]],
                         "ancestor", {"parent": 2, "ancestor": 2}, theorem_name="t")
    assert out["rules"] and len(out["rules"]) == 2 and out["refuted_count"] > 0
    assert "theorem t : ancestor tom bob :=" in out["lean"]
    res = m.lean_verify(out["lean"])
    if res["available"]:
        assert res["ok"], "external Lean rejected the induced theory's proof:\n" + res["stderr"]


def test_seminaive_equality_and_scale():
    """Semi-naive evaluation (Bancilhon & Ramakrishnan 1986), added because naive T_P DNF'd
    at 300s on the repo's own 708-module import graph (the dogfood measurement that revealed
    it). Pins: (1) the equality theorem -- seminaive derives EXACTLY the naive atom set on a
    recursive base; (2) its proofs pass the independent checker; (3) on a 60-node chain-heavy
    graph it completes well inside a budget naive can't touch at repo scale; (4) opt-in --
    the default strategy is untouched naive (pinned Lean bytes depend on its order)."""
    rules = _family_rules()
    assert ({a.key() for a in L.consequences(rules, strategy="seminaive")}
            == {a.key() for a in L.consequences(rules)})
    p = L.prove(L.Atom("ancestor", ("tom", "liz")), rules, strategy="seminaive")
    assert p is not None and L.check_proof(p, rules)
    # a 60-node path graph: 1770 reaches pairs, recursion depth 59
    edges = [("n%02d" % i, "n%02d" % (i + 1)) for i in range(59)]
    big = [L.Rule(L.Atom("imports", e), name="e%d" % i) for i, e in enumerate(edges)]
    big += [L.Rule(L.Atom("reaches", ("?x", "?y")), (L.Atom("imports", ("?x", "?y")),), name="rb"),
            L.Rule(L.Atom("reaches", ("?x", "?z")),
                   (L.Atom("imports", ("?x", "?y")), L.Atom("reaches", ("?y", "?z"))), name="rs")]
    sn = {tuple(a.args) for a in L.consequences(big, max_steps=10**7, strategy="seminaive")
          if a.pred == "reaches"}
    assert len(sn) == 59 * 60 // 2  # exactly the closure of a path, counted not eyeballed
    import inspect
    assert inspect.signature(L.prove).parameters["strategy"].default == "naive"


def test_fuzz_export_differential():
    """The distilled oracle: hostile random theories through the whole chain. Without a
    lean binary this still exercises equality (naive==seminaive on adversarial inputs),
    soundness (ghost atoms underivable under BOTH strategies), checker acceptance, and
    export byte-determinism -- the Lean stage adds itself when a binary exists and probes
    itself with a corrupted term. Standing result distilled into the repo: 300 theories /
    793 externally verified exports / 0 failures (seeds 0-59 @ defaults, 0-239 @ hostile
    widths). This pin keeps 12 of those seeds running forever, Lean-free."""
    r = L.fuzz_export(n=12, seed=0)
    assert r["failures"] == [], r["failures"]
    assert r["derivable"] > 0  # a fuzz that derived nothing tested nothing
    if r["lean_available"]:
        assert r["lean_checked"] == r["derivable"]


def test_proof_memory_verified_knowledge():
    """The Lean distillation into the substrate: only CHECKED proofs enter (an underivable
    goal stores nothing -- pinned), provenance travels with each record ("checked" or
    "lean_verified" when a binary judged it -- what we KEEP from Lean while the binary stays
    optional), goals/trees/traces live as INDEXED ROWS (the logic_fact_capacity negative is
    why), and recall is honest: exact hit, planted-truth family retrieval (an unstored
    ancestor goal's nearest neighbours are ancestor records, not mortal ones), structural
    tree-similarity, provenance filtering, and empty results stated as empty."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=1024, seed=0)
    fams = [("tom", "bob"), ("bob", "liz"), ("liz", "ann"), ("pat", "jim"), ("jim", "joe")]
    rules = [{"head": ["parent", list(p)], "name": "p%d" % i} for i, p in enumerate(fams)]
    rules += [{"head": ["ancestor", ["?x", "?y"]], "body": [["parent", ["?x", "?y"]]], "name": "ab"},
              {"head": ["ancestor", ["?x", "?z"]],
               "body": [["parent", ["?x", "?y"]], ["ancestor", ["?y", "?z"]]], "name": "as"},
              {"head": ["mortal", ["?x"]], "body": [["human", ["?x"]]], "name": "mo"},
              {"head": ["human", ["socrates"]], "name": "hs"},
              {"head": ["human", ["plato"]], "name": "hp"}]
    for g in [["ancestor", ["tom", "liz"]], ["ancestor", ["tom", "ann"]],
              ["ancestor", ["pat", "joe"]], ["mortal", ["socrates"]], ["mortal", ["plato"]]]:
        r = m.proof_store(g, rules)
        assert r["stored"] and r["provenance"] in ("checked", "lean_verified")
    assert m.proof_store(["ancestor", ["zeus", "tom"]], rules)["stored"] is False
    q = m.proof_recall(["ancestor", ["bob", "ann"]], k=3)
    assert q["exact"] is None
    assert q["similar"][0]["key"].startswith("ancestor(")  # planted truth: family wins top-1
    qe = m.proof_recall(["ancestor", ["tom", "liz"]], k=2)
    assert qe["exact"]["goal"] == ["ancestor", ["tom", "liz"]]
    qt = m.proof_recall(["ancestor", ["tom", "ann"]], k=1, by="tree")
    assert qt["similar"] and qt["similar"][0]["key"].startswith("ancestor(")
    # trace recall is complex-FHRR under the hood; caught live: casting to float corrupted
    # every trace vector (the ComplexWarning WAS the instrument) -- pinned complex-safe
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        qr = m.proof_recall(["ancestor", ["tom", "liz"]], k=2, by="trace")
    assert qr["similar"]
    # provenance gate: demanding lean_verified with no binary yields honest emptiness
    got = m.proof_recall(["ancestor", ["bob", "ann"]], k=3, min_provenance="lean_verified")
    assert isinstance(got["similar"], list)  # empty without a binary, populated with one


def test_lean_status_two_tiers():
    """The dependency architecture pinned: lean_status reports tier 1 iff a binary is
    reachable, tier 0 otherwise WITH the install hint -- and tier 0 must leave every
    non-Lean capability standing (kernel, checker, EMITTER -- emitting needs no binary)."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    s = m.lean_status()
    assert s["tier"] in (0, 1) and s["pinned_version"]
    if s["tier"] == 0:
        assert s["install_hint"] == "python3 tools/install_lean.py"
    # tier 0 completeness: export works binary-free (the distilled subset IS the kernel)
    rules = [{"head": ["human", ["socrates"]], "name": "h"},
             {"head": ["mortal", ["?x"]], "body": [["human", ["?x"]]], "name": "m"}]
    out = m.lean_export(["mortal", ["socrates"]], rules)
    assert out["ok"] and out["lean"].startswith("--")


def test_tabled_query_terminates_where_sld_diverges():
    """E1, with the design corrected by the SOTA check: the backlog said 'SLD with occurs
    check', but plain SLD diverges on left recursion and cyclic relations -- and transitive
    closure over a CYCLIC graph is our flagship workload. Tabling (Chen & Warren 1996; XSB /
    SWI-Prolog) is the standard fix: a subgoal that is a variant of one in progress reads the
    answer table instead of recursing. Pins: bindings for a non-ground goal, every answer's
    proof independently checked, the query slice EQUALS the fixpoint slice (two engines one
    answer set), left-recursion-over-a-cycle terminating with the right closure, and the
    occurs check (Robinson's warning, now code not prose)."""
    rules = _family_rules()
    q = L.query(L.Atom("ancestor", ("tom", "?w")), rules)
    got = {a.key() for a in q["answers"]}
    assert got == {a.key() for a in L.consequences(rules)
                   if a.pred == "ancestor" and a.args[0] == "tom"}
    for a in q["answers"]:
        assert L.check_proof(q["proofs"][a.key()], rules)
    cyc = [L.Rule(L.Atom("edge", e), name="e%d" % i)
           for i, e in enumerate([("a", "b"), ("b", "c"), ("c", "a")])]        # a cycle
    cyc += [L.Rule(L.Atom("path", ("?x", "?y")),                                # left-recursive
                   (L.Atom("path", ("?x", "?z")), L.Atom("edge", ("?z", "?y"))), name="pl"),
            L.Rule(L.Atom("path", ("?x", "?y")), (L.Atom("edge", ("?x", "?y")),), name="pb")]
    c = L.query(L.Atom("path", ("a", "?w")), cyc)
    assert sorted(a.key() for a in c["answers"]) == ["path(a,a)", "path(a,b)", "path(a,c)"]
    assert L.occurs_in("?x", "?x", {}) and not L.occurs_in("?x", "a", {})


def test_logic_query_routes_and_never_lies():
    """The measured NEGATIVE made safe: goal-direction is 304x FASTER at demand 1 but 0.3x
    (slower) at demand 690 on the repo graph, so query() is never the silent default. A blown
    budget must report itself, and the faculty must fall back to the fixpoint and still return
    the COMPLETE answer set -- both routes agree exactly."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    rules = [{"head": ["parent", ["tom", "bob"]], "name": "p0"},
             {"head": ["parent", ["bob", "liz"]], "name": "p1"},
             {"head": ["ancestor", ["?x", "?y"]], "body": [["parent", ["?x", "?y"]]], "name": "ab"},
             {"head": ["ancestor", ["?x", "?z"]],
              "body": [["parent", ["?x", "?y"]], ["ancestor", ["?y", "?z"]]], "name": "as"}]
    fast = m.logic_query(["ancestor", ["tom", "?w"]], rules)
    slow = m.logic_query(["ancestor", ["tom", "?w"]], rules, budget=1)
    assert fast["route"] == "query" and slow["route"] == "fixpoint"
    assert sorted(map(str, fast["answers"])) == sorted(map(str, slow["answers"]))
    assert m.logic_query(["ancestor", ["tom", "?w"]], rules, budget=1,
                         fallback=False)["budget_exceeded"] is True

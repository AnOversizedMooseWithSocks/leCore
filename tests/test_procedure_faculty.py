"""HoloMachine wired into UnifiedMind as a faculty -- the de-silo. Pins that procedures (executable
recipes of VSA operations) run THROUGH the mind, that the mind truly DELEGATES to the machine (not a
re-implementation: bit-identical results), that CALL-composition and program-as-data inspection work
through the mind, and that a procedure is also a typed B7 structure (bit-exact).

This file grows as the procedure work is upgraded (richer opcodes, goal-addressable recall, generation).
"""
import numpy as np
import pytest
import holographic.agents_and_reasoning.holographic_machine as hm
from holographic.agents_and_reasoning.holographic_machine import HoloMachine
from holographic.misc.holographic_unified import UnifiedMind


def test_procedure_delegates_to_machine_bit_identical():
    """The mind's procedure faculty DELEGATES to HoloMachine -- a procedure run through the mind is
    bit-for-bit the same as through a bare machine at the same dim & seed. (Wiring, not a fork.)"""
    prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "a")]
    m = UnifiedMind(dim=1024, seed=0)
    m.learn_procedure("p", prog)
    acc_mind, tr_mind = m.run_procedure("p")
    bare = HoloMachine(dim=1024, seed=0)
    bare.define("p", prog)
    acc_bare, tr_bare = bare.run(bare.functions["p"])
    assert np.array_equal(acc_mind, acc_bare)
    assert tr_mind == tr_bare


def test_procedure_runs_real_vsa_ops():
    """A procedure is a sequence of actual VSA operations: the accumulator equals the hand-computed
    algebra (LOAD a; BIND b; BUNDLE c == bundle(bind(a,b), c))."""
    m = UnifiedMind(dim=1024, seed=0)
    acc, tr = m.run_procedure([("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "a")])
    M = m._machine()
    expected = hm.bundle([hm.bind(M.data_atoms["a"], M.data_atoms["b"]), M.data_atoms["c"]])
    assert hm.cosine(acc, expected) > 0.999
    assert tr == [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c")]


def test_procedure_seeds_accumulator_from_mind_vector():
    """init_acc lets a procedure transform a vector from the mind's own space -- the bridge that makes
    a procedure an operation ON the mind's data, not just on the machine's data atoms."""
    m = UnifiedMind(dim=1024, seed=0)
    m.learn_procedure("wrap_b", [("BIND", "b"), ("HALT", "b")])
    X = hm.derived_atom(0, "some_mind_vector", 1024, unitary=True)
    acc, _ = m.run_procedure("wrap_b", init_acc=X)
    assert hm.cosine(acc, hm.bind(X, m._machine().data_atoms["b"])) > 0.999


def test_procedure_call_composition_through_mind():
    """A procedure may CALL procedures defined earlier -- a recipe of recipes, composed through the
    mind, computing the right result."""
    m = UnifiedMind(dim=1024, seed=0)
    m.learn_procedure("wrap_b", [("BIND", "b"), ("HALT", "b")])
    m.learn_procedure("add_c", [("BUNDLE", "c"), ("HALT", "c")])
    acc, tr = m.run_procedure([("LOAD", "a"), ("CALL", "wrap_b"), ("CALL", "add_c"), ("HALT", "a")])
    M = m._machine()
    expected = hm.bundle([hm.bind(M.data_atoms["a"], M.data_atoms["b"]), M.data_atoms["c"]])
    assert hm.cosine(acc, expected) > 0.999
    assert ("CALL", "wrap_b") in tr and ("CALL", "add_c") in tr


def test_decode_step_reads_procedure_as_data():
    """The von Neumann encoding lets a stored procedure be INSPECTED, not just run: decode_step reads
    instruction i back as (opcode, operand)."""
    m = UnifiedMind(dim=1024, seed=0)
    prog = [("LOAD", "a"), ("BIND", "b"), ("PERMUTE", "a"), ("HALT", "a")]
    m.learn_procedure("p", prog)
    assert [m.decode_step("p", i) for i in range(3)] == [("LOAD", "a"), ("BIND", "b"), ("PERMUTE", "a")]


def test_procedure_to_recipe_is_bit_exact():
    """A procedure is also a typed B7 structure: procedure_to_recipe reproduces the assembled program
    bit-exactly (realized recipe == machine.assemble)."""
    m = UnifiedMind(dim=1024, seed=0)
    prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "a")]
    r = m.procedure_to_recipe(prog)
    assert hm.cosine(m.realize(r), m._machine().assemble(prog)) > 0.999


def test_run_unknown_procedure_raises():
    m = UnifiedMind(dim=1024, seed=0)
    with pytest.raises(KeyError):
        m.run_procedure("nonexistent")


# ---- M2: richer opcodes -- APPLY <faculty> invokes a mind faculty as a procedure step ----

def test_apply_cleanup_recovers_noisy_accumulator():
    """APPLY cleanup delegates to the mind's dense associative cleanup: a procedure can self-correct a
    noisy accumulator back toward the nearest known value atom -- a real step a plain list of kernel
    ops cannot take."""
    m = UnifiedMind(dim=1024, seed=0)
    M = m._machine()
    before, after = [], []
    for t in range(20):
        rng = np.random.default_rng(t)
        true = M.data_atoms["c"]
        noisy = true + 0.5 * rng.standard_normal(1024)
        acc, _ = m.run_procedure([("APPLY", "cleanup"), ("HALT", "c")], init_acc=noisy)
        before.append(hm.cosine(noisy, true)); after.append(hm.cosine(acc, true))
    assert np.mean(before) < 0.2 and np.mean(after) > 0.5      # cleanup substantially recovers


def test_apply_backward_compatible_bit_identical():
    """Adding APPLY did not change existing programs: one without APPLY is still bit-for-bit identical
    to a bare HoloMachine (the run() signature change is backward-compatible)."""
    prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "a")]
    m = UnifiedMind(dim=1024, seed=0)
    m.learn_procedure("p", prog)
    am, _ = m.run_procedure("p")
    bare = HoloMachine(dim=1024, seed=0)
    bare.define("p", prog)
    ab, _ = bare.run(bare.functions["p"])
    assert np.array_equal(am, ab)


def test_apply_decodes_and_bare_vm_runs_it_as_noop():
    """APPLY decodes back as data, and the bare VM (no handlers) runs an APPLY program as a safe no-op
    -- so a procedure with APPLY is still a valid, runnable program everywhere."""
    m = UnifiedMind(dim=1024, seed=0)
    prog = [("LOAD", "a"), ("APPLY", "cleanup"), ("HALT", "a")]
    assert m.decode_step(prog, 1) == ("APPLY", "cleanup")
    bare = HoloMachine(dim=1024, seed=0)
    acc, tr = bare.run(bare.assemble(prog))
    assert hm.cosine(acc, bare.data_atoms["a"]) > 0.999       # no handler -> acc untouched by APPLY
    assert ("APPLY", "cleanup") in tr


def test_apply_procedure_to_recipe_bit_exact():
    """A procedure containing APPLY is still a typed B7 structure, reproduced bit-exactly."""
    m = UnifiedMind(dim=1024, seed=0)
    prog = [("LOAD", "a"), ("APPLY", "cleanup"), ("HALT", "a")]
    r = m.procedure_to_recipe(prog)
    assert hm.cosine(m.realize(r), m._machine().assemble(prog)) > 0.999


# ---- M3: procedure memory -- goal-addressable recall over the library ----

def _mixed_library(seed=0):
    m = UnifiedMind(dim=1024, seed=seed)
    m.learn_procedure("t_b", [("BIND", "b"), ("HALT", "b")])
    m.learn_procedure("t_d", [("BIND", "d"), ("HALT", "d")])
    m.learn_procedure("t_perm", [("PERMUTE", "a"), ("HALT", "a")])
    m.learn_procedure("t_e", [("BUNDLE", "e"), ("HALT", "e")])
    return m


def test_recall_procedure_identifies_from_one_example():
    """Given ONE (input -> output) example, recall which stored procedure produced it -- behaviourally,
    so it works across a MIXED library (bind / permute / bundle), not just single-bind transforms."""
    m = _mixed_library()
    for trial, name in enumerate(["t_b", "t_perm", "t_e", "t_d"]):
        X = hm.derived_atom(0, f"x{trial}", 1024, unitary=True)
        Y, _ = m.run_procedure(name, init_acc=X)
        got, score = m.recall_procedure(X, Y)
        assert got == name and score > 0.99


def test_recall_and_apply_transfers_to_new_input():
    """Learn the operation from one example, then apply it to NEW input (analogy/transfer)."""
    m = _mixed_library()
    M = m._machine()
    X = hm.derived_atom(0, "x", 1024, unitary=True)
    Y, _ = m.run_procedure("t_b", init_acc=X)            # demonstrate the bind-b transform
    Z = hm.derived_atom(0, "z", 1024, unitary=True)
    W, used, score = m.recall_and_apply(X, Y, Z)
    assert used == "t_b"
    assert hm.cosine(W, hm.bind(Z, M.data_atoms["b"])) > 0.99    # same transform, new input


def test_recall_on_empty_library_returns_none():
    m = UnifiedMind(dim=1024, seed=0)
    X = hm.derived_atom(0, "x", 1024, unitary=True)
    name, score = m.recall_procedure(X, X)
    assert name is None


# ---- M4: recipe generation/completion -- predict the next opcode from a partial recipe ----

def _grammar_mind():
    """Grammar: LOAD, BIND, then 0-2 of {BUNDLE,APPLY,PERMUTE}, then HALT."""
    m = UnifiedMind(dim=512, seed=0)
    MID = ["BUNDLE", "APPLY", "PERMUTE"]
    rng = np.random.default_rng(0)
    recipes = []
    for _ in range(60):
        k = int(rng.integers(0, 3))
        ops = ["LOAD", "BIND"] + list(rng.choice(MID, k)) + ["HALT"]
        recipes.append([(o, "a") for o in ops])
    m.learn_recipe_grammar(recipes, order=3)
    return m, MID


def test_recipe_grammar_predicts_valid_next_opcode():
    """The learned grammar anticipates the next opcode: BIND must follow LOAD, and HALT must follow
    two middle ops -- the hard constraints of the training grammar."""
    m, MID = _grammar_mind()
    assert m.complete_procedure(["LOAD"])[0] == "BIND"
    assert m.complete_procedure(["LOAD", "BIND", "BUNDLE", "APPLY"])[0] == "HALT"
    # held-out: every predicted next opcode is a grammar-valid continuation
    rng = np.random.default_rng(123)
    valid = []
    for _ in range(40):
        k = int(rng.integers(0, 3))
        ops = ["LOAD", "BIND"] + list(rng.choice(MID, k)) + ["HALT"]
        for i in range(len(ops)):
            pred = m.complete_procedure(ops[:i])[0]
            if i == 0:
                ok = pred == "LOAD"
            elif ops[:i] == ["LOAD"]:
                ok = pred == "BIND"
            else:
                mids = [o for o in ops[2:i] if o in MID]
                ok = pred == "HALT" if len(mids) >= 2 else pred in set(MID) | {"HALT"}
            valid.append(ok)
    assert np.mean(valid) > 0.95


def test_complete_procedure_without_grammar_returns_none():
    m = UnifiedMind(dim=512, seed=0)
    assert m.complete_procedure(["LOAD"]) == (None, 0.0)


# ---- M5: fingerprint fast-path for recall (zero-run shortcut, confidence-gated) ----

def _five_proc_mind():
    m = UnifiedMind(dim=1024, seed=0)
    m.learn_procedure("t_b", [("BIND", "b"), ("HALT", "b")])
    m.learn_procedure("t_c", [("BIND", "c"), ("HALT", "c")])
    m.learn_procedure("t_perm", [("PERMUTE", "a"), ("HALT", "a")])
    m.learn_procedure("t_clean", [("BIND", "b"), ("APPLY", "cleanup"), ("HALT", "b")])
    return m


def test_fingerprint_auto_matches_behavioral_across_mixed_library():
    """With a fingerprint index, 'auto' identifies the right procedure across a MIXED library -- binds
    via the zero-run shortcut, permute/nonlinear via the behavioural fallback (the confidence gate
    routes them). And it stays backward-compatible: with no index, 'auto' is the behavioural scan."""
    m = _five_proc_mind()
    # backward-compat: no index -> behavioural result
    X = hm.derived_atom(0, "x", 1024, unitary=True)
    Y, _ = m.run_procedure("t_b", init_acc=X)
    assert m.recall_procedure(X, Y)[0] == "t_b"
    # with index: every kind resolves correctly under auto
    m.index_procedures()
    for nm in ["t_b", "t_c", "t_perm", "t_clean"]:
        Xi = hm.derived_atom(0, f"x_{nm}", 1024, unitary=True)
        Yi, _ = m.run_procedure(nm, init_acc=Xi)
        assert m.recall_procedure(Xi, Yi, method="auto")[0] == nm


def test_fingerprint_shortcut_is_exact_for_linear_and_gated_for_nonlinear():
    """method='fingerprint' identifies a LINEAR transform with high confidence and ZERO program runs --
    and that class is bind AND permute (permutation is convolution by a shifted delta, so it commutes
    with binding the same way: bind(P, unbind(permute(X), X)) == permute(P)). A genuinely NONLINEAR
    procedure (one with an APPLY cleanup step) scores near zero, which is exactly why 'auto' falls back
    to the behavioural scan there."""
    m = _five_proc_mind()
    m.index_procedures()
    X = hm.derived_atom(0, "x", 1024, unitary=True)
    for nm in ["t_b", "t_perm"]:                                  # both linear -> shortcut nails them
        Y, _ = m.run_procedure(nm, init_acc=X)
        name, score = m.recall_procedure(X, Y, method="fingerprint")
        assert name == nm and score > 0.5
    Yc, _ = m.run_procedure("t_clean", init_acc=X)                # nonlinear -> below the 0.5 gate
    _, cscore = m.recall_procedure(X, Yc, method="fingerprint")
    assert cscore < 0.5


# ---- M6: procedure synthesis -- CONSTRUCT a procedure for a goal (not just recall one) ----

def test_synthesize_single_and_composite_procedures():
    """Synthesis constructs a verified program mapping input -> output for single-op and composite
    targets, including the order-sensitive PERMUTE-then-BUNDLE case."""
    m = UnifiedMind(dim=1024, seed=0)
    M = m._machine(); A = M.data_atoms
    X = hm.derived_atom(0, "X", 1024, unitary=True)
    cases = {
        "bind": hm.bind(X, A["b"]),
        "bind_bind": hm.bind(hm.bind(X, A["b"]), A["c"]),
        "perm_bundle": hm.bundle([hm.permute(X, 1), A["d"]]),
    }
    for _, Y in cases.items():
        p = m.synthesize_procedure(X, Y, max_depth=2)
        assert p is not None and p[-1][0] == "HALT"
        out, _ = m.run_procedure(p, init_acc=X)
        assert hm.cosine(out, Y) > 0.99                       # verified to map X -> Y


def test_synthesized_procedure_generalizes_to_new_input():
    """A synthesized program captures the TRANSFORM, not the example pair: it does the same operation
    on a fresh input."""
    m = UnifiedMind(dim=1024, seed=0)
    M = m._machine(); A = M.data_atoms
    X = hm.derived_atom(0, "X", 1024, unitary=True)
    Y = hm.bind(hm.bind(X, A["b"]), A["c"])
    p = m.synthesize_procedure(X, Y, max_depth=2)
    X2 = hm.derived_atom(0, "X2", 1024, unitary=True)
    out2, _ = m.run_procedure(p, init_acc=X2)
    assert hm.cosine(out2, hm.bind(hm.bind(X2, A["b"]), A["c"])) > 0.99


def test_synthesize_returns_none_when_unreachable():
    """Honest negative: a target not reachable within max_depth returns None."""
    m = UnifiedMind(dim=1024, seed=0)
    X = hm.derived_atom(0, "X", 1024, unitary=True)
    Y = hm.derived_atom(0, "unreachable", 1024, unitary=True)
    assert m.synthesize_procedure(X, Y, max_depth=2) is None


# ---- M7: control flow -- IFMATCH (conditional) and ITERATE (fixed-point loop) ----

def test_iterate_converges_to_fixed_point():
    """ITERATE re-applies a body to ACC until it stops changing -- the input->process->feed-back loop.
    A cleanup body on a low-noise input converges to the clean atom (reason 'converged')."""
    m = UnifiedMind(dim=1024, seed=0)
    M = m._machine()
    m.learn_procedure("clean_step", [("APPLY", "cleanup"), ("HALT", "c")])
    true = M.data_atoms["c"]
    noisy = true + 0.3 * np.random.default_rng(0).standard_normal(1024)
    acc, tr = m.run_procedure([("ITERATE", "clean_step"), ("HALT", "c")], init_acc=noisy)
    it = [x for x in tr if x[0] == "ITERATE"][0]
    assert it[3] == "converged" and it[2] >= 1
    assert hm.cosine(acc, true) > 0.9                       # converged to the right attractor


def test_iterate_goal_and_cap_exits():
    """ITERATE exits early when a host 'stop' predicate marks the desired OUTPUT reached, and caps at
    max_loop for a body that never converges."""
    m = UnifiedMind(dim=1024, seed=0)
    M = m._machine()
    m.learn_procedure("clean_step", [("APPLY", "cleanup"), ("HALT", "c")])
    m.learn_procedure("rotate", [("PERMUTE", "a"), ("HALT", "a")])     # never reaches a fixed point
    true = M.data_atoms["c"]
    noisy = true + 0.3 * np.random.default_rng(0).standard_normal(1024)
    _, tg = m.run_procedure([("ITERATE", "clean_step"), ("HALT", "c")], init_acc=noisy,
                            stop=lambda a: hm.cosine(a, true) >= 0.9)
    assert [x for x in tg if x[0] == "ITERATE"][0][3] == "goal"
    X = hm.derived_atom(0, "x", 1024, unitary=True)
    _, tc = m.run_procedure([("ITERATE", "rotate"), ("HALT", "a")], init_acc=X, max_loop=6)
    cap = [x for x in tc if x[0] == "ITERATE"][0]
    assert cap[3] == "maxloop" and cap[2] == 6


def test_ifmatch_conditional_branches_both_ways():
    """IFMATCH gates the next instruction on a predicate over ACC: the guarded CALL runs on a match and
    is skipped on a mismatch."""
    m = UnifiedMind(dim=1024, seed=0)
    M = m._machine()
    m.learn_procedure("mark", [("BIND", "b"), ("HALT", "b")])
    acc_m, tr_m = m.run_procedure([("LOAD", "a"), ("IFMATCH", "a"), ("CALL", "mark"), ("HALT", "a")])
    acc_n, tr_n = m.run_procedure([("LOAD", "a"), ("IFMATCH", "c"), ("CALL", "mark"), ("HALT", "a")])
    assert hm.cosine(acc_m, hm.bind(M.data_atoms["a"], M.data_atoms["b"])) > 0.99   # match -> CALL ran
    assert [t[0] for t in tr_m] == ["LOAD", "IFMATCH", "CALL"]
    assert hm.cosine(acc_n, M.data_atoms["a"]) > 0.99                                # mismatch -> CALL skipped
    assert [t[0] for t in tr_n] == ["LOAD", "IFMATCH"]


def test_control_flow_backward_compatible_and_recipe_rules():
    """A program with no control flow is still bit-identical to a bare VM; IFMATCH (data operand) is a
    typed structure bit-exactly, while ITERATE (runtime library lookup, like CALL) is out of scope."""
    import pytest
    prog = [("LOAD", "a"), ("BIND", "b"), ("HALT", "a")]
    m = UnifiedMind(dim=1024, seed=0)
    m.learn_procedure("p", prog)
    am, _ = m.run_procedure("p")
    bare = HoloMachine(dim=1024, seed=0)
    bare.define("p", prog)
    ab, _ = bare.run(bare.functions["p"])
    assert np.array_equal(am, ab)
    # IFMATCH decodes and bridges to a typed structure bit-exactly
    ifp = [("LOAD", "a"), ("IFMATCH", "c"), ("HALT", "a")]
    assert m.decode_step(ifp, 1) == ("IFMATCH", "c")
    r = m.procedure_to_recipe(ifp)
    assert hm.cosine(m.realize(r), m._machine().assemble(ifp)) > 0.999
    # ITERATE is runtime -> recipe bridge rejects it
    m.learn_procedure("body", [("PERMUTE", "a"), ("HALT", "a")])
    with pytest.raises(ValueError):
        m.procedure_to_recipe([("ITERATE", "body"), ("HALT", "a")])


# ---- VM-1: matmul in the loop -- exact_matmul as an APPLY faculty ----

def test_matmul_apply_iterates_to_stationary_distribution():
    """ITERATE [APPLY matmul] with a column-stochastic matrix is power iteration: it converges to the
    stationary distribution (the dominant eigenvector) -- a real iterative algorithm as a VM program,
    with the engine's exact matmul as the process step."""
    D = 64
    m = UnifiedMind(dim=D, seed=0)
    rng = np.random.default_rng(0)
    P = np.abs(rng.standard_normal((D, D))) + 0.05
    P = P / P.sum(axis=0, keepdims=True)
    evals, evecs = np.linalg.eig(P)
    k = int(np.argmin(np.abs(evals - 1.0)))
    stationary = np.abs(np.real(evecs[:, k])); stationary /= stationary.sum()
    m.set_matmul(P)
    m.learn_procedure("mm_step", [("APPLY", "matmul"), ("HALT", "a")])
    start = np.abs(rng.standard_normal(D)); start /= start.sum()
    acc, tr = m.run_procedure([("ITERATE", "mm_step"), ("HALT", "a")], init_acc=start,
                              converge_tol=0.99999, max_loop=200)
    it = [t for t in tr if t[0] == "ITERATE"][0]
    acc_dist = np.abs(acc) / np.abs(acc).sum()
    assert it[3] == "converged"
    assert hm.cosine(acc_dist, stationary) > 0.99


def test_matmul_apply_disabled_is_noop():
    """With no matrix configured, APPLY matmul is a safe no-op (so the opcode is harmless until used)."""
    m = UnifiedMind(dim=64, seed=0)
    x = np.abs(np.random.default_rng(0).standard_normal(64))
    acc, tr = m.run_procedure([("APPLY", "matmul"), ("HALT", "a")], init_acc=x)
    assert np.array_equal(acc, x)
    assert ("APPLY", "matmul") in tr


# ---- VM-2: counted loop -- REPEAT n runs the next CALL n times ----

def test_repeat_runs_the_next_call_n_times():
    """REPEAT n; CALL f runs f exactly n times. With a one-permute body, the result is permute(X, n) --
    an exact, countable check that the loop ran the right number of times."""
    m = UnifiedMind(dim=1024, seed=0)
    m.learn_procedure("shiftone", [("PERMUTE", "a"), ("HALT", "a")])
    X = hm.derived_atom(0, "X", 1024, unitary=True)
    for n in (1, 3, 5):
        acc, tr = m.run_procedure([("REPEAT", n), ("CALL", "shiftone"), ("HALT", "a")], init_acc=X)
        assert hm.cosine(acc, hm.permute(X, n)) > 0.99
        assert [t[0] for t in tr] == ["REPEAT", "CALL"]


def test_repeat_decodes_and_runs_on_bare_vm():
    """REPEAT decodes back as (REPEAT, count) data, and the bare VM runs a REPEAT;CALL program."""
    m = UnifiedMind(dim=1024, seed=0)
    m.learn_procedure("shiftone", [("PERMUTE", "a"), ("HALT", "a")])
    assert m.decode_step([("REPEAT", 3), ("CALL", "shiftone"), ("HALT", "a")], 0) == ("REPEAT", 3)
    bare = HoloMachine(dim=1024, seed=0)
    bare.define("shiftone", [("PERMUTE", "a"), ("HALT", "a")])
    X = hm.derived_atom(0, "X", 1024, unitary=True)
    acc, tr = bare.run(bare.assemble([("REPEAT", 2), ("CALL", "shiftone"), ("HALT", "a")]), init_acc=X)
    assert hm.cosine(acc, hm.permute(X, 2)) > 0.99
    assert [t[0] for t in tr] == ["REPEAT", "CALL"]


# ---- VM-3: control flow composes -- nesting + a worked program ----

def test_nested_control_flow_composes():
    """Control flow nests: a counted loop of convergence loops (REPEAT>CALL>ITERATE) and a convergence
    loop whose body CALLs (ITERATE>CALL) both reach the fixed point."""
    m = UnifiedMind(dim=1024, seed=0)
    M = m._machine()
    true = M.data_atoms["c"]
    m.learn_procedure("clean_step", [("APPLY", "cleanup"), ("HALT", "c")])
    m.learn_procedure("refine", [("ITERATE", "clean_step"), ("HALT", "c")])
    m.learn_procedure("double_clean", [("CALL", "clean_step"), ("CALL", "clean_step"), ("HALT", "c")])
    n0 = true + 0.3 * np.random.default_rng(0).standard_normal(1024)
    acc1, _ = m.run_procedure([("REPEAT", 2), ("CALL", "refine"), ("HALT", "c")], init_acc=n0)
    assert hm.cosine(acc1, true) > 0.9
    n1 = true + 0.3 * np.random.default_rng(1).standard_normal(1024)
    acc2, tr2 = m.run_procedure([("ITERATE", "double_clean"), ("HALT", "c")], init_acc=n1)
    assert [t for t in tr2 if t[0] == "ITERATE"][0][3] == "converged" and hm.cosine(acc2, true) > 0.9
    # determinism: identical program + input runs bit-identically
    a, _ = m.run_procedure([("REPEAT", 2), ("CALL", "refine"), ("HALT", "c")], init_acc=n0)
    b, _ = m.run_procedure([("REPEAT", 2), ("CALL", "refine"), ("HALT", "c")], init_acc=n0)
    assert np.array_equal(a, b)


def test_worked_program_denoise_classify_tag():
    """A complete routine in one procedure: ITERATE a cleanup to denoise the input, IFMATCH the cleaned
    result against a target, and CALL a tag only when it matches -- loop + conditional + call together."""
    m = UnifiedMind(dim=1024, seed=0)
    M = m._machine()
    m.learn_procedure("clean_step", [("APPLY", "cleanup"), ("HALT", "c")])
    m.learn_procedure("tag", [("BIND", "b"), ("HALT", "b")])
    prog = [("ITERATE", "clean_step"), ("IFMATCH", "c"), ("CALL", "tag"), ("HALT", "c")]
    nc = M.data_atoms["c"] + 0.3 * np.random.default_rng(0).standard_normal(1024)
    nd = M.data_atoms["d"] + 0.3 * np.random.default_rng(1).standard_normal(1024)
    acc_c, tr_c = m.run_procedure(prog, init_acc=nc)
    acc_d, tr_d = m.run_procedure(prog, init_acc=nd)
    assert hm.cosine(acc_c, hm.bind(M.data_atoms["c"], M.data_atoms["b"])) > 0.99   # cleaned to c, then tagged
    assert [t[0] for t in tr_c] == ["ITERATE", "IFMATCH", "CALL"]
    assert hm.cosine(acc_d, M.data_atoms["d"]) > 0.99                                # cleaned to d, not tagged
    assert [t[0] for t in tr_d] == ["ITERATE", "IFMATCH"]


# ---- PIPE-1: the automatic data-analysis pipeline (a VSA program) -----------------------------------

def _grid(n=256):
    return np.linspace(0, 1, n)


def test_analysis_pipeline_finds_structure_and_branches_to_training():
    """On a STRUCTURED signal the pipeline PROGRAM runs analyze -> denoise-loop -> decompose, the IFMATCH
    fires, and the CALL'd train+validate runs: a generative law is found and it generalizes held-out."""
    t = _grid()
    sig = 1 + 2 * t + 3 * t ** 2 + 0.3 * np.random.default_rng(0).standard_normal(t.size)
    rep = UnifiedMind(dim=256, seed=1).run_analysis_pipeline(sig)
    assert rep["explained_var"] > 0.9 and rep["n_terms"] >= 1     # a law was found
    assert "CALL" in rep["_ops"]                                  # IFMATCH structured -> CALL train+validate
    assert rep.get("trained") is True
    assert rep["heldout_rel"] < 0.5                               # the law extrapolates to unseen tail
    assert rep["saved_as"] == "generative_law"
    assert rep["law_bytes"] < sig.nbytes                          # stored far smaller than the raw samples


def test_analysis_pipeline_on_noise_skips_training_honestly():
    """On PURE NOISE decompose reports no structure, so the SAME program's IFMATCH SKIPS the CALL --
    train+validate do not run and save reports raw_only. The branch is real, driven by the data."""
    rep = UnifiedMind(dim=256, seed=2).run_analysis_pipeline(
        np.random.default_rng(0).standard_normal(256))
    assert rep["explained_var"] < 0.1 and rep["n_terms"] == 0     # nothing found
    assert "CALL" not in rep["_ops"]                              # the conditional skipped training
    assert rep.get("trained") is None                            # train never ran
    assert rep["saved_as"] == "raw_only"                         # honest: no compressible law


def test_pipeline_denoise_is_self_similar_and_converges():
    """The loop body denoises a signal against its OWN trajectory structure (no external prior) -- it
    reduces noise on a structured signal and is ~idempotent, so the ITERATE converges."""
    t = _grid()
    clean = 1 + 2 * t + 3 * t ** 2
    noisy = clean + 0.3 * np.random.default_rng(0).standard_normal(t.size)
    m = UnifiedMind(dim=256, seed=0)
    d1 = m._denoise_signal(noisy)
    assert np.sqrt(np.mean((d1 - clean) ** 2)) < 0.6 * np.sqrt(np.mean((noisy - clean) ** 2))  # noise down
    d2 = m._denoise_signal(d1)
    assert hm.cosine(d1, d2) > 0.99                              # iterating is near-fixed -> ITERATE settles


def test_pipeline_decompose_delegates_to_decompose_signal():
    """The pipeline's decompose step DELEGATES to decompose_signal, it does not re-implement it: the law
    it recorded matches a direct decompose_signal on the final denoised signal (same n_terms)."""
    t = _grid()
    sig = np.exp(1.5 * t) + 0.2 * np.random.default_rng(3).standard_normal(t.size)
    m = UnifiedMind(dim=256, seed=4)
    rep = m.run_analysis_pipeline(sig)
    denoised = m._pipe["signal"]                                  # the signal decompose actually saw
    _, info = m.decompose_signal(denoised)
    assert rep["n_terms"] == info["n_terms"]                     # above (pipeline) == below (faculty)


def test_pipeline_is_a_vsa_program_not_python_control_flow():
    """The pipeline is a HoloMachine PROGRAM run by the VM, not Python branching: a custom program of
    (opcode, operand) tuples runs through run_procedure and its faculties still fire."""
    t = _grid()
    sig = 1 + 2 * t + 3 * t ** 2
    m = UnifiedMind(dim=256, seed=5)
    # a minimal hand-written pipeline program (analyze then decompose, no loop/branch) -- pure VSA opcodes
    prog = [("APPLY", "analyze"), ("APPLY", "decompose"), ("HALT", "a")]
    rep = m.run_analysis_pipeline(sig, program=prog)
    assert rep["_ops"] == ["APPLY", "APPLY"]                     # exactly the two APPLYs we wrote, via the VM
    assert "explained_var" in rep and "topology" in rep         # both faculties delegated and recorded


# ---- PIPE-1 recursive peel: access structure on EVERY level -----------------------------------------

def test_recursive_peel_finds_cross_basis_structure_in_layers():
    """A line trend + a periodic part is structure ONE decompose cannot fit together (the trend explains
    only a modest fraction). recursive=True peels it layer by layer -- trend, then the periodic component --
    driving the residual down across several levels. Two ITERATE loops run (denoise, then peel)."""
    t = _grid()
    sig = 0.5 + 2 * t + np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.default_rng(0).standard_normal(t.size)
    rep = UnifiedMind(dim=256, seed=1).run_analysis_pipeline(sig, recursive=True)
    assert rep["n_levels"] >= 2                                   # caught on SEPARATE levels
    assert rep["cumulative_explained"] > 0.9                      # together they explain almost all of it
    assert rep["_ops"].count("ITERATE") == 2                      # denoise loop AND peel loop both ran
    assert "CALL" in rep["_ops"] and rep["saved_as"] == "law_ladder"


def test_recursive_peel_stops_when_one_basis_suffices():
    """When a SINGLE decompose captures everything (its additive dictionary fits poly+exp at once), peeling
    correctly stops at one level -- it does not invent spurious extra layers."""
    t = _grid()
    rep = UnifiedMind(dim=256, seed=2).run_analysis_pipeline((1 + 2 * t + 3 * t ** 2) + np.exp(2 * t),
                                                             recursive=True)
    assert rep["n_levels"] == 1 and rep["cumulative_explained"] > 0.99
    assert rep["saved_as"] == "generative_law"                   # one law, not a ladder


def test_recursive_peel_on_noise_finds_no_levels():
    """On pure noise the MDL gate admits no term at the first level, so peeling finds nothing, the IFMATCH
    skips training, and save reports raw_only -- the recursive mode is honest about emptiness too."""
    rep = UnifiedMind(dim=256, seed=3).run_analysis_pipeline(
        np.random.default_rng(0).standard_normal(256), recursive=True)
    assert rep["n_levels"] == 0 and rep["cumulative_explained"] == 0.0
    assert "CALL" not in rep["_ops"] and rep["saved_as"] == "raw_only"


def test_recursive_peel_levels_delegate_to_decompose_signal():
    """Each peel level is a real decompose_signal call, not a re-implementation: the first level's topology
    and term count match a direct decompose_signal on the (denoised) signal the peel started from."""
    t = _grid()
    sig = 0.5 + 2 * t + np.sin(2 * np.pi * 5 * t)
    m = UnifiedMind(dim=256, seed=4)
    rep = m.run_analysis_pipeline(sig, recursive=True)
    first = rep["levels"][0]
    _, info = m.decompose_signal(m._pipe["signal"])               # the denoised signal peeling began on
    assert first["topology"] == info["topology"] and first["n_terms"] == info["n_terms"]


# ---- SYN-1: canonicalize (the constructive flip of "deep synthesis is unnecessary") -----------------

def test_canonicalize_collapses_bind_permute_program():
    """Any bind/permute interleaving flattens to permute(x, net) bound by the operand product. A deep
    program reduces to a verified-equivalent shorter one (binds -> one bind; permutes stay unit shifts)."""
    m = UnifiedMind(dim=1024, seed=0)
    prog = [("BIND", "a"), ("PERMUTE", "a"), ("BIND", "b"), ("PERMUTE", "a"), ("BIND", "c"), ("HALT", "a")]
    canon, info = m.canonicalize_procedure(prog)
    assert info["fully_collapsible"] and info["verified"] and info["equivalence_cosine"] > 0.999
    assert info["net_shift"] == 2 and info["n_bind"] == 3
    assert info["canonical_len"] < info["original_len"]          # genuinely shorter (3 binds -> 1)


def test_canonicalize_binds_collapse_to_single_product():
    """k binds with no permutes collapse to ONE bind by the product of operands."""
    m = UnifiedMind(dim=1024, seed=1)
    _, info = m.canonicalize_procedure([("BIND", c) for c in "abcde"] + [("HALT", "a")])
    assert info["canonical_len"] == 1 and info["n_bind"] == 5 and info["verified"]


def test_canonicalize_refuses_bundle_and_nonlinear_barriers():
    """BUNDLE (normalization breaks commutativity) and nonlinear ops do not collapse -- canonicalization is
    refused honestly rather than returning a wrong partial answer."""
    m = UnifiedMind(dim=512, seed=2)
    canon, info = m.canonicalize_procedure([("BIND", "a"), ("BUNDLE", "b"), ("BIND", "c"), ("HALT", "a")])
    assert canon is None and info["fully_collapsible"] is False
    assert any(op == "BUNDLE" for _, op in info["barriers"])


def test_canonicalize_detects_equivalent_programs():
    """Two differently-written bind/permute programs that compute the same function reduce to the SAME
    canonical form -- canonicalization is an equivalence oracle for the invertible algebra."""
    m = UnifiedMind(dim=1024, seed=3)
    ca, ia = m.canonicalize_procedure([("BIND", "a"), ("PERMUTE", "a"), ("BIND", "b"), ("HALT", "a")])
    cb, ib = m.canonicalize_procedure([("PERMUTE", "a"), ("BIND", "b"), ("BIND", "a"), ("HALT", "a")])
    assert ca == cb and ia["verified"] and ib["verified"]


# ---- REC-1: vectorized fingerprint recall (forest-indexing measured premature, rejected) ------------

def test_fingerprint_recall_is_vectorized_and_matches_loop():
    """index_procedures caches a unit-normalized fingerprint MATRIX, and recall does one matrix-vector
    product (cosine vs every fingerprint at once) instead of a Python loop -- giving the SAME identity the
    per-candidate scan would, including the exact score. (Vectorizing the O(N) scan was the right fix; a
    HoloForest index was measured 3-7x slower for realistic libraries and is not used.)"""
    m = UnifiedMind(dim=1024, seed=0)
    for nm, prog in [("shift", [("PERMUTE", "a")]), ("bindb", [("BIND", "b")]), ("bindc", [("BIND", "c")])]:
        m.learn_procedure(nm, prog + [("HALT", "a")])
    m.index_procedures()
    assert m._proc_fp_mat.shape == (3, 1024)                      # the cached matrix exists
    assert np.allclose(np.linalg.norm(m._proc_fp_mat, axis=1), 1.0)   # rows unit-normalized -> matvec == cosine
    x = hm.derived_atom(7, "x", 1024, unitary=True)
    for nm in ("shift", "bindb", "bindc"):
        out, _ = m.run_procedure(nm, init_acc=x)
        # vectorized (full) path and the dict-loop (subset) path must agree on the winner
        full, sf = m.recall_procedure(x, out, method="fingerprint")
        subset, ss = m.recall_procedure(x, out, names=["shift", "bindb", "bindc"], method="fingerprint")
        assert full == nm and subset == nm and abs(sf - ss) < 1e-6


# ---- GEN-1: operand prediction in recipe completion -------------------------------------------------

_tA = [("BIND", "a"), ("BIND", "b"), ("BIND", "c"), ("HALT", "a")]
_tB = [("BIND", "d"), ("BIND", "e"), ("BIND", "f"), ("HALT", "a")]


def _operand_acc(m, test):
    c = t = 0
    for r in test:
        for i in range(1, len(r)):
            _, operand, _ = m.complete_instruction(r[:i])
            t += 1; c += (operand == r[i][1])
    return c / t


def test_complete_instruction_predicts_patterned_operands():
    """When operand USAGE is patterned (two templates a->b->c and d->e->f, the operand determined by
    context), complete_instruction predicts the full next instruction -- opcode AND the right operand --
    and it generalizes to held-out recipes of the same patterns."""
    m = UnifiedMind(dim=1024, seed=0)
    m.learn_recipe_grammar([_tA] * 10 + [_tB] * 10)
    op, operand, conf = m.complete_instruction([("BIND", "a")])
    assert (op, operand) == ("BIND", "b")                         # context-correct operand, not just opcode
    assert m.complete_instruction([("BIND", "d")])[:2] == ("BIND", "e")   # the OTHER template's operand
    assert _operand_acc(m, [_tA] * 3 + [_tB] * 3) > 0.9          # generalizes


def test_operand_prediction_fails_on_random_operands_but_shape_holds():
    """When operands are arbitrary per recipe, the operand is unknowable -- operand-prediction accuracy
    falls to chance -- yet the opcode SHAPE is still predicted (the opcode grammar ignores operands). So
    the honest discriminator is generalization, not the (n-gram-overconfident) score."""
    import random
    random.seed(2)
    mk = lambda: [("BIND", random.choice("abcdef")) for _ in range(3)] + [("HALT", "a")]
    train = [mk() for _ in range(20)]; test = [mk() for _ in range(20)]
    m = UnifiedMind(dim=1024, seed=0); m.learn_recipe_grammar(train)
    assert _operand_acc(m, test) < 0.5                           # operand ~chance (1/6) -> not learnable
    # but opcode shape is still anticipated (operand-independent)
    assert m.complete_procedure([("BIND", "a")])[0] == "BIND"


def test_complete_instruction_no_grammar_returns_empty():
    """With no grammar learned, complete_instruction returns (None, None, 0.0) -- backward-safe."""
    assert UnifiedMind(dim=256, seed=0).complete_instruction([("BIND", "a")]) == (None, None, 0.0)

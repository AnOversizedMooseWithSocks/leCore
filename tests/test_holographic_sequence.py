"""ORDER as a queryable property -- the PB&J problem. The bag-of-everything
stores deliberately discard order (correct for topic/class/record), but some
meaning lives only in the sequence: a recipe with steps in the wrong order is
not a recipe. SequenceMemory makes order queryable with the same primitives."""
import numpy as np
from holographic.misc.holographic_sequence import SequenceMemory


def test_sequence_encoding_is_order_sensitive():
    # the foundation: a scrambled sequence must be near-orthogonal to the
    # correct one (measured cosine ~0.03), or order is being lost
    from holographic.agents_and_reasoning.holographic_ai import cosine
    m = SequenceMemory(dim=2048, seed=0)
    correct = ["a", "b", "c", "d", "e"]
    scrambled = ["c", "a", "e", "b", "d"]
    vc = m.encode(correct)
    assert cosine(vc, m.encode(correct)) > 0.99       # itself
    assert cosine(vc, m.encode(scrambled)) < 0.2      # scramble -> orthogonal


def test_step_and_position_queries_are_exact():
    m = SequenceMemory(dim=2048, seed=0)
    recipe = ["bread", "peanut_butter", "jelly", "close", "cut"]
    m.add("pbj", recipe)
    for i, s in enumerate(recipe):
        assert m.step("pbj", i) == s                  # read each position back
        assert m.position_of("pbj", s) == i


def test_precedence_and_validation_catch_bad_order():
    m = SequenceMemory(dim=2048, seed=0)
    recipe = ["bread", "peanut_butter", "jelly", "close", "cut"]
    m.add("pbj", recipe)
    assert m.precedes("pbj", "jelly", "cut") is True
    assert m.precedes("pbj", "cut", "jelly") is False
    # the PB&J test: a plan that cuts before assembling is INVALID, and the
    # validator names the violated rule
    constraints = [("peanut_butter", "close"), ("jelly", "close"), ("close", "cut")]
    assert m.validate("pbj", constraints) == (True, [])
    bad = ["bread", "cut", "peanut_butter", "jelly", "close"]
    ok, viol = m.validate(bad, constraints)
    assert ok is False and ("close", "cut") in viol


def test_order_queries_accurate_across_random_plans():
    m = SequenceMemory(dim=2048, seed=0)
    rng = np.random.default_rng(1)
    steps = [f"s{i}" for i in range(12)]
    pos_ok = prec_ok = tot = 0
    for _ in range(200):
        L = rng.integers(4, 9)
        seq = list(rng.choice(steps, L, replace=False))
        i = rng.integers(0, L)
        pos_ok += (m.step(seq, i) == seq[i])
        a, b = rng.choice(seq, 2, replace=False)
        prec_ok += (m.precedes(seq, a, b) == (seq.index(a) < seq.index(b)))
        tot += 1
    assert pos_ok / tot >= 0.95                        # measured 100%
    assert prec_ok / tot >= 0.95                       # measured 100%


def test_unified_mind_learns_and_validates_plans():
    # ORDER reaches the unified brain: learn_plan stores an ordered plan in the
    # shared holographic space (steps are the encoder's own symbol atoms), and
    # validate_plan answers the PB&J question.
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=2048, seed=0)
    m.learn_plan("pbj", ["bread", "peanut_butter", "jelly", "close", "cut"])
    assert m.step_at("pbj", 2) == "jelly"
    assert m.precedes("pbj", "jelly", "cut") is True
    ok, viol = m.validate_plan(["bread", "cut", "peanut_butter", "jelly", "close"],
                               [("peanut_butter", "close"), ("close", "cut")])
    assert ok is False and ("close", "cut") in viol


def test_sequentiality_discovered_by_permutation_test():
    # SELF-DISCOVERY of order, no magic threshold: a class proves sequential by
    # predicting its next element better than its OWN shuffled null (a
    # permutation test). Genuinely ordered members score high z; an order-free
    # bag of the same elements scores ~0 (real order indistinguishable from
    # shuffled). z>2 is the standard significance bar, not a tuned constant.
    import numpy as np
    from holographic.agents_and_reasoning.holographic_ai import Vocabulary
    from holographic.misc.holographic_sequence import sequentiality_z
    v = Vocabulary(1024, seed=0)
    rng = np.random.default_rng(0)
    steps = [f"e{i}" for i in range(15)]
    base = list(rng.choice(steps, 8, replace=False))
    sequential = [base[rng.integers(0, 2):rng.integers(6, 9)] for _ in range(10)]
    bag = [list(rng.permutation(base))[:6] for _ in range(10)]
    assert sequentiality_z(sequential, v) > 2.0       # real order predicts
    assert sequentiality_z(bag, v) < 2.0              # shuffled is no worse


def test_unified_mind_discovers_which_classes_are_sequential():
    # The mind absorbs a sequential class and a bag class WITHOUT being told
    # which is which, and discover_sequential() finds out -- registering only
    # the true sequence for order queries. Self-discovery end to end.
    import numpy as np
    from holographic.misc.holographic_unified import UnifiedMind
    rng = np.random.default_rng(0)
    m = UnifiedMind(dim=1024, seed=0)
    canon = ["bread", "pb", "jelly", "close", "cut", "plate"]
    recipe = [canon[rng.integers(0, 2):rng.integers(4, 7)] for _ in range(10)]
    ingredients = [list(rng.permutation(canon))[:5] for _ in range(10)]
    m.learn_sequences([(s, "recipe") for s in recipe]
                      + [(s, "ingredients") for s in ingredients])
    v = m.discover_sequential()
    # verdicts are (z, status) for sequential classes, bare z for non-sequential
    rz = v["recipe"][0] if isinstance(v["recipe"], tuple) else v["recipe"]
    iz = v["ingredients"][0] if isinstance(v["ingredients"], tuple) else v["ingredients"]
    assert rz > 2.0 and iz < 2.0
    assert "recipe" in m._seq_mem().seqs            # registered for order queries
    assert "ingredients" not in m._seq_mem().seqs   # a bag, left alone
    assert m.precedes("recipe", "jelly", "cut") is True


def test_canonical_order_self_assembles_from_partial_observations():
    # SELF-ASSEMBLY: the canonical order is reconstructed from noisy partial
    # member sequences by a pairwise-precedence vote -- the mind recovers a
    # sequence it was never shown whole. Measured exact on drop-one observations.
    import numpy as np
    from holographic.misc.holographic_unified import UnifiedMind
    rng = np.random.default_rng(3)
    m = UnifiedMind(dim=1024, seed=0)
    canon = ["wake", "shower", "dress", "breakfast", "commute", "work"]
    members = []
    for _ in range(12):
        seq = list(canon)
        if rng.random() < 0.5:
            seq.pop(rng.integers(0, len(seq)))
        members.append(seq)
    m.learn_sequences([(s, "morning") for s in members])
    m.discover_sequential()
    assert m._seq_mem().seqs["morning"][1] == canon  # exact recovery


def test_recursive_hierarchy_discovery_unfolds_layers():
    # RECURSION + FRACTAL: the same permutation test applied at every layer. A
    # nested recipe (top steps, some expanding into ordered sub-recipes, some
    # atomic) unfolds into a tree the mind was never given the shape of -- and
    # the recursion STOPS honestly at atomic leaves (no sub-observations) rather
    # than at a chosen depth.
    import numpy as np
    from holographic.misc.holographic_unified import UnifiedMind
    rng = np.random.default_rng(0)
    sub_sauce = ["heat_oil", "add_garlic", "add_tomato", "simmer"]
    sub_prep = ["chop_onion", "chop_garlic", "measure_flour"]

    def noisy(canon):
        s = list(canon)
        if rng.random() < 0.4 and len(s) > 2:
            s.pop(rng.integers(0, len(s)))
        return s

    obs = [[("prep", noisy(sub_prep)), ("make_sauce", noisy(sub_sauce)),
            "assemble", "bake"] for _ in range(12)]
    m = UnifiedMind(dim=1024, seed=0)
    m.learn_hierarchical("dinner", obs)
    tree = m.discover_hierarchy("dinner")

    assert list(tree.keys())[:2] == ["prep", "make_sauce"]   # top order recovered
    assert tree["assemble"] is None and tree["bake"] is None  # atomic leaves stop
    # expandable steps recursed: make_sauce's sub-order recovered
    sauce = tree["make_sauce"]
    sauce_order = list(sauce.keys()) if isinstance(sauce, dict) else sauce
    assert sauce_order[0] == "heat_oil" and sauce_order[-1] == "simmer"


def test_recursion_stops_at_unordered_substep():
    # HONEST TERMINATION: a step WITH sub-observations that are an unordered bag
    # must NOT be falsely expanded -- the permutation test stops there. The mind
    # tells an ordered sub-recipe from an unordered ingredient list by
    # measurement, not by assumption.
    import numpy as np
    from holographic.misc.holographic_unified import UnifiedMind
    rng = np.random.default_rng(1)
    sub_sauce = ["heat_oil", "add_garlic", "add_tomato", "simmer"]
    garnish = ["parsley", "basil", "pepper", "salt"]
    obs = [[("make_sauce", list(sub_sauce[rng.integers(0, 2):])),
            ("garnish", list(rng.permutation(garnish))[:3]), "serve"]
           for _ in range(12)]
    m = UnifiedMind(dim=1024, seed=0)
    m.learn_hierarchical("plate", obs)
    tree = m.discover_hierarchy("plate")
    # make_sauce expands; garnish (random order) does not
    assert tree["make_sauce"] is not None
    assert tree["garnish"] is None


def test_executability_proof_catches_cycles():
    # SELF-PROOF: a discovered order must prove EXECUTABLE, not merely score z>2.
    # A precedence cycle (A before B, B before C, C before A) admits no
    # consistent ordering -- the proof catches it where the statistical test
    # might not. Structure earns trust by passing this.
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=512, seed=0)
    ok, viol = m.prove_executable([["a", "b", "c"], ["a", "b"], ["b", "c"]])
    assert ok is True and viol == []
    bad_ok, bad_viol = m.prove_executable([["a", "b"], ["b", "c"], ["c", "a"]] * 3)
    assert bad_ok is False and bad_viol            # a cycle is named


def test_canonical_order_respects_strong_majority_edges():
    # The proof surfaced a real bug: a score-heuristic sort could misplace a
    # rare element against a 4-0 majority. The topological sort respects every
    # consistent majority edge, so a clean dataset yields its exact order.
    import numpy as np
    from holographic.misc.holographic_unified import UnifiedMind
    rng = np.random.default_rng(0)
    m = UnifiedMind(dim=512, seed=0)
    canon = ["bread", "pb", "jelly", "close", "cut", "plate"]
    recipe = [canon[rng.integers(0, 2):rng.integers(4, 7)] for _ in range(10)]
    assert m._canonical_order(recipe) == canon       # exact, majority-respecting
    assert m.prove_executable(recipe) == (True, [])


def test_extract_template_finds_context_slots():
    # CONTEXT-BINDING: a step is a generic SCHEMA plus context-filled SLOTS.
    # 'the material has density X' -- the schema words are stable, X varies. The
    # mind separates them by per-position entropy, splitting at the natural gap
    # (no magic threshold). This is 'F = m*a' generic until a scenario binds the
    # values; the schema is the law, the slot is where context enters.
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=512, seed=0)
    obs = [["the", "material", "has", "density", "5g"],
           ["the", "material", "has", "density", "3g"],
           ["the", "material", "has", "density", "8g"],
           ["the", "material", "has", "density", "2g"]]
    template, slots = m.extract_template(obs)
    assert template == ["the", "material", "has", "density", "<_>"]
    assert set(slots.keys()) == {4}                  # only the value position
    assert set(slots[4]) == {"5g", "3g", "8g", "2g"}
    # a fully-fixed step has no slots
    t2, s2 = m.extract_template([["wash", "hands", "then", "begin"]] * 4)
    assert s2 == {} and t2 == ["wash", "hands", "then", "begin"]


def test_execute_plan_fires_on_preconditions_and_binds_context():
    # THE CLOSED LOOP: a discovered, proven plan is RUN. A step fires only when
    # its preconditions (earlier steps) have fired AND its context slots bind;
    # otherwise it blocks with a reason. No assumed success.
    import numpy as np
    from holographic.misc.holographic_unified import UnifiedMind
    rng = np.random.default_rng(0)
    m = UnifiedMind(dim=1024, seed=0)
    canon = ["bread", "pb", "jelly", "close", "cut", "plate"]
    recipe = [canon[rng.integers(0, 2):rng.integers(4, 7)] for _ in range(10)]
    m.learn_sequences([(s, "recipe") for s in recipe])
    m.discover_sequential()
    templates = {"cut": (["cut", "into", "<_>", "pieces"], ["pieces"])}

    # correct order with the binding present: everything fires, slot filled
    log = m.execute_plan("recipe", context={"pieces": "2"}, templates=templates)
    assert all(st == "fired" for _, st, _ in log)
    cut_line = [d for s, st, d in log if s == "cut"][0]
    assert cut_line == "cut into 2 pieces"            # context bound into the step

    # without the binding, cut blocks and plate cascades-blocks behind it
    log2 = m.execute_plan("recipe", context={}, templates=templates)
    status = {s: st for s, st, _ in log2}
    assert status["cut"] == "blocked" and status["plate"] == "blocked"
    assert status["bread"] == "fired"                 # earlier steps still fire


def test_execute_plan_blocks_out_of_order_attempts():
    # Preconditions are real: attempting a late step first BLOCKS it, naming the
    # steps it still needs -- the plan cannot be cheated out of order.
    import numpy as np
    from holographic.misc.holographic_unified import UnifiedMind
    rng = np.random.default_rng(0)
    m = UnifiedMind(dim=1024, seed=0)
    canon = ["bread", "pb", "jelly", "close", "cut", "plate"]
    recipe = [canon[rng.integers(0, 2):rng.integers(4, 7)] for _ in range(10)]
    m.learn_sequences([(s, "recipe") for s in recipe])
    m.discover_sequential()
    order = m._seq_mem().seqs["recipe"][1]
    attempt = ["cut"] + [s for s in order if s != "cut"]
    log = m.execute_plan("recipe", attempt_order=attempt)
    first = log[0]
    assert first[0] == "cut" and first[1] == "blocked"


def test_execute_refuses_unproven_plan():
    # You cannot run what was never proven: executing an unregistered plan raises.
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=512, seed=0)
    import pytest
    with pytest.raises(ValueError):
        m.execute_plan("never_discovered")


def test_absorb_auto_discovers_sequential_classes():
    # WIRED IN, not just a manual method: absorb() itself auto-discovers which
    # absorbed list-classes are sequential -- order becomes a property of
    # self-assembly. Mixed procedures and bags, one call, correct separation.
    import numpy as np
    from holographic.misc.holographic_unified import UnifiedMind
    rng = np.random.default_rng(0)
    procs = {"tea": ["kettle", "boil", "pour", "steep", "serve"],
             "wash": ["sort", "load", "soap", "start", "dry"]}
    bags = {"fruit": ["apple", "pear", "plum", "fig", "lime"]}
    examples = []
    for name, canon in procs.items():
        for _ in range(30):
            s = list(canon)
            if rng.random() < 0.4 and len(s) > 2:
                s.pop(rng.integers(0, len(s)))
            examples.append((s, name))
    for name, pool in bags.items():
        for _ in range(30):
            examples.append((list(rng.permutation(pool))[:4], name))
    rng.shuffle(examples)
    m = UnifiedMind(dim=2048, seed=0)
    m.absorb(examples, maintain=False)               # auto-discovery happens here
    registered = set(m._seq_mem().seqs)
    assert registered == set(procs)                  # procedures found
    assert not (registered & set(bags))              # bags left alone
    assert m._seq_mem().seqs["tea"][1] == procs["tea"]   # exact order


def test_creature_route_capture_and_discovery():
    # CREATURE USES THE LATEST BRAIN: a trained maze solver's successful escape
    # routes are captured and the sequence machinery discovers + proves their
    # canonical structure. Acting, then understanding the action's structure.
    from holographic.misc.holographic_creature import GridWorld, learn_maze, capture_route
    from holographic.misc.holographic_unified import UnifiedMind
    import io, contextlib

    def make_world():
        return GridWorld(width=9, height=9, maze=True, seed=5)
    with contextlib.redirect_stdout(io.StringIO()):
        enc, mind, rate = learn_maze(make_world, dim=256, episodes=150, mem=2)
        routes = capture_route(make_world, enc, mind, mem=2, trials=8)
    assert len(routes) >= 3                           # the brain escapes reliably
    m = UnifiedMind(dim=2048, seed=0)
    m.learn_sequences([(r, "route") for r in routes])
    v = m.discover_sequential()
    rz = v["route"][0] if isinstance(v["route"], tuple) else v["route"]
    assert rz > 2.0                                   # route is genuinely ordered
    assert "route" in m._seq_mem().seqs
    assert m.prove_executable(routes)[0] is True      # and proven executable


def test_replay_plan_navigates_and_detects_breaks():
    # COMPOSITION: a discovered, proven route plan drives navigation by replay,
    # and knows the boundary of its own validity. In its own maze it escapes; in
    # a DIFFERENT maze it detects exactly where the plan breaks (a blocked move)
    # rather than falsely succeeding -- the break point is the seam where reality
    # changed. Acting on discovered structure, honestly.
    import io, contextlib
    from holographic.misc.holographic_creature import GridWorld, learn_maze, capture_route, replay_plan
    from holographic.misc.holographic_unified import UnifiedMind

    def world_a():
        return GridWorld(width=9, height=9, maze=True, seed=5)

    def world_b():
        return GridWorld(width=9, height=9, maze=True, seed=9)

    with contextlib.redirect_stdout(io.StringIO()):
        enc, mind, rate = learn_maze(world_a, dim=256, episodes=150, mem=2)
        routes = capture_route(world_a, enc, mind, mem=2, trials=8)
    m = UnifiedMind(dim=2048, seed=0)
    m.learn_sequences([(r, "route") for r in routes])
    m.discover_sequential()
    canon = m._seq_mem().seqs["route"][1]

    # in its own maze: the plan escapes
    assert replay_plan(world_a(), canon)[0] == "escaped"
    # in a different maze: the plan breaks, and says where -- not a false escape
    statuses = [replay_plan(world_b(), canon)[0] for _ in range(5)]
    assert all(s == "broke" for s in statuses)


def test_ordered_recall_on_realistic_lists():
    """Recipes / directions / instructions: full ordered recall is exact, and 'what is step i',
    precedence, and constraint validation all hold on realistic (<= 20 step) lists."""
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=2048, seed=0)
    recipe = ["mix flour and sugar", "beat the eggs", "add the milk", "whisk the batter",
              "heat the pan", "pour the batter", "flip the pancake", "serve warm"]
    m.learn_plan("pancakes", recipe)
    assert [m.step_at("pancakes", i) for i in range(len(recipe))] == recipe   # exact ordered recall
    assert m.precedes("pancakes", "beat the eggs", "serve warm") is True
    assert m.precedes("pancakes", "serve warm", "heat the pan") is False
    ok, viol = m.validate_plan("pancakes", [("heat the pan", "pour the batter")])
    assert ok and not viol
    bad_ok, bad_viol = m.validate_plan("pancakes", [("serve warm", "beat the eggs")])
    assert (not bad_ok) and ("serve warm", "beat the eggs") in bad_viol


def test_sequence_capacity_is_far_past_eight():
    """Pins the MEASURED capacity (the old '~8' note was far too conservative): at dim 2048,
    forced-choice step recall is exact at length 20 and still strong (>= 88%) at length 120."""
    short = SequenceMemory(dim=2048, seed=0)
    s20 = [f"s{j}" for j in range(20)]
    short.add("p", s20)
    assert all(short.step("p", i) == s20[i] for i in range(20))               # exact at 20
    accs = []
    for t in range(5):
        sm = SequenceMemory(dim=2048, seed=t)
        s120 = [f"s{j}" for j in range(120)]
        sm.add("p", s120)
        accs.append(np.mean([sm.step("p", i) == s120[i] for i in range(120)]))
    assert np.mean(accs) >= 0.88                                              # graceful, not a cliff


def test_repeated_step_recall_and_the_position_of_limit():
    """A recurring step: position -> element is correct at EVERY occurrence (the encoding is
    position-indexed), but element -> position (position_of) collapses to a single slot. Kept
    negative, pinned so it cannot silently change."""
    sm = SequenceMemory(dim=2048, seed=0)
    rep = ["add water", "stir", "add flour", "stir", "add eggs", "stir", "bake"]   # stir at 1,3,5
    sm.add("dough", rep)
    assert [sm.step("dough", i) for i in range(len(rep))] == rep              # every slot recalls right
    assert sm.position_of("dough", "stir") in (1, 3, 5)                       # returns ONE of them only


def test_chunked_storage_keeps_long_sequence_queries_exact():
    """add(..., chunk=K) stores a long sequence as positional blocks so vector-only position/order queries
    stay EXACT past the single-bundle cap, where the unchunked positional encoding decays badly with length
    (measured: ~36% single vs 100% chunked at length 400, dim 2048). chunk=0 (default) is unchanged."""
    sm = SequenceMemory(dim=2048, seed=0)
    seq = [f"op{i}" for i in range(120)]                        # past where a single bundle stays reliable
    sm.add("plan", seq, chunk=14)
    # step_at is exact at every position
    assert all(sm.step("plan", i) == seq[i] for i in range(0, 120, 7))
    # order relation exact across a long gap (single-bundle would be unreliable here)
    assert sm.precedes("plan", "op5", "op110") is True
    assert sm.precedes("plan", "op110", "op5") is False
    # the kept element list is still at index 1 (backward-compatible storage shape)
    assert sm.seqs["plan"][1] == seq
    # chunked vs single is a no-op on a SHORT sequence (same answers)
    short = ["a", "b", "c", "d"]
    sm.add("s0", short); sm.add("s1", short, chunk=14)
    assert [sm.step("s0", i) for i in range(4)] == [sm.step("s1", i) for i in range(4)] == short


def test_chunked_storage_is_backward_compatible_default():
    # default add (no chunk) stores a single vector and indexes elements at [1], exactly as before.
    sm = SequenceMemory(dim=2048, seed=0)
    sm.add("p", ["x", "y", "z"])
    assert sm.seqs["p"][1] == ["x", "y", "z"] and sm.seqs["p"][2] == 0    # (vector, elems, chunk=0)
    assert sm.step("p", 1) == "y"
